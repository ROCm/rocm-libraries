// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2,@function
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 18573) 
v_and_b32 v3, 48, v1 // (op 18573) 
v_sub_u32 v4, v0, v3 // (op 18573) 

v_lshrrev_b32 v3, 4, v0 // (op 18573) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 18573) 


v_lshrrev_b32 v3, 7, v1 // (op 18573) 
v_lshlrev_b32 v4, 4, v3 // (op 18573) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 18573) 


v_lshlrev_b32 v0, 2, v3 // (op 18573) 

v_lshrrev_b32 v3, 3, v0 // (op 18573) 

v_and_b32 v0, 48, v1 // (op 18648) 
v_and_b32 v4, 63, v1 // (op 18648) 
v_sub_u32 v5, v4, v0 // (op 18648) 

v_lshrrev_b32 v4, 1, v5 // (op 18648) 
v_add_lshl_u32 v6, v0, v4, 7 // (op 18648) 


v_and_b32 v0, 1, v5 // (op 18648) 

v_lshlrev_b32 v4, 3, v0 // (op 18648) 

v_lshrrev_b32 v0, 6, v1 // (op 18648) 
v_and_b32 v5, 1, v0 // (op 18648) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 18648) 


v_add_u32 v4, v6, v0 // (op 18648) 


v_lshrrev_b32 v0, 7, v1 // (op 18724) 
v_and_b32 v5, 15, v1 // (op 18724) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 18724) 


v_and_b32 v0, 63, v1 // (op 18724) 
v_lshrrev_b32 v5, 4, v0 // (op 18724) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 18724) 


v_lshrrev_b32 v5, 6, v1 // (op 18800) 
v_and_b32 v6, 1, v5 // (op 18800) 

v_and_b32 v5, 15, v1 // (op 18800) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 18800) 


v_and_b32 v5, 63, v1 // (op 18800) 
v_lshrrev_b32 v6, 4, v5 // (op 18800) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 18800) 


v_and_b32 v6, 63, v1 // (op 18927) 
v_and_b32 v7, 48, v1 // (op 18927) 
v_sub_u32 v8, v6, v7 // (op 18927) 

v_lshrrev_b32 v7, 4, v6 // (op 18927) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 18927) 


v_lshrrev_b32 v7, 7, v1 // (op 18927) 
v_lshlrev_b32 v8, 4, v7 // (op 18927) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 18927) 


v_lshlrev_b32 v6, 2, v7 // (op 18927) 

v_lshrrev_b32 v7, 3, v6 // (op 18927) 

v_and_b32 v6, 48, v1 // (op 19002) 
v_and_b32 v8, 63, v1 // (op 19002) 
v_sub_u32 v9, v8, v6 // (op 19002) 

v_lshrrev_b32 v8, 1, v9 // (op 19002) 
v_add_lshl_u32 v10, v6, v8, 7 // (op 19002) 


v_and_b32 v6, 1, v9 // (op 19002) 

v_lshlrev_b32 v8, 3, v6 // (op 19002) 

v_lshrrev_b32 v6, 6, v1 // (op 19002) 
v_and_b32 v9, 1, v6 // (op 19002) 

v_lshl_add_u32 v6, v9, 4, v8 // (op 19002) 


v_add_u32 v8, v10, v6 // (op 19002) 


v_lshrrev_b32 v6, 7, v1 // (op 19078) 
v_and_b32 v9, 15, v1 // (op 19078) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 19078) 


v_and_b32 v6, 63, v1 // (op 19078) 
v_lshrrev_b32 v9, 4, v6 // (op 19078) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 19078) 


v_lshrrev_b32 v9, 6, v1 // (op 19154) 
v_and_b32 v10, 1, v9 // (op 19154) 

v_and_b32 v9, 15, v1 // (op 19154) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 19154) 


v_and_b32 v9, 63, v1 // (op 19154) 
v_lshrrev_b32 v10, 4, v9 // (op 19154) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 19154) 


v_and_b32 v10, 63, v1 // (op 19268) 
v_and_b32 v11, 48, v1 // (op 19268) 
v_sub_u32 v12, v10, v11 // (op 19268) 

v_lshrrev_b32 v11, 4, v10 // (op 19268) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 19268) 


v_lshrrev_b32 v11, 7, v1 // (op 19268) 
v_lshlrev_b32 v12, 4, v11 // (op 19268) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 19268) 


v_lshlrev_b32 v10, 2, v11 // (op 19268) 

v_lshrrev_b32 v11, 3, v10 // (op 19268) 

v_and_b32 v10, 48, v1 // (op 19312) 
v_and_b32 v12, 63, v1 // (op 19312) 
v_sub_u32 v13, v12, v10 // (op 19312) 

v_lshrrev_b32 v12, 1, v13 // (op 19312) 
v_add_lshl_u32 v14, v10, v12, 7 // (op 19312) 


v_and_b32 v10, 1, v13 // (op 19312) 

v_lshlrev_b32 v12, 3, v10 // (op 19312) 

v_lshrrev_b32 v10, 6, v1 // (op 19312) 
v_and_b32 v13, 1, v10 // (op 19312) 

v_lshl_add_u32 v10, v13, 4, v12 // (op 19312) 


v_add_u32 v12, v14, v10 // (op 19312) 


v_lshrrev_b32 v10, 7, v1 // (op 19341) 
v_and_b32 v13, 15, v1 // (op 19341) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 19341) 


v_and_b32 v10, 63, v1 // (op 19341) 
v_lshrrev_b32 v13, 4, v10 // (op 19341) 

v_lshl_add_u32 v10, v13, 7, v14 // (op 19341) 


v_lshrrev_b32 v13, 6, v1 // (op 19370) 
v_and_b32 v14, 1, v13 // (op 19370) 

v_and_b32 v13, 15, v1 // (op 19370) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 19370) 


v_and_b32 v13, 63, v1 // (op 19370) 
v_lshrrev_b32 v14, 4, v13 // (op 19370) 

v_lshl_add_u32 v13, v14, 7, v15 // (op 19370) 


v_and_b32 v14, 63, v1 // (op 19414) 
v_and_b32 v15, 48, v1 // (op 19414) 
v_sub_u32 v16, v14, v15 // (op 19414) 

v_lshrrev_b32 v15, 4, v14 // (op 19414) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 19414) 


v_lshrrev_b32 v15, 7, v1 // (op 19414) 
v_lshlrev_b32 v16, 4, v15 // (op 19414) 

v_lshl_add_u32 v15, v14, 7, v16 // (op 19414) 


v_lshlrev_b32 v14, 2, v15 // (op 19414) 

v_lshrrev_b32 v15, 3, v14 // (op 19414) 

v_and_b32 v14, 48, v1 // (op 19458) 
v_and_b32 v16, 63, v1 // (op 19458) 
v_sub_u32 v17, v16, v14 // (op 19458) 

v_lshrrev_b32 v16, 1, v17 // (op 19458) 
v_add_lshl_u32 v18, v14, v16, 7 // (op 19458) 


v_and_b32 v14, 1, v17 // (op 19458) 

v_lshlrev_b32 v16, 3, v14 // (op 19458) 

v_lshrrev_b32 v14, 6, v1 // (op 19458) 
v_and_b32 v17, 1, v14 // (op 19458) 

v_lshl_add_u32 v14, v17, 4, v16 // (op 19458) 


v_add_u32 v16, v18, v14 // (op 19458) 


v_lshrrev_b32 v14, 7, v1 // (op 19487) 
v_and_b32 v17, 15, v1 // (op 19487) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 19487) 


v_and_b32 v14, 63, v1 // (op 19487) 
v_lshrrev_b32 v17, 4, v14 // (op 19487) 

v_lshl_add_u32 v14, v17, 7, v18 // (op 19487) 


v_lshrrev_b32 v17, 6, v1 // (op 19516) 
v_and_b32 v18, 1, v17 // (op 19516) 

v_and_b32 v17, 15, v1 // (op 19516) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 19516) 


v_and_b32 v17, 63, v1 // (op 19516) 
v_lshrrev_b32 v18, 4, v17 // (op 19516) 

v_lshl_add_u32 v17, v18, 7, v19 // (op 19516) 


v_mov_b32 v18, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 395) call()
s_mov_b32 s1, 0 // (op 396) call()
s_mov_b32 s6, 0 // (op 397) call()
s_mov_b32 s7, 0 // (op 398) call()
s_mov_b32 s18, 0 // (op 399) call()
s_mov_b32 s19, 0 // (op 400) call()
s_mov_b32 s0, 0 // (op 295) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 7, v1 // (op 18484) 
s_mul_i32 s52, s16, s8 // (op 18484) 
s_add_u32 s52, s52, s0 // (op 18484) 
v_and_b32 v21, 63, v1 // (op 18484) 
s_mul_hi_u32 s53, s52, s13 // (op 18484) 
v_lshrrev_b32 v22, 4, v21 // (op 18484) 
s_sub_u32 s54, s52, s53 // (op 18484) 
v_lshlrev_b32 v23, 2, v22 // (op 18484) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 18484) 


s_lshr_b32 s55, s54, 1 // (op 18484) 

s_add_u32 s54, s55, s53 // (op 18484) 


v_lshrrev_b32 v20, 2, v21 // (op 18484) 

s_lshr_b32 s53, s54, s11 // (op 18484) 

v_and_b32 v21, 3, v20 // (op 18484) 

s_and_b32 s54, s11, 2147483648 // (op 18484) 
v_and_b32 v20, 3, v1 // (op 18484) 
s_cmp_eq_i32 s54, 0 // (op 18484) 
s_mov_b32 s55, scc // (op 18484) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 18484) 


s_cmp_eq_i32 s55, 0 // (op 18484) 
s_cselect_b32 s54, s52, s53 // (op 18484) 



v_lshrrev_b32 v20, 6, v1 // (op 18484) 
s_mul_hi_u32 s52, s54, s12 // (op 18484) 
v_and_b32 v21, 1, v20 // (op 18484) 

s_sub_u32 s53, s54, s52 // (op 18484) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 18484) 


s_lshr_b32 s55, s53, 1 // (op 18484) 

s_add_u32 s53, s55, s52 // (op 18484) 


s_lshr_b32 s52, s53, s10 // (op 18484) 

s_and_b32 s53, s10, 2147483648 // (op 18484) 
s_cmp_eq_i32 s53, 0 // (op 18484) 
s_mov_b32 s55, scc // (op 18484) 

s_cmp_eq_i32 s55, 0 // (op 18484) 
s_cselect_b32 s53, s54, s52 // (op 18484) 


s_mul_i32 s52, s53, s14 // (op 18484) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 18484) 


s_sub_u32 s53, s54, s52 // (op 18484) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 18484) 


v_mul_lo_u32 v20, v22, s4 // (op 18484) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 18484) 


s_mul_i32 s53, s4, s23 // (op 18486) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 18486) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 18486) least significant
s_mul_hi_u32 s55, s4, s22 // (op 18486) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18486) most significant: sum
s_add_u32 s53, s53, s55 // (op 18486) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 18486) 

s_mov_b32 s56, s20 // (op 18486) reg expression
s_mov_b32 s57, s21 // (op 18486) reg expression
s_mov_b32 s58, s54 // (op 18486) reg expression
s_mov_b32 s59, 131072 // (op 18486) call()

v_and_b32 v20, 63, v1 // (op 18590) 
v_and_b32 v21, 48, v1 // (op 18590) 
v_sub_u32 v23, v20, v21 // (op 18590) 

v_lshrrev_b32 v21, 4, v20 // (op 18590) 

v_lshl_add_u32 v20, v21, 5, v23 // (op 18590) 


v_lshrrev_b32 v21, 7, v1 // (op 18590) 
v_lshlrev_b32 v23, 4, v21 // (op 18590) 

v_lshl_add_u32 v21, v20, 7, v23 // (op 18590) 


v_lshlrev_b32 v20, 2, v21 // (op 18590) 

v_lshrrev_b32 v21, 3, v20 // (op 18590) 

v_and_b32 v20, 48, v1 // (op 18665) 
v_and_b32 v23, 63, v1 // (op 18665) 
v_sub_u32 v24, v23, v20 // (op 18665) 

v_lshrrev_b32 v23, 1, v24 // (op 18665) 
v_add_lshl_u32 v25, v20, v23, 7 // (op 18665) 


v_and_b32 v20, 1, v24 // (op 18665) 

v_lshlrev_b32 v23, 3, v20 // (op 18665) 

v_lshrrev_b32 v20, 6, v1 // (op 18665) 
v_and_b32 v24, 1, v20 // (op 18665) 

v_lshl_add_u32 v20, v24, 4, v23 // (op 18665) 


v_add_u32 v23, v25, v20 // (op 18665) 


v_lshrrev_b32 v20, 7, v1 // (op 18741) 
v_and_b32 v24, 15, v1 // (op 18741) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 18741) 


v_and_b32 v20, 63, v1 // (op 18741) 
v_lshrrev_b32 v24, 4, v20 // (op 18741) 

v_lshl_add_u32 v20, v24, 7, v25 // (op 18741) 


v_lshrrev_b32 v24, 6, v1 // (op 18817) 
v_and_b32 v25, 1, v24 // (op 18817) 

v_and_b32 v24, 15, v1 // (op 18817) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 18817) 


v_and_b32 v24, 63, v1 // (op 18817) 
v_lshrrev_b32 v25, 4, v24 // (op 18817) 

v_lshl_add_u32 v24, v25, 7, v26 // (op 18817) 


v_lshrrev_b32 v25, 7, v1 // (op 18840) 
s_mul_i32 s52, s16, s8 // (op 18840) 
s_add_u32 s52, s52, s0 // (op 18840) 
v_and_b32 v26, 63, v1 // (op 18840) 
s_mul_hi_u32 s53, s52, s13 // (op 18840) 
v_lshrrev_b32 v27, 4, v26 // (op 18840) 
s_sub_u32 s54, s52, s53 // (op 18840) 
v_lshlrev_b32 v28, 2, v27 // (op 18840) 

s_lshr_b32 s55, s54, 1 // (op 18840) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 18840) 


s_add_u32 s54, s55, s53 // (op 18840) 


v_lshrrev_b32 v25, 2, v26 // (op 18840) 

s_lshr_b32 s53, s54, s11 // (op 18840) 

v_and_b32 v26, 3, v25 // (op 18840) 

s_and_b32 s54, s11, 2147483648 // (op 18840) 
v_and_b32 v25, 3, v1 // (op 18840) 
s_cmp_eq_i32 s54, 0 // (op 18840) 
s_mov_b32 s55, scc // (op 18840) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 18840) 


s_cmp_eq_i32 s55, 0 // (op 18840) 
s_cselect_b32 s54, s52, s53 // (op 18840) 



v_lshrrev_b32 v25, 6, v1 // (op 18840) 
s_mul_hi_u32 s55, s54, s12 // (op 18840) 
v_and_b32 v26, 1, v25 // (op 18840) 

s_sub_u32 s52, s54, s55 // (op 18840) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 18840) 


s_lshr_b32 s53, s52, 1 // (op 18840) 

s_add_u32 s52, s53, s55 // (op 18840) 


s_lshr_b32 s53, s52, s10 // (op 18840) 

s_and_b32 s52, s10, 2147483648 // (op 18840) 
s_cmp_eq_i32 s52, 0 // (op 18840) 
s_mov_b32 s55, scc // (op 18840) 

s_cmp_eq_i32 s55, 0 // (op 18840) 
s_cselect_b32 s52, s54, s53 // (op 18840) 


s_mul_i32 s53, s52, s14 // (op 18840) 
v_lshl_add_u32 v26, s52, 7, v27 // (op 18840) 


s_sub_u32 s52, s54, s53 // (op 18840) 


v_lshl_add_u32 v27, s52, 7, v25 // (op 18840) 


v_mul_lo_u32 v25, v27, s24 // (op 18840) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 18840) 


s_mul_i32 s53, s24, s23 // (op 18842) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 18842) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 18842) least significant
s_mul_hi_u32 s55, s24, s22 // (op 18842) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18842) most significant: sum
s_add_u32 s53, s53, s55 // (op 18842) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 18842) 

s_mov_b32 s60, s26 // (op 18842) reg expression
s_mov_b32 s61, s27 // (op 18842) reg expression
s_mov_b32 s62, s54 // (op 18842) reg expression
s_mov_b32 s63, 131072 // (op 18842) call()

v_lshrrev_b32 v25, 5, v1 // (op 19194) 
v_lshlrev_b32 v26, 2, v25 // (op 19194) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 19194) 

v_and_b32 v26, -32, v1 // (op 19194) 
v_sub_u32 v28, v1, v26 // (op 19194) 

v_lshl_add_u32 v26, v25, 7, v28 // (op 19194) 


v_lshlrev_b32 v25, 2, v26 // (op 19194) 

s_lshl_b32 s52, s15, 16 // (op 19196) 
s_mov_b32 s64, s28 // (op 19196) reg expression
s_mov_b32 s65, s29 // (op 19196) reg expression
s_mov_b32 s66, s52 // (op 19196) reg expression
s_mov_b32 s67, 131072 // (op 19196) call()

s_mul_i32 s6, s8, s16 // (op 299) 
s_add_u32 s6, s6, s0 // (op 299) 
s_mul_hi_u32 s52, s6, s13 // (op 300) 
s_sub_u32 s53, s6, s52 // (op 300) 
s_lshr_b32 s54, s53, 1 // (op 300) 

s_add_u32 s53, s54, s52 // (op 300) 


s_lshr_b32 s52, s53, s11 // (op 300) 

s_and_b32 s53, s11, 2147483648 // (op 300) 
s_cmp_eq_i32 s53, 0 // (op 300) 
s_mov_b32 s54, scc // (op 300) 

s_cmp_eq_i32 s54, 0 // (op 300) 
s_cselect_b32 s53, s6, s52 // (op 300) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 300) 
s_mul_i32 s52, s53, s54 // (op 300) 


s_sub_u32 s18, s6, s52 // (op 300) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 301) 
s_cmp_gt_u32 s52, s18 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_sub_u32 s55, s52, s18 // (op 301) 

s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s52, 0, s55 // (op 301) 


s_cmp_gt_u32 s8, s0 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_sub_u32 s54, s8, s0 // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s55, 0, s54 // (op 301) 


s_cmp_lt_u32 s52, s55 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s54, s55, s52 // (op 301) 



s_cmp_gt_u32 s9, s6 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_sub_u32 s52, s9, s6 // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s53, 0, s52 // (op 301) 


s_cmp_lt_u32 s54, s53 // (op 301) 
s_mov_b32 s52, scc // (op 301) 
s_cmp_eq_i32 s52, 0 // (op 301) 
s_cselect_b32 s7, s53, s54 // (op 301) 



s_add_u32 s52, s18, s7 // (op 302) 
s_sub_u32 s19, s52, 1 // (op 302) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 4802) call()
v_accvgpr_write a5, 0.00000 // (op 4802) call()
v_accvgpr_write a6, 0.00000 // (op 4802) call()
v_accvgpr_write a7, 0.00000 // (op 4802) call()
v_accvgpr_write a8, 0.00000 // (op 4978) call()
v_accvgpr_write a9, 0.00000 // (op 4978) call()
v_accvgpr_write a10, 0.00000 // (op 4978) call()
v_accvgpr_write a11, 0.00000 // (op 4978) call()
v_accvgpr_write a12, 0.00000 // (op 5154) call()
v_accvgpr_write a13, 0.00000 // (op 5154) call()
v_accvgpr_write a14, 0.00000 // (op 5154) call()
v_accvgpr_write a15, 0.00000 // (op 5154) call()
v_accvgpr_write a16, 0.00000 // (op 5418) call()
v_accvgpr_write a17, 0.00000 // (op 5418) call()
v_accvgpr_write a18, 0.00000 // (op 5418) call()
v_accvgpr_write a19, 0.00000 // (op 5418) call()
v_accvgpr_write a20, 0.00000 // (op 5727) call()
v_accvgpr_write a21, 0.00000 // (op 5727) call()
v_accvgpr_write a22, 0.00000 // (op 5727) call()
v_accvgpr_write a23, 0.00000 // (op 5727) call()
v_accvgpr_write a24, 0.00000 // (op 5728) call()
v_accvgpr_write a25, 0.00000 // (op 5728) call()
v_accvgpr_write a26, 0.00000 // (op 5728) call()
v_accvgpr_write a27, 0.00000 // (op 5728) call()
v_accvgpr_write a28, 0.00000 // (op 5729) call()
v_accvgpr_write a29, 0.00000 // (op 5729) call()
v_accvgpr_write a30, 0.00000 // (op 5729) call()
v_accvgpr_write a31, 0.00000 // (op 5729) call()
v_accvgpr_write a32, 0.00000 // (op 6188) call()
v_accvgpr_write a33, 0.00000 // (op 6188) call()
v_accvgpr_write a34, 0.00000 // (op 6188) call()
v_accvgpr_write a35, 0.00000 // (op 6188) call()
v_accvgpr_write a36, 0.00000 // (op 6497) call()
v_accvgpr_write a37, 0.00000 // (op 6497) call()
v_accvgpr_write a38, 0.00000 // (op 6497) call()
v_accvgpr_write a39, 0.00000 // (op 6497) call()
v_accvgpr_write a40, 0.00000 // (op 6498) call()
v_accvgpr_write a41, 0.00000 // (op 6498) call()
v_accvgpr_write a42, 0.00000 // (op 6498) call()
v_accvgpr_write a43, 0.00000 // (op 6498) call()
v_accvgpr_write a44, 0.00000 // (op 6499) call()
v_accvgpr_write a45, 0.00000 // (op 6499) call()
v_accvgpr_write a46, 0.00000 // (op 6499) call()
v_accvgpr_write a47, 0.00000 // (op 6499) call()
v_accvgpr_write a48, 0.00000 // (op 6958) call()
v_accvgpr_write a49, 0.00000 // (op 6958) call()
v_accvgpr_write a50, 0.00000 // (op 6958) call()
v_accvgpr_write a51, 0.00000 // (op 6958) call()
v_accvgpr_write a52, 0.00000 // (op 7267) call()
v_accvgpr_write a53, 0.00000 // (op 7267) call()
v_accvgpr_write a54, 0.00000 // (op 7267) call()
v_accvgpr_write a55, 0.00000 // (op 7267) call()
v_accvgpr_write a56, 0.00000 // (op 7268) call()
v_accvgpr_write a57, 0.00000 // (op 7268) call()
v_accvgpr_write a58, 0.00000 // (op 7268) call()
v_accvgpr_write a59, 0.00000 // (op 7268) call()
v_accvgpr_write a60, 0.00000 // (op 7269) call()
v_accvgpr_write a61, 0.00000 // (op 7269) call()
v_accvgpr_write a62, 0.00000 // (op 7269) call()
v_accvgpr_write a63, 0.00000 // (op 7269) call()
v_and_b32 v26, -4, v1 // (op 18051) 
s_mul_i32 s52, s16, s8 // (op 18051) 
s_add_u32 s52, s52, s0 // (op 18051) 
v_sub_u32 v28, v1, v26 // (op 18051) 

s_mul_hi_u32 s53, s52, s13 // (op 18051) 
v_lshlrev_b32 v26, 5, v28 // (op 18051) 

s_sub_u32 s54, s52, s53 // (op 18051) 
v_lshrrev_b32 v28, 2, v1 // (op 18051) 
s_lshr_b32 s55, s54, 1 // (op 18051) 

s_add_u32 s54, s55, s53 // (op 18051) 


s_lshr_b32 s53, s54, s11 // (op 18051) 

s_and_b32 s54, s11, 2147483648 // (op 18051) 
s_cmp_eq_i32 s54, 0 // (op 18051) 
s_mov_b32 s55, scc // (op 18051) 

s_cmp_eq_i32 s55, 0 // (op 18051) 
s_cselect_b32 s54, s52, s53 // (op 18051) 


s_mul_hi_u32 s53, s54, s12 // (op 18051) 
s_sub_u32 s55, s54, s53 // (op 18051) 
s_lshr_b32 s68, s55, 1 // (op 18051) 

s_add_u32 s55, s68, s53 // (op 18051) 


s_lshr_b32 s53, s55, s10 // (op 18051) 

s_and_b32 s55, s10, 2147483648 // (op 18051) 
s_cmp_eq_i32 s55, 0 // (op 18051) 
s_mov_b32 s68, scc // (op 18051) 

s_cmp_eq_i32 s68, 0 // (op 18051) 
s_cselect_b32 s55, s54, s53 // (op 18051) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 18051) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 18051) 
v_mov_b32 v31, 0 // (op 18051) convert
v_mov_b32 v30, v29 // (op 18051) convert

s_mul_i32 s53, s54, s68 // (op 18051) 


s_sub_u32 s54, s52, s53 // (op 18051) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 18051) 


v_mul_lo_u32 v29, v26, s31 // (op 18051) most significant: low of low * high
v_mul_hi_u32 v32, v26, s30 // (op 18051) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 18051) least significant: low of low * low
v_add_u32 v29, v29, v32 // (op 18051) most significant: sum


v_add_co_u32 v32, s[52:53], v30, v28 // (op 18051) least significant half
v_addc_co_u32 v33, s[52:53], v31, v29, s[52:53] // (op 18051) most significant half

v_lshlrev_b64 v[32:33], 2, v[32:33] // (op 18051) 


v_lshrrev_b64 v[28:29], 3, v[32:33] // (op 18051) 

s_mul_i32 s53, s30, s3 // (op 18053) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 18053) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 18053) least significant
s_mul_hi_u32 s55, s30, s2 // (op 18053) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18053) most significant: sum
s_add_u32 s53, s53, s55 // (op 18053) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 18053) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 18053) 

s_mov_b32 s68, s32 // (op 18053) reg expression
s_mov_b32 s69, s33 // (op 18053) reg expression
s_mov_b32 s70, s52 // (op 18053) reg expression
s_mov_b32 s71, 131072 // (op 18053) call()

v_and_b32 v26, -4, v1 // (op 18070) 
s_mul_i32 s52, s16, s8 // (op 18070) 
s_add_u32 s52, s52, s0 // (op 18070) 
v_sub_u32 v30, v1, v26 // (op 18070) 

s_mul_hi_u32 s53, s52, s13 // (op 18070) 
v_lshlrev_b32 v26, 5, v30 // (op 18070) 

s_sub_u32 s54, s52, s53 // (op 18070) 
v_lshrrev_b32 v30, 2, v1 // (op 18070) 
s_lshr_b32 s55, s54, 1 // (op 18070) 

s_add_u32 s54, s55, s53 // (op 18070) 


s_lshr_b32 s53, s54, s11 // (op 18070) 

s_and_b32 s54, s11, 2147483648 // (op 18070) 
s_cmp_eq_i32 s54, 0 // (op 18070) 
s_mov_b32 s55, scc // (op 18070) 

s_cmp_eq_i32 s55, 0 // (op 18070) 
s_cselect_b32 s54, s52, s53 // (op 18070) 


s_mul_hi_u32 s53, s54, s12 // (op 18070) 
s_sub_u32 s55, s54, s53 // (op 18070) 
s_lshr_b32 s72, s55, 1 // (op 18070) 

s_add_u32 s55, s72, s53 // (op 18070) 


s_lshr_b32 s53, s55, s10 // (op 18070) 

s_and_b32 s55, s10, 2147483648 // (op 18070) 
s_cmp_eq_i32 s55, 0 // (op 18070) 
s_mov_b32 s72, scc // (op 18070) 

s_cmp_eq_i32 s72, 0 // (op 18070) 
s_cselect_b32 s55, s54, s53 // (op 18070) 


v_lshl_add_u32 v31, s55, 7, v26 // (op 18070) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 18070) 
v_mov_b32 v33, 0 // (op 18070) convert
v_mov_b32 v32, v31 // (op 18070) convert

s_mul_i32 s53, s54, s72 // (op 18070) 


s_sub_u32 s54, s52, s53 // (op 18070) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 18070) 


v_mul_lo_u32 v31, v26, s31 // (op 18070) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 18070) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 18070) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 18070) most significant: sum


v_add_co_u32 v34, s[52:53], v32, v30 // (op 18070) least significant half
v_addc_co_u32 v35, s[52:53], v33, v31, s[52:53] // (op 18070) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 18070) 


v_lshrrev_b64 v[30:31], 3, v[34:35] // (op 18070) 

s_mul_i32 s53, s30, s3 // (op 18072) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 18072) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 18072) least significant
s_mul_hi_u32 s55, s30, s2 // (op 18072) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18072) most significant: sum
s_add_u32 s53, s53, s55 // (op 18072) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 18072) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 18072) 

s_mov_b32 s68, s32 // (op 18072) reg expression
s_mov_b32 s69, s33 // (op 18072) reg expression
s_mov_b32 s70, s52 // (op 18072) reg expression
s_mov_b32 s71, 131072 // (op 18072) call()

v_and_b32 v26, -8, v1 // (op 18179) 
s_mul_i32 s52, s16, s8 // (op 18179) 
s_add_u32 s52, s52, s0 // (op 18179) 
v_sub_u32 v32, v1, v26 // (op 18179) 

s_mul_hi_u32 s53, s52, s13 // (op 18179) 
v_lshlrev_b32 v26, 4, v32 // (op 18179) 

s_sub_u32 s54, s52, s53 // (op 18179) 
v_lshrrev_b32 v32, 3, v1 // (op 18179) 
s_lshr_b32 s55, s54, 1 // (op 18179) 

s_add_u32 s54, s55, s53 // (op 18179) 


s_lshr_b32 s53, s54, s11 // (op 18179) 

s_and_b32 s54, s11, 2147483648 // (op 18179) 
s_cmp_eq_i32 s54, 0 // (op 18179) 
s_mov_b32 s55, scc // (op 18179) 

s_cmp_eq_i32 s55, 0 // (op 18179) 
s_cselect_b32 s54, s52, s53 // (op 18179) 


s_mul_hi_u32 s53, s54, s12 // (op 18179) 
s_sub_u32 s55, s54, s53 // (op 18179) 
s_lshr_b32 s72, s55, 1 // (op 18179) 

s_add_u32 s55, s72, s53 // (op 18179) 


s_lshr_b32 s53, s55, s10 // (op 18179) 

s_and_b32 s55, s10, 2147483648 // (op 18179) 
s_cmp_eq_i32 s55, 0 // (op 18179) 
s_mov_b32 s72, scc // (op 18179) 

s_cmp_eq_i32 s72, 0 // (op 18179) 
s_cselect_b32 s55, s54, s53 // (op 18179) 


s_mul_i32 s53, s55, s14 // (op 18179) 

s_sub_u32 s55, s54, s53 // (op 18179) 

v_lshl_add_u32 v33, s55, 7, v26 // (op 18179) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 18179) 
v_mov_b32 v35, 0 // (op 18179) convert
v_mov_b32 v34, v33 // (op 18179) convert

s_mul_i32 s53, s54, s72 // (op 18179) 


s_sub_u32 s54, s52, s53 // (op 18179) 


v_lshl_add_u32 v26, s54, 7, v32 // (op 18179) 


v_mul_lo_u32 v33, v26, s35 // (op 18179) most significant: low of low * high
v_mul_hi_u32 v36, v26, s34 // (op 18179) most significant: high of low * low
v_mul_lo_u32 v32, v26, s34 // (op 18179) least significant: low of low * low
v_add_u32 v33, v33, v36 // (op 18179) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v34 // (op 18179) least significant half
v_addc_co_u32 v33, s[52:53], v33, v35, s[52:53] // (op 18179) most significant half



s_mul_i32 s53, s34, s3 // (op 18181) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 18181) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 18181) least significant
s_mul_hi_u32 s55, s34, s2 // (op 18181) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18181) most significant: sum
s_add_u32 s53, s53, s55 // (op 18181) most significant: sum


s_mov_b32 s72, s36 // (op 18181) reg expression
s_mov_b32 s73, s37 // (op 18181) reg expression
s_mov_b32 s74, s52 // (op 18181) reg expression
s_mov_b32 s75, 131072 // (op 18181) call()

v_and_b32 v26, -8, v1 // (op 18198) 
s_mul_i32 s52, s16, s8 // (op 18198) 
s_add_u32 s52, s52, s0 // (op 18198) 
v_sub_u32 v34, v1, v26 // (op 18198) 

s_mul_hi_u32 s53, s52, s13 // (op 18198) 
v_lshlrev_b32 v26, 4, v34 // (op 18198) 

s_sub_u32 s54, s52, s53 // (op 18198) 
v_lshrrev_b32 v34, 3, v1 // (op 18198) 
s_lshr_b32 s55, s54, 1 // (op 18198) 

s_add_u32 s54, s55, s53 // (op 18198) 


s_lshr_b32 s53, s54, s11 // (op 18198) 

s_and_b32 s54, s11, 2147483648 // (op 18198) 
s_cmp_eq_i32 s54, 0 // (op 18198) 
s_mov_b32 s55, scc // (op 18198) 

s_cmp_eq_i32 s55, 0 // (op 18198) 
s_cselect_b32 s54, s52, s53 // (op 18198) 


s_mul_hi_u32 s53, s54, s12 // (op 18198) 
s_sub_u32 s55, s54, s53 // (op 18198) 
s_lshr_b32 s76, s55, 1 // (op 18198) 

s_add_u32 s55, s76, s53 // (op 18198) 


s_lshr_b32 s53, s55, s10 // (op 18198) 

s_and_b32 s55, s10, 2147483648 // (op 18198) 
s_cmp_eq_i32 s55, 0 // (op 18198) 
s_mov_b32 s76, scc // (op 18198) 

s_cmp_eq_i32 s76, 0 // (op 18198) 
s_cselect_b32 s55, s54, s53 // (op 18198) 


s_mul_i32 s53, s55, s14 // (op 18198) 

s_sub_u32 s55, s54, s53 // (op 18198) 

v_lshl_add_u32 v35, s55, 7, v26 // (op 18198) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 18198) 
v_mov_b32 v37, 0 // (op 18198) convert
v_mov_b32 v36, v35 // (op 18198) convert

s_mul_i32 s53, s54, s76 // (op 18198) 


s_sub_u32 s54, s52, s53 // (op 18198) 


v_lshl_add_u32 v26, s54, 7, v34 // (op 18198) 


v_mul_lo_u32 v35, v26, s35 // (op 18198) most significant: low of low * high
v_mul_hi_u32 v38, v26, s34 // (op 18198) most significant: high of low * low
v_mul_lo_u32 v34, v26, s34 // (op 18198) least significant: low of low * low
v_add_u32 v35, v35, v38 // (op 18198) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v36 // (op 18198) least significant half
v_addc_co_u32 v35, s[52:53], v35, v37, s[52:53] // (op 18198) most significant half



s_mul_i32 s53, s34, s3 // (op 18200) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 18200) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 18200) least significant
s_mul_hi_u32 s55, s34, s2 // (op 18200) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18200) most significant: sum
s_add_u32 s53, s53, s55 // (op 18200) most significant: sum


s_mov_b32 s72, s36 // (op 18200) reg expression
s_mov_b32 s73, s37 // (op 18200) reg expression
s_mov_b32 s74, s52 // (op 18200) reg expression
s_mov_b32 s75, 131072 // (op 18200) call()

v_and_b32 v26, -128, v1 // (op 18307) 
s_mul_i32 s52, s16, s8 // (op 18307) 
s_add_u32 s52, s52, s0 // (op 18307) 
v_sub_u32 v36, v1, v26 // (op 18307) 

s_mul_hi_u32 s53, s52, s13 // (op 18307) 
v_lshrrev_b32 v26, 7, v1 // (op 18307) 
s_sub_u32 s54, s52, s53 // (op 18307) 
v_lshlrev_b32 v37, 1, v26 // (op 18307) 

s_lshr_b32 s55, s54, 1 // (op 18307) 

s_add_u32 s54, s55, s53 // (op 18307) 


s_lshr_b32 s53, s54, s11 // (op 18307) 

s_and_b32 s54, s11, 2147483648 // (op 18307) 
s_cmp_eq_i32 s54, 0 // (op 18307) 
s_mov_b32 s55, scc // (op 18307) 

s_cmp_eq_i32 s55, 0 // (op 18307) 
s_cselect_b32 s54, s52, s53 // (op 18307) 


s_mul_hi_u32 s53, s54, s12 // (op 18307) 
s_sub_u32 s55, s54, s53 // (op 18307) 
s_lshr_b32 s76, s55, 1 // (op 18307) 

s_add_u32 s55, s76, s53 // (op 18307) 


s_lshr_b32 s53, s55, s10 // (op 18307) 

s_and_b32 s55, s10, 2147483648 // (op 18307) 
s_cmp_eq_i32 s55, 0 // (op 18307) 
s_mov_b32 s76, scc // (op 18307) 

s_cmp_eq_i32 s76, 0 // (op 18307) 
s_cselect_b32 s55, s54, s53 // (op 18307) 


v_lshl_add_u32 v26, s55, 7, v36 // (op 18307) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 18307) 
s_mul_i32 s53, s54, s76 // (op 18307) 


s_sub_u32 s54, s52, s53 // (op 18307) 


v_lshl_add_u32 v36, s54, 2, v37 // (op 18307) 


v_mul_lo_u32 v37, v36, s38 // (op 18307) 

v_add_u32 v36, v26, v37 // (op 18307) 


s_mul_i32 s53, s38, s43 // (op 18309) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18309) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18309) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18309) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18309) most significant: sum
s_add_u32 s53, s53, s55 // (op 18309) most significant: sum


s_mov_b32 s76, s40 // (op 18309) reg expression
s_mov_b32 s77, s41 // (op 18309) reg expression
s_mov_b32 s78, s52 // (op 18309) reg expression
s_mov_b32 s79, 131072 // (op 18309) call()

v_and_b32 v26, -128, v1 // (op 18326) 
s_mul_i32 s52, s16, s8 // (op 18326) 
s_add_u32 s52, s52, s0 // (op 18326) 
v_sub_u32 v37, v1, v26 // (op 18326) 

s_mul_hi_u32 s53, s52, s13 // (op 18326) 
v_lshrrev_b32 v26, 7, v1 // (op 18326) 
s_sub_u32 s54, s52, s53 // (op 18326) 
v_lshlrev_b32 v38, 1, v26 // (op 18326) 

s_lshr_b32 s55, s54, 1 // (op 18326) 

s_add_u32 s54, s55, s53 // (op 18326) 


s_lshr_b32 s53, s54, s11 // (op 18326) 

s_and_b32 s54, s11, 2147483648 // (op 18326) 
s_cmp_eq_i32 s54, 0 // (op 18326) 
s_mov_b32 s55, scc // (op 18326) 

s_cmp_eq_i32 s55, 0 // (op 18326) 
s_cselect_b32 s54, s52, s53 // (op 18326) 


s_mul_hi_u32 s53, s54, s12 // (op 18326) 
s_sub_u32 s55, s54, s53 // (op 18326) 
s_lshr_b32 s80, s55, 1 // (op 18326) 

s_add_u32 s55, s80, s53 // (op 18326) 


s_lshr_b32 s53, s55, s10 // (op 18326) 

s_and_b32 s55, s10, 2147483648 // (op 18326) 
s_cmp_eq_i32 s55, 0 // (op 18326) 
s_mov_b32 s80, scc // (op 18326) 

s_cmp_eq_i32 s80, 0 // (op 18326) 
s_cselect_b32 s55, s54, s53 // (op 18326) 


v_lshl_add_u32 v26, s55, 7, v37 // (op 18326) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 18326) 
s_mul_i32 s53, s54, s80 // (op 18326) 


s_sub_u32 s54, s52, s53 // (op 18326) 


v_lshl_add_u32 v37, s54, 2, v38 // (op 18326) 


v_mul_lo_u32 v38, v37, s38 // (op 18326) 

v_add_u32 v37, v26, v38 // (op 18326) 


s_mul_i32 s53, s38, s43 // (op 18328) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18328) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18328) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18328) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18328) most significant: sum
s_add_u32 s53, s53, s55 // (op 18328) most significant: sum


s_mov_b32 s76, s40 // (op 18328) reg expression
s_mov_b32 s77, s41 // (op 18328) reg expression
s_mov_b32 s78, s52 // (op 18328) reg expression
s_mov_b32 s79, 131072 // (op 18328) call()

v_and_b32 v26, -128, v1 // (op 18435) 
s_mul_i32 s52, s16, s8 // (op 18435) 
s_add_u32 s52, s52, s0 // (op 18435) 
v_sub_u32 v38, v1, v26 // (op 18435) 

s_mul_hi_u32 s53, s52, s13 // (op 18435) 
v_lshrrev_b32 v26, 7, v1 // (op 18435) 
s_sub_u32 s54, s52, s53 // (op 18435) 
v_lshlrev_b32 v39, 1, v26 // (op 18435) 

s_lshr_b32 s55, s54, 1 // (op 18435) 

s_add_u32 s54, s55, s53 // (op 18435) 


s_lshr_b32 s53, s54, s11 // (op 18435) 

s_and_b32 s54, s11, 2147483648 // (op 18435) 
s_cmp_eq_i32 s54, 0 // (op 18435) 
s_mov_b32 s55, scc // (op 18435) 

s_cmp_eq_i32 s55, 0 // (op 18435) 
s_cselect_b32 s54, s52, s53 // (op 18435) 


s_mul_hi_u32 s53, s54, s12 // (op 18435) 
s_sub_u32 s55, s54, s53 // (op 18435) 
s_lshr_b32 s80, s55, 1 // (op 18435) 

s_add_u32 s55, s80, s53 // (op 18435) 


s_lshr_b32 s53, s55, s10 // (op 18435) 

s_and_b32 s55, s10, 2147483648 // (op 18435) 
s_cmp_eq_i32 s55, 0 // (op 18435) 
s_mov_b32 s80, scc // (op 18435) 

s_cmp_eq_i32 s80, 0 // (op 18435) 
s_cselect_b32 s55, s54, s53 // (op 18435) 


s_mul_i32 s53, s55, s14 // (op 18435) 

s_sub_u32 s55, s54, s53 // (op 18435) 

v_lshl_add_u32 v26, s55, 7, v38 // (op 18435) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 18435) 
s_mul_i32 s53, s54, s80 // (op 18435) 


s_sub_u32 s54, s52, s53 // (op 18435) 


v_lshl_add_u32 v38, s54, 2, v39 // (op 18435) 


v_mul_lo_u32 v39, v38, s44 // (op 18435) 
v_add_u32 v39, v39, v26 // (op 18435) 


s_mul_i32 s53, s44, s43 // (op 18437) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 18437) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 18437) least significant
s_mul_hi_u32 s55, s44, s42 // (op 18437) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18437) most significant: sum
s_add_u32 s53, s53, s55 // (op 18437) most significant: sum


s_mov_b32 s80, s46 // (op 18437) reg expression
s_mov_b32 s81, s47 // (op 18437) reg expression
s_mov_b32 s82, s52 // (op 18437) reg expression
s_mov_b32 s83, 131072 // (op 18437) call()

v_and_b32 v26, -128, v1 // (op 18454) 
s_mul_i32 s52, s16, s8 // (op 18454) 
s_add_u32 s52, s52, s0 // (op 18454) 
v_sub_u32 v38, v1, v26 // (op 18454) 

s_mul_hi_u32 s53, s52, s13 // (op 18454) 
v_lshrrev_b32 v26, 7, v1 // (op 18454) 
s_sub_u32 s54, s52, s53 // (op 18454) 
v_lshlrev_b32 v40, 1, v26 // (op 18454) 

s_lshr_b32 s55, s54, 1 // (op 18454) 

s_add_u32 s54, s55, s53 // (op 18454) 


s_lshr_b32 s53, s54, s11 // (op 18454) 

s_and_b32 s54, s11, 2147483648 // (op 18454) 
s_cmp_eq_i32 s54, 0 // (op 18454) 
s_mov_b32 s55, scc // (op 18454) 

s_cmp_eq_i32 s55, 0 // (op 18454) 
s_cselect_b32 s54, s52, s53 // (op 18454) 


s_mul_hi_u32 s53, s54, s12 // (op 18454) 
s_sub_u32 s55, s54, s53 // (op 18454) 
s_lshr_b32 s84, s55, 1 // (op 18454) 

s_add_u32 s55, s84, s53 // (op 18454) 


s_lshr_b32 s53, s55, s10 // (op 18454) 

s_and_b32 s55, s10, 2147483648 // (op 18454) 
s_cmp_eq_i32 s55, 0 // (op 18454) 
s_mov_b32 s84, scc // (op 18454) 

s_cmp_eq_i32 s84, 0 // (op 18454) 
s_cselect_b32 s55, s54, s53 // (op 18454) 


s_mul_i32 s53, s55, s14 // (op 18454) 

s_sub_u32 s55, s54, s53 // (op 18454) 

v_lshl_add_u32 v26, s55, 7, v38 // (op 18454) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 18454) 
s_mul_i32 s53, s54, s84 // (op 18454) 


s_sub_u32 s54, s52, s53 // (op 18454) 


v_lshl_add_u32 v38, s54, 2, v40 // (op 18454) 


v_mul_lo_u32 v40, v38, s44 // (op 18454) 
v_add_u32 v40, v40, v26 // (op 18454) 


s_mul_i32 s53, s44, s43 // (op 18456) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 18456) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 18456) least significant
s_mul_hi_u32 s55, s44, s42 // (op 18456) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18456) most significant: sum
s_add_u32 s53, s53, s55 // (op 18456) most significant: sum


s_mov_b32 s80, s46 // (op 18456) reg expression
s_mov_b32 s81, s47 // (op 18456) reg expression
s_mov_b32 s82, s52 // (op 18456) reg expression
s_mov_b32 s83, 131072 // (op 18456) call()

v_and_b32 v26, -8, v1 // (op 18631) 
v_sub_u32 v38, v1, v26 // (op 18631) 

v_lshlrev_b32 v26, 4, v38 // (op 18631) 

v_lshrrev_b32 v38, 3, v1 // (op 18631) 
v_lshl_add_u32 v41, v38, 7, v26 // (op 18631) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 18631) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19770) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 17901) 
v_mov_b64 v[42:43], s[54:55] // (op 17901) convert


v_mov_b64 v[44:45], v[32:33] // (op 17901) reg expression
s_mov_b32 m0, s52 // (op 17901) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17901) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17901) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17901) most significant half

s_add_u32 m0, m0, 4096 // (op 17901) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17901) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17901) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17901) most significant half

s_add_u32 m0, m0, 4096 // (op 17901) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17901) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17901) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17901) most significant half

s_add_u32 m0, m0, 4096 // (op 17901) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17901) Load value direct to lds




v_mov_b64 v[42:43], s[44:45] // (op 17572) convert

v_mov_b32 v26, v39 // (op 17572) reg expression
buffer_load_ubyte v44, v26, s[80:83], 0 offen offset: 0 // (op 17572) Load value
v_add_u32 v26, v26, v42 // (op 17572) 
buffer_load_ubyte v45, v26, s[80:83], 0 offen offset: 0 // (op 17572) Load value


v_and_b32 v26, -4, v1 // (op 18556) 
v_sub_u32 v38, v1, v26 // (op 18556) 

v_lshlrev_b32 v26, 5, v38 // (op 18556) 

v_lshrrev_b32 v38, 2, v1 // (op 18556) 
v_lshl_add_u32 v41, v38, 7, v26 // (op 18556) 


v_lshlrev_b32 v26, 2, v41 // (op 18556) 

v_lshrrev_b32 v38, 3, v26 // (op 18556) 

s_nop 0
v_readfirstlane_b32 s52, v38 // (op 18556) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19751) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 17906) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17906) 

v_lshrrev_b64 v[42:43], 3, s[84:85] // (op 17906) 


v_mov_b64 v[46:47], v[28:29] // (op 17906) reg expression
s_add_u32 s53, 16384, s52 // (op 17906) 
s_mov_b32 m0, s53 // (op 17906) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 17906) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 17906) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 17906) most significant half

s_add_u32 m0, m0, 4096 // (op 17906) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 17906) Load value direct to lds





v_mov_b64 v[42:43], s[38:39] // (op 17578) convert

v_mov_b32 v26, v36 // (op 17578) reg expression
buffer_load_ubyte v46, v26, s[76:79], 0 offen offset: 0 // (op 17578) Load value
v_add_u32 v26, v26, v42 // (op 17578) 
buffer_load_ubyte v47, v26, s[76:79], 0 offen offset: 0 // (op 17578) Load value


v_and_b32 v26, -8, v1 // (op 19441) 
v_sub_u32 v38, v1, v26 // (op 19441) 

v_lshlrev_b32 v26, 4, v38 // (op 19441) 

v_lshrrev_b32 v38, 3, v1 // (op 19441) 
v_lshl_add_u32 v41, v38, 7, v26 // (op 19441) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 19441) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19950) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 17911) 
v_mov_b64 v[42:43], s[54:55] // (op 17911) convert


s_lshl_b64 s[54:55], s[34:35], 7 // (op 17911) 
v_mov_b32 v26, s54 // (op 17911) 
v_mov_b32 v38, s55 // (op 17911) 
v_add_co_u32 v48, s[84:85], v32, v26 // (op 17911) least significant half
v_addc_co_u32 v49, s[84:85], v33, v38, s[84:85] // (op 17911) most significant half




s_add_u32 s53, 24576, s52 // (op 17911) 
s_mov_b32 m0, s53 // (op 17911) reg expression
buffer_load_dwordx4 v48, s[72:75], 0 offen offset: 0 lds // (op 17911) Load value direct to lds
v_add_co_u32 v48, s[54:55], v48, v42 // (op 17911) least significant half
v_addc_co_u32 v49, s[54:55], 0, v49, s[54:55] // (op 17911) most significant half

s_add_u32 m0, m0, 4096 // (op 17911) 
buffer_load_dwordx4 v48, s[72:75], 0 offen offset: 0 lds // (op 17911) Load value direct to lds
v_add_co_u32 v48, s[54:55], v48, v42 // (op 17911) least significant half
v_addc_co_u32 v49, s[54:55], 0, v49, s[54:55] // (op 17911) most significant half

s_add_u32 m0, m0, 4096 // (op 17911) 
buffer_load_dwordx4 v48, s[72:75], 0 offen offset: 0 lds // (op 17911) Load value direct to lds
v_add_co_u32 v48, s[54:55], v48, v42 // (op 17911) least significant half
v_addc_co_u32 v49, s[54:55], 0, v49, s[54:55] // (op 17911) most significant half

s_add_u32 m0, m0, 4096 // (op 17911) 
buffer_load_dwordx4 v48, s[72:75], 0 offen offset: 0 lds // (op 17911) Load value direct to lds






v_mov_b64 v[32:33], s[44:45] // (op 17584) convert

s_lshl_b32 s52, s44, 2 // (op 17584) 
v_add_u32 v26, v39, s52 // (op 17584) 

buffer_load_ubyte v42, v26, s[80:83], 0 offen offset: 0 // (op 17584) Load value
v_add_u32 v26, v26, v32 // (op 17584) 
buffer_load_ubyte v43, v26, s[80:83], 0 offen offset: 0 // (op 17584) Load value



v_and_b32 v26, -4, v1 // (op 19397) 
v_sub_u32 v41, v1, v26 // (op 19397) 

v_lshlrev_b32 v26, 5, v41 // (op 19397) 

v_lshrrev_b32 v41, 2, v1 // (op 19397) 
v_lshl_add_u32 v32, v41, 7, v26 // (op 19397) 


v_lshlrev_b32 v26, 2, v32 // (op 19397) 

v_lshrrev_b32 v41, 3, v26 // (op 19397) 

s_nop 0
v_readfirstlane_b32 s52, v41 // (op 19397) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19940) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 17913) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17913) 

v_lshrrev_b64 v[32:33], 3, s[84:85] // (op 17913) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 17913) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17913) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 17913) 

v_mov_b32 v26, s54 // (op 17913) 
v_mov_b32 v41, s55 // (op 17913) 
v_add_co_u32 v38, s[84:85], v28, v26 // (op 17913) least significant half
v_addc_co_u32 v39, s[84:85], v29, v41, s[84:85] // (op 17913) most significant half




s_add_u32 s53, 40960, s52 // (op 17913) 
s_mov_b32 m0, s53 // (op 17913) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17913) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v32 // (op 17913) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 17913) most significant half

s_add_u32 m0, m0, 4096 // (op 17913) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17913) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 17590) convert

s_lshl_b32 s52, s38, 2 // (op 17590) 
v_add_u32 v26, v36, s52 // (op 17590) 

buffer_load_ubyte v32, v26, s[76:79], 0 offen offset: 0 // (op 17590) Load value
v_add_u32 v26, v26, v28 // (op 17590) 
buffer_load_ubyte v33, v26, s[76:79], 0 offen offset: 0 // (op 17590) Load value



v_and_b32 v26, -128, v1 // (op 18783) 
v_sub_u32 v36, v1, v26 // (op 18783) 

v_lshrrev_b32 v26, 7, v1 // (op 18783) 
v_lshl_add_u32 v41, v26, 8, v36 // (op 18783) 


s_barrier  // (op 19810) 



s_waitcnt vmcnt(15)
ds_write_b8 v41, v44 offset:49152 // (op 17593) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v41, v45 offset:49280 // (op 17593) Store local data 

v_and_b32 v26, -128, v1 // (op 18707) 
v_sub_u32 v36, v1, v26 // (op 18707) 

v_lshrrev_b32 v26, 7, v1 // (op 18707) 
v_lshl_add_u32 v41, v26, 8, v36 // (op 18707) 


s_barrier  // (op 19790) 



s_waitcnt vmcnt(11)
ds_write_b8 v41, v46 offset:49664 // (op 17595) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v41, v47 offset:49792 // (op 17595) Store local data 

s_barrier  // (op 19767) 



ds_read_b64_tr_b8 v[48:49], v23 // (op 17630) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v23 offset:1024 // (op 17630) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v23 offset:8192 // (op 17630) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v23 offset:9216 // (op 17630) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19807) 


ds_read_u8 v26, v24 offset:49152 // (op 17670) Load local data 



ds_read_b64_tr_b8 v[56:57], v23 offset:32 // (op 17635) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v23 offset:1056 // (op 17635) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v23 offset:8224 // (op 17635) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v23 offset:9248 // (op 17635) Transpose load local data 


ds_read_u8 v36, v24 offset:49184 // (op 17675) Load local data 



ds_read_b64_tr_b8 v[64:65], v23 offset:64 // (op 17640) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v23 offset:1088 // (op 17640) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v23 offset:8256 // (op 17640) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v23 offset:9280 // (op 17640) Transpose load local data 


ds_read_u8 v41, v24 offset:49216 // (op 17680) Load local data 



ds_read_b64_tr_b8 v[72:73], v23 offset:96 // (op 17645) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v23 offset:1120 // (op 17645) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v23 offset:8288 // (op 17645) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v23 offset:9312 // (op 17645) Transpose load local data 


ds_read_u8 v28, v24 offset:49248 // (op 17685) Load local data 
s_barrier  // (op 19748) 



ds_read_b64_tr_b4 v[80:81], v21 offset:16384 // (op 17610) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v21 offset:17408 // (op 17610) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19787) 


ds_read_u8 v29, v20 offset:49664 // (op 17650) Load local data 



ds_read_b64_tr_b4 v[84:85], v21 offset:16400 // (op 17615) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v21 offset:17424 // (op 17615) Transpose load local data 


ds_read_u8 v38, v20 offset:49696 // (op 17655) Load local data 



ds_read_b64_tr_b4 v[88:89], v21 offset:16416 // (op 17620) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v21 offset:17440 // (op 17620) Transpose load local data 


ds_read_u8 v39, v20 offset:49728 // (op 17660) Load local data 



ds_read_b64_tr_b4 v[92:93], v21 offset:16432 // (op 17625) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v21 offset:17456 // (op 17625) Transpose load local data 


ds_read_u8 v96, v20 offset:49760 // (op 17665) Load local data 
v_and_b32 v97, -4, v1 // (op 18528) 
v_sub_u32 v98, v1, v97 // (op 18528) 

v_lshlrev_b32 v97, 5, v98 // (op 18528) 

v_lshrrev_b32 v98, 2, v1 // (op 18528) 
v_lshl_add_u32 v99, v98, 7, v97 // (op 18528) 


v_lshlrev_b32 v97, 2, v99 // (op 18528) 

v_lshrrev_b32 v98, 3, v97 // (op 18528) 

s_nop 0
v_readfirstlane_b32 s52, v98 // (op 18528) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 18603) 
v_sub_u32 v98, v1, v97 // (op 18603) 

v_lshlrev_b32 v97, 4, v98 // (op 18603) 

v_lshrrev_b32 v98, 3, v1 // (op 18603) 
v_lshl_add_u32 v99, v98, 7, v97 // (op 18603) 


s_nop 0
v_readfirstlane_b32 s53, v99 // (op 18603) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -128, v1 // (op 18678) 
v_sub_u32 v98, v1, v97 // (op 18678) 

v_lshrrev_b32 v97, 7, v1 // (op 18678) 
v_lshl_add_u32 v99, v97, 8, v98 // (op 18678) 


v_and_b32 v97, -128, v1 // (op 18754) 
v_sub_u32 v98, v1, v97 // (op 18754) 

v_lshrrev_b32 v97, 7, v1 // (op 18754) 
v_lshl_add_u32 v100, v97, 8, v98 // (op 18754) 


v_and_b32 v97, -4, v1 // (op 19382) 
v_sub_u32 v98, v1, v97 // (op 19382) 

v_lshlrev_b32 v97, 5, v98 // (op 19382) 

v_lshrrev_b32 v98, 2, v1 // (op 19382) 
v_lshl_add_u32 v101, v98, 7, v97 // (op 19382) 


v_lshlrev_b32 v97, 2, v101 // (op 19382) 

v_lshrrev_b32 v98, 3, v97 // (op 19382) 

s_nop 0
v_readfirstlane_b32 s54, v98 // (op 19382) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 19426) 
v_sub_u32 v98, v1, v97 // (op 19426) 

v_lshlrev_b32 v97, 4, v98 // (op 19426) 

v_lshrrev_b32 v98, 3, v1 // (op 19426) 
v_lshl_add_u32 v101, v98, 7, v97 // (op 19426) 


s_nop 0
v_readfirstlane_b32 s55, v101 // (op 19426) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -128, v1 // (op 19470) 
v_sub_u32 v98, v1, v97 // (op 19470) 

v_lshrrev_b32 v97, 7, v1 // (op 19470) 
v_lshl_add_u32 v101, v97, 8, v98 // (op 19470) 


v_and_b32 v97, -128, v1 // (op 19499) 
v_sub_u32 v98, v1, v97 // (op 19499) 

v_lshrrev_b32 v97, 7, v1 // (op 19499) 
v_lshl_add_u32 v102, v97, 8, v98 // (op 19499) 


s_mov_b32 s1, 0 // (op 7053) call()
s_and_b32 s84, s7, 4294967294 // (op 7052) 
s_cmp_lt_u32 s1, s84 // (op 7052) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_7052 // (op 7052) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_7052 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_7052:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[48:55], a[0:3], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 90) 
s_barrier  // (op 19773) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[56:63], a[4:7], v29, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4842) 
s_add_u32 s1, s1, 2 // (op 7054) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:83], v[64:71], a[8:11], v29, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5018) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 17826) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:83], v[72:79], a[12:15], v29, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5194) 
v_mov_b64 v[104:105], s[84:85] // (op 17826) convert


v_mov_b64 v[106:107], v[34:35] // (op 17826) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 17826) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 17826) 

v_mov_b32 v103, s86 // (op 17826) 
v_mov_b32 v97, s87 // (op 17826) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[48:55], a[16:19], v38, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5689) 
v_add_co_u32 v108, s[84:85], v106, v103 // (op 17826) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[56:63], a[20:23], v38, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5643) 
v_addc_co_u32 v109, s[84:85], v107, v97, s[84:85] // (op 17826) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[64:71], a[24:27], v38, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5597) 
s_mov_b32 m0, s53 // (op 17826) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[72:79], a[28:31], v38, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5551) 
buffer_load_dwordx4 v108, s[72:75], 0 offen offset: 0 lds // (op 17826) Load value direct to lds
v_add_co_u32 v108, s[84:85], v108, v104 // (op 17826) least significant half
v_addc_co_u32 v109, s[84:85], 0, v109, s[84:85] // (op 17826) most significant half

s_add_u32 m0, m0, 4096 // (op 17826) 
buffer_load_dwordx4 v108, s[72:75], 0 offen offset: 0 lds // (op 17826) Load value direct to lds
v_add_co_u32 v108, s[84:85], v108, v104 // (op 17826) least significant half
v_addc_co_u32 v109, s[84:85], 0, v109, s[84:85] // (op 17826) most significant half

s_add_u32 m0, m0, 4096 // (op 17826) 
buffer_load_dwordx4 v108, s[72:75], 0 offen offset: 0 lds // (op 17826) Load value direct to lds
v_add_co_u32 v108, s[84:85], v108, v104 // (op 17826) least significant half
v_addc_co_u32 v109, s[84:85], 0, v109, s[84:85] // (op 17826) most significant half

s_add_u32 m0, m0, 4096 // (op 17826) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[48:55], a[32:35], v39, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6459) 
buffer_load_dwordx4 v108, s[72:75], 0 offen offset: 0 lds // (op 17826) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[56:63], a[36:39], v39, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6413) 
v_mov_b64 v[104:105], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[64:71], a[40:43], v39, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6367) 
v_mov_b32 v103, v40 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[72:79], a[44:47], v39, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6321) 
s_lshl_b32 s84, s44, 3 // (op 25) 
v_add_u32 v97, v103, s84 // (op 25) 

buffer_load_ubyte v44, v97, s[80:83], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v97, v97, v104 // (op 25) 
buffer_load_ubyte v45, v97, s[80:83], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[48:55], a[48:51], v96, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7229) 
s_barrier  // (op 19754) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[56:63], a[52:55], v96, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7183) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[64:71], a[56:59], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7137) 
s_lshl_b64 s[84:85], s[30:31], 6 // (op 17821) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[72:79], a[60:63], v96, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7091) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17821) 

v_lshrrev_b64 v[104:105], 3, s[86:87] // (op 17821) 


v_mov_b64 v[108:109], v[30:31] // (op 17821) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 17821) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17821) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17821) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 17821) 

v_mov_b32 v97, s86 // (op 17821) 
v_mov_b32 v98, s87 // (op 17821) 
v_add_co_u32 v110, s[84:85], v108, v97 // (op 17821) least significant half
v_addc_co_u32 v111, s[84:85], v109, v98, s[84:85] // (op 17821) most significant half




s_add_u32 s84, 16384, s52 // (op 17821) 
s_nop 0
s_mov_b32 m0, s84 // (op 17821) reg expression
buffer_load_dwordx4 v110, s[68:71], 0 offen offset: 0 lds // (op 17821) Load value direct to lds
v_add_co_u32 v110, s[86:87], v110, v104 // (op 17821) least significant half
v_addc_co_u32 v111, s[86:87], 0, v111, s[86:87] // (op 17821) most significant half

s_add_u32 m0, m0, 4096 // (op 17821) 
buffer_load_dwordx4 v110, s[68:71], 0 offen offset: 0 lds // (op 17821) Load value direct to lds




v_mov_b64 v[104:105], s[38:39] // (op 17) convert

v_mov_b32 v97, v37 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v98, v97, s84 // (op 17) 

buffer_load_ubyte v46, v98, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v98, v98, v104 // (op 17) 
buffer_load_ubyte v47, v98, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 19965) 



s_waitcnt vmcnt(15)
ds_write_b8 v102, v42 offset:50176 // (op 4721) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v102, v43 offset:50304 // (op 4721) Store local data 
s_barrier  // (op 19959) 



s_waitcnt vmcnt(11)
ds_write_b8 v101, v32 offset:50688 // (op 4720) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v101, v33 offset:50816 // (op 4720) Store local data 
s_barrier  // (op 19946) 



ds_read_b64_tr_b8 v[110:111], v16 offset:24576 // (op 4713) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v16 offset:25600 // (op 4713) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v16 offset:32768 // (op 4713) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v16 offset:33792 // (op 4713) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19962) 


ds_read_u8 v98, v17 offset:50176 // (op 4717) Load local data 



ds_read_b64_tr_b8 v[118:119], v16 offset:24608 // (op 4847) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v16 offset:25632 // (op 4847) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v16 offset:32800 // (op 4847) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v16 offset:33824 // (op 4847) Transpose load local data 


ds_read_u8 v38, v17 offset:50208 // (op 4851) Load local data 



ds_read_b64_tr_b8 v[126:127], v16 offset:24640 // (op 5023) Transpose load local data 
ds_read_b64_tr_b8 v[128:129], v16 offset:25664 // (op 5023) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v16 offset:32832 // (op 5023) Transpose load local data 
ds_read_b64_tr_b8 v[132:133], v16 offset:33856 // (op 5023) Transpose load local data 


ds_read_u8 v39, v17 offset:50240 // (op 5027) Load local data 



ds_read_b64_tr_b8 v[134:135], v16 offset:24672 // (op 5199) Transpose load local data 
ds_read_b64_tr_b8 v[136:137], v16 offset:25696 // (op 5199) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v16 offset:32864 // (op 5199) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v16 offset:33888 // (op 5199) Transpose load local data 


ds_read_u8 v104, v17 offset:50272 // (op 5203) Load local data 
s_barrier  // (op 19937) 
s_barrier  // (op 20308) 



ds_read_b64_tr_b4 v[80:81], v15 offset:40960 // (op 4711) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v15 offset:41984 // (op 4711) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19956) 


ds_read_u8 v29, v14 offset:50688 // (op 4715) Load local data 



ds_read_b64_tr_b4 v[84:85], v15 offset:40976 // (op 5554) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v15 offset:42000 // (op 5554) Transpose load local data 


ds_read_u8 v36, v14 offset:50720 // (op 5557) Load local data 



ds_read_b64_tr_b4 v[88:89], v15 offset:40992 // (op 6324) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v15 offset:42016 // (op 6324) Transpose load local data 


ds_read_u8 v41, v14 offset:50752 // (op 6327) Load local data 



ds_read_b64_tr_b4 v[92:93], v15 offset:41008 // (op 7094) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v15 offset:42032 // (op 7094) Transpose load local data 


ds_read_u8 v96, v14 offset:50784 // (op 7097) Load local data 
s_barrier  // (op 20342) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[110:117], a[0:3], v29, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4718) 
s_barrier  // (op 19953) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[118:125], a[4:7], v29, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4843) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 17882) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:83], v[126:133], a[8:11], v29, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5019) 
v_mov_b64 v[142:143], s[84:85] // (op 17882) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:83], v[134:141], a[12:15], v29, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5195) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 17882) 
s_mul_i32 s87, 3, s85 // (op 17882) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 17882) least significant
s_mul_hi_u32 s88, 3, s84 // (op 17882) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 17882) most significant: sum


v_mov_b32 v105, s86 // (op 17882) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[110:117], a[16:19], v36, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5690) 
v_mov_b32 v144, s87 // (op 17882) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[118:125], a[20:23], v36, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5644) 
v_add_co_u32 v146, s[84:85], v106, v105 // (op 17882) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[126:133], a[24:27], v36, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5598) 
v_addc_co_u32 v147, s[84:85], v107, v144, s[84:85] // (op 17882) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[134:141], a[28:31], v36, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5552) 
s_add_u32 s84, 24576, s55 // (op 17882) 
s_mov_b32 m0, s84 // (op 17882) reg expression
buffer_load_dwordx4 v146, s[72:75], 0 offen offset: 0 lds // (op 17882) Load value direct to lds
v_add_co_u32 v146, s[86:87], v146, v142 // (op 17882) least significant half
v_addc_co_u32 v147, s[86:87], 0, v147, s[86:87] // (op 17882) most significant half

s_add_u32 m0, m0, 4096 // (op 17882) 
buffer_load_dwordx4 v146, s[72:75], 0 offen offset: 0 lds // (op 17882) Load value direct to lds
v_add_co_u32 v146, s[86:87], v146, v142 // (op 17882) least significant half
v_addc_co_u32 v147, s[86:87], 0, v147, s[86:87] // (op 17882) most significant half

s_add_u32 m0, m0, 4096 // (op 17882) 
buffer_load_dwordx4 v146, s[72:75], 0 offen offset: 0 lds // (op 17882) Load value direct to lds
v_add_co_u32 v146, s[86:87], v146, v142 // (op 17882) least significant half
v_addc_co_u32 v147, s[86:87], 0, v147, s[86:87] // (op 17882) most significant half

s_add_u32 m0, m0, 4096 // (op 17882) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[110:117], a[32:35], v41, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6460) 
buffer_load_dwordx4 v146, s[72:75], 0 offen offset: 0 lds // (op 17882) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[118:125], a[36:39], v41, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6414) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[126:133], a[40:43], v41, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6368) 
v_mov_b64 v[106:107], s[44:45] // (op 4724) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[134:141], a[44:47], v41, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6322) 
s_lshl_b32 s84, s44, 2 // (op 4724) 
s_mul_i32 s85, 3, s84 // (op 4724) 

v_add_u32 v105, v103, s85 // (op 4724) 

buffer_load_ubyte v42, v105, s[80:83], 0 offen offset: 0 // (op 4724) Load value
v_add_u32 v105, v105, v106 // (op 4724) 
buffer_load_ubyte v43, v105, s[80:83], 0 offen offset: 0 // (op 4724) Load value


s_lshl_b64 s[84:85], s[34:35], 7 // (op 18190) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 18190) 

s_lshl_b32 s84, s44, 3 // (op 18446) 
v_add_u32 v40, v40, s84 // (op 18446) 

v_mov_b32 v103, s86 // (op 18190) 
v_mov_b32 v105, s87 // (op 18190) 
v_add_co_u32 v34, s[84:85], v34, v103 // (op 18190) least significant half
v_addc_co_u32 v35, s[84:85], v35, v105, s[84:85] // (op 18190) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[110:117], a[48:51], v96, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7230) 
s_barrier  // (op 19943) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[118:125], a[52:55], v96, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7184) 
s_lshl_b64 s[84:85], s[30:31], 6 // (op 17877) 

s_lshl_b64 s[86:87], s[84:85], 2 // (op 17877) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[126:133], a[56:59], v96, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7138) 
v_lshrrev_b64 v[106:107], 3, s[86:87] // (op 17877) 



s_lshl_b64 s[84:85], s[30:31], 7 // (op 17877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[134:141], a[60:63], v96, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7092) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17877) 



s_lshr_b64 s[84:85], s[86:87], 3 // (op 17877) 

s_mul_i32 s87, 3, s85 // (op 17877) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 17877) least significant
s_mul_hi_u32 s88, 3, s84 // (op 17877) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 17877) most significant: sum


v_mov_b32 v98, s86 // (op 17877) 
v_mov_b32 v103, s87 // (op 17877) 
v_add_co_u32 v104, s[84:85], v108, v98 // (op 17877) least significant half
v_addc_co_u32 v105, s[84:85], v109, v103, s[84:85] // (op 17877) most significant half




s_add_u32 s84, 40960, s54 // (op 17877) 
s_nop 0
s_mov_b32 m0, s84 // (op 17877) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 17877) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v106 // (op 17877) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 17877) most significant half

s_add_u32 m0, m0, 4096 // (op 17877) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 17877) Load value direct to lds





v_mov_b64 v[104:105], s[38:39] // (op 4723) convert

s_lshl_b32 s84, s38, 2 // (op 4723) 
s_mul_i32 s85, 3, s84 // (op 4723) 

v_add_u32 v98, v97, s85 // (op 4723) 

buffer_load_ubyte v32, v98, s[76:79], 0 offen offset: 0 // (op 4723) Load value
v_add_u32 v98, v98, v104 // (op 4723) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 18062) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 18062) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 18062) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 18062) 

v_mov_b32 v103, s86 // (op 18062) 
v_mov_b32 v106, s87 // (op 18062) 
v_add_co_u32 v30, s[84:85], v30, v103 // (op 18062) least significant half
v_addc_co_u32 v31, s[84:85], v31, v106, s[84:85] // (op 18062) most significant half




buffer_load_ubyte v33, v98, s[76:79], 0 offen offset: 0 // (op 4723) Load value



s_barrier  // (op 19813) 
s_lshl_b32 s84, s38, 3 // (op 18318) 



v_add_u32 v37, v37, s84 // (op 18318) 

s_waitcnt vmcnt(15)
ds_write_b8 v100, v44 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v100, v45 offset:49280 // (op 30) Store local data 
s_barrier  // (op 19793) 



s_waitcnt vmcnt(11)
ds_write_b8 v99, v46 offset:49664 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v99, v47 offset:49792 // (op 21) Store local data 
s_barrier  // (op 19779) 



ds_read_b64_tr_b8 v[48:49], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:8192 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:9216 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19820) 


ds_read_u8 v26, v5 offset:49152 // (op 87) Load local data 



ds_read_b64_tr_b8 v[56:57], v4 offset:32 // (op 4837) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:1056 // (op 4837) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:8224 // (op 4837) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:9248 // (op 4837) Transpose load local data 


ds_read_u8 v36, v5 offset:49184 // (op 4841) Load local data 



ds_read_b64_tr_b8 v[64:65], v4 offset:64 // (op 5013) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v4 offset:1088 // (op 5013) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:8256 // (op 5013) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v4 offset:9280 // (op 5013) Transpose load local data 


ds_read_u8 v41, v5 offset:49216 // (op 5017) Load local data 



ds_read_b64_tr_b8 v[72:73], v4 offset:96 // (op 5189) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v4 offset:1120 // (op 5189) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v4 offset:8288 // (op 5189) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v4 offset:9312 // (op 5189) Transpose load local data 


ds_read_u8 v28, v5 offset:49248 // (op 5193) Load local data 
s_barrier  // (op 19760) 



ds_read_b64_tr_b4 v[80:81], v3 offset:16384 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:17408 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19800) 


ds_read_u8 v29, v0 offset:49664 // (op 82) Load local data 



ds_read_b64_tr_b4 v[84:85], v3 offset:16400 // (op 5546) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:17424 // (op 5546) Transpose load local data 


ds_read_u8 v38, v0 offset:49696 // (op 5549) Load local data 



ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 6316) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:17440 // (op 6316) Transpose load local data 


ds_read_u8 v39, v0 offset:49728 // (op 6319) Load local data 



ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 7086) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:17456 // (op 7086) Transpose load local data 


ds_read_u8 v96, v0 offset:49760 // (op 7089) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 7052) 
s_cmp_lt_u32 s1, s84 // (op 7052) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_7052 // (op 7052) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_7052 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_7052:




















s_barrier  // (op 20326) 
s_and_b32 s52, s7, 4294967294 // (op 17950) 
v_and_b32 v26, -4, v1 // (op 17950) 
s_add_u32 s53, s0, s52 // (op 17950) 

v_sub_u32 v30, v1, v26 // (op 17950) 

v_lshlrev_b32 v26, 5, v30 // (op 17950) 

s_mul_i32 s52, s16, s8 // (op 17950) 
s_add_u32 s52, s52, s53 // (op 17950) 

v_lshrrev_b32 v30, 2, v1 // (op 17950) 
s_mul_hi_u32 s53, s52, s13 // (op 17950) 
s_sub_u32 s54, s52, s53 // (op 17950) 
s_lshr_b32 s55, s54, 1 // (op 17950) 

s_add_u32 s54, s55, s53 // (op 17950) 


s_lshr_b32 s53, s54, s11 // (op 17950) 

s_and_b32 s54, s11, 2147483648 // (op 17950) 
s_cmp_eq_i32 s54, 0 // (op 17950) 
s_mov_b32 s55, scc // (op 17950) 

s_cmp_eq_i32 s55, 0 // (op 17950) 
s_cselect_b32 s54, s52, s53 // (op 17950) 


s_mul_hi_u32 s53, s54, s12 // (op 17950) 
s_sub_u32 s55, s54, s53 // (op 17950) 
s_lshr_b32 s84, s55, 1 // (op 17950) 

s_add_u32 s55, s84, s53 // (op 17950) 


s_lshr_b32 s53, s55, s10 // (op 17950) 

s_and_b32 s55, s10, 2147483648 // (op 17950) 
s_cmp_eq_i32 s55, 0 // (op 17950) 
s_mov_b32 s84, scc // (op 17950) 

s_cmp_eq_i32 s84, 0 // (op 17950) 
s_cselect_b32 s55, s54, s53 // (op 17950) 


v_lshl_add_u32 v31, s55, 7, v26 // (op 17950) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 17950) 
v_mov_b32 v41, 0 // (op 17950) convert
v_mov_b32 v40, v31 // (op 17950) convert

s_mul_i32 s53, s54, s84 // (op 17950) 


s_sub_u32 s54, s52, s53 // (op 17950) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 17950) 


v_mul_lo_u32 v31, v26, s31 // (op 17950) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 17950) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 17950) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 17950) most significant: sum


v_add_co_u32 v34, s[52:53], v40, v30 // (op 17950) least significant half
v_addc_co_u32 v35, s[52:53], v41, v31, s[52:53] // (op 17950) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 17950) 


v_lshrrev_b64 v[30:31], 3, v[34:35] // (op 17950) 

s_mul_i32 s53, s30, s3 // (op 17952) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 17952) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 17952) least significant
s_mul_hi_u32 s55, s30, s2 // (op 17952) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 17952) most significant: sum
s_add_u32 s53, s53, s55 // (op 17952) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 17952) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 17952) 

s_mov_b32 s68, s32 // (op 17952) reg expression
s_mov_b32 s69, s33 // (op 17952) reg expression
s_mov_b32 s70, s52 // (op 17952) reg expression
s_mov_b32 s71, 131072 // (op 17952) call()

s_and_b32 s52, s7, 4294967294 // (op 18114) 
v_and_b32 v26, -8, v1 // (op 18114) 
s_add_u32 s53, s0, s52 // (op 18114) 

v_sub_u32 v34, v1, v26 // (op 18114) 

v_lshlrev_b32 v26, 4, v34 // (op 18114) 

s_mul_i32 s52, s16, s8 // (op 18114) 
s_add_u32 s52, s52, s53 // (op 18114) 

v_lshrrev_b32 v34, 3, v1 // (op 18114) 
s_mul_hi_u32 s53, s52, s13 // (op 18114) 
s_sub_u32 s54, s52, s53 // (op 18114) 
s_lshr_b32 s55, s54, 1 // (op 18114) 

s_add_u32 s54, s55, s53 // (op 18114) 


s_lshr_b32 s53, s54, s11 // (op 18114) 

s_and_b32 s54, s11, 2147483648 // (op 18114) 
s_cmp_eq_i32 s54, 0 // (op 18114) 
s_mov_b32 s55, scc // (op 18114) 

s_cmp_eq_i32 s55, 0 // (op 18114) 
s_cselect_b32 s54, s52, s53 // (op 18114) 


s_mul_hi_u32 s53, s54, s12 // (op 18114) 
s_sub_u32 s55, s54, s53 // (op 18114) 
s_lshr_b32 s84, s55, 1 // (op 18114) 

s_add_u32 s55, s84, s53 // (op 18114) 


s_lshr_b32 s53, s55, s10 // (op 18114) 

s_and_b32 s55, s10, 2147483648 // (op 18114) 
s_cmp_eq_i32 s55, 0 // (op 18114) 
s_mov_b32 s84, scc // (op 18114) 

s_cmp_eq_i32 s84, 0 // (op 18114) 
s_cselect_b32 s55, s54, s53 // (op 18114) 


s_mul_i32 s53, s55, s14 // (op 18114) 

s_sub_u32 s55, s54, s53 // (op 18114) 

v_lshl_add_u32 v35, s55, 7, v26 // (op 18114) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 18114) 
v_mov_b32 v37, 0 // (op 18114) convert
v_mov_b32 v36, v35 // (op 18114) convert

s_mul_i32 s53, s54, s84 // (op 18114) 


s_sub_u32 s54, s52, s53 // (op 18114) 


v_lshl_add_u32 v26, s54, 7, v34 // (op 18114) 


v_mul_lo_u32 v35, v26, s35 // (op 18114) most significant: low of low * high
v_mul_hi_u32 v40, v26, s34 // (op 18114) most significant: high of low * low
v_mul_lo_u32 v34, v26, s34 // (op 18114) least significant: low of low * low
v_add_u32 v35, v35, v40 // (op 18114) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v36 // (op 18114) least significant half
v_addc_co_u32 v35, s[52:53], v35, v37, s[52:53] // (op 18114) most significant half



s_mul_i32 s53, s34, s3 // (op 18116) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 18116) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 18116) least significant
s_mul_hi_u32 s55, s34, s2 // (op 18116) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18116) most significant: sum
s_add_u32 s53, s53, s55 // (op 18116) most significant: sum


s_mov_b32 s72, s36 // (op 18116) reg expression
s_mov_b32 s73, s37 // (op 18116) reg expression
s_mov_b32 s74, s52 // (op 18116) reg expression
s_mov_b32 s75, 131072 // (op 18116) call()

s_and_b32 s52, s7, 4294967294 // (op 18242) 
v_and_b32 v26, -128, v1 // (op 18242) 
s_add_u32 s53, s0, s52 // (op 18242) 

v_sub_u32 v36, v1, v26 // (op 18242) 

v_lshrrev_b32 v26, 7, v1 // (op 18242) 
s_mul_i32 s52, s16, s8 // (op 18242) 
s_add_u32 s52, s52, s53 // (op 18242) 

v_lshlrev_b32 v37, 1, v26 // (op 18242) 

s_mul_hi_u32 s53, s52, s13 // (op 18242) 
s_sub_u32 s54, s52, s53 // (op 18242) 
s_lshr_b32 s55, s54, 1 // (op 18242) 

s_add_u32 s54, s55, s53 // (op 18242) 


s_lshr_b32 s53, s54, s11 // (op 18242) 

s_and_b32 s54, s11, 2147483648 // (op 18242) 
s_cmp_eq_i32 s54, 0 // (op 18242) 
s_mov_b32 s55, scc // (op 18242) 

s_cmp_eq_i32 s55, 0 // (op 18242) 
s_cselect_b32 s54, s52, s53 // (op 18242) 


s_mul_hi_u32 s53, s54, s12 // (op 18242) 
s_sub_u32 s55, s54, s53 // (op 18242) 
s_lshr_b32 s84, s55, 1 // (op 18242) 

s_add_u32 s55, s84, s53 // (op 18242) 


s_lshr_b32 s53, s55, s10 // (op 18242) 

s_and_b32 s55, s10, 2147483648 // (op 18242) 
s_cmp_eq_i32 s55, 0 // (op 18242) 
s_mov_b32 s84, scc // (op 18242) 

s_cmp_eq_i32 s84, 0 // (op 18242) 
s_cselect_b32 s55, s54, s53 // (op 18242) 


v_lshl_add_u32 v26, s55, 7, v36 // (op 18242) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 18242) 
s_mul_i32 s53, s54, s84 // (op 18242) 


s_sub_u32 s54, s52, s53 // (op 18242) 


v_lshl_add_u32 v36, s54, 2, v37 // (op 18242) 


v_mul_lo_u32 v37, v36, s38 // (op 18242) 

v_add_u32 v36, v26, v37 // (op 18242) 


s_mul_i32 s53, s38, s43 // (op 18244) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18244) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18244) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18244) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18244) most significant: sum
s_add_u32 s53, s53, s55 // (op 18244) most significant: sum


s_mov_b32 s76, s40 // (op 18244) reg expression
s_mov_b32 s77, s41 // (op 18244) reg expression
s_mov_b32 s78, s52 // (op 18244) reg expression
s_mov_b32 s79, 131072 // (op 18244) call()

s_and_b32 s52, s7, 4294967294 // (op 18370) 
v_and_b32 v26, -128, v1 // (op 18370) 
s_add_u32 s53, s0, s52 // (op 18370) 

v_sub_u32 v37, v1, v26 // (op 18370) 

v_lshrrev_b32 v26, 7, v1 // (op 18370) 
s_mul_i32 s52, s16, s8 // (op 18370) 
s_add_u32 s52, s52, s53 // (op 18370) 

v_lshlrev_b32 v40, 1, v26 // (op 18370) 

s_mul_hi_u32 s53, s52, s13 // (op 18370) 
s_sub_u32 s54, s52, s53 // (op 18370) 
s_lshr_b32 s55, s54, 1 // (op 18370) 

s_add_u32 s54, s55, s53 // (op 18370) 


s_lshr_b32 s53, s54, s11 // (op 18370) 

s_and_b32 s54, s11, 2147483648 // (op 18370) 
s_cmp_eq_i32 s54, 0 // (op 18370) 
s_mov_b32 s55, scc // (op 18370) 

s_cmp_eq_i32 s55, 0 // (op 18370) 
s_cselect_b32 s54, s52, s53 // (op 18370) 


s_mul_hi_u32 s53, s54, s12 // (op 18370) 
s_sub_u32 s55, s54, s53 // (op 18370) 
s_lshr_b32 s84, s55, 1 // (op 18370) 

s_add_u32 s55, s84, s53 // (op 18370) 


s_lshr_b32 s53, s55, s10 // (op 18370) 

s_and_b32 s55, s10, 2147483648 // (op 18370) 
s_cmp_eq_i32 s55, 0 // (op 18370) 
s_mov_b32 s84, scc // (op 18370) 

s_cmp_eq_i32 s84, 0 // (op 18370) 
s_cselect_b32 s55, s54, s53 // (op 18370) 


s_mul_i32 s53, s55, s14 // (op 18370) 

s_sub_u32 s55, s54, s53 // (op 18370) 

v_lshl_add_u32 v26, s55, 7, v37 // (op 18370) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 18370) 
s_mul_i32 s53, s54, s84 // (op 18370) 


s_sub_u32 s54, s52, s53 // (op 18370) 


v_lshl_add_u32 v37, s54, 2, v40 // (op 18370) 


v_mul_lo_u32 v40, v37, s44 // (op 18370) 
v_add_u32 v40, v40, v26 // (op 18370) 


s_mul_i32 s53, s44, s43 // (op 18372) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 18372) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 18372) least significant
s_mul_hi_u32 s55, s44, s42 // (op 18372) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18372) most significant: sum
s_add_u32 s53, s53, s55 // (op 18372) most significant: sum


s_mov_b32 s80, s46 // (op 18372) reg expression
s_mov_b32 s81, s47 // (op 18372) reg expression
s_mov_b32 s82, s52 // (op 18372) reg expression
s_mov_b32 s83, 131072 // (op 18372) call()

v_and_b32 v26, -4, v1 // (op 18541) 
v_sub_u32 v37, v1, v26 // (op 18541) 

v_lshlrev_b32 v26, 5, v37 // (op 18541) 

v_lshrrev_b32 v37, 2, v1 // (op 18541) 
v_lshl_add_u32 v41, v37, 7, v26 // (op 18541) 


v_lshlrev_b32 v26, 2, v41 // (op 18541) 

v_lshrrev_b32 v37, 3, v26 // (op 18541) 

s_nop 0
v_readfirstlane_b32 s52, v37 // (op 18541) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 18616) 
v_sub_u32 v37, v1, v26 // (op 18616) 

v_lshlrev_b32 v26, 4, v37 // (op 18616) 

v_lshrrev_b32 v37, 3, v1 // (op 18616) 
v_lshl_add_u32 v41, v37, 7, v26 // (op 18616) 


s_nop 0
v_readfirstlane_b32 s53, v41 // (op 18616) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -128, v1 // (op 18691) 
v_sub_u32 v37, v1, v26 // (op 18691) 

v_lshrrev_b32 v26, 7, v1 // (op 18691) 
v_lshl_add_u32 v41, v26, 8, v37 // (op 18691) 


v_and_b32 v26, -128, v1 // (op 18767) 
v_sub_u32 v37, v1, v26 // (op 18767) 

v_lshrrev_b32 v26, 7, v1 // (op 18767) 
v_lshl_add_u32 v44, v26, 8, v37 // (op 18767) 


s_and_b32 s54, s7, 4294967294 // (op 7058) 
s_cmp_lt_u32 s54, s7 // (op 7057) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_7057 // (op 7057) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_7057 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_7057:
s_barrier  // (op 19757) 
s_add_u32 s54, s54, 1 // (op 7059) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 17864) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17864) 

v_lshrrev_b64 v[28:29], 3, s[86:87] // (op 17864) 


v_mov_b64 v[32:33], v[30:31] // (op 17864) 
v_mov_b64 v[38:39], v[32:33] // (op 17864) reg expression
s_add_u32 s55, 16384, s52 // (op 17864) 
s_mov_b32 m0, s55 // (op 17864) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17864) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 17864) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 17864) most significant half

s_add_u32 m0, m0, 4096 // (op 17864) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17864) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 17940) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17940) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17940) 

v_mov_b32 v26, s84 // (op 17940) 
v_mov_b32 v28, s85 // (op 17940) 
v_add_co_u32 v30, s[86:87], v30, v26 // (op 17940) least significant half
v_addc_co_u32 v31, s[86:87], v31, v28, s[86:87] // (op 17940) most significant half




s_barrier  // (op 19776) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 17870) 
v_mov_b64 v[28:29], s[84:85] // (op 17870) convert


v_mov_b64 v[32:33], v[34:35] // (op 17870) 
v_mov_b64 v[38:39], v[32:33] // (op 17870) reg expression
s_mov_b32 m0, s53 // (op 17870) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17870) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 17870) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 17870) most significant half

s_add_u32 m0, m0, 4096 // (op 17870) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17870) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 17870) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 17870) most significant half

s_add_u32 m0, m0, 4096 // (op 17870) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17870) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 17870) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 17870) most significant half

s_add_u32 m0, m0, 4096 // (op 17870) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17870) Load value direct to lds



s_lshl_b64 s[84:85], s[34:35], 7 // (op 18106) 
v_mov_b32 v26, s84 // (op 18106) 
v_mov_b32 v28, s85 // (op 18106) 
v_add_co_u32 v34, s[86:87], v34, v26 // (op 18106) least significant half
v_addc_co_u32 v35, s[86:87], v35, v28, s[86:87] // (op 18106) most significant half





v_mov_b64 v[28:29], s[38:39] // (op 4657) convert

v_mov_b32 v26, v36 // (op 4657) 
v_mov_b32 v32, v26 // (op 4657) reg expression
buffer_load_ubyte v38, v32, s[76:79], 0 offen offset: 0 // (op 4657) Load value
v_add_u32 v32, v32, v28 // (op 4657) 
buffer_load_ubyte v39, v32, s[76:79], 0 offen offset: 0 // (op 4657) Load value



s_lshl_b32 s55, s38, 2 // (op 18234) 
v_add_u32 v36, v36, s55 // (op 18234) 


s_barrier  // (op 19796) 
v_mov_b64 v[28:29], s[44:45] // (op 4658) convert




v_mov_b32 v26, v40 // (op 4658) 
v_mov_b32 v37, v26 // (op 4658) reg expression
buffer_load_ubyte v32, v37, s[80:83], 0 offen offset: 0 // (op 4658) Load value
v_add_u32 v37, v37, v28 // (op 4658) 
s_waitcnt vmcnt(2)
ds_write_b8 v41, v38 offset:49664 // (op 4654) Store local data 
buffer_load_ubyte v33, v37, s[80:83], 0 offen offset: 0 // (op 4658) Load value



s_lshl_b32 s55, s44, 2 // (op 18362) 
v_add_u32 v40, v40, s55 // (op 18362) 

s_waitcnt vmcnt(2)
ds_write_b8 v41, v39 offset:49792 // (op 4654) Store local data 

s_barrier  // (op 19763) 



s_barrier  // (op 19816) 
ds_read_b64_tr_b4 v[46:47], v3 offset:16384 // (op 4645) Transpose load local data 



ds_read_b64_tr_b4 v[48:49], v3 offset:17408 // (op 4645) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v44, v32 offset:49152 // (op 4655) Store local data 
s_barrier  // (op 19783) 
s_waitcnt vmcnt(0)
ds_write_b8 v44, v33 offset:49280 // (op 4655) Store local data 




ds_read_b64_tr_b8 v[50:51], v4 // (op 4647) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:1024 // (op 4647) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:8192 // (op 4647) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:9216 // (op 4647) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 19803) 


ds_read_u8 v26, v0 offset:49664 // (op 4649) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 19823) 


ds_read_u8 v45, v5 offset:49152 // (op 4651) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:57], a[0:3], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4652) 
ds_read_b64_tr_b8 v[58:59], v4 offset:32 // (op 4816) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:1056 // (op 4816) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:8224 // (op 4816) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:9248 // (op 4816) Transpose load local data 


ds_read_u8 v28, v5 offset:49184 // (op 4820) Load local data 



ds_read_b64_tr_b8 v[66:67], v4 offset:64 // (op 4992) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[58:65], a[4:7], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4821) 
ds_read_b64_tr_b8 v[68:69], v4 offset:1088 // (op 4992) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v4 offset:8256 // (op 4992) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v4 offset:9280 // (op 4992) Transpose load local data 


ds_read_u8 v29, v5 offset:49216 // (op 4996) Load local data 



ds_read_b64_tr_b8 v[74:75], v4 offset:96 // (op 5168) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v4 offset:1120 // (op 5168) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:73], a[8:11], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 4997) 
ds_read_b64_tr_b8 v[78:79], v4 offset:8288 // (op 5168) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v4 offset:9312 // (op 5168) Transpose load local data 


ds_read_u8 v32, v5 offset:49248 // (op 5172) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[74:81], a[12:15], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5173) 
ds_read_b64_tr_b4 v[82:83], v3 offset:16400 // (op 5430) Transpose load local data 


ds_read_b64_tr_b4 v[84:85], v3 offset:17424 // (op 5430) Transpose load local data 


ds_read_u8 v26, v0 offset:49696 // (op 5433) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:85], v[50:57], a[16:19], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5435) 
ds_read_b64_tr_b4 v[46:47], v3 offset:16416 // (op 6200) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[82:85], v[58:65], a[20:23], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5466) 
ds_read_b64_tr_b4 v[48:49], v3 offset:17440 // (op 6200) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[66:73], a[24:27], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5497) 
ds_read_u8 v33, v0 offset:49728 // (op 6203) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[74:81], a[28:31], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 5528) 
ds_read_b64_tr_b4 v[86:87], v3 offset:16432 // (op 6970) Transpose load local data 


ds_read_b64_tr_b4 v[88:89], v3 offset:17456 // (op 6970) Transpose load local data 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[46:49], v[50:57], a[32:35], v33, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6205) 
ds_read_u8 v26, v0 offset:49760 // (op 6973) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[46:49], v[58:65], a[36:39], v33, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6236) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[46:49], v[66:73], a[40:43], v33, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6267) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[46:49], v[74:81], a[44:47], v33, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6298) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[50:57], a[48:51], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 6975) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[58:65], a[52:55], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7006) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[66:73], a[56:59], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7037) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[74:81], a[60:63], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 7068) 




s_cmp_lt_u32 s54, s7 // (op 7057) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_7057 // (op 7057) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_7057 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_7057:









s_barrier  // (op 20330) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_322



v_add_u32 v26, 49152, v25 // (op 8100) 
buffer_store_dword a48, v26, s[64:67], 0 offen sc1 // (op 8100) Store value

v_add_u32 v26, 49664, v25 // (op 8100) 
buffer_store_dword a49, v26, s[64:67], 0 offen sc1 // (op 8100) Store value

v_add_u32 v26, 50176, v25 // (op 8100) 
buffer_store_dword a50, v26, s[64:67], 0 offen sc1 // (op 8100) Store value

v_add_u32 v26, 50688, v25 // (op 8100) 
buffer_store_dword a51, v26, s[64:67], 0 offen sc1 // (op 8100) Store value




v_add_u32 v26, 49280, v25 // (op 8102) 
buffer_store_dword a52, v26, s[64:67], 0 offen sc1 // (op 8102) Store value

v_add_u32 v26, 49792, v25 // (op 8102) 
buffer_store_dword a53, v26, s[64:67], 0 offen sc1 // (op 8102) Store value

v_add_u32 v26, 50304, v25 // (op 8102) 
buffer_store_dword a54, v26, s[64:67], 0 offen sc1 // (op 8102) Store value

v_add_u32 v26, 50816, v25 // (op 8102) 
buffer_store_dword a55, v26, s[64:67], 0 offen sc1 // (op 8102) Store value




v_add_u32 v26, 49408, v25 // (op 8104) 
buffer_store_dword a56, v26, s[64:67], 0 offen sc1 // (op 8104) Store value

v_add_u32 v26, 49920, v25 // (op 8104) 
buffer_store_dword a57, v26, s[64:67], 0 offen sc1 // (op 8104) Store value

v_add_u32 v26, 50432, v25 // (op 8104) 
buffer_store_dword a58, v26, s[64:67], 0 offen sc1 // (op 8104) Store value

v_add_u32 v26, 50944, v25 // (op 8104) 
buffer_store_dword a59, v26, s[64:67], 0 offen sc1 // (op 8104) Store value




v_add_u32 v26, 49536, v25 // (op 8106) 
buffer_store_dword a60, v26, s[64:67], 0 offen sc1 // (op 8106) Store value

v_add_u32 v26, 50048, v25 // (op 8106) 
buffer_store_dword a61, v26, s[64:67], 0 offen sc1 // (op 8106) Store value

v_add_u32 v26, 50560, v25 // (op 8106) 
buffer_store_dword a62, v26, s[64:67], 0 offen sc1 // (op 8106) Store value

v_add_u32 v26, 51072, v25 // (op 8106) 
buffer_store_dword a63, v26, s[64:67], 0 offen sc1 // (op 8106) Store value




v_add_u32 v26, 32768, v25 // (op 8076) 
buffer_store_dword a32, v26, s[64:67], 0 offen sc1 // (op 8076) Store value

v_add_u32 v26, 33280, v25 // (op 8076) 
buffer_store_dword a33, v26, s[64:67], 0 offen sc1 // (op 8076) Store value

v_add_u32 v26, 33792, v25 // (op 8076) 
buffer_store_dword a34, v26, s[64:67], 0 offen sc1 // (op 8076) Store value

v_add_u32 v26, 34304, v25 // (op 8076) 
buffer_store_dword a35, v26, s[64:67], 0 offen sc1 // (op 8076) Store value




v_add_u32 v26, 32896, v25 // (op 8078) 
buffer_store_dword a36, v26, s[64:67], 0 offen sc1 // (op 8078) Store value

v_add_u32 v26, 33408, v25 // (op 8078) 
buffer_store_dword a37, v26, s[64:67], 0 offen sc1 // (op 8078) Store value

v_add_u32 v26, 33920, v25 // (op 8078) 
buffer_store_dword a38, v26, s[64:67], 0 offen sc1 // (op 8078) Store value

v_add_u32 v26, 34432, v25 // (op 8078) 
buffer_store_dword a39, v26, s[64:67], 0 offen sc1 // (op 8078) Store value




v_add_u32 v26, 33024, v25 // (op 8080) 
buffer_store_dword a40, v26, s[64:67], 0 offen sc1 // (op 8080) Store value

v_add_u32 v26, 33536, v25 // (op 8080) 
buffer_store_dword a41, v26, s[64:67], 0 offen sc1 // (op 8080) Store value

v_add_u32 v26, 34048, v25 // (op 8080) 
buffer_store_dword a42, v26, s[64:67], 0 offen sc1 // (op 8080) Store value

v_add_u32 v26, 34560, v25 // (op 8080) 
buffer_store_dword a43, v26, s[64:67], 0 offen sc1 // (op 8080) Store value




v_add_u32 v26, 33152, v25 // (op 8082) 
buffer_store_dword a44, v26, s[64:67], 0 offen sc1 // (op 8082) Store value

v_add_u32 v26, 33664, v25 // (op 8082) 
buffer_store_dword a45, v26, s[64:67], 0 offen sc1 // (op 8082) Store value

v_add_u32 v26, 34176, v25 // (op 8082) 
buffer_store_dword a46, v26, s[64:67], 0 offen sc1 // (op 8082) Store value

v_add_u32 v26, 34688, v25 // (op 8082) 
buffer_store_dword a47, v26, s[64:67], 0 offen sc1 // (op 8082) Store value




v_add_u32 v26, 16384, v25 // (op 8052) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 8052) Store value

v_add_u32 v26, 16896, v25 // (op 8052) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 8052) Store value

v_add_u32 v26, 17408, v25 // (op 8052) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 8052) Store value

v_add_u32 v26, 17920, v25 // (op 8052) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 8052) Store value




v_add_u32 v26, 16512, v25 // (op 8054) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 8054) Store value

v_add_u32 v26, 17024, v25 // (op 8054) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 8054) Store value

v_add_u32 v26, 17536, v25 // (op 8054) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 8054) Store value

v_add_u32 v26, 18048, v25 // (op 8054) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 8054) Store value




v_add_u32 v26, 16640, v25 // (op 8056) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 8056) Store value

v_add_u32 v26, 17152, v25 // (op 8056) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 8056) Store value

v_add_u32 v26, 17664, v25 // (op 8056) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 8056) Store value

v_add_u32 v26, 18176, v25 // (op 8056) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 8056) Store value




v_add_u32 v26, 16768, v25 // (op 8058) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 8058) Store value

v_add_u32 v26, 17280, v25 // (op 8058) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 8058) Store value

v_add_u32 v26, 17792, v25 // (op 8058) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 8058) Store value

v_add_u32 v26, 18304, v25 // (op 8058) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 8058) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:1024 sc1 // (op 319) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:1536 sc1 // (op 319) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 8026) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:640 sc1 // (op 8026) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:1152 sc1 // (op 8026) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:1664 sc1 // (op 8026) Store value



buffer_store_dword a8, v25, s[64:67], 0 offen offset:256 sc1 // (op 8027) Store value
buffer_store_dword a9, v25, s[64:67], 0 offen offset:768 sc1 // (op 8027) Store value
buffer_store_dword a10, v25, s[64:67], 0 offen offset:1280 sc1 // (op 8027) Store value
buffer_store_dword a11, v25, s[64:67], 0 offen offset:1792 sc1 // (op 8027) Store value



buffer_store_dword a12, v25, s[64:67], 0 offen offset:384 sc1 // (op 8028) Store value
buffer_store_dword a13, v25, s[64:67], 0 offen offset:896 sc1 // (op 8028) Store value
buffer_store_dword a14, v25, s[64:67], 0 offen offset:1408 sc1 // (op 8028) Store value
buffer_store_dword a15, v25, s[64:67], 0 offen offset:1920 sc1 // (op 8028) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
s_mov_b32 s52, 1 // (op 325) call()
s_mov_b32 s55, 0 // (op 326) convert
s_mov_b32 s54, s16 // (op 326) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 326) 
s_add_u32 s84, s48, s54 // (op 326) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 326) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 326) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_322:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 338) 
s_mov_b32 s52, scc // (op 338) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 338) 
s_mov_b32 s53, s54 // (op 338) convert

s_sub_u32 s54, s53, 1 // (op 338) 

s_cmp_lt_u32 s19, s54 // (op 338) 
s_mov_b32 s53, scc // (op 338) 

s_and_b32 s54, s52, s53 // (op 338) 
s_cmp_lg_u32 s54, 0 // (op 338) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 19221) 
v_lshrrev_b32 v20, 5, v1 // (op 19221) 
s_add_u32 s54, s53, s52 // (op 19221) 
s_lshl_b32 s54, s54, 7 // (op 19221) 

v_lshlrev_b32 v21, 2, v20 // (op 19221) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 19221) 


v_and_b32 v21, -32, v1 // (op 19221) 
v_sub_u32 v23, v1, v21 // (op 19221) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 19221) 


s_lshl_b32 s53, s15, 16 // (op 19223) 
s_mov_b32 s64, s28 // (op 19223) reg expression
s_mov_b32 s65, s29 // (op 19223) reg expression
s_mov_b32 s66, s53 // (op 19223) reg expression
s_mov_b32 s67, 131072 // (op 19223) call()

RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
s_add_u32 s53, s16, 1 // (op 340) 
s_add_u32 s54, s53, s52 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 340) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 340) 

s_add_u32 s84, s48, s54 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 340) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 340) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 342) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
s_mov_b32 s53, 0 // (op 343) call()
s_add_u32 s54, s16, 1 // (op 344) 
s_add_u32 s84, s54, s52 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 344) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 344) 

s_add_u32 s54, s48, s84 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 344) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 344) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 357) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:512 // (op 357) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1024 // (op 357) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1536 // (op 357) Load value
v_accvgpr_read v32, a0 // (op 360) reg expression
v_accvgpr_read v33, a1 // (op 360) reg expression
v_accvgpr_read v34, a2 // (op 360) reg expression
v_accvgpr_read v35, a3 // (op 360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 362) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 362) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 362) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 362) 

v_accvgpr_write a0, v32 // (op 364) reg expression
v_accvgpr_write a1, v33 // (op 364) reg expression
v_accvgpr_write a2, v34 // (op 364) reg expression
v_accvgpr_write a3, v35 // (op 364) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 8175) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:640 // (op 8175) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1152 // (op 8175) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1664 // (op 8175) Load value
v_accvgpr_read v32, a4 // (op 8176) reg expression
v_accvgpr_read v33, a5 // (op 8176) reg expression
v_accvgpr_read v34, a6 // (op 8176) reg expression
v_accvgpr_read v35, a7 // (op 8176) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8177) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8177) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8177) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8177) 

v_accvgpr_write a4, v32 // (op 8178) reg expression
v_accvgpr_write a5, v33 // (op 8178) reg expression
v_accvgpr_write a6, v34 // (op 8178) reg expression
v_accvgpr_write a7, v35 // (op 8178) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:256 // (op 8182) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:768 // (op 8182) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1280 // (op 8182) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1792 // (op 8182) Load value
v_accvgpr_read v32, a8 // (op 8183) reg expression
v_accvgpr_read v33, a9 // (op 8183) reg expression
v_accvgpr_read v34, a10 // (op 8183) reg expression
v_accvgpr_read v35, a11 // (op 8183) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8184) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8184) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8184) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8184) 

v_accvgpr_write a8, v32 // (op 8185) reg expression
v_accvgpr_write a9, v33 // (op 8185) reg expression
v_accvgpr_write a10, v34 // (op 8185) reg expression
v_accvgpr_write a11, v35 // (op 8185) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:384 // (op 8189) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:896 // (op 8189) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1408 // (op 8189) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1920 // (op 8189) Load value
v_accvgpr_read v32, a12 // (op 8190) reg expression
v_accvgpr_read v33, a13 // (op 8190) reg expression
v_accvgpr_read v34, a14 // (op 8190) reg expression
v_accvgpr_read v35, a15 // (op 8190) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8191) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8191) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8191) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8191) 

v_accvgpr_write a12, v32 // (op 8192) reg expression
v_accvgpr_write a13, v33 // (op 8192) reg expression
v_accvgpr_write a14, v34 // (op 8192) reg expression
v_accvgpr_write a15, v35 // (op 8192) reg expression




v_mov_b32 v20, 16384 // (op 8226) 
v_add_i32 v23, v21, v20 // (op 8226) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8226) Load value

v_mov_b32 v20, 16896 // (op 8226) 
v_add_i32 v23, v21, v20 // (op 8226) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8226) Load value

v_mov_b32 v20, 17408 // (op 8226) 
v_add_i32 v23, v21, v20 // (op 8226) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8226) Load value

v_mov_b32 v20, 17920 // (op 8226) 
v_add_i32 v23, v21, v20 // (op 8226) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8226) Load value

v_accvgpr_read v32, a16 // (op 8223) reg expression
v_accvgpr_read v33, a17 // (op 8223) reg expression
v_accvgpr_read v34, a18 // (op 8223) reg expression
v_accvgpr_read v35, a19 // (op 8223) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8224) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8224) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8224) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8224) 

v_accvgpr_write a16, v32 // (op 8225) reg expression
v_accvgpr_write a17, v33 // (op 8225) reg expression
v_accvgpr_write a18, v34 // (op 8225) reg expression
v_accvgpr_write a19, v35 // (op 8225) reg expression




v_mov_b32 v20, 16512 // (op 8231) 
v_add_i32 v23, v21, v20 // (op 8231) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8231) Load value

v_mov_b32 v20, 17024 // (op 8231) 
v_add_i32 v23, v21, v20 // (op 8231) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8231) Load value

v_mov_b32 v20, 17536 // (op 8231) 
v_add_i32 v23, v21, v20 // (op 8231) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8231) Load value

v_mov_b32 v20, 18048 // (op 8231) 
v_add_i32 v23, v21, v20 // (op 8231) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8231) Load value

v_accvgpr_read v32, a20 // (op 8228) reg expression
v_accvgpr_read v33, a21 // (op 8228) reg expression
v_accvgpr_read v34, a22 // (op 8228) reg expression
v_accvgpr_read v35, a23 // (op 8228) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8229) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8229) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8229) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8229) 

v_accvgpr_write a20, v32 // (op 8230) reg expression
v_accvgpr_write a21, v33 // (op 8230) reg expression
v_accvgpr_write a22, v34 // (op 8230) reg expression
v_accvgpr_write a23, v35 // (op 8230) reg expression




v_mov_b32 v20, 16640 // (op 8236) 
v_add_i32 v23, v21, v20 // (op 8236) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8236) Load value

v_mov_b32 v20, 17152 // (op 8236) 
v_add_i32 v23, v21, v20 // (op 8236) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8236) Load value

v_mov_b32 v20, 17664 // (op 8236) 
v_add_i32 v23, v21, v20 // (op 8236) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8236) Load value

v_mov_b32 v20, 18176 // (op 8236) 
v_add_i32 v23, v21, v20 // (op 8236) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8236) Load value

v_accvgpr_read v32, a24 // (op 8233) reg expression
v_accvgpr_read v33, a25 // (op 8233) reg expression
v_accvgpr_read v34, a26 // (op 8233) reg expression
v_accvgpr_read v35, a27 // (op 8233) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8234) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8234) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8234) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8234) 

v_accvgpr_write a24, v32 // (op 8235) reg expression
v_accvgpr_write a25, v33 // (op 8235) reg expression
v_accvgpr_write a26, v34 // (op 8235) reg expression
v_accvgpr_write a27, v35 // (op 8235) reg expression




v_mov_b32 v20, 16768 // (op 8241) 
v_add_i32 v23, v21, v20 // (op 8241) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8241) Load value

v_mov_b32 v20, 17280 // (op 8241) 
v_add_i32 v23, v21, v20 // (op 8241) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8241) Load value

v_mov_b32 v20, 17792 // (op 8241) 
v_add_i32 v23, v21, v20 // (op 8241) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8241) Load value

v_mov_b32 v20, 18304 // (op 8241) 
v_add_i32 v23, v21, v20 // (op 8241) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8241) Load value

v_accvgpr_read v32, a28 // (op 8238) reg expression
v_accvgpr_read v33, a29 // (op 8238) reg expression
v_accvgpr_read v34, a30 // (op 8238) reg expression
v_accvgpr_read v35, a31 // (op 8238) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8239) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8239) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8239) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8239) 

v_accvgpr_write a28, v32 // (op 8240) reg expression
v_accvgpr_write a29, v33 // (op 8240) reg expression
v_accvgpr_write a30, v34 // (op 8240) reg expression
v_accvgpr_write a31, v35 // (op 8240) reg expression




v_mov_b32 v20, 32768 // (op 8274) 
v_add_i32 v23, v21, v20 // (op 8274) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8274) Load value

v_mov_b32 v20, 33280 // (op 8274) 
v_add_i32 v23, v21, v20 // (op 8274) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8274) Load value

v_mov_b32 v20, 33792 // (op 8274) 
v_add_i32 v23, v21, v20 // (op 8274) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8274) Load value

v_mov_b32 v20, 34304 // (op 8274) 
v_add_i32 v23, v21, v20 // (op 8274) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8274) Load value

v_accvgpr_read v32, a32 // (op 8271) reg expression
v_accvgpr_read v33, a33 // (op 8271) reg expression
v_accvgpr_read v34, a34 // (op 8271) reg expression
v_accvgpr_read v35, a35 // (op 8271) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8272) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8272) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8272) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8272) 

v_accvgpr_write a32, v32 // (op 8273) reg expression
v_accvgpr_write a33, v33 // (op 8273) reg expression
v_accvgpr_write a34, v34 // (op 8273) reg expression
v_accvgpr_write a35, v35 // (op 8273) reg expression




v_mov_b32 v20, 32896 // (op 8279) 
v_add_i32 v23, v21, v20 // (op 8279) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8279) Load value

v_mov_b32 v20, 33408 // (op 8279) 
v_add_i32 v23, v21, v20 // (op 8279) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8279) Load value

v_mov_b32 v20, 33920 // (op 8279) 
v_add_i32 v23, v21, v20 // (op 8279) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8279) Load value

v_mov_b32 v20, 34432 // (op 8279) 
v_add_i32 v23, v21, v20 // (op 8279) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8279) Load value

v_accvgpr_read v32, a36 // (op 8276) reg expression
v_accvgpr_read v33, a37 // (op 8276) reg expression
v_accvgpr_read v34, a38 // (op 8276) reg expression
v_accvgpr_read v35, a39 // (op 8276) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8277) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8277) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8277) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8277) 

v_accvgpr_write a36, v32 // (op 8278) reg expression
v_accvgpr_write a37, v33 // (op 8278) reg expression
v_accvgpr_write a38, v34 // (op 8278) reg expression
v_accvgpr_write a39, v35 // (op 8278) reg expression




v_mov_b32 v20, 33024 // (op 8284) 
v_add_i32 v23, v21, v20 // (op 8284) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8284) Load value

v_mov_b32 v20, 33536 // (op 8284) 
v_add_i32 v23, v21, v20 // (op 8284) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8284) Load value

v_mov_b32 v20, 34048 // (op 8284) 
v_add_i32 v23, v21, v20 // (op 8284) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8284) Load value

v_mov_b32 v20, 34560 // (op 8284) 
v_add_i32 v23, v21, v20 // (op 8284) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8284) Load value

v_accvgpr_read v32, a40 // (op 8281) reg expression
v_accvgpr_read v33, a41 // (op 8281) reg expression
v_accvgpr_read v34, a42 // (op 8281) reg expression
v_accvgpr_read v35, a43 // (op 8281) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8282) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8282) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8282) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8282) 

v_accvgpr_write a40, v32 // (op 8283) reg expression
v_accvgpr_write a41, v33 // (op 8283) reg expression
v_accvgpr_write a42, v34 // (op 8283) reg expression
v_accvgpr_write a43, v35 // (op 8283) reg expression




v_mov_b32 v20, 33152 // (op 8289) 
v_add_i32 v23, v21, v20 // (op 8289) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8289) Load value

v_mov_b32 v20, 33664 // (op 8289) 
v_add_i32 v23, v21, v20 // (op 8289) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8289) Load value

v_mov_b32 v20, 34176 // (op 8289) 
v_add_i32 v23, v21, v20 // (op 8289) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8289) Load value

v_mov_b32 v20, 34688 // (op 8289) 
v_add_i32 v23, v21, v20 // (op 8289) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8289) Load value

v_accvgpr_read v32, a44 // (op 8286) reg expression
v_accvgpr_read v33, a45 // (op 8286) reg expression
v_accvgpr_read v34, a46 // (op 8286) reg expression
v_accvgpr_read v35, a47 // (op 8286) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8287) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8287) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8287) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8287) 

v_accvgpr_write a44, v32 // (op 8288) reg expression
v_accvgpr_write a45, v33 // (op 8288) reg expression
v_accvgpr_write a46, v34 // (op 8288) reg expression
v_accvgpr_write a47, v35 // (op 8288) reg expression




v_mov_b32 v20, 49152 // (op 8322) 
v_add_i32 v23, v21, v20 // (op 8322) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8322) Load value

v_mov_b32 v20, 49664 // (op 8322) 
v_add_i32 v23, v21, v20 // (op 8322) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8322) Load value

v_mov_b32 v20, 50176 // (op 8322) 
v_add_i32 v23, v21, v20 // (op 8322) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8322) Load value

v_mov_b32 v20, 50688 // (op 8322) 
v_add_i32 v23, v21, v20 // (op 8322) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8322) Load value

v_accvgpr_read v32, a48 // (op 8319) reg expression
v_accvgpr_read v33, a49 // (op 8319) reg expression
v_accvgpr_read v34, a50 // (op 8319) reg expression
v_accvgpr_read v35, a51 // (op 8319) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8320) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8320) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8320) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8320) 

v_accvgpr_write a48, v32 // (op 8321) reg expression
v_accvgpr_write a49, v33 // (op 8321) reg expression
v_accvgpr_write a50, v34 // (op 8321) reg expression
v_accvgpr_write a51, v35 // (op 8321) reg expression




v_mov_b32 v20, 49280 // (op 8327) 
v_add_i32 v23, v21, v20 // (op 8327) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8327) Load value

v_mov_b32 v20, 49792 // (op 8327) 
v_add_i32 v23, v21, v20 // (op 8327) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8327) Load value

v_mov_b32 v20, 50304 // (op 8327) 
v_add_i32 v23, v21, v20 // (op 8327) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8327) Load value

v_mov_b32 v20, 50816 // (op 8327) 
v_add_i32 v23, v21, v20 // (op 8327) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8327) Load value

v_accvgpr_read v32, a52 // (op 8324) reg expression
v_accvgpr_read v33, a53 // (op 8324) reg expression
v_accvgpr_read v34, a54 // (op 8324) reg expression
v_accvgpr_read v35, a55 // (op 8324) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8325) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8325) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8325) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8325) 

v_accvgpr_write a52, v32 // (op 8326) reg expression
v_accvgpr_write a53, v33 // (op 8326) reg expression
v_accvgpr_write a54, v34 // (op 8326) reg expression
v_accvgpr_write a55, v35 // (op 8326) reg expression




v_mov_b32 v20, 49408 // (op 8332) 
v_add_i32 v23, v21, v20 // (op 8332) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8332) Load value

v_mov_b32 v20, 49920 // (op 8332) 
v_add_i32 v23, v21, v20 // (op 8332) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8332) Load value

v_mov_b32 v20, 50432 // (op 8332) 
v_add_i32 v23, v21, v20 // (op 8332) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8332) Load value

v_mov_b32 v20, 50944 // (op 8332) 
v_add_i32 v23, v21, v20 // (op 8332) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8332) Load value

v_accvgpr_read v32, a56 // (op 8329) reg expression
v_accvgpr_read v33, a57 // (op 8329) reg expression
v_accvgpr_read v34, a58 // (op 8329) reg expression
v_accvgpr_read v35, a59 // (op 8329) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8330) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8330) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8330) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8330) 

v_accvgpr_write a56, v32 // (op 8331) reg expression
v_accvgpr_write a57, v33 // (op 8331) reg expression
v_accvgpr_write a58, v34 // (op 8331) reg expression
v_accvgpr_write a59, v35 // (op 8331) reg expression




v_mov_b32 v20, 49536 // (op 8337) 
v_add_i32 v23, v21, v20 // (op 8337) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8337) Load value

v_mov_b32 v20, 50048 // (op 8337) 
v_add_i32 v23, v21, v20 // (op 8337) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8337) Load value

v_mov_b32 v20, 50560 // (op 8337) 
v_add_i32 v23, v21, v20 // (op 8337) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8337) Load value

v_mov_b32 v20, 51072 // (op 8337) 
v_add_i32 v23, v21, v20 // (op 8337) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8337) Load value

v_accvgpr_read v32, a60 // (op 8334) reg expression
v_accvgpr_read v33, a61 // (op 8334) reg expression
v_accvgpr_read v34, a62 // (op 8334) reg expression
v_accvgpr_read v35, a63 // (op 8334) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8335) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8335) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8335) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8335) 

v_accvgpr_write a60, v32 // (op 8336) reg expression
v_accvgpr_write a61, v33 // (op 8336) reg expression
v_accvgpr_write a62, v34 // (op 8336) reg expression
v_accvgpr_write a63, v35 // (op 8336) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135
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



v_accvgpr_read v28, a4 // (op 8419) 
v_accvgpr_read v29, a5 // (op 8419) 
v_accvgpr_read v30, a6 // (op 8419) 
v_accvgpr_read v31, a7 // (op 8419) 
v_mul_f32 v32, v19, v28 // (op 8419) 
v_mul_f32 v33, v19, v29 // (op 8419) 
v_mul_f32 v34, v19, v30 // (op 8419) 
v_mul_f32 v35, v19, v31 // (op 8419) 

v_cvt_f16_f32 v28, v32 // (op 8421) 
v_cvt_f16_f32 v29, v33 // (op 8421) 
v_cvt_f16_f32 v30, v34 // (op 8421) 
v_cvt_f16_f32 v31, v35 // (op 8421) 


v_pack_b32_f16 v20, v28, v29 // (op 8422) 
v_pack_b32_f16 v21, v30, v31 // (op 8422) 

s_lshl_b32 s52, s24, 6 // (op 8422) 
v_add_u32 v23, v27, s52 // (op 8422) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8422) Store value



v_accvgpr_read v28, a8 // (op 8433) 
v_accvgpr_read v29, a9 // (op 8433) 
v_accvgpr_read v30, a10 // (op 8433) 
v_accvgpr_read v31, a11 // (op 8433) 
v_mul_f32 v32, v19, v28 // (op 8433) 
v_mul_f32 v33, v19, v29 // (op 8433) 
v_mul_f32 v34, v19, v30 // (op 8433) 
v_mul_f32 v35, v19, v31 // (op 8433) 

v_cvt_f16_f32 v28, v32 // (op 8435) 
v_cvt_f16_f32 v29, v33 // (op 8435) 
v_cvt_f16_f32 v30, v34 // (op 8435) 
v_cvt_f16_f32 v31, v35 // (op 8435) 


v_pack_b32_f16 v20, v28, v29 // (op 8436) 
v_pack_b32_f16 v21, v30, v31 // (op 8436) 

s_lshl_b32 s52, s24, 7 // (op 8436) 
v_add_u32 v23, v27, s52 // (op 8436) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8436) Store value



v_accvgpr_read v28, a12 // (op 8447) 
v_accvgpr_read v29, a13 // (op 8447) 
v_accvgpr_read v30, a14 // (op 8447) 
v_accvgpr_read v31, a15 // (op 8447) 
v_mul_f32 v32, v19, v28 // (op 8447) 
v_mul_f32 v33, v19, v29 // (op 8447) 
v_mul_f32 v34, v19, v30 // (op 8447) 
v_mul_f32 v35, v19, v31 // (op 8447) 

v_cvt_f16_f32 v28, v32 // (op 8449) 
v_cvt_f16_f32 v29, v33 // (op 8449) 
v_cvt_f16_f32 v30, v34 // (op 8449) 
v_cvt_f16_f32 v31, v35 // (op 8449) 


v_pack_b32_f16 v20, v28, v29 // (op 8450) 
v_pack_b32_f16 v21, v30, v31 // (op 8450) 

s_lshl_b32 s52, s24, 6 // (op 8450) 
s_mul_i32 s53, 3, s52 // (op 8450) 

v_add_u32 v23, v27, s53 // (op 8450) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8450) Store value



v_accvgpr_read v28, a16 // (op 8483) 
v_accvgpr_read v29, a17 // (op 8483) 
v_accvgpr_read v30, a18 // (op 8483) 
v_accvgpr_read v31, a19 // (op 8483) 
v_mul_f32 v32, v19, v28 // (op 8483) 
v_mul_f32 v33, v19, v29 // (op 8483) 
v_mul_f32 v34, v19, v30 // (op 8483) 
v_mul_f32 v35, v19, v31 // (op 8483) 

v_cvt_f16_f32 v28, v32 // (op 8485) 
v_cvt_f16_f32 v29, v33 // (op 8485) 
v_cvt_f16_f32 v30, v34 // (op 8485) 
v_cvt_f16_f32 v31, v35 // (op 8485) 


v_pack_b32_f16 v20, v28, v29 // (op 8487) 
v_pack_b32_f16 v21, v30, v31 // (op 8487) 

v_add_u32 v23, 64, v27 // (op 8487) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8487) Store value



v_accvgpr_read v28, a20 // (op 8497) 
v_accvgpr_read v29, a21 // (op 8497) 
v_accvgpr_read v30, a22 // (op 8497) 
v_accvgpr_read v31, a23 // (op 8497) 
v_mul_f32 v32, v19, v28 // (op 8497) 
v_mul_f32 v33, v19, v29 // (op 8497) 
v_mul_f32 v34, v19, v30 // (op 8497) 
v_mul_f32 v35, v19, v31 // (op 8497) 

v_cvt_f16_f32 v28, v32 // (op 8499) 
v_cvt_f16_f32 v29, v33 // (op 8499) 
v_cvt_f16_f32 v30, v34 // (op 8499) 
v_cvt_f16_f32 v31, v35 // (op 8499) 


v_pack_b32_f16 v20, v28, v29 // (op 8489) 
v_pack_b32_f16 v21, v30, v31 // (op 8489) 

s_lshl_b32 s52, s24, 6 // (op 8489) 
s_add_u32 s52, s52, 64 // (op 8489) 
v_add_u32 v23, v27, s52 // (op 8489) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8489) Store value



v_accvgpr_read v28, a24 // (op 8503) 
v_accvgpr_read v29, a25 // (op 8503) 
v_accvgpr_read v30, a26 // (op 8503) 
v_accvgpr_read v31, a27 // (op 8503) 
v_mul_f32 v32, v19, v28 // (op 8503) 
v_mul_f32 v33, v19, v29 // (op 8503) 
v_mul_f32 v34, v19, v30 // (op 8503) 
v_mul_f32 v35, v19, v31 // (op 8503) 

v_cvt_f16_f32 v28, v32 // (op 8505) 
v_cvt_f16_f32 v29, v33 // (op 8505) 
v_cvt_f16_f32 v30, v34 // (op 8505) 
v_cvt_f16_f32 v31, v35 // (op 8505) 


v_pack_b32_f16 v20, v28, v29 // (op 8491) 
v_pack_b32_f16 v21, v30, v31 // (op 8491) 

s_lshl_b32 s52, s24, 7 // (op 8491) 
s_add_u32 s52, s52, 64 // (op 8491) 
v_add_u32 v23, v27, s52 // (op 8491) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8491) Store value



v_accvgpr_read v28, a28 // (op 8509) 
v_accvgpr_read v29, a29 // (op 8509) 
v_accvgpr_read v30, a30 // (op 8509) 
v_accvgpr_read v31, a31 // (op 8509) 
v_mul_f32 v32, v19, v28 // (op 8509) 
v_mul_f32 v33, v19, v29 // (op 8509) 
v_mul_f32 v34, v19, v30 // (op 8509) 
v_mul_f32 v35, v19, v31 // (op 8509) 

v_cvt_f16_f32 v28, v32 // (op 8511) 
v_cvt_f16_f32 v29, v33 // (op 8511) 
v_cvt_f16_f32 v30, v34 // (op 8511) 
v_cvt_f16_f32 v31, v35 // (op 8511) 


v_pack_b32_f16 v20, v28, v29 // (op 8493) 
v_pack_b32_f16 v21, v30, v31 // (op 8493) 

s_lshl_b32 s52, s24, 6 // (op 8493) 
s_mul_i32 s53, 3, s52 // (op 8493) 
s_add_u32 s53, s53, 64 // (op 8493) 

v_add_u32 v23, v27, s53 // (op 8493) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8493) Store value



v_accvgpr_read v28, a32 // (op 8559) 
v_accvgpr_read v29, a33 // (op 8559) 
v_accvgpr_read v30, a34 // (op 8559) 
v_accvgpr_read v31, a35 // (op 8559) 
v_mul_f32 v32, v19, v28 // (op 8559) 
v_mul_f32 v33, v19, v29 // (op 8559) 
v_mul_f32 v34, v19, v30 // (op 8559) 
v_mul_f32 v35, v19, v31 // (op 8559) 

v_cvt_f16_f32 v28, v32 // (op 8561) 
v_cvt_f16_f32 v29, v33 // (op 8561) 
v_cvt_f16_f32 v30, v34 // (op 8561) 
v_cvt_f16_f32 v31, v35 // (op 8561) 


v_pack_b32_f16 v20, v28, v29 // (op 8563) 
v_pack_b32_f16 v21, v30, v31 // (op 8563) 

v_add_u32 v23, 128, v27 // (op 8563) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8563) Store value



v_accvgpr_read v28, a36 // (op 8573) 
v_accvgpr_read v29, a37 // (op 8573) 
v_accvgpr_read v30, a38 // (op 8573) 
v_accvgpr_read v31, a39 // (op 8573) 
v_mul_f32 v32, v19, v28 // (op 8573) 
v_mul_f32 v33, v19, v29 // (op 8573) 
v_mul_f32 v34, v19, v30 // (op 8573) 
v_mul_f32 v35, v19, v31 // (op 8573) 

v_cvt_f16_f32 v28, v32 // (op 8575) 
v_cvt_f16_f32 v29, v33 // (op 8575) 
v_cvt_f16_f32 v30, v34 // (op 8575) 
v_cvt_f16_f32 v31, v35 // (op 8575) 


v_pack_b32_f16 v20, v28, v29 // (op 8565) 
v_pack_b32_f16 v21, v30, v31 // (op 8565) 

s_lshl_b32 s52, s24, 6 // (op 8565) 
s_add_u32 s52, s52, 128 // (op 8565) 
v_add_u32 v23, v27, s52 // (op 8565) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8565) Store value



v_accvgpr_read v28, a40 // (op 8579) 
v_accvgpr_read v29, a41 // (op 8579) 
v_accvgpr_read v30, a42 // (op 8579) 
v_accvgpr_read v31, a43 // (op 8579) 
v_mul_f32 v32, v19, v28 // (op 8579) 
v_mul_f32 v33, v19, v29 // (op 8579) 
v_mul_f32 v34, v19, v30 // (op 8579) 
v_mul_f32 v35, v19, v31 // (op 8579) 

v_cvt_f16_f32 v28, v32 // (op 8581) 
v_cvt_f16_f32 v29, v33 // (op 8581) 
v_cvt_f16_f32 v30, v34 // (op 8581) 
v_cvt_f16_f32 v31, v35 // (op 8581) 


v_pack_b32_f16 v20, v28, v29 // (op 8567) 
v_pack_b32_f16 v21, v30, v31 // (op 8567) 

s_lshl_b32 s52, s24, 7 // (op 8567) 
s_add_u32 s52, s52, 128 // (op 8567) 
v_add_u32 v23, v27, s52 // (op 8567) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8567) Store value



v_accvgpr_read v28, a44 // (op 8585) 
v_accvgpr_read v29, a45 // (op 8585) 
v_accvgpr_read v30, a46 // (op 8585) 
v_accvgpr_read v31, a47 // (op 8585) 
v_mul_f32 v32, v19, v28 // (op 8585) 
v_mul_f32 v33, v19, v29 // (op 8585) 
v_mul_f32 v34, v19, v30 // (op 8585) 
v_mul_f32 v35, v19, v31 // (op 8585) 

v_cvt_f16_f32 v28, v32 // (op 8587) 
v_cvt_f16_f32 v29, v33 // (op 8587) 
v_cvt_f16_f32 v30, v34 // (op 8587) 
v_cvt_f16_f32 v31, v35 // (op 8587) 


v_pack_b32_f16 v20, v28, v29 // (op 8569) 
v_pack_b32_f16 v21, v30, v31 // (op 8569) 

s_lshl_b32 s52, s24, 6 // (op 8569) 
s_mul_i32 s53, 3, s52 // (op 8569) 
s_add_u32 s53, s53, 128 // (op 8569) 

v_add_u32 v23, v27, s53 // (op 8569) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8569) Store value



v_accvgpr_read v28, a48 // (op 8635) 
v_accvgpr_read v29, a49 // (op 8635) 
v_accvgpr_read v30, a50 // (op 8635) 
v_accvgpr_read v31, a51 // (op 8635) 
v_mul_f32 v32, v19, v28 // (op 8635) 
v_mul_f32 v33, v19, v29 // (op 8635) 
v_mul_f32 v34, v19, v30 // (op 8635) 
v_mul_f32 v35, v19, v31 // (op 8635) 

v_cvt_f16_f32 v28, v32 // (op 8637) 
v_cvt_f16_f32 v29, v33 // (op 8637) 
v_cvt_f16_f32 v30, v34 // (op 8637) 
v_cvt_f16_f32 v31, v35 // (op 8637) 


v_pack_b32_f16 v20, v28, v29 // (op 8639) 
v_pack_b32_f16 v21, v30, v31 // (op 8639) 

v_add_u32 v23, 192, v27 // (op 8639) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8639) Store value



v_accvgpr_read v28, a52 // (op 8649) 
v_accvgpr_read v29, a53 // (op 8649) 
v_accvgpr_read v30, a54 // (op 8649) 
v_accvgpr_read v31, a55 // (op 8649) 
v_mul_f32 v32, v19, v28 // (op 8649) 
v_mul_f32 v33, v19, v29 // (op 8649) 
v_mul_f32 v34, v19, v30 // (op 8649) 
v_mul_f32 v35, v19, v31 // (op 8649) 

v_cvt_f16_f32 v28, v32 // (op 8651) 
v_cvt_f16_f32 v29, v33 // (op 8651) 
v_cvt_f16_f32 v30, v34 // (op 8651) 
v_cvt_f16_f32 v31, v35 // (op 8651) 


v_pack_b32_f16 v20, v28, v29 // (op 8641) 
v_pack_b32_f16 v21, v30, v31 // (op 8641) 

s_lshl_b32 s52, s24, 6 // (op 8641) 
s_add_u32 s52, s52, 192 // (op 8641) 
v_add_u32 v23, v27, s52 // (op 8641) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8641) Store value



v_accvgpr_read v28, a56 // (op 8655) 
v_accvgpr_read v29, a57 // (op 8655) 
v_accvgpr_read v30, a58 // (op 8655) 
v_accvgpr_read v31, a59 // (op 8655) 
v_mul_f32 v32, v19, v28 // (op 8655) 
v_mul_f32 v33, v19, v29 // (op 8655) 
v_mul_f32 v34, v19, v30 // (op 8655) 
v_mul_f32 v35, v19, v31 // (op 8655) 

v_cvt_f16_f32 v28, v32 // (op 8657) 
v_cvt_f16_f32 v29, v33 // (op 8657) 
v_cvt_f16_f32 v30, v34 // (op 8657) 
v_cvt_f16_f32 v31, v35 // (op 8657) 


v_pack_b32_f16 v20, v28, v29 // (op 8643) 
v_pack_b32_f16 v21, v30, v31 // (op 8643) 

s_lshl_b32 s52, s24, 7 // (op 8643) 
s_add_u32 s52, s52, 192 // (op 8643) 
v_add_u32 v23, v27, s52 // (op 8643) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8643) Store value



v_accvgpr_read v28, a60 // (op 8661) 
v_accvgpr_read v29, a61 // (op 8661) 
v_accvgpr_read v30, a62 // (op 8661) 
v_accvgpr_read v31, a63 // (op 8661) 
v_mul_f32 v32, v19, v28 // (op 8661) 
v_mul_f32 v33, v19, v29 // (op 8661) 
v_mul_f32 v34, v19, v30 // (op 8661) 
v_mul_f32 v35, v19, v31 // (op 8661) 

v_cvt_f16_f32 v28, v32 // (op 8663) 
v_cvt_f16_f32 v29, v33 // (op 8663) 
v_cvt_f16_f32 v30, v34 // (op 8663) 
v_cvt_f16_f32 v31, v35 // (op 8663) 


v_pack_b32_f16 v20, v28, v29 // (op 8645) 
v_pack_b32_f16 v21, v30, v31 // (op 8645) 

s_lshl_b32 s52, s24, 6 // (op 8645) 
s_mul_i32 s53, 3, s52 // (op 8645) 
s_add_u32 s53, s53, 192 // (op 8645) 

v_add_u32 v23, v27, s53 // (op 8645) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8645) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v20, 192, v22 // (op 9033) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 9033) Load value

v_accvgpr_read v28, a48 // (op 9020) 
v_accvgpr_read v29, a49 // (op 9020) 
v_accvgpr_read v30, a50 // (op 9020) 
v_accvgpr_read v31, a51 // (op 9020) 
v_mul_f32 v32, v19, v28 // (op 9020) 
v_mul_f32 v33, v19, v29 // (op 9020) 
v_mul_f32 v34, v19, v30 // (op 9020) 
v_mul_f32 v35, v19, v31 // (op 9020) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 9032) 
v_lshrrev_b32 v21, 16, v24 // (op 9032) 
v_cvt_f32_f16 v21, v21 // (op 9032) 
v_mul_f32 v28, v18, v20 // (op 9032) 
v_mul_f32 v29, v18, v21 // (op 9032) 
v_cvt_f32_f16 v20, v25 // (op 9032) 
v_lshrrev_b32 v21, 16, v25 // (op 9032) 
v_cvt_f32_f16 v21, v21 // (op 9032) 
v_mul_f32 v30, v18, v20 // (op 9032) 
v_mul_f32 v31, v18, v21 // (op 9032) 


v_add_f32 v36, v28, v32 // (op 9021) 
v_add_f32 v37, v29, v33 // (op 9021) 
v_add_f32 v38, v30, v34 // (op 9021) 
v_add_f32 v39, v31, v35 // (op 9021) 


v_cvt_f16_f32 v28, v36 // (op 9022) 
v_cvt_f16_f32 v29, v37 // (op 9022) 
v_cvt_f16_f32 v30, v38 // (op 9022) 
v_cvt_f16_f32 v31, v39 // (op 9022) 


v_pack_b32_f16 v20, v28, v29 // (op 9024) 
v_pack_b32_f16 v21, v30, v31 // (op 9024) 

v_add_u32 v23, 192, v27 // (op 9024) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 9024) Store value





s_lshl_b32 s52, s4, 6 // (op 9038) 
s_add_u32 s52, s52, 192 // (op 9038) 
v_add_u32 v20, v22, s52 // (op 9038) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 9038) Load value

v_accvgpr_read v28, a52 // (op 9050) 
v_accvgpr_read v29, a53 // (op 9050) 
v_accvgpr_read v30, a54 // (op 9050) 
v_accvgpr_read v31, a55 // (op 9050) 
v_mul_f32 v32, v19, v28 // (op 9050) 
v_mul_f32 v33, v19, v29 // (op 9050) 
v_mul_f32 v34, v19, v30 // (op 9050) 
v_mul_f32 v35, v19, v31 // (op 9050) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 9035) 
v_lshrrev_b32 v21, 16, v24 // (op 9035) 
v_cvt_f32_f16 v21, v21 // (op 9035) 
v_mul_f32 v28, v18, v20 // (op 9035) 
v_mul_f32 v29, v18, v21 // (op 9035) 
v_cvt_f32_f16 v20, v25 // (op 9035) 
v_lshrrev_b32 v21, 16, v25 // (op 9035) 
v_cvt_f32_f16 v21, v21 // (op 9035) 
v_mul_f32 v30, v18, v20 // (op 9035) 
v_mul_f32 v31, v18, v21 // (op 9035) 


v_add_f32 v36, v28, v32 // (op 9036) 
v_add_f32 v37, v29, v33 // (op 9036) 
v_add_f32 v38, v30, v34 // (op 9036) 
v_add_f32 v39, v31, v35 // (op 9036) 


v_cvt_f16_f32 v28, v36 // (op 9037) 
v_cvt_f16_f32 v29, v37 // (op 9037) 
v_cvt_f16_f32 v30, v38 // (op 9037) 
v_cvt_f16_f32 v31, v39 // (op 9037) 


v_pack_b32_f16 v20, v28, v29 // (op 9026) 
v_pack_b32_f16 v21, v30, v31 // (op 9026) 

s_lshl_b32 s52, s24, 6 // (op 9026) 
s_add_u32 s52, s52, 192 // (op 9026) 
v_add_u32 v23, v27, s52 // (op 9026) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 9026) Store value





s_lshl_b32 s52, s4, 7 // (op 9043) 
s_add_u32 s52, s52, 192 // (op 9043) 
v_add_u32 v20, v22, s52 // (op 9043) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 9043) Load value

v_accvgpr_read v28, a56 // (op 9052) 
v_accvgpr_read v29, a57 // (op 9052) 
v_accvgpr_read v30, a58 // (op 9052) 
v_accvgpr_read v31, a59 // (op 9052) 
v_mul_f32 v32, v19, v28 // (op 9052) 
v_mul_f32 v33, v19, v29 // (op 9052) 
v_mul_f32 v34, v19, v30 // (op 9052) 
v_mul_f32 v35, v19, v31 // (op 9052) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 9040) 
v_lshrrev_b32 v21, 16, v24 // (op 9040) 
v_cvt_f32_f16 v21, v21 // (op 9040) 
v_mul_f32 v28, v18, v20 // (op 9040) 
v_mul_f32 v29, v18, v21 // (op 9040) 
v_cvt_f32_f16 v20, v25 // (op 9040) 
v_lshrrev_b32 v21, 16, v25 // (op 9040) 
v_cvt_f32_f16 v21, v21 // (op 9040) 
v_mul_f32 v30, v18, v20 // (op 9040) 
v_mul_f32 v31, v18, v21 // (op 9040) 


v_add_f32 v36, v28, v32 // (op 9041) 
v_add_f32 v37, v29, v33 // (op 9041) 
v_add_f32 v38, v30, v34 // (op 9041) 
v_add_f32 v39, v31, v35 // (op 9041) 


v_cvt_f16_f32 v28, v36 // (op 9042) 
v_cvt_f16_f32 v29, v37 // (op 9042) 
v_cvt_f16_f32 v30, v38 // (op 9042) 
v_cvt_f16_f32 v31, v39 // (op 9042) 


v_pack_b32_f16 v20, v28, v29 // (op 9028) 
v_pack_b32_f16 v21, v30, v31 // (op 9028) 

s_lshl_b32 s52, s24, 7 // (op 9028) 
s_add_u32 s52, s52, 192 // (op 9028) 
v_add_u32 v23, v27, s52 // (op 9028) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 9028) Store value





s_lshl_b32 s52, s4, 6 // (op 9048) 
s_mul_i32 s53, 3, s52 // (op 9048) 
s_add_u32 s53, s53, 192 // (op 9048) 

v_add_u32 v20, v22, s53 // (op 9048) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 9048) Load value

v_accvgpr_read v28, a60 // (op 9054) 
v_accvgpr_read v29, a61 // (op 9054) 
v_accvgpr_read v30, a62 // (op 9054) 
v_accvgpr_read v31, a63 // (op 9054) 
v_mul_f32 v32, v19, v28 // (op 9054) 
v_mul_f32 v33, v19, v29 // (op 9054) 
v_mul_f32 v34, v19, v30 // (op 9054) 
v_mul_f32 v35, v19, v31 // (op 9054) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 9045) 
v_lshrrev_b32 v21, 16, v24 // (op 9045) 
v_cvt_f32_f16 v21, v21 // (op 9045) 
v_mul_f32 v28, v18, v20 // (op 9045) 
v_mul_f32 v29, v18, v21 // (op 9045) 
v_cvt_f32_f16 v20, v25 // (op 9045) 
v_lshrrev_b32 v21, 16, v25 // (op 9045) 
v_cvt_f32_f16 v21, v21 // (op 9045) 
v_mul_f32 v30, v18, v20 // (op 9045) 
v_mul_f32 v31, v18, v21 // (op 9045) 


v_add_f32 v36, v28, v32 // (op 9046) 
v_add_f32 v37, v29, v33 // (op 9046) 
v_add_f32 v38, v30, v34 // (op 9046) 
v_add_f32 v39, v31, v35 // (op 9046) 


v_cvt_f16_f32 v28, v36 // (op 9047) 
v_cvt_f16_f32 v29, v37 // (op 9047) 
v_cvt_f16_f32 v30, v38 // (op 9047) 
v_cvt_f16_f32 v31, v39 // (op 9047) 


v_pack_b32_f16 v20, v28, v29 // (op 9030) 
v_pack_b32_f16 v21, v30, v31 // (op 9030) 

s_lshl_b32 s52, s24, 6 // (op 9030) 
s_mul_i32 s53, 3, s52 // (op 9030) 
s_add_u32 s53, s53, 192 // (op 9030) 

v_add_u32 v23, v27, s53 // (op 9030) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 9030) Store value





v_add_u32 v20, 128, v22 // (op 8946) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8946) Load value

v_accvgpr_read v28, a32 // (op 8933) 
v_accvgpr_read v29, a33 // (op 8933) 
v_accvgpr_read v30, a34 // (op 8933) 
v_accvgpr_read v31, a35 // (op 8933) 
v_mul_f32 v32, v19, v28 // (op 8933) 
v_mul_f32 v33, v19, v29 // (op 8933) 
v_mul_f32 v34, v19, v30 // (op 8933) 
v_mul_f32 v35, v19, v31 // (op 8933) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8945) 
v_lshrrev_b32 v21, 16, v24 // (op 8945) 
v_cvt_f32_f16 v21, v21 // (op 8945) 
v_mul_f32 v28, v18, v20 // (op 8945) 
v_mul_f32 v29, v18, v21 // (op 8945) 
v_cvt_f32_f16 v20, v25 // (op 8945) 
v_lshrrev_b32 v21, 16, v25 // (op 8945) 
v_cvt_f32_f16 v21, v21 // (op 8945) 
v_mul_f32 v30, v18, v20 // (op 8945) 
v_mul_f32 v31, v18, v21 // (op 8945) 


v_add_f32 v36, v28, v32 // (op 8934) 
v_add_f32 v37, v29, v33 // (op 8934) 
v_add_f32 v38, v30, v34 // (op 8934) 
v_add_f32 v39, v31, v35 // (op 8934) 


v_cvt_f16_f32 v28, v36 // (op 8935) 
v_cvt_f16_f32 v29, v37 // (op 8935) 
v_cvt_f16_f32 v30, v38 // (op 8935) 
v_cvt_f16_f32 v31, v39 // (op 8935) 


v_pack_b32_f16 v20, v28, v29 // (op 8937) 
v_pack_b32_f16 v21, v30, v31 // (op 8937) 

v_add_u32 v23, 128, v27 // (op 8937) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8937) Store value





s_lshl_b32 s52, s4, 6 // (op 8951) 
s_add_u32 s52, s52, 128 // (op 8951) 
v_add_u32 v20, v22, s52 // (op 8951) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8951) Load value

v_accvgpr_read v28, a36 // (op 8963) 
v_accvgpr_read v29, a37 // (op 8963) 
v_accvgpr_read v30, a38 // (op 8963) 
v_accvgpr_read v31, a39 // (op 8963) 
v_mul_f32 v32, v19, v28 // (op 8963) 
v_mul_f32 v33, v19, v29 // (op 8963) 
v_mul_f32 v34, v19, v30 // (op 8963) 
v_mul_f32 v35, v19, v31 // (op 8963) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8948) 
v_lshrrev_b32 v21, 16, v24 // (op 8948) 
v_cvt_f32_f16 v21, v21 // (op 8948) 
v_mul_f32 v28, v18, v20 // (op 8948) 
v_mul_f32 v29, v18, v21 // (op 8948) 
v_cvt_f32_f16 v20, v25 // (op 8948) 
v_lshrrev_b32 v21, 16, v25 // (op 8948) 
v_cvt_f32_f16 v21, v21 // (op 8948) 
v_mul_f32 v30, v18, v20 // (op 8948) 
v_mul_f32 v31, v18, v21 // (op 8948) 


v_add_f32 v36, v28, v32 // (op 8949) 
v_add_f32 v37, v29, v33 // (op 8949) 
v_add_f32 v38, v30, v34 // (op 8949) 
v_add_f32 v39, v31, v35 // (op 8949) 


v_cvt_f16_f32 v28, v36 // (op 8950) 
v_cvt_f16_f32 v29, v37 // (op 8950) 
v_cvt_f16_f32 v30, v38 // (op 8950) 
v_cvt_f16_f32 v31, v39 // (op 8950) 


v_pack_b32_f16 v20, v28, v29 // (op 8939) 
v_pack_b32_f16 v21, v30, v31 // (op 8939) 

s_lshl_b32 s52, s24, 6 // (op 8939) 
s_add_u32 s52, s52, 128 // (op 8939) 
v_add_u32 v23, v27, s52 // (op 8939) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8939) Store value





s_lshl_b32 s52, s4, 7 // (op 8956) 
s_add_u32 s52, s52, 128 // (op 8956) 
v_add_u32 v20, v22, s52 // (op 8956) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8956) Load value

v_accvgpr_read v28, a40 // (op 8965) 
v_accvgpr_read v29, a41 // (op 8965) 
v_accvgpr_read v30, a42 // (op 8965) 
v_accvgpr_read v31, a43 // (op 8965) 
v_mul_f32 v32, v19, v28 // (op 8965) 
v_mul_f32 v33, v19, v29 // (op 8965) 
v_mul_f32 v34, v19, v30 // (op 8965) 
v_mul_f32 v35, v19, v31 // (op 8965) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8953) 
v_lshrrev_b32 v21, 16, v24 // (op 8953) 
v_cvt_f32_f16 v21, v21 // (op 8953) 
v_mul_f32 v28, v18, v20 // (op 8953) 
v_mul_f32 v29, v18, v21 // (op 8953) 
v_cvt_f32_f16 v20, v25 // (op 8953) 
v_lshrrev_b32 v21, 16, v25 // (op 8953) 
v_cvt_f32_f16 v21, v21 // (op 8953) 
v_mul_f32 v30, v18, v20 // (op 8953) 
v_mul_f32 v31, v18, v21 // (op 8953) 


v_add_f32 v36, v28, v32 // (op 8954) 
v_add_f32 v37, v29, v33 // (op 8954) 
v_add_f32 v38, v30, v34 // (op 8954) 
v_add_f32 v39, v31, v35 // (op 8954) 


v_cvt_f16_f32 v28, v36 // (op 8955) 
v_cvt_f16_f32 v29, v37 // (op 8955) 
v_cvt_f16_f32 v30, v38 // (op 8955) 
v_cvt_f16_f32 v31, v39 // (op 8955) 


v_pack_b32_f16 v20, v28, v29 // (op 8941) 
v_pack_b32_f16 v21, v30, v31 // (op 8941) 

s_lshl_b32 s52, s24, 7 // (op 8941) 
s_add_u32 s52, s52, 128 // (op 8941) 
v_add_u32 v23, v27, s52 // (op 8941) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8941) Store value





s_lshl_b32 s52, s4, 6 // (op 8961) 
s_mul_i32 s53, 3, s52 // (op 8961) 
s_add_u32 s53, s53, 128 // (op 8961) 

v_add_u32 v20, v22, s53 // (op 8961) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8961) Load value

v_accvgpr_read v28, a44 // (op 8967) 
v_accvgpr_read v29, a45 // (op 8967) 
v_accvgpr_read v30, a46 // (op 8967) 
v_accvgpr_read v31, a47 // (op 8967) 
v_mul_f32 v32, v19, v28 // (op 8967) 
v_mul_f32 v33, v19, v29 // (op 8967) 
v_mul_f32 v34, v19, v30 // (op 8967) 
v_mul_f32 v35, v19, v31 // (op 8967) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8958) 
v_lshrrev_b32 v21, 16, v24 // (op 8958) 
v_cvt_f32_f16 v21, v21 // (op 8958) 
v_mul_f32 v28, v18, v20 // (op 8958) 
v_mul_f32 v29, v18, v21 // (op 8958) 
v_cvt_f32_f16 v20, v25 // (op 8958) 
v_lshrrev_b32 v21, 16, v25 // (op 8958) 
v_cvt_f32_f16 v21, v21 // (op 8958) 
v_mul_f32 v30, v18, v20 // (op 8958) 
v_mul_f32 v31, v18, v21 // (op 8958) 


v_add_f32 v36, v28, v32 // (op 8959) 
v_add_f32 v37, v29, v33 // (op 8959) 
v_add_f32 v38, v30, v34 // (op 8959) 
v_add_f32 v39, v31, v35 // (op 8959) 


v_cvt_f16_f32 v28, v36 // (op 8960) 
v_cvt_f16_f32 v29, v37 // (op 8960) 
v_cvt_f16_f32 v30, v38 // (op 8960) 
v_cvt_f16_f32 v31, v39 // (op 8960) 


v_pack_b32_f16 v20, v28, v29 // (op 8943) 
v_pack_b32_f16 v21, v30, v31 // (op 8943) 

s_lshl_b32 s52, s24, 6 // (op 8943) 
s_mul_i32 s53, 3, s52 // (op 8943) 
s_add_u32 s53, s53, 128 // (op 8943) 

v_add_u32 v23, v27, s53 // (op 8943) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8943) Store value





v_add_u32 v20, 64, v22 // (op 8859) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8859) Load value

v_accvgpr_read v28, a16 // (op 8846) 
v_accvgpr_read v29, a17 // (op 8846) 
v_accvgpr_read v30, a18 // (op 8846) 
v_accvgpr_read v31, a19 // (op 8846) 
v_mul_f32 v32, v19, v28 // (op 8846) 
v_mul_f32 v33, v19, v29 // (op 8846) 
v_mul_f32 v34, v19, v30 // (op 8846) 
v_mul_f32 v35, v19, v31 // (op 8846) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8858) 
v_lshrrev_b32 v21, 16, v24 // (op 8858) 
v_cvt_f32_f16 v21, v21 // (op 8858) 
v_mul_f32 v28, v18, v20 // (op 8858) 
v_mul_f32 v29, v18, v21 // (op 8858) 
v_cvt_f32_f16 v20, v25 // (op 8858) 
v_lshrrev_b32 v21, 16, v25 // (op 8858) 
v_cvt_f32_f16 v21, v21 // (op 8858) 
v_mul_f32 v30, v18, v20 // (op 8858) 
v_mul_f32 v31, v18, v21 // (op 8858) 


v_add_f32 v36, v28, v32 // (op 8847) 
v_add_f32 v37, v29, v33 // (op 8847) 
v_add_f32 v38, v30, v34 // (op 8847) 
v_add_f32 v39, v31, v35 // (op 8847) 


v_cvt_f16_f32 v28, v36 // (op 8848) 
v_cvt_f16_f32 v29, v37 // (op 8848) 
v_cvt_f16_f32 v30, v38 // (op 8848) 
v_cvt_f16_f32 v31, v39 // (op 8848) 


v_pack_b32_f16 v20, v28, v29 // (op 8850) 
v_pack_b32_f16 v21, v30, v31 // (op 8850) 

v_add_u32 v23, 64, v27 // (op 8850) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8850) Store value





s_lshl_b32 s52, s4, 6 // (op 8864) 
s_add_u32 s52, s52, 64 // (op 8864) 
v_add_u32 v20, v22, s52 // (op 8864) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8864) Load value

v_accvgpr_read v28, a20 // (op 8876) 
v_accvgpr_read v29, a21 // (op 8876) 
v_accvgpr_read v30, a22 // (op 8876) 
v_accvgpr_read v31, a23 // (op 8876) 
v_mul_f32 v32, v19, v28 // (op 8876) 
v_mul_f32 v33, v19, v29 // (op 8876) 
v_mul_f32 v34, v19, v30 // (op 8876) 
v_mul_f32 v35, v19, v31 // (op 8876) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8861) 
v_lshrrev_b32 v21, 16, v24 // (op 8861) 
v_cvt_f32_f16 v21, v21 // (op 8861) 
v_mul_f32 v28, v18, v20 // (op 8861) 
v_mul_f32 v29, v18, v21 // (op 8861) 
v_cvt_f32_f16 v20, v25 // (op 8861) 
v_lshrrev_b32 v21, 16, v25 // (op 8861) 
v_cvt_f32_f16 v21, v21 // (op 8861) 
v_mul_f32 v30, v18, v20 // (op 8861) 
v_mul_f32 v31, v18, v21 // (op 8861) 


v_add_f32 v36, v28, v32 // (op 8862) 
v_add_f32 v37, v29, v33 // (op 8862) 
v_add_f32 v38, v30, v34 // (op 8862) 
v_add_f32 v39, v31, v35 // (op 8862) 


v_cvt_f16_f32 v28, v36 // (op 8863) 
v_cvt_f16_f32 v29, v37 // (op 8863) 
v_cvt_f16_f32 v30, v38 // (op 8863) 
v_cvt_f16_f32 v31, v39 // (op 8863) 


v_pack_b32_f16 v20, v28, v29 // (op 8852) 
v_pack_b32_f16 v21, v30, v31 // (op 8852) 

s_lshl_b32 s52, s24, 6 // (op 8852) 
s_add_u32 s52, s52, 64 // (op 8852) 
v_add_u32 v23, v27, s52 // (op 8852) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8852) Store value





s_lshl_b32 s52, s4, 7 // (op 8869) 
s_add_u32 s52, s52, 64 // (op 8869) 
v_add_u32 v20, v22, s52 // (op 8869) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8869) Load value

v_accvgpr_read v28, a24 // (op 8878) 
v_accvgpr_read v29, a25 // (op 8878) 
v_accvgpr_read v30, a26 // (op 8878) 
v_accvgpr_read v31, a27 // (op 8878) 
v_mul_f32 v32, v19, v28 // (op 8878) 
v_mul_f32 v33, v19, v29 // (op 8878) 
v_mul_f32 v34, v19, v30 // (op 8878) 
v_mul_f32 v35, v19, v31 // (op 8878) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8866) 
v_lshrrev_b32 v21, 16, v24 // (op 8866) 
v_cvt_f32_f16 v21, v21 // (op 8866) 
v_mul_f32 v28, v18, v20 // (op 8866) 
v_mul_f32 v29, v18, v21 // (op 8866) 
v_cvt_f32_f16 v20, v25 // (op 8866) 
v_lshrrev_b32 v21, 16, v25 // (op 8866) 
v_cvt_f32_f16 v21, v21 // (op 8866) 
v_mul_f32 v30, v18, v20 // (op 8866) 
v_mul_f32 v31, v18, v21 // (op 8866) 


v_add_f32 v36, v28, v32 // (op 8867) 
v_add_f32 v37, v29, v33 // (op 8867) 
v_add_f32 v38, v30, v34 // (op 8867) 
v_add_f32 v39, v31, v35 // (op 8867) 


v_cvt_f16_f32 v28, v36 // (op 8868) 
v_cvt_f16_f32 v29, v37 // (op 8868) 
v_cvt_f16_f32 v30, v38 // (op 8868) 
v_cvt_f16_f32 v31, v39 // (op 8868) 


v_pack_b32_f16 v20, v28, v29 // (op 8854) 
v_pack_b32_f16 v21, v30, v31 // (op 8854) 

s_lshl_b32 s52, s24, 7 // (op 8854) 
s_add_u32 s52, s52, 64 // (op 8854) 
v_add_u32 v23, v27, s52 // (op 8854) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8854) Store value





s_lshl_b32 s52, s4, 6 // (op 8874) 
s_mul_i32 s53, 3, s52 // (op 8874) 
s_add_u32 s53, s53, 64 // (op 8874) 

v_add_u32 v20, v22, s53 // (op 8874) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8874) Load value

v_accvgpr_read v28, a28 // (op 8880) 
v_accvgpr_read v29, a29 // (op 8880) 
v_accvgpr_read v30, a30 // (op 8880) 
v_accvgpr_read v31, a31 // (op 8880) 
v_mul_f32 v32, v19, v28 // (op 8880) 
v_mul_f32 v33, v19, v29 // (op 8880) 
v_mul_f32 v34, v19, v30 // (op 8880) 
v_mul_f32 v35, v19, v31 // (op 8880) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8871) 
v_lshrrev_b32 v21, 16, v24 // (op 8871) 
v_cvt_f32_f16 v21, v21 // (op 8871) 
v_mul_f32 v28, v18, v20 // (op 8871) 
v_mul_f32 v29, v18, v21 // (op 8871) 
v_cvt_f32_f16 v20, v25 // (op 8871) 
v_lshrrev_b32 v21, 16, v25 // (op 8871) 
v_cvt_f32_f16 v21, v21 // (op 8871) 
v_mul_f32 v30, v18, v20 // (op 8871) 
v_mul_f32 v31, v18, v21 // (op 8871) 


v_add_f32 v36, v28, v32 // (op 8872) 
v_add_f32 v37, v29, v33 // (op 8872) 
v_add_f32 v38, v30, v34 // (op 8872) 
v_add_f32 v39, v31, v35 // (op 8872) 


v_cvt_f16_f32 v28, v36 // (op 8873) 
v_cvt_f16_f32 v29, v37 // (op 8873) 
v_cvt_f16_f32 v30, v38 // (op 8873) 
v_cvt_f16_f32 v31, v39 // (op 8873) 


v_pack_b32_f16 v20, v28, v29 // (op 8856) 
v_pack_b32_f16 v21, v30, v31 // (op 8856) 

s_lshl_b32 s52, s24, 6 // (op 8856) 
s_mul_i32 s53, 3, s52 // (op 8856) 
s_add_u32 s53, s53, 64 // (op 8856) 

v_add_u32 v23, v27, s53 // (op 8856) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8856) Store value





v_mov_b32 v20, v22 // (op 148) reg expression

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v28, a0 // (op 144) 
v_accvgpr_read v29, a1 // (op 144) 
v_accvgpr_read v30, a2 // (op 144) 
v_accvgpr_read v31, a3 // (op 144) 
v_mul_f32 v32, v19, v28 // (op 144) 
v_mul_f32 v33, v19, v29 // (op 144) 
v_mul_f32 v34, v19, v30 // (op 144) 
v_mul_f32 v35, v19, v31 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 149) 
v_lshrrev_b32 v21, 16, v24 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v28, v18, v20 // (op 149) 
v_mul_f32 v29, v18, v21 // (op 149) 
v_cvt_f32_f16 v20, v25 // (op 149) 
v_lshrrev_b32 v21, 16, v25 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v30, v18, v20 // (op 149) 
v_mul_f32 v31, v18, v21 // (op 149) 


v_add_f32 v36, v28, v32 // (op 145) 
v_add_f32 v37, v29, v33 // (op 145) 
v_add_f32 v38, v30, v34 // (op 145) 
v_add_f32 v39, v31, v35 // (op 145) 


v_cvt_f16_f32 v28, v36 // (op 146) 
v_cvt_f16_f32 v29, v37 // (op 146) 
v_cvt_f16_f32 v30, v38 // (op 146) 
v_cvt_f16_f32 v31, v39 // (op 146) 


v_pack_b32_f16 v20, v28, v29 // (op 147) 
v_pack_b32_f16 v21, v30, v31 // (op 147) 

v_mov_b32 v23, v27 // (op 147) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s52, s4, 6 // (op 8767) 
v_add_u32 v20, v22, s52 // (op 8767) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8767) Load value

v_accvgpr_read v28, a4 // (op 8763) 
v_accvgpr_read v29, a5 // (op 8763) 
v_accvgpr_read v30, a6 // (op 8763) 
v_accvgpr_read v31, a7 // (op 8763) 
v_mul_f32 v32, v19, v28 // (op 8763) 
v_mul_f32 v33, v19, v29 // (op 8763) 
v_mul_f32 v34, v19, v30 // (op 8763) 
v_mul_f32 v35, v19, v31 // (op 8763) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8768) 
v_lshrrev_b32 v21, 16, v24 // (op 8768) 
v_cvt_f32_f16 v21, v21 // (op 8768) 
v_mul_f32 v28, v18, v20 // (op 8768) 
v_mul_f32 v29, v18, v21 // (op 8768) 
v_cvt_f32_f16 v20, v25 // (op 8768) 
v_lshrrev_b32 v21, 16, v25 // (op 8768) 
v_cvt_f32_f16 v21, v21 // (op 8768) 
v_mul_f32 v30, v18, v20 // (op 8768) 
v_mul_f32 v31, v18, v21 // (op 8768) 


v_add_f32 v36, v28, v32 // (op 8764) 
v_add_f32 v37, v29, v33 // (op 8764) 
v_add_f32 v38, v30, v34 // (op 8764) 
v_add_f32 v39, v31, v35 // (op 8764) 


v_cvt_f16_f32 v28, v36 // (op 8765) 
v_cvt_f16_f32 v29, v37 // (op 8765) 
v_cvt_f16_f32 v30, v38 // (op 8765) 
v_cvt_f16_f32 v31, v39 // (op 8765) 


v_pack_b32_f16 v20, v28, v29 // (op 8766) 
v_pack_b32_f16 v21, v30, v31 // (op 8766) 

s_lshl_b32 s52, s24, 6 // (op 8766) 
v_add_u32 v23, v27, s52 // (op 8766) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8766) Store value





s_lshl_b32 s52, s4, 7 // (op 8781) 
v_add_u32 v20, v22, s52 // (op 8781) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8781) Load value

v_accvgpr_read v28, a8 // (op 8777) 
v_accvgpr_read v29, a9 // (op 8777) 
v_accvgpr_read v30, a10 // (op 8777) 
v_accvgpr_read v31, a11 // (op 8777) 
v_mul_f32 v32, v19, v28 // (op 8777) 
v_mul_f32 v33, v19, v29 // (op 8777) 
v_mul_f32 v34, v19, v30 // (op 8777) 
v_mul_f32 v35, v19, v31 // (op 8777) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8782) 
v_lshrrev_b32 v21, 16, v24 // (op 8782) 
v_cvt_f32_f16 v21, v21 // (op 8782) 
v_mul_f32 v28, v18, v20 // (op 8782) 
v_mul_f32 v29, v18, v21 // (op 8782) 
v_cvt_f32_f16 v20, v25 // (op 8782) 
v_lshrrev_b32 v21, 16, v25 // (op 8782) 
v_cvt_f32_f16 v21, v21 // (op 8782) 
v_mul_f32 v30, v18, v20 // (op 8782) 
v_mul_f32 v31, v18, v21 // (op 8782) 


v_add_f32 v36, v28, v32 // (op 8778) 
v_add_f32 v37, v29, v33 // (op 8778) 
v_add_f32 v38, v30, v34 // (op 8778) 
v_add_f32 v39, v31, v35 // (op 8778) 


v_cvt_f16_f32 v28, v36 // (op 8779) 
v_cvt_f16_f32 v29, v37 // (op 8779) 
v_cvt_f16_f32 v30, v38 // (op 8779) 
v_cvt_f16_f32 v31, v39 // (op 8779) 


v_pack_b32_f16 v20, v28, v29 // (op 8780) 
v_pack_b32_f16 v21, v30, v31 // (op 8780) 

s_lshl_b32 s52, s24, 7 // (op 8780) 
v_add_u32 v23, v27, s52 // (op 8780) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8780) Store value





s_lshl_b32 s52, s4, 6 // (op 8795) 
s_mul_i32 s53, 3, s52 // (op 8795) 

v_add_u32 v20, v22, s53 // (op 8795) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 8795) Load value

v_accvgpr_read v28, a12 // (op 8791) 
v_accvgpr_read v29, a13 // (op 8791) 
v_accvgpr_read v30, a14 // (op 8791) 
v_accvgpr_read v31, a15 // (op 8791) 
v_mul_f32 v32, v19, v28 // (op 8791) 
v_mul_f32 v33, v19, v29 // (op 8791) 
v_mul_f32 v34, v19, v30 // (op 8791) 
v_mul_f32 v35, v19, v31 // (op 8791) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 8796) 
v_lshrrev_b32 v21, 16, v24 // (op 8796) 
v_cvt_f32_f16 v21, v21 // (op 8796) 
v_mul_f32 v28, v18, v20 // (op 8796) 
v_mul_f32 v29, v18, v21 // (op 8796) 
v_cvt_f32_f16 v20, v25 // (op 8796) 
v_lshrrev_b32 v21, 16, v25 // (op 8796) 
v_cvt_f32_f16 v21, v21 // (op 8796) 
v_mul_f32 v30, v18, v20 // (op 8796) 
v_mul_f32 v31, v18, v21 // (op 8796) 


v_add_f32 v36, v28, v32 // (op 8792) 
v_add_f32 v37, v29, v33 // (op 8792) 
v_add_f32 v38, v30, v34 // (op 8792) 
v_add_f32 v39, v31, v35 // (op 8792) 


v_cvt_f16_f32 v28, v36 // (op 8793) 
v_cvt_f16_f32 v29, v37 // (op 8793) 
v_cvt_f16_f32 v30, v38 // (op 8793) 
v_cvt_f16_f32 v31, v39 // (op 8793) 


v_pack_b32_f16 v20, v28, v29 // (op 8794) 
v_pack_b32_f16 v21, v30, v31 // (op 8794) 

s_lshl_b32 s52, s24, 6 // (op 8794) 
s_mul_i32 s53, 3, s52 // (op 8794) 

v_add_u32 v23, v27, s53 // (op 8794) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 8794) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:


s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s0, 0 // (op 450) call()
s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 18511) 
s_mul_i32 s15, s16, s50 // (op 18511) 
s_add_u32 s15, s15, s0 // (op 18511) 
v_and_b32 v3, 63, v1 // (op 18511) 
s_add_u32 s6, s15, s9 // (op 18511) 

v_lshrrev_b32 v4, 4, v3 // (op 18511) 
s_mul_hi_u32 s15, s6, s13 // (op 18511) 
v_lshlrev_b32 v5, 2, v4 // (op 18511) 

s_sub_u32 s7, s6, s15 // (op 18511) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 18511) 


s_lshr_b32 s8, s7, 1 // (op 18511) 

v_lshrrev_b32 v0, 2, v3 // (op 18511) 

s_add_u32 s7, s8, s15 // (op 18511) 


v_and_b32 v3, 3, v0 // (op 18511) 

s_lshr_b32 s8, s7, s11 // (op 18511) 

v_and_b32 v0, 3, v1 // (op 18511) 
s_and_b32 s7, s11, 2147483648 // (op 18511) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 18511) 


s_cmp_eq_i32 s7, 0 // (op 18511) 
s_mov_b32 s15, scc // (op 18511) 

v_lshrrev_b32 v0, 6, v1 // (op 18511) 
s_cmp_eq_i32 s15, 0 // (op 18511) 
s_cselect_b32 s7, s6, s8 // (op 18511) 



v_and_b32 v3, 1, v0 // (op 18511) 

s_mul_hi_u32 s6, s7, s12 // (op 18511) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 18511) 


s_sub_u32 s8, s7, s6 // (op 18511) 
s_lshr_b32 s15, s8, 1 // (op 18511) 

s_add_u32 s8, s15, s6 // (op 18511) 


s_lshr_b32 s6, s8, s10 // (op 18511) 

s_and_b32 s8, s10, 2147483648 // (op 18511) 
s_cmp_eq_i32 s8, 0 // (op 18511) 
s_mov_b32 s15, scc // (op 18511) 

s_cmp_eq_i32 s15, 0 // (op 18511) 
s_cselect_b32 s8, s7, s6 // (op 18511) 


s_mul_i32 s6, s8, s14 // (op 18511) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 18511) 


s_sub_u32 s8, s7, s6 // (op 18511) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 18511) 


v_mul_lo_u32 v0, v4, s4 // (op 18511) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 18511) 


s_mul_i32 s19, s4, s23 // (op 18513) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 18513) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 18513) least significant
s_mul_hi_u32 s6, s4, s22 // (op 18513) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18513) most significant: sum
s_add_u32 s19, s19, s6 // (op 18513) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 18513) 

s_mov_b32 s56, s20 // (op 18513) reg expression
s_mov_b32 s57, s21 // (op 18513) reg expression
s_mov_b32 s58, s28 // (op 18513) reg expression
s_mov_b32 s59, 131072 // (op 18513) call()

v_lshrrev_b32 v0, 7, v1 // (op 18865) 
s_mul_i32 s15, s16, s50 // (op 18865) 
s_add_u32 s15, s15, s0 // (op 18865) 
v_and_b32 v3, 63, v1 // (op 18865) 
s_add_u32 s6, s15, s9 // (op 18865) 

v_lshrrev_b32 v5, 4, v3 // (op 18865) 
s_mul_hi_u32 s15, s6, s13 // (op 18865) 
v_lshlrev_b32 v14, 2, v5 // (op 18865) 

s_sub_u32 s7, s6, s15 // (op 18865) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 18865) 


s_lshr_b32 s8, s7, 1 // (op 18865) 

v_lshrrev_b32 v0, 2, v3 // (op 18865) 

s_add_u32 s7, s8, s15 // (op 18865) 


v_and_b32 v3, 3, v0 // (op 18865) 

s_lshr_b32 s8, s7, s11 // (op 18865) 

v_and_b32 v0, 3, v1 // (op 18865) 
s_and_b32 s7, s11, 2147483648 // (op 18865) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 18865) 


s_cmp_eq_i32 s7, 0 // (op 18865) 
s_mov_b32 s15, scc // (op 18865) 

v_lshrrev_b32 v0, 6, v1 // (op 18865) 
s_cmp_eq_i32 s15, 0 // (op 18865) 
s_cselect_b32 s7, s6, s8 // (op 18865) 



v_and_b32 v3, 1, v0 // (op 18865) 

s_mul_hi_u32 s6, s7, s12 // (op 18865) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 18865) 


s_sub_u32 s8, s7, s6 // (op 18865) 
s_lshr_b32 s15, s8, 1 // (op 18865) 

s_add_u32 s8, s15, s6 // (op 18865) 


s_lshr_b32 s6, s8, s10 // (op 18865) 

s_and_b32 s8, s10, 2147483648 // (op 18865) 
s_cmp_eq_i32 s8, 0 // (op 18865) 
s_mov_b32 s15, scc // (op 18865) 

s_cmp_eq_i32 s15, 0 // (op 18865) 
s_cselect_b32 s8, s7, s6 // (op 18865) 


s_mul_i32 s6, s8, s14 // (op 18865) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 18865) 


s_sub_u32 s8, s7, s6 // (op 18865) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 18865) 


v_mul_lo_u32 v0, v5, s24 // (op 18865) 

v_add_lshl_u32 v5, v3, v0, 1 // (op 18865) 


s_mul_i32 s19, s24, s23 // (op 18867) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 18867) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 18867) least significant
s_mul_hi_u32 s6, s24, s22 // (op 18867) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18867) most significant: sum
s_add_u32 s19, s19, s6 // (op 18867) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 18867) 

s_mov_b32 s60, s26 // (op 18867) reg expression
s_mov_b32 s61, s27 // (op 18867) reg expression
s_mov_b32 s62, s28 // (op 18867) reg expression
s_mov_b32 s63, 131072 // (op 18867) call()

v_and_b32 v0, 63, v1 // (op 18944) 
v_and_b32 v3, 48, v1 // (op 18944) 
v_sub_u32 v14, v0, v3 // (op 18944) 

v_lshrrev_b32 v3, 4, v0 // (op 18944) 

v_lshl_add_u32 v0, v3, 5, v14 // (op 18944) 


v_lshrrev_b32 v3, 7, v1 // (op 18944) 
v_lshlrev_b32 v14, 4, v3 // (op 18944) 

v_lshl_add_u32 v3, v0, 7, v14 // (op 18944) 


v_lshlrev_b32 v0, 2, v3 // (op 18944) 

v_lshrrev_b32 v3, 3, v0 // (op 18944) 

v_and_b32 v0, 48, v1 // (op 19019) 
v_and_b32 v14, 63, v1 // (op 19019) 
v_sub_u32 v15, v14, v0 // (op 19019) 

v_lshrrev_b32 v14, 1, v15 // (op 19019) 
v_add_lshl_u32 v16, v0, v14, 7 // (op 19019) 


v_and_b32 v0, 1, v15 // (op 19019) 

v_lshlrev_b32 v17, 3, v0 // (op 19019) 

v_lshrrev_b32 v0, 6, v1 // (op 19019) 
v_and_b32 v14, 1, v0 // (op 19019) 

v_lshl_add_u32 v0, v14, 4, v17 // (op 19019) 


v_add_u32 v17, v16, v0 // (op 19019) 


v_lshrrev_b32 v0, 7, v1 // (op 19095) 
v_and_b32 v14, 15, v1 // (op 19095) 
v_lshl_add_u32 v15, v0, 4, v14 // (op 19095) 


v_and_b32 v0, 63, v1 // (op 19095) 
v_lshrrev_b32 v14, 4, v0 // (op 19095) 

v_lshl_add_u32 v0, v14, 7, v15 // (op 19095) 


v_lshrrev_b32 v14, 6, v1 // (op 19171) 
v_and_b32 v15, 1, v14 // (op 19171) 

v_and_b32 v14, 15, v1 // (op 19171) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 19171) 


v_and_b32 v14, 63, v1 // (op 19171) 
v_lshrrev_b32 v15, 4, v14 // (op 19171) 

v_lshl_add_u32 v14, v15, 7, v16 // (op 19171) 


v_accvgpr_write a0, 0.00000 // (op 184) call()
v_accvgpr_write a1, 0.00000 // (op 184) call()
v_accvgpr_write a2, 0.00000 // (op 184) call()
v_accvgpr_write a3, 0.00000 // (op 184) call()
v_accvgpr_write a4, 0.00000 // (op 622) call()
v_accvgpr_write a5, 0.00000 // (op 622) call()
v_accvgpr_write a6, 0.00000 // (op 622) call()
v_accvgpr_write a7, 0.00000 // (op 622) call()
v_accvgpr_write a8, 0.00000 // (op 798) call()
v_accvgpr_write a9, 0.00000 // (op 798) call()
v_accvgpr_write a10, 0.00000 // (op 798) call()
v_accvgpr_write a11, 0.00000 // (op 798) call()
v_accvgpr_write a12, 0.00000 // (op 974) call()
v_accvgpr_write a13, 0.00000 // (op 974) call()
v_accvgpr_write a14, 0.00000 // (op 974) call()
v_accvgpr_write a15, 0.00000 // (op 974) call()
v_accvgpr_write a16, 0.00000 // (op 1238) call()
v_accvgpr_write a17, 0.00000 // (op 1238) call()
v_accvgpr_write a18, 0.00000 // (op 1238) call()
v_accvgpr_write a19, 0.00000 // (op 1238) call()
v_accvgpr_write a20, 0.00000 // (op 1547) call()
v_accvgpr_write a21, 0.00000 // (op 1547) call()
v_accvgpr_write a22, 0.00000 // (op 1547) call()
v_accvgpr_write a23, 0.00000 // (op 1547) call()
v_accvgpr_write a24, 0.00000 // (op 1548) call()
v_accvgpr_write a25, 0.00000 // (op 1548) call()
v_accvgpr_write a26, 0.00000 // (op 1548) call()
v_accvgpr_write a27, 0.00000 // (op 1548) call()
v_accvgpr_write a28, 0.00000 // (op 1549) call()
v_accvgpr_write a29, 0.00000 // (op 1549) call()
v_accvgpr_write a30, 0.00000 // (op 1549) call()
v_accvgpr_write a31, 0.00000 // (op 1549) call()
v_accvgpr_write a32, 0.00000 // (op 2008) call()
v_accvgpr_write a33, 0.00000 // (op 2008) call()
v_accvgpr_write a34, 0.00000 // (op 2008) call()
v_accvgpr_write a35, 0.00000 // (op 2008) call()
v_accvgpr_write a36, 0.00000 // (op 2317) call()
v_accvgpr_write a37, 0.00000 // (op 2317) call()
v_accvgpr_write a38, 0.00000 // (op 2317) call()
v_accvgpr_write a39, 0.00000 // (op 2317) call()
v_accvgpr_write a40, 0.00000 // (op 2318) call()
v_accvgpr_write a41, 0.00000 // (op 2318) call()
v_accvgpr_write a42, 0.00000 // (op 2318) call()
v_accvgpr_write a43, 0.00000 // (op 2318) call()
v_accvgpr_write a44, 0.00000 // (op 2319) call()
v_accvgpr_write a45, 0.00000 // (op 2319) call()
v_accvgpr_write a46, 0.00000 // (op 2319) call()
v_accvgpr_write a47, 0.00000 // (op 2319) call()
v_accvgpr_write a48, 0.00000 // (op 2778) call()
v_accvgpr_write a49, 0.00000 // (op 2778) call()
v_accvgpr_write a50, 0.00000 // (op 2778) call()
v_accvgpr_write a51, 0.00000 // (op 2778) call()
v_accvgpr_write a52, 0.00000 // (op 3087) call()
v_accvgpr_write a53, 0.00000 // (op 3087) call()
v_accvgpr_write a54, 0.00000 // (op 3087) call()
v_accvgpr_write a55, 0.00000 // (op 3087) call()
v_accvgpr_write a56, 0.00000 // (op 3088) call()
v_accvgpr_write a57, 0.00000 // (op 3088) call()
v_accvgpr_write a58, 0.00000 // (op 3088) call()
v_accvgpr_write a59, 0.00000 // (op 3088) call()
v_accvgpr_write a60, 0.00000 // (op 3089) call()
v_accvgpr_write a61, 0.00000 // (op 3089) call()
v_accvgpr_write a62, 0.00000 // (op 3089) call()
v_accvgpr_write a63, 0.00000 // (op 3089) call()
v_and_b32 v15, -4, v1 // (op 17991) 
s_mul_i32 s15, s16, s50 // (op 17991) 
s_add_u32 s15, s15, s0 // (op 17991) 
v_sub_u32 v16, v1, v15 // (op 17991) 

s_add_u32 s6, s15, s9 // (op 17991) 

v_lshlrev_b32 v15, 5, v16 // (op 17991) 

s_mul_hi_u32 s15, s6, s13 // (op 17991) 
v_lshrrev_b32 v16, 2, v1 // (op 17991) 
s_sub_u32 s7, s6, s15 // (op 17991) 
s_lshr_b32 s8, s7, 1 // (op 17991) 

s_add_u32 s7, s8, s15 // (op 17991) 


s_lshr_b32 s8, s7, s11 // (op 17991) 

s_and_b32 s7, s11, 2147483648 // (op 17991) 
s_cmp_eq_i32 s7, 0 // (op 17991) 
s_mov_b32 s15, scc // (op 17991) 

s_cmp_eq_i32 s15, 0 // (op 17991) 
s_cselect_b32 s7, s6, s8 // (op 17991) 


s_mul_hi_u32 s8, s7, s12 // (op 17991) 
s_sub_u32 s15, s7, s8 // (op 17991) 
s_lshr_b32 s18, s15, 1 // (op 17991) 

s_add_u32 s15, s18, s8 // (op 17991) 


s_lshr_b32 s8, s15, s10 // (op 17991) 

s_and_b32 s15, s10, 2147483648 // (op 17991) 
s_cmp_eq_i32 s15, 0 // (op 17991) 
s_mov_b32 s18, scc // (op 17991) 

s_cmp_eq_i32 s18, 0 // (op 17991) 
s_cselect_b32 s15, s7, s8 // (op 17991) 


v_lshl_add_u32 v20, s15, 7, v15 // (op 17991) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 17991) 
v_mov_b32 v23, 0 // (op 17991) convert
v_mov_b32 v22, v20 // (op 17991) convert

s_mul_i32 s8, s7, s18 // (op 17991) 


s_sub_u32 s7, s6, s8 // (op 17991) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 17991) 


v_mul_lo_u32 v21, v15, s31 // (op 17991) most significant: low of low * high
v_mul_hi_u32 v16, v15, s30 // (op 17991) most significant: high of low * low
v_mul_lo_u32 v20, v15, s30 // (op 17991) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 17991) most significant: sum


v_add_co_u32 v24, s[18:19], v22, v20 // (op 17991) least significant half
v_addc_co_u32 v25, s[18:19], v23, v21, s[18:19] // (op 17991) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 17991) 


v_lshrrev_b64 v[20:21], 3, v[24:25] // (op 17991) 

s_mul_i32 s19, s30, s3 // (op 17993) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 17993) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 17993) least significant
s_mul_hi_u32 s6, s30, s2 // (op 17993) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17993) most significant: sum
s_add_u32 s19, s19, s6 // (op 17993) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17993) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 17993) 

s_mov_b32 s68, s32 // (op 17993) reg expression
s_mov_b32 s69, s33 // (op 17993) reg expression
s_mov_b32 s70, s18 // (op 17993) reg expression
s_mov_b32 s71, 131072 // (op 17993) call()

v_and_b32 v15, -4, v1 // (op 18010) 
s_mul_i32 s15, s16, s50 // (op 18010) 
s_add_u32 s15, s15, s0 // (op 18010) 
v_sub_u32 v16, v1, v15 // (op 18010) 

s_add_u32 s6, s15, s9 // (op 18010) 

v_lshlrev_b32 v15, 5, v16 // (op 18010) 

s_mul_hi_u32 s15, s6, s13 // (op 18010) 
v_lshrrev_b32 v16, 2, v1 // (op 18010) 
s_sub_u32 s7, s6, s15 // (op 18010) 
s_lshr_b32 s8, s7, 1 // (op 18010) 

s_add_u32 s7, s8, s15 // (op 18010) 


s_lshr_b32 s8, s7, s11 // (op 18010) 

s_and_b32 s7, s11, 2147483648 // (op 18010) 
s_cmp_eq_i32 s7, 0 // (op 18010) 
s_mov_b32 s15, scc // (op 18010) 

s_cmp_eq_i32 s15, 0 // (op 18010) 
s_cselect_b32 s7, s6, s8 // (op 18010) 


s_mul_hi_u32 s8, s7, s12 // (op 18010) 
s_sub_u32 s15, s7, s8 // (op 18010) 
s_lshr_b32 s18, s15, 1 // (op 18010) 

s_add_u32 s15, s18, s8 // (op 18010) 


s_lshr_b32 s8, s15, s10 // (op 18010) 

s_and_b32 s15, s10, 2147483648 // (op 18010) 
s_cmp_eq_i32 s15, 0 // (op 18010) 
s_mov_b32 s18, scc // (op 18010) 

s_cmp_eq_i32 s18, 0 // (op 18010) 
s_cselect_b32 s15, s7, s8 // (op 18010) 


v_lshl_add_u32 v22, s15, 7, v15 // (op 18010) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18010) 
v_mov_b32 v25, 0 // (op 18010) convert
v_mov_b32 v24, v22 // (op 18010) convert

s_mul_i32 s8, s7, s18 // (op 18010) 


s_sub_u32 s7, s6, s8 // (op 18010) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 18010) 


v_mul_lo_u32 v23, v15, s31 // (op 18010) most significant: low of low * high
v_mul_hi_u32 v16, v15, s30 // (op 18010) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 18010) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 18010) most significant: sum


v_add_co_u32 v26, s[18:19], v24, v22 // (op 18010) least significant half
v_addc_co_u32 v27, s[18:19], v25, v23, s[18:19] // (op 18010) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 18010) 


v_lshrrev_b64 v[22:23], 3, v[26:27] // (op 18010) 

s_mul_i32 s19, s30, s3 // (op 18012) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 18012) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 18012) least significant
s_mul_hi_u32 s6, s30, s2 // (op 18012) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18012) most significant: sum
s_add_u32 s19, s19, s6 // (op 18012) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 18012) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 18012) 

s_mov_b32 s68, s32 // (op 18012) reg expression
s_mov_b32 s69, s33 // (op 18012) reg expression
s_mov_b32 s70, s18 // (op 18012) reg expression
s_mov_b32 s71, 131072 // (op 18012) call()

v_and_b32 v15, -8, v1 // (op 18137) 
s_mul_i32 s15, s16, s50 // (op 18137) 
s_add_u32 s15, s15, s0 // (op 18137) 
v_sub_u32 v16, v1, v15 // (op 18137) 

s_add_u32 s6, s15, s9 // (op 18137) 

v_lshlrev_b32 v15, 4, v16 // (op 18137) 

s_mul_hi_u32 s15, s6, s13 // (op 18137) 
v_lshrrev_b32 v16, 3, v1 // (op 18137) 
s_sub_u32 s7, s6, s15 // (op 18137) 
s_lshr_b32 s8, s7, 1 // (op 18137) 

s_add_u32 s7, s8, s15 // (op 18137) 


s_lshr_b32 s8, s7, s11 // (op 18137) 

s_and_b32 s7, s11, 2147483648 // (op 18137) 
s_cmp_eq_i32 s7, 0 // (op 18137) 
s_mov_b32 s15, scc // (op 18137) 

s_cmp_eq_i32 s15, 0 // (op 18137) 
s_cselect_b32 s7, s6, s8 // (op 18137) 


s_mul_hi_u32 s8, s7, s12 // (op 18137) 
s_sub_u32 s15, s7, s8 // (op 18137) 
s_lshr_b32 s18, s15, 1 // (op 18137) 

s_add_u32 s15, s18, s8 // (op 18137) 


s_lshr_b32 s8, s15, s10 // (op 18137) 

s_and_b32 s15, s10, 2147483648 // (op 18137) 
s_cmp_eq_i32 s15, 0 // (op 18137) 
s_mov_b32 s18, scc // (op 18137) 

s_cmp_eq_i32 s18, 0 // (op 18137) 
s_cselect_b32 s15, s7, s8 // (op 18137) 


s_mul_i32 s8, s15, s14 // (op 18137) 

s_sub_u32 s15, s7, s8 // (op 18137) 

v_lshl_add_u32 v24, s15, 7, v15 // (op 18137) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18137) 
v_mov_b32 v27, 0 // (op 18137) convert
v_mov_b32 v26, v24 // (op 18137) convert

s_mul_i32 s8, s7, s18 // (op 18137) 


s_sub_u32 s7, s6, s8 // (op 18137) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 18137) 


v_mul_lo_u32 v25, v15, s35 // (op 18137) most significant: low of low * high
v_mul_hi_u32 v16, v15, s34 // (op 18137) most significant: high of low * low
v_mul_lo_u32 v24, v15, s34 // (op 18137) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 18137) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v26 // (op 18137) least significant half
v_addc_co_u32 v25, s[18:19], v25, v27, s[18:19] // (op 18137) most significant half



s_mul_i32 s19, s34, s3 // (op 18139) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 18139) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 18139) least significant
s_mul_hi_u32 s6, s34, s2 // (op 18139) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18139) most significant: sum
s_add_u32 s19, s19, s6 // (op 18139) most significant: sum


s_mov_b32 s72, s36 // (op 18139) reg expression
s_mov_b32 s73, s37 // (op 18139) reg expression
s_mov_b32 s74, s18 // (op 18139) reg expression
s_mov_b32 s75, 131072 // (op 18139) call()

v_and_b32 v15, -8, v1 // (op 18156) 
s_mul_i32 s15, s16, s50 // (op 18156) 
s_add_u32 s15, s15, s0 // (op 18156) 
v_sub_u32 v16, v1, v15 // (op 18156) 

s_add_u32 s6, s15, s9 // (op 18156) 

v_lshlrev_b32 v15, 4, v16 // (op 18156) 

s_mul_hi_u32 s15, s6, s13 // (op 18156) 
v_lshrrev_b32 v16, 3, v1 // (op 18156) 
s_sub_u32 s7, s6, s15 // (op 18156) 
s_lshr_b32 s8, s7, 1 // (op 18156) 

s_add_u32 s7, s8, s15 // (op 18156) 


s_lshr_b32 s8, s7, s11 // (op 18156) 

s_and_b32 s7, s11, 2147483648 // (op 18156) 
s_cmp_eq_i32 s7, 0 // (op 18156) 
s_mov_b32 s15, scc // (op 18156) 

s_cmp_eq_i32 s15, 0 // (op 18156) 
s_cselect_b32 s7, s6, s8 // (op 18156) 


s_mul_hi_u32 s8, s7, s12 // (op 18156) 
s_sub_u32 s15, s7, s8 // (op 18156) 
s_lshr_b32 s18, s15, 1 // (op 18156) 

s_add_u32 s15, s18, s8 // (op 18156) 


s_lshr_b32 s8, s15, s10 // (op 18156) 

s_and_b32 s15, s10, 2147483648 // (op 18156) 
s_cmp_eq_i32 s15, 0 // (op 18156) 
s_mov_b32 s18, scc // (op 18156) 

s_cmp_eq_i32 s18, 0 // (op 18156) 
s_cselect_b32 s15, s7, s8 // (op 18156) 


s_mul_i32 s8, s15, s14 // (op 18156) 

s_sub_u32 s15, s7, s8 // (op 18156) 

v_lshl_add_u32 v26, s15, 7, v15 // (op 18156) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18156) 
v_mov_b32 v29, 0 // (op 18156) convert
v_mov_b32 v28, v26 // (op 18156) convert

s_mul_i32 s8, s7, s18 // (op 18156) 


s_sub_u32 s7, s6, s8 // (op 18156) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 18156) 


v_mul_lo_u32 v27, v15, s35 // (op 18156) most significant: low of low * high
v_mul_hi_u32 v16, v15, s34 // (op 18156) most significant: high of low * low
v_mul_lo_u32 v26, v15, s34 // (op 18156) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 18156) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v28 // (op 18156) least significant half
v_addc_co_u32 v27, s[18:19], v27, v29, s[18:19] // (op 18156) most significant half



s_mul_i32 s19, s34, s3 // (op 18158) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 18158) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 18158) least significant
s_mul_hi_u32 s6, s34, s2 // (op 18158) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18158) most significant: sum
s_add_u32 s19, s19, s6 // (op 18158) most significant: sum


s_mov_b32 s72, s36 // (op 18158) reg expression
s_mov_b32 s73, s37 // (op 18158) reg expression
s_mov_b32 s74, s18 // (op 18158) reg expression
s_mov_b32 s75, 131072 // (op 18158) call()

v_and_b32 v15, -128, v1 // (op 18265) 
s_mul_i32 s15, s16, s50 // (op 18265) 
s_add_u32 s15, s15, s0 // (op 18265) 
v_sub_u32 v16, v1, v15 // (op 18265) 

s_add_u32 s6, s15, s9 // (op 18265) 

v_lshrrev_b32 v15, 7, v1 // (op 18265) 
s_mul_hi_u32 s15, s6, s13 // (op 18265) 
v_lshlrev_b32 v28, 1, v15 // (op 18265) 

s_sub_u32 s7, s6, s15 // (op 18265) 
s_lshr_b32 s8, s7, 1 // (op 18265) 

s_add_u32 s7, s8, s15 // (op 18265) 


s_lshr_b32 s8, s7, s11 // (op 18265) 

s_and_b32 s7, s11, 2147483648 // (op 18265) 
s_cmp_eq_i32 s7, 0 // (op 18265) 
s_mov_b32 s15, scc // (op 18265) 

s_cmp_eq_i32 s15, 0 // (op 18265) 
s_cselect_b32 s7, s6, s8 // (op 18265) 


s_mul_hi_u32 s8, s7, s12 // (op 18265) 
s_sub_u32 s15, s7, s8 // (op 18265) 
s_lshr_b32 s18, s15, 1 // (op 18265) 

s_add_u32 s15, s18, s8 // (op 18265) 


s_lshr_b32 s8, s15, s10 // (op 18265) 

s_and_b32 s15, s10, 2147483648 // (op 18265) 
s_cmp_eq_i32 s15, 0 // (op 18265) 
s_mov_b32 s18, scc // (op 18265) 

s_cmp_eq_i32 s18, 0 // (op 18265) 
s_cselect_b32 s15, s7, s8 // (op 18265) 


v_lshl_add_u32 v15, s15, 7, v16 // (op 18265) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18265) 
s_mul_i32 s8, s7, s18 // (op 18265) 


s_sub_u32 s7, s6, s8 // (op 18265) 


v_lshl_add_u32 v16, s7, 2, v28 // (op 18265) 


v_mul_lo_u32 v28, v16, s38 // (op 18265) 

v_add_u32 v16, v15, v28 // (op 18265) 


s_mul_i32 s19, s38, s43 // (op 18267) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18267) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18267) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18267) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18267) most significant: sum
s_add_u32 s19, s19, s6 // (op 18267) most significant: sum


s_mov_b32 s76, s40 // (op 18267) reg expression
s_mov_b32 s77, s41 // (op 18267) reg expression
s_mov_b32 s78, s18 // (op 18267) reg expression
s_mov_b32 s79, 131072 // (op 18267) call()

v_and_b32 v15, -128, v1 // (op 18284) 
s_mul_i32 s15, s16, s50 // (op 18284) 
s_add_u32 s15, s15, s0 // (op 18284) 
v_sub_u32 v28, v1, v15 // (op 18284) 

s_add_u32 s6, s15, s9 // (op 18284) 

v_lshrrev_b32 v15, 7, v1 // (op 18284) 
s_mul_hi_u32 s15, s6, s13 // (op 18284) 
v_lshlrev_b32 v29, 1, v15 // (op 18284) 

s_sub_u32 s7, s6, s15 // (op 18284) 
s_lshr_b32 s8, s7, 1 // (op 18284) 

s_add_u32 s7, s8, s15 // (op 18284) 


s_lshr_b32 s8, s7, s11 // (op 18284) 

s_and_b32 s7, s11, 2147483648 // (op 18284) 
s_cmp_eq_i32 s7, 0 // (op 18284) 
s_mov_b32 s15, scc // (op 18284) 

s_cmp_eq_i32 s15, 0 // (op 18284) 
s_cselect_b32 s7, s6, s8 // (op 18284) 


s_mul_hi_u32 s8, s7, s12 // (op 18284) 
s_sub_u32 s15, s7, s8 // (op 18284) 
s_lshr_b32 s18, s15, 1 // (op 18284) 

s_add_u32 s15, s18, s8 // (op 18284) 


s_lshr_b32 s8, s15, s10 // (op 18284) 

s_and_b32 s15, s10, 2147483648 // (op 18284) 
s_cmp_eq_i32 s15, 0 // (op 18284) 
s_mov_b32 s18, scc // (op 18284) 

s_cmp_eq_i32 s18, 0 // (op 18284) 
s_cselect_b32 s15, s7, s8 // (op 18284) 


v_lshl_add_u32 v15, s15, 7, v28 // (op 18284) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18284) 
s_mul_i32 s8, s7, s18 // (op 18284) 


s_sub_u32 s7, s6, s8 // (op 18284) 


v_lshl_add_u32 v28, s7, 2, v29 // (op 18284) 


v_mul_lo_u32 v29, v28, s38 // (op 18284) 

v_add_u32 v28, v15, v29 // (op 18284) 


s_mul_i32 s19, s38, s43 // (op 18286) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18286) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18286) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18286) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18286) most significant: sum
s_add_u32 s19, s19, s6 // (op 18286) most significant: sum


s_mov_b32 s76, s40 // (op 18286) reg expression
s_mov_b32 s77, s41 // (op 18286) reg expression
s_mov_b32 s78, s18 // (op 18286) reg expression
s_mov_b32 s79, 131072 // (op 18286) call()

v_and_b32 v15, -128, v1 // (op 18393) 
s_mul_i32 s15, s16, s50 // (op 18393) 
s_add_u32 s15, s15, s0 // (op 18393) 
v_sub_u32 v29, v1, v15 // (op 18393) 

s_add_u32 s6, s15, s9 // (op 18393) 

v_lshrrev_b32 v15, 7, v1 // (op 18393) 
s_mul_hi_u32 s15, s6, s13 // (op 18393) 
v_lshlrev_b32 v30, 1, v15 // (op 18393) 

s_sub_u32 s7, s6, s15 // (op 18393) 
s_lshr_b32 s8, s7, 1 // (op 18393) 

s_add_u32 s7, s8, s15 // (op 18393) 


s_lshr_b32 s8, s7, s11 // (op 18393) 

s_and_b32 s7, s11, 2147483648 // (op 18393) 
s_cmp_eq_i32 s7, 0 // (op 18393) 
s_mov_b32 s15, scc // (op 18393) 

s_cmp_eq_i32 s15, 0 // (op 18393) 
s_cselect_b32 s7, s6, s8 // (op 18393) 


s_mul_hi_u32 s8, s7, s12 // (op 18393) 
s_sub_u32 s15, s7, s8 // (op 18393) 
s_lshr_b32 s18, s15, 1 // (op 18393) 

s_add_u32 s15, s18, s8 // (op 18393) 


s_lshr_b32 s8, s15, s10 // (op 18393) 

s_and_b32 s15, s10, 2147483648 // (op 18393) 
s_cmp_eq_i32 s15, 0 // (op 18393) 
s_mov_b32 s18, scc // (op 18393) 

s_cmp_eq_i32 s18, 0 // (op 18393) 
s_cselect_b32 s15, s7, s8 // (op 18393) 


s_mul_i32 s8, s15, s14 // (op 18393) 

s_sub_u32 s15, s7, s8 // (op 18393) 

v_lshl_add_u32 v15, s15, 7, v29 // (op 18393) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18393) 
s_mul_i32 s8, s7, s18 // (op 18393) 


s_sub_u32 s7, s6, s8 // (op 18393) 


v_lshl_add_u32 v29, s7, 2, v30 // (op 18393) 


v_mul_lo_u32 v30, v29, s44 // (op 18393) 
v_add_u32 v30, v30, v15 // (op 18393) 


s_mul_i32 s19, s44, s43 // (op 18395) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 18395) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 18395) least significant
s_mul_hi_u32 s6, s44, s42 // (op 18395) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18395) most significant: sum
s_add_u32 s19, s19, s6 // (op 18395) most significant: sum


s_mov_b32 s80, s46 // (op 18395) reg expression
s_mov_b32 s81, s47 // (op 18395) reg expression
s_mov_b32 s82, s18 // (op 18395) reg expression
s_mov_b32 s83, 131072 // (op 18395) call()

v_and_b32 v15, -128, v1 // (op 18412) 
s_mul_i32 s15, s16, s50 // (op 18412) 
s_add_u32 s15, s15, s0 // (op 18412) 
v_sub_u32 v29, v1, v15 // (op 18412) 

s_add_u32 s6, s15, s9 // (op 18412) 

v_lshrrev_b32 v15, 7, v1 // (op 18412) 
s_mul_hi_u32 s15, s6, s13 // (op 18412) 
v_lshlrev_b32 v31, 1, v15 // (op 18412) 

s_sub_u32 s7, s6, s15 // (op 18412) 
s_lshr_b32 s8, s7, 1 // (op 18412) 

s_add_u32 s7, s8, s15 // (op 18412) 


s_lshr_b32 s8, s7, s11 // (op 18412) 

s_and_b32 s7, s11, 2147483648 // (op 18412) 
s_cmp_eq_i32 s7, 0 // (op 18412) 
s_mov_b32 s15, scc // (op 18412) 

s_cmp_eq_i32 s15, 0 // (op 18412) 
s_cselect_b32 s7, s6, s8 // (op 18412) 


s_mul_hi_u32 s8, s7, s12 // (op 18412) 
s_sub_u32 s15, s7, s8 // (op 18412) 
s_lshr_b32 s18, s15, 1 // (op 18412) 

s_add_u32 s15, s18, s8 // (op 18412) 


s_lshr_b32 s8, s15, s10 // (op 18412) 

s_and_b32 s15, s10, 2147483648 // (op 18412) 
s_cmp_eq_i32 s15, 0 // (op 18412) 
s_mov_b32 s18, scc // (op 18412) 

s_cmp_eq_i32 s18, 0 // (op 18412) 
s_cselect_b32 s15, s7, s8 // (op 18412) 


s_mul_i32 s8, s15, s14 // (op 18412) 

s_sub_u32 s15, s7, s8 // (op 18412) 

v_lshl_add_u32 v15, s15, 7, v29 // (op 18412) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18412) 
s_mul_i32 s8, s7, s18 // (op 18412) 


s_sub_u32 s7, s6, s8 // (op 18412) 


v_lshl_add_u32 v29, s7, 2, v31 // (op 18412) 


v_mul_lo_u32 v31, v29, s44 // (op 18412) 
v_add_u32 v31, v31, v15 // (op 18412) 


s_mul_i32 s19, s44, s43 // (op 18414) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 18414) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 18414) least significant
s_mul_hi_u32 s6, s44, s42 // (op 18414) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18414) most significant: sum
s_add_u32 s19, s19, s6 // (op 18414) most significant: sum


s_mov_b32 s80, s46 // (op 18414) reg expression
s_mov_b32 s81, s47 // (op 18414) reg expression
s_mov_b32 s82, s18 // (op 18414) reg expression
s_mov_b32 s83, 131072 // (op 18414) call()

v_and_b32 v15, -8, v1 // (op 18985) 
v_sub_u32 v29, v1, v15 // (op 18985) 

v_lshlrev_b32 v15, 4, v29 // (op 18985) 

v_lshrrev_b32 v29, 3, v1 // (op 18985) 
v_lshl_add_u32 v32, v29, 7, v15 // (op 18985) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 18985) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19849) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17887) 
v_mov_b64 v[32:33], s[18:19] // (op 17887) convert


v_mov_b64 v[34:35], v[24:25] // (op 17887) reg expression
s_mov_b32 m0, s15 // (op 17887) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17887) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17887) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17887) most significant half

s_add_u32 m0, m0, 4096 // (op 17887) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17887) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17887) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17887) most significant half

s_add_u32 m0, m0, 4096 // (op 17887) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17887) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17887) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17887) most significant half

s_add_u32 m0, m0, 4096 // (op 17887) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17887) Load value direct to lds




v_mov_b64 v[32:33], s[44:45] // (op 17299) convert

v_mov_b32 v15, v30 // (op 17299) reg expression
buffer_load_ubyte v34, v15, s[80:83], 0 offen offset: 0 // (op 17299) Load value
v_add_u32 v15, v15, v32 // (op 17299) 
buffer_load_ubyte v35, v15, s[80:83], 0 offen offset: 0 // (op 17299) Load value


v_and_b32 v15, -4, v1 // (op 18910) 
v_sub_u32 v29, v1, v15 // (op 18910) 

v_lshlrev_b32 v15, 5, v29 // (op 18910) 

v_lshrrev_b32 v29, 2, v1 // (op 18910) 
v_lshl_add_u32 v32, v29, 7, v15 // (op 18910) 


v_lshlrev_b32 v15, 2, v32 // (op 18910) 

v_lshrrev_b32 v29, 3, v15 // (op 18910) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 18910) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19830) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17892) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17892) 

v_lshrrev_b64 v[32:33], 3, s[28:29] // (op 17892) 


v_mov_b64 v[36:37], v[20:21] // (op 17892) reg expression
s_add_u32 s6, 16384, s15 // (op 17892) 
s_mov_b32 m0, s6 // (op 17892) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17892) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 17892) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 17892) most significant half

s_add_u32 m0, m0, 4096 // (op 17892) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17892) Load value direct to lds





v_mov_b64 v[32:33], s[38:39] // (op 17305) convert

v_mov_b32 v15, v16 // (op 17305) reg expression
buffer_load_ubyte v36, v15, s[76:79], 0 offen offset: 0 // (op 17305) Load value
v_add_u32 v15, v15, v32 // (op 17305) 
buffer_load_ubyte v37, v15, s[76:79], 0 offen offset: 0 // (op 17305) Load value


v_and_b32 v15, -8, v1 // (op 19295) 
v_sub_u32 v29, v1, v15 // (op 19295) 

v_lshlrev_b32 v15, 4, v29 // (op 19295) 

v_lshrrev_b32 v29, 3, v1 // (op 19295) 
v_lshl_add_u32 v32, v29, 7, v15 // (op 19295) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 19295) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19919) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17897) 
v_mov_b64 v[32:33], s[18:19] // (op 17897) convert


s_lshl_b64 s[18:19], s[34:35], 7 // (op 17897) 
v_mov_b32 v15, s18 // (op 17897) 
v_mov_b32 v29, s19 // (op 17897) 
v_add_co_u32 v38, s[28:29], v24, v15 // (op 17897) least significant half
v_addc_co_u32 v39, s[28:29], v25, v29, s[28:29] // (op 17897) most significant half




s_add_u32 s6, 24576, s15 // (op 17897) 
s_mov_b32 m0, s6 // (op 17897) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17897) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17897) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17897) most significant half

s_add_u32 m0, m0, 4096 // (op 17897) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17897) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17897) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17897) most significant half

s_add_u32 m0, m0, 4096 // (op 17897) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17897) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17897) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17897) most significant half

s_add_u32 m0, m0, 4096 // (op 17897) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 17897) Load value direct to lds






v_mov_b64 v[24:25], s[44:45] // (op 17311) convert

s_lshl_b32 s15, s44, 2 // (op 17311) 
v_add_u32 v15, v30, s15 // (op 17311) 

buffer_load_ubyte v32, v15, s[80:83], 0 offen offset: 0 // (op 17311) Load value
v_add_u32 v15, v15, v24 // (op 17311) 
buffer_load_ubyte v33, v15, s[80:83], 0 offen offset: 0 // (op 17311) Load value



v_and_b32 v15, -4, v1 // (op 19251) 
v_sub_u32 v24, v1, v15 // (op 19251) 

v_lshlrev_b32 v15, 5, v24 // (op 19251) 

v_lshrrev_b32 v24, 2, v1 // (op 19251) 
v_lshl_add_u32 v25, v24, 7, v15 // (op 19251) 


v_lshlrev_b32 v15, 2, v25 // (op 19251) 

v_lshrrev_b32 v24, 3, v15 // (op 19251) 

s_nop 0
v_readfirstlane_b32 s15, v24 // (op 19251) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19909) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17899) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17899) 

v_lshrrev_b64 v[24:25], 3, s[28:29] // (op 17899) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 17899) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17899) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17899) 

v_mov_b32 v15, s18 // (op 17899) 
v_mov_b32 v29, s19 // (op 17899) 
v_add_co_u32 v38, s[28:29], v20, v15 // (op 17899) least significant half
v_addc_co_u32 v39, s[28:29], v21, v29, s[28:29] // (op 17899) most significant half




s_add_u32 s6, 40960, s15 // (op 17899) 
s_mov_b32 m0, s6 // (op 17899) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17899) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v24 // (op 17899) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17899) most significant half

s_add_u32 m0, m0, 4096 // (op 17899) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17899) Load value direct to lds






v_mov_b64 v[20:21], s[38:39] // (op 17317) convert

s_lshl_b32 s15, s38, 2 // (op 17317) 
v_add_u32 v15, v16, s15 // (op 17317) 

buffer_load_ubyte v24, v15, s[76:79], 0 offen offset: 0 // (op 17317) Load value
v_add_u32 v15, v15, v20 // (op 17317) 
buffer_load_ubyte v25, v15, s[76:79], 0 offen offset: 0 // (op 17317) Load value



v_and_b32 v15, -128, v1 // (op 19137) 
v_sub_u32 v16, v1, v15 // (op 19137) 

v_lshrrev_b32 v15, 7, v1 // (op 19137) 
v_lshl_add_u32 v20, v15, 8, v16 // (op 19137) 


s_barrier  // (op 19889) 



s_waitcnt vmcnt(15)
ds_write_b8 v20, v34 offset:49152 // (op 17320) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v20, v35 offset:49280 // (op 17320) Store local data 

v_and_b32 v15, -128, v1 // (op 19061) 
v_sub_u32 v16, v1, v15 // (op 19061) 

v_lshrrev_b32 v15, 7, v1 // (op 19061) 
v_lshl_add_u32 v20, v15, 8, v16 // (op 19061) 


s_barrier  // (op 19869) 



s_waitcnt vmcnt(11)
ds_write_b8 v20, v36 offset:49664 // (op 17322) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v20, v37 offset:49792 // (op 17322) Store local data 

s_barrier  // (op 19846) 



ds_read_b64_tr_b8 v[38:39], v17 // (op 17357) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v17 offset:1024 // (op 17357) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v17 offset:8192 // (op 17357) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:9216 // (op 17357) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19886) 


ds_read_u8 v15, v14 offset:49152 // (op 17397) Load local data 



ds_read_b64_tr_b8 v[46:47], v17 offset:32 // (op 17362) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v17 offset:1056 // (op 17362) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v17 offset:8224 // (op 17362) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v17 offset:9248 // (op 17362) Transpose load local data 


ds_read_u8 v16, v14 offset:49184 // (op 17402) Load local data 



ds_read_b64_tr_b8 v[54:55], v17 offset:64 // (op 17367) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v17 offset:1088 // (op 17367) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v17 offset:8256 // (op 17367) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v17 offset:9280 // (op 17367) Transpose load local data 


ds_read_u8 v20, v14 offset:49216 // (op 17407) Load local data 



ds_read_b64_tr_b8 v[62:63], v17 offset:96 // (op 17372) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v17 offset:1120 // (op 17372) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v17 offset:8288 // (op 17372) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v17 offset:9312 // (op 17372) Transpose load local data 


ds_read_u8 v21, v14 offset:49248 // (op 17412) Load local data 
s_barrier  // (op 19827) 



ds_read_b64_tr_b4 v[70:71], v3 offset:16384 // (op 17337) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:17408 // (op 17337) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19866) 


ds_read_u8 v29, v0 offset:49664 // (op 17377) Load local data 



ds_read_b64_tr_b4 v[74:75], v3 offset:16400 // (op 17342) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:17424 // (op 17342) Transpose load local data 


ds_read_u8 v30, v0 offset:49696 // (op 17382) Load local data 



ds_read_b64_tr_b4 v[78:79], v3 offset:16416 // (op 17347) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:17440 // (op 17347) Transpose load local data 


ds_read_u8 v82, v0 offset:49728 // (op 17387) Load local data 



ds_read_b64_tr_b4 v[84:85], v3 offset:16432 // (op 17352) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:17456 // (op 17352) Transpose load local data 


ds_read_u8 v83, v0 offset:49760 // (op 17392) Load local data 
v_and_b32 v88, -4, v1 // (op 18882) 
v_sub_u32 v89, v1, v88 // (op 18882) 

v_lshlrev_b32 v88, 5, v89 // (op 18882) 

v_lshrrev_b32 v89, 2, v1 // (op 18882) 
v_lshl_add_u32 v90, v89, 7, v88 // (op 18882) 


v_lshlrev_b32 v88, 2, v90 // (op 18882) 

v_lshrrev_b32 v89, 3, v88 // (op 18882) 

s_nop 0
v_readfirstlane_b32 s15, v89 // (op 18882) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -8, v1 // (op 18957) 
v_sub_u32 v89, v1, v88 // (op 18957) 

v_lshlrev_b32 v88, 4, v89 // (op 18957) 

v_lshrrev_b32 v89, 3, v1 // (op 18957) 
v_lshl_add_u32 v90, v89, 7, v88 // (op 18957) 


s_nop 0
v_readfirstlane_b32 s6, v90 // (op 18957) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -128, v1 // (op 19032) 
v_sub_u32 v89, v1, v88 // (op 19032) 

v_lshrrev_b32 v88, 7, v1 // (op 19032) 
v_lshl_add_u32 v90, v88, 8, v89 // (op 19032) 


v_and_b32 v88, -128, v1 // (op 19108) 
v_sub_u32 v89, v1, v88 // (op 19108) 

v_lshrrev_b32 v88, 7, v1 // (op 19108) 
v_lshl_add_u32 v91, v88, 8, v89 // (op 19108) 


v_and_b32 v88, -4, v1 // (op 19236) 
v_sub_u32 v89, v1, v88 // (op 19236) 

v_lshlrev_b32 v88, 5, v89 // (op 19236) 

v_lshrrev_b32 v89, 2, v1 // (op 19236) 
v_lshl_add_u32 v92, v89, 7, v88 // (op 19236) 


v_lshlrev_b32 v88, 2, v92 // (op 19236) 

v_lshrrev_b32 v89, 3, v88 // (op 19236) 

s_nop 0
v_readfirstlane_b32 s7, v89 // (op 19236) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -8, v1 // (op 19280) 
v_sub_u32 v89, v1, v88 // (op 19280) 

v_lshlrev_b32 v88, 4, v89 // (op 19280) 

v_lshrrev_b32 v89, 3, v1 // (op 19280) 
v_lshl_add_u32 v92, v89, 7, v88 // (op 19280) 


s_nop 0
v_readfirstlane_b32 s8, v92 // (op 19280) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -128, v1 // (op 19324) 
v_sub_u32 v89, v1, v88 // (op 19324) 

v_lshrrev_b32 v88, 7, v1 // (op 19324) 
v_lshl_add_u32 v92, v88, 8, v89 // (op 19324) 


v_and_b32 v88, -128, v1 // (op 19353) 
v_sub_u32 v89, v1, v88 // (op 19353) 

v_lshrrev_b32 v88, 7, v1 // (op 19353) 
v_lshl_add_u32 v93, v88, 8, v89 // (op 19353) 


s_mov_b32 s1, 0 // (op 2873) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2872) 
s_mov_b32 s28, s18 // (op 2872) convert

s_and_b32 s29, s28, 4294967294 // (op 2872) 

s_cmp_lt_u32 s1, s29 // (op 2872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2872 // (op 2872) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2872 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2872:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[38:45], a[0:3], v29, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 198) 
s_barrier  // (op 19852) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[46:53], a[4:7], v29, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 662) 
s_add_u32 s1, s1, 2 // (op 2874) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[54:61], a[8:11], v29, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 838) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 17836) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[62:69], a[12:15], v29, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1014) 
v_mov_b64 v[88:89], s[18:19] // (op 17836) convert


v_mov_b64 v[94:95], v[26:27] // (op 17836) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 17836) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 17836) 

v_mov_b32 v96, s28 // (op 17836) 
v_mov_b32 v97, s29 // (op 17836) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[38:45], a[16:19], v30, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1509) 
v_add_co_u32 v98, s[18:19], v94, v96 // (op 17836) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[46:53], a[20:23], v30, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1463) 
v_addc_co_u32 v99, s[18:19], v95, v97, s[18:19] // (op 17836) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[54:61], a[24:27], v30, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1417) 
s_mov_b32 m0, s6 // (op 17836) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[62:69], a[28:31], v30, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1371) 
buffer_load_dwordx4 v98, s[72:75], 0 offen offset: 0 lds // (op 17836) Load value direct to lds
v_add_co_u32 v98, s[18:19], v98, v88 // (op 17836) least significant half
v_addc_co_u32 v99, s[18:19], 0, v99, s[18:19] // (op 17836) most significant half

s_add_u32 m0, m0, 4096 // (op 17836) 
buffer_load_dwordx4 v98, s[72:75], 0 offen offset: 0 lds // (op 17836) Load value direct to lds
v_add_co_u32 v98, s[18:19], v98, v88 // (op 17836) least significant half
v_addc_co_u32 v99, s[18:19], 0, v99, s[18:19] // (op 17836) most significant half

s_add_u32 m0, m0, 4096 // (op 17836) 
buffer_load_dwordx4 v98, s[72:75], 0 offen offset: 0 lds // (op 17836) Load value direct to lds
v_add_co_u32 v98, s[18:19], v98, v88 // (op 17836) least significant half
v_addc_co_u32 v99, s[18:19], 0, v99, s[18:19] // (op 17836) most significant half

s_add_u32 m0, m0, 4096 // (op 17836) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[38:45], a[32:35], v82, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2279) 
buffer_load_dwordx4 v98, s[72:75], 0 offen offset: 0 lds // (op 17836) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[46:53], a[36:39], v82, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2233) 
v_mov_b64 v[88:89], s[44:45] // (op 204) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[54:61], a[40:43], v82, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2187) 
v_mov_b32 v96, v31 // (op 204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[62:69], a[44:47], v82, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2141) 
s_lshl_b32 s18, s44, 3 // (op 204) 
v_add_u32 v97, v96, s18 // (op 204) 

buffer_load_ubyte v34, v97, s[80:83], 0 offen offset: 0 // (op 204) Load value
v_add_u32 v97, v97, v88 // (op 204) 
buffer_load_ubyte v35, v97, s[80:83], 0 offen offset: 0 // (op 204) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[38:45], a[48:51], v83, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 3049) 
s_barrier  // (op 19833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[46:53], a[52:55], v83, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 3003) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[54:61], a[56:59], v83, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2957) 
s_lshl_b64 s[18:19], s[30:31], 6 // (op 17831) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[62:69], a[60:63], v83, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2911) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17831) 

v_lshrrev_b64 v[88:89], 3, s[28:29] // (op 17831) 


v_mov_b64 v[98:99], v[22:23] // (op 17831) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 17831) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17831) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17831) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 17831) 

v_mov_b32 v97, s28 // (op 17831) 
v_mov_b32 v100, s29 // (op 17831) 
v_add_co_u32 v102, s[18:19], v98, v97 // (op 17831) least significant half
v_addc_co_u32 v103, s[18:19], v99, v100, s[18:19] // (op 17831) most significant half




s_add_u32 s18, 16384, s15 // (op 17831) 
s_nop 0
s_mov_b32 m0, s18 // (op 17831) reg expression
buffer_load_dwordx4 v102, s[68:71], 0 offen offset: 0 lds // (op 17831) Load value direct to lds
v_add_co_u32 v102, s[28:29], v102, v88 // (op 17831) least significant half
v_addc_co_u32 v103, s[28:29], 0, v103, s[28:29] // (op 17831) most significant half

s_add_u32 m0, m0, 4096 // (op 17831) 
buffer_load_dwordx4 v102, s[68:71], 0 offen offset: 0 lds // (op 17831) Load value direct to lds




v_mov_b64 v[88:89], s[38:39] // (op 203) convert

v_mov_b32 v97, v28 // (op 203) 
s_lshl_b32 s18, s38, 3 // (op 203) 
v_add_u32 v100, v97, s18 // (op 203) 

buffer_load_ubyte v36, v100, s[76:79], 0 offen offset: 0 // (op 203) Load value
v_add_u32 v100, v100, v88 // (op 203) 
buffer_load_ubyte v37, v100, s[76:79], 0 offen offset: 0 // (op 203) Load value


s_barrier  // (op 19934) 



s_waitcnt vmcnt(15)
ds_write_b8 v93, v32 offset:50176 // (op 541) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v93, v33 offset:50304 // (op 541) Store local data 
s_barrier  // (op 19928) 



s_waitcnt vmcnt(11)
ds_write_b8 v92, v24 offset:50688 // (op 540) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v92, v25 offset:50816 // (op 540) Store local data 
s_barrier  // (op 19915) 



ds_read_b64_tr_b8 v[100:101], v12 offset:24576 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v12 offset:25600 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v12 offset:32768 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v12 offset:33792 // (op 533) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19931) 


ds_read_u8 v88, v13 offset:50176 // (op 537) Load local data 



ds_read_b64_tr_b8 v[108:109], v12 offset:24608 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v12 offset:25632 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v12 offset:32800 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v12 offset:33824 // (op 667) Transpose load local data 


ds_read_u8 v30, v13 offset:50208 // (op 671) Load local data 



ds_read_b64_tr_b8 v[116:117], v12 offset:24640 // (op 843) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v12 offset:25664 // (op 843) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v12 offset:32832 // (op 843) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v12 offset:33856 // (op 843) Transpose load local data 


ds_read_u8 v82, v13 offset:50240 // (op 847) Load local data 



ds_read_b64_tr_b8 v[124:125], v12 offset:24672 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v12 offset:25696 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[128:129], v12 offset:32864 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v12 offset:33888 // (op 1019) Transpose load local data 


ds_read_u8 v89, v13 offset:50272 // (op 1023) Load local data 
s_barrier  // (op 19906) 
s_barrier  // (op 20052) 



ds_read_b64_tr_b4 v[70:71], v11 offset:40960 // (op 531) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v11 offset:41984 // (op 531) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19925) 


ds_read_u8 v29, v10 offset:50688 // (op 535) Load local data 



ds_read_b64_tr_b4 v[74:75], v11 offset:40976 // (op 1374) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v11 offset:42000 // (op 1374) Transpose load local data 


ds_read_u8 v16, v10 offset:50720 // (op 1377) Load local data 



ds_read_b64_tr_b4 v[78:79], v11 offset:40992 // (op 2144) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v11 offset:42016 // (op 2144) Transpose load local data 


ds_read_u8 v20, v10 offset:50752 // (op 2147) Load local data 



ds_read_b64_tr_b4 v[84:85], v11 offset:41008 // (op 2914) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v11 offset:42032 // (op 2914) Transpose load local data 


ds_read_u8 v83, v10 offset:50784 // (op 2917) Load local data 
s_barrier  // (op 20086) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[100:107], a[0:3], v29, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 538) 
s_barrier  // (op 19922) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[108:115], a[4:7], v29, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 663) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 17859) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[116:123], a[8:11], v29, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 839) 
v_mov_b64 v[132:133], s[18:19] // (op 17859) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[124:131], a[12:15], v29, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1015) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 17859) 
s_mul_i32 s29, 3, s19 // (op 17859) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 17859) least significant
s_mul_hi_u32 s48, 3, s18 // (op 17859) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 17859) most significant: sum


v_mov_b32 v134, s28 // (op 17859) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[100:107], a[16:19], v16, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1510) 
v_mov_b32 v135, s29 // (op 17859) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[108:115], a[20:23], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1464) 
v_add_co_u32 v136, s[18:19], v94, v134 // (op 17859) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[116:123], a[24:27], v16, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1418) 
v_addc_co_u32 v137, s[18:19], v95, v135, s[18:19] // (op 17859) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[124:131], a[28:31], v16, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1372) 
s_add_u32 s18, 24576, s8 // (op 17859) 
s_mov_b32 m0, s18 // (op 17859) reg expression
buffer_load_dwordx4 v136, s[72:75], 0 offen offset: 0 lds // (op 17859) Load value direct to lds
v_add_co_u32 v136, s[28:29], v136, v132 // (op 17859) least significant half
v_addc_co_u32 v137, s[28:29], 0, v137, s[28:29] // (op 17859) most significant half

s_add_u32 m0, m0, 4096 // (op 17859) 
buffer_load_dwordx4 v136, s[72:75], 0 offen offset: 0 lds // (op 17859) Load value direct to lds
v_add_co_u32 v136, s[28:29], v136, v132 // (op 17859) least significant half
v_addc_co_u32 v137, s[28:29], 0, v137, s[28:29] // (op 17859) most significant half

s_add_u32 m0, m0, 4096 // (op 17859) 
buffer_load_dwordx4 v136, s[72:75], 0 offen offset: 0 lds // (op 17859) Load value direct to lds
v_add_co_u32 v136, s[28:29], v136, v132 // (op 17859) least significant half
v_addc_co_u32 v137, s[28:29], 0, v137, s[28:29] // (op 17859) most significant half

s_add_u32 m0, m0, 4096 // (op 17859) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[100:107], a[32:35], v20, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2280) 
buffer_load_dwordx4 v136, s[72:75], 0 offen offset: 0 lds // (op 17859) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[108:115], a[36:39], v20, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2234) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[116:123], a[40:43], v20, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2188) 
v_mov_b64 v[94:95], s[44:45] // (op 544) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[124:131], a[44:47], v20, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2142) 
s_lshl_b32 s18, s44, 2 // (op 544) 
s_mul_i32 s19, 3, s18 // (op 544) 

v_add_u32 v132, v96, s19 // (op 544) 

buffer_load_ubyte v32, v132, s[80:83], 0 offen offset: 0 // (op 544) Load value
v_add_u32 v132, v132, v94 // (op 544) 
buffer_load_ubyte v33, v132, s[80:83], 0 offen offset: 0 // (op 544) Load value


s_lshl_b64 s[18:19], s[34:35], 7 // (op 18148) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 18148) 

s_lshl_b32 s18, s44, 3 // (op 18404) 
v_add_u32 v31, v31, s18 // (op 18404) 

v_mov_b32 v94, s28 // (op 18148) 
v_mov_b32 v95, s29 // (op 18148) 
v_add_co_u32 v26, s[18:19], v26, v94 // (op 18148) least significant half
v_addc_co_u32 v27, s[18:19], v27, v95, s[18:19] // (op 18148) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[100:107], a[48:51], v83, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 3050) 
s_barrier  // (op 19912) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[108:115], a[52:55], v83, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 3004) 
s_lshl_b64 s[18:19], s[30:31], 6 // (op 17854) 

s_lshl_b64 s[28:29], s[18:19], 2 // (op 17854) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[116:123], a[56:59], v83, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2958) 
v_lshrrev_b64 v[94:95], 3, s[28:29] // (op 17854) 



s_lshl_b64 s[18:19], s[30:31], 7 // (op 17854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[124:131], a[60:63], v83, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2912) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17854) 



s_lshr_b64 s[18:19], s[28:29], 3 // (op 17854) 

s_mul_i32 s29, 3, s19 // (op 17854) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 17854) least significant
s_mul_hi_u32 s48, 3, s18 // (op 17854) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 17854) most significant: sum


v_mov_b32 v96, s28 // (op 17854) 
v_mov_b32 v88, s29 // (op 17854) 
v_add_co_u32 v100, s[18:19], v98, v96 // (op 17854) least significant half
v_addc_co_u32 v101, s[18:19], v99, v88, s[18:19] // (op 17854) most significant half




s_add_u32 s18, 40960, s7 // (op 17854) 
s_nop 0
s_mov_b32 m0, s18 // (op 17854) reg expression
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 17854) Load value direct to lds
v_add_co_u32 v100, s[28:29], v100, v94 // (op 17854) least significant half
v_addc_co_u32 v101, s[28:29], 0, v101, s[28:29] // (op 17854) most significant half

s_add_u32 m0, m0, 4096 // (op 17854) 
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 17854) Load value direct to lds





v_mov_b64 v[88:89], s[38:39] // (op 543) convert

s_lshl_b32 s18, s38, 2 // (op 543) 
s_mul_i32 s19, 3, s18 // (op 543) 

v_add_u32 v94, v97, s19 // (op 543) 

buffer_load_ubyte v24, v94, s[76:79], 0 offen offset: 0 // (op 543) Load value
v_add_u32 v94, v94, v88 // (op 543) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 18002) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 18002) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 18002) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 18002) 

v_mov_b32 v95, s28 // (op 18002) 
v_mov_b32 v96, s29 // (op 18002) 
v_add_co_u32 v22, s[18:19], v22, v95 // (op 18002) least significant half
v_addc_co_u32 v23, s[18:19], v23, v96, s[18:19] // (op 18002) most significant half




buffer_load_ubyte v25, v94, s[76:79], 0 offen offset: 0 // (op 543) Load value



s_barrier  // (op 19892) 
s_lshl_b32 s18, s38, 3 // (op 18276) 



v_add_u32 v28, v28, s18 // (op 18276) 

s_waitcnt vmcnt(15)
ds_write_b8 v91, v34 offset:49152 // (op 201) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v91, v35 offset:49280 // (op 201) Store local data 
s_barrier  // (op 19872) 



s_waitcnt vmcnt(11)
ds_write_b8 v90, v36 offset:49664 // (op 200) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v90, v37 offset:49792 // (op 200) Store local data 
s_barrier  // (op 19858) 



ds_read_b64_tr_b8 v[38:39], v8 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v8 offset:1024 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:8192 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:9216 // (op 193) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19899) 


ds_read_u8 v15, v9 offset:49152 // (op 197) Load local data 



ds_read_b64_tr_b8 v[46:47], v8 offset:32 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v8 offset:1056 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:8224 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:9248 // (op 657) Transpose load local data 


ds_read_u8 v16, v9 offset:49184 // (op 661) Load local data 



ds_read_b64_tr_b8 v[54:55], v8 offset:64 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:1088 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v8 offset:8256 // (op 833) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v8 offset:9280 // (op 833) Transpose load local data 


ds_read_u8 v20, v9 offset:49216 // (op 837) Load local data 



ds_read_b64_tr_b8 v[62:63], v8 offset:96 // (op 1009) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v8 offset:1120 // (op 1009) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v8 offset:8288 // (op 1009) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v8 offset:9312 // (op 1009) Transpose load local data 


ds_read_u8 v21, v9 offset:49248 // (op 1013) Load local data 
s_barrier  // (op 19839) 



ds_read_b64_tr_b4 v[70:71], v7 offset:16384 // (op 191) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v7 offset:17408 // (op 191) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 19879) 


ds_read_u8 v29, v6 offset:49664 // (op 195) Load local data 



ds_read_b64_tr_b4 v[74:75], v7 offset:16400 // (op 1366) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:17424 // (op 1366) Transpose load local data 


ds_read_u8 v30, v6 offset:49696 // (op 1369) Load local data 



ds_read_b64_tr_b4 v[78:79], v7 offset:16416 // (op 2136) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:17440 // (op 2136) Transpose load local data 


ds_read_u8 v82, v6 offset:49728 // (op 2139) Load local data 



ds_read_b64_tr_b4 v[84:85], v7 offset:16432 // (op 2906) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:17456 // (op 2906) Transpose load local data 


ds_read_u8 v83, v6 offset:49760 // (op 2909) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2872) 
s_mov_b32 s28, s18 // (op 2872) convert

s_and_b32 s29, s28, 4294967294 // (op 2872) 

s_cmp_lt_u32 s1, s29 // (op 2872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2872 // (op 2872) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2872 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2872:




















s_barrier  // (op 20070) 
v_and_b32 v20, -4, v1 // (op 17926) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 17926) 
v_sub_u32 v31, v1, v20 // (op 17926) 

s_and_b32 s15, s18, 4294967294 // (op 17926) 
v_lshlrev_b32 v20, 5, v31 // (op 17926) 

s_add_u32 s6, s0, s15 // (op 17926) 

v_lshrrev_b32 v31, 2, v1 // (op 17926) 
s_mul_i32 s15, s16, s50 // (op 17926) 
s_add_u32 s15, s15, s6 // (op 17926) 

s_add_u32 s6, s15, s9 // (op 17926) 

s_mul_hi_u32 s15, s6, s13 // (op 17926) 
s_sub_u32 s7, s6, s15 // (op 17926) 
s_lshr_b32 s8, s7, 1 // (op 17926) 

s_add_u32 s7, s8, s15 // (op 17926) 


s_lshr_b32 s8, s7, s11 // (op 17926) 

s_and_b32 s7, s11, 2147483648 // (op 17926) 
s_cmp_eq_i32 s7, 0 // (op 17926) 
s_mov_b32 s15, scc // (op 17926) 

s_cmp_eq_i32 s15, 0 // (op 17926) 
s_cselect_b32 s7, s6, s8 // (op 17926) 


s_mul_hi_u32 s8, s7, s12 // (op 17926) 
s_sub_u32 s15, s7, s8 // (op 17926) 
s_lshr_b32 s28, s15, 1 // (op 17926) 

s_add_u32 s15, s28, s8 // (op 17926) 


s_lshr_b32 s8, s15, s10 // (op 17926) 

s_and_b32 s15, s10, 2147483648 // (op 17926) 
s_cmp_eq_i32 s15, 0 // (op 17926) 
s_mov_b32 s28, scc // (op 17926) 

s_cmp_eq_i32 s28, 0 // (op 17926) 
s_cselect_b32 s15, s7, s8 // (op 17926) 


s_mul_i32 s8, s7, s18 // (op 17926) 


v_lshl_add_u32 v15, s15, 7, v20 // (op 17926) 


s_sub_u32 s7, s6, s8 // (op 17926) 


v_mov_b32 v23, 0 // (op 17926) convert
v_mov_b32 v22, v15 // (op 17926) convert

v_lshl_add_u32 v20, s7, 7, v31 // (op 17926) 


v_mul_lo_u32 v27, v20, s31 // (op 17926) most significant: low of low * high
v_mul_hi_u32 v28, v20, s30 // (op 17926) most significant: high of low * low
v_mul_lo_u32 v26, v20, s30 // (op 17926) least significant: low of low * low
v_add_u32 v27, v27, v28 // (op 17926) most significant: sum


v_add_co_u32 v34, s[18:19], v22, v26 // (op 17926) least significant half
v_addc_co_u32 v35, s[18:19], v23, v27, s[18:19] // (op 17926) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 17926) 


v_lshrrev_b64 v[22:23], 3, v[34:35] // (op 17926) 

s_mul_i32 s19, s30, s3 // (op 17928) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 17928) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 17928) least significant
s_mul_hi_u32 s6, s30, s2 // (op 17928) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17928) most significant: sum
s_add_u32 s19, s19, s6 // (op 17928) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17928) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 17928) 

s_mov_b32 s68, s32 // (op 17928) reg expression
s_mov_b32 s69, s33 // (op 17928) reg expression
s_mov_b32 s70, s18 // (op 17928) reg expression
s_mov_b32 s71, 131072 // (op 17928) call()

v_and_b32 v20, -8, v1 // (op 18092) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 18092) 
v_sub_u32 v31, v1, v20 // (op 18092) 

s_and_b32 s15, s18, 4294967294 // (op 18092) 
v_lshlrev_b32 v20, 4, v31 // (op 18092) 

s_add_u32 s6, s0, s15 // (op 18092) 

v_lshrrev_b32 v31, 3, v1 // (op 18092) 
s_mul_i32 s15, s16, s50 // (op 18092) 
s_add_u32 s15, s15, s6 // (op 18092) 

s_add_u32 s6, s15, s9 // (op 18092) 

s_mul_hi_u32 s15, s6, s13 // (op 18092) 
s_sub_u32 s7, s6, s15 // (op 18092) 
s_lshr_b32 s8, s7, 1 // (op 18092) 

s_add_u32 s7, s8, s15 // (op 18092) 


s_lshr_b32 s8, s7, s11 // (op 18092) 

s_and_b32 s7, s11, 2147483648 // (op 18092) 
s_cmp_eq_i32 s7, 0 // (op 18092) 
s_mov_b32 s15, scc // (op 18092) 

s_cmp_eq_i32 s15, 0 // (op 18092) 
s_cselect_b32 s7, s6, s8 // (op 18092) 


s_mul_hi_u32 s8, s7, s12 // (op 18092) 
s_sub_u32 s15, s7, s8 // (op 18092) 
s_lshr_b32 s28, s15, 1 // (op 18092) 

s_add_u32 s15, s28, s8 // (op 18092) 


s_lshr_b32 s8, s15, s10 // (op 18092) 

s_and_b32 s15, s10, 2147483648 // (op 18092) 
s_cmp_eq_i32 s15, 0 // (op 18092) 
s_mov_b32 s28, scc // (op 18092) 

s_cmp_eq_i32 s28, 0 // (op 18092) 
s_cselect_b32 s15, s7, s8 // (op 18092) 


s_mul_i32 s8, s15, s14 // (op 18092) 

s_sub_u32 s15, s7, s8 // (op 18092) 

s_mul_i32 s8, s7, s18 // (op 18092) 


v_lshl_add_u32 v15, s15, 7, v20 // (op 18092) 


s_sub_u32 s7, s6, s8 // (op 18092) 


v_mov_b32 v27, 0 // (op 18092) convert
v_mov_b32 v26, v15 // (op 18092) convert

v_lshl_add_u32 v20, s7, 7, v31 // (op 18092) 


v_mul_lo_u32 v35, v20, s35 // (op 18092) most significant: low of low * high
v_mul_hi_u32 v28, v20, s34 // (op 18092) most significant: high of low * low
v_mul_lo_u32 v34, v20, s34 // (op 18092) least significant: low of low * low
v_add_u32 v35, v35, v28 // (op 18092) most significant: sum

v_add_co_u32 v34, s[18:19], v34, v26 // (op 18092) least significant half
v_addc_co_u32 v35, s[18:19], v35, v27, s[18:19] // (op 18092) most significant half



s_mul_i32 s19, s34, s3 // (op 18094) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 18094) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 18094) least significant
s_mul_hi_u32 s6, s34, s2 // (op 18094) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18094) most significant: sum
s_add_u32 s19, s19, s6 // (op 18094) most significant: sum


s_mov_b32 s72, s36 // (op 18094) reg expression
s_mov_b32 s73, s37 // (op 18094) reg expression
s_mov_b32 s74, s18 // (op 18094) reg expression
s_mov_b32 s75, 131072 // (op 18094) call()

v_and_b32 v20, -128, v1 // (op 18220) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 18220) 
v_sub_u32 v31, v1, v20 // (op 18220) 

s_and_b32 s15, s18, 4294967294 // (op 18220) 
v_lshrrev_b32 v20, 7, v1 // (op 18220) 
s_add_u32 s6, s0, s15 // (op 18220) 

v_lshlrev_b32 v15, 1, v20 // (op 18220) 

s_mul_i32 s15, s16, s50 // (op 18220) 
s_add_u32 s15, s15, s6 // (op 18220) 

s_add_u32 s6, s15, s9 // (op 18220) 

s_mul_hi_u32 s15, s6, s13 // (op 18220) 
s_sub_u32 s7, s6, s15 // (op 18220) 
s_lshr_b32 s8, s7, 1 // (op 18220) 

s_add_u32 s7, s8, s15 // (op 18220) 


s_lshr_b32 s8, s7, s11 // (op 18220) 

s_and_b32 s7, s11, 2147483648 // (op 18220) 
s_cmp_eq_i32 s7, 0 // (op 18220) 
s_mov_b32 s15, scc // (op 18220) 

s_cmp_eq_i32 s15, 0 // (op 18220) 
s_cselect_b32 s7, s6, s8 // (op 18220) 


s_mul_hi_u32 s8, s7, s12 // (op 18220) 
s_sub_u32 s15, s7, s8 // (op 18220) 
s_lshr_b32 s28, s15, 1 // (op 18220) 

s_add_u32 s15, s28, s8 // (op 18220) 


s_lshr_b32 s8, s15, s10 // (op 18220) 

s_and_b32 s15, s10, 2147483648 // (op 18220) 
s_cmp_eq_i32 s15, 0 // (op 18220) 
s_mov_b32 s28, scc // (op 18220) 

s_cmp_eq_i32 s28, 0 // (op 18220) 
s_cselect_b32 s15, s7, s8 // (op 18220) 


s_mul_i32 s8, s7, s18 // (op 18220) 


v_lshl_add_u32 v16, s15, 7, v31 // (op 18220) 


s_sub_u32 s7, s6, s8 // (op 18220) 


v_lshl_add_u32 v20, s7, 2, v15 // (op 18220) 


v_mul_lo_u32 v15, v20, s38 // (op 18220) 

v_add_u32 v20, v16, v15 // (op 18220) 


s_mul_i32 s19, s38, s43 // (op 18222) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18222) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18222) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18222) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18222) most significant: sum
s_add_u32 s19, s19, s6 // (op 18222) most significant: sum


s_mov_b32 s76, s40 // (op 18222) reg expression
s_mov_b32 s77, s41 // (op 18222) reg expression
s_mov_b32 s78, s18 // (op 18222) reg expression
s_mov_b32 s79, 131072 // (op 18222) call()

v_and_b32 v31, -128, v1 // (op 18348) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 18348) 
v_sub_u32 v15, v1, v31 // (op 18348) 

s_and_b32 s15, s18, 4294967294 // (op 18348) 
v_lshrrev_b32 v16, 7, v1 // (op 18348) 
s_add_u32 s6, s0, s15 // (op 18348) 

v_lshlrev_b32 v31, 1, v16 // (op 18348) 

s_mul_i32 s15, s16, s50 // (op 18348) 
s_add_u32 s15, s15, s6 // (op 18348) 

s_add_u32 s6, s15, s9 // (op 18348) 

s_mul_hi_u32 s15, s6, s13 // (op 18348) 
s_sub_u32 s7, s6, s15 // (op 18348) 
s_lshr_b32 s8, s7, 1 // (op 18348) 

s_add_u32 s7, s8, s15 // (op 18348) 


s_lshr_b32 s8, s7, s11 // (op 18348) 

s_and_b32 s7, s11, 2147483648 // (op 18348) 
s_cmp_eq_i32 s7, 0 // (op 18348) 
s_mov_b32 s15, scc // (op 18348) 

s_cmp_eq_i32 s15, 0 // (op 18348) 
s_cselect_b32 s7, s6, s8 // (op 18348) 


s_mul_hi_u32 s8, s7, s12 // (op 18348) 
s_sub_u32 s15, s7, s8 // (op 18348) 
s_lshr_b32 s28, s15, 1 // (op 18348) 

s_add_u32 s15, s28, s8 // (op 18348) 


s_lshr_b32 s8, s15, s10 // (op 18348) 

s_and_b32 s15, s10, 2147483648 // (op 18348) 
s_cmp_eq_i32 s15, 0 // (op 18348) 
s_mov_b32 s28, scc // (op 18348) 

s_cmp_eq_i32 s28, 0 // (op 18348) 
s_cselect_b32 s15, s7, s8 // (op 18348) 


s_mul_i32 s8, s15, s14 // (op 18348) 

s_sub_u32 s15, s7, s8 // (op 18348) 

s_mul_i32 s8, s7, s18 // (op 18348) 


v_lshl_add_u32 v16, s15, 7, v15 // (op 18348) 


s_sub_u32 s7, s6, s8 // (op 18348) 


v_lshl_add_u32 v15, s7, 2, v31 // (op 18348) 


v_mul_lo_u32 v31, v15, s44 // (op 18348) 
v_add_u32 v31, v31, v16 // (op 18348) 


s_mul_i32 s19, s44, s43 // (op 18350) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 18350) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 18350) least significant
s_mul_hi_u32 s6, s44, s42 // (op 18350) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18350) most significant: sum
s_add_u32 s19, s19, s6 // (op 18350) most significant: sum


s_mov_b32 s80, s46 // (op 18350) reg expression
s_mov_b32 s81, s47 // (op 18350) reg expression
s_mov_b32 s82, s18 // (op 18350) reg expression
s_mov_b32 s83, 131072 // (op 18350) call()

v_and_b32 v15, -4, v1 // (op 18895) 
v_sub_u32 v16, v1, v15 // (op 18895) 

v_lshlrev_b32 v15, 5, v16 // (op 18895) 

v_lshrrev_b32 v16, 2, v1 // (op 18895) 
v_lshl_add_u32 v26, v16, 7, v15 // (op 18895) 


v_lshlrev_b32 v15, 2, v26 // (op 18895) 

v_lshrrev_b32 v16, 3, v15 // (op 18895) 

s_nop 0
v_readfirstlane_b32 s15, v16 // (op 18895) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 18970) 
v_sub_u32 v16, v1, v15 // (op 18970) 

v_lshlrev_b32 v15, 4, v16 // (op 18970) 

v_lshrrev_b32 v16, 3, v1 // (op 18970) 
v_lshl_add_u32 v26, v16, 7, v15 // (op 18970) 


s_nop 0
v_readfirstlane_b32 s6, v26 // (op 18970) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -128, v1 // (op 19045) 
v_sub_u32 v16, v1, v15 // (op 19045) 

v_lshrrev_b32 v15, 7, v1 // (op 19045) 
v_lshl_add_u32 v26, v15, 8, v16 // (op 19045) 


v_and_b32 v15, -128, v1 // (op 19121) 
v_sub_u32 v16, v1, v15 // (op 19121) 

v_lshrrev_b32 v15, 7, v1 // (op 19121) 
v_lshl_add_u32 v27, v15, 8, v16 // (op 19121) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2878) 
s_and_b32 s7, s18, 4294967294 // (op 2878) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2877) 
s_mov_b32 s8, s18 // (op 2877) convert

s_cmp_lt_u32 s7, s8 // (op 2877) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2877 // (op 2877) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2877 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2877:
s_barrier  // (op 19836) 
s_add_u32 s7, s7, 1 // (op 2879) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17841) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17841) 

v_lshrrev_b64 v[24:25], 3, s[28:29] // (op 17841) 


v_mov_b64 v[32:33], v[22:23] // (op 17841) 
v_mov_b64 v[28:29], v[32:33] // (op 17841) reg expression
s_add_u32 s8, 16384, s15 // (op 17841) 
s_mov_b32 m0, s8 // (op 17841) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 17841) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 17841) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 17841) most significant half

s_add_u32 m0, m0, 4096 // (op 17841) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 17841) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 17916) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17916) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17916) 

v_mov_b32 v21, s18 // (op 17916) 
v_mov_b32 v15, s19 // (op 17916) 
v_add_co_u32 v22, s[28:29], v22, v21 // (op 17916) least significant half
v_addc_co_u32 v23, s[28:29], v23, v15, s[28:29] // (op 17916) most significant half




s_barrier  // (op 19855) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17847) 
v_mov_b64 v[24:25], s[18:19] // (op 17847) convert


v_mov_b64 v[32:33], v[34:35] // (op 17847) 
v_mov_b64 v[28:29], v[32:33] // (op 17847) reg expression
s_mov_b32 m0, s6 // (op 17847) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 17847) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 17847) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 17847) most significant half

s_add_u32 m0, m0, 4096 // (op 17847) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 17847) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 17847) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 17847) most significant half

s_add_u32 m0, m0, 4096 // (op 17847) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 17847) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 17847) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 17847) most significant half

s_add_u32 m0, m0, 4096 // (op 17847) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 17847) Load value direct to lds



s_lshl_b64 s[18:19], s[34:35], 7 // (op 18084) 
v_mov_b32 v21, s18 // (op 18084) 
v_mov_b32 v15, s19 // (op 18084) 
v_add_co_u32 v34, s[28:29], v34, v21 // (op 18084) least significant half
v_addc_co_u32 v35, s[28:29], v35, v15, s[28:29] // (op 18084) most significant half





v_mov_b64 v[24:25], s[38:39] // (op 477) convert

v_mov_b32 v21, v20 // (op 477) 
v_mov_b32 v15, v21 // (op 477) reg expression
buffer_load_ubyte v32, v15, s[76:79], 0 offen offset: 0 // (op 477) Load value
v_add_u32 v15, v15, v24 // (op 477) 
buffer_load_ubyte v33, v15, s[76:79], 0 offen offset: 0 // (op 477) Load value



s_lshl_b32 s8, s38, 2 // (op 18212) 
v_add_u32 v20, v20, s8 // (op 18212) 


s_barrier  // (op 19875) 
v_mov_b64 v[24:25], s[44:45] // (op 478) convert




v_mov_b32 v21, v31 // (op 478) 
v_mov_b32 v15, v21 // (op 478) reg expression
buffer_load_ubyte v28, v15, s[80:83], 0 offen offset: 0 // (op 478) Load value
v_add_u32 v15, v15, v24 // (op 478) 
s_waitcnt vmcnt(2)
ds_write_b8 v26, v32 offset:49664 // (op 474) Store local data 
buffer_load_ubyte v29, v15, s[80:83], 0 offen offset: 0 // (op 478) Load value



s_lshl_b32 s8, s44, 2 // (op 18340) 
v_add_u32 v31, v31, s8 // (op 18340) 

s_waitcnt vmcnt(2)
ds_write_b8 v26, v33 offset:49792 // (op 474) Store local data 

s_barrier  // (op 19842) 



s_barrier  // (op 19895) 
ds_read_b64_tr_b4 v[36:37], v7 offset:16384 // (op 465) Transpose load local data 



ds_read_b64_tr_b4 v[38:39], v7 offset:17408 // (op 465) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v28 offset:49152 // (op 475) Store local data 
s_barrier  // (op 19862) 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v29 offset:49280 // (op 475) Store local data 




ds_read_b64_tr_b8 v[40:41], v8 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:1024 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:8192 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:9216 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 19882) 


ds_read_u8 v21, v6 offset:49664 // (op 469) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 19902) 


ds_read_u8 v15, v9 offset:49152 // (op 471) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:39], v[40:47], a[0:3], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 472) 
ds_read_b64_tr_b8 v[48:49], v8 offset:32 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:1056 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:8224 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:9248 // (op 636) Transpose load local data 


ds_read_u8 v16, v9 offset:49184 // (op 640) Load local data 



ds_read_b64_tr_b8 v[56:57], v8 offset:64 // (op 812) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:39], v[48:55], a[4:7], v21, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 641) 
ds_read_b64_tr_b8 v[58:59], v8 offset:1088 // (op 812) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v8 offset:8256 // (op 812) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v8 offset:9280 // (op 812) Transpose load local data 


ds_read_u8 v24, v9 offset:49216 // (op 816) Load local data 



ds_read_b64_tr_b8 v[64:65], v8 offset:96 // (op 988) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v8 offset:1120 // (op 988) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:39], v[56:63], a[8:11], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 817) 
ds_read_b64_tr_b8 v[68:69], v8 offset:8288 // (op 988) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v8 offset:9312 // (op 988) Transpose load local data 


ds_read_u8 v25, v9 offset:49248 // (op 992) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:39], v[64:71], a[12:15], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 993) 
ds_read_b64_tr_b4 v[72:73], v7 offset:16400 // (op 1250) Transpose load local data 


ds_read_b64_tr_b4 v[74:75], v7 offset:17424 // (op 1250) Transpose load local data 


ds_read_u8 v21, v6 offset:49696 // (op 1253) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[40:47], a[16:19], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1255) 
ds_read_b64_tr_b4 v[36:37], v7 offset:16416 // (op 2020) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[48:55], a[20:23], v21, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1286) 
ds_read_b64_tr_b4 v[38:39], v7 offset:17440 // (op 2020) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[56:63], a[24:27], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1317) 
ds_read_u8 v28, v6 offset:49728 // (op 2023) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[64:71], a[28:31], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1348) 
ds_read_b64_tr_b4 v[76:77], v7 offset:16432 // (op 2790) Transpose load local data 


ds_read_b64_tr_b4 v[78:79], v7 offset:17456 // (op 2790) Transpose load local data 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[36:39], v[40:47], a[32:35], v28, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2025) 
ds_read_u8 v21, v6 offset:49760 // (op 2793) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[36:39], v[48:55], a[36:39], v28, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2056) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:39], v[56:63], a[40:43], v28, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2087) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[36:39], v[64:71], a[44:47], v28, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2118) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[40:47], a[48:51], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2795) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[48:55], a[52:55], v21, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2826) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[56:63], a[56:59], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2857) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[64:71], a[60:63], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2888) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 2877) 
s_mov_b32 s8, s18 // (op 2877) convert

s_cmp_lt_u32 s7, s8 // (op 2877) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2877 // (op 2877) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2877 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2877:









s_barrier  // (op 20074) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_238
v_accvgpr_read v14, a0 // (op 246) 
v_accvgpr_read v15, a1 // (op 246) 
v_accvgpr_read v16, a2 // (op 246) 
v_accvgpr_read v17, a3 // (op 246) 
v_mul_f32 v20, v19, v14 // (op 246) 
v_mul_f32 v21, v19, v15 // (op 246) 
v_mul_f32 v22, v19, v16 // (op 246) 
v_mul_f32 v23, v19, v17 // (op 246) 

v_cvt_f16_f32 v14, v20 // (op 248) 
v_cvt_f16_f32 v15, v21 // (op 248) 
v_cvt_f16_f32 v16, v22 // (op 248) 
v_cvt_f16_f32 v17, v23 // (op 248) 


v_pack_b32_f16 v20, v14, v15 // (op 249) 
v_pack_b32_f16 v21, v16, v17 // (op 249) 

v_mov_b32 v0, v5 // (op 249) reg expression

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 249) Store value



v_accvgpr_read v14, a4 // (op 3847) 
v_accvgpr_read v15, a5 // (op 3847) 
v_accvgpr_read v16, a6 // (op 3847) 
v_accvgpr_read v17, a7 // (op 3847) 
v_mul_f32 v20, v19, v14 // (op 3847) 
v_mul_f32 v21, v19, v15 // (op 3847) 
v_mul_f32 v22, v19, v16 // (op 3847) 
v_mul_f32 v23, v19, v17 // (op 3847) 

v_cvt_f16_f32 v14, v20 // (op 3849) 
v_cvt_f16_f32 v15, v21 // (op 3849) 
v_cvt_f16_f32 v16, v22 // (op 3849) 
v_cvt_f16_f32 v17, v23 // (op 3849) 


v_pack_b32_f16 v20, v14, v15 // (op 3850) 
v_pack_b32_f16 v21, v16, v17 // (op 3850) 

s_lshl_b32 s15, s24, 6 // (op 3850) 
v_add_u32 v0, v5, s15 // (op 3850) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3850) Store value



v_accvgpr_read v14, a8 // (op 3861) 
v_accvgpr_read v15, a9 // (op 3861) 
v_accvgpr_read v16, a10 // (op 3861) 
v_accvgpr_read v17, a11 // (op 3861) 
v_mul_f32 v20, v19, v14 // (op 3861) 
v_mul_f32 v21, v19, v15 // (op 3861) 
v_mul_f32 v22, v19, v16 // (op 3861) 
v_mul_f32 v23, v19, v17 // (op 3861) 

v_cvt_f16_f32 v14, v20 // (op 3863) 
v_cvt_f16_f32 v15, v21 // (op 3863) 
v_cvt_f16_f32 v16, v22 // (op 3863) 
v_cvt_f16_f32 v17, v23 // (op 3863) 


v_pack_b32_f16 v20, v14, v15 // (op 3864) 
v_pack_b32_f16 v21, v16, v17 // (op 3864) 

s_lshl_b32 s15, s24, 7 // (op 3864) 
v_add_u32 v0, v5, s15 // (op 3864) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3864) Store value



v_accvgpr_read v14, a12 // (op 3875) 
v_accvgpr_read v15, a13 // (op 3875) 
v_accvgpr_read v16, a14 // (op 3875) 
v_accvgpr_read v17, a15 // (op 3875) 
v_mul_f32 v20, v19, v14 // (op 3875) 
v_mul_f32 v21, v19, v15 // (op 3875) 
v_mul_f32 v22, v19, v16 // (op 3875) 
v_mul_f32 v23, v19, v17 // (op 3875) 

v_cvt_f16_f32 v14, v20 // (op 3877) 
v_cvt_f16_f32 v15, v21 // (op 3877) 
v_cvt_f16_f32 v16, v22 // (op 3877) 
v_cvt_f16_f32 v17, v23 // (op 3877) 


v_pack_b32_f16 v20, v14, v15 // (op 3878) 
v_pack_b32_f16 v21, v16, v17 // (op 3878) 

s_lshl_b32 s15, s24, 6 // (op 3878) 
s_mul_i32 s6, 3, s15 // (op 3878) 

v_add_u32 v0, v5, s6 // (op 3878) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3878) Store value



v_accvgpr_read v14, a16 // (op 3911) 
v_accvgpr_read v15, a17 // (op 3911) 
v_accvgpr_read v16, a18 // (op 3911) 
v_accvgpr_read v17, a19 // (op 3911) 
v_mul_f32 v20, v19, v14 // (op 3911) 
v_mul_f32 v21, v19, v15 // (op 3911) 
v_mul_f32 v22, v19, v16 // (op 3911) 
v_mul_f32 v23, v19, v17 // (op 3911) 

v_cvt_f16_f32 v14, v20 // (op 3913) 
v_cvt_f16_f32 v15, v21 // (op 3913) 
v_cvt_f16_f32 v16, v22 // (op 3913) 
v_cvt_f16_f32 v17, v23 // (op 3913) 


v_pack_b32_f16 v20, v14, v15 // (op 3915) 
v_pack_b32_f16 v21, v16, v17 // (op 3915) 

v_add_u32 v0, 64, v5 // (op 3915) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3915) Store value



v_accvgpr_read v14, a20 // (op 3925) 
v_accvgpr_read v15, a21 // (op 3925) 
v_accvgpr_read v16, a22 // (op 3925) 
v_accvgpr_read v17, a23 // (op 3925) 
v_mul_f32 v20, v19, v14 // (op 3925) 
v_mul_f32 v21, v19, v15 // (op 3925) 
v_mul_f32 v22, v19, v16 // (op 3925) 
v_mul_f32 v23, v19, v17 // (op 3925) 

v_cvt_f16_f32 v14, v20 // (op 3927) 
v_cvt_f16_f32 v15, v21 // (op 3927) 
v_cvt_f16_f32 v16, v22 // (op 3927) 
v_cvt_f16_f32 v17, v23 // (op 3927) 


v_pack_b32_f16 v20, v14, v15 // (op 3917) 
v_pack_b32_f16 v21, v16, v17 // (op 3917) 

s_lshl_b32 s15, s24, 6 // (op 3917) 
s_add_u32 s15, s15, 64 // (op 3917) 
v_add_u32 v0, v5, s15 // (op 3917) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3917) Store value



v_accvgpr_read v14, a24 // (op 3931) 
v_accvgpr_read v15, a25 // (op 3931) 
v_accvgpr_read v16, a26 // (op 3931) 
v_accvgpr_read v17, a27 // (op 3931) 
v_mul_f32 v20, v19, v14 // (op 3931) 
v_mul_f32 v21, v19, v15 // (op 3931) 
v_mul_f32 v22, v19, v16 // (op 3931) 
v_mul_f32 v23, v19, v17 // (op 3931) 

v_cvt_f16_f32 v14, v20 // (op 3933) 
v_cvt_f16_f32 v15, v21 // (op 3933) 
v_cvt_f16_f32 v16, v22 // (op 3933) 
v_cvt_f16_f32 v17, v23 // (op 3933) 


v_pack_b32_f16 v20, v14, v15 // (op 3919) 
v_pack_b32_f16 v21, v16, v17 // (op 3919) 

s_lshl_b32 s15, s24, 7 // (op 3919) 
s_add_u32 s15, s15, 64 // (op 3919) 
v_add_u32 v0, v5, s15 // (op 3919) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3919) Store value



v_accvgpr_read v14, a28 // (op 3937) 
v_accvgpr_read v15, a29 // (op 3937) 
v_accvgpr_read v16, a30 // (op 3937) 
v_accvgpr_read v17, a31 // (op 3937) 
v_mul_f32 v20, v19, v14 // (op 3937) 
v_mul_f32 v21, v19, v15 // (op 3937) 
v_mul_f32 v22, v19, v16 // (op 3937) 
v_mul_f32 v23, v19, v17 // (op 3937) 

v_cvt_f16_f32 v14, v20 // (op 3939) 
v_cvt_f16_f32 v15, v21 // (op 3939) 
v_cvt_f16_f32 v16, v22 // (op 3939) 
v_cvt_f16_f32 v17, v23 // (op 3939) 


v_pack_b32_f16 v20, v14, v15 // (op 3921) 
v_pack_b32_f16 v21, v16, v17 // (op 3921) 

s_lshl_b32 s15, s24, 6 // (op 3921) 
s_mul_i32 s6, 3, s15 // (op 3921) 
s_add_u32 s6, s6, 64 // (op 3921) 

v_add_u32 v0, v5, s6 // (op 3921) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3921) Store value



v_accvgpr_read v14, a32 // (op 3987) 
v_accvgpr_read v15, a33 // (op 3987) 
v_accvgpr_read v16, a34 // (op 3987) 
v_accvgpr_read v17, a35 // (op 3987) 
v_mul_f32 v20, v19, v14 // (op 3987) 
v_mul_f32 v21, v19, v15 // (op 3987) 
v_mul_f32 v22, v19, v16 // (op 3987) 
v_mul_f32 v23, v19, v17 // (op 3987) 

v_cvt_f16_f32 v14, v20 // (op 3989) 
v_cvt_f16_f32 v15, v21 // (op 3989) 
v_cvt_f16_f32 v16, v22 // (op 3989) 
v_cvt_f16_f32 v17, v23 // (op 3989) 


v_pack_b32_f16 v20, v14, v15 // (op 3991) 
v_pack_b32_f16 v21, v16, v17 // (op 3991) 

v_add_u32 v0, 128, v5 // (op 3991) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3991) Store value



v_accvgpr_read v14, a36 // (op 4001) 
v_accvgpr_read v15, a37 // (op 4001) 
v_accvgpr_read v16, a38 // (op 4001) 
v_accvgpr_read v17, a39 // (op 4001) 
v_mul_f32 v20, v19, v14 // (op 4001) 
v_mul_f32 v21, v19, v15 // (op 4001) 
v_mul_f32 v22, v19, v16 // (op 4001) 
v_mul_f32 v23, v19, v17 // (op 4001) 

v_cvt_f16_f32 v14, v20 // (op 4003) 
v_cvt_f16_f32 v15, v21 // (op 4003) 
v_cvt_f16_f32 v16, v22 // (op 4003) 
v_cvt_f16_f32 v17, v23 // (op 4003) 


v_pack_b32_f16 v20, v14, v15 // (op 3993) 
v_pack_b32_f16 v21, v16, v17 // (op 3993) 

s_lshl_b32 s15, s24, 6 // (op 3993) 
s_add_u32 s15, s15, 128 // (op 3993) 
v_add_u32 v0, v5, s15 // (op 3993) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3993) Store value



v_accvgpr_read v14, a40 // (op 4007) 
v_accvgpr_read v15, a41 // (op 4007) 
v_accvgpr_read v16, a42 // (op 4007) 
v_accvgpr_read v17, a43 // (op 4007) 
v_mul_f32 v20, v19, v14 // (op 4007) 
v_mul_f32 v21, v19, v15 // (op 4007) 
v_mul_f32 v22, v19, v16 // (op 4007) 
v_mul_f32 v23, v19, v17 // (op 4007) 

v_cvt_f16_f32 v14, v20 // (op 4009) 
v_cvt_f16_f32 v15, v21 // (op 4009) 
v_cvt_f16_f32 v16, v22 // (op 4009) 
v_cvt_f16_f32 v17, v23 // (op 4009) 


v_pack_b32_f16 v20, v14, v15 // (op 3995) 
v_pack_b32_f16 v21, v16, v17 // (op 3995) 

s_lshl_b32 s15, s24, 7 // (op 3995) 
s_add_u32 s15, s15, 128 // (op 3995) 
v_add_u32 v0, v5, s15 // (op 3995) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3995) Store value



v_accvgpr_read v14, a44 // (op 4013) 
v_accvgpr_read v15, a45 // (op 4013) 
v_accvgpr_read v16, a46 // (op 4013) 
v_accvgpr_read v17, a47 // (op 4013) 
v_mul_f32 v20, v19, v14 // (op 4013) 
v_mul_f32 v21, v19, v15 // (op 4013) 
v_mul_f32 v22, v19, v16 // (op 4013) 
v_mul_f32 v23, v19, v17 // (op 4013) 

v_cvt_f16_f32 v14, v20 // (op 4015) 
v_cvt_f16_f32 v15, v21 // (op 4015) 
v_cvt_f16_f32 v16, v22 // (op 4015) 
v_cvt_f16_f32 v17, v23 // (op 4015) 


v_pack_b32_f16 v20, v14, v15 // (op 3997) 
v_pack_b32_f16 v21, v16, v17 // (op 3997) 

s_lshl_b32 s15, s24, 6 // (op 3997) 
s_mul_i32 s6, 3, s15 // (op 3997) 
s_add_u32 s6, s6, 128 // (op 3997) 

v_add_u32 v0, v5, s6 // (op 3997) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 3997) Store value



v_accvgpr_read v14, a48 // (op 4063) 
v_accvgpr_read v15, a49 // (op 4063) 
v_accvgpr_read v16, a50 // (op 4063) 
v_accvgpr_read v17, a51 // (op 4063) 
v_mul_f32 v20, v19, v14 // (op 4063) 
v_mul_f32 v21, v19, v15 // (op 4063) 
v_mul_f32 v22, v19, v16 // (op 4063) 
v_mul_f32 v23, v19, v17 // (op 4063) 

v_cvt_f16_f32 v14, v20 // (op 4065) 
v_cvt_f16_f32 v15, v21 // (op 4065) 
v_cvt_f16_f32 v16, v22 // (op 4065) 
v_cvt_f16_f32 v17, v23 // (op 4065) 


v_pack_b32_f16 v20, v14, v15 // (op 4067) 
v_pack_b32_f16 v21, v16, v17 // (op 4067) 

v_add_u32 v0, 192, v5 // (op 4067) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 4067) Store value



v_accvgpr_read v14, a52 // (op 4077) 
v_accvgpr_read v15, a53 // (op 4077) 
v_accvgpr_read v16, a54 // (op 4077) 
v_accvgpr_read v17, a55 // (op 4077) 
v_mul_f32 v20, v19, v14 // (op 4077) 
v_mul_f32 v21, v19, v15 // (op 4077) 
v_mul_f32 v22, v19, v16 // (op 4077) 
v_mul_f32 v23, v19, v17 // (op 4077) 

v_cvt_f16_f32 v14, v20 // (op 4079) 
v_cvt_f16_f32 v15, v21 // (op 4079) 
v_cvt_f16_f32 v16, v22 // (op 4079) 
v_cvt_f16_f32 v17, v23 // (op 4079) 


v_pack_b32_f16 v20, v14, v15 // (op 4069) 
v_pack_b32_f16 v21, v16, v17 // (op 4069) 

s_lshl_b32 s15, s24, 6 // (op 4069) 
s_add_u32 s15, s15, 192 // (op 4069) 
v_add_u32 v0, v5, s15 // (op 4069) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 4069) Store value



v_accvgpr_read v14, a56 // (op 4083) 
v_accvgpr_read v15, a57 // (op 4083) 
v_accvgpr_read v16, a58 // (op 4083) 
v_accvgpr_read v17, a59 // (op 4083) 
v_mul_f32 v20, v19, v14 // (op 4083) 
v_mul_f32 v21, v19, v15 // (op 4083) 
v_mul_f32 v22, v19, v16 // (op 4083) 
v_mul_f32 v23, v19, v17 // (op 4083) 

v_cvt_f16_f32 v14, v20 // (op 4085) 
v_cvt_f16_f32 v15, v21 // (op 4085) 
v_cvt_f16_f32 v16, v22 // (op 4085) 
v_cvt_f16_f32 v17, v23 // (op 4085) 


v_pack_b32_f16 v20, v14, v15 // (op 4071) 
v_pack_b32_f16 v21, v16, v17 // (op 4071) 

s_lshl_b32 s15, s24, 7 // (op 4071) 
s_add_u32 s15, s15, 192 // (op 4071) 
v_add_u32 v0, v5, s15 // (op 4071) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 4071) Store value



v_accvgpr_read v14, a60 // (op 4089) 
v_accvgpr_read v15, a61 // (op 4089) 
v_accvgpr_read v16, a62 // (op 4089) 
v_accvgpr_read v17, a63 // (op 4089) 
v_mul_f32 v20, v19, v14 // (op 4089) 
v_mul_f32 v21, v19, v15 // (op 4089) 
v_mul_f32 v22, v19, v16 // (op 4089) 
v_mul_f32 v23, v19, v17 // (op 4089) 

v_cvt_f16_f32 v14, v20 // (op 4091) 
v_cvt_f16_f32 v15, v21 // (op 4091) 
v_cvt_f16_f32 v16, v22 // (op 4091) 
v_cvt_f16_f32 v17, v23 // (op 4091) 


v_pack_b32_f16 v20, v14, v15 // (op 4073) 
v_pack_b32_f16 v21, v16, v17 // (op 4073) 

s_lshl_b32 s15, s24, 6 // (op 4073) 
s_mul_i32 s6, 3, s15 // (op 4073) 
s_add_u32 s6, s6, 192 // (op 4073) 

v_add_u32 v0, v5, s6 // (op 4073) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 4073) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_238:


v_add_u32 v0, 192, v4 // (op 4461) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4461) Load value

v_accvgpr_read v20, a48 // (op 4448) 
v_accvgpr_read v21, a49 // (op 4448) 
v_accvgpr_read v22, a50 // (op 4448) 
v_accvgpr_read v23, a51 // (op 4448) 
v_mul_f32 v24, v19, v20 // (op 4448) 
v_mul_f32 v25, v19, v21 // (op 4448) 
v_mul_f32 v26, v19, v22 // (op 4448) 
v_mul_f32 v27, v19, v23 // (op 4448) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4460) 
v_lshrrev_b32 v17, 16, v14 // (op 4460) 
v_cvt_f32_f16 v17, v17 // (op 4460) 
v_mul_f32 v20, v18, v16 // (op 4460) 
v_mul_f32 v21, v18, v17 // (op 4460) 
v_cvt_f32_f16 v16, v15 // (op 4460) 
v_lshrrev_b32 v17, 16, v15 // (op 4460) 
v_cvt_f32_f16 v17, v17 // (op 4460) 
v_mul_f32 v22, v18, v16 // (op 4460) 
v_mul_f32 v23, v18, v17 // (op 4460) 


v_add_f32 v14, v20, v24 // (op 4449) 
v_add_f32 v15, v21, v25 // (op 4449) 
v_add_f32 v16, v22, v26 // (op 4449) 
v_add_f32 v17, v23, v27 // (op 4449) 


v_cvt_f16_f32 v20, v14 // (op 4450) 
v_cvt_f16_f32 v21, v15 // (op 4450) 
v_cvt_f16_f32 v22, v16 // (op 4450) 
v_cvt_f16_f32 v23, v17 // (op 4450) 


v_pack_b32_f16 v14, v20, v21 // (op 4452) 
v_pack_b32_f16 v15, v22, v23 // (op 4452) 

v_add_u32 v0, 192, v5 // (op 4452) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4452) Store value





s_lshl_b32 s15, s4, 6 // (op 4466) 
s_add_u32 s15, s15, 192 // (op 4466) 
v_add_u32 v0, v4, s15 // (op 4466) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4466) Load value

v_accvgpr_read v20, a52 // (op 4478) 
v_accvgpr_read v21, a53 // (op 4478) 
v_accvgpr_read v22, a54 // (op 4478) 
v_accvgpr_read v23, a55 // (op 4478) 
v_mul_f32 v24, v19, v20 // (op 4478) 
v_mul_f32 v25, v19, v21 // (op 4478) 
v_mul_f32 v26, v19, v22 // (op 4478) 
v_mul_f32 v27, v19, v23 // (op 4478) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4463) 
v_lshrrev_b32 v17, 16, v14 // (op 4463) 
v_cvt_f32_f16 v17, v17 // (op 4463) 
v_mul_f32 v20, v18, v16 // (op 4463) 
v_mul_f32 v21, v18, v17 // (op 4463) 
v_cvt_f32_f16 v16, v15 // (op 4463) 
v_lshrrev_b32 v17, 16, v15 // (op 4463) 
v_cvt_f32_f16 v17, v17 // (op 4463) 
v_mul_f32 v22, v18, v16 // (op 4463) 
v_mul_f32 v23, v18, v17 // (op 4463) 


v_add_f32 v14, v20, v24 // (op 4464) 
v_add_f32 v15, v21, v25 // (op 4464) 
v_add_f32 v16, v22, v26 // (op 4464) 
v_add_f32 v17, v23, v27 // (op 4464) 


v_cvt_f16_f32 v20, v14 // (op 4465) 
v_cvt_f16_f32 v21, v15 // (op 4465) 
v_cvt_f16_f32 v22, v16 // (op 4465) 
v_cvt_f16_f32 v23, v17 // (op 4465) 


v_pack_b32_f16 v14, v20, v21 // (op 4454) 
v_pack_b32_f16 v15, v22, v23 // (op 4454) 

s_lshl_b32 s15, s24, 6 // (op 4454) 
s_add_u32 s15, s15, 192 // (op 4454) 
v_add_u32 v0, v5, s15 // (op 4454) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4454) Store value





s_lshl_b32 s15, s4, 7 // (op 4471) 
s_add_u32 s15, s15, 192 // (op 4471) 
v_add_u32 v0, v4, s15 // (op 4471) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4471) Load value

v_accvgpr_read v20, a56 // (op 4480) 
v_accvgpr_read v21, a57 // (op 4480) 
v_accvgpr_read v22, a58 // (op 4480) 
v_accvgpr_read v23, a59 // (op 4480) 
v_mul_f32 v24, v19, v20 // (op 4480) 
v_mul_f32 v25, v19, v21 // (op 4480) 
v_mul_f32 v26, v19, v22 // (op 4480) 
v_mul_f32 v27, v19, v23 // (op 4480) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4468) 
v_lshrrev_b32 v17, 16, v14 // (op 4468) 
v_cvt_f32_f16 v17, v17 // (op 4468) 
v_mul_f32 v20, v18, v16 // (op 4468) 
v_mul_f32 v21, v18, v17 // (op 4468) 
v_cvt_f32_f16 v16, v15 // (op 4468) 
v_lshrrev_b32 v17, 16, v15 // (op 4468) 
v_cvt_f32_f16 v17, v17 // (op 4468) 
v_mul_f32 v22, v18, v16 // (op 4468) 
v_mul_f32 v23, v18, v17 // (op 4468) 


v_add_f32 v14, v20, v24 // (op 4469) 
v_add_f32 v15, v21, v25 // (op 4469) 
v_add_f32 v16, v22, v26 // (op 4469) 
v_add_f32 v17, v23, v27 // (op 4469) 


v_cvt_f16_f32 v20, v14 // (op 4470) 
v_cvt_f16_f32 v21, v15 // (op 4470) 
v_cvt_f16_f32 v22, v16 // (op 4470) 
v_cvt_f16_f32 v23, v17 // (op 4470) 


v_pack_b32_f16 v14, v20, v21 // (op 4456) 
v_pack_b32_f16 v15, v22, v23 // (op 4456) 

s_lshl_b32 s15, s24, 7 // (op 4456) 
s_add_u32 s15, s15, 192 // (op 4456) 
v_add_u32 v0, v5, s15 // (op 4456) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4456) Store value





s_lshl_b32 s15, s4, 6 // (op 4476) 
s_mul_i32 s6, 3, s15 // (op 4476) 
s_add_u32 s6, s6, 192 // (op 4476) 

v_add_u32 v0, v4, s6 // (op 4476) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4476) Load value

v_accvgpr_read v20, a60 // (op 4482) 
v_accvgpr_read v21, a61 // (op 4482) 
v_accvgpr_read v22, a62 // (op 4482) 
v_accvgpr_read v23, a63 // (op 4482) 
v_mul_f32 v24, v19, v20 // (op 4482) 
v_mul_f32 v25, v19, v21 // (op 4482) 
v_mul_f32 v26, v19, v22 // (op 4482) 
v_mul_f32 v27, v19, v23 // (op 4482) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4473) 
v_lshrrev_b32 v17, 16, v14 // (op 4473) 
v_cvt_f32_f16 v17, v17 // (op 4473) 
v_mul_f32 v20, v18, v16 // (op 4473) 
v_mul_f32 v21, v18, v17 // (op 4473) 
v_cvt_f32_f16 v16, v15 // (op 4473) 
v_lshrrev_b32 v17, 16, v15 // (op 4473) 
v_cvt_f32_f16 v17, v17 // (op 4473) 
v_mul_f32 v22, v18, v16 // (op 4473) 
v_mul_f32 v23, v18, v17 // (op 4473) 


v_add_f32 v14, v20, v24 // (op 4474) 
v_add_f32 v15, v21, v25 // (op 4474) 
v_add_f32 v16, v22, v26 // (op 4474) 
v_add_f32 v17, v23, v27 // (op 4474) 


v_cvt_f16_f32 v20, v14 // (op 4475) 
v_cvt_f16_f32 v21, v15 // (op 4475) 
v_cvt_f16_f32 v22, v16 // (op 4475) 
v_cvt_f16_f32 v23, v17 // (op 4475) 


v_pack_b32_f16 v14, v20, v21 // (op 4458) 
v_pack_b32_f16 v15, v22, v23 // (op 4458) 

s_lshl_b32 s15, s24, 6 // (op 4458) 
s_mul_i32 s6, 3, s15 // (op 4458) 
s_add_u32 s6, s6, 192 // (op 4458) 

v_add_u32 v0, v5, s6 // (op 4458) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4458) Store value





v_add_u32 v0, 128, v4 // (op 4374) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4374) Load value

v_accvgpr_read v20, a32 // (op 4361) 
v_accvgpr_read v21, a33 // (op 4361) 
v_accvgpr_read v22, a34 // (op 4361) 
v_accvgpr_read v23, a35 // (op 4361) 
v_mul_f32 v24, v19, v20 // (op 4361) 
v_mul_f32 v25, v19, v21 // (op 4361) 
v_mul_f32 v26, v19, v22 // (op 4361) 
v_mul_f32 v27, v19, v23 // (op 4361) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4373) 
v_lshrrev_b32 v17, 16, v14 // (op 4373) 
v_cvt_f32_f16 v17, v17 // (op 4373) 
v_mul_f32 v20, v18, v16 // (op 4373) 
v_mul_f32 v21, v18, v17 // (op 4373) 
v_cvt_f32_f16 v16, v15 // (op 4373) 
v_lshrrev_b32 v17, 16, v15 // (op 4373) 
v_cvt_f32_f16 v17, v17 // (op 4373) 
v_mul_f32 v22, v18, v16 // (op 4373) 
v_mul_f32 v23, v18, v17 // (op 4373) 


v_add_f32 v14, v20, v24 // (op 4362) 
v_add_f32 v15, v21, v25 // (op 4362) 
v_add_f32 v16, v22, v26 // (op 4362) 
v_add_f32 v17, v23, v27 // (op 4362) 


v_cvt_f16_f32 v20, v14 // (op 4363) 
v_cvt_f16_f32 v21, v15 // (op 4363) 
v_cvt_f16_f32 v22, v16 // (op 4363) 
v_cvt_f16_f32 v23, v17 // (op 4363) 


v_pack_b32_f16 v14, v20, v21 // (op 4365) 
v_pack_b32_f16 v15, v22, v23 // (op 4365) 

v_add_u32 v0, 128, v5 // (op 4365) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4365) Store value





s_lshl_b32 s15, s4, 6 // (op 4379) 
s_add_u32 s15, s15, 128 // (op 4379) 
v_add_u32 v0, v4, s15 // (op 4379) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4379) Load value

v_accvgpr_read v20, a36 // (op 4391) 
v_accvgpr_read v21, a37 // (op 4391) 
v_accvgpr_read v22, a38 // (op 4391) 
v_accvgpr_read v23, a39 // (op 4391) 
v_mul_f32 v24, v19, v20 // (op 4391) 
v_mul_f32 v25, v19, v21 // (op 4391) 
v_mul_f32 v26, v19, v22 // (op 4391) 
v_mul_f32 v27, v19, v23 // (op 4391) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4376) 
v_lshrrev_b32 v17, 16, v14 // (op 4376) 
v_cvt_f32_f16 v17, v17 // (op 4376) 
v_mul_f32 v20, v18, v16 // (op 4376) 
v_mul_f32 v21, v18, v17 // (op 4376) 
v_cvt_f32_f16 v16, v15 // (op 4376) 
v_lshrrev_b32 v17, 16, v15 // (op 4376) 
v_cvt_f32_f16 v17, v17 // (op 4376) 
v_mul_f32 v22, v18, v16 // (op 4376) 
v_mul_f32 v23, v18, v17 // (op 4376) 


v_add_f32 v14, v20, v24 // (op 4377) 
v_add_f32 v15, v21, v25 // (op 4377) 
v_add_f32 v16, v22, v26 // (op 4377) 
v_add_f32 v17, v23, v27 // (op 4377) 


v_cvt_f16_f32 v20, v14 // (op 4378) 
v_cvt_f16_f32 v21, v15 // (op 4378) 
v_cvt_f16_f32 v22, v16 // (op 4378) 
v_cvt_f16_f32 v23, v17 // (op 4378) 


v_pack_b32_f16 v14, v20, v21 // (op 4367) 
v_pack_b32_f16 v15, v22, v23 // (op 4367) 

s_lshl_b32 s15, s24, 6 // (op 4367) 
s_add_u32 s15, s15, 128 // (op 4367) 
v_add_u32 v0, v5, s15 // (op 4367) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4367) Store value





s_lshl_b32 s15, s4, 7 // (op 4384) 
s_add_u32 s15, s15, 128 // (op 4384) 
v_add_u32 v0, v4, s15 // (op 4384) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4384) Load value

v_accvgpr_read v20, a40 // (op 4393) 
v_accvgpr_read v21, a41 // (op 4393) 
v_accvgpr_read v22, a42 // (op 4393) 
v_accvgpr_read v23, a43 // (op 4393) 
v_mul_f32 v24, v19, v20 // (op 4393) 
v_mul_f32 v25, v19, v21 // (op 4393) 
v_mul_f32 v26, v19, v22 // (op 4393) 
v_mul_f32 v27, v19, v23 // (op 4393) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4381) 
v_lshrrev_b32 v17, 16, v14 // (op 4381) 
v_cvt_f32_f16 v17, v17 // (op 4381) 
v_mul_f32 v20, v18, v16 // (op 4381) 
v_mul_f32 v21, v18, v17 // (op 4381) 
v_cvt_f32_f16 v16, v15 // (op 4381) 
v_lshrrev_b32 v17, 16, v15 // (op 4381) 
v_cvt_f32_f16 v17, v17 // (op 4381) 
v_mul_f32 v22, v18, v16 // (op 4381) 
v_mul_f32 v23, v18, v17 // (op 4381) 


v_add_f32 v14, v20, v24 // (op 4382) 
v_add_f32 v15, v21, v25 // (op 4382) 
v_add_f32 v16, v22, v26 // (op 4382) 
v_add_f32 v17, v23, v27 // (op 4382) 


v_cvt_f16_f32 v20, v14 // (op 4383) 
v_cvt_f16_f32 v21, v15 // (op 4383) 
v_cvt_f16_f32 v22, v16 // (op 4383) 
v_cvt_f16_f32 v23, v17 // (op 4383) 


v_pack_b32_f16 v14, v20, v21 // (op 4369) 
v_pack_b32_f16 v15, v22, v23 // (op 4369) 

s_lshl_b32 s15, s24, 7 // (op 4369) 
s_add_u32 s15, s15, 128 // (op 4369) 
v_add_u32 v0, v5, s15 // (op 4369) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4369) Store value





s_lshl_b32 s15, s4, 6 // (op 4389) 
s_mul_i32 s6, 3, s15 // (op 4389) 
s_add_u32 s6, s6, 128 // (op 4389) 

v_add_u32 v0, v4, s6 // (op 4389) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4389) Load value

v_accvgpr_read v20, a44 // (op 4395) 
v_accvgpr_read v21, a45 // (op 4395) 
v_accvgpr_read v22, a46 // (op 4395) 
v_accvgpr_read v23, a47 // (op 4395) 
v_mul_f32 v24, v19, v20 // (op 4395) 
v_mul_f32 v25, v19, v21 // (op 4395) 
v_mul_f32 v26, v19, v22 // (op 4395) 
v_mul_f32 v27, v19, v23 // (op 4395) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4386) 
v_lshrrev_b32 v17, 16, v14 // (op 4386) 
v_cvt_f32_f16 v17, v17 // (op 4386) 
v_mul_f32 v20, v18, v16 // (op 4386) 
v_mul_f32 v21, v18, v17 // (op 4386) 
v_cvt_f32_f16 v16, v15 // (op 4386) 
v_lshrrev_b32 v17, 16, v15 // (op 4386) 
v_cvt_f32_f16 v17, v17 // (op 4386) 
v_mul_f32 v22, v18, v16 // (op 4386) 
v_mul_f32 v23, v18, v17 // (op 4386) 


v_add_f32 v14, v20, v24 // (op 4387) 
v_add_f32 v15, v21, v25 // (op 4387) 
v_add_f32 v16, v22, v26 // (op 4387) 
v_add_f32 v17, v23, v27 // (op 4387) 


v_cvt_f16_f32 v20, v14 // (op 4388) 
v_cvt_f16_f32 v21, v15 // (op 4388) 
v_cvt_f16_f32 v22, v16 // (op 4388) 
v_cvt_f16_f32 v23, v17 // (op 4388) 


v_pack_b32_f16 v14, v20, v21 // (op 4371) 
v_pack_b32_f16 v15, v22, v23 // (op 4371) 

s_lshl_b32 s15, s24, 6 // (op 4371) 
s_mul_i32 s6, 3, s15 // (op 4371) 
s_add_u32 s6, s6, 128 // (op 4371) 

v_add_u32 v0, v5, s6 // (op 4371) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4371) Store value





v_add_u32 v0, 64, v4 // (op 4287) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4287) Load value

v_accvgpr_read v20, a16 // (op 4274) 
v_accvgpr_read v21, a17 // (op 4274) 
v_accvgpr_read v22, a18 // (op 4274) 
v_accvgpr_read v23, a19 // (op 4274) 
v_mul_f32 v24, v19, v20 // (op 4274) 
v_mul_f32 v25, v19, v21 // (op 4274) 
v_mul_f32 v26, v19, v22 // (op 4274) 
v_mul_f32 v27, v19, v23 // (op 4274) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4286) 
v_lshrrev_b32 v17, 16, v14 // (op 4286) 
v_cvt_f32_f16 v17, v17 // (op 4286) 
v_mul_f32 v20, v18, v16 // (op 4286) 
v_mul_f32 v21, v18, v17 // (op 4286) 
v_cvt_f32_f16 v16, v15 // (op 4286) 
v_lshrrev_b32 v17, 16, v15 // (op 4286) 
v_cvt_f32_f16 v17, v17 // (op 4286) 
v_mul_f32 v22, v18, v16 // (op 4286) 
v_mul_f32 v23, v18, v17 // (op 4286) 


v_add_f32 v14, v20, v24 // (op 4275) 
v_add_f32 v15, v21, v25 // (op 4275) 
v_add_f32 v16, v22, v26 // (op 4275) 
v_add_f32 v17, v23, v27 // (op 4275) 


v_cvt_f16_f32 v20, v14 // (op 4276) 
v_cvt_f16_f32 v21, v15 // (op 4276) 
v_cvt_f16_f32 v22, v16 // (op 4276) 
v_cvt_f16_f32 v23, v17 // (op 4276) 


v_pack_b32_f16 v14, v20, v21 // (op 4278) 
v_pack_b32_f16 v15, v22, v23 // (op 4278) 

v_add_u32 v0, 64, v5 // (op 4278) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4278) Store value





s_lshl_b32 s15, s4, 6 // (op 4292) 
s_add_u32 s15, s15, 64 // (op 4292) 
v_add_u32 v0, v4, s15 // (op 4292) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4292) Load value

v_accvgpr_read v20, a20 // (op 4304) 
v_accvgpr_read v21, a21 // (op 4304) 
v_accvgpr_read v22, a22 // (op 4304) 
v_accvgpr_read v23, a23 // (op 4304) 
v_mul_f32 v24, v19, v20 // (op 4304) 
v_mul_f32 v25, v19, v21 // (op 4304) 
v_mul_f32 v26, v19, v22 // (op 4304) 
v_mul_f32 v27, v19, v23 // (op 4304) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4289) 
v_lshrrev_b32 v17, 16, v14 // (op 4289) 
v_cvt_f32_f16 v17, v17 // (op 4289) 
v_mul_f32 v20, v18, v16 // (op 4289) 
v_mul_f32 v21, v18, v17 // (op 4289) 
v_cvt_f32_f16 v16, v15 // (op 4289) 
v_lshrrev_b32 v17, 16, v15 // (op 4289) 
v_cvt_f32_f16 v17, v17 // (op 4289) 
v_mul_f32 v22, v18, v16 // (op 4289) 
v_mul_f32 v23, v18, v17 // (op 4289) 


v_add_f32 v14, v20, v24 // (op 4290) 
v_add_f32 v15, v21, v25 // (op 4290) 
v_add_f32 v16, v22, v26 // (op 4290) 
v_add_f32 v17, v23, v27 // (op 4290) 


v_cvt_f16_f32 v20, v14 // (op 4291) 
v_cvt_f16_f32 v21, v15 // (op 4291) 
v_cvt_f16_f32 v22, v16 // (op 4291) 
v_cvt_f16_f32 v23, v17 // (op 4291) 


v_pack_b32_f16 v14, v20, v21 // (op 4280) 
v_pack_b32_f16 v15, v22, v23 // (op 4280) 

s_lshl_b32 s15, s24, 6 // (op 4280) 
s_add_u32 s15, s15, 64 // (op 4280) 
v_add_u32 v0, v5, s15 // (op 4280) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4280) Store value





s_lshl_b32 s15, s4, 7 // (op 4297) 
s_add_u32 s15, s15, 64 // (op 4297) 
v_add_u32 v0, v4, s15 // (op 4297) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4297) Load value

v_accvgpr_read v20, a24 // (op 4306) 
v_accvgpr_read v21, a25 // (op 4306) 
v_accvgpr_read v22, a26 // (op 4306) 
v_accvgpr_read v23, a27 // (op 4306) 
v_mul_f32 v24, v19, v20 // (op 4306) 
v_mul_f32 v25, v19, v21 // (op 4306) 
v_mul_f32 v26, v19, v22 // (op 4306) 
v_mul_f32 v27, v19, v23 // (op 4306) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4294) 
v_lshrrev_b32 v17, 16, v14 // (op 4294) 
v_cvt_f32_f16 v17, v17 // (op 4294) 
v_mul_f32 v20, v18, v16 // (op 4294) 
v_mul_f32 v21, v18, v17 // (op 4294) 
v_cvt_f32_f16 v16, v15 // (op 4294) 
v_lshrrev_b32 v17, 16, v15 // (op 4294) 
v_cvt_f32_f16 v17, v17 // (op 4294) 
v_mul_f32 v22, v18, v16 // (op 4294) 
v_mul_f32 v23, v18, v17 // (op 4294) 


v_add_f32 v14, v20, v24 // (op 4295) 
v_add_f32 v15, v21, v25 // (op 4295) 
v_add_f32 v16, v22, v26 // (op 4295) 
v_add_f32 v17, v23, v27 // (op 4295) 


v_cvt_f16_f32 v20, v14 // (op 4296) 
v_cvt_f16_f32 v21, v15 // (op 4296) 
v_cvt_f16_f32 v22, v16 // (op 4296) 
v_cvt_f16_f32 v23, v17 // (op 4296) 


v_pack_b32_f16 v14, v20, v21 // (op 4282) 
v_pack_b32_f16 v15, v22, v23 // (op 4282) 

s_lshl_b32 s15, s24, 7 // (op 4282) 
s_add_u32 s15, s15, 64 // (op 4282) 
v_add_u32 v0, v5, s15 // (op 4282) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4282) Store value





s_lshl_b32 s15, s4, 6 // (op 4302) 
s_mul_i32 s6, 3, s15 // (op 4302) 
s_add_u32 s6, s6, 64 // (op 4302) 

v_add_u32 v0, v4, s6 // (op 4302) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4302) Load value

v_accvgpr_read v20, a28 // (op 4308) 
v_accvgpr_read v21, a29 // (op 4308) 
v_accvgpr_read v22, a30 // (op 4308) 
v_accvgpr_read v23, a31 // (op 4308) 
v_mul_f32 v24, v19, v20 // (op 4308) 
v_mul_f32 v25, v19, v21 // (op 4308) 
v_mul_f32 v26, v19, v22 // (op 4308) 
v_mul_f32 v27, v19, v23 // (op 4308) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4299) 
v_lshrrev_b32 v17, 16, v14 // (op 4299) 
v_cvt_f32_f16 v17, v17 // (op 4299) 
v_mul_f32 v20, v18, v16 // (op 4299) 
v_mul_f32 v21, v18, v17 // (op 4299) 
v_cvt_f32_f16 v16, v15 // (op 4299) 
v_lshrrev_b32 v17, 16, v15 // (op 4299) 
v_cvt_f32_f16 v17, v17 // (op 4299) 
v_mul_f32 v22, v18, v16 // (op 4299) 
v_mul_f32 v23, v18, v17 // (op 4299) 


v_add_f32 v14, v20, v24 // (op 4300) 
v_add_f32 v15, v21, v25 // (op 4300) 
v_add_f32 v16, v22, v26 // (op 4300) 
v_add_f32 v17, v23, v27 // (op 4300) 


v_cvt_f16_f32 v20, v14 // (op 4301) 
v_cvt_f16_f32 v21, v15 // (op 4301) 
v_cvt_f16_f32 v22, v16 // (op 4301) 
v_cvt_f16_f32 v23, v17 // (op 4301) 


v_pack_b32_f16 v14, v20, v21 // (op 4284) 
v_pack_b32_f16 v15, v22, v23 // (op 4284) 

s_lshl_b32 s15, s24, 6 // (op 4284) 
s_mul_i32 s6, 3, s15 // (op 4284) 
s_add_u32 s6, s6, 64 // (op 4284) 

v_add_u32 v0, v5, s6 // (op 4284) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4284) Store value





v_mov_b32 v0, v4 // (op 263) reg expression

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 263) Load value

v_accvgpr_read v20, a0 // (op 259) 
v_accvgpr_read v21, a1 // (op 259) 
v_accvgpr_read v22, a2 // (op 259) 
v_accvgpr_read v23, a3 // (op 259) 
v_mul_f32 v24, v19, v20 // (op 259) 
v_mul_f32 v25, v19, v21 // (op 259) 
v_mul_f32 v26, v19, v22 // (op 259) 
v_mul_f32 v27, v19, v23 // (op 259) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 264) 
v_lshrrev_b32 v17, 16, v14 // (op 264) 
v_cvt_f32_f16 v17, v17 // (op 264) 
v_mul_f32 v20, v18, v16 // (op 264) 
v_mul_f32 v21, v18, v17 // (op 264) 
v_cvt_f32_f16 v16, v15 // (op 264) 
v_lshrrev_b32 v17, 16, v15 // (op 264) 
v_cvt_f32_f16 v17, v17 // (op 264) 
v_mul_f32 v22, v18, v16 // (op 264) 
v_mul_f32 v23, v18, v17 // (op 264) 


v_add_f32 v14, v20, v24 // (op 260) 
v_add_f32 v15, v21, v25 // (op 260) 
v_add_f32 v16, v22, v26 // (op 260) 
v_add_f32 v17, v23, v27 // (op 260) 


v_cvt_f16_f32 v20, v14 // (op 261) 
v_cvt_f16_f32 v21, v15 // (op 261) 
v_cvt_f16_f32 v22, v16 // (op 261) 
v_cvt_f16_f32 v23, v17 // (op 261) 


v_pack_b32_f16 v14, v20, v21 // (op 262) 
v_pack_b32_f16 v15, v22, v23 // (op 262) 

v_mov_b32 v0, v5 // (op 262) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 262) Store value





s_lshl_b32 s15, s4, 6 // (op 4195) 
v_add_u32 v0, v4, s15 // (op 4195) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4195) Load value

v_accvgpr_read v20, a4 // (op 4191) 
v_accvgpr_read v21, a5 // (op 4191) 
v_accvgpr_read v22, a6 // (op 4191) 
v_accvgpr_read v23, a7 // (op 4191) 
v_mul_f32 v24, v19, v20 // (op 4191) 
v_mul_f32 v25, v19, v21 // (op 4191) 
v_mul_f32 v26, v19, v22 // (op 4191) 
v_mul_f32 v27, v19, v23 // (op 4191) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4196) 
v_lshrrev_b32 v17, 16, v14 // (op 4196) 
v_cvt_f32_f16 v17, v17 // (op 4196) 
v_mul_f32 v20, v18, v16 // (op 4196) 
v_mul_f32 v21, v18, v17 // (op 4196) 
v_cvt_f32_f16 v16, v15 // (op 4196) 
v_lshrrev_b32 v17, 16, v15 // (op 4196) 
v_cvt_f32_f16 v17, v17 // (op 4196) 
v_mul_f32 v22, v18, v16 // (op 4196) 
v_mul_f32 v23, v18, v17 // (op 4196) 


v_add_f32 v14, v20, v24 // (op 4192) 
v_add_f32 v15, v21, v25 // (op 4192) 
v_add_f32 v16, v22, v26 // (op 4192) 
v_add_f32 v17, v23, v27 // (op 4192) 


v_cvt_f16_f32 v20, v14 // (op 4193) 
v_cvt_f16_f32 v21, v15 // (op 4193) 
v_cvt_f16_f32 v22, v16 // (op 4193) 
v_cvt_f16_f32 v23, v17 // (op 4193) 


v_pack_b32_f16 v14, v20, v21 // (op 4194) 
v_pack_b32_f16 v15, v22, v23 // (op 4194) 

s_lshl_b32 s15, s24, 6 // (op 4194) 
v_add_u32 v0, v5, s15 // (op 4194) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4194) Store value





s_lshl_b32 s15, s4, 7 // (op 4209) 
v_add_u32 v0, v4, s15 // (op 4209) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4209) Load value

v_accvgpr_read v20, a8 // (op 4205) 
v_accvgpr_read v21, a9 // (op 4205) 
v_accvgpr_read v22, a10 // (op 4205) 
v_accvgpr_read v23, a11 // (op 4205) 
v_mul_f32 v24, v19, v20 // (op 4205) 
v_mul_f32 v25, v19, v21 // (op 4205) 
v_mul_f32 v26, v19, v22 // (op 4205) 
v_mul_f32 v27, v19, v23 // (op 4205) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4210) 
v_lshrrev_b32 v17, 16, v14 // (op 4210) 
v_cvt_f32_f16 v17, v17 // (op 4210) 
v_mul_f32 v20, v18, v16 // (op 4210) 
v_mul_f32 v21, v18, v17 // (op 4210) 
v_cvt_f32_f16 v16, v15 // (op 4210) 
v_lshrrev_b32 v17, 16, v15 // (op 4210) 
v_cvt_f32_f16 v17, v17 // (op 4210) 
v_mul_f32 v22, v18, v16 // (op 4210) 
v_mul_f32 v23, v18, v17 // (op 4210) 


v_add_f32 v14, v20, v24 // (op 4206) 
v_add_f32 v15, v21, v25 // (op 4206) 
v_add_f32 v16, v22, v26 // (op 4206) 
v_add_f32 v17, v23, v27 // (op 4206) 


v_cvt_f16_f32 v20, v14 // (op 4207) 
v_cvt_f16_f32 v21, v15 // (op 4207) 
v_cvt_f16_f32 v22, v16 // (op 4207) 
v_cvt_f16_f32 v23, v17 // (op 4207) 


v_pack_b32_f16 v14, v20, v21 // (op 4208) 
v_pack_b32_f16 v15, v22, v23 // (op 4208) 

s_lshl_b32 s15, s24, 7 // (op 4208) 
v_add_u32 v0, v5, s15 // (op 4208) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4208) Store value





s_lshl_b32 s15, s4, 6 // (op 4223) 
s_mul_i32 s6, 3, s15 // (op 4223) 

v_add_u32 v0, v4, s6 // (op 4223) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 4223) Load value

v_accvgpr_read v20, a12 // (op 4219) 
v_accvgpr_read v21, a13 // (op 4219) 
v_accvgpr_read v22, a14 // (op 4219) 
v_accvgpr_read v23, a15 // (op 4219) 
v_mul_f32 v24, v19, v20 // (op 4219) 
v_mul_f32 v25, v19, v21 // (op 4219) 
v_mul_f32 v26, v19, v22 // (op 4219) 
v_mul_f32 v27, v19, v23 // (op 4219) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 4224) 
v_lshrrev_b32 v17, 16, v14 // (op 4224) 
v_cvt_f32_f16 v17, v17 // (op 4224) 
v_mul_f32 v20, v18, v16 // (op 4224) 
v_mul_f32 v21, v18, v17 // (op 4224) 
v_cvt_f32_f16 v16, v15 // (op 4224) 
v_lshrrev_b32 v17, 16, v15 // (op 4224) 
v_cvt_f32_f16 v17, v17 // (op 4224) 
v_mul_f32 v22, v18, v16 // (op 4224) 
v_mul_f32 v23, v18, v17 // (op 4224) 


v_add_f32 v14, v20, v24 // (op 4220) 
v_add_f32 v15, v21, v25 // (op 4220) 
v_add_f32 v16, v22, v26 // (op 4220) 
v_add_f32 v17, v23, v27 // (op 4220) 


v_cvt_f16_f32 v20, v14 // (op 4221) 
v_cvt_f16_f32 v21, v15 // (op 4221) 
v_cvt_f16_f32 v22, v16 // (op 4221) 
v_cvt_f16_f32 v23, v17 // (op 4221) 


v_pack_b32_f16 v14, v20, v21 // (op 4222) 
v_pack_b32_f16 v15, v22, v23 // (op 4222) 

s_lshl_b32 s15, s24, 6 // (op 4222) 
s_mul_i32 s6, 3, s15 // (op 4222) 

v_add_u32 v0, v5, s6 // (op 4222) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 4222) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_238:
s_add_u32 s0, s0, s1 // (op 451) 


s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:

























































s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end:
.size RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2, .LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end-RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51200 // lds bytes
  .amdhsa_accum_offset 148
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
  - .name: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 51200
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 148
    .agpr_count: 64
    .max_flat_workgroup_size: 256
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

