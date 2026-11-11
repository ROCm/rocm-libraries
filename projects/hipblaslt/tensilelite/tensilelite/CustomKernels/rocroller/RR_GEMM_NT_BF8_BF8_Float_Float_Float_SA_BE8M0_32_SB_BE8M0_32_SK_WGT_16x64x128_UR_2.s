// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 2619) 
v_and_b32 v3, 63, v1 // (op 2619) 
v_sub_u32 v4, v3, v0 // (op 2619) 

v_lshrrev_b32 v3, 1, v4 // (op 2619) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 2619) 


v_and_b32 v0, 1, v4 // (op 2619) 

v_lshlrev_b32 v3, 3, v0 // (op 2619) 

v_lshrrev_b32 v0, 7, v1 // (op 2619) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2619) 


v_add_u32 v0, v5, v4 // (op 2619) 


v_and_b32 v3, 48, v1 // (op 2877) 
v_and_b32 v4, 63, v1 // (op 2877) 
v_sub_u32 v5, v4, v3 // (op 2877) 

v_lshrrev_b32 v4, 1, v5 // (op 2877) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 2877) 


v_and_b32 v3, 1, v5 // (op 2877) 

v_lshlrev_b32 v4, 3, v3 // (op 2877) 

v_lshrrev_b32 v3, 7, v1 // (op 2877) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 2877) 


v_add_u32 v3, v6, v5 // (op 2877) 


v_and_b32 v4, 48, v1 // (op 3122) 
v_and_b32 v5, 63, v1 // (op 3122) 
v_sub_u32 v6, v5, v4 // (op 3122) 

v_lshrrev_b32 v5, 1, v6 // (op 3122) 
v_add_lshl_u32 v7, v4, v5, 4 // (op 3122) 


v_and_b32 v4, 1, v6 // (op 3122) 

v_lshlrev_b32 v5, 3, v4 // (op 3122) 

v_lshrrev_b32 v4, 7, v1 // (op 3122) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 3122) 


v_add_u32 v4, v7, v6 // (op 3122) 


v_and_b32 v5, 48, v1 // (op 3234) 
v_and_b32 v6, 63, v1 // (op 3234) 
v_sub_u32 v7, v6, v5 // (op 3234) 

v_lshrrev_b32 v6, 1, v7 // (op 3234) 
v_add_lshl_u32 v8, v5, v6, 4 // (op 3234) 


v_and_b32 v5, 1, v7 // (op 3234) 

v_lshlrev_b32 v6, 3, v5 // (op 3234) 

v_lshrrev_b32 v5, 7, v1 // (op 3234) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 3234) 


v_add_u32 v5, v8, v7 // (op 3234) 


v_and_b32 v6, 48, v1 // (op 2689) 
v_and_b32 v7, 63, v1 // (op 2689) 
v_sub_u32 v8, v7, v6 // (op 2689) 

v_lshrrev_b32 v7, 1, v8 // (op 2689) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 2689) 


v_and_b32 v6, 1, v8 // (op 2689) 

v_lshlrev_b32 v7, 3, v6 // (op 2689) 

v_lshrrev_b32 v6, 6, v1 // (op 2689) 
v_and_b32 v8, 1, v6 // (op 2689) 

v_lshl_add_u32 v6, v8, 4, v7 // (op 2689) 


v_add_u32 v7, v9, v6 // (op 2689) 


v_lshrrev_b32 v6, 6, v1 // (op 2765) 
v_and_b32 v8, 1, v6 // (op 2765) 

v_and_b32 v6, 15, v1 // (op 2765) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 2765) 


v_and_b32 v6, 63, v1 // (op 2765) 
v_lshrrev_b32 v8, 4, v6 // (op 2765) 

v_lshl_add_u32 v6, v8, 6, v9 // (op 2765) 


v_and_b32 v8, 48, v1 // (op 2947) 
v_and_b32 v9, 63, v1 // (op 2947) 
v_sub_u32 v10, v9, v8 // (op 2947) 

v_lshrrev_b32 v9, 1, v10 // (op 2947) 
v_add_lshl_u32 v11, v8, v9, 6 // (op 2947) 


v_and_b32 v8, 1, v10 // (op 2947) 

v_lshlrev_b32 v9, 3, v8 // (op 2947) 

v_lshrrev_b32 v8, 6, v1 // (op 2947) 
v_and_b32 v10, 1, v8 // (op 2947) 

v_lshl_add_u32 v8, v10, 4, v9 // (op 2947) 


v_add_u32 v9, v11, v8 // (op 2947) 


v_lshrrev_b32 v8, 6, v1 // (op 3023) 
v_and_b32 v10, 1, v8 // (op 3023) 

v_and_b32 v8, 15, v1 // (op 3023) 
v_lshl_add_u32 v11, v10, 4, v8 // (op 3023) 


v_and_b32 v8, 63, v1 // (op 3023) 
v_lshrrev_b32 v10, 4, v8 // (op 3023) 

v_lshl_add_u32 v8, v10, 6, v11 // (op 3023) 


v_and_b32 v10, 48, v1 // (op 3166) 
v_and_b32 v11, 63, v1 // (op 3166) 
v_sub_u32 v12, v11, v10 // (op 3166) 

v_lshrrev_b32 v11, 1, v12 // (op 3166) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 3166) 


v_and_b32 v10, 1, v12 // (op 3166) 

v_lshlrev_b32 v11, 3, v10 // (op 3166) 

v_lshrrev_b32 v10, 6, v1 // (op 3166) 
v_and_b32 v12, 1, v10 // (op 3166) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 3166) 


v_add_u32 v11, v13, v10 // (op 3166) 


v_lshrrev_b32 v10, 6, v1 // (op 3195) 
v_and_b32 v12, 1, v10 // (op 3195) 

v_and_b32 v10, 15, v1 // (op 3195) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3195) 


v_and_b32 v10, 63, v1 // (op 3195) 
v_lshrrev_b32 v12, 4, v10 // (op 3195) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 3195) 


v_and_b32 v12, 48, v1 // (op 3278) 
v_and_b32 v13, 63, v1 // (op 3278) 
v_sub_u32 v14, v13, v12 // (op 3278) 

v_lshrrev_b32 v13, 1, v14 // (op 3278) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3278) 


v_and_b32 v12, 1, v14 // (op 3278) 

v_lshlrev_b32 v13, 3, v12 // (op 3278) 

v_lshrrev_b32 v12, 6, v1 // (op 3278) 
v_and_b32 v14, 1, v12 // (op 3278) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 3278) 


v_add_u32 v13, v15, v12 // (op 3278) 


v_lshrrev_b32 v12, 6, v1 // (op 3307) 
v_and_b32 v14, 1, v12 // (op 3307) 

v_and_b32 v12, 15, v1 // (op 3307) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 3307) 


v_and_b32 v12, 63, v1 // (op 3307) 
v_lshrrev_b32 v14, 4, v12 // (op 3307) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 3307) 


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
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 7, v1 // (op 2540) 
s_mul_i32 s52, s16, s8 // (op 2540) 
s_add_u32 s52, s52, s0 // (op 2540) 
v_and_b32 v17, 63, v1 // (op 2540) 
s_mul_hi_u32 s53, s52, s13 // (op 2540) 
v_lshrrev_b32 v18, 4, v17 // (op 2540) 
s_sub_u32 s54, s52, s53 // (op 2540) 
v_lshlrev_b32 v19, 2, v18 // (op 2540) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2540) 


s_lshr_b32 s55, s54, 1 // (op 2540) 

s_add_u32 s54, s55, s53 // (op 2540) 


v_lshrrev_b32 v16, 2, v17 // (op 2540) 

s_lshr_b32 s53, s54, s11 // (op 2540) 

v_and_b32 v17, 3, v16 // (op 2540) 

s_and_b32 s54, s11, 2147483648 // (op 2540) 
v_and_b32 v16, 3, v1 // (op 2540) 
s_cmp_eq_i32 s54, 0 // (op 2540) 
s_mov_b32 s55, scc // (op 2540) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 2540) 


s_cmp_eq_i32 s55, 0 // (op 2540) 
s_cselect_b32 s54, s52, s53 // (op 2540) 



v_lshrrev_b32 v16, 6, v1 // (op 2540) 
s_mul_hi_u32 s52, s54, s12 // (op 2540) 
v_and_b32 v17, 1, v16 // (op 2540) 

s_sub_u32 s53, s54, s52 // (op 2540) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 2540) 


s_lshr_b32 s55, s53, 1 // (op 2540) 

s_add_u32 s53, s55, s52 // (op 2540) 


s_lshr_b32 s52, s53, s10 // (op 2540) 

s_and_b32 s53, s10, 2147483648 // (op 2540) 
s_cmp_eq_i32 s53, 0 // (op 2540) 
s_mov_b32 s55, scc // (op 2540) 

s_cmp_eq_i32 s55, 0 // (op 2540) 
s_cselect_b32 s53, s54, s52 // (op 2540) 


s_mul_i32 s52, s53, s14 // (op 2540) 
v_lshl_add_u32 v17, s53, 4, v18 // (op 2540) 


s_sub_u32 s53, s54, s52 // (op 2540) 


v_lshl_add_u32 v18, s53, 6, v16 // (op 2540) 


v_mul_lo_u32 v16, v18, s4 // (op 2540) 

v_add_lshl_u32 v18, v17, v16, 2 // (op 2540) 


s_mul_i32 s53, s4, s23 // (op 2542) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2542) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2542) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2542) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2542) most significant: sum
s_add_u32 s53, s53, s55 // (op 2542) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2542) 

s_mov_b32 s56, s20 // (op 2542) reg expression
s_mov_b32 s57, s21 // (op 2542) reg expression
s_mov_b32 s58, s54 // (op 2542) reg expression
s_mov_b32 s59, 131072 // (op 2542) call()

v_and_b32 v16, 48, v1 // (op 2631) 
v_and_b32 v17, 63, v1 // (op 2631) 
v_sub_u32 v19, v17, v16 // (op 2631) 

v_lshrrev_b32 v17, 1, v19 // (op 2631) 
v_add_lshl_u32 v20, v16, v17, 4 // (op 2631) 


v_and_b32 v16, 1, v19 // (op 2631) 

v_lshlrev_b32 v17, 3, v16 // (op 2631) 

v_lshrrev_b32 v16, 7, v1 // (op 2631) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 2631) 


v_add_u32 v16, v20, v19 // (op 2631) 


v_and_b32 v17, 48, v1 // (op 2706) 
v_and_b32 v19, 63, v1 // (op 2706) 
v_sub_u32 v20, v19, v17 // (op 2706) 

v_lshrrev_b32 v19, 1, v20 // (op 2706) 
v_add_lshl_u32 v21, v17, v19, 6 // (op 2706) 


v_and_b32 v17, 1, v20 // (op 2706) 

v_lshlrev_b32 v19, 3, v17 // (op 2706) 

v_lshrrev_b32 v17, 6, v1 // (op 2706) 
v_and_b32 v20, 1, v17 // (op 2706) 

v_lshl_add_u32 v17, v20, 4, v19 // (op 2706) 


v_add_u32 v19, v21, v17 // (op 2706) 


v_lshrrev_b32 v17, 6, v1 // (op 2782) 
v_and_b32 v20, 1, v17 // (op 2782) 

v_and_b32 v17, 15, v1 // (op 2782) 
v_lshl_add_u32 v21, v20, 4, v17 // (op 2782) 


v_and_b32 v17, 63, v1 // (op 2782) 
v_lshrrev_b32 v20, 4, v17 // (op 2782) 

v_lshl_add_u32 v17, v20, 6, v21 // (op 2782) 


v_lshrrev_b32 v20, 7, v1 // (op 2800) 
s_mul_i32 s52, s16, s8 // (op 2800) 
s_add_u32 s52, s52, s0 // (op 2800) 
v_and_b32 v21, 63, v1 // (op 2800) 
s_mul_hi_u32 s53, s52, s13 // (op 2800) 
v_lshrrev_b32 v22, 4, v21 // (op 2800) 
s_sub_u32 s54, s52, s53 // (op 2800) 
v_lshlrev_b32 v23, 2, v22 // (op 2800) 

s_lshr_b32 s55, s54, 1 // (op 2800) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2800) 


s_add_u32 s54, s55, s53 // (op 2800) 


v_lshrrev_b32 v20, 2, v21 // (op 2800) 

s_lshr_b32 s53, s54, s11 // (op 2800) 

v_and_b32 v21, 3, v20 // (op 2800) 

s_and_b32 s54, s11, 2147483648 // (op 2800) 
v_and_b32 v20, 3, v1 // (op 2800) 
s_cmp_eq_i32 s54, 0 // (op 2800) 
s_mov_b32 s55, scc // (op 2800) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 2800) 


s_cmp_eq_i32 s55, 0 // (op 2800) 
s_cselect_b32 s54, s52, s53 // (op 2800) 



v_lshrrev_b32 v20, 6, v1 // (op 2800) 
s_mul_hi_u32 s55, s54, s12 // (op 2800) 
v_and_b32 v21, 1, v20 // (op 2800) 

s_sub_u32 s52, s54, s55 // (op 2800) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 2800) 


s_lshr_b32 s53, s52, 1 // (op 2800) 

s_add_u32 s52, s53, s55 // (op 2800) 


s_lshr_b32 s53, s52, s10 // (op 2800) 

s_and_b32 s52, s10, 2147483648 // (op 2800) 
s_cmp_eq_i32 s52, 0 // (op 2800) 
s_mov_b32 s55, scc // (op 2800) 

s_cmp_eq_i32 s55, 0 // (op 2800) 
s_cselect_b32 s52, s54, s53 // (op 2800) 


s_mul_i32 s53, s52, s14 // (op 2800) 
v_lshl_add_u32 v21, s52, 4, v22 // (op 2800) 


s_sub_u32 s52, s54, s53 // (op 2800) 


v_lshl_add_u32 v22, s52, 6, v20 // (op 2800) 


v_mul_lo_u32 v20, v22, s24 // (op 2800) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 2800) 


s_mul_i32 s53, s24, s23 // (op 2802) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2802) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2802) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2802) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2802) most significant: sum
s_add_u32 s53, s53, s55 // (op 2802) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2802) 

s_mov_b32 s60, s26 // (op 2802) reg expression
s_mov_b32 s61, s27 // (op 2802) reg expression
s_mov_b32 s62, s54 // (op 2802) reg expression
s_mov_b32 s63, 131072 // (op 2802) call()

v_lshrrev_b32 v20, 5, v1 // (op 3058) 
v_lshlrev_b32 v21, 2, v20 // (op 3058) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 3058) 

v_and_b32 v21, -32, v1 // (op 3058) 
v_sub_u32 v23, v1, v21 // (op 3058) 

v_lshl_add_u32 v21, v20, 6, v23 // (op 3058) 


v_lshlrev_b32 v20, 2, v21 // (op 3058) 

s_lshl_b32 s52, s15, 12 // (op 3060) 
s_mov_b32 s64, s28 // (op 3060) reg expression
s_mov_b32 s65, s29 // (op 3060) reg expression
s_mov_b32 s66, s52 // (op 3060) reg expression
s_mov_b32 s67, 131072 // (op 3060) call()

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
v_accvgpr_write a4, 0.00000 // (op 971) call()
v_accvgpr_write a5, 0.00000 // (op 971) call()
v_accvgpr_write a6, 0.00000 // (op 971) call()
v_accvgpr_write a7, 0.00000 // (op 971) call()
s_mul_i32 s52, s16, s8 // (op 2112) 
s_add_u32 s52, s52, s0 // (op 2112) 
s_mul_hi_u32 s53, s52, s13 // (op 2112) 
s_sub_u32 s54, s52, s53 // (op 2112) 
s_lshr_b32 s55, s54, 1 // (op 2112) 

s_add_u32 s54, s55, s53 // (op 2112) 


s_lshr_b32 s53, s54, s11 // (op 2112) 

s_and_b32 s54, s11, 2147483648 // (op 2112) 
s_cmp_eq_i32 s54, 0 // (op 2112) 
s_mov_b32 s55, scc // (op 2112) 

s_cmp_eq_i32 s55, 0 // (op 2112) 
s_cselect_b32 s54, s52, s53 // (op 2112) 


s_mul_hi_u32 s53, s54, s12 // (op 2112) 
s_sub_u32 s55, s54, s53 // (op 2112) 
s_lshr_b32 s68, s55, 1 // (op 2112) 

s_add_u32 s55, s68, s53 // (op 2112) 


s_lshr_b32 s53, s55, s10 // (op 2112) 

s_and_b32 s55, s10, 2147483648 // (op 2112) 
s_cmp_eq_i32 s55, 0 // (op 2112) 
s_mov_b32 s68, scc // (op 2112) 

s_cmp_eq_i32 s68, 0 // (op 2112) 
s_cselect_b32 s55, s54, s53 // (op 2112) 


s_lshl_b32 s53, s55, 4 // (op 2112) 

s_mov_b32 s69, 0 // (op 2112) convert
s_mov_b32 s68, s53 // (op 2112) convert

s_ashr_i64 s[70:71], s[2:3], 7 // (op 2112) 
s_mul_i32 s53, s54, s70 // (op 2112) 


s_sub_u32 s54, s52, s53 // (op 2112) 


v_lshl_add_u32 v21, s54, 7, v1 // (op 2112) 

v_mul_lo_u32 v25, v21, s31 // (op 2112) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 2112) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 2112) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2112) most significant: sum


v_mov_b32 v21, s68 // (op 2112) 
v_mov_b32 v23, s69 // (op 2112) 
v_add_co_u32 v26, s[52:53], v21, v24 // (op 2112) least significant half
v_addc_co_u32 v27, s[52:53], v23, v25, s[52:53] // (op 2112) most significant half





s_mul_i32 s53, s30, s3 // (op 2114) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2114) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2114) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2114) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2114) most significant: sum
s_add_u32 s53, s53, s55 // (op 2114) most significant: sum


s_mov_b32 s68, s32 // (op 2114) reg expression
s_mov_b32 s69, s33 // (op 2114) reg expression
s_mov_b32 s70, s52 // (op 2114) reg expression
s_mov_b32 s71, 131072 // (op 2114) call()

s_mul_i32 s52, s16, s8 // (op 2131) 
s_add_u32 s52, s52, s0 // (op 2131) 
s_mul_hi_u32 s53, s52, s13 // (op 2131) 
s_sub_u32 s54, s52, s53 // (op 2131) 
s_lshr_b32 s55, s54, 1 // (op 2131) 

s_add_u32 s54, s55, s53 // (op 2131) 


s_lshr_b32 s53, s54, s11 // (op 2131) 

s_and_b32 s54, s11, 2147483648 // (op 2131) 
s_cmp_eq_i32 s54, 0 // (op 2131) 
s_mov_b32 s55, scc // (op 2131) 

s_cmp_eq_i32 s55, 0 // (op 2131) 
s_cselect_b32 s54, s52, s53 // (op 2131) 


s_mul_hi_u32 s53, s54, s12 // (op 2131) 
s_sub_u32 s55, s54, s53 // (op 2131) 
s_lshr_b32 s72, s55, 1 // (op 2131) 

s_add_u32 s55, s72, s53 // (op 2131) 


s_lshr_b32 s53, s55, s10 // (op 2131) 

s_and_b32 s55, s10, 2147483648 // (op 2131) 
s_cmp_eq_i32 s55, 0 // (op 2131) 
s_mov_b32 s72, scc // (op 2131) 

s_cmp_eq_i32 s72, 0 // (op 2131) 
s_cselect_b32 s55, s54, s53 // (op 2131) 


s_lshl_b32 s53, s55, 4 // (op 2131) 

s_mov_b32 s73, 0 // (op 2131) convert
s_mov_b32 s72, s53 // (op 2131) convert

s_ashr_i64 s[74:75], s[2:3], 7 // (op 2131) 
s_mul_i32 s53, s54, s74 // (op 2131) 


s_sub_u32 s54, s52, s53 // (op 2131) 


v_lshl_add_u32 v21, s54, 7, v1 // (op 2131) 

v_mul_lo_u32 v25, v21, s31 // (op 2131) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 2131) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 2131) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2131) most significant: sum


v_mov_b32 v21, s72 // (op 2131) 
v_mov_b32 v23, s73 // (op 2131) 
v_add_co_u32 v28, s[52:53], v21, v24 // (op 2131) least significant half
v_addc_co_u32 v29, s[52:53], v23, v25, s[52:53] // (op 2131) most significant half





s_mul_i32 s53, s30, s3 // (op 2133) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2133) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2133) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2133) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2133) most significant: sum
s_add_u32 s53, s53, s55 // (op 2133) most significant: sum


s_mov_b32 s68, s32 // (op 2133) reg expression
s_mov_b32 s69, s33 // (op 2133) reg expression
s_mov_b32 s70, s52 // (op 2133) reg expression
s_mov_b32 s71, 131072 // (op 2133) call()

v_and_b32 v21, -4, v1 // (op 2240) 
s_mul_i32 s52, s16, s8 // (op 2240) 
s_add_u32 s52, s52, s0 // (op 2240) 
v_sub_u32 v23, v1, v21 // (op 2240) 

s_mul_hi_u32 s53, s52, s13 // (op 2240) 
v_lshlrev_b32 v21, 4, v23 // (op 2240) 

s_sub_u32 s54, s52, s53 // (op 2240) 
v_lshrrev_b32 v23, 2, v1 // (op 2240) 
s_lshr_b32 s55, s54, 1 // (op 2240) 

s_add_u32 s54, s55, s53 // (op 2240) 


s_lshr_b32 s53, s54, s11 // (op 2240) 

s_and_b32 s54, s11, 2147483648 // (op 2240) 
s_cmp_eq_i32 s54, 0 // (op 2240) 
s_mov_b32 s55, scc // (op 2240) 

s_cmp_eq_i32 s55, 0 // (op 2240) 
s_cselect_b32 s54, s52, s53 // (op 2240) 


s_mul_hi_u32 s53, s54, s12 // (op 2240) 
s_sub_u32 s55, s54, s53 // (op 2240) 
s_lshr_b32 s72, s55, 1 // (op 2240) 

s_add_u32 s55, s72, s53 // (op 2240) 


s_lshr_b32 s53, s55, s10 // (op 2240) 

s_and_b32 s55, s10, 2147483648 // (op 2240) 
s_cmp_eq_i32 s55, 0 // (op 2240) 
s_mov_b32 s72, scc // (op 2240) 

s_cmp_eq_i32 s72, 0 // (op 2240) 
s_cselect_b32 s55, s54, s53 // (op 2240) 


s_mul_i32 s53, s55, s14 // (op 2240) 

s_sub_u32 s55, s54, s53 // (op 2240) 

v_lshl_add_u32 v24, s55, 6, v21 // (op 2240) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2240) 
v_mov_b32 v31, 0 // (op 2240) convert
v_mov_b32 v30, v24 // (op 2240) convert

s_mul_i32 s53, s54, s72 // (op 2240) 


s_sub_u32 s54, s52, s53 // (op 2240) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2240) 


v_mul_lo_u32 v25, v21, s35 // (op 2240) most significant: low of low * high
v_mul_hi_u32 v23, v21, s34 // (op 2240) most significant: high of low * low
v_mul_lo_u32 v24, v21, s34 // (op 2240) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2240) most significant: sum

v_add_co_u32 v24, s[52:53], v24, v30 // (op 2240) least significant half
v_addc_co_u32 v25, s[52:53], v25, v31, s[52:53] // (op 2240) most significant half



s_mul_i32 s53, s34, s3 // (op 2242) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2242) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2242) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2242) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2242) most significant: sum
s_add_u32 s53, s53, s55 // (op 2242) most significant: sum


s_mov_b32 s72, s36 // (op 2242) reg expression
s_mov_b32 s73, s37 // (op 2242) reg expression
s_mov_b32 s74, s52 // (op 2242) reg expression
s_mov_b32 s75, 131072 // (op 2242) call()

v_and_b32 v21, -4, v1 // (op 2259) 
s_mul_i32 s52, s16, s8 // (op 2259) 
s_add_u32 s52, s52, s0 // (op 2259) 
v_sub_u32 v23, v1, v21 // (op 2259) 

s_mul_hi_u32 s53, s52, s13 // (op 2259) 
v_lshlrev_b32 v21, 4, v23 // (op 2259) 

s_sub_u32 s54, s52, s53 // (op 2259) 
v_lshrrev_b32 v23, 2, v1 // (op 2259) 
s_lshr_b32 s55, s54, 1 // (op 2259) 

s_add_u32 s54, s55, s53 // (op 2259) 


s_lshr_b32 s53, s54, s11 // (op 2259) 

s_and_b32 s54, s11, 2147483648 // (op 2259) 
s_cmp_eq_i32 s54, 0 // (op 2259) 
s_mov_b32 s55, scc // (op 2259) 

s_cmp_eq_i32 s55, 0 // (op 2259) 
s_cselect_b32 s54, s52, s53 // (op 2259) 


s_mul_hi_u32 s53, s54, s12 // (op 2259) 
s_sub_u32 s55, s54, s53 // (op 2259) 
s_lshr_b32 s76, s55, 1 // (op 2259) 

s_add_u32 s55, s76, s53 // (op 2259) 


s_lshr_b32 s53, s55, s10 // (op 2259) 

s_and_b32 s55, s10, 2147483648 // (op 2259) 
s_cmp_eq_i32 s55, 0 // (op 2259) 
s_mov_b32 s76, scc // (op 2259) 

s_cmp_eq_i32 s76, 0 // (op 2259) 
s_cselect_b32 s55, s54, s53 // (op 2259) 


s_mul_i32 s53, s55, s14 // (op 2259) 

s_sub_u32 s55, s54, s53 // (op 2259) 

v_lshl_add_u32 v30, s55, 6, v21 // (op 2259) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2259) 
v_mov_b32 v33, 0 // (op 2259) convert
v_mov_b32 v32, v30 // (op 2259) convert

s_mul_i32 s53, s54, s76 // (op 2259) 


s_sub_u32 s54, s52, s53 // (op 2259) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2259) 


v_mul_lo_u32 v31, v21, s35 // (op 2259) most significant: low of low * high
v_mul_hi_u32 v23, v21, s34 // (op 2259) most significant: high of low * low
v_mul_lo_u32 v30, v21, s34 // (op 2259) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 2259) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v32 // (op 2259) least significant half
v_addc_co_u32 v31, s[52:53], v31, v33, s[52:53] // (op 2259) most significant half



s_mul_i32 s53, s34, s3 // (op 2261) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2261) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2261) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2261) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2261) most significant: sum
s_add_u32 s53, s53, s55 // (op 2261) most significant: sum


s_mov_b32 s72, s36 // (op 2261) reg expression
s_mov_b32 s73, s37 // (op 2261) reg expression
s_mov_b32 s74, s52 // (op 2261) reg expression
s_mov_b32 s75, 131072 // (op 2261) call()

v_lshrrev_b32 v21, 7, v1 // (op 2368) 
s_mul_i32 s52, s16, s8 // (op 2368) 
s_add_u32 s52, s52, s0 // (op 2368) 
v_and_b32 v23, 15, v1 // (op 2368) 
s_mul_hi_u32 s53, s52, s13 // (op 2368) 
v_lshl_add_u32 v32, v21, 4, v23 // (op 2368) 


s_sub_u32 s54, s52, s53 // (op 2368) 
v_and_b32 v21, 63, v1 // (op 2368) 
s_lshr_b32 s55, s54, 1 // (op 2368) 

v_lshrrev_b32 v23, 4, v21 // (op 2368) 

s_add_u32 s54, s55, s53 // (op 2368) 


s_lshr_b32 s53, s54, s11 // (op 2368) 

s_and_b32 s54, s11, 2147483648 // (op 2368) 
s_cmp_eq_i32 s54, 0 // (op 2368) 
s_mov_b32 s55, scc // (op 2368) 

s_cmp_eq_i32 s55, 0 // (op 2368) 
s_cselect_b32 s54, s52, s53 // (op 2368) 


s_mul_hi_u32 s53, s54, s12 // (op 2368) 
s_sub_u32 s55, s54, s53 // (op 2368) 
s_lshr_b32 s76, s55, 1 // (op 2368) 

s_add_u32 s55, s76, s53 // (op 2368) 


s_lshr_b32 s53, s55, s10 // (op 2368) 

s_and_b32 s55, s10, 2147483648 // (op 2368) 
s_cmp_eq_i32 s55, 0 // (op 2368) 
s_mov_b32 s76, scc // (op 2368) 

s_cmp_eq_i32 s76, 0 // (op 2368) 
s_cselect_b32 s55, s54, s53 // (op 2368) 


v_lshl_add_u32 v21, s55, 4, v32 // (op 2368) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2368) 
s_mul_i32 s53, s54, s76 // (op 2368) 


s_sub_u32 s54, s52, s53 // (op 2368) 


v_lshl_add_u32 v32, s54, 2, v23 // (op 2368) 


v_mul_lo_u32 v23, v32, s38 // (op 2368) 

v_add_u32 v32, v21, v23 // (op 2368) 


s_mul_i32 s53, s38, s43 // (op 2370) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2370) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2370) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2370) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2370) most significant: sum
s_add_u32 s53, s53, s55 // (op 2370) most significant: sum


s_mov_b32 s76, s40 // (op 2370) reg expression
s_mov_b32 s77, s41 // (op 2370) reg expression
s_mov_b32 s78, s52 // (op 2370) reg expression
s_mov_b32 s79, 131072 // (op 2370) call()

v_lshrrev_b32 v21, 7, v1 // (op 2387) 
s_mul_i32 s52, s16, s8 // (op 2387) 
s_add_u32 s52, s52, s0 // (op 2387) 
v_and_b32 v23, 15, v1 // (op 2387) 
s_mul_hi_u32 s53, s52, s13 // (op 2387) 
v_lshl_add_u32 v33, v21, 4, v23 // (op 2387) 


s_sub_u32 s54, s52, s53 // (op 2387) 
v_and_b32 v21, 63, v1 // (op 2387) 
s_lshr_b32 s55, s54, 1 // (op 2387) 

v_lshrrev_b32 v23, 4, v21 // (op 2387) 

s_add_u32 s54, s55, s53 // (op 2387) 


s_lshr_b32 s53, s54, s11 // (op 2387) 

s_and_b32 s54, s11, 2147483648 // (op 2387) 
s_cmp_eq_i32 s54, 0 // (op 2387) 
s_mov_b32 s55, scc // (op 2387) 

s_cmp_eq_i32 s55, 0 // (op 2387) 
s_cselect_b32 s54, s52, s53 // (op 2387) 


s_mul_hi_u32 s53, s54, s12 // (op 2387) 
s_sub_u32 s55, s54, s53 // (op 2387) 
s_lshr_b32 s80, s55, 1 // (op 2387) 

s_add_u32 s55, s80, s53 // (op 2387) 


s_lshr_b32 s53, s55, s10 // (op 2387) 

s_and_b32 s55, s10, 2147483648 // (op 2387) 
s_cmp_eq_i32 s55, 0 // (op 2387) 
s_mov_b32 s80, scc // (op 2387) 

s_cmp_eq_i32 s80, 0 // (op 2387) 
s_cselect_b32 s55, s54, s53 // (op 2387) 


v_lshl_add_u32 v21, s55, 4, v33 // (op 2387) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2387) 
s_mul_i32 s53, s54, s80 // (op 2387) 


s_sub_u32 s54, s52, s53 // (op 2387) 


v_lshl_add_u32 v33, s54, 2, v23 // (op 2387) 


v_mul_lo_u32 v23, v33, s38 // (op 2387) 

v_add_u32 v33, v21, v23 // (op 2387) 


s_mul_i32 s53, s38, s43 // (op 2389) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2389) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2389) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2389) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2389) most significant: sum
s_add_u32 s53, s53, s55 // (op 2389) most significant: sum


s_mov_b32 s76, s40 // (op 2389) reg expression
s_mov_b32 s77, s41 // (op 2389) reg expression
s_mov_b32 s78, s52 // (op 2389) reg expression
s_mov_b32 s79, 131072 // (op 2389) call()

v_and_b32 v21, -64, v1 // (op 2496) 
s_mul_i32 s52, s16, s8 // (op 2496) 
s_add_u32 s52, s52, s0 // (op 2496) 
v_sub_u32 v23, v1, v21 // (op 2496) 

s_mul_hi_u32 s53, s52, s13 // (op 2496) 
v_lshrrev_b32 v21, 6, v1 // (op 2496) 
s_sub_u32 s54, s52, s53 // (op 2496) 
v_lshlrev_b32 v34, 1, v21 // (op 2496) 

s_lshr_b32 s55, s54, 1 // (op 2496) 

s_add_u32 s54, s55, s53 // (op 2496) 


s_lshr_b32 s53, s54, s11 // (op 2496) 

s_and_b32 s54, s11, 2147483648 // (op 2496) 
s_cmp_eq_i32 s54, 0 // (op 2496) 
s_mov_b32 s55, scc // (op 2496) 

s_cmp_eq_i32 s55, 0 // (op 2496) 
s_cselect_b32 s54, s52, s53 // (op 2496) 


s_mul_hi_u32 s53, s54, s12 // (op 2496) 
s_sub_u32 s55, s54, s53 // (op 2496) 
s_lshr_b32 s80, s55, 1 // (op 2496) 

s_add_u32 s55, s80, s53 // (op 2496) 


s_lshr_b32 s53, s55, s10 // (op 2496) 

s_and_b32 s55, s10, 2147483648 // (op 2496) 
s_cmp_eq_i32 s55, 0 // (op 2496) 
s_mov_b32 s80, scc // (op 2496) 

s_cmp_eq_i32 s80, 0 // (op 2496) 
s_cselect_b32 s55, s54, s53 // (op 2496) 


s_mul_i32 s53, s55, s14 // (op 2496) 

s_sub_u32 s55, s54, s53 // (op 2496) 

v_lshl_add_u32 v21, s55, 6, v23 // (op 2496) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2496) 
s_mul_i32 s53, s54, s80 // (op 2496) 


s_sub_u32 s54, s52, s53 // (op 2496) 


v_lshl_add_u32 v23, s54, 2, v34 // (op 2496) 


v_mul_lo_u32 v34, v23, s44 // (op 2496) 
v_add_u32 v34, v34, v21 // (op 2496) 


s_mul_i32 s53, s44, s43 // (op 2498) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2498) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2498) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2498) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2498) most significant: sum
s_add_u32 s53, s53, s55 // (op 2498) most significant: sum


s_mov_b32 s80, s46 // (op 2498) reg expression
s_mov_b32 s81, s47 // (op 2498) reg expression
s_mov_b32 s82, s52 // (op 2498) reg expression
s_mov_b32 s83, 131072 // (op 2498) call()

v_and_b32 v21, -64, v1 // (op 2515) 
s_mul_i32 s52, s16, s8 // (op 2515) 
s_add_u32 s52, s52, s0 // (op 2515) 
v_sub_u32 v23, v1, v21 // (op 2515) 

s_mul_hi_u32 s53, s52, s13 // (op 2515) 
v_lshrrev_b32 v21, 6, v1 // (op 2515) 
s_sub_u32 s54, s52, s53 // (op 2515) 
v_lshlrev_b32 v35, 1, v21 // (op 2515) 

s_lshr_b32 s55, s54, 1 // (op 2515) 

s_add_u32 s54, s55, s53 // (op 2515) 


s_lshr_b32 s53, s54, s11 // (op 2515) 

s_and_b32 s54, s11, 2147483648 // (op 2515) 
s_cmp_eq_i32 s54, 0 // (op 2515) 
s_mov_b32 s55, scc // (op 2515) 

s_cmp_eq_i32 s55, 0 // (op 2515) 
s_cselect_b32 s54, s52, s53 // (op 2515) 


s_mul_hi_u32 s53, s54, s12 // (op 2515) 
s_sub_u32 s55, s54, s53 // (op 2515) 
s_lshr_b32 s84, s55, 1 // (op 2515) 

s_add_u32 s55, s84, s53 // (op 2515) 


s_lshr_b32 s53, s55, s10 // (op 2515) 

s_and_b32 s55, s10, 2147483648 // (op 2515) 
s_cmp_eq_i32 s55, 0 // (op 2515) 
s_mov_b32 s84, scc // (op 2515) 

s_cmp_eq_i32 s84, 0 // (op 2515) 
s_cselect_b32 s55, s54, s53 // (op 2515) 


s_mul_i32 s53, s55, s14 // (op 2515) 

s_sub_u32 s55, s54, s53 // (op 2515) 

v_lshl_add_u32 v21, s55, 6, v23 // (op 2515) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2515) 
s_mul_i32 s53, s54, s84 // (op 2515) 


s_sub_u32 s54, s52, s53 // (op 2515) 


v_lshl_add_u32 v23, s54, 2, v35 // (op 2515) 


v_mul_lo_u32 v35, v23, s44 // (op 2515) 
v_add_u32 v35, v35, v21 // (op 2515) 


s_mul_i32 s53, s44, s43 // (op 2517) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2517) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2517) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2517) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2517) most significant: sum
s_add_u32 s53, s53, s55 // (op 2517) most significant: sum


s_mov_b32 s80, s46 // (op 2517) reg expression
s_mov_b32 s81, s47 // (op 2517) reg expression
s_mov_b32 s82, s52 // (op 2517) reg expression
s_mov_b32 s83, 131072 // (op 2517) call()


v_mov_b64 v[36:37], s[38:39] // (op 1801) convert

v_mov_b32 v21, v32 // (op 1801) reg expression
v_mov_b32 v23, v21 // (op 1801) 
buffer_load_ubyte v38, v23, s[76:79], 0 offen offset: 0 // (op 1801) Load value



v_and_b32 v21, -4, v1 // (op 2672) 
v_sub_u32 v23, v1, v21 // (op 2672) 

v_lshlrev_b32 v21, 4, v23 // (op 2672) 

v_lshrrev_b32 v23, 2, v1 // (op 2672) 
v_lshl_add_u32 v36, v23, 6, v21 // (op 2672) 


s_nop 0
v_readfirstlane_b32 s52, v36 // (op 2672) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3407) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 1990) 
v_mov_b64 v[36:37], s[54:55] // (op 1990) convert


v_mov_b64 v[40:41], v[24:25] // (op 1990) reg expression
s_mov_b32 m0, s52 // (op 1990) reg expression
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v36 // (op 1990) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 1990) most significant half

s_add_u32 m0, m0, 2048 // (op 1990) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v36 // (op 1990) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 1990) most significant half

s_add_u32 m0, m0, 2048 // (op 1990) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v36 // (op 1990) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 1990) most significant half

s_add_u32 m0, m0, 2048 // (op 1990) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds




v_mov_b64 v[36:37], s[44:45] // (op 1808) convert

v_mov_b32 v21, v34 // (op 1808) reg expression
buffer_load_ubyte v40, v21, s[80:83], 0 offen offset: 0 // (op 1808) Load value
v_add_u32 v21, v21, v36 // (op 1808) 
buffer_load_ubyte v41, v21, s[80:83], 0 offen offset: 0 // (op 1808) Load value


v_lshlrev_b32 v21, 4, v1 // (op 2607) 
s_nop 0
v_readfirstlane_b32 s52, v21 // (op 2607) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3388) 


v_mov_b64 v[36:37], v[26:27] // (op 1995) reg expression
s_add_u32 s53, 8192, s52 // (op 1995) 

s_mov_b32 m0, s53 // (op 1995) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 1995) Load value direct to lds




v_mov_b64 v[36:37], s[38:39] // (op 1815) convert

s_lshl_b32 s52, s38, 2 // (op 1815) 
v_add_u32 v21, v32, s52 // (op 1815) 

v_mov_b32 v23, v21 // (op 1815) 
buffer_load_ubyte v39, v23, s[76:79], 0 offen offset: 0 // (op 1815) Load value




v_and_b32 v21, -4, v1 // (op 3261) 
v_sub_u32 v23, v1, v21 // (op 3261) 

v_lshlrev_b32 v21, 4, v23 // (op 3261) 

v_lshrrev_b32 v23, 2, v1 // (op 3261) 
v_lshl_add_u32 v32, v23, 6, v21 // (op 3261) 


s_nop 0
v_readfirstlane_b32 s52, v32 // (op 3261) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3537) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 2000) 
v_mov_b64 v[36:37], s[54:55] // (op 2000) convert


s_lshl_b64 s[54:55], s[34:35], 7 // (op 2000) 
v_mov_b32 v21, s54 // (op 2000) 
v_mov_b32 v23, s55 // (op 2000) 
v_add_co_u32 v42, s[84:85], v24, v21 // (op 2000) least significant half
v_addc_co_u32 v43, s[84:85], v25, v23, s[84:85] // (op 2000) most significant half




s_add_u32 s53, 10240, s52 // (op 2000) 
s_mov_b32 m0, s53 // (op 2000) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v36 // (op 2000) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2000) most significant half

s_add_u32 m0, m0, 2048 // (op 2000) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v36 // (op 2000) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2000) most significant half

s_add_u32 m0, m0, 2048 // (op 2000) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v36 // (op 2000) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2000) most significant half

s_add_u32 m0, m0, 2048 // (op 2000) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds






v_mov_b64 v[24:25], s[44:45] // (op 1822) convert

s_lshl_b32 s52, s44, 2 // (op 1822) 
v_add_u32 v21, v34, s52 // (op 1822) 

buffer_load_ubyte v36, v21, s[80:83], 0 offen offset: 0 // (op 1822) Load value
v_add_u32 v21, v21, v24 // (op 1822) 
buffer_load_ubyte v37, v21, s[80:83], 0 offen offset: 0 // (op 1822) Load value



v_lshlrev_b32 v21, 4, v1 // (op 3222) 
s_nop 0
v_readfirstlane_b32 s52, v21 // (op 3222) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3528) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 2002) 
v_mov_b32 v21, s54 // (op 2002) 
v_mov_b32 v32, s55 // (op 2002) 
v_add_co_u32 v24, s[84:85], v26, v21 // (op 2002) least significant half
v_addc_co_u32 v25, s[84:85], v27, v32, s[84:85] // (op 2002) most significant half




s_add_u32 s53, 18432, s52 // (op 2002) 

s_mov_b32 m0, s53 // (op 2002) reg expression
buffer_load_dwordx4 v24, s[68:71], 0 offen offset: 0 lds // (op 2002) Load value direct to lds




v_and_b32 v21, -64, v1 // (op 2748) 
v_sub_u32 v32, v1, v21 // (op 2748) 

v_lshrrev_b32 v21, 6, v1 // (op 2748) 
v_lshl_add_u32 v34, v21, 7, v32 // (op 2748) 


s_barrier  // (op 3426) 



s_waitcnt vmcnt(10)
ds_write_b8 v34, v40 offset:20480 // (op 1828) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v34, v41 offset:20544 // (op 1828) Store local data 

s_barrier  // (op 3404) 



ds_read_b64_tr_b8 v[42:43], v19 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v19 offset:512 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v19 offset:4096 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v19 offset:4608 // (op 1846) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3423) 


ds_read_u8 v21, v17 offset:20480 // (op 1856) Load local data 



ds_read_b64_tr_b8 v[50:51], v19 offset:32 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v19 offset:544 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v19 offset:4128 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v19 offset:4640 // (op 1851) Transpose load local data 


ds_read_u8 v32, v17 offset:20512 // (op 1861) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 3385) 



ds_read_b64_tr_b8 v[58:59], v16 offset:8192 // (op 1842) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v16 offset:8320 // (op 1842) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v16 offset:9216 // (op 1842) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v16 offset:9344 // (op 1842) Transpose load local data 
v_lshlrev_b32 v34, 4, v1 // (op 2579) 
s_nop 0
v_readfirstlane_b32 s52, v34 // (op 2579) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v34, -4, v1 // (op 2644) 
v_sub_u32 v23, v1, v34 // (op 2644) 

v_lshlrev_b32 v34, 4, v23 // (op 2644) 

v_lshrrev_b32 v23, 2, v1 // (op 2644) 
v_lshl_add_u32 v24, v23, 6, v34 // (op 2644) 


s_nop 0
v_readfirstlane_b32 s53, v24 // (op 2644) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v34, -64, v1 // (op 2719) 
v_sub_u32 v23, v1, v34 // (op 2719) 

v_lshrrev_b32 v34, 6, v1 // (op 2719) 
v_lshl_add_u32 v24, v34, 7, v23 // (op 2719) 


v_lshlrev_b32 v23, 4, v1 // (op 3207) 
s_nop 0
v_readfirstlane_b32 s54, v23 // (op 3207) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -4, v1 // (op 3246) 
v_sub_u32 v34, v1, v23 // (op 3246) 

v_lshlrev_b32 v23, 4, v34 // (op 3246) 

v_lshrrev_b32 v34, 2, v1 // (op 3246) 
v_lshl_add_u32 v25, v34, 6, v23 // (op 3246) 


s_nop 0
v_readfirstlane_b32 s55, v25 // (op 3246) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 3290) 
v_sub_u32 v34, v1, v23 // (op 3290) 

v_lshrrev_b32 v23, 6, v1 // (op 3290) 
v_lshl_add_u32 v25, v23, 7, v34 // (op 3290) 


s_mov_b32 s1, 0 // (op 973) call()
s_and_b32 s84, s7, 4294967294 // (op 972) 
s_cmp_lt_u32 s1, s84 // (op 972) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_972 // (op 972) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_972 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_972:
s_add_u32 s1, s1, 2 // (op 974) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v38, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v38, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1008) 

v_mov_b64 v[26:27], s[38:39] // (op 75) convert

v_mov_b32 v23, v33 // (op 75) 
s_lshl_b32 s84, s38, 3 // (op 75) 
v_add_u32 v34, v23, s84 // (op 75) 

v_mov_b32 v66, v34 // (op 75) 
buffer_load_ubyte v38, v66, s[76:79], 0 offen offset: 0 // (op 75) Load value



s_barrier  // (op 3410) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 1915) 
v_mov_b64 v[26:27], s[84:85] // (op 1915) convert


v_mov_b64 v[66:67], v[30:31] // (op 1915) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 1915) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 1915) 

v_mov_b32 v34, s86 // (op 1915) 
v_mov_b32 v68, s87 // (op 1915) 
v_add_co_u32 v70, s[84:85], v66, v34 // (op 1915) least significant half
v_addc_co_u32 v71, s[84:85], v67, v68, s[84:85] // (op 1915) most significant half




s_mov_b32 m0, s53 // (op 1915) reg expression
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds
v_add_co_u32 v70, s[84:85], v70, v26 // (op 1915) least significant half
v_addc_co_u32 v71, s[84:85], 0, v71, s[84:85] // (op 1915) most significant half

s_add_u32 m0, m0, 2048 // (op 1915) 
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds
v_add_co_u32 v70, s[84:85], v70, v26 // (op 1915) least significant half
v_addc_co_u32 v71, s[84:85], 0, v71, s[84:85] // (op 1915) most significant half

s_add_u32 m0, m0, 2048 // (op 1915) 
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds
v_add_co_u32 v70, s[84:85], v70, v26 // (op 1915) least significant half
v_addc_co_u32 v71, s[84:85], 0, v71, s[84:85] // (op 1915) most significant half

s_add_u32 m0, m0, 2048 // (op 1915) 
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds



v_mov_b64 v[26:27], s[44:45] // (op 21) convert

v_mov_b32 v34, v35 // (op 21) 
s_lshl_b32 s84, s44, 3 // (op 21) 
v_add_u32 v68, v34, s84 // (op 21) 

buffer_load_ubyte v40, v68, s[80:83], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v68, v68, v26 // (op 21) 
buffer_load_ubyte v41, v68, s[80:83], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 3391) 


v_mov_b64 v[26:27], v[28:29] // (op 1910) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 1910) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 1910) 

v_mov_b32 v68, s86 // (op 1910) 
v_mov_b32 v69, s87 // (op 1910) 
v_add_co_u32 v70, s[84:85], v26, v68 // (op 1910) least significant half
v_addc_co_u32 v71, s[84:85], v27, v69, s[84:85] // (op 1910) most significant half




s_add_u32 s84, 8192, s52 // (op 1910) 

s_nop 0
s_mov_b32 m0, s84 // (op 1910) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 1910) Load value direct to lds


s_barrier  // (op 3546) 



s_waitcnt vmcnt(10)
ds_write_b8 v25, v36 offset:20736 // (op 901) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v25, v37 offset:20800 // (op 901) Store local data 
s_barrier  // (op 3534) 



ds_read_b64_tr_b8 v[42:43], v13 offset:10240 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:10752 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:14336 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:14848 // (op 894) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3543) 


ds_read_u8 v21, v12 offset:20736 // (op 898) Load local data 



ds_read_b64_tr_b8 v[50:51], v13 offset:10272 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:10784 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:14368 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v13 offset:14880 // (op 1013) Transpose load local data 


ds_read_u8 v32, v12 offset:20768 // (op 1016) Load local data 
s_barrier  // (op 3689) 
s_waitcnt vmcnt(8)
s_barrier  // (op 3525) 



ds_read_b64_tr_b8 v[58:59], v5 offset:18432 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v5 offset:18560 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v5 offset:19456 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v5 offset:19584 // (op 892) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v39, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 899) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1009) 

v_mov_b64 v[68:69], s[38:39] // (op 896) convert

s_lshl_b32 s84, s38, 2 // (op 896) 
s_mul_i32 s85, 3, s84 // (op 896) 

v_add_u32 v70, v23, s85 // (op 896) 

v_mov_b32 v71, v70 // (op 896) 
buffer_load_ubyte v39, v71, s[76:79], 0 offen offset: 0 // (op 896) Load value




s_barrier  // (op 3540) 
s_lshl_b32 s84, s38, 3 // (op 2379) 

v_add_u32 v33, v33, s84 // (op 2379) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 1971) 
v_mov_b64 v[68:69], s[84:85] // (op 1971) convert


s_lshl_b64 s[84:85], s[34:35], 7 // (op 1971) 
s_mul_i32 s87, 3, s85 // (op 1971) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 1971) least significant
s_mul_hi_u32 s88, 3, s84 // (op 1971) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 1971) most significant: sum


v_mov_b32 v23, s86 // (op 1971) 
v_mov_b32 v70, s87 // (op 1971) 
v_add_co_u32 v72, s[84:85], v66, v23 // (op 1971) least significant half
v_addc_co_u32 v73, s[84:85], v67, v70, s[84:85] // (op 1971) most significant half




s_add_u32 s84, 10240, s55 // (op 1971) 
s_nop 0
s_mov_b32 m0, s84 // (op 1971) reg expression
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v68 // (op 1971) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1971) most significant half

s_add_u32 m0, m0, 2048 // (op 1971) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v68 // (op 1971) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1971) most significant half

s_add_u32 m0, m0, 2048 // (op 1971) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v68 // (op 1971) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1971) most significant half

s_add_u32 m0, m0, 2048 // (op 1971) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds





v_mov_b64 v[66:67], s[44:45] // (op 903) convert

s_lshl_b32 s84, s44, 2 // (op 903) 
s_mul_i32 s85, 3, s84 // (op 903) 

v_add_u32 v23, v34, s85 // (op 903) 

buffer_load_ubyte v36, v23, s[80:83], 0 offen offset: 0 // (op 903) Load value
v_add_u32 v23, v23, v66 // (op 903) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 2251) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2251) 

v_mov_b32 v68, s86 // (op 2251) 
v_mov_b32 v69, s87 // (op 2251) 
v_add_co_u32 v30, s[84:85], v30, v68 // (op 2251) least significant half
v_addc_co_u32 v31, s[84:85], v31, v69, s[84:85] // (op 2251) most significant half




buffer_load_ubyte v37, v23, s[80:83], 0 offen offset: 0 // (op 903) Load value



s_barrier  // (op 3531) 
s_lshl_b32 s84, s44, 3 // (op 2507) 


v_add_u32 v35, v35, s84 // (op 2507) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 1966) 
s_mul_i32 s87, 3, s85 // (op 1966) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 1966) least significant
s_mul_hi_u32 s88, 3, s84 // (op 1966) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 1966) most significant: sum


v_mov_b32 v23, s86 // (op 1966) 
v_mov_b32 v34, s87 // (op 1966) 
v_add_co_u32 v66, s[84:85], v26, v23 // (op 1966) least significant half
v_addc_co_u32 v67, s[84:85], v27, v34, s[84:85] // (op 1966) most significant half




s_add_u32 s84, 18432, s54 // (op 1966) 

s_nop 0
s_mov_b32 m0, s84 // (op 1966) reg expression
buffer_load_dwordx4 v66, s[68:71], 0 offen offset: 0 lds // (op 1966) Load value direct to lds



s_barrier  // (op 3429) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 2123) 



s_lshl_b64 s[86:87], s[84:85], 1 // (op 2123) 

v_mov_b32 v23, s86 // (op 2123) 
v_mov_b32 v34, s87 // (op 2123) 
v_add_co_u32 v28, s[84:85], v28, v23 // (op 2123) least significant half
v_addc_co_u32 v29, s[84:85], v29, v34, s[84:85] // (op 2123) most significant half




s_waitcnt vmcnt(10)
ds_write_b8 v24, v40 offset:20480 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v24, v41 offset:20544 // (op 26) Store local data 
s_barrier  // (op 3416) 



ds_read_b64_tr_b8 v[42:43], v7 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4608 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3436) 


ds_read_u8 v21, v6 offset:20480 // (op 79) Load local data 



ds_read_b64_tr_b8 v[50:51], v7 offset:32 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:544 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:4128 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:4640 // (op 1004) Transpose load local data 


ds_read_u8 v32, v6 offset:20512 // (op 1007) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 3397) 



ds_read_b64_tr_b8 v[58:59], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:8320 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:9216 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:9344 // (op 65) Transpose load local data 
s_and_b32 s84, s7, 4294967294 // (op 972) 
s_cmp_lt_u32 s1, s84 // (op 972) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_972 // (op 972) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_972 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_972:












s_barrier  // (op 3673) 
s_and_b32 s52, s7, 4294967294 // (op 2039) 
s_add_u32 s53, s0, s52 // (op 2039) 

s_mul_i32 s52, s16, s8 // (op 2039) 
s_add_u32 s52, s52, s53 // (op 2039) 

s_mul_hi_u32 s53, s52, s13 // (op 2039) 
s_sub_u32 s54, s52, s53 // (op 2039) 
s_lshr_b32 s55, s54, 1 // (op 2039) 

s_add_u32 s54, s55, s53 // (op 2039) 


s_lshr_b32 s53, s54, s11 // (op 2039) 

s_and_b32 s54, s11, 2147483648 // (op 2039) 
s_cmp_eq_i32 s54, 0 // (op 2039) 
s_mov_b32 s55, scc // (op 2039) 

s_cmp_eq_i32 s55, 0 // (op 2039) 
s_cselect_b32 s54, s52, s53 // (op 2039) 


s_mul_hi_u32 s53, s54, s12 // (op 2039) 
s_sub_u32 s55, s54, s53 // (op 2039) 
s_lshr_b32 s84, s55, 1 // (op 2039) 

s_add_u32 s55, s84, s53 // (op 2039) 


s_lshr_b32 s53, s55, s10 // (op 2039) 

s_and_b32 s55, s10, 2147483648 // (op 2039) 
s_cmp_eq_i32 s55, 0 // (op 2039) 
s_mov_b32 s84, scc // (op 2039) 

s_cmp_eq_i32 s84, 0 // (op 2039) 
s_cselect_b32 s55, s54, s53 // (op 2039) 


s_lshl_b32 s53, s55, 4 // (op 2039) 

s_mov_b32 s85, 0 // (op 2039) convert
s_mov_b32 s84, s53 // (op 2039) convert

s_ashr_i64 s[86:87], s[2:3], 7 // (op 2039) 
s_mul_i32 s53, s54, s86 // (op 2039) 


s_sub_u32 s54, s52, s53 // (op 2039) 


v_lshl_add_u32 v38, s54, 7, v1 // (op 2039) 

v_mul_lo_u32 v35, v38, s31 // (op 2039) most significant: low of low * high
v_mul_hi_u32 v33, v38, s30 // (op 2039) most significant: high of low * low
v_mul_lo_u32 v34, v38, s30 // (op 2039) least significant: low of low * low
v_add_u32 v35, v35, v33 // (op 2039) most significant: sum


v_mov_b32 v33, s84 // (op 2039) 
v_mov_b32 v38, s85 // (op 2039) 
v_add_co_u32 v40, s[52:53], v33, v34 // (op 2039) least significant half
v_addc_co_u32 v41, s[52:53], v38, v35, s[52:53] // (op 2039) most significant half





s_mul_i32 s53, s30, s3 // (op 2041) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2041) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2041) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2041) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2041) most significant: sum
s_add_u32 s53, s53, s55 // (op 2041) most significant: sum


s_mov_b32 s68, s32 // (op 2041) reg expression
s_mov_b32 s69, s33 // (op 2041) reg expression
s_mov_b32 s70, s52 // (op 2041) reg expression
s_mov_b32 s71, 131072 // (op 2041) call()

s_and_b32 s52, s7, 4294967294 // (op 2175) 
v_and_b32 v38, -4, v1 // (op 2175) 
s_add_u32 s53, s0, s52 // (op 2175) 

v_sub_u32 v23, v1, v38 // (op 2175) 

v_lshlrev_b32 v38, 4, v23 // (op 2175) 

s_mul_i32 s52, s16, s8 // (op 2175) 
s_add_u32 s52, s52, s53 // (op 2175) 

v_lshrrev_b32 v23, 2, v1 // (op 2175) 
s_mul_hi_u32 s53, s52, s13 // (op 2175) 
s_sub_u32 s54, s52, s53 // (op 2175) 
s_lshr_b32 s55, s54, 1 // (op 2175) 

s_add_u32 s54, s55, s53 // (op 2175) 


s_lshr_b32 s53, s54, s11 // (op 2175) 

s_and_b32 s54, s11, 2147483648 // (op 2175) 
s_cmp_eq_i32 s54, 0 // (op 2175) 
s_mov_b32 s55, scc // (op 2175) 

s_cmp_eq_i32 s55, 0 // (op 2175) 
s_cselect_b32 s54, s52, s53 // (op 2175) 


s_mul_hi_u32 s53, s54, s12 // (op 2175) 
s_sub_u32 s55, s54, s53 // (op 2175) 
s_lshr_b32 s84, s55, 1 // (op 2175) 

s_add_u32 s55, s84, s53 // (op 2175) 


s_lshr_b32 s53, s55, s10 // (op 2175) 

s_and_b32 s55, s10, 2147483648 // (op 2175) 
s_cmp_eq_i32 s55, 0 // (op 2175) 
s_mov_b32 s84, scc // (op 2175) 

s_cmp_eq_i32 s84, 0 // (op 2175) 
s_cselect_b32 s55, s54, s53 // (op 2175) 


s_mul_i32 s53, s55, s14 // (op 2175) 

s_sub_u32 s55, s54, s53 // (op 2175) 

v_lshl_add_u32 v24, s55, 6, v38 // (op 2175) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2175) 
v_mov_b32 v35, 0 // (op 2175) convert
v_mov_b32 v34, v24 // (op 2175) convert

s_mul_i32 s53, s54, s84 // (op 2175) 


s_sub_u32 s54, s52, s53 // (op 2175) 


v_lshl_add_u32 v33, s54, 7, v23 // (op 2175) 


v_mul_lo_u32 v31, v33, s35 // (op 2175) most significant: low of low * high
v_mul_hi_u32 v38, v33, s34 // (op 2175) most significant: high of low * low
v_mul_lo_u32 v30, v33, s34 // (op 2175) least significant: low of low * low
v_add_u32 v31, v31, v38 // (op 2175) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v34 // (op 2175) least significant half
v_addc_co_u32 v31, s[52:53], v31, v35, s[52:53] // (op 2175) most significant half



s_mul_i32 s53, s34, s3 // (op 2177) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2177) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2177) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2177) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2177) most significant: sum
s_add_u32 s53, s53, s55 // (op 2177) most significant: sum


s_mov_b32 s72, s36 // (op 2177) reg expression
s_mov_b32 s73, s37 // (op 2177) reg expression
s_mov_b32 s74, s52 // (op 2177) reg expression
s_mov_b32 s75, 131072 // (op 2177) call()

s_and_b32 s52, s7, 4294967294 // (op 2303) 
v_lshrrev_b32 v38, 7, v1 // (op 2303) 
s_add_u32 s53, s0, s52 // (op 2303) 

v_and_b32 v23, 15, v1 // (op 2303) 
v_lshl_add_u32 v24, v38, 4, v23 // (op 2303) 


s_mul_i32 s52, s16, s8 // (op 2303) 
s_add_u32 s52, s52, s53 // (op 2303) 

v_and_b32 v23, 63, v1 // (op 2303) 
s_mul_hi_u32 s53, s52, s13 // (op 2303) 
v_lshrrev_b32 v38, 4, v23 // (op 2303) 

s_sub_u32 s54, s52, s53 // (op 2303) 
s_lshr_b32 s55, s54, 1 // (op 2303) 

s_add_u32 s54, s55, s53 // (op 2303) 


s_lshr_b32 s53, s54, s11 // (op 2303) 

s_and_b32 s54, s11, 2147483648 // (op 2303) 
s_cmp_eq_i32 s54, 0 // (op 2303) 
s_mov_b32 s55, scc // (op 2303) 

s_cmp_eq_i32 s55, 0 // (op 2303) 
s_cselect_b32 s54, s52, s53 // (op 2303) 


s_mul_hi_u32 s53, s54, s12 // (op 2303) 
s_sub_u32 s55, s54, s53 // (op 2303) 
s_lshr_b32 s84, s55, 1 // (op 2303) 

s_add_u32 s55, s84, s53 // (op 2303) 


s_lshr_b32 s53, s55, s10 // (op 2303) 

s_and_b32 s55, s10, 2147483648 // (op 2303) 
s_cmp_eq_i32 s55, 0 // (op 2303) 
s_mov_b32 s84, scc // (op 2303) 

s_cmp_eq_i32 s84, 0 // (op 2303) 
s_cselect_b32 s55, s54, s53 // (op 2303) 


v_lshl_add_u32 v23, s55, 4, v24 // (op 2303) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2303) 
s_mul_i32 s53, s54, s84 // (op 2303) 


s_sub_u32 s54, s52, s53 // (op 2303) 


v_lshl_add_u32 v24, s54, 2, v38 // (op 2303) 


v_mul_lo_u32 v38, v24, s38 // (op 2303) 

v_add_u32 v24, v23, v38 // (op 2303) 


s_mul_i32 s53, s38, s43 // (op 2305) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2305) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2305) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2305) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2305) most significant: sum
s_add_u32 s53, s53, s55 // (op 2305) most significant: sum


s_mov_b32 s76, s40 // (op 2305) reg expression
s_mov_b32 s77, s41 // (op 2305) reg expression
s_mov_b32 s78, s52 // (op 2305) reg expression
s_mov_b32 s79, 131072 // (op 2305) call()

s_and_b32 s52, s7, 4294967294 // (op 2431) 
v_and_b32 v23, -64, v1 // (op 2431) 
s_add_u32 s53, s0, s52 // (op 2431) 

v_sub_u32 v38, v1, v23 // (op 2431) 

v_lshrrev_b32 v23, 6, v1 // (op 2431) 
s_mul_i32 s52, s16, s8 // (op 2431) 
s_add_u32 s52, s52, s53 // (op 2431) 

v_lshlrev_b32 v25, 1, v23 // (op 2431) 

s_mul_hi_u32 s53, s52, s13 // (op 2431) 
s_sub_u32 s54, s52, s53 // (op 2431) 
s_lshr_b32 s55, s54, 1 // (op 2431) 

s_add_u32 s54, s55, s53 // (op 2431) 


s_lshr_b32 s53, s54, s11 // (op 2431) 

s_and_b32 s54, s11, 2147483648 // (op 2431) 
s_cmp_eq_i32 s54, 0 // (op 2431) 
s_mov_b32 s55, scc // (op 2431) 

s_cmp_eq_i32 s55, 0 // (op 2431) 
s_cselect_b32 s54, s52, s53 // (op 2431) 


s_mul_hi_u32 s53, s54, s12 // (op 2431) 
s_sub_u32 s55, s54, s53 // (op 2431) 
s_lshr_b32 s84, s55, 1 // (op 2431) 

s_add_u32 s55, s84, s53 // (op 2431) 


s_lshr_b32 s53, s55, s10 // (op 2431) 

s_and_b32 s55, s10, 2147483648 // (op 2431) 
s_cmp_eq_i32 s55, 0 // (op 2431) 
s_mov_b32 s84, scc // (op 2431) 

s_cmp_eq_i32 s84, 0 // (op 2431) 
s_cselect_b32 s55, s54, s53 // (op 2431) 


s_mul_i32 s53, s55, s14 // (op 2431) 

s_sub_u32 s55, s54, s53 // (op 2431) 

v_lshl_add_u32 v23, s55, 6, v38 // (op 2431) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2431) 
s_mul_i32 s53, s54, s84 // (op 2431) 


s_sub_u32 s54, s52, s53 // (op 2431) 


v_lshl_add_u32 v38, s54, 2, v25 // (op 2431) 


v_mul_lo_u32 v25, v38, s44 // (op 2431) 
v_add_u32 v25, v25, v23 // (op 2431) 


s_mul_i32 s53, s44, s43 // (op 2433) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2433) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2433) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2433) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2433) most significant: sum
s_add_u32 s53, s53, s55 // (op 2433) most significant: sum


s_mov_b32 s80, s46 // (op 2433) reg expression
s_mov_b32 s81, s47 // (op 2433) reg expression
s_mov_b32 s82, s52 // (op 2433) reg expression
s_mov_b32 s83, 131072 // (op 2433) call()

v_lshlrev_b32 v23, 4, v1 // (op 2592) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 2592) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -4, v1 // (op 2657) 
v_sub_u32 v38, v1, v23 // (op 2657) 

v_lshlrev_b32 v23, 4, v38 // (op 2657) 

v_lshrrev_b32 v38, 2, v1 // (op 2657) 
v_lshl_add_u32 v26, v38, 6, v23 // (op 2657) 


s_nop 0
v_readfirstlane_b32 s53, v26 // (op 2657) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 2732) 
v_sub_u32 v38, v1, v23 // (op 2732) 

v_lshrrev_b32 v23, 6, v1 // (op 2732) 
v_lshl_add_u32 v26, v23, 7, v38 // (op 2732) 


s_and_b32 s54, s7, 4294967294 // (op 978) 
s_cmp_lt_u32 s54, s7 // (op 977) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_977 // (op 977) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_977 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_977:
s_barrier  // (op 3394) 
s_add_u32 s54, s54, 1 // (op 979) 


v_mov_b64 v[28:29], v[40:41] // (op 1953) 
v_mov_b64 v[32:33], v[28:29] // (op 1953) reg expression
s_add_u32 s55, 8192, s52 // (op 1953) 

s_mov_b32 m0, s55 // (op 1953) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 1953) Load value direct to lds



s_lshl_b64 s[84:85], s[30:31], 7 // (op 2029) 
v_mov_b32 v21, s84 // (op 2029) 
v_mov_b32 v23, s85 // (op 2029) 
v_add_co_u32 v40, s[86:87], v40, v21 // (op 2029) least significant half
v_addc_co_u32 v41, s[86:87], v41, v23, s[86:87] // (op 2029) most significant half




s_barrier  // (op 3413) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 1959) 
v_mov_b64 v[28:29], s[84:85] // (op 1959) convert


v_mov_b64 v[32:33], v[30:31] // (op 1959) 
v_mov_b64 v[34:35], v[32:33] // (op 1959) reg expression
s_mov_b32 m0, s53 // (op 1959) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 1959) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 1959) most significant half

s_add_u32 m0, m0, 2048 // (op 1959) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 1959) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 1959) most significant half

s_add_u32 m0, m0, 2048 // (op 1959) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 1959) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 1959) most significant half

s_add_u32 m0, m0, 2048 // (op 1959) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds



s_lshl_b64 s[84:85], s[34:35], 7 // (op 2167) 
v_mov_b32 v21, s84 // (op 2167) 
v_mov_b32 v23, s85 // (op 2167) 
v_add_co_u32 v30, s[86:87], v30, v21 // (op 2167) least significant half
v_addc_co_u32 v31, s[86:87], v31, v23, s[86:87] // (op 2167) most significant half





v_mov_b64 v[28:29], s[44:45] // (op 845) convert

v_mov_b32 v21, v25 // (op 845) 
v_mov_b32 v23, v21 // (op 845) reg expression
buffer_load_ubyte v32, v23, s[80:83], 0 offen offset: 0 // (op 845) Load value
v_add_u32 v23, v23, v28 // (op 845) 
buffer_load_ubyte v33, v23, s[80:83], 0 offen offset: 0 // (op 845) Load value



s_lshl_b32 s55, s44, 2 // (op 2423) 
v_add_u32 v25, v25, s55 // (op 2423) 

s_barrier  // (op 3432) 
s_waitcnt vmcnt(6)
s_barrier  // (op 3400) 






ds_read_b64_tr_b8 v[42:43], v0 offset:8192 // (op 834) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:8320 // (op 834) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:9216 // (op 834) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:9344 // (op 834) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v26, v32 offset:20480 // (op 843) Store local data 
s_barrier  // (op 3419) 
s_waitcnt vmcnt(0)
ds_write_b8 v26, v33 offset:20544 // (op 843) Store local data 




ds_read_b64_tr_b8 v[32:33], v7 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:512 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:4096 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:4608 // (op 836) Transpose load local data 

v_mov_b64 v[28:29], s[38:39] // (op 838) convert

v_mov_b32 v21, v24 // (op 838) 
v_mov_b32 v23, v21 // (op 838) reg expression
v_mov_b32 v27, v23 // (op 838) 
buffer_load_ubyte v50, v27, s[76:79], 0 offen offset: 0 // (op 838) Load value




s_lshl_b32 s55, s38, 2 // (op 2295) 
v_add_u32 v24, v24, s55 // (op 2295) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3439) 


ds_read_u8 v21, v6 offset:20480 // (op 840) Load local data 



ds_read_b64_tr_b8 v[52:53], v7 offset:32 // (op 985) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[32:39], a[0:3], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 841) 
ds_read_b64_tr_b8 v[54:55], v7 offset:544 // (op 985) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v7 offset:4128 // (op 985) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:4640 // (op 985) Transpose load local data 


ds_read_u8 v21, v6 offset:20512 // (op 988) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[52:59], a[4:7], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 989) 




s_cmp_lt_u32 s54, s7 // (op 977) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_977 // (op 977) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_977 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_977:








s_barrier  // (op 3677) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 272) 
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_298



buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:256 sc1 // (op 295) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:512 sc1 // (op 295) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:768 sc1 // (op 295) Store value



buffer_store_dword a4, v20, s[64:67], 0 offen offset:128 sc1 // (op 1168) Store value
buffer_store_dword a5, v20, s[64:67], 0 offen offset:384 sc1 // (op 1168) Store value
buffer_store_dword a6, v20, s[64:67], 0 offen offset:640 sc1 // (op 1168) Store value
buffer_store_dword a7, v20, s[64:67], 0 offen offset:896 sc1 // (op 1168) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
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
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_298:

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
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s52, 0 // (op 280) call()
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s53, s16, 1 // (op 3080) 
v_lshrrev_b32 v16, 5, v1 // (op 3080) 
s_add_u32 s54, s53, s52 // (op 3080) 
s_lshl_b32 s54, s54, 4 // (op 3080) 

v_lshlrev_b32 v17, 2, v16 // (op 3080) 

v_add_lshl_u32 v16, s54, v17, 6 // (op 3080) 


v_and_b32 v17, -32, v1 // (op 3080) 
v_sub_u32 v19, v1, v17 // (op 3080) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 3080) 


s_lshl_b32 s53, s15, 12 // (op 3082) 
s_mov_b32 s64, s28 // (op 3082) reg expression
s_mov_b32 s65, s29 // (op 3082) reg expression
s_mov_b32 s66, s53 // (op 3082) reg expression
s_mov_b32 s67, 131072 // (op 3082) call()

RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
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
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
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
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




buffer_load_dword v24, v17, s[64:67], 0 offen // (op 333) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:256 // (op 333) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:512 // (op 333) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:768 // (op 333) Load value
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




buffer_load_dword v24, v17, s[64:67], 0 offen offset:128 // (op 1178) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:384 // (op 1178) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:640 // (op 1178) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:896 // (op 1178) Load value
v_accvgpr_read v28, a4 // (op 1179) reg expression
v_accvgpr_read v29, a5 // (op 1179) reg expression
v_accvgpr_read v30, a6 // (op 1179) reg expression
v_accvgpr_read v31, a7 // (op 1179) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1180) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1180) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1180) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1180) 

v_accvgpr_write a4, v28 // (op 1181) reg expression
v_accvgpr_write a5, v29 // (op 1181) reg expression
v_accvgpr_write a6, v30 // (op 1181) reg expression
v_accvgpr_write a7, v31 // (op 1181) reg expression

s_add_u32 s52, s52, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 



v_mov_b32 v16, v22 // (op 52) reg expression

buffer_store_dwordx4 v[28:31], v16, s[60:63], 0 offen // (op 52) Store value


v_accvgpr_read v24, a4 // (op 1197) 
v_accvgpr_read v25, a5 // (op 1197) 
v_accvgpr_read v26, a6 // (op 1197) 
v_accvgpr_read v27, a7 // (op 1197) 
v_mul_f32 v28, v15, v24 // (op 1197) 
v_mul_f32 v29, v15, v25 // (op 1197) 
v_mul_f32 v30, v15, v26 // (op 1197) 
v_mul_f32 v31, v15, v27 // (op 1197) 



s_lshl_b32 s52, s24, 7 // (op 1199) 
v_add_u32 v16, v22, s52 // (op 1199) 


buffer_store_dwordx4 v[28:31], v16, s[60:63], 0 offen // (op 1199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v16, v18 // (op 135) reg expression

buffer_load_dwordx4 v[24:27], v16, s[56:59], 0 offen // (op 135) Load value

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

buffer_store_dwordx4 v[24:27], v23, s[60:63], 0 offen // (op 134) Store value




s_lshl_b32 s52, s4, 7 // (op 1221) 
v_add_u32 v16, v18, s52 // (op 1221) 


buffer_load_dwordx4 v[24:27], v16, s[56:59], 0 offen // (op 1221) Load value

v_accvgpr_read v28, a4 // (op 1218) 
v_accvgpr_read v29, a5 // (op 1218) 
v_accvgpr_read v30, a6 // (op 1218) 
v_accvgpr_read v31, a7 // (op 1218) 
v_mul_f32 v32, v15, v28 // (op 1218) 
v_mul_f32 v33, v15, v29 // (op 1218) 
v_mul_f32 v34, v15, v30 // (op 1218) 
v_mul_f32 v35, v15, v31 // (op 1218) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 1222) 
v_mul_f32 v29, v14, v25 // (op 1222) 
v_mul_f32 v30, v14, v26 // (op 1222) 
v_mul_f32 v31, v14, v27 // (op 1222) 

v_add_f32 v24, v28, v32 // (op 1219) 
v_add_f32 v25, v29, v33 // (op 1219) 
v_add_f32 v26, v30, v34 // (op 1219) 
v_add_f32 v27, v31, v35 // (op 1219) 




s_lshl_b32 s52, s24, 7 // (op 1220) 
v_add_u32 v23, v22, s52 // (op 1220) 


buffer_store_dwordx4 v[24:27], v23, s[60:63], 0 offen // (op 1220) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:


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
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















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
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 7, v1 // (op 2562) 
s_mul_i32 s15, s16, s50 // (op 2562) 
s_add_u32 s15, s15, s0 // (op 2562) 
v_and_b32 v5, 63, v1 // (op 2562) 
s_add_u32 s6, s15, s9 // (op 2562) 

v_lshrrev_b32 v6, 4, v5 // (op 2562) 
s_mul_hi_u32 s15, s6, s13 // (op 2562) 
v_lshlrev_b32 v7, 2, v6 // (op 2562) 

s_sub_u32 s7, s6, s15 // (op 2562) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 2562) 


s_lshr_b32 s8, s7, 1 // (op 2562) 

v_lshrrev_b32 v0, 2, v5 // (op 2562) 

s_add_u32 s7, s8, s15 // (op 2562) 


v_and_b32 v5, 3, v0 // (op 2562) 

s_lshr_b32 s8, s7, s11 // (op 2562) 

v_and_b32 v0, 3, v1 // (op 2562) 
s_and_b32 s7, s11, 2147483648 // (op 2562) 
v_lshl_add_u32 v7, v5, 2, v0 // (op 2562) 


s_cmp_eq_i32 s7, 0 // (op 2562) 
s_mov_b32 s15, scc // (op 2562) 

v_lshrrev_b32 v0, 6, v1 // (op 2562) 
s_cmp_eq_i32 s15, 0 // (op 2562) 
s_cselect_b32 s7, s6, s8 // (op 2562) 



v_and_b32 v5, 1, v0 // (op 2562) 

s_mul_hi_u32 s6, s7, s12 // (op 2562) 
v_lshl_add_u32 v0, v5, 4, v7 // (op 2562) 


s_sub_u32 s8, s7, s6 // (op 2562) 
s_lshr_b32 s15, s8, 1 // (op 2562) 

s_add_u32 s8, s15, s6 // (op 2562) 


s_lshr_b32 s6, s8, s10 // (op 2562) 

s_and_b32 s8, s10, 2147483648 // (op 2562) 
s_cmp_eq_i32 s8, 0 // (op 2562) 
s_mov_b32 s15, scc // (op 2562) 

s_cmp_eq_i32 s15, 0 // (op 2562) 
s_cselect_b32 s8, s7, s6 // (op 2562) 


s_mul_i32 s6, s8, s14 // (op 2562) 
v_lshl_add_u32 v5, s8, 4, v6 // (op 2562) 


s_sub_u32 s8, s7, s6 // (op 2562) 


v_lshl_add_u32 v6, s8, 6, v0 // (op 2562) 


v_mul_lo_u32 v0, v6, s4 // (op 2562) 

v_add_lshl_u32 v6, v5, v0, 2 // (op 2562) 


s_mul_i32 s19, s4, s23 // (op 2564) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2564) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2564) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2564) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2564) most significant: sum
s_add_u32 s19, s19, s6 // (op 2564) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2564) 

s_mov_b32 s56, s20 // (op 2564) reg expression
s_mov_b32 s57, s21 // (op 2564) reg expression
s_mov_b32 s58, s28 // (op 2564) reg expression
s_mov_b32 s59, 131072 // (op 2564) call()

v_lshrrev_b32 v0, 7, v1 // (op 2820) 
s_mul_i32 s15, s16, s50 // (op 2820) 
s_add_u32 s15, s15, s0 // (op 2820) 
v_and_b32 v5, 63, v1 // (op 2820) 
s_add_u32 s6, s15, s9 // (op 2820) 

v_lshrrev_b32 v7, 4, v5 // (op 2820) 
s_mul_hi_u32 s15, s6, s13 // (op 2820) 
v_lshlrev_b32 v12, 2, v7 // (op 2820) 

s_sub_u32 s7, s6, s15 // (op 2820) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2820) 


s_lshr_b32 s8, s7, 1 // (op 2820) 

v_lshrrev_b32 v0, 2, v5 // (op 2820) 

s_add_u32 s7, s8, s15 // (op 2820) 


v_and_b32 v5, 3, v0 // (op 2820) 

s_lshr_b32 s8, s7, s11 // (op 2820) 

v_and_b32 v0, 3, v1 // (op 2820) 
s_and_b32 s7, s11, 2147483648 // (op 2820) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 2820) 


s_cmp_eq_i32 s7, 0 // (op 2820) 
s_mov_b32 s15, scc // (op 2820) 

v_lshrrev_b32 v0, 6, v1 // (op 2820) 
s_cmp_eq_i32 s15, 0 // (op 2820) 
s_cselect_b32 s7, s6, s8 // (op 2820) 



v_and_b32 v5, 1, v0 // (op 2820) 

s_mul_hi_u32 s6, s7, s12 // (op 2820) 
v_lshl_add_u32 v0, v5, 4, v12 // (op 2820) 


s_sub_u32 s8, s7, s6 // (op 2820) 
s_lshr_b32 s15, s8, 1 // (op 2820) 

s_add_u32 s8, s15, s6 // (op 2820) 


s_lshr_b32 s6, s8, s10 // (op 2820) 

s_and_b32 s8, s10, 2147483648 // (op 2820) 
s_cmp_eq_i32 s8, 0 // (op 2820) 
s_mov_b32 s15, scc // (op 2820) 

s_cmp_eq_i32 s15, 0 // (op 2820) 
s_cselect_b32 s8, s7, s6 // (op 2820) 


s_mul_i32 s6, s8, s14 // (op 2820) 
v_lshl_add_u32 v5, s8, 4, v7 // (op 2820) 


s_sub_u32 s8, s7, s6 // (op 2820) 


v_lshl_add_u32 v7, s8, 6, v0 // (op 2820) 


v_mul_lo_u32 v0, v7, s24 // (op 2820) 

v_add_lshl_u32 v7, v5, v0, 2 // (op 2820) 


s_mul_i32 s19, s24, s23 // (op 2822) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2822) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2822) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2822) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2822) most significant: sum
s_add_u32 s19, s19, s6 // (op 2822) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2822) 

s_mov_b32 s60, s26 // (op 2822) reg expression
s_mov_b32 s61, s27 // (op 2822) reg expression
s_mov_b32 s62, s28 // (op 2822) reg expression
s_mov_b32 s63, 131072 // (op 2822) call()

v_and_b32 v0, 48, v1 // (op 2889) 
v_and_b32 v5, 63, v1 // (op 2889) 
v_sub_u32 v12, v5, v0 // (op 2889) 

v_lshrrev_b32 v5, 1, v12 // (op 2889) 
v_add_lshl_u32 v13, v0, v5, 4 // (op 2889) 


v_and_b32 v0, 1, v12 // (op 2889) 

v_lshlrev_b32 v5, 3, v0 // (op 2889) 

v_lshrrev_b32 v0, 7, v1 // (op 2889) 
v_lshl_add_u32 v12, v0, 4, v5 // (op 2889) 


v_add_u32 v0, v13, v12 // (op 2889) 


v_and_b32 v5, 48, v1 // (op 2964) 
v_and_b32 v12, 63, v1 // (op 2964) 
v_sub_u32 v13, v12, v5 // (op 2964) 

v_lshrrev_b32 v12, 1, v13 // (op 2964) 
v_add_lshl_u32 v16, v5, v12, 6 // (op 2964) 


v_and_b32 v5, 1, v13 // (op 2964) 

v_lshlrev_b32 v12, 3, v5 // (op 2964) 

v_lshrrev_b32 v5, 6, v1 // (op 2964) 
v_and_b32 v13, 1, v5 // (op 2964) 

v_lshl_add_u32 v5, v13, 4, v12 // (op 2964) 


v_add_u32 v12, v16, v5 // (op 2964) 


v_lshrrev_b32 v5, 6, v1 // (op 3040) 
v_and_b32 v13, 1, v5 // (op 3040) 

v_and_b32 v5, 15, v1 // (op 3040) 
v_lshl_add_u32 v16, v13, 4, v5 // (op 3040) 


v_and_b32 v5, 63, v1 // (op 3040) 
v_lshrrev_b32 v13, 4, v5 // (op 3040) 

v_lshl_add_u32 v5, v13, 6, v16 // (op 3040) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_accvgpr_write a4, 0.00000 // (op 578) call()
v_accvgpr_write a5, 0.00000 // (op 578) call()
v_accvgpr_write a6, 0.00000 // (op 578) call()
v_accvgpr_write a7, 0.00000 // (op 578) call()
s_mul_i32 s15, s16, s50 // (op 2066) 
s_add_u32 s15, s15, s0 // (op 2066) 
s_add_u32 s6, s15, s9 // (op 2066) 

s_mul_hi_u32 s15, s6, s13 // (op 2066) 
s_sub_u32 s7, s6, s15 // (op 2066) 
s_lshr_b32 s8, s7, 1 // (op 2066) 

s_add_u32 s7, s8, s15 // (op 2066) 


s_lshr_b32 s8, s7, s11 // (op 2066) 

s_and_b32 s7, s11, 2147483648 // (op 2066) 
s_cmp_eq_i32 s7, 0 // (op 2066) 
s_mov_b32 s15, scc // (op 2066) 

s_cmp_eq_i32 s15, 0 // (op 2066) 
s_cselect_b32 s7, s6, s8 // (op 2066) 


s_mul_hi_u32 s8, s7, s12 // (op 2066) 
s_sub_u32 s15, s7, s8 // (op 2066) 
s_lshr_b32 s18, s15, 1 // (op 2066) 

s_add_u32 s15, s18, s8 // (op 2066) 


s_lshr_b32 s8, s15, s10 // (op 2066) 

s_and_b32 s15, s10, 2147483648 // (op 2066) 
s_cmp_eq_i32 s15, 0 // (op 2066) 
s_mov_b32 s18, scc // (op 2066) 

s_cmp_eq_i32 s18, 0 // (op 2066) 
s_cselect_b32 s15, s7, s8 // (op 2066) 


s_lshl_b32 s8, s15, 4 // (op 2066) 

s_mov_b32 s19, 0 // (op 2066) convert
s_mov_b32 s18, s8 // (op 2066) convert

s_ashr_i64 s[28:29], s[2:3], 7 // (op 2066) 
s_mul_i32 s8, s7, s28 // (op 2066) 


s_sub_u32 s7, s6, s8 // (op 2066) 


v_lshl_add_u32 v13, s7, 7, v1 // (op 2066) 

v_mul_lo_u32 v17, v13, s31 // (op 2066) most significant: low of low * high
v_mul_hi_u32 v18, v13, s30 // (op 2066) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 2066) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 2066) most significant: sum


v_mov_b32 v13, s18 // (op 2066) 
v_mov_b32 v18, s19 // (op 2066) 
v_add_co_u32 v20, s[28:29], v13, v16 // (op 2066) least significant half
v_addc_co_u32 v21, s[28:29], v18, v17, s[28:29] // (op 2066) most significant half





s_mul_i32 s19, s30, s3 // (op 2068) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2068) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2068) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2068) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2068) most significant: sum
s_add_u32 s19, s19, s6 // (op 2068) most significant: sum


s_mov_b32 s68, s32 // (op 2068) reg expression
s_mov_b32 s69, s33 // (op 2068) reg expression
s_mov_b32 s70, s18 // (op 2068) reg expression
s_mov_b32 s71, 131072 // (op 2068) call()

s_mul_i32 s15, s16, s50 // (op 2085) 
s_add_u32 s15, s15, s0 // (op 2085) 
s_add_u32 s6, s15, s9 // (op 2085) 

s_mul_hi_u32 s15, s6, s13 // (op 2085) 
s_sub_u32 s7, s6, s15 // (op 2085) 
s_lshr_b32 s8, s7, 1 // (op 2085) 

s_add_u32 s7, s8, s15 // (op 2085) 


s_lshr_b32 s8, s7, s11 // (op 2085) 

s_and_b32 s7, s11, 2147483648 // (op 2085) 
s_cmp_eq_i32 s7, 0 // (op 2085) 
s_mov_b32 s15, scc // (op 2085) 

s_cmp_eq_i32 s15, 0 // (op 2085) 
s_cselect_b32 s7, s6, s8 // (op 2085) 


s_mul_hi_u32 s8, s7, s12 // (op 2085) 
s_sub_u32 s15, s7, s8 // (op 2085) 
s_lshr_b32 s18, s15, 1 // (op 2085) 

s_add_u32 s15, s18, s8 // (op 2085) 


s_lshr_b32 s8, s15, s10 // (op 2085) 

s_and_b32 s15, s10, 2147483648 // (op 2085) 
s_cmp_eq_i32 s15, 0 // (op 2085) 
s_mov_b32 s18, scc // (op 2085) 

s_cmp_eq_i32 s18, 0 // (op 2085) 
s_cselect_b32 s15, s7, s8 // (op 2085) 


s_lshl_b32 s8, s15, 4 // (op 2085) 

s_mov_b32 s19, 0 // (op 2085) convert
s_mov_b32 s18, s8 // (op 2085) convert

s_ashr_i64 s[28:29], s[2:3], 7 // (op 2085) 
s_mul_i32 s8, s7, s28 // (op 2085) 


s_sub_u32 s7, s6, s8 // (op 2085) 


v_lshl_add_u32 v13, s7, 7, v1 // (op 2085) 

v_mul_lo_u32 v17, v13, s31 // (op 2085) most significant: low of low * high
v_mul_hi_u32 v18, v13, s30 // (op 2085) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 2085) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 2085) most significant: sum


v_mov_b32 v13, s18 // (op 2085) 
v_mov_b32 v18, s19 // (op 2085) 
v_add_co_u32 v22, s[28:29], v13, v16 // (op 2085) least significant half
v_addc_co_u32 v23, s[28:29], v18, v17, s[28:29] // (op 2085) most significant half





s_mul_i32 s19, s30, s3 // (op 2087) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2087) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2087) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2087) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2087) most significant: sum
s_add_u32 s19, s19, s6 // (op 2087) most significant: sum


s_mov_b32 s68, s32 // (op 2087) reg expression
s_mov_b32 s69, s33 // (op 2087) reg expression
s_mov_b32 s70, s18 // (op 2087) reg expression
s_mov_b32 s71, 131072 // (op 2087) call()

v_and_b32 v13, -4, v1 // (op 2198) 
s_mul_i32 s15, s16, s50 // (op 2198) 
s_add_u32 s15, s15, s0 // (op 2198) 
v_sub_u32 v16, v1, v13 // (op 2198) 

s_add_u32 s6, s15, s9 // (op 2198) 

v_lshlrev_b32 v13, 4, v16 // (op 2198) 

s_mul_hi_u32 s15, s6, s13 // (op 2198) 
v_lshrrev_b32 v16, 2, v1 // (op 2198) 
s_sub_u32 s7, s6, s15 // (op 2198) 
s_lshr_b32 s8, s7, 1 // (op 2198) 

s_add_u32 s7, s8, s15 // (op 2198) 


s_lshr_b32 s8, s7, s11 // (op 2198) 

s_and_b32 s7, s11, 2147483648 // (op 2198) 
s_cmp_eq_i32 s7, 0 // (op 2198) 
s_mov_b32 s15, scc // (op 2198) 

s_cmp_eq_i32 s15, 0 // (op 2198) 
s_cselect_b32 s7, s6, s8 // (op 2198) 


s_mul_hi_u32 s8, s7, s12 // (op 2198) 
s_sub_u32 s15, s7, s8 // (op 2198) 
s_lshr_b32 s18, s15, 1 // (op 2198) 

s_add_u32 s15, s18, s8 // (op 2198) 


s_lshr_b32 s8, s15, s10 // (op 2198) 

s_and_b32 s15, s10, 2147483648 // (op 2198) 
s_cmp_eq_i32 s15, 0 // (op 2198) 
s_mov_b32 s18, scc // (op 2198) 

s_cmp_eq_i32 s18, 0 // (op 2198) 
s_cselect_b32 s15, s7, s8 // (op 2198) 


s_mul_i32 s8, s15, s14 // (op 2198) 

s_sub_u32 s15, s7, s8 // (op 2198) 

v_lshl_add_u32 v17, s15, 6, v13 // (op 2198) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2198) 
v_mov_b32 v19, 0 // (op 2198) convert
v_mov_b32 v18, v17 // (op 2198) convert

s_mul_i32 s8, s7, s18 // (op 2198) 


s_sub_u32 s7, s6, s8 // (op 2198) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2198) 


v_mul_lo_u32 v17, v13, s35 // (op 2198) most significant: low of low * high
v_mul_hi_u32 v24, v13, s34 // (op 2198) most significant: high of low * low
v_mul_lo_u32 v16, v13, s34 // (op 2198) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 2198) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v18 // (op 2198) least significant half
v_addc_co_u32 v17, s[18:19], v17, v19, s[18:19] // (op 2198) most significant half



s_mul_i32 s19, s34, s3 // (op 2200) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2200) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2200) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2200) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2200) most significant: sum
s_add_u32 s19, s19, s6 // (op 2200) most significant: sum


s_mov_b32 s72, s36 // (op 2200) reg expression
s_mov_b32 s73, s37 // (op 2200) reg expression
s_mov_b32 s74, s18 // (op 2200) reg expression
s_mov_b32 s75, 131072 // (op 2200) call()

v_and_b32 v13, -4, v1 // (op 2217) 
s_mul_i32 s15, s16, s50 // (op 2217) 
s_add_u32 s15, s15, s0 // (op 2217) 
v_sub_u32 v18, v1, v13 // (op 2217) 

s_add_u32 s6, s15, s9 // (op 2217) 

v_lshlrev_b32 v13, 4, v18 // (op 2217) 

s_mul_hi_u32 s15, s6, s13 // (op 2217) 
v_lshrrev_b32 v18, 2, v1 // (op 2217) 
s_sub_u32 s7, s6, s15 // (op 2217) 
s_lshr_b32 s8, s7, 1 // (op 2217) 

s_add_u32 s7, s8, s15 // (op 2217) 


s_lshr_b32 s8, s7, s11 // (op 2217) 

s_and_b32 s7, s11, 2147483648 // (op 2217) 
s_cmp_eq_i32 s7, 0 // (op 2217) 
s_mov_b32 s15, scc // (op 2217) 

s_cmp_eq_i32 s15, 0 // (op 2217) 
s_cselect_b32 s7, s6, s8 // (op 2217) 


s_mul_hi_u32 s8, s7, s12 // (op 2217) 
s_sub_u32 s15, s7, s8 // (op 2217) 
s_lshr_b32 s18, s15, 1 // (op 2217) 

s_add_u32 s15, s18, s8 // (op 2217) 


s_lshr_b32 s8, s15, s10 // (op 2217) 

s_and_b32 s15, s10, 2147483648 // (op 2217) 
s_cmp_eq_i32 s15, 0 // (op 2217) 
s_mov_b32 s18, scc // (op 2217) 

s_cmp_eq_i32 s18, 0 // (op 2217) 
s_cselect_b32 s15, s7, s8 // (op 2217) 


s_mul_i32 s8, s15, s14 // (op 2217) 

s_sub_u32 s15, s7, s8 // (op 2217) 

v_lshl_add_u32 v19, s15, 6, v13 // (op 2217) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2217) 
v_mov_b32 v25, 0 // (op 2217) convert
v_mov_b32 v24, v19 // (op 2217) convert

s_mul_i32 s8, s7, s18 // (op 2217) 


s_sub_u32 s7, s6, s8 // (op 2217) 


v_lshl_add_u32 v13, s7, 7, v18 // (op 2217) 


v_mul_lo_u32 v19, v13, s35 // (op 2217) most significant: low of low * high
v_mul_hi_u32 v26, v13, s34 // (op 2217) most significant: high of low * low
v_mul_lo_u32 v18, v13, s34 // (op 2217) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 2217) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v24 // (op 2217) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 2217) most significant half



s_mul_i32 s19, s34, s3 // (op 2219) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2219) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2219) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2219) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2219) most significant: sum
s_add_u32 s19, s19, s6 // (op 2219) most significant: sum


s_mov_b32 s72, s36 // (op 2219) reg expression
s_mov_b32 s73, s37 // (op 2219) reg expression
s_mov_b32 s74, s18 // (op 2219) reg expression
s_mov_b32 s75, 131072 // (op 2219) call()

v_lshrrev_b32 v13, 7, v1 // (op 2326) 
s_mul_i32 s15, s16, s50 // (op 2326) 
s_add_u32 s15, s15, s0 // (op 2326) 
v_and_b32 v24, 15, v1 // (op 2326) 
s_add_u32 s6, s15, s9 // (op 2326) 

v_lshl_add_u32 v25, v13, 4, v24 // (op 2326) 


s_mul_hi_u32 s15, s6, s13 // (op 2326) 
v_and_b32 v13, 63, v1 // (op 2326) 
s_sub_u32 s7, s6, s15 // (op 2326) 
v_lshrrev_b32 v24, 4, v13 // (op 2326) 

s_lshr_b32 s8, s7, 1 // (op 2326) 

s_add_u32 s7, s8, s15 // (op 2326) 


s_lshr_b32 s8, s7, s11 // (op 2326) 

s_and_b32 s7, s11, 2147483648 // (op 2326) 
s_cmp_eq_i32 s7, 0 // (op 2326) 
s_mov_b32 s15, scc // (op 2326) 

s_cmp_eq_i32 s15, 0 // (op 2326) 
s_cselect_b32 s7, s6, s8 // (op 2326) 


s_mul_hi_u32 s8, s7, s12 // (op 2326) 
s_sub_u32 s15, s7, s8 // (op 2326) 
s_lshr_b32 s18, s15, 1 // (op 2326) 

s_add_u32 s15, s18, s8 // (op 2326) 


s_lshr_b32 s8, s15, s10 // (op 2326) 

s_and_b32 s15, s10, 2147483648 // (op 2326) 
s_cmp_eq_i32 s15, 0 // (op 2326) 
s_mov_b32 s18, scc // (op 2326) 

s_cmp_eq_i32 s18, 0 // (op 2326) 
s_cselect_b32 s15, s7, s8 // (op 2326) 


v_lshl_add_u32 v13, s15, 4, v25 // (op 2326) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2326) 
s_mul_i32 s8, s7, s18 // (op 2326) 


s_sub_u32 s7, s6, s8 // (op 2326) 


v_lshl_add_u32 v25, s7, 2, v24 // (op 2326) 


v_mul_lo_u32 v24, v25, s38 // (op 2326) 

v_add_u32 v25, v13, v24 // (op 2326) 


s_mul_i32 s19, s38, s43 // (op 2328) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2328) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2328) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2328) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2328) most significant: sum
s_add_u32 s19, s19, s6 // (op 2328) most significant: sum


s_mov_b32 s76, s40 // (op 2328) reg expression
s_mov_b32 s77, s41 // (op 2328) reg expression
s_mov_b32 s78, s18 // (op 2328) reg expression
s_mov_b32 s79, 131072 // (op 2328) call()

v_lshrrev_b32 v13, 7, v1 // (op 2345) 
s_mul_i32 s15, s16, s50 // (op 2345) 
s_add_u32 s15, s15, s0 // (op 2345) 
v_and_b32 v24, 15, v1 // (op 2345) 
s_add_u32 s6, s15, s9 // (op 2345) 

v_lshl_add_u32 v26, v13, 4, v24 // (op 2345) 


s_mul_hi_u32 s15, s6, s13 // (op 2345) 
v_and_b32 v13, 63, v1 // (op 2345) 
s_sub_u32 s7, s6, s15 // (op 2345) 
v_lshrrev_b32 v24, 4, v13 // (op 2345) 

s_lshr_b32 s8, s7, 1 // (op 2345) 

s_add_u32 s7, s8, s15 // (op 2345) 


s_lshr_b32 s8, s7, s11 // (op 2345) 

s_and_b32 s7, s11, 2147483648 // (op 2345) 
s_cmp_eq_i32 s7, 0 // (op 2345) 
s_mov_b32 s15, scc // (op 2345) 

s_cmp_eq_i32 s15, 0 // (op 2345) 
s_cselect_b32 s7, s6, s8 // (op 2345) 


s_mul_hi_u32 s8, s7, s12 // (op 2345) 
s_sub_u32 s15, s7, s8 // (op 2345) 
s_lshr_b32 s18, s15, 1 // (op 2345) 

s_add_u32 s15, s18, s8 // (op 2345) 


s_lshr_b32 s8, s15, s10 // (op 2345) 

s_and_b32 s15, s10, 2147483648 // (op 2345) 
s_cmp_eq_i32 s15, 0 // (op 2345) 
s_mov_b32 s18, scc // (op 2345) 

s_cmp_eq_i32 s18, 0 // (op 2345) 
s_cselect_b32 s15, s7, s8 // (op 2345) 


v_lshl_add_u32 v13, s15, 4, v26 // (op 2345) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2345) 
s_mul_i32 s8, s7, s18 // (op 2345) 


s_sub_u32 s7, s6, s8 // (op 2345) 


v_lshl_add_u32 v26, s7, 2, v24 // (op 2345) 


v_mul_lo_u32 v24, v26, s38 // (op 2345) 

v_add_u32 v26, v13, v24 // (op 2345) 


s_mul_i32 s19, s38, s43 // (op 2347) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2347) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2347) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2347) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2347) most significant: sum
s_add_u32 s19, s19, s6 // (op 2347) most significant: sum


s_mov_b32 s76, s40 // (op 2347) reg expression
s_mov_b32 s77, s41 // (op 2347) reg expression
s_mov_b32 s78, s18 // (op 2347) reg expression
s_mov_b32 s79, 131072 // (op 2347) call()

v_and_b32 v13, -64, v1 // (op 2454) 
s_mul_i32 s15, s16, s50 // (op 2454) 
s_add_u32 s15, s15, s0 // (op 2454) 
v_sub_u32 v24, v1, v13 // (op 2454) 

s_add_u32 s6, s15, s9 // (op 2454) 

v_lshrrev_b32 v13, 6, v1 // (op 2454) 
s_mul_hi_u32 s15, s6, s13 // (op 2454) 
v_lshlrev_b32 v27, 1, v13 // (op 2454) 

s_sub_u32 s7, s6, s15 // (op 2454) 
s_lshr_b32 s8, s7, 1 // (op 2454) 

s_add_u32 s7, s8, s15 // (op 2454) 


s_lshr_b32 s8, s7, s11 // (op 2454) 

s_and_b32 s7, s11, 2147483648 // (op 2454) 
s_cmp_eq_i32 s7, 0 // (op 2454) 
s_mov_b32 s15, scc // (op 2454) 

s_cmp_eq_i32 s15, 0 // (op 2454) 
s_cselect_b32 s7, s6, s8 // (op 2454) 


s_mul_hi_u32 s8, s7, s12 // (op 2454) 
s_sub_u32 s15, s7, s8 // (op 2454) 
s_lshr_b32 s18, s15, 1 // (op 2454) 

s_add_u32 s15, s18, s8 // (op 2454) 


s_lshr_b32 s8, s15, s10 // (op 2454) 

s_and_b32 s15, s10, 2147483648 // (op 2454) 
s_cmp_eq_i32 s15, 0 // (op 2454) 
s_mov_b32 s18, scc // (op 2454) 

s_cmp_eq_i32 s18, 0 // (op 2454) 
s_cselect_b32 s15, s7, s8 // (op 2454) 


s_mul_i32 s8, s15, s14 // (op 2454) 

s_sub_u32 s15, s7, s8 // (op 2454) 

v_lshl_add_u32 v13, s15, 6, v24 // (op 2454) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2454) 
s_mul_i32 s8, s7, s18 // (op 2454) 


s_sub_u32 s7, s6, s8 // (op 2454) 


v_lshl_add_u32 v24, s7, 2, v27 // (op 2454) 


v_mul_lo_u32 v27, v24, s44 // (op 2454) 
v_add_u32 v27, v27, v13 // (op 2454) 


s_mul_i32 s19, s44, s43 // (op 2456) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2456) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2456) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2456) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2456) most significant: sum
s_add_u32 s19, s19, s6 // (op 2456) most significant: sum


s_mov_b32 s80, s46 // (op 2456) reg expression
s_mov_b32 s81, s47 // (op 2456) reg expression
s_mov_b32 s82, s18 // (op 2456) reg expression
s_mov_b32 s83, 131072 // (op 2456) call()

v_and_b32 v13, -64, v1 // (op 2473) 
s_mul_i32 s15, s16, s50 // (op 2473) 
s_add_u32 s15, s15, s0 // (op 2473) 
v_sub_u32 v24, v1, v13 // (op 2473) 

s_add_u32 s6, s15, s9 // (op 2473) 

v_lshrrev_b32 v13, 6, v1 // (op 2473) 
s_mul_hi_u32 s15, s6, s13 // (op 2473) 
v_lshlrev_b32 v28, 1, v13 // (op 2473) 

s_sub_u32 s7, s6, s15 // (op 2473) 
s_lshr_b32 s8, s7, 1 // (op 2473) 

s_add_u32 s7, s8, s15 // (op 2473) 


s_lshr_b32 s8, s7, s11 // (op 2473) 

s_and_b32 s7, s11, 2147483648 // (op 2473) 
s_cmp_eq_i32 s7, 0 // (op 2473) 
s_mov_b32 s15, scc // (op 2473) 

s_cmp_eq_i32 s15, 0 // (op 2473) 
s_cselect_b32 s7, s6, s8 // (op 2473) 


s_mul_hi_u32 s8, s7, s12 // (op 2473) 
s_sub_u32 s15, s7, s8 // (op 2473) 
s_lshr_b32 s18, s15, 1 // (op 2473) 

s_add_u32 s15, s18, s8 // (op 2473) 


s_lshr_b32 s8, s15, s10 // (op 2473) 

s_and_b32 s15, s10, 2147483648 // (op 2473) 
s_cmp_eq_i32 s15, 0 // (op 2473) 
s_mov_b32 s18, scc // (op 2473) 

s_cmp_eq_i32 s18, 0 // (op 2473) 
s_cselect_b32 s15, s7, s8 // (op 2473) 


s_mul_i32 s8, s15, s14 // (op 2473) 

s_sub_u32 s15, s7, s8 // (op 2473) 

v_lshl_add_u32 v13, s15, 6, v24 // (op 2473) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2473) 
s_mul_i32 s8, s7, s18 // (op 2473) 


s_sub_u32 s7, s6, s8 // (op 2473) 


v_lshl_add_u32 v24, s7, 2, v28 // (op 2473) 


v_mul_lo_u32 v28, v24, s44 // (op 2473) 
v_add_u32 v28, v28, v13 // (op 2473) 


s_mul_i32 s19, s44, s43 // (op 2475) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2475) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2475) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2475) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2475) most significant: sum
s_add_u32 s19, s19, s6 // (op 2475) most significant: sum


s_mov_b32 s80, s46 // (op 2475) reg expression
s_mov_b32 s81, s47 // (op 2475) reg expression
s_mov_b32 s82, s18 // (op 2475) reg expression
s_mov_b32 s83, 131072 // (op 2475) call()


v_mov_b64 v[30:31], s[38:39] // (op 1684) convert

v_mov_b32 v13, v25 // (op 1684) reg expression
v_mov_b32 v24, v13 // (op 1684) 
buffer_load_ubyte v29, v24, s[76:79], 0 offen offset: 0 // (op 1684) Load value



v_and_b32 v13, -4, v1 // (op 2930) 
v_sub_u32 v24, v1, v13 // (op 2930) 

v_lshlrev_b32 v13, 4, v24 // (op 2930) 

v_lshrrev_b32 v24, 2, v1 // (op 2930) 
v_lshl_add_u32 v30, v24, 6, v13 // (op 2930) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2930) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3465) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 1976) 
v_mov_b64 v[30:31], s[18:19] // (op 1976) convert


v_mov_b64 v[32:33], v[16:17] // (op 1976) reg expression
s_mov_b32 m0, s15 // (op 1976) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1976) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1976) most significant half

s_add_u32 m0, m0, 2048 // (op 1976) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1976) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1976) most significant half

s_add_u32 m0, m0, 2048 // (op 1976) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1976) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1976) most significant half

s_add_u32 m0, m0, 2048 // (op 1976) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds




v_mov_b64 v[30:31], s[44:45] // (op 1691) convert

v_mov_b32 v13, v27 // (op 1691) reg expression
buffer_load_ubyte v32, v13, s[80:83], 0 offen offset: 0 // (op 1691) Load value
v_add_u32 v13, v13, v30 // (op 1691) 
buffer_load_ubyte v33, v13, s[80:83], 0 offen offset: 0 // (op 1691) Load value


v_lshlrev_b32 v13, 4, v1 // (op 2865) 
s_nop 0
v_readfirstlane_b32 s15, v13 // (op 2865) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3446) 


v_mov_b64 v[30:31], v[20:21] // (op 1981) reg expression
s_add_u32 s6, 8192, s15 // (op 1981) 

s_mov_b32 m0, s6 // (op 1981) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 1981) Load value direct to lds




v_mov_b64 v[30:31], s[38:39] // (op 1698) convert

s_lshl_b32 s15, s38, 2 // (op 1698) 
v_add_u32 v13, v25, s15 // (op 1698) 

v_mov_b32 v24, v13 // (op 1698) 
buffer_load_ubyte v34, v24, s[76:79], 0 offen offset: 0 // (op 1698) Load value




v_and_b32 v13, -4, v1 // (op 3149) 
v_sub_u32 v24, v1, v13 // (op 3149) 

v_lshlrev_b32 v13, 4, v24 // (op 3149) 

v_lshrrev_b32 v24, 2, v1 // (op 3149) 
v_lshl_add_u32 v25, v24, 6, v13 // (op 3149) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 3149) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3513) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 1986) 
v_mov_b64 v[24:25], s[18:19] // (op 1986) convert


s_lshl_b64 s[18:19], s[34:35], 7 // (op 1986) 
v_mov_b32 v13, s18 // (op 1986) 
v_mov_b32 v30, s19 // (op 1986) 
v_add_co_u32 v36, s[28:29], v16, v13 // (op 1986) least significant half
v_addc_co_u32 v37, s[28:29], v17, v30, s[28:29] // (op 1986) most significant half




s_add_u32 s6, 10240, s15 // (op 1986) 
s_mov_b32 m0, s6 // (op 1986) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v24 // (op 1986) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1986) most significant half

s_add_u32 m0, m0, 2048 // (op 1986) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v24 // (op 1986) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1986) most significant half

s_add_u32 m0, m0, 2048 // (op 1986) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v24 // (op 1986) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1986) most significant half

s_add_u32 m0, m0, 2048 // (op 1986) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds






v_mov_b64 v[16:17], s[44:45] // (op 1705) convert

s_lshl_b32 s15, s44, 2 // (op 1705) 
v_add_u32 v13, v27, s15 // (op 1705) 

buffer_load_ubyte v24, v13, s[80:83], 0 offen offset: 0 // (op 1705) Load value
v_add_u32 v13, v13, v16 // (op 1705) 
buffer_load_ubyte v25, v13, s[80:83], 0 offen offset: 0 // (op 1705) Load value



v_lshlrev_b32 v13, 4, v1 // (op 3110) 
s_nop 0
v_readfirstlane_b32 s15, v13 // (op 3110) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3504) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 1988) 
v_mov_b32 v13, s18 // (op 1988) 
v_mov_b32 v27, s19 // (op 1988) 
v_add_co_u32 v16, s[28:29], v20, v13 // (op 1988) least significant half
v_addc_co_u32 v17, s[28:29], v21, v27, s[28:29] // (op 1988) most significant half




s_add_u32 s6, 18432, s15 // (op 1988) 

s_mov_b32 m0, s6 // (op 1988) reg expression
buffer_load_dwordx4 v16, s[68:71], 0 offen offset: 0 lds // (op 1988) Load value direct to lds




v_and_b32 v13, -64, v1 // (op 3006) 
v_sub_u32 v27, v1, v13 // (op 3006) 

v_lshrrev_b32 v13, 6, v1 // (op 3006) 
v_lshl_add_u32 v16, v13, 7, v27 // (op 3006) 


s_barrier  // (op 3484) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v32 offset:20480 // (op 1711) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v33 offset:20544 // (op 1711) Store local data 

s_barrier  // (op 3462) 



ds_read_b64_tr_b8 v[36:37], v12 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v12 offset:512 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v12 offset:4096 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v12 offset:4608 // (op 1729) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3481) 


ds_read_u8 v13, v5 offset:20480 // (op 1739) Load local data 



ds_read_b64_tr_b8 v[44:45], v12 offset:32 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v12 offset:544 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v12 offset:4128 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v12 offset:4640 // (op 1734) Transpose load local data 


ds_read_u8 v27, v5 offset:20512 // (op 1744) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 3443) 



ds_read_b64_tr_b8 v[52:53], v0 offset:8192 // (op 1725) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:8320 // (op 1725) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:9216 // (op 1725) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:9344 // (op 1725) Transpose load local data 
v_lshlrev_b32 v35, 4, v1 // (op 2837) 
s_nop 0
v_readfirstlane_b32 s15, v35 // (op 2837) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 2902) 
v_sub_u32 v16, v1, v35 // (op 2902) 

v_lshlrev_b32 v17, 4, v16 // (op 2902) 

v_lshrrev_b32 v16, 2, v1 // (op 2902) 
v_lshl_add_u32 v35, v16, 6, v17 // (op 2902) 


s_nop 0
v_readfirstlane_b32 s6, v35 // (op 2902) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -64, v1 // (op 2977) 
v_sub_u32 v16, v1, v35 // (op 2977) 

v_lshrrev_b32 v17, 6, v1 // (op 2977) 
v_lshl_add_u32 v35, v17, 7, v16 // (op 2977) 


v_lshlrev_b32 v16, 4, v1 // (op 3095) 
s_nop 0
v_readfirstlane_b32 s7, v16 // (op 3095) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 3134) 
v_sub_u32 v17, v1, v16 // (op 3134) 

v_lshlrev_b32 v16, 4, v17 // (op 3134) 

v_lshrrev_b32 v17, 2, v1 // (op 3134) 
v_lshl_add_u32 v20, v17, 6, v16 // (op 3134) 


s_nop 0
v_readfirstlane_b32 s8, v20 // (op 3134) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -64, v1 // (op 3178) 
v_sub_u32 v17, v1, v16 // (op 3178) 

v_lshrrev_b32 v16, 6, v1 // (op 3178) 
v_lshl_add_u32 v20, v16, 7, v17 // (op 3178) 


s_mov_b32 s1, 0 // (op 580) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 579) 
s_mov_b32 s28, s18 // (op 579) convert

s_and_b32 s29, s28, 4294967294 // (op 579) 

s_cmp_lt_u32 s1, s29 // (op 579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_579 // (op 579) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_579 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_579:
s_add_u32 s1, s1, 2 // (op 581) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[36:43], a[0:3], v29, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 184) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[44:51], a[4:7], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 615) 

v_mov_b64 v[16:17], s[38:39] // (op 181) convert

v_mov_b32 v21, v26 // (op 181) 
s_lshl_b32 s18, s38, 3 // (op 181) 
v_add_u32 v30, v21, s18 // (op 181) 

v_mov_b32 v31, v30 // (op 181) 
buffer_load_ubyte v29, v31, s[76:79], 0 offen offset: 0 // (op 181) Load value



s_barrier  // (op 3468) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 1925) 
v_mov_b64 v[16:17], s[18:19] // (op 1925) convert


v_mov_b64 v[30:31], v[18:19] // (op 1925) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 1925) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1925) 

v_mov_b32 v60, s28 // (op 1925) 
v_mov_b32 v61, s29 // (op 1925) 
v_add_co_u32 v62, s[18:19], v30, v60 // (op 1925) least significant half
v_addc_co_u32 v63, s[18:19], v31, v61, s[18:19] // (op 1925) most significant half




s_mov_b32 m0, s6 // (op 1925) reg expression
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v16 // (op 1925) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 1925) most significant half

s_add_u32 m0, m0, 2048 // (op 1925) 
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v16 // (op 1925) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 1925) most significant half

s_add_u32 m0, m0, 2048 // (op 1925) 
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v16 // (op 1925) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 1925) most significant half

s_add_u32 m0, m0, 2048 // (op 1925) 
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds



v_mov_b64 v[16:17], s[44:45] // (op 188) convert

v_mov_b32 v60, v28 // (op 188) 
s_lshl_b32 s18, s44, 3 // (op 188) 
v_add_u32 v61, v60, s18 // (op 188) 

buffer_load_ubyte v32, v61, s[80:83], 0 offen offset: 0 // (op 188) Load value
v_add_u32 v61, v61, v16 // (op 188) 
buffer_load_ubyte v33, v61, s[80:83], 0 offen offset: 0 // (op 188) Load value


s_barrier  // (op 3449) 


v_mov_b64 v[16:17], v[22:23] // (op 1920) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 1920) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1920) 

v_mov_b32 v61, s28 // (op 1920) 
v_mov_b32 v62, s29 // (op 1920) 
v_add_co_u32 v64, s[18:19], v16, v61 // (op 1920) least significant half
v_addc_co_u32 v65, s[18:19], v17, v62, s[18:19] // (op 1920) most significant half




s_add_u32 s18, 8192, s15 // (op 1920) 

s_nop 0
s_mov_b32 m0, s18 // (op 1920) reg expression
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 1920) Load value direct to lds


s_barrier  // (op 3522) 



s_waitcnt vmcnt(10)
ds_write_b8 v20, v24 offset:20736 // (op 508) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v20, v25 offset:20800 // (op 508) Store local data 
s_barrier  // (op 3510) 



ds_read_b64_tr_b8 v[36:37], v11 offset:10240 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v11 offset:10752 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v11 offset:14336 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v11 offset:14848 // (op 501) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3519) 


ds_read_u8 v13, v10 offset:20736 // (op 505) Load local data 



ds_read_b64_tr_b8 v[44:45], v11 offset:10272 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:10784 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:14368 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v11 offset:14880 // (op 620) Transpose load local data 


ds_read_u8 v27, v10 offset:20768 // (op 623) Load local data 
s_barrier  // (op 3633) 
s_waitcnt vmcnt(8)
s_barrier  // (op 3501) 



ds_read_b64_tr_b8 v[52:53], v4 offset:18432 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:18560 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:19456 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:19584 // (op 499) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[36:43], a[0:3], v34, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 506) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[44:51], a[4:7], v34, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 616) 

v_mov_b64 v[62:63], s[38:39] // (op 503) convert

s_lshl_b32 s18, s38, 2 // (op 503) 
s_mul_i32 s19, 3, s18 // (op 503) 

v_add_u32 v61, v21, s19 // (op 503) 

v_mov_b32 v64, v61 // (op 503) 
buffer_load_ubyte v34, v64, s[76:79], 0 offen offset: 0 // (op 503) Load value




s_barrier  // (op 3516) 
s_lshl_b32 s18, s38, 3 // (op 2337) 

v_add_u32 v26, v26, s18 // (op 2337) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 1948) 
v_mov_b64 v[62:63], s[18:19] // (op 1948) convert


s_lshl_b64 s[18:19], s[34:35], 7 // (op 1948) 
s_mul_i32 s29, 3, s19 // (op 1948) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 1948) least significant
s_mul_hi_u32 s48, 3, s18 // (op 1948) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 1948) most significant: sum


v_mov_b32 v21, s28 // (op 1948) 
v_mov_b32 v61, s29 // (op 1948) 
v_add_co_u32 v64, s[18:19], v30, v21 // (op 1948) least significant half
v_addc_co_u32 v65, s[18:19], v31, v61, s[18:19] // (op 1948) most significant half




s_add_u32 s18, 10240, s8 // (op 1948) 
s_nop 0
s_mov_b32 m0, s18 // (op 1948) reg expression
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v62 // (op 1948) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1948) most significant half

s_add_u32 m0, m0, 2048 // (op 1948) 
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v62 // (op 1948) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1948) most significant half

s_add_u32 m0, m0, 2048 // (op 1948) 
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v62 // (op 1948) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1948) most significant half

s_add_u32 m0, m0, 2048 // (op 1948) 
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds





v_mov_b64 v[30:31], s[44:45] // (op 510) convert

s_lshl_b32 s18, s44, 2 // (op 510) 
s_mul_i32 s19, 3, s18 // (op 510) 

v_add_u32 v21, v60, s19 // (op 510) 

buffer_load_ubyte v24, v21, s[80:83], 0 offen offset: 0 // (op 510) Load value
v_add_u32 v21, v21, v30 // (op 510) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 2209) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2209) 

v_mov_b32 v61, s28 // (op 2209) 
v_mov_b32 v62, s29 // (op 2209) 
v_add_co_u32 v18, s[18:19], v18, v61 // (op 2209) least significant half
v_addc_co_u32 v19, s[18:19], v19, v62, s[18:19] // (op 2209) most significant half




buffer_load_ubyte v25, v21, s[80:83], 0 offen offset: 0 // (op 510) Load value



s_barrier  // (op 3507) 
s_lshl_b32 s18, s44, 3 // (op 2465) 


v_add_u32 v28, v28, s18 // (op 2465) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 1943) 
s_mul_i32 s29, 3, s19 // (op 1943) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 1943) least significant
s_mul_hi_u32 s48, 3, s18 // (op 1943) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 1943) most significant: sum


v_mov_b32 v21, s28 // (op 1943) 
v_mov_b32 v30, s29 // (op 1943) 
v_add_co_u32 v60, s[18:19], v16, v21 // (op 1943) least significant half
v_addc_co_u32 v61, s[18:19], v17, v30, s[18:19] // (op 1943) most significant half




s_add_u32 s18, 18432, s7 // (op 1943) 

s_nop 0
s_mov_b32 m0, s18 // (op 1943) reg expression
buffer_load_dwordx4 v60, s[68:71], 0 offen offset: 0 lds // (op 1943) Load value direct to lds



s_barrier  // (op 3487) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 2077) 



s_lshl_b64 s[28:29], s[18:19], 1 // (op 2077) 

v_mov_b32 v21, s28 // (op 2077) 
v_mov_b32 v16, s29 // (op 2077) 
v_add_co_u32 v22, s[18:19], v22, v21 // (op 2077) least significant half
v_addc_co_u32 v23, s[18:19], v23, v16, s[18:19] // (op 2077) most significant half




s_waitcnt vmcnt(10)
ds_write_b8 v35, v32 offset:20480 // (op 186) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v33 offset:20544 // (op 186) Store local data 
s_barrier  // (op 3474) 



ds_read_b64_tr_b8 v[36:37], v9 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:512 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:4096 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4608 // (op 179) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3494) 


ds_read_u8 v13, v8 offset:20480 // (op 183) Load local data 



ds_read_b64_tr_b8 v[44:45], v9 offset:32 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v9 offset:544 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:4128 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:4640 // (op 611) Transpose load local data 


ds_read_u8 v27, v8 offset:20512 // (op 614) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 3455) 



ds_read_b64_tr_b8 v[52:53], v3 offset:8192 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:8320 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v3 offset:9216 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:9344 // (op 177) Transpose load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 579) 
s_mov_b32 s28, s18 // (op 579) convert

s_and_b32 s29, s28, 4294967294 // (op 579) 

s_cmp_lt_u32 s1, s29 // (op 579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_579 // (op 579) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_579 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_579:












s_barrier  // (op 3617) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2015) 
s_and_b32 s15, s18, 4294967294 // (op 2015) 
s_add_u32 s6, s0, s15 // (op 2015) 

s_mul_i32 s15, s16, s50 // (op 2015) 
s_add_u32 s15, s15, s6 // (op 2015) 

s_add_u32 s6, s15, s9 // (op 2015) 

s_mul_hi_u32 s15, s6, s13 // (op 2015) 
s_sub_u32 s7, s6, s15 // (op 2015) 
s_lshr_b32 s8, s7, 1 // (op 2015) 

s_add_u32 s7, s8, s15 // (op 2015) 


s_lshr_b32 s8, s7, s11 // (op 2015) 

s_and_b32 s7, s11, 2147483648 // (op 2015) 
s_cmp_eq_i32 s7, 0 // (op 2015) 
s_mov_b32 s15, scc // (op 2015) 

s_cmp_eq_i32 s15, 0 // (op 2015) 
s_cselect_b32 s7, s6, s8 // (op 2015) 


s_mul_hi_u32 s8, s7, s12 // (op 2015) 
s_sub_u32 s15, s7, s8 // (op 2015) 
s_lshr_b32 s28, s15, 1 // (op 2015) 

s_add_u32 s15, s28, s8 // (op 2015) 


s_lshr_b32 s8, s15, s10 // (op 2015) 

s_and_b32 s15, s10, 2147483648 // (op 2015) 
s_cmp_eq_i32 s15, 0 // (op 2015) 
s_mov_b32 s28, scc // (op 2015) 

s_cmp_eq_i32 s28, 0 // (op 2015) 
s_cselect_b32 s15, s7, s8 // (op 2015) 


s_lshl_b32 s8, s15, 4 // (op 2015) 

s_mov_b32 s29, 0 // (op 2015) convert
s_mov_b32 s28, s8 // (op 2015) convert

s_mul_i32 s8, s7, s18 // (op 2015) 


s_sub_u32 s7, s6, s8 // (op 2015) 


v_lshl_add_u32 v26, s7, 7, v1 // (op 2015) 

v_mul_lo_u32 v17, v26, s31 // (op 2015) most significant: low of low * high
v_mul_hi_u32 v35, v26, s30 // (op 2015) most significant: high of low * low
v_mul_lo_u32 v16, v26, s30 // (op 2015) least significant: low of low * low
v_add_u32 v17, v17, v35 // (op 2015) most significant: sum


v_mov_b32 v26, s28 // (op 2015) 
v_mov_b32 v35, s29 // (op 2015) 
v_add_co_u32 v18, s[18:19], v26, v16 // (op 2015) least significant half
v_addc_co_u32 v19, s[18:19], v35, v17, s[18:19] // (op 2015) most significant half





s_mul_i32 s19, s30, s3 // (op 2017) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2017) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2017) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2017) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2017) most significant: sum
s_add_u32 s19, s19, s6 // (op 2017) most significant: sum


s_mov_b32 s68, s32 // (op 2017) reg expression
s_mov_b32 s69, s33 // (op 2017) reg expression
s_mov_b32 s70, s18 // (op 2017) reg expression
s_mov_b32 s71, 131072 // (op 2017) call()

v_and_b32 v26, -4, v1 // (op 2153) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2153) 
v_sub_u32 v35, v1, v26 // (op 2153) 

s_and_b32 s15, s18, 4294967294 // (op 2153) 
v_lshlrev_b32 v26, 4, v35 // (op 2153) 

s_add_u32 s6, s0, s15 // (op 2153) 

v_lshrrev_b32 v35, 2, v1 // (op 2153) 
s_mul_i32 s15, s16, s50 // (op 2153) 
s_add_u32 s15, s15, s6 // (op 2153) 

s_add_u32 s6, s15, s9 // (op 2153) 

s_mul_hi_u32 s15, s6, s13 // (op 2153) 
s_sub_u32 s7, s6, s15 // (op 2153) 
s_lshr_b32 s8, s7, 1 // (op 2153) 

s_add_u32 s7, s8, s15 // (op 2153) 


s_lshr_b32 s8, s7, s11 // (op 2153) 

s_and_b32 s7, s11, 2147483648 // (op 2153) 
s_cmp_eq_i32 s7, 0 // (op 2153) 
s_mov_b32 s15, scc // (op 2153) 

s_cmp_eq_i32 s15, 0 // (op 2153) 
s_cselect_b32 s7, s6, s8 // (op 2153) 


s_mul_hi_u32 s8, s7, s12 // (op 2153) 
s_sub_u32 s15, s7, s8 // (op 2153) 
s_lshr_b32 s28, s15, 1 // (op 2153) 

s_add_u32 s15, s28, s8 // (op 2153) 


s_lshr_b32 s8, s15, s10 // (op 2153) 

s_and_b32 s15, s10, 2147483648 // (op 2153) 
s_cmp_eq_i32 s15, 0 // (op 2153) 
s_mov_b32 s28, scc // (op 2153) 

s_cmp_eq_i32 s28, 0 // (op 2153) 
s_cselect_b32 s15, s7, s8 // (op 2153) 


s_mul_i32 s8, s15, s14 // (op 2153) 

s_sub_u32 s15, s7, s8 // (op 2153) 

s_mul_i32 s8, s7, s18 // (op 2153) 


v_lshl_add_u32 v16, s15, 6, v26 // (op 2153) 


s_sub_u32 s7, s6, s8 // (op 2153) 


v_mov_b32 v21, 0 // (op 2153) convert
v_mov_b32 v20, v16 // (op 2153) convert

v_lshl_add_u32 v26, s7, 7, v35 // (op 2153) 


v_mul_lo_u32 v17, v26, s35 // (op 2153) most significant: low of low * high
v_mul_hi_u32 v35, v26, s34 // (op 2153) most significant: high of low * low
v_mul_lo_u32 v16, v26, s34 // (op 2153) least significant: low of low * low
v_add_u32 v17, v17, v35 // (op 2153) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v20 // (op 2153) least significant half
v_addc_co_u32 v17, s[18:19], v17, v21, s[18:19] // (op 2153) most significant half



s_mul_i32 s19, s34, s3 // (op 2155) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2155) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2155) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2155) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2155) most significant: sum
s_add_u32 s19, s19, s6 // (op 2155) most significant: sum


s_mov_b32 s72, s36 // (op 2155) reg expression
s_mov_b32 s73, s37 // (op 2155) reg expression
s_mov_b32 s74, s18 // (op 2155) reg expression
s_mov_b32 s75, 131072 // (op 2155) call()

v_lshrrev_b32 v26, 7, v1 // (op 2281) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2281) 
v_and_b32 v35, 15, v1 // (op 2281) 
s_and_b32 s15, s18, 4294967294 // (op 2281) 
v_lshl_add_u32 v20, v26, 4, v35 // (op 2281) 


s_add_u32 s6, s0, s15 // (op 2281) 

v_and_b32 v26, 63, v1 // (op 2281) 
v_lshrrev_b32 v35, 4, v26 // (op 2281) 

s_mul_i32 s15, s16, s50 // (op 2281) 
s_add_u32 s15, s15, s6 // (op 2281) 

s_add_u32 s6, s15, s9 // (op 2281) 

s_mul_hi_u32 s15, s6, s13 // (op 2281) 
s_sub_u32 s7, s6, s15 // (op 2281) 
s_lshr_b32 s8, s7, 1 // (op 2281) 

s_add_u32 s7, s8, s15 // (op 2281) 


s_lshr_b32 s8, s7, s11 // (op 2281) 

s_and_b32 s7, s11, 2147483648 // (op 2281) 
s_cmp_eq_i32 s7, 0 // (op 2281) 
s_mov_b32 s15, scc // (op 2281) 

s_cmp_eq_i32 s15, 0 // (op 2281) 
s_cselect_b32 s7, s6, s8 // (op 2281) 


s_mul_hi_u32 s8, s7, s12 // (op 2281) 
s_sub_u32 s15, s7, s8 // (op 2281) 
s_lshr_b32 s28, s15, 1 // (op 2281) 

s_add_u32 s15, s28, s8 // (op 2281) 


s_lshr_b32 s8, s15, s10 // (op 2281) 

s_and_b32 s15, s10, 2147483648 // (op 2281) 
s_cmp_eq_i32 s15, 0 // (op 2281) 
s_mov_b32 s28, scc // (op 2281) 

s_cmp_eq_i32 s28, 0 // (op 2281) 
s_cselect_b32 s15, s7, s8 // (op 2281) 


s_mul_i32 s8, s7, s18 // (op 2281) 


v_lshl_add_u32 v26, s15, 4, v20 // (op 2281) 


s_sub_u32 s7, s6, s8 // (op 2281) 


v_lshl_add_u32 v20, s7, 2, v35 // (op 2281) 


v_mul_lo_u32 v35, v20, s38 // (op 2281) 

v_add_u32 v20, v26, v35 // (op 2281) 


s_mul_i32 s19, s38, s43 // (op 2283) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2283) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2283) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2283) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2283) most significant: sum
s_add_u32 s19, s19, s6 // (op 2283) most significant: sum


s_mov_b32 s76, s40 // (op 2283) reg expression
s_mov_b32 s77, s41 // (op 2283) reg expression
s_mov_b32 s78, s18 // (op 2283) reg expression
s_mov_b32 s79, 131072 // (op 2283) call()

v_and_b32 v26, -64, v1 // (op 2409) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2409) 
v_sub_u32 v35, v1, v26 // (op 2409) 

s_and_b32 s15, s18, 4294967294 // (op 2409) 
v_lshrrev_b32 v26, 6, v1 // (op 2409) 
s_add_u32 s6, s0, s15 // (op 2409) 

v_lshlrev_b32 v21, 1, v26 // (op 2409) 

s_mul_i32 s15, s16, s50 // (op 2409) 
s_add_u32 s15, s15, s6 // (op 2409) 

s_add_u32 s6, s15, s9 // (op 2409) 

s_mul_hi_u32 s15, s6, s13 // (op 2409) 
s_sub_u32 s7, s6, s15 // (op 2409) 
s_lshr_b32 s8, s7, 1 // (op 2409) 

s_add_u32 s7, s8, s15 // (op 2409) 


s_lshr_b32 s8, s7, s11 // (op 2409) 

s_and_b32 s7, s11, 2147483648 // (op 2409) 
s_cmp_eq_i32 s7, 0 // (op 2409) 
s_mov_b32 s15, scc // (op 2409) 

s_cmp_eq_i32 s15, 0 // (op 2409) 
s_cselect_b32 s7, s6, s8 // (op 2409) 


s_mul_hi_u32 s8, s7, s12 // (op 2409) 
s_sub_u32 s15, s7, s8 // (op 2409) 
s_lshr_b32 s28, s15, 1 // (op 2409) 

s_add_u32 s15, s28, s8 // (op 2409) 


s_lshr_b32 s8, s15, s10 // (op 2409) 

s_and_b32 s15, s10, 2147483648 // (op 2409) 
s_cmp_eq_i32 s15, 0 // (op 2409) 
s_mov_b32 s28, scc // (op 2409) 

s_cmp_eq_i32 s28, 0 // (op 2409) 
s_cselect_b32 s15, s7, s8 // (op 2409) 


s_mul_i32 s8, s15, s14 // (op 2409) 

s_sub_u32 s15, s7, s8 // (op 2409) 

s_mul_i32 s8, s7, s18 // (op 2409) 


v_lshl_add_u32 v26, s15, 6, v35 // (op 2409) 


s_sub_u32 s7, s6, s8 // (op 2409) 


v_lshl_add_u32 v35, s7, 2, v21 // (op 2409) 


v_mul_lo_u32 v21, v35, s44 // (op 2409) 
v_add_u32 v21, v21, v26 // (op 2409) 


s_mul_i32 s19, s44, s43 // (op 2411) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2411) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2411) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2411) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2411) most significant: sum
s_add_u32 s19, s19, s6 // (op 2411) most significant: sum


s_mov_b32 s80, s46 // (op 2411) reg expression
s_mov_b32 s81, s47 // (op 2411) reg expression
s_mov_b32 s82, s18 // (op 2411) reg expression
s_mov_b32 s83, 131072 // (op 2411) call()

v_lshlrev_b32 v26, 4, v1 // (op 2850) 
s_nop 0
v_readfirstlane_b32 s15, v26 // (op 2850) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 2915) 
v_sub_u32 v35, v1, v26 // (op 2915) 

v_lshlrev_b32 v26, 4, v35 // (op 2915) 

v_lshrrev_b32 v35, 2, v1 // (op 2915) 
v_lshl_add_u32 v22, v35, 6, v26 // (op 2915) 


s_nop 0
v_readfirstlane_b32 s6, v22 // (op 2915) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -64, v1 // (op 2990) 
v_sub_u32 v35, v1, v26 // (op 2990) 

v_lshrrev_b32 v26, 6, v1 // (op 2990) 
v_lshl_add_u32 v22, v26, 7, v35 // (op 2990) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 585) 
s_and_b32 s7, s18, 4294967294 // (op 585) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 584) 
s_mov_b32 s8, s18 // (op 584) convert

s_cmp_lt_u32 s7, s8 // (op 584) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_584 // (op 584) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_584 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_584:
s_barrier  // (op 3452) 
s_add_u32 s7, s7, 1 // (op 586) 


v_mov_b64 v[24:25], v[18:19] // (op 1930) 
v_mov_b64 v[26:27], v[24:25] // (op 1930) reg expression
s_add_u32 s8, 8192, s15 // (op 1930) 

s_mov_b32 m0, s8 // (op 1930) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 1930) Load value direct to lds



s_lshl_b64 s[18:19], s[30:31], 7 // (op 2005) 
v_mov_b32 v13, s18 // (op 2005) 
v_mov_b32 v23, s19 // (op 2005) 
v_add_co_u32 v18, s[28:29], v18, v13 // (op 2005) least significant half
v_addc_co_u32 v19, s[28:29], v19, v23, s[28:29] // (op 2005) most significant half




s_barrier  // (op 3471) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 1936) 
v_mov_b64 v[24:25], s[18:19] // (op 1936) convert


v_mov_b64 v[26:27], v[16:17] // (op 1936) 
v_mov_b64 v[28:29], v[26:27] // (op 1936) reg expression
s_mov_b32 m0, s6 // (op 1936) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 1936) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1936) most significant half

s_add_u32 m0, m0, 2048 // (op 1936) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 1936) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1936) most significant half

s_add_u32 m0, m0, 2048 // (op 1936) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 1936) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1936) most significant half

s_add_u32 m0, m0, 2048 // (op 1936) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds



s_lshl_b64 s[18:19], s[34:35], 7 // (op 2145) 
v_mov_b32 v13, s18 // (op 2145) 
v_mov_b32 v23, s19 // (op 2145) 
v_add_co_u32 v16, s[28:29], v16, v13 // (op 2145) least significant half
v_addc_co_u32 v17, s[28:29], v17, v23, s[28:29] // (op 2145) most significant half





v_mov_b64 v[24:25], s[44:45] // (op 452) convert

v_mov_b32 v13, v21 // (op 452) 
v_mov_b32 v23, v13 // (op 452) reg expression
buffer_load_ubyte v26, v23, s[80:83], 0 offen offset: 0 // (op 452) Load value
v_add_u32 v23, v23, v24 // (op 452) 
buffer_load_ubyte v27, v23, s[80:83], 0 offen offset: 0 // (op 452) Load value



s_lshl_b32 s8, s44, 2 // (op 2401) 
v_add_u32 v21, v21, s8 // (op 2401) 

s_barrier  // (op 3490) 
s_waitcnt vmcnt(6)
s_barrier  // (op 3458) 






ds_read_b64_tr_b8 v[28:29], v3 offset:8192 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:8320 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:9216 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:9344 // (op 441) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v26 offset:20480 // (op 450) Store local data 
s_barrier  // (op 3477) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v27 offset:20544 // (op 450) Store local data 




ds_read_b64_tr_b8 v[36:37], v9 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:512 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:4096 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4608 // (op 443) Transpose load local data 

v_mov_b64 v[26:27], s[38:39] // (op 445) convert

v_mov_b32 v13, v20 // (op 445) 
v_mov_b32 v23, v13 // (op 445) reg expression
v_mov_b32 v24, v23 // (op 445) 
buffer_load_ubyte v25, v24, s[76:79], 0 offen offset: 0 // (op 445) Load value




s_lshl_b32 s8, s38, 2 // (op 2273) 
v_add_u32 v20, v20, s8 // (op 2273) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3497) 


ds_read_u8 v13, v8 offset:20480 // (op 447) Load local data 



ds_read_b64_tr_b8 v[44:45], v9 offset:32 // (op 592) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 448) 
ds_read_b64_tr_b8 v[46:47], v9 offset:544 // (op 592) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v9 offset:4128 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:4640 // (op 592) Transpose load local data 


ds_read_u8 v13, v8 offset:20512 // (op 595) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[44:51], a[4:7], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 596) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 584) 
s_mov_b32 s8, s18 // (op 584) convert

s_cmp_lt_u32 s7, s8 // (op 584) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_584 // (op 584) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_584 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_584:








s_barrier  // (op 3621) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_218
v_accvgpr_read v16, a0 // (op 226) 
v_accvgpr_read v17, a1 // (op 226) 
v_accvgpr_read v18, a2 // (op 226) 
v_accvgpr_read v19, a3 // (op 226) 
v_mul_f32 v20, v15, v16 // (op 226) 
v_mul_f32 v21, v15, v17 // (op 226) 
v_mul_f32 v22, v15, v18 // (op 226) 
v_mul_f32 v23, v15, v19 // (op 226) 



v_mov_b32 v0, v7 // (op 228) reg expression

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 228) Store value


v_accvgpr_read v16, a4 // (op 776) 
v_accvgpr_read v17, a5 // (op 776) 
v_accvgpr_read v18, a6 // (op 776) 
v_accvgpr_read v19, a7 // (op 776) 
v_mul_f32 v20, v15, v16 // (op 776) 
v_mul_f32 v21, v15, v17 // (op 776) 
v_mul_f32 v22, v15, v18 // (op 776) 
v_mul_f32 v23, v15, v19 // (op 776) 



s_lshl_b32 s15, s24, 7 // (op 778) 
v_add_u32 v0, v7, s15 // (op 778) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 778) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_218:


v_mov_b32 v0, v6 // (op 241) reg expression

buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen // (op 241) Load value

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

buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 240) Store value




s_lshl_b32 s15, s4, 7 // (op 800) 
v_add_u32 v0, v6, s15 // (op 800) 


buffer_load_dwordx4 v[16:19], v0, s[56:59], 0 offen // (op 800) Load value

v_accvgpr_read v20, a4 // (op 797) 
v_accvgpr_read v21, a5 // (op 797) 
v_accvgpr_read v22, a6 // (op 797) 
v_accvgpr_read v23, a7 // (op 797) 
v_mul_f32 v24, v15, v20 // (op 797) 
v_mul_f32 v25, v15, v21 // (op 797) 
v_mul_f32 v26, v15, v22 // (op 797) 
v_mul_f32 v27, v15, v23 // (op 797) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 801) 
v_mul_f32 v21, v14, v17 // (op 801) 
v_mul_f32 v22, v14, v18 // (op 801) 
v_mul_f32 v23, v14, v19 // (op 801) 

v_add_f32 v16, v20, v24 // (op 798) 
v_add_f32 v17, v21, v25 // (op 798) 
v_add_f32 v18, v22, v26 // (op 798) 
v_add_f32 v19, v23, v27 // (op 798) 




s_lshl_b32 s15, s24, 7 // (op 799) 
v_add_u32 v0, v7, s15 // (op 799) 


buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 799) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_218:
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
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:









































s_endpgm  // End of RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2, .LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end-RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 76
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
    DestDataType: s
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
    macrotile: [16, 64, 128]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 7
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
  - .name: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 74
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

