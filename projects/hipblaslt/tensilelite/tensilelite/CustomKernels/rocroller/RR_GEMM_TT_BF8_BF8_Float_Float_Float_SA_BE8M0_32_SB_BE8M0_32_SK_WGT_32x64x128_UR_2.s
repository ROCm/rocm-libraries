// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2,@function
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2619) 
v_and_b32 v3, 15, v1 // (op 2619) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2619) 


v_and_b32 v0, 48, v1 // (op 2619) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2619) 


v_lshrrev_b32 v0, 7, v1 // (op 2877) 
v_and_b32 v4, 15, v1 // (op 2877) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2877) 


v_and_b32 v0, 48, v1 // (op 2877) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2877) 


v_lshrrev_b32 v0, 7, v1 // (op 3122) 
v_and_b32 v5, 15, v1 // (op 3122) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 3122) 


v_and_b32 v0, 48, v1 // (op 3122) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 3122) 


v_lshrrev_b32 v0, 7, v1 // (op 3234) 
v_and_b32 v6, 15, v1 // (op 3234) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 3234) 


v_and_b32 v0, 48, v1 // (op 3234) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 3234) 


v_and_b32 v0, 48, v1 // (op 2689) 
v_and_b32 v7, 63, v1 // (op 2689) 
v_sub_u32 v8, v7, v0 // (op 2689) 

v_lshrrev_b32 v7, 1, v8 // (op 2689) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 2689) 


v_and_b32 v0, 1, v8 // (op 2689) 

v_lshlrev_b32 v7, 3, v0 // (op 2689) 

v_lshrrev_b32 v0, 6, v1 // (op 2689) 
v_and_b32 v8, 1, v0 // (op 2689) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 2689) 


v_add_u32 v7, v9, v0 // (op 2689) 


v_lshrrev_b32 v0, 6, v1 // (op 2765) 
v_and_b32 v8, 1, v0 // (op 2765) 

v_and_b32 v0, 15, v1 // (op 2765) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 2765) 


v_and_b32 v0, 63, v1 // (op 2765) 
v_lshrrev_b32 v8, 4, v0 // (op 2765) 

v_lshl_add_u32 v0, v8, 6, v9 // (op 2765) 


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
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 7, v1 // (op 2540) 
s_mul_i32 s54, s16, s8 // (op 2540) 
s_add_u32 s54, s54, s0 // (op 2540) 
v_and_b32 v17, 63, v1 // (op 2540) 
s_mul_hi_u32 s55, s54, s13 // (op 2540) 
v_lshrrev_b32 v18, 4, v17 // (op 2540) 
s_sub_u32 s56, s54, s55 // (op 2540) 
v_lshlrev_b32 v19, 2, v18 // (op 2540) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2540) 


s_lshr_b32 s57, s56, 1 // (op 2540) 

s_add_u32 s56, s57, s55 // (op 2540) 


v_lshrrev_b32 v16, 2, v17 // (op 2540) 

s_lshr_b32 s55, s56, s11 // (op 2540) 

v_and_b32 v17, 3, v16 // (op 2540) 

s_and_b32 s56, s11, 2147483648 // (op 2540) 
v_and_b32 v16, 3, v1 // (op 2540) 
s_cmp_eq_i32 s56, 0 // (op 2540) 
s_mov_b32 s57, scc // (op 2540) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 2540) 


s_cmp_eq_i32 s57, 0 // (op 2540) 
s_cselect_b32 s56, s54, s55 // (op 2540) 



v_lshrrev_b32 v16, 6, v1 // (op 2540) 
s_mul_hi_u32 s54, s56, s12 // (op 2540) 
v_and_b32 v17, 1, v16 // (op 2540) 

s_sub_u32 s55, s56, s54 // (op 2540) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 2540) 


s_lshr_b32 s57, s55, 1 // (op 2540) 

s_add_u32 s55, s57, s54 // (op 2540) 


s_lshr_b32 s54, s55, s10 // (op 2540) 

s_and_b32 s55, s10, 2147483648 // (op 2540) 
s_cmp_eq_i32 s55, 0 // (op 2540) 
s_mov_b32 s57, scc // (op 2540) 

s_cmp_eq_i32 s57, 0 // (op 2540) 
s_cselect_b32 s55, s56, s54 // (op 2540) 


s_mul_i32 s54, s55, s14 // (op 2540) 
v_lshl_add_u32 v17, s55, 5, v18 // (op 2540) 


s_sub_u32 s55, s56, s54 // (op 2540) 


v_lshl_add_u32 v18, s55, 6, v16 // (op 2540) 


v_mul_lo_u32 v16, v18, s4 // (op 2540) 

v_add_lshl_u32 v18, v17, v16, 2 // (op 2540) 


s_mul_i32 s55, s4, s23 // (op 2542) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 2542) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 2542) least significant
s_mul_hi_u32 s57, s4, s22 // (op 2542) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 2542) most significant: sum
s_add_u32 s55, s55, s57 // (op 2542) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 2 // (op 2542) 

s_mov_b32 s60, s20 // (op 2542) reg expression
s_mov_b32 s61, s21 // (op 2542) reg expression
s_mov_b32 s62, s56 // (op 2542) reg expression
s_mov_b32 s63, 131072 // (op 2542) call()

v_lshrrev_b32 v16, 7, v1 // (op 2631) 
v_and_b32 v17, 15, v1 // (op 2631) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 2631) 


v_and_b32 v16, 48, v1 // (op 2631) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 2631) 


v_and_b32 v16, 48, v1 // (op 2706) 
v_and_b32 v19, 63, v1 // (op 2706) 
v_sub_u32 v20, v19, v16 // (op 2706) 

v_lshrrev_b32 v19, 1, v20 // (op 2706) 
v_add_lshl_u32 v21, v16, v19, 6 // (op 2706) 


v_and_b32 v16, 1, v20 // (op 2706) 

v_lshlrev_b32 v19, 3, v16 // (op 2706) 

v_lshrrev_b32 v16, 6, v1 // (op 2706) 
v_and_b32 v20, 1, v16 // (op 2706) 

v_lshl_add_u32 v16, v20, 4, v19 // (op 2706) 


v_add_u32 v19, v21, v16 // (op 2706) 


v_lshrrev_b32 v16, 6, v1 // (op 2782) 
v_and_b32 v20, 1, v16 // (op 2782) 

v_and_b32 v16, 15, v1 // (op 2782) 
v_lshl_add_u32 v21, v20, 4, v16 // (op 2782) 


v_and_b32 v16, 63, v1 // (op 2782) 
v_lshrrev_b32 v20, 4, v16 // (op 2782) 

v_lshl_add_u32 v16, v20, 6, v21 // (op 2782) 


v_lshrrev_b32 v20, 7, v1 // (op 2800) 
s_mul_i32 s54, s16, s8 // (op 2800) 
s_add_u32 s54, s54, s0 // (op 2800) 
v_and_b32 v21, 63, v1 // (op 2800) 
s_mul_hi_u32 s55, s54, s13 // (op 2800) 
v_lshrrev_b32 v22, 4, v21 // (op 2800) 
s_sub_u32 s56, s54, s55 // (op 2800) 
v_lshlrev_b32 v23, 2, v22 // (op 2800) 

s_lshr_b32 s57, s56, 1 // (op 2800) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2800) 


s_add_u32 s56, s57, s55 // (op 2800) 


v_lshrrev_b32 v20, 2, v21 // (op 2800) 

s_lshr_b32 s55, s56, s11 // (op 2800) 

v_and_b32 v21, 3, v20 // (op 2800) 

s_and_b32 s56, s11, 2147483648 // (op 2800) 
v_and_b32 v20, 3, v1 // (op 2800) 
s_cmp_eq_i32 s56, 0 // (op 2800) 
s_mov_b32 s57, scc // (op 2800) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 2800) 


s_cmp_eq_i32 s57, 0 // (op 2800) 
s_cselect_b32 s56, s54, s55 // (op 2800) 



v_lshrrev_b32 v20, 6, v1 // (op 2800) 
s_mul_hi_u32 s54, s56, s12 // (op 2800) 
v_and_b32 v21, 1, v20 // (op 2800) 

s_sub_u32 s55, s56, s54 // (op 2800) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 2800) 


s_lshr_b32 s57, s55, 1 // (op 2800) 

s_add_u32 s55, s57, s54 // (op 2800) 


s_lshr_b32 s54, s55, s10 // (op 2800) 

s_and_b32 s55, s10, 2147483648 // (op 2800) 
s_cmp_eq_i32 s55, 0 // (op 2800) 
s_mov_b32 s57, scc // (op 2800) 

s_cmp_eq_i32 s57, 0 // (op 2800) 
s_cselect_b32 s55, s56, s54 // (op 2800) 


s_mul_i32 s54, s55, s14 // (op 2800) 
v_lshl_add_u32 v21, s55, 5, v22 // (op 2800) 


s_sub_u32 s55, s56, s54 // (op 2800) 


v_lshl_add_u32 v22, s55, 6, v20 // (op 2800) 


v_mul_lo_u32 v20, v22, s24 // (op 2800) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 2800) 


s_mul_i32 s55, s24, s23 // (op 2802) most significant: low of low * high
s_mul_i32 s56, s25, s22 // (op 2802) most significant: low of high * low
s_mul_i32 s54, s24, s22 // (op 2802) least significant
s_mul_hi_u32 s57, s24, s22 // (op 2802) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 2802) most significant: sum
s_add_u32 s55, s55, s57 // (op 2802) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 2 // (op 2802) 

s_mov_b32 s64, s26 // (op 2802) reg expression
s_mov_b32 s65, s27 // (op 2802) reg expression
s_mov_b32 s66, s56 // (op 2802) reg expression
s_mov_b32 s67, 131072 // (op 2802) call()

v_lshrrev_b32 v20, 5, v1 // (op 3058) 
v_lshlrev_b32 v21, 2, v20 // (op 3058) 

v_lshl_add_u32 v20, s16, 5, v21 // (op 3058) 

v_and_b32 v21, -32, v1 // (op 3058) 
v_sub_u32 v23, v1, v21 // (op 3058) 

v_lshl_add_u32 v21, v20, 6, v23 // (op 3058) 


v_lshlrev_b32 v20, 2, v21 // (op 3058) 

s_lshl_b32 s54, s15, 13 // (op 3060) 
s_mov_b32 s56, s28 // (op 3060) reg expression
s_mov_b32 s57, s29 // (op 3060) reg expression
s_mov_b32 s58, s54 // (op 3060) reg expression
s_mov_b32 s59, 131072 // (op 3060) call()

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
v_accvgpr_write a4, 0.00000 // (op 971) call()
v_accvgpr_write a5, 0.00000 // (op 971) call()
v_accvgpr_write a6, 0.00000 // (op 971) call()
v_accvgpr_write a7, 0.00000 // (op 971) call()
v_lshrrev_b32 v21, 3, v1 // (op 2112) 
s_mul_i32 s54, s16, s8 // (op 2112) 
s_add_u32 s54, s54, s0 // (op 2112) 
v_and_b32 v23, -8, v1 // (op 2112) 
s_mul_hi_u32 s55, s54, s13 // (op 2112) 
v_sub_u32 v24, v1, v23 // (op 2112) 

s_sub_u32 s68, s54, s55 // (op 2112) 
v_lshlrev_b32 v23, 4, v24 // (op 2112) 

s_lshr_b32 s69, s68, 1 // (op 2112) 

s_add_u32 s68, s69, s55 // (op 2112) 


s_lshr_b32 s55, s68, s11 // (op 2112) 

s_and_b32 s68, s11, 2147483648 // (op 2112) 
s_cmp_eq_i32 s68, 0 // (op 2112) 
s_mov_b32 s69, scc // (op 2112) 

s_cmp_eq_i32 s69, 0 // (op 2112) 
s_cselect_b32 s68, s54, s55 // (op 2112) 


s_mul_hi_u32 s55, s68, s12 // (op 2112) 
s_sub_u32 s69, s68, s55 // (op 2112) 
s_lshr_b32 s70, s69, 1 // (op 2112) 

s_add_u32 s69, s70, s55 // (op 2112) 


s_lshr_b32 s55, s69, s10 // (op 2112) 

s_and_b32 s69, s10, 2147483648 // (op 2112) 
s_cmp_eq_i32 s69, 0 // (op 2112) 
s_mov_b32 s70, scc // (op 2112) 

s_cmp_eq_i32 s70, 0 // (op 2112) 
s_cselect_b32 s69, s68, s55 // (op 2112) 


v_lshl_add_u32 v24, s69, 5, v21 // (op 2112) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 2112) 
s_mul_i32 s55, s68, s70 // (op 2112) 


s_sub_u32 s68, s54, s55 // (op 2112) 


v_lshl_add_u32 v21, s68, 7, v23 // (op 2112) 


v_mov_b32 v27, 0 // (op 2112) convert
v_mov_b32 v26, v21 // (op 2112) convert

v_mul_lo_u32 v29, v24, s31 // (op 2112) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 2112) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 2112) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 2112) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v26 // (op 2112) least significant half
v_addc_co_u32 v29, s[54:55], v29, v27, s[54:55] // (op 2112) most significant half



s_mul_i32 s55, s30, s35 // (op 2114) most significant: low of low * high
s_mul_i32 s68, s31, s34 // (op 2114) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2114) least significant
s_mul_hi_u32 s69, s30, s34 // (op 2114) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 2114) most significant: sum
s_add_u32 s55, s55, s69 // (op 2114) most significant: sum


s_mov_b32 s68, s32 // (op 2114) reg expression
s_mov_b32 s69, s33 // (op 2114) reg expression
s_mov_b32 s70, s54 // (op 2114) reg expression
s_mov_b32 s71, 131072 // (op 2114) call()

v_lshrrev_b32 v21, 3, v1 // (op 2131) 
s_mul_i32 s54, s16, s8 // (op 2131) 
s_add_u32 s54, s54, s0 // (op 2131) 
v_and_b32 v23, -8, v1 // (op 2131) 
s_mul_hi_u32 s55, s54, s13 // (op 2131) 
v_sub_u32 v24, v1, v23 // (op 2131) 

v_lshlrev_b32 v23, 4, v24 // (op 2131) 

s_sub_u32 s72, s54, s55 // (op 2131) 
s_lshr_b32 s73, s72, 1 // (op 2131) 

s_add_u32 s72, s73, s55 // (op 2131) 


s_lshr_b32 s55, s72, s11 // (op 2131) 

s_and_b32 s72, s11, 2147483648 // (op 2131) 
s_cmp_eq_i32 s72, 0 // (op 2131) 
s_mov_b32 s73, scc // (op 2131) 

s_cmp_eq_i32 s73, 0 // (op 2131) 
s_cselect_b32 s72, s54, s55 // (op 2131) 


s_mul_hi_u32 s55, s72, s12 // (op 2131) 
s_sub_u32 s73, s72, s55 // (op 2131) 
s_lshr_b32 s74, s73, 1 // (op 2131) 

s_add_u32 s73, s74, s55 // (op 2131) 


s_lshr_b32 s55, s73, s10 // (op 2131) 

s_and_b32 s73, s10, 2147483648 // (op 2131) 
s_cmp_eq_i32 s73, 0 // (op 2131) 
s_mov_b32 s74, scc // (op 2131) 

s_cmp_eq_i32 s74, 0 // (op 2131) 
s_cselect_b32 s73, s72, s55 // (op 2131) 


v_lshl_add_u32 v24, s73, 5, v21 // (op 2131) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 2131) 
s_mul_i32 s55, s72, s74 // (op 2131) 


s_sub_u32 s72, s54, s55 // (op 2131) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 2131) 


v_mov_b32 v27, 0 // (op 2131) convert
v_mov_b32 v26, v21 // (op 2131) convert

v_mul_lo_u32 v31, v24, s31 // (op 2131) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 2131) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 2131) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 2131) most significant: sum

v_add_co_u32 v30, s[54:55], v30, v26 // (op 2131) least significant half
v_addc_co_u32 v31, s[54:55], v31, v27, s[54:55] // (op 2131) most significant half



s_mul_i32 s55, s30, s35 // (op 2133) most significant: low of low * high
s_mul_i32 s72, s31, s34 // (op 2133) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2133) least significant
s_mul_hi_u32 s73, s30, s34 // (op 2133) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 2133) most significant: sum
s_add_u32 s55, s55, s73 // (op 2133) most significant: sum


s_mov_b32 s68, s32 // (op 2133) reg expression
s_mov_b32 s69, s33 // (op 2133) reg expression
s_mov_b32 s70, s54 // (op 2133) reg expression
s_mov_b32 s71, 131072 // (op 2133) call()

v_and_b32 v21, -4, v1 // (op 2240) 
s_mul_i32 s54, s16, s8 // (op 2240) 
s_add_u32 s54, s54, s0 // (op 2240) 
v_sub_u32 v23, v1, v21 // (op 2240) 

s_mul_hi_u32 s55, s54, s13 // (op 2240) 
v_lshlrev_b32 v21, 4, v23 // (op 2240) 

s_sub_u32 s72, s54, s55 // (op 2240) 
v_lshrrev_b32 v23, 2, v1 // (op 2240) 
s_lshr_b32 s73, s72, 1 // (op 2240) 

s_add_u32 s72, s73, s55 // (op 2240) 


s_lshr_b32 s55, s72, s11 // (op 2240) 

s_and_b32 s72, s11, 2147483648 // (op 2240) 
s_cmp_eq_i32 s72, 0 // (op 2240) 
s_mov_b32 s73, scc // (op 2240) 

s_cmp_eq_i32 s73, 0 // (op 2240) 
s_cselect_b32 s72, s54, s55 // (op 2240) 


s_mul_hi_u32 s55, s72, s12 // (op 2240) 
s_sub_u32 s73, s72, s55 // (op 2240) 
s_lshr_b32 s74, s73, 1 // (op 2240) 

s_add_u32 s73, s74, s55 // (op 2240) 


s_lshr_b32 s55, s73, s10 // (op 2240) 

s_and_b32 s73, s10, 2147483648 // (op 2240) 
s_cmp_eq_i32 s73, 0 // (op 2240) 
s_mov_b32 s74, scc // (op 2240) 

s_cmp_eq_i32 s74, 0 // (op 2240) 
s_cselect_b32 s73, s72, s55 // (op 2240) 


s_mul_i32 s55, s73, s14 // (op 2240) 

s_sub_u32 s73, s72, s55 // (op 2240) 

v_lshl_add_u32 v24, s73, 6, v21 // (op 2240) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 2240) 
v_mov_b32 v27, 0 // (op 2240) convert
v_mov_b32 v26, v24 // (op 2240) convert

s_mul_i32 s55, s72, s74 // (op 2240) 


s_sub_u32 s72, s54, s55 // (op 2240) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 2240) 


v_mul_lo_u32 v25, v21, s37 // (op 2240) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 2240) most significant: high of low * low
v_mul_lo_u32 v24, v21, s36 // (op 2240) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2240) most significant: sum

v_add_co_u32 v24, s[54:55], v24, v26 // (op 2240) least significant half
v_addc_co_u32 v25, s[54:55], v25, v27, s[54:55] // (op 2240) most significant half



s_mul_i32 s55, s36, s3 // (op 2242) most significant: low of low * high
s_mul_i32 s72, s37, s2 // (op 2242) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2242) least significant
s_mul_hi_u32 s73, s36, s2 // (op 2242) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 2242) most significant: sum
s_add_u32 s55, s55, s73 // (op 2242) most significant: sum


s_mov_b32 s72, s38 // (op 2242) reg expression
s_mov_b32 s73, s39 // (op 2242) reg expression
s_mov_b32 s74, s54 // (op 2242) reg expression
s_mov_b32 s75, 131072 // (op 2242) call()

v_and_b32 v21, -4, v1 // (op 2259) 
s_mul_i32 s54, s16, s8 // (op 2259) 
s_add_u32 s54, s54, s0 // (op 2259) 
v_sub_u32 v23, v1, v21 // (op 2259) 

s_mul_hi_u32 s55, s54, s13 // (op 2259) 
v_lshlrev_b32 v21, 4, v23 // (op 2259) 

v_lshrrev_b32 v23, 2, v1 // (op 2259) 
s_sub_u32 s76, s54, s55 // (op 2259) 
s_lshr_b32 s77, s76, 1 // (op 2259) 

s_add_u32 s76, s77, s55 // (op 2259) 


s_lshr_b32 s55, s76, s11 // (op 2259) 

s_and_b32 s76, s11, 2147483648 // (op 2259) 
s_cmp_eq_i32 s76, 0 // (op 2259) 
s_mov_b32 s77, scc // (op 2259) 

s_cmp_eq_i32 s77, 0 // (op 2259) 
s_cselect_b32 s76, s54, s55 // (op 2259) 


s_mul_hi_u32 s55, s76, s12 // (op 2259) 
s_sub_u32 s77, s76, s55 // (op 2259) 
s_lshr_b32 s78, s77, 1 // (op 2259) 

s_add_u32 s77, s78, s55 // (op 2259) 


s_lshr_b32 s55, s77, s10 // (op 2259) 

s_and_b32 s77, s10, 2147483648 // (op 2259) 
s_cmp_eq_i32 s77, 0 // (op 2259) 
s_mov_b32 s78, scc // (op 2259) 

s_cmp_eq_i32 s78, 0 // (op 2259) 
s_cselect_b32 s77, s76, s55 // (op 2259) 


s_mul_i32 s55, s77, s14 // (op 2259) 

s_sub_u32 s77, s76, s55 // (op 2259) 

v_lshl_add_u32 v26, s77, 6, v21 // (op 2259) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 2259) 
v_mov_b32 v33, 0 // (op 2259) convert
v_mov_b32 v32, v26 // (op 2259) convert

s_mul_i32 s55, s76, s78 // (op 2259) 


s_sub_u32 s76, s54, s55 // (op 2259) 


v_lshl_add_u32 v21, s76, 7, v23 // (op 2259) 


v_mul_lo_u32 v27, v21, s37 // (op 2259) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 2259) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 2259) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 2259) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v32 // (op 2259) least significant half
v_addc_co_u32 v27, s[54:55], v27, v33, s[54:55] // (op 2259) most significant half



s_mul_i32 s55, s36, s3 // (op 2261) most significant: low of low * high
s_mul_i32 s76, s37, s2 // (op 2261) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2261) least significant
s_mul_hi_u32 s77, s36, s2 // (op 2261) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2261) most significant: sum
s_add_u32 s55, s55, s77 // (op 2261) most significant: sum


s_mov_b32 s72, s38 // (op 2261) reg expression
s_mov_b32 s73, s39 // (op 2261) reg expression
s_mov_b32 s74, s54 // (op 2261) reg expression
s_mov_b32 s75, 131072 // (op 2261) call()

v_lshrrev_b32 v21, 7, v1 // (op 2368) 
s_mul_i32 s54, s16, s8 // (op 2368) 
s_add_u32 s54, s54, s0 // (op 2368) 
v_and_b32 v23, 15, v1 // (op 2368) 
s_mul_hi_u32 s55, s54, s13 // (op 2368) 
v_lshl_add_u32 v32, v21, 4, v23 // (op 2368) 


s_sub_u32 s76, s54, s55 // (op 2368) 
v_and_b32 v21, 63, v1 // (op 2368) 
s_lshr_b32 s77, s76, 1 // (op 2368) 

v_lshrrev_b32 v23, 4, v21 // (op 2368) 

s_add_u32 s76, s77, s55 // (op 2368) 


s_lshr_b32 s55, s76, s11 // (op 2368) 

s_and_b32 s76, s11, 2147483648 // (op 2368) 
s_cmp_eq_i32 s76, 0 // (op 2368) 
s_mov_b32 s77, scc // (op 2368) 

s_cmp_eq_i32 s77, 0 // (op 2368) 
s_cselect_b32 s76, s54, s55 // (op 2368) 


s_mul_hi_u32 s55, s76, s12 // (op 2368) 
s_sub_u32 s77, s76, s55 // (op 2368) 
s_lshr_b32 s78, s77, 1 // (op 2368) 

s_add_u32 s77, s78, s55 // (op 2368) 


s_lshr_b32 s55, s77, s10 // (op 2368) 

s_and_b32 s77, s10, 2147483648 // (op 2368) 
s_cmp_eq_i32 s77, 0 // (op 2368) 
s_mov_b32 s78, scc // (op 2368) 

s_cmp_eq_i32 s78, 0 // (op 2368) 
s_cselect_b32 s77, s76, s55 // (op 2368) 


v_lshl_add_u32 v21, s77, 5, v32 // (op 2368) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 2368) 
s_mul_i32 s55, s76, s78 // (op 2368) 


s_sub_u32 s76, s54, s55 // (op 2368) 


v_lshl_add_u32 v32, s76, 2, v23 // (op 2368) 


v_mul_lo_u32 v23, v21, s40 // (op 2368) 
v_add_u32 v23, v23, v32 // (op 2368) 


s_mul_i32 s55, s40, s35 // (op 2370) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 2370) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2370) least significant
s_mul_hi_u32 s77, s40, s34 // (op 2370) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2370) most significant: sum
s_add_u32 s55, s55, s77 // (op 2370) most significant: sum


s_mov_b32 s76, s42 // (op 2370) reg expression
s_mov_b32 s77, s43 // (op 2370) reg expression
s_mov_b32 s78, s54 // (op 2370) reg expression
s_mov_b32 s79, 131072 // (op 2370) call()

v_lshrrev_b32 v21, 7, v1 // (op 2387) 
s_mul_i32 s54, s16, s8 // (op 2387) 
s_add_u32 s54, s54, s0 // (op 2387) 
v_and_b32 v32, 15, v1 // (op 2387) 
s_mul_hi_u32 s55, s54, s13 // (op 2387) 
v_lshl_add_u32 v33, v21, 4, v32 // (op 2387) 


v_and_b32 v21, 63, v1 // (op 2387) 
s_sub_u32 s80, s54, s55 // (op 2387) 
v_lshrrev_b32 v32, 4, v21 // (op 2387) 

s_lshr_b32 s81, s80, 1 // (op 2387) 

s_add_u32 s80, s81, s55 // (op 2387) 


s_lshr_b32 s55, s80, s11 // (op 2387) 

s_and_b32 s80, s11, 2147483648 // (op 2387) 
s_cmp_eq_i32 s80, 0 // (op 2387) 
s_mov_b32 s81, scc // (op 2387) 

s_cmp_eq_i32 s81, 0 // (op 2387) 
s_cselect_b32 s80, s54, s55 // (op 2387) 


s_mul_hi_u32 s55, s80, s12 // (op 2387) 
s_sub_u32 s81, s80, s55 // (op 2387) 
s_lshr_b32 s82, s81, 1 // (op 2387) 

s_add_u32 s81, s82, s55 // (op 2387) 


s_lshr_b32 s55, s81, s10 // (op 2387) 

s_and_b32 s81, s10, 2147483648 // (op 2387) 
s_cmp_eq_i32 s81, 0 // (op 2387) 
s_mov_b32 s82, scc // (op 2387) 

s_cmp_eq_i32 s82, 0 // (op 2387) 
s_cselect_b32 s81, s80, s55 // (op 2387) 


v_lshl_add_u32 v21, s81, 5, v33 // (op 2387) 


s_ashr_i64 s[82:83], s[2:3], 7 // (op 2387) 
s_mul_i32 s55, s80, s82 // (op 2387) 


s_sub_u32 s80, s54, s55 // (op 2387) 


v_lshl_add_u32 v33, s80, 2, v32 // (op 2387) 


v_mul_lo_u32 v32, v21, s40 // (op 2387) 
v_add_u32 v32, v32, v33 // (op 2387) 


s_mul_i32 s55, s40, s35 // (op 2389) most significant: low of low * high
s_mul_i32 s80, s41, s34 // (op 2389) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2389) least significant
s_mul_hi_u32 s81, s40, s34 // (op 2389) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 2389) most significant: sum
s_add_u32 s55, s55, s81 // (op 2389) most significant: sum


s_mov_b32 s76, s42 // (op 2389) reg expression
s_mov_b32 s77, s43 // (op 2389) reg expression
s_mov_b32 s78, s54 // (op 2389) reg expression
s_mov_b32 s79, 131072 // (op 2389) call()

v_and_b32 v21, -64, v1 // (op 2496) 
s_mul_i32 s54, s16, s8 // (op 2496) 
s_add_u32 s54, s54, s0 // (op 2496) 
v_sub_u32 v33, v1, v21 // (op 2496) 

s_mul_hi_u32 s55, s54, s13 // (op 2496) 
v_lshrrev_b32 v21, 6, v1 // (op 2496) 
s_sub_u32 s80, s54, s55 // (op 2496) 
s_lshr_b32 s81, s80, 1 // (op 2496) 

s_add_u32 s80, s81, s55 // (op 2496) 


s_lshr_b32 s55, s80, s11 // (op 2496) 

s_and_b32 s80, s11, 2147483648 // (op 2496) 
s_cmp_eq_i32 s80, 0 // (op 2496) 
s_mov_b32 s81, scc // (op 2496) 

s_cmp_eq_i32 s81, 0 // (op 2496) 
s_cselect_b32 s80, s54, s55 // (op 2496) 


s_mul_hi_u32 s55, s80, s12 // (op 2496) 
s_sub_u32 s81, s80, s55 // (op 2496) 
s_lshr_b32 s82, s81, 1 // (op 2496) 

s_add_u32 s81, s82, s55 // (op 2496) 


s_lshr_b32 s55, s81, s10 // (op 2496) 

s_and_b32 s81, s10, 2147483648 // (op 2496) 
s_cmp_eq_i32 s81, 0 // (op 2496) 
s_mov_b32 s82, scc // (op 2496) 

s_cmp_eq_i32 s82, 0 // (op 2496) 
s_cselect_b32 s81, s80, s55 // (op 2496) 


s_mul_i32 s55, s81, s14 // (op 2496) 

s_sub_u32 s81, s80, s55 // (op 2496) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 2496) 
v_lshl_add_u32 v34, s81, 6, v33 // (op 2496) 


s_mul_i32 s55, s80, s82 // (op 2496) 


s_sub_u32 s80, s54, s55 // (op 2496) 


v_lshl_add_u32 v33, s80, 2, v21 // (op 2496) 


v_mul_lo_u32 v21, v33, s44 // (op 2496) 
v_add_u32 v21, v21, v34 // (op 2496) 


s_mul_i32 s55, s44, s49 // (op 2498) most significant: low of low * high
s_mul_i32 s80, s45, s48 // (op 2498) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2498) least significant
s_mul_hi_u32 s81, s44, s48 // (op 2498) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 2498) most significant: sum
s_add_u32 s55, s55, s81 // (op 2498) most significant: sum


s_mov_b32 s80, s46 // (op 2498) reg expression
s_mov_b32 s81, s47 // (op 2498) reg expression
s_mov_b32 s82, s54 // (op 2498) reg expression
s_mov_b32 s83, 131072 // (op 2498) call()

v_and_b32 v33, -64, v1 // (op 2515) 
s_mul_i32 s54, s16, s8 // (op 2515) 
s_add_u32 s54, s54, s0 // (op 2515) 
v_sub_u32 v34, v1, v33 // (op 2515) 

s_mul_hi_u32 s55, s54, s13 // (op 2515) 
v_lshrrev_b32 v33, 6, v1 // (op 2515) 
s_sub_u32 s84, s54, s55 // (op 2515) 
s_lshr_b32 s85, s84, 1 // (op 2515) 

s_add_u32 s84, s85, s55 // (op 2515) 


s_lshr_b32 s55, s84, s11 // (op 2515) 

s_and_b32 s84, s11, 2147483648 // (op 2515) 
s_cmp_eq_i32 s84, 0 // (op 2515) 
s_mov_b32 s85, scc // (op 2515) 

s_cmp_eq_i32 s85, 0 // (op 2515) 
s_cselect_b32 s84, s54, s55 // (op 2515) 


s_mul_hi_u32 s55, s84, s12 // (op 2515) 
s_sub_u32 s85, s84, s55 // (op 2515) 
s_lshr_b32 s86, s85, 1 // (op 2515) 

s_add_u32 s85, s86, s55 // (op 2515) 


s_lshr_b32 s55, s85, s10 // (op 2515) 

s_and_b32 s85, s10, 2147483648 // (op 2515) 
s_cmp_eq_i32 s85, 0 // (op 2515) 
s_mov_b32 s86, scc // (op 2515) 

s_cmp_eq_i32 s86, 0 // (op 2515) 
s_cselect_b32 s85, s84, s55 // (op 2515) 


s_mul_i32 s55, s85, s14 // (op 2515) 

s_sub_u32 s85, s84, s55 // (op 2515) 

v_lshl_add_u32 v35, s85, 6, v34 // (op 2515) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2515) 
s_mul_i32 s55, s84, s86 // (op 2515) 


s_sub_u32 s84, s54, s55 // (op 2515) 


v_lshl_add_u32 v34, s84, 2, v33 // (op 2515) 


v_mul_lo_u32 v33, v34, s44 // (op 2515) 
v_add_u32 v33, v33, v35 // (op 2515) 


s_mul_i32 s55, s44, s49 // (op 2517) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 2517) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2517) least significant
s_mul_hi_u32 s85, s44, s48 // (op 2517) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2517) most significant: sum
s_add_u32 s55, s55, s85 // (op 2517) most significant: sum


s_mov_b32 s80, s46 // (op 2517) reg expression
s_mov_b32 s81, s47 // (op 2517) reg expression
s_mov_b32 s82, s54 // (op 2517) reg expression
s_mov_b32 s83, 131072 // (op 2517) call()




buffer_load_ubyte v34, v23, s[76:79], 0 offen offset: 0 // (op 1801) Load value
v_and_b32 v35, -4, v1 // (op 2672) 
v_sub_u32 v36, v1, v35 // (op 2672) 

v_lshlrev_b32 v35, 4, v36 // (op 2672) 

v_lshrrev_b32 v36, 2, v1 // (op 2672) 
v_lshl_add_u32 v37, v36, 6, v35 // (op 2672) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 2672) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3407) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 1990) 
v_mov_b64 v[36:37], s[84:85] // (op 1990) convert


v_mov_b64 v[38:39], v[24:25] // (op 1990) reg expression
s_mov_b32 m0, s54 // (op 1990) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 1990) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 1990) most significant half

s_add_u32 m0, m0, 4096 // (op 1990) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1990) Load value direct to lds





v_mov_b32 v35, v21 // (op 1808) reg expression

buffer_load_ubyte v36, v35, s[80:83], 0 offen offset: 0 // (op 1808) Load value

v_and_b32 v35, -8, v1 // (op 2607) 
v_sub_u32 v37, v1, v35 // (op 2607) 

v_lshlrev_b32 v35, 4, v37 // (op 2607) 

v_lshrrev_b32 v37, 3, v1 // (op 2607) 
v_lshl_add_u32 v38, v37, 7, v35 // (op 2607) 


s_nop 0
v_readfirstlane_b32 s54, v38 // (op 2607) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3388) 


s_nop 0
s_add_u32 s55, 8192, s54 // (op 1995) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 1995) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 1995) Load value direct to lds





buffer_load_ubyte v35, v23, s[76:79], 0 offen offset:4 // (op 1815) Load value

v_and_b32 v23, -4, v1 // (op 3261) 
v_sub_u32 v37, v1, v23 // (op 3261) 

v_lshlrev_b32 v23, 4, v37 // (op 3261) 

v_lshrrev_b32 v37, 2, v1 // (op 3261) 
v_lshl_add_u32 v38, v37, 6, v23 // (op 3261) 


s_nop 0
v_readfirstlane_b32 s54, v38 // (op 3261) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3537) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 2000) 
v_mov_b64 v[38:39], s[84:85] // (op 2000) convert


s_lshl_b64 s[84:85], s[36:37], 7 // (op 2000) 
v_mov_b32 v23, s84 // (op 2000) 
v_mov_b32 v37, s85 // (op 2000) 
v_add_co_u32 v40, s[86:87], v24, v23 // (op 2000) least significant half
v_addc_co_u32 v41, s[86:87], v25, v37, s[86:87] // (op 2000) most significant half




s_add_u32 s55, 12288, s54 // (op 2000) 
s_mov_b32 m0, s55 // (op 2000) reg expression
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds
v_add_co_u32 v40, s[84:85], v40, v38 // (op 2000) least significant half
v_addc_co_u32 v41, s[84:85], 0, v41, s[84:85] // (op 2000) most significant half

s_add_u32 m0, m0, 4096 // (op 2000) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds







s_lshl_b32 s54, s44, 2 // (op 1822) 
v_add_u32 v23, v21, s54 // (op 1822) 


buffer_load_ubyte v24, v23, s[80:83], 0 offen offset: 0 // (op 1822) Load value


v_and_b32 v21, -8, v1 // (op 3222) 
v_sub_u32 v23, v1, v21 // (op 3222) 

v_lshlrev_b32 v21, 4, v23 // (op 3222) 

v_lshrrev_b32 v23, 3, v1 // (op 3222) 
v_lshl_add_u32 v25, v23, 7, v21 // (op 3222) 


s_nop 0
v_readfirstlane_b32 s54, v25 // (op 3222) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3528) 


s_nop 0
s_add_u32 s55, 20480, s54 // (op 2002) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 2002) reg expression
s_mov_b32 s84, 128 // (op 2002) call()
buffer_load_dwordx4 v28, s[68:71], s84 offen offset: 0 lds // (op 2002) Load value direct to lds




v_and_b32 v21, -64, v1 // (op 2748) 
v_sub_u32 v23, v1, v21 // (op 2748) 

v_and_b32 v21, 4294967232, v1 // (op 2748) 
v_add_u32 v25, v21, v23 // (op 2748) 


s_barrier  // (op 3426) 



s_waitcnt vmcnt(6)
ds_write_b8 v25, v36 offset:24576 // (op 1828) Store local data 

s_barrier  // (op 3404) 



ds_read_b64_tr_b8 v[38:39], v19 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v19 offset:512 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v19 offset:4096 // (op 1846) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v19 offset:4608 // (op 1846) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3423) 


ds_read_u8 v21, v16 offset:24576 // (op 1856) Load local data 



ds_read_b64_tr_b8 v[46:47], v19 offset:32 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v19 offset:544 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v19 offset:4128 // (op 1851) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v19 offset:4640 // (op 1851) Transpose load local data 


ds_read_u8 v23, v16 offset:24608 // (op 1861) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3385) 



ds_read_b128 v[54:57], v17 offset:8192 // (op 1842) Load local data 
ds_read_b128 v[58:61], v17 offset:8256 // (op 1842) Load local data 
v_and_b32 v25, -8, v1 // (op 2579) 
v_sub_u32 v37, v1, v25 // (op 2579) 

v_lshlrev_b32 v25, 4, v37 // (op 2579) 

v_lshrrev_b32 v37, 3, v1 // (op 2579) 
v_lshl_add_u32 v28, v37, 7, v25 // (op 2579) 


s_nop 0
v_readfirstlane_b32 s54, v28 // (op 2579) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 2644) 
v_sub_u32 v37, v1, v25 // (op 2644) 

v_lshlrev_b32 v25, 4, v37 // (op 2644) 

v_lshrrev_b32 v37, 2, v1 // (op 2644) 
v_lshl_add_u32 v28, v37, 6, v25 // (op 2644) 


s_nop 0
v_readfirstlane_b32 s55, v28 // (op 2644) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 2719) 
v_sub_u32 v37, v1, v25 // (op 2719) 

v_and_b32 v25, 4294967232, v1 // (op 2719) 
v_add_u32 v28, v25, v37 // (op 2719) 


v_and_b32 v25, -8, v1 // (op 3207) 
v_sub_u32 v29, v1, v25 // (op 3207) 

v_lshlrev_b32 v25, 4, v29 // (op 3207) 

v_lshrrev_b32 v29, 3, v1 // (op 3207) 
v_lshl_add_u32 v37, v29, 7, v25 // (op 3207) 


s_nop 0
v_readfirstlane_b32 s84, v37 // (op 3207) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 3246) 
v_sub_u32 v29, v1, v25 // (op 3246) 

v_lshlrev_b32 v25, 4, v29 // (op 3246) 

v_lshrrev_b32 v29, 2, v1 // (op 3246) 
v_lshl_add_u32 v37, v29, 6, v25 // (op 3246) 


s_nop 0
v_readfirstlane_b32 s85, v37 // (op 3246) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 3290) 
v_sub_u32 v29, v1, v25 // (op 3290) 

v_and_b32 v25, 4294967232, v1 // (op 3290) 
v_add_u32 v37, v25, v29 // (op 3290) 


s_mov_b32 s1, 0 // (op 973) call()
s_and_b32 s86, s7, 4294967294 // (op 972) 
s_cmp_lt_u32 s1, s86 // (op 972) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_972 // (op 972) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_972 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_972:
s_add_u32 s1, s1, 2 // (op 974) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v34, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[46:53], a[4:7], v34, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1008) 


v_mov_b32 v25, v32 // (op 75) 

buffer_load_ubyte v34, v25, s[76:79], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 3410) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1915) 
v_mov_b64 v[62:63], s[86:87] // (op 1915) convert


v_mov_b64 v[64:65], v[26:27] // (op 1915) 
s_lshl_b64 s[86:87], s[36:37], 7 // (op 1915) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 1915) 

v_mov_b32 v29, s88 // (op 1915) 
v_mov_b32 v66, s89 // (op 1915) 
v_add_co_u32 v68, s[86:87], v64, v29 // (op 1915) least significant half
v_addc_co_u32 v69, s[86:87], v65, v66, s[86:87] // (op 1915) most significant half




s_mov_b32 m0, s55 // (op 1915) reg expression
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds
v_add_co_u32 v68, s[86:87], v68, v62 // (op 1915) least significant half
v_addc_co_u32 v69, s[86:87], 0, v69, s[86:87] // (op 1915) most significant half

s_add_u32 m0, m0, 4096 // (op 1915) 
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 1915) Load value direct to lds




v_mov_b32 v29, v33 // (op 21) 
s_lshl_b32 s86, s44, 3 // (op 21) 
v_add_u32 v62, v29, s86 // (op 21) 


buffer_load_ubyte v36, v62, s[80:83], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 3391) 


v_mov_b64 v[62:63], v[30:31] // (op 1910) 
s_add_u32 s86, 8192, s54 // (op 1910) 

s_mov_b32 m0, s86 // (op 1910) reg expression
s_mov_b32 s87, 256 // (op 1910) call()
buffer_load_dwordx4 v62, s[68:71], s87 offen offset: 0 lds // (op 1910) Load value direct to lds


s_barrier  // (op 3546) 



s_waitcnt vmcnt(6)
ds_write_b8 v37, v24 offset:24832 // (op 901) Store local data 
s_barrier  // (op 3534) 



ds_read_b64_tr_b8 v[38:39], v13 offset:12288 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:12800 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v13 offset:16384 // (op 894) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:16896 // (op 894) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3543) 


ds_read_u8 v21, v12 offset:24832 // (op 898) Load local data 



ds_read_b64_tr_b8 v[46:47], v13 offset:12320 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:12832 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v13 offset:16416 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:16928 // (op 1013) Transpose load local data 


ds_read_u8 v23, v12 offset:24864 // (op 1016) Load local data 
s_barrier  // (op 3689) 
s_waitcnt vmcnt(5)
s_barrier  // (op 3525) 



ds_read_b128 v[54:57], v6 offset:20480 // (op 892) Load local data 
ds_read_b128 v[58:61], v6 offset:20544 // (op 892) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v35, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 899) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[46:53], a[4:7], v35, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1009) 



buffer_load_ubyte v35, v25, s[76:79], 0 offen offset:12 // (op 896) Load value

s_barrier  // (op 3540) 
v_add_u32 v32, 8, v32 // (op 2379) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1971) 
v_mov_b64 v[66:67], s[86:87] // (op 1971) convert


s_lshl_b64 s[86:87], s[36:37], 7 // (op 1971) 
s_mul_i32 s89, 3, s87 // (op 1971) most significant: low of low * high
s_mul_i32 s88, 3, s86 // (op 1971) least significant
s_mul_hi_u32 s90, 3, s86 // (op 1971) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 1971) most significant: sum


v_mov_b32 v25, s88 // (op 1971) 
v_mov_b32 v68, s89 // (op 1971) 
v_add_co_u32 v70, s[86:87], v64, v25 // (op 1971) least significant half
v_addc_co_u32 v71, s[86:87], v65, v68, s[86:87] // (op 1971) most significant half




s_add_u32 s86, 12288, s85 // (op 1971) 
s_nop 0
s_mov_b32 m0, s86 // (op 1971) reg expression
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds
v_add_co_u32 v70, s[88:89], v70, v66 // (op 1971) least significant half
v_addc_co_u32 v71, s[88:89], 0, v71, s[88:89] // (op 1971) most significant half

s_add_u32 m0, m0, 4096 // (op 1971) 
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1971) Load value direct to lds






s_lshl_b32 s86, s44, 2 // (op 903) 
s_mul_i32 s87, 3, s86 // (op 903) 

v_add_u32 v25, v29, s87 // (op 903) 


s_lshl_b64 s[86:87], s[36:37], 7 // (op 2251) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 2251) 

v_mov_b32 v64, s88 // (op 2251) 
v_mov_b32 v65, s89 // (op 2251) 
v_add_co_u32 v26, s[86:87], v26, v64 // (op 2251) least significant half
v_addc_co_u32 v27, s[86:87], v27, v65, s[86:87] // (op 2251) most significant half




buffer_load_ubyte v24, v25, s[80:83], 0 offen offset: 0 // (op 903) Load value


s_barrier  // (op 3531) 
s_lshl_b32 s86, s44, 3 // (op 2507) 


v_add_u32 v33, v33, s86 // (op 2507) 

s_add_u32 s86, 20480, s84 // (op 1966) 

s_mov_b32 m0, s86 // (op 1966) reg expression
s_mov_b32 s87, 384 // (op 1966) call()
buffer_load_dwordx4 v62, s[68:71], s87 offen offset: 0 lds // (op 1966) Load value direct to lds



s_barrier  // (op 3429) 
v_add_co_u32 v30, vcc, 256, v30 // (op 2123) least significant half



v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 2123) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v28, v36 offset:24576 // (op 26) Store local data 
s_barrier  // (op 3416) 



ds_read_b64_tr_b8 v[38:39], v7 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:4608 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3436) 


ds_read_u8 v21, v0 offset:24576 // (op 79) Load local data 



ds_read_b64_tr_b8 v[46:47], v7 offset:32 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:544 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4128 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:4640 // (op 1004) Transpose load local data 


ds_read_u8 v23, v0 offset:24608 // (op 1007) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3397) 



ds_read_b128 v[54:57], v3 offset:8192 // (op 65) Load local data 
ds_read_b128 v[58:61], v3 offset:8256 // (op 65) Load local data 
s_and_b32 s86, s7, 4294967294 // (op 972) 
s_cmp_lt_u32 s1, s86 // (op 972) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_972 // (op 972) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_972 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_972:












s_barrier  // (op 3673) 
s_and_b32 s54, s7, 4294967294 // (op 2039) 
v_lshrrev_b32 v25, 3, v1 // (op 2039) 
s_add_u32 s55, s0, s54 // (op 2039) 

v_and_b32 v26, -8, v1 // (op 2039) 
v_sub_u32 v27, v1, v26 // (op 2039) 

s_mul_i32 s54, s16, s8 // (op 2039) 
s_add_u32 s54, s54, s55 // (op 2039) 

v_lshlrev_b32 v26, 4, v27 // (op 2039) 

s_mul_hi_u32 s55, s54, s13 // (op 2039) 
s_sub_u32 s84, s54, s55 // (op 2039) 
s_lshr_b32 s85, s84, 1 // (op 2039) 

s_add_u32 s84, s85, s55 // (op 2039) 


s_lshr_b32 s55, s84, s11 // (op 2039) 

s_and_b32 s84, s11, 2147483648 // (op 2039) 
s_cmp_eq_i32 s84, 0 // (op 2039) 
s_mov_b32 s85, scc // (op 2039) 

s_cmp_eq_i32 s85, 0 // (op 2039) 
s_cselect_b32 s84, s54, s55 // (op 2039) 


s_mul_hi_u32 s55, s84, s12 // (op 2039) 
s_sub_u32 s85, s84, s55 // (op 2039) 
s_lshr_b32 s86, s85, 1 // (op 2039) 

s_add_u32 s85, s86, s55 // (op 2039) 


s_lshr_b32 s55, s85, s10 // (op 2039) 

s_and_b32 s85, s10, 2147483648 // (op 2039) 
s_cmp_eq_i32 s85, 0 // (op 2039) 
s_mov_b32 s86, scc // (op 2039) 

s_cmp_eq_i32 s86, 0 // (op 2039) 
s_cselect_b32 s85, s84, s55 // (op 2039) 


v_lshl_add_u32 v27, s85, 5, v25 // (op 2039) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2039) 
s_mul_i32 s55, s84, s86 // (op 2039) 


s_sub_u32 s84, s54, s55 // (op 2039) 


v_lshl_add_u32 v25, s84, 7, v26 // (op 2039) 


v_mov_b32 v37, 0 // (op 2039) convert
v_mov_b32 v36, v25 // (op 2039) convert

v_mul_lo_u32 v29, v27, s31 // (op 2039) most significant: low of low * high
v_mul_hi_u32 v25, v27, s30 // (op 2039) most significant: high of low * low
v_mul_lo_u32 v28, v27, s30 // (op 2039) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 2039) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v36 // (op 2039) least significant half
v_addc_co_u32 v29, s[54:55], v29, v37, s[54:55] // (op 2039) most significant half



s_mul_i32 s55, s30, s35 // (op 2041) most significant: low of low * high
s_mul_i32 s84, s31, s34 // (op 2041) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2041) least significant
s_mul_hi_u32 s85, s30, s34 // (op 2041) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2041) most significant: sum
s_add_u32 s55, s55, s85 // (op 2041) most significant: sum


s_mov_b32 s68, s32 // (op 2041) reg expression
s_mov_b32 s69, s33 // (op 2041) reg expression
s_mov_b32 s70, s54 // (op 2041) reg expression
s_mov_b32 s71, 131072 // (op 2041) call()

s_and_b32 s54, s7, 4294967294 // (op 2175) 
v_and_b32 v25, -4, v1 // (op 2175) 
s_add_u32 s55, s0, s54 // (op 2175) 

v_sub_u32 v26, v1, v25 // (op 2175) 

v_lshlrev_b32 v25, 4, v26 // (op 2175) 

s_mul_i32 s54, s16, s8 // (op 2175) 
s_add_u32 s54, s54, s55 // (op 2175) 

v_lshrrev_b32 v26, 2, v1 // (op 2175) 
s_mul_hi_u32 s55, s54, s13 // (op 2175) 
s_sub_u32 s84, s54, s55 // (op 2175) 
s_lshr_b32 s85, s84, 1 // (op 2175) 

s_add_u32 s84, s85, s55 // (op 2175) 


s_lshr_b32 s55, s84, s11 // (op 2175) 

s_and_b32 s84, s11, 2147483648 // (op 2175) 
s_cmp_eq_i32 s84, 0 // (op 2175) 
s_mov_b32 s85, scc // (op 2175) 

s_cmp_eq_i32 s85, 0 // (op 2175) 
s_cselect_b32 s84, s54, s55 // (op 2175) 


s_mul_hi_u32 s55, s84, s12 // (op 2175) 
s_sub_u32 s85, s84, s55 // (op 2175) 
s_lshr_b32 s86, s85, 1 // (op 2175) 

s_add_u32 s85, s86, s55 // (op 2175) 


s_lshr_b32 s55, s85, s10 // (op 2175) 

s_and_b32 s85, s10, 2147483648 // (op 2175) 
s_cmp_eq_i32 s85, 0 // (op 2175) 
s_mov_b32 s86, scc // (op 2175) 

s_cmp_eq_i32 s86, 0 // (op 2175) 
s_cselect_b32 s85, s84, s55 // (op 2175) 


s_mul_i32 s55, s85, s14 // (op 2175) 

s_sub_u32 s85, s84, s55 // (op 2175) 

v_lshl_add_u32 v27, s85, 6, v25 // (op 2175) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2175) 
v_mov_b32 v37, 0 // (op 2175) convert
v_mov_b32 v36, v27 // (op 2175) convert

s_mul_i32 s55, s84, s86 // (op 2175) 


s_sub_u32 s84, s54, s55 // (op 2175) 


v_lshl_add_u32 v25, s84, 7, v26 // (op 2175) 


v_mul_lo_u32 v27, v25, s37 // (op 2175) most significant: low of low * high
v_mul_hi_u32 v30, v25, s36 // (op 2175) most significant: high of low * low
v_mul_lo_u32 v26, v25, s36 // (op 2175) least significant: low of low * low
v_add_u32 v27, v27, v30 // (op 2175) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v36 // (op 2175) least significant half
v_addc_co_u32 v27, s[54:55], v27, v37, s[54:55] // (op 2175) most significant half



s_mul_i32 s55, s36, s3 // (op 2177) most significant: low of low * high
s_mul_i32 s84, s37, s2 // (op 2177) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2177) least significant
s_mul_hi_u32 s85, s36, s2 // (op 2177) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2177) most significant: sum
s_add_u32 s55, s55, s85 // (op 2177) most significant: sum


s_mov_b32 s72, s38 // (op 2177) reg expression
s_mov_b32 s73, s39 // (op 2177) reg expression
s_mov_b32 s74, s54 // (op 2177) reg expression
s_mov_b32 s75, 131072 // (op 2177) call()

s_and_b32 s54, s7, 4294967294 // (op 2303) 
v_lshrrev_b32 v25, 7, v1 // (op 2303) 
s_add_u32 s55, s0, s54 // (op 2303) 

v_and_b32 v30, 15, v1 // (op 2303) 
v_lshl_add_u32 v31, v25, 4, v30 // (op 2303) 


s_mul_i32 s54, s16, s8 // (op 2303) 
s_add_u32 s54, s54, s55 // (op 2303) 

v_and_b32 v25, 63, v1 // (op 2303) 
s_mul_hi_u32 s55, s54, s13 // (op 2303) 
v_lshrrev_b32 v30, 4, v25 // (op 2303) 

s_sub_u32 s84, s54, s55 // (op 2303) 
s_lshr_b32 s85, s84, 1 // (op 2303) 

s_add_u32 s84, s85, s55 // (op 2303) 


s_lshr_b32 s55, s84, s11 // (op 2303) 

s_and_b32 s84, s11, 2147483648 // (op 2303) 
s_cmp_eq_i32 s84, 0 // (op 2303) 
s_mov_b32 s85, scc // (op 2303) 

s_cmp_eq_i32 s85, 0 // (op 2303) 
s_cselect_b32 s84, s54, s55 // (op 2303) 


s_mul_hi_u32 s55, s84, s12 // (op 2303) 
s_sub_u32 s85, s84, s55 // (op 2303) 
s_lshr_b32 s86, s85, 1 // (op 2303) 

s_add_u32 s85, s86, s55 // (op 2303) 


s_lshr_b32 s55, s85, s10 // (op 2303) 

s_and_b32 s85, s10, 2147483648 // (op 2303) 
s_cmp_eq_i32 s85, 0 // (op 2303) 
s_mov_b32 s86, scc // (op 2303) 

s_cmp_eq_i32 s86, 0 // (op 2303) 
s_cselect_b32 s85, s84, s55 // (op 2303) 


v_lshl_add_u32 v25, s85, 5, v31 // (op 2303) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2303) 
s_mul_i32 s55, s84, s86 // (op 2303) 


s_sub_u32 s84, s54, s55 // (op 2303) 


v_lshl_add_u32 v31, s84, 2, v30 // (op 2303) 


v_mul_lo_u32 v30, v25, s40 // (op 2303) 
v_add_u32 v30, v30, v31 // (op 2303) 


s_mul_i32 s55, s40, s35 // (op 2305) most significant: low of low * high
s_mul_i32 s84, s41, s34 // (op 2305) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2305) least significant
s_mul_hi_u32 s85, s40, s34 // (op 2305) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2305) most significant: sum
s_add_u32 s55, s55, s85 // (op 2305) most significant: sum


s_mov_b32 s76, s42 // (op 2305) reg expression
s_mov_b32 s77, s43 // (op 2305) reg expression
s_mov_b32 s78, s54 // (op 2305) reg expression
s_mov_b32 s79, 131072 // (op 2305) call()

s_and_b32 s54, s7, 4294967294 // (op 2431) 
v_and_b32 v25, -64, v1 // (op 2431) 
s_add_u32 s55, s0, s54 // (op 2431) 

v_sub_u32 v31, v1, v25 // (op 2431) 

v_lshrrev_b32 v25, 6, v1 // (op 2431) 
s_mul_i32 s54, s16, s8 // (op 2431) 
s_add_u32 s54, s54, s55 // (op 2431) 

s_mul_hi_u32 s55, s54, s13 // (op 2431) 
s_sub_u32 s84, s54, s55 // (op 2431) 
s_lshr_b32 s85, s84, 1 // (op 2431) 

s_add_u32 s84, s85, s55 // (op 2431) 


s_lshr_b32 s55, s84, s11 // (op 2431) 

s_and_b32 s84, s11, 2147483648 // (op 2431) 
s_cmp_eq_i32 s84, 0 // (op 2431) 
s_mov_b32 s85, scc // (op 2431) 

s_cmp_eq_i32 s85, 0 // (op 2431) 
s_cselect_b32 s84, s54, s55 // (op 2431) 


s_mul_hi_u32 s55, s84, s12 // (op 2431) 
s_sub_u32 s85, s84, s55 // (op 2431) 
s_lshr_b32 s86, s85, 1 // (op 2431) 

s_add_u32 s85, s86, s55 // (op 2431) 


s_lshr_b32 s55, s85, s10 // (op 2431) 

s_and_b32 s85, s10, 2147483648 // (op 2431) 
s_cmp_eq_i32 s85, 0 // (op 2431) 
s_mov_b32 s86, scc // (op 2431) 

s_cmp_eq_i32 s86, 0 // (op 2431) 
s_cselect_b32 s85, s84, s55 // (op 2431) 


s_mul_i32 s55, s85, s14 // (op 2431) 

s_sub_u32 s85, s84, s55 // (op 2431) 

v_lshl_add_u32 v32, s85, 6, v31 // (op 2431) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2431) 
s_mul_i32 s55, s84, s86 // (op 2431) 


s_sub_u32 s84, s54, s55 // (op 2431) 


v_lshl_add_u32 v31, s84, 2, v25 // (op 2431) 


v_mul_lo_u32 v25, v31, s44 // (op 2431) 
v_add_u32 v25, v25, v32 // (op 2431) 


s_mul_i32 s55, s44, s49 // (op 2433) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 2433) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2433) least significant
s_mul_hi_u32 s85, s44, s48 // (op 2433) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2433) most significant: sum
s_add_u32 s55, s55, s85 // (op 2433) most significant: sum


s_mov_b32 s80, s46 // (op 2433) reg expression
s_mov_b32 s81, s47 // (op 2433) reg expression
s_mov_b32 s82, s54 // (op 2433) reg expression
s_mov_b32 s83, 131072 // (op 2433) call()

v_and_b32 v31, -8, v1 // (op 2592) 
v_sub_u32 v32, v1, v31 // (op 2592) 

v_lshlrev_b32 v31, 4, v32 // (op 2592) 

v_lshrrev_b32 v32, 3, v1 // (op 2592) 
v_lshl_add_u32 v33, v32, 7, v31 // (op 2592) 


s_nop 0
v_readfirstlane_b32 s54, v33 // (op 2592) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -4, v1 // (op 2657) 
v_sub_u32 v32, v1, v31 // (op 2657) 

v_lshlrev_b32 v31, 4, v32 // (op 2657) 

v_lshrrev_b32 v32, 2, v1 // (op 2657) 
v_lshl_add_u32 v33, v32, 6, v31 // (op 2657) 


s_nop 0
v_readfirstlane_b32 s55, v33 // (op 2657) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -64, v1 // (op 2732) 
v_sub_u32 v32, v1, v31 // (op 2732) 

v_and_b32 v31, 4294967232, v1 // (op 2732) 
v_add_u32 v33, v31, v32 // (op 2732) 


s_and_b32 s84, s7, 4294967294 // (op 978) 
s_cmp_lt_u32 s84, s7 // (op 977) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_977 // (op 977) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_977 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_977:
s_barrier  // (op 3394) 
s_add_u32 s84, s84, 1 // (op 979) 


v_mov_b64 v[34:35], v[28:29] // (op 1953) 
s_add_u32 s85, 8192, s54 // (op 1953) 

s_mov_b32 m0, s85 // (op 1953) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 1953) Load value direct to lds


v_add_co_u32 v28, vcc, 128, v28 // (op 2029) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 2029) most significant half
s_barrier  // (op 3413) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1959) 
v_mov_b64 v[34:35], s[86:87] // (op 1959) convert


v_mov_b64 v[36:37], v[26:27] // (op 1959) 
v_mov_b64 v[38:39], v[36:37] // (op 1959) reg expression
s_mov_b32 m0, s55 // (op 1959) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds
v_add_co_u32 v38, s[86:87], v38, v34 // (op 1959) least significant half
v_addc_co_u32 v39, s[86:87], 0, v39, s[86:87] // (op 1959) most significant half

s_add_u32 m0, m0, 4096 // (op 1959) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1959) Load value direct to lds



s_lshl_b64 s[86:87], s[36:37], 7 // (op 2167) 
v_mov_b32 v21, s86 // (op 2167) 
v_mov_b32 v23, s87 // (op 2167) 
v_add_co_u32 v26, s[88:89], v26, v21 // (op 2167) least significant half
v_addc_co_u32 v27, s[88:89], v27, v23, s[88:89] // (op 2167) most significant half






v_mov_b32 v21, v25 // (op 845) 
v_mov_b32 v23, v21 // (op 845) reg expression

buffer_load_ubyte v24, v23, s[80:83], 0 offen offset: 0 // (op 845) Load value


s_lshl_b32 s85, s44, 2 // (op 2423) 
v_add_u32 v25, v25, s85 // (op 2423) 

s_barrier  // (op 3432) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3400) 






ds_read_b128 v[34:37], v3 offset:8192 // (op 834) Load local data 
ds_read_b128 v[38:41], v3 offset:8256 // (op 834) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v24 offset:24576 // (op 843) Store local data 

s_barrier  // (op 3419) 



ds_read_b64_tr_b8 v[42:43], v7 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:512 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4096 // (op 836) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4608 // (op 836) Transpose load local data 


v_mov_b32 v21, v30 // (op 838) 

buffer_load_ubyte v23, v21, s[76:79], 0 offen offset: 0 // (op 838) Load value

v_add_u32 v30, 4, v30 // (op 2295) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3439) 


ds_read_u8 v21, v0 offset:24576 // (op 840) Load local data 



ds_read_b64_tr_b8 v[50:51], v7 offset:32 // (op 985) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 841) 
ds_read_b64_tr_b8 v[52:53], v7 offset:544 // (op 985) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v7 offset:4128 // (op 985) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:4640 // (op 985) Transpose load local data 


ds_read_u8 v21, v0 offset:24608 // (op 988) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[50:57], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 989) 




s_cmp_lt_u32 s84, s7 // (op 977) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_977 // (op 977) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_977 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_977:








s_barrier  // (op 3677) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 272) 
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_298



buffer_store_dword a0, v20, s[56:59], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v20, s[56:59], 0 offen offset:256 sc1 // (op 295) Store value
buffer_store_dword a2, v20, s[56:59], 0 offen offset:512 sc1 // (op 295) Store value
buffer_store_dword a3, v20, s[56:59], 0 offen offset:768 sc1 // (op 295) Store value



buffer_store_dword a4, v20, s[56:59], 0 offen offset:128 sc1 // (op 1168) Store value
buffer_store_dword a5, v20, s[56:59], 0 offen offset:384 sc1 // (op 1168) Store value
buffer_store_dword a6, v20, s[56:59], 0 offen offset:640 sc1 // (op 1168) Store value
buffer_store_dword a7, v20, s[56:59], 0 offen offset:896 sc1 // (op 1168) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
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
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_298:

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
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s54, 0 // (op 280) call()
s_mul_i32 s55, s54, s8 // (op 279) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 279) 
s_mov_b32 s86, s84 // (op 279) convert

s_sub_u32 s84, s86, s19 // (op 279) 

s_sub_u32 s85, s84, 1 // (op 279) 

s_cmp_lt_u32 s55, s85 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s55, s16, 1 // (op 3080) 
v_lshrrev_b32 v16, 5, v1 // (op 3080) 
s_add_u32 s84, s55, s54 // (op 3080) 
s_lshl_b32 s84, s84, 5 // (op 3080) 

v_lshlrev_b32 v17, 2, v16 // (op 3080) 

v_add_lshl_u32 v16, s84, v17, 6 // (op 3080) 


v_and_b32 v17, -32, v1 // (op 3080) 
v_sub_u32 v19, v1, v17 // (op 3080) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 3080) 


s_lshl_b32 s55, s15, 13 // (op 3082) 
s_mov_b32 s56, s28 // (op 3082) reg expression
s_mov_b32 s57, s29 // (op 3082) reg expression
s_mov_b32 s58, s55 // (op 3082) reg expression
s_mov_b32 s59, 131072 // (op 3082) call()

RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
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
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
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
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




buffer_load_dword v24, v17, s[56:59], 0 offen // (op 333) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:256 // (op 333) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:512 // (op 333) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:768 // (op 333) Load value
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




buffer_load_dword v24, v17, s[56:59], 0 offen offset:128 // (op 1178) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:384 // (op 1178) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:640 // (op 1178) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:896 // (op 1178) Load value
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

s_add_u32 s54, s54, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 279) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 279) 
s_mov_b32 s86, s84 // (op 279) convert

s_sub_u32 s84, s86, s19 // (op 279) 

s_sub_u32 s85, s84, 1 // (op 279) 

s_cmp_lt_u32 s55, s85 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123
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


v_accvgpr_read v24, a4 // (op 1197) 
v_accvgpr_read v25, a5 // (op 1197) 
v_accvgpr_read v26, a6 // (op 1197) 
v_accvgpr_read v27, a7 // (op 1197) 
v_mul_f32 v28, v15, v24 // (op 1197) 
v_mul_f32 v29, v15, v25 // (op 1197) 
v_mul_f32 v30, v15, v26 // (op 1197) 
v_mul_f32 v31, v15, v27 // (op 1197) 



s_lshl_b32 s54, s24, 7 // (op 1199) 
v_add_u32 v16, v22, s54 // (op 1199) 


buffer_store_dwordx4 v[28:31], v16, s[64:67], 0 offen // (op 1199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123:


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




s_lshl_b32 s54, s4, 7 // (op 1221) 
v_add_u32 v16, v18, s54 // (op 1221) 


buffer_load_dwordx4 v[24:27], v16, s[60:63], 0 offen // (op 1221) Load value

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




s_lshl_b32 s54, s24, 7 // (op 1220) 
v_add_u32 v23, v22, s54 // (op 1220) 


buffer_store_dwordx4 v[24:27], v23, s[64:67], 0 offen // (op 1220) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:


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
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















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
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 7, v1 // (op 2562) 
s_mul_i32 s15, s16, s52 // (op 2562) 
s_add_u32 s15, s15, s0 // (op 2562) 
v_and_b32 v3, 63, v1 // (op 2562) 
s_add_u32 s6, s15, s9 // (op 2562) 

v_lshrrev_b32 v6, 4, v3 // (op 2562) 
s_mul_hi_u32 s15, s6, s13 // (op 2562) 
v_lshlrev_b32 v7, 2, v6 // (op 2562) 

s_sub_u32 s7, s6, s15 // (op 2562) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 2562) 


s_lshr_b32 s8, s7, 1 // (op 2562) 

v_lshrrev_b32 v0, 2, v3 // (op 2562) 

s_add_u32 s7, s8, s15 // (op 2562) 


v_and_b32 v3, 3, v0 // (op 2562) 

s_lshr_b32 s8, s7, s11 // (op 2562) 

v_and_b32 v0, 3, v1 // (op 2562) 
s_and_b32 s7, s11, 2147483648 // (op 2562) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 2562) 


s_cmp_eq_i32 s7, 0 // (op 2562) 
s_mov_b32 s15, scc // (op 2562) 

v_lshrrev_b32 v0, 6, v1 // (op 2562) 
s_cmp_eq_i32 s15, 0 // (op 2562) 
s_cselect_b32 s7, s6, s8 // (op 2562) 



v_and_b32 v3, 1, v0 // (op 2562) 

s_mul_hi_u32 s6, s7, s12 // (op 2562) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 2562) 


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
v_lshl_add_u32 v3, s8, 5, v6 // (op 2562) 


s_sub_u32 s8, s7, s6 // (op 2562) 


v_lshl_add_u32 v6, s8, 6, v0 // (op 2562) 


v_mul_lo_u32 v0, v6, s4 // (op 2562) 

v_add_lshl_u32 v6, v3, v0, 2 // (op 2562) 


s_mul_i32 s19, s4, s23 // (op 2564) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2564) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2564) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2564) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2564) most significant: sum
s_add_u32 s19, s19, s6 // (op 2564) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2564) 

s_mov_b32 s60, s20 // (op 2564) reg expression
s_mov_b32 s61, s21 // (op 2564) reg expression
s_mov_b32 s62, s28 // (op 2564) reg expression
s_mov_b32 s63, 131072 // (op 2564) call()

v_lshrrev_b32 v0, 7, v1 // (op 2820) 
s_mul_i32 s15, s16, s52 // (op 2820) 
s_add_u32 s15, s15, s0 // (op 2820) 
v_and_b32 v3, 63, v1 // (op 2820) 
s_add_u32 s6, s15, s9 // (op 2820) 

v_lshrrev_b32 v7, 4, v3 // (op 2820) 
s_mul_hi_u32 s15, s6, s13 // (op 2820) 
v_lshlrev_b32 v12, 2, v7 // (op 2820) 

s_sub_u32 s7, s6, s15 // (op 2820) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2820) 


s_lshr_b32 s8, s7, 1 // (op 2820) 

v_lshrrev_b32 v0, 2, v3 // (op 2820) 

s_add_u32 s7, s8, s15 // (op 2820) 


v_and_b32 v3, 3, v0 // (op 2820) 

s_lshr_b32 s8, s7, s11 // (op 2820) 

v_and_b32 v0, 3, v1 // (op 2820) 
s_and_b32 s7, s11, 2147483648 // (op 2820) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 2820) 


s_cmp_eq_i32 s7, 0 // (op 2820) 
s_mov_b32 s15, scc // (op 2820) 

v_lshrrev_b32 v0, 6, v1 // (op 2820) 
s_cmp_eq_i32 s15, 0 // (op 2820) 
s_cselect_b32 s7, s6, s8 // (op 2820) 



v_and_b32 v3, 1, v0 // (op 2820) 

s_mul_hi_u32 s6, s7, s12 // (op 2820) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 2820) 


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
v_lshl_add_u32 v3, s8, 5, v7 // (op 2820) 


s_sub_u32 s8, s7, s6 // (op 2820) 


v_lshl_add_u32 v7, s8, 6, v0 // (op 2820) 


v_mul_lo_u32 v0, v7, s24 // (op 2820) 

v_add_lshl_u32 v7, v3, v0, 2 // (op 2820) 


s_mul_i32 s19, s24, s23 // (op 2822) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2822) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2822) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2822) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2822) most significant: sum
s_add_u32 s19, s19, s6 // (op 2822) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2822) 

s_mov_b32 s64, s26 // (op 2822) reg expression
s_mov_b32 s65, s27 // (op 2822) reg expression
s_mov_b32 s66, s28 // (op 2822) reg expression
s_mov_b32 s67, 131072 // (op 2822) call()

v_lshrrev_b32 v0, 7, v1 // (op 2889) 
v_and_b32 v3, 15, v1 // (op 2889) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 2889) 


v_and_b32 v0, 48, v1 // (op 2889) 
v_lshl_add_u32 v3, v12, 7, v0 // (op 2889) 


v_and_b32 v0, 48, v1 // (op 2964) 
v_and_b32 v12, 63, v1 // (op 2964) 
v_sub_u32 v13, v12, v0 // (op 2964) 

v_lshrrev_b32 v12, 1, v13 // (op 2964) 
v_add_lshl_u32 v16, v0, v12, 6 // (op 2964) 


v_and_b32 v0, 1, v13 // (op 2964) 

v_lshlrev_b32 v12, 3, v0 // (op 2964) 

v_lshrrev_b32 v0, 6, v1 // (op 2964) 
v_and_b32 v13, 1, v0 // (op 2964) 

v_lshl_add_u32 v0, v13, 4, v12 // (op 2964) 


v_add_u32 v12, v16, v0 // (op 2964) 


v_lshrrev_b32 v0, 6, v1 // (op 3040) 
v_and_b32 v13, 1, v0 // (op 3040) 

v_and_b32 v0, 15, v1 // (op 3040) 
v_lshl_add_u32 v16, v13, 4, v0 // (op 3040) 


v_and_b32 v0, 63, v1 // (op 3040) 
v_lshrrev_b32 v13, 4, v0 // (op 3040) 

v_lshl_add_u32 v0, v13, 6, v16 // (op 3040) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_accvgpr_write a4, 0.00000 // (op 578) call()
v_accvgpr_write a5, 0.00000 // (op 578) call()
v_accvgpr_write a6, 0.00000 // (op 578) call()
v_accvgpr_write a7, 0.00000 // (op 578) call()
v_lshrrev_b32 v13, 3, v1 // (op 2066) 
s_mul_i32 s15, s16, s52 // (op 2066) 
s_add_u32 s15, s15, s0 // (op 2066) 
v_and_b32 v16, -8, v1 // (op 2066) 
s_add_u32 s6, s15, s9 // (op 2066) 

v_sub_u32 v17, v1, v16 // (op 2066) 

s_mul_hi_u32 s15, s6, s13 // (op 2066) 
v_lshlrev_b32 v16, 4, v17 // (op 2066) 

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


v_lshl_add_u32 v17, s15, 5, v13 // (op 2066) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2066) 
s_mul_i32 s8, s7, s18 // (op 2066) 


s_sub_u32 s7, s6, s8 // (op 2066) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2066) 


v_mov_b32 v19, 0 // (op 2066) convert
v_mov_b32 v18, v13 // (op 2066) convert

v_mul_lo_u32 v21, v17, s31 // (op 2066) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 2066) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 2066) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 2066) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 2066) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 2066) most significant half



s_mul_i32 s19, s30, s35 // (op 2068) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2068) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2068) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2068) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2068) most significant: sum
s_add_u32 s19, s19, s6 // (op 2068) most significant: sum


s_mov_b32 s68, s32 // (op 2068) reg expression
s_mov_b32 s69, s33 // (op 2068) reg expression
s_mov_b32 s70, s18 // (op 2068) reg expression
s_mov_b32 s71, 131072 // (op 2068) call()

v_lshrrev_b32 v13, 3, v1 // (op 2085) 
s_mul_i32 s15, s16, s52 // (op 2085) 
s_add_u32 s15, s15, s0 // (op 2085) 
v_and_b32 v16, -8, v1 // (op 2085) 
s_add_u32 s6, s15, s9 // (op 2085) 

v_sub_u32 v17, v1, v16 // (op 2085) 

s_mul_hi_u32 s15, s6, s13 // (op 2085) 
v_lshlrev_b32 v16, 4, v17 // (op 2085) 

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


v_lshl_add_u32 v17, s15, 5, v13 // (op 2085) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2085) 
s_mul_i32 s8, s7, s18 // (op 2085) 


s_sub_u32 s7, s6, s8 // (op 2085) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2085) 


v_mov_b32 v19, 0 // (op 2085) convert
v_mov_b32 v18, v13 // (op 2085) convert

v_mul_lo_u32 v23, v17, s31 // (op 2085) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 2085) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 2085) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 2085) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 2085) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 2085) most significant half



s_mul_i32 s19, s30, s35 // (op 2087) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2087) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2087) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2087) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2087) most significant: sum
s_add_u32 s19, s19, s6 // (op 2087) most significant: sum


s_mov_b32 s68, s32 // (op 2087) reg expression
s_mov_b32 s69, s33 // (op 2087) reg expression
s_mov_b32 s70, s18 // (op 2087) reg expression
s_mov_b32 s71, 131072 // (op 2087) call()

v_and_b32 v13, -4, v1 // (op 2198) 
s_mul_i32 s15, s16, s52 // (op 2198) 
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


v_mul_lo_u32 v17, v13, s37 // (op 2198) most significant: low of low * high
v_mul_hi_u32 v24, v13, s36 // (op 2198) most significant: high of low * low
v_mul_lo_u32 v16, v13, s36 // (op 2198) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 2198) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v18 // (op 2198) least significant half
v_addc_co_u32 v17, s[18:19], v17, v19, s[18:19] // (op 2198) most significant half



s_mul_i32 s19, s36, s3 // (op 2200) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2200) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2200) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2200) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2200) most significant: sum
s_add_u32 s19, s19, s6 // (op 2200) most significant: sum


s_mov_b32 s72, s38 // (op 2200) reg expression
s_mov_b32 s73, s39 // (op 2200) reg expression
s_mov_b32 s74, s18 // (op 2200) reg expression
s_mov_b32 s75, 131072 // (op 2200) call()

v_and_b32 v13, -4, v1 // (op 2217) 
s_mul_i32 s15, s16, s52 // (op 2217) 
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


v_mul_lo_u32 v19, v13, s37 // (op 2217) most significant: low of low * high
v_mul_hi_u32 v26, v13, s36 // (op 2217) most significant: high of low * low
v_mul_lo_u32 v18, v13, s36 // (op 2217) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 2217) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v24 // (op 2217) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 2217) most significant half



s_mul_i32 s19, s36, s3 // (op 2219) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2219) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2219) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2219) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2219) most significant: sum
s_add_u32 s19, s19, s6 // (op 2219) most significant: sum


s_mov_b32 s72, s38 // (op 2219) reg expression
s_mov_b32 s73, s39 // (op 2219) reg expression
s_mov_b32 s74, s18 // (op 2219) reg expression
s_mov_b32 s75, 131072 // (op 2219) call()

v_lshrrev_b32 v13, 7, v1 // (op 2326) 
s_mul_i32 s15, s16, s52 // (op 2326) 
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


v_lshl_add_u32 v13, s15, 5, v25 // (op 2326) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2326) 
s_mul_i32 s8, s7, s18 // (op 2326) 


s_sub_u32 s7, s6, s8 // (op 2326) 


v_lshl_add_u32 v25, s7, 2, v24 // (op 2326) 


v_mul_lo_u32 v24, v13, s40 // (op 2326) 
v_add_u32 v24, v24, v25 // (op 2326) 


s_mul_i32 s19, s40, s35 // (op 2328) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2328) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2328) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2328) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2328) most significant: sum
s_add_u32 s19, s19, s6 // (op 2328) most significant: sum


s_mov_b32 s76, s42 // (op 2328) reg expression
s_mov_b32 s77, s43 // (op 2328) reg expression
s_mov_b32 s78, s18 // (op 2328) reg expression
s_mov_b32 s79, 131072 // (op 2328) call()

v_lshrrev_b32 v13, 7, v1 // (op 2345) 
s_mul_i32 s15, s16, s52 // (op 2345) 
s_add_u32 s15, s15, s0 // (op 2345) 
v_and_b32 v25, 15, v1 // (op 2345) 
s_add_u32 s6, s15, s9 // (op 2345) 

v_lshl_add_u32 v26, v13, 4, v25 // (op 2345) 


s_mul_hi_u32 s15, s6, s13 // (op 2345) 
v_and_b32 v13, 63, v1 // (op 2345) 
s_sub_u32 s7, s6, s15 // (op 2345) 
v_lshrrev_b32 v25, 4, v13 // (op 2345) 

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


v_lshl_add_u32 v13, s15, 5, v26 // (op 2345) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2345) 
s_mul_i32 s8, s7, s18 // (op 2345) 


s_sub_u32 s7, s6, s8 // (op 2345) 


v_lshl_add_u32 v26, s7, 2, v25 // (op 2345) 


v_mul_lo_u32 v25, v13, s40 // (op 2345) 
v_add_u32 v25, v25, v26 // (op 2345) 


s_mul_i32 s19, s40, s35 // (op 2347) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2347) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2347) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2347) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2347) most significant: sum
s_add_u32 s19, s19, s6 // (op 2347) most significant: sum


s_mov_b32 s76, s42 // (op 2347) reg expression
s_mov_b32 s77, s43 // (op 2347) reg expression
s_mov_b32 s78, s18 // (op 2347) reg expression
s_mov_b32 s79, 131072 // (op 2347) call()

v_and_b32 v13, -64, v1 // (op 2454) 
s_mul_i32 s15, s16, s52 // (op 2454) 
s_add_u32 s15, s15, s0 // (op 2454) 
v_sub_u32 v26, v1, v13 // (op 2454) 

s_add_u32 s6, s15, s9 // (op 2454) 

v_lshrrev_b32 v13, 6, v1 // (op 2454) 
s_mul_hi_u32 s15, s6, s13 // (op 2454) 
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

v_lshl_add_u32 v27, s15, 6, v26 // (op 2454) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2454) 
s_mul_i32 s8, s7, s18 // (op 2454) 


s_sub_u32 s7, s6, s8 // (op 2454) 


v_lshl_add_u32 v26, s7, 2, v13 // (op 2454) 


v_mul_lo_u32 v13, v26, s44 // (op 2454) 
v_add_u32 v13, v13, v27 // (op 2454) 


s_mul_i32 s19, s44, s49 // (op 2456) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2456) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2456) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2456) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2456) most significant: sum
s_add_u32 s19, s19, s6 // (op 2456) most significant: sum


s_mov_b32 s80, s46 // (op 2456) reg expression
s_mov_b32 s81, s47 // (op 2456) reg expression
s_mov_b32 s82, s18 // (op 2456) reg expression
s_mov_b32 s83, 131072 // (op 2456) call()

v_and_b32 v26, -64, v1 // (op 2473) 
s_mul_i32 s15, s16, s52 // (op 2473) 
s_add_u32 s15, s15, s0 // (op 2473) 
v_sub_u32 v27, v1, v26 // (op 2473) 

s_add_u32 s6, s15, s9 // (op 2473) 

v_lshrrev_b32 v26, 6, v1 // (op 2473) 
s_mul_hi_u32 s15, s6, s13 // (op 2473) 
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

v_lshl_add_u32 v28, s15, 6, v27 // (op 2473) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2473) 
s_mul_i32 s8, s7, s18 // (op 2473) 


s_sub_u32 s7, s6, s8 // (op 2473) 


v_lshl_add_u32 v27, s7, 2, v26 // (op 2473) 


v_mul_lo_u32 v26, v27, s44 // (op 2473) 
v_add_u32 v26, v26, v28 // (op 2473) 


s_mul_i32 s19, s44, s49 // (op 2475) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2475) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2475) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2475) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2475) most significant: sum
s_add_u32 s19, s19, s6 // (op 2475) most significant: sum


s_mov_b32 s80, s46 // (op 2475) reg expression
s_mov_b32 s81, s47 // (op 2475) reg expression
s_mov_b32 s82, s18 // (op 2475) reg expression
s_mov_b32 s83, 131072 // (op 2475) call()




buffer_load_ubyte v27, v24, s[76:79], 0 offen offset: 0 // (op 1684) Load value
v_and_b32 v28, -4, v1 // (op 2930) 
v_sub_u32 v29, v1, v28 // (op 2930) 

v_lshlrev_b32 v28, 4, v29 // (op 2930) 

v_lshrrev_b32 v29, 2, v1 // (op 2930) 
v_lshl_add_u32 v30, v29, 6, v28 // (op 2930) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2930) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3465) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1976) 
v_mov_b64 v[28:29], s[18:19] // (op 1976) convert


v_mov_b64 v[30:31], v[16:17] // (op 1976) reg expression
s_mov_b32 m0, s15 // (op 1976) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1976) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1976) most significant half

s_add_u32 m0, m0, 4096 // (op 1976) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1976) Load value direct to lds





v_mov_b32 v28, v13 // (op 1691) reg expression

buffer_load_ubyte v29, v28, s[80:83], 0 offen offset: 0 // (op 1691) Load value

v_and_b32 v28, -8, v1 // (op 2865) 
v_sub_u32 v30, v1, v28 // (op 2865) 

v_lshlrev_b32 v28, 4, v30 // (op 2865) 

v_lshrrev_b32 v30, 3, v1 // (op 2865) 
v_lshl_add_u32 v31, v30, 7, v28 // (op 2865) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 2865) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3446) 


s_nop 0
s_add_u32 s6, 8192, s15 // (op 1981) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1981) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 1981) Load value direct to lds





buffer_load_ubyte v28, v24, s[76:79], 0 offen offset:4 // (op 1698) Load value

v_and_b32 v24, -4, v1 // (op 3149) 
v_sub_u32 v30, v1, v24 // (op 3149) 

v_lshlrev_b32 v24, 4, v30 // (op 3149) 

v_lshrrev_b32 v30, 2, v1 // (op 3149) 
v_lshl_add_u32 v31, v30, 6, v24 // (op 3149) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 3149) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3513) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1986) 
v_mov_b64 v[30:31], s[18:19] // (op 1986) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 1986) 
v_mov_b32 v24, s18 // (op 1986) 
v_mov_b32 v32, s19 // (op 1986) 
v_add_co_u32 v34, s[28:29], v16, v24 // (op 1986) least significant half
v_addc_co_u32 v35, s[28:29], v17, v32, s[28:29] // (op 1986) most significant half




s_add_u32 s6, 12288, s15 // (op 1986) 
s_mov_b32 m0, s6 // (op 1986) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 1986) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1986) most significant half

s_add_u32 m0, m0, 4096 // (op 1986) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 1986) Load value direct to lds







s_lshl_b32 s15, s44, 2 // (op 1705) 
v_add_u32 v24, v13, s15 // (op 1705) 


buffer_load_ubyte v16, v24, s[80:83], 0 offen offset: 0 // (op 1705) Load value


v_and_b32 v13, -8, v1 // (op 3110) 
v_sub_u32 v17, v1, v13 // (op 3110) 

v_lshlrev_b32 v13, 4, v17 // (op 3110) 

v_lshrrev_b32 v17, 3, v1 // (op 3110) 
v_lshl_add_u32 v24, v17, 7, v13 // (op 3110) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 3110) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3504) 


s_nop 0
s_add_u32 s6, 20480, s15 // (op 1988) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1988) reg expression
s_mov_b32 s7, 128 // (op 1988) call()
buffer_load_dwordx4 v20, s[68:71], s7 offen offset: 0 lds // (op 1988) Load value direct to lds




v_and_b32 v13, -64, v1 // (op 3006) 
v_sub_u32 v17, v1, v13 // (op 3006) 

v_and_b32 v13, 4294967232, v1 // (op 3006) 
v_add_u32 v24, v13, v17 // (op 3006) 


s_barrier  // (op 3484) 



s_waitcnt vmcnt(6)
ds_write_b8 v24, v29 offset:24576 // (op 1711) Store local data 

s_barrier  // (op 3462) 



ds_read_b64_tr_b8 v[30:31], v12 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v12 offset:512 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v12 offset:4096 // (op 1729) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v12 offset:4608 // (op 1729) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3481) 


ds_read_u8 v13, v0 offset:24576 // (op 1739) Load local data 



ds_read_b64_tr_b8 v[38:39], v12 offset:32 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v12 offset:544 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v12 offset:4128 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v12 offset:4640 // (op 1734) Transpose load local data 


ds_read_u8 v17, v0 offset:24608 // (op 1744) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3443) 



ds_read_b128 v[46:49], v3 offset:8192 // (op 1725) Load local data 
ds_read_b128 v[50:53], v3 offset:8256 // (op 1725) Load local data 
v_and_b32 v24, -8, v1 // (op 2837) 
v_sub_u32 v20, v1, v24 // (op 2837) 

v_lshlrev_b32 v21, 4, v20 // (op 2837) 

v_lshrrev_b32 v20, 3, v1 // (op 2837) 
v_lshl_add_u32 v24, v20, 7, v21 // (op 2837) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 2837) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 2902) 
v_sub_u32 v20, v1, v24 // (op 2902) 

v_lshlrev_b32 v21, 4, v20 // (op 2902) 

v_lshrrev_b32 v20, 2, v1 // (op 2902) 
v_lshl_add_u32 v24, v20, 6, v21 // (op 2902) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 2902) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -64, v1 // (op 2977) 
v_sub_u32 v20, v1, v24 // (op 2977) 

v_and_b32 v21, 4294967232, v1 // (op 2977) 
v_add_u32 v24, v21, v20 // (op 2977) 


v_and_b32 v20, -8, v1 // (op 3095) 
v_sub_u32 v21, v1, v20 // (op 3095) 

v_lshlrev_b32 v20, 4, v21 // (op 3095) 

v_lshrrev_b32 v21, 3, v1 // (op 3095) 
v_lshl_add_u32 v54, v21, 7, v20 // (op 3095) 


s_nop 0
v_readfirstlane_b32 s7, v54 // (op 3095) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 3134) 
v_sub_u32 v21, v1, v20 // (op 3134) 

v_lshlrev_b32 v20, 4, v21 // (op 3134) 

v_lshrrev_b32 v21, 2, v1 // (op 3134) 
v_lshl_add_u32 v54, v21, 6, v20 // (op 3134) 


s_nop 0
v_readfirstlane_b32 s8, v54 // (op 3134) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 3178) 
v_sub_u32 v21, v1, v20 // (op 3178) 

v_and_b32 v20, 4294967232, v1 // (op 3178) 
v_add_u32 v54, v20, v21 // (op 3178) 


s_mov_b32 s1, 0 // (op 580) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 579) 
s_mov_b32 s28, s18 // (op 579) convert

s_and_b32 s29, s28, 4294967294 // (op 579) 

s_cmp_lt_u32 s1, s29 // (op 579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_579 // (op 579) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_579 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_579:
s_add_u32 s1, s1, 2 // (op 581) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v27, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 184) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v27, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 615) 


v_mov_b32 v20, v25 // (op 181) 

buffer_load_ubyte v27, v20, s[76:79], 0 offen offset:8 // (op 181) Load value
s_barrier  // (op 3468) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1925) 
v_mov_b64 v[56:57], s[18:19] // (op 1925) convert


v_mov_b64 v[58:59], v[18:19] // (op 1925) 
s_lshl_b64 s[18:19], s[36:37], 7 // (op 1925) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1925) 

v_mov_b32 v21, s28 // (op 1925) 
v_mov_b32 v55, s29 // (op 1925) 
v_add_co_u32 v60, s[18:19], v58, v21 // (op 1925) least significant half
v_addc_co_u32 v61, s[18:19], v59, v55, s[18:19] // (op 1925) most significant half




s_mov_b32 m0, s6 // (op 1925) reg expression
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 1925) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 1925) most significant half

s_add_u32 m0, m0, 4096 // (op 1925) 
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 1925) Load value direct to lds




v_mov_b32 v21, v26 // (op 188) 
s_lshl_b32 s18, s44, 3 // (op 188) 
v_add_u32 v55, v21, s18 // (op 188) 


buffer_load_ubyte v29, v55, s[80:83], 0 offen offset: 0 // (op 188) Load value

s_barrier  // (op 3449) 


v_mov_b64 v[56:57], v[22:23] // (op 1920) 
s_add_u32 s18, 8192, s15 // (op 1920) 

s_mov_b32 m0, s18 // (op 1920) reg expression
s_mov_b32 s19, 256 // (op 1920) call()
buffer_load_dwordx4 v56, s[68:71], s19 offen offset: 0 lds // (op 1920) Load value direct to lds


s_barrier  // (op 3522) 



s_waitcnt vmcnt(6)
ds_write_b8 v54, v16 offset:24832 // (op 508) Store local data 
s_barrier  // (op 3510) 



ds_read_b64_tr_b8 v[30:31], v11 offset:12288 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v11 offset:12800 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v11 offset:16384 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v11 offset:16896 // (op 501) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3519) 


ds_read_u8 v13, v10 offset:24832 // (op 505) Load local data 



ds_read_b64_tr_b8 v[38:39], v11 offset:12320 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v11 offset:12832 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v11 offset:16416 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v11 offset:16928 // (op 620) Transpose load local data 


ds_read_u8 v17, v10 offset:24864 // (op 623) Load local data 
s_barrier  // (op 3633) 
s_waitcnt vmcnt(5)
s_barrier  // (op 3501) 



ds_read_b128 v[46:49], v5 offset:20480 // (op 499) Load local data 
ds_read_b128 v[50:53], v5 offset:20544 // (op 499) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v28, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 506) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v28, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 616) 



buffer_load_ubyte v28, v20, s[76:79], 0 offen offset:12 // (op 503) Load value

s_barrier  // (op 3516) 
v_add_u32 v25, 8, v25 // (op 2337) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1948) 
v_mov_b64 v[60:61], s[18:19] // (op 1948) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 1948) 
s_mul_i32 s29, 3, s19 // (op 1948) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 1948) least significant
s_mul_hi_u32 s50, 3, s18 // (op 1948) most significant: high of low * low
s_add_u32 s29, s29, s50 // (op 1948) most significant: sum


v_mov_b32 v20, s28 // (op 1948) 
v_mov_b32 v55, s29 // (op 1948) 
v_add_co_u32 v62, s[18:19], v58, v20 // (op 1948) least significant half
v_addc_co_u32 v63, s[18:19], v59, v55, s[18:19] // (op 1948) most significant half




s_add_u32 s18, 12288, s8 // (op 1948) 
s_nop 0
s_mov_b32 m0, s18 // (op 1948) reg expression
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds
v_add_co_u32 v62, s[28:29], v62, v60 // (op 1948) least significant half
v_addc_co_u32 v63, s[28:29], 0, v63, s[28:29] // (op 1948) most significant half

s_add_u32 m0, m0, 4096 // (op 1948) 
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1948) Load value direct to lds






s_lshl_b32 s18, s44, 2 // (op 510) 
s_mul_i32 s19, 3, s18 // (op 510) 

v_add_u32 v20, v21, s19 // (op 510) 


s_lshl_b64 s[18:19], s[36:37], 7 // (op 2209) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2209) 

v_mov_b32 v55, s28 // (op 2209) 
v_mov_b32 v58, s29 // (op 2209) 
v_add_co_u32 v18, s[18:19], v18, v55 // (op 2209) least significant half
v_addc_co_u32 v19, s[18:19], v19, v58, s[18:19] // (op 2209) most significant half




buffer_load_ubyte v16, v20, s[80:83], 0 offen offset: 0 // (op 510) Load value


s_barrier  // (op 3507) 
s_lshl_b32 s18, s44, 3 // (op 2465) 


v_add_u32 v26, v26, s18 // (op 2465) 

s_add_u32 s18, 20480, s7 // (op 1943) 

s_mov_b32 m0, s18 // (op 1943) reg expression
s_mov_b32 s19, 384 // (op 1943) call()
buffer_load_dwordx4 v56, s[68:71], s19 offen offset: 0 lds // (op 1943) Load value direct to lds



s_barrier  // (op 3487) 
v_add_co_u32 v22, vcc, 256, v22 // (op 2077) least significant half



v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 2077) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v24, v29 offset:24576 // (op 186) Store local data 
s_barrier  // (op 3474) 



ds_read_b64_tr_b8 v[30:31], v9 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v9 offset:512 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v9 offset:4096 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v9 offset:4608 // (op 179) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3494) 


ds_read_u8 v13, v8 offset:24576 // (op 183) Load local data 



ds_read_b64_tr_b8 v[38:39], v9 offset:32 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:544 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4128 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:4640 // (op 611) Transpose load local data 


ds_read_u8 v17, v8 offset:24608 // (op 614) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3455) 



ds_read_b128 v[46:49], v4 offset:8192 // (op 177) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 177) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 579) 
s_mov_b32 s28, s18 // (op 579) convert

s_and_b32 s29, s28, 4294967294 // (op 579) 

s_cmp_lt_u32 s1, s29 // (op 579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_579 // (op 579) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_579 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_579:












s_barrier  // (op 3617) 
v_lshrrev_b32 v29, 3, v1 // (op 2015) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2015) 
v_and_b32 v18, -8, v1 // (op 2015) 
s_and_b32 s15, s18, 4294967294 // (op 2015) 
v_sub_u32 v19, v1, v18 // (op 2015) 

s_add_u32 s6, s0, s15 // (op 2015) 

v_lshlrev_b32 v18, 4, v19 // (op 2015) 

s_mul_i32 s15, s16, s52 // (op 2015) 
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


s_mul_i32 s8, s7, s18 // (op 2015) 


v_lshl_add_u32 v19, s15, 5, v29 // (op 2015) 


s_sub_u32 s7, s6, s8 // (op 2015) 


v_lshl_add_u32 v29, s7, 7, v18 // (op 2015) 


v_mov_b32 v21, 0 // (op 2015) convert
v_mov_b32 v20, v29 // (op 2015) convert

v_mul_lo_u32 v23, v19, s31 // (op 2015) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 2015) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 2015) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 2015) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 2015) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 2015) most significant half



s_mul_i32 s19, s30, s35 // (op 2017) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2017) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2017) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2017) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2017) most significant: sum
s_add_u32 s19, s19, s6 // (op 2017) most significant: sum


s_mov_b32 s68, s32 // (op 2017) reg expression
s_mov_b32 s69, s33 // (op 2017) reg expression
s_mov_b32 s70, s18 // (op 2017) reg expression
s_mov_b32 s71, 131072 // (op 2017) call()

v_and_b32 v29, -4, v1 // (op 2153) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2153) 
v_sub_u32 v18, v1, v29 // (op 2153) 

s_and_b32 s15, s18, 4294967294 // (op 2153) 
v_lshlrev_b32 v29, 4, v18 // (op 2153) 

s_add_u32 s6, s0, s15 // (op 2153) 

v_lshrrev_b32 v18, 2, v1 // (op 2153) 
s_mul_i32 s15, s16, s52 // (op 2153) 
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


v_lshl_add_u32 v19, s15, 6, v29 // (op 2153) 


s_sub_u32 s7, s6, s8 // (op 2153) 


v_mov_b32 v21, 0 // (op 2153) convert
v_mov_b32 v20, v19 // (op 2153) convert

v_lshl_add_u32 v19, s7, 7, v18 // (op 2153) 


v_mul_lo_u32 v25, v19, s37 // (op 2153) most significant: low of low * high
v_mul_hi_u32 v18, v19, s36 // (op 2153) most significant: high of low * low
v_mul_lo_u32 v24, v19, s36 // (op 2153) least significant: low of low * low
v_add_u32 v25, v25, v18 // (op 2153) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v20 // (op 2153) least significant half
v_addc_co_u32 v25, s[18:19], v25, v21, s[18:19] // (op 2153) most significant half



s_mul_i32 s19, s36, s3 // (op 2155) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2155) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2155) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2155) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2155) most significant: sum
s_add_u32 s19, s19, s6 // (op 2155) most significant: sum


s_mov_b32 s72, s38 // (op 2155) reg expression
s_mov_b32 s73, s39 // (op 2155) reg expression
s_mov_b32 s74, s18 // (op 2155) reg expression
s_mov_b32 s75, 131072 // (op 2155) call()

v_lshrrev_b32 v29, 7, v1 // (op 2281) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2281) 
v_and_b32 v18, 15, v1 // (op 2281) 
s_and_b32 s15, s18, 4294967294 // (op 2281) 
v_lshl_add_u32 v19, v29, 4, v18 // (op 2281) 


s_add_u32 s6, s0, s15 // (op 2281) 

v_and_b32 v18, 63, v1 // (op 2281) 
v_lshrrev_b32 v29, 4, v18 // (op 2281) 

s_mul_i32 s15, s16, s52 // (op 2281) 
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


v_lshl_add_u32 v18, s15, 5, v19 // (op 2281) 


s_sub_u32 s7, s6, s8 // (op 2281) 


v_lshl_add_u32 v19, s7, 2, v29 // (op 2281) 


v_mul_lo_u32 v29, v18, s40 // (op 2281) 
v_add_u32 v29, v29, v19 // (op 2281) 


s_mul_i32 s19, s40, s35 // (op 2283) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2283) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2283) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2283) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2283) most significant: sum
s_add_u32 s19, s19, s6 // (op 2283) most significant: sum


s_mov_b32 s76, s42 // (op 2283) reg expression
s_mov_b32 s77, s43 // (op 2283) reg expression
s_mov_b32 s78, s18 // (op 2283) reg expression
s_mov_b32 s79, 131072 // (op 2283) call()

v_and_b32 v18, -64, v1 // (op 2409) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2409) 
v_sub_u32 v19, v1, v18 // (op 2409) 

s_and_b32 s15, s18, 4294967294 // (op 2409) 
v_lshrrev_b32 v18, 6, v1 // (op 2409) 
s_add_u32 s6, s0, s15 // (op 2409) 

s_mul_i32 s15, s16, s52 // (op 2409) 
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


v_lshl_add_u32 v20, s15, 6, v19 // (op 2409) 


s_sub_u32 s7, s6, s8 // (op 2409) 


v_lshl_add_u32 v19, s7, 2, v18 // (op 2409) 


v_mul_lo_u32 v18, v19, s44 // (op 2409) 
v_add_u32 v18, v18, v20 // (op 2409) 


s_mul_i32 s19, s44, s49 // (op 2411) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2411) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2411) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2411) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2411) most significant: sum
s_add_u32 s19, s19, s6 // (op 2411) most significant: sum


s_mov_b32 s80, s46 // (op 2411) reg expression
s_mov_b32 s81, s47 // (op 2411) reg expression
s_mov_b32 s82, s18 // (op 2411) reg expression
s_mov_b32 s83, 131072 // (op 2411) call()

v_and_b32 v19, -8, v1 // (op 2850) 
v_sub_u32 v20, v1, v19 // (op 2850) 

v_lshlrev_b32 v19, 4, v20 // (op 2850) 

v_lshrrev_b32 v20, 3, v1 // (op 2850) 
v_lshl_add_u32 v21, v20, 7, v19 // (op 2850) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2850) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 2915) 
v_sub_u32 v20, v1, v19 // (op 2915) 

v_lshlrev_b32 v19, 4, v20 // (op 2915) 

v_lshrrev_b32 v20, 2, v1 // (op 2915) 
v_lshl_add_u32 v21, v20, 6, v19 // (op 2915) 


s_nop 0
v_readfirstlane_b32 s6, v21 // (op 2915) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -64, v1 // (op 2990) 
v_sub_u32 v20, v1, v19 // (op 2990) 

v_and_b32 v19, 4294967232, v1 // (op 2990) 
v_add_u32 v21, v19, v20 // (op 2990) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 585) 
s_and_b32 s7, s18, 4294967294 // (op 585) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 584) 
s_mov_b32 s8, s18 // (op 584) convert

s_cmp_lt_u32 s7, s8 // (op 584) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_584 // (op 584) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_584 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_584:
s_barrier  // (op 3452) 
s_add_u32 s7, s7, 1 // (op 586) 


v_mov_b64 v[16:17], v[22:23] // (op 1930) 
s_add_u32 s8, 8192, s15 // (op 1930) 

s_mov_b32 m0, s8 // (op 1930) reg expression
buffer_load_dwordx4 v16, s[68:71], 0 offen offset: 0 lds // (op 1930) Load value direct to lds


v_add_co_u32 v22, vcc, 128, v22 // (op 2005) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 2005) most significant half
s_barrier  // (op 3471) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1936) 
v_mov_b64 v[16:17], s[18:19] // (op 1936) convert


v_mov_b64 v[26:27], v[24:25] // (op 1936) 
v_mov_b64 v[30:31], v[26:27] // (op 1936) reg expression
s_mov_b32 m0, s6 // (op 1936) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v16 // (op 1936) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1936) most significant half

s_add_u32 m0, m0, 4096 // (op 1936) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1936) Load value direct to lds



s_lshl_b64 s[18:19], s[36:37], 7 // (op 2145) 
v_mov_b32 v13, s18 // (op 2145) 
v_mov_b32 v16, s19 // (op 2145) 
v_add_co_u32 v24, s[28:29], v24, v13 // (op 2145) least significant half
v_addc_co_u32 v25, s[28:29], v25, v16, s[28:29] // (op 2145) most significant half






v_mov_b32 v13, v18 // (op 452) 
v_mov_b32 v16, v13 // (op 452) reg expression

buffer_load_ubyte v17, v16, s[80:83], 0 offen offset: 0 // (op 452) Load value


s_lshl_b32 s8, s44, 2 // (op 2401) 
v_add_u32 v18, v18, s8 // (op 2401) 

s_barrier  // (op 3490) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3458) 






ds_read_b128 v[30:33], v4 offset:8192 // (op 441) Load local data 
ds_read_b128 v[34:37], v4 offset:8256 // (op 441) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v17 offset:24576 // (op 450) Store local data 

s_barrier  // (op 3477) 



ds_read_b64_tr_b8 v[38:39], v9 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:512 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4096 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:4608 // (op 443) Transpose load local data 


v_mov_b32 v13, v29 // (op 445) 

buffer_load_ubyte v16, v13, s[76:79], 0 offen offset: 0 // (op 445) Load value

v_add_u32 v29, 4, v29 // (op 2273) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3497) 


ds_read_u8 v13, v8 offset:24576 // (op 447) Load local data 



ds_read_b64_tr_b8 v[46:47], v9 offset:32 // (op 592) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 448) 
ds_read_b64_tr_b8 v[48:49], v9 offset:544 // (op 592) Transpose load local data 


ds_read_b64_tr_b8 v[50:51], v9 offset:4128 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:4640 // (op 592) Transpose load local data 


ds_read_u8 v13, v8 offset:24608 // (op 595) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[46:53], a[4:7], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 596) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 584) 
s_mov_b32 s8, s18 // (op 584) convert

s_cmp_lt_u32 s7, s8 // (op 584) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_584 // (op 584) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_584 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_584:








s_barrier  // (op 3621) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_218
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


buffer_store_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 778) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_218:


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




s_lshl_b32 s15, s4, 7 // (op 800) 
v_add_u32 v0, v6, s15 // (op 800) 


buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 800) Load value

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


buffer_store_dwordx4 v[16:19], v0, s[64:67], 0 offen // (op 799) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_218:
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
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:










































s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_end:
.size RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2, .LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_end-RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 72
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
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 6
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
  - .name: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
    .symbol: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 91
    .vgpr_count: 72
    .agpr_count: 8
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

