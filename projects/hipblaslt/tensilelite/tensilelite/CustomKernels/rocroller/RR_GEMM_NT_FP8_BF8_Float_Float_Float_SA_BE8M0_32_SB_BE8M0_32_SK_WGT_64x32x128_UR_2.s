// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2,@function
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 2742) 
v_and_b32 v3, 63, v1 // (op 2742) 
v_sub_u32 v4, v3, v0 // (op 2742) 

v_lshrrev_b32 v3, 1, v4 // (op 2742) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 2742) 


v_and_b32 v0, 1, v4 // (op 2742) 

v_lshlrev_b32 v3, 3, v0 // (op 2742) 

v_lshrrev_b32 v0, 6, v1 // (op 2742) 
v_and_b32 v4, 1, v0 // (op 2742) 

v_lshl_add_u32 v0, v4, 4, v3 // (op 2742) 


v_add_u32 v3, v5, v0 // (op 2742) 


v_and_b32 v0, 48, v1 // (op 3000) 
v_and_b32 v4, 63, v1 // (op 3000) 
v_sub_u32 v5, v4, v0 // (op 3000) 

v_lshrrev_b32 v4, 1, v5 // (op 3000) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 3000) 


v_and_b32 v0, 1, v5 // (op 3000) 

v_lshlrev_b32 v4, 3, v0 // (op 3000) 

v_lshrrev_b32 v0, 6, v1 // (op 3000) 
v_and_b32 v5, 1, v0 // (op 3000) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 3000) 


v_add_u32 v4, v6, v0 // (op 3000) 


v_and_b32 v0, 48, v1 // (op 3214) 
v_and_b32 v5, 63, v1 // (op 3214) 
v_sub_u32 v6, v5, v0 // (op 3214) 

v_lshrrev_b32 v5, 1, v6 // (op 3214) 
v_add_lshl_u32 v7, v0, v5, 5 // (op 3214) 


v_and_b32 v0, 1, v6 // (op 3214) 

v_lshlrev_b32 v5, 3, v0 // (op 3214) 

v_lshrrev_b32 v0, 6, v1 // (op 3214) 
v_and_b32 v6, 1, v0 // (op 3214) 

v_lshl_add_u32 v0, v6, 4, v5 // (op 3214) 


v_add_u32 v5, v7, v0 // (op 3214) 


v_and_b32 v0, 48, v1 // (op 3326) 
v_and_b32 v6, 63, v1 // (op 3326) 
v_sub_u32 v7, v6, v0 // (op 3326) 

v_lshrrev_b32 v6, 1, v7 // (op 3326) 
v_add_lshl_u32 v8, v0, v6, 5 // (op 3326) 


v_and_b32 v0, 1, v7 // (op 3326) 

v_lshlrev_b32 v6, 3, v0 // (op 3326) 

v_lshrrev_b32 v0, 6, v1 // (op 3326) 
v_and_b32 v7, 1, v0 // (op 3326) 

v_lshl_add_u32 v0, v7, 4, v6 // (op 3326) 


v_add_u32 v6, v8, v0 // (op 3326) 


v_and_b32 v0, 48, v1 // (op 2672) 
v_and_b32 v7, 63, v1 // (op 2672) 
v_sub_u32 v8, v7, v0 // (op 2672) 

v_lshrrev_b32 v7, 1, v8 // (op 2672) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 2672) 


v_and_b32 v0, 1, v8 // (op 2672) 

v_lshlrev_b32 v7, 3, v0 // (op 2672) 

v_lshrrev_b32 v0, 7, v1 // (op 2672) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 2672) 


v_add_u32 v0, v9, v8 // (op 2672) 


v_lshrrev_b32 v7, 7, v1 // (op 2813) 
v_and_b32 v8, 15, v1 // (op 2813) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 2813) 


v_and_b32 v7, 63, v1 // (op 2813) 
v_lshrrev_b32 v8, 4, v7 // (op 2813) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 2813) 


v_and_b32 v8, 48, v1 // (op 2930) 
v_and_b32 v9, 63, v1 // (op 2930) 
v_sub_u32 v10, v9, v8 // (op 2930) 

v_lshrrev_b32 v9, 1, v10 // (op 2930) 
v_add_lshl_u32 v11, v8, v9, 6 // (op 2930) 


v_and_b32 v8, 1, v10 // (op 2930) 

v_lshlrev_b32 v9, 3, v8 // (op 2930) 

v_lshrrev_b32 v8, 7, v1 // (op 2930) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2930) 


v_add_u32 v8, v11, v10 // (op 2930) 


v_lshrrev_b32 v9, 7, v1 // (op 3071) 
v_and_b32 v10, 15, v1 // (op 3071) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 3071) 


v_and_b32 v9, 63, v1 // (op 3071) 
v_lshrrev_b32 v10, 4, v9 // (op 3071) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 3071) 


v_and_b32 v10, 48, v1 // (op 3175) 
v_and_b32 v11, 63, v1 // (op 3175) 
v_sub_u32 v12, v11, v10 // (op 3175) 

v_lshrrev_b32 v11, 1, v12 // (op 3175) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 3175) 


v_and_b32 v10, 1, v12 // (op 3175) 

v_lshlrev_b32 v11, 3, v10 // (op 3175) 

v_lshrrev_b32 v10, 7, v1 // (op 3175) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 3175) 


v_add_u32 v10, v13, v12 // (op 3175) 


v_lshrrev_b32 v11, 7, v1 // (op 3243) 
v_and_b32 v12, 15, v1 // (op 3243) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 3243) 


v_and_b32 v11, 63, v1 // (op 3243) 
v_lshrrev_b32 v12, 4, v11 // (op 3243) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 3243) 


v_and_b32 v12, 48, v1 // (op 3287) 
v_and_b32 v13, 63, v1 // (op 3287) 
v_sub_u32 v14, v13, v12 // (op 3287) 

v_lshrrev_b32 v13, 1, v14 // (op 3287) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3287) 


v_and_b32 v12, 1, v14 // (op 3287) 

v_lshlrev_b32 v13, 3, v12 // (op 3287) 

v_lshrrev_b32 v12, 7, v1 // (op 3287) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 3287) 


v_add_u32 v12, v15, v14 // (op 3287) 


v_lshrrev_b32 v13, 7, v1 // (op 3355) 
v_and_b32 v14, 15, v1 // (op 3355) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 3355) 


v_and_b32 v13, 63, v1 // (op 3355) 
v_lshrrev_b32 v14, 4, v13 // (op 3355) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 3355) 


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
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 7, v1 // (op 2588) 
s_mul_i32 s52, s16, s8 // (op 2588) 
s_add_u32 s52, s52, s0 // (op 2588) 
v_and_b32 v17, 63, v1 // (op 2588) 
s_mul_hi_u32 s53, s52, s13 // (op 2588) 
v_lshrrev_b32 v18, 4, v17 // (op 2588) 
s_sub_u32 s54, s52, s53 // (op 2588) 
v_lshlrev_b32 v19, 2, v18 // (op 2588) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2588) 


s_lshr_b32 s55, s54, 1 // (op 2588) 

s_add_u32 s54, s55, s53 // (op 2588) 


v_lshrrev_b32 v16, 2, v17 // (op 2588) 

s_lshr_b32 s53, s54, s11 // (op 2588) 

v_and_b32 v17, 3, v16 // (op 2588) 

s_and_b32 s54, s11, 2147483648 // (op 2588) 
v_and_b32 v16, 3, v1 // (op 2588) 
s_cmp_eq_i32 s54, 0 // (op 2588) 
s_mov_b32 s55, scc // (op 2588) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 2588) 


s_cmp_eq_i32 s55, 0 // (op 2588) 
s_cselect_b32 s54, s52, s53 // (op 2588) 



v_lshrrev_b32 v16, 6, v1 // (op 2588) 
s_mul_hi_u32 s52, s54, s12 // (op 2588) 
v_and_b32 v17, 1, v16 // (op 2588) 

s_sub_u32 s53, s54, s52 // (op 2588) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 2588) 


s_lshr_b32 s55, s53, 1 // (op 2588) 

s_add_u32 s53, s55, s52 // (op 2588) 


s_lshr_b32 s52, s53, s10 // (op 2588) 

s_and_b32 s53, s10, 2147483648 // (op 2588) 
s_cmp_eq_i32 s53, 0 // (op 2588) 
s_mov_b32 s55, scc // (op 2588) 

s_cmp_eq_i32 s55, 0 // (op 2588) 
s_cselect_b32 s53, s54, s52 // (op 2588) 


s_mul_i32 s52, s53, s14 // (op 2588) 
v_lshl_add_u32 v17, s53, 6, v18 // (op 2588) 


s_sub_u32 s53, s54, s52 // (op 2588) 


v_lshl_add_u32 v18, s53, 5, v16 // (op 2588) 


v_mul_lo_u32 v16, v18, s4 // (op 2588) 

v_add_lshl_u32 v18, v17, v16, 2 // (op 2588) 


s_mul_i32 s53, s4, s23 // (op 2590) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2590) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2590) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2590) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2590) most significant: sum
s_add_u32 s53, s53, s55 // (op 2590) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2590) 

s_mov_b32 s56, s20 // (op 2590) reg expression
s_mov_b32 s57, s21 // (op 2590) reg expression
s_mov_b32 s58, s54 // (op 2590) reg expression
s_mov_b32 s59, 131072 // (op 2590) call()

v_and_b32 v16, 48, v1 // (op 2689) 
v_and_b32 v17, 63, v1 // (op 2689) 
v_sub_u32 v19, v17, v16 // (op 2689) 

v_lshrrev_b32 v17, 1, v19 // (op 2689) 
v_add_lshl_u32 v20, v16, v17, 6 // (op 2689) 


v_and_b32 v16, 1, v19 // (op 2689) 

v_lshlrev_b32 v17, 3, v16 // (op 2689) 

v_lshrrev_b32 v16, 7, v1 // (op 2689) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 2689) 


v_add_u32 v16, v20, v19 // (op 2689) 


v_and_b32 v17, 48, v1 // (op 2754) 
v_and_b32 v19, 63, v1 // (op 2754) 
v_sub_u32 v20, v19, v17 // (op 2754) 

v_lshrrev_b32 v19, 1, v20 // (op 2754) 
v_add_lshl_u32 v21, v17, v19, 5 // (op 2754) 


v_and_b32 v17, 1, v20 // (op 2754) 

v_lshlrev_b32 v19, 3, v17 // (op 2754) 

v_lshrrev_b32 v17, 6, v1 // (op 2754) 
v_and_b32 v20, 1, v17 // (op 2754) 

v_lshl_add_u32 v17, v20, 4, v19 // (op 2754) 


v_add_u32 v19, v21, v17 // (op 2754) 


v_lshrrev_b32 v17, 7, v1 // (op 2830) 
v_and_b32 v20, 15, v1 // (op 2830) 
v_lshl_add_u32 v21, v17, 4, v20 // (op 2830) 


v_and_b32 v17, 63, v1 // (op 2830) 
v_lshrrev_b32 v20, 4, v17 // (op 2830) 

v_lshl_add_u32 v17, v20, 6, v21 // (op 2830) 


v_lshrrev_b32 v20, 7, v1 // (op 2848) 
s_mul_i32 s52, s16, s8 // (op 2848) 
s_add_u32 s52, s52, s0 // (op 2848) 
v_and_b32 v21, 63, v1 // (op 2848) 
s_mul_hi_u32 s53, s52, s13 // (op 2848) 
v_lshrrev_b32 v22, 4, v21 // (op 2848) 
s_sub_u32 s54, s52, s53 // (op 2848) 
v_lshlrev_b32 v23, 2, v22 // (op 2848) 

s_lshr_b32 s55, s54, 1 // (op 2848) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2848) 


s_add_u32 s54, s55, s53 // (op 2848) 


v_lshrrev_b32 v20, 2, v21 // (op 2848) 

s_lshr_b32 s53, s54, s11 // (op 2848) 

v_and_b32 v21, 3, v20 // (op 2848) 

s_and_b32 s54, s11, 2147483648 // (op 2848) 
v_and_b32 v20, 3, v1 // (op 2848) 
s_cmp_eq_i32 s54, 0 // (op 2848) 
s_mov_b32 s55, scc // (op 2848) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 2848) 


s_cmp_eq_i32 s55, 0 // (op 2848) 
s_cselect_b32 s54, s52, s53 // (op 2848) 



v_lshrrev_b32 v20, 6, v1 // (op 2848) 
s_mul_hi_u32 s55, s54, s12 // (op 2848) 
v_and_b32 v21, 1, v20 // (op 2848) 

s_sub_u32 s52, s54, s55 // (op 2848) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 2848) 


s_lshr_b32 s53, s52, 1 // (op 2848) 

s_add_u32 s52, s53, s55 // (op 2848) 


s_lshr_b32 s53, s52, s10 // (op 2848) 

s_and_b32 s52, s10, 2147483648 // (op 2848) 
s_cmp_eq_i32 s52, 0 // (op 2848) 
s_mov_b32 s55, scc // (op 2848) 

s_cmp_eq_i32 s55, 0 // (op 2848) 
s_cselect_b32 s52, s54, s53 // (op 2848) 


s_mul_i32 s53, s52, s14 // (op 2848) 
v_lshl_add_u32 v21, s52, 6, v22 // (op 2848) 


s_sub_u32 s52, s54, s53 // (op 2848) 


v_lshl_add_u32 v22, s52, 5, v20 // (op 2848) 


v_mul_lo_u32 v20, v22, s24 // (op 2848) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 2848) 


s_mul_i32 s53, s24, s23 // (op 2850) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2850) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2850) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2850) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2850) most significant: sum
s_add_u32 s53, s53, s55 // (op 2850) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2850) 

s_mov_b32 s60, s26 // (op 2850) reg expression
s_mov_b32 s61, s27 // (op 2850) reg expression
s_mov_b32 s62, s54 // (op 2850) reg expression
s_mov_b32 s63, 131072 // (op 2850) call()

v_lshrrev_b32 v20, 5, v1 // (op 3106) 
v_lshlrev_b32 v21, 2, v20 // (op 3106) 

v_lshl_add_u32 v20, s16, 6, v21 // (op 3106) 

v_and_b32 v21, -32, v1 // (op 3106) 
v_sub_u32 v23, v1, v21 // (op 3106) 

v_lshl_add_u32 v21, v20, 5, v23 // (op 3106) 


v_lshlrev_b32 v20, 2, v21 // (op 3106) 

s_lshl_b32 s52, s15, 13 // (op 3108) 
s_mov_b32 s64, s28 // (op 3108) reg expression
s_mov_b32 s65, s29 // (op 3108) reg expression
s_mov_b32 s66, s52 // (op 3108) reg expression
s_mov_b32 s67, 131072 // (op 3108) call()

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
v_accvgpr_write a4, 0.00000 // (op 992) call()
v_accvgpr_write a5, 0.00000 // (op 992) call()
v_accvgpr_write a6, 0.00000 // (op 992) call()
v_accvgpr_write a7, 0.00000 // (op 992) call()
v_and_b32 v21, -4, v1 // (op 2160) 
s_mul_i32 s52, s16, s8 // (op 2160) 
s_add_u32 s52, s52, s0 // (op 2160) 
v_sub_u32 v23, v1, v21 // (op 2160) 

s_mul_hi_u32 s53, s52, s13 // (op 2160) 
v_lshlrev_b32 v21, 4, v23 // (op 2160) 

s_sub_u32 s54, s52, s53 // (op 2160) 
v_lshrrev_b32 v23, 2, v1 // (op 2160) 
s_lshr_b32 s55, s54, 1 // (op 2160) 

s_add_u32 s54, s55, s53 // (op 2160) 


s_lshr_b32 s53, s54, s11 // (op 2160) 

s_and_b32 s54, s11, 2147483648 // (op 2160) 
s_cmp_eq_i32 s54, 0 // (op 2160) 
s_mov_b32 s55, scc // (op 2160) 

s_cmp_eq_i32 s55, 0 // (op 2160) 
s_cselect_b32 s54, s52, s53 // (op 2160) 


s_mul_hi_u32 s53, s54, s12 // (op 2160) 
s_sub_u32 s55, s54, s53 // (op 2160) 
s_lshr_b32 s68, s55, 1 // (op 2160) 

s_add_u32 s55, s68, s53 // (op 2160) 


s_lshr_b32 s53, s55, s10 // (op 2160) 

s_and_b32 s55, s10, 2147483648 // (op 2160) 
s_cmp_eq_i32 s55, 0 // (op 2160) 
s_mov_b32 s68, scc // (op 2160) 

s_cmp_eq_i32 s68, 0 // (op 2160) 
s_cselect_b32 s55, s54, s53 // (op 2160) 


v_lshl_add_u32 v24, s55, 6, v21 // (op 2160) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 2160) 
v_mov_b32 v27, 0 // (op 2160) convert
v_mov_b32 v26, v24 // (op 2160) convert

s_mul_i32 s53, s54, s68 // (op 2160) 


s_sub_u32 s54, s52, s53 // (op 2160) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2160) 


v_mul_lo_u32 v25, v21, s31 // (op 2160) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 2160) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 2160) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2160) most significant: sum


v_add_co_u32 v28, s[52:53], v26, v24 // (op 2160) least significant half
v_addc_co_u32 v29, s[52:53], v27, v25, s[52:53] // (op 2160) most significant half



s_mul_i32 s53, s30, s3 // (op 2162) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2162) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2162) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2162) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2162) most significant: sum
s_add_u32 s53, s53, s55 // (op 2162) most significant: sum


s_mov_b32 s68, s32 // (op 2162) reg expression
s_mov_b32 s69, s33 // (op 2162) reg expression
s_mov_b32 s70, s52 // (op 2162) reg expression
s_mov_b32 s71, 131072 // (op 2162) call()

v_and_b32 v21, -4, v1 // (op 2179) 
s_mul_i32 s52, s16, s8 // (op 2179) 
s_add_u32 s52, s52, s0 // (op 2179) 
v_sub_u32 v23, v1, v21 // (op 2179) 

s_mul_hi_u32 s53, s52, s13 // (op 2179) 
v_lshlrev_b32 v21, 4, v23 // (op 2179) 

s_sub_u32 s54, s52, s53 // (op 2179) 
v_lshrrev_b32 v23, 2, v1 // (op 2179) 
s_lshr_b32 s55, s54, 1 // (op 2179) 

s_add_u32 s54, s55, s53 // (op 2179) 


s_lshr_b32 s53, s54, s11 // (op 2179) 

s_and_b32 s54, s11, 2147483648 // (op 2179) 
s_cmp_eq_i32 s54, 0 // (op 2179) 
s_mov_b32 s55, scc // (op 2179) 

s_cmp_eq_i32 s55, 0 // (op 2179) 
s_cselect_b32 s54, s52, s53 // (op 2179) 


s_mul_hi_u32 s53, s54, s12 // (op 2179) 
s_sub_u32 s55, s54, s53 // (op 2179) 
s_lshr_b32 s72, s55, 1 // (op 2179) 

s_add_u32 s55, s72, s53 // (op 2179) 


s_lshr_b32 s53, s55, s10 // (op 2179) 

s_and_b32 s55, s10, 2147483648 // (op 2179) 
s_cmp_eq_i32 s55, 0 // (op 2179) 
s_mov_b32 s72, scc // (op 2179) 

s_cmp_eq_i32 s72, 0 // (op 2179) 
s_cselect_b32 s55, s54, s53 // (op 2179) 


v_lshl_add_u32 v24, s55, 6, v21 // (op 2179) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2179) 
v_mov_b32 v27, 0 // (op 2179) convert
v_mov_b32 v26, v24 // (op 2179) convert

s_mul_i32 s53, s54, s72 // (op 2179) 


s_sub_u32 s54, s52, s53 // (op 2179) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2179) 


v_mul_lo_u32 v25, v21, s31 // (op 2179) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 2179) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 2179) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2179) most significant: sum


v_add_co_u32 v30, s[52:53], v26, v24 // (op 2179) least significant half
v_addc_co_u32 v31, s[52:53], v27, v25, s[52:53] // (op 2179) most significant half



s_mul_i32 s53, s30, s3 // (op 2181) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2181) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2181) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2181) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2181) most significant: sum
s_add_u32 s53, s53, s55 // (op 2181) most significant: sum


s_mov_b32 s68, s32 // (op 2181) reg expression
s_mov_b32 s69, s33 // (op 2181) reg expression
s_mov_b32 s70, s52 // (op 2181) reg expression
s_mov_b32 s71, 131072 // (op 2181) call()

v_and_b32 v21, -2, v1 // (op 2288) 
s_mul_i32 s52, s16, s8 // (op 2288) 
s_add_u32 s52, s52, s0 // (op 2288) 
v_sub_u32 v23, v1, v21 // (op 2288) 

s_mul_hi_u32 s53, s52, s13 // (op 2288) 
v_lshlrev_b32 v21, 4, v23 // (op 2288) 

s_sub_u32 s54, s52, s53 // (op 2288) 
v_lshrrev_b32 v23, 1, v1 // (op 2288) 
s_lshr_b32 s55, s54, 1 // (op 2288) 

s_add_u32 s54, s55, s53 // (op 2288) 


s_lshr_b32 s53, s54, s11 // (op 2288) 

s_and_b32 s54, s11, 2147483648 // (op 2288) 
s_cmp_eq_i32 s54, 0 // (op 2288) 
s_mov_b32 s55, scc // (op 2288) 

s_cmp_eq_i32 s55, 0 // (op 2288) 
s_cselect_b32 s54, s52, s53 // (op 2288) 


s_mul_hi_u32 s53, s54, s12 // (op 2288) 
s_sub_u32 s55, s54, s53 // (op 2288) 
s_lshr_b32 s72, s55, 1 // (op 2288) 

s_add_u32 s55, s72, s53 // (op 2288) 


s_lshr_b32 s53, s55, s10 // (op 2288) 

s_and_b32 s55, s10, 2147483648 // (op 2288) 
s_cmp_eq_i32 s55, 0 // (op 2288) 
s_mov_b32 s72, scc // (op 2288) 

s_cmp_eq_i32 s72, 0 // (op 2288) 
s_cselect_b32 s55, s54, s53 // (op 2288) 


s_mul_i32 s53, s55, s14 // (op 2288) 

s_sub_u32 s55, s54, s53 // (op 2288) 

v_lshl_add_u32 v24, s55, 5, v21 // (op 2288) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2288) 
v_mov_b32 v27, 0 // (op 2288) convert
v_mov_b32 v26, v24 // (op 2288) convert

s_mul_i32 s53, s54, s72 // (op 2288) 


s_sub_u32 s54, s52, s53 // (op 2288) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2288) 


v_mul_lo_u32 v25, v21, s35 // (op 2288) most significant: low of low * high
v_mul_hi_u32 v23, v21, s34 // (op 2288) most significant: high of low * low
v_mul_lo_u32 v24, v21, s34 // (op 2288) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2288) most significant: sum

v_add_co_u32 v24, s[52:53], v24, v26 // (op 2288) least significant half
v_addc_co_u32 v25, s[52:53], v25, v27, s[52:53] // (op 2288) most significant half



s_mul_i32 s53, s34, s3 // (op 2290) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2290) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2290) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2290) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2290) most significant: sum
s_add_u32 s53, s53, s55 // (op 2290) most significant: sum


s_mov_b32 s72, s36 // (op 2290) reg expression
s_mov_b32 s73, s37 // (op 2290) reg expression
s_mov_b32 s74, s52 // (op 2290) reg expression
s_mov_b32 s75, 131072 // (op 2290) call()

v_and_b32 v21, -2, v1 // (op 2307) 
s_mul_i32 s52, s16, s8 // (op 2307) 
s_add_u32 s52, s52, s0 // (op 2307) 
v_sub_u32 v23, v1, v21 // (op 2307) 

s_mul_hi_u32 s53, s52, s13 // (op 2307) 
v_lshlrev_b32 v21, 4, v23 // (op 2307) 

s_sub_u32 s54, s52, s53 // (op 2307) 
v_lshrrev_b32 v23, 1, v1 // (op 2307) 
s_lshr_b32 s55, s54, 1 // (op 2307) 

s_add_u32 s54, s55, s53 // (op 2307) 


s_lshr_b32 s53, s54, s11 // (op 2307) 

s_and_b32 s54, s11, 2147483648 // (op 2307) 
s_cmp_eq_i32 s54, 0 // (op 2307) 
s_mov_b32 s55, scc // (op 2307) 

s_cmp_eq_i32 s55, 0 // (op 2307) 
s_cselect_b32 s54, s52, s53 // (op 2307) 


s_mul_hi_u32 s53, s54, s12 // (op 2307) 
s_sub_u32 s55, s54, s53 // (op 2307) 
s_lshr_b32 s76, s55, 1 // (op 2307) 

s_add_u32 s55, s76, s53 // (op 2307) 


s_lshr_b32 s53, s55, s10 // (op 2307) 

s_and_b32 s55, s10, 2147483648 // (op 2307) 
s_cmp_eq_i32 s55, 0 // (op 2307) 
s_mov_b32 s76, scc // (op 2307) 

s_cmp_eq_i32 s76, 0 // (op 2307) 
s_cselect_b32 s55, s54, s53 // (op 2307) 


s_mul_i32 s53, s55, s14 // (op 2307) 

s_sub_u32 s55, s54, s53 // (op 2307) 

v_lshl_add_u32 v26, s55, 5, v21 // (op 2307) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2307) 
v_mov_b32 v33, 0 // (op 2307) convert
v_mov_b32 v32, v26 // (op 2307) convert

s_mul_i32 s53, s54, s76 // (op 2307) 


s_sub_u32 s54, s52, s53 // (op 2307) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 2307) 


v_mul_lo_u32 v27, v21, s35 // (op 2307) most significant: low of low * high
v_mul_hi_u32 v23, v21, s34 // (op 2307) most significant: high of low * low
v_mul_lo_u32 v26, v21, s34 // (op 2307) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 2307) most significant: sum

v_add_co_u32 v26, s[52:53], v26, v32 // (op 2307) least significant half
v_addc_co_u32 v27, s[52:53], v27, v33, s[52:53] // (op 2307) most significant half



s_mul_i32 s53, s34, s3 // (op 2309) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2309) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2309) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2309) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2309) most significant: sum
s_add_u32 s53, s53, s55 // (op 2309) most significant: sum


s_mov_b32 s72, s36 // (op 2309) reg expression
s_mov_b32 s73, s37 // (op 2309) reg expression
s_mov_b32 s74, s52 // (op 2309) reg expression
s_mov_b32 s75, 131072 // (op 2309) call()

v_and_b32 v21, -64, v1 // (op 2416) 
s_mul_i32 s52, s16, s8 // (op 2416) 
s_add_u32 s52, s52, s0 // (op 2416) 
v_sub_u32 v23, v1, v21 // (op 2416) 

s_mul_hi_u32 s53, s52, s13 // (op 2416) 
v_lshrrev_b32 v21, 6, v1 // (op 2416) 
s_sub_u32 s54, s52, s53 // (op 2416) 
s_lshr_b32 s55, s54, 1 // (op 2416) 

s_add_u32 s54, s55, s53 // (op 2416) 


s_lshr_b32 s53, s54, s11 // (op 2416) 

s_and_b32 s54, s11, 2147483648 // (op 2416) 
s_cmp_eq_i32 s54, 0 // (op 2416) 
s_mov_b32 s55, scc // (op 2416) 

s_cmp_eq_i32 s55, 0 // (op 2416) 
s_cselect_b32 s54, s52, s53 // (op 2416) 


s_mul_hi_u32 s53, s54, s12 // (op 2416) 
s_sub_u32 s55, s54, s53 // (op 2416) 
s_lshr_b32 s76, s55, 1 // (op 2416) 

s_add_u32 s55, s76, s53 // (op 2416) 


s_lshr_b32 s53, s55, s10 // (op 2416) 

s_and_b32 s55, s10, 2147483648 // (op 2416) 
s_cmp_eq_i32 s55, 0 // (op 2416) 
s_mov_b32 s76, scc // (op 2416) 

s_cmp_eq_i32 s76, 0 // (op 2416) 
s_cselect_b32 s55, s54, s53 // (op 2416) 


v_lshl_add_u32 v32, s55, 6, v23 // (op 2416) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2416) 
s_mul_i32 s53, s54, s76 // (op 2416) 


s_sub_u32 s54, s52, s53 // (op 2416) 


v_lshl_add_u32 v23, s54, 2, v21 // (op 2416) 


v_mul_lo_u32 v21, v23, s38 // (op 2416) 

v_add_u32 v23, v32, v21 // (op 2416) 


s_mul_i32 s53, s38, s43 // (op 2418) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2418) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2418) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2418) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2418) most significant: sum
s_add_u32 s53, s53, s55 // (op 2418) most significant: sum


s_mov_b32 s76, s40 // (op 2418) reg expression
s_mov_b32 s77, s41 // (op 2418) reg expression
s_mov_b32 s78, s52 // (op 2418) reg expression
s_mov_b32 s79, 131072 // (op 2418) call()

v_and_b32 v21, -64, v1 // (op 2435) 
s_mul_i32 s52, s16, s8 // (op 2435) 
s_add_u32 s52, s52, s0 // (op 2435) 
v_sub_u32 v32, v1, v21 // (op 2435) 

s_mul_hi_u32 s53, s52, s13 // (op 2435) 
v_lshrrev_b32 v21, 6, v1 // (op 2435) 
s_sub_u32 s54, s52, s53 // (op 2435) 
s_lshr_b32 s55, s54, 1 // (op 2435) 

s_add_u32 s54, s55, s53 // (op 2435) 


s_lshr_b32 s53, s54, s11 // (op 2435) 

s_and_b32 s54, s11, 2147483648 // (op 2435) 
s_cmp_eq_i32 s54, 0 // (op 2435) 
s_mov_b32 s55, scc // (op 2435) 

s_cmp_eq_i32 s55, 0 // (op 2435) 
s_cselect_b32 s54, s52, s53 // (op 2435) 


s_mul_hi_u32 s53, s54, s12 // (op 2435) 
s_sub_u32 s55, s54, s53 // (op 2435) 
s_lshr_b32 s80, s55, 1 // (op 2435) 

s_add_u32 s55, s80, s53 // (op 2435) 


s_lshr_b32 s53, s55, s10 // (op 2435) 

s_and_b32 s55, s10, 2147483648 // (op 2435) 
s_cmp_eq_i32 s55, 0 // (op 2435) 
s_mov_b32 s80, scc // (op 2435) 

s_cmp_eq_i32 s80, 0 // (op 2435) 
s_cselect_b32 s55, s54, s53 // (op 2435) 


v_lshl_add_u32 v33, s55, 6, v32 // (op 2435) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2435) 
s_mul_i32 s53, s54, s80 // (op 2435) 


s_sub_u32 s54, s52, s53 // (op 2435) 


v_lshl_add_u32 v32, s54, 2, v21 // (op 2435) 


v_mul_lo_u32 v21, v32, s38 // (op 2435) 

v_add_u32 v32, v33, v21 // (op 2435) 


s_mul_i32 s53, s38, s43 // (op 2437) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2437) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2437) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2437) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2437) most significant: sum
s_add_u32 s53, s53, s55 // (op 2437) most significant: sum


s_mov_b32 s76, s40 // (op 2437) reg expression
s_mov_b32 s77, s41 // (op 2437) reg expression
s_mov_b32 s78, s52 // (op 2437) reg expression
s_mov_b32 s79, 131072 // (op 2437) call()

v_lshrrev_b32 v21, 6, v1 // (op 2544) 
s_mul_i32 s52, s16, s8 // (op 2544) 
s_add_u32 s52, s52, s0 // (op 2544) 
v_and_b32 v33, 1, v21 // (op 2544) 

s_mul_hi_u32 s53, s52, s13 // (op 2544) 
v_and_b32 v21, 15, v1 // (op 2544) 
s_sub_u32 s54, s52, s53 // (op 2544) 
v_lshl_add_u32 v34, v33, 4, v21 // (op 2544) 


s_lshr_b32 s55, s54, 1 // (op 2544) 

v_and_b32 v21, 63, v1 // (op 2544) 
s_add_u32 s54, s55, s53 // (op 2544) 


v_lshrrev_b32 v33, 4, v21 // (op 2544) 

s_lshr_b32 s53, s54, s11 // (op 2544) 

s_and_b32 s54, s11, 2147483648 // (op 2544) 
s_cmp_eq_i32 s54, 0 // (op 2544) 
s_mov_b32 s55, scc // (op 2544) 

s_cmp_eq_i32 s55, 0 // (op 2544) 
s_cselect_b32 s54, s52, s53 // (op 2544) 


s_mul_hi_u32 s53, s54, s12 // (op 2544) 
s_sub_u32 s55, s54, s53 // (op 2544) 
s_lshr_b32 s80, s55, 1 // (op 2544) 

s_add_u32 s55, s80, s53 // (op 2544) 


s_lshr_b32 s53, s55, s10 // (op 2544) 

s_and_b32 s55, s10, 2147483648 // (op 2544) 
s_cmp_eq_i32 s55, 0 // (op 2544) 
s_mov_b32 s80, scc // (op 2544) 

s_cmp_eq_i32 s80, 0 // (op 2544) 
s_cselect_b32 s55, s54, s53 // (op 2544) 


s_mul_i32 s53, s55, s14 // (op 2544) 

s_sub_u32 s55, s54, s53 // (op 2544) 

v_lshl_add_u32 v21, s55, 5, v34 // (op 2544) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2544) 
s_mul_i32 s53, s54, s80 // (op 2544) 


s_sub_u32 s54, s52, s53 // (op 2544) 


v_lshl_add_u32 v34, s54, 2, v33 // (op 2544) 


v_mul_lo_u32 v33, v34, s44 // (op 2544) 
v_add_u32 v33, v33, v21 // (op 2544) 


s_mul_i32 s53, s44, s43 // (op 2546) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2546) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2546) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2546) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2546) most significant: sum
s_add_u32 s53, s53, s55 // (op 2546) most significant: sum


s_mov_b32 s80, s46 // (op 2546) reg expression
s_mov_b32 s81, s47 // (op 2546) reg expression
s_mov_b32 s82, s52 // (op 2546) reg expression
s_mov_b32 s83, 131072 // (op 2546) call()

v_lshrrev_b32 v21, 6, v1 // (op 2563) 
s_mul_i32 s52, s16, s8 // (op 2563) 
s_add_u32 s52, s52, s0 // (op 2563) 
v_and_b32 v34, 1, v21 // (op 2563) 

s_mul_hi_u32 s53, s52, s13 // (op 2563) 
v_and_b32 v21, 15, v1 // (op 2563) 
s_sub_u32 s54, s52, s53 // (op 2563) 
v_lshl_add_u32 v35, v34, 4, v21 // (op 2563) 


s_lshr_b32 s55, s54, 1 // (op 2563) 

v_and_b32 v21, 63, v1 // (op 2563) 
s_add_u32 s54, s55, s53 // (op 2563) 


v_lshrrev_b32 v34, 4, v21 // (op 2563) 

s_lshr_b32 s53, s54, s11 // (op 2563) 

s_and_b32 s54, s11, 2147483648 // (op 2563) 
s_cmp_eq_i32 s54, 0 // (op 2563) 
s_mov_b32 s55, scc // (op 2563) 

s_cmp_eq_i32 s55, 0 // (op 2563) 
s_cselect_b32 s54, s52, s53 // (op 2563) 


s_mul_hi_u32 s53, s54, s12 // (op 2563) 
s_sub_u32 s55, s54, s53 // (op 2563) 
s_lshr_b32 s84, s55, 1 // (op 2563) 

s_add_u32 s55, s84, s53 // (op 2563) 


s_lshr_b32 s53, s55, s10 // (op 2563) 

s_and_b32 s55, s10, 2147483648 // (op 2563) 
s_cmp_eq_i32 s55, 0 // (op 2563) 
s_mov_b32 s84, scc // (op 2563) 

s_cmp_eq_i32 s84, 0 // (op 2563) 
s_cselect_b32 s55, s54, s53 // (op 2563) 


s_mul_i32 s53, s55, s14 // (op 2563) 

s_sub_u32 s55, s54, s53 // (op 2563) 

v_lshl_add_u32 v21, s55, 5, v35 // (op 2563) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2563) 
s_mul_i32 s53, s54, s84 // (op 2563) 


s_sub_u32 s54, s52, s53 // (op 2563) 


v_lshl_add_u32 v35, s54, 2, v34 // (op 2563) 


v_mul_lo_u32 v34, v35, s44 // (op 2563) 
v_add_u32 v34, v34, v21 // (op 2563) 


s_mul_i32 s53, s44, s43 // (op 2565) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2565) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2565) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2565) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2565) most significant: sum
s_add_u32 s53, s53, s55 // (op 2565) most significant: sum


s_mov_b32 s80, s46 // (op 2565) reg expression
s_mov_b32 s81, s47 // (op 2565) reg expression
s_mov_b32 s82, s52 // (op 2565) reg expression
s_mov_b32 s83, 131072 // (op 2565) call()


v_mov_b64 v[36:37], s[44:45] // (op 1849) convert

v_mov_b32 v21, v33 // (op 1849) reg expression
v_mov_b32 v35, v21 // (op 1849) 
buffer_load_ubyte v38, v35, s[80:83], 0 offen offset: 0 // (op 1849) Load value



v_and_b32 v21, -2, v1 // (op 2730) 
v_sub_u32 v35, v1, v21 // (op 2730) 

v_lshlrev_b32 v21, 4, v35 // (op 2730) 

v_lshrrev_b32 v35, 1, v1 // (op 2730) 
v_lshl_add_u32 v36, v35, 5, v21 // (op 2730) 


s_nop 0
v_readfirstlane_b32 s52, v36 // (op 2730) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3455) 


v_mov_b64 v[36:37], v[24:25] // (op 2038) reg expression

s_mov_b32 m0, s52 // (op 2038) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2038) Load value direct to lds


v_and_b32 v21, -4, v1 // (op 2655) 
v_sub_u32 v35, v1, v21 // (op 2655) 

v_lshlrev_b32 v21, 4, v35 // (op 2655) 

v_lshrrev_b32 v35, 2, v1 // (op 2655) 
v_lshl_add_u32 v36, v35, 6, v21 // (op 2655) 


s_nop 0
v_readfirstlane_b32 s52, v36 // (op 2655) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3436) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 2043) 
v_mov_b64 v[36:37], s[54:55] // (op 2043) convert


v_mov_b64 v[40:41], v[28:29] // (op 2043) reg expression
s_add_u32 s53, 4096, s52 // (op 2043) 
s_mov_b32 m0, s53 // (op 2043) reg expression
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2043) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v36 // (op 2043) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 2043) most significant half

s_add_u32 m0, m0, 4096 // (op 2043) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2043) Load value direct to lds






v_mov_b32 v21, v23 // (op 1859) reg expression

buffer_load_ubyte v35, v21, s[76:79], 0 offen offset: 0 // (op 1859) Load value


v_mov_b64 v[36:37], s[44:45] // (op 1863) convert

s_lshl_b32 s52, s44, 2 // (op 1863) 
v_add_u32 v21, v33, s52 // (op 1863) 

v_mov_b32 v39, v21 // (op 1863) 
buffer_load_ubyte v40, v39, s[80:83], 0 offen offset: 0 // (op 1863) Load value




v_and_b32 v21, -2, v1 // (op 3314) 
v_sub_u32 v33, v1, v21 // (op 3314) 

v_lshlrev_b32 v21, 4, v33 // (op 3314) 

v_lshrrev_b32 v33, 1, v1 // (op 3314) 
v_lshl_add_u32 v39, v33, 5, v21 // (op 3314) 


s_nop 0
v_readfirstlane_b32 s52, v39 // (op 3314) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3585) 


s_lshl_b64 s[54:55], s[34:35], 7 // (op 2048) 
v_mov_b32 v21, s54 // (op 2048) 
v_mov_b32 v33, s55 // (op 2048) 
v_add_co_u32 v36, s[84:85], v24, v21 // (op 2048) least significant half
v_addc_co_u32 v37, s[84:85], v25, v33, s[84:85] // (op 2048) most significant half




s_add_u32 s53, 12288, s52 // (op 2048) 

s_mov_b32 m0, s53 // (op 2048) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2048) Load value direct to lds




v_and_b32 v21, -4, v1 // (op 3270) 
v_sub_u32 v33, v1, v21 // (op 3270) 

v_lshlrev_b32 v21, 4, v33 // (op 3270) 

v_lshrrev_b32 v33, 2, v1 // (op 3270) 
v_lshl_add_u32 v39, v33, 6, v21 // (op 3270) 


s_nop 0
v_readfirstlane_b32 s52, v39 // (op 3270) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3576) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 2050) 
v_mov_b64 v[24:25], s[54:55] // (op 2050) convert


s_lshl_b64 s[54:55], s[30:31], 7 // (op 2050) 
v_mov_b32 v21, s54 // (op 2050) 
v_mov_b32 v33, s55 // (op 2050) 
v_add_co_u32 v36, s[84:85], v28, v21 // (op 2050) least significant half
v_addc_co_u32 v37, s[84:85], v29, v33, s[84:85] // (op 2050) most significant half




s_add_u32 s53, 16384, s52 // (op 2050) 
s_mov_b32 m0, s53 // (op 2050) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2050) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v24 // (op 2050) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 2050) most significant half

s_add_u32 m0, m0, 4096 // (op 2050) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2050) Load value direct to lds







s_lshl_b32 s52, s38, 2 // (op 1873) 
v_add_u32 v21, v23, s52 // (op 1873) 


buffer_load_ubyte v33, v21, s[76:79], 0 offen offset: 0 // (op 1873) Load value


v_and_b32 v21, -64, v1 // (op 2796) 
v_sub_u32 v39, v1, v21 // (op 2796) 

v_and_b32 v21, 4294967232, v1 // (op 2796) 
v_add_u32 v23, v21, v39 // (op 2796) 


s_barrier  // (op 3474) 



s_waitcnt vmcnt(5)
ds_write_b8 v23, v35 offset:24576 // (op 1877) Store local data 

s_barrier  // (op 3452) 



ds_read_b64_tr_b8 v[42:43], v19 // (op 1900) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v19 offset:256 // (op 1900) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v19 offset:2048 // (op 1900) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v19 offset:2304 // (op 1900) Transpose load local data 
s_barrier  // (op 3433) 



ds_read_b64_tr_b8 v[50:51], v16 offset:4096 // (op 1891) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v16 offset:4608 // (op 1891) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v16 offset:8192 // (op 1891) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v16 offset:8704 // (op 1891) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3471) 


ds_read_u8 v21, v17 offset:24576 // (op 1904) Load local data 



ds_read_b64_tr_b8 v[58:59], v16 offset:4128 // (op 1896) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v16 offset:4640 // (op 1896) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v16 offset:8224 // (op 1896) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v16 offset:8736 // (op 1896) Transpose load local data 


ds_read_u8 v39, v17 offset:24608 // (op 1909) Load local data 
v_and_b32 v41, -4, v1 // (op 2627) 
v_sub_u32 v23, v1, v41 // (op 2627) 

v_lshlrev_b32 v41, 4, v23 // (op 2627) 

v_lshrrev_b32 v23, 2, v1 // (op 2627) 
v_lshl_add_u32 v24, v23, 6, v41 // (op 2627) 


s_nop 0
v_readfirstlane_b32 s52, v24 // (op 2627) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v41, -2, v1 // (op 2702) 
v_sub_u32 v23, v1, v41 // (op 2702) 

v_lshlrev_b32 v41, 4, v23 // (op 2702) 

v_lshrrev_b32 v23, 1, v1 // (op 2702) 
v_lshl_add_u32 v24, v23, 5, v41 // (op 2702) 


s_nop 0
v_readfirstlane_b32 s53, v24 // (op 2702) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v41, -64, v1 // (op 2767) 
v_sub_u32 v23, v1, v41 // (op 2767) 

v_and_b32 v41, 4294967232, v1 // (op 2767) 
v_add_u32 v24, v41, v23 // (op 2767) 


v_and_b32 v23, -4, v1 // (op 3255) 
v_sub_u32 v25, v1, v23 // (op 3255) 

v_lshlrev_b32 v23, 4, v25 // (op 3255) 

v_lshrrev_b32 v25, 2, v1 // (op 3255) 
v_lshl_add_u32 v41, v25, 6, v23 // (op 3255) 


s_nop 0
v_readfirstlane_b32 s54, v41 // (op 3255) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -2, v1 // (op 3299) 
v_sub_u32 v25, v1, v23 // (op 3299) 

v_lshlrev_b32 v23, 4, v25 // (op 3299) 

v_lshrrev_b32 v25, 1, v1 // (op 3299) 
v_lshl_add_u32 v41, v25, 5, v23 // (op 3299) 


s_nop 0
v_readfirstlane_b32 s55, v41 // (op 3299) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 3338) 
v_sub_u32 v25, v1, v23 // (op 3338) 

v_and_b32 v23, 4294967232, v1 // (op 3338) 
v_add_u32 v41, v23, v25 // (op 3338) 


s_mov_b32 s1, 0 // (op 994) call()
s_and_b32 s84, s7, 4294967294 // (op 993) 
s_cmp_lt_u32 s1, s84 // (op 993) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_993 // (op 993) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_993 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_993:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[42:49], a[0:3], v21, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 82) 
s_add_u32 s1, s1, 2 // (op 995) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[42:49], a[4:7], v39, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1029) 

v_mov_b64 v[28:29], s[44:45] // (op 80) convert

v_mov_b32 v23, v34 // (op 80) 
s_lshl_b32 s84, s44, 3 // (op 80) 
v_add_u32 v25, v23, s84 // (op 80) 

v_mov_b32 v36, v25 // (op 80) 
buffer_load_ubyte v38, v36, s[80:83], 0 offen offset: 0 // (op 80) Load value



s_barrier  // (op 3458) 


v_mov_b64 v[28:29], v[26:27] // (op 1963) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 1963) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 1963) 

v_mov_b32 v25, s86 // (op 1963) 
v_mov_b32 v36, s87 // (op 1963) 
v_add_co_u32 v66, s[84:85], v28, v25 // (op 1963) least significant half
v_addc_co_u32 v67, s[84:85], v29, v36, s[84:85] // (op 1963) most significant half





s_mov_b32 m0, s53 // (op 1963) reg expression
buffer_load_dwordx4 v66, s[72:75], 0 offen offset: 0 lds // (op 1963) Load value direct to lds

s_barrier  // (op 3439) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 1958) 
v_mov_b64 v[36:37], s[84:85] // (op 1958) convert


v_mov_b64 v[66:67], v[30:31] // (op 1958) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 1958) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 1958) 

v_mov_b32 v25, s86 // (op 1958) 
v_mov_b32 v68, s87 // (op 1958) 
v_add_co_u32 v70, s[84:85], v66, v25 // (op 1958) least significant half
v_addc_co_u32 v71, s[84:85], v67, v68, s[84:85] // (op 1958) most significant half




s_add_u32 s84, 4096, s52 // (op 1958) 
s_nop 0
s_mov_b32 m0, s84 // (op 1958) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 1958) Load value direct to lds
v_add_co_u32 v70, s[86:87], v70, v36 // (op 1958) least significant half
v_addc_co_u32 v71, s[86:87], 0, v71, s[86:87] // (op 1958) most significant half

s_add_u32 m0, m0, 4096 // (op 1958) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 1958) Load value direct to lds





v_mov_b32 v25, v32 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v36, v25, s84 // (op 17) 


buffer_load_ubyte v35, v36, s[76:79], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 3594) 



s_waitcnt vmcnt(5)
ds_write_b8 v41, v33 offset:24832 // (op 919) Store local data 
s_barrier  // (op 3582) 



ds_read_b64_tr_b8 v[42:43], v6 offset:12288 // (op 912) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:12544 // (op 912) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:14336 // (op 912) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:14592 // (op 912) Transpose load local data 
s_barrier  // (op 3573) 



ds_read_b64_tr_b8 v[50:51], v12 offset:16384 // (op 910) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:16896 // (op 910) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:20480 // (op 910) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v12 offset:20992 // (op 910) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3591) 


ds_read_u8 v21, v13 offset:24832 // (op 914) Load local data 



ds_read_b64_tr_b8 v[58:59], v12 offset:16416 // (op 1036) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v12 offset:16928 // (op 1036) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v12 offset:20512 // (op 1036) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v12 offset:21024 // (op 1036) Transpose load local data 


ds_read_u8 v39, v13 offset:24864 // (op 1040) Load local data 
s_barrier  // (op 3737) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[42:49], a[0:3], v21, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 917) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[42:49], a[4:7], v39, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1030) 

v_mov_b64 v[36:37], s[44:45] // (op 916) convert

s_lshl_b32 s84, s44, 2 // (op 916) 
s_mul_i32 s85, 3, s84 // (op 916) 

v_add_u32 v68, v23, s85 // (op 916) 

v_mov_b32 v69, v68 // (op 916) 
buffer_load_ubyte v40, v69, s[80:83], 0 offen offset: 0 // (op 916) Load value




s_barrier  // (op 3588) 
s_lshl_b32 s84, s44, 3 // (op 2555) 


v_add_u32 v34, v34, s84 // (op 2555) 

s_lshl_b64 s[84:85], s[34:35], 7 // (op 2019) 
s_mul_i32 s87, 3, s85 // (op 2019) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 2019) least significant
s_mul_hi_u32 s88, 3, s84 // (op 2019) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 2019) most significant: sum


v_mov_b32 v23, s86 // (op 2019) 
v_mov_b32 v36, s87 // (op 2019) 
v_add_co_u32 v68, s[84:85], v28, v23 // (op 2019) least significant half
v_addc_co_u32 v69, s[84:85], v29, v36, s[84:85] // (op 2019) most significant half




s_add_u32 s84, 12288, s55 // (op 2019) 

s_nop 0
s_mov_b32 m0, s84 // (op 2019) reg expression
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 2019) Load value direct to lds



s_barrier  // (op 3579) 
s_lshl_b64 s[84:85], s[34:35], 7 // (op 2299) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 2299) 

v_mov_b32 v23, s86 // (op 2299) 
v_mov_b32 v28, s87 // (op 2299) 
s_lshl_b64 s[84:85], s[30:31], 6 // (op 2014) 
v_mov_b64 v[36:37], s[84:85] // (op 2014) convert


s_lshl_b64 s[84:85], s[30:31], 7 // (op 2014) 
s_mul_i32 s89, 3, s85 // (op 2014) most significant: low of low * high
s_mul_i32 s88, 3, s84 // (op 2014) least significant
s_mul_hi_u32 s90, 3, s84 // (op 2014) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 2014) most significant: sum


v_mov_b32 v29, s88 // (op 2014) 
v_mov_b32 v68, s89 // (op 2014) 
v_add_co_u32 v26, s[84:85], v26, v23 // (op 2299) least significant half
v_addc_co_u32 v27, s[84:85], v27, v28, s[84:85] // (op 2299) most significant half




v_add_co_u32 v70, s[84:85], v66, v29 // (op 2014) least significant half
v_addc_co_u32 v71, s[84:85], v67, v68, s[84:85] // (op 2014) most significant half




s_add_u32 s84, 16384, s54 // (op 2014) 
s_nop 0
s_mov_b32 m0, s84 // (op 2014) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2014) Load value direct to lds
v_add_co_u32 v70, s[86:87], v70, v36 // (op 2014) least significant half
v_addc_co_u32 v71, s[86:87], 0, v71, s[86:87] // (op 2014) most significant half

s_add_u32 m0, m0, 4096 // (op 2014) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2014) Load value direct to lds






s_lshl_b32 s84, s38, 2 // (op 921) 
s_mul_i32 s85, 3, s84 // (op 921) 

v_add_u32 v23, v25, s85 // (op 921) 


s_lshl_b64 s[84:85], s[30:31], 7 // (op 2171) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2171) 

v_mov_b32 v28, s86 // (op 2171) 
v_mov_b32 v29, s87 // (op 2171) 
v_add_co_u32 v30, s[84:85], v30, v28 // (op 2171) least significant half
v_addc_co_u32 v31, s[84:85], v31, v29, s[84:85] // (op 2171) most significant half




buffer_load_ubyte v33, v23, s[76:79], 0 offen offset: 0 // (op 921) Load value


s_lshl_b32 s84, s38, 3 // (op 2427) 
s_barrier  // (op 3477) 
v_add_u32 v32, v32, s84 // (op 2427) 




s_waitcnt vmcnt(5)
ds_write_b8 v24, v35 offset:24576 // (op 21) Store local data 
s_barrier  // (op 3464) 



ds_read_b64_tr_b8 v[42:43], v3 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:256 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:2304 // (op 70) Transpose load local data 
s_barrier  // (op 3445) 



ds_read_b64_tr_b8 v[50:51], v0 offset:4096 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:4608 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8704 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3484) 


ds_read_u8 v21, v7 offset:24576 // (op 75) Load local data 



ds_read_b64_tr_b8 v[58:59], v0 offset:4128 // (op 1023) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:4640 // (op 1023) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:8224 // (op 1023) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:8736 // (op 1023) Transpose load local data 


ds_read_u8 v39, v7 offset:24608 // (op 1027) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 993) 
s_cmp_lt_u32 s1, s84 // (op 993) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_993 // (op 993) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_993 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_993:












s_barrier  // (op 3721) 
s_and_b32 s52, s7, 4294967294 // (op 2087) 
v_and_b32 v41, -4, v1 // (op 2087) 
s_add_u32 s53, s0, s52 // (op 2087) 

v_sub_u32 v23, v1, v41 // (op 2087) 

v_lshlrev_b32 v41, 4, v23 // (op 2087) 

s_mul_i32 s52, s16, s8 // (op 2087) 
s_add_u32 s52, s52, s53 // (op 2087) 

v_lshrrev_b32 v23, 2, v1 // (op 2087) 
s_mul_hi_u32 s53, s52, s13 // (op 2087) 
s_sub_u32 s54, s52, s53 // (op 2087) 
s_lshr_b32 s55, s54, 1 // (op 2087) 

s_add_u32 s54, s55, s53 // (op 2087) 


s_lshr_b32 s53, s54, s11 // (op 2087) 

s_and_b32 s54, s11, 2147483648 // (op 2087) 
s_cmp_eq_i32 s54, 0 // (op 2087) 
s_mov_b32 s55, scc // (op 2087) 

s_cmp_eq_i32 s55, 0 // (op 2087) 
s_cselect_b32 s54, s52, s53 // (op 2087) 


s_mul_hi_u32 s53, s54, s12 // (op 2087) 
s_sub_u32 s55, s54, s53 // (op 2087) 
s_lshr_b32 s84, s55, 1 // (op 2087) 

s_add_u32 s55, s84, s53 // (op 2087) 


s_lshr_b32 s53, s55, s10 // (op 2087) 

s_and_b32 s55, s10, 2147483648 // (op 2087) 
s_cmp_eq_i32 s55, 0 // (op 2087) 
s_mov_b32 s84, scc // (op 2087) 

s_cmp_eq_i32 s84, 0 // (op 2087) 
s_cselect_b32 s55, s54, s53 // (op 2087) 


v_lshl_add_u32 v24, s55, 6, v41 // (op 2087) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2087) 
v_mov_b32 v35, 0 // (op 2087) convert
v_mov_b32 v34, v24 // (op 2087) convert

s_mul_i32 s53, s54, s84 // (op 2087) 


s_sub_u32 s54, s52, s53 // (op 2087) 


v_lshl_add_u32 v41, s54, 7, v23 // (op 2087) 


v_mul_lo_u32 v37, v41, s31 // (op 2087) most significant: low of low * high
v_mul_hi_u32 v38, v41, s30 // (op 2087) most significant: high of low * low
v_mul_lo_u32 v36, v41, s30 // (op 2087) least significant: low of low * low
v_add_u32 v37, v37, v38 // (op 2087) most significant: sum


v_add_co_u32 v24, s[52:53], v34, v36 // (op 2087) least significant half
v_addc_co_u32 v25, s[52:53], v35, v37, s[52:53] // (op 2087) most significant half



s_mul_i32 s53, s30, s3 // (op 2089) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2089) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2089) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2089) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2089) most significant: sum
s_add_u32 s53, s53, s55 // (op 2089) most significant: sum


s_mov_b32 s68, s32 // (op 2089) reg expression
s_mov_b32 s69, s33 // (op 2089) reg expression
s_mov_b32 s70, s52 // (op 2089) reg expression
s_mov_b32 s71, 131072 // (op 2089) call()

s_and_b32 s52, s7, 4294967294 // (op 2223) 
v_and_b32 v23, -2, v1 // (op 2223) 
s_add_u32 s53, s0, s52 // (op 2223) 

v_sub_u32 v41, v1, v23 // (op 2223) 

v_lshlrev_b32 v23, 4, v41 // (op 2223) 

s_mul_i32 s52, s16, s8 // (op 2223) 
s_add_u32 s52, s52, s53 // (op 2223) 

v_lshrrev_b32 v41, 1, v1 // (op 2223) 
s_mul_hi_u32 s53, s52, s13 // (op 2223) 
s_sub_u32 s54, s52, s53 // (op 2223) 
s_lshr_b32 s55, s54, 1 // (op 2223) 

s_add_u32 s54, s55, s53 // (op 2223) 


s_lshr_b32 s53, s54, s11 // (op 2223) 

s_and_b32 s54, s11, 2147483648 // (op 2223) 
s_cmp_eq_i32 s54, 0 // (op 2223) 
s_mov_b32 s55, scc // (op 2223) 

s_cmp_eq_i32 s55, 0 // (op 2223) 
s_cselect_b32 s54, s52, s53 // (op 2223) 


s_mul_hi_u32 s53, s54, s12 // (op 2223) 
s_sub_u32 s55, s54, s53 // (op 2223) 
s_lshr_b32 s84, s55, 1 // (op 2223) 

s_add_u32 s55, s84, s53 // (op 2223) 


s_lshr_b32 s53, s55, s10 // (op 2223) 

s_and_b32 s55, s10, 2147483648 // (op 2223) 
s_cmp_eq_i32 s55, 0 // (op 2223) 
s_mov_b32 s84, scc // (op 2223) 

s_cmp_eq_i32 s84, 0 // (op 2223) 
s_cselect_b32 s55, s54, s53 // (op 2223) 


s_mul_i32 s53, s55, s14 // (op 2223) 

s_sub_u32 s55, s54, s53 // (op 2223) 

v_lshl_add_u32 v26, s55, 5, v23 // (op 2223) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2223) 
v_mov_b32 v35, 0 // (op 2223) convert
v_mov_b32 v34, v26 // (op 2223) convert

s_mul_i32 s53, s54, s84 // (op 2223) 


s_sub_u32 s54, s52, s53 // (op 2223) 


v_lshl_add_u32 v23, s54, 7, v41 // (op 2223) 


v_mul_lo_u32 v27, v23, s35 // (op 2223) most significant: low of low * high
v_mul_hi_u32 v41, v23, s34 // (op 2223) most significant: high of low * low
v_mul_lo_u32 v26, v23, s34 // (op 2223) least significant: low of low * low
v_add_u32 v27, v27, v41 // (op 2223) most significant: sum

v_add_co_u32 v26, s[52:53], v26, v34 // (op 2223) least significant half
v_addc_co_u32 v27, s[52:53], v27, v35, s[52:53] // (op 2223) most significant half



s_mul_i32 s53, s34, s3 // (op 2225) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2225) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2225) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2225) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2225) most significant: sum
s_add_u32 s53, s53, s55 // (op 2225) most significant: sum


s_mov_b32 s72, s36 // (op 2225) reg expression
s_mov_b32 s73, s37 // (op 2225) reg expression
s_mov_b32 s74, s52 // (op 2225) reg expression
s_mov_b32 s75, 131072 // (op 2225) call()

s_and_b32 s52, s7, 4294967294 // (op 2351) 
v_and_b32 v23, -64, v1 // (op 2351) 
s_add_u32 s53, s0, s52 // (op 2351) 

v_sub_u32 v41, v1, v23 // (op 2351) 

v_lshrrev_b32 v23, 6, v1 // (op 2351) 
s_mul_i32 s52, s16, s8 // (op 2351) 
s_add_u32 s52, s52, s53 // (op 2351) 

s_mul_hi_u32 s53, s52, s13 // (op 2351) 
s_sub_u32 s54, s52, s53 // (op 2351) 
s_lshr_b32 s55, s54, 1 // (op 2351) 

s_add_u32 s54, s55, s53 // (op 2351) 


s_lshr_b32 s53, s54, s11 // (op 2351) 

s_and_b32 s54, s11, 2147483648 // (op 2351) 
s_cmp_eq_i32 s54, 0 // (op 2351) 
s_mov_b32 s55, scc // (op 2351) 

s_cmp_eq_i32 s55, 0 // (op 2351) 
s_cselect_b32 s54, s52, s53 // (op 2351) 


s_mul_hi_u32 s53, s54, s12 // (op 2351) 
s_sub_u32 s55, s54, s53 // (op 2351) 
s_lshr_b32 s84, s55, 1 // (op 2351) 

s_add_u32 s55, s84, s53 // (op 2351) 


s_lshr_b32 s53, s55, s10 // (op 2351) 

s_and_b32 s55, s10, 2147483648 // (op 2351) 
s_cmp_eq_i32 s55, 0 // (op 2351) 
s_mov_b32 s84, scc // (op 2351) 

s_cmp_eq_i32 s84, 0 // (op 2351) 
s_cselect_b32 s55, s54, s53 // (op 2351) 


v_lshl_add_u32 v28, s55, 6, v41 // (op 2351) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2351) 
s_mul_i32 s53, s54, s84 // (op 2351) 


s_sub_u32 s54, s52, s53 // (op 2351) 


v_lshl_add_u32 v41, s54, 2, v23 // (op 2351) 


v_mul_lo_u32 v23, v41, s38 // (op 2351) 

v_add_u32 v41, v28, v23 // (op 2351) 


s_mul_i32 s53, s38, s43 // (op 2353) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2353) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2353) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2353) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2353) most significant: sum
s_add_u32 s53, s53, s55 // (op 2353) most significant: sum


s_mov_b32 s76, s40 // (op 2353) reg expression
s_mov_b32 s77, s41 // (op 2353) reg expression
s_mov_b32 s78, s52 // (op 2353) reg expression
s_mov_b32 s79, 131072 // (op 2353) call()

s_and_b32 s52, s7, 4294967294 // (op 2479) 
v_lshrrev_b32 v23, 6, v1 // (op 2479) 
s_add_u32 s53, s0, s52 // (op 2479) 

v_and_b32 v28, 1, v23 // (op 2479) 

v_and_b32 v23, 15, v1 // (op 2479) 
s_mul_i32 s52, s16, s8 // (op 2479) 
s_add_u32 s52, s52, s53 // (op 2479) 

v_lshl_add_u32 v29, v28, 4, v23 // (op 2479) 


s_mul_hi_u32 s53, s52, s13 // (op 2479) 
v_and_b32 v23, 63, v1 // (op 2479) 
s_sub_u32 s54, s52, s53 // (op 2479) 
v_lshrrev_b32 v28, 4, v23 // (op 2479) 

s_lshr_b32 s55, s54, 1 // (op 2479) 

s_add_u32 s54, s55, s53 // (op 2479) 


s_lshr_b32 s53, s54, s11 // (op 2479) 

s_and_b32 s54, s11, 2147483648 // (op 2479) 
s_cmp_eq_i32 s54, 0 // (op 2479) 
s_mov_b32 s55, scc // (op 2479) 

s_cmp_eq_i32 s55, 0 // (op 2479) 
s_cselect_b32 s54, s52, s53 // (op 2479) 


s_mul_hi_u32 s53, s54, s12 // (op 2479) 
s_sub_u32 s55, s54, s53 // (op 2479) 
s_lshr_b32 s84, s55, 1 // (op 2479) 

s_add_u32 s55, s84, s53 // (op 2479) 


s_lshr_b32 s53, s55, s10 // (op 2479) 

s_and_b32 s55, s10, 2147483648 // (op 2479) 
s_cmp_eq_i32 s55, 0 // (op 2479) 
s_mov_b32 s84, scc // (op 2479) 

s_cmp_eq_i32 s84, 0 // (op 2479) 
s_cselect_b32 s55, s54, s53 // (op 2479) 


s_mul_i32 s53, s55, s14 // (op 2479) 

s_sub_u32 s55, s54, s53 // (op 2479) 

v_lshl_add_u32 v23, s55, 5, v29 // (op 2479) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2479) 
s_mul_i32 s53, s54, s84 // (op 2479) 


s_sub_u32 s54, s52, s53 // (op 2479) 


v_lshl_add_u32 v29, s54, 2, v28 // (op 2479) 


v_mul_lo_u32 v28, v29, s44 // (op 2479) 
v_add_u32 v28, v28, v23 // (op 2479) 


s_mul_i32 s53, s44, s43 // (op 2481) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 2481) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 2481) least significant
s_mul_hi_u32 s55, s44, s42 // (op 2481) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2481) most significant: sum
s_add_u32 s53, s53, s55 // (op 2481) most significant: sum


s_mov_b32 s80, s46 // (op 2481) reg expression
s_mov_b32 s81, s47 // (op 2481) reg expression
s_mov_b32 s82, s52 // (op 2481) reg expression
s_mov_b32 s83, 131072 // (op 2481) call()

v_and_b32 v23, -4, v1 // (op 2640) 
v_sub_u32 v29, v1, v23 // (op 2640) 

v_lshlrev_b32 v23, 4, v29 // (op 2640) 

v_lshrrev_b32 v29, 2, v1 // (op 2640) 
v_lshl_add_u32 v30, v29, 6, v23 // (op 2640) 


s_nop 0
v_readfirstlane_b32 s52, v30 // (op 2640) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -2, v1 // (op 2715) 
v_sub_u32 v29, v1, v23 // (op 2715) 

v_lshlrev_b32 v23, 4, v29 // (op 2715) 

v_lshrrev_b32 v29, 1, v1 // (op 2715) 
v_lshl_add_u32 v30, v29, 5, v23 // (op 2715) 


s_nop 0
v_readfirstlane_b32 s53, v30 // (op 2715) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 2780) 
v_sub_u32 v29, v1, v23 // (op 2780) 

v_and_b32 v23, 4294967232, v1 // (op 2780) 
v_add_u32 v30, v23, v29 // (op 2780) 


s_and_b32 s54, s7, 4294967294 // (op 999) 
s_cmp_lt_u32 s54, s7 // (op 998) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_998 // (op 998) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_998 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_998:
s_barrier  // (op 3442) 
s_add_u32 s54, s54, 1 // (op 1000) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 2001) 
v_mov_b64 v[32:33], s[84:85] // (op 2001) convert


v_mov_b64 v[34:35], v[24:25] // (op 2001) 
v_mov_b64 v[36:37], v[34:35] // (op 2001) reg expression
s_add_u32 s55, 4096, s52 // (op 2001) 
s_mov_b32 m0, s55 // (op 2001) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2001) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v32 // (op 2001) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 2001) most significant half

s_add_u32 m0, m0, 4096 // (op 2001) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2001) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 2077) 
v_mov_b32 v21, s84 // (op 2077) 
v_mov_b32 v23, s85 // (op 2077) 
v_add_co_u32 v24, s[86:87], v24, v21 // (op 2077) least significant half
v_addc_co_u32 v25, s[86:87], v25, v23, s[86:87] // (op 2077) most significant half




s_barrier  // (op 3461) 


v_mov_b64 v[32:33], v[26:27] // (op 2007) 
v_mov_b64 v[34:35], v[32:33] // (op 2007) reg expression

s_mov_b32 m0, s53 // (op 2007) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2007) Load value direct to lds


s_lshl_b64 s[84:85], s[34:35], 7 // (op 2215) 
v_mov_b32 v21, s84 // (op 2215) 
v_mov_b32 v23, s85 // (op 2215) 
v_add_co_u32 v26, s[86:87], v26, v21 // (op 2215) least significant half
v_addc_co_u32 v27, s[86:87], v27, v23, s[86:87] // (op 2215) most significant half






v_mov_b32 v21, v41 // (op 863) 
v_mov_b32 v23, v21 // (op 863) reg expression

buffer_load_ubyte v29, v23, s[76:79], 0 offen offset: 0 // (op 863) Load value


s_lshl_b32 s55, s38, 2 // (op 2343) 
v_add_u32 v41, v41, s55 // (op 2343) 

s_barrier  // (op 3480) 
s_waitcnt vmcnt(2)
s_barrier  // (op 3448) 






ds_read_b64_tr_b8 v[32:33], v0 offset:4096 // (op 852) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:4608 // (op 852) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:8192 // (op 852) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:8704 // (op 852) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v30, v29 offset:24576 // (op 861) Store local data 

s_barrier  // (op 3467) 



ds_read_b64_tr_b8 v[42:43], v3 // (op 854) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:256 // (op 854) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:2048 // (op 854) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:2304 // (op 854) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3487) 


ds_read_u8 v21, v7 offset:24576 // (op 856) Load local data 

v_mov_b64 v[50:51], s[44:45] // (op 858) convert

v_mov_b32 v23, v28 // (op 858) 
v_mov_b32 v29, v23 // (op 858) reg expression
v_mov_b32 v31, v29 // (op 858) 
buffer_load_ubyte v40, v31, s[80:83], 0 offen offset: 0 // (op 858) Load value




s_lshl_b32 s55, s44, 2 // (op 2471) 
v_add_u32 v28, v28, s55 // (op 2471) 




ds_read_b64_tr_b8 v[50:51], v0 offset:4128 // (op 1004) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:4640 // (op 1004) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[42:49], a[0:3], v21, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 859) 
ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 1004) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 1004) Transpose load local data 


ds_read_u8 v21, v7 offset:24608 // (op 1008) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[42:49], a[4:7], v21, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1010) 




s_cmp_lt_u32 s54, s7 // (op 998) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_998 // (op 998) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_998 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_998:








s_barrier  // (op 3725) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 272) 
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_298



v_add_u32 v19, 4096, v20 // (op 1195) 
s_nop 0
buffer_store_dword a4, v19, s[64:67], 0 offen sc1 // (op 1195) Store value

v_add_u32 v19, 4224, v20 // (op 1195) 
buffer_store_dword a5, v19, s[64:67], 0 offen sc1 // (op 1195) Store value

v_add_u32 v19, 4352, v20 // (op 1195) 
buffer_store_dword a6, v19, s[64:67], 0 offen sc1 // (op 1195) Store value

v_add_u32 v19, 4480, v20 // (op 1195) 
buffer_store_dword a7, v19, s[64:67], 0 offen sc1 // (op 1195) Store value




buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:128 sc1 // (op 295) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:256 sc1 // (op 295) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:384 sc1 // (op 295) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
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
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_298:

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
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s52, 0 // (op 280) call()
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s53, s16, 1 // (op 3128) 
v_lshrrev_b32 v16, 5, v1 // (op 3128) 
s_add_u32 s54, s53, s52 // (op 3128) 
s_lshl_b32 s54, s54, 6 // (op 3128) 

v_lshlrev_b32 v17, 2, v16 // (op 3128) 

v_add_lshl_u32 v16, s54, v17, 5 // (op 3128) 


v_and_b32 v17, -32, v1 // (op 3128) 
v_sub_u32 v19, v1, v17 // (op 3128) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 3128) 


s_lshl_b32 s53, s15, 13 // (op 3130) 
s_mov_b32 s64, s28 // (op 3130) reg expression
s_mov_b32 s65, s29 // (op 3130) reg expression
s_mov_b32 s66, s53 // (op 3130) reg expression
s_mov_b32 s67, 131072 // (op 3130) call()

RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
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
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
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
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




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




v_mov_b32 v16, 4096 // (op 1211) 
v_add_i32 v19, v17, v16 // (op 1211) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 1211) Load value

v_mov_b32 v16, 4224 // (op 1211) 
v_add_i32 v23, v17, v16 // (op 1211) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 1211) Load value

v_mov_b32 v16, 4352 // (op 1211) 
v_add_i32 v23, v17, v16 // (op 1211) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 1211) Load value

v_mov_b32 v16, 4480 // (op 1211) 
v_add_i32 v23, v17, v16 // (op 1211) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 1211) Load value

v_accvgpr_read v28, a4 // (op 1212) reg expression
v_accvgpr_read v29, a5 // (op 1212) reg expression
v_accvgpr_read v30, a6 // (op 1212) reg expression
v_accvgpr_read v31, a7 // (op 1212) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1213) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1213) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1213) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1213) 

v_accvgpr_write a4, v28 // (op 1214) reg expression
v_accvgpr_write a5, v29 // (op 1214) reg expression
v_accvgpr_write a6, v30 // (op 1214) reg expression
v_accvgpr_write a7, v31 // (op 1214) reg expression

s_add_u32 s52, s52, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v24, a4 // (op 1236) 
v_accvgpr_read v25, a5 // (op 1236) 
v_accvgpr_read v26, a6 // (op 1236) 
v_accvgpr_read v27, a7 // (op 1236) 
v_mul_f32 v28, v15, v24 // (op 1236) 
v_mul_f32 v29, v15, v25 // (op 1236) 
v_mul_f32 v30, v15, v26 // (op 1236) 
v_mul_f32 v31, v15, v27 // (op 1236) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen offset:128 // (op 1238) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen offset:128 // (op 1266) Load value
v_accvgpr_read v28, a4 // (op 1263) 
v_accvgpr_read v29, a5 // (op 1263) 
v_accvgpr_read v30, a6 // (op 1263) 
v_accvgpr_read v31, a7 // (op 1263) 
v_mul_f32 v32, v15, v28 // (op 1263) 
v_mul_f32 v33, v15, v29 // (op 1263) 
v_mul_f32 v34, v15, v30 // (op 1263) 
v_mul_f32 v35, v15, v31 // (op 1263) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 1267) 
v_mul_f32 v29, v14, v25 // (op 1267) 
v_mul_f32 v30, v14, v26 // (op 1267) 
v_mul_f32 v31, v14, v27 // (op 1267) 

v_add_f32 v24, v28, v32 // (op 1264) 
v_add_f32 v25, v29, v33 // (op 1264) 
v_add_f32 v26, v30, v34 // (op 1264) 
v_add_f32 v27, v31, v35 // (op 1264) 





buffer_store_dwordx4 v[24:27], v22, s[60:63], 0 offen offset:128 // (op 1265) Store value




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
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:


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
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















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
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 7, v1 // (op 2610) 
s_mul_i32 s15, s16, s50 // (op 2610) 
s_add_u32 s15, s15, s0 // (op 2610) 
v_and_b32 v3, 63, v1 // (op 2610) 
s_add_u32 s6, s15, s9 // (op 2610) 

v_lshrrev_b32 v6, 4, v3 // (op 2610) 
s_mul_hi_u32 s15, s6, s13 // (op 2610) 
v_lshlrev_b32 v7, 2, v6 // (op 2610) 

s_sub_u32 s7, s6, s15 // (op 2610) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 2610) 


s_lshr_b32 s8, s7, 1 // (op 2610) 

v_lshrrev_b32 v0, 2, v3 // (op 2610) 

s_add_u32 s7, s8, s15 // (op 2610) 


v_and_b32 v3, 3, v0 // (op 2610) 

s_lshr_b32 s8, s7, s11 // (op 2610) 

v_and_b32 v0, 3, v1 // (op 2610) 
s_and_b32 s7, s11, 2147483648 // (op 2610) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 2610) 


s_cmp_eq_i32 s7, 0 // (op 2610) 
s_mov_b32 s15, scc // (op 2610) 

v_lshrrev_b32 v0, 6, v1 // (op 2610) 
s_cmp_eq_i32 s15, 0 // (op 2610) 
s_cselect_b32 s7, s6, s8 // (op 2610) 



v_and_b32 v3, 1, v0 // (op 2610) 

s_mul_hi_u32 s6, s7, s12 // (op 2610) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 2610) 


s_sub_u32 s8, s7, s6 // (op 2610) 
s_lshr_b32 s15, s8, 1 // (op 2610) 

s_add_u32 s8, s15, s6 // (op 2610) 


s_lshr_b32 s6, s8, s10 // (op 2610) 

s_and_b32 s8, s10, 2147483648 // (op 2610) 
s_cmp_eq_i32 s8, 0 // (op 2610) 
s_mov_b32 s15, scc // (op 2610) 

s_cmp_eq_i32 s15, 0 // (op 2610) 
s_cselect_b32 s8, s7, s6 // (op 2610) 


s_mul_i32 s6, s8, s14 // (op 2610) 
v_lshl_add_u32 v3, s8, 6, v6 // (op 2610) 


s_sub_u32 s8, s7, s6 // (op 2610) 


v_lshl_add_u32 v6, s8, 5, v0 // (op 2610) 


v_mul_lo_u32 v0, v6, s4 // (op 2610) 

v_add_lshl_u32 v6, v3, v0, 2 // (op 2610) 


s_mul_i32 s19, s4, s23 // (op 2612) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2612) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2612) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2612) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2612) most significant: sum
s_add_u32 s19, s19, s6 // (op 2612) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2612) 

s_mov_b32 s56, s20 // (op 2612) reg expression
s_mov_b32 s57, s21 // (op 2612) reg expression
s_mov_b32 s58, s28 // (op 2612) reg expression
s_mov_b32 s59, 131072 // (op 2612) call()

v_lshrrev_b32 v0, 7, v1 // (op 2868) 
s_mul_i32 s15, s16, s50 // (op 2868) 
s_add_u32 s15, s15, s0 // (op 2868) 
v_and_b32 v3, 63, v1 // (op 2868) 
s_add_u32 s6, s15, s9 // (op 2868) 

v_lshrrev_b32 v7, 4, v3 // (op 2868) 
s_mul_hi_u32 s15, s6, s13 // (op 2868) 
v_lshlrev_b32 v12, 2, v7 // (op 2868) 

s_sub_u32 s7, s6, s15 // (op 2868) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2868) 


s_lshr_b32 s8, s7, 1 // (op 2868) 

v_lshrrev_b32 v0, 2, v3 // (op 2868) 

s_add_u32 s7, s8, s15 // (op 2868) 


v_and_b32 v3, 3, v0 // (op 2868) 

s_lshr_b32 s8, s7, s11 // (op 2868) 

v_and_b32 v0, 3, v1 // (op 2868) 
s_and_b32 s7, s11, 2147483648 // (op 2868) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 2868) 


s_cmp_eq_i32 s7, 0 // (op 2868) 
s_mov_b32 s15, scc // (op 2868) 

v_lshrrev_b32 v0, 6, v1 // (op 2868) 
s_cmp_eq_i32 s15, 0 // (op 2868) 
s_cselect_b32 s7, s6, s8 // (op 2868) 



v_and_b32 v3, 1, v0 // (op 2868) 

s_mul_hi_u32 s6, s7, s12 // (op 2868) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 2868) 


s_sub_u32 s8, s7, s6 // (op 2868) 
s_lshr_b32 s15, s8, 1 // (op 2868) 

s_add_u32 s8, s15, s6 // (op 2868) 


s_lshr_b32 s6, s8, s10 // (op 2868) 

s_and_b32 s8, s10, 2147483648 // (op 2868) 
s_cmp_eq_i32 s8, 0 // (op 2868) 
s_mov_b32 s15, scc // (op 2868) 

s_cmp_eq_i32 s15, 0 // (op 2868) 
s_cselect_b32 s8, s7, s6 // (op 2868) 


s_mul_i32 s6, s8, s14 // (op 2868) 
v_lshl_add_u32 v3, s8, 6, v7 // (op 2868) 


s_sub_u32 s8, s7, s6 // (op 2868) 


v_lshl_add_u32 v7, s8, 5, v0 // (op 2868) 


v_mul_lo_u32 v0, v7, s24 // (op 2868) 

v_add_lshl_u32 v7, v3, v0, 2 // (op 2868) 


s_mul_i32 s19, s24, s23 // (op 2870) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2870) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2870) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2870) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2870) most significant: sum
s_add_u32 s19, s19, s6 // (op 2870) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2870) 

s_mov_b32 s60, s26 // (op 2870) reg expression
s_mov_b32 s61, s27 // (op 2870) reg expression
s_mov_b32 s62, s28 // (op 2870) reg expression
s_mov_b32 s63, 131072 // (op 2870) call()

v_and_b32 v0, 48, v1 // (op 2947) 
v_and_b32 v3, 63, v1 // (op 2947) 
v_sub_u32 v12, v3, v0 // (op 2947) 

v_lshrrev_b32 v3, 1, v12 // (op 2947) 
v_add_lshl_u32 v13, v0, v3, 6 // (op 2947) 


v_and_b32 v0, 1, v12 // (op 2947) 

v_lshlrev_b32 v3, 3, v0 // (op 2947) 

v_lshrrev_b32 v0, 7, v1 // (op 2947) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 2947) 


v_add_u32 v0, v13, v12 // (op 2947) 


v_and_b32 v3, 48, v1 // (op 3012) 
v_and_b32 v12, 63, v1 // (op 3012) 
v_sub_u32 v13, v12, v3 // (op 3012) 

v_lshrrev_b32 v12, 1, v13 // (op 3012) 
v_add_lshl_u32 v16, v3, v12, 5 // (op 3012) 


v_and_b32 v3, 1, v13 // (op 3012) 

v_lshlrev_b32 v12, 3, v3 // (op 3012) 

v_lshrrev_b32 v3, 6, v1 // (op 3012) 
v_and_b32 v13, 1, v3 // (op 3012) 

v_lshl_add_u32 v3, v13, 4, v12 // (op 3012) 


v_add_u32 v12, v16, v3 // (op 3012) 


v_lshrrev_b32 v3, 7, v1 // (op 3088) 
v_and_b32 v13, 15, v1 // (op 3088) 
v_lshl_add_u32 v16, v3, 4, v13 // (op 3088) 


v_and_b32 v3, 63, v1 // (op 3088) 
v_lshrrev_b32 v13, 4, v3 // (op 3088) 

v_lshl_add_u32 v3, v13, 6, v16 // (op 3088) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_accvgpr_write a4, 0.00000 // (op 581) call()
v_accvgpr_write a5, 0.00000 // (op 581) call()
v_accvgpr_write a6, 0.00000 // (op 581) call()
v_accvgpr_write a7, 0.00000 // (op 581) call()
v_and_b32 v13, -4, v1 // (op 2114) 
s_mul_i32 s15, s16, s50 // (op 2114) 
s_add_u32 s15, s15, s0 // (op 2114) 
v_sub_u32 v16, v1, v13 // (op 2114) 

s_add_u32 s6, s15, s9 // (op 2114) 

v_lshlrev_b32 v13, 4, v16 // (op 2114) 

s_mul_hi_u32 s15, s6, s13 // (op 2114) 
v_lshrrev_b32 v16, 2, v1 // (op 2114) 
s_sub_u32 s7, s6, s15 // (op 2114) 
s_lshr_b32 s8, s7, 1 // (op 2114) 

s_add_u32 s7, s8, s15 // (op 2114) 


s_lshr_b32 s8, s7, s11 // (op 2114) 

s_and_b32 s7, s11, 2147483648 // (op 2114) 
s_cmp_eq_i32 s7, 0 // (op 2114) 
s_mov_b32 s15, scc // (op 2114) 

s_cmp_eq_i32 s15, 0 // (op 2114) 
s_cselect_b32 s7, s6, s8 // (op 2114) 


s_mul_hi_u32 s8, s7, s12 // (op 2114) 
s_sub_u32 s15, s7, s8 // (op 2114) 
s_lshr_b32 s18, s15, 1 // (op 2114) 

s_add_u32 s15, s18, s8 // (op 2114) 


s_lshr_b32 s8, s15, s10 // (op 2114) 

s_and_b32 s15, s10, 2147483648 // (op 2114) 
s_cmp_eq_i32 s15, 0 // (op 2114) 
s_mov_b32 s18, scc // (op 2114) 

s_cmp_eq_i32 s18, 0 // (op 2114) 
s_cselect_b32 s15, s7, s8 // (op 2114) 


v_lshl_add_u32 v17, s15, 6, v13 // (op 2114) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2114) 
v_mov_b32 v19, 0 // (op 2114) convert
v_mov_b32 v18, v17 // (op 2114) convert

s_mul_i32 s8, s7, s18 // (op 2114) 


s_sub_u32 s7, s6, s8 // (op 2114) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2114) 


v_mul_lo_u32 v17, v13, s31 // (op 2114) most significant: low of low * high
v_mul_hi_u32 v20, v13, s30 // (op 2114) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 2114) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 2114) most significant: sum


v_add_co_u32 v20, s[18:19], v18, v16 // (op 2114) least significant half
v_addc_co_u32 v21, s[18:19], v19, v17, s[18:19] // (op 2114) most significant half



s_mul_i32 s19, s30, s3 // (op 2116) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2116) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2116) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2116) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2116) most significant: sum
s_add_u32 s19, s19, s6 // (op 2116) most significant: sum


s_mov_b32 s68, s32 // (op 2116) reg expression
s_mov_b32 s69, s33 // (op 2116) reg expression
s_mov_b32 s70, s18 // (op 2116) reg expression
s_mov_b32 s71, 131072 // (op 2116) call()

v_and_b32 v13, -4, v1 // (op 2133) 
s_mul_i32 s15, s16, s50 // (op 2133) 
s_add_u32 s15, s15, s0 // (op 2133) 
v_sub_u32 v16, v1, v13 // (op 2133) 

s_add_u32 s6, s15, s9 // (op 2133) 

v_lshlrev_b32 v13, 4, v16 // (op 2133) 

s_mul_hi_u32 s15, s6, s13 // (op 2133) 
v_lshrrev_b32 v16, 2, v1 // (op 2133) 
s_sub_u32 s7, s6, s15 // (op 2133) 
s_lshr_b32 s8, s7, 1 // (op 2133) 

s_add_u32 s7, s8, s15 // (op 2133) 


s_lshr_b32 s8, s7, s11 // (op 2133) 

s_and_b32 s7, s11, 2147483648 // (op 2133) 
s_cmp_eq_i32 s7, 0 // (op 2133) 
s_mov_b32 s15, scc // (op 2133) 

s_cmp_eq_i32 s15, 0 // (op 2133) 
s_cselect_b32 s7, s6, s8 // (op 2133) 


s_mul_hi_u32 s8, s7, s12 // (op 2133) 
s_sub_u32 s15, s7, s8 // (op 2133) 
s_lshr_b32 s18, s15, 1 // (op 2133) 

s_add_u32 s15, s18, s8 // (op 2133) 


s_lshr_b32 s8, s15, s10 // (op 2133) 

s_and_b32 s15, s10, 2147483648 // (op 2133) 
s_cmp_eq_i32 s15, 0 // (op 2133) 
s_mov_b32 s18, scc // (op 2133) 

s_cmp_eq_i32 s18, 0 // (op 2133) 
s_cselect_b32 s15, s7, s8 // (op 2133) 


v_lshl_add_u32 v17, s15, 6, v13 // (op 2133) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2133) 
v_mov_b32 v19, 0 // (op 2133) convert
v_mov_b32 v18, v17 // (op 2133) convert

s_mul_i32 s8, s7, s18 // (op 2133) 


s_sub_u32 s7, s6, s8 // (op 2133) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2133) 


v_mul_lo_u32 v17, v13, s31 // (op 2133) most significant: low of low * high
v_mul_hi_u32 v22, v13, s30 // (op 2133) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 2133) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 2133) most significant: sum


v_add_co_u32 v22, s[18:19], v18, v16 // (op 2133) least significant half
v_addc_co_u32 v23, s[18:19], v19, v17, s[18:19] // (op 2133) most significant half



s_mul_i32 s19, s30, s3 // (op 2135) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2135) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2135) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2135) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2135) most significant: sum
s_add_u32 s19, s19, s6 // (op 2135) most significant: sum


s_mov_b32 s68, s32 // (op 2135) reg expression
s_mov_b32 s69, s33 // (op 2135) reg expression
s_mov_b32 s70, s18 // (op 2135) reg expression
s_mov_b32 s71, 131072 // (op 2135) call()

v_and_b32 v13, -2, v1 // (op 2246) 
s_mul_i32 s15, s16, s50 // (op 2246) 
s_add_u32 s15, s15, s0 // (op 2246) 
v_sub_u32 v16, v1, v13 // (op 2246) 

s_add_u32 s6, s15, s9 // (op 2246) 

v_lshlrev_b32 v13, 4, v16 // (op 2246) 

s_mul_hi_u32 s15, s6, s13 // (op 2246) 
v_lshrrev_b32 v16, 1, v1 // (op 2246) 
s_sub_u32 s7, s6, s15 // (op 2246) 
s_lshr_b32 s8, s7, 1 // (op 2246) 

s_add_u32 s7, s8, s15 // (op 2246) 


s_lshr_b32 s8, s7, s11 // (op 2246) 

s_and_b32 s7, s11, 2147483648 // (op 2246) 
s_cmp_eq_i32 s7, 0 // (op 2246) 
s_mov_b32 s15, scc // (op 2246) 

s_cmp_eq_i32 s15, 0 // (op 2246) 
s_cselect_b32 s7, s6, s8 // (op 2246) 


s_mul_hi_u32 s8, s7, s12 // (op 2246) 
s_sub_u32 s15, s7, s8 // (op 2246) 
s_lshr_b32 s18, s15, 1 // (op 2246) 

s_add_u32 s15, s18, s8 // (op 2246) 


s_lshr_b32 s8, s15, s10 // (op 2246) 

s_and_b32 s15, s10, 2147483648 // (op 2246) 
s_cmp_eq_i32 s15, 0 // (op 2246) 
s_mov_b32 s18, scc // (op 2246) 

s_cmp_eq_i32 s18, 0 // (op 2246) 
s_cselect_b32 s15, s7, s8 // (op 2246) 


s_mul_i32 s8, s15, s14 // (op 2246) 

s_sub_u32 s15, s7, s8 // (op 2246) 

v_lshl_add_u32 v17, s15, 5, v13 // (op 2246) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2246) 
v_mov_b32 v19, 0 // (op 2246) convert
v_mov_b32 v18, v17 // (op 2246) convert

s_mul_i32 s8, s7, s18 // (op 2246) 


s_sub_u32 s7, s6, s8 // (op 2246) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2246) 


v_mul_lo_u32 v17, v13, s35 // (op 2246) most significant: low of low * high
v_mul_hi_u32 v24, v13, s34 // (op 2246) most significant: high of low * low
v_mul_lo_u32 v16, v13, s34 // (op 2246) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 2246) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v18 // (op 2246) least significant half
v_addc_co_u32 v17, s[18:19], v17, v19, s[18:19] // (op 2246) most significant half



s_mul_i32 s19, s34, s3 // (op 2248) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2248) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2248) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2248) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2248) most significant: sum
s_add_u32 s19, s19, s6 // (op 2248) most significant: sum


s_mov_b32 s72, s36 // (op 2248) reg expression
s_mov_b32 s73, s37 // (op 2248) reg expression
s_mov_b32 s74, s18 // (op 2248) reg expression
s_mov_b32 s75, 131072 // (op 2248) call()

v_and_b32 v13, -2, v1 // (op 2265) 
s_mul_i32 s15, s16, s50 // (op 2265) 
s_add_u32 s15, s15, s0 // (op 2265) 
v_sub_u32 v18, v1, v13 // (op 2265) 

s_add_u32 s6, s15, s9 // (op 2265) 

v_lshlrev_b32 v13, 4, v18 // (op 2265) 

s_mul_hi_u32 s15, s6, s13 // (op 2265) 
v_lshrrev_b32 v18, 1, v1 // (op 2265) 
s_sub_u32 s7, s6, s15 // (op 2265) 
s_lshr_b32 s8, s7, 1 // (op 2265) 

s_add_u32 s7, s8, s15 // (op 2265) 


s_lshr_b32 s8, s7, s11 // (op 2265) 

s_and_b32 s7, s11, 2147483648 // (op 2265) 
s_cmp_eq_i32 s7, 0 // (op 2265) 
s_mov_b32 s15, scc // (op 2265) 

s_cmp_eq_i32 s15, 0 // (op 2265) 
s_cselect_b32 s7, s6, s8 // (op 2265) 


s_mul_hi_u32 s8, s7, s12 // (op 2265) 
s_sub_u32 s15, s7, s8 // (op 2265) 
s_lshr_b32 s18, s15, 1 // (op 2265) 

s_add_u32 s15, s18, s8 // (op 2265) 


s_lshr_b32 s8, s15, s10 // (op 2265) 

s_and_b32 s15, s10, 2147483648 // (op 2265) 
s_cmp_eq_i32 s15, 0 // (op 2265) 
s_mov_b32 s18, scc // (op 2265) 

s_cmp_eq_i32 s18, 0 // (op 2265) 
s_cselect_b32 s15, s7, s8 // (op 2265) 


s_mul_i32 s8, s15, s14 // (op 2265) 

s_sub_u32 s15, s7, s8 // (op 2265) 

v_lshl_add_u32 v19, s15, 5, v13 // (op 2265) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2265) 
v_mov_b32 v25, 0 // (op 2265) convert
v_mov_b32 v24, v19 // (op 2265) convert

s_mul_i32 s8, s7, s18 // (op 2265) 


s_sub_u32 s7, s6, s8 // (op 2265) 


v_lshl_add_u32 v13, s7, 7, v18 // (op 2265) 


v_mul_lo_u32 v19, v13, s35 // (op 2265) most significant: low of low * high
v_mul_hi_u32 v26, v13, s34 // (op 2265) most significant: high of low * low
v_mul_lo_u32 v18, v13, s34 // (op 2265) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 2265) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v24 // (op 2265) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 2265) most significant half



s_mul_i32 s19, s34, s3 // (op 2267) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2267) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2267) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2267) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2267) most significant: sum
s_add_u32 s19, s19, s6 // (op 2267) most significant: sum


s_mov_b32 s72, s36 // (op 2267) reg expression
s_mov_b32 s73, s37 // (op 2267) reg expression
s_mov_b32 s74, s18 // (op 2267) reg expression
s_mov_b32 s75, 131072 // (op 2267) call()

v_and_b32 v13, -64, v1 // (op 2374) 
s_mul_i32 s15, s16, s50 // (op 2374) 
s_add_u32 s15, s15, s0 // (op 2374) 
v_sub_u32 v24, v1, v13 // (op 2374) 

s_add_u32 s6, s15, s9 // (op 2374) 

v_lshrrev_b32 v13, 6, v1 // (op 2374) 
s_mul_hi_u32 s15, s6, s13 // (op 2374) 
s_sub_u32 s7, s6, s15 // (op 2374) 
s_lshr_b32 s8, s7, 1 // (op 2374) 

s_add_u32 s7, s8, s15 // (op 2374) 


s_lshr_b32 s8, s7, s11 // (op 2374) 

s_and_b32 s7, s11, 2147483648 // (op 2374) 
s_cmp_eq_i32 s7, 0 // (op 2374) 
s_mov_b32 s15, scc // (op 2374) 

s_cmp_eq_i32 s15, 0 // (op 2374) 
s_cselect_b32 s7, s6, s8 // (op 2374) 


s_mul_hi_u32 s8, s7, s12 // (op 2374) 
s_sub_u32 s15, s7, s8 // (op 2374) 
s_lshr_b32 s18, s15, 1 // (op 2374) 

s_add_u32 s15, s18, s8 // (op 2374) 


s_lshr_b32 s8, s15, s10 // (op 2374) 

s_and_b32 s15, s10, 2147483648 // (op 2374) 
s_cmp_eq_i32 s15, 0 // (op 2374) 
s_mov_b32 s18, scc // (op 2374) 

s_cmp_eq_i32 s18, 0 // (op 2374) 
s_cselect_b32 s15, s7, s8 // (op 2374) 


v_lshl_add_u32 v25, s15, 6, v24 // (op 2374) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2374) 
s_mul_i32 s8, s7, s18 // (op 2374) 


s_sub_u32 s7, s6, s8 // (op 2374) 


v_lshl_add_u32 v24, s7, 2, v13 // (op 2374) 


v_mul_lo_u32 v13, v24, s38 // (op 2374) 

v_add_u32 v24, v25, v13 // (op 2374) 


s_mul_i32 s19, s38, s43 // (op 2376) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2376) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2376) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2376) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2376) most significant: sum
s_add_u32 s19, s19, s6 // (op 2376) most significant: sum


s_mov_b32 s76, s40 // (op 2376) reg expression
s_mov_b32 s77, s41 // (op 2376) reg expression
s_mov_b32 s78, s18 // (op 2376) reg expression
s_mov_b32 s79, 131072 // (op 2376) call()

v_and_b32 v13, -64, v1 // (op 2393) 
s_mul_i32 s15, s16, s50 // (op 2393) 
s_add_u32 s15, s15, s0 // (op 2393) 
v_sub_u32 v25, v1, v13 // (op 2393) 

s_add_u32 s6, s15, s9 // (op 2393) 

v_lshrrev_b32 v13, 6, v1 // (op 2393) 
s_mul_hi_u32 s15, s6, s13 // (op 2393) 
s_sub_u32 s7, s6, s15 // (op 2393) 
s_lshr_b32 s8, s7, 1 // (op 2393) 

s_add_u32 s7, s8, s15 // (op 2393) 


s_lshr_b32 s8, s7, s11 // (op 2393) 

s_and_b32 s7, s11, 2147483648 // (op 2393) 
s_cmp_eq_i32 s7, 0 // (op 2393) 
s_mov_b32 s15, scc // (op 2393) 

s_cmp_eq_i32 s15, 0 // (op 2393) 
s_cselect_b32 s7, s6, s8 // (op 2393) 


s_mul_hi_u32 s8, s7, s12 // (op 2393) 
s_sub_u32 s15, s7, s8 // (op 2393) 
s_lshr_b32 s18, s15, 1 // (op 2393) 

s_add_u32 s15, s18, s8 // (op 2393) 


s_lshr_b32 s8, s15, s10 // (op 2393) 

s_and_b32 s15, s10, 2147483648 // (op 2393) 
s_cmp_eq_i32 s15, 0 // (op 2393) 
s_mov_b32 s18, scc // (op 2393) 

s_cmp_eq_i32 s18, 0 // (op 2393) 
s_cselect_b32 s15, s7, s8 // (op 2393) 


v_lshl_add_u32 v26, s15, 6, v25 // (op 2393) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2393) 
s_mul_i32 s8, s7, s18 // (op 2393) 


s_sub_u32 s7, s6, s8 // (op 2393) 


v_lshl_add_u32 v25, s7, 2, v13 // (op 2393) 


v_mul_lo_u32 v13, v25, s38 // (op 2393) 

v_add_u32 v25, v26, v13 // (op 2393) 


s_mul_i32 s19, s38, s43 // (op 2395) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2395) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2395) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2395) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2395) most significant: sum
s_add_u32 s19, s19, s6 // (op 2395) most significant: sum


s_mov_b32 s76, s40 // (op 2395) reg expression
s_mov_b32 s77, s41 // (op 2395) reg expression
s_mov_b32 s78, s18 // (op 2395) reg expression
s_mov_b32 s79, 131072 // (op 2395) call()

v_lshrrev_b32 v13, 6, v1 // (op 2502) 
s_mul_i32 s15, s16, s50 // (op 2502) 
s_add_u32 s15, s15, s0 // (op 2502) 
v_and_b32 v26, 1, v13 // (op 2502) 

s_add_u32 s6, s15, s9 // (op 2502) 

v_and_b32 v13, 15, v1 // (op 2502) 
s_mul_hi_u32 s15, s6, s13 // (op 2502) 
v_lshl_add_u32 v27, v26, 4, v13 // (op 2502) 


s_sub_u32 s7, s6, s15 // (op 2502) 
v_and_b32 v13, 63, v1 // (op 2502) 
s_lshr_b32 s8, s7, 1 // (op 2502) 

v_lshrrev_b32 v26, 4, v13 // (op 2502) 

s_add_u32 s7, s8, s15 // (op 2502) 


s_lshr_b32 s8, s7, s11 // (op 2502) 

s_and_b32 s7, s11, 2147483648 // (op 2502) 
s_cmp_eq_i32 s7, 0 // (op 2502) 
s_mov_b32 s15, scc // (op 2502) 

s_cmp_eq_i32 s15, 0 // (op 2502) 
s_cselect_b32 s7, s6, s8 // (op 2502) 


s_mul_hi_u32 s8, s7, s12 // (op 2502) 
s_sub_u32 s15, s7, s8 // (op 2502) 
s_lshr_b32 s18, s15, 1 // (op 2502) 

s_add_u32 s15, s18, s8 // (op 2502) 


s_lshr_b32 s8, s15, s10 // (op 2502) 

s_and_b32 s15, s10, 2147483648 // (op 2502) 
s_cmp_eq_i32 s15, 0 // (op 2502) 
s_mov_b32 s18, scc // (op 2502) 

s_cmp_eq_i32 s18, 0 // (op 2502) 
s_cselect_b32 s15, s7, s8 // (op 2502) 


s_mul_i32 s8, s15, s14 // (op 2502) 

s_sub_u32 s15, s7, s8 // (op 2502) 

v_lshl_add_u32 v13, s15, 5, v27 // (op 2502) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2502) 
s_mul_i32 s8, s7, s18 // (op 2502) 


s_sub_u32 s7, s6, s8 // (op 2502) 


v_lshl_add_u32 v27, s7, 2, v26 // (op 2502) 


v_mul_lo_u32 v26, v27, s44 // (op 2502) 
v_add_u32 v26, v26, v13 // (op 2502) 


s_mul_i32 s19, s44, s43 // (op 2504) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2504) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2504) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2504) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2504) most significant: sum
s_add_u32 s19, s19, s6 // (op 2504) most significant: sum


s_mov_b32 s80, s46 // (op 2504) reg expression
s_mov_b32 s81, s47 // (op 2504) reg expression
s_mov_b32 s82, s18 // (op 2504) reg expression
s_mov_b32 s83, 131072 // (op 2504) call()

v_lshrrev_b32 v13, 6, v1 // (op 2521) 
s_mul_i32 s15, s16, s50 // (op 2521) 
s_add_u32 s15, s15, s0 // (op 2521) 
v_and_b32 v27, 1, v13 // (op 2521) 

s_add_u32 s6, s15, s9 // (op 2521) 

v_and_b32 v13, 15, v1 // (op 2521) 
s_mul_hi_u32 s15, s6, s13 // (op 2521) 
v_lshl_add_u32 v28, v27, 4, v13 // (op 2521) 


s_sub_u32 s7, s6, s15 // (op 2521) 
v_and_b32 v13, 63, v1 // (op 2521) 
s_lshr_b32 s8, s7, 1 // (op 2521) 

v_lshrrev_b32 v27, 4, v13 // (op 2521) 

s_add_u32 s7, s8, s15 // (op 2521) 


s_lshr_b32 s8, s7, s11 // (op 2521) 

s_and_b32 s7, s11, 2147483648 // (op 2521) 
s_cmp_eq_i32 s7, 0 // (op 2521) 
s_mov_b32 s15, scc // (op 2521) 

s_cmp_eq_i32 s15, 0 // (op 2521) 
s_cselect_b32 s7, s6, s8 // (op 2521) 


s_mul_hi_u32 s8, s7, s12 // (op 2521) 
s_sub_u32 s15, s7, s8 // (op 2521) 
s_lshr_b32 s18, s15, 1 // (op 2521) 

s_add_u32 s15, s18, s8 // (op 2521) 


s_lshr_b32 s8, s15, s10 // (op 2521) 

s_and_b32 s15, s10, 2147483648 // (op 2521) 
s_cmp_eq_i32 s15, 0 // (op 2521) 
s_mov_b32 s18, scc // (op 2521) 

s_cmp_eq_i32 s18, 0 // (op 2521) 
s_cselect_b32 s15, s7, s8 // (op 2521) 


s_mul_i32 s8, s15, s14 // (op 2521) 

s_sub_u32 s15, s7, s8 // (op 2521) 

v_lshl_add_u32 v13, s15, 5, v28 // (op 2521) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2521) 
s_mul_i32 s8, s7, s18 // (op 2521) 


s_sub_u32 s7, s6, s8 // (op 2521) 


v_lshl_add_u32 v28, s7, 2, v27 // (op 2521) 


v_mul_lo_u32 v27, v28, s44 // (op 2521) 
v_add_u32 v27, v27, v13 // (op 2521) 


s_mul_i32 s19, s44, s43 // (op 2523) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2523) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2523) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2523) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2523) most significant: sum
s_add_u32 s19, s19, s6 // (op 2523) most significant: sum


s_mov_b32 s80, s46 // (op 2523) reg expression
s_mov_b32 s81, s47 // (op 2523) reg expression
s_mov_b32 s82, s18 // (op 2523) reg expression
s_mov_b32 s83, 131072 // (op 2523) call()


v_mov_b64 v[28:29], s[44:45] // (op 1732) convert

v_mov_b32 v13, v26 // (op 1732) reg expression
v_mov_b32 v30, v13 // (op 1732) 
buffer_load_ubyte v31, v30, s[80:83], 0 offen offset: 0 // (op 1732) Load value



v_and_b32 v13, -2, v1 // (op 2988) 
v_sub_u32 v28, v1, v13 // (op 2988) 

v_lshlrev_b32 v13, 4, v28 // (op 2988) 

v_lshrrev_b32 v28, 1, v1 // (op 2988) 
v_lshl_add_u32 v29, v28, 5, v13 // (op 2988) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2988) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3513) 


v_mov_b64 v[28:29], v[16:17] // (op 2024) reg expression

s_mov_b32 m0, s15 // (op 2024) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 2024) Load value direct to lds


v_and_b32 v13, -4, v1 // (op 2913) 
v_sub_u32 v28, v1, v13 // (op 2913) 

v_lshlrev_b32 v13, 4, v28 // (op 2913) 

v_lshrrev_b32 v28, 2, v1 // (op 2913) 
v_lshl_add_u32 v29, v28, 6, v13 // (op 2913) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2913) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3494) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 2029) 
v_mov_b64 v[28:29], s[18:19] // (op 2029) convert


v_mov_b64 v[32:33], v[20:21] // (op 2029) reg expression
s_add_u32 s6, 4096, s15 // (op 2029) 
s_mov_b32 m0, s6 // (op 2029) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2029) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 2029) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2029) most significant half

s_add_u32 m0, m0, 4096 // (op 2029) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2029) Load value direct to lds






v_mov_b32 v13, v24 // (op 1742) reg expression

buffer_load_ubyte v28, v13, s[76:79], 0 offen offset: 0 // (op 1742) Load value


v_mov_b64 v[32:33], s[44:45] // (op 1746) convert

s_lshl_b32 s15, s44, 2 // (op 1746) 
v_add_u32 v13, v26, s15 // (op 1746) 

v_mov_b32 v29, v13 // (op 1746) 
buffer_load_ubyte v30, v29, s[80:83], 0 offen offset: 0 // (op 1746) Load value




v_and_b32 v13, -2, v1 // (op 3202) 
v_sub_u32 v26, v1, v13 // (op 3202) 

v_lshlrev_b32 v13, 4, v26 // (op 3202) 

v_lshrrev_b32 v26, 1, v1 // (op 3202) 
v_lshl_add_u32 v29, v26, 5, v13 // (op 3202) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 3202) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3561) 


s_lshl_b64 s[18:19], s[34:35], 7 // (op 2034) 
v_mov_b32 v13, s18 // (op 2034) 
v_mov_b32 v26, s19 // (op 2034) 
v_add_co_u32 v32, s[28:29], v16, v13 // (op 2034) least significant half
v_addc_co_u32 v33, s[28:29], v17, v26, s[28:29] // (op 2034) most significant half




s_add_u32 s6, 12288, s15 // (op 2034) 

s_mov_b32 m0, s6 // (op 2034) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 2034) Load value direct to lds




v_and_b32 v13, -4, v1 // (op 3158) 
v_sub_u32 v26, v1, v13 // (op 3158) 

v_lshlrev_b32 v13, 4, v26 // (op 3158) 

v_lshrrev_b32 v26, 2, v1 // (op 3158) 
v_lshl_add_u32 v29, v26, 6, v13 // (op 3158) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 3158) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3552) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 2036) 
v_mov_b64 v[16:17], s[18:19] // (op 2036) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 2036) 
v_mov_b32 v13, s18 // (op 2036) 
v_mov_b32 v26, s19 // (op 2036) 
v_add_co_u32 v32, s[28:29], v20, v13 // (op 2036) least significant half
v_addc_co_u32 v33, s[28:29], v21, v26, s[28:29] // (op 2036) most significant half




s_add_u32 s6, 16384, s15 // (op 2036) 
s_mov_b32 m0, s6 // (op 2036) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2036) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v16 // (op 2036) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2036) most significant half

s_add_u32 m0, m0, 4096 // (op 2036) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2036) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 1756) 
v_add_u32 v13, v24, s15 // (op 1756) 


buffer_load_ubyte v26, v13, s[76:79], 0 offen offset: 0 // (op 1756) Load value


v_and_b32 v13, -64, v1 // (op 3054) 
v_sub_u32 v24, v1, v13 // (op 3054) 

v_and_b32 v13, 4294967232, v1 // (op 3054) 
v_add_u32 v29, v13, v24 // (op 3054) 


s_barrier  // (op 3532) 



s_waitcnt vmcnt(5)
ds_write_b8 v29, v28 offset:24576 // (op 1760) Store local data 

s_barrier  // (op 3510) 



ds_read_b64_tr_b8 v[32:33], v12 // (op 1783) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v12 offset:256 // (op 1783) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v12 offset:2048 // (op 1783) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v12 offset:2304 // (op 1783) Transpose load local data 
s_barrier  // (op 3491) 



ds_read_b64_tr_b8 v[40:41], v0 offset:4096 // (op 1774) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4608 // (op 1774) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:8192 // (op 1774) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:8704 // (op 1774) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3529) 


ds_read_u8 v13, v3 offset:24576 // (op 1787) Load local data 



ds_read_b64_tr_b8 v[48:49], v0 offset:4128 // (op 1779) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4640 // (op 1779) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:8224 // (op 1779) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:8736 // (op 1779) Transpose load local data 


ds_read_u8 v24, v3 offset:24608 // (op 1792) Load local data 
v_and_b32 v29, -4, v1 // (op 2885) 
v_sub_u32 v16, v1, v29 // (op 2885) 

v_lshlrev_b32 v17, 4, v16 // (op 2885) 

v_lshrrev_b32 v16, 2, v1 // (op 2885) 
v_lshl_add_u32 v29, v16, 6, v17 // (op 2885) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -2, v1 // (op 2960) 
v_sub_u32 v16, v1, v29 // (op 2960) 

v_lshlrev_b32 v17, 4, v16 // (op 2960) 

v_lshrrev_b32 v16, 1, v1 // (op 2960) 
v_lshl_add_u32 v29, v16, 5, v17 // (op 2960) 


s_nop 0
v_readfirstlane_b32 s6, v29 // (op 2960) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -64, v1 // (op 3025) 
v_sub_u32 v16, v1, v29 // (op 3025) 

v_and_b32 v17, 4294967232, v1 // (op 3025) 
v_add_u32 v29, v17, v16 // (op 3025) 


v_and_b32 v16, -4, v1 // (op 3143) 
v_sub_u32 v17, v1, v16 // (op 3143) 

v_lshlrev_b32 v16, 4, v17 // (op 3143) 

v_lshrrev_b32 v17, 2, v1 // (op 3143) 
v_lshl_add_u32 v20, v17, 6, v16 // (op 3143) 


s_nop 0
v_readfirstlane_b32 s7, v20 // (op 3143) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 3187) 
v_sub_u32 v17, v1, v16 // (op 3187) 

v_lshlrev_b32 v16, 4, v17 // (op 3187) 

v_lshrrev_b32 v17, 1, v1 // (op 3187) 
v_lshl_add_u32 v20, v17, 5, v16 // (op 3187) 


s_nop 0
v_readfirstlane_b32 s8, v20 // (op 3187) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -64, v1 // (op 3226) 
v_sub_u32 v17, v1, v16 // (op 3226) 

v_and_b32 v16, 4294967232, v1 // (op 3226) 
v_add_u32 v20, v16, v17 // (op 3226) 


s_mov_b32 s1, 0 // (op 583) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 582) 
s_mov_b32 s28, s18 // (op 582) convert

s_and_b32 s29, s28, 4294967294 // (op 582) 

s_cmp_lt_u32 s1, s29 // (op 582) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_582 // (op 582) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_582 if false)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_582:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v13, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 184) 
s_add_u32 s1, s1, 2 // (op 584) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v24, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 618) 

v_mov_b64 v[16:17], s[44:45] // (op 183) convert

v_mov_b32 v21, v27 // (op 183) 
s_lshl_b32 s18, s44, 3 // (op 183) 
v_add_u32 v56, v21, s18 // (op 183) 

v_mov_b32 v57, v56 // (op 183) 
buffer_load_ubyte v31, v57, s[80:83], 0 offen offset: 0 // (op 183) Load value



s_barrier  // (op 3516) 


v_mov_b64 v[16:17], v[18:19] // (op 1973) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 1973) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1973) 

v_mov_b32 v56, s28 // (op 1973) 
v_mov_b32 v57, s29 // (op 1973) 
v_add_co_u32 v58, s[18:19], v16, v56 // (op 1973) least significant half
v_addc_co_u32 v59, s[18:19], v17, v57, s[18:19] // (op 1973) most significant half





s_mov_b32 m0, s6 // (op 1973) reg expression
buffer_load_dwordx4 v58, s[72:75], 0 offen offset: 0 lds // (op 1973) Load value direct to lds

s_barrier  // (op 3497) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1968) 
v_mov_b64 v[56:57], s[18:19] // (op 1968) convert


v_mov_b64 v[58:59], v[22:23] // (op 1968) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 1968) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1968) 

v_mov_b32 v60, s28 // (op 1968) 
v_mov_b32 v61, s29 // (op 1968) 
v_add_co_u32 v62, s[18:19], v58, v60 // (op 1968) least significant half
v_addc_co_u32 v63, s[18:19], v59, v61, s[18:19] // (op 1968) most significant half




s_add_u32 s18, 4096, s15 // (op 1968) 
s_nop 0
s_mov_b32 m0, s18 // (op 1968) reg expression
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 1968) Load value direct to lds
v_add_co_u32 v62, s[28:29], v62, v56 // (op 1968) least significant half
v_addc_co_u32 v63, s[28:29], 0, v63, s[28:29] // (op 1968) most significant half

s_add_u32 m0, m0, 4096 // (op 1968) 
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 1968) Load value direct to lds





v_mov_b32 v56, v25 // (op 188) 
s_lshl_b32 s18, s38, 3 // (op 188) 
v_add_u32 v57, v56, s18 // (op 188) 


buffer_load_ubyte v28, v57, s[76:79], 0 offen offset: 0 // (op 188) Load value

s_barrier  // (op 3570) 



s_waitcnt vmcnt(5)
ds_write_b8 v20, v26 offset:24832 // (op 508) Store local data 
s_barrier  // (op 3558) 



ds_read_b64_tr_b8 v[32:33], v5 offset:12288 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:12544 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:14336 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:14592 // (op 501) Transpose load local data 
s_barrier  // (op 3549) 



ds_read_b64_tr_b8 v[40:41], v10 offset:16384 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:16896 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:20480 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:20992 // (op 499) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3567) 


ds_read_u8 v13, v11 offset:24832 // (op 503) Load local data 



ds_read_b64_tr_b8 v[48:49], v10 offset:16416 // (op 625) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v10 offset:16928 // (op 625) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v10 offset:20512 // (op 625) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v10 offset:21024 // (op 625) Transpose load local data 


ds_read_u8 v24, v11 offset:24864 // (op 629) Load local data 
s_barrier  // (op 3681) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v13, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 506) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[32:39], a[4:7], v24, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 619) 

v_mov_b64 v[60:61], s[44:45] // (op 505) convert

s_lshl_b32 s18, s44, 2 // (op 505) 
s_mul_i32 s19, 3, s18 // (op 505) 

v_add_u32 v57, v21, s19 // (op 505) 

v_mov_b32 v62, v57 // (op 505) 
buffer_load_ubyte v30, v62, s[80:83], 0 offen offset: 0 // (op 505) Load value




s_barrier  // (op 3564) 
s_lshl_b32 s18, s44, 3 // (op 2513) 


v_add_u32 v27, v27, s18 // (op 2513) 

s_lshl_b64 s[18:19], s[34:35], 7 // (op 1996) 
s_mul_i32 s29, 3, s19 // (op 1996) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 1996) least significant
s_mul_hi_u32 s48, 3, s18 // (op 1996) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 1996) most significant: sum


v_mov_b32 v21, s28 // (op 1996) 
v_mov_b32 v57, s29 // (op 1996) 
v_add_co_u32 v60, s[18:19], v16, v21 // (op 1996) least significant half
v_addc_co_u32 v61, s[18:19], v17, v57, s[18:19] // (op 1996) most significant half




s_add_u32 s18, 12288, s8 // (op 1996) 

s_nop 0
s_mov_b32 m0, s18 // (op 1996) reg expression
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 1996) Load value direct to lds



s_barrier  // (op 3555) 
s_lshl_b64 s[18:19], s[34:35], 7 // (op 2257) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 2257) 

v_mov_b32 v21, s28 // (op 2257) 
v_mov_b32 v57, s29 // (op 2257) 
s_lshl_b64 s[18:19], s[30:31], 6 // (op 1991) 
v_mov_b64 v[16:17], s[18:19] // (op 1991) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 1991) 
s_mul_i32 s49, 3, s19 // (op 1991) most significant: low of low * high
s_mul_i32 s48, 3, s18 // (op 1991) least significant
s_mul_hi_u32 s52, 3, s18 // (op 1991) most significant: high of low * low
s_add_u32 s49, s49, s52 // (op 1991) most significant: sum


v_mov_b32 v60, s48 // (op 1991) 
v_mov_b32 v61, s49 // (op 1991) 
v_add_co_u32 v18, s[18:19], v18, v21 // (op 2257) least significant half
v_addc_co_u32 v19, s[18:19], v19, v57, s[18:19] // (op 2257) most significant half




v_add_co_u32 v62, s[18:19], v58, v60 // (op 1991) least significant half
v_addc_co_u32 v63, s[18:19], v59, v61, s[18:19] // (op 1991) most significant half




s_add_u32 s18, 16384, s7 // (op 1991) 
s_nop 0
s_mov_b32 m0, s18 // (op 1991) reg expression
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 1991) Load value direct to lds
v_add_co_u32 v62, s[28:29], v62, v16 // (op 1991) least significant half
v_addc_co_u32 v63, s[28:29], 0, v63, s[28:29] // (op 1991) most significant half

s_add_u32 m0, m0, 4096 // (op 1991) 
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 1991) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 510) 
s_mul_i32 s19, 3, s18 // (op 510) 

v_add_u32 v21, v56, s19 // (op 510) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 2125) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2125) 

v_mov_b32 v16, s28 // (op 2125) 
v_mov_b32 v17, s29 // (op 2125) 
v_add_co_u32 v22, s[18:19], v22, v16 // (op 2125) least significant half
v_addc_co_u32 v23, s[18:19], v23, v17, s[18:19] // (op 2125) most significant half




buffer_load_ubyte v26, v21, s[76:79], 0 offen offset: 0 // (op 510) Load value


s_lshl_b32 s18, s38, 3 // (op 2385) 
s_barrier  // (op 3535) 
v_add_u32 v25, v25, s18 // (op 2385) 




s_waitcnt vmcnt(5)
ds_write_b8 v29, v28 offset:24576 // (op 186) Store local data 
s_barrier  // (op 3522) 



ds_read_b64_tr_b8 v[32:33], v4 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:256 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:2048 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:2304 // (op 179) Transpose load local data 
s_barrier  // (op 3503) 



ds_read_b64_tr_b8 v[40:41], v8 offset:4096 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:4608 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:8192 // (op 177) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:8704 // (op 177) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 3542) 


ds_read_u8 v13, v9 offset:24576 // (op 181) Load local data 



ds_read_b64_tr_b8 v[48:49], v8 offset:4128 // (op 612) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:4640 // (op 612) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:8224 // (op 612) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:8736 // (op 612) Transpose load local data 


ds_read_u8 v24, v9 offset:24608 // (op 616) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 582) 
s_mov_b32 s28, s18 // (op 582) convert

s_and_b32 s29, s28, 4294967294 // (op 582) 

s_cmp_lt_u32 s1, s29 // (op 582) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_582 // (op 582) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_582 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_582:












s_barrier  // (op 3665) 
v_and_b32 v25, -4, v1 // (op 2063) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2063) 
v_sub_u32 v31, v1, v25 // (op 2063) 

s_and_b32 s15, s18, 4294967294 // (op 2063) 
v_lshlrev_b32 v25, 4, v31 // (op 2063) 

s_add_u32 s6, s0, s15 // (op 2063) 

v_lshrrev_b32 v31, 2, v1 // (op 2063) 
s_mul_i32 s15, s16, s50 // (op 2063) 
s_add_u32 s15, s15, s6 // (op 2063) 

s_add_u32 s6, s15, s9 // (op 2063) 

s_mul_hi_u32 s15, s6, s13 // (op 2063) 
s_sub_u32 s7, s6, s15 // (op 2063) 
s_lshr_b32 s8, s7, 1 // (op 2063) 

s_add_u32 s7, s8, s15 // (op 2063) 


s_lshr_b32 s8, s7, s11 // (op 2063) 

s_and_b32 s7, s11, 2147483648 // (op 2063) 
s_cmp_eq_i32 s7, 0 // (op 2063) 
s_mov_b32 s15, scc // (op 2063) 

s_cmp_eq_i32 s15, 0 // (op 2063) 
s_cselect_b32 s7, s6, s8 // (op 2063) 


s_mul_hi_u32 s8, s7, s12 // (op 2063) 
s_sub_u32 s15, s7, s8 // (op 2063) 
s_lshr_b32 s28, s15, 1 // (op 2063) 

s_add_u32 s15, s28, s8 // (op 2063) 


s_lshr_b32 s8, s15, s10 // (op 2063) 

s_and_b32 s15, s10, 2147483648 // (op 2063) 
s_cmp_eq_i32 s15, 0 // (op 2063) 
s_mov_b32 s28, scc // (op 2063) 

s_cmp_eq_i32 s28, 0 // (op 2063) 
s_cselect_b32 s15, s7, s8 // (op 2063) 


s_mul_i32 s8, s7, s18 // (op 2063) 


v_lshl_add_u32 v16, s15, 6, v25 // (op 2063) 


s_sub_u32 s7, s6, s8 // (op 2063) 


v_mov_b32 v29, 0 // (op 2063) convert
v_mov_b32 v28, v16 // (op 2063) convert

v_lshl_add_u32 v25, s7, 7, v31 // (op 2063) 


v_mul_lo_u32 v17, v25, s31 // (op 2063) most significant: low of low * high
v_mul_hi_u32 v27, v25, s30 // (op 2063) most significant: high of low * low
v_mul_lo_u32 v16, v25, s30 // (op 2063) least significant: low of low * low
v_add_u32 v17, v17, v27 // (op 2063) most significant: sum


v_add_co_u32 v18, s[18:19], v28, v16 // (op 2063) least significant half
v_addc_co_u32 v19, s[18:19], v29, v17, s[18:19] // (op 2063) most significant half



s_mul_i32 s19, s30, s3 // (op 2065) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2065) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2065) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2065) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2065) most significant: sum
s_add_u32 s19, s19, s6 // (op 2065) most significant: sum


s_mov_b32 s68, s32 // (op 2065) reg expression
s_mov_b32 s69, s33 // (op 2065) reg expression
s_mov_b32 s70, s18 // (op 2065) reg expression
s_mov_b32 s71, 131072 // (op 2065) call()

v_and_b32 v25, -2, v1 // (op 2201) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2201) 
v_sub_u32 v31, v1, v25 // (op 2201) 

s_and_b32 s15, s18, 4294967294 // (op 2201) 
v_lshlrev_b32 v25, 4, v31 // (op 2201) 

s_add_u32 s6, s0, s15 // (op 2201) 

v_lshrrev_b32 v31, 1, v1 // (op 2201) 
s_mul_i32 s15, s16, s50 // (op 2201) 
s_add_u32 s15, s15, s6 // (op 2201) 

s_add_u32 s6, s15, s9 // (op 2201) 

s_mul_hi_u32 s15, s6, s13 // (op 2201) 
s_sub_u32 s7, s6, s15 // (op 2201) 
s_lshr_b32 s8, s7, 1 // (op 2201) 

s_add_u32 s7, s8, s15 // (op 2201) 


s_lshr_b32 s8, s7, s11 // (op 2201) 

s_and_b32 s7, s11, 2147483648 // (op 2201) 
s_cmp_eq_i32 s7, 0 // (op 2201) 
s_mov_b32 s15, scc // (op 2201) 

s_cmp_eq_i32 s15, 0 // (op 2201) 
s_cselect_b32 s7, s6, s8 // (op 2201) 


s_mul_hi_u32 s8, s7, s12 // (op 2201) 
s_sub_u32 s15, s7, s8 // (op 2201) 
s_lshr_b32 s28, s15, 1 // (op 2201) 

s_add_u32 s15, s28, s8 // (op 2201) 


s_lshr_b32 s8, s15, s10 // (op 2201) 

s_and_b32 s15, s10, 2147483648 // (op 2201) 
s_cmp_eq_i32 s15, 0 // (op 2201) 
s_mov_b32 s28, scc // (op 2201) 

s_cmp_eq_i32 s28, 0 // (op 2201) 
s_cselect_b32 s15, s7, s8 // (op 2201) 


s_mul_i32 s8, s15, s14 // (op 2201) 

s_sub_u32 s15, s7, s8 // (op 2201) 

s_mul_i32 s8, s7, s18 // (op 2201) 


v_lshl_add_u32 v16, s15, 5, v25 // (op 2201) 


s_sub_u32 s7, s6, s8 // (op 2201) 


v_mov_b32 v29, 0 // (op 2201) convert
v_mov_b32 v28, v16 // (op 2201) convert

v_lshl_add_u32 v25, s7, 7, v31 // (op 2201) 


v_mul_lo_u32 v17, v25, s35 // (op 2201) most significant: low of low * high
v_mul_hi_u32 v27, v25, s34 // (op 2201) most significant: high of low * low
v_mul_lo_u32 v16, v25, s34 // (op 2201) least significant: low of low * low
v_add_u32 v17, v17, v27 // (op 2201) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v28 // (op 2201) least significant half
v_addc_co_u32 v17, s[18:19], v17, v29, s[18:19] // (op 2201) most significant half



s_mul_i32 s19, s34, s3 // (op 2203) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2203) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2203) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2203) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2203) most significant: sum
s_add_u32 s19, s19, s6 // (op 2203) most significant: sum


s_mov_b32 s72, s36 // (op 2203) reg expression
s_mov_b32 s73, s37 // (op 2203) reg expression
s_mov_b32 s74, s18 // (op 2203) reg expression
s_mov_b32 s75, 131072 // (op 2203) call()

v_and_b32 v25, -64, v1 // (op 2329) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2329) 
v_sub_u32 v31, v1, v25 // (op 2329) 

s_and_b32 s15, s18, 4294967294 // (op 2329) 
v_lshrrev_b32 v25, 6, v1 // (op 2329) 
s_add_u32 s6, s0, s15 // (op 2329) 

s_mul_i32 s15, s16, s50 // (op 2329) 
s_add_u32 s15, s15, s6 // (op 2329) 

s_add_u32 s6, s15, s9 // (op 2329) 

s_mul_hi_u32 s15, s6, s13 // (op 2329) 
s_sub_u32 s7, s6, s15 // (op 2329) 
s_lshr_b32 s8, s7, 1 // (op 2329) 

s_add_u32 s7, s8, s15 // (op 2329) 


s_lshr_b32 s8, s7, s11 // (op 2329) 

s_and_b32 s7, s11, 2147483648 // (op 2329) 
s_cmp_eq_i32 s7, 0 // (op 2329) 
s_mov_b32 s15, scc // (op 2329) 

s_cmp_eq_i32 s15, 0 // (op 2329) 
s_cselect_b32 s7, s6, s8 // (op 2329) 


s_mul_hi_u32 s8, s7, s12 // (op 2329) 
s_sub_u32 s15, s7, s8 // (op 2329) 
s_lshr_b32 s28, s15, 1 // (op 2329) 

s_add_u32 s15, s28, s8 // (op 2329) 


s_lshr_b32 s8, s15, s10 // (op 2329) 

s_and_b32 s15, s10, 2147483648 // (op 2329) 
s_cmp_eq_i32 s15, 0 // (op 2329) 
s_mov_b32 s28, scc // (op 2329) 

s_cmp_eq_i32 s28, 0 // (op 2329) 
s_cselect_b32 s15, s7, s8 // (op 2329) 


s_mul_i32 s8, s7, s18 // (op 2329) 


v_lshl_add_u32 v20, s15, 6, v31 // (op 2329) 


s_sub_u32 s7, s6, s8 // (op 2329) 


v_lshl_add_u32 v31, s7, 2, v25 // (op 2329) 


v_mul_lo_u32 v25, v31, s38 // (op 2329) 

v_add_u32 v31, v20, v25 // (op 2329) 


s_mul_i32 s19, s38, s43 // (op 2331) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2331) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2331) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2331) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2331) most significant: sum
s_add_u32 s19, s19, s6 // (op 2331) most significant: sum


s_mov_b32 s76, s40 // (op 2331) reg expression
s_mov_b32 s77, s41 // (op 2331) reg expression
s_mov_b32 s78, s18 // (op 2331) reg expression
s_mov_b32 s79, 131072 // (op 2331) call()

v_lshrrev_b32 v25, 6, v1 // (op 2457) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2457) 
v_and_b32 v20, 1, v25 // (op 2457) 

s_and_b32 s15, s18, 4294967294 // (op 2457) 
v_and_b32 v25, 15, v1 // (op 2457) 
s_add_u32 s6, s0, s15 // (op 2457) 

v_lshl_add_u32 v21, v20, 4, v25 // (op 2457) 


v_and_b32 v20, 63, v1 // (op 2457) 
s_mul_i32 s15, s16, s50 // (op 2457) 
s_add_u32 s15, s15, s6 // (op 2457) 

v_lshrrev_b32 v25, 4, v20 // (op 2457) 

s_add_u32 s6, s15, s9 // (op 2457) 

s_mul_hi_u32 s15, s6, s13 // (op 2457) 
s_sub_u32 s7, s6, s15 // (op 2457) 
s_lshr_b32 s8, s7, 1 // (op 2457) 

s_add_u32 s7, s8, s15 // (op 2457) 


s_lshr_b32 s8, s7, s11 // (op 2457) 

s_and_b32 s7, s11, 2147483648 // (op 2457) 
s_cmp_eq_i32 s7, 0 // (op 2457) 
s_mov_b32 s15, scc // (op 2457) 

s_cmp_eq_i32 s15, 0 // (op 2457) 
s_cselect_b32 s7, s6, s8 // (op 2457) 


s_mul_hi_u32 s8, s7, s12 // (op 2457) 
s_sub_u32 s15, s7, s8 // (op 2457) 
s_lshr_b32 s28, s15, 1 // (op 2457) 

s_add_u32 s15, s28, s8 // (op 2457) 


s_lshr_b32 s8, s15, s10 // (op 2457) 

s_and_b32 s15, s10, 2147483648 // (op 2457) 
s_cmp_eq_i32 s15, 0 // (op 2457) 
s_mov_b32 s28, scc // (op 2457) 

s_cmp_eq_i32 s28, 0 // (op 2457) 
s_cselect_b32 s15, s7, s8 // (op 2457) 


s_mul_i32 s8, s15, s14 // (op 2457) 

s_sub_u32 s15, s7, s8 // (op 2457) 

s_mul_i32 s8, s7, s18 // (op 2457) 


v_lshl_add_u32 v20, s15, 5, v21 // (op 2457) 


s_sub_u32 s7, s6, s8 // (op 2457) 


v_lshl_add_u32 v21, s7, 2, v25 // (op 2457) 


v_mul_lo_u32 v25, v21, s44 // (op 2457) 
v_add_u32 v25, v25, v20 // (op 2457) 


s_mul_i32 s19, s44, s43 // (op 2459) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 2459) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 2459) least significant
s_mul_hi_u32 s6, s44, s42 // (op 2459) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2459) most significant: sum
s_add_u32 s19, s19, s6 // (op 2459) most significant: sum


s_mov_b32 s80, s46 // (op 2459) reg expression
s_mov_b32 s81, s47 // (op 2459) reg expression
s_mov_b32 s82, s18 // (op 2459) reg expression
s_mov_b32 s83, 131072 // (op 2459) call()

v_and_b32 v20, -4, v1 // (op 2898) 
v_sub_u32 v21, v1, v20 // (op 2898) 

v_lshlrev_b32 v20, 4, v21 // (op 2898) 

v_lshrrev_b32 v21, 2, v1 // (op 2898) 
v_lshl_add_u32 v22, v21, 6, v20 // (op 2898) 


s_nop 0
v_readfirstlane_b32 s15, v22 // (op 2898) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -2, v1 // (op 2973) 
v_sub_u32 v21, v1, v20 // (op 2973) 

v_lshlrev_b32 v20, 4, v21 // (op 2973) 

v_lshrrev_b32 v21, 1, v1 // (op 2973) 
v_lshl_add_u32 v22, v21, 5, v20 // (op 2973) 


s_nop 0
v_readfirstlane_b32 s6, v22 // (op 2973) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 3038) 
v_sub_u32 v21, v1, v20 // (op 3038) 

v_and_b32 v20, 4294967232, v1 // (op 3038) 
v_add_u32 v22, v20, v21 // (op 3038) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 588) 
s_and_b32 s7, s18, 4294967294 // (op 588) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 587) 
s_mov_b32 s8, s18 // (op 587) convert

s_cmp_lt_u32 s7, s8 // (op 587) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_587 // (op 587) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_587 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_587:
s_barrier  // (op 3500) 
s_add_u32 s7, s7, 1 // (op 589) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1978) 
v_mov_b64 v[20:21], s[18:19] // (op 1978) convert


v_mov_b64 v[26:27], v[18:19] // (op 1978) 
v_mov_b64 v[28:29], v[26:27] // (op 1978) reg expression
s_add_u32 s8, 4096, s15 // (op 1978) 
s_mov_b32 m0, s8 // (op 1978) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 1978) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 1978) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1978) most significant half

s_add_u32 m0, m0, 4096 // (op 1978) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 1978) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 2053) 
v_mov_b32 v13, s18 // (op 2053) 
v_mov_b32 v20, s19 // (op 2053) 
v_add_co_u32 v18, s[28:29], v18, v13 // (op 2053) least significant half
v_addc_co_u32 v19, s[28:29], v19, v20, s[28:29] // (op 2053) most significant half




s_barrier  // (op 3519) 


v_mov_b64 v[20:21], v[16:17] // (op 1984) 
v_mov_b64 v[26:27], v[20:21] // (op 1984) reg expression

s_mov_b32 m0, s6 // (op 1984) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 1984) Load value direct to lds


s_lshl_b64 s[18:19], s[34:35], 7 // (op 2193) 
v_mov_b32 v13, s18 // (op 2193) 
v_mov_b32 v20, s19 // (op 2193) 
v_add_co_u32 v16, s[28:29], v16, v13 // (op 2193) least significant half
v_addc_co_u32 v17, s[28:29], v17, v20, s[28:29] // (op 2193) most significant half






v_mov_b32 v13, v31 // (op 452) 
v_mov_b32 v20, v13 // (op 452) reg expression

buffer_load_ubyte v21, v20, s[76:79], 0 offen offset: 0 // (op 452) Load value


s_lshl_b32 s8, s38, 2 // (op 2321) 
v_add_u32 v31, v31, s8 // (op 2321) 

s_barrier  // (op 3538) 
s_waitcnt vmcnt(2)
s_barrier  // (op 3506) 






ds_read_b64_tr_b8 v[32:33], v8 offset:4096 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:4608 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v8 offset:8192 // (op 441) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v8 offset:8704 // (op 441) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v21 offset:24576 // (op 450) Store local data 

s_barrier  // (op 3525) 



ds_read_b64_tr_b8 v[40:41], v4 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:256 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:2048 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:2304 // (op 443) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3545) 


ds_read_u8 v13, v9 offset:24576 // (op 445) Load local data 

v_mov_b64 v[20:21], s[44:45] // (op 447) convert

v_mov_b32 v23, v25 // (op 447) 
v_mov_b32 v24, v23 // (op 447) reg expression
v_mov_b32 v26, v24 // (op 447) 
buffer_load_ubyte v27, v26, s[80:83], 0 offen offset: 0 // (op 447) Load value




s_lshl_b32 s8, s44, 2 // (op 2449) 
v_add_u32 v25, v25, s8 // (op 2449) 




ds_read_b64_tr_b8 v[48:49], v8 offset:4128 // (op 593) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:4640 // (op 593) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v13, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 448) 
ds_read_b64_tr_b8 v[52:53], v8 offset:8224 // (op 593) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v8 offset:8736 // (op 593) Transpose load local data 


ds_read_u8 v13, v9 offset:24608 // (op 597) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v13, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 599) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 587) 
s_mov_b32 s8, s18 // (op 587) convert

s_cmp_lt_u32 s7, s8 // (op 587) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_587 // (op 587) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_587 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_587:








s_barrier  // (op 3669) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_218
v_accvgpr_read v16, a0 // (op 226) 
v_accvgpr_read v17, a1 // (op 226) 
v_accvgpr_read v18, a2 // (op 226) 
v_accvgpr_read v19, a3 // (op 226) 
v_mul_f32 v20, v15, v16 // (op 226) 
v_mul_f32 v21, v15, v17 // (op 226) 
v_mul_f32 v22, v15, v18 // (op 226) 
v_mul_f32 v23, v15, v19 // (op 226) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen // (op 228) Store value

v_accvgpr_read v16, a4 // (op 785) 
v_accvgpr_read v17, a5 // (op 785) 
v_accvgpr_read v18, a6 // (op 785) 
v_accvgpr_read v19, a7 // (op 785) 
v_mul_f32 v20, v15, v16 // (op 785) 
v_mul_f32 v21, v15, v17 // (op 785) 
v_mul_f32 v22, v15, v18 // (op 785) 
v_mul_f32 v23, v15, v19 // (op 785) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen offset:128 // (op 787) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_218:



buffer_load_dwordx4 v[16:19], v6, s[56:59], 0 offen offset:128 // (op 815) Load value
v_accvgpr_read v20, a4 // (op 812) 
v_accvgpr_read v21, a5 // (op 812) 
v_accvgpr_read v22, a6 // (op 812) 
v_accvgpr_read v23, a7 // (op 812) 
v_mul_f32 v24, v15, v20 // (op 812) 
v_mul_f32 v25, v15, v21 // (op 812) 
v_mul_f32 v26, v15, v22 // (op 812) 
v_mul_f32 v27, v15, v23 // (op 812) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 816) 
v_mul_f32 v21, v14, v17 // (op 816) 
v_mul_f32 v22, v14, v18 // (op 816) 
v_mul_f32 v23, v14, v19 // (op 816) 

v_add_f32 v16, v20, v24 // (op 813) 
v_add_f32 v17, v21, v25 // (op 813) 
v_add_f32 v18, v22, v26 // (op 813) 
v_add_f32 v19, v23, v27 // (op 813) 





buffer_store_dwordx4 v[16:19], v7, s[60:63], 0 offen offset:128 // (op 814) Store value




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
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_218:
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
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:









































s_endpgm  // End of RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.LRR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2, .LRR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_end-RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
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
    DataType: F8
    DataTypeA: F8
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
    macrotile: [64, 32, 128]
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
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 91
    .vgpr_count: 72
    .agpr_count: 8
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

