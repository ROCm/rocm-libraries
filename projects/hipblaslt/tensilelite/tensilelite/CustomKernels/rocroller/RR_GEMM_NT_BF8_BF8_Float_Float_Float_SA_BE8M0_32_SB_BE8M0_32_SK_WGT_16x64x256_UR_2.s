// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2,@function
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 3260) 
v_and_b32 v3, 63, v1 // (op 3260) 
v_sub_u32 v4, v3, v0 // (op 3260) 

v_lshrrev_b32 v3, 1, v4 // (op 3260) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 3260) 


v_and_b32 v0, 1, v4 // (op 3260) 

v_lshlrev_b32 v3, 3, v0 // (op 3260) 

v_lshrrev_b32 v0, 7, v1 // (op 3260) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3260) 


v_add_u32 v0, v5, v4 // (op 3260) 


v_lshrrev_b32 v3, 7, v1 // (op 3401) 
v_and_b32 v4, 15, v1 // (op 3401) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 3401) 


v_and_b32 v3, 48, v1 // (op 3401) 
v_add_u32 v4, v3, v5 // (op 3401) 


v_and_b32 v3, 48, v1 // (op 3584) 
v_and_b32 v5, 63, v1 // (op 3584) 
v_sub_u32 v6, v5, v3 // (op 3584) 

v_lshrrev_b32 v5, 1, v6 // (op 3584) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 3584) 


v_and_b32 v3, 1, v6 // (op 3584) 

v_lshlrev_b32 v5, 3, v3 // (op 3584) 

v_lshrrev_b32 v3, 7, v1 // (op 3584) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 3584) 


v_add_u32 v3, v7, v6 // (op 3584) 


v_lshrrev_b32 v5, 7, v1 // (op 3725) 
v_and_b32 v6, 15, v1 // (op 3725) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 3725) 


v_and_b32 v5, 48, v1 // (op 3725) 
v_add_u32 v6, v5, v7 // (op 3725) 


v_and_b32 v5, 48, v1 // (op 3895) 
v_and_b32 v7, 63, v1 // (op 3895) 
v_sub_u32 v8, v7, v5 // (op 3895) 

v_lshrrev_b32 v7, 1, v8 // (op 3895) 
v_add_lshl_u32 v9, v5, v7, 4 // (op 3895) 


v_and_b32 v5, 1, v8 // (op 3895) 

v_lshlrev_b32 v7, 3, v5 // (op 3895) 

v_lshrrev_b32 v5, 7, v1 // (op 3895) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 3895) 


v_add_u32 v5, v9, v8 // (op 3895) 


v_lshrrev_b32 v7, 7, v1 // (op 3963) 
v_and_b32 v8, 15, v1 // (op 3963) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 3963) 


v_and_b32 v7, 48, v1 // (op 3963) 
v_add_u32 v8, v7, v9 // (op 3963) 


v_and_b32 v7, 48, v1 // (op 4031) 
v_and_b32 v9, 63, v1 // (op 4031) 
v_sub_u32 v10, v9, v7 // (op 4031) 

v_lshrrev_b32 v9, 1, v10 // (op 4031) 
v_add_lshl_u32 v11, v7, v9, 4 // (op 4031) 


v_and_b32 v7, 1, v10 // (op 4031) 

v_lshlrev_b32 v9, 3, v7 // (op 4031) 

v_lshrrev_b32 v7, 7, v1 // (op 4031) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 4031) 


v_add_u32 v7, v11, v10 // (op 4031) 


v_lshrrev_b32 v9, 7, v1 // (op 4099) 
v_and_b32 v10, 15, v1 // (op 4099) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 4099) 


v_and_b32 v9, 48, v1 // (op 4099) 
v_add_u32 v10, v9, v11 // (op 4099) 


v_and_b32 v9, 48, v1 // (op 3330) 
v_and_b32 v11, 63, v1 // (op 3330) 
v_sub_u32 v12, v11, v9 // (op 3330) 

v_lshrrev_b32 v11, 1, v12 // (op 3330) 
v_add_lshl_u32 v13, v9, v11, 6 // (op 3330) 


v_and_b32 v9, 1, v12 // (op 3330) 

v_lshlrev_b32 v11, 3, v9 // (op 3330) 

v_lshrrev_b32 v9, 6, v1 // (op 3330) 
v_and_b32 v12, 1, v9 // (op 3330) 

v_lshl_add_u32 v9, v12, 4, v11 // (op 3330) 


v_add_u32 v11, v13, v9 // (op 3330) 


v_lshrrev_b32 v9, 6, v1 // (op 3472) 
v_and_b32 v12, 1, v9 // (op 3472) 

v_and_b32 v9, 15, v1 // (op 3472) 
v_lshl_add_u32 v13, v12, 4, v9 // (op 3472) 


v_and_b32 v9, 63, v1 // (op 3472) 
v_lshrrev_b32 v12, 4, v9 // (op 3472) 

v_lshl_add_u32 v9, v12, 6, v13 // (op 3472) 


v_and_b32 v12, 48, v1 // (op 3654) 
v_and_b32 v13, 63, v1 // (op 3654) 
v_sub_u32 v14, v13, v12 // (op 3654) 

v_lshrrev_b32 v13, 1, v14 // (op 3654) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3654) 


v_and_b32 v12, 1, v14 // (op 3654) 

v_lshlrev_b32 v13, 3, v12 // (op 3654) 

v_lshrrev_b32 v12, 6, v1 // (op 3654) 
v_and_b32 v14, 1, v12 // (op 3654) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 3654) 


v_add_u32 v13, v15, v12 // (op 3654) 


v_lshrrev_b32 v12, 6, v1 // (op 3796) 
v_and_b32 v14, 1, v12 // (op 3796) 

v_and_b32 v12, 15, v1 // (op 3796) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 3796) 


v_and_b32 v12, 63, v1 // (op 3796) 
v_lshrrev_b32 v14, 4, v12 // (op 3796) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 3796) 


v_and_b32 v14, 48, v1 // (op 3939) 
v_and_b32 v15, 63, v1 // (op 3939) 
v_sub_u32 v16, v15, v14 // (op 3939) 

v_lshrrev_b32 v15, 1, v16 // (op 3939) 
v_add_lshl_u32 v17, v14, v15, 6 // (op 3939) 


v_and_b32 v14, 1, v16 // (op 3939) 

v_lshlrev_b32 v15, 3, v14 // (op 3939) 

v_lshrrev_b32 v14, 6, v1 // (op 3939) 
v_and_b32 v16, 1, v14 // (op 3939) 

v_lshl_add_u32 v14, v16, 4, v15 // (op 3939) 


v_add_u32 v15, v17, v14 // (op 3939) 


v_lshrrev_b32 v14, 6, v1 // (op 3992) 
v_and_b32 v16, 1, v14 // (op 3992) 

v_and_b32 v14, 15, v1 // (op 3992) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 3992) 


v_and_b32 v14, 63, v1 // (op 3992) 
v_lshrrev_b32 v16, 4, v14 // (op 3992) 

v_lshl_add_u32 v14, v16, 6, v17 // (op 3992) 


v_and_b32 v16, 48, v1 // (op 4075) 
v_and_b32 v17, 63, v1 // (op 4075) 
v_sub_u32 v18, v17, v16 // (op 4075) 

v_lshrrev_b32 v17, 1, v18 // (op 4075) 
v_add_lshl_u32 v19, v16, v17, 6 // (op 4075) 


v_and_b32 v16, 1, v18 // (op 4075) 

v_lshlrev_b32 v17, 3, v16 // (op 4075) 

v_lshrrev_b32 v16, 6, v1 // (op 4075) 
v_and_b32 v18, 1, v16 // (op 4075) 

v_lshl_add_u32 v16, v18, 4, v17 // (op 4075) 


v_add_u32 v17, v19, v16 // (op 4075) 


v_lshrrev_b32 v16, 6, v1 // (op 4128) 
v_and_b32 v18, 1, v16 // (op 4128) 

v_and_b32 v16, 15, v1 // (op 4128) 
v_lshl_add_u32 v19, v18, 4, v16 // (op 4128) 


v_and_b32 v16, 63, v1 // (op 4128) 
v_lshrrev_b32 v18, 4, v16 // (op 4128) 

v_lshl_add_u32 v16, v18, 6, v19 // (op 4128) 


v_mov_b32 v18, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 427) call()
s_mov_b32 s1, 0 // (op 428) call()
s_mov_b32 s6, 0 // (op 429) call()
s_mov_b32 s7, 0 // (op 430) call()
s_mov_b32 s18, 0 // (op 431) call()
s_mov_b32 s19, 0 // (op 432) call()
s_mov_b32 s0, 0 // (op 327) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 326) 
s_cmp_gt_u32 s9, s52 // (op 326) 
s_mov_b32 s53, scc // (op 326) 
s_sub_u32 s54, s9, s52 // (op 326) 

s_cmp_eq_i32 s53, 0 // (op 326) 
s_cselect_b32 s52, 0, s54 // (op 326) 


s_cmp_lt_u32 s8, s52 // (op 326) 
s_mov_b32 s53, scc // (op 326) 
s_cmp_eq_i32 s53, 0 // (op 326) 
s_cselect_b32 s54, s52, s8 // (op 326) 


s_cmp_lt_u32 s0, s54 // (op 326) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_326 // (op 326) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_326 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_326:
v_lshrrev_b32 v20, 7, v1 // (op 3181) 
s_mul_i32 s52, s16, s8 // (op 3181) 
s_add_u32 s52, s52, s0 // (op 3181) 
v_and_b32 v21, 63, v1 // (op 3181) 
s_mul_hi_u32 s53, s52, s13 // (op 3181) 
v_lshrrev_b32 v22, 4, v21 // (op 3181) 
s_sub_u32 s54, s52, s53 // (op 3181) 
v_lshlrev_b32 v23, 2, v22 // (op 3181) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 3181) 


s_lshr_b32 s55, s54, 1 // (op 3181) 

s_add_u32 s54, s55, s53 // (op 3181) 


v_lshrrev_b32 v20, 2, v21 // (op 3181) 

s_lshr_b32 s53, s54, s11 // (op 3181) 

v_and_b32 v21, 3, v20 // (op 3181) 

s_and_b32 s54, s11, 2147483648 // (op 3181) 
v_and_b32 v20, 3, v1 // (op 3181) 
s_cmp_eq_i32 s54, 0 // (op 3181) 
s_mov_b32 s55, scc // (op 3181) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 3181) 


s_cmp_eq_i32 s55, 0 // (op 3181) 
s_cselect_b32 s54, s52, s53 // (op 3181) 



v_lshrrev_b32 v20, 6, v1 // (op 3181) 
s_mul_hi_u32 s52, s54, s12 // (op 3181) 
v_and_b32 v21, 1, v20 // (op 3181) 

s_sub_u32 s53, s54, s52 // (op 3181) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 3181) 


s_lshr_b32 s55, s53, 1 // (op 3181) 

s_add_u32 s53, s55, s52 // (op 3181) 


s_lshr_b32 s52, s53, s10 // (op 3181) 

s_and_b32 s53, s10, 2147483648 // (op 3181) 
s_cmp_eq_i32 s53, 0 // (op 3181) 
s_mov_b32 s55, scc // (op 3181) 

s_cmp_eq_i32 s55, 0 // (op 3181) 
s_cselect_b32 s53, s54, s52 // (op 3181) 


s_mul_i32 s52, s53, s14 // (op 3181) 
v_lshl_add_u32 v21, s53, 4, v22 // (op 3181) 


s_sub_u32 s53, s54, s52 // (op 3181) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 3181) 


v_mul_lo_u32 v20, v22, s4 // (op 3181) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 3181) 


s_mul_i32 s53, s4, s23 // (op 3183) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3183) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3183) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3183) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3183) most significant: sum
s_add_u32 s53, s53, s55 // (op 3183) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3183) 

s_mov_b32 s56, s20 // (op 3183) reg expression
s_mov_b32 s57, s21 // (op 3183) reg expression
s_mov_b32 s58, s54 // (op 3183) reg expression
s_mov_b32 s59, 131072 // (op 3183) call()

v_and_b32 v20, 48, v1 // (op 3272) 
v_and_b32 v21, 63, v1 // (op 3272) 
v_sub_u32 v23, v21, v20 // (op 3272) 

v_lshrrev_b32 v21, 1, v23 // (op 3272) 
v_add_lshl_u32 v24, v20, v21, 4 // (op 3272) 


v_and_b32 v20, 1, v23 // (op 3272) 

v_lshlrev_b32 v21, 3, v20 // (op 3272) 

v_lshrrev_b32 v20, 7, v1 // (op 3272) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 3272) 


v_add_u32 v20, v24, v23 // (op 3272) 


v_and_b32 v21, 48, v1 // (op 3347) 
v_and_b32 v23, 63, v1 // (op 3347) 
v_sub_u32 v24, v23, v21 // (op 3347) 

v_lshrrev_b32 v23, 1, v24 // (op 3347) 
v_add_lshl_u32 v25, v21, v23, 6 // (op 3347) 


v_and_b32 v21, 1, v24 // (op 3347) 

v_lshlrev_b32 v23, 3, v21 // (op 3347) 

v_lshrrev_b32 v21, 6, v1 // (op 3347) 
v_and_b32 v24, 1, v21 // (op 3347) 

v_lshl_add_u32 v21, v24, 4, v23 // (op 3347) 


v_add_u32 v23, v25, v21 // (op 3347) 


v_lshrrev_b32 v21, 7, v1 // (op 3413) 
v_and_b32 v24, 15, v1 // (op 3413) 
v_lshl_add_u32 v25, v21, 4, v24 // (op 3413) 


v_and_b32 v21, 48, v1 // (op 3413) 
v_add_u32 v24, v21, v25 // (op 3413) 


v_lshrrev_b32 v21, 6, v1 // (op 3489) 
v_and_b32 v25, 1, v21 // (op 3489) 

v_and_b32 v21, 15, v1 // (op 3489) 
v_lshl_add_u32 v26, v25, 4, v21 // (op 3489) 


v_and_b32 v21, 63, v1 // (op 3489) 
v_lshrrev_b32 v25, 4, v21 // (op 3489) 

v_lshl_add_u32 v21, v25, 6, v26 // (op 3489) 


v_lshrrev_b32 v25, 7, v1 // (op 3507) 
s_mul_i32 s52, s16, s8 // (op 3507) 
s_add_u32 s52, s52, s0 // (op 3507) 
v_and_b32 v26, 63, v1 // (op 3507) 
s_mul_hi_u32 s53, s52, s13 // (op 3507) 
v_lshrrev_b32 v27, 4, v26 // (op 3507) 
s_sub_u32 s54, s52, s53 // (op 3507) 
v_lshlrev_b32 v28, 2, v27 // (op 3507) 

s_lshr_b32 s55, s54, 1 // (op 3507) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 3507) 


s_add_u32 s54, s55, s53 // (op 3507) 


v_lshrrev_b32 v25, 2, v26 // (op 3507) 

s_lshr_b32 s53, s54, s11 // (op 3507) 

v_and_b32 v26, 3, v25 // (op 3507) 

s_and_b32 s54, s11, 2147483648 // (op 3507) 
v_and_b32 v25, 3, v1 // (op 3507) 
s_cmp_eq_i32 s54, 0 // (op 3507) 
s_mov_b32 s55, scc // (op 3507) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 3507) 


s_cmp_eq_i32 s55, 0 // (op 3507) 
s_cselect_b32 s54, s52, s53 // (op 3507) 



v_lshrrev_b32 v25, 6, v1 // (op 3507) 
s_mul_hi_u32 s55, s54, s12 // (op 3507) 
v_and_b32 v26, 1, v25 // (op 3507) 

s_sub_u32 s52, s54, s55 // (op 3507) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 3507) 


s_lshr_b32 s53, s52, 1 // (op 3507) 

s_add_u32 s52, s53, s55 // (op 3507) 


s_lshr_b32 s53, s52, s10 // (op 3507) 

s_and_b32 s52, s10, 2147483648 // (op 3507) 
s_cmp_eq_i32 s52, 0 // (op 3507) 
s_mov_b32 s55, scc // (op 3507) 

s_cmp_eq_i32 s55, 0 // (op 3507) 
s_cselect_b32 s52, s54, s53 // (op 3507) 


s_mul_i32 s53, s52, s14 // (op 3507) 
v_lshl_add_u32 v26, s52, 4, v27 // (op 3507) 


s_sub_u32 s52, s54, s53 // (op 3507) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 3507) 


v_mul_lo_u32 v25, v27, s24 // (op 3507) 

v_add_lshl_u32 v27, v26, v25, 2 // (op 3507) 


s_mul_i32 s53, s24, s23 // (op 3509) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3509) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3509) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3509) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3509) most significant: sum
s_add_u32 s53, s53, s55 // (op 3509) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3509) 

s_mov_b32 s60, s26 // (op 3509) reg expression
s_mov_b32 s61, s27 // (op 3509) reg expression
s_mov_b32 s62, s54 // (op 3509) reg expression
s_mov_b32 s63, 131072 // (op 3509) call()

v_lshrrev_b32 v25, 5, v1 // (op 3831) 
v_lshlrev_b32 v26, 2, v25 // (op 3831) 

v_lshl_add_u32 v25, s16, 4, v26 // (op 3831) 

v_and_b32 v26, -32, v1 // (op 3831) 
v_sub_u32 v28, v1, v26 // (op 3831) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 3831) 


v_lshlrev_b32 v25, 2, v26 // (op 3831) 

s_lshl_b32 s52, s15, 12 // (op 3833) 
s_mov_b32 s64, s28 // (op 3833) reg expression
s_mov_b32 s65, s29 // (op 3833) reg expression
s_mov_b32 s66, s52 // (op 3833) reg expression
s_mov_b32 s67, 131072 // (op 3833) call()

s_mul_i32 s6, s8, s16 // (op 331) 
s_add_u32 s6, s6, s0 // (op 331) 
s_mul_hi_u32 s52, s6, s13 // (op 332) 
s_sub_u32 s53, s6, s52 // (op 332) 
s_lshr_b32 s54, s53, 1 // (op 332) 

s_add_u32 s53, s54, s52 // (op 332) 


s_lshr_b32 s52, s53, s11 // (op 332) 

s_and_b32 s53, s11, 2147483648 // (op 332) 
s_cmp_eq_i32 s53, 0 // (op 332) 
s_mov_b32 s54, scc // (op 332) 

s_cmp_eq_i32 s54, 0 // (op 332) 
s_cselect_b32 s53, s6, s52 // (op 332) 


s_ashr_i64 s[54:55], s[2:3], 8 // (op 332) 
s_mul_i32 s52, s53, s54 // (op 332) 


s_sub_u32 s18, s6, s52 // (op 332) 

s_ashr_i64 s[52:53], s[2:3], 8 // (op 333) 
s_cmp_gt_u32 s52, s18 // (op 333) 
s_mov_b32 s54, scc // (op 333) 
s_sub_u32 s55, s52, s18 // (op 333) 

s_cmp_eq_i32 s54, 0 // (op 333) 
s_cselect_b32 s52, 0, s55 // (op 333) 


s_cmp_gt_u32 s8, s0 // (op 333) 
s_mov_b32 s53, scc // (op 333) 
s_sub_u32 s54, s8, s0 // (op 333) 
s_cmp_eq_i32 s53, 0 // (op 333) 
s_cselect_b32 s55, 0, s54 // (op 333) 


s_cmp_lt_u32 s52, s55 // (op 333) 
s_mov_b32 s53, scc // (op 333) 
s_cmp_eq_i32 s53, 0 // (op 333) 
s_cselect_b32 s54, s55, s52 // (op 333) 



s_cmp_gt_u32 s9, s6 // (op 333) 
s_mov_b32 s55, scc // (op 333) 
s_sub_u32 s52, s9, s6 // (op 333) 
s_cmp_eq_i32 s55, 0 // (op 333) 
s_cselect_b32 s53, 0, s52 // (op 333) 


s_cmp_lt_u32 s54, s53 // (op 333) 
s_mov_b32 s52, scc // (op 333) 
s_cmp_eq_i32 s52, 0 // (op 333) 
s_cselect_b32 s7, s53, s54 // (op 333) 



s_add_u32 s52, s18, s7 // (op 334) 
s_sub_u32 s19, s52, 1 // (op 334) 

v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 1261) call()
v_accvgpr_write a5, 0.00000 // (op 1261) call()
v_accvgpr_write a6, 0.00000 // (op 1261) call()
v_accvgpr_write a7, 0.00000 // (op 1261) call()
s_mul_i32 s52, s16, s8 // (op 2753) 
s_add_u32 s52, s52, s0 // (op 2753) 
s_mul_hi_u32 s53, s52, s13 // (op 2753) 
s_sub_u32 s54, s52, s53 // (op 2753) 
s_lshr_b32 s55, s54, 1 // (op 2753) 

s_add_u32 s54, s55, s53 // (op 2753) 


s_lshr_b32 s53, s54, s11 // (op 2753) 

s_and_b32 s54, s11, 2147483648 // (op 2753) 
s_cmp_eq_i32 s54, 0 // (op 2753) 
s_mov_b32 s55, scc // (op 2753) 

s_cmp_eq_i32 s55, 0 // (op 2753) 
s_cselect_b32 s54, s52, s53 // (op 2753) 


s_mul_hi_u32 s53, s54, s12 // (op 2753) 
s_sub_u32 s55, s54, s53 // (op 2753) 
s_lshr_b32 s68, s55, 1 // (op 2753) 

s_add_u32 s55, s68, s53 // (op 2753) 


s_lshr_b32 s53, s55, s10 // (op 2753) 

s_and_b32 s55, s10, 2147483648 // (op 2753) 
s_cmp_eq_i32 s55, 0 // (op 2753) 
s_mov_b32 s68, scc // (op 2753) 

s_cmp_eq_i32 s68, 0 // (op 2753) 
s_cselect_b32 s55, s54, s53 // (op 2753) 


s_lshl_b32 s53, s55, 4 // (op 2753) 

s_mov_b32 s69, 0 // (op 2753) convert
s_mov_b32 s68, s53 // (op 2753) convert

s_ashr_i64 s[70:71], s[2:3], 8 // (op 2753) 
s_mul_i32 s53, s54, s70 // (op 2753) 


s_sub_u32 s54, s52, s53 // (op 2753) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2753) 

v_mul_lo_u32 v29, v26, s31 // (op 2753) most significant: low of low * high
v_mul_hi_u32 v30, v26, s30 // (op 2753) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 2753) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 2753) most significant: sum


v_mov_b32 v26, s68 // (op 2753) 
v_mov_b32 v30, s69 // (op 2753) 
v_add_co_u32 v32, s[52:53], v26, v28 // (op 2753) least significant half
v_addc_co_u32 v33, s[52:53], v30, v29, s[52:53] // (op 2753) most significant half





s_mul_i32 s53, s30, s3 // (op 2755) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2755) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2755) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2755) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2755) most significant: sum
s_add_u32 s53, s53, s55 // (op 2755) most significant: sum


s_mov_b32 s68, s32 // (op 2755) reg expression
s_mov_b32 s69, s33 // (op 2755) reg expression
s_mov_b32 s70, s52 // (op 2755) reg expression
s_mov_b32 s71, 131072 // (op 2755) call()

s_mul_i32 s52, s16, s8 // (op 2772) 
s_add_u32 s52, s52, s0 // (op 2772) 
s_mul_hi_u32 s53, s52, s13 // (op 2772) 
s_sub_u32 s54, s52, s53 // (op 2772) 
s_lshr_b32 s55, s54, 1 // (op 2772) 

s_add_u32 s54, s55, s53 // (op 2772) 


s_lshr_b32 s53, s54, s11 // (op 2772) 

s_and_b32 s54, s11, 2147483648 // (op 2772) 
s_cmp_eq_i32 s54, 0 // (op 2772) 
s_mov_b32 s55, scc // (op 2772) 

s_cmp_eq_i32 s55, 0 // (op 2772) 
s_cselect_b32 s54, s52, s53 // (op 2772) 


s_mul_hi_u32 s53, s54, s12 // (op 2772) 
s_sub_u32 s55, s54, s53 // (op 2772) 
s_lshr_b32 s72, s55, 1 // (op 2772) 

s_add_u32 s55, s72, s53 // (op 2772) 


s_lshr_b32 s53, s55, s10 // (op 2772) 

s_and_b32 s55, s10, 2147483648 // (op 2772) 
s_cmp_eq_i32 s55, 0 // (op 2772) 
s_mov_b32 s72, scc // (op 2772) 

s_cmp_eq_i32 s72, 0 // (op 2772) 
s_cselect_b32 s55, s54, s53 // (op 2772) 


s_lshl_b32 s53, s55, 4 // (op 2772) 

s_mov_b32 s73, 0 // (op 2772) convert
s_mov_b32 s72, s53 // (op 2772) convert

s_ashr_i64 s[74:75], s[2:3], 8 // (op 2772) 
s_mul_i32 s53, s54, s74 // (op 2772) 


s_sub_u32 s54, s52, s53 // (op 2772) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2772) 

v_mul_lo_u32 v29, v26, s31 // (op 2772) most significant: low of low * high
v_mul_hi_u32 v30, v26, s30 // (op 2772) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 2772) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 2772) most significant: sum


v_mov_b32 v26, s72 // (op 2772) 
v_mov_b32 v30, s73 // (op 2772) 
v_add_co_u32 v34, s[52:53], v26, v28 // (op 2772) least significant half
v_addc_co_u32 v35, s[52:53], v30, v29, s[52:53] // (op 2772) most significant half





s_mul_i32 s53, s30, s3 // (op 2774) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2774) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2774) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2774) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2774) most significant: sum
s_add_u32 s53, s53, s55 // (op 2774) most significant: sum


s_mov_b32 s68, s32 // (op 2774) reg expression
s_mov_b32 s69, s33 // (op 2774) reg expression
s_mov_b32 s70, s52 // (op 2774) reg expression
s_mov_b32 s71, 131072 // (op 2774) call()

v_and_b32 v26, -4, v1 // (op 2881) 
s_mul_i32 s52, s16, s8 // (op 2881) 
s_add_u32 s52, s52, s0 // (op 2881) 
v_sub_u32 v28, v1, v26 // (op 2881) 

s_mul_hi_u32 s53, s52, s13 // (op 2881) 
v_lshlrev_b32 v26, 4, v28 // (op 2881) 

s_sub_u32 s54, s52, s53 // (op 2881) 
v_lshrrev_b32 v28, 2, v1 // (op 2881) 
s_lshr_b32 s55, s54, 1 // (op 2881) 

s_add_u32 s54, s55, s53 // (op 2881) 


s_lshr_b32 s53, s54, s11 // (op 2881) 

s_and_b32 s54, s11, 2147483648 // (op 2881) 
s_cmp_eq_i32 s54, 0 // (op 2881) 
s_mov_b32 s55, scc // (op 2881) 

s_cmp_eq_i32 s55, 0 // (op 2881) 
s_cselect_b32 s54, s52, s53 // (op 2881) 


s_mul_hi_u32 s53, s54, s12 // (op 2881) 
s_sub_u32 s55, s54, s53 // (op 2881) 
s_lshr_b32 s72, s55, 1 // (op 2881) 

s_add_u32 s55, s72, s53 // (op 2881) 


s_lshr_b32 s53, s55, s10 // (op 2881) 

s_and_b32 s55, s10, 2147483648 // (op 2881) 
s_cmp_eq_i32 s55, 0 // (op 2881) 
s_mov_b32 s72, scc // (op 2881) 

s_cmp_eq_i32 s72, 0 // (op 2881) 
s_cselect_b32 s55, s54, s53 // (op 2881) 


s_mul_i32 s53, s55, s14 // (op 2881) 

s_sub_u32 s55, s54, s53 // (op 2881) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 2881) 


s_ashr_i64 s[72:73], s[2:3], 8 // (op 2881) 
v_mov_b32 v31, 0 // (op 2881) convert
v_mov_b32 v30, v29 // (op 2881) convert

s_mul_i32 s53, s54, s72 // (op 2881) 


s_sub_u32 s54, s52, s53 // (op 2881) 


v_lshl_add_u32 v26, s54, 8, v28 // (op 2881) 


v_mul_lo_u32 v29, v26, s35 // (op 2881) most significant: low of low * high
v_mul_hi_u32 v36, v26, s34 // (op 2881) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 2881) least significant: low of low * low
v_add_u32 v29, v29, v36 // (op 2881) most significant: sum

v_add_co_u32 v28, s[52:53], v28, v30 // (op 2881) least significant half
v_addc_co_u32 v29, s[52:53], v29, v31, s[52:53] // (op 2881) most significant half



s_mul_i32 s53, s34, s3 // (op 2883) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2883) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2883) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2883) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2883) most significant: sum
s_add_u32 s53, s53, s55 // (op 2883) most significant: sum


s_mov_b32 s72, s36 // (op 2883) reg expression
s_mov_b32 s73, s37 // (op 2883) reg expression
s_mov_b32 s74, s52 // (op 2883) reg expression
s_mov_b32 s75, 131072 // (op 2883) call()

v_and_b32 v26, -4, v1 // (op 2900) 
s_mul_i32 s52, s16, s8 // (op 2900) 
s_add_u32 s52, s52, s0 // (op 2900) 
v_sub_u32 v30, v1, v26 // (op 2900) 

s_mul_hi_u32 s53, s52, s13 // (op 2900) 
v_lshlrev_b32 v26, 4, v30 // (op 2900) 

s_sub_u32 s54, s52, s53 // (op 2900) 
v_lshrrev_b32 v30, 2, v1 // (op 2900) 
s_lshr_b32 s55, s54, 1 // (op 2900) 

s_add_u32 s54, s55, s53 // (op 2900) 


s_lshr_b32 s53, s54, s11 // (op 2900) 

s_and_b32 s54, s11, 2147483648 // (op 2900) 
s_cmp_eq_i32 s54, 0 // (op 2900) 
s_mov_b32 s55, scc // (op 2900) 

s_cmp_eq_i32 s55, 0 // (op 2900) 
s_cselect_b32 s54, s52, s53 // (op 2900) 


s_mul_hi_u32 s53, s54, s12 // (op 2900) 
s_sub_u32 s55, s54, s53 // (op 2900) 
s_lshr_b32 s76, s55, 1 // (op 2900) 

s_add_u32 s55, s76, s53 // (op 2900) 


s_lshr_b32 s53, s55, s10 // (op 2900) 

s_and_b32 s55, s10, 2147483648 // (op 2900) 
s_cmp_eq_i32 s55, 0 // (op 2900) 
s_mov_b32 s76, scc // (op 2900) 

s_cmp_eq_i32 s76, 0 // (op 2900) 
s_cselect_b32 s55, s54, s53 // (op 2900) 


s_mul_i32 s53, s55, s14 // (op 2900) 

s_sub_u32 s55, s54, s53 // (op 2900) 

v_lshl_add_u32 v31, s55, 6, v26 // (op 2900) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 2900) 
v_mov_b32 v37, 0 // (op 2900) convert
v_mov_b32 v36, v31 // (op 2900) convert

s_mul_i32 s53, s54, s76 // (op 2900) 


s_sub_u32 s54, s52, s53 // (op 2900) 


v_lshl_add_u32 v26, s54, 8, v30 // (op 2900) 


v_mul_lo_u32 v31, v26, s35 // (op 2900) most significant: low of low * high
v_mul_hi_u32 v38, v26, s34 // (op 2900) most significant: high of low * low
v_mul_lo_u32 v30, v26, s34 // (op 2900) least significant: low of low * low
v_add_u32 v31, v31, v38 // (op 2900) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v36 // (op 2900) least significant half
v_addc_co_u32 v31, s[52:53], v31, v37, s[52:53] // (op 2900) most significant half



s_mul_i32 s53, s34, s3 // (op 2902) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2902) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2902) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2902) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2902) most significant: sum
s_add_u32 s53, s53, s55 // (op 2902) most significant: sum


s_mov_b32 s72, s36 // (op 2902) reg expression
s_mov_b32 s73, s37 // (op 2902) reg expression
s_mov_b32 s74, s52 // (op 2902) reg expression
s_mov_b32 s75, 131072 // (op 2902) call()

v_and_b32 v26, -16, v1 // (op 3009) 
s_mul_i32 s52, s16, s8 // (op 3009) 
s_add_u32 s52, s52, s0 // (op 3009) 
v_sub_u32 v36, v1, v26 // (op 3009) 

s_mul_hi_u32 s53, s52, s13 // (op 3009) 
v_lshrrev_b32 v26, 4, v1 // (op 3009) 
s_sub_u32 s54, s52, s53 // (op 3009) 
s_lshr_b32 s55, s54, 1 // (op 3009) 

s_add_u32 s54, s55, s53 // (op 3009) 


s_lshr_b32 s53, s54, s11 // (op 3009) 

s_and_b32 s54, s11, 2147483648 // (op 3009) 
s_cmp_eq_i32 s54, 0 // (op 3009) 
s_mov_b32 s55, scc // (op 3009) 

s_cmp_eq_i32 s55, 0 // (op 3009) 
s_cselect_b32 s54, s52, s53 // (op 3009) 


s_mul_hi_u32 s53, s54, s12 // (op 3009) 
s_sub_u32 s55, s54, s53 // (op 3009) 
s_lshr_b32 s76, s55, 1 // (op 3009) 

s_add_u32 s55, s76, s53 // (op 3009) 


s_lshr_b32 s53, s55, s10 // (op 3009) 

s_and_b32 s55, s10, 2147483648 // (op 3009) 
s_cmp_eq_i32 s55, 0 // (op 3009) 
s_mov_b32 s76, scc // (op 3009) 

s_cmp_eq_i32 s76, 0 // (op 3009) 
s_cselect_b32 s55, s54, s53 // (op 3009) 


v_lshl_add_u32 v37, s55, 4, v36 // (op 3009) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 3009) 
s_mul_i32 s53, s54, s76 // (op 3009) 


s_sub_u32 s54, s52, s53 // (op 3009) 


v_lshl_add_u32 v36, s54, 3, v26 // (op 3009) 


v_mul_lo_u32 v26, v36, s38 // (op 3009) 

v_add_u32 v36, v37, v26 // (op 3009) 


s_mul_i32 s53, s38, s43 // (op 3011) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 3011) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 3011) least significant
s_mul_hi_u32 s55, s38, s42 // (op 3011) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3011) most significant: sum
s_add_u32 s53, s53, s55 // (op 3011) most significant: sum


s_mov_b32 s76, s40 // (op 3011) reg expression
s_mov_b32 s77, s41 // (op 3011) reg expression
s_mov_b32 s78, s52 // (op 3011) reg expression
s_mov_b32 s79, 131072 // (op 3011) call()

v_and_b32 v26, -16, v1 // (op 3028) 
s_mul_i32 s52, s16, s8 // (op 3028) 
s_add_u32 s52, s52, s0 // (op 3028) 
v_sub_u32 v37, v1, v26 // (op 3028) 

s_mul_hi_u32 s53, s52, s13 // (op 3028) 
v_lshrrev_b32 v26, 4, v1 // (op 3028) 
s_sub_u32 s54, s52, s53 // (op 3028) 
s_lshr_b32 s55, s54, 1 // (op 3028) 

s_add_u32 s54, s55, s53 // (op 3028) 


s_lshr_b32 s53, s54, s11 // (op 3028) 

s_and_b32 s54, s11, 2147483648 // (op 3028) 
s_cmp_eq_i32 s54, 0 // (op 3028) 
s_mov_b32 s55, scc // (op 3028) 

s_cmp_eq_i32 s55, 0 // (op 3028) 
s_cselect_b32 s54, s52, s53 // (op 3028) 


s_mul_hi_u32 s53, s54, s12 // (op 3028) 
s_sub_u32 s55, s54, s53 // (op 3028) 
s_lshr_b32 s80, s55, 1 // (op 3028) 

s_add_u32 s55, s80, s53 // (op 3028) 


s_lshr_b32 s53, s55, s10 // (op 3028) 

s_and_b32 s55, s10, 2147483648 // (op 3028) 
s_cmp_eq_i32 s55, 0 // (op 3028) 
s_mov_b32 s80, scc // (op 3028) 

s_cmp_eq_i32 s80, 0 // (op 3028) 
s_cselect_b32 s55, s54, s53 // (op 3028) 


v_lshl_add_u32 v38, s55, 4, v37 // (op 3028) 


s_ashr_i64 s[80:81], s[2:3], 8 // (op 3028) 
s_mul_i32 s53, s54, s80 // (op 3028) 


s_sub_u32 s54, s52, s53 // (op 3028) 


v_lshl_add_u32 v37, s54, 3, v26 // (op 3028) 


v_mul_lo_u32 v26, v37, s38 // (op 3028) 

v_add_u32 v37, v38, v26 // (op 3028) 


s_mul_i32 s53, s38, s43 // (op 3030) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 3030) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 3030) least significant
s_mul_hi_u32 s55, s38, s42 // (op 3030) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3030) most significant: sum
s_add_u32 s53, s53, s55 // (op 3030) most significant: sum


s_mov_b32 s76, s40 // (op 3030) reg expression
s_mov_b32 s77, s41 // (op 3030) reg expression
s_mov_b32 s78, s52 // (op 3030) reg expression
s_mov_b32 s79, 131072 // (op 3030) call()

v_and_b32 v26, -16, v1 // (op 3137) 
s_mul_i32 s52, s16, s8 // (op 3137) 
s_add_u32 s52, s52, s0 // (op 3137) 
v_sub_u32 v38, v1, v26 // (op 3137) 

s_mul_hi_u32 s53, s52, s13 // (op 3137) 
v_lshlrev_b32 v26, 2, v38 // (op 3137) 

s_sub_u32 s54, s52, s53 // (op 3137) 
v_lshrrev_b32 v38, 4, v1 // (op 3137) 
s_lshr_b32 s55, s54, 1 // (op 3137) 

s_add_u32 s54, s55, s53 // (op 3137) 


s_lshr_b32 s53, s54, s11 // (op 3137) 

s_and_b32 s54, s11, 2147483648 // (op 3137) 
s_cmp_eq_i32 s54, 0 // (op 3137) 
s_mov_b32 s55, scc // (op 3137) 

s_cmp_eq_i32 s55, 0 // (op 3137) 
s_cselect_b32 s54, s52, s53 // (op 3137) 


s_mul_hi_u32 s53, s54, s12 // (op 3137) 
s_sub_u32 s55, s54, s53 // (op 3137) 
s_lshr_b32 s80, s55, 1 // (op 3137) 

s_add_u32 s55, s80, s53 // (op 3137) 


s_lshr_b32 s53, s55, s10 // (op 3137) 

s_and_b32 s55, s10, 2147483648 // (op 3137) 
s_cmp_eq_i32 s55, 0 // (op 3137) 
s_mov_b32 s80, scc // (op 3137) 

s_cmp_eq_i32 s80, 0 // (op 3137) 
s_cselect_b32 s55, s54, s53 // (op 3137) 


s_mul_i32 s53, s55, s14 // (op 3137) 

s_sub_u32 s55, s54, s53 // (op 3137) 

s_ashr_i64 s[80:81], s[2:3], 8 // (op 3137) 
v_lshl_add_u32 v39, s55, 6, v26 // (op 3137) 


s_mul_i32 s53, s54, s80 // (op 3137) 


s_sub_u32 s54, s52, s53 // (op 3137) 


v_lshl_add_u32 v26, s54, 3, v38 // (op 3137) 


v_mul_lo_u32 v38, v26, s44 // (op 3137) 
v_add_u32 v38, v38, v39 // (op 3137) 


s_mul_i32 s53, s44, s43 // (op 3139) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3139) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3139) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3139) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3139) most significant: sum
s_add_u32 s53, s53, s55 // (op 3139) most significant: sum


s_mov_b32 s80, s46 // (op 3139) reg expression
s_mov_b32 s81, s47 // (op 3139) reg expression
s_mov_b32 s82, s52 // (op 3139) reg expression
s_mov_b32 s83, 131072 // (op 3139) call()

v_and_b32 v26, -16, v1 // (op 3156) 
s_mul_i32 s52, s16, s8 // (op 3156) 
s_add_u32 s52, s52, s0 // (op 3156) 
v_sub_u32 v39, v1, v26 // (op 3156) 

s_mul_hi_u32 s53, s52, s13 // (op 3156) 
v_lshlrev_b32 v26, 2, v39 // (op 3156) 

s_sub_u32 s54, s52, s53 // (op 3156) 
v_lshrrev_b32 v39, 4, v1 // (op 3156) 
s_lshr_b32 s55, s54, 1 // (op 3156) 

s_add_u32 s54, s55, s53 // (op 3156) 


s_lshr_b32 s53, s54, s11 // (op 3156) 

s_and_b32 s54, s11, 2147483648 // (op 3156) 
s_cmp_eq_i32 s54, 0 // (op 3156) 
s_mov_b32 s55, scc // (op 3156) 

s_cmp_eq_i32 s55, 0 // (op 3156) 
s_cselect_b32 s54, s52, s53 // (op 3156) 


s_mul_hi_u32 s53, s54, s12 // (op 3156) 
s_sub_u32 s55, s54, s53 // (op 3156) 
s_lshr_b32 s84, s55, 1 // (op 3156) 

s_add_u32 s55, s84, s53 // (op 3156) 


s_lshr_b32 s53, s55, s10 // (op 3156) 

s_and_b32 s55, s10, 2147483648 // (op 3156) 
s_cmp_eq_i32 s55, 0 // (op 3156) 
s_mov_b32 s84, scc // (op 3156) 

s_cmp_eq_i32 s84, 0 // (op 3156) 
s_cselect_b32 s55, s54, s53 // (op 3156) 


s_mul_i32 s53, s55, s14 // (op 3156) 

s_sub_u32 s55, s54, s53 // (op 3156) 

v_lshl_add_u32 v40, s55, 6, v26 // (op 3156) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 3156) 
s_mul_i32 s53, s54, s84 // (op 3156) 


s_sub_u32 s54, s52, s53 // (op 3156) 


v_lshl_add_u32 v26, s54, 3, v39 // (op 3156) 


v_mul_lo_u32 v39, v26, s44 // (op 3156) 
v_add_u32 v39, v39, v40 // (op 3156) 


s_mul_i32 s53, s44, s43 // (op 3158) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3158) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3158) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3158) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3158) most significant: sum
s_add_u32 s53, s53, s55 // (op 3158) most significant: sum


s_mov_b32 s80, s46 // (op 3158) reg expression
s_mov_b32 s81, s47 // (op 3158) reg expression
s_mov_b32 s82, s52 // (op 3158) reg expression
s_mov_b32 s83, 131072 // (op 3158) call()

v_and_b32 v26, -4, v1 // (op 3313) 
v_sub_u32 v40, v1, v26 // (op 3313) 

v_lshlrev_b32 v26, 4, v40 // (op 3313) 

v_lshrrev_b32 v40, 2, v1 // (op 3313) 
v_lshl_add_u32 v41, v40, 6, v26 // (op 3313) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 3313) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4295) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 2631) 
v_mov_b64 v[40:41], s[54:55] // (op 2631) convert


v_mov_b64 v[42:43], v[28:29] // (op 2631) reg expression
s_mov_b32 m0, s52 // (op 2631) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2631) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2631) most significant half

s_add_u32 m0, m0, 2048 // (op 2631) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2631) Load value direct to lds





v_mov_b32 v26, v38 // (op 2390) reg expression

buffer_load_dword v40, v26, s[80:83], 0 offen // (op 2390) Load value

v_lshlrev_b32 v26, 4, v1 // (op 3248) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 3248) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4276) 

s_lshl_b64 s[54:55], s[30:31], 7 // (op 2636) 
v_mov_b64 v[42:43], s[54:55] // (op 2636) convert


v_mov_b64 v[44:45], v[32:33] // (op 2636) reg expression
s_add_u32 s53, 16384, s52 // (op 2636) 
s_mov_b32 m0, s53 // (op 2636) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2636) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 2636) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 2636) most significant half

s_add_u32 m0, m0, 2048 // (op 2636) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2636) Load value direct to lds






v_mov_b32 v26, v36 // (op 2396) reg expression

buffer_load_ubyte v41, v26, s[76:79], 0 offen offset: 0 // (op 2396) Load value

v_and_b32 v26, -4, v1 // (op 4058) 
v_sub_u32 v42, v1, v26 // (op 4058) 

v_lshlrev_b32 v26, 4, v42 // (op 4058) 

v_lshrrev_b32 v42, 2, v1 // (op 4058) 
v_lshl_add_u32 v43, v42, 6, v26 // (op 4058) 


s_nop 0
v_readfirstlane_b32 s52, v43 // (op 4058) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4478) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 2641) 
v_mov_b64 v[42:43], s[54:55] // (op 2641) convert


s_lshl_b64 s[54:55], s[34:35], 8 // (op 2641) 
v_mov_b32 v26, s54 // (op 2641) 
v_mov_b32 v44, s55 // (op 2641) 
v_add_co_u32 v46, s[84:85], v28, v26 // (op 2641) least significant half
v_addc_co_u32 v47, s[84:85], v29, v44, s[84:85] // (op 2641) most significant half




s_add_u32 s53, 20480, s52 // (op 2641) 
s_mov_b32 m0, s53 // (op 2641) reg expression
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2641) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2641) most significant half

s_add_u32 m0, m0, 2048 // (op 2641) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2641) Load value direct to lds







s_lshl_b32 s52, s44, 3 // (op 2402) 
v_add_u32 v26, v38, s52 // (op 2402) 


buffer_load_dword v28, v26, s[80:83], 0 offen // (op 2402) Load value


v_lshlrev_b32 v26, 4, v1 // (op 4019) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 4019) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4468) 

s_lshl_b64 s[54:55], s[30:31], 7 // (op 2643) 
v_mov_b64 v[42:43], s[54:55] // (op 2643) convert


s_lshl_b64 s[54:55], s[30:31], 8 // (op 2643) 
v_mov_b32 v26, s54 // (op 2643) 
v_mov_b32 v29, s55 // (op 2643) 
v_add_co_u32 v44, s[84:85], v32, v26 // (op 2643) least significant half
v_addc_co_u32 v45, s[84:85], v33, v29, s[84:85] // (op 2643) most significant half




s_add_u32 s53, 36864, s52 // (op 2643) 
s_mov_b32 m0, s53 // (op 2643) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 2643) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2643) Load value direct to lds







s_lshl_b32 s52, s38, 3 // (op 2408) 
v_add_u32 v26, v36, s52 // (op 2408) 


buffer_load_ubyte v29, v26, s[76:79], 0 offen offset: 0 // (op 2408) Load value


v_and_b32 v26, -16, v1 // (op 3455) 
v_sub_u32 v36, v1, v26 // (op 3455) 

v_lshlrev_b32 v26, 2, v36 // (op 3455) 

v_lshrrev_b32 v36, 4, v1 // (op 3455) 
v_lshl_add_u32 v38, v36, 6, v26 // (op 3455) 


s_barrier  // (op 4336) 



s_waitcnt vmcnt(15)
ds_write_b32 v38, v40 offset:40960 // (op 2411) Store local data 

v_and_b32 v26, -16, v1 // (op 3389) 
v_sub_u32 v36, v1, v26 // (op 3389) 

v_and_b32 v26, 4294967280, v1 // (op 3389) 
v_add_u32 v38, v26, v36 // (op 3389) 


s_barrier  // (op 4315) 



s_waitcnt vmcnt(12)
ds_write_b8 v38, v41 offset:41472 // (op 2413) Store local data 

s_barrier  // (op 4292) 



ds_read_b64_tr_b8 v[42:43], v23 // (op 2434) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v23 offset:512 // (op 2434) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v23 offset:4096 // (op 2434) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v23 offset:4608 // (op 2434) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v23 offset:8192 // (op 2444) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v23 offset:8704 // (op 2444) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v23 offset:12288 // (op 2444) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v23 offset:12800 // (op 2444) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4333) 


ds_read_u8 v26, v21 offset:40960 // (op 2460) Load local data 


ds_read_u8 v36, v21 offset:41216 // (op 2470) Load local data 



ds_read_b64_tr_b8 v[58:59], v23 offset:32 // (op 2439) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v23 offset:544 // (op 2439) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v23 offset:4128 // (op 2439) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v23 offset:4640 // (op 2439) Transpose load local data 



ds_read_b64_tr_b8 v[66:67], v23 offset:8224 // (op 2449) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v23 offset:8736 // (op 2449) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v23 offset:12320 // (op 2449) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v23 offset:12832 // (op 2449) Transpose load local data 


ds_read_u8 v38, v21 offset:40992 // (op 2465) Load local data 


ds_read_u8 v32, v21 offset:41248 // (op 2475) Load local data 
s_barrier  // (op 4273) 



ds_read_b64_tr_b8 v[74:75], v20 offset:16384 // (op 2427) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v20 offset:16512 // (op 2427) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v20 offset:17408 // (op 2427) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v20 offset:17536 // (op 2427) Transpose load local data 



ds_read_b64_tr_b8 v[82:83], v20 offset:18432 // (op 2430) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v20 offset:18560 // (op 2430) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v20 offset:19456 // (op 2430) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v20 offset:19584 // (op 2430) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4312) 


ds_read_u8 v33, v24 offset:41472 // (op 2453) Load local data 


ds_read_u8 v90, v24 offset:41536 // (op 2456) Load local data 
v_lshlrev_b32 v91, 4, v1 // (op 3220) 
s_nop 0
v_readfirstlane_b32 s52, v91 // (op 3220) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -4, v1 // (op 3285) 
v_sub_u32 v92, v1, v91 // (op 3285) 

v_lshlrev_b32 v91, 4, v92 // (op 3285) 

v_lshrrev_b32 v92, 2, v1 // (op 3285) 
v_lshl_add_u32 v93, v92, 6, v91 // (op 3285) 


s_nop 0
v_readfirstlane_b32 s53, v93 // (op 3285) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -16, v1 // (op 3360) 
v_sub_u32 v92, v1, v91 // (op 3360) 

v_and_b32 v91, 4294967280, v1 // (op 3360) 
v_add_u32 v93, v91, v92 // (op 3360) 


v_and_b32 v91, -16, v1 // (op 3426) 
v_sub_u32 v92, v1, v91 // (op 3426) 

v_lshlrev_b32 v91, 2, v92 // (op 3426) 

v_lshrrev_b32 v92, 4, v1 // (op 3426) 
v_lshl_add_u32 v94, v92, 6, v91 // (op 3426) 


v_lshlrev_b32 v91, 4, v1 // (op 4004) 
s_nop 0
v_readfirstlane_b32 s54, v91 // (op 4004) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -4, v1 // (op 4043) 
v_sub_u32 v92, v1, v91 // (op 4043) 

v_lshlrev_b32 v91, 4, v92 // (op 4043) 

v_lshrrev_b32 v92, 2, v1 // (op 4043) 
v_lshl_add_u32 v95, v92, 6, v91 // (op 4043) 


s_nop 0
v_readfirstlane_b32 s55, v95 // (op 4043) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -16, v1 // (op 4087) 
v_sub_u32 v92, v1, v91 // (op 4087) 

v_and_b32 v91, 4294967280, v1 // (op 4087) 
v_add_u32 v95, v91, v92 // (op 4087) 


v_and_b32 v91, -16, v1 // (op 4111) 
v_sub_u32 v92, v1, v91 // (op 4111) 

v_lshlrev_b32 v91, 2, v92 // (op 4111) 

v_lshrrev_b32 v92, 4, v1 // (op 4111) 
v_lshl_add_u32 v96, v92, 6, v91 // (op 4111) 


s_mov_b32 s1, 0 // (op 1263) call()
s_and_b32 s84, s7, 4294967294 // (op 1262) 
s_cmp_lt_u32 s1, s84 // (op 1262) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1262 // (op 1262) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1262 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1262:
s_add_u32 s1, s1, 2 // (op 1264) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v33, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 4298) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v33, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1310) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 2556) 
v_mov_b64 v[98:99], s[84:85] // (op 2556) convert


v_mov_b64 v[100:101], v[30:31] // (op 2556) 
s_lshl_b64 s[84:85], s[34:35], 8 // (op 2556) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2556) 

v_mov_b32 v97, s86 // (op 2556) 
v_mov_b32 v91, s87 // (op 2556) 
v_add_co_u32 v102, s[84:85], v100, v97 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], v101, v91, s[84:85] // (op 2556) most significant half




s_mov_b32 m0, s53 // (op 2556) reg expression
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[50:57], a[0:3], v90, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 105) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[66:73], a[4:7], v90, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1319) 
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2556) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2556) most significant half

s_add_u32 m0, m0, 2048 // (op 2556) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2556) Load value direct to lds




v_mov_b32 v91, v39 // (op 25) 
s_lshl_b32 s84, s44, 4 // (op 25) 
v_add_u32 v92, v91, s84 // (op 25) 


buffer_load_dword v40, v92, s[80:83], 0 offen // (op 25) Load value

s_barrier  // (op 4279) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 2551) 
v_mov_b64 v[98:99], s[84:85] // (op 2551) convert


v_mov_b64 v[102:103], v[34:35] // (op 2551) 
s_lshl_b64 s[84:85], s[30:31], 8 // (op 2551) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2551) 

v_mov_b32 v92, s86 // (op 2551) 
v_mov_b32 v97, s87 // (op 2551) 
v_add_co_u32 v104, s[84:85], v102, v92 // (op 2551) least significant half
v_addc_co_u32 v105, s[84:85], v103, v97, s[84:85] // (op 2551) most significant half




s_add_u32 s84, 16384, s52 // (op 2551) 
s_nop 0
s_mov_b32 m0, s84 // (op 2551) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2551) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v98 // (op 2551) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 2551) most significant half

s_add_u32 m0, m0, 2048 // (op 2551) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2551) Load value direct to lds





v_mov_b32 v92, v37 // (op 17) 
s_lshl_b32 s84, s38, 4 // (op 17) 
v_add_u32 v97, v92, s84 // (op 17) 


buffer_load_ubyte v41, v97, s[76:79], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 4494) 



s_waitcnt vmcnt(15)
ds_write_b32 v96, v28 offset:41600 // (op 1167) Store local data 
s_barrier  // (op 4488) 



s_waitcnt vmcnt(12)
ds_write_b8 v95, v29 offset:42112 // (op 1166) Store local data 
s_barrier  // (op 4474) 



ds_read_b64_tr_b8 v[42:43], v17 offset:20480 // (op 1150) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:20992 // (op 1150) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v17 offset:24576 // (op 1150) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v17 offset:25088 // (op 1150) Transpose load local data 



ds_read_b64_tr_b8 v[104:105], v17 offset:28672 // (op 1159) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v17 offset:29184 // (op 1159) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v17 offset:32768 // (op 1159) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v17 offset:33280 // (op 1159) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4491) 


ds_read_u8 v26, v16 offset:41600 // (op 1154) Load local data 


ds_read_u8 v97, v16 offset:41856 // (op 1163) Load local data 



ds_read_b64_tr_b8 v[58:59], v17 offset:20512 // (op 1333) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v17 offset:21024 // (op 1333) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v17 offset:24608 // (op 1333) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v17 offset:25120 // (op 1333) Transpose load local data 



ds_read_b64_tr_b8 v[112:113], v17 offset:28704 // (op 1324) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v17 offset:29216 // (op 1324) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v17 offset:32800 // (op 1324) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v17 offset:33312 // (op 1324) Transpose load local data 


ds_read_u8 v38, v16 offset:41632 // (op 1337) Load local data 


ds_read_u8 v98, v16 offset:41888 // (op 1328) Load local data 
s_barrier  // (op 4465) 
s_barrier  // (op 4671) 



ds_read_b64_tr_b8 v[74:75], v7 offset:36864 // (op 1148) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v7 offset:36992 // (op 1148) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v7 offset:37888 // (op 1148) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v7 offset:38016 // (op 1148) Transpose load local data 



ds_read_b64_tr_b8 v[120:121], v7 offset:38912 // (op 1157) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v7 offset:39040 // (op 1157) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v7 offset:39936 // (op 1157) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v7 offset:40064 // (op 1157) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4484) 


ds_read_u8 v33, v10 offset:42112 // (op 1152) Load local data 


ds_read_u8 v90, v10 offset:42176 // (op 1161) Load local data 
s_barrier  // (op 4635) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v33, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1155) 
s_barrier  // (op 4481) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v33, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1320) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 2612) 
v_mov_b64 v[128:129], s[84:85] // (op 2612) convert


s_lshl_b64 s[84:85], s[34:35], 8 // (op 2612) 
s_mul_i32 s87, 3, s85 // (op 2612) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 2612) least significant
s_mul_hi_u32 s88, 3, s84 // (op 2612) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 2612) most significant: sum


v_mov_b32 v99, s86 // (op 2612) 
v_mov_b32 v130, s87 // (op 2612) 
v_add_co_u32 v132, s[84:85], v100, v99 // (op 2612) least significant half
v_addc_co_u32 v133, s[84:85], v101, v130, s[84:85] // (op 2612) most significant half




s_add_u32 s84, 20480, s55 // (op 2612) 
s_nop 0
s_mov_b32 m0, s84 // (op 2612) reg expression
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:127], v[104:111], a[0:3], v90, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1164) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds


v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:127], v[112:119], a[4:7], v90, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1329) 
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half




s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2612) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2612) most significant half

s_add_u32 m0, m0, 2048 // (op 2612) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2612) Load value direct to lds






s_lshl_b32 s84, s44, 3 // (op 1170) 
s_mul_i32 s85, 3, s84 // (op 1170) 

v_add_u32 v97, v91, s85 // (op 1170) 


buffer_load_dword v28, v97, s[80:83], 0 offen // (op 1170) Load value

s_lshl_b64 s[84:85], s[34:35], 8 // (op 2892) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 2892) 

s_barrier  // (op 4471) 
s_lshl_b32 s84, s44, 4 // (op 3148) 

v_add_u32 v39, v39, s84 // (op 3148) 

v_mov_b32 v91, s86 // (op 2892) 
v_mov_b32 v97, s87 // (op 2892) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 2607) 
v_mov_b64 v[98:99], s[84:85] // (op 2607) convert


s_lshl_b64 s[84:85], s[30:31], 8 // (op 2607) 
s_mul_i32 s89, 3, s85 // (op 2607) most significant: low of low * high
s_mul_i32 s88, 3, s84 // (op 2607) least significant
s_mul_hi_u32 s90, 3, s84 // (op 2607) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 2607) most significant: sum


v_mov_b32 v100, s88 // (op 2607) 
v_mov_b32 v101, s89 // (op 2607) 
v_add_co_u32 v30, s[84:85], v30, v91 // (op 2892) least significant half
v_addc_co_u32 v31, s[84:85], v31, v97, s[84:85] // (op 2892) most significant half




v_add_co_u32 v104, s[84:85], v102, v100 // (op 2607) least significant half
v_addc_co_u32 v105, s[84:85], v103, v101, s[84:85] // (op 2607) most significant half




s_add_u32 s84, 36864, s54 // (op 2607) 
s_nop 0
s_mov_b32 m0, s84 // (op 2607) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2607) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v98 // (op 2607) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 2607) most significant half

s_add_u32 m0, m0, 2048 // (op 2607) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2607) Load value direct to lds






s_lshl_b32 s84, s38, 3 // (op 1169) 
s_mul_i32 s85, 3, s84 // (op 1169) 

v_add_u32 v91, v92, s85 // (op 1169) 


buffer_load_ubyte v29, v91, s[76:79], 0 offen offset: 0 // (op 1169) Load value

s_lshl_b64 s[84:85], s[30:31], 8 // (op 2764) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 2764) 

s_barrier  // (op 4339) 
s_lshl_b32 s84, s38, 4 // (op 3020) 



v_add_u32 v37, v37, s84 // (op 3020) 

v_mov_b32 v91, s86 // (op 2764) 
v_mov_b32 v92, s87 // (op 2764) 
v_add_co_u32 v34, s[84:85], v34, v91 // (op 2764) least significant half
v_addc_co_u32 v35, s[84:85], v35, v92, s[84:85] // (op 2764) most significant half




s_waitcnt vmcnt(15)
ds_write_b32 v94, v40 offset:40960 // (op 30) Store local data 
s_barrier  // (op 4318) 



s_waitcnt vmcnt(12)
ds_write_b8 v93, v41 offset:41472 // (op 21) Store local data 
s_barrier  // (op 4304) 



ds_read_b64_tr_b8 v[42:43], v11 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v11 offset:512 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:4096 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:4608 // (op 75) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v11 offset:8192 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:8704 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v11 offset:12288 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:12800 // (op 95) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4346) 


ds_read_u8 v26, v9 offset:40960 // (op 85) Load local data 


ds_read_u8 v36, v9 offset:41216 // (op 103) Load local data 



ds_read_b64_tr_b8 v[58:59], v11 offset:32 // (op 1305) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v11 offset:544 // (op 1305) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v11 offset:4128 // (op 1305) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v11 offset:4640 // (op 1305) Transpose load local data 



ds_read_b64_tr_b8 v[66:67], v11 offset:8224 // (op 1314) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v11 offset:8736 // (op 1314) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v11 offset:12320 // (op 1314) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v11 offset:12832 // (op 1314) Transpose load local data 


ds_read_u8 v38, v9 offset:40992 // (op 1309) Load local data 


ds_read_u8 v32, v9 offset:41248 // (op 1318) Load local data 
s_barrier  // (op 4285) 



ds_read_b64_tr_b8 v[74:75], v0 offset:16384 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:16512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:17408 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:17536 // (op 70) Transpose load local data 



ds_read_b64_tr_b8 v[82:83], v0 offset:18432 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:18560 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:19456 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:19584 // (op 91) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4325) 


ds_read_u8 v33, v4 offset:41472 // (op 80) Load local data 


ds_read_u8 v90, v4 offset:41536 // (op 99) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1262) 
s_cmp_lt_u32 s1, s84 // (op 1262) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1262 // (op 1262) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1262 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1262:



















s_barrier  // (op 4653) 
s_and_b32 s52, s7, 4294967294 // (op 2680) 
s_add_u32 s53, s0, s52 // (op 2680) 

s_mul_i32 s52, s16, s8 // (op 2680) 
s_add_u32 s52, s52, s53 // (op 2680) 

s_mul_hi_u32 s53, s52, s13 // (op 2680) 
s_sub_u32 s54, s52, s53 // (op 2680) 
s_lshr_b32 s55, s54, 1 // (op 2680) 

s_add_u32 s54, s55, s53 // (op 2680) 


s_lshr_b32 s53, s54, s11 // (op 2680) 

s_and_b32 s54, s11, 2147483648 // (op 2680) 
s_cmp_eq_i32 s54, 0 // (op 2680) 
s_mov_b32 s55, scc // (op 2680) 

s_cmp_eq_i32 s55, 0 // (op 2680) 
s_cselect_b32 s54, s52, s53 // (op 2680) 


s_mul_hi_u32 s53, s54, s12 // (op 2680) 
s_sub_u32 s55, s54, s53 // (op 2680) 
s_lshr_b32 s84, s55, 1 // (op 2680) 

s_add_u32 s55, s84, s53 // (op 2680) 


s_lshr_b32 s53, s55, s10 // (op 2680) 

s_and_b32 s55, s10, 2147483648 // (op 2680) 
s_cmp_eq_i32 s55, 0 // (op 2680) 
s_mov_b32 s84, scc // (op 2680) 

s_cmp_eq_i32 s84, 0 // (op 2680) 
s_cselect_b32 s55, s54, s53 // (op 2680) 


s_lshl_b32 s53, s55, 4 // (op 2680) 

s_mov_b32 s85, 0 // (op 2680) convert
s_mov_b32 s84, s53 // (op 2680) convert

s_ashr_i64 s[86:87], s[2:3], 8 // (op 2680) 
s_mul_i32 s53, s54, s86 // (op 2680) 


s_sub_u32 s54, s52, s53 // (op 2680) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2680) 

v_mul_lo_u32 v31, v26, s31 // (op 2680) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 2680) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 2680) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 2680) most significant: sum


v_mov_b32 v26, s84 // (op 2680) 
v_mov_b32 v34, s85 // (op 2680) 
v_add_co_u32 v36, s[52:53], v26, v30 // (op 2680) least significant half
v_addc_co_u32 v37, s[52:53], v34, v31, s[52:53] // (op 2680) most significant half





s_mul_i32 s53, s30, s3 // (op 2682) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2682) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2682) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2682) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2682) most significant: sum
s_add_u32 s53, s53, s55 // (op 2682) most significant: sum


s_mov_b32 s68, s32 // (op 2682) reg expression
s_mov_b32 s69, s33 // (op 2682) reg expression
s_mov_b32 s70, s52 // (op 2682) reg expression
s_mov_b32 s71, 131072 // (op 2682) call()

s_and_b32 s52, s7, 4294967294 // (op 2816) 
v_and_b32 v26, -4, v1 // (op 2816) 
s_add_u32 s53, s0, s52 // (op 2816) 

v_sub_u32 v30, v1, v26 // (op 2816) 

v_lshlrev_b32 v26, 4, v30 // (op 2816) 

s_mul_i32 s52, s16, s8 // (op 2816) 
s_add_u32 s52, s52, s53 // (op 2816) 

v_lshrrev_b32 v30, 2, v1 // (op 2816) 
s_mul_hi_u32 s53, s52, s13 // (op 2816) 
s_sub_u32 s54, s52, s53 // (op 2816) 
s_lshr_b32 s55, s54, 1 // (op 2816) 

s_add_u32 s54, s55, s53 // (op 2816) 


s_lshr_b32 s53, s54, s11 // (op 2816) 

s_and_b32 s54, s11, 2147483648 // (op 2816) 
s_cmp_eq_i32 s54, 0 // (op 2816) 
s_mov_b32 s55, scc // (op 2816) 

s_cmp_eq_i32 s55, 0 // (op 2816) 
s_cselect_b32 s54, s52, s53 // (op 2816) 


s_mul_hi_u32 s53, s54, s12 // (op 2816) 
s_sub_u32 s55, s54, s53 // (op 2816) 
s_lshr_b32 s84, s55, 1 // (op 2816) 

s_add_u32 s55, s84, s53 // (op 2816) 


s_lshr_b32 s53, s55, s10 // (op 2816) 

s_and_b32 s55, s10, 2147483648 // (op 2816) 
s_cmp_eq_i32 s55, 0 // (op 2816) 
s_mov_b32 s84, scc // (op 2816) 

s_cmp_eq_i32 s84, 0 // (op 2816) 
s_cselect_b32 s55, s54, s53 // (op 2816) 


s_mul_i32 s53, s55, s14 // (op 2816) 

s_sub_u32 s55, s54, s53 // (op 2816) 

v_lshl_add_u32 v31, s55, 6, v26 // (op 2816) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 2816) 
v_mov_b32 v35, 0 // (op 2816) convert
v_mov_b32 v34, v31 // (op 2816) convert

s_mul_i32 s53, s54, s84 // (op 2816) 


s_sub_u32 s54, s52, s53 // (op 2816) 


v_lshl_add_u32 v26, s54, 8, v30 // (op 2816) 


v_mul_lo_u32 v31, v26, s35 // (op 2816) most significant: low of low * high
v_mul_hi_u32 v39, v26, s34 // (op 2816) most significant: high of low * low
v_mul_lo_u32 v30, v26, s34 // (op 2816) least significant: low of low * low
v_add_u32 v31, v31, v39 // (op 2816) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v34 // (op 2816) least significant half
v_addc_co_u32 v31, s[52:53], v31, v35, s[52:53] // (op 2816) most significant half



s_mul_i32 s53, s34, s3 // (op 2818) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2818) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2818) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2818) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2818) most significant: sum
s_add_u32 s53, s53, s55 // (op 2818) most significant: sum


s_mov_b32 s72, s36 // (op 2818) reg expression
s_mov_b32 s73, s37 // (op 2818) reg expression
s_mov_b32 s74, s52 // (op 2818) reg expression
s_mov_b32 s75, 131072 // (op 2818) call()

s_and_b32 s52, s7, 4294967294 // (op 2944) 
v_and_b32 v26, -16, v1 // (op 2944) 
s_add_u32 s53, s0, s52 // (op 2944) 

v_sub_u32 v34, v1, v26 // (op 2944) 

v_lshrrev_b32 v26, 4, v1 // (op 2944) 
s_mul_i32 s52, s16, s8 // (op 2944) 
s_add_u32 s52, s52, s53 // (op 2944) 

s_mul_hi_u32 s53, s52, s13 // (op 2944) 
s_sub_u32 s54, s52, s53 // (op 2944) 
s_lshr_b32 s55, s54, 1 // (op 2944) 

s_add_u32 s54, s55, s53 // (op 2944) 


s_lshr_b32 s53, s54, s11 // (op 2944) 

s_and_b32 s54, s11, 2147483648 // (op 2944) 
s_cmp_eq_i32 s54, 0 // (op 2944) 
s_mov_b32 s55, scc // (op 2944) 

s_cmp_eq_i32 s55, 0 // (op 2944) 
s_cselect_b32 s54, s52, s53 // (op 2944) 


s_mul_hi_u32 s53, s54, s12 // (op 2944) 
s_sub_u32 s55, s54, s53 // (op 2944) 
s_lshr_b32 s84, s55, 1 // (op 2944) 

s_add_u32 s55, s84, s53 // (op 2944) 


s_lshr_b32 s53, s55, s10 // (op 2944) 

s_and_b32 s55, s10, 2147483648 // (op 2944) 
s_cmp_eq_i32 s55, 0 // (op 2944) 
s_mov_b32 s84, scc // (op 2944) 

s_cmp_eq_i32 s84, 0 // (op 2944) 
s_cselect_b32 s55, s54, s53 // (op 2944) 


v_lshl_add_u32 v35, s55, 4, v34 // (op 2944) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 2944) 
s_mul_i32 s53, s54, s84 // (op 2944) 


s_sub_u32 s54, s52, s53 // (op 2944) 


v_lshl_add_u32 v34, s54, 3, v26 // (op 2944) 


v_mul_lo_u32 v26, v34, s38 // (op 2944) 

v_add_u32 v34, v35, v26 // (op 2944) 


s_mul_i32 s53, s38, s43 // (op 2946) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2946) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2946) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2946) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2946) most significant: sum
s_add_u32 s53, s53, s55 // (op 2946) most significant: sum


s_mov_b32 s76, s40 // (op 2946) reg expression
s_mov_b32 s77, s41 // (op 2946) reg expression
s_mov_b32 s78, s52 // (op 2946) reg expression
s_mov_b32 s79, 131072 // (op 2946) call()

s_and_b32 s52, s7, 4294967294 // (op 3072) 
v_and_b32 v26, -16, v1 // (op 3072) 
s_add_u32 s53, s0, s52 // (op 3072) 

v_sub_u32 v35, v1, v26 // (op 3072) 

v_lshlrev_b32 v26, 2, v35 // (op 3072) 

s_mul_i32 s52, s16, s8 // (op 3072) 
s_add_u32 s52, s52, s53 // (op 3072) 

v_lshrrev_b32 v35, 4, v1 // (op 3072) 
s_mul_hi_u32 s53, s52, s13 // (op 3072) 
s_sub_u32 s54, s52, s53 // (op 3072) 
s_lshr_b32 s55, s54, 1 // (op 3072) 

s_add_u32 s54, s55, s53 // (op 3072) 


s_lshr_b32 s53, s54, s11 // (op 3072) 

s_and_b32 s54, s11, 2147483648 // (op 3072) 
s_cmp_eq_i32 s54, 0 // (op 3072) 
s_mov_b32 s55, scc // (op 3072) 

s_cmp_eq_i32 s55, 0 // (op 3072) 
s_cselect_b32 s54, s52, s53 // (op 3072) 


s_mul_hi_u32 s53, s54, s12 // (op 3072) 
s_sub_u32 s55, s54, s53 // (op 3072) 
s_lshr_b32 s84, s55, 1 // (op 3072) 

s_add_u32 s55, s84, s53 // (op 3072) 


s_lshr_b32 s53, s55, s10 // (op 3072) 

s_and_b32 s55, s10, 2147483648 // (op 3072) 
s_cmp_eq_i32 s55, 0 // (op 3072) 
s_mov_b32 s84, scc // (op 3072) 

s_cmp_eq_i32 s84, 0 // (op 3072) 
s_cselect_b32 s55, s54, s53 // (op 3072) 


s_mul_i32 s53, s55, s14 // (op 3072) 

s_sub_u32 s55, s54, s53 // (op 3072) 

v_lshl_add_u32 v39, s55, 6, v26 // (op 3072) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 3072) 
s_mul_i32 s53, s54, s84 // (op 3072) 


s_sub_u32 s54, s52, s53 // (op 3072) 


v_lshl_add_u32 v26, s54, 3, v35 // (op 3072) 


v_mul_lo_u32 v35, v26, s44 // (op 3072) 
v_add_u32 v35, v35, v39 // (op 3072) 


s_mul_i32 s53, s44, s43 // (op 3074) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3074) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3074) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3074) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3074) most significant: sum
s_add_u32 s53, s53, s55 // (op 3074) most significant: sum


s_mov_b32 s80, s46 // (op 3074) reg expression
s_mov_b32 s81, s47 // (op 3074) reg expression
s_mov_b32 s82, s52 // (op 3074) reg expression
s_mov_b32 s83, 131072 // (op 3074) call()

v_lshlrev_b32 v26, 4, v1 // (op 3233) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 3233) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 3298) 
v_sub_u32 v39, v1, v26 // (op 3298) 

v_lshlrev_b32 v26, 4, v39 // (op 3298) 

v_lshrrev_b32 v39, 2, v1 // (op 3298) 
v_lshl_add_u32 v40, v39, 6, v26 // (op 3298) 


s_nop 0
v_readfirstlane_b32 s53, v40 // (op 3298) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -16, v1 // (op 3373) 
v_sub_u32 v39, v1, v26 // (op 3373) 

v_and_b32 v26, 4294967280, v1 // (op 3373) 
v_add_u32 v40, v26, v39 // (op 3373) 


v_and_b32 v26, -16, v1 // (op 3439) 
v_sub_u32 v39, v1, v26 // (op 3439) 

v_lshlrev_b32 v26, 2, v39 // (op 3439) 

v_lshrrev_b32 v39, 4, v1 // (op 3439) 
v_lshl_add_u32 v41, v39, 6, v26 // (op 3439) 


s_and_b32 s54, s7, 4294967294 // (op 1268) 
s_cmp_lt_u32 s54, s7 // (op 1267) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1267 // (op 1267) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1267 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1267:
s_barrier  // (op 4282) 
s_add_u32 s54, s54, 1 // (op 1269) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 2594) 
v_mov_b64 v[28:29], s[84:85] // (op 2594) convert


v_mov_b64 v[32:33], v[36:37] // (op 2594) 
v_mov_b64 v[38:39], v[32:33] // (op 2594) reg expression
s_add_u32 s55, 16384, s52 // (op 2594) 
s_mov_b32 m0, s55 // (op 2594) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2594) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2594) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2594) most significant half

s_add_u32 m0, m0, 2048 // (op 2594) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2594) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 8 // (op 2670) 
v_mov_b32 v26, s84 // (op 2670) 
v_mov_b32 v28, s85 // (op 2670) 
v_add_co_u32 v36, s[86:87], v36, v26 // (op 2670) least significant half
v_addc_co_u32 v37, s[86:87], v37, v28, s[86:87] // (op 2670) most significant half




s_barrier  // (op 4301) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 2600) 
v_mov_b64 v[28:29], s[84:85] // (op 2600) convert


v_mov_b64 v[32:33], v[30:31] // (op 2600) 
v_mov_b64 v[38:39], v[32:33] // (op 2600) reg expression
s_mov_b32 m0, s53 // (op 2600) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2600) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2600) Load value direct to lds



s_lshl_b64 s[84:85], s[34:35], 8 // (op 2808) 
v_mov_b32 v26, s84 // (op 2808) 
v_mov_b32 v28, s85 // (op 2808) 
v_add_co_u32 v30, s[86:87], v30, v26 // (op 2808) least significant half
v_addc_co_u32 v31, s[86:87], v31, v28, s[86:87] // (op 2808) most significant half






v_mov_b32 v26, v34 // (op 1085) 
v_mov_b32 v28, v26 // (op 1085) reg expression

buffer_load_ubyte v29, v28, s[76:79], 0 offen offset: 0 // (op 1085) Load value


s_lshl_b32 s55, s38, 3 // (op 2936) 
v_add_u32 v34, v34, s55 // (op 2936) 



s_barrier  // (op 4321) 
v_mov_b32 v26, v35 // (op 1086) 



v_mov_b32 v28, v26 // (op 1086) reg expression

buffer_load_dword v32, v28, s[80:83], 0 offen // (op 1086) Load value


s_lshl_b32 s55, s44, 3 // (op 3064) 
v_add_u32 v35, v35, s55 // (op 3064) 

s_waitcnt vmcnt(1)
ds_write_b8 v40, v29 offset:41472 // (op 1082) Store local data 

s_barrier  // (op 4288) 



s_barrier  // (op 4342) 
ds_read_b64_tr_b8 v[42:43], v0 offset:16384 // (op 1064) Transpose load local data 



ds_read_b64_tr_b8 v[44:45], v0 offset:16512 // (op 1064) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:17408 // (op 1064) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:17536 // (op 1064) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v41, v32 offset:40960 // (op 1083) Store local data 

s_barrier  // (op 4308) 



ds_read_b64_tr_b8 v[50:51], v11 // (op 1066) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:512 // (op 1066) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v11 offset:4096 // (op 1066) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:4608 // (op 1066) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4329) 


ds_read_u8 v26, v4 offset:41472 // (op 1068) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4349) 


ds_read_u8 v28, v9 offset:40960 // (op 1070) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:57], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1071) 





ds_read_b64_tr_b8 v[50:51], v0 offset:18432 // (op 1073) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:18560 // (op 1073) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:19456 // (op 1073) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:19584 // (op 1073) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v11 offset:8192 // (op 1075) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v11 offset:8704 // (op 1075) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v11 offset:12288 // (op 1075) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v11 offset:12800 // (op 1075) Transpose load local data 


ds_read_u8 v28, v4 offset:41536 // (op 1077) Load local data 


ds_read_u8 v29, v9 offset:41216 // (op 1079) Load local data 



ds_read_b64_tr_b8 v[66:67], v11 offset:32 // (op 1275) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v11 offset:544 // (op 1275) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v11 offset:4128 // (op 1275) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v11 offset:4640 // (op 1275) Transpose load local data 


ds_read_u8 v32, v9 offset:40992 // (op 1279) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[66:73], a[4:7], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1280) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[58:65], a[0:3], v28, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1080) 
ds_read_b64_tr_b8 v[42:43], v11 offset:8224 // (op 1284) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v11 offset:8736 // (op 1284) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:12320 // (op 1284) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:12832 // (op 1284) Transpose load local data 


ds_read_u8 v26, v9 offset:41248 // (op 1288) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[42:49], a[4:7], v28, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1289) 




s_cmp_lt_u32 s54, s7 // (op 1267) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1267 // (op 1267) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1267 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1267:









s_barrier  // (op 4657) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 328) 
s_cmp_lg_u32 s18, 0 // (op 354) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_354 // (op 354) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_354



buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 351) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 351) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 351) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 351) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 1572) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 1572) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 1572) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 1572) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 355) 
v_cmp_eq_i32 vcc, v1, 0 // (op 359) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359 // (op 359) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359
s_mov_b32 s52, 1 // (op 357) call()
s_mov_b32 s55, 0 // (op 358) convert
s_mov_b32 s54, s16 // (op 358) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 358) 
s_add_u32 s84, s48, s54 // (op 358) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 358) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 358) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359 // (op 359) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_354 // (op 354) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_354
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_354:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_354:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 370) 
s_mov_b32 s52, scc // (op 370) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 370) 
s_mov_b32 s53, s54 // (op 370) convert

s_sub_u32 s54, s53, 1 // (op 370) 

s_cmp_lt_u32 s19, s54 // (op 370) 
s_mov_b32 s53, scc // (op 370) 

s_and_b32 s54, s52, s53 // (op 370) 
s_cmp_lg_u32 s54, 0 // (op 370) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_370 // (op 370) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_370
s_mov_b32 s52, 0 // (op 336) call()
s_mul_i32 s53, s52, s8 // (op 335) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 335) 
s_mov_b32 s84, s54 // (op 335) convert

s_sub_u32 s54, s84, s19 // (op 335) 

s_sub_u32 s55, s54, 1 // (op 335) 

s_cmp_lt_u32 s53, s55 // (op 335) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_335 // (op 335) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_335 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_335:
s_add_u32 s53, s16, 1 // (op 3853) 
v_lshrrev_b32 v20, 5, v1 // (op 3853) 
s_add_u32 s54, s53, s52 // (op 3853) 
s_lshl_b32 s54, s54, 4 // (op 3853) 

v_lshlrev_b32 v21, 2, v20 // (op 3853) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 3853) 


v_and_b32 v21, -32, v1 // (op 3853) 
v_sub_u32 v23, v1, v21 // (op 3853) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 3853) 


s_lshl_b32 s53, s15, 12 // (op 3855) 
s_mov_b32 s64, s28 // (op 3855) reg expression
s_mov_b32 s65, s29 // (op 3855) reg expression
s_mov_b32 s66, s53 // (op 3855) reg expression
s_mov_b32 s67, 131072 // (op 3855) call()

RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_374:
s_add_u32 s53, s16, 1 // (op 372) 
s_add_u32 s54, s53, s52 // (op 372) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 372) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 372) 

s_add_u32 s84, s48, s54 // (op 372) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 372) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 372) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 374) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_374 // (op 374) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_374 if true)
s_barrier  // (op 378) 
v_cmp_eq_i32 vcc, v1, 0 // (op 377) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377 // (op 377) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377
s_mov_b32 s53, 0 // (op 375) call()
s_add_u32 s54, s16, 1 // (op 376) 
s_add_u32 s84, s54, s52 // (op 376) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 376) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 376) 

s_add_u32 s54, s48, s84 // (op 376) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 376) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 376) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377 // (op 377) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 389) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:256 // (op 389) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:512 // (op 389) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:768 // (op 389) Load value
v_accvgpr_read v32, a0 // (op 392) reg expression
v_accvgpr_read v33, a1 // (op 392) reg expression
v_accvgpr_read v34, a2 // (op 392) reg expression
v_accvgpr_read v35, a3 // (op 392) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 394) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 394) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 394) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 394) 

v_accvgpr_write a0, v32 // (op 396) reg expression
v_accvgpr_write a1, v33 // (op 396) reg expression
v_accvgpr_write a2, v34 // (op 396) reg expression
v_accvgpr_write a3, v35 // (op 396) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 1582) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 1582) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 1582) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 1582) Load value
v_accvgpr_read v32, a4 // (op 1583) reg expression
v_accvgpr_read v33, a5 // (op 1583) reg expression
v_accvgpr_read v34, a6 // (op 1583) reg expression
v_accvgpr_read v35, a7 // (op 1583) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 1584) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 1584) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 1584) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 1584) 

v_accvgpr_write a4, v32 // (op 1585) reg expression
v_accvgpr_write a5, v33 // (op 1585) reg expression
v_accvgpr_write a6, v34 // (op 1585) reg expression
v_accvgpr_write a7, v35 // (op 1585) reg expression

s_add_u32 s52, s52, 1 // (op 337) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 335) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 335) 
s_mov_b32 s84, s54 // (op 335) convert

s_sub_u32 s54, s84, s19 // (op 335) 

s_sub_u32 s55, s54, 1 // (op 335) 

s_cmp_lt_u32 s53, s55 // (op 335) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_335 // (op 335) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_335 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_335:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_370 // (op 370) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_370
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_370:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_370:

s_cmp_eq_i32 s18, 0 // (op 413) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413 // (op 413) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v28, a0 // (op 50) 
v_accvgpr_read v29, a1 // (op 50) 
v_accvgpr_read v30, a2 // (op 50) 
v_accvgpr_read v31, a3 // (op 50) 
v_mul_f32 v32, v19, v28 // (op 50) 
v_mul_f32 v33, v19, v29 // (op 50) 
v_mul_f32 v34, v19, v30 // (op 50) 
v_mul_f32 v35, v19, v31 // (op 50) 



v_mov_b32 v20, v27 // (op 56) reg expression

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v28, a4 // (op 1601) 
v_accvgpr_read v29, a5 // (op 1601) 
v_accvgpr_read v30, a6 // (op 1601) 
v_accvgpr_read v31, a7 // (op 1601) 
v_mul_f32 v32, v19, v28 // (op 1601) 
v_mul_f32 v33, v19, v29 // (op 1601) 
v_mul_f32 v34, v19, v30 // (op 1601) 
v_mul_f32 v35, v19, v31 // (op 1601) 



s_lshl_b32 s52, s24, 7 // (op 1603) 
v_add_u32 v20, v27, s52 // (op 1603) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 1603) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


v_mov_b32 v20, v22 // (op 167) reg expression

buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 167) Load value

v_accvgpr_read v32, a0 // (op 164) 
v_accvgpr_read v33, a1 // (op 164) 
v_accvgpr_read v34, a2 // (op 164) 
v_accvgpr_read v35, a3 // (op 164) 
v_mul_f32 v36, v19, v32 // (op 164) 
v_mul_f32 v37, v19, v33 // (op 164) 
v_mul_f32 v38, v19, v34 // (op 164) 
v_mul_f32 v39, v19, v35 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 168) 
v_mul_f32 v33, v18, v29 // (op 168) 
v_mul_f32 v34, v18, v30 // (op 168) 
v_mul_f32 v35, v18, v31 // (op 168) 

v_add_f32 v28, v32, v36 // (op 165) 
v_add_f32 v29, v33, v37 // (op 165) 
v_add_f32 v30, v34, v38 // (op 165) 
v_add_f32 v31, v35, v39 // (op 165) 




v_mov_b32 v20, v27 // (op 166) reg expression

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 166) Store value




s_lshl_b32 s52, s4, 7 // (op 1625) 
v_add_u32 v20, v22, s52 // (op 1625) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 1625) Load value

v_accvgpr_read v32, a4 // (op 1622) 
v_accvgpr_read v33, a5 // (op 1622) 
v_accvgpr_read v34, a6 // (op 1622) 
v_accvgpr_read v35, a7 // (op 1622) 
v_mul_f32 v36, v19, v32 // (op 1622) 
v_mul_f32 v37, v19, v33 // (op 1622) 
v_mul_f32 v38, v19, v34 // (op 1622) 
v_mul_f32 v39, v19, v35 // (op 1622) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 1626) 
v_mul_f32 v33, v18, v29 // (op 1626) 
v_mul_f32 v34, v18, v30 // (op 1626) 
v_mul_f32 v35, v18, v31 // (op 1626) 

v_add_f32 v28, v32, v36 // (op 1623) 
v_add_f32 v29, v33, v37 // (op 1623) 
v_add_f32 v30, v34, v38 // (op 1623) 
v_add_f32 v31, v35, v39 // (op 1623) 




s_lshl_b32 s52, s24, 7 // (op 1624) 
v_add_u32 v20, v27, s52 // (op 1624) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 1624) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413 // (op 413) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413:
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413:


s_mul_i32 s52, s8, s16 // (op 326) 
s_cmp_gt_u32 s9, s52 // (op 326) 
s_mov_b32 s53, scc // (op 326) 
s_sub_u32 s54, s9, s52 // (op 326) 

s_cmp_eq_i32 s53, 0 // (op 326) 
s_cselect_b32 s52, 0, s54 // (op 326) 


s_cmp_lt_u32 s8, s52 // (op 326) 
s_mov_b32 s53, scc // (op 326) 
s_cmp_eq_i32 s53, 0 // (op 326) 
s_cselect_b32 s54, s52, s8 // (op 326) 


s_cmp_lt_u32 s0, s54 // (op 326) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_326 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_326:

















s_mov_b32 s0, 0 // (op 467) call()
s_mov_b32 s1, 0 // (op 468) call()
s_mov_b32 s0, 0 // (op 482) call()
s_mul_i32 s15, s50, s16 // (op 481) 
s_cmp_gt_u32 s51, s15 // (op 481) 
s_mov_b32 s6, scc // (op 481) 
s_sub_u32 s7, s51, s15 // (op 481) 

s_cmp_eq_i32 s6, 0 // (op 481) 
s_cselect_b32 s8, 0, s7 // (op 481) 


s_cmp_lt_u32 s50, s8 // (op 481) 
s_mov_b32 s15, scc // (op 481) 
s_cmp_eq_i32 s15, 0 // (op 481) 
s_cselect_b32 s6, s8, s50 // (op 481) 


s_cmp_lt_u32 s0, s6 // (op 481) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_481 // (op 481) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_481 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_481:
v_lshrrev_b32 v0, 7, v1 // (op 3203) 
s_mul_i32 s15, s16, s50 // (op 3203) 
s_add_u32 s15, s15, s0 // (op 3203) 
v_and_b32 v4, 63, v1 // (op 3203) 
s_add_u32 s6, s15, s9 // (op 3203) 

v_lshrrev_b32 v7, 4, v4 // (op 3203) 
s_mul_hi_u32 s15, s6, s13 // (op 3203) 
v_lshlrev_b32 v9, 2, v7 // (op 3203) 

s_sub_u32 s7, s6, s15 // (op 3203) 
v_lshl_add_u32 v7, v0, 4, v9 // (op 3203) 


s_lshr_b32 s8, s7, 1 // (op 3203) 

v_lshrrev_b32 v0, 2, v4 // (op 3203) 

s_add_u32 s7, s8, s15 // (op 3203) 


v_and_b32 v4, 3, v0 // (op 3203) 

s_lshr_b32 s8, s7, s11 // (op 3203) 

v_and_b32 v0, 3, v1 // (op 3203) 
s_and_b32 s7, s11, 2147483648 // (op 3203) 
v_lshl_add_u32 v9, v4, 2, v0 // (op 3203) 


s_cmp_eq_i32 s7, 0 // (op 3203) 
s_mov_b32 s15, scc // (op 3203) 

v_lshrrev_b32 v0, 6, v1 // (op 3203) 
s_cmp_eq_i32 s15, 0 // (op 3203) 
s_cselect_b32 s7, s6, s8 // (op 3203) 



v_and_b32 v4, 1, v0 // (op 3203) 

s_mul_hi_u32 s6, s7, s12 // (op 3203) 
v_lshl_add_u32 v0, v4, 4, v9 // (op 3203) 


s_sub_u32 s8, s7, s6 // (op 3203) 
s_lshr_b32 s15, s8, 1 // (op 3203) 

s_add_u32 s8, s15, s6 // (op 3203) 


s_lshr_b32 s6, s8, s10 // (op 3203) 

s_and_b32 s8, s10, 2147483648 // (op 3203) 
s_cmp_eq_i32 s8, 0 // (op 3203) 
s_mov_b32 s15, scc // (op 3203) 

s_cmp_eq_i32 s15, 0 // (op 3203) 
s_cselect_b32 s8, s7, s6 // (op 3203) 


s_mul_i32 s6, s8, s14 // (op 3203) 
v_lshl_add_u32 v4, s8, 4, v7 // (op 3203) 


s_sub_u32 s8, s7, s6 // (op 3203) 


v_lshl_add_u32 v7, s8, 6, v0 // (op 3203) 


v_mul_lo_u32 v0, v7, s4 // (op 3203) 

v_add_lshl_u32 v7, v4, v0, 2 // (op 3203) 


s_mul_i32 s19, s4, s23 // (op 3205) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3205) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3205) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3205) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3205) most significant: sum
s_add_u32 s19, s19, s6 // (op 3205) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3205) 

s_mov_b32 s56, s20 // (op 3205) reg expression
s_mov_b32 s57, s21 // (op 3205) reg expression
s_mov_b32 s58, s28 // (op 3205) reg expression
s_mov_b32 s59, 131072 // (op 3205) call()

v_lshrrev_b32 v0, 7, v1 // (op 3527) 
s_mul_i32 s15, s16, s50 // (op 3527) 
s_add_u32 s15, s15, s0 // (op 3527) 
v_and_b32 v4, 63, v1 // (op 3527) 
s_add_u32 s6, s15, s9 // (op 3527) 

v_lshrrev_b32 v9, 4, v4 // (op 3527) 
s_mul_hi_u32 s15, s6, s13 // (op 3527) 
v_lshlrev_b32 v10, 2, v9 // (op 3527) 

s_sub_u32 s7, s6, s15 // (op 3527) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3527) 


s_lshr_b32 s8, s7, 1 // (op 3527) 

v_lshrrev_b32 v0, 2, v4 // (op 3527) 

s_add_u32 s7, s8, s15 // (op 3527) 


v_and_b32 v4, 3, v0 // (op 3527) 

s_lshr_b32 s8, s7, s11 // (op 3527) 

v_and_b32 v0, 3, v1 // (op 3527) 
s_and_b32 s7, s11, 2147483648 // (op 3527) 
v_lshl_add_u32 v10, v4, 2, v0 // (op 3527) 


s_cmp_eq_i32 s7, 0 // (op 3527) 
s_mov_b32 s15, scc // (op 3527) 

v_lshrrev_b32 v0, 6, v1 // (op 3527) 
s_cmp_eq_i32 s15, 0 // (op 3527) 
s_cselect_b32 s7, s6, s8 // (op 3527) 



v_and_b32 v4, 1, v0 // (op 3527) 

s_mul_hi_u32 s6, s7, s12 // (op 3527) 
v_lshl_add_u32 v0, v4, 4, v10 // (op 3527) 


s_sub_u32 s8, s7, s6 // (op 3527) 
s_lshr_b32 s15, s8, 1 // (op 3527) 

s_add_u32 s8, s15, s6 // (op 3527) 


s_lshr_b32 s6, s8, s10 // (op 3527) 

s_and_b32 s8, s10, 2147483648 // (op 3527) 
s_cmp_eq_i32 s8, 0 // (op 3527) 
s_mov_b32 s15, scc // (op 3527) 

s_cmp_eq_i32 s15, 0 // (op 3527) 
s_cselect_b32 s8, s7, s6 // (op 3527) 


s_mul_i32 s6, s8, s14 // (op 3527) 
v_lshl_add_u32 v4, s8, 4, v9 // (op 3527) 


s_sub_u32 s8, s7, s6 // (op 3527) 


v_lshl_add_u32 v9, s8, 6, v0 // (op 3527) 


v_mul_lo_u32 v0, v9, s24 // (op 3527) 

v_add_lshl_u32 v9, v4, v0, 2 // (op 3527) 


s_mul_i32 s19, s24, s23 // (op 3529) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3529) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3529) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3529) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3529) most significant: sum
s_add_u32 s19, s19, s6 // (op 3529) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3529) 

s_mov_b32 s60, s26 // (op 3529) reg expression
s_mov_b32 s61, s27 // (op 3529) reg expression
s_mov_b32 s62, s28 // (op 3529) reg expression
s_mov_b32 s63, 131072 // (op 3529) call()

v_and_b32 v0, 48, v1 // (op 3596) 
v_and_b32 v4, 63, v1 // (op 3596) 
v_sub_u32 v10, v4, v0 // (op 3596) 

v_lshrrev_b32 v4, 1, v10 // (op 3596) 
v_add_lshl_u32 v11, v0, v4, 4 // (op 3596) 


v_and_b32 v0, 1, v10 // (op 3596) 

v_lshlrev_b32 v4, 3, v0 // (op 3596) 

v_lshrrev_b32 v0, 7, v1 // (op 3596) 
v_lshl_add_u32 v10, v0, 4, v4 // (op 3596) 


v_add_u32 v0, v11, v10 // (op 3596) 


v_and_b32 v4, 48, v1 // (op 3671) 
v_and_b32 v10, 63, v1 // (op 3671) 
v_sub_u32 v11, v10, v4 // (op 3671) 

v_lshrrev_b32 v10, 1, v11 // (op 3671) 
v_add_lshl_u32 v16, v4, v10, 6 // (op 3671) 


v_and_b32 v4, 1, v11 // (op 3671) 

v_lshlrev_b32 v17, 3, v4 // (op 3671) 

v_lshrrev_b32 v4, 6, v1 // (op 3671) 
v_and_b32 v10, 1, v4 // (op 3671) 

v_lshl_add_u32 v4, v10, 4, v17 // (op 3671) 


v_add_u32 v17, v16, v4 // (op 3671) 


v_lshrrev_b32 v4, 7, v1 // (op 3737) 
v_and_b32 v16, 15, v1 // (op 3737) 
v_lshl_add_u32 v10, v4, 4, v16 // (op 3737) 


v_and_b32 v4, 48, v1 // (op 3737) 
v_add_u32 v11, v4, v10 // (op 3737) 


v_lshrrev_b32 v4, 6, v1 // (op 3813) 
v_and_b32 v10, 1, v4 // (op 3813) 

v_and_b32 v4, 15, v1 // (op 3813) 
v_lshl_add_u32 v16, v10, 4, v4 // (op 3813) 


v_and_b32 v4, 63, v1 // (op 3813) 
v_lshrrev_b32 v10, 4, v4 // (op 3813) 

v_lshl_add_u32 v4, v10, 6, v16 // (op 3813) 


v_accvgpr_write a0, 0.00000 // (op 202) call()
v_accvgpr_write a1, 0.00000 // (op 202) call()
v_accvgpr_write a2, 0.00000 // (op 202) call()
v_accvgpr_write a3, 0.00000 // (op 202) call()
v_accvgpr_write a4, 0.00000 // (op 694) call()
v_accvgpr_write a5, 0.00000 // (op 694) call()
v_accvgpr_write a6, 0.00000 // (op 694) call()
v_accvgpr_write a7, 0.00000 // (op 694) call()
s_mul_i32 s15, s16, s50 // (op 2707) 
s_add_u32 s15, s15, s0 // (op 2707) 
s_add_u32 s6, s15, s9 // (op 2707) 

s_mul_hi_u32 s15, s6, s13 // (op 2707) 
s_sub_u32 s7, s6, s15 // (op 2707) 
s_lshr_b32 s8, s7, 1 // (op 2707) 

s_add_u32 s7, s8, s15 // (op 2707) 


s_lshr_b32 s8, s7, s11 // (op 2707) 

s_and_b32 s7, s11, 2147483648 // (op 2707) 
s_cmp_eq_i32 s7, 0 // (op 2707) 
s_mov_b32 s15, scc // (op 2707) 

s_cmp_eq_i32 s15, 0 // (op 2707) 
s_cselect_b32 s7, s6, s8 // (op 2707) 


s_mul_hi_u32 s8, s7, s12 // (op 2707) 
s_sub_u32 s15, s7, s8 // (op 2707) 
s_lshr_b32 s18, s15, 1 // (op 2707) 

s_add_u32 s15, s18, s8 // (op 2707) 


s_lshr_b32 s8, s15, s10 // (op 2707) 

s_and_b32 s15, s10, 2147483648 // (op 2707) 
s_cmp_eq_i32 s15, 0 // (op 2707) 
s_mov_b32 s18, scc // (op 2707) 

s_cmp_eq_i32 s18, 0 // (op 2707) 
s_cselect_b32 s15, s7, s8 // (op 2707) 


s_lshl_b32 s8, s15, 4 // (op 2707) 

s_mov_b32 s19, 0 // (op 2707) convert
s_mov_b32 s18, s8 // (op 2707) convert

s_ashr_i64 s[28:29], s[2:3], 8 // (op 2707) 
s_mul_i32 s8, s7, s28 // (op 2707) 


s_sub_u32 s7, s6, s8 // (op 2707) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2707) 

v_mul_lo_u32 v21, v10, s31 // (op 2707) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2707) most significant: high of low * low
v_mul_lo_u32 v20, v10, s30 // (op 2707) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2707) most significant: sum


v_mov_b32 v10, s18 // (op 2707) 
v_mov_b32 v16, s19 // (op 2707) 
v_add_co_u32 v22, s[28:29], v10, v20 // (op 2707) least significant half
v_addc_co_u32 v23, s[28:29], v16, v21, s[28:29] // (op 2707) most significant half





s_mul_i32 s19, s30, s3 // (op 2709) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2709) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2709) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2709) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2709) most significant: sum
s_add_u32 s19, s19, s6 // (op 2709) most significant: sum


s_mov_b32 s68, s32 // (op 2709) reg expression
s_mov_b32 s69, s33 // (op 2709) reg expression
s_mov_b32 s70, s18 // (op 2709) reg expression
s_mov_b32 s71, 131072 // (op 2709) call()

s_mul_i32 s15, s16, s50 // (op 2726) 
s_add_u32 s15, s15, s0 // (op 2726) 
s_add_u32 s6, s15, s9 // (op 2726) 

s_mul_hi_u32 s15, s6, s13 // (op 2726) 
s_sub_u32 s7, s6, s15 // (op 2726) 
s_lshr_b32 s8, s7, 1 // (op 2726) 

s_add_u32 s7, s8, s15 // (op 2726) 


s_lshr_b32 s8, s7, s11 // (op 2726) 

s_and_b32 s7, s11, 2147483648 // (op 2726) 
s_cmp_eq_i32 s7, 0 // (op 2726) 
s_mov_b32 s15, scc // (op 2726) 

s_cmp_eq_i32 s15, 0 // (op 2726) 
s_cselect_b32 s7, s6, s8 // (op 2726) 


s_mul_hi_u32 s8, s7, s12 // (op 2726) 
s_sub_u32 s15, s7, s8 // (op 2726) 
s_lshr_b32 s18, s15, 1 // (op 2726) 

s_add_u32 s15, s18, s8 // (op 2726) 


s_lshr_b32 s8, s15, s10 // (op 2726) 

s_and_b32 s15, s10, 2147483648 // (op 2726) 
s_cmp_eq_i32 s15, 0 // (op 2726) 
s_mov_b32 s18, scc // (op 2726) 

s_cmp_eq_i32 s18, 0 // (op 2726) 
s_cselect_b32 s15, s7, s8 // (op 2726) 


s_lshl_b32 s8, s15, 4 // (op 2726) 

s_mov_b32 s19, 0 // (op 2726) convert
s_mov_b32 s18, s8 // (op 2726) convert

s_ashr_i64 s[28:29], s[2:3], 8 // (op 2726) 
s_mul_i32 s8, s7, s28 // (op 2726) 


s_sub_u32 s7, s6, s8 // (op 2726) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2726) 

v_mul_lo_u32 v21, v10, s31 // (op 2726) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2726) most significant: high of low * low
v_mul_lo_u32 v20, v10, s30 // (op 2726) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2726) most significant: sum


v_mov_b32 v10, s18 // (op 2726) 
v_mov_b32 v16, s19 // (op 2726) 
v_add_co_u32 v24, s[28:29], v10, v20 // (op 2726) least significant half
v_addc_co_u32 v25, s[28:29], v16, v21, s[28:29] // (op 2726) most significant half





s_mul_i32 s19, s30, s3 // (op 2728) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2728) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2728) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2728) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2728) most significant: sum
s_add_u32 s19, s19, s6 // (op 2728) most significant: sum


s_mov_b32 s68, s32 // (op 2728) reg expression
s_mov_b32 s69, s33 // (op 2728) reg expression
s_mov_b32 s70, s18 // (op 2728) reg expression
s_mov_b32 s71, 131072 // (op 2728) call()

v_and_b32 v10, -4, v1 // (op 2839) 
s_mul_i32 s15, s16, s50 // (op 2839) 
s_add_u32 s15, s15, s0 // (op 2839) 
v_sub_u32 v16, v1, v10 // (op 2839) 

s_add_u32 s6, s15, s9 // (op 2839) 

v_lshlrev_b32 v10, 4, v16 // (op 2839) 

s_mul_hi_u32 s15, s6, s13 // (op 2839) 
v_lshrrev_b32 v16, 2, v1 // (op 2839) 
s_sub_u32 s7, s6, s15 // (op 2839) 
s_lshr_b32 s8, s7, 1 // (op 2839) 

s_add_u32 s7, s8, s15 // (op 2839) 


s_lshr_b32 s8, s7, s11 // (op 2839) 

s_and_b32 s7, s11, 2147483648 // (op 2839) 
s_cmp_eq_i32 s7, 0 // (op 2839) 
s_mov_b32 s15, scc // (op 2839) 

s_cmp_eq_i32 s15, 0 // (op 2839) 
s_cselect_b32 s7, s6, s8 // (op 2839) 


s_mul_hi_u32 s8, s7, s12 // (op 2839) 
s_sub_u32 s15, s7, s8 // (op 2839) 
s_lshr_b32 s18, s15, 1 // (op 2839) 

s_add_u32 s15, s18, s8 // (op 2839) 


s_lshr_b32 s8, s15, s10 // (op 2839) 

s_and_b32 s15, s10, 2147483648 // (op 2839) 
s_cmp_eq_i32 s15, 0 // (op 2839) 
s_mov_b32 s18, scc // (op 2839) 

s_cmp_eq_i32 s18, 0 // (op 2839) 
s_cselect_b32 s15, s7, s8 // (op 2839) 


s_mul_i32 s8, s15, s14 // (op 2839) 

s_sub_u32 s15, s7, s8 // (op 2839) 

v_lshl_add_u32 v20, s15, 6, v10 // (op 2839) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2839) 
v_mov_b32 v27, 0 // (op 2839) convert
v_mov_b32 v26, v20 // (op 2839) convert

s_mul_i32 s8, s7, s18 // (op 2839) 


s_sub_u32 s7, s6, s8 // (op 2839) 


v_lshl_add_u32 v10, s7, 8, v16 // (op 2839) 


v_mul_lo_u32 v21, v10, s35 // (op 2839) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2839) most significant: high of low * low
v_mul_lo_u32 v20, v10, s34 // (op 2839) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2839) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v26 // (op 2839) least significant half
v_addc_co_u32 v21, s[18:19], v21, v27, s[18:19] // (op 2839) most significant half



s_mul_i32 s19, s34, s3 // (op 2841) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2841) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2841) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2841) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2841) most significant: sum
s_add_u32 s19, s19, s6 // (op 2841) most significant: sum


s_mov_b32 s72, s36 // (op 2841) reg expression
s_mov_b32 s73, s37 // (op 2841) reg expression
s_mov_b32 s74, s18 // (op 2841) reg expression
s_mov_b32 s75, 131072 // (op 2841) call()

v_and_b32 v10, -4, v1 // (op 2858) 
s_mul_i32 s15, s16, s50 // (op 2858) 
s_add_u32 s15, s15, s0 // (op 2858) 
v_sub_u32 v16, v1, v10 // (op 2858) 

s_add_u32 s6, s15, s9 // (op 2858) 

v_lshlrev_b32 v10, 4, v16 // (op 2858) 

s_mul_hi_u32 s15, s6, s13 // (op 2858) 
v_lshrrev_b32 v16, 2, v1 // (op 2858) 
s_sub_u32 s7, s6, s15 // (op 2858) 
s_lshr_b32 s8, s7, 1 // (op 2858) 

s_add_u32 s7, s8, s15 // (op 2858) 


s_lshr_b32 s8, s7, s11 // (op 2858) 

s_and_b32 s7, s11, 2147483648 // (op 2858) 
s_cmp_eq_i32 s7, 0 // (op 2858) 
s_mov_b32 s15, scc // (op 2858) 

s_cmp_eq_i32 s15, 0 // (op 2858) 
s_cselect_b32 s7, s6, s8 // (op 2858) 


s_mul_hi_u32 s8, s7, s12 // (op 2858) 
s_sub_u32 s15, s7, s8 // (op 2858) 
s_lshr_b32 s18, s15, 1 // (op 2858) 

s_add_u32 s15, s18, s8 // (op 2858) 


s_lshr_b32 s8, s15, s10 // (op 2858) 

s_and_b32 s15, s10, 2147483648 // (op 2858) 
s_cmp_eq_i32 s15, 0 // (op 2858) 
s_mov_b32 s18, scc // (op 2858) 

s_cmp_eq_i32 s18, 0 // (op 2858) 
s_cselect_b32 s15, s7, s8 // (op 2858) 


s_mul_i32 s8, s15, s14 // (op 2858) 

s_sub_u32 s15, s7, s8 // (op 2858) 

v_lshl_add_u32 v26, s15, 6, v10 // (op 2858) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2858) 
v_mov_b32 v29, 0 // (op 2858) convert
v_mov_b32 v28, v26 // (op 2858) convert

s_mul_i32 s8, s7, s18 // (op 2858) 


s_sub_u32 s7, s6, s8 // (op 2858) 


v_lshl_add_u32 v10, s7, 8, v16 // (op 2858) 


v_mul_lo_u32 v27, v10, s35 // (op 2858) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2858) most significant: high of low * low
v_mul_lo_u32 v26, v10, s34 // (op 2858) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 2858) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v28 // (op 2858) least significant half
v_addc_co_u32 v27, s[18:19], v27, v29, s[18:19] // (op 2858) most significant half



s_mul_i32 s19, s34, s3 // (op 2860) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2860) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2860) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2860) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2860) most significant: sum
s_add_u32 s19, s19, s6 // (op 2860) most significant: sum


s_mov_b32 s72, s36 // (op 2860) reg expression
s_mov_b32 s73, s37 // (op 2860) reg expression
s_mov_b32 s74, s18 // (op 2860) reg expression
s_mov_b32 s75, 131072 // (op 2860) call()

v_and_b32 v10, -16, v1 // (op 2967) 
s_mul_i32 s15, s16, s50 // (op 2967) 
s_add_u32 s15, s15, s0 // (op 2967) 
v_sub_u32 v16, v1, v10 // (op 2967) 

s_add_u32 s6, s15, s9 // (op 2967) 

v_lshrrev_b32 v10, 4, v1 // (op 2967) 
s_mul_hi_u32 s15, s6, s13 // (op 2967) 
s_sub_u32 s7, s6, s15 // (op 2967) 
s_lshr_b32 s8, s7, 1 // (op 2967) 

s_add_u32 s7, s8, s15 // (op 2967) 


s_lshr_b32 s8, s7, s11 // (op 2967) 

s_and_b32 s7, s11, 2147483648 // (op 2967) 
s_cmp_eq_i32 s7, 0 // (op 2967) 
s_mov_b32 s15, scc // (op 2967) 

s_cmp_eq_i32 s15, 0 // (op 2967) 
s_cselect_b32 s7, s6, s8 // (op 2967) 


s_mul_hi_u32 s8, s7, s12 // (op 2967) 
s_sub_u32 s15, s7, s8 // (op 2967) 
s_lshr_b32 s18, s15, 1 // (op 2967) 

s_add_u32 s15, s18, s8 // (op 2967) 


s_lshr_b32 s8, s15, s10 // (op 2967) 

s_and_b32 s15, s10, 2147483648 // (op 2967) 
s_cmp_eq_i32 s15, 0 // (op 2967) 
s_mov_b32 s18, scc // (op 2967) 

s_cmp_eq_i32 s18, 0 // (op 2967) 
s_cselect_b32 s15, s7, s8 // (op 2967) 


v_lshl_add_u32 v28, s15, 4, v16 // (op 2967) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2967) 
s_mul_i32 s8, s7, s18 // (op 2967) 


s_sub_u32 s7, s6, s8 // (op 2967) 


v_lshl_add_u32 v16, s7, 3, v10 // (op 2967) 


v_mul_lo_u32 v10, v16, s38 // (op 2967) 

v_add_u32 v16, v28, v10 // (op 2967) 


s_mul_i32 s19, s38, s43 // (op 2969) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2969) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2969) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2969) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2969) most significant: sum
s_add_u32 s19, s19, s6 // (op 2969) most significant: sum


s_mov_b32 s76, s40 // (op 2969) reg expression
s_mov_b32 s77, s41 // (op 2969) reg expression
s_mov_b32 s78, s18 // (op 2969) reg expression
s_mov_b32 s79, 131072 // (op 2969) call()

v_and_b32 v10, -16, v1 // (op 2986) 
s_mul_i32 s15, s16, s50 // (op 2986) 
s_add_u32 s15, s15, s0 // (op 2986) 
v_sub_u32 v28, v1, v10 // (op 2986) 

s_add_u32 s6, s15, s9 // (op 2986) 

v_lshrrev_b32 v10, 4, v1 // (op 2986) 
s_mul_hi_u32 s15, s6, s13 // (op 2986) 
s_sub_u32 s7, s6, s15 // (op 2986) 
s_lshr_b32 s8, s7, 1 // (op 2986) 

s_add_u32 s7, s8, s15 // (op 2986) 


s_lshr_b32 s8, s7, s11 // (op 2986) 

s_and_b32 s7, s11, 2147483648 // (op 2986) 
s_cmp_eq_i32 s7, 0 // (op 2986) 
s_mov_b32 s15, scc // (op 2986) 

s_cmp_eq_i32 s15, 0 // (op 2986) 
s_cselect_b32 s7, s6, s8 // (op 2986) 


s_mul_hi_u32 s8, s7, s12 // (op 2986) 
s_sub_u32 s15, s7, s8 // (op 2986) 
s_lshr_b32 s18, s15, 1 // (op 2986) 

s_add_u32 s15, s18, s8 // (op 2986) 


s_lshr_b32 s8, s15, s10 // (op 2986) 

s_and_b32 s15, s10, 2147483648 // (op 2986) 
s_cmp_eq_i32 s15, 0 // (op 2986) 
s_mov_b32 s18, scc // (op 2986) 

s_cmp_eq_i32 s18, 0 // (op 2986) 
s_cselect_b32 s15, s7, s8 // (op 2986) 


v_lshl_add_u32 v29, s15, 4, v28 // (op 2986) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2986) 
s_mul_i32 s8, s7, s18 // (op 2986) 


s_sub_u32 s7, s6, s8 // (op 2986) 


v_lshl_add_u32 v28, s7, 3, v10 // (op 2986) 


v_mul_lo_u32 v10, v28, s38 // (op 2986) 

v_add_u32 v28, v29, v10 // (op 2986) 


s_mul_i32 s19, s38, s43 // (op 2988) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2988) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2988) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2988) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2988) most significant: sum
s_add_u32 s19, s19, s6 // (op 2988) most significant: sum


s_mov_b32 s76, s40 // (op 2988) reg expression
s_mov_b32 s77, s41 // (op 2988) reg expression
s_mov_b32 s78, s18 // (op 2988) reg expression
s_mov_b32 s79, 131072 // (op 2988) call()

v_and_b32 v10, -16, v1 // (op 3095) 
s_mul_i32 s15, s16, s50 // (op 3095) 
s_add_u32 s15, s15, s0 // (op 3095) 
v_sub_u32 v29, v1, v10 // (op 3095) 

s_add_u32 s6, s15, s9 // (op 3095) 

v_lshlrev_b32 v10, 2, v29 // (op 3095) 

s_mul_hi_u32 s15, s6, s13 // (op 3095) 
v_lshrrev_b32 v29, 4, v1 // (op 3095) 
s_sub_u32 s7, s6, s15 // (op 3095) 
s_lshr_b32 s8, s7, 1 // (op 3095) 

s_add_u32 s7, s8, s15 // (op 3095) 


s_lshr_b32 s8, s7, s11 // (op 3095) 

s_and_b32 s7, s11, 2147483648 // (op 3095) 
s_cmp_eq_i32 s7, 0 // (op 3095) 
s_mov_b32 s15, scc // (op 3095) 

s_cmp_eq_i32 s15, 0 // (op 3095) 
s_cselect_b32 s7, s6, s8 // (op 3095) 


s_mul_hi_u32 s8, s7, s12 // (op 3095) 
s_sub_u32 s15, s7, s8 // (op 3095) 
s_lshr_b32 s18, s15, 1 // (op 3095) 

s_add_u32 s15, s18, s8 // (op 3095) 


s_lshr_b32 s8, s15, s10 // (op 3095) 

s_and_b32 s15, s10, 2147483648 // (op 3095) 
s_cmp_eq_i32 s15, 0 // (op 3095) 
s_mov_b32 s18, scc // (op 3095) 

s_cmp_eq_i32 s18, 0 // (op 3095) 
s_cselect_b32 s15, s7, s8 // (op 3095) 


s_mul_i32 s8, s15, s14 // (op 3095) 

s_sub_u32 s15, s7, s8 // (op 3095) 

v_lshl_add_u32 v30, s15, 6, v10 // (op 3095) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 3095) 
s_mul_i32 s8, s7, s18 // (op 3095) 


s_sub_u32 s7, s6, s8 // (op 3095) 


v_lshl_add_u32 v10, s7, 3, v29 // (op 3095) 


v_mul_lo_u32 v29, v10, s44 // (op 3095) 
v_add_u32 v29, v29, v30 // (op 3095) 


s_mul_i32 s19, s44, s43 // (op 3097) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3097) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3097) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3097) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3097) most significant: sum
s_add_u32 s19, s19, s6 // (op 3097) most significant: sum


s_mov_b32 s80, s46 // (op 3097) reg expression
s_mov_b32 s81, s47 // (op 3097) reg expression
s_mov_b32 s82, s18 // (op 3097) reg expression
s_mov_b32 s83, 131072 // (op 3097) call()

v_and_b32 v10, -16, v1 // (op 3114) 
s_mul_i32 s15, s16, s50 // (op 3114) 
s_add_u32 s15, s15, s0 // (op 3114) 
v_sub_u32 v30, v1, v10 // (op 3114) 

s_add_u32 s6, s15, s9 // (op 3114) 

v_lshlrev_b32 v10, 2, v30 // (op 3114) 

s_mul_hi_u32 s15, s6, s13 // (op 3114) 
v_lshrrev_b32 v30, 4, v1 // (op 3114) 
s_sub_u32 s7, s6, s15 // (op 3114) 
s_lshr_b32 s8, s7, 1 // (op 3114) 

s_add_u32 s7, s8, s15 // (op 3114) 


s_lshr_b32 s8, s7, s11 // (op 3114) 

s_and_b32 s7, s11, 2147483648 // (op 3114) 
s_cmp_eq_i32 s7, 0 // (op 3114) 
s_mov_b32 s15, scc // (op 3114) 

s_cmp_eq_i32 s15, 0 // (op 3114) 
s_cselect_b32 s7, s6, s8 // (op 3114) 


s_mul_hi_u32 s8, s7, s12 // (op 3114) 
s_sub_u32 s15, s7, s8 // (op 3114) 
s_lshr_b32 s18, s15, 1 // (op 3114) 

s_add_u32 s15, s18, s8 // (op 3114) 


s_lshr_b32 s8, s15, s10 // (op 3114) 

s_and_b32 s15, s10, 2147483648 // (op 3114) 
s_cmp_eq_i32 s15, 0 // (op 3114) 
s_mov_b32 s18, scc // (op 3114) 

s_cmp_eq_i32 s18, 0 // (op 3114) 
s_cselect_b32 s15, s7, s8 // (op 3114) 


s_mul_i32 s8, s15, s14 // (op 3114) 

s_sub_u32 s15, s7, s8 // (op 3114) 

v_lshl_add_u32 v31, s15, 6, v10 // (op 3114) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 3114) 
s_mul_i32 s8, s7, s18 // (op 3114) 


s_sub_u32 s7, s6, s8 // (op 3114) 


v_lshl_add_u32 v10, s7, 3, v30 // (op 3114) 


v_mul_lo_u32 v30, v10, s44 // (op 3114) 
v_add_u32 v30, v30, v31 // (op 3114) 


s_mul_i32 s19, s44, s43 // (op 3116) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3116) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3116) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3116) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3116) most significant: sum
s_add_u32 s19, s19, s6 // (op 3116) most significant: sum


s_mov_b32 s80, s46 // (op 3116) reg expression
s_mov_b32 s81, s47 // (op 3116) reg expression
s_mov_b32 s82, s18 // (op 3116) reg expression
s_mov_b32 s83, 131072 // (op 3116) call()

v_and_b32 v10, -4, v1 // (op 3637) 
v_sub_u32 v31, v1, v10 // (op 3637) 

v_lshlrev_b32 v10, 4, v31 // (op 3637) 

v_lshrrev_b32 v31, 2, v1 // (op 3637) 
v_lshl_add_u32 v32, v31, 6, v10 // (op 3637) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 3637) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4375) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2617) 
v_mov_b64 v[32:33], s[18:19] // (op 2617) convert


v_mov_b64 v[34:35], v[20:21] // (op 2617) reg expression
s_mov_b32 m0, s15 // (op 2617) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2617) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2617) most significant half

s_add_u32 m0, m0, 2048 // (op 2617) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2617) Load value direct to lds





v_mov_b32 v10, v29 // (op 2219) reg expression

buffer_load_dword v31, v10, s[80:83], 0 offen // (op 2219) Load value

v_lshlrev_b32 v10, 4, v1 // (op 3572) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3572) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4356) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2622) 
v_mov_b64 v[32:33], s[18:19] // (op 2622) convert


v_mov_b64 v[34:35], v[22:23] // (op 2622) reg expression
s_add_u32 s6, 16384, s15 // (op 2622) 
s_mov_b32 m0, s6 // (op 2622) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2622) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2622) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2622) most significant half

s_add_u32 m0, m0, 2048 // (op 2622) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2622) Load value direct to lds






v_mov_b32 v10, v16 // (op 2225) reg expression

buffer_load_ubyte v32, v10, s[76:79], 0 offen offset: 0 // (op 2225) Load value

v_and_b32 v10, -4, v1 // (op 3922) 
v_sub_u32 v33, v1, v10 // (op 3922) 

v_lshlrev_b32 v10, 4, v33 // (op 3922) 

v_lshrrev_b32 v33, 2, v1 // (op 3922) 
v_lshl_add_u32 v34, v33, 6, v10 // (op 3922) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 3922) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4446) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2627) 
v_mov_b64 v[34:35], s[18:19] // (op 2627) convert


s_lshl_b64 s[18:19], s[34:35], 8 // (op 2627) 
v_mov_b32 v10, s18 // (op 2627) 
v_mov_b32 v33, s19 // (op 2627) 
v_add_co_u32 v36, s[28:29], v20, v10 // (op 2627) least significant half
v_addc_co_u32 v37, s[28:29], v21, v33, s[28:29] // (op 2627) most significant half




s_add_u32 s6, 20480, s15 // (op 2627) 
s_mov_b32 m0, s6 // (op 2627) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2627) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2627) most significant half

s_add_u32 m0, m0, 2048 // (op 2627) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2627) Load value direct to lds







s_lshl_b32 s15, s44, 3 // (op 2231) 
v_add_u32 v10, v29, s15 // (op 2231) 


buffer_load_dword v20, v10, s[80:83], 0 offen // (op 2231) Load value


v_lshlrev_b32 v10, 4, v1 // (op 3883) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3883) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4436) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2629) 
v_mov_b64 v[34:35], s[18:19] // (op 2629) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 2629) 
v_mov_b32 v10, s18 // (op 2629) 
v_mov_b32 v21, s19 // (op 2629) 
v_add_co_u32 v36, s[28:29], v22, v10 // (op 2629) least significant half
v_addc_co_u32 v37, s[28:29], v23, v21, s[28:29] // (op 2629) most significant half




s_add_u32 s6, 36864, s15 // (op 2629) 
s_mov_b32 m0, s6 // (op 2629) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2629) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2629) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2629) most significant half

s_add_u32 m0, m0, 2048 // (op 2629) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2629) Load value direct to lds







s_lshl_b32 s15, s38, 3 // (op 2237) 
v_add_u32 v10, v16, s15 // (op 2237) 


buffer_load_ubyte v29, v10, s[76:79], 0 offen offset: 0 // (op 2237) Load value


v_and_b32 v10, -16, v1 // (op 3779) 
v_sub_u32 v16, v1, v10 // (op 3779) 

v_lshlrev_b32 v10, 2, v16 // (op 3779) 

v_lshrrev_b32 v16, 4, v1 // (op 3779) 
v_lshl_add_u32 v21, v16, 6, v10 // (op 3779) 


s_barrier  // (op 4416) 



s_waitcnt vmcnt(15)
ds_write_b32 v21, v31 offset:40960 // (op 2240) Store local data 

v_and_b32 v10, -16, v1 // (op 3713) 
v_sub_u32 v16, v1, v10 // (op 3713) 

v_and_b32 v10, 4294967280, v1 // (op 3713) 
v_add_u32 v21, v10, v16 // (op 3713) 


s_barrier  // (op 4395) 



s_waitcnt vmcnt(12)
ds_write_b8 v21, v32 offset:41472 // (op 2242) Store local data 

s_barrier  // (op 4372) 



ds_read_b64_tr_b8 v[34:35], v17 // (op 2263) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v17 offset:512 // (op 2263) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v17 offset:4096 // (op 2263) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v17 offset:4608 // (op 2263) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v17 offset:8192 // (op 2273) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:8704 // (op 2273) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v17 offset:12288 // (op 2273) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v17 offset:12800 // (op 2273) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4413) 


ds_read_u8 v10, v4 offset:40960 // (op 2289) Load local data 


ds_read_u8 v16, v4 offset:41216 // (op 2299) Load local data 



ds_read_b64_tr_b8 v[50:51], v17 offset:32 // (op 2268) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v17 offset:544 // (op 2268) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v17 offset:4128 // (op 2268) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v17 offset:4640 // (op 2268) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v17 offset:8224 // (op 2278) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v17 offset:8736 // (op 2278) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v17 offset:12320 // (op 2278) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v17 offset:12832 // (op 2278) Transpose load local data 


ds_read_u8 v33, v4 offset:40992 // (op 2294) Load local data 


ds_read_u8 v21, v4 offset:41248 // (op 2304) Load local data 
s_barrier  // (op 4353) 



ds_read_b64_tr_b8 v[66:67], v0 offset:16384 // (op 2256) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:16512 // (op 2256) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:17408 // (op 2256) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:17536 // (op 2256) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v0 offset:18432 // (op 2259) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:18560 // (op 2259) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:19456 // (op 2259) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:19584 // (op 2259) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4392) 


ds_read_u8 v22, v11 offset:41472 // (op 2282) Load local data 


ds_read_u8 v23, v11 offset:41536 // (op 2285) Load local data 
v_lshlrev_b32 v82, 4, v1 // (op 3544) 
s_nop 0
v_readfirstlane_b32 s15, v82 // (op 3544) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -4, v1 // (op 3609) 
v_sub_u32 v83, v1, v82 // (op 3609) 

v_lshlrev_b32 v82, 4, v83 // (op 3609) 

v_lshrrev_b32 v83, 2, v1 // (op 3609) 
v_lshl_add_u32 v84, v83, 6, v82 // (op 3609) 


s_nop 0
v_readfirstlane_b32 s6, v84 // (op 3609) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -16, v1 // (op 3684) 
v_sub_u32 v83, v1, v82 // (op 3684) 

v_and_b32 v82, 4294967280, v1 // (op 3684) 
v_add_u32 v84, v82, v83 // (op 3684) 


v_and_b32 v82, -16, v1 // (op 3750) 
v_sub_u32 v83, v1, v82 // (op 3750) 

v_lshlrev_b32 v82, 2, v83 // (op 3750) 

v_lshrrev_b32 v83, 4, v1 // (op 3750) 
v_lshl_add_u32 v85, v83, 6, v82 // (op 3750) 


v_lshlrev_b32 v82, 4, v1 // (op 3868) 
s_nop 0
v_readfirstlane_b32 s7, v82 // (op 3868) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -4, v1 // (op 3907) 
v_sub_u32 v83, v1, v82 // (op 3907) 

v_lshlrev_b32 v82, 4, v83 // (op 3907) 

v_lshrrev_b32 v83, 2, v1 // (op 3907) 
v_lshl_add_u32 v86, v83, 6, v82 // (op 3907) 


s_nop 0
v_readfirstlane_b32 s8, v86 // (op 3907) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -16, v1 // (op 3951) 
v_sub_u32 v83, v1, v82 // (op 3951) 

v_and_b32 v82, 4294967280, v1 // (op 3951) 
v_add_u32 v86, v82, v83 // (op 3951) 


v_and_b32 v82, -16, v1 // (op 3975) 
v_sub_u32 v83, v1, v82 // (op 3975) 

v_lshlrev_b32 v82, 2, v83 // (op 3975) 

v_lshrrev_b32 v83, 4, v1 // (op 3975) 
v_lshl_add_u32 v87, v83, 6, v82 // (op 3975) 


s_mov_b32 s1, 0 // (op 696) call()
s_ashr_i64 s[18:19], s[2:3], 8 // (op 695) 
s_mov_b32 s28, s18 // (op 695) convert

s_and_b32 s29, s28, 4294967294 // (op 695) 

s_cmp_lt_u32 s1, s29 // (op 695) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_695 // (op 695) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_695 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_695:
s_add_u32 s1, s1, 2 // (op 697) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v22, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 216) 
s_barrier  // (op 4378) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v22, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 743) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 2566) 
v_mov_b64 v[82:83], s[18:19] // (op 2566) convert


v_mov_b64 v[88:89], v[26:27] // (op 2566) 
s_lshl_b64 s[18:19], s[34:35], 8 // (op 2566) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2566) 

v_mov_b32 v90, s28 // (op 2566) 
v_mov_b32 v91, s29 // (op 2566) 
v_add_co_u32 v92, s[18:19], v88, v90 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], v89, v91, s[18:19] // (op 2566) most significant half




s_mov_b32 m0, s6 // (op 2566) reg expression
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 225) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 752) 
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2566) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2566) most significant half

s_add_u32 m0, m0, 2048 // (op 2566) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2566) Load value direct to lds




v_mov_b32 v82, v30 // (op 231) 
s_lshl_b32 s18, s44, 4 // (op 231) 
v_add_u32 v83, v82, s18 // (op 231) 


buffer_load_dword v31, v83, s[80:83], 0 offen // (op 231) Load value

s_barrier  // (op 4359) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2561) 
v_mov_b64 v[90:91], s[18:19] // (op 2561) convert


v_mov_b64 v[92:93], v[24:25] // (op 2561) 
s_lshl_b64 s[18:19], s[30:31], 8 // (op 2561) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2561) 

v_mov_b32 v83, s28 // (op 2561) 
v_mov_b32 v94, s29 // (op 2561) 
v_add_co_u32 v96, s[18:19], v92, v83 // (op 2561) least significant half
v_addc_co_u32 v97, s[18:19], v93, v94, s[18:19] // (op 2561) most significant half




s_add_u32 s18, 16384, s15 // (op 2561) 
s_nop 0
s_mov_b32 m0, s18 // (op 2561) reg expression
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2561) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v90 // (op 2561) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 2561) most significant half

s_add_u32 m0, m0, 2048 // (op 2561) 
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2561) Load value direct to lds





v_mov_b32 v83, v28 // (op 230) 
s_lshl_b32 s18, s38, 4 // (op 230) 
v_add_u32 v90, v83, s18 // (op 230) 


buffer_load_ubyte v32, v90, s[76:79], 0 offen offset: 0 // (op 230) Load value

s_barrier  // (op 4462) 



s_waitcnt vmcnt(15)
ds_write_b32 v87, v20 offset:41600 // (op 600) Store local data 
s_barrier  // (op 4456) 



s_waitcnt vmcnt(12)
ds_write_b8 v86, v29 offset:42112 // (op 599) Store local data 
s_barrier  // (op 4442) 



ds_read_b64_tr_b8 v[34:35], v15 offset:20480 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v15 offset:20992 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v15 offset:24576 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v15 offset:25088 // (op 583) Transpose load local data 



ds_read_b64_tr_b8 v[94:95], v15 offset:28672 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v15 offset:29184 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v15 offset:32768 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v15 offset:33280 // (op 592) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4459) 


ds_read_u8 v10, v14 offset:41600 // (op 587) Load local data 


ds_read_u8 v90, v14 offset:41856 // (op 596) Load local data 



ds_read_b64_tr_b8 v[50:51], v15 offset:20512 // (op 766) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v15 offset:21024 // (op 766) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v15 offset:24608 // (op 766) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v15 offset:25120 // (op 766) Transpose load local data 



ds_read_b64_tr_b8 v[102:103], v15 offset:28704 // (op 757) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v15 offset:29216 // (op 757) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v15 offset:32800 // (op 757) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v15 offset:33312 // (op 757) Transpose load local data 


ds_read_u8 v33, v14 offset:41632 // (op 770) Load local data 


ds_read_u8 v91, v14 offset:41888 // (op 761) Load local data 
s_barrier  // (op 4433) 
s_barrier  // (op 4599) 



ds_read_b64_tr_b8 v[66:67], v5 offset:36864 // (op 581) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v5 offset:36992 // (op 581) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v5 offset:37888 // (op 581) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v5 offset:38016 // (op 581) Transpose load local data 



ds_read_b64_tr_b8 v[110:111], v5 offset:38912 // (op 590) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v5 offset:39040 // (op 590) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v5 offset:39936 // (op 590) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v5 offset:40064 // (op 590) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4452) 


ds_read_u8 v22, v8 offset:42112 // (op 585) Load local data 


ds_read_u8 v23, v8 offset:42176 // (op 594) Load local data 
s_barrier  // (op 4563) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v22, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 588) 
s_barrier  // (op 4449) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v22, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 753) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 2589) 
v_mov_b64 v[118:119], s[18:19] // (op 2589) convert


s_lshl_b64 s[18:19], s[34:35], 8 // (op 2589) 
s_mul_i32 s29, 3, s19 // (op 2589) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 2589) least significant
s_mul_hi_u32 s48, 3, s18 // (op 2589) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 2589) most significant: sum


v_mov_b32 v120, s28 // (op 2589) 
v_mov_b32 v121, s29 // (op 2589) 
v_add_co_u32 v122, s[18:19], v88, v120 // (op 2589) least significant half
v_addc_co_u32 v123, s[18:19], v89, v121, s[18:19] // (op 2589) most significant half




s_add_u32 s18, 20480, s8 // (op 2589) 
s_nop 0
s_mov_b32 m0, s18 // (op 2589) reg expression
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:117], v[94:101], a[0:3], v23, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 597) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds


v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:117], v[102:109], a[4:7], v23, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 762) 
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half




s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2589) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2589) most significant half

s_add_u32 m0, m0, 2048 // (op 2589) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2589) Load value direct to lds






s_lshl_b32 s18, s44, 3 // (op 603) 
s_mul_i32 s19, 3, s18 // (op 603) 

v_add_u32 v88, v82, s19 // (op 603) 


buffer_load_dword v20, v88, s[80:83], 0 offen // (op 603) Load value

s_lshl_b64 s[18:19], s[34:35], 8 // (op 2850) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 2850) 

s_barrier  // (op 4439) 
s_lshl_b32 s18, s44, 4 // (op 3106) 

v_add_u32 v30, v30, s18 // (op 3106) 

v_mov_b32 v82, s28 // (op 2850) 
v_mov_b32 v88, s29 // (op 2850) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 2584) 
v_mov_b64 v[90:91], s[18:19] // (op 2584) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 2584) 
s_mul_i32 s49, 3, s19 // (op 2584) most significant: low of low * high
s_mul_i32 s48, 3, s18 // (op 2584) least significant
s_mul_hi_u32 s52, 3, s18 // (op 2584) most significant: high of low * low
s_add_u32 s49, s49, s52 // (op 2584) most significant: sum


v_mov_b32 v89, s48 // (op 2584) 
v_mov_b32 v94, s49 // (op 2584) 
v_add_co_u32 v26, s[18:19], v26, v82 // (op 2850) least significant half
v_addc_co_u32 v27, s[18:19], v27, v88, s[18:19] // (op 2850) most significant half




v_add_co_u32 v96, s[18:19], v92, v89 // (op 2584) least significant half
v_addc_co_u32 v97, s[18:19], v93, v94, s[18:19] // (op 2584) most significant half




s_add_u32 s18, 36864, s7 // (op 2584) 
s_nop 0
s_mov_b32 m0, s18 // (op 2584) reg expression
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2584) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v90 // (op 2584) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 2584) most significant half

s_add_u32 m0, m0, 2048 // (op 2584) 
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2584) Load value direct to lds






s_lshl_b32 s18, s38, 3 // (op 602) 
s_mul_i32 s19, 3, s18 // (op 602) 

v_add_u32 v82, v83, s19 // (op 602) 


buffer_load_ubyte v29, v82, s[76:79], 0 offen offset: 0 // (op 602) Load value

s_lshl_b64 s[18:19], s[30:31], 8 // (op 2718) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 2718) 

s_barrier  // (op 4419) 
s_lshl_b32 s18, s38, 4 // (op 2978) 



v_add_u32 v28, v28, s18 // (op 2978) 

v_mov_b32 v82, s28 // (op 2718) 
v_mov_b32 v83, s29 // (op 2718) 
v_add_co_u32 v24, s[18:19], v24, v82 // (op 2718) least significant half
v_addc_co_u32 v25, s[18:19], v25, v83, s[18:19] // (op 2718) most significant half




s_waitcnt vmcnt(15)
ds_write_b32 v85, v31 offset:40960 // (op 228) Store local data 
s_barrier  // (op 4398) 



s_waitcnt vmcnt(12)
ds_write_b8 v84, v32 offset:41472 // (op 227) Store local data 
s_barrier  // (op 4384) 



ds_read_b64_tr_b8 v[34:35], v13 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v13 offset:512 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v13 offset:4096 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:4608 // (op 211) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v13 offset:8192 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:8704 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:12288 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:12800 // (op 220) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4426) 


ds_read_u8 v10, v12 offset:40960 // (op 215) Load local data 


ds_read_u8 v16, v12 offset:41216 // (op 224) Load local data 



ds_read_b64_tr_b8 v[50:51], v13 offset:32 // (op 738) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:544 // (op 738) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:4128 // (op 738) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v13 offset:4640 // (op 738) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v13 offset:8224 // (op 747) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v13 offset:8736 // (op 747) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v13 offset:12320 // (op 747) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v13 offset:12832 // (op 747) Transpose load local data 


ds_read_u8 v33, v12 offset:40992 // (op 742) Load local data 


ds_read_u8 v21, v12 offset:41248 // (op 751) Load local data 
s_barrier  // (op 4365) 



ds_read_b64_tr_b8 v[66:67], v3 offset:16384 // (op 209) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:16512 // (op 209) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:17408 // (op 209) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v3 offset:17536 // (op 209) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v3 offset:18432 // (op 218) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v3 offset:18560 // (op 218) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v3 offset:19456 // (op 218) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v3 offset:19584 // (op 218) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4405) 


ds_read_u8 v22, v6 offset:41472 // (op 213) Load local data 


ds_read_u8 v23, v6 offset:41536 // (op 222) Load local data 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 695) 
s_mov_b32 s28, s18 // (op 695) convert

s_and_b32 s29, s28, 4294967294 // (op 695) 

s_cmp_lt_u32 s1, s29 // (op 695) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_695 // (op 695) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_695 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_695:



















s_barrier  // (op 4581) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2656) 
s_and_b32 s15, s18, 4294967294 // (op 2656) 
s_add_u32 s6, s0, s15 // (op 2656) 

s_mul_i32 s15, s16, s50 // (op 2656) 
s_add_u32 s15, s15, s6 // (op 2656) 

s_add_u32 s6, s15, s9 // (op 2656) 

s_mul_hi_u32 s15, s6, s13 // (op 2656) 
s_sub_u32 s7, s6, s15 // (op 2656) 
s_lshr_b32 s8, s7, 1 // (op 2656) 

s_add_u32 s7, s8, s15 // (op 2656) 


s_lshr_b32 s8, s7, s11 // (op 2656) 

s_and_b32 s7, s11, 2147483648 // (op 2656) 
s_cmp_eq_i32 s7, 0 // (op 2656) 
s_mov_b32 s15, scc // (op 2656) 

s_cmp_eq_i32 s15, 0 // (op 2656) 
s_cselect_b32 s7, s6, s8 // (op 2656) 


s_mul_hi_u32 s8, s7, s12 // (op 2656) 
s_sub_u32 s15, s7, s8 // (op 2656) 
s_lshr_b32 s28, s15, 1 // (op 2656) 

s_add_u32 s15, s28, s8 // (op 2656) 


s_lshr_b32 s8, s15, s10 // (op 2656) 

s_and_b32 s15, s10, 2147483648 // (op 2656) 
s_cmp_eq_i32 s15, 0 // (op 2656) 
s_mov_b32 s28, scc // (op 2656) 

s_cmp_eq_i32 s28, 0 // (op 2656) 
s_cselect_b32 s15, s7, s8 // (op 2656) 


s_lshl_b32 s8, s15, 4 // (op 2656) 

s_mov_b32 s29, 0 // (op 2656) convert
s_mov_b32 s28, s8 // (op 2656) convert

s_mul_i32 s8, s7, s18 // (op 2656) 


s_sub_u32 s7, s6, s8 // (op 2656) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2656) 

v_mul_lo_u32 v25, v10, s31 // (op 2656) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2656) most significant: high of low * low
v_mul_lo_u32 v24, v10, s30 // (op 2656) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 2656) most significant: sum


v_mov_b32 v10, s28 // (op 2656) 
v_mov_b32 v16, s29 // (op 2656) 
v_add_co_u32 v26, s[18:19], v10, v24 // (op 2656) least significant half
v_addc_co_u32 v27, s[18:19], v16, v25, s[18:19] // (op 2656) most significant half





s_mul_i32 s19, s30, s3 // (op 2658) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2658) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2658) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2658) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2658) most significant: sum
s_add_u32 s19, s19, s6 // (op 2658) most significant: sum


s_mov_b32 s68, s32 // (op 2658) reg expression
s_mov_b32 s69, s33 // (op 2658) reg expression
s_mov_b32 s70, s18 // (op 2658) reg expression
s_mov_b32 s71, 131072 // (op 2658) call()

v_and_b32 v10, -4, v1 // (op 2794) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2794) 
v_sub_u32 v16, v1, v10 // (op 2794) 

s_and_b32 s15, s18, 4294967294 // (op 2794) 
v_lshlrev_b32 v10, 4, v16 // (op 2794) 

s_add_u32 s6, s0, s15 // (op 2794) 

v_lshrrev_b32 v16, 2, v1 // (op 2794) 
s_mul_i32 s15, s16, s50 // (op 2794) 
s_add_u32 s15, s15, s6 // (op 2794) 

s_add_u32 s6, s15, s9 // (op 2794) 

s_mul_hi_u32 s15, s6, s13 // (op 2794) 
s_sub_u32 s7, s6, s15 // (op 2794) 
s_lshr_b32 s8, s7, 1 // (op 2794) 

s_add_u32 s7, s8, s15 // (op 2794) 


s_lshr_b32 s8, s7, s11 // (op 2794) 

s_and_b32 s7, s11, 2147483648 // (op 2794) 
s_cmp_eq_i32 s7, 0 // (op 2794) 
s_mov_b32 s15, scc // (op 2794) 

s_cmp_eq_i32 s15, 0 // (op 2794) 
s_cselect_b32 s7, s6, s8 // (op 2794) 


s_mul_hi_u32 s8, s7, s12 // (op 2794) 
s_sub_u32 s15, s7, s8 // (op 2794) 
s_lshr_b32 s28, s15, 1 // (op 2794) 

s_add_u32 s15, s28, s8 // (op 2794) 


s_lshr_b32 s8, s15, s10 // (op 2794) 

s_and_b32 s15, s10, 2147483648 // (op 2794) 
s_cmp_eq_i32 s15, 0 // (op 2794) 
s_mov_b32 s28, scc // (op 2794) 

s_cmp_eq_i32 s28, 0 // (op 2794) 
s_cselect_b32 s15, s7, s8 // (op 2794) 


s_mul_i32 s8, s15, s14 // (op 2794) 

s_sub_u32 s15, s7, s8 // (op 2794) 

s_mul_i32 s8, s7, s18 // (op 2794) 


v_lshl_add_u32 v28, s15, 6, v10 // (op 2794) 


s_sub_u32 s7, s6, s8 // (op 2794) 


v_mov_b32 v25, 0 // (op 2794) convert
v_mov_b32 v24, v28 // (op 2794) convert

v_lshl_add_u32 v10, s7, 8, v16 // (op 2794) 


v_mul_lo_u32 v31, v10, s35 // (op 2794) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2794) most significant: high of low * low
v_mul_lo_u32 v30, v10, s34 // (op 2794) least significant: low of low * low
v_add_u32 v31, v31, v16 // (op 2794) most significant: sum

v_add_co_u32 v30, s[18:19], v30, v24 // (op 2794) least significant half
v_addc_co_u32 v31, s[18:19], v31, v25, s[18:19] // (op 2794) most significant half



s_mul_i32 s19, s34, s3 // (op 2796) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2796) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2796) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2796) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2796) most significant: sum
s_add_u32 s19, s19, s6 // (op 2796) most significant: sum


s_mov_b32 s72, s36 // (op 2796) reg expression
s_mov_b32 s73, s37 // (op 2796) reg expression
s_mov_b32 s74, s18 // (op 2796) reg expression
s_mov_b32 s75, 131072 // (op 2796) call()

v_and_b32 v10, -16, v1 // (op 2922) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2922) 
v_sub_u32 v16, v1, v10 // (op 2922) 

s_and_b32 s15, s18, 4294967294 // (op 2922) 
v_lshrrev_b32 v10, 4, v1 // (op 2922) 
s_add_u32 s6, s0, s15 // (op 2922) 

s_mul_i32 s15, s16, s50 // (op 2922) 
s_add_u32 s15, s15, s6 // (op 2922) 

s_add_u32 s6, s15, s9 // (op 2922) 

s_mul_hi_u32 s15, s6, s13 // (op 2922) 
s_sub_u32 s7, s6, s15 // (op 2922) 
s_lshr_b32 s8, s7, 1 // (op 2922) 

s_add_u32 s7, s8, s15 // (op 2922) 


s_lshr_b32 s8, s7, s11 // (op 2922) 

s_and_b32 s7, s11, 2147483648 // (op 2922) 
s_cmp_eq_i32 s7, 0 // (op 2922) 
s_mov_b32 s15, scc // (op 2922) 

s_cmp_eq_i32 s15, 0 // (op 2922) 
s_cselect_b32 s7, s6, s8 // (op 2922) 


s_mul_hi_u32 s8, s7, s12 // (op 2922) 
s_sub_u32 s15, s7, s8 // (op 2922) 
s_lshr_b32 s28, s15, 1 // (op 2922) 

s_add_u32 s15, s28, s8 // (op 2922) 


s_lshr_b32 s8, s15, s10 // (op 2922) 

s_and_b32 s15, s10, 2147483648 // (op 2922) 
s_cmp_eq_i32 s15, 0 // (op 2922) 
s_mov_b32 s28, scc // (op 2922) 

s_cmp_eq_i32 s28, 0 // (op 2922) 
s_cselect_b32 s15, s7, s8 // (op 2922) 


s_mul_i32 s8, s7, s18 // (op 2922) 


v_lshl_add_u32 v28, s15, 4, v16 // (op 2922) 


s_sub_u32 s7, s6, s8 // (op 2922) 


v_lshl_add_u32 v16, s7, 3, v10 // (op 2922) 


v_mul_lo_u32 v10, v16, s38 // (op 2922) 

v_add_u32 v16, v28, v10 // (op 2922) 


s_mul_i32 s19, s38, s43 // (op 2924) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2924) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2924) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2924) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2924) most significant: sum
s_add_u32 s19, s19, s6 // (op 2924) most significant: sum


s_mov_b32 s76, s40 // (op 2924) reg expression
s_mov_b32 s77, s41 // (op 2924) reg expression
s_mov_b32 s78, s18 // (op 2924) reg expression
s_mov_b32 s79, 131072 // (op 2924) call()

v_and_b32 v10, -16, v1 // (op 3050) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 3050) 
v_sub_u32 v28, v1, v10 // (op 3050) 

s_and_b32 s15, s18, 4294967294 // (op 3050) 
v_lshlrev_b32 v10, 2, v28 // (op 3050) 

s_add_u32 s6, s0, s15 // (op 3050) 

v_lshrrev_b32 v28, 4, v1 // (op 3050) 
s_mul_i32 s15, s16, s50 // (op 3050) 
s_add_u32 s15, s15, s6 // (op 3050) 

s_add_u32 s6, s15, s9 // (op 3050) 

s_mul_hi_u32 s15, s6, s13 // (op 3050) 
s_sub_u32 s7, s6, s15 // (op 3050) 
s_lshr_b32 s8, s7, 1 // (op 3050) 

s_add_u32 s7, s8, s15 // (op 3050) 


s_lshr_b32 s8, s7, s11 // (op 3050) 

s_and_b32 s7, s11, 2147483648 // (op 3050) 
s_cmp_eq_i32 s7, 0 // (op 3050) 
s_mov_b32 s15, scc // (op 3050) 

s_cmp_eq_i32 s15, 0 // (op 3050) 
s_cselect_b32 s7, s6, s8 // (op 3050) 


s_mul_hi_u32 s8, s7, s12 // (op 3050) 
s_sub_u32 s15, s7, s8 // (op 3050) 
s_lshr_b32 s28, s15, 1 // (op 3050) 

s_add_u32 s15, s28, s8 // (op 3050) 


s_lshr_b32 s8, s15, s10 // (op 3050) 

s_and_b32 s15, s10, 2147483648 // (op 3050) 
s_cmp_eq_i32 s15, 0 // (op 3050) 
s_mov_b32 s28, scc // (op 3050) 

s_cmp_eq_i32 s28, 0 // (op 3050) 
s_cselect_b32 s15, s7, s8 // (op 3050) 


s_mul_i32 s8, s15, s14 // (op 3050) 

s_sub_u32 s15, s7, s8 // (op 3050) 

s_mul_i32 s8, s7, s18 // (op 3050) 


v_lshl_add_u32 v32, s15, 6, v10 // (op 3050) 


s_sub_u32 s7, s6, s8 // (op 3050) 


v_lshl_add_u32 v10, s7, 3, v28 // (op 3050) 


v_mul_lo_u32 v28, v10, s44 // (op 3050) 
v_add_u32 v28, v28, v32 // (op 3050) 


s_mul_i32 s19, s44, s43 // (op 3052) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3052) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3052) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3052) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3052) most significant: sum
s_add_u32 s19, s19, s6 // (op 3052) most significant: sum


s_mov_b32 s80, s46 // (op 3052) reg expression
s_mov_b32 s81, s47 // (op 3052) reg expression
s_mov_b32 s82, s18 // (op 3052) reg expression
s_mov_b32 s83, 131072 // (op 3052) call()

v_lshlrev_b32 v10, 4, v1 // (op 3557) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3557) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 3622) 
v_sub_u32 v32, v1, v10 // (op 3622) 

v_lshlrev_b32 v10, 4, v32 // (op 3622) 

v_lshrrev_b32 v32, 2, v1 // (op 3622) 
v_lshl_add_u32 v24, v32, 6, v10 // (op 3622) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 3622) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -16, v1 // (op 3697) 
v_sub_u32 v32, v1, v10 // (op 3697) 

v_and_b32 v10, 4294967280, v1 // (op 3697) 
v_add_u32 v24, v10, v32 // (op 3697) 


v_and_b32 v10, -16, v1 // (op 3763) 
v_sub_u32 v25, v1, v10 // (op 3763) 

v_lshlrev_b32 v10, 2, v25 // (op 3763) 

v_lshrrev_b32 v25, 4, v1 // (op 3763) 
v_lshl_add_u32 v32, v25, 6, v10 // (op 3763) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 701) 
s_and_b32 s7, s18, 4294967294 // (op 701) 

s_ashr_i64 s[18:19], s[2:3], 8 // (op 700) 
s_mov_b32 s8, s18 // (op 700) convert

s_cmp_lt_u32 s7, s8 // (op 700) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_700 // (op 700) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_700 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_700:
s_barrier  // (op 4362) 
s_add_u32 s7, s7, 1 // (op 702) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2571) 
v_mov_b64 v[20:21], s[18:19] // (op 2571) convert


v_mov_b64 v[22:23], v[26:27] // (op 2571) 
v_mov_b64 v[34:35], v[22:23] // (op 2571) reg expression
s_add_u32 s8, 16384, s15 // (op 2571) 
s_mov_b32 m0, s8 // (op 2571) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2571) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2571) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2571) most significant half

s_add_u32 m0, m0, 2048 // (op 2571) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2571) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 8 // (op 2646) 
v_mov_b32 v10, s18 // (op 2646) 
v_mov_b32 v25, s19 // (op 2646) 
v_add_co_u32 v26, s[28:29], v26, v10 // (op 2646) least significant half
v_addc_co_u32 v27, s[28:29], v27, v25, s[28:29] // (op 2646) most significant half




s_barrier  // (op 4381) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2577) 
v_mov_b64 v[20:21], s[18:19] // (op 2577) convert


v_mov_b64 v[22:23], v[30:31] // (op 2577) 
v_mov_b64 v[34:35], v[22:23] // (op 2577) reg expression
s_mov_b32 m0, s6 // (op 2577) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2577) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2577) Load value direct to lds



s_lshl_b64 s[18:19], s[34:35], 8 // (op 2786) 
v_mov_b32 v10, s18 // (op 2786) 
v_mov_b32 v25, s19 // (op 2786) 
v_add_co_u32 v30, s[28:29], v30, v10 // (op 2786) least significant half
v_addc_co_u32 v31, s[28:29], v31, v25, s[28:29] // (op 2786) most significant half






v_mov_b32 v10, v16 // (op 518) 
v_mov_b32 v25, v10 // (op 518) reg expression

buffer_load_ubyte v29, v25, s[76:79], 0 offen offset: 0 // (op 518) Load value


s_lshl_b32 s8, s38, 3 // (op 2914) 
v_add_u32 v16, v16, s8 // (op 2914) 



s_barrier  // (op 4401) 
v_mov_b32 v10, v28 // (op 519) 



v_mov_b32 v25, v10 // (op 519) reg expression

buffer_load_dword v20, v25, s[80:83], 0 offen // (op 519) Load value


s_lshl_b32 s8, s44, 3 // (op 3042) 
v_add_u32 v28, v28, s8 // (op 3042) 

s_waitcnt vmcnt(1)
ds_write_b8 v24, v29 offset:41472 // (op 515) Store local data 

s_barrier  // (op 4368) 



s_barrier  // (op 4422) 
ds_read_b64_tr_b8 v[34:35], v3 offset:16384 // (op 497) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v3 offset:16512 // (op 497) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:17408 // (op 497) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:17536 // (op 497) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v32, v20 offset:40960 // (op 516) Store local data 

s_barrier  // (op 4388) 



ds_read_b64_tr_b8 v[42:43], v13 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:512 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:4096 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:4608 // (op 499) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4409) 


ds_read_u8 v10, v6 offset:41472 // (op 501) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4429) 


ds_read_u8 v25, v12 offset:40960 // (op 503) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v10, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 504) 





ds_read_b64_tr_b8 v[42:43], v3 offset:18432 // (op 506) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:18560 // (op 506) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:19456 // (op 506) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:19584 // (op 506) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v13 offset:8192 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:8704 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:12288 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v13 offset:12800 // (op 508) Transpose load local data 


ds_read_u8 v25, v6 offset:41536 // (op 510) Load local data 


ds_read_u8 v29, v12 offset:41216 // (op 512) Load local data 



ds_read_b64_tr_b8 v[58:59], v13 offset:32 // (op 708) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v13 offset:544 // (op 708) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v13 offset:4128 // (op 708) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v13 offset:4640 // (op 708) Transpose load local data 


ds_read_u8 v33, v12 offset:40992 // (op 712) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[58:65], a[4:7], v10, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 713) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:57], a[0:3], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 513) 
ds_read_b64_tr_b8 v[34:35], v13 offset:8224 // (op 717) Transpose load local data 


ds_read_b64_tr_b8 v[36:37], v13 offset:8736 // (op 717) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v13 offset:12320 // (op 717) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:12832 // (op 717) Transpose load local data 


ds_read_u8 v10, v12 offset:41248 // (op 721) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v25, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 722) 




s_ashr_i64 s[18:19], s[2:3], 8 // (op 700) 
s_mov_b32 s8, s18 // (op 700) convert

s_cmp_lt_u32 s7, s8 // (op 700) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_700 // (op 700) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_700 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_700:









s_barrier  // (op 4585) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_274 // (op 274) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_274
v_accvgpr_read v20, a0 // (op 282) 
v_accvgpr_read v21, a1 // (op 282) 
v_accvgpr_read v22, a2 // (op 282) 
v_accvgpr_read v23, a3 // (op 282) 
v_mul_f32 v24, v19, v20 // (op 282) 
v_mul_f32 v25, v19, v21 // (op 282) 
v_mul_f32 v26, v19, v22 // (op 282) 
v_mul_f32 v27, v19, v23 // (op 282) 



v_mov_b32 v0, v9 // (op 284) reg expression

buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen // (op 284) Store value


v_accvgpr_read v20, a4 // (op 1006) 
v_accvgpr_read v21, a5 // (op 1006) 
v_accvgpr_read v22, a6 // (op 1006) 
v_accvgpr_read v23, a7 // (op 1006) 
v_mul_f32 v24, v19, v20 // (op 1006) 
v_mul_f32 v25, v19, v21 // (op 1006) 
v_mul_f32 v26, v19, v22 // (op 1006) 
v_mul_f32 v27, v19, v23 // (op 1006) 



s_lshl_b32 s15, s24, 7 // (op 1008) 
v_add_u32 v0, v9, s15 // (op 1008) 


buffer_store_dwordx4 v[24:27], v0, s[60:63], 0 offen // (op 1008) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_274 // (op 274) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_274
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_274:


v_mov_b32 v0, v7 // (op 297) reg expression

buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen // (op 297) Load value

v_accvgpr_read v24, a0 // (op 294) 
v_accvgpr_read v25, a1 // (op 294) 
v_accvgpr_read v26, a2 // (op 294) 
v_accvgpr_read v27, a3 // (op 294) 
v_mul_f32 v28, v19, v24 // (op 294) 
v_mul_f32 v29, v19, v25 // (op 294) 
v_mul_f32 v30, v19, v26 // (op 294) 
v_mul_f32 v31, v19, v27 // (op 294) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v18, v20 // (op 298) 
v_mul_f32 v25, v18, v21 // (op 298) 
v_mul_f32 v26, v18, v22 // (op 298) 
v_mul_f32 v27, v18, v23 // (op 298) 

v_add_f32 v20, v24, v28 // (op 295) 
v_add_f32 v21, v25, v29 // (op 295) 
v_add_f32 v22, v26, v30 // (op 295) 
v_add_f32 v23, v27, v31 // (op 295) 




v_mov_b32 v0, v9 // (op 296) reg expression

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 296) Store value




s_lshl_b32 s15, s4, 7 // (op 1030) 
v_add_u32 v0, v7, s15 // (op 1030) 


buffer_load_dwordx4 v[20:23], v0, s[56:59], 0 offen // (op 1030) Load value

v_accvgpr_read v24, a4 // (op 1027) 
v_accvgpr_read v25, a5 // (op 1027) 
v_accvgpr_read v26, a6 // (op 1027) 
v_accvgpr_read v27, a7 // (op 1027) 
v_mul_f32 v28, v19, v24 // (op 1027) 
v_mul_f32 v29, v19, v25 // (op 1027) 
v_mul_f32 v30, v19, v26 // (op 1027) 
v_mul_f32 v31, v19, v27 // (op 1027) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v18, v20 // (op 1031) 
v_mul_f32 v25, v18, v21 // (op 1031) 
v_mul_f32 v26, v18, v22 // (op 1031) 
v_mul_f32 v27, v18, v23 // (op 1031) 

v_add_f32 v20, v24, v28 // (op 1028) 
v_add_f32 v21, v25, v29 // (op 1028) 
v_add_f32 v22, v26, v30 // (op 1028) 
v_add_f32 v23, v27, v31 // (op 1028) 




s_lshl_b32 s15, s24, 7 // (op 1029) 
v_add_u32 v0, v9, s15 // (op 1029) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 1029) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_274:
s_add_u32 s0, s0, s1 // (op 483) 


s_mul_i32 s15, s50, s16 // (op 481) 
s_cmp_gt_u32 s51, s15 // (op 481) 
s_mov_b32 s6, scc // (op 481) 
s_sub_u32 s7, s51, s15 // (op 481) 

s_cmp_eq_i32 s6, 0 // (op 481) 
s_cselect_b32 s8, 0, s7 // (op 481) 


s_cmp_lt_u32 s50, s8 // (op 481) 
s_mov_b32 s15, scc // (op 481) 
s_cmp_eq_i32 s15, 0 // (op 481) 
s_cselect_b32 s6, s8, s50 // (op 481) 


s_cmp_lt_u32 s0, s6 // (op 481) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_481 // (op 481) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_481 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_481:











































s_endpgm  // End of RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2, .LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_end-RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
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
  - .name: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2.kd
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

