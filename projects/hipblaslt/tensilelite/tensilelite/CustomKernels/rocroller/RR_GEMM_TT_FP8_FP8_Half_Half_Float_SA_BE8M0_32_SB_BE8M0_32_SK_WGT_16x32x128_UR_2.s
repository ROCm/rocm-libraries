// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2,@function
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1676) 
v_and_b32 v3, 15, v1 // (op 1676) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1676) 


v_and_b32 v0, 48, v1 // (op 1676) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1676) 


v_and_b32 v0, 48, v1 // (op 1741) 
v_and_b32 v4, 63, v1 // (op 1741) 
v_sub_u32 v5, v4, v0 // (op 1741) 

v_lshrrev_b32 v4, 1, v5 // (op 1741) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 1741) 


v_and_b32 v0, 1, v5 // (op 1741) 

v_lshlrev_b32 v4, 3, v0 // (op 1741) 

v_lshrrev_b32 v0, 6, v1 // (op 1741) 
v_and_b32 v5, 1, v0 // (op 1741) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 1741) 


v_add_u32 v4, v6, v0 // (op 1741) 


v_lshrrev_b32 v0, 6, v1 // (op 1807) 
v_and_b32 v5, 1, v0 // (op 1807) 

v_and_b32 v0, 15, v1 // (op 1807) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 1807) 


v_and_b32 v0, 63, v1 // (op 1807) 
v_lshrrev_b32 v5, 4, v0 // (op 1807) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 1807) 


v_lshrrev_b32 v5, 7, v1 // (op 1942) 
v_and_b32 v6, 15, v1 // (op 1942) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1942) 


v_and_b32 v5, 48, v1 // (op 1942) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 1942) 


v_and_b32 v5, 48, v1 // (op 2007) 
v_and_b32 v7, 63, v1 // (op 2007) 
v_sub_u32 v8, v7, v5 // (op 2007) 

v_lshrrev_b32 v7, 1, v8 // (op 2007) 
v_add_lshl_u32 v9, v5, v7, 5 // (op 2007) 


v_and_b32 v5, 1, v8 // (op 2007) 

v_lshlrev_b32 v7, 3, v5 // (op 2007) 

v_lshrrev_b32 v5, 6, v1 // (op 2007) 
v_and_b32 v8, 1, v5 // (op 2007) 

v_lshl_add_u32 v5, v8, 4, v7 // (op 2007) 


v_add_u32 v7, v9, v5 // (op 2007) 


v_lshrrev_b32 v5, 6, v1 // (op 2073) 
v_and_b32 v8, 1, v5 // (op 2073) 

v_and_b32 v5, 15, v1 // (op 2073) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 2073) 


v_and_b32 v5, 63, v1 // (op 2073) 
v_lshrrev_b32 v8, 4, v5 // (op 2073) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 2073) 


v_lshrrev_b32 v8, 7, v1 // (op 2159) 
v_and_b32 v9, 15, v1 // (op 2159) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2159) 


v_and_b32 v8, 48, v1 // (op 2159) 
v_lshl_add_u32 v9, v10, 7, v8 // (op 2159) 


v_and_b32 v8, 48, v1 // (op 2198) 
v_and_b32 v10, 63, v1 // (op 2198) 
v_sub_u32 v11, v10, v8 // (op 2198) 

v_lshrrev_b32 v10, 1, v11 // (op 2198) 
v_add_lshl_u32 v12, v8, v10, 5 // (op 2198) 


v_and_b32 v8, 1, v11 // (op 2198) 

v_lshlrev_b32 v10, 3, v8 // (op 2198) 

v_lshrrev_b32 v8, 6, v1 // (op 2198) 
v_and_b32 v11, 1, v8 // (op 2198) 

v_lshl_add_u32 v8, v11, 4, v10 // (op 2198) 


v_add_u32 v10, v12, v8 // (op 2198) 


v_lshrrev_b32 v8, 6, v1 // (op 2222) 
v_and_b32 v11, 1, v8 // (op 2222) 

v_and_b32 v8, 15, v1 // (op 2222) 
v_lshl_add_u32 v12, v11, 4, v8 // (op 2222) 


v_and_b32 v8, 63, v1 // (op 2222) 
v_lshrrev_b32 v11, 4, v8 // (op 2222) 

v_lshl_add_u32 v8, v11, 5, v12 // (op 2222) 


v_lshrrev_b32 v11, 7, v1 // (op 2261) 
v_and_b32 v12, 15, v1 // (op 2261) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 2261) 


v_and_b32 v11, 48, v1 // (op 2261) 
v_lshl_add_u32 v12, v13, 7, v11 // (op 2261) 


v_and_b32 v11, 48, v1 // (op 2300) 
v_and_b32 v13, 63, v1 // (op 2300) 
v_sub_u32 v14, v13, v11 // (op 2300) 

v_lshrrev_b32 v13, 1, v14 // (op 2300) 
v_add_lshl_u32 v15, v11, v13, 5 // (op 2300) 


v_and_b32 v11, 1, v14 // (op 2300) 

v_lshlrev_b32 v13, 3, v11 // (op 2300) 

v_lshrrev_b32 v11, 6, v1 // (op 2300) 
v_and_b32 v14, 1, v11 // (op 2300) 

v_lshl_add_u32 v11, v14, 4, v13 // (op 2300) 


v_add_u32 v13, v15, v11 // (op 2300) 


v_lshrrev_b32 v11, 6, v1 // (op 2324) 
v_and_b32 v14, 1, v11 // (op 2324) 

v_and_b32 v11, 15, v1 // (op 2324) 
v_lshl_add_u32 v15, v14, 4, v11 // (op 2324) 


v_and_b32 v11, 63, v1 // (op 2324) 
v_lshrrev_b32 v14, 4, v11 // (op 2324) 

v_lshl_add_u32 v11, v14, 5, v15 // (op 2324) 


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
s_mul_i32 s54, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_sub_u32 s56, s9, s54 // (op 278) 

s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s54, 0, s56 // (op 278) 


s_cmp_lt_u32 s8, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s56, s54, s8 // (op 278) 


s_cmp_lt_u32 s0, s56 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 // (op 278) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278:
v_lshrrev_b32 v16, 7, v1 // (op 1602) 
s_mul_i32 s54, s16, s8 // (op 1602) 
s_add_u32 s54, s54, s0 // (op 1602) 
v_and_b32 v17, 63, v1 // (op 1602) 
s_mul_hi_u32 s55, s54, s13 // (op 1602) 
v_lshrrev_b32 v18, 4, v17 // (op 1602) 
s_sub_u32 s56, s54, s55 // (op 1602) 
v_lshlrev_b32 v19, 2, v18 // (op 1602) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 1602) 


s_lshr_b32 s57, s56, 1 // (op 1602) 

s_add_u32 s56, s57, s55 // (op 1602) 


v_lshrrev_b32 v16, 2, v17 // (op 1602) 

s_lshr_b32 s55, s56, s11 // (op 1602) 

v_and_b32 v17, 3, v16 // (op 1602) 

s_and_b32 s56, s11, 2147483648 // (op 1602) 
v_and_b32 v16, 3, v1 // (op 1602) 
s_cmp_eq_i32 s56, 0 // (op 1602) 
s_mov_b32 s57, scc // (op 1602) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 1602) 


s_cmp_eq_i32 s57, 0 // (op 1602) 
s_cselect_b32 s56, s54, s55 // (op 1602) 



v_lshrrev_b32 v16, 6, v1 // (op 1602) 
s_mul_hi_u32 s54, s56, s12 // (op 1602) 
v_and_b32 v17, 1, v16 // (op 1602) 

s_sub_u32 s55, s56, s54 // (op 1602) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 1602) 


s_lshr_b32 s57, s55, 1 // (op 1602) 

s_add_u32 s55, s57, s54 // (op 1602) 


s_lshr_b32 s54, s55, s10 // (op 1602) 

s_and_b32 s55, s10, 2147483648 // (op 1602) 
s_cmp_eq_i32 s55, 0 // (op 1602) 
s_mov_b32 s57, scc // (op 1602) 

s_cmp_eq_i32 s57, 0 // (op 1602) 
s_cselect_b32 s55, s56, s54 // (op 1602) 


s_mul_i32 s54, s55, s14 // (op 1602) 
v_lshl_add_u32 v17, s55, 4, v18 // (op 1602) 


s_sub_u32 s55, s56, s54 // (op 1602) 


v_lshl_add_u32 v18, s55, 5, v16 // (op 1602) 


v_mul_lo_u32 v16, v18, s4 // (op 1602) 

v_add_lshl_u32 v18, v17, v16, 1 // (op 1602) 


s_mul_i32 s55, s4, s23 // (op 1604) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 1604) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 1604) least significant
s_mul_hi_u32 s57, s4, s22 // (op 1604) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 1604) most significant: sum
s_add_u32 s55, s55, s57 // (op 1604) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 1 // (op 1604) 

s_mov_b32 s60, s20 // (op 1604) reg expression
s_mov_b32 s61, s21 // (op 1604) reg expression
s_mov_b32 s62, s56 // (op 1604) reg expression
s_mov_b32 s63, 131072 // (op 1604) call()

v_lshrrev_b32 v16, 7, v1 // (op 1688) 
v_and_b32 v17, 15, v1 // (op 1688) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 1688) 


v_and_b32 v16, 48, v1 // (op 1688) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 1688) 


v_and_b32 v16, 48, v1 // (op 1753) 
v_and_b32 v19, 63, v1 // (op 1753) 
v_sub_u32 v20, v19, v16 // (op 1753) 

v_lshrrev_b32 v19, 1, v20 // (op 1753) 
v_add_lshl_u32 v21, v16, v19, 5 // (op 1753) 


v_and_b32 v16, 1, v20 // (op 1753) 

v_lshlrev_b32 v19, 3, v16 // (op 1753) 

v_lshrrev_b32 v16, 6, v1 // (op 1753) 
v_and_b32 v20, 1, v16 // (op 1753) 

v_lshl_add_u32 v16, v20, 4, v19 // (op 1753) 


v_add_u32 v19, v21, v16 // (op 1753) 


v_lshrrev_b32 v16, 6, v1 // (op 1819) 
v_and_b32 v20, 1, v16 // (op 1819) 

v_and_b32 v16, 15, v1 // (op 1819) 
v_lshl_add_u32 v21, v20, 4, v16 // (op 1819) 


v_and_b32 v16, 63, v1 // (op 1819) 
v_lshrrev_b32 v20, 4, v16 // (op 1819) 

v_lshl_add_u32 v16, v20, 5, v21 // (op 1819) 


s_mul_i32 s6, s8, s16 // (op 283) 
s_add_u32 s6, s6, s0 // (op 283) 
s_mul_hi_u32 s54, s6, s13 // (op 284) 
s_sub_u32 s55, s6, s54 // (op 284) 
s_lshr_b32 s56, s55, 1 // (op 284) 

s_add_u32 s55, s56, s54 // (op 284) 


s_lshr_b32 s54, s55, s11 // (op 284) 

s_and_b32 s55, s11, 2147483648 // (op 284) 
s_cmp_eq_i32 s55, 0 // (op 284) 
s_mov_b32 s56, scc // (op 284) 

s_cmp_eq_i32 s56, 0 // (op 284) 
s_cselect_b32 s55, s6, s54 // (op 284) 


s_ashr_i64 s[56:57], s[2:3], 7 // (op 284) 
s_mul_i32 s54, s55, s56 // (op 284) 


s_sub_u32 s18, s6, s54 // (op 284) 

s_ashr_i64 s[54:55], s[2:3], 7 // (op 285) 
s_cmp_gt_u32 s54, s18 // (op 285) 
s_mov_b32 s56, scc // (op 285) 
s_sub_u32 s57, s54, s18 // (op 285) 

s_cmp_eq_i32 s56, 0 // (op 285) 
s_cselect_b32 s54, 0, s57 // (op 285) 


s_cmp_gt_u32 s8, s0 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_sub_u32 s56, s8, s0 // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s57, 0, s56 // (op 285) 


s_cmp_lt_u32 s54, s57 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s56, s57, s54 // (op 285) 



s_cmp_gt_u32 s9, s6 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_sub_u32 s55, s9, s6 // (op 285) 
s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s57, 0, s55 // (op 285) 


s_cmp_lt_u32 s56, s57 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s7, s57, s56 // (op 285) 



s_add_u32 s54, s18, s7 // (op 286) 
s_sub_u32 s19, s54, 1 // (op 286) 

v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_lshrrev_b32 v20, 3, v1 // (op 1134) 
s_mul_i32 s54, s16, s8 // (op 1134) 
s_add_u32 s54, s54, s0 // (op 1134) 
v_and_b32 v21, -8, v1 // (op 1134) 
s_mul_hi_u32 s55, s54, s13 // (op 1134) 
v_sub_u32 v22, v1, v21 // (op 1134) 

s_sub_u32 s56, s54, s55 // (op 1134) 
v_lshlrev_b32 v21, 4, v22 // (op 1134) 

s_lshr_b32 s57, s56, 1 // (op 1134) 

s_add_u32 s56, s57, s55 // (op 1134) 


s_lshr_b32 s55, s56, s11 // (op 1134) 

s_and_b32 s56, s11, 2147483648 // (op 1134) 
s_cmp_eq_i32 s56, 0 // (op 1134) 
s_mov_b32 s57, scc // (op 1134) 

s_cmp_eq_i32 s57, 0 // (op 1134) 
s_cselect_b32 s56, s54, s55 // (op 1134) 


s_mul_hi_u32 s55, s56, s12 // (op 1134) 
s_sub_u32 s57, s56, s55 // (op 1134) 
s_lshr_b32 s58, s57, 1 // (op 1134) 

s_add_u32 s57, s58, s55 // (op 1134) 


s_lshr_b32 s55, s57, s10 // (op 1134) 

s_and_b32 s57, s10, 2147483648 // (op 1134) 
s_cmp_eq_i32 s57, 0 // (op 1134) 
s_mov_b32 s58, scc // (op 1134) 

s_cmp_eq_i32 s58, 0 // (op 1134) 
s_cselect_b32 s57, s56, s55 // (op 1134) 


v_lshl_add_u32 v22, s57, 4, v20 // (op 1134) 


s_ashr_i64 s[58:59], s[2:3], 7 // (op 1134) 
s_mul_i32 s55, s56, s58 // (op 1134) 


s_sub_u32 s56, s54, s55 // (op 1134) 


v_lshl_add_u32 v20, s56, 7, v21 // (op 1134) 


v_mov_b32 v25, 0 // (op 1134) convert
v_mov_b32 v24, v20 // (op 1134) convert

v_mul_lo_u32 v21, v22, s25 // (op 1134) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 1134) most significant: high of low * low
v_mul_lo_u32 v20, v22, s24 // (op 1134) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 1134) most significant: sum

v_add_co_u32 v20, s[54:55], v20, v24 // (op 1134) least significant half
v_addc_co_u32 v21, s[54:55], v21, v25, s[54:55] // (op 1134) most significant half



s_mul_i32 s55, s24, s29 // (op 1136) most significant: low of low * high
s_mul_i32 s56, s25, s28 // (op 1136) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1136) least significant
s_mul_hi_u32 s57, s24, s28 // (op 1136) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 1136) most significant: sum
s_add_u32 s55, s55, s57 // (op 1136) most significant: sum


s_mov_b32 s56, s26 // (op 1136) reg expression
s_mov_b32 s57, s27 // (op 1136) reg expression
s_mov_b32 s58, s54 // (op 1136) reg expression
s_mov_b32 s59, 131072 // (op 1136) call()

v_lshrrev_b32 v22, 3, v1 // (op 1153) 
s_mul_i32 s54, s16, s8 // (op 1153) 
s_add_u32 s54, s54, s0 // (op 1153) 
v_and_b32 v23, -8, v1 // (op 1153) 
s_mul_hi_u32 s55, s54, s13 // (op 1153) 
v_sub_u32 v24, v1, v23 // (op 1153) 

v_lshlrev_b32 v23, 4, v24 // (op 1153) 

s_sub_u32 s64, s54, s55 // (op 1153) 
s_lshr_b32 s65, s64, 1 // (op 1153) 

s_add_u32 s64, s65, s55 // (op 1153) 


s_lshr_b32 s55, s64, s11 // (op 1153) 

s_and_b32 s64, s11, 2147483648 // (op 1153) 
s_cmp_eq_i32 s64, 0 // (op 1153) 
s_mov_b32 s65, scc // (op 1153) 

s_cmp_eq_i32 s65, 0 // (op 1153) 
s_cselect_b32 s64, s54, s55 // (op 1153) 


s_mul_hi_u32 s55, s64, s12 // (op 1153) 
s_sub_u32 s65, s64, s55 // (op 1153) 
s_lshr_b32 s66, s65, 1 // (op 1153) 

s_add_u32 s65, s66, s55 // (op 1153) 


s_lshr_b32 s55, s65, s10 // (op 1153) 

s_and_b32 s65, s10, 2147483648 // (op 1153) 
s_cmp_eq_i32 s65, 0 // (op 1153) 
s_mov_b32 s66, scc // (op 1153) 

s_cmp_eq_i32 s66, 0 // (op 1153) 
s_cselect_b32 s65, s64, s55 // (op 1153) 


v_lshl_add_u32 v24, s65, 4, v22 // (op 1153) 


s_ashr_i64 s[66:67], s[2:3], 7 // (op 1153) 
s_mul_i32 s55, s64, s66 // (op 1153) 


s_sub_u32 s64, s54, s55 // (op 1153) 


v_lshl_add_u32 v22, s64, 7, v23 // (op 1153) 


v_mov_b32 v27, 0 // (op 1153) convert
v_mov_b32 v26, v22 // (op 1153) convert

v_mul_lo_u32 v23, v24, s25 // (op 1153) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 1153) most significant: high of low * low
v_mul_lo_u32 v22, v24, s24 // (op 1153) least significant: low of low * low
v_add_u32 v23, v23, v25 // (op 1153) most significant: sum

v_add_co_u32 v22, s[54:55], v22, v26 // (op 1153) least significant half
v_addc_co_u32 v23, s[54:55], v23, v27, s[54:55] // (op 1153) most significant half



s_mul_i32 s55, s24, s29 // (op 1155) most significant: low of low * high
s_mul_i32 s64, s25, s28 // (op 1155) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1155) least significant
s_mul_hi_u32 s65, s24, s28 // (op 1155) most significant: high of low * low
s_add_u32 s55, s55, s64 // (op 1155) most significant: sum
s_add_u32 s55, s55, s65 // (op 1155) most significant: sum


s_mov_b32 s56, s26 // (op 1155) reg expression
s_mov_b32 s57, s27 // (op 1155) reg expression
s_mov_b32 s58, s54 // (op 1155) reg expression
s_mov_b32 s59, 131072 // (op 1155) call()

v_and_b32 v24, -2, v1 // (op 1265) 
s_mul_i32 s54, s16, s8 // (op 1265) 
s_add_u32 s54, s54, s0 // (op 1265) 
v_sub_u32 v25, v1, v24 // (op 1265) 

s_mul_hi_u32 s55, s54, s13 // (op 1265) 
v_lshlrev_b32 v24, 4, v25 // (op 1265) 

s_sub_u32 s64, s54, s55 // (op 1265) 
v_lshrrev_b32 v25, 1, v1 // (op 1265) 
s_lshr_b32 s65, s64, 1 // (op 1265) 

s_add_u32 s64, s65, s55 // (op 1265) 


s_lshr_b32 s55, s64, s11 // (op 1265) 

s_and_b32 s64, s11, 2147483648 // (op 1265) 
s_cmp_eq_i32 s64, 0 // (op 1265) 
s_mov_b32 s65, scc // (op 1265) 

s_cmp_eq_i32 s65, 0 // (op 1265) 
s_cselect_b32 s64, s54, s55 // (op 1265) 


s_mul_hi_u32 s55, s64, s12 // (op 1265) 
s_sub_u32 s65, s64, s55 // (op 1265) 
s_lshr_b32 s66, s65, 1 // (op 1265) 

s_add_u32 s65, s66, s55 // (op 1265) 


s_lshr_b32 s55, s65, s10 // (op 1265) 

s_and_b32 s65, s10, 2147483648 // (op 1265) 
s_cmp_eq_i32 s65, 0 // (op 1265) 
s_mov_b32 s66, scc // (op 1265) 

s_cmp_eq_i32 s66, 0 // (op 1265) 
s_cselect_b32 s65, s64, s55 // (op 1265) 


s_mul_i32 s55, s65, s14 // (op 1265) 

s_sub_u32 s65, s64, s55 // (op 1265) 

v_lshl_add_u32 v26, s65, 5, v24 // (op 1265) 


s_ashr_i64 s[66:67], s[2:3], 7 // (op 1265) 
v_mov_b32 v29, 0 // (op 1265) convert
v_mov_b32 v28, v26 // (op 1265) convert

s_mul_i32 s55, s64, s66 // (op 1265) 


s_sub_u32 s64, s54, s55 // (op 1265) 


v_lshl_add_u32 v24, s64, 7, v25 // (op 1265) 


v_mul_lo_u32 v27, v24, s31 // (op 1265) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 1265) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 1265) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 1265) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v28 // (op 1265) least significant half
v_addc_co_u32 v27, s[54:55], v27, v29, s[54:55] // (op 1265) most significant half



s_mul_i32 s55, s30, s3 // (op 1267) most significant: low of low * high
s_mul_i32 s64, s31, s2 // (op 1267) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1267) least significant
s_mul_hi_u32 s65, s30, s2 // (op 1267) most significant: high of low * low
s_add_u32 s55, s55, s64 // (op 1267) most significant: sum
s_add_u32 s55, s55, s65 // (op 1267) most significant: sum


s_mov_b32 s64, s32 // (op 1267) reg expression
s_mov_b32 s65, s33 // (op 1267) reg expression
s_mov_b32 s66, s54 // (op 1267) reg expression
s_mov_b32 s67, 131072 // (op 1267) call()

v_and_b32 v24, -2, v1 // (op 1284) 
s_mul_i32 s54, s16, s8 // (op 1284) 
s_add_u32 s54, s54, s0 // (op 1284) 
v_sub_u32 v25, v1, v24 // (op 1284) 

s_mul_hi_u32 s55, s54, s13 // (op 1284) 
v_lshlrev_b32 v24, 4, v25 // (op 1284) 

v_lshrrev_b32 v25, 1, v1 // (op 1284) 
s_sub_u32 s68, s54, s55 // (op 1284) 
s_lshr_b32 s69, s68, 1 // (op 1284) 

s_add_u32 s68, s69, s55 // (op 1284) 


s_lshr_b32 s55, s68, s11 // (op 1284) 

s_and_b32 s68, s11, 2147483648 // (op 1284) 
s_cmp_eq_i32 s68, 0 // (op 1284) 
s_mov_b32 s69, scc // (op 1284) 

s_cmp_eq_i32 s69, 0 // (op 1284) 
s_cselect_b32 s68, s54, s55 // (op 1284) 


s_mul_hi_u32 s55, s68, s12 // (op 1284) 
s_sub_u32 s69, s68, s55 // (op 1284) 
s_lshr_b32 s70, s69, 1 // (op 1284) 

s_add_u32 s69, s70, s55 // (op 1284) 


s_lshr_b32 s55, s69, s10 // (op 1284) 

s_and_b32 s69, s10, 2147483648 // (op 1284) 
s_cmp_eq_i32 s69, 0 // (op 1284) 
s_mov_b32 s70, scc // (op 1284) 

s_cmp_eq_i32 s70, 0 // (op 1284) 
s_cselect_b32 s69, s68, s55 // (op 1284) 


s_mul_i32 s55, s69, s14 // (op 1284) 

s_sub_u32 s69, s68, s55 // (op 1284) 

v_lshl_add_u32 v28, s69, 5, v24 // (op 1284) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 1284) 
v_mov_b32 v31, 0 // (op 1284) convert
v_mov_b32 v30, v28 // (op 1284) convert

s_mul_i32 s55, s68, s70 // (op 1284) 


s_sub_u32 s68, s54, s55 // (op 1284) 


v_lshl_add_u32 v24, s68, 7, v25 // (op 1284) 


v_mul_lo_u32 v29, v24, s31 // (op 1284) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 1284) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 1284) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1284) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v30 // (op 1284) least significant half
v_addc_co_u32 v29, s[54:55], v29, v31, s[54:55] // (op 1284) most significant half



s_mul_i32 s55, s30, s3 // (op 1286) most significant: low of low * high
s_mul_i32 s68, s31, s2 // (op 1286) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1286) least significant
s_mul_hi_u32 s69, s30, s2 // (op 1286) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 1286) most significant: sum
s_add_u32 s55, s55, s69 // (op 1286) most significant: sum


s_mov_b32 s64, s32 // (op 1286) reg expression
s_mov_b32 s65, s33 // (op 1286) reg expression
s_mov_b32 s66, s54 // (op 1286) reg expression
s_mov_b32 s67, 131072 // (op 1286) call()

v_lshrrev_b32 v24, 7, v1 // (op 1393) 
s_mul_i32 s54, s16, s8 // (op 1393) 
s_add_u32 s54, s54, s0 // (op 1393) 
v_and_b32 v25, 15, v1 // (op 1393) 
s_mul_hi_u32 s55, s54, s13 // (op 1393) 
v_lshl_add_u32 v30, v24, 4, v25 // (op 1393) 


s_sub_u32 s68, s54, s55 // (op 1393) 
v_and_b32 v24, 63, v1 // (op 1393) 
s_lshr_b32 s69, s68, 1 // (op 1393) 

v_lshrrev_b32 v25, 4, v24 // (op 1393) 

s_add_u32 s68, s69, s55 // (op 1393) 


s_lshr_b32 s55, s68, s11 // (op 1393) 

s_and_b32 s68, s11, 2147483648 // (op 1393) 
s_cmp_eq_i32 s68, 0 // (op 1393) 
s_mov_b32 s69, scc // (op 1393) 

s_cmp_eq_i32 s69, 0 // (op 1393) 
s_cselect_b32 s68, s54, s55 // (op 1393) 


s_mul_hi_u32 s55, s68, s12 // (op 1393) 
s_sub_u32 s69, s68, s55 // (op 1393) 
s_lshr_b32 s70, s69, 1 // (op 1393) 

s_add_u32 s69, s70, s55 // (op 1393) 


s_lshr_b32 s55, s69, s10 // (op 1393) 

s_and_b32 s69, s10, 2147483648 // (op 1393) 
s_cmp_eq_i32 s69, 0 // (op 1393) 
s_mov_b32 s70, scc // (op 1393) 

s_cmp_eq_i32 s70, 0 // (op 1393) 
s_cselect_b32 s69, s68, s55 // (op 1393) 


v_lshl_add_u32 v24, s69, 4, v30 // (op 1393) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 1393) 
s_mul_i32 s55, s68, s70 // (op 1393) 


s_sub_u32 s68, s54, s55 // (op 1393) 


v_lshl_add_u32 v30, s68, 2, v25 // (op 1393) 


v_mul_lo_u32 v25, v24, s34 // (op 1393) 
v_add_u32 v25, v25, v30 // (op 1393) 


s_mul_i32 s55, s34, s29 // (op 1395) most significant: low of low * high
s_mul_i32 s68, s35, s28 // (op 1395) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1395) least significant
s_mul_hi_u32 s69, s34, s28 // (op 1395) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 1395) most significant: sum
s_add_u32 s55, s55, s69 // (op 1395) most significant: sum


s_mov_b32 s68, s36 // (op 1395) reg expression
s_mov_b32 s69, s37 // (op 1395) reg expression
s_mov_b32 s70, s54 // (op 1395) reg expression
s_mov_b32 s71, 131072 // (op 1395) call()

v_lshrrev_b32 v24, 7, v1 // (op 1412) 
s_mul_i32 s54, s16, s8 // (op 1412) 
s_add_u32 s54, s54, s0 // (op 1412) 
v_and_b32 v30, 15, v1 // (op 1412) 
s_mul_hi_u32 s55, s54, s13 // (op 1412) 
v_lshl_add_u32 v31, v24, 4, v30 // (op 1412) 


v_and_b32 v24, 63, v1 // (op 1412) 
s_sub_u32 s72, s54, s55 // (op 1412) 
v_lshrrev_b32 v30, 4, v24 // (op 1412) 

s_lshr_b32 s73, s72, 1 // (op 1412) 

s_add_u32 s72, s73, s55 // (op 1412) 


s_lshr_b32 s55, s72, s11 // (op 1412) 

s_and_b32 s72, s11, 2147483648 // (op 1412) 
s_cmp_eq_i32 s72, 0 // (op 1412) 
s_mov_b32 s73, scc // (op 1412) 

s_cmp_eq_i32 s73, 0 // (op 1412) 
s_cselect_b32 s72, s54, s55 // (op 1412) 


s_mul_hi_u32 s55, s72, s12 // (op 1412) 
s_sub_u32 s73, s72, s55 // (op 1412) 
s_lshr_b32 s74, s73, 1 // (op 1412) 

s_add_u32 s73, s74, s55 // (op 1412) 


s_lshr_b32 s55, s73, s10 // (op 1412) 

s_and_b32 s73, s10, 2147483648 // (op 1412) 
s_cmp_eq_i32 s73, 0 // (op 1412) 
s_mov_b32 s74, scc // (op 1412) 

s_cmp_eq_i32 s74, 0 // (op 1412) 
s_cselect_b32 s73, s72, s55 // (op 1412) 


v_lshl_add_u32 v24, s73, 4, v31 // (op 1412) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 1412) 
s_mul_i32 s55, s72, s74 // (op 1412) 


s_sub_u32 s72, s54, s55 // (op 1412) 


v_lshl_add_u32 v31, s72, 2, v30 // (op 1412) 


v_mul_lo_u32 v30, v24, s34 // (op 1412) 
v_add_u32 v30, v30, v31 // (op 1412) 


s_mul_i32 s55, s34, s29 // (op 1414) most significant: low of low * high
s_mul_i32 s72, s35, s28 // (op 1414) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1414) least significant
s_mul_hi_u32 s73, s34, s28 // (op 1414) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 1414) most significant: sum
s_add_u32 s55, s55, s73 // (op 1414) most significant: sum


s_mov_b32 s68, s36 // (op 1414) reg expression
s_mov_b32 s69, s37 // (op 1414) reg expression
s_mov_b32 s70, s54 // (op 1414) reg expression
s_mov_b32 s71, 131072 // (op 1414) call()

v_and_b32 v24, -32, v1 // (op 1521) 
s_mul_i32 s54, s16, s8 // (op 1521) 
s_add_u32 s54, s54, s0 // (op 1521) 
v_sub_u32 v31, v1, v24 // (op 1521) 

s_mul_hi_u32 s55, s54, s13 // (op 1521) 
v_lshrrev_b32 v24, 5, v1 // (op 1521) 
s_sub_u32 s72, s54, s55 // (op 1521) 
s_lshr_b32 s73, s72, 1 // (op 1521) 

s_add_u32 s72, s73, s55 // (op 1521) 


s_lshr_b32 s55, s72, s11 // (op 1521) 

s_and_b32 s72, s11, 2147483648 // (op 1521) 
s_cmp_eq_i32 s72, 0 // (op 1521) 
s_mov_b32 s73, scc // (op 1521) 

s_cmp_eq_i32 s73, 0 // (op 1521) 
s_cselect_b32 s72, s54, s55 // (op 1521) 


s_mul_hi_u32 s55, s72, s12 // (op 1521) 
s_sub_u32 s73, s72, s55 // (op 1521) 
s_lshr_b32 s74, s73, 1 // (op 1521) 

s_add_u32 s73, s74, s55 // (op 1521) 


s_lshr_b32 s55, s73, s10 // (op 1521) 

s_and_b32 s73, s10, 2147483648 // (op 1521) 
s_cmp_eq_i32 s73, 0 // (op 1521) 
s_mov_b32 s74, scc // (op 1521) 

s_cmp_eq_i32 s74, 0 // (op 1521) 
s_cselect_b32 s73, s72, s55 // (op 1521) 


s_mul_i32 s55, s73, s14 // (op 1521) 

s_sub_u32 s73, s72, s55 // (op 1521) 

s_ashr_i64 s[74:75], s[2:3], 7 // (op 1521) 
v_lshl_add_u32 v32, s73, 5, v31 // (op 1521) 


s_mul_i32 s55, s72, s74 // (op 1521) 


s_sub_u32 s72, s54, s55 // (op 1521) 


v_lshl_add_u32 v31, s72, 2, v24 // (op 1521) 


v_mul_lo_u32 v24, v31, s38 // (op 1521) 
v_add_u32 v24, v24, v32 // (op 1521) 


s_mul_i32 s55, s38, s43 // (op 1523) most significant: low of low * high
s_mul_i32 s72, s39, s42 // (op 1523) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1523) least significant
s_mul_hi_u32 s73, s38, s42 // (op 1523) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 1523) most significant: sum
s_add_u32 s55, s55, s73 // (op 1523) most significant: sum


s_mov_b32 s72, s40 // (op 1523) reg expression
s_mov_b32 s73, s41 // (op 1523) reg expression
s_mov_b32 s74, s54 // (op 1523) reg expression
s_mov_b32 s75, 131072 // (op 1523) call()

v_and_b32 v31, -32, v1 // (op 1540) 
s_mul_i32 s54, s16, s8 // (op 1540) 
s_add_u32 s54, s54, s0 // (op 1540) 
v_sub_u32 v32, v1, v31 // (op 1540) 

s_mul_hi_u32 s55, s54, s13 // (op 1540) 
v_lshrrev_b32 v31, 5, v1 // (op 1540) 
s_sub_u32 s76, s54, s55 // (op 1540) 
s_lshr_b32 s77, s76, 1 // (op 1540) 

s_add_u32 s76, s77, s55 // (op 1540) 


s_lshr_b32 s55, s76, s11 // (op 1540) 

s_and_b32 s76, s11, 2147483648 // (op 1540) 
s_cmp_eq_i32 s76, 0 // (op 1540) 
s_mov_b32 s77, scc // (op 1540) 

s_cmp_eq_i32 s77, 0 // (op 1540) 
s_cselect_b32 s76, s54, s55 // (op 1540) 


s_mul_hi_u32 s55, s76, s12 // (op 1540) 
s_sub_u32 s77, s76, s55 // (op 1540) 
s_lshr_b32 s78, s77, 1 // (op 1540) 

s_add_u32 s77, s78, s55 // (op 1540) 


s_lshr_b32 s55, s77, s10 // (op 1540) 

s_and_b32 s77, s10, 2147483648 // (op 1540) 
s_cmp_eq_i32 s77, 0 // (op 1540) 
s_mov_b32 s78, scc // (op 1540) 

s_cmp_eq_i32 s78, 0 // (op 1540) 
s_cselect_b32 s77, s76, s55 // (op 1540) 


s_mul_i32 s55, s77, s14 // (op 1540) 

s_sub_u32 s77, s76, s55 // (op 1540) 

v_lshl_add_u32 v33, s77, 5, v32 // (op 1540) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 1540) 
s_mul_i32 s55, s76, s78 // (op 1540) 


s_sub_u32 s76, s54, s55 // (op 1540) 


v_lshl_add_u32 v32, s76, 2, v31 // (op 1540) 


v_mul_lo_u32 v31, v32, s38 // (op 1540) 
v_add_u32 v31, v31, v33 // (op 1540) 


s_mul_i32 s55, s38, s43 // (op 1542) most significant: low of low * high
s_mul_i32 s76, s39, s42 // (op 1542) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1542) least significant
s_mul_hi_u32 s77, s38, s42 // (op 1542) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1542) most significant: sum
s_add_u32 s55, s55, s77 // (op 1542) most significant: sum


s_mov_b32 s72, s40 // (op 1542) reg expression
s_mov_b32 s73, s41 // (op 1542) reg expression
s_mov_b32 s74, s54 // (op 1542) reg expression
s_mov_b32 s75, 131072 // (op 1542) call()




buffer_load_ubyte v32, v25, s[68:71], 0 offen offset: 0 // (op 802) Load value
v_and_b32 v33, -2, v1 // (op 1729) 
v_sub_u32 v34, v1, v33 // (op 1729) 

v_lshlrev_b32 v33, 4, v34 // (op 1729) 

v_lshrrev_b32 v34, 1, v1 // (op 1729) 
v_lshl_add_u32 v35, v34, 5, v33 // (op 1729) 


s_nop 0
v_readfirstlane_b32 s54, v35 // (op 1729) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2416) 

s_lshl_b64 s[76:77], s[30:31], 6 // (op 1045) 
v_mov_b64 v[34:35], s[76:77] // (op 1045) convert


v_mov_b64 v[36:37], v[26:27] // (op 1045) reg expression
s_mov_b32 m0, s54 // (op 1045) reg expression
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1045) Load value direct to lds
v_add_co_u32 v36, s[76:77], v36, v34 // (op 1045) least significant half
v_addc_co_u32 v37, s[76:77], 0, v37, s[76:77] // (op 1045) most significant half

s_add_u32 m0, m0, 2048 // (op 1045) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1045) Load value direct to lds





v_mov_b32 v33, v24 // (op 809) reg expression

buffer_load_ubyte v34, v33, s[72:75], 0 offen offset: 0 // (op 809) Load value

v_and_b32 v33, -8, v1 // (op 1664) 
v_sub_u32 v35, v1, v33 // (op 1664) 

v_lshlrev_b32 v33, 4, v35 // (op 1664) 

v_lshrrev_b32 v35, 3, v1 // (op 1664) 
v_lshl_add_u32 v36, v35, 7, v33 // (op 1664) 


s_nop 0
v_readfirstlane_b32 s54, v36 // (op 1664) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2397) 


s_nop 0
s_add_u32 s55, 4096, s54 // (op 1050) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 1050) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 1050) Load value direct to lds





buffer_load_ubyte v33, v25, s[68:71], 0 offen offset:4 // (op 816) Load value

v_and_b32 v25, -2, v1 // (op 2288) 
v_sub_u32 v35, v1, v25 // (op 2288) 

v_lshlrev_b32 v25, 4, v35 // (op 2288) 

v_lshrrev_b32 v35, 1, v1 // (op 2288) 
v_lshl_add_u32 v36, v35, 5, v25 // (op 2288) 


s_nop 0
v_readfirstlane_b32 s54, v36 // (op 2288) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2546) 

s_lshl_b64 s[76:77], s[30:31], 6 // (op 1055) 
v_mov_b64 v[36:37], s[76:77] // (op 1055) convert


s_lshl_b64 s[76:77], s[30:31], 7 // (op 1055) 
v_mov_b32 v25, s76 // (op 1055) 
v_mov_b32 v35, s77 // (op 1055) 
v_add_co_u32 v38, s[78:79], v26, v25 // (op 1055) least significant half
v_addc_co_u32 v39, s[78:79], v27, v35, s[78:79] // (op 1055) most significant half




s_add_u32 s55, 6144, s54 // (op 1055) 
s_mov_b32 m0, s55 // (op 1055) reg expression
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1055) Load value direct to lds
v_add_co_u32 v38, s[76:77], v38, v36 // (op 1055) least significant half
v_addc_co_u32 v39, s[76:77], 0, v39, s[76:77] // (op 1055) most significant half

s_add_u32 m0, m0, 2048 // (op 1055) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1055) Load value direct to lds







s_lshl_b32 s54, s38, 2 // (op 823) 
v_add_u32 v25, v24, s54 // (op 823) 


buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 823) Load value


v_and_b32 v27, -8, v1 // (op 2249) 
v_sub_u32 v24, v1, v27 // (op 2249) 

v_lshlrev_b32 v25, 4, v24 // (op 2249) 

v_lshrrev_b32 v24, 3, v1 // (op 2249) 
v_lshl_add_u32 v27, v24, 7, v25 // (op 2249) 


s_nop 0
v_readfirstlane_b32 s54, v27 // (op 2249) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2537) 


s_nop 0
s_add_u32 s55, 10240, s54 // (op 1057) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 1057) reg expression
s_mov_b32 s76, 128 // (op 1057) call()
buffer_load_dwordx4 v20, s[56:59], s76 offen offset: 0 lds // (op 1057) Load value direct to lds




v_and_b32 v27, -32, v1 // (op 1795) 
v_sub_u32 v20, v1, v27 // (op 1795) 

v_and_b32 v21, 4294967264, v1 // (op 1795) 
v_add_u32 v27, v21, v20 // (op 1795) 


s_barrier  // (op 2435) 



s_waitcnt vmcnt(6)
ds_write_b8 v27, v34 offset:12288 // (op 829) Store local data 

s_barrier  // (op 2413) 



ds_read_b64_tr_b8 v[36:37], v19 // (op 846) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v19 offset:256 // (op 846) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v19 offset:2048 // (op 846) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v19 offset:2304 // (op 846) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2432) 


ds_read_u8 v27, v16 offset:12288 // (op 849) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 2394) 



ds_read_b128 v[44:47], v17 offset:4096 // (op 843) Load local data 
ds_read_b128 v[48:51], v17 offset:4160 // (op 843) Load local data 
v_and_b32 v35, -8, v1 // (op 1636) 
v_sub_u32 v20, v1, v35 // (op 1636) 

v_lshlrev_b32 v21, 4, v20 // (op 1636) 

v_lshrrev_b32 v20, 3, v1 // (op 1636) 
v_lshl_add_u32 v35, v20, 7, v21 // (op 1636) 


s_nop 0
v_readfirstlane_b32 s54, v35 // (op 1636) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -2, v1 // (op 1701) 
v_sub_u32 v20, v1, v35 // (op 1701) 

v_lshlrev_b32 v21, 4, v20 // (op 1701) 

v_lshrrev_b32 v20, 1, v1 // (op 1701) 
v_lshl_add_u32 v35, v20, 5, v21 // (op 1701) 


s_nop 0
v_readfirstlane_b32 s55, v35 // (op 1701) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -32, v1 // (op 1766) 
v_sub_u32 v20, v1, v35 // (op 1766) 

v_and_b32 v21, 4294967264, v1 // (op 1766) 
v_add_u32 v35, v21, v20 // (op 1766) 


v_and_b32 v20, -8, v1 // (op 2234) 
v_sub_u32 v21, v1, v20 // (op 2234) 

v_lshlrev_b32 v20, 4, v21 // (op 2234) 

v_lshrrev_b32 v21, 3, v1 // (op 2234) 
v_lshl_add_u32 v24, v21, 7, v20 // (op 2234) 


s_nop 0
v_readfirstlane_b32 s76, v24 // (op 2234) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -2, v1 // (op 2273) 
v_sub_u32 v21, v1, v20 // (op 2273) 

v_lshlrev_b32 v20, 4, v21 // (op 2273) 

v_lshrrev_b32 v21, 1, v1 // (op 2273) 
v_lshl_add_u32 v24, v21, 5, v20 // (op 2273) 


s_nop 0
v_readfirstlane_b32 s77, v24 // (op 2273) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -32, v1 // (op 2312) 
v_sub_u32 v21, v1, v20 // (op 2312) 

v_and_b32 v20, 4294967264, v1 // (op 2312) 
v_add_u32 v24, v20, v21 // (op 2312) 


s_mov_b32 s1, 0 // (op 57) call()
s_and_b32 s78, s7, 4294967294 // (op 56) 
s_cmp_lt_u32 s1, s78 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s1, s1, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[36:43], a[0:3], v32, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 


v_mov_b32 v25, v30 // (op 77) 

buffer_load_ubyte v32, v25, s[68:71], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 2419) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 984) 
v_mov_b64 v[20:21], s[78:79] // (op 984) convert


v_mov_b64 v[52:53], v[28:29] // (op 984) 
s_lshl_b64 s[78:79], s[30:31], 7 // (op 984) 
s_lshl_b64 s[80:81], s[78:79], 1 // (op 984) 

v_mov_b32 v54, s80 // (op 984) 
v_mov_b32 v55, s81 // (op 984) 
v_add_co_u32 v56, s[78:79], v52, v54 // (op 984) least significant half
v_addc_co_u32 v57, s[78:79], v53, v55, s[78:79] // (op 984) most significant half




s_mov_b32 m0, s55 // (op 984) reg expression
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 984) Load value direct to lds
v_add_co_u32 v56, s[78:79], v56, v20 // (op 984) least significant half
v_addc_co_u32 v57, s[78:79], 0, v57, s[78:79] // (op 984) most significant half

s_add_u32 m0, m0, 2048 // (op 984) 
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 984) Load value direct to lds




v_mov_b32 v20, v31 // (op 21) 
s_lshl_b32 s78, s38, 3 // (op 21) 
v_add_u32 v21, v20, s78 // (op 21) 


buffer_load_ubyte v34, v21, s[72:75], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 2400) 


v_mov_b64 v[54:55], v[22:23] // (op 979) 
s_add_u32 s78, 4096, s54 // (op 979) 

s_mov_b32 m0, s78 // (op 979) reg expression
s_mov_b32 s79, 256 // (op 979) call()
buffer_load_dwordx4 v54, s[56:59], s79 offen offset: 0 lds // (op 979) Load value direct to lds


s_barrier  // (op 2555) 



s_waitcnt vmcnt(6)
ds_write_b8 v24, v26 offset:12416 // (op 661) Store local data 
s_barrier  // (op 2543) 



ds_read_b64_tr_b8 v[36:37], v13 offset:6144 // (op 654) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v13 offset:6400 // (op 654) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:8192 // (op 654) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v13 offset:8448 // (op 654) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2552) 


ds_read_u8 v27, v11 offset:12416 // (op 658) Load local data 
s_barrier  // (op 2688) 
s_waitcnt vmcnt(5)
s_barrier  // (op 2534) 



ds_read_b128 v[44:47], v12 offset:10240 // (op 652) Load local data 
ds_read_b128 v[48:51], v12 offset:10304 // (op 652) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[36:43], a[0:3], v33, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 659) 



buffer_load_ubyte v33, v25, s[68:71], 0 offen offset:12 // (op 656) Load value

s_barrier  // (op 2549) 
v_add_u32 v30, 8, v30 // (op 1404) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 1040) 
v_mov_b64 v[56:57], s[78:79] // (op 1040) convert


s_lshl_b64 s[78:79], s[30:31], 7 // (op 1040) 
s_mul_i32 s81, 3, s79 // (op 1040) most significant: low of low * high
s_mul_i32 s80, 3, s78 // (op 1040) least significant
s_mul_hi_u32 s82, 3, s78 // (op 1040) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 1040) most significant: sum


v_mov_b32 v21, s80 // (op 1040) 
v_mov_b32 v25, s81 // (op 1040) 
v_add_co_u32 v58, s[78:79], v52, v21 // (op 1040) least significant half
v_addc_co_u32 v59, s[78:79], v53, v25, s[78:79] // (op 1040) most significant half




s_add_u32 s78, 6144, s77 // (op 1040) 
s_nop 0
s_mov_b32 m0, s78 // (op 1040) reg expression
buffer_load_dwordx4 v58, s[64:67], 0 offen offset: 0 lds // (op 1040) Load value direct to lds
v_add_co_u32 v58, s[80:81], v58, v56 // (op 1040) least significant half
v_addc_co_u32 v59, s[80:81], 0, v59, s[80:81] // (op 1040) most significant half

s_add_u32 m0, m0, 2048 // (op 1040) 
buffer_load_dwordx4 v58, s[64:67], 0 offen offset: 0 lds // (op 1040) Load value direct to lds






s_lshl_b32 s78, s38, 2 // (op 663) 
s_mul_i32 s79, 3, s78 // (op 663) 

v_add_u32 v21, v20, s79 // (op 663) 


s_lshl_b64 s[78:79], s[30:31], 7 // (op 1276) 
s_lshl_b64 s[80:81], s[78:79], 1 // (op 1276) 

v_mov_b32 v25, s80 // (op 1276) 
v_mov_b32 v52, s81 // (op 1276) 
v_add_co_u32 v28, s[78:79], v28, v25 // (op 1276) least significant half
v_addc_co_u32 v29, s[78:79], v29, v52, s[78:79] // (op 1276) most significant half




buffer_load_ubyte v26, v21, s[72:75], 0 offen offset: 0 // (op 663) Load value


s_barrier  // (op 2540) 
s_lshl_b32 s78, s38, 3 // (op 1532) 


v_add_u32 v31, v31, s78 // (op 1532) 

s_add_u32 s78, 10240, s76 // (op 1035) 

s_mov_b32 m0, s78 // (op 1035) reg expression
s_mov_b32 s79, 384 // (op 1035) call()
buffer_load_dwordx4 v54, s[56:59], s79 offen offset: 0 lds // (op 1035) Load value direct to lds



s_barrier  // (op 2438) 
v_add_co_u32 v22, vcc, 256, v22 // (op 1145) least significant half



v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 1145) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v35, v34 offset:12288 // (op 26) Store local data 
s_barrier  // (op 2425) 



ds_read_b64_tr_b8 v[36:37], v4 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:256 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2304 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2445) 


ds_read_u8 v27, v0 offset:12288 // (op 81) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 2406) 



ds_read_b128 v[44:47], v3 offset:4096 // (op 67) Load local data 
ds_read_b128 v[48:51], v3 offset:4160 // (op 67) Load local data 
s_and_b32 s78, s7, 4294967294 // (op 56) 
s_cmp_lt_u32 s1, s78 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:












s_barrier  // (op 2566) 
s_and_b32 s54, s7, 4294967294 // (op 1108) 
v_lshrrev_b32 v20, 3, v1 // (op 1108) 
s_add_u32 s55, s0, s54 // (op 1108) 

v_and_b32 v21, -8, v1 // (op 1108) 
v_sub_u32 v22, v1, v21 // (op 1108) 

s_mul_i32 s54, s16, s8 // (op 1108) 
s_add_u32 s54, s54, s55 // (op 1108) 

v_lshlrev_b32 v21, 4, v22 // (op 1108) 

s_mul_hi_u32 s55, s54, s13 // (op 1108) 
s_sub_u32 s76, s54, s55 // (op 1108) 
s_lshr_b32 s77, s76, 1 // (op 1108) 

s_add_u32 s76, s77, s55 // (op 1108) 


s_lshr_b32 s55, s76, s11 // (op 1108) 

s_and_b32 s76, s11, 2147483648 // (op 1108) 
s_cmp_eq_i32 s76, 0 // (op 1108) 
s_mov_b32 s77, scc // (op 1108) 

s_cmp_eq_i32 s77, 0 // (op 1108) 
s_cselect_b32 s76, s54, s55 // (op 1108) 


s_mul_hi_u32 s55, s76, s12 // (op 1108) 
s_sub_u32 s77, s76, s55 // (op 1108) 
s_lshr_b32 s78, s77, 1 // (op 1108) 

s_add_u32 s77, s78, s55 // (op 1108) 


s_lshr_b32 s55, s77, s10 // (op 1108) 

s_and_b32 s77, s10, 2147483648 // (op 1108) 
s_cmp_eq_i32 s77, 0 // (op 1108) 
s_mov_b32 s78, scc // (op 1108) 

s_cmp_eq_i32 s78, 0 // (op 1108) 
s_cselect_b32 s77, s76, s55 // (op 1108) 


v_lshl_add_u32 v22, s77, 4, v20 // (op 1108) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 1108) 
s_mul_i32 s55, s76, s78 // (op 1108) 


s_sub_u32 s76, s54, s55 // (op 1108) 


v_lshl_add_u32 v20, s76, 7, v21 // (op 1108) 


v_mov_b32 v25, 0 // (op 1108) convert
v_mov_b32 v24, v20 // (op 1108) convert

v_mul_lo_u32 v21, v22, s25 // (op 1108) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 1108) most significant: high of low * low
v_mul_lo_u32 v20, v22, s24 // (op 1108) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 1108) most significant: sum

v_add_co_u32 v20, s[54:55], v20, v24 // (op 1108) least significant half
v_addc_co_u32 v21, s[54:55], v21, v25, s[54:55] // (op 1108) most significant half



s_mul_i32 s55, s24, s29 // (op 1110) most significant: low of low * high
s_mul_i32 s76, s25, s28 // (op 1110) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1110) least significant
s_mul_hi_u32 s77, s24, s28 // (op 1110) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1110) most significant: sum
s_add_u32 s55, s55, s77 // (op 1110) most significant: sum


s_mov_b32 s56, s26 // (op 1110) reg expression
s_mov_b32 s57, s27 // (op 1110) reg expression
s_mov_b32 s58, s54 // (op 1110) reg expression
s_mov_b32 s59, 131072 // (op 1110) call()

s_and_b32 s54, s7, 4294967294 // (op 1242) 
v_and_b32 v22, -2, v1 // (op 1242) 
s_add_u32 s55, s0, s54 // (op 1242) 

v_sub_u32 v23, v1, v22 // (op 1242) 

v_lshlrev_b32 v22, 4, v23 // (op 1242) 

s_mul_i32 s54, s16, s8 // (op 1242) 
s_add_u32 s54, s54, s55 // (op 1242) 

v_lshrrev_b32 v23, 1, v1 // (op 1242) 
s_mul_hi_u32 s55, s54, s13 // (op 1242) 
s_sub_u32 s76, s54, s55 // (op 1242) 
s_lshr_b32 s77, s76, 1 // (op 1242) 

s_add_u32 s76, s77, s55 // (op 1242) 


s_lshr_b32 s55, s76, s11 // (op 1242) 

s_and_b32 s76, s11, 2147483648 // (op 1242) 
s_cmp_eq_i32 s76, 0 // (op 1242) 
s_mov_b32 s77, scc // (op 1242) 

s_cmp_eq_i32 s77, 0 // (op 1242) 
s_cselect_b32 s76, s54, s55 // (op 1242) 


s_mul_hi_u32 s55, s76, s12 // (op 1242) 
s_sub_u32 s77, s76, s55 // (op 1242) 
s_lshr_b32 s78, s77, 1 // (op 1242) 

s_add_u32 s77, s78, s55 // (op 1242) 


s_lshr_b32 s55, s77, s10 // (op 1242) 

s_and_b32 s77, s10, 2147483648 // (op 1242) 
s_cmp_eq_i32 s77, 0 // (op 1242) 
s_mov_b32 s78, scc // (op 1242) 

s_cmp_eq_i32 s78, 0 // (op 1242) 
s_cselect_b32 s77, s76, s55 // (op 1242) 


s_mul_i32 s55, s77, s14 // (op 1242) 

s_sub_u32 s77, s76, s55 // (op 1242) 

v_lshl_add_u32 v24, s77, 5, v22 // (op 1242) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 1242) 
v_mov_b32 v35, 0 // (op 1242) convert
v_mov_b32 v34, v24 // (op 1242) convert

s_mul_i32 s55, s76, s78 // (op 1242) 


s_sub_u32 s76, s54, s55 // (op 1242) 


v_lshl_add_u32 v22, s76, 7, v23 // (op 1242) 


v_mul_lo_u32 v25, v22, s31 // (op 1242) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 1242) most significant: high of low * low
v_mul_lo_u32 v24, v22, s30 // (op 1242) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 1242) most significant: sum

v_add_co_u32 v24, s[54:55], v24, v34 // (op 1242) least significant half
v_addc_co_u32 v25, s[54:55], v25, v35, s[54:55] // (op 1242) most significant half



s_mul_i32 s55, s30, s3 // (op 1244) most significant: low of low * high
s_mul_i32 s76, s31, s2 // (op 1244) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1244) least significant
s_mul_hi_u32 s77, s30, s2 // (op 1244) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1244) most significant: sum
s_add_u32 s55, s55, s77 // (op 1244) most significant: sum


s_mov_b32 s64, s32 // (op 1244) reg expression
s_mov_b32 s65, s33 // (op 1244) reg expression
s_mov_b32 s66, s54 // (op 1244) reg expression
s_mov_b32 s67, 131072 // (op 1244) call()

s_and_b32 s54, s7, 4294967294 // (op 1370) 
v_lshrrev_b32 v22, 7, v1 // (op 1370) 
s_add_u32 s55, s0, s54 // (op 1370) 

v_and_b32 v23, 15, v1 // (op 1370) 
v_lshl_add_u32 v28, v22, 4, v23 // (op 1370) 


s_mul_i32 s54, s16, s8 // (op 1370) 
s_add_u32 s54, s54, s55 // (op 1370) 

v_and_b32 v22, 63, v1 // (op 1370) 
s_mul_hi_u32 s55, s54, s13 // (op 1370) 
v_lshrrev_b32 v23, 4, v22 // (op 1370) 

s_sub_u32 s76, s54, s55 // (op 1370) 
s_lshr_b32 s77, s76, 1 // (op 1370) 

s_add_u32 s76, s77, s55 // (op 1370) 


s_lshr_b32 s55, s76, s11 // (op 1370) 

s_and_b32 s76, s11, 2147483648 // (op 1370) 
s_cmp_eq_i32 s76, 0 // (op 1370) 
s_mov_b32 s77, scc // (op 1370) 

s_cmp_eq_i32 s77, 0 // (op 1370) 
s_cselect_b32 s76, s54, s55 // (op 1370) 


s_mul_hi_u32 s55, s76, s12 // (op 1370) 
s_sub_u32 s77, s76, s55 // (op 1370) 
s_lshr_b32 s78, s77, 1 // (op 1370) 

s_add_u32 s77, s78, s55 // (op 1370) 


s_lshr_b32 s55, s77, s10 // (op 1370) 

s_and_b32 s77, s10, 2147483648 // (op 1370) 
s_cmp_eq_i32 s77, 0 // (op 1370) 
s_mov_b32 s78, scc // (op 1370) 

s_cmp_eq_i32 s78, 0 // (op 1370) 
s_cselect_b32 s77, s76, s55 // (op 1370) 


v_lshl_add_u32 v22, s77, 4, v28 // (op 1370) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 1370) 
s_mul_i32 s55, s76, s78 // (op 1370) 


s_sub_u32 s76, s54, s55 // (op 1370) 


v_lshl_add_u32 v28, s76, 2, v23 // (op 1370) 


v_mul_lo_u32 v23, v22, s34 // (op 1370) 
v_add_u32 v23, v23, v28 // (op 1370) 


s_mul_i32 s55, s34, s29 // (op 1372) most significant: low of low * high
s_mul_i32 s76, s35, s28 // (op 1372) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1372) least significant
s_mul_hi_u32 s77, s34, s28 // (op 1372) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1372) most significant: sum
s_add_u32 s55, s55, s77 // (op 1372) most significant: sum


s_mov_b32 s68, s36 // (op 1372) reg expression
s_mov_b32 s69, s37 // (op 1372) reg expression
s_mov_b32 s70, s54 // (op 1372) reg expression
s_mov_b32 s71, 131072 // (op 1372) call()

s_and_b32 s54, s7, 4294967294 // (op 1498) 
v_and_b32 v22, -32, v1 // (op 1498) 
s_add_u32 s55, s0, s54 // (op 1498) 

v_sub_u32 v28, v1, v22 // (op 1498) 

v_lshrrev_b32 v22, 5, v1 // (op 1498) 
s_mul_i32 s54, s16, s8 // (op 1498) 
s_add_u32 s54, s54, s55 // (op 1498) 

s_mul_hi_u32 s55, s54, s13 // (op 1498) 
s_sub_u32 s76, s54, s55 // (op 1498) 
s_lshr_b32 s77, s76, 1 // (op 1498) 

s_add_u32 s76, s77, s55 // (op 1498) 


s_lshr_b32 s55, s76, s11 // (op 1498) 

s_and_b32 s76, s11, 2147483648 // (op 1498) 
s_cmp_eq_i32 s76, 0 // (op 1498) 
s_mov_b32 s77, scc // (op 1498) 

s_cmp_eq_i32 s77, 0 // (op 1498) 
s_cselect_b32 s76, s54, s55 // (op 1498) 


s_mul_hi_u32 s55, s76, s12 // (op 1498) 
s_sub_u32 s77, s76, s55 // (op 1498) 
s_lshr_b32 s78, s77, 1 // (op 1498) 

s_add_u32 s77, s78, s55 // (op 1498) 


s_lshr_b32 s55, s77, s10 // (op 1498) 

s_and_b32 s77, s10, 2147483648 // (op 1498) 
s_cmp_eq_i32 s77, 0 // (op 1498) 
s_mov_b32 s78, scc // (op 1498) 

s_cmp_eq_i32 s78, 0 // (op 1498) 
s_cselect_b32 s77, s76, s55 // (op 1498) 


s_mul_i32 s55, s77, s14 // (op 1498) 

s_sub_u32 s77, s76, s55 // (op 1498) 

v_lshl_add_u32 v29, s77, 5, v28 // (op 1498) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 1498) 
s_mul_i32 s55, s76, s78 // (op 1498) 


s_sub_u32 s76, s54, s55 // (op 1498) 


v_lshl_add_u32 v28, s76, 2, v22 // (op 1498) 


v_mul_lo_u32 v22, v28, s38 // (op 1498) 
v_add_u32 v22, v22, v29 // (op 1498) 


s_mul_i32 s55, s38, s43 // (op 1500) most significant: low of low * high
s_mul_i32 s76, s39, s42 // (op 1500) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1500) least significant
s_mul_hi_u32 s77, s38, s42 // (op 1500) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1500) most significant: sum
s_add_u32 s55, s55, s77 // (op 1500) most significant: sum


s_mov_b32 s72, s40 // (op 1500) reg expression
s_mov_b32 s73, s41 // (op 1500) reg expression
s_mov_b32 s74, s54 // (op 1500) reg expression
s_mov_b32 s75, 131072 // (op 1500) call()

v_and_b32 v28, -8, v1 // (op 1649) 
v_sub_u32 v29, v1, v28 // (op 1649) 

v_lshlrev_b32 v28, 4, v29 // (op 1649) 

v_lshrrev_b32 v29, 3, v1 // (op 1649) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 1649) 


s_nop 0
v_readfirstlane_b32 s54, v30 // (op 1649) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -2, v1 // (op 1714) 
v_sub_u32 v29, v1, v28 // (op 1714) 

v_lshlrev_b32 v28, 4, v29 // (op 1714) 

v_lshrrev_b32 v29, 1, v1 // (op 1714) 
v_lshl_add_u32 v30, v29, 5, v28 // (op 1714) 


s_nop 0
v_readfirstlane_b32 s55, v30 // (op 1714) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -32, v1 // (op 1779) 
v_sub_u32 v29, v1, v28 // (op 1779) 

v_and_b32 v28, 4294967264, v1 // (op 1779) 
v_add_u32 v30, v28, v29 // (op 1779) 


s_and_b32 s76, s7, 4294967294 // (op 587) 
s_cmp_lt_u32 s76, s7 // (op 586) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_586 // (op 586) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_586 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_586:
s_barrier  // (op 2403) 
s_add_u32 s76, s76, 1 // (op 588) 


v_mov_b64 v[26:27], v[20:21] // (op 1022) 
s_add_u32 s77, 4096, s54 // (op 1022) 

s_mov_b32 m0, s77 // (op 1022) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 1022) Load value direct to lds


v_add_co_u32 v20, vcc, 128, v20 // (op 1098) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1098) most significant half
s_barrier  // (op 2422) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 1028) 
v_mov_b64 v[26:27], s[78:79] // (op 1028) convert


v_mov_b64 v[28:29], v[24:25] // (op 1028) 
v_mov_b64 v[32:33], v[28:29] // (op 1028) reg expression
s_mov_b32 m0, s55 // (op 1028) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v32, s[78:79], v32, v26 // (op 1028) least significant half
v_addc_co_u32 v33, s[78:79], 0, v33, s[78:79] // (op 1028) most significant half

s_add_u32 m0, m0, 2048 // (op 1028) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1028) Load value direct to lds



s_lshl_b64 s[78:79], s[30:31], 7 // (op 1234) 
v_mov_b32 v26, s78 // (op 1234) 
v_mov_b32 v27, s79 // (op 1234) 
v_add_co_u32 v24, s[80:81], v24, v26 // (op 1234) least significant half
v_addc_co_u32 v25, s[80:81], v25, v27, s[80:81] // (op 1234) most significant half






v_mov_b32 v26, v22 // (op 605) 
v_mov_b32 v27, v26 // (op 605) reg expression

buffer_load_ubyte v28, v27, s[72:75], 0 offen offset: 0 // (op 605) Load value


s_lshl_b32 s77, s38, 2 // (op 1490) 
v_add_u32 v22, v22, s77 // (op 1490) 

s_barrier  // (op 2441) 
s_waitcnt vmcnt(3)
s_barrier  // (op 2409) 






ds_read_b128 v[32:35], v3 offset:4096 // (op 594) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 594) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v30, v28 offset:12288 // (op 603) Store local data 

s_barrier  // (op 2428) 



ds_read_b64_tr_b8 v[40:41], v4 // (op 596) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:256 // (op 596) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:2048 // (op 596) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:2304 // (op 596) Transpose load local data 


v_mov_b32 v31, v23 // (op 598) 

buffer_load_ubyte v26, v31, s[68:71], 0 offen offset: 0 // (op 598) Load value

v_add_u32 v23, 4, v23 // (op 1362) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2448) 


ds_read_u8 v31, v0 offset:12288 // (op 600) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 601) 




s_cmp_lt_u32 s76, s7 // (op 586) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_586 // (op 586) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_586 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_586:








s_barrier  // (op 2678) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_306 // (op 306) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_306
v_lshrrev_b32 v16, 5, v1 // (op 2100) 
v_lshlrev_b32 v17, 2, v16 // (op 2100) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 2100) 

v_and_b32 v17, -32, v1 // (op 2100) 
v_sub_u32 v19, v1, v17 // (op 2100) 

v_lshl_add_u32 v17, v16, 5, v19 // (op 2100) 


v_lshlrev_b32 v16, 2, v17 // (op 2100) 

s_lshl_b32 s54, s15, 11 // (op 2102) 
s_mov_b32 s76, s44 // (op 2102) reg expression
s_mov_b32 s77, s45 // (op 2102) reg expression
s_mov_b32 s78, s54 // (op 2102) reg expression
s_mov_b32 s79, 131072 // (op 2102) call()




buffer_store_dword a0, v16, s[76:79], 0 offen sc1 // (op 303) Store value
buffer_store_dword a1, v16, s[76:79], 0 offen offset:128 sc1 // (op 303) Store value
buffer_store_dword a2, v16, s[76:79], 0 offen offset:256 sc1 // (op 303) Store value
buffer_store_dword a3, v16, s[76:79], 0 offen offset:384 sc1 // (op 303) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 311) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311 // (op 311) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311
s_mov_b32 s54, 1 // (op 309) call()
s_mov_b32 s81, 0 // (op 310) convert
s_mov_b32 s80, s16 // (op 310) convert
s_lshl_b64 s[80:81], s[80:81], 2 // (op 310) 
s_add_u32 s82, s46, s80 // (op 310) Start of Int64 add, locking SCC
s_addc_u32 s83, s47, s81 // (op 310) End of Int64 add, unlocking SCC
s_store_dword s54, s[82:83], 0 glc // (op 310) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311 // (op 311) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311:
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_306 // (op 306) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_306
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_306:
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_306:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 322) 
s_mov_b32 s54, scc // (op 322) 
s_ashr_i64 s[80:81], s[2:3], 7 // (op 322) 
s_mov_b32 s55, s80 // (op 322) convert

s_sub_u32 s80, s55, 1 // (op 322) 

s_cmp_lt_u32 s19, s80 // (op 322) 
s_mov_b32 s55, scc // (op 322) 

s_and_b32 s80, s54, s55 // (op 322) 
s_cmp_lg_u32 s80, 0 // (op 322) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_322
s_mov_b32 s54, 0 // (op 288) call()
s_mul_i32 s55, s54, s8 // (op 287) 
s_ashr_i64 s[80:81], s[2:3], 7 // (op 287) 
s_mov_b32 s82, s80 // (op 287) convert

s_sub_u32 s80, s82, s19 // (op 287) 

s_sub_u32 s81, s80, 1 // (op 287) 

s_cmp_lt_u32 s55, s81 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 // (op 287) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287:
s_add_u32 s55, s16, 1 // (op 2117) 
v_lshrrev_b32 v16, 5, v1 // (op 2117) 
s_add_u32 s80, s55, s54 // (op 2117) 
s_lshl_b32 s80, s80, 4 // (op 2117) 

v_lshlrev_b32 v17, 2, v16 // (op 2117) 

v_add_lshl_u32 v16, s80, v17, 5 // (op 2117) 


v_and_b32 v17, -32, v1 // (op 2117) 
v_sub_u32 v19, v1, v17 // (op 2117) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 2117) 


s_lshl_b32 s55, s15, 11 // (op 2119) 
s_mov_b32 s76, s44 // (op 2119) reg expression
s_mov_b32 s77, s45 // (op 2119) reg expression
s_mov_b32 s78, s55 // (op 2119) reg expression
s_mov_b32 s79, 131072 // (op 2119) call()

RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326:
s_add_u32 s55, s16, 1 // (op 324) 
s_add_u32 s80, s55, s54 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 324) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 324) 

s_add_u32 s82, s46, s80 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s83, s47, s81 // (op 324) End of Int64 add, unlocking SCC
s_load_dword s55, s[82:83], 0 glc // (op 324) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s55, 0 // (op 326) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 if true)
s_barrier  // (op 330) 
v_cmp_eq_i32 vcc, v1, 0 // (op 329) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329 // (op 329) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329
s_mov_b32 s55, 0 // (op 327) call()
s_add_u32 s80, s16, 1 // (op 328) 
s_add_u32 s82, s80, s54 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s83, 0, 0 // (op 328) End of Int64 add, unlocking SCC
s_lshl_b64 s[82:83], s[82:83], 2 // (op 328) 

s_add_u32 s80, s46, s82 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s81, s47, s83 // (op 328) End of Int64 add, unlocking SCC
s_store_dword s55, s[80:81], 0 glc // (op 328) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329 // (op 329) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329:
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329:




buffer_load_dword v20, v17, s[76:79], 0 offen // (op 341) Load value
buffer_load_dword v21, v17, s[76:79], 0 offen offset:128 // (op 341) Load value
buffer_load_dword v22, v17, s[76:79], 0 offen offset:256 // (op 341) Load value
buffer_load_dword v23, v17, s[76:79], 0 offen offset:384 // (op 341) Load value
v_accvgpr_read v24, a0 // (op 344) reg expression
v_accvgpr_read v25, a1 // (op 344) reg expression
v_accvgpr_read v26, a2 // (op 344) reg expression
v_accvgpr_read v27, a3 // (op 344) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v24, v24, v20 // (op 346) 
s_waitcnt vmcnt(2)
v_add_f32 v25, v25, v21 // (op 346) 
s_waitcnt vmcnt(1)
v_add_f32 v26, v26, v22 // (op 346) 
s_waitcnt vmcnt(0)
v_add_f32 v27, v27, v23 // (op 346) 

v_accvgpr_write a0, v24 // (op 348) reg expression
v_accvgpr_write a1, v25 // (op 348) reg expression
v_accvgpr_write a2, v26 // (op 348) reg expression
v_accvgpr_write a3, v27 // (op 348) reg expression

s_add_u32 s54, s54, 1 // (op 289) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 287) 
s_ashr_i64 s[80:81], s[2:3], 7 // (op 287) 
s_mov_b32 s82, s80 // (op 287) convert

s_sub_u32 s80, s82, s19 // (op 287) 

s_sub_u32 s81, s80, 1 // (op 287) 

s_cmp_lt_u32 s55, s81 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287 // (op 287) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_322
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_322:
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_322:

s_cmp_eq_i32 s18, 0 // (op 365) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365 // (op 365) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v20, a0 // (op 46) 
v_accvgpr_read v21, a1 // (op 46) 
v_accvgpr_read v22, a2 // (op 46) 
v_accvgpr_read v23, a3 // (op 46) 
v_mul_f32 v24, v15, v20 // (op 46) 
v_mul_f32 v25, v15, v21 // (op 46) 
v_mul_f32 v26, v15, v22 // (op 46) 
v_mul_f32 v27, v15, v23 // (op 46) 

v_cvt_f16_f32 v20, v24 // (op 52) 
v_cvt_f16_f32 v21, v25 // (op 52) 
v_cvt_f16_f32 v22, v26 // (op 52) 
v_cvt_f16_f32 v23, v27 // (op 52) 

v_lshrrev_b32 v19, 7, v1 // (op 1834) 
s_mul_i32 s54, s16, s8 // (op 1834) 
s_add_u32 s54, s54, s0 // (op 1834) 
v_and_b32 v16, 63, v1 // (op 1834) 
s_mul_hi_u32 s55, s54, s13 // (op 1834) 
v_lshrrev_b32 v17, 4, v16 // (op 1834) 
s_sub_u32 s76, s54, s55 // (op 1834) 
v_lshlrev_b32 v24, 2, v17 // (op 1834) 

s_lshr_b32 s77, s76, 1 // (op 1834) 

v_lshl_add_u32 v17, v19, 4, v24 // (op 1834) 


s_add_u32 s76, s77, s55 // (op 1834) 


v_lshrrev_b32 v19, 2, v16 // (op 1834) 

s_lshr_b32 s55, s76, s11 // (op 1834) 

v_and_b32 v16, 3, v19 // (op 1834) 

s_and_b32 s76, s11, 2147483648 // (op 1834) 
v_and_b32 v19, 3, v1 // (op 1834) 
s_cmp_eq_i32 s76, 0 // (op 1834) 
s_mov_b32 s77, scc // (op 1834) 

v_lshl_add_u32 v24, v16, 2, v19 // (op 1834) 


s_cmp_eq_i32 s77, 0 // (op 1834) 
s_cselect_b32 s76, s54, s55 // (op 1834) 



v_lshrrev_b32 v16, 6, v1 // (op 1834) 
s_mul_hi_u32 s54, s76, s12 // (op 1834) 
v_and_b32 v19, 1, v16 // (op 1834) 

s_sub_u32 s55, s76, s54 // (op 1834) 
v_lshl_add_u32 v16, v19, 4, v24 // (op 1834) 


s_lshr_b32 s77, s55, 1 // (op 1834) 

s_add_u32 s55, s77, s54 // (op 1834) 


s_lshr_b32 s54, s55, s10 // (op 1834) 

s_and_b32 s55, s10, 2147483648 // (op 1834) 
s_cmp_eq_i32 s55, 0 // (op 1834) 
s_mov_b32 s77, scc // (op 1834) 

s_cmp_eq_i32 s77, 0 // (op 1834) 
s_cselect_b32 s55, s76, s54 // (op 1834) 


s_mul_i32 s54, s55, s14 // (op 1834) 
v_lshl_add_u32 v19, s55, 4, v17 // (op 1834) 


s_sub_u32 s55, s76, s54 // (op 1834) 


v_lshl_add_u32 v17, s55, 5, v16 // (op 1834) 


v_mul_lo_u32 v16, v17, s48 // (op 1834) 

v_add_lshl_u32 v17, v19, v16, 1 // (op 1834) 


s_mul_i32 s55, s48, s23 // (op 1836) most significant: low of low * high
s_mul_i32 s76, s49, s22 // (op 1836) most significant: low of high * low
s_mul_i32 s54, s48, s22 // (op 1836) least significant
s_mul_hi_u32 s77, s48, s22 // (op 1836) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1836) most significant: sum
s_add_u32 s55, s55, s77 // (op 1836) most significant: sum


s_lshl_b64 s[76:77], s[54:55], 1 // (op 1836) 

s_mov_b32 s80, s50 // (op 1836) reg expression
s_mov_b32 s81, s51 // (op 1836) reg expression
s_mov_b32 s82, s76 // (op 1836) reg expression
s_mov_b32 s83, 131072 // (op 1836) call()


v_pack_b32_f16 v24, v20, v21 // (op 54) 
v_pack_b32_f16 v25, v22, v23 // (op 54) 


buffer_store_dwordx2 v[24:25], v17, s[80:83], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[16:17], v18, s[60:63], 0 offen // (op 138) Load value
v_accvgpr_read v20, a0 // (op 134) 
v_accvgpr_read v21, a1 // (op 134) 
v_accvgpr_read v22, a2 // (op 134) 
v_accvgpr_read v23, a3 // (op 134) 
v_mul_f32 v24, v15, v20 // (op 134) 
v_mul_f32 v25, v15, v21 // (op 134) 
v_mul_f32 v26, v15, v22 // (op 134) 
v_mul_f32 v27, v15, v23 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v22, v16 // (op 139) 
v_lshrrev_b32 v23, 16, v16 // (op 139) 
v_cvt_f32_f16 v23, v23 // (op 139) 
v_mul_f32 v28, v14, v22 // (op 139) 
v_mul_f32 v29, v14, v23 // (op 139) 
v_cvt_f32_f16 v22, v17 // (op 139) 
v_lshrrev_b32 v23, 16, v17 // (op 139) 
v_cvt_f32_f16 v23, v23 // (op 139) 
v_mul_f32 v30, v14, v22 // (op 139) 
v_mul_f32 v31, v14, v23 // (op 139) 


v_add_f32 v20, v28, v24 // (op 135) 
v_add_f32 v21, v29, v25 // (op 135) 
v_add_f32 v22, v30, v26 // (op 135) 
v_add_f32 v23, v31, v27 // (op 135) 


v_cvt_f16_f32 v24, v20 // (op 136) 
v_cvt_f16_f32 v25, v21 // (op 136) 
v_cvt_f16_f32 v26, v22 // (op 136) 
v_cvt_f16_f32 v27, v23 // (op 136) 

v_lshrrev_b32 v16, 7, v1 // (op 1851) 
s_mul_i32 s54, s16, s8 // (op 1851) 
s_add_u32 s54, s54, s0 // (op 1851) 
v_and_b32 v17, 63, v1 // (op 1851) 
s_mul_hi_u32 s55, s54, s13 // (op 1851) 
v_lshrrev_b32 v19, 4, v17 // (op 1851) 
s_sub_u32 s76, s54, s55 // (op 1851) 
v_lshlrev_b32 v20, 2, v19 // (op 1851) 

s_lshr_b32 s77, s76, 1 // (op 1851) 

v_lshl_add_u32 v19, v16, 4, v20 // (op 1851) 


s_add_u32 s76, s77, s55 // (op 1851) 


v_lshrrev_b32 v16, 2, v17 // (op 1851) 

s_lshr_b32 s55, s76, s11 // (op 1851) 

v_and_b32 v17, 3, v16 // (op 1851) 

s_and_b32 s76, s11, 2147483648 // (op 1851) 
v_and_b32 v16, 3, v1 // (op 1851) 
s_cmp_eq_i32 s76, 0 // (op 1851) 
s_mov_b32 s77, scc // (op 1851) 

v_lshl_add_u32 v20, v17, 2, v16 // (op 1851) 


s_cmp_eq_i32 s77, 0 // (op 1851) 
s_cselect_b32 s76, s54, s55 // (op 1851) 



v_lshrrev_b32 v16, 6, v1 // (op 1851) 
s_mul_hi_u32 s54, s76, s12 // (op 1851) 
v_and_b32 v17, 1, v16 // (op 1851) 

s_sub_u32 s55, s76, s54 // (op 1851) 
v_lshl_add_u32 v16, v17, 4, v20 // (op 1851) 


s_lshr_b32 s77, s55, 1 // (op 1851) 

s_add_u32 s55, s77, s54 // (op 1851) 


s_lshr_b32 s54, s55, s10 // (op 1851) 

s_and_b32 s55, s10, 2147483648 // (op 1851) 
s_cmp_eq_i32 s55, 0 // (op 1851) 
s_mov_b32 s77, scc // (op 1851) 

s_cmp_eq_i32 s77, 0 // (op 1851) 
s_cselect_b32 s55, s76, s54 // (op 1851) 


s_mul_i32 s54, s55, s14 // (op 1851) 
v_lshl_add_u32 v17, s55, 4, v19 // (op 1851) 


s_sub_u32 s55, s76, s54 // (op 1851) 


v_lshl_add_u32 v19, s55, 5, v16 // (op 1851) 


v_mul_lo_u32 v16, v19, s48 // (op 1851) 

v_add_lshl_u32 v19, v17, v16, 1 // (op 1851) 


s_mul_i32 s55, s48, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s76, s49, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s54, s48, s22 // (op 1853) least significant
s_mul_hi_u32 s77, s48, s22 // (op 1853) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1853) most significant: sum
s_add_u32 s55, s55, s77 // (op 1853) most significant: sum


s_lshl_b64 s[76:77], s[54:55], 1 // (op 1853) 

s_mov_b32 s80, s50 // (op 1853) reg expression
s_mov_b32 s81, s51 // (op 1853) reg expression
s_mov_b32 s82, s76 // (op 1853) reg expression
s_mov_b32 s83, 131072 // (op 1853) call()


v_pack_b32_f16 v16, v24, v25 // (op 137) 
v_pack_b32_f16 v17, v26, v27 // (op 137) 


buffer_store_dwordx2 v[16:17], v19, s[80:83], 0 offen // (op 137) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365 // (op 365) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365:
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365:

s_add_u32 s0, s0, s7 // (op 280) 
s_mul_i32 s54, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_sub_u32 s76, s9, s54 // (op 278) 

s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s54, 0, s76 // (op 278) 


s_cmp_lt_u32 s8, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s76, s54, s8 // (op 278) 


s_cmp_lt_u32 s0, s76 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278 // (op 278) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278:















s_mov_b32 s0, 0 // (op 419) call()
s_mov_b32 s1, 0 // (op 420) call()
s_mov_b32 s0, 0 // (op 434) call()
s_mul_i32 s15, s52, s16 // (op 433) 
s_cmp_gt_u32 s53, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s53, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s52, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s52 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 // (op 433) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433:
v_lshrrev_b32 v0, 7, v1 // (op 1619) 
s_mul_i32 s15, s16, s52 // (op 1619) 
s_add_u32 s15, s15, s0 // (op 1619) 
v_and_b32 v3, 63, v1 // (op 1619) 
s_add_u32 s6, s15, s9 // (op 1619) 

v_lshrrev_b32 v4, 4, v3 // (op 1619) 
s_mul_hi_u32 s15, s6, s13 // (op 1619) 
v_lshlrev_b32 v11, 2, v4 // (op 1619) 

s_sub_u32 s7, s6, s15 // (op 1619) 
v_lshl_add_u32 v4, v0, 4, v11 // (op 1619) 


s_lshr_b32 s8, s7, 1 // (op 1619) 

v_lshrrev_b32 v0, 2, v3 // (op 1619) 

s_add_u32 s7, s8, s15 // (op 1619) 


v_and_b32 v3, 3, v0 // (op 1619) 

s_lshr_b32 s8, s7, s11 // (op 1619) 

v_and_b32 v0, 3, v1 // (op 1619) 
s_and_b32 s7, s11, 2147483648 // (op 1619) 
v_lshl_add_u32 v11, v3, 2, v0 // (op 1619) 


s_cmp_eq_i32 s7, 0 // (op 1619) 
s_mov_b32 s15, scc // (op 1619) 

v_lshrrev_b32 v0, 6, v1 // (op 1619) 
s_cmp_eq_i32 s15, 0 // (op 1619) 
s_cselect_b32 s7, s6, s8 // (op 1619) 



v_and_b32 v3, 1, v0 // (op 1619) 

s_mul_hi_u32 s6, s7, s12 // (op 1619) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 1619) 


s_sub_u32 s8, s7, s6 // (op 1619) 
s_lshr_b32 s15, s8, 1 // (op 1619) 

s_add_u32 s8, s15, s6 // (op 1619) 


s_lshr_b32 s6, s8, s10 // (op 1619) 

s_and_b32 s8, s10, 2147483648 // (op 1619) 
s_cmp_eq_i32 s8, 0 // (op 1619) 
s_mov_b32 s15, scc // (op 1619) 

s_cmp_eq_i32 s15, 0 // (op 1619) 
s_cselect_b32 s8, s7, s6 // (op 1619) 


s_mul_i32 s6, s8, s14 // (op 1619) 
v_lshl_add_u32 v3, s8, 4, v4 // (op 1619) 


s_sub_u32 s8, s7, s6 // (op 1619) 


v_lshl_add_u32 v4, s8, 5, v0 // (op 1619) 


v_mul_lo_u32 v0, v4, s4 // (op 1619) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 1619) 


s_mul_i32 s19, s4, s23 // (op 1621) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1621) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1621) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1621) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1621) most significant: sum
s_add_u32 s19, s19, s6 // (op 1621) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 1 // (op 1621) 

s_mov_b32 s60, s20 // (op 1621) reg expression
s_mov_b32 s61, s21 // (op 1621) reg expression
s_mov_b32 s62, s54 // (op 1621) reg expression
s_mov_b32 s63, 131072 // (op 1621) call()

v_lshrrev_b32 v0, 7, v1 // (op 1954) 
v_and_b32 v3, 15, v1 // (op 1954) 
v_lshl_add_u32 v11, v0, 4, v3 // (op 1954) 


v_and_b32 v0, 48, v1 // (op 1954) 
v_lshl_add_u32 v3, v11, 7, v0 // (op 1954) 


v_and_b32 v0, 48, v1 // (op 2019) 
v_and_b32 v11, 63, v1 // (op 2019) 
v_sub_u32 v12, v11, v0 // (op 2019) 

v_lshrrev_b32 v11, 1, v12 // (op 2019) 
v_add_lshl_u32 v13, v0, v11, 5 // (op 2019) 


v_and_b32 v0, 1, v12 // (op 2019) 

v_lshlrev_b32 v11, 3, v0 // (op 2019) 

v_lshrrev_b32 v0, 6, v1 // (op 2019) 
v_and_b32 v12, 1, v0 // (op 2019) 

v_lshl_add_u32 v0, v12, 4, v11 // (op 2019) 


v_add_u32 v11, v13, v0 // (op 2019) 


v_lshrrev_b32 v0, 6, v1 // (op 2085) 
v_and_b32 v12, 1, v0 // (op 2085) 

v_and_b32 v0, 15, v1 // (op 2085) 
v_lshl_add_u32 v13, v12, 4, v0 // (op 2085) 


v_and_b32 v0, 63, v1 // (op 2085) 
v_lshrrev_b32 v12, 4, v0 // (op 2085) 

v_lshl_add_u32 v0, v12, 5, v13 // (op 2085) 


v_accvgpr_write a0, 0.00000 // (op 174) call()
v_accvgpr_write a1, 0.00000 // (op 174) call()
v_accvgpr_write a2, 0.00000 // (op 174) call()
v_accvgpr_write a3, 0.00000 // (op 174) call()
v_lshrrev_b32 v12, 3, v1 // (op 1179) 
s_mul_i32 s15, s16, s52 // (op 1179) 
s_add_u32 s15, s15, s0 // (op 1179) 
v_and_b32 v13, -8, v1 // (op 1179) 
s_add_u32 s6, s15, s9 // (op 1179) 

v_sub_u32 v16, v1, v13 // (op 1179) 

s_mul_hi_u32 s15, s6, s13 // (op 1179) 
v_lshlrev_b32 v13, 4, v16 // (op 1179) 

s_sub_u32 s7, s6, s15 // (op 1179) 
s_lshr_b32 s8, s7, 1 // (op 1179) 

s_add_u32 s7, s8, s15 // (op 1179) 


s_lshr_b32 s8, s7, s11 // (op 1179) 

s_and_b32 s7, s11, 2147483648 // (op 1179) 
s_cmp_eq_i32 s7, 0 // (op 1179) 
s_mov_b32 s15, scc // (op 1179) 

s_cmp_eq_i32 s15, 0 // (op 1179) 
s_cselect_b32 s7, s6, s8 // (op 1179) 


s_mul_hi_u32 s8, s7, s12 // (op 1179) 
s_sub_u32 s15, s7, s8 // (op 1179) 
s_lshr_b32 s18, s15, 1 // (op 1179) 

s_add_u32 s15, s18, s8 // (op 1179) 


s_lshr_b32 s8, s15, s10 // (op 1179) 

s_and_b32 s15, s10, 2147483648 // (op 1179) 
s_cmp_eq_i32 s15, 0 // (op 1179) 
s_mov_b32 s18, scc // (op 1179) 

s_cmp_eq_i32 s18, 0 // (op 1179) 
s_cselect_b32 s15, s7, s8 // (op 1179) 


v_lshl_add_u32 v16, s15, 4, v12 // (op 1179) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1179) 
s_mul_i32 s8, s7, s18 // (op 1179) 


s_sub_u32 s7, s6, s8 // (op 1179) 


v_lshl_add_u32 v12, s7, 7, v13 // (op 1179) 


v_mov_b32 v19, 0 // (op 1179) convert
v_mov_b32 v18, v12 // (op 1179) convert

v_mul_lo_u32 v13, v16, s25 // (op 1179) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1179) most significant: high of low * low
v_mul_lo_u32 v12, v16, s24 // (op 1179) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 1179) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v18 // (op 1179) least significant half
v_addc_co_u32 v13, s[18:19], v13, v19, s[18:19] // (op 1179) most significant half



s_mul_i32 s19, s24, s29 // (op 1181) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1181) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1181) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1181) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1181) most significant: sum
s_add_u32 s19, s19, s6 // (op 1181) most significant: sum


s_mov_b32 s56, s26 // (op 1181) reg expression
s_mov_b32 s57, s27 // (op 1181) reg expression
s_mov_b32 s58, s18 // (op 1181) reg expression
s_mov_b32 s59, 131072 // (op 1181) call()

v_lshrrev_b32 v16, 3, v1 // (op 1198) 
s_mul_i32 s15, s16, s52 // (op 1198) 
s_add_u32 s15, s15, s0 // (op 1198) 
v_and_b32 v17, -8, v1 // (op 1198) 
s_add_u32 s6, s15, s9 // (op 1198) 

v_sub_u32 v18, v1, v17 // (op 1198) 

s_mul_hi_u32 s15, s6, s13 // (op 1198) 
v_lshlrev_b32 v17, 4, v18 // (op 1198) 

s_sub_u32 s7, s6, s15 // (op 1198) 
s_lshr_b32 s8, s7, 1 // (op 1198) 

s_add_u32 s7, s8, s15 // (op 1198) 


s_lshr_b32 s8, s7, s11 // (op 1198) 

s_and_b32 s7, s11, 2147483648 // (op 1198) 
s_cmp_eq_i32 s7, 0 // (op 1198) 
s_mov_b32 s15, scc // (op 1198) 

s_cmp_eq_i32 s15, 0 // (op 1198) 
s_cselect_b32 s7, s6, s8 // (op 1198) 


s_mul_hi_u32 s8, s7, s12 // (op 1198) 
s_sub_u32 s15, s7, s8 // (op 1198) 
s_lshr_b32 s18, s15, 1 // (op 1198) 

s_add_u32 s15, s18, s8 // (op 1198) 


s_lshr_b32 s8, s15, s10 // (op 1198) 

s_and_b32 s15, s10, 2147483648 // (op 1198) 
s_cmp_eq_i32 s15, 0 // (op 1198) 
s_mov_b32 s18, scc // (op 1198) 

s_cmp_eq_i32 s18, 0 // (op 1198) 
s_cselect_b32 s15, s7, s8 // (op 1198) 


v_lshl_add_u32 v18, s15, 4, v16 // (op 1198) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1198) 
s_mul_i32 s8, s7, s18 // (op 1198) 


s_sub_u32 s7, s6, s8 // (op 1198) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 1198) 


v_mov_b32 v21, 0 // (op 1198) convert
v_mov_b32 v20, v16 // (op 1198) convert

v_mul_lo_u32 v17, v18, s25 // (op 1198) most significant: low of low * high
v_mul_hi_u32 v19, v18, s24 // (op 1198) most significant: high of low * low
v_mul_lo_u32 v16, v18, s24 // (op 1198) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 1198) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v20 // (op 1198) least significant half
v_addc_co_u32 v17, s[18:19], v17, v21, s[18:19] // (op 1198) most significant half



s_mul_i32 s19, s24, s29 // (op 1200) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1200) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1200) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1200) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1200) most significant: sum
s_add_u32 s19, s19, s6 // (op 1200) most significant: sum


s_mov_b32 s56, s26 // (op 1200) reg expression
s_mov_b32 s57, s27 // (op 1200) reg expression
s_mov_b32 s58, s18 // (op 1200) reg expression
s_mov_b32 s59, 131072 // (op 1200) call()

v_and_b32 v18, -2, v1 // (op 1307) 
s_mul_i32 s15, s16, s52 // (op 1307) 
s_add_u32 s15, s15, s0 // (op 1307) 
v_sub_u32 v19, v1, v18 // (op 1307) 

s_add_u32 s6, s15, s9 // (op 1307) 

v_lshlrev_b32 v18, 4, v19 // (op 1307) 

s_mul_hi_u32 s15, s6, s13 // (op 1307) 
v_lshrrev_b32 v19, 1, v1 // (op 1307) 
s_sub_u32 s7, s6, s15 // (op 1307) 
s_lshr_b32 s8, s7, 1 // (op 1307) 

s_add_u32 s7, s8, s15 // (op 1307) 


s_lshr_b32 s8, s7, s11 // (op 1307) 

s_and_b32 s7, s11, 2147483648 // (op 1307) 
s_cmp_eq_i32 s7, 0 // (op 1307) 
s_mov_b32 s15, scc // (op 1307) 

s_cmp_eq_i32 s15, 0 // (op 1307) 
s_cselect_b32 s7, s6, s8 // (op 1307) 


s_mul_hi_u32 s8, s7, s12 // (op 1307) 
s_sub_u32 s15, s7, s8 // (op 1307) 
s_lshr_b32 s18, s15, 1 // (op 1307) 

s_add_u32 s15, s18, s8 // (op 1307) 


s_lshr_b32 s8, s15, s10 // (op 1307) 

s_and_b32 s15, s10, 2147483648 // (op 1307) 
s_cmp_eq_i32 s15, 0 // (op 1307) 
s_mov_b32 s18, scc // (op 1307) 

s_cmp_eq_i32 s18, 0 // (op 1307) 
s_cselect_b32 s15, s7, s8 // (op 1307) 


s_mul_i32 s8, s15, s14 // (op 1307) 

s_sub_u32 s15, s7, s8 // (op 1307) 

v_lshl_add_u32 v20, s15, 5, v18 // (op 1307) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1307) 
v_mov_b32 v23, 0 // (op 1307) convert
v_mov_b32 v22, v20 // (op 1307) convert

s_mul_i32 s8, s7, s18 // (op 1307) 


s_sub_u32 s7, s6, s8 // (op 1307) 


v_lshl_add_u32 v18, s7, 7, v19 // (op 1307) 


v_mul_lo_u32 v21, v18, s31 // (op 1307) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 1307) most significant: high of low * low
v_mul_lo_u32 v20, v18, s30 // (op 1307) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1307) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v22 // (op 1307) least significant half
v_addc_co_u32 v21, s[18:19], v21, v23, s[18:19] // (op 1307) most significant half



s_mul_i32 s19, s30, s3 // (op 1309) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1309) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1309) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1309) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1309) most significant: sum
s_add_u32 s19, s19, s6 // (op 1309) most significant: sum


s_mov_b32 s64, s32 // (op 1309) reg expression
s_mov_b32 s65, s33 // (op 1309) reg expression
s_mov_b32 s66, s18 // (op 1309) reg expression
s_mov_b32 s67, 131072 // (op 1309) call()

v_and_b32 v18, -2, v1 // (op 1326) 
s_mul_i32 s15, s16, s52 // (op 1326) 
s_add_u32 s15, s15, s0 // (op 1326) 
v_sub_u32 v19, v1, v18 // (op 1326) 

s_add_u32 s6, s15, s9 // (op 1326) 

v_lshlrev_b32 v18, 4, v19 // (op 1326) 

s_mul_hi_u32 s15, s6, s13 // (op 1326) 
v_lshrrev_b32 v19, 1, v1 // (op 1326) 
s_sub_u32 s7, s6, s15 // (op 1326) 
s_lshr_b32 s8, s7, 1 // (op 1326) 

s_add_u32 s7, s8, s15 // (op 1326) 


s_lshr_b32 s8, s7, s11 // (op 1326) 

s_and_b32 s7, s11, 2147483648 // (op 1326) 
s_cmp_eq_i32 s7, 0 // (op 1326) 
s_mov_b32 s15, scc // (op 1326) 

s_cmp_eq_i32 s15, 0 // (op 1326) 
s_cselect_b32 s7, s6, s8 // (op 1326) 


s_mul_hi_u32 s8, s7, s12 // (op 1326) 
s_sub_u32 s15, s7, s8 // (op 1326) 
s_lshr_b32 s18, s15, 1 // (op 1326) 

s_add_u32 s15, s18, s8 // (op 1326) 


s_lshr_b32 s8, s15, s10 // (op 1326) 

s_and_b32 s15, s10, 2147483648 // (op 1326) 
s_cmp_eq_i32 s15, 0 // (op 1326) 
s_mov_b32 s18, scc // (op 1326) 

s_cmp_eq_i32 s18, 0 // (op 1326) 
s_cselect_b32 s15, s7, s8 // (op 1326) 


s_mul_i32 s8, s15, s14 // (op 1326) 

s_sub_u32 s15, s7, s8 // (op 1326) 

v_lshl_add_u32 v22, s15, 5, v18 // (op 1326) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1326) 
v_mov_b32 v25, 0 // (op 1326) convert
v_mov_b32 v24, v22 // (op 1326) convert

s_mul_i32 s8, s7, s18 // (op 1326) 


s_sub_u32 s7, s6, s8 // (op 1326) 


v_lshl_add_u32 v18, s7, 7, v19 // (op 1326) 


v_mul_lo_u32 v23, v18, s31 // (op 1326) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 1326) most significant: high of low * low
v_mul_lo_u32 v22, v18, s30 // (op 1326) least significant: low of low * low
v_add_u32 v23, v23, v19 // (op 1326) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v24 // (op 1326) least significant half
v_addc_co_u32 v23, s[18:19], v23, v25, s[18:19] // (op 1326) most significant half



s_mul_i32 s19, s30, s3 // (op 1328) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1328) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1328) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1328) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1328) most significant: sum
s_add_u32 s19, s19, s6 // (op 1328) most significant: sum


s_mov_b32 s64, s32 // (op 1328) reg expression
s_mov_b32 s65, s33 // (op 1328) reg expression
s_mov_b32 s66, s18 // (op 1328) reg expression
s_mov_b32 s67, 131072 // (op 1328) call()

v_lshrrev_b32 v18, 7, v1 // (op 1435) 
s_mul_i32 s15, s16, s52 // (op 1435) 
s_add_u32 s15, s15, s0 // (op 1435) 
v_and_b32 v19, 15, v1 // (op 1435) 
s_add_u32 s6, s15, s9 // (op 1435) 

v_lshl_add_u32 v24, v18, 4, v19 // (op 1435) 


s_mul_hi_u32 s15, s6, s13 // (op 1435) 
v_and_b32 v18, 63, v1 // (op 1435) 
s_sub_u32 s7, s6, s15 // (op 1435) 
v_lshrrev_b32 v19, 4, v18 // (op 1435) 

s_lshr_b32 s8, s7, 1 // (op 1435) 

s_add_u32 s7, s8, s15 // (op 1435) 


s_lshr_b32 s8, s7, s11 // (op 1435) 

s_and_b32 s7, s11, 2147483648 // (op 1435) 
s_cmp_eq_i32 s7, 0 // (op 1435) 
s_mov_b32 s15, scc // (op 1435) 

s_cmp_eq_i32 s15, 0 // (op 1435) 
s_cselect_b32 s7, s6, s8 // (op 1435) 


s_mul_hi_u32 s8, s7, s12 // (op 1435) 
s_sub_u32 s15, s7, s8 // (op 1435) 
s_lshr_b32 s18, s15, 1 // (op 1435) 

s_add_u32 s15, s18, s8 // (op 1435) 


s_lshr_b32 s8, s15, s10 // (op 1435) 

s_and_b32 s15, s10, 2147483648 // (op 1435) 
s_cmp_eq_i32 s15, 0 // (op 1435) 
s_mov_b32 s18, scc // (op 1435) 

s_cmp_eq_i32 s18, 0 // (op 1435) 
s_cselect_b32 s15, s7, s8 // (op 1435) 


v_lshl_add_u32 v18, s15, 4, v24 // (op 1435) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1435) 
s_mul_i32 s8, s7, s18 // (op 1435) 


s_sub_u32 s7, s6, s8 // (op 1435) 


v_lshl_add_u32 v24, s7, 2, v19 // (op 1435) 


v_mul_lo_u32 v19, v18, s34 // (op 1435) 
v_add_u32 v19, v19, v24 // (op 1435) 


s_mul_i32 s19, s34, s29 // (op 1437) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1437) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1437) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1437) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1437) most significant: sum
s_add_u32 s19, s19, s6 // (op 1437) most significant: sum


s_mov_b32 s68, s36 // (op 1437) reg expression
s_mov_b32 s69, s37 // (op 1437) reg expression
s_mov_b32 s70, s18 // (op 1437) reg expression
s_mov_b32 s71, 131072 // (op 1437) call()

v_lshrrev_b32 v18, 7, v1 // (op 1454) 
s_mul_i32 s15, s16, s52 // (op 1454) 
s_add_u32 s15, s15, s0 // (op 1454) 
v_and_b32 v24, 15, v1 // (op 1454) 
s_add_u32 s6, s15, s9 // (op 1454) 

v_lshl_add_u32 v25, v18, 4, v24 // (op 1454) 


s_mul_hi_u32 s15, s6, s13 // (op 1454) 
v_and_b32 v18, 63, v1 // (op 1454) 
s_sub_u32 s7, s6, s15 // (op 1454) 
v_lshrrev_b32 v24, 4, v18 // (op 1454) 

s_lshr_b32 s8, s7, 1 // (op 1454) 

s_add_u32 s7, s8, s15 // (op 1454) 


s_lshr_b32 s8, s7, s11 // (op 1454) 

s_and_b32 s7, s11, 2147483648 // (op 1454) 
s_cmp_eq_i32 s7, 0 // (op 1454) 
s_mov_b32 s15, scc // (op 1454) 

s_cmp_eq_i32 s15, 0 // (op 1454) 
s_cselect_b32 s7, s6, s8 // (op 1454) 


s_mul_hi_u32 s8, s7, s12 // (op 1454) 
s_sub_u32 s15, s7, s8 // (op 1454) 
s_lshr_b32 s18, s15, 1 // (op 1454) 

s_add_u32 s15, s18, s8 // (op 1454) 


s_lshr_b32 s8, s15, s10 // (op 1454) 

s_and_b32 s15, s10, 2147483648 // (op 1454) 
s_cmp_eq_i32 s15, 0 // (op 1454) 
s_mov_b32 s18, scc // (op 1454) 

s_cmp_eq_i32 s18, 0 // (op 1454) 
s_cselect_b32 s15, s7, s8 // (op 1454) 


v_lshl_add_u32 v18, s15, 4, v25 // (op 1454) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1454) 
s_mul_i32 s8, s7, s18 // (op 1454) 


s_sub_u32 s7, s6, s8 // (op 1454) 


v_lshl_add_u32 v25, s7, 2, v24 // (op 1454) 


v_mul_lo_u32 v24, v18, s34 // (op 1454) 
v_add_u32 v24, v24, v25 // (op 1454) 


s_mul_i32 s19, s34, s29 // (op 1456) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1456) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1456) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1456) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1456) most significant: sum
s_add_u32 s19, s19, s6 // (op 1456) most significant: sum


s_mov_b32 s68, s36 // (op 1456) reg expression
s_mov_b32 s69, s37 // (op 1456) reg expression
s_mov_b32 s70, s18 // (op 1456) reg expression
s_mov_b32 s71, 131072 // (op 1456) call()

v_and_b32 v18, -32, v1 // (op 1563) 
s_mul_i32 s15, s16, s52 // (op 1563) 
s_add_u32 s15, s15, s0 // (op 1563) 
v_sub_u32 v25, v1, v18 // (op 1563) 

s_add_u32 s6, s15, s9 // (op 1563) 

v_lshrrev_b32 v18, 5, v1 // (op 1563) 
s_mul_hi_u32 s15, s6, s13 // (op 1563) 
s_sub_u32 s7, s6, s15 // (op 1563) 
s_lshr_b32 s8, s7, 1 // (op 1563) 

s_add_u32 s7, s8, s15 // (op 1563) 


s_lshr_b32 s8, s7, s11 // (op 1563) 

s_and_b32 s7, s11, 2147483648 // (op 1563) 
s_cmp_eq_i32 s7, 0 // (op 1563) 
s_mov_b32 s15, scc // (op 1563) 

s_cmp_eq_i32 s15, 0 // (op 1563) 
s_cselect_b32 s7, s6, s8 // (op 1563) 


s_mul_hi_u32 s8, s7, s12 // (op 1563) 
s_sub_u32 s15, s7, s8 // (op 1563) 
s_lshr_b32 s18, s15, 1 // (op 1563) 

s_add_u32 s15, s18, s8 // (op 1563) 


s_lshr_b32 s8, s15, s10 // (op 1563) 

s_and_b32 s15, s10, 2147483648 // (op 1563) 
s_cmp_eq_i32 s15, 0 // (op 1563) 
s_mov_b32 s18, scc // (op 1563) 

s_cmp_eq_i32 s18, 0 // (op 1563) 
s_cselect_b32 s15, s7, s8 // (op 1563) 


s_mul_i32 s8, s15, s14 // (op 1563) 

s_sub_u32 s15, s7, s8 // (op 1563) 

v_lshl_add_u32 v26, s15, 5, v25 // (op 1563) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1563) 
s_mul_i32 s8, s7, s18 // (op 1563) 


s_sub_u32 s7, s6, s8 // (op 1563) 


v_lshl_add_u32 v25, s7, 2, v18 // (op 1563) 


v_mul_lo_u32 v18, v25, s38 // (op 1563) 
v_add_u32 v18, v18, v26 // (op 1563) 


s_mul_i32 s19, s38, s43 // (op 1565) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1565) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1565) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1565) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1565) most significant: sum
s_add_u32 s19, s19, s6 // (op 1565) most significant: sum


s_mov_b32 s72, s40 // (op 1565) reg expression
s_mov_b32 s73, s41 // (op 1565) reg expression
s_mov_b32 s74, s18 // (op 1565) reg expression
s_mov_b32 s75, 131072 // (op 1565) call()

v_and_b32 v25, -32, v1 // (op 1582) 
s_mul_i32 s15, s16, s52 // (op 1582) 
s_add_u32 s15, s15, s0 // (op 1582) 
v_sub_u32 v26, v1, v25 // (op 1582) 

s_add_u32 s6, s15, s9 // (op 1582) 

v_lshrrev_b32 v25, 5, v1 // (op 1582) 
s_mul_hi_u32 s15, s6, s13 // (op 1582) 
s_sub_u32 s7, s6, s15 // (op 1582) 
s_lshr_b32 s8, s7, 1 // (op 1582) 

s_add_u32 s7, s8, s15 // (op 1582) 


s_lshr_b32 s8, s7, s11 // (op 1582) 

s_and_b32 s7, s11, 2147483648 // (op 1582) 
s_cmp_eq_i32 s7, 0 // (op 1582) 
s_mov_b32 s15, scc // (op 1582) 

s_cmp_eq_i32 s15, 0 // (op 1582) 
s_cselect_b32 s7, s6, s8 // (op 1582) 


s_mul_hi_u32 s8, s7, s12 // (op 1582) 
s_sub_u32 s15, s7, s8 // (op 1582) 
s_lshr_b32 s18, s15, 1 // (op 1582) 

s_add_u32 s15, s18, s8 // (op 1582) 


s_lshr_b32 s8, s15, s10 // (op 1582) 

s_and_b32 s15, s10, 2147483648 // (op 1582) 
s_cmp_eq_i32 s15, 0 // (op 1582) 
s_mov_b32 s18, scc // (op 1582) 

s_cmp_eq_i32 s18, 0 // (op 1582) 
s_cselect_b32 s15, s7, s8 // (op 1582) 


s_mul_i32 s8, s15, s14 // (op 1582) 

s_sub_u32 s15, s7, s8 // (op 1582) 

v_lshl_add_u32 v27, s15, 5, v26 // (op 1582) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1582) 
s_mul_i32 s8, s7, s18 // (op 1582) 


s_sub_u32 s7, s6, s8 // (op 1582) 


v_lshl_add_u32 v26, s7, 2, v25 // (op 1582) 


v_mul_lo_u32 v25, v26, s38 // (op 1582) 
v_add_u32 v25, v25, v27 // (op 1582) 


s_mul_i32 s19, s38, s43 // (op 1584) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1584) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1584) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1584) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1584) most significant: sum
s_add_u32 s19, s19, s6 // (op 1584) most significant: sum


s_mov_b32 s72, s40 // (op 1584) reg expression
s_mov_b32 s73, s41 // (op 1584) reg expression
s_mov_b32 s74, s18 // (op 1584) reg expression
s_mov_b32 s75, 131072 // (op 1584) call()




buffer_load_ubyte v26, v19, s[68:71], 0 offen offset: 0 // (op 894) Load value
v_and_b32 v27, -2, v1 // (op 1995) 
v_sub_u32 v28, v1, v27 // (op 1995) 

v_lshlrev_b32 v27, 4, v28 // (op 1995) 

v_lshrrev_b32 v28, 1, v1 // (op 1995) 
v_lshl_add_u32 v29, v28, 5, v27 // (op 1995) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1995) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2474) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1059) 
v_mov_b64 v[28:29], s[18:19] // (op 1059) convert


v_mov_b64 v[30:31], v[20:21] // (op 1059) reg expression
s_mov_b32 m0, s15 // (op 1059) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 1059) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1059) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1059) most significant half

s_add_u32 m0, m0, 2048 // (op 1059) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 1059) Load value direct to lds





v_mov_b32 v27, v18 // (op 901) reg expression

buffer_load_ubyte v28, v27, s[72:75], 0 offen offset: 0 // (op 901) Load value

v_and_b32 v27, -8, v1 // (op 1930) 
v_sub_u32 v29, v1, v27 // (op 1930) 

v_lshlrev_b32 v27, 4, v29 // (op 1930) 

v_lshrrev_b32 v29, 3, v1 // (op 1930) 
v_lshl_add_u32 v30, v29, 7, v27 // (op 1930) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 1930) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2455) 


s_nop 0
s_add_u32 s6, 4096, s15 // (op 1064) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1064) reg expression
buffer_load_dwordx4 v12, s[56:59], 0 offen offset: 0 lds // (op 1064) Load value direct to lds





buffer_load_ubyte v27, v19, s[68:71], 0 offen offset:4 // (op 908) Load value

v_and_b32 v19, -2, v1 // (op 2186) 
v_sub_u32 v29, v1, v19 // (op 2186) 

v_lshlrev_b32 v19, 4, v29 // (op 2186) 

v_lshrrev_b32 v29, 1, v1 // (op 2186) 
v_lshl_add_u32 v30, v29, 5, v19 // (op 2186) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2186) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2522) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1069) 
v_mov_b64 v[30:31], s[18:19] // (op 1069) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 1069) 
v_mov_b32 v19, s18 // (op 1069) 
v_mov_b32 v29, s19 // (op 1069) 
v_add_co_u32 v32, s[54:55], v20, v19 // (op 1069) least significant half
v_addc_co_u32 v33, s[54:55], v21, v29, s[54:55] // (op 1069) most significant half




s_add_u32 s6, 6144, s15 // (op 1069) 
s_mov_b32 m0, s6 // (op 1069) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1069) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1069) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1069) most significant half

s_add_u32 m0, m0, 2048 // (op 1069) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1069) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 915) 
v_add_u32 v19, v18, s15 // (op 915) 


buffer_load_ubyte v20, v19, s[72:75], 0 offen offset: 0 // (op 915) Load value


v_and_b32 v21, -8, v1 // (op 2147) 
v_sub_u32 v18, v1, v21 // (op 2147) 

v_lshlrev_b32 v19, 4, v18 // (op 2147) 

v_lshrrev_b32 v18, 3, v1 // (op 2147) 
v_lshl_add_u32 v21, v18, 7, v19 // (op 2147) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2147) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2513) 


s_nop 0
s_add_u32 s6, 10240, s15 // (op 1071) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1071) reg expression
s_mov_b32 s7, 128 // (op 1071) call()
buffer_load_dwordx4 v12, s[56:59], s7 offen offset: 0 lds // (op 1071) Load value direct to lds




v_and_b32 v21, -32, v1 // (op 2061) 
v_sub_u32 v12, v1, v21 // (op 2061) 

v_and_b32 v13, 4294967264, v1 // (op 2061) 
v_add_u32 v21, v13, v12 // (op 2061) 


s_barrier  // (op 2493) 



s_waitcnt vmcnt(6)
ds_write_b8 v21, v28 offset:12288 // (op 921) Store local data 

s_barrier  // (op 2471) 



ds_read_b64_tr_b8 v[30:31], v11 // (op 938) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v11 offset:256 // (op 938) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v11 offset:2048 // (op 938) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v11 offset:2304 // (op 938) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2490) 


ds_read_u8 v21, v0 offset:12288 // (op 941) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 2452) 



ds_read_b128 v[38:41], v3 offset:4096 // (op 935) Load local data 
ds_read_b128 v[42:45], v3 offset:4160 // (op 935) Load local data 
v_and_b32 v29, -8, v1 // (op 1902) 
v_sub_u32 v12, v1, v29 // (op 1902) 

v_lshlrev_b32 v13, 4, v12 // (op 1902) 

v_lshrrev_b32 v12, 3, v1 // (op 1902) 
v_lshl_add_u32 v29, v12, 7, v13 // (op 1902) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1902) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -2, v1 // (op 1967) 
v_sub_u32 v12, v1, v29 // (op 1967) 

v_lshlrev_b32 v13, 4, v12 // (op 1967) 

v_lshrrev_b32 v12, 1, v1 // (op 1967) 
v_lshl_add_u32 v29, v12, 5, v13 // (op 1967) 


s_nop 0
v_readfirstlane_b32 s6, v29 // (op 1967) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -32, v1 // (op 2032) 
v_sub_u32 v12, v1, v29 // (op 2032) 

v_and_b32 v13, 4294967264, v1 // (op 2032) 
v_add_u32 v29, v13, v12 // (op 2032) 


v_and_b32 v12, -8, v1 // (op 2132) 
v_sub_u32 v13, v1, v12 // (op 2132) 

v_lshlrev_b32 v12, 4, v13 // (op 2132) 

v_lshrrev_b32 v13, 3, v1 // (op 2132) 
v_lshl_add_u32 v18, v13, 7, v12 // (op 2132) 


s_nop 0
v_readfirstlane_b32 s7, v18 // (op 2132) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 2171) 
v_sub_u32 v13, v1, v12 // (op 2171) 

v_lshlrev_b32 v12, 4, v13 // (op 2171) 

v_lshrrev_b32 v13, 1, v1 // (op 2171) 
v_lshl_add_u32 v18, v13, 5, v12 // (op 2171) 


s_nop 0
v_readfirstlane_b32 s8, v18 // (op 2171) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -32, v1 // (op 2210) 
v_sub_u32 v13, v1, v12 // (op 2210) 

v_and_b32 v12, 4294967264, v1 // (op 2210) 
v_add_u32 v18, v12, v13 // (op 2210) 


s_mov_b32 s1, 0 // (op 176) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 175) 
s_mov_b32 s44, s18 // (op 175) convert

s_and_b32 s18, s44, 4294967294 // (op 175) 

s_cmp_lt_u32 s1, s18 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_175 // (op 175) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_175 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_175:
s_add_u32 s1, s1, 2 // (op 177) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v26, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 188) 


v_mov_b32 v19, v24 // (op 185) 

buffer_load_ubyte v26, v19, s[68:71], 0 offen offset:8 // (op 185) Load value
s_barrier  // (op 2477) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 994) 
v_mov_b64 v[12:13], s[18:19] // (op 994) convert


v_mov_b64 v[46:47], v[22:23] // (op 994) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 994) 
s_lshl_b64 s[54:55], s[18:19], 1 // (op 994) 

v_mov_b32 v48, s54 // (op 994) 
v_mov_b32 v49, s55 // (op 994) 
v_add_co_u32 v50, s[18:19], v46, v48 // (op 994) least significant half
v_addc_co_u32 v51, s[18:19], v47, v49, s[18:19] // (op 994) most significant half




s_mov_b32 m0, s6 // (op 994) reg expression
buffer_load_dwordx4 v50, s[64:67], 0 offen offset: 0 lds // (op 994) Load value direct to lds
v_add_co_u32 v50, s[18:19], v50, v12 // (op 994) least significant half
v_addc_co_u32 v51, s[18:19], 0, v51, s[18:19] // (op 994) most significant half

s_add_u32 m0, m0, 2048 // (op 994) 
buffer_load_dwordx4 v50, s[64:67], 0 offen offset: 0 lds // (op 994) Load value direct to lds




v_mov_b32 v12, v25 // (op 192) 
s_lshl_b32 s18, s38, 3 // (op 192) 
v_add_u32 v13, v12, s18 // (op 192) 


buffer_load_ubyte v28, v13, s[72:75], 0 offen offset: 0 // (op 192) Load value

s_barrier  // (op 2458) 


v_mov_b64 v[48:49], v[16:17] // (op 989) 
s_add_u32 s18, 4096, s15 // (op 989) 

s_mov_b32 m0, s18 // (op 989) reg expression
s_mov_b32 s19, 256 // (op 989) call()
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 989) Load value direct to lds


s_barrier  // (op 2531) 



s_waitcnt vmcnt(6)
ds_write_b8 v18, v20 offset:12416 // (op 516) Store local data 
s_barrier  // (op 2519) 



ds_read_b64_tr_b8 v[30:31], v10 offset:6144 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v10 offset:6400 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:8192 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v10 offset:8448 // (op 509) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2528) 


ds_read_u8 v21, v8 offset:12416 // (op 513) Load local data 
s_barrier  // (op 2652) 
s_waitcnt vmcnt(5)
s_barrier  // (op 2510) 



ds_read_b128 v[38:41], v9 offset:10240 // (op 507) Load local data 
ds_read_b128 v[42:45], v9 offset:10304 // (op 507) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 514) 



buffer_load_ubyte v27, v19, s[68:71], 0 offen offset:12 // (op 511) Load value

s_barrier  // (op 2525) 
v_add_u32 v24, 8, v24 // (op 1446) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1017) 
v_mov_b64 v[50:51], s[18:19] // (op 1017) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 1017) 
s_mul_i32 s55, 3, s19 // (op 1017) most significant: low of low * high
s_mul_i32 s54, 3, s18 // (op 1017) least significant
s_mul_hi_u32 s44, 3, s18 // (op 1017) most significant: high of low * low
s_add_u32 s55, s55, s44 // (op 1017) most significant: sum


v_mov_b32 v13, s54 // (op 1017) 
v_mov_b32 v19, s55 // (op 1017) 
v_add_co_u32 v52, s[18:19], v46, v13 // (op 1017) least significant half
v_addc_co_u32 v53, s[18:19], v47, v19, s[18:19] // (op 1017) most significant half




s_add_u32 s18, 6144, s8 // (op 1017) 
s_nop 0
s_mov_b32 m0, s18 // (op 1017) reg expression
buffer_load_dwordx4 v52, s[64:67], 0 offen offset: 0 lds // (op 1017) Load value direct to lds
v_add_co_u32 v52, s[54:55], v52, v50 // (op 1017) least significant half
v_addc_co_u32 v53, s[54:55], 0, v53, s[54:55] // (op 1017) most significant half

s_add_u32 m0, m0, 2048 // (op 1017) 
buffer_load_dwordx4 v52, s[64:67], 0 offen offset: 0 lds // (op 1017) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 518) 
s_mul_i32 s19, 3, s18 // (op 518) 

v_add_u32 v13, v12, s19 // (op 518) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 1318) 
s_lshl_b64 s[54:55], s[18:19], 1 // (op 1318) 

v_mov_b32 v19, s54 // (op 1318) 
v_mov_b32 v46, s55 // (op 1318) 
v_add_co_u32 v22, s[18:19], v22, v19 // (op 1318) least significant half
v_addc_co_u32 v23, s[18:19], v23, v46, s[18:19] // (op 1318) most significant half




buffer_load_ubyte v20, v13, s[72:75], 0 offen offset: 0 // (op 518) Load value


s_barrier  // (op 2516) 
s_lshl_b32 s18, s38, 3 // (op 1574) 


v_add_u32 v25, v25, s18 // (op 1574) 

s_add_u32 s18, 10240, s7 // (op 1012) 

s_mov_b32 m0, s18 // (op 1012) reg expression
s_mov_b32 s19, 384 // (op 1012) call()
buffer_load_dwordx4 v48, s[56:59], s19 offen offset: 0 lds // (op 1012) Load value direct to lds



s_barrier  // (op 2496) 
v_add_co_u32 v16, vcc, 256, v16 // (op 1190) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1190) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v29, v28 offset:12288 // (op 190) Store local data 
s_barrier  // (op 2483) 



ds_read_b64_tr_b8 v[30:31], v7 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:256 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:2048 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:2304 // (op 183) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2503) 


ds_read_u8 v21, v5 offset:12288 // (op 187) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 2464) 



ds_read_b128 v[38:41], v6 offset:4096 // (op 181) Load local data 
ds_read_b128 v[42:45], v6 offset:4160 // (op 181) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 175) 
s_mov_b32 s44, s18 // (op 175) convert

s_and_b32 s18, s44, 4294967294 // (op 175) 

s_cmp_lt_u32 s1, s18 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_175 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_175:












s_barrier  // (op 2580) 
v_lshrrev_b32 v12, 3, v1 // (op 1084) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1084) 
v_and_b32 v13, -8, v1 // (op 1084) 
s_and_b32 s15, s18, 4294967294 // (op 1084) 
v_sub_u32 v16, v1, v13 // (op 1084) 

s_add_u32 s6, s0, s15 // (op 1084) 

v_lshlrev_b32 v13, 4, v16 // (op 1084) 

s_mul_i32 s15, s16, s52 // (op 1084) 
s_add_u32 s15, s15, s6 // (op 1084) 

s_add_u32 s6, s15, s9 // (op 1084) 

s_mul_hi_u32 s15, s6, s13 // (op 1084) 
s_sub_u32 s7, s6, s15 // (op 1084) 
s_lshr_b32 s8, s7, 1 // (op 1084) 

s_add_u32 s7, s8, s15 // (op 1084) 


s_lshr_b32 s8, s7, s11 // (op 1084) 

s_and_b32 s7, s11, 2147483648 // (op 1084) 
s_cmp_eq_i32 s7, 0 // (op 1084) 
s_mov_b32 s15, scc // (op 1084) 

s_cmp_eq_i32 s15, 0 // (op 1084) 
s_cselect_b32 s7, s6, s8 // (op 1084) 


s_mul_hi_u32 s8, s7, s12 // (op 1084) 
s_sub_u32 s15, s7, s8 // (op 1084) 
s_lshr_b32 s44, s15, 1 // (op 1084) 

s_add_u32 s15, s44, s8 // (op 1084) 


s_lshr_b32 s8, s15, s10 // (op 1084) 

s_and_b32 s15, s10, 2147483648 // (op 1084) 
s_cmp_eq_i32 s15, 0 // (op 1084) 
s_mov_b32 s44, scc // (op 1084) 

s_cmp_eq_i32 s44, 0 // (op 1084) 
s_cselect_b32 s15, s7, s8 // (op 1084) 


s_mul_i32 s8, s7, s18 // (op 1084) 


v_lshl_add_u32 v16, s15, 4, v12 // (op 1084) 


s_sub_u32 s7, s6, s8 // (op 1084) 


v_lshl_add_u32 v12, s7, 7, v13 // (op 1084) 


v_mov_b32 v19, 0 // (op 1084) convert
v_mov_b32 v18, v12 // (op 1084) convert

v_mul_lo_u32 v13, v16, s25 // (op 1084) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1084) most significant: high of low * low
v_mul_lo_u32 v12, v16, s24 // (op 1084) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 1084) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v18 // (op 1084) least significant half
v_addc_co_u32 v13, s[18:19], v13, v19, s[18:19] // (op 1084) most significant half



s_mul_i32 s19, s24, s29 // (op 1086) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1086) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1086) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1086) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1086) most significant: sum
s_add_u32 s19, s19, s6 // (op 1086) most significant: sum


s_mov_b32 s56, s26 // (op 1086) reg expression
s_mov_b32 s57, s27 // (op 1086) reg expression
s_mov_b32 s58, s18 // (op 1086) reg expression
s_mov_b32 s59, 131072 // (op 1086) call()

v_and_b32 v16, -2, v1 // (op 1220) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1220) 
v_sub_u32 v17, v1, v16 // (op 1220) 

s_and_b32 s15, s18, 4294967294 // (op 1220) 
v_lshlrev_b32 v16, 4, v17 // (op 1220) 

s_add_u32 s6, s0, s15 // (op 1220) 

v_lshrrev_b32 v17, 1, v1 // (op 1220) 
s_mul_i32 s15, s16, s52 // (op 1220) 
s_add_u32 s15, s15, s6 // (op 1220) 

s_add_u32 s6, s15, s9 // (op 1220) 

s_mul_hi_u32 s15, s6, s13 // (op 1220) 
s_sub_u32 s7, s6, s15 // (op 1220) 
s_lshr_b32 s8, s7, 1 // (op 1220) 

s_add_u32 s7, s8, s15 // (op 1220) 


s_lshr_b32 s8, s7, s11 // (op 1220) 

s_and_b32 s7, s11, 2147483648 // (op 1220) 
s_cmp_eq_i32 s7, 0 // (op 1220) 
s_mov_b32 s15, scc // (op 1220) 

s_cmp_eq_i32 s15, 0 // (op 1220) 
s_cselect_b32 s7, s6, s8 // (op 1220) 


s_mul_hi_u32 s8, s7, s12 // (op 1220) 
s_sub_u32 s15, s7, s8 // (op 1220) 
s_lshr_b32 s44, s15, 1 // (op 1220) 

s_add_u32 s15, s44, s8 // (op 1220) 


s_lshr_b32 s8, s15, s10 // (op 1220) 

s_and_b32 s15, s10, 2147483648 // (op 1220) 
s_cmp_eq_i32 s15, 0 // (op 1220) 
s_mov_b32 s44, scc // (op 1220) 

s_cmp_eq_i32 s44, 0 // (op 1220) 
s_cselect_b32 s15, s7, s8 // (op 1220) 


s_mul_i32 s8, s15, s14 // (op 1220) 

s_sub_u32 s15, s7, s8 // (op 1220) 

s_mul_i32 s8, s7, s18 // (op 1220) 


v_lshl_add_u32 v18, s15, 5, v16 // (op 1220) 


s_sub_u32 s7, s6, s8 // (op 1220) 


v_mov_b32 v29, 0 // (op 1220) convert
v_mov_b32 v28, v18 // (op 1220) convert

v_lshl_add_u32 v16, s7, 7, v17 // (op 1220) 


v_mul_lo_u32 v19, v16, s31 // (op 1220) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 1220) most significant: high of low * low
v_mul_lo_u32 v18, v16, s30 // (op 1220) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1220) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v28 // (op 1220) least significant half
v_addc_co_u32 v19, s[18:19], v19, v29, s[18:19] // (op 1220) most significant half



s_mul_i32 s19, s30, s3 // (op 1222) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1222) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1222) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1222) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1222) most significant: sum
s_add_u32 s19, s19, s6 // (op 1222) most significant: sum


s_mov_b32 s64, s32 // (op 1222) reg expression
s_mov_b32 s65, s33 // (op 1222) reg expression
s_mov_b32 s66, s18 // (op 1222) reg expression
s_mov_b32 s67, 131072 // (op 1222) call()

v_lshrrev_b32 v16, 7, v1 // (op 1348) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1348) 
v_and_b32 v17, 15, v1 // (op 1348) 
s_and_b32 s15, s18, 4294967294 // (op 1348) 
v_lshl_add_u32 v22, v16, 4, v17 // (op 1348) 


s_add_u32 s6, s0, s15 // (op 1348) 

v_and_b32 v16, 63, v1 // (op 1348) 
v_lshrrev_b32 v17, 4, v16 // (op 1348) 

s_mul_i32 s15, s16, s52 // (op 1348) 
s_add_u32 s15, s15, s6 // (op 1348) 

s_add_u32 s6, s15, s9 // (op 1348) 

s_mul_hi_u32 s15, s6, s13 // (op 1348) 
s_sub_u32 s7, s6, s15 // (op 1348) 
s_lshr_b32 s8, s7, 1 // (op 1348) 

s_add_u32 s7, s8, s15 // (op 1348) 


s_lshr_b32 s8, s7, s11 // (op 1348) 

s_and_b32 s7, s11, 2147483648 // (op 1348) 
s_cmp_eq_i32 s7, 0 // (op 1348) 
s_mov_b32 s15, scc // (op 1348) 

s_cmp_eq_i32 s15, 0 // (op 1348) 
s_cselect_b32 s7, s6, s8 // (op 1348) 


s_mul_hi_u32 s8, s7, s12 // (op 1348) 
s_sub_u32 s15, s7, s8 // (op 1348) 
s_lshr_b32 s44, s15, 1 // (op 1348) 

s_add_u32 s15, s44, s8 // (op 1348) 


s_lshr_b32 s8, s15, s10 // (op 1348) 

s_and_b32 s15, s10, 2147483648 // (op 1348) 
s_cmp_eq_i32 s15, 0 // (op 1348) 
s_mov_b32 s44, scc // (op 1348) 

s_cmp_eq_i32 s44, 0 // (op 1348) 
s_cselect_b32 s15, s7, s8 // (op 1348) 


s_mul_i32 s8, s7, s18 // (op 1348) 


v_lshl_add_u32 v16, s15, 4, v22 // (op 1348) 


s_sub_u32 s7, s6, s8 // (op 1348) 


v_lshl_add_u32 v22, s7, 2, v17 // (op 1348) 


v_mul_lo_u32 v17, v16, s34 // (op 1348) 
v_add_u32 v17, v17, v22 // (op 1348) 


s_mul_i32 s19, s34, s29 // (op 1350) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1350) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1350) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1350) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1350) most significant: sum
s_add_u32 s19, s19, s6 // (op 1350) most significant: sum


s_mov_b32 s68, s36 // (op 1350) reg expression
s_mov_b32 s69, s37 // (op 1350) reg expression
s_mov_b32 s70, s18 // (op 1350) reg expression
s_mov_b32 s71, 131072 // (op 1350) call()

v_and_b32 v16, -32, v1 // (op 1476) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1476) 
v_sub_u32 v22, v1, v16 // (op 1476) 

s_and_b32 s15, s18, 4294967294 // (op 1476) 
v_lshrrev_b32 v16, 5, v1 // (op 1476) 
s_add_u32 s6, s0, s15 // (op 1476) 

s_mul_i32 s15, s16, s52 // (op 1476) 
s_add_u32 s15, s15, s6 // (op 1476) 

s_add_u32 s6, s15, s9 // (op 1476) 

s_mul_hi_u32 s15, s6, s13 // (op 1476) 
s_sub_u32 s7, s6, s15 // (op 1476) 
s_lshr_b32 s8, s7, 1 // (op 1476) 

s_add_u32 s7, s8, s15 // (op 1476) 


s_lshr_b32 s8, s7, s11 // (op 1476) 

s_and_b32 s7, s11, 2147483648 // (op 1476) 
s_cmp_eq_i32 s7, 0 // (op 1476) 
s_mov_b32 s15, scc // (op 1476) 

s_cmp_eq_i32 s15, 0 // (op 1476) 
s_cselect_b32 s7, s6, s8 // (op 1476) 


s_mul_hi_u32 s8, s7, s12 // (op 1476) 
s_sub_u32 s15, s7, s8 // (op 1476) 
s_lshr_b32 s44, s15, 1 // (op 1476) 

s_add_u32 s15, s44, s8 // (op 1476) 


s_lshr_b32 s8, s15, s10 // (op 1476) 

s_and_b32 s15, s10, 2147483648 // (op 1476) 
s_cmp_eq_i32 s15, 0 // (op 1476) 
s_mov_b32 s44, scc // (op 1476) 

s_cmp_eq_i32 s44, 0 // (op 1476) 
s_cselect_b32 s15, s7, s8 // (op 1476) 


s_mul_i32 s8, s15, s14 // (op 1476) 

s_sub_u32 s15, s7, s8 // (op 1476) 

s_mul_i32 s8, s7, s18 // (op 1476) 


v_lshl_add_u32 v23, s15, 5, v22 // (op 1476) 


s_sub_u32 s7, s6, s8 // (op 1476) 


v_lshl_add_u32 v22, s7, 2, v16 // (op 1476) 


v_mul_lo_u32 v16, v22, s38 // (op 1476) 
v_add_u32 v16, v16, v23 // (op 1476) 


s_mul_i32 s19, s38, s43 // (op 1478) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1478) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1478) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1478) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1478) most significant: sum
s_add_u32 s19, s19, s6 // (op 1478) most significant: sum


s_mov_b32 s72, s40 // (op 1478) reg expression
s_mov_b32 s73, s41 // (op 1478) reg expression
s_mov_b32 s74, s18 // (op 1478) reg expression
s_mov_b32 s75, 131072 // (op 1478) call()

v_and_b32 v22, -8, v1 // (op 1915) 
v_sub_u32 v23, v1, v22 // (op 1915) 

v_lshlrev_b32 v22, 4, v23 // (op 1915) 

v_lshrrev_b32 v23, 3, v1 // (op 1915) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 1915) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 1915) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -2, v1 // (op 1980) 
v_sub_u32 v23, v1, v22 // (op 1980) 

v_lshlrev_b32 v22, 4, v23 // (op 1980) 

v_lshrrev_b32 v23, 1, v1 // (op 1980) 
v_lshl_add_u32 v24, v23, 5, v22 // (op 1980) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 1980) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -32, v1 // (op 2045) 
v_sub_u32 v23, v1, v22 // (op 2045) 

v_and_b32 v22, 4294967264, v1 // (op 2045) 
v_add_u32 v24, v22, v23 // (op 2045) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 442) 
s_and_b32 s7, s18, 4294967294 // (op 442) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 441) 
s_mov_b32 s8, s18 // (op 441) convert

s_cmp_lt_u32 s7, s8 // (op 441) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_441 // (op 441) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_441 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_441:
s_barrier  // (op 2461) 
s_add_u32 s7, s7, 1 // (op 443) 


v_mov_b64 v[20:21], v[12:13] // (op 999) 
s_add_u32 s8, 4096, s15 // (op 999) 

s_mov_b32 m0, s8 // (op 999) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 999) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 1074) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 1074) most significant half
s_barrier  // (op 2480) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1005) 
v_mov_b64 v[20:21], s[18:19] // (op 1005) convert


v_mov_b64 v[22:23], v[18:19] // (op 1005) 
v_mov_b64 v[26:27], v[22:23] // (op 1005) reg expression
s_mov_b32 m0, s6 // (op 1005) reg expression
buffer_load_dwordx4 v26, s[64:67], 0 offen offset: 0 lds // (op 1005) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 1005) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1005) most significant half

s_add_u32 m0, m0, 2048 // (op 1005) 
buffer_load_dwordx4 v26, s[64:67], 0 offen offset: 0 lds // (op 1005) Load value direct to lds



s_lshl_b64 s[18:19], s[30:31], 7 // (op 1212) 
v_mov_b32 v20, s18 // (op 1212) 
v_mov_b32 v21, s19 // (op 1212) 
v_add_co_u32 v18, s[54:55], v18, v20 // (op 1212) least significant half
v_addc_co_u32 v19, s[54:55], v19, v21, s[54:55] // (op 1212) most significant half






v_mov_b32 v20, v16 // (op 460) 
v_mov_b32 v21, v20 // (op 460) reg expression

buffer_load_ubyte v22, v21, s[72:75], 0 offen offset: 0 // (op 460) Load value


s_lshl_b32 s8, s38, 2 // (op 1468) 
v_add_u32 v16, v16, s8 // (op 1468) 

s_barrier  // (op 2499) 
s_waitcnt vmcnt(3)
s_barrier  // (op 2467) 






ds_read_b128 v[26:29], v6 offset:4096 // (op 449) Load local data 
ds_read_b128 v[30:33], v6 offset:4160 // (op 449) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v22 offset:12288 // (op 458) Store local data 

s_barrier  // (op 2486) 



ds_read_b64_tr_b8 v[34:35], v7 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:256 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:2048 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:2304 // (op 451) Transpose load local data 


v_mov_b32 v25, v17 // (op 453) 

buffer_load_ubyte v20, v25, s[68:71], 0 offen offset: 0 // (op 453) Load value

v_add_u32 v17, 4, v17 // (op 1340) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2506) 


ds_read_u8 v25, v5 offset:12288 // (op 455) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 456) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 441) 
s_mov_b32 s8, s18 // (op 441) convert

s_cmp_lt_u32 s7, s8 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_441 // (op 441) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_441 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_441:








s_barrier  // (op 2642) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 222) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_222 // (op 222) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_222
s_nop 0
v_accvgpr_read v16, a0 // (op 230) Wait state hazard: XDL Write Hazard
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

v_lshrrev_b32 v0, 7, v1 // (op 1868) 
s_mul_i32 s15, s16, s52 // (op 1868) 
s_add_u32 s15, s15, s0 // (op 1868) 
v_and_b32 v3, 63, v1 // (op 1868) 
s_add_u32 s6, s15, s9 // (op 1868) 

v_lshrrev_b32 v11, 4, v3 // (op 1868) 
s_mul_hi_u32 s15, s6, s13 // (op 1868) 
v_lshlrev_b32 v12, 2, v11 // (op 1868) 

s_sub_u32 s7, s6, s15 // (op 1868) 
v_lshl_add_u32 v11, v0, 4, v12 // (op 1868) 


s_lshr_b32 s8, s7, 1 // (op 1868) 

v_lshrrev_b32 v0, 2, v3 // (op 1868) 

s_add_u32 s7, s8, s15 // (op 1868) 


v_and_b32 v3, 3, v0 // (op 1868) 

s_lshr_b32 s8, s7, s11 // (op 1868) 

v_and_b32 v0, 3, v1 // (op 1868) 
s_and_b32 s7, s11, 2147483648 // (op 1868) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 1868) 


s_cmp_eq_i32 s7, 0 // (op 1868) 
s_mov_b32 s15, scc // (op 1868) 

v_lshrrev_b32 v0, 6, v1 // (op 1868) 
s_cmp_eq_i32 s15, 0 // (op 1868) 
s_cselect_b32 s7, s6, s8 // (op 1868) 



v_and_b32 v3, 1, v0 // (op 1868) 

s_mul_hi_u32 s6, s7, s12 // (op 1868) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 1868) 


s_sub_u32 s8, s7, s6 // (op 1868) 
s_lshr_b32 s15, s8, 1 // (op 1868) 

s_add_u32 s8, s15, s6 // (op 1868) 


s_lshr_b32 s6, s8, s10 // (op 1868) 

s_and_b32 s8, s10, 2147483648 // (op 1868) 
s_cmp_eq_i32 s8, 0 // (op 1868) 
s_mov_b32 s15, scc // (op 1868) 

s_cmp_eq_i32 s15, 0 // (op 1868) 
s_cselect_b32 s8, s7, s6 // (op 1868) 


s_mul_i32 s6, s8, s14 // (op 1868) 
v_lshl_add_u32 v3, s8, 4, v11 // (op 1868) 


s_sub_u32 s8, s7, s6 // (op 1868) 


v_lshl_add_u32 v11, s8, 5, v0 // (op 1868) 


v_mul_lo_u32 v0, v11, s48 // (op 1868) 

v_add_lshl_u32 v11, v3, v0, 1 // (op 1868) 


s_mul_i32 s19, s48, s23 // (op 1870) most significant: low of low * high
s_mul_i32 s15, s49, s22 // (op 1870) most significant: low of high * low
s_mul_i32 s18, s48, s22 // (op 1870) least significant
s_mul_hi_u32 s6, s48, s22 // (op 1870) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1870) most significant: sum
s_add_u32 s19, s19, s6 // (op 1870) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 1 // (op 1870) 

s_mov_b32 s80, s50 // (op 1870) reg expression
s_mov_b32 s81, s51 // (op 1870) reg expression
s_mov_b32 s82, s54 // (op 1870) reg expression
s_mov_b32 s83, 131072 // (op 1870) call()


v_pack_b32_f16 v12, v16, v17 // (op 233) 
v_pack_b32_f16 v13, v18, v19 // (op 233) 


buffer_store_dwordx2 v[12:13], v11, s[80:83], 0 offen // (op 233) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_222 // (op 222) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_222
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_222:



buffer_load_dwordx2 v[12:13], v4, s[60:63], 0 offen // (op 247) Load value
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

v_lshrrev_b32 v0, 7, v1 // (op 1885) 
s_mul_i32 s15, s16, s52 // (op 1885) 
s_add_u32 s15, s15, s0 // (op 1885) 
v_and_b32 v3, 63, v1 // (op 1885) 
s_add_u32 s6, s15, s9 // (op 1885) 

v_lshrrev_b32 v11, 4, v3 // (op 1885) 
s_mul_hi_u32 s15, s6, s13 // (op 1885) 
v_lshlrev_b32 v12, 2, v11 // (op 1885) 

s_sub_u32 s7, s6, s15 // (op 1885) 
v_lshl_add_u32 v11, v0, 4, v12 // (op 1885) 


s_lshr_b32 s8, s7, 1 // (op 1885) 

v_lshrrev_b32 v0, 2, v3 // (op 1885) 

s_add_u32 s7, s8, s15 // (op 1885) 


v_and_b32 v3, 3, v0 // (op 1885) 

s_lshr_b32 s8, s7, s11 // (op 1885) 

v_and_b32 v0, 3, v1 // (op 1885) 
s_and_b32 s7, s11, 2147483648 // (op 1885) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 1885) 


s_cmp_eq_i32 s7, 0 // (op 1885) 
s_mov_b32 s15, scc // (op 1885) 

v_lshrrev_b32 v0, 6, v1 // (op 1885) 
s_cmp_eq_i32 s15, 0 // (op 1885) 
s_cselect_b32 s7, s6, s8 // (op 1885) 



v_and_b32 v3, 1, v0 // (op 1885) 

s_mul_hi_u32 s6, s7, s12 // (op 1885) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 1885) 


s_sub_u32 s8, s7, s6 // (op 1885) 
s_lshr_b32 s15, s8, 1 // (op 1885) 

s_add_u32 s8, s15, s6 // (op 1885) 


s_lshr_b32 s6, s8, s10 // (op 1885) 

s_and_b32 s8, s10, 2147483648 // (op 1885) 
s_cmp_eq_i32 s8, 0 // (op 1885) 
s_mov_b32 s15, scc // (op 1885) 

s_cmp_eq_i32 s15, 0 // (op 1885) 
s_cselect_b32 s8, s7, s6 // (op 1885) 


s_mul_i32 s6, s8, s14 // (op 1885) 
v_lshl_add_u32 v3, s8, 4, v11 // (op 1885) 


s_sub_u32 s8, s7, s6 // (op 1885) 


v_lshl_add_u32 v11, s8, 5, v0 // (op 1885) 


v_mul_lo_u32 v0, v11, s48 // (op 1885) 

v_add_lshl_u32 v11, v3, v0, 1 // (op 1885) 


s_mul_i32 s19, s48, s23 // (op 1887) most significant: low of low * high
s_mul_i32 s15, s49, s22 // (op 1887) most significant: low of high * low
s_mul_i32 s18, s48, s22 // (op 1887) least significant
s_mul_hi_u32 s6, s48, s22 // (op 1887) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1887) most significant: sum
s_add_u32 s19, s19, s6 // (op 1887) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 1 // (op 1887) 

s_mov_b32 s80, s50 // (op 1887) reg expression
s_mov_b32 s81, s51 // (op 1887) reg expression
s_mov_b32 s82, s54 // (op 1887) reg expression
s_mov_b32 s83, 131072 // (op 1887) call()


v_pack_b32_f16 v12, v20, v21 // (op 246) 
v_pack_b32_f16 v13, v22, v23 // (op 246) 


buffer_store_dwordx2 v[12:13], v11, s[80:83], 0 offen // (op 246) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_222:

s_add_u32 s0, s0, s1 // (op 435) 
s_mul_i32 s15, s52, s16 // (op 433) 
s_cmp_gt_u32 s53, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s53, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s52, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s52 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433 // (op 433) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433:









































s_endpgm  // End of RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.LRR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2, .LRR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end-RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
  .amdhsa_accum_offset 60
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
    DestDataType: h
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
            { type: address, semantic: RR_SKAddressPartials },
            { type: address, semantic: RR_SKAddressFlags },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 13
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 84
    .vgpr_count: 60
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_16
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_17
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: SCRATCH_None
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 168
        .value_kind: by_value
      - .name: Tensor_21_pointer
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

