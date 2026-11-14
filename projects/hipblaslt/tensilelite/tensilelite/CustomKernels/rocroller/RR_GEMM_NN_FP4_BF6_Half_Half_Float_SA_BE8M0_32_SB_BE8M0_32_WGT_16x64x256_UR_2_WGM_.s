// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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
s_load_dword s50, s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s10 // (op 1446) 
s_mul_hi_i32 s1, s16, s10 // (op 1688) 
v_and_b32 v0, 63, v1 // (op 1503) 
v_and_b32 v3, 48, v1 // (op 1503) 
v_sub_u32 v4, v0, v3 // (op 1503) 

v_lshrrev_b32 v3, 4, v0 // (op 1503) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1503) 


v_lshrrev_b32 v3, 7, v1 // (op 1503) 
v_lshlrev_b32 v4, 4, v3 // (op 1503) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1503) 


v_lshlrev_b32 v0, 2, v3 // (op 1503) 

v_lshrrev_b32 v3, 3, v0 // (op 1503) 

v_lshrrev_b32 v0, 7, v1 // (op 1614) 
v_and_b32 v4, 15, v1 // (op 1614) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1614) 


v_and_b32 v0, 48, v1 // (op 1614) 
v_add_u32 v4, v0, v5 // (op 1614) 


v_and_b32 v0, 63, v1 // (op 1714) 
v_and_b32 v5, 48, v1 // (op 1714) 
v_sub_u32 v6, v0, v5 // (op 1714) 

v_lshrrev_b32 v5, 4, v0 // (op 1714) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 1714) 


v_lshrrev_b32 v5, 7, v1 // (op 1714) 
v_lshlrev_b32 v6, 4, v5 // (op 1714) 

v_lshl_add_u32 v5, v0, 4, v6 // (op 1714) 


v_lshlrev_b32 v0, 2, v5 // (op 1714) 

v_lshrrev_b32 v5, 3, v0 // (op 1714) 

v_lshrrev_b32 v0, 7, v1 // (op 1767) 
v_and_b32 v6, 15, v1 // (op 1767) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 1767) 


v_and_b32 v0, 48, v1 // (op 1767) 
v_add_u32 v6, v0, v7 // (op 1767) 


v_lshrrev_b32 v0, 7, v1 // (op 1446) 
s_add_i32 s18, s0, s16 // (op 1446) 

v_and_b32 v7, 63, v1 // (op 1446) 
s_ashr_i32 s0, s18, 31 // (op 1446) 
v_lshrrev_b32 v8, 4, v7 // (op 1446) 
s_cmp_eq_i32 s10, 0 // (op 1446) 
s_mov_b32 s19, scc // (op 1446) 
v_lshlrev_b32 v9, 2, v8 // (op 1446) 

s_cmp_eq_i32 s19, 0 // (op 1446) 
s_cselect_b32 s51, 0, -1 // (op 1446) 

v_lshl_add_u32 v8, v0, 4, v9 // (op 1446) 


s_and_b32 s19, s11, 31 // (op 1446) 
v_lshrrev_b32 v0, 2, v7 // (op 1446) 

v_and_b32 v7, 3, v0 // (op 1446) 

v_lshrrev_b32 v0, 6, v1 // (op 1561) 
v_and_b32 v9, 1, v0 // (op 1561) 

v_and_b32 v0, 15, v1 // (op 1561) 
v_lshl_add_u32 v10, v9, 4, v0 // (op 1561) 


v_and_b32 v0, 63, v1 // (op 1561) 
v_lshrrev_b32 v9, 4, v0 // (op 1561) 

v_lshlrev_b32 v0, 5, v9 // (op 1561) 

v_lshl_add_u32 v9, v10, 8, v0 // (op 1561) 


v_mul_lo_u32 v0, v9, 6 // (op 1561) 

v_lshrrev_b32 v9, 3, v0 // (op 1561) 

v_lshrrev_b32 v0, 6, v1 // (op 1672) 
v_and_b32 v10, 1, v0 // (op 1672) 

v_and_b32 v0, 15, v1 // (op 1672) 
v_lshl_add_u32 v11, v10, 4, v0 // (op 1672) 


v_and_b32 v0, 63, v1 // (op 1672) 
v_lshrrev_b32 v10, 4, v0 // (op 1672) 

v_lshl_add_u32 v0, v11, 3, v10 // (op 1672) 


v_lshrrev_b32 v10, 7, v1 // (op 1688) 
v_and_b32 v11, 63, v1 // (op 1688) 
v_lshrrev_b32 v12, 6, v1 // (op 1743) 
v_and_b32 v13, 1, v12 // (op 1743) 

v_and_b32 v12, 15, v1 // (op 1743) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1743) 


v_and_b32 v12, 63, v1 // (op 1743) 
v_lshrrev_b32 v13, 4, v12 // (op 1743) 

v_lshlrev_b32 v12, 5, v13 // (op 1743) 

v_lshl_add_u32 v13, v14, 8, v12 // (op 1743) 


v_mul_lo_u32 v12, v13, 6 // (op 1743) 

v_lshrrev_b32 v13, 3, v12 // (op 1743) 

v_lshrrev_b32 v12, 6, v1 // (op 1796) 
v_and_b32 v14, 1, v12 // (op 1796) 

v_and_b32 v12, 15, v1 // (op 1796) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1796) 


v_and_b32 v12, 63, v1 // (op 1796) 
v_lshrrev_b32 v14, 4, v12 // (op 1796) 

v_lshl_add_u32 v12, v15, 3, v14 // (op 1796) 


s_lshl_b32 s52, 1, s19 // (op 1446) 
v_and_b32 v14, 3, v1 // (op 1446) 
s_add_i32 s53, s52, s51 // (op 1446) 


s_and_b32 s51, s0, s53 // (op 1446) 


s_add_i32 s0, s1, s16 // (op 1688) 

s_ashr_i32 s1, s0, 31 // (op 1688) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1446) 


s_add_i32 s52, s18, s51 // (op 1446) 


v_lshrrev_b32 v7, 6, v1 // (op 1446) 
s_ashr_i32 s18, s52, s19 // (op 1446) 


v_and_b32 v14, 1, v7 // (op 1446) 

s_lshl_b32 s19, s11, 24 // (op 1446) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1446) 


s_ashr_i32 s51, s19, 31 // (op 1446) 

s_xor_b32 s19, s18, s51 // (op 1446) 

s_sub_i32 s18, s19, s51 // (op 1446) 


s_mul_hi_i32 s19, s18, s12 // (op 1446) 
s_add_i32 s51, s19, s18 // (op 1446) 

s_ashr_i32 s19, s51, 31 // (op 1446) 
s_cmp_eq_i32 s12, 0 // (op 1446) 
s_mov_b32 s52, scc // (op 1446) 
s_cmp_eq_i32 s52, 0 // (op 1446) 
s_cselect_b32 s53, 0, -1 // (op 1446) 

s_and_b32 s52, s13, 31 // (op 1446) 
v_lshrrev_b32 v14, 4, v11 // (op 1688) 
s_cmp_eq_i32 s10, 0 // (op 1688) 
s_mov_b32 s54, scc // (op 1688) 
v_lshlrev_b32 v15, 2, v14 // (op 1688) 

s_cmp_eq_i32 s54, 0 // (op 1688) 
s_cselect_b32 s55, 0, -1 // (op 1688) 

s_lshl_b32 s54, 1, s52 // (op 1446) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1688) 


v_lshrrev_b32 v10, 2, v11 // (op 1688) 

s_add_i32 s56, s54, s53 // (op 1446) 


s_and_b32 s53, s19, s56 // (op 1446) 


s_add_i32 s19, s51, s53 // (op 1446) 


s_ashr_i32 s51, s19, s52 // (op 1446) 


s_lshl_b32 s19, s13, 24 // (op 1446) 
s_ashr_i32 s52, s19, 31 // (op 1446) 

s_xor_b32 s19, s51, s52 // (op 1446) 

s_sub_i32 s51, s19, s52 // (op 1446) 


s_mul_i32 s19, s18, s15 // (op 1446) 
s_sub_u32 s52, s16, s19 // (op 1446) 

s_mul_i32 s19, s51, s15 // (op 1446) 
s_add_u32 s19, s19, s52 // (op 1446) 

s_mul_hi_i32 s52, s16, s9 // (op 1446) 
s_add_i32 s53, s52, s16 // (op 1446) 

s_ashr_i32 s52, s53, 31 // (op 1446) 
s_cmp_eq_i32 s9, 0 // (op 1446) 
s_mov_b32 s54, scc // (op 1446) 
s_cmp_eq_i32 s54, 0 // (op 1446) 
s_cselect_b32 s56, 0, -1 // (op 1446) 

s_and_b32 s54, s11, 31 // (op 1688) 
v_and_b32 v11, 3, v10 // (op 1688) 

s_lshl_b32 s57, 1, s54 // (op 1688) 
v_and_b32 v10, 3, v1 // (op 1688) 
s_add_i32 s58, s57, s55 // (op 1688) 


s_and_b32 s55, s1, s58 // (op 1688) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1688) 


s_add_i32 s1, s0, s55 // (op 1688) 


v_lshrrev_b32 v10, 6, v1 // (op 1688) 
s_ashr_i32 s0, s1, s54 // (op 1688) 


v_and_b32 v11, 1, v10 // (op 1688) 

s_lshl_b32 s1, s11, 24 // (op 1688) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1688) 


s_ashr_i32 s54, s1, 31 // (op 1688) 

s_xor_b32 s1, s0, s54 // (op 1688) 

s_sub_i32 s0, s1, s54 // (op 1688) 


s_mul_hi_i32 s1, s0, s12 // (op 1688) 
s_add_i32 s54, s1, s0 // (op 1688) 

s_ashr_i32 s1, s54, 31 // (op 1688) 
s_cmp_eq_i32 s12, 0 // (op 1688) 
s_mov_b32 s55, scc // (op 1688) 
s_cmp_eq_i32 s55, 0 // (op 1688) 
s_cselect_b32 s57, 0, -1 // (op 1688) 

s_and_b32 s55, s13, 31 // (op 1688) 
s_lshl_b32 s58, 1, s55 // (op 1688) 
s_add_i32 s59, s58, s57 // (op 1688) 


s_and_b32 s57, s1, s59 // (op 1688) 


s_add_i32 s1, s54, s57 // (op 1688) 


s_ashr_i32 s54, s1, s55 // (op 1688) 


s_lshl_b32 s1, s13, 24 // (op 1688) 
s_ashr_i32 s55, s1, 31 // (op 1688) 

s_xor_b32 s1, s54, s55 // (op 1688) 

s_sub_i32 s54, s1, s55 // (op 1688) 


s_mul_i32 s1, s0, s15 // (op 1688) 
s_sub_u32 s55, s16, s1 // (op 1688) 

s_mul_i32 s1, s54, s15 // (op 1688) 
s_add_u32 s1, s1, s55 // (op 1688) 

s_mul_hi_i32 s55, s16, s9 // (op 1688) 
s_add_i32 s57, s55, s16 // (op 1688) 

s_ashr_i32 s55, s57, 31 // (op 1688) 
s_cmp_eq_i32 s9, 0 // (op 1688) 
s_mov_b32 s58, scc // (op 1688) 
s_cmp_eq_i32 s58, 0 // (op 1688) 
s_cselect_b32 s59, 0, -1 // (op 1688) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 1446) 
s_lshl_b32 s60, 1, s58 // (op 1446) 
s_add_i32 s61, s60, s56 // (op 1446) 


s_and_b32 s56, s52, s61 // (op 1446) 


s_add_i32 s52, s53, s56 // (op 1446) 


s_ashr_i32 s53, s52, s58 // (op 1446) 


s_lshl_b32 s52, s46, 24 // (op 1446) 
s_ashr_i32 s56, s52, 31 // (op 1446) 

s_xor_b32 s52, s53, s56 // (op 1446) 

s_sub_i32 s53, s52, s56 // (op 1446) 


s_and_b32 s52, s46, 31 // (op 1688) 
s_lshl_b32 s56, 1, s52 // (op 1688) 
s_add_i32 s58, s56, s59 // (op 1688) 


s_and_b32 s56, s55, s58 // (op 1688) 


s_add_i32 s55, s57, s56 // (op 1688) 


s_ashr_i32 s56, s55, s52 // (op 1688) 


s_lshl_b32 s52, s46, 24 // (op 1688) 
s_ashr_i32 s55, s52, 31 // (op 1688) 

s_xor_b32 s52, s56, s55 // (op 1688) 

s_sub_i32 s56, s52, s55 // (op 1688) 


s_add_u32 s58, s6, 64 // (op 1446) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1446) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1446) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1446) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1446) 

s_mul_i32 s52, s15, s58 // (op 1446) 
s_mul_i32 s55, s8, s52 // (op 1446) 

s_mul_i32 s52, s53, s55 // (op 1446) 

s_sub_u32 s55, s16, s52 // (op 1446) 

s_mov_b32 s52, s55 // (op 1446) 
s_mul_hi_i32 s57, s52, s14 // (op 1446) 
s_add_i32 s60, s57, s52 // (op 1446) 


s_ashr_i32 s52, s60, 31 // (op 1446) 
s_cmp_eq_i32 s14, 0 // (op 1446) 
s_mov_b32 s57, scc // (op 1446) 
s_cmp_eq_i32 s57, 0 // (op 1446) 
s_cselect_b32 s61, 0, -1 // (op 1446) 

s_and_b32 s57, s48, 31 // (op 1446) 
s_lshl_b32 s62, 1, s57 // (op 1446) 
s_add_i32 s63, s62, s61 // (op 1446) 


s_and_b32 s61, s52, s63 // (op 1446) 


s_add_i32 s52, s60, s61 // (op 1446) 


s_ashr_i32 s60, s52, s57 // (op 1446) 


s_lshl_b32 s52, s48, 24 // (op 1446) 
s_ashr_i32 s57, s52, 31 // (op 1446) 

s_xor_b32 s52, s60, s57 // (op 1446) 

s_sub_i32 s60, s52, s57 // (op 1446) 


s_mul_i32 s52, s47, s15 // (op 1446) 
s_add_u32 s62, s20, 16 // (op 1446) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 1446) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1446) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1446) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1446) 

s_sub_i32 s57, s62, s52 // (op 1446) 

s_mul_i32 s61, s60, s57 // (op 1446) 

s_sub_u32 s57, s55, s61 // (op 1446) 


s_add_u32 s55, s57, s52 // (op 1446) 


s_cmp_eq_i32 s53, 0 // (op 1446) 
s_mov_b32 s52, scc // (op 1446) 

s_cmp_eq_i32 s52, 0 // (op 1446) 
s_cselect_b32 s53, s55, s19 // (op 1446) 


s_mul_i32 s19, s51, s58 // (op 1446) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1446) 


s_sub_i32 s51, s18, s19 // (op 1446) 


s_cmp_eq_i32 s52, 0 // (op 1446) 
s_cselect_b32 s53, s60, s51 // (op 1446) 



v_lshl_add_u32 v8, s53, 6, v7 // (op 1446) 


v_mul_lo_u32 v7, v8, s2 // (op 1446) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 1446) 


s_add_u32 s18, s6, 64 // (op 1688) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1688) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1688) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1688) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 1688) 

s_mul_i32 s55, s15, s18 // (op 1688) 
s_mul_i32 s51, s8, s55 // (op 1688) 

s_mul_i32 s55, s56, s51 // (op 1688) 

s_sub_u32 s51, s16, s55 // (op 1688) 

s_mov_b32 s55, s51 // (op 1688) 
s_mul_hi_i32 s52, s55, s14 // (op 1688) 
s_add_i32 s53, s52, s55 // (op 1688) 


s_ashr_i32 s52, s53, 31 // (op 1688) 
s_cmp_eq_i32 s14, 0 // (op 1688) 
s_mov_b32 s55, scc // (op 1688) 
s_cmp_eq_i32 s55, 0 // (op 1688) 
s_cselect_b32 s57, 0, -1 // (op 1688) 

s_and_b32 s55, s48, 31 // (op 1688) 
s_lshl_b32 s58, 1, s55 // (op 1688) 
s_add_i32 s59, s58, s57 // (op 1688) 


s_and_b32 s57, s52, s59 // (op 1688) 


s_add_i32 s52, s53, s57 // (op 1688) 


s_ashr_i32 s53, s52, s55 // (op 1688) 


s_lshl_b32 s52, s48, 24 // (op 1688) 
s_ashr_i32 s55, s52, 31 // (op 1688) 

s_xor_b32 s52, s53, s55 // (op 1688) 

s_sub_i32 s53, s52, s55 // (op 1688) 


s_mul_i32 s52, s47, s15 // (op 1688) 
s_add_u32 s58, s20, 16 // (op 1688) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 1688) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1688) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1688) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1688) 

s_sub_i32 s55, s58, s52 // (op 1688) 

s_mul_i32 s57, s53, s55 // (op 1688) 

s_sub_u32 s55, s51, s57 // (op 1688) 


s_add_u32 s51, s55, s52 // (op 1688) 


s_cmp_eq_i32 s56, 0 // (op 1688) 
s_mov_b32 s52, scc // (op 1688) 

s_cmp_eq_i32 s52, 0 // (op 1688) 
s_cselect_b32 s55, s51, s1 // (op 1688) 


s_mul_i32 s1, s54, s18 // (op 1688) 


v_lshl_add_u32 v7, s55, 4, v14 // (op 1688) 


s_sub_i32 s51, s0, s1 // (op 1688) 


s_cmp_eq_i32 s52, 0 // (op 1688) 
s_cselect_b32 s0, s53, s51 // (op 1688) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1688) 


v_mul_lo_u32 v10, v11, s4 // (op 1688) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 1688) 


s_mul_i32 s1, s2, s7 // (op 1448) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1448) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1448) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1448) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1448) most significant: sum
s_add_u32 s1, s1, s19 // (op 1448) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1448) 

s_mul_i32 s1, s4, s7 // (op 1690) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1690) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1690) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1690) most significant: sum
s_add_u32 s1, s1, s52 // (op 1690) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1690) 

s_mov_b32 s56, s22 // (op 1448) reg expression
s_mov_b32 s57, s23 // (op 1448) reg expression
s_mov_b32 s58, s18 // (op 1448) reg expression
s_mov_b32 s59, 131072 // (op 1448) call()

s_mov_b32 s60, s24 // (op 1690) reg expression

s_mov_b32 s61, s25 // (op 1690) reg expression
s_mov_b32 s62, s52 // (op 1690) reg expression
s_mov_b32 s63, 131072 // (op 1690) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 402) call()
v_accvgpr_write a5, 0.00000 // (op 402) call()
v_accvgpr_write a6, 0.00000 // (op 402) call()
v_accvgpr_write a7, 0.00000 // (op 402) call()
s_mul_hi_i32 s0, s16, s10 // (op 1213) 
v_lshlrev_b32 v14, 1, v1 // (op 1213) 
s_add_i32 s1, s0, s16 // (op 1213) 

s_ashr_i32 s0, s1, 31 // (op 1213) 
s_cmp_eq_i32 s10, 0 // (op 1213) 
s_mov_b32 s18, scc // (op 1213) 
s_cmp_eq_i32 s18, 0 // (op 1213) 
s_cselect_b32 s19, 0, -1 // (op 1213) 

s_and_b32 s18, s11, 31 // (op 1213) 
s_lshl_b32 s22, 1, s18 // (op 1213) 
s_add_i32 s23, s22, s19 // (op 1213) 


s_and_b32 s19, s0, s23 // (op 1213) 


s_add_i32 s0, s1, s19 // (op 1213) 


s_ashr_i32 s1, s0, s18 // (op 1213) 


s_lshl_b32 s0, s11, 24 // (op 1213) 
s_ashr_i32 s18, s0, 31 // (op 1213) 

s_xor_b32 s0, s1, s18 // (op 1213) 

s_sub_i32 s1, s0, s18 // (op 1213) 


s_mul_hi_i32 s0, s1, s12 // (op 1213) 
s_add_i32 s18, s0, s1 // (op 1213) 

s_ashr_i32 s0, s18, 31 // (op 1213) 
s_cmp_eq_i32 s12, 0 // (op 1213) 
s_mov_b32 s19, scc // (op 1213) 
s_cmp_eq_i32 s19, 0 // (op 1213) 
s_cselect_b32 s22, 0, -1 // (op 1213) 

s_and_b32 s19, s13, 31 // (op 1213) 
s_lshl_b32 s23, 1, s19 // (op 1213) 
s_add_i32 s24, s23, s22 // (op 1213) 


s_and_b32 s25, s0, s24 // (op 1213) 


s_add_i32 s0, s18, s25 // (op 1213) 


s_ashr_i32 s18, s0, s19 // (op 1213) 


s_lshl_b32 s0, s13, 24 // (op 1213) 
s_ashr_i32 s19, s0, 31 // (op 1213) 

s_xor_b32 s0, s18, s19 // (op 1213) 

s_sub_i32 s18, s0, s19 // (op 1213) 


s_mul_i32 s0, s1, s15 // (op 1213) 

s_sub_u32 s1, s16, s0 // (op 1213) 

s_mul_i32 s0, s18, s15 // (op 1213) 
s_add_u32 s0, s0, s1 // (op 1213) 


s_mul_hi_i32 s1, s16, s9 // (op 1213) 
s_add_i32 s18, s1, s16 // (op 1213) 

s_ashr_i32 s1, s18, 31 // (op 1213) 
s_cmp_eq_i32 s9, 0 // (op 1213) 
s_mov_b32 s19, scc // (op 1213) 
s_cmp_eq_i32 s19, 0 // (op 1213) 
s_cselect_b32 s22, 0, -1 // (op 1213) 

s_and_b32 s19, s46, 31 // (op 1213) 
s_lshl_b32 s23, 1, s19 // (op 1213) 
s_add_i32 s24, s23, s22 // (op 1213) 


s_and_b32 s25, s1, s24 // (op 1213) 


s_add_i32 s1, s18, s25 // (op 1213) 


s_ashr_i32 s18, s1, s19 // (op 1213) 


s_lshl_b32 s1, s46, 24 // (op 1213) 
s_ashr_i32 s19, s1, 31 // (op 1213) 

s_xor_b32 s1, s18, s19 // (op 1213) 

s_sub_i32 s18, s1, s19 // (op 1213) 


s_add_u32 s22, s6, 64 // (op 1213) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1213) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1213) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1213) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1213) 

s_mul_i32 s1, s15, s22 // (op 1213) 

s_mul_i32 s19, s8, s1 // (op 1213) 

s_mul_i32 s1, s18, s19 // (op 1213) 

s_sub_u32 s19, s16, s1 // (op 1213) 

s_mov_b32 s1, s19 // (op 1213) 
s_mul_hi_i32 s22, s1, s14 // (op 1213) 
s_add_i32 s23, s22, s1 // (op 1213) 


s_ashr_i32 s1, s23, 31 // (op 1213) 
s_cmp_eq_i32 s14, 0 // (op 1213) 
s_mov_b32 s22, scc // (op 1213) 
s_cmp_eq_i32 s22, 0 // (op 1213) 
s_cselect_b32 s24, 0, -1 // (op 1213) 

s_and_b32 s22, s48, 31 // (op 1213) 
s_lshl_b32 s25, 1, s22 // (op 1213) 
s_add_i32 s49, s25, s24 // (op 1213) 


s_and_b32 s24, s1, s49 // (op 1213) 


s_add_i32 s1, s23, s24 // (op 1213) 


s_ashr_i32 s23, s1, s22 // (op 1213) 


s_lshl_b32 s1, s48, 24 // (op 1213) 
s_ashr_i32 s22, s1, 31 // (op 1213) 

s_xor_b32 s1, s23, s22 // (op 1213) 

s_sub_i32 s23, s1, s22 // (op 1213) 


s_mul_i32 s1, s47, s15 // (op 1213) 
s_add_u32 s24, s20, 16 // (op 1213) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1213) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 1213) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 1213) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 4 // (op 1213) 

s_sub_i32 s22, s24, s1 // (op 1213) 

s_mul_i32 s24, s23, s22 // (op 1213) 


s_sub_u32 s25, s19, s24 // (op 1213) 


s_add_u32 s19, s25, s1 // (op 1213) 


s_cmp_eq_i32 s18, 0 // (op 1213) 
s_mov_b32 s1, scc // (op 1213) 

s_cmp_eq_i32 s1, 0 // (op 1213) 
s_cselect_b32 s18, s19, s0 // (op 1213) 



s_lshl_b32 s19, s18, 4 // (op 1213) 

s_mov_b32 s1, 0 // (op 1213) convert
s_mov_b32 s0, s19 // (op 1213) convert

v_mul_lo_u32 v17, v14, s27 // (op 1213) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 1213) most significant: high of low * low
v_mul_lo_u32 v16, v14, s26 // (op 1213) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1213) most significant: sum


v_mov_b32 v14, s0 // (op 1213) 
v_mov_b32 v15, s1 // (op 1213) 
v_add_co_u32 v18, s[18:19], v14, v16 // (op 1213) least significant half
v_addc_co_u32 v19, s[18:19], v15, v17, s[18:19] // (op 1213) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 1213) 


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1213) 

v_mov_b32 v16, v14 // (op 1213) convert

s_mul_i32 s1, s26, s31 // (op 1215) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1215) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1215) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1215) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1215) most significant: sum
s_add_u32 s1, s1, s19 // (op 1215) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1215) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1215) 

s_mov_b32 s52, s28 // (op 1215) reg expression
s_mov_b32 s53, s29 // (op 1215) reg expression
s_mov_b32 s54, s0 // (op 1215) reg expression
s_mov_b32 s55, 131072 // (op 1215) call()

s_mul_hi_i32 s0, s16, s10 // (op 1232) 
v_lshlrev_b32 v14, 1, v1 // (op 1232) 
s_add_i32 s1, s0, s16 // (op 1232) 

s_ashr_i32 s0, s1, 31 // (op 1232) 
s_cmp_eq_i32 s10, 0 // (op 1232) 
s_mov_b32 s18, scc // (op 1232) 
s_cmp_eq_i32 s18, 0 // (op 1232) 
s_cselect_b32 s19, 0, -1 // (op 1232) 

s_and_b32 s18, s11, 31 // (op 1232) 
s_lshl_b32 s22, 1, s18 // (op 1232) 
s_add_i32 s23, s22, s19 // (op 1232) 


s_and_b32 s19, s0, s23 // (op 1232) 


s_add_i32 s0, s1, s19 // (op 1232) 


s_ashr_i32 s1, s0, s18 // (op 1232) 


s_lshl_b32 s0, s11, 24 // (op 1232) 
s_ashr_i32 s18, s0, 31 // (op 1232) 

s_xor_b32 s0, s1, s18 // (op 1232) 

s_sub_i32 s1, s0, s18 // (op 1232) 


s_mul_hi_i32 s0, s1, s12 // (op 1232) 
s_add_i32 s18, s0, s1 // (op 1232) 

s_ashr_i32 s0, s18, 31 // (op 1232) 
s_cmp_eq_i32 s12, 0 // (op 1232) 
s_mov_b32 s19, scc // (op 1232) 
s_cmp_eq_i32 s19, 0 // (op 1232) 
s_cselect_b32 s22, 0, -1 // (op 1232) 

s_and_b32 s19, s13, 31 // (op 1232) 
s_lshl_b32 s23, 1, s19 // (op 1232) 
s_add_i32 s24, s23, s22 // (op 1232) 


s_and_b32 s25, s0, s24 // (op 1232) 


s_add_i32 s0, s18, s25 // (op 1232) 


s_ashr_i32 s18, s0, s19 // (op 1232) 


s_lshl_b32 s0, s13, 24 // (op 1232) 
s_ashr_i32 s19, s0, 31 // (op 1232) 

s_xor_b32 s0, s18, s19 // (op 1232) 

s_sub_i32 s18, s0, s19 // (op 1232) 


s_mul_i32 s0, s1, s15 // (op 1232) 

s_sub_u32 s1, s16, s0 // (op 1232) 

s_mul_i32 s0, s18, s15 // (op 1232) 
s_add_u32 s0, s0, s1 // (op 1232) 


s_mul_hi_i32 s1, s16, s9 // (op 1232) 
s_add_i32 s18, s1, s16 // (op 1232) 

s_ashr_i32 s1, s18, 31 // (op 1232) 
s_cmp_eq_i32 s9, 0 // (op 1232) 
s_mov_b32 s19, scc // (op 1232) 
s_cmp_eq_i32 s19, 0 // (op 1232) 
s_cselect_b32 s22, 0, -1 // (op 1232) 

s_and_b32 s19, s46, 31 // (op 1232) 
s_lshl_b32 s23, 1, s19 // (op 1232) 
s_add_i32 s24, s23, s22 // (op 1232) 


s_and_b32 s25, s1, s24 // (op 1232) 


s_add_i32 s1, s18, s25 // (op 1232) 


s_ashr_i32 s18, s1, s19 // (op 1232) 


s_lshl_b32 s1, s46, 24 // (op 1232) 
s_ashr_i32 s19, s1, 31 // (op 1232) 

s_xor_b32 s1, s18, s19 // (op 1232) 

s_sub_i32 s18, s1, s19 // (op 1232) 


s_add_u32 s50, s6, 64 // (op 1232) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1232) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1232) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1232) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1232) 

s_mul_i32 s1, s15, s24 // (op 1232) 

s_mul_i32 s19, s8, s1 // (op 1232) 

s_mul_i32 s1, s18, s19 // (op 1232) 

s_sub_u32 s19, s16, s1 // (op 1232) 

s_mov_b32 s1, s19 // (op 1232) 
s_mul_hi_i32 s22, s1, s14 // (op 1232) 
s_add_i32 s23, s22, s1 // (op 1232) 


s_ashr_i32 s1, s23, 31 // (op 1232) 
s_cmp_eq_i32 s14, 0 // (op 1232) 
s_mov_b32 s22, scc // (op 1232) 
s_cmp_eq_i32 s22, 0 // (op 1232) 
s_cselect_b32 s24, 0, -1 // (op 1232) 

s_and_b32 s22, s48, 31 // (op 1232) 
s_lshl_b32 s25, 1, s22 // (op 1232) 
s_add_i32 s49, s25, s24 // (op 1232) 


s_and_b32 s24, s1, s49 // (op 1232) 


s_add_i32 s1, s23, s24 // (op 1232) 


s_ashr_i32 s23, s1, s22 // (op 1232) 


s_lshl_b32 s1, s48, 24 // (op 1232) 
s_ashr_i32 s22, s1, 31 // (op 1232) 

s_xor_b32 s1, s23, s22 // (op 1232) 

s_sub_i32 s23, s1, s22 // (op 1232) 


s_mul_i32 s1, s47, s15 // (op 1232) 
s_add_u32 s24, s20, 16 // (op 1232) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1232) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1232) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1232) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1232) 

s_sub_i32 s22, s24, s1 // (op 1232) 

s_mul_i32 s24, s23, s22 // (op 1232) 


s_sub_u32 s25, s19, s24 // (op 1232) 


s_add_u32 s19, s25, s1 // (op 1232) 


s_cmp_eq_i32 s18, 0 // (op 1232) 
s_mov_b32 s1, scc // (op 1232) 

s_cmp_eq_i32 s1, 0 // (op 1232) 
s_cselect_b32 s18, s19, s0 // (op 1232) 



s_lshl_b32 s19, s18, 4 // (op 1232) 

s_mov_b32 s1, 0 // (op 1232) convert
s_mov_b32 s0, s19 // (op 1232) convert

v_mul_lo_u32 v19, v14, s27 // (op 1232) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 1232) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 1232) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1232) most significant: sum


v_mov_b32 v17, s0 // (op 1232) 
v_mov_b32 v14, s1 // (op 1232) 
v_add_co_u32 v20, s[18:19], v17, v18 // (op 1232) least significant half
v_addc_co_u32 v21, s[18:19], v14, v19, s[18:19] // (op 1232) most significant half



v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 1232) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 1232) 

v_mov_b32 v17, v14 // (op 1232) convert

s_mul_i32 s1, s26, s31 // (op 1234) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1234) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1234) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1234) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1234) most significant: sum
s_add_u32 s1, s1, s19 // (op 1234) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1234) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1234) 

s_mov_b32 s52, s28 // (op 1234) reg expression
s_mov_b32 s53, s29 // (op 1234) reg expression
s_mov_b32 s54, s0 // (op 1234) reg expression
s_mov_b32 s55, 131072 // (op 1234) call()

s_mul_hi_i32 s0, s16, s10 // (op 1277) 
v_lshrrev_b32 v14, 2, v1 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

v_lshlrev_b32 v15, 1, v14 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
v_and_b32 v14, -4, v1 // (op 1277) 
s_cmp_eq_i32 s10, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
v_sub_u32 v18, v1, v14 // (op 1277) 

s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

v_lshlrev_b32 v14, 6, v18 // (op 1277) 

v_mov_b32 v19, 0 // (op 1277) convert
v_mov_b32 v18, v14 // (op 1277) convert

s_and_b32 s18, s11, 31 // (op 1277) 
s_lshl_b32 s22, 1, s18 // (op 1277) 
s_add_i32 s23, s22, s19 // (op 1277) 


s_and_b32 s19, s0, s23 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s11, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_hi_i32 s0, s1, s12 // (op 1277) 
s_add_i32 s18, s0, s1 // (op 1277) 

s_ashr_i32 s0, s18, 31 // (op 1277) 
s_cmp_eq_i32 s12, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s22, 0, -1 // (op 1277) 

s_and_b32 s19, s13, 31 // (op 1277) 
s_lshl_b32 s23, 1, s19 // (op 1277) 
s_add_i32 s24, s23, s22 // (op 1277) 


s_and_b32 s25, s0, s24 // (op 1277) 


s_add_i32 s0, s18, s25 // (op 1277) 


s_ashr_i32 s18, s0, s19 // (op 1277) 


s_lshl_b32 s0, s13, 24 // (op 1277) 
s_ashr_i32 s19, s0, 31 // (op 1277) 

s_xor_b32 s0, s18, s19 // (op 1277) 

s_sub_i32 s18, s0, s19 // (op 1277) 


s_add_u32 s50, s6, 64 // (op 1277) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1277) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1277) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1277) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1277) 

s_mul_i32 s0, s18, s24 // (op 1277) 

s_sub_i32 s18, s1, s0 // (op 1277) 


s_mul_hi_i32 s19, s16, s9 // (op 1277) 
s_add_i32 s0, s19, s16 // (op 1277) 

s_ashr_i32 s1, s0, 31 // (op 1277) 
s_cmp_eq_i32 s9, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s22, 0, -1 // (op 1277) 

s_and_b32 s19, s46, 31 // (op 1277) 
s_lshl_b32 s23, 1, s19 // (op 1277) 
s_add_i32 s49, s23, s22 // (op 1277) 


s_and_b32 s22, s1, s49 // (op 1277) 


s_add_i32 s1, s0, s22 // (op 1277) 


s_ashr_i32 s0, s1, s19 // (op 1277) 


s_lshl_b32 s1, s46, 24 // (op 1277) 
s_ashr_i32 s19, s1, 31 // (op 1277) 

s_xor_b32 s1, s0, s19 // (op 1277) 

s_sub_i32 s0, s1, s19 // (op 1277) 


s_mul_i32 s1, s15, s24 // (op 1277) 

s_mul_i32 s19, s8, s1 // (op 1277) 

s_mul_i32 s1, s0, s19 // (op 1277) 

s_sub_u32 s19, s16, s1 // (op 1277) 

s_mul_hi_i32 s1, s19, s14 // (op 1277) 
s_add_i32 s22, s1, s19 // (op 1277) 


s_ashr_i32 s1, s22, 31 // (op 1277) 
s_cmp_eq_i32 s14, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s23, 0, -1 // (op 1277) 

s_and_b32 s19, s48, 31 // (op 1277) 
s_lshl_b32 s24, 1, s19 // (op 1277) 
s_add_i32 s25, s24, s23 // (op 1277) 


s_and_b32 s23, s1, s25 // (op 1277) 


s_add_i32 s1, s22, s23 // (op 1277) 


s_ashr_i32 s22, s1, s19 // (op 1277) 


s_lshl_b32 s1, s48, 24 // (op 1277) 
s_ashr_i32 s19, s1, 31 // (op 1277) 

s_xor_b32 s1, s22, s19 // (op 1277) 

s_sub_i32 s22, s1, s19 // (op 1277) 


s_cmp_eq_i32 s0, 0 // (op 1277) 
s_mov_b32 s1, scc // (op 1277) 

s_cmp_eq_i32 s1, 0 // (op 1277) 
s_cselect_b32 s0, s22, s18 // (op 1277) 



v_lshl_add_u32 v14, s0, 6, v15 // (op 1277) 


v_mul_lo_u32 v21, v14, s33 // (op 1277) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v20, v14, s32 // (op 1277) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1277) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 1277) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 1277) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 1277) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 1277) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1277) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1277) 

v_mov_b32 v18, v14 // (op 1277) convert

s_mul_i32 s1, s7, 6 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1279) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1279) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1279) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1279) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1279) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1279) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1279) most significant: sum
s_add_u32 s19, s19, s23 // (op 1279) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1279) 

s_mov_b32 s64, s34 // (op 1279) reg expression
s_mov_b32 s65, s35 // (op 1279) reg expression
s_mov_b32 s66, s0 // (op 1279) reg expression
s_mov_b32 s67, 131072 // (op 1279) call()

s_mul_hi_i32 s0, s16, s10 // (op 1296) 
v_lshrrev_b32 v14, 2, v1 // (op 1296) 
s_add_i32 s1, s0, s16 // (op 1296) 

v_lshlrev_b32 v15, 1, v14 // (op 1296) 

s_ashr_i32 s0, s1, 31 // (op 1296) 
v_and_b32 v14, -4, v1 // (op 1296) 
s_cmp_eq_i32 s10, 0 // (op 1296) 
s_mov_b32 s18, scc // (op 1296) 
v_sub_u32 v19, v1, v14 // (op 1296) 

s_cmp_eq_i32 s18, 0 // (op 1296) 
s_cselect_b32 s19, 0, -1 // (op 1296) 

v_lshlrev_b32 v14, 6, v19 // (op 1296) 

v_mov_b32 v21, 0 // (op 1296) convert
v_mov_b32 v20, v14 // (op 1296) convert

s_and_b32 s18, s11, 31 // (op 1296) 
s_lshl_b32 s22, 1, s18 // (op 1296) 
s_add_i32 s23, s22, s19 // (op 1296) 


s_and_b32 s19, s0, s23 // (op 1296) 


s_add_i32 s0, s1, s19 // (op 1296) 


s_ashr_i32 s1, s0, s18 // (op 1296) 


s_lshl_b32 s0, s11, 24 // (op 1296) 
s_ashr_i32 s18, s0, 31 // (op 1296) 

s_xor_b32 s0, s1, s18 // (op 1296) 

s_sub_i32 s1, s0, s18 // (op 1296) 


s_mul_hi_i32 s0, s1, s12 // (op 1296) 
s_add_i32 s18, s0, s1 // (op 1296) 

s_ashr_i32 s0, s18, 31 // (op 1296) 
s_cmp_eq_i32 s12, 0 // (op 1296) 
s_mov_b32 s19, scc // (op 1296) 
s_cmp_eq_i32 s19, 0 // (op 1296) 
s_cselect_b32 s22, 0, -1 // (op 1296) 

s_and_b32 s19, s13, 31 // (op 1296) 
s_lshl_b32 s23, 1, s19 // (op 1296) 
s_add_i32 s24, s23, s22 // (op 1296) 


s_and_b32 s25, s0, s24 // (op 1296) 


s_add_i32 s0, s18, s25 // (op 1296) 


s_ashr_i32 s18, s0, s19 // (op 1296) 


s_lshl_b32 s0, s13, 24 // (op 1296) 
s_ashr_i32 s19, s0, 31 // (op 1296) 

s_xor_b32 s0, s18, s19 // (op 1296) 

s_sub_i32 s18, s0, s19 // (op 1296) 


s_add_u32 s50, s6, 64 // (op 1296) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1296) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1296) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1296) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1296) 

s_mul_i32 s0, s18, s24 // (op 1296) 

s_sub_i32 s18, s1, s0 // (op 1296) 


s_mul_hi_i32 s19, s16, s9 // (op 1296) 
s_add_i32 s0, s19, s16 // (op 1296) 

s_ashr_i32 s1, s0, 31 // (op 1296) 
s_cmp_eq_i32 s9, 0 // (op 1296) 
s_mov_b32 s19, scc // (op 1296) 
s_cmp_eq_i32 s19, 0 // (op 1296) 
s_cselect_b32 s22, 0, -1 // (op 1296) 

s_and_b32 s19, s46, 31 // (op 1296) 
s_lshl_b32 s23, 1, s19 // (op 1296) 
s_add_i32 s49, s23, s22 // (op 1296) 


s_and_b32 s22, s1, s49 // (op 1296) 


s_add_i32 s1, s0, s22 // (op 1296) 


s_ashr_i32 s0, s1, s19 // (op 1296) 


s_lshl_b32 s1, s46, 24 // (op 1296) 
s_ashr_i32 s19, s1, 31 // (op 1296) 

s_xor_b32 s1, s0, s19 // (op 1296) 

s_sub_i32 s0, s1, s19 // (op 1296) 


s_mul_i32 s1, s15, s24 // (op 1296) 

s_mul_i32 s19, s8, s1 // (op 1296) 

s_mul_i32 s1, s0, s19 // (op 1296) 

s_sub_u32 s19, s16, s1 // (op 1296) 

s_mul_hi_i32 s1, s19, s14 // (op 1296) 
s_add_i32 s22, s1, s19 // (op 1296) 


s_ashr_i32 s1, s22, 31 // (op 1296) 
s_cmp_eq_i32 s14, 0 // (op 1296) 
s_mov_b32 s19, scc // (op 1296) 
s_cmp_eq_i32 s19, 0 // (op 1296) 
s_cselect_b32 s23, 0, -1 // (op 1296) 

s_and_b32 s19, s48, 31 // (op 1296) 
s_lshl_b32 s24, 1, s19 // (op 1296) 
s_add_i32 s25, s24, s23 // (op 1296) 


s_and_b32 s23, s1, s25 // (op 1296) 


s_add_i32 s1, s22, s23 // (op 1296) 


s_ashr_i32 s22, s1, s19 // (op 1296) 


s_lshl_b32 s1, s48, 24 // (op 1296) 
s_ashr_i32 s19, s1, 31 // (op 1296) 

s_xor_b32 s1, s22, s19 // (op 1296) 

s_sub_i32 s22, s1, s19 // (op 1296) 


s_cmp_eq_i32 s0, 0 // (op 1296) 
s_mov_b32 s1, scc // (op 1296) 

s_cmp_eq_i32 s1, 0 // (op 1296) 
s_cselect_b32 s0, s22, s18 // (op 1296) 



v_lshl_add_u32 v14, s0, 6, v15 // (op 1296) 


v_mul_lo_u32 v23, v14, s33 // (op 1296) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 1296) most significant: high of low * low
v_mul_lo_u32 v22, v14, s32 // (op 1296) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1296) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1296) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1296) most significant half



v_mul_lo_u32 v21, v15, 6 // (op 1296) most significant: low of high * low
v_mul_hi_u32 v19, v14, 6 // (op 1296) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 1296) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1296) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 1296) 

v_mov_b32 v19, v14 // (op 1296) convert

s_mul_i32 s1, s7, 6 // (op 1298) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1298) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1298) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1298) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1298) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1298) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1298) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1298) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1298) most significant: sum
s_add_u32 s19, s19, s23 // (op 1298) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1298) 

s_mov_b32 s64, s34 // (op 1298) reg expression
s_mov_b32 s65, s35 // (op 1298) reg expression
s_mov_b32 s66, s0 // (op 1298) reg expression
s_mov_b32 s67, 131072 // (op 1298) call()

s_mul_hi_i32 s0, s16, s10 // (op 1341) 
v_and_b32 v14, -16, v1 // (op 1341) 
s_add_i32 s1, s0, s16 // (op 1341) 

v_sub_u32 v15, v1, v14 // (op 1341) 

s_ashr_i32 s0, s1, 31 // (op 1341) 
v_lshrrev_b32 v14, 4, v1 // (op 1341) 
s_cmp_eq_i32 s10, 0 // (op 1341) 
s_mov_b32 s18, scc // (op 1341) 
s_cmp_eq_i32 s18, 0 // (op 1341) 
s_cselect_b32 s19, 0, -1 // (op 1341) 

s_and_b32 s18, s11, 31 // (op 1341) 
s_lshl_b32 s22, 1, s18 // (op 1341) 
s_add_i32 s23, s22, s19 // (op 1341) 


s_and_b32 s19, s0, s23 // (op 1341) 


s_add_i32 s0, s1, s19 // (op 1341) 


s_ashr_i32 s1, s0, s18 // (op 1341) 


s_lshl_b32 s0, s11, 24 // (op 1341) 
s_ashr_i32 s18, s0, 31 // (op 1341) 

s_xor_b32 s0, s1, s18 // (op 1341) 

s_sub_i32 s1, s0, s18 // (op 1341) 


s_mul_hi_i32 s0, s1, s12 // (op 1341) 
s_add_i32 s18, s0, s1 // (op 1341) 

s_ashr_i32 s0, s18, 31 // (op 1341) 
s_cmp_eq_i32 s12, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s22, 0, -1 // (op 1341) 

s_and_b32 s19, s13, 31 // (op 1341) 
s_lshl_b32 s23, 1, s19 // (op 1341) 
s_add_i32 s24, s23, s22 // (op 1341) 


s_and_b32 s25, s0, s24 // (op 1341) 


s_add_i32 s0, s18, s25 // (op 1341) 


s_ashr_i32 s18, s0, s19 // (op 1341) 


s_lshl_b32 s0, s13, 24 // (op 1341) 
s_ashr_i32 s19, s0, 31 // (op 1341) 

s_xor_b32 s0, s18, s19 // (op 1341) 

s_sub_i32 s18, s0, s19 // (op 1341) 


s_mul_i32 s0, s1, s15 // (op 1341) 

s_sub_u32 s1, s16, s0 // (op 1341) 

s_mul_i32 s0, s18, s15 // (op 1341) 
s_add_u32 s0, s0, s1 // (op 1341) 


s_mul_hi_i32 s1, s16, s9 // (op 1341) 
s_add_i32 s18, s1, s16 // (op 1341) 

s_ashr_i32 s1, s18, 31 // (op 1341) 
s_cmp_eq_i32 s9, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s22, 0, -1 // (op 1341) 

s_and_b32 s19, s46, 31 // (op 1341) 
s_lshl_b32 s23, 1, s19 // (op 1341) 
s_add_i32 s24, s23, s22 // (op 1341) 


s_and_b32 s25, s1, s24 // (op 1341) 


s_add_i32 s1, s18, s25 // (op 1341) 


s_ashr_i32 s18, s1, s19 // (op 1341) 


s_lshl_b32 s1, s46, 24 // (op 1341) 
s_ashr_i32 s19, s1, 31 // (op 1341) 

s_xor_b32 s1, s18, s19 // (op 1341) 

s_sub_i32 s18, s1, s19 // (op 1341) 


s_add_u32 s50, s6, 64 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1341) 

s_mul_i32 s1, s15, s24 // (op 1341) 

s_mul_i32 s19, s8, s1 // (op 1341) 

s_mul_i32 s1, s18, s19 // (op 1341) 

s_sub_u32 s19, s16, s1 // (op 1341) 

s_mov_b32 s1, s19 // (op 1341) 
s_mul_hi_i32 s22, s1, s14 // (op 1341) 
s_add_i32 s23, s22, s1 // (op 1341) 


s_ashr_i32 s1, s23, 31 // (op 1341) 
s_cmp_eq_i32 s14, 0 // (op 1341) 
s_mov_b32 s22, scc // (op 1341) 
s_cmp_eq_i32 s22, 0 // (op 1341) 
s_cselect_b32 s24, 0, -1 // (op 1341) 

s_and_b32 s22, s48, 31 // (op 1341) 
s_lshl_b32 s25, 1, s22 // (op 1341) 
s_add_i32 s49, s25, s24 // (op 1341) 


s_and_b32 s24, s1, s49 // (op 1341) 


s_add_i32 s1, s23, s24 // (op 1341) 


s_ashr_i32 s23, s1, s22 // (op 1341) 


s_lshl_b32 s1, s48, 24 // (op 1341) 
s_ashr_i32 s22, s1, 31 // (op 1341) 

s_xor_b32 s1, s23, s22 // (op 1341) 

s_sub_i32 s23, s1, s22 // (op 1341) 


s_mul_i32 s1, s47, s15 // (op 1341) 
s_add_u32 s24, s20, 16 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1341) 

s_sub_i32 s22, s24, s1 // (op 1341) 

s_mul_i32 s24, s23, s22 // (op 1341) 


s_sub_u32 s25, s19, s24 // (op 1341) 


s_add_u32 s19, s25, s1 // (op 1341) 


s_cmp_eq_i32 s18, 0 // (op 1341) 
s_mov_b32 s1, scc // (op 1341) 

s_cmp_eq_i32 s1, 0 // (op 1341) 
s_cselect_b32 s18, s19, s0 // (op 1341) 



v_lshl_add_u32 v20, s18, 4, v15 // (op 1341) 


v_mul_lo_u32 v15, v14, s36 // (op 1341) 

v_add_u32 v14, v20, v15 // (op 1341) 


s_mul_i32 s1, s36, s41 // (op 1343) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1343) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1343) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1343) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1343) most significant: sum
s_add_u32 s1, s1, s19 // (op 1343) most significant: sum


s_mov_b32 s68, s38 // (op 1343) reg expression
s_mov_b32 s69, s39 // (op 1343) reg expression
s_mov_b32 s70, s0 // (op 1343) reg expression
s_mov_b32 s71, 131072 // (op 1343) call()

s_mul_hi_i32 s0, s16, s10 // (op 1360) 
v_and_b32 v15, -16, v1 // (op 1360) 
s_add_i32 s1, s0, s16 // (op 1360) 

v_sub_u32 v20, v1, v15 // (op 1360) 

s_ashr_i32 s0, s1, 31 // (op 1360) 
v_lshrrev_b32 v15, 4, v1 // (op 1360) 
s_cmp_eq_i32 s10, 0 // (op 1360) 
s_mov_b32 s18, scc // (op 1360) 
s_cmp_eq_i32 s18, 0 // (op 1360) 
s_cselect_b32 s19, 0, -1 // (op 1360) 

s_and_b32 s18, s11, 31 // (op 1360) 
s_lshl_b32 s22, 1, s18 // (op 1360) 
s_add_i32 s23, s22, s19 // (op 1360) 


s_and_b32 s19, s0, s23 // (op 1360) 


s_add_i32 s0, s1, s19 // (op 1360) 


s_ashr_i32 s1, s0, s18 // (op 1360) 


s_lshl_b32 s0, s11, 24 // (op 1360) 
s_ashr_i32 s18, s0, 31 // (op 1360) 

s_xor_b32 s0, s1, s18 // (op 1360) 

s_sub_i32 s1, s0, s18 // (op 1360) 


s_mul_hi_i32 s0, s1, s12 // (op 1360) 
s_add_i32 s18, s0, s1 // (op 1360) 

s_ashr_i32 s0, s18, 31 // (op 1360) 
s_cmp_eq_i32 s12, 0 // (op 1360) 
s_mov_b32 s19, scc // (op 1360) 
s_cmp_eq_i32 s19, 0 // (op 1360) 
s_cselect_b32 s22, 0, -1 // (op 1360) 

s_and_b32 s19, s13, 31 // (op 1360) 
s_lshl_b32 s23, 1, s19 // (op 1360) 
s_add_i32 s24, s23, s22 // (op 1360) 


s_and_b32 s25, s0, s24 // (op 1360) 


s_add_i32 s0, s18, s25 // (op 1360) 


s_ashr_i32 s18, s0, s19 // (op 1360) 


s_lshl_b32 s0, s13, 24 // (op 1360) 
s_ashr_i32 s19, s0, 31 // (op 1360) 

s_xor_b32 s0, s18, s19 // (op 1360) 

s_sub_i32 s18, s0, s19 // (op 1360) 


s_mul_i32 s0, s1, s15 // (op 1360) 

s_sub_u32 s1, s16, s0 // (op 1360) 

s_mul_i32 s0, s18, s15 // (op 1360) 
s_add_u32 s0, s0, s1 // (op 1360) 


s_mul_hi_i32 s1, s16, s9 // (op 1360) 
s_add_i32 s18, s1, s16 // (op 1360) 

s_ashr_i32 s1, s18, 31 // (op 1360) 
s_cmp_eq_i32 s9, 0 // (op 1360) 
s_mov_b32 s19, scc // (op 1360) 
s_cmp_eq_i32 s19, 0 // (op 1360) 
s_cselect_b32 s22, 0, -1 // (op 1360) 

s_and_b32 s19, s46, 31 // (op 1360) 
s_lshl_b32 s23, 1, s19 // (op 1360) 
s_add_i32 s24, s23, s22 // (op 1360) 


s_and_b32 s25, s1, s24 // (op 1360) 


s_add_i32 s1, s18, s25 // (op 1360) 


s_ashr_i32 s18, s1, s19 // (op 1360) 


s_lshl_b32 s1, s46, 24 // (op 1360) 
s_ashr_i32 s19, s1, 31 // (op 1360) 

s_xor_b32 s1, s18, s19 // (op 1360) 

s_sub_i32 s18, s1, s19 // (op 1360) 


s_add_u32 s50, s6, 64 // (op 1360) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1360) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1360) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1360) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1360) 

s_mul_i32 s1, s15, s24 // (op 1360) 

s_mul_i32 s19, s8, s1 // (op 1360) 

s_mul_i32 s1, s18, s19 // (op 1360) 

s_sub_u32 s19, s16, s1 // (op 1360) 

s_mov_b32 s1, s19 // (op 1360) 
s_mul_hi_i32 s22, s1, s14 // (op 1360) 
s_add_i32 s23, s22, s1 // (op 1360) 


s_ashr_i32 s1, s23, 31 // (op 1360) 
s_cmp_eq_i32 s14, 0 // (op 1360) 
s_mov_b32 s22, scc // (op 1360) 
s_cmp_eq_i32 s22, 0 // (op 1360) 
s_cselect_b32 s24, 0, -1 // (op 1360) 

s_and_b32 s22, s48, 31 // (op 1360) 
s_lshl_b32 s25, 1, s22 // (op 1360) 
s_add_i32 s49, s25, s24 // (op 1360) 


s_and_b32 s24, s1, s49 // (op 1360) 


s_add_i32 s1, s23, s24 // (op 1360) 


s_ashr_i32 s23, s1, s22 // (op 1360) 


s_lshl_b32 s1, s48, 24 // (op 1360) 
s_ashr_i32 s22, s1, 31 // (op 1360) 

s_xor_b32 s1, s23, s22 // (op 1360) 

s_sub_i32 s23, s1, s22 // (op 1360) 


s_mul_i32 s1, s47, s15 // (op 1360) 
s_add_u32 s24, s20, 16 // (op 1360) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1360) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1360) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1360) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1360) 

s_sub_i32 s22, s24, s1 // (op 1360) 

s_mul_i32 s24, s23, s22 // (op 1360) 


s_sub_u32 s25, s19, s24 // (op 1360) 


s_add_u32 s19, s25, s1 // (op 1360) 


s_cmp_eq_i32 s18, 0 // (op 1360) 
s_mov_b32 s1, scc // (op 1360) 

s_cmp_eq_i32 s1, 0 // (op 1360) 
s_cselect_b32 s18, s19, s0 // (op 1360) 



v_lshl_add_u32 v21, s18, 4, v20 // (op 1360) 


v_mul_lo_u32 v20, v15, s36 // (op 1360) 

v_add_u32 v15, v21, v20 // (op 1360) 


s_mul_i32 s1, s36, s41 // (op 1362) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1362) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1362) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1362) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1362) most significant: sum
s_add_u32 s1, s1, s19 // (op 1362) most significant: sum


s_mov_b32 s68, s38 // (op 1362) reg expression
s_mov_b32 s69, s39 // (op 1362) reg expression
s_mov_b32 s70, s0 // (op 1362) reg expression
s_mov_b32 s71, 131072 // (op 1362) call()

s_mul_hi_i32 s0, s16, s10 // (op 1405) 
v_lshrrev_b32 v20, 1, v1 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

v_and_b32 v21, -2, v1 // (op 1405) 
s_ashr_i32 s0, s1, 31 // (op 1405) 
v_sub_u32 v22, v1, v21 // (op 1405) 

s_cmp_eq_i32 s10, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s11, 31 // (op 1405) 
s_lshl_b32 s22, 1, s18 // (op 1405) 
s_add_i32 s23, s22, s19 // (op 1405) 


s_and_b32 s19, s0, s23 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s11, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_hi_i32 s0, s1, s12 // (op 1405) 
s_add_i32 s18, s0, s1 // (op 1405) 

s_ashr_i32 s0, s18, 31 // (op 1405) 
s_cmp_eq_i32 s12, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s22, 0, -1 // (op 1405) 

s_and_b32 s19, s13, 31 // (op 1405) 
s_lshl_b32 s23, 1, s19 // (op 1405) 
s_add_i32 s24, s23, s22 // (op 1405) 


s_and_b32 s25, s0, s24 // (op 1405) 


s_add_i32 s0, s18, s25 // (op 1405) 


s_ashr_i32 s18, s0, s19 // (op 1405) 


s_lshl_b32 s0, s13, 24 // (op 1405) 
s_ashr_i32 s19, s0, 31 // (op 1405) 

s_xor_b32 s0, s18, s19 // (op 1405) 

s_sub_i32 s18, s0, s19 // (op 1405) 


s_add_u32 s50, s6, 64 // (op 1405) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1405) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1405) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1405) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1405) 

s_mul_i32 s0, s18, s24 // (op 1405) 

s_sub_i32 s18, s1, s0 // (op 1405) 


s_mul_hi_i32 s19, s16, s9 // (op 1405) 
s_add_i32 s0, s19, s16 // (op 1405) 

s_ashr_i32 s1, s0, 31 // (op 1405) 
s_cmp_eq_i32 s9, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s22, 0, -1 // (op 1405) 

s_and_b32 s19, s46, 31 // (op 1405) 
s_lshl_b32 s23, 1, s19 // (op 1405) 
s_add_i32 s49, s23, s22 // (op 1405) 


s_and_b32 s22, s1, s49 // (op 1405) 


s_add_i32 s1, s0, s22 // (op 1405) 


s_ashr_i32 s0, s1, s19 // (op 1405) 


s_lshl_b32 s1, s46, 24 // (op 1405) 
s_ashr_i32 s19, s1, 31 // (op 1405) 

s_xor_b32 s1, s0, s19 // (op 1405) 

s_sub_i32 s0, s1, s19 // (op 1405) 


s_mul_i32 s1, s15, s24 // (op 1405) 

s_mul_i32 s19, s8, s1 // (op 1405) 

s_mul_i32 s1, s0, s19 // (op 1405) 

s_sub_u32 s19, s16, s1 // (op 1405) 

s_mul_hi_i32 s1, s19, s14 // (op 1405) 
s_add_i32 s22, s1, s19 // (op 1405) 


s_ashr_i32 s1, s22, 31 // (op 1405) 
s_cmp_eq_i32 s14, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s23, 0, -1 // (op 1405) 

s_and_b32 s19, s48, 31 // (op 1405) 
s_lshl_b32 s24, 1, s19 // (op 1405) 
s_add_i32 s25, s24, s23 // (op 1405) 


s_and_b32 s23, s1, s25 // (op 1405) 


s_add_i32 s1, s22, s23 // (op 1405) 


s_ashr_i32 s22, s1, s19 // (op 1405) 


s_lshl_b32 s1, s48, 24 // (op 1405) 
s_ashr_i32 s19, s1, 31 // (op 1405) 

s_xor_b32 s1, s22, s19 // (op 1405) 

s_sub_i32 s22, s1, s19 // (op 1405) 


s_cmp_eq_i32 s0, 0 // (op 1405) 
s_mov_b32 s1, scc // (op 1405) 

s_cmp_eq_i32 s1, 0 // (op 1405) 
s_cselect_b32 s0, s22, s18 // (op 1405) 



v_lshl_add_u32 v21, s0, 6, v20 // (op 1405) 


v_mul_lo_u32 v20, v21, s42 // (op 1405) 

v_lshl_add_u32 v21, v22, 2, v20 // (op 1405) 


s_mul_i32 s1, s42, s7 // (op 1407) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1407) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1407) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1407) most significant: sum
s_add_u32 s1, s1, s19 // (op 1407) most significant: sum


s_mov_b32 s72, s44 // (op 1407) reg expression
s_mov_b32 s73, s45 // (op 1407) reg expression
s_mov_b32 s74, s0 // (op 1407) reg expression
s_mov_b32 s75, 131072 // (op 1407) call()

s_mul_hi_i32 s0, s16, s10 // (op 1424) 
v_lshrrev_b32 v20, 1, v1 // (op 1424) 
s_add_i32 s1, s0, s16 // (op 1424) 

v_and_b32 v22, -2, v1 // (op 1424) 
s_ashr_i32 s0, s1, 31 // (op 1424) 
v_sub_u32 v23, v1, v22 // (op 1424) 

s_cmp_eq_i32 s10, 0 // (op 1424) 
s_mov_b32 s18, scc // (op 1424) 
s_cmp_eq_i32 s18, 0 // (op 1424) 
s_cselect_b32 s19, 0, -1 // (op 1424) 

s_and_b32 s18, s11, 31 // (op 1424) 
s_lshl_b32 s22, 1, s18 // (op 1424) 
s_add_i32 s23, s22, s19 // (op 1424) 


s_and_b32 s19, s0, s23 // (op 1424) 


s_add_i32 s0, s1, s19 // (op 1424) 


s_ashr_i32 s1, s0, s18 // (op 1424) 


s_lshl_b32 s0, s11, 24 // (op 1424) 
s_ashr_i32 s18, s0, 31 // (op 1424) 

s_xor_b32 s0, s1, s18 // (op 1424) 

s_sub_i32 s1, s0, s18 // (op 1424) 


s_mul_hi_i32 s0, s1, s12 // (op 1424) 
s_add_i32 s18, s0, s1 // (op 1424) 

s_ashr_i32 s0, s18, 31 // (op 1424) 
s_cmp_eq_i32 s12, 0 // (op 1424) 
s_mov_b32 s19, scc // (op 1424) 
s_cmp_eq_i32 s19, 0 // (op 1424) 
s_cselect_b32 s22, 0, -1 // (op 1424) 

s_and_b32 s19, s13, 31 // (op 1424) 
s_lshl_b32 s23, 1, s19 // (op 1424) 
s_add_i32 s24, s23, s22 // (op 1424) 


s_and_b32 s25, s0, s24 // (op 1424) 


s_add_i32 s0, s18, s25 // (op 1424) 


s_ashr_i32 s18, s0, s19 // (op 1424) 


s_lshl_b32 s0, s13, 24 // (op 1424) 
s_ashr_i32 s19, s0, 31 // (op 1424) 

s_xor_b32 s0, s18, s19 // (op 1424) 

s_sub_i32 s18, s0, s19 // (op 1424) 


s_add_u32 s50, s6, 64 // (op 1424) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1424) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1424) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1424) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1424) 

s_mul_i32 s0, s18, s24 // (op 1424) 

s_sub_i32 s18, s1, s0 // (op 1424) 


s_mul_hi_i32 s19, s16, s9 // (op 1424) 
s_add_i32 s0, s19, s16 // (op 1424) 

s_ashr_i32 s1, s0, 31 // (op 1424) 
s_cmp_eq_i32 s9, 0 // (op 1424) 
s_mov_b32 s19, scc // (op 1424) 
s_cmp_eq_i32 s19, 0 // (op 1424) 
s_cselect_b32 s22, 0, -1 // (op 1424) 

s_and_b32 s19, s46, 31 // (op 1424) 
s_lshl_b32 s23, 1, s19 // (op 1424) 
s_add_i32 s49, s23, s22 // (op 1424) 


s_and_b32 s22, s1, s49 // (op 1424) 


s_add_i32 s1, s0, s22 // (op 1424) 


s_ashr_i32 s0, s1, s19 // (op 1424) 


s_lshl_b32 s1, s46, 24 // (op 1424) 
s_ashr_i32 s19, s1, 31 // (op 1424) 

s_xor_b32 s1, s0, s19 // (op 1424) 

s_sub_i32 s0, s1, s19 // (op 1424) 


s_mul_i32 s1, s15, s24 // (op 1424) 

s_mul_i32 s19, s8, s1 // (op 1424) 

s_mul_i32 s1, s0, s19 // (op 1424) 

s_sub_u32 s19, s16, s1 // (op 1424) 

s_mul_hi_i32 s1, s19, s14 // (op 1424) 
s_add_i32 s22, s1, s19 // (op 1424) 


s_ashr_i32 s1, s22, 31 // (op 1424) 
s_cmp_eq_i32 s14, 0 // (op 1424) 
s_mov_b32 s19, scc // (op 1424) 
s_cmp_eq_i32 s19, 0 // (op 1424) 
s_cselect_b32 s23, 0, -1 // (op 1424) 

s_and_b32 s19, s48, 31 // (op 1424) 
s_lshl_b32 s24, 1, s19 // (op 1424) 
s_add_i32 s25, s24, s23 // (op 1424) 


s_and_b32 s23, s1, s25 // (op 1424) 


s_add_i32 s1, s22, s23 // (op 1424) 


s_ashr_i32 s22, s1, s19 // (op 1424) 


s_lshl_b32 s1, s48, 24 // (op 1424) 
s_ashr_i32 s19, s1, 31 // (op 1424) 

s_xor_b32 s1, s22, s19 // (op 1424) 

s_sub_i32 s22, s1, s19 // (op 1424) 


s_cmp_eq_i32 s0, 0 // (op 1424) 
s_mov_b32 s1, scc // (op 1424) 

s_cmp_eq_i32 s1, 0 // (op 1424) 
s_cselect_b32 s0, s22, s18 // (op 1424) 



v_lshl_add_u32 v22, s0, 6, v20 // (op 1424) 


v_mul_lo_u32 v20, v22, s42 // (op 1424) 

v_lshl_add_u32 v22, v23, 2, v20 // (op 1424) 


s_mul_i32 s1, s42, s7 // (op 1426) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1426) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1426) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1426) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1426) most significant: sum
s_add_u32 s1, s1, s19 // (op 1426) most significant: sum


s_mov_b32 s72, s44 // (op 1426) reg expression
s_mov_b32 s73, s45 // (op 1426) reg expression
s_mov_b32 s74, s0 // (op 1426) reg expression
s_mov_b32 s75, 131072 // (op 1426) call()


s_mul_i32 s1, s33, 6 // (op 1037) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1037) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1037) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1037) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1037) 


v_mov_b32 v20, v18 // (op 1037) 
buffer_load_dwordx4 v[26:29], v20, s[64:67], 0 offen // (op 1037) Load value
buffer_load_dwordx2 v[30:31], v20, s[64:67], 0 offen offset:16 // (op 1037) Load value
buffer_load_dwordx4 v[32:35], v20, s[64:67], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx2 v[36:37], v20, s[64:67], 0 offen offset:40 // (op 1037) Load value
v_add_u32 v20, v20, v24 // (op 1037) 
buffer_load_dwordx4 v[38:41], v20, s[64:67], 0 offen // (op 1037) Load value
buffer_load_dwordx2 v[42:43], v20, s[64:67], 0 offen offset:16 // (op 1037) Load value
buffer_load_dwordx4 v[44:47], v20, s[64:67], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx2 v[48:49], v20, s[64:67], 0 offen offset:40 // (op 1037) Load value





buffer_load_dword v20, v21, s[72:75], 0 offen // (op 1040) Load value

s_lshl_b64 s[0:1], s[26:27], 2 // (op 1043) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1043) 


v_mov_b32 v23, v16 // (op 1043) reg expression
buffer_load_dwordx2 v[50:51], v23, s[52:55], 0 offen // (op 1043) Load value
v_add_u32 v23, v23, v24 // (op 1043) 
buffer_load_dwordx2 v[52:53], v23, s[52:55], 0 offen // (op 1043) Load value




v_mov_b32 v23, v14 // (op 1046) reg expression

buffer_load_ubyte v24, v23, s[68:71], 0 offen offset: 0 // (op 1046) Load value


s_mul_i32 s1, s33, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1049) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1049) most significant: sum

v_lshrrev_b64 v[54:55], 3, s[0:1] // (op 1049) 


v_mov_b32 v23, v18 // (op 1049) 
buffer_load_dwordx4 v[56:59], v23, s[64:67], 0 offen offset:192 // (op 1049) Load value
buffer_load_dwordx2 v[60:61], v23, s[64:67], 0 offen offset:208 // (op 1049) Load value
buffer_load_dwordx4 v[62:65], v23, s[64:67], 0 offen offset:216 // (op 1049) Load value
buffer_load_dwordx2 v[66:67], v23, s[64:67], 0 offen offset:232 // (op 1049) Load value
v_add_u32 v23, v23, v54 // (op 1049) 
buffer_load_dwordx4 v[68:71], v23, s[64:67], 0 offen offset:192 // (op 1049) Load value
buffer_load_dwordx2 v[72:73], v23, s[64:67], 0 offen offset:208 // (op 1049) Load value
buffer_load_dwordx4 v[74:77], v23, s[64:67], 0 offen offset:216 // (op 1049) Load value
buffer_load_dwordx2 v[78:79], v23, s[64:67], 0 offen offset:232 // (op 1049) Load value






buffer_load_dword v18, v21, s[72:75], 0 offen offset:8 // (op 1052) Load value


s_lshl_b64 s[0:1], s[26:27], 2 // (op 1055) 
v_lshrrev_b64 v[54:55], 3, s[0:1] // (op 1055) 


s_lshl_b64 s[0:1], s[26:27], 8 // (op 1055) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 1055) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 1055) 

v_add_u32 v21, v16, s0 // (op 1055) 

buffer_load_dwordx2 v[80:81], v21, s[52:55], 0 offen // (op 1055) Load value
v_add_u32 v21, v21, v54 // (op 1055) 
buffer_load_dwordx2 v[82:83], v21, s[52:55], 0 offen // (op 1055) Load value





s_lshl_b32 s0, s36, 3 // (op 1058) 
v_add_u32 v16, v14, s0 // (op 1058) 


buffer_load_ubyte v21, v16, s[68:71], 0 offen offset: 0 // (op 1058) Load value


v_and_b32 v14, -4, v1 // (op 1544) 
v_sub_u32 v16, v1, v14 // (op 1544) 

v_lshlrev_b32 v14, 6, v16 // (op 1544) 

v_lshrrev_b32 v16, 2, v1 // (op 1544) 
v_lshl_add_u32 v23, v16, 9, v14 // (op 1544) 


v_mul_lo_u32 v14, v23, 6 // (op 1544) 

v_lshrrev_b32 v16, 3, v14 // (op 1544) 




s_waitcnt vmcnt(23)
ds_write_b128 v16, v[26:29] // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v16, v[30:31] offset:16 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v16, v[32:35] offset:24 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v16, v[36:37] offset:40 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v16, v[38:41] offset:192 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v16, v[42:43] offset:208 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v16, v[44:47] offset:216 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v16, v[48:49] offset:232 // (op 1060) Store local data 

v_and_b32 v14, -2, v1 // (op 1655) 
v_sub_u32 v16, v1, v14 // (op 1655) 

v_lshlrev_b32 v14, 2, v16 // (op 1655) 

v_lshrrev_b32 v16, 1, v1 // (op 1655) 
v_lshl_add_u32 v23, v16, 3, v14 // (op 1655) 





s_waitcnt vmcnt(15)
ds_write_b32 v23, v20 offset:12288 // (op 1061) Store local data 

v_lshlrev_b32 v14, 7, v1 // (op 1491) 
v_lshrrev_b32 v16, 3, v14 // (op 1491) 




s_waitcnt vmcnt(14)
ds_write_b64 v16, v[50:51] offset:12800 // (op 1062) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v16, v[52:53] offset:12808 // (op 1062) Store local data 

v_and_b32 v14, -16, v1 // (op 1602) 
v_sub_u32 v16, v1, v14 // (op 1602) 

v_and_b32 v14, 4294967280, v1 // (op 1602) 
v_add_u32 v23, v14, v16 // (op 1602) 





s_waitcnt vmcnt(12)
ds_write_b8 v23, v24 offset:14848 // (op 1063) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1866) 



ds_read_b128 v[84:87], v9 // (op 1081) Load local data 
ds_read_b64 v[88:89], v9 offset:16 // (op 1081) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1907) 



ds_read_u8 v14, v0 offset:12288 // (op 1094) Load local data 



ds_read_b128 v[90:93], v9 offset:6144 // (op 1086) Load local data 
ds_read_b64 v[94:95], v9 offset:6160 // (op 1086) Load local data 



ds_read_u8 v16, v0 offset:12544 // (op 1099) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1847) 



ds_read_b64_tr_b4 v[96:97], v3 offset:12800 // (op 1077) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v3 offset:12928 // (op 1077) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1887) 


ds_read_u8 v23, v4 offset:14848 // (op 1090) Load local data 
v_lshlrev_b32 v25, 7, v1 // (op 1462) 
v_lshrrev_b32 v54, 3, v25 // (op 1462) 

v_and_b32 v25, -4, v1 // (op 1515) 
v_sub_u32 v55, v1, v25 // (op 1515) 

v_lshlrev_b32 v25, 6, v55 // (op 1515) 

v_lshrrev_b32 v55, 2, v1 // (op 1515) 
v_lshl_add_u32 v100, v55, 9, v25 // (op 1515) 


v_mul_lo_u32 v25, v100, 6 // (op 1515) 

v_lshrrev_b32 v55, 3, v25 // (op 1515) 

v_and_b32 v25, -16, v1 // (op 1573) 
v_sub_u32 v100, v1, v25 // (op 1573) 

v_and_b32 v25, 4294967280, v1 // (op 1573) 
v_add_u32 v101, v25, v100 // (op 1573) 


v_and_b32 v25, -2, v1 // (op 1626) 
v_sub_u32 v100, v1, v25 // (op 1626) 

v_lshlrev_b32 v25, 2, v100 // (op 1626) 

v_lshrrev_b32 v100, 1, v1 // (op 1626) 
v_lshl_add_u32 v102, v100, 3, v25 // (op 1626) 


v_lshlrev_b32 v25, 7, v1 // (op 1702) 
v_lshrrev_b32 v100, 3, v25 // (op 1702) 

v_and_b32 v25, -4, v1 // (op 1726) 
v_sub_u32 v103, v1, v25 // (op 1726) 

v_lshlrev_b32 v25, 6, v103 // (op 1726) 

v_lshrrev_b32 v103, 2, v1 // (op 1726) 
v_lshl_add_u32 v104, v103, 9, v25 // (op 1726) 


v_mul_lo_u32 v25, v104, 6 // (op 1726) 

v_lshrrev_b32 v103, 3, v25 // (op 1726) 

v_and_b32 v25, -16, v1 // (op 1755) 
v_sub_u32 v104, v1, v25 // (op 1755) 

v_and_b32 v25, 4294967280, v1 // (op 1755) 
v_add_u32 v105, v25, v104 // (op 1755) 


v_and_b32 v25, -2, v1 // (op 1779) 
v_sub_u32 v104, v1, v25 // (op 1779) 

v_lshlrev_b32 v25, 2, v104 // (op 1779) 

v_lshrrev_b32 v104, 1, v1 // (op 1779) 
v_lshl_add_u32 v106, v104, 3, v25 // (op 1779) 


s_mov_b32 s0, 0 // (op 404) call()
s_ashr_i64 s[18:19], s[30:31], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 if false)
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403:

s_add_u32 s0, s0, 2 // (op 405) 
s_mul_i32 s19, s33, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s32, 6 // (op 10) least significant
s_mul_hi_u32 s1, s32, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[108:109], 3, s[18:19] // (op 10) 


v_mov_b32 v25, v19 // (op 10) 
v_mov_b32 v104, v25 // (op 10) 
buffer_load_dwordx4 v[26:29], v104, s[64:67], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[30:31], v104, s[64:67], 0 offen offset:400 // (op 10) Load value
buffer_load_dwordx4 v[32:35], v104, s[64:67], 0 offen offset:408 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[84:89], a[0:3], v23, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 90) 
buffer_load_dwordx2 v[36:37], v104, s[64:67], 0 offen offset:424 // (op 10) Load value
s_barrier  // (op 1856) 
v_add_u32 v104, v104, v108 // (op 10) 



buffer_load_dwordx4 v[38:41], v104, s[64:67], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[42:43], v104, s[64:67], 0 offen offset:400 // (op 10) Load value
ds_read_b64_tr_b4 v[110:111], v3 offset:13824 // (op 93) Transpose load local data 
ds_read_b64_tr_b4 v[112:113], v3 offset:13952 // (op 93) Transpose load local data 
buffer_load_dwordx4 v[44:47], v104, s[64:67], 0 offen offset:408 // (op 10) Load value
s_barrier  // (op 1876) 
buffer_load_dwordx2 v[48:49], v104, s[64:67], 0 offen offset:424 // (op 10) Load value







ds_read_b128 v[84:87], v9 offset:96 // (op 97) Load local data 
ds_read_b64 v[88:89], v9 offset:112 // (op 97) Load local data 
v_mov_b32 v104, v22 // (op 25) 

s_barrier  // (op 1897) 
buffer_load_dword v20, v104, s[72:75], 0 offen offset:16 // (op 25) Load value



ds_read_u8 v107, v4 offset:14912 // (op 101) Load local data 
s_lshl_b64 s[18:19], s[26:27], 2 // (op 4) 
s_barrier  // (op 1917) 
v_lshrrev_b64 v[108:109], 3, s[18:19] // (op 4) 





v_mov_b32 v114, v17 // (op 4) 
s_lshl_b64 s[18:19], s[26:27], 8 // (op 4) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v115, v114, s1 // (op 4) 

ds_read_u8 v14, v0 offset:12292 // (op 105) Load local data 
buffer_load_dwordx2 v[50:51], v115, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v115, v115, v108 // (op 4) 
buffer_load_dwordx2 v[52:53], v115, s[52:55], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:113], v[84:89], a[0:3], v107, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 107) 


v_mov_b32 v108, v15 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[90:95], a[4:7], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 451) 
s_lshl_b32 s1, s36, 4 // (op 17) 



ds_read_b128 v[90:93], v9 offset:6240 // (op 455) Load local data 
ds_read_b64 v[94:95], v9 offset:6256 // (op 455) Load local data 
v_add_u32 v109, v108, s1 // (op 17) 





ds_read_u8 v16, v0 offset:12548 // (op 459) Load local data 
buffer_load_ubyte v24, v109, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1937) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:113], v[90:95], a[4:7], v107, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 460) 





s_waitcnt vmcnt(23)
ds_write_b128 v103, v[56:59] offset:14976 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v103, v[60:61] offset:14992 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v103, v[62:65] offset:15000 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v103, v[66:67] offset:15016 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v103, v[68:71] offset:15168 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v103, v[72:73] offset:15184 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v103, v[74:77] offset:15192 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v103, v[78:79] offset:15208 // (op 306) Store local data 
s_barrier  // (op 1949) 



s_waitcnt vmcnt(15)
ds_write_b32 v106, v18 offset:27264 // (op 308) Store local data 
s_barrier  // (op 1930) 



s_waitcnt vmcnt(14)
ds_write_b64 v100, v[80:81] offset:27776 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v100, v[82:83] offset:27784 // (op 287) Store local data 
s_barrier  // (op 1943) 



s_waitcnt vmcnt(12)
ds_write_b8 v105, v21 offset:29824 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1933) 



ds_read_b128 v[84:87], v13 offset:14976 // (op 291) Load local data 
ds_read_b64 v[88:89], v13 offset:14992 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1946) 



ds_read_u8 v14, v12 offset:27264 // (op 295) Load local data 



ds_read_b128 v[116:119], v13 offset:21120 // (op 474) Load local data 
ds_read_b64 v[120:121], v13 offset:21136 // (op 474) Load local data 



ds_read_u8 v107, v12 offset:27520 // (op 478) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1927) 



ds_read_b64_tr_b4 v[96:97], v5 offset:27776 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v5 offset:27904 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1940) 


ds_read_u8 v23, v6 offset:29824 // (op 293) Load local data 

s_mul_i32 s19, s33, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s32, 6 // (op 309) least significant
s_mul_hi_u32 s1, s32, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 309) most significant: sum

v_lshrrev_b64 v[112:113], 3, s[18:19] // (op 309) 


v_mov_b32 v115, v25 // (op 309) 
buffer_load_dwordx4 v[56:59], v115, s[64:67], 0 offen offset:576 // (op 309) Load value
buffer_load_dwordx2 v[60:61], v115, s[64:67], 0 offen offset:592 // (op 309) Load value
buffer_load_dwordx4 v[62:65], v115, s[64:67], 0 offen offset:600 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[84:89], a[0:3], v23, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 296) 
buffer_load_dwordx2 v[66:67], v115, s[64:67], 0 offen offset:616 // (op 309) Load value
v_add_u32 v115, v115, v112 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[116:121], a[4:7], v23, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 461) 
buffer_load_dwordx4 v[68:71], v115, s[64:67], 0 offen offset:576 // (op 309) Load value


buffer_load_dwordx2 v[72:73], v115, s[64:67], 0 offen offset:592 // (op 309) Load value
ds_read_b64_tr_b4 v[116:117], v5 offset:28800 // (op 298) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v5 offset:28928 // (op 298) Transpose load local data 
buffer_load_dwordx4 v[74:77], v115, s[64:67], 0 offen offset:600 // (op 309) Load value
buffer_load_dwordx2 v[78:79], v115, s[64:67], 0 offen offset:616 // (op 309) Load value






s_barrier  // (op 1998) 



buffer_load_dword v18, v104, s[72:75], 0 offen offset:24 // (op 311) Load value
v_add_u32 v19, 384, v19 // (op 1288) 

ds_read_b128 v[120:123], v13 offset:15072 // (op 300) Load local data 

ds_read_b64 v[124:125], v13 offset:15088 // (op 300) Load local data 
v_add_u32 v22, 16, v22 // (op 1416) 


ds_read_u8 v25, v6 offset:29888 // (op 302) Load local data 
s_lshl_b64 s[18:19], s[26:27], 2 // (op 286) 
v_lshrrev_b64 v[112:113], 3, s[18:19] // (op 286) 





s_barrier  // (op 2002) 
s_lshl_b64 s[18:19], s[26:27], 8 // (op 286) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 286) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 286) 

s_mul_i32 s1, 3, s18 // (op 286) 

v_add_u32 v104, v114, s1 // (op 286) 

ds_read_u8 v107, v12 offset:27268 // (op 304) Load local data 
buffer_load_dwordx2 v[80:81], v104, s[52:55], 0 offen // (op 286) Load value



v_add_u32 v104, v104, v112 // (op 286) 
ds_read_b128 v[126:129], v13 offset:21216 // (op 465) Load local data 
ds_read_b64 v[130:131], v13 offset:21232 // (op 465) Load local data 
buffer_load_dwordx2 v[82:83], v104, s[52:55], 0 offen // (op 286) Load value








s_barrier  // (op 2038) 
ds_read_u8 v104, v12 offset:27524 // (op 469) Load local data 
s_lshl_b32 s1, s36, 3 // (op 310) 
s_mul_i32 s18, 3, s1 // (op 310) 

s_barrier  // (op 2042) 
v_add_u32 v109, v108, s18 // (op 310) 


buffer_load_ubyte v21, v109, s[68:71], 0 offen offset: 0 // (op 310) Load value

s_lshl_b64 s[18:19], s[26:27], 8 // (op 1224) 

s_lshl_b64 s[50:51], s[18:19], 2 // (op 1224) 

s_barrier  // (op 1869) 
s_lshl_b32 s1, s36, 4 // (op 1352) 



v_add_u32 v15, v15, s1 // (op 1352) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 1224) 

s_lshl_b32 s1, s18, 1 // (op 1224) 

v_add_u32 v17, v17, s1 // (op 1224) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[120:125], a[0:3], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 305) 
s_waitcnt vmcnt(23)
ds_write_b128 v55, v[26:29] // (op 13) Store local data 


s_waitcnt vmcnt(22)
ds_write_b64 v55, v[30:31] offset:16 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[126:131], a[4:7], v25, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 470) 
s_waitcnt vmcnt(21)
ds_write_b128 v55, v[32:35] offset:24 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b64 v55, v[36:37] offset:40 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v55, v[38:41] offset:192 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v55, v[42:43] offset:208 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v55, v[44:47] offset:216 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v55, v[48:49] offset:232 // (op 13) Store local data 
s_barrier  // (op 1910) 



s_waitcnt vmcnt(15)
ds_write_b32 v102, v20 offset:12288 // (op 30) Store local data 
s_barrier  // (op 1850) 



s_waitcnt vmcnt(14)
ds_write_b64 v54, v[50:51] offset:12800 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v54, v[52:53] offset:12808 // (op 6) Store local data 
s_barrier  // (op 1890) 



s_waitcnt vmcnt(12)
ds_write_b8 v101, v24 offset:14848 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1879) 



ds_read_b128 v[84:87], v9 // (op 77) Load local data 
ds_read_b64 v[88:89], v9 offset:16 // (op 77) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1920) 



ds_read_u8 v14, v0 offset:12288 // (op 87) Load local data 



ds_read_b128 v[90:93], v9 offset:6144 // (op 446) Load local data 
ds_read_b64 v[94:95], v9 offset:6160 // (op 446) Load local data 



ds_read_u8 v16, v0 offset:12544 // (op 450) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1859) 



ds_read_b64_tr_b4 v[96:97], v3 offset:12800 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v3 offset:12928 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1900) 


ds_read_u8 v23, v4 offset:14848 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[30:31], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 if true)
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403:

















s_mul_hi_i32 s0, s16, s10 // (op 1186) 
v_lshlrev_b32 v15, 1, v1 // (op 1186) 
s_add_i32 s1, s0, s16 // (op 1186) 

s_ashr_i32 s0, s1, 31 // (op 1186) 
s_cmp_eq_i32 s10, 0 // (op 1186) 
s_mov_b32 s18, scc // (op 1186) 
s_cmp_eq_i32 s18, 0 // (op 1186) 
s_cselect_b32 s19, 0, -1 // (op 1186) 

s_and_b32 s18, s11, 31 // (op 1186) 
s_lshl_b32 s22, 1, s18 // (op 1186) 
s_add_i32 s23, s22, s19 // (op 1186) 


s_and_b32 s19, s0, s23 // (op 1186) 


s_add_i32 s0, s1, s19 // (op 1186) 


s_ashr_i32 s1, s0, s18 // (op 1186) 


s_lshl_b32 s0, s11, 24 // (op 1186) 
s_ashr_i32 s18, s0, 31 // (op 1186) 

s_xor_b32 s0, s1, s18 // (op 1186) 

s_sub_i32 s1, s0, s18 // (op 1186) 


s_mul_hi_i32 s0, s1, s12 // (op 1186) 
s_add_i32 s18, s0, s1 // (op 1186) 

s_ashr_i32 s0, s18, 31 // (op 1186) 
s_cmp_eq_i32 s12, 0 // (op 1186) 
s_mov_b32 s19, scc // (op 1186) 
s_cmp_eq_i32 s19, 0 // (op 1186) 
s_cselect_b32 s22, 0, -1 // (op 1186) 

s_and_b32 s19, s13, 31 // (op 1186) 
s_lshl_b32 s23, 1, s19 // (op 1186) 
s_add_i32 s24, s23, s22 // (op 1186) 


s_and_b32 s25, s0, s24 // (op 1186) 


s_add_i32 s0, s18, s25 // (op 1186) 


s_ashr_i32 s18, s0, s19 // (op 1186) 


s_lshl_b32 s0, s13, 24 // (op 1186) 
s_ashr_i32 s19, s0, 31 // (op 1186) 

s_xor_b32 s0, s18, s19 // (op 1186) 

s_sub_i32 s18, s0, s19 // (op 1186) 


s_mul_i32 s0, s1, s15 // (op 1186) 

s_sub_u32 s1, s16, s0 // (op 1186) 

s_mul_i32 s0, s18, s15 // (op 1186) 
s_add_u32 s0, s0, s1 // (op 1186) 


s_mul_hi_i32 s1, s16, s9 // (op 1186) 
s_add_i32 s18, s1, s16 // (op 1186) 

s_ashr_i32 s1, s18, 31 // (op 1186) 
s_cmp_eq_i32 s9, 0 // (op 1186) 
s_mov_b32 s19, scc // (op 1186) 
s_cmp_eq_i32 s19, 0 // (op 1186) 
s_cselect_b32 s22, 0, -1 // (op 1186) 

s_and_b32 s19, s46, 31 // (op 1186) 
s_lshl_b32 s23, 1, s19 // (op 1186) 
s_add_i32 s24, s23, s22 // (op 1186) 


s_and_b32 s25, s1, s24 // (op 1186) 


s_add_i32 s1, s18, s25 // (op 1186) 


s_ashr_i32 s18, s1, s19 // (op 1186) 


s_lshl_b32 s1, s46, 24 // (op 1186) 
s_ashr_i32 s19, s1, 31 // (op 1186) 

s_xor_b32 s1, s18, s19 // (op 1186) 

s_sub_i32 s18, s1, s19 // (op 1186) 


s_add_u32 s50, s6, 64 // (op 1186) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1186) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1186) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1186) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1186) 

s_mul_i32 s1, s15, s24 // (op 1186) 

s_mul_i32 s19, s8, s1 // (op 1186) 

s_mul_i32 s1, s18, s19 // (op 1186) 

s_sub_u32 s19, s16, s1 // (op 1186) 

s_mov_b32 s1, s19 // (op 1186) 
s_mul_hi_i32 s22, s1, s14 // (op 1186) 
s_add_i32 s23, s22, s1 // (op 1186) 


s_ashr_i32 s1, s23, 31 // (op 1186) 
s_cmp_eq_i32 s14, 0 // (op 1186) 
s_mov_b32 s22, scc // (op 1186) 
s_cmp_eq_i32 s22, 0 // (op 1186) 
s_cselect_b32 s24, 0, -1 // (op 1186) 

s_and_b32 s22, s48, 31 // (op 1186) 
s_lshl_b32 s25, 1, s22 // (op 1186) 
s_add_i32 s49, s25, s24 // (op 1186) 


s_and_b32 s24, s1, s49 // (op 1186) 


s_add_i32 s1, s23, s24 // (op 1186) 


s_ashr_i32 s23, s1, s22 // (op 1186) 


s_lshl_b32 s1, s48, 24 // (op 1186) 
s_ashr_i32 s22, s1, 31 // (op 1186) 

s_xor_b32 s1, s23, s22 // (op 1186) 

s_sub_i32 s23, s1, s22 // (op 1186) 


s_mul_i32 s1, s47, s15 // (op 1186) 
s_add_u32 s24, s20, 16 // (op 1186) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1186) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1186) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1186) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1186) 

s_sub_i32 s22, s24, s1 // (op 1186) 

s_mul_i32 s24, s23, s22 // (op 1186) 


s_sub_u32 s25, s19, s24 // (op 1186) 


s_add_u32 s19, s25, s1 // (op 1186) 


s_cmp_eq_i32 s18, 0 // (op 1186) 
s_mov_b32 s1, scc // (op 1186) 

s_cmp_eq_i32 s1, 0 // (op 1186) 
s_cselect_b32 s18, s19, s0 // (op 1186) 



s_lshl_b32 s19, s18, 4 // (op 1186) 

s_mov_b32 s1, 0 // (op 1186) convert
s_mov_b32 s0, s19 // (op 1186) convert

s_ashr_i64 s[18:19], s[30:31], 8 // (op 1186) 
s_lshr_b32 s22, s18, 1 // (op 1186) 

v_lshl_add_u32 v17, s22, 9, v15 // (op 1186) 


v_mul_lo_u32 v25, v17, s27 // (op 1186) most significant: low of low * high
v_mul_hi_u32 v15, v17, s26 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v24, v17, s26 // (op 1186) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 1186) most significant: sum


v_mov_b32 v15, s0 // (op 1186) 
v_mov_b32 v17, s1 // (op 1186) 
v_add_co_u32 v26, s[18:19], v15, v24 // (op 1186) least significant half
v_addc_co_u32 v27, s[18:19], v17, v25, s[18:19] // (op 1186) most significant half



v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 1186) 


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 1186) 

v_mov_b32 v15, v24 // (op 1186) convert

s_mul_i32 s1, s26, s31 // (op 1188) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1188) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1188) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1188) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1188) most significant: sum
s_add_u32 s1, s1, s19 // (op 1188) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1188) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1188) 

s_mov_b32 s52, s28 // (op 1188) reg expression
s_mov_b32 s53, s29 // (op 1188) reg expression
s_mov_b32 s54, s0 // (op 1188) reg expression
s_mov_b32 s55, 131072 // (op 1188) call()

s_mul_hi_i32 s0, s16, s10 // (op 1254) 
v_lshrrev_b32 v17, 2, v1 // (op 1254) 
s_add_i32 s1, s0, s16 // (op 1254) 

v_lshlrev_b32 v22, 1, v17 // (op 1254) 

s_ashr_i32 s0, s1, 31 // (op 1254) 
v_and_b32 v17, -4, v1 // (op 1254) 
s_cmp_eq_i32 s10, 0 // (op 1254) 
s_mov_b32 s18, scc // (op 1254) 
v_sub_u32 v19, v1, v17 // (op 1254) 

s_cmp_eq_i32 s18, 0 // (op 1254) 
s_cselect_b32 s19, 0, -1 // (op 1254) 

v_lshlrev_b32 v17, 6, v19 // (op 1254) 

s_and_b32 s18, s11, 31 // (op 1254) 
s_lshl_b32 s22, 1, s18 // (op 1254) 
s_add_i32 s23, s22, s19 // (op 1254) 


s_and_b32 s19, s0, s23 // (op 1254) 


s_add_i32 s0, s1, s19 // (op 1254) 


s_ashr_i32 s1, s0, s18 // (op 1254) 


s_lshl_b32 s0, s11, 24 // (op 1254) 
s_ashr_i32 s18, s0, 31 // (op 1254) 

s_xor_b32 s0, s1, s18 // (op 1254) 

s_sub_i32 s1, s0, s18 // (op 1254) 


s_mul_hi_i32 s0, s1, s12 // (op 1254) 
s_add_i32 s18, s0, s1 // (op 1254) 

s_ashr_i32 s0, s18, 31 // (op 1254) 
s_cmp_eq_i32 s12, 0 // (op 1254) 
s_mov_b32 s19, scc // (op 1254) 
s_cmp_eq_i32 s19, 0 // (op 1254) 
s_cselect_b32 s22, 0, -1 // (op 1254) 

s_and_b32 s19, s13, 31 // (op 1254) 
s_lshl_b32 s23, 1, s19 // (op 1254) 
s_add_i32 s24, s23, s22 // (op 1254) 


s_and_b32 s25, s0, s24 // (op 1254) 


s_add_i32 s0, s18, s25 // (op 1254) 


s_ashr_i32 s18, s0, s19 // (op 1254) 


s_lshl_b32 s0, s13, 24 // (op 1254) 
s_ashr_i32 s19, s0, 31 // (op 1254) 

s_xor_b32 s0, s18, s19 // (op 1254) 

s_sub_i32 s18, s0, s19 // (op 1254) 


s_add_u32 s50, s6, 64 // (op 1254) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1254) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1254) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1254) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1254) 

s_mul_i32 s0, s18, s24 // (op 1254) 

s_sub_i32 s18, s1, s0 // (op 1254) 


s_mul_hi_i32 s19, s16, s9 // (op 1254) 
s_add_i32 s0, s19, s16 // (op 1254) 

s_ashr_i32 s1, s0, 31 // (op 1254) 
s_cmp_eq_i32 s9, 0 // (op 1254) 
s_mov_b32 s19, scc // (op 1254) 
s_cmp_eq_i32 s19, 0 // (op 1254) 
s_cselect_b32 s22, 0, -1 // (op 1254) 

s_and_b32 s19, s46, 31 // (op 1254) 
s_lshl_b32 s23, 1, s19 // (op 1254) 
s_add_i32 s49, s23, s22 // (op 1254) 


s_and_b32 s22, s1, s49 // (op 1254) 


s_add_i32 s1, s0, s22 // (op 1254) 


s_ashr_i32 s0, s1, s19 // (op 1254) 


s_lshl_b32 s1, s46, 24 // (op 1254) 
s_ashr_i32 s19, s1, 31 // (op 1254) 

s_xor_b32 s1, s0, s19 // (op 1254) 

s_sub_i32 s0, s1, s19 // (op 1254) 


s_mul_i32 s1, s15, s24 // (op 1254) 

s_mul_i32 s19, s8, s1 // (op 1254) 

s_mul_i32 s1, s0, s19 // (op 1254) 

s_sub_u32 s19, s16, s1 // (op 1254) 

s_mul_hi_i32 s1, s19, s14 // (op 1254) 
s_add_i32 s22, s1, s19 // (op 1254) 


s_ashr_i32 s1, s22, 31 // (op 1254) 
s_cmp_eq_i32 s14, 0 // (op 1254) 
s_mov_b32 s19, scc // (op 1254) 
s_cmp_eq_i32 s19, 0 // (op 1254) 
s_cselect_b32 s23, 0, -1 // (op 1254) 

s_and_b32 s19, s48, 31 // (op 1254) 
s_lshl_b32 s24, 1, s19 // (op 1254) 
s_add_i32 s25, s24, s23 // (op 1254) 


s_and_b32 s23, s1, s25 // (op 1254) 


s_add_i32 s1, s22, s23 // (op 1254) 


s_ashr_i32 s22, s1, s19 // (op 1254) 


s_lshl_b32 s1, s48, 24 // (op 1254) 
s_ashr_i32 s19, s1, 31 // (op 1254) 

s_xor_b32 s1, s22, s19 // (op 1254) 

s_sub_i32 s22, s1, s19 // (op 1254) 


s_cmp_eq_i32 s0, 0 // (op 1254) 
s_mov_b32 s1, scc // (op 1254) 

s_cmp_eq_i32 s1, 0 // (op 1254) 
s_cselect_b32 s0, s22, s18 // (op 1254) 



v_lshl_add_u32 v19, s0, 6, v22 // (op 1254) 


v_mul_lo_u32 v25, v19, s33 // (op 1254) most significant: low of low * high
v_mul_hi_u32 v20, v19, s32 // (op 1254) most significant: high of low * low
v_mul_lo_u32 v24, v19, s32 // (op 1254) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 1254) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1254) 
s_lshr_b32 s18, s0, 1 // (op 1254) 

v_lshl_add_u32 v22, s18, 9, v17 // (op 1254) 


v_mov_b32 v27, 0 // (op 1254) convert
v_mov_b32 v26, v22 // (op 1254) convert

v_add_co_u32 v28, s[0:1], v26, v24 // (op 1254) least significant half
v_addc_co_u32 v29, s[0:1], v27, v25, s[0:1] // (op 1254) most significant half



v_mul_lo_u32 v25, v29, 6 // (op 1254) most significant: low of high * low
v_mul_hi_u32 v17, v28, 6 // (op 1254) most significant: high of low * low
v_mul_lo_u32 v24, v28, 6 // (op 1254) least significant: low of low * low
v_add_u32 v25, v25, v17 // (op 1254) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 1254) 

v_mov_b32 v17, v26 // (op 1254) convert

s_mul_i32 s1, s7, 6 // (op 1256) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1256) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1256) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1256) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1256) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1256) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1256) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1256) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1256) most significant: sum
s_add_u32 s19, s19, s23 // (op 1256) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1256) 

s_mov_b32 s64, s34 // (op 1256) reg expression
s_mov_b32 s65, s35 // (op 1256) reg expression
s_mov_b32 s66, s0 // (op 1256) reg expression
s_mov_b32 s67, 131072 // (op 1256) call()

s_mul_hi_i32 s0, s16, s10 // (op 1318) 
v_and_b32 v22, -16, v1 // (op 1318) 
s_add_i32 s1, s0, s16 // (op 1318) 

v_sub_u32 v19, v1, v22 // (op 1318) 

s_ashr_i32 s0, s1, 31 // (op 1318) 
v_lshrrev_b32 v20, 4, v1 // (op 1318) 
s_cmp_eq_i32 s10, 0 // (op 1318) 
s_mov_b32 s18, scc // (op 1318) 
s_cmp_eq_i32 s18, 0 // (op 1318) 
s_cselect_b32 s19, 0, -1 // (op 1318) 

s_and_b32 s18, s11, 31 // (op 1318) 
s_lshl_b32 s22, 1, s18 // (op 1318) 
s_add_i32 s23, s22, s19 // (op 1318) 


s_and_b32 s19, s0, s23 // (op 1318) 


s_add_i32 s0, s1, s19 // (op 1318) 


s_ashr_i32 s1, s0, s18 // (op 1318) 


s_lshl_b32 s0, s11, 24 // (op 1318) 
s_ashr_i32 s18, s0, 31 // (op 1318) 

s_xor_b32 s0, s1, s18 // (op 1318) 

s_sub_i32 s1, s0, s18 // (op 1318) 


s_mul_hi_i32 s0, s1, s12 // (op 1318) 
s_add_i32 s18, s0, s1 // (op 1318) 

s_ashr_i32 s0, s18, 31 // (op 1318) 
s_cmp_eq_i32 s12, 0 // (op 1318) 
s_mov_b32 s19, scc // (op 1318) 
s_cmp_eq_i32 s19, 0 // (op 1318) 
s_cselect_b32 s22, 0, -1 // (op 1318) 

s_and_b32 s19, s13, 31 // (op 1318) 
s_lshl_b32 s23, 1, s19 // (op 1318) 
s_add_i32 s24, s23, s22 // (op 1318) 


s_and_b32 s25, s0, s24 // (op 1318) 


s_add_i32 s0, s18, s25 // (op 1318) 


s_ashr_i32 s18, s0, s19 // (op 1318) 


s_lshl_b32 s0, s13, 24 // (op 1318) 
s_ashr_i32 s19, s0, 31 // (op 1318) 

s_xor_b32 s0, s18, s19 // (op 1318) 

s_sub_i32 s18, s0, s19 // (op 1318) 


s_mul_i32 s0, s1, s15 // (op 1318) 

s_sub_u32 s1, s16, s0 // (op 1318) 

s_mul_i32 s0, s18, s15 // (op 1318) 
s_add_u32 s0, s0, s1 // (op 1318) 


s_mul_hi_i32 s1, s16, s9 // (op 1318) 
s_add_i32 s18, s1, s16 // (op 1318) 

s_ashr_i32 s1, s18, 31 // (op 1318) 
s_cmp_eq_i32 s9, 0 // (op 1318) 
s_mov_b32 s19, scc // (op 1318) 
s_cmp_eq_i32 s19, 0 // (op 1318) 
s_cselect_b32 s22, 0, -1 // (op 1318) 

s_and_b32 s19, s46, 31 // (op 1318) 
s_lshl_b32 s23, 1, s19 // (op 1318) 
s_add_i32 s24, s23, s22 // (op 1318) 


s_and_b32 s25, s1, s24 // (op 1318) 


s_add_i32 s1, s18, s25 // (op 1318) 


s_ashr_i32 s18, s1, s19 // (op 1318) 


s_lshl_b32 s1, s46, 24 // (op 1318) 
s_ashr_i32 s19, s1, 31 // (op 1318) 

s_xor_b32 s1, s18, s19 // (op 1318) 

s_sub_i32 s18, s1, s19 // (op 1318) 


s_add_u32 s50, s6, 64 // (op 1318) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1318) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1318) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1318) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1318) 

s_mul_i32 s1, s15, s24 // (op 1318) 

s_mul_i32 s19, s8, s1 // (op 1318) 

s_mul_i32 s1, s18, s19 // (op 1318) 

s_sub_u32 s19, s16, s1 // (op 1318) 

s_mov_b32 s1, s19 // (op 1318) 
s_mul_hi_i32 s22, s1, s14 // (op 1318) 
s_add_i32 s23, s22, s1 // (op 1318) 


s_ashr_i32 s1, s23, 31 // (op 1318) 
s_cmp_eq_i32 s14, 0 // (op 1318) 
s_mov_b32 s22, scc // (op 1318) 
s_cmp_eq_i32 s22, 0 // (op 1318) 
s_cselect_b32 s24, 0, -1 // (op 1318) 

s_and_b32 s22, s48, 31 // (op 1318) 
s_lshl_b32 s25, 1, s22 // (op 1318) 
s_add_i32 s49, s25, s24 // (op 1318) 


s_and_b32 s24, s1, s49 // (op 1318) 


s_add_i32 s1, s23, s24 // (op 1318) 


s_ashr_i32 s23, s1, s22 // (op 1318) 


s_lshl_b32 s1, s48, 24 // (op 1318) 
s_ashr_i32 s22, s1, 31 // (op 1318) 

s_xor_b32 s1, s23, s22 // (op 1318) 

s_sub_i32 s23, s1, s22 // (op 1318) 


s_mul_i32 s1, s47, s15 // (op 1318) 
s_add_u32 s24, s20, 16 // (op 1318) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1318) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1318) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1318) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1318) 

s_sub_i32 s22, s24, s1 // (op 1318) 

s_mul_i32 s24, s23, s22 // (op 1318) 


s_sub_u32 s25, s19, s24 // (op 1318) 


s_add_u32 s19, s25, s1 // (op 1318) 


s_cmp_eq_i32 s18, 0 // (op 1318) 
s_mov_b32 s1, scc // (op 1318) 

s_cmp_eq_i32 s1, 0 // (op 1318) 
s_cselect_b32 s18, s19, s0 // (op 1318) 



v_lshl_add_u32 v22, s18, 4, v19 // (op 1318) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1318) 
s_lshr_b32 s18, s0, 1 // (op 1318) 

v_lshl_add_u32 v19, s18, 4, v20 // (op 1318) 


v_mul_lo_u32 v20, v19, s36 // (op 1318) 

v_add_u32 v19, v22, v20 // (op 1318) 


s_mul_i32 s1, s36, s41 // (op 1320) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1320) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1320) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1320) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1320) most significant: sum
s_add_u32 s1, s1, s19 // (op 1320) most significant: sum


s_mov_b32 s68, s38 // (op 1320) reg expression
s_mov_b32 s69, s39 // (op 1320) reg expression
s_mov_b32 s70, s0 // (op 1320) reg expression
s_mov_b32 s71, 131072 // (op 1320) call()

s_mul_hi_i32 s0, s16, s10 // (op 1382) 
v_lshrrev_b32 v20, 1, v1 // (op 1382) 
s_add_i32 s1, s0, s16 // (op 1382) 

v_and_b32 v22, -2, v1 // (op 1382) 
s_ashr_i32 s0, s1, 31 // (op 1382) 
v_sub_u32 v24, v1, v22 // (op 1382) 

s_cmp_eq_i32 s10, 0 // (op 1382) 
s_mov_b32 s18, scc // (op 1382) 
v_lshlrev_b32 v22, 2, v24 // (op 1382) 

s_cmp_eq_i32 s18, 0 // (op 1382) 
s_cselect_b32 s19, 0, -1 // (op 1382) 

s_and_b32 s18, s11, 31 // (op 1382) 
s_lshl_b32 s22, 1, s18 // (op 1382) 
s_add_i32 s23, s22, s19 // (op 1382) 


s_and_b32 s19, s0, s23 // (op 1382) 


s_add_i32 s0, s1, s19 // (op 1382) 


s_ashr_i32 s1, s0, s18 // (op 1382) 


s_lshl_b32 s0, s11, 24 // (op 1382) 
s_ashr_i32 s18, s0, 31 // (op 1382) 

s_xor_b32 s0, s1, s18 // (op 1382) 

s_sub_i32 s1, s0, s18 // (op 1382) 


s_mul_hi_i32 s0, s1, s12 // (op 1382) 
s_add_i32 s18, s0, s1 // (op 1382) 

s_ashr_i32 s0, s18, 31 // (op 1382) 
s_cmp_eq_i32 s12, 0 // (op 1382) 
s_mov_b32 s19, scc // (op 1382) 
s_cmp_eq_i32 s19, 0 // (op 1382) 
s_cselect_b32 s22, 0, -1 // (op 1382) 

s_and_b32 s19, s13, 31 // (op 1382) 
s_lshl_b32 s23, 1, s19 // (op 1382) 
s_add_i32 s24, s23, s22 // (op 1382) 


s_and_b32 s25, s0, s24 // (op 1382) 


s_add_i32 s0, s18, s25 // (op 1382) 


s_ashr_i32 s18, s0, s19 // (op 1382) 


s_lshl_b32 s0, s13, 24 // (op 1382) 
s_ashr_i32 s19, s0, 31 // (op 1382) 

s_xor_b32 s0, s18, s19 // (op 1382) 

s_sub_i32 s18, s0, s19 // (op 1382) 


s_add_u32 s50, s6, 64 // (op 1382) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1382) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1382) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1382) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1382) 

s_mul_i32 s0, s18, s24 // (op 1382) 

s_sub_i32 s18, s1, s0 // (op 1382) 


s_mul_hi_i32 s19, s16, s9 // (op 1382) 
s_add_i32 s0, s19, s16 // (op 1382) 

s_ashr_i32 s1, s0, 31 // (op 1382) 
s_cmp_eq_i32 s9, 0 // (op 1382) 
s_mov_b32 s19, scc // (op 1382) 
s_cmp_eq_i32 s19, 0 // (op 1382) 
s_cselect_b32 s22, 0, -1 // (op 1382) 

s_and_b32 s19, s46, 31 // (op 1382) 
s_lshl_b32 s23, 1, s19 // (op 1382) 
s_add_i32 s49, s23, s22 // (op 1382) 


s_and_b32 s22, s1, s49 // (op 1382) 


s_add_i32 s1, s0, s22 // (op 1382) 


s_ashr_i32 s0, s1, s19 // (op 1382) 


s_lshl_b32 s1, s46, 24 // (op 1382) 
s_ashr_i32 s19, s1, 31 // (op 1382) 

s_xor_b32 s1, s0, s19 // (op 1382) 

s_sub_i32 s0, s1, s19 // (op 1382) 


s_mul_i32 s1, s15, s24 // (op 1382) 

s_mul_i32 s19, s8, s1 // (op 1382) 

s_mul_i32 s1, s0, s19 // (op 1382) 

s_sub_u32 s19, s16, s1 // (op 1382) 

s_mul_hi_i32 s1, s19, s14 // (op 1382) 
s_add_i32 s22, s1, s19 // (op 1382) 


s_ashr_i32 s1, s22, 31 // (op 1382) 
s_cmp_eq_i32 s14, 0 // (op 1382) 
s_mov_b32 s19, scc // (op 1382) 
s_cmp_eq_i32 s19, 0 // (op 1382) 
s_cselect_b32 s23, 0, -1 // (op 1382) 

s_and_b32 s19, s48, 31 // (op 1382) 
s_lshl_b32 s24, 1, s19 // (op 1382) 
s_add_i32 s25, s24, s23 // (op 1382) 


s_and_b32 s23, s1, s25 // (op 1382) 


s_add_i32 s1, s22, s23 // (op 1382) 


s_ashr_i32 s22, s1, s19 // (op 1382) 


s_lshl_b32 s1, s48, 24 // (op 1382) 
s_ashr_i32 s19, s1, 31 // (op 1382) 

s_xor_b32 s1, s22, s19 // (op 1382) 

s_sub_i32 s22, s1, s19 // (op 1382) 


s_cmp_eq_i32 s0, 0 // (op 1382) 
s_mov_b32 s1, scc // (op 1382) 

s_cmp_eq_i32 s1, 0 // (op 1382) 
s_cselect_b32 s0, s22, s18 // (op 1382) 



v_lshl_add_u32 v24, s0, 6, v20 // (op 1382) 


v_mul_lo_u32 v20, v24, s42 // (op 1382) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 1382) 
s_lshr_b32 s18, s0, 1 // (op 1382) 

v_lshl_add_u32 v24, s18, 4, v22 // (op 1382) 


v_add_u32 v22, v24, v20 // (op 1382) 


s_mul_i32 s1, s42, s7 // (op 1384) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1384) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1384) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1384) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1384) most significant: sum
s_add_u32 s1, s1, s19 // (op 1384) most significant: sum


s_mov_b32 s72, s44 // (op 1384) reg expression
s_mov_b32 s73, s45 // (op 1384) reg expression
s_mov_b32 s74, s0 // (op 1384) reg expression
s_mov_b32 s75, 131072 // (op 1384) call()

v_lshlrev_b32 v20, 7, v1 // (op 1475) 
v_lshrrev_b32 v24, 3, v20 // (op 1475) 

v_and_b32 v20, -4, v1 // (op 1528) 
v_sub_u32 v25, v1, v20 // (op 1528) 

v_lshlrev_b32 v20, 6, v25 // (op 1528) 

v_lshrrev_b32 v25, 2, v1 // (op 1528) 
v_lshl_add_u32 v26, v25, 9, v20 // (op 1528) 


v_mul_lo_u32 v20, v26, 6 // (op 1528) 

v_lshrrev_b32 v25, 3, v20 // (op 1528) 

v_and_b32 v20, -16, v1 // (op 1586) 
v_sub_u32 v26, v1, v20 // (op 1586) 

v_and_b32 v20, 4294967280, v1 // (op 1586) 
v_add_u32 v27, v20, v26 // (op 1586) 


v_and_b32 v20, -2, v1 // (op 1639) 
v_sub_u32 v26, v1, v20 // (op 1639) 

v_lshlrev_b32 v20, 2, v26 // (op 1639) 

v_lshrrev_b32 v26, 1, v1 // (op 1639) 
v_lshl_add_u32 v28, v26, 3, v20 // (op 1639) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 409) 
s_and_b32 s18, s0, 4294967294 // (op 409) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408:

s_add_u32 s18, s18, 1 // (op 410) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 202) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v16, v14 // (op 202) reg expression
buffer_load_dwordx2 v[30:31], v16, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v16, v16, v20 // (op 202) 
buffer_load_dwordx2 v[32:33], v16, s[52:55], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[26:27], 8 // (op 1176) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 1176) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 1176) 

v_add_u32 v15, v15, s0 // (op 1176) 


s_barrier  // (op 1853) 
s_mul_i32 s1, s33, 6 // (op 225) most significant: low of high * low



s_mul_i32 s0, s32, 6 // (op 225) least significant
s_mul_hi_u32 s19, s32, 6 // (op 225) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 225) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 225) 


v_mov_b32 v14, v17 // (op 225) 
v_mov_b32 v16, v14 // (op 225) 
buffer_load_dwordx4 v[34:37], v16, s[64:67], 0 offen // (op 225) Load value
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[30:31] offset:12800 // (op 203) Store local data 
buffer_load_dwordx2 v[38:39], v16, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[40:43], v16, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[44:45], v16, s[64:67], 0 offen offset:40 // (op 225) Load value
v_add_u32 v16, v16, v20 // (op 225) 
s_waitcnt vmcnt(4)
ds_write_b64 v24, v[32:33] offset:12808 // (op 203) Store local data 
buffer_load_dwordx4 v[46:49], v16, s[64:67], 0 offen // (op 225) Load value

buffer_load_dwordx2 v[50:51], v16, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[52:55], v16, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[56:57], v16, s[64:67], 0 offen offset:40 // (op 225) Load value



v_add_u32 v17, 192, v17 // (op 1246) 


s_barrier  // (op 1872) 
v_mov_b32 v14, v19 // (op 226) 



v_mov_b32 v16, v14 // (op 226) reg expression

buffer_load_ubyte v18, v16, s[68:71], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s19, s36, 3 // (op 1310) 
v_add_u32 v19, v19, s19 // (op 1310) 

s_waitcnt vmcnt(8)
ds_write_b128 v25, v[34:37] // (op 222) Store local data 


v_mov_b32 v14, v22 // (op 227) 

buffer_load_dword v16, v14, s[72:75], 0 offen // (op 227) Load value

v_add_u32 v22, 8, v22 // (op 1374) 
s_waitcnt vmcnt(8)
ds_write_b64 v25, v[38:39] offset:16 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1862) 
s_waitcnt vmcnt(7)
ds_write_b128 v25, v[40:43] offset:24 // (op 222) Store local data 



ds_read_b64_tr_b4 v[30:31], v3 offset:12800 // (op 205) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v3 offset:12928 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v25, v[44:45] offset:40 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v25, v[46:49] offset:192 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v25, v[50:51] offset:208 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v25, v[52:55] offset:216 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v25, v[56:57] offset:232 // (op 222) Store local data 

s_barrier  // (op 1893) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1883) 






ds_read_b128 v[34:37], v9 // (op 207) Load local data 
ds_read_b64 v[38:39], v9 offset:16 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v18 offset:14848 // (op 223) Store local data 

s_barrier  // (op 1913) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1903) 





ds_read_u8 v14, v4 offset:14848 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v28, v16 offset:12288 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1923) 



ds_read_u8 v16, v0 offset:12288 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:39], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 212) 





ds_read_b64_tr_b4 v[34:35], v3 offset:13824 // (op 214) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:13952 // (op 214) Transpose load local data 



ds_read_b128 v[38:41], v9 offset:96 // (op 216) Load local data 
ds_read_b64 v[42:43], v9 offset:112 // (op 216) Load local data 


ds_read_u8 v16, v4 offset:14912 // (op 218) Load local data 



ds_read_u8 v18, v0 offset:12292 // (op 220) Load local data 



ds_read_b128 v[44:47], v9 offset:6144 // (op 416) Load local data 
ds_read_b64 v[48:49], v9 offset:6160 // (op 416) Load local data 



ds_read_u8 v23, v0 offset:12544 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[44:49], a[4:7], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:43], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 221) 
ds_read_b128 v[44:47], v9 offset:6240 // (op 425) Load local data 


ds_read_b64 v[48:49], v9 offset:6256 // (op 425) Load local data 



ds_read_u8 v14, v0 offset:12548 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[44:49], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 430) 




s_ashr_i64 s[0:1], s[30:31], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 if true)
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2024) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

v_cvt_f16_f32 v12, v16 // (op 56) 
v_cvt_f16_f32 v13, v17 // (op 56) 
v_cvt_f16_f32 v14, v18 // (op 56) 
v_cvt_f16_f32 v15, v19 // (op 56) 


v_pack_b32_f16 v4, v12, v13 // (op 58) 
v_pack_b32_f16 v5, v14, v15 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v12, a4 // (op 714) 
v_accvgpr_read v13, a5 // (op 714) 
v_accvgpr_read v14, a6 // (op 714) 
v_accvgpr_read v15, a7 // (op 714) 
v_mul_f32 v16, v10, v12 // (op 714) 
v_mul_f32 v17, v10, v13 // (op 714) 
v_mul_f32 v18, v10, v14 // (op 714) 
v_mul_f32 v19, v10, v15 // (op 714) 

v_cvt_f16_f32 v12, v16 // (op 716) 
v_cvt_f16_f32 v13, v17 // (op 716) 
v_cvt_f16_f32 v14, v18 // (op 716) 
v_cvt_f16_f32 v15, v19 // (op 716) 


v_pack_b32_f16 v4, v12, v13 // (op 717) 
v_pack_b32_f16 v5, v14, v15 // (op 717) 

s_lshl_b32 s0, s4, 6 // (op 717) 
v_add_u32 v0, v11, s0 // (op 717) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 717) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 170) Load value

v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v10, v12 // (op 166) 
v_mul_f32 v17, v10, v13 // (op 166) 
v_mul_f32 v18, v10, v14 // (op 166) 
v_mul_f32 v19, v10, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 171) 
v_lshrrev_b32 v13, 16, v4 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v20, v7, v12 // (op 171) 
v_mul_f32 v21, v7, v13 // (op 171) 
v_cvt_f32_f16 v12, v5 // (op 171) 
v_lshrrev_b32 v13, 16, v5 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v22, v7, v12 // (op 171) 
v_mul_f32 v23, v7, v13 // (op 171) 


v_add_f32 v12, v20, v16 // (op 167) 
v_add_f32 v13, v21, v17 // (op 167) 
v_add_f32 v14, v22, v18 // (op 167) 
v_add_f32 v15, v23, v19 // (op 167) 


v_cvt_f16_f32 v16, v12 // (op 168) 
v_cvt_f16_f32 v17, v13 // (op 168) 
v_cvt_f16_f32 v18, v14 // (op 168) 
v_cvt_f16_f32 v19, v15 // (op 168) 


v_pack_b32_f16 v12, v16, v17 // (op 169) 
v_pack_b32_f16 v13, v18, v19 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 169) Store value





s_lshl_b32 s0, s2, 6 // (op 741) 
v_add_u32 v0, v8, s0 // (op 741) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 741) Load value

v_accvgpr_read v12, a4 // (op 737) 
v_accvgpr_read v13, a5 // (op 737) 
v_accvgpr_read v14, a6 // (op 737) 
v_accvgpr_read v15, a7 // (op 737) 
v_mul_f32 v16, v10, v12 // (op 737) 
v_mul_f32 v17, v10, v13 // (op 737) 
v_mul_f32 v18, v10, v14 // (op 737) 
v_mul_f32 v19, v10, v15 // (op 737) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 742) 
v_lshrrev_b32 v13, 16, v4 // (op 742) 
v_cvt_f32_f16 v13, v13 // (op 742) 
v_mul_f32 v20, v7, v12 // (op 742) 
v_mul_f32 v21, v7, v13 // (op 742) 
v_cvt_f32_f16 v12, v5 // (op 742) 
v_lshrrev_b32 v13, 16, v5 // (op 742) 
v_cvt_f32_f16 v13, v13 // (op 742) 
v_mul_f32 v22, v7, v12 // (op 742) 
v_mul_f32 v23, v7, v13 // (op 742) 


v_add_f32 v12, v20, v16 // (op 738) 
v_add_f32 v13, v21, v17 // (op 738) 
v_add_f32 v14, v22, v18 // (op 738) 
v_add_f32 v15, v23, v19 // (op 738) 


v_cvt_f16_f32 v16, v12 // (op 739) 
v_cvt_f16_f32 v17, v13 // (op 739) 
v_cvt_f16_f32 v18, v14 // (op 739) 
v_cvt_f16_f32 v19, v15 // (op 739) 


v_pack_b32_f16 v12, v16, v17 // (op 740) 
v_pack_b32_f16 v13, v18, v19 // (op 740) 

s_lshl_b32 s0, s4, 6 // (op 740) 
v_add_u32 v0, v11, s0 // (op 740) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 740) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 29952 // lds bytes
  .amdhsa_accum_offset 132
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
    DataTypeB: B6
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [16, 64, 256]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 29952
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 132
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

