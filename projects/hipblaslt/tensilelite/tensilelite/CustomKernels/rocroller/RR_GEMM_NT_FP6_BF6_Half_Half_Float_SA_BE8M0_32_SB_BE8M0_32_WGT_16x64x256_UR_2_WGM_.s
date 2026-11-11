// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 1446) 
s_mul_hi_i32 s1, s16, s12 // (op 1688) 
v_and_b32 v0, 63, v1 // (op 1503) 
v_and_b32 v3, 48, v1 // (op 1503) 
v_sub_u32 v4, v0, v3 // (op 1503) 

v_lshrrev_b32 v3, 4, v0 // (op 1503) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1503) 


v_lshrrev_b32 v3, 7, v1 // (op 1503) 
v_lshlrev_b32 v4, 4, v3 // (op 1503) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1503) 


v_mul_lo_u32 v0, v3, 6 // (op 1503) 
v_lshrrev_b32 v4, 3, v0 // (op 1503) 

v_lshrrev_b32 v0, 4, v3 // (op 1503) 

v_lshlrev_b32 v3, 2, v0 // (op 1503) 

v_add_u32 v0, v4, v3 // (op 1503) 


v_lshrrev_b32 v3, 7, v1 // (op 1614) 
v_and_b32 v4, 15, v1 // (op 1614) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1614) 


v_and_b32 v3, 48, v1 // (op 1614) 
v_add_u32 v4, v3, v5 // (op 1614) 


v_and_b32 v3, 63, v1 // (op 1714) 
v_and_b32 v5, 48, v1 // (op 1714) 
v_sub_u32 v6, v3, v5 // (op 1714) 

v_lshrrev_b32 v5, 4, v3 // (op 1714) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 1714) 


v_lshrrev_b32 v5, 7, v1 // (op 1714) 
v_lshlrev_b32 v6, 4, v5 // (op 1714) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 1714) 


v_mul_lo_u32 v3, v5, 6 // (op 1714) 
v_lshrrev_b32 v6, 3, v3 // (op 1714) 

v_lshrrev_b32 v3, 4, v5 // (op 1714) 

v_lshlrev_b32 v5, 2, v3 // (op 1714) 

v_add_u32 v3, v6, v5 // (op 1714) 


v_lshrrev_b32 v5, 7, v1 // (op 1767) 
v_and_b32 v6, 15, v1 // (op 1767) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1767) 


v_and_b32 v5, 48, v1 // (op 1767) 
v_add_u32 v6, v5, v7 // (op 1767) 


v_lshrrev_b32 v5, 7, v1 // (op 1446) 
s_add_i32 s18, s0, s16 // (op 1446) 

v_and_b32 v7, 63, v1 // (op 1446) 
s_ashr_i32 s0, s18, 31 // (op 1446) 
v_lshrrev_b32 v8, 4, v7 // (op 1446) 
s_cmp_eq_i32 s12, 0 // (op 1446) 
s_mov_b32 s19, scc // (op 1446) 
v_lshlrev_b32 v9, 2, v8 // (op 1446) 

s_cmp_eq_i32 s19, 0 // (op 1446) 
s_cselect_b32 s51, 0, -1 // (op 1446) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1446) 


s_and_b32 s19, s15, 31 // (op 1446) 
v_lshrrev_b32 v5, 2, v7 // (op 1446) 

v_and_b32 v7, 3, v5 // (op 1446) 

v_and_b32 v5, 63, v1 // (op 1561) 
v_and_b32 v9, 48, v1 // (op 1561) 
v_sub_u32 v10, v5, v9 // (op 1561) 

v_lshrrev_b32 v9, 4, v5 // (op 1561) 

v_lshl_add_u32 v5, v9, 5, v10 // (op 1561) 


v_lshrrev_b32 v9, 6, v1 // (op 1561) 
v_and_b32 v10, 1, v9 // (op 1561) 

v_lshlrev_b32 v9, 4, v10 // (op 1561) 

v_lshl_add_u32 v10, v5, 6, v9 // (op 1561) 


v_mul_lo_u32 v5, v10, 6 // (op 1561) 
v_lshrrev_b32 v9, 3, v5 // (op 1561) 

v_lshrrev_b32 v5, 4, v10 // (op 1561) 

v_lshlrev_b32 v10, 2, v5 // (op 1561) 

v_add_u32 v5, v9, v10 // (op 1561) 


v_lshrrev_b32 v9, 6, v1 // (op 1672) 
v_and_b32 v10, 1, v9 // (op 1672) 

v_and_b32 v9, 15, v1 // (op 1672) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 1672) 


v_and_b32 v9, 63, v1 // (op 1672) 
v_lshrrev_b32 v10, 4, v9 // (op 1672) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1672) 


v_lshrrev_b32 v10, 7, v1 // (op 1688) 
v_and_b32 v11, 63, v1 // (op 1688) 
v_and_b32 v12, 63, v1 // (op 1743) 
v_and_b32 v13, 48, v1 // (op 1743) 
v_sub_u32 v14, v12, v13 // (op 1743) 

v_lshrrev_b32 v13, 4, v12 // (op 1743) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 1743) 


v_lshrrev_b32 v13, 6, v1 // (op 1743) 
v_and_b32 v14, 1, v13 // (op 1743) 

v_lshlrev_b32 v13, 4, v14 // (op 1743) 

v_lshl_add_u32 v14, v12, 6, v13 // (op 1743) 


v_mul_lo_u32 v12, v14, 6 // (op 1743) 
v_lshrrev_b32 v13, 3, v12 // (op 1743) 

v_lshrrev_b32 v12, 4, v14 // (op 1743) 

v_lshlrev_b32 v14, 2, v12 // (op 1743) 

v_add_u32 v12, v13, v14 // (op 1743) 


v_lshrrev_b32 v13, 6, v1 // (op 1796) 
v_and_b32 v14, 1, v13 // (op 1796) 

v_and_b32 v13, 15, v1 // (op 1796) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1796) 


v_and_b32 v13, 63, v1 // (op 1796) 
v_lshrrev_b32 v14, 4, v13 // (op 1796) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 1796) 


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

s_lshl_b32 s19, s15, 24 // (op 1446) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1446) 


s_ashr_i32 s51, s19, 31 // (op 1446) 

s_xor_b32 s19, s18, s51 // (op 1446) 

s_sub_i32 s18, s19, s51 // (op 1446) 


v_lshrrev_b32 v14, 4, v11 // (op 1688) 
s_cmp_eq_i32 s12, 0 // (op 1688) 
s_mov_b32 s19, scc // (op 1688) 
v_lshlrev_b32 v15, 2, v14 // (op 1688) 

s_cmp_eq_i32 s19, 0 // (op 1688) 
s_cselect_b32 s51, 0, -1 // (op 1688) 

v_lshl_add_u32 v14, v10, 4, v15 // (op 1688) 


s_and_b32 s19, s15, 31 // (op 1688) 
v_lshrrev_b32 v10, 2, v11 // (op 1688) 

s_lshl_b32 s52, 1, s19 // (op 1688) 
v_and_b32 v11, 3, v10 // (op 1688) 

s_add_i32 s53, s52, s51 // (op 1688) 


v_and_b32 v10, 3, v1 // (op 1688) 
s_and_b32 s51, s1, s53 // (op 1688) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1688) 


s_add_i32 s1, s0, s51 // (op 1688) 


v_lshrrev_b32 v10, 6, v1 // (op 1688) 
s_ashr_i32 s0, s1, s19 // (op 1688) 


v_and_b32 v11, 1, v10 // (op 1688) 

s_lshl_b32 s1, s15, 24 // (op 1688) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1688) 


s_ashr_i32 s19, s1, 31 // (op 1688) 

s_xor_b32 s1, s0, s19 // (op 1688) 

s_sub_i32 s0, s1, s19 // (op 1688) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 1446) 
s_add_i32 s19, s1, s18 // (op 1446) 

s_ashr_i32 s1, s19, 31 // (op 1446) 
s_cmp_eq_i32 s44, 0 // (op 1446) 
s_mov_b32 s51, scc // (op 1446) 
s_cmp_eq_i32 s51, 0 // (op 1446) 
s_cselect_b32 s52, 0, -1 // (op 1446) 

s_and_b32 s51, s45, 31 // (op 1446) 
s_lshl_b32 s53, 1, s51 // (op 1446) 
s_add_i32 s54, s53, s52 // (op 1446) 


s_and_b32 s52, s1, s54 // (op 1446) 


s_add_i32 s1, s19, s52 // (op 1446) 


s_ashr_i32 s19, s1, s51 // (op 1446) 


s_lshl_b32 s1, s45, 24 // (op 1446) 
s_ashr_i32 s51, s1, 31 // (op 1446) 

s_xor_b32 s1, s19, s51 // (op 1446) 

s_sub_i32 s19, s1, s51 // (op 1446) 


s_mul_i32 s1, s18, s13 // (op 1446) 
s_sub_u32 s51, s16, s1 // (op 1446) 

s_mul_i32 s1, s19, s13 // (op 1446) 
s_add_u32 s1, s1, s51 // (op 1446) 

s_mul_hi_i32 s51, s16, s11 // (op 1446) 
s_add_i32 s52, s51, s16 // (op 1446) 

s_ashr_i32 s51, s52, 31 // (op 1446) 
s_cmp_eq_i32 s11, 0 // (op 1446) 
s_mov_b32 s53, scc // (op 1446) 
s_cmp_eq_i32 s53, 0 // (op 1446) 
s_cselect_b32 s54, 0, -1 // (op 1446) 

s_and_b32 s53, s47, 31 // (op 1446) 
s_lshl_b32 s55, 1, s53 // (op 1446) 
s_add_i32 s56, s55, s54 // (op 1446) 


s_and_b32 s54, s51, s56 // (op 1446) 


s_add_i32 s51, s52, s54 // (op 1446) 


s_ashr_i32 s52, s51, s53 // (op 1446) 


s_lshl_b32 s51, s47, 24 // (op 1446) 
s_ashr_i32 s53, s51, 31 // (op 1446) 

s_xor_b32 s51, s52, s53 // (op 1446) 

s_sub_i32 s52, s51, s53 // (op 1446) 


s_mul_hi_i32 s51, s0, s44 // (op 1688) 
s_add_i32 s53, s51, s0 // (op 1688) 

s_ashr_i32 s51, s53, 31 // (op 1688) 
s_cmp_eq_i32 s44, 0 // (op 1688) 
s_mov_b32 s54, scc // (op 1688) 
s_cmp_eq_i32 s54, 0 // (op 1688) 
s_cselect_b32 s55, 0, -1 // (op 1688) 

s_and_b32 s54, s45, 31 // (op 1688) 
s_lshl_b32 s56, 1, s54 // (op 1688) 
s_add_i32 s57, s56, s55 // (op 1688) 


s_and_b32 s55, s51, s57 // (op 1688) 


s_add_i32 s51, s53, s55 // (op 1688) 


s_ashr_i32 s53, s51, s54 // (op 1688) 


s_lshl_b32 s51, s45, 24 // (op 1688) 
s_ashr_i32 s54, s51, 31 // (op 1688) 

s_xor_b32 s51, s53, s54 // (op 1688) 

s_sub_i32 s53, s51, s54 // (op 1688) 


s_mul_i32 s51, s0, s13 // (op 1688) 
s_sub_u32 s54, s16, s51 // (op 1688) 

s_mul_i32 s51, s53, s13 // (op 1688) 
s_add_u32 s51, s51, s54 // (op 1688) 

s_mul_hi_i32 s54, s16, s11 // (op 1688) 
s_add_i32 s55, s54, s16 // (op 1688) 

s_ashr_i32 s54, s55, 31 // (op 1688) 
s_cmp_eq_i32 s11, 0 // (op 1688) 
s_mov_b32 s56, scc // (op 1688) 
s_cmp_eq_i32 s56, 0 // (op 1688) 
s_cselect_b32 s57, 0, -1 // (op 1688) 

s_and_b32 s56, s47, 31 // (op 1688) 
s_lshl_b32 s58, 1, s56 // (op 1688) 
s_add_i32 s59, s58, s57 // (op 1688) 


s_and_b32 s57, s54, s59 // (op 1688) 


s_add_i32 s54, s55, s57 // (op 1688) 


s_ashr_i32 s55, s54, s56 // (op 1688) 


s_lshl_b32 s54, s47, 24 // (op 1688) 
s_ashr_i32 s56, s54, 31 // (op 1688) 

s_xor_b32 s54, s55, s56 // (op 1688) 

s_sub_i32 s55, s54, s56 // (op 1688) 


s_add_u32 s56, s8, 64 // (op 1446) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 1446) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1446) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1446) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1446) 

s_mul_i32 s54, s13, s56 // (op 1446) 
s_mul_i32 s58, s10, s54 // (op 1446) 

s_mul_i32 s54, s52, s58 // (op 1446) 

s_sub_u32 s58, s16, s54 // (op 1446) 

s_mov_b32 s54, s58 // (op 1446) 
s_mul_hi_i32 s59, s54, s14 // (op 1446) 
s_add_i32 s60, s59, s54 // (op 1446) 


s_ashr_i32 s54, s60, 31 // (op 1446) 
s_cmp_eq_i32 s14, 0 // (op 1446) 
s_mov_b32 s59, scc // (op 1446) 
s_cmp_eq_i32 s59, 0 // (op 1446) 
s_cselect_b32 s61, 0, -1 // (op 1446) 

s_and_b32 s59, s46, 31 // (op 1446) 
s_lshl_b32 s62, 1, s59 // (op 1446) 
s_add_i32 s63, s62, s61 // (op 1446) 


s_and_b32 s61, s54, s63 // (op 1446) 


s_add_i32 s54, s60, s61 // (op 1446) 


s_ashr_i32 s60, s54, s59 // (op 1446) 


s_lshl_b32 s54, s46, 24 // (op 1446) 
s_ashr_i32 s59, s54, 31 // (op 1446) 

s_xor_b32 s54, s60, s59 // (op 1446) 

s_sub_i32 s60, s54, s59 // (op 1446) 


s_mul_i32 s54, s48, s13 // (op 1446) 
s_add_u32 s62, s6, 16 // (op 1446) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 1446) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1446) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1446) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1446) 

s_sub_i32 s59, s62, s54 // (op 1446) 

s_mul_i32 s61, s60, s59 // (op 1446) 

s_sub_u32 s59, s58, s61 // (op 1446) 


s_add_u32 s58, s59, s54 // (op 1446) 


s_cmp_eq_i32 s52, 0 // (op 1446) 
s_mov_b32 s54, scc // (op 1446) 

s_cmp_eq_i32 s54, 0 // (op 1446) 
s_cselect_b32 s52, s58, s1 // (op 1446) 


s_mul_i32 s1, s19, s56 // (op 1446) 


v_lshl_add_u32 v11, s52, 4, v8 // (op 1446) 


s_sub_i32 s19, s18, s1 // (op 1446) 


s_cmp_eq_i32 s54, 0 // (op 1446) 
s_cselect_b32 s1, s60, s19 // (op 1446) 



v_lshl_add_u32 v8, s1, 6, v7 // (op 1446) 


v_mul_lo_u32 v7, v8, s2 // (op 1446) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 1446) 


s_add_u32 s18, s8, 64 // (op 1688) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 1688) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 1688) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 1688) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 6 // (op 1688) 

s_mul_i32 s1, s13, s18 // (op 1688) 
s_mul_i32 s52, s10, s1 // (op 1688) 

s_mul_i32 s1, s55, s52 // (op 1688) 

s_sub_u32 s52, s16, s1 // (op 1688) 

s_mov_b32 s1, s52 // (op 1688) 
s_mul_hi_i32 s54, s1, s14 // (op 1688) 
s_add_i32 s56, s54, s1 // (op 1688) 


s_ashr_i32 s1, s56, 31 // (op 1688) 
s_cmp_eq_i32 s14, 0 // (op 1688) 
s_mov_b32 s54, scc // (op 1688) 
s_cmp_eq_i32 s54, 0 // (op 1688) 
s_cselect_b32 s57, 0, -1 // (op 1688) 

s_and_b32 s54, s46, 31 // (op 1688) 
s_lshl_b32 s58, 1, s54 // (op 1688) 
s_add_i32 s59, s58, s57 // (op 1688) 


s_and_b32 s57, s1, s59 // (op 1688) 


s_add_i32 s1, s56, s57 // (op 1688) 


s_ashr_i32 s56, s1, s54 // (op 1688) 


s_lshl_b32 s1, s46, 24 // (op 1688) 
s_ashr_i32 s54, s1, 31 // (op 1688) 

s_xor_b32 s1, s56, s54 // (op 1688) 

s_sub_i32 s56, s1, s54 // (op 1688) 


s_mul_i32 s1, s48, s13 // (op 1688) 
s_add_u32 s58, s6, 16 // (op 1688) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1688) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1688) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1688) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1688) 

s_sub_i32 s54, s58, s1 // (op 1688) 

s_mul_i32 s57, s56, s54 // (op 1688) 

s_sub_u32 s54, s52, s57 // (op 1688) 


s_add_u32 s52, s54, s1 // (op 1688) 


s_cmp_eq_i32 s55, 0 // (op 1688) 
s_mov_b32 s1, scc // (op 1688) 

s_cmp_eq_i32 s1, 0 // (op 1688) 
s_cselect_b32 s54, s52, s51 // (op 1688) 


s_mul_i32 s55, s53, s18 // (op 1688) 


v_lshl_add_u32 v7, s54, 4, v14 // (op 1688) 


s_sub_i32 s18, s0, s55 // (op 1688) 


s_cmp_eq_i32 s1, 0 // (op 1688) 
s_cselect_b32 s0, s56, s18 // (op 1688) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1688) 


v_mul_lo_u32 v10, v11, s4 // (op 1688) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 1688) 


s_mul_i32 s1, s2, s9 // (op 1448) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1448) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1448) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1448) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1448) most significant: sum
s_add_u32 s1, s1, s19 // (op 1448) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1448) 

s_mul_i32 s1, s4, s9 // (op 1690) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 1690) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1690) least significant
s_mul_hi_u32 s52, s4, s8 // (op 1690) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1690) most significant: sum
s_add_u32 s1, s1, s52 // (op 1690) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1690) 

s_mov_b32 s56, s20 // (op 1448) reg expression
s_mov_b32 s57, s21 // (op 1448) reg expression
s_mov_b32 s58, s18 // (op 1448) reg expression
s_mov_b32 s59, 131072 // (op 1448) call()

s_mov_b32 s60, s22 // (op 1690) reg expression

s_mov_b32 s61, s23 // (op 1690) reg expression
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
s_mul_hi_i32 s0, s16, s12 // (op 1213) 
v_lshlrev_b32 v14, 1, v1 // (op 1213) 
s_add_i32 s1, s0, s16 // (op 1213) 

s_ashr_i32 s0, s1, 31 // (op 1213) 
s_cmp_eq_i32 s12, 0 // (op 1213) 
s_mov_b32 s18, scc // (op 1213) 
s_cmp_eq_i32 s18, 0 // (op 1213) 
s_cselect_b32 s19, 0, -1 // (op 1213) 

s_and_b32 s18, s15, 31 // (op 1213) 
s_lshl_b32 s20, 1, s18 // (op 1213) 
s_add_i32 s21, s20, s19 // (op 1213) 


s_and_b32 s19, s0, s21 // (op 1213) 


s_add_i32 s0, s1, s19 // (op 1213) 


s_ashr_i32 s1, s0, s18 // (op 1213) 


s_lshl_b32 s0, s15, 24 // (op 1213) 
s_ashr_i32 s18, s0, 31 // (op 1213) 

s_xor_b32 s0, s1, s18 // (op 1213) 

s_sub_i32 s1, s0, s18 // (op 1213) 


s_mul_hi_i32 s0, s1, s44 // (op 1213) 
s_add_i32 s18, s0, s1 // (op 1213) 

s_ashr_i32 s0, s18, 31 // (op 1213) 
s_cmp_eq_i32 s44, 0 // (op 1213) 
s_mov_b32 s19, scc // (op 1213) 
s_cmp_eq_i32 s19, 0 // (op 1213) 
s_cselect_b32 s20, 0, -1 // (op 1213) 

s_and_b32 s19, s45, 31 // (op 1213) 
s_lshl_b32 s21, 1, s19 // (op 1213) 
s_add_i32 s22, s21, s20 // (op 1213) 


s_and_b32 s23, s0, s22 // (op 1213) 


s_add_i32 s0, s18, s23 // (op 1213) 


s_ashr_i32 s18, s0, s19 // (op 1213) 


s_lshl_b32 s0, s45, 24 // (op 1213) 
s_ashr_i32 s19, s0, 31 // (op 1213) 

s_xor_b32 s0, s18, s19 // (op 1213) 

s_sub_i32 s18, s0, s19 // (op 1213) 


s_mul_i32 s0, s1, s13 // (op 1213) 

s_sub_u32 s1, s16, s0 // (op 1213) 

s_mul_i32 s0, s18, s13 // (op 1213) 
s_add_u32 s0, s0, s1 // (op 1213) 


s_mul_hi_i32 s1, s16, s11 // (op 1213) 
s_add_i32 s18, s1, s16 // (op 1213) 

s_ashr_i32 s1, s18, 31 // (op 1213) 
s_cmp_eq_i32 s11, 0 // (op 1213) 
s_mov_b32 s19, scc // (op 1213) 
s_cmp_eq_i32 s19, 0 // (op 1213) 
s_cselect_b32 s20, 0, -1 // (op 1213) 

s_and_b32 s19, s47, 31 // (op 1213) 
s_lshl_b32 s21, 1, s19 // (op 1213) 
s_add_i32 s22, s21, s20 // (op 1213) 


s_and_b32 s23, s1, s22 // (op 1213) 


s_add_i32 s1, s18, s23 // (op 1213) 


s_ashr_i32 s18, s1, s19 // (op 1213) 


s_lshl_b32 s1, s47, 24 // (op 1213) 
s_ashr_i32 s19, s1, 31 // (op 1213) 

s_xor_b32 s1, s18, s19 // (op 1213) 

s_sub_i32 s18, s1, s19 // (op 1213) 


s_add_u32 s22, s8, 64 // (op 1213) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 1213) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 1213) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 1213) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 1213) 

s_mul_i32 s1, s13, s22 // (op 1213) 

s_mul_i32 s19, s10, s1 // (op 1213) 

s_mul_i32 s1, s18, s19 // (op 1213) 

s_sub_u32 s19, s16, s1 // (op 1213) 

s_mov_b32 s1, s19 // (op 1213) 
s_mul_hi_i32 s20, s1, s14 // (op 1213) 
s_add_i32 s21, s20, s1 // (op 1213) 


s_ashr_i32 s1, s21, 31 // (op 1213) 
s_cmp_eq_i32 s14, 0 // (op 1213) 
s_mov_b32 s20, scc // (op 1213) 
s_cmp_eq_i32 s20, 0 // (op 1213) 
s_cselect_b32 s22, 0, -1 // (op 1213) 

s_and_b32 s20, s46, 31 // (op 1213) 
s_lshl_b32 s23, 1, s20 // (op 1213) 
s_add_i32 s49, s23, s22 // (op 1213) 


s_and_b32 s22, s1, s49 // (op 1213) 


s_add_i32 s1, s21, s22 // (op 1213) 


s_ashr_i32 s21, s1, s20 // (op 1213) 


s_lshl_b32 s1, s46, 24 // (op 1213) 
s_ashr_i32 s20, s1, 31 // (op 1213) 

s_xor_b32 s1, s21, s20 // (op 1213) 

s_sub_i32 s21, s1, s20 // (op 1213) 


s_mul_i32 s1, s48, s13 // (op 1213) 
s_add_u32 s22, s6, 16 // (op 1213) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1213) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 1213) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 1213) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 1213) 

s_sub_i32 s20, s22, s1 // (op 1213) 

s_mul_i32 s22, s21, s20 // (op 1213) 


s_sub_u32 s23, s19, s22 // (op 1213) 


s_add_u32 s19, s23, s1 // (op 1213) 


s_cmp_eq_i32 s18, 0 // (op 1213) 
s_mov_b32 s1, scc // (op 1213) 

s_cmp_eq_i32 s1, 0 // (op 1213) 
s_cselect_b32 s18, s19, s0 // (op 1213) 



s_lshl_b32 s0, s18, 4 // (op 1213) 

s_mov_b32 s19, 0 // (op 1213) convert
s_mov_b32 s18, s0 // (op 1213) convert

v_mul_lo_u32 v17, v14, s25 // (op 1213) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1213) most significant: high of low * low
v_mul_lo_u32 v16, v14, s24 // (op 1213) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1213) most significant: sum


v_mov_b32 v14, s18 // (op 1213) 
v_mov_b32 v15, s19 // (op 1213) 
v_add_co_u32 v18, s[0:1], v14, v16 // (op 1213) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1213) most significant half





v_mul_lo_u32 v15, v19, 6 // (op 1213) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 1213) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 1213) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1213) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1213) 

v_mov_b32 v14, v16 // (op 1213) convert

s_mul_i32 s1, s29, 6 // (op 1215) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1215) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1215) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1215) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1215) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 1215) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1215) least significant
s_mul_hi_u32 s21, s24, s0 // (op 1215) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 1215) most significant: sum
s_add_u32 s19, s19, s21 // (op 1215) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1215) 

s_mov_b32 s20, s26 // (op 1215) reg expression
s_mov_b32 s21, s27 // (op 1215) reg expression
s_mov_b32 s22, s0 // (op 1215) reg expression
s_mov_b32 s23, 131072 // (op 1215) call()

s_mul_hi_i32 s0, s16, s12 // (op 1232) 
v_lshlrev_b32 v15, 1, v1 // (op 1232) 
s_add_i32 s1, s0, s16 // (op 1232) 

s_ashr_i32 s0, s1, 31 // (op 1232) 
s_cmp_eq_i32 s12, 0 // (op 1232) 
s_mov_b32 s18, scc // (op 1232) 
s_cmp_eq_i32 s18, 0 // (op 1232) 
s_cselect_b32 s19, 0, -1 // (op 1232) 

s_and_b32 s18, s15, 31 // (op 1232) 
s_lshl_b32 s49, 1, s18 // (op 1232) 
s_add_i32 s50, s49, s19 // (op 1232) 


s_and_b32 s19, s0, s50 // (op 1232) 


s_add_i32 s0, s1, s19 // (op 1232) 


s_ashr_i32 s1, s0, s18 // (op 1232) 


s_lshl_b32 s0, s15, 24 // (op 1232) 
s_ashr_i32 s18, s0, 31 // (op 1232) 

s_xor_b32 s0, s1, s18 // (op 1232) 

s_sub_i32 s1, s0, s18 // (op 1232) 


s_mul_hi_i32 s0, s1, s44 // (op 1232) 
s_add_i32 s18, s0, s1 // (op 1232) 

s_ashr_i32 s0, s18, 31 // (op 1232) 
s_cmp_eq_i32 s44, 0 // (op 1232) 
s_mov_b32 s19, scc // (op 1232) 
s_cmp_eq_i32 s19, 0 // (op 1232) 
s_cselect_b32 s49, 0, -1 // (op 1232) 

s_and_b32 s19, s45, 31 // (op 1232) 
s_lshl_b32 s50, 1, s19 // (op 1232) 
s_add_i32 s51, s50, s49 // (op 1232) 


s_and_b32 s49, s0, s51 // (op 1232) 


s_add_i32 s0, s18, s49 // (op 1232) 


s_ashr_i32 s18, s0, s19 // (op 1232) 


s_lshl_b32 s0, s45, 24 // (op 1232) 
s_ashr_i32 s19, s0, 31 // (op 1232) 

s_xor_b32 s0, s18, s19 // (op 1232) 

s_sub_i32 s18, s0, s19 // (op 1232) 


s_mul_i32 s0, s1, s13 // (op 1232) 

s_sub_u32 s1, s16, s0 // (op 1232) 

s_mul_i32 s0, s18, s13 // (op 1232) 
s_add_u32 s0, s0, s1 // (op 1232) 


s_mul_hi_i32 s1, s16, s11 // (op 1232) 
s_add_i32 s18, s1, s16 // (op 1232) 

s_ashr_i32 s1, s18, 31 // (op 1232) 
s_cmp_eq_i32 s11, 0 // (op 1232) 
s_mov_b32 s19, scc // (op 1232) 
s_cmp_eq_i32 s19, 0 // (op 1232) 
s_cselect_b32 s49, 0, -1 // (op 1232) 

s_and_b32 s19, s47, 31 // (op 1232) 
s_lshl_b32 s50, 1, s19 // (op 1232) 
s_add_i32 s51, s50, s49 // (op 1232) 


s_and_b32 s49, s1, s51 // (op 1232) 


s_add_i32 s1, s18, s49 // (op 1232) 


s_ashr_i32 s18, s1, s19 // (op 1232) 


s_lshl_b32 s1, s47, 24 // (op 1232) 
s_ashr_i32 s19, s1, 31 // (op 1232) 

s_xor_b32 s1, s18, s19 // (op 1232) 

s_sub_i32 s18, s1, s19 // (op 1232) 


s_add_u32 s54, s8, 64 // (op 1232) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 1232) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1232) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1232) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1232) 

s_mul_i32 s1, s13, s50 // (op 1232) 

s_mul_i32 s19, s10, s1 // (op 1232) 

s_mul_i32 s1, s18, s19 // (op 1232) 

s_sub_u32 s19, s16, s1 // (op 1232) 

s_mov_b32 s1, s19 // (op 1232) 
s_mul_hi_i32 s49, s1, s14 // (op 1232) 
s_add_i32 s50, s49, s1 // (op 1232) 


s_ashr_i32 s1, s50, 31 // (op 1232) 
s_cmp_eq_i32 s14, 0 // (op 1232) 
s_mov_b32 s49, scc // (op 1232) 
s_cmp_eq_i32 s49, 0 // (op 1232) 
s_cselect_b32 s51, 0, -1 // (op 1232) 

s_and_b32 s49, s46, 31 // (op 1232) 
s_lshl_b32 s52, 1, s49 // (op 1232) 
s_add_i32 s53, s52, s51 // (op 1232) 


s_and_b32 s51, s1, s53 // (op 1232) 


s_add_i32 s1, s50, s51 // (op 1232) 


s_ashr_i32 s50, s1, s49 // (op 1232) 


s_lshl_b32 s1, s46, 24 // (op 1232) 
s_ashr_i32 s49, s1, 31 // (op 1232) 

s_xor_b32 s1, s50, s49 // (op 1232) 

s_sub_i32 s50, s1, s49 // (op 1232) 


s_mul_i32 s1, s48, s13 // (op 1232) 
s_add_u32 s54, s6, 16 // (op 1232) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1232) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1232) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1232) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 1232) 

s_sub_i32 s49, s54, s1 // (op 1232) 

s_mul_i32 s51, s50, s49 // (op 1232) 


s_sub_u32 s49, s19, s51 // (op 1232) 


s_add_u32 s19, s49, s1 // (op 1232) 


s_cmp_eq_i32 s18, 0 // (op 1232) 
s_mov_b32 s1, scc // (op 1232) 

s_cmp_eq_i32 s1, 0 // (op 1232) 
s_cselect_b32 s18, s19, s0 // (op 1232) 



s_lshl_b32 s19, s18, 4 // (op 1232) 

s_mov_b32 s1, 0 // (op 1232) convert
s_mov_b32 s0, s19 // (op 1232) convert

v_mul_lo_u32 v17, v15, s25 // (op 1232) most significant: low of low * high
v_mul_hi_u32 v18, v15, s24 // (op 1232) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 1232) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1232) most significant: sum


v_mov_b32 v15, s0 // (op 1232) 
v_mov_b32 v18, s1 // (op 1232) 
v_add_co_u32 v20, s[18:19], v15, v16 // (op 1232) least significant half
v_addc_co_u32 v21, s[18:19], v18, v17, s[18:19] // (op 1232) most significant half





v_mul_lo_u32 v19, v21, 6 // (op 1232) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 1232) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 1232) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1232) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1232) 

v_mov_b32 v15, v16 // (op 1232) convert

s_mul_i32 s1, s29, 6 // (op 1234) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1234) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1234) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1234) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1234) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 1234) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1234) least significant
s_mul_hi_u32 s50, s24, s0 // (op 1234) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1234) most significant: sum
s_add_u32 s19, s19, s50 // (op 1234) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1234) 

s_mov_b32 s20, s26 // (op 1234) reg expression
s_mov_b32 s21, s27 // (op 1234) reg expression
s_mov_b32 s22, s0 // (op 1234) reg expression
s_mov_b32 s23, 131072 // (op 1234) call()

s_mul_hi_i32 s0, s16, s12 // (op 1277) 
v_lshlrev_b32 v16, 1, v1 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
s_cmp_eq_i32 s12, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

s_and_b32 s18, s15, 31 // (op 1277) 
s_lshl_b32 s49, 1, s18 // (op 1277) 
s_add_i32 s50, s49, s19 // (op 1277) 


s_and_b32 s19, s0, s50 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s15, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_hi_i32 s0, s1, s44 // (op 1277) 
s_add_i32 s18, s0, s1 // (op 1277) 

s_ashr_i32 s0, s18, 31 // (op 1277) 
s_cmp_eq_i32 s44, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s49, 0, -1 // (op 1277) 

s_and_b32 s19, s45, 31 // (op 1277) 
s_lshl_b32 s50, 1, s19 // (op 1277) 
s_add_i32 s51, s50, s49 // (op 1277) 


s_and_b32 s49, s0, s51 // (op 1277) 


s_add_i32 s0, s18, s49 // (op 1277) 


s_ashr_i32 s18, s0, s19 // (op 1277) 


s_lshl_b32 s0, s45, 24 // (op 1277) 
s_ashr_i32 s19, s0, 31 // (op 1277) 

s_xor_b32 s0, s18, s19 // (op 1277) 

s_sub_i32 s18, s0, s19 // (op 1277) 


s_add_u32 s54, s8, 64 // (op 1277) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 1277) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1277) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1277) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1277) 

s_mul_i32 s0, s18, s50 // (op 1277) 

s_sub_i32 s49, s1, s0 // (op 1277) 


s_mul_hi_i32 s0, s16, s11 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
s_cmp_eq_i32 s11, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

s_and_b32 s18, s47, 31 // (op 1277) 
s_lshl_b32 s52, 1, s18 // (op 1277) 
s_add_i32 s53, s52, s19 // (op 1277) 


s_and_b32 s19, s0, s53 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s47, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_i32 s0, s13, s50 // (op 1277) 

s_mul_i32 s18, s10, s0 // (op 1277) 

s_mul_i32 s0, s1, s18 // (op 1277) 

s_sub_u32 s18, s16, s0 // (op 1277) 

s_mul_hi_i32 s0, s18, s14 // (op 1277) 
s_add_i32 s19, s0, s18 // (op 1277) 


s_ashr_i32 s0, s19, 31 // (op 1277) 
s_cmp_eq_i32 s14, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s50, 0, -1 // (op 1277) 

s_and_b32 s18, s46, 31 // (op 1277) 
s_lshl_b32 s51, 1, s18 // (op 1277) 
s_add_i32 s52, s51, s50 // (op 1277) 


s_and_b32 s50, s0, s52 // (op 1277) 


s_add_i32 s0, s19, s50 // (op 1277) 


s_ashr_i32 s19, s0, s18 // (op 1277) 


s_lshl_b32 s0, s46, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s19, s18 // (op 1277) 

s_sub_i32 s19, s0, s18 // (op 1277) 


s_cmp_eq_i32 s1, 0 // (op 1277) 
s_mov_b32 s0, scc // (op 1277) 

s_cmp_eq_i32 s0, 0 // (op 1277) 
s_cselect_b32 s1, s19, s49 // (op 1277) 



s_lshl_b32 s0, s1, 6 // (op 1277) 

s_mov_b32 s19, 0 // (op 1277) convert
s_mov_b32 s18, s0 // (op 1277) convert

v_mul_lo_u32 v19, v16, s31 // (op 1277) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v18, v16, s30 // (op 1277) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1277) most significant: sum

v_mov_b32 v17, s18 // (op 1277) 
v_mov_b32 v20, s19 // (op 1277) 
v_add_co_u32 v18, s[0:1], v18, v17 // (op 1277) least significant half
v_addc_co_u32 v19, s[0:1], v19, v20, s[0:1] // (op 1277) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 1277) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1277) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 1277) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1277) 

v_mov_b32 v16, v18 // (op 1277) convert

s_mul_i32 s1, s29, 6 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1279) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1279) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1279) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 1279) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1279) least significant
s_mul_hi_u32 s50, s30, s0 // (op 1279) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1279) most significant: sum
s_add_u32 s19, s19, s50 // (op 1279) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1279) 

s_mov_b32 s52, s32 // (op 1279) reg expression
s_mov_b32 s53, s33 // (op 1279) reg expression
s_mov_b32 s54, s0 // (op 1279) reg expression
s_mov_b32 s55, 131072 // (op 1279) call()

s_mul_hi_i32 s0, s16, s12 // (op 1296) 
v_lshlrev_b32 v17, 1, v1 // (op 1296) 
s_add_i32 s1, s0, s16 // (op 1296) 

s_ashr_i32 s0, s1, 31 // (op 1296) 
s_cmp_eq_i32 s12, 0 // (op 1296) 
s_mov_b32 s18, scc // (op 1296) 
s_cmp_eq_i32 s18, 0 // (op 1296) 
s_cselect_b32 s19, 0, -1 // (op 1296) 

s_and_b32 s18, s15, 31 // (op 1296) 
s_lshl_b32 s49, 1, s18 // (op 1296) 
s_add_i32 s50, s49, s19 // (op 1296) 


s_and_b32 s19, s0, s50 // (op 1296) 


s_add_i32 s0, s1, s19 // (op 1296) 


s_ashr_i32 s1, s0, s18 // (op 1296) 


s_lshl_b32 s0, s15, 24 // (op 1296) 
s_ashr_i32 s18, s0, 31 // (op 1296) 

s_xor_b32 s0, s1, s18 // (op 1296) 

s_sub_i32 s1, s0, s18 // (op 1296) 


s_mul_hi_i32 s0, s1, s44 // (op 1296) 
s_add_i32 s18, s0, s1 // (op 1296) 

s_ashr_i32 s0, s18, 31 // (op 1296) 
s_cmp_eq_i32 s44, 0 // (op 1296) 
s_mov_b32 s19, scc // (op 1296) 
s_cmp_eq_i32 s19, 0 // (op 1296) 
s_cselect_b32 s49, 0, -1 // (op 1296) 

s_and_b32 s19, s45, 31 // (op 1296) 
s_lshl_b32 s50, 1, s19 // (op 1296) 
s_add_i32 s51, s50, s49 // (op 1296) 


s_and_b32 s49, s0, s51 // (op 1296) 


s_add_i32 s0, s18, s49 // (op 1296) 


s_ashr_i32 s18, s0, s19 // (op 1296) 


s_lshl_b32 s0, s45, 24 // (op 1296) 
s_ashr_i32 s19, s0, 31 // (op 1296) 

s_xor_b32 s0, s18, s19 // (op 1296) 

s_sub_i32 s18, s0, s19 // (op 1296) 


s_add_u32 s50, s8, 64 // (op 1296) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1296) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1296) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1296) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1296) 

s_mul_i32 s0, s18, s50 // (op 1296) 

s_sub_i32 s49, s1, s0 // (op 1296) 


s_mul_hi_i32 s0, s16, s11 // (op 1296) 
s_add_i32 s1, s0, s16 // (op 1296) 

s_ashr_i32 s0, s1, 31 // (op 1296) 
s_cmp_eq_i32 s11, 0 // (op 1296) 
s_mov_b32 s18, scc // (op 1296) 
s_cmp_eq_i32 s18, 0 // (op 1296) 
s_cselect_b32 s19, 0, -1 // (op 1296) 

s_and_b32 s18, s47, 31 // (op 1296) 
s_lshl_b32 s64, 1, s18 // (op 1296) 
s_add_i32 s65, s64, s19 // (op 1296) 


s_and_b32 s19, s0, s65 // (op 1296) 


s_add_i32 s0, s1, s19 // (op 1296) 


s_ashr_i32 s1, s0, s18 // (op 1296) 


s_lshl_b32 s0, s47, 24 // (op 1296) 
s_ashr_i32 s18, s0, 31 // (op 1296) 

s_xor_b32 s0, s1, s18 // (op 1296) 

s_sub_i32 s1, s0, s18 // (op 1296) 


s_mul_i32 s0, s13, s50 // (op 1296) 

s_mul_i32 s18, s10, s0 // (op 1296) 

s_mul_i32 s0, s1, s18 // (op 1296) 

s_sub_u32 s18, s16, s0 // (op 1296) 

s_mul_hi_i32 s0, s18, s14 // (op 1296) 
s_add_i32 s19, s0, s18 // (op 1296) 


s_ashr_i32 s0, s19, 31 // (op 1296) 
s_cmp_eq_i32 s14, 0 // (op 1296) 
s_mov_b32 s18, scc // (op 1296) 
s_cmp_eq_i32 s18, 0 // (op 1296) 
s_cselect_b32 s50, 0, -1 // (op 1296) 

s_and_b32 s18, s46, 31 // (op 1296) 
s_lshl_b32 s51, 1, s18 // (op 1296) 
s_add_i32 s64, s51, s50 // (op 1296) 


s_and_b32 s50, s0, s64 // (op 1296) 


s_add_i32 s0, s19, s50 // (op 1296) 


s_ashr_i32 s19, s0, s18 // (op 1296) 


s_lshl_b32 s0, s46, 24 // (op 1296) 
s_ashr_i32 s18, s0, 31 // (op 1296) 

s_xor_b32 s0, s19, s18 // (op 1296) 

s_sub_i32 s19, s0, s18 // (op 1296) 


s_cmp_eq_i32 s1, 0 // (op 1296) 
s_mov_b32 s0, scc // (op 1296) 

s_cmp_eq_i32 s0, 0 // (op 1296) 
s_cselect_b32 s1, s19, s49 // (op 1296) 



s_lshl_b32 s0, s1, 6 // (op 1296) 

s_mov_b32 s19, 0 // (op 1296) convert
s_mov_b32 s18, s0 // (op 1296) convert

v_mul_lo_u32 v19, v17, s31 // (op 1296) most significant: low of low * high
v_mul_hi_u32 v20, v17, s30 // (op 1296) most significant: high of low * low
v_mul_lo_u32 v18, v17, s30 // (op 1296) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1296) most significant: sum

v_mov_b32 v20, s18 // (op 1296) 
v_mov_b32 v21, s19 // (op 1296) 
v_add_co_u32 v18, s[0:1], v18, v20 // (op 1296) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1296) most significant half





v_mul_lo_u32 v21, v19, 6 // (op 1296) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 1296) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 1296) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1296) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1296) 

v_mov_b32 v17, v18 // (op 1296) convert

s_mul_i32 s1, s29, 6 // (op 1298) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1298) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1298) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1298) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1298) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 1298) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1298) least significant
s_mul_hi_u32 s50, s30, s0 // (op 1298) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1298) most significant: sum
s_add_u32 s19, s19, s50 // (op 1298) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1298) 

s_mov_b32 s52, s32 // (op 1298) reg expression
s_mov_b32 s53, s33 // (op 1298) reg expression
s_mov_b32 s54, s0 // (op 1298) reg expression
s_mov_b32 s55, 131072 // (op 1298) call()

s_mul_hi_i32 s0, s16, s12 // (op 1341) 
v_and_b32 v18, -16, v1 // (op 1341) 
s_add_i32 s1, s0, s16 // (op 1341) 

v_sub_u32 v19, v1, v18 // (op 1341) 

s_ashr_i32 s0, s1, 31 // (op 1341) 
v_lshrrev_b32 v18, 4, v1 // (op 1341) 
s_cmp_eq_i32 s12, 0 // (op 1341) 
s_mov_b32 s18, scc // (op 1341) 
s_cmp_eq_i32 s18, 0 // (op 1341) 
s_cselect_b32 s19, 0, -1 // (op 1341) 

s_and_b32 s18, s15, 31 // (op 1341) 
s_lshl_b32 s49, 1, s18 // (op 1341) 
s_add_i32 s50, s49, s19 // (op 1341) 


s_and_b32 s19, s0, s50 // (op 1341) 


s_add_i32 s0, s1, s19 // (op 1341) 


s_ashr_i32 s1, s0, s18 // (op 1341) 


s_lshl_b32 s0, s15, 24 // (op 1341) 
s_ashr_i32 s18, s0, 31 // (op 1341) 

s_xor_b32 s0, s1, s18 // (op 1341) 

s_sub_i32 s1, s0, s18 // (op 1341) 


s_mul_hi_i32 s0, s1, s44 // (op 1341) 
s_add_i32 s18, s0, s1 // (op 1341) 

s_ashr_i32 s0, s18, 31 // (op 1341) 
s_cmp_eq_i32 s44, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s49, 0, -1 // (op 1341) 

s_and_b32 s19, s45, 31 // (op 1341) 
s_lshl_b32 s50, 1, s19 // (op 1341) 
s_add_i32 s51, s50, s49 // (op 1341) 


s_and_b32 s49, s0, s51 // (op 1341) 


s_add_i32 s0, s18, s49 // (op 1341) 


s_ashr_i32 s18, s0, s19 // (op 1341) 


s_lshl_b32 s0, s45, 24 // (op 1341) 
s_ashr_i32 s19, s0, 31 // (op 1341) 

s_xor_b32 s0, s18, s19 // (op 1341) 

s_sub_i32 s18, s0, s19 // (op 1341) 


s_mul_i32 s0, s1, s13 // (op 1341) 

s_sub_u32 s1, s16, s0 // (op 1341) 

s_mul_i32 s0, s18, s13 // (op 1341) 
s_add_u32 s0, s0, s1 // (op 1341) 


s_mul_hi_i32 s1, s16, s11 // (op 1341) 
s_add_i32 s18, s1, s16 // (op 1341) 

s_ashr_i32 s1, s18, 31 // (op 1341) 
s_cmp_eq_i32 s11, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s49, 0, -1 // (op 1341) 

s_and_b32 s19, s47, 31 // (op 1341) 
s_lshl_b32 s50, 1, s19 // (op 1341) 
s_add_i32 s51, s50, s49 // (op 1341) 


s_and_b32 s49, s1, s51 // (op 1341) 


s_add_i32 s1, s18, s49 // (op 1341) 


s_ashr_i32 s18, s1, s19 // (op 1341) 


s_lshl_b32 s1, s47, 24 // (op 1341) 
s_ashr_i32 s19, s1, 31 // (op 1341) 

s_xor_b32 s1, s18, s19 // (op 1341) 

s_sub_i32 s18, s1, s19 // (op 1341) 


s_add_u32 s50, s8, 64 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1341) 

s_mul_i32 s1, s13, s50 // (op 1341) 

s_mul_i32 s19, s10, s1 // (op 1341) 

s_mul_i32 s1, s18, s19 // (op 1341) 

s_sub_u32 s19, s16, s1 // (op 1341) 

s_mov_b32 s1, s19 // (op 1341) 
s_mul_hi_i32 s49, s1, s14 // (op 1341) 
s_add_i32 s50, s49, s1 // (op 1341) 


s_ashr_i32 s1, s50, 31 // (op 1341) 
s_cmp_eq_i32 s14, 0 // (op 1341) 
s_mov_b32 s49, scc // (op 1341) 
s_cmp_eq_i32 s49, 0 // (op 1341) 
s_cselect_b32 s51, 0, -1 // (op 1341) 

s_and_b32 s49, s46, 31 // (op 1341) 
s_lshl_b32 s64, 1, s49 // (op 1341) 
s_add_i32 s65, s64, s51 // (op 1341) 


s_and_b32 s51, s1, s65 // (op 1341) 


s_add_i32 s1, s50, s51 // (op 1341) 


s_ashr_i32 s50, s1, s49 // (op 1341) 


s_lshl_b32 s1, s46, 24 // (op 1341) 
s_ashr_i32 s49, s1, 31 // (op 1341) 

s_xor_b32 s1, s50, s49 // (op 1341) 

s_sub_i32 s50, s1, s49 // (op 1341) 


s_mul_i32 s1, s48, s13 // (op 1341) 
s_add_u32 s64, s6, 16 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1341) 

s_sub_i32 s49, s64, s1 // (op 1341) 

s_mul_i32 s51, s50, s49 // (op 1341) 


s_sub_u32 s49, s19, s51 // (op 1341) 


s_add_u32 s19, s49, s1 // (op 1341) 


s_cmp_eq_i32 s18, 0 // (op 1341) 
s_mov_b32 s1, scc // (op 1341) 

s_cmp_eq_i32 s1, 0 // (op 1341) 
s_cselect_b32 s18, s19, s0 // (op 1341) 



v_lshl_add_u32 v20, s18, 4, v19 // (op 1341) 


v_mul_lo_u32 v19, v18, s34 // (op 1341) 

v_add_u32 v18, v20, v19 // (op 1341) 


s_mul_i32 s1, s34, s39 // (op 1343) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1343) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1343) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1343) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1343) most significant: sum
s_add_u32 s1, s1, s19 // (op 1343) most significant: sum


s_mov_b32 s64, s36 // (op 1343) reg expression
s_mov_b32 s65, s37 // (op 1343) reg expression
s_mov_b32 s66, s0 // (op 1343) reg expression
s_mov_b32 s67, 131072 // (op 1343) call()

s_mul_hi_i32 s0, s16, s12 // (op 1360) 
v_and_b32 v19, -16, v1 // (op 1360) 
s_add_i32 s1, s0, s16 // (op 1360) 

v_sub_u32 v20, v1, v19 // (op 1360) 

s_ashr_i32 s0, s1, 31 // (op 1360) 
v_lshrrev_b32 v19, 4, v1 // (op 1360) 
s_cmp_eq_i32 s12, 0 // (op 1360) 
s_mov_b32 s18, scc // (op 1360) 
s_cmp_eq_i32 s18, 0 // (op 1360) 
s_cselect_b32 s19, 0, -1 // (op 1360) 

s_and_b32 s18, s15, 31 // (op 1360) 
s_lshl_b32 s49, 1, s18 // (op 1360) 
s_add_i32 s50, s49, s19 // (op 1360) 


s_and_b32 s19, s0, s50 // (op 1360) 


s_add_i32 s0, s1, s19 // (op 1360) 


s_ashr_i32 s1, s0, s18 // (op 1360) 


s_lshl_b32 s0, s15, 24 // (op 1360) 
s_ashr_i32 s18, s0, 31 // (op 1360) 

s_xor_b32 s0, s1, s18 // (op 1360) 

s_sub_i32 s1, s0, s18 // (op 1360) 


s_mul_hi_i32 s0, s1, s44 // (op 1360) 
s_add_i32 s18, s0, s1 // (op 1360) 

s_ashr_i32 s0, s18, 31 // (op 1360) 
s_cmp_eq_i32 s44, 0 // (op 1360) 
s_mov_b32 s19, scc // (op 1360) 
s_cmp_eq_i32 s19, 0 // (op 1360) 
s_cselect_b32 s49, 0, -1 // (op 1360) 

s_and_b32 s19, s45, 31 // (op 1360) 
s_lshl_b32 s50, 1, s19 // (op 1360) 
s_add_i32 s51, s50, s49 // (op 1360) 


s_and_b32 s49, s0, s51 // (op 1360) 


s_add_i32 s0, s18, s49 // (op 1360) 


s_ashr_i32 s18, s0, s19 // (op 1360) 


s_lshl_b32 s0, s45, 24 // (op 1360) 
s_ashr_i32 s19, s0, 31 // (op 1360) 

s_xor_b32 s0, s18, s19 // (op 1360) 

s_sub_i32 s18, s0, s19 // (op 1360) 


s_mul_i32 s0, s1, s13 // (op 1360) 

s_sub_u32 s1, s16, s0 // (op 1360) 

s_mul_i32 s0, s18, s13 // (op 1360) 
s_add_u32 s0, s0, s1 // (op 1360) 


s_mul_hi_i32 s1, s16, s11 // (op 1360) 
s_add_i32 s18, s1, s16 // (op 1360) 

s_ashr_i32 s1, s18, 31 // (op 1360) 
s_cmp_eq_i32 s11, 0 // (op 1360) 
s_mov_b32 s19, scc // (op 1360) 
s_cmp_eq_i32 s19, 0 // (op 1360) 
s_cselect_b32 s49, 0, -1 // (op 1360) 

s_and_b32 s19, s47, 31 // (op 1360) 
s_lshl_b32 s50, 1, s19 // (op 1360) 
s_add_i32 s51, s50, s49 // (op 1360) 


s_and_b32 s49, s1, s51 // (op 1360) 


s_add_i32 s1, s18, s49 // (op 1360) 


s_ashr_i32 s18, s1, s19 // (op 1360) 


s_lshl_b32 s1, s47, 24 // (op 1360) 
s_ashr_i32 s19, s1, 31 // (op 1360) 

s_xor_b32 s1, s18, s19 // (op 1360) 

s_sub_i32 s18, s1, s19 // (op 1360) 


s_add_u32 s50, s8, 64 // (op 1360) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1360) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1360) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1360) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1360) 

s_mul_i32 s1, s13, s50 // (op 1360) 

s_mul_i32 s19, s10, s1 // (op 1360) 

s_mul_i32 s1, s18, s19 // (op 1360) 

s_sub_u32 s19, s16, s1 // (op 1360) 

s_mov_b32 s1, s19 // (op 1360) 
s_mul_hi_i32 s49, s1, s14 // (op 1360) 
s_add_i32 s50, s49, s1 // (op 1360) 


s_ashr_i32 s1, s50, 31 // (op 1360) 
s_cmp_eq_i32 s14, 0 // (op 1360) 
s_mov_b32 s49, scc // (op 1360) 
s_cmp_eq_i32 s49, 0 // (op 1360) 
s_cselect_b32 s51, 0, -1 // (op 1360) 

s_and_b32 s49, s46, 31 // (op 1360) 
s_lshl_b32 s68, 1, s49 // (op 1360) 
s_add_i32 s69, s68, s51 // (op 1360) 


s_and_b32 s51, s1, s69 // (op 1360) 


s_add_i32 s1, s50, s51 // (op 1360) 


s_ashr_i32 s50, s1, s49 // (op 1360) 


s_lshl_b32 s1, s46, 24 // (op 1360) 
s_ashr_i32 s49, s1, 31 // (op 1360) 

s_xor_b32 s1, s50, s49 // (op 1360) 

s_sub_i32 s50, s1, s49 // (op 1360) 


s_mul_i32 s1, s48, s13 // (op 1360) 
s_add_u32 s68, s6, 16 // (op 1360) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 1360) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1360) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1360) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1360) 

s_sub_i32 s49, s68, s1 // (op 1360) 

s_mul_i32 s51, s50, s49 // (op 1360) 


s_sub_u32 s49, s19, s51 // (op 1360) 


s_add_u32 s19, s49, s1 // (op 1360) 


s_cmp_eq_i32 s18, 0 // (op 1360) 
s_mov_b32 s1, scc // (op 1360) 

s_cmp_eq_i32 s1, 0 // (op 1360) 
s_cselect_b32 s18, s19, s0 // (op 1360) 



v_lshl_add_u32 v21, s18, 4, v20 // (op 1360) 


v_mul_lo_u32 v20, v19, s34 // (op 1360) 

v_add_u32 v19, v21, v20 // (op 1360) 


s_mul_i32 s1, s34, s39 // (op 1362) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1362) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1362) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1362) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1362) most significant: sum
s_add_u32 s1, s1, s19 // (op 1362) most significant: sum


s_mov_b32 s64, s36 // (op 1362) reg expression
s_mov_b32 s65, s37 // (op 1362) reg expression
s_mov_b32 s66, s0 // (op 1362) reg expression
s_mov_b32 s67, 131072 // (op 1362) call()

s_mul_hi_i32 s0, s16, s12 // (op 1405) 
v_and_b32 v20, -16, v1 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

v_sub_u32 v21, v1, v20 // (op 1405) 

s_ashr_i32 s0, s1, 31 // (op 1405) 
v_lshlrev_b32 v20, 2, v21 // (op 1405) 

s_cmp_eq_i32 s12, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
v_lshrrev_b32 v21, 4, v1 // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s15, 31 // (op 1405) 
s_lshl_b32 s49, 1, s18 // (op 1405) 
s_add_i32 s50, s49, s19 // (op 1405) 


s_and_b32 s19, s0, s50 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s15, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_hi_i32 s0, s1, s44 // (op 1405) 
s_add_i32 s18, s0, s1 // (op 1405) 

s_ashr_i32 s0, s18, 31 // (op 1405) 
s_cmp_eq_i32 s44, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s49, 0, -1 // (op 1405) 

s_and_b32 s19, s45, 31 // (op 1405) 
s_lshl_b32 s50, 1, s19 // (op 1405) 
s_add_i32 s51, s50, s49 // (op 1405) 


s_and_b32 s49, s0, s51 // (op 1405) 


s_add_i32 s0, s18, s49 // (op 1405) 


s_ashr_i32 s18, s0, s19 // (op 1405) 


s_lshl_b32 s0, s45, 24 // (op 1405) 
s_ashr_i32 s19, s0, 31 // (op 1405) 

s_xor_b32 s0, s18, s19 // (op 1405) 

s_sub_i32 s18, s0, s19 // (op 1405) 


s_add_u32 s50, s8, 64 // (op 1405) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1405) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1405) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1405) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1405) 

s_mul_i32 s0, s18, s50 // (op 1405) 

s_sub_i32 s49, s1, s0 // (op 1405) 


s_mul_hi_i32 s0, s16, s11 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

s_ashr_i32 s0, s1, 31 // (op 1405) 
s_cmp_eq_i32 s11, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s47, 31 // (op 1405) 
s_lshl_b32 s68, 1, s18 // (op 1405) 
s_add_i32 s69, s68, s19 // (op 1405) 


s_and_b32 s19, s0, s69 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s47, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_i32 s0, s13, s50 // (op 1405) 

s_mul_i32 s18, s10, s0 // (op 1405) 

s_mul_i32 s0, s1, s18 // (op 1405) 

s_sub_u32 s18, s16, s0 // (op 1405) 

s_mul_hi_i32 s0, s18, s14 // (op 1405) 
s_add_i32 s19, s0, s18 // (op 1405) 


s_ashr_i32 s0, s19, 31 // (op 1405) 
s_cmp_eq_i32 s14, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s50, 0, -1 // (op 1405) 

s_and_b32 s18, s46, 31 // (op 1405) 
s_lshl_b32 s51, 1, s18 // (op 1405) 
s_add_i32 s68, s51, s50 // (op 1405) 


s_and_b32 s50, s0, s68 // (op 1405) 


s_add_i32 s0, s19, s50 // (op 1405) 


s_ashr_i32 s19, s0, s18 // (op 1405) 


s_lshl_b32 s0, s46, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s19, s18 // (op 1405) 

s_sub_i32 s19, s0, s18 // (op 1405) 


s_cmp_eq_i32 s1, 0 // (op 1405) 
s_mov_b32 s0, scc // (op 1405) 

s_cmp_eq_i32 s0, 0 // (op 1405) 
s_cselect_b32 s1, s19, s49 // (op 1405) 



v_lshl_add_u32 v22, s1, 6, v20 // (op 1405) 


v_mul_lo_u32 v20, v21, s40 // (op 1405) 
v_add_u32 v20, v20, v22 // (op 1405) 


s_mul_i32 s1, s40, s39 // (op 1407) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1407) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1407) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1407) most significant: sum
s_add_u32 s1, s1, s19 // (op 1407) most significant: sum


s_mov_b32 s68, s42 // (op 1407) reg expression
s_mov_b32 s69, s43 // (op 1407) reg expression
s_mov_b32 s70, s0 // (op 1407) reg expression
s_mov_b32 s71, 131072 // (op 1407) call()

s_mul_hi_i32 s0, s16, s12 // (op 1424) 
v_and_b32 v21, -16, v1 // (op 1424) 
s_add_i32 s1, s0, s16 // (op 1424) 

v_sub_u32 v22, v1, v21 // (op 1424) 

s_ashr_i32 s0, s1, 31 // (op 1424) 
v_lshlrev_b32 v21, 2, v22 // (op 1424) 

s_cmp_eq_i32 s12, 0 // (op 1424) 
s_mov_b32 s18, scc // (op 1424) 
v_lshrrev_b32 v22, 4, v1 // (op 1424) 
s_cmp_eq_i32 s18, 0 // (op 1424) 
s_cselect_b32 s19, 0, -1 // (op 1424) 

s_and_b32 s18, s15, 31 // (op 1424) 
s_lshl_b32 s49, 1, s18 // (op 1424) 
s_add_i32 s50, s49, s19 // (op 1424) 


s_and_b32 s19, s0, s50 // (op 1424) 


s_add_i32 s0, s1, s19 // (op 1424) 


s_ashr_i32 s1, s0, s18 // (op 1424) 


s_lshl_b32 s0, s15, 24 // (op 1424) 
s_ashr_i32 s18, s0, 31 // (op 1424) 

s_xor_b32 s0, s1, s18 // (op 1424) 

s_sub_i32 s1, s0, s18 // (op 1424) 


s_mul_hi_i32 s0, s1, s44 // (op 1424) 
s_add_i32 s18, s0, s1 // (op 1424) 

s_ashr_i32 s0, s18, 31 // (op 1424) 
s_cmp_eq_i32 s44, 0 // (op 1424) 
s_mov_b32 s19, scc // (op 1424) 
s_cmp_eq_i32 s19, 0 // (op 1424) 
s_cselect_b32 s49, 0, -1 // (op 1424) 

s_and_b32 s19, s45, 31 // (op 1424) 
s_lshl_b32 s50, 1, s19 // (op 1424) 
s_add_i32 s51, s50, s49 // (op 1424) 


s_and_b32 s49, s0, s51 // (op 1424) 


s_add_i32 s0, s18, s49 // (op 1424) 


s_ashr_i32 s18, s0, s19 // (op 1424) 


s_lshl_b32 s0, s45, 24 // (op 1424) 
s_ashr_i32 s19, s0, 31 // (op 1424) 

s_xor_b32 s0, s18, s19 // (op 1424) 

s_sub_i32 s18, s0, s19 // (op 1424) 


s_add_u32 s50, s8, 64 // (op 1424) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1424) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1424) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1424) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1424) 

s_mul_i32 s0, s18, s50 // (op 1424) 

s_sub_i32 s49, s1, s0 // (op 1424) 


s_mul_hi_i32 s0, s16, s11 // (op 1424) 
s_add_i32 s1, s0, s16 // (op 1424) 

s_ashr_i32 s0, s1, 31 // (op 1424) 
s_cmp_eq_i32 s11, 0 // (op 1424) 
s_mov_b32 s18, scc // (op 1424) 
s_cmp_eq_i32 s18, 0 // (op 1424) 
s_cselect_b32 s19, 0, -1 // (op 1424) 

s_and_b32 s18, s47, 31 // (op 1424) 
s_lshl_b32 s72, 1, s18 // (op 1424) 
s_add_i32 s73, s72, s19 // (op 1424) 


s_and_b32 s19, s0, s73 // (op 1424) 


s_add_i32 s0, s1, s19 // (op 1424) 


s_ashr_i32 s1, s0, s18 // (op 1424) 


s_lshl_b32 s0, s47, 24 // (op 1424) 
s_ashr_i32 s18, s0, 31 // (op 1424) 

s_xor_b32 s0, s1, s18 // (op 1424) 

s_sub_i32 s1, s0, s18 // (op 1424) 


s_mul_i32 s0, s13, s50 // (op 1424) 

s_mul_i32 s18, s10, s0 // (op 1424) 

s_mul_i32 s0, s1, s18 // (op 1424) 

s_sub_u32 s18, s16, s0 // (op 1424) 

s_mul_hi_i32 s0, s18, s14 // (op 1424) 
s_add_i32 s19, s0, s18 // (op 1424) 


s_ashr_i32 s0, s19, 31 // (op 1424) 
s_cmp_eq_i32 s14, 0 // (op 1424) 
s_mov_b32 s18, scc // (op 1424) 
s_cmp_eq_i32 s18, 0 // (op 1424) 
s_cselect_b32 s50, 0, -1 // (op 1424) 

s_and_b32 s18, s46, 31 // (op 1424) 
s_lshl_b32 s51, 1, s18 // (op 1424) 
s_add_i32 s72, s51, s50 // (op 1424) 


s_and_b32 s50, s0, s72 // (op 1424) 


s_add_i32 s0, s19, s50 // (op 1424) 


s_ashr_i32 s19, s0, s18 // (op 1424) 


s_lshl_b32 s0, s46, 24 // (op 1424) 
s_ashr_i32 s18, s0, 31 // (op 1424) 

s_xor_b32 s0, s19, s18 // (op 1424) 

s_sub_i32 s19, s0, s18 // (op 1424) 


s_cmp_eq_i32 s1, 0 // (op 1424) 
s_mov_b32 s0, scc // (op 1424) 

s_cmp_eq_i32 s0, 0 // (op 1424) 
s_cselect_b32 s1, s19, s49 // (op 1424) 



v_lshl_add_u32 v23, s1, 6, v21 // (op 1424) 


v_mul_lo_u32 v21, v22, s40 // (op 1424) 
v_add_u32 v21, v21, v23 // (op 1424) 


s_mul_i32 s1, s40, s39 // (op 1426) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1426) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1426) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1426) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1426) most significant: sum
s_add_u32 s1, s1, s19 // (op 1426) most significant: sum


s_mov_b32 s68, s42 // (op 1426) reg expression
s_mov_b32 s69, s43 // (op 1426) reg expression
s_mov_b32 s70, s0 // (op 1426) reg expression
s_mov_b32 s71, 131072 // (op 1426) call()


s_mul_i32 s1, s31, 6 // (op 1037) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 1037) least significant
s_mul_hi_u32 s18, s30, 6 // (op 1037) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1037) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1037) 


v_mov_b32 v24, v16 // (op 1037) reg expression
buffer_load_dwordx3 v[26:28], v24, s[52:55], 0 offen // (op 1037) Load value
buffer_load_dwordx3 v[30:32], v24, s[52:55], 0 offen offset:12 // (op 1037) Load value
buffer_load_dwordx3 v[34:36], v24, s[52:55], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx3 v[38:40], v24, s[52:55], 0 offen offset:36 // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx3 v[42:44], v24, s[52:55], 0 offen // (op 1037) Load value
buffer_load_dwordx3 v[46:48], v24, s[52:55], 0 offen offset:12 // (op 1037) Load value
buffer_load_dwordx3 v[50:52], v24, s[52:55], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx3 v[54:56], v24, s[52:55], 0 offen offset:36 // (op 1037) Load value




v_mov_b32 v29, v20 // (op 1040) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 1040) Load value


s_mul_i32 s1, s25, 6 // (op 1043) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1043) least significant
s_mul_hi_u32 s18, s24, 6 // (op 1043) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1043) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1043) 


v_mov_b32 v29, v14 // (op 1043) reg expression
buffer_load_dwordx3 v[58:60], v29, s[20:23], 0 offen // (op 1043) Load value
v_add_u32 v29, v29, v22 // (op 1043) 
buffer_load_dwordx3 v[62:64], v29, s[20:23], 0 offen // (op 1043) Load value




v_mov_b32 v29, v18 // (op 1046) reg expression

buffer_load_ubyte v37, v29, s[64:67], 0 offen offset: 0 // (op 1046) Load value


s_mul_i32 s1, s31, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 1049) least significant
s_mul_hi_u32 s18, s30, 6 // (op 1049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1049) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1049) 


s_lshl_b64 s[0:1], s[30:31], 8 // (op 1049) 
s_mul_i32 s19, s1, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 1049) least significant
s_mul_hi_u32 s49, s0, 6 // (op 1049) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1049) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 1049) 

v_add_u32 v29, v16, s0 // (op 1049) 

buffer_load_dwordx3 v[66:68], v29, s[52:55], 0 offen // (op 1049) Load value
buffer_load_dwordx3 v[70:72], v29, s[52:55], 0 offen offset:12 // (op 1049) Load value
buffer_load_dwordx3 v[74:76], v29, s[52:55], 0 offen offset:24 // (op 1049) Load value
buffer_load_dwordx3 v[78:80], v29, s[52:55], 0 offen offset:36 // (op 1049) Load value
v_add_u32 v29, v29, v22 // (op 1049) 
buffer_load_dwordx3 v[82:84], v29, s[52:55], 0 offen // (op 1049) Load value
buffer_load_dwordx3 v[86:88], v29, s[52:55], 0 offen offset:12 // (op 1049) Load value
buffer_load_dwordx3 v[90:92], v29, s[52:55], 0 offen offset:24 // (op 1049) Load value
buffer_load_dwordx3 v[94:96], v29, s[52:55], 0 offen offset:36 // (op 1049) Load value





s_lshl_b32 s0, s40, 3 // (op 1052) 
v_add_u32 v16, v20, s0 // (op 1052) 


buffer_load_dword v29, v16, s[68:71], 0 offen // (op 1052) Load value



s_mul_i32 s1, s25, 6 // (op 1055) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1055) least significant
s_mul_hi_u32 s18, s24, 6 // (op 1055) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1055) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1055) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1055) 
s_mul_i32 s19, s1, 6 // (op 1055) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 1055) least significant
s_mul_hi_u32 s49, s0, 6 // (op 1055) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1055) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 1055) 

v_add_u32 v16, v14, s0 // (op 1055) 

buffer_load_dwordx3 v[98:100], v16, s[20:23], 0 offen // (op 1055) Load value
v_add_u32 v16, v16, v22 // (op 1055) 
buffer_load_dwordx3 v[102:104], v16, s[20:23], 0 offen // (op 1055) Load value





s_lshl_b32 s0, s34, 3 // (op 1058) 
v_add_u32 v14, v18, s0 // (op 1058) 


buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 1058) Load value


v_lshlrev_b32 v14, 7, v1 // (op 1544) 
v_mul_lo_u32 v18, v14, 6 // (op 1544) 
v_lshrrev_b32 v20, 3, v18 // (op 1544) 

v_lshrrev_b32 v18, 4, v14 // (op 1544) 

v_lshlrev_b32 v14, 2, v18 // (op 1544) 

v_add_u32 v18, v20, v14 // (op 1544) 





s_waitcnt vmcnt(23)
ds_write_b96 v18, v[26:28] // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v18, v[30:32] offset:16 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v18, v[34:36] offset:32 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v18, v[38:40] offset:48 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v18, v[42:44] offset:64 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v18, v[46:48] offset:80 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v18, v[50:52] offset:96 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v18, v[54:56] offset:112 // (op 1060) Store local data 

v_and_b32 v14, -16, v1 // (op 1655) 
v_sub_u32 v18, v1, v14 // (op 1655) 

v_lshlrev_b32 v14, 2, v18 // (op 1655) 

v_lshrrev_b32 v18, 4, v1 // (op 1655) 
v_lshl_add_u32 v20, v18, 6, v14 // (op 1655) 





s_waitcnt vmcnt(15)
ds_write_b32 v20, v33 offset:16384 // (op 1061) Store local data 

v_lshlrev_b32 v14, 5, v1 // (op 1491) 
v_mul_lo_u32 v18, v14, 6 // (op 1491) 
v_lshrrev_b32 v20, 3, v18 // (op 1491) 

v_lshrrev_b32 v18, 4, v14 // (op 1491) 

v_lshlrev_b32 v14, 2, v18 // (op 1491) 

v_add_u32 v18, v20, v14 // (op 1491) 





s_waitcnt vmcnt(14)
ds_write_b96 v18, v[58:60] offset:16896 // (op 1062) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v18, v[62:64] offset:16912 // (op 1062) Store local data 

v_and_b32 v14, -16, v1 // (op 1602) 
v_sub_u32 v18, v1, v14 // (op 1602) 

v_and_b32 v14, 4294967280, v1 // (op 1602) 
v_add_u32 v20, v14, v18 // (op 1602) 





s_waitcnt vmcnt(12)
ds_write_b8 v20, v37 offset:20992 // (op 1063) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1866) 



ds_read_b96_tr_b6 v[106:108], v5 // (op 1081) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:1024 // (op 1081) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1907) 


ds_read_u8 v14, v9 offset:16384 // (op 1094) Load local data 



ds_read_b96_tr_b6 v[112:114], v5 offset:32 // (op 1086) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:1056 // (op 1086) Transpose load local data 


ds_read_u8 v18, v9 offset:16416 // (op 1099) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1847) 



ds_read_b96_tr_b6 v[118:120], v0 offset:16896 // (op 1077) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:17152 // (op 1077) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1887) 


ds_read_u8 v20, v4 offset:20992 // (op 1090) Load local data 
v_lshlrev_b32 v41, 5, v1 // (op 1462) 
v_mul_lo_u32 v45, v41, 6 // (op 1462) 
v_lshrrev_b32 v49, 3, v45 // (op 1462) 

v_lshrrev_b32 v45, 4, v41 // (op 1462) 

v_lshlrev_b32 v41, 2, v45 // (op 1462) 

v_add_u32 v45, v49, v41 // (op 1462) 


v_lshlrev_b32 v41, 7, v1 // (op 1515) 
v_mul_lo_u32 v49, v41, 6 // (op 1515) 
v_lshrrev_b32 v53, 3, v49 // (op 1515) 

v_lshrrev_b32 v49, 4, v41 // (op 1515) 

v_lshlrev_b32 v41, 2, v49 // (op 1515) 

v_add_u32 v49, v53, v41 // (op 1515) 


v_and_b32 v41, -16, v1 // (op 1573) 
v_sub_u32 v53, v1, v41 // (op 1573) 

v_and_b32 v41, 4294967280, v1 // (op 1573) 
v_add_u32 v57, v41, v53 // (op 1573) 


v_and_b32 v41, -16, v1 // (op 1626) 
v_sub_u32 v53, v1, v41 // (op 1626) 

v_lshlrev_b32 v41, 2, v53 // (op 1626) 

v_lshrrev_b32 v53, 4, v1 // (op 1626) 
v_lshl_add_u32 v61, v53, 6, v41 // (op 1626) 


v_lshlrev_b32 v41, 5, v1 // (op 1702) 
v_mul_lo_u32 v53, v41, 6 // (op 1702) 
v_lshrrev_b32 v65, 3, v53 // (op 1702) 

v_lshrrev_b32 v53, 4, v41 // (op 1702) 

v_lshlrev_b32 v41, 2, v53 // (op 1702) 

v_add_u32 v53, v65, v41 // (op 1702) 


v_lshlrev_b32 v41, 7, v1 // (op 1726) 
v_mul_lo_u32 v65, v41, 6 // (op 1726) 
v_lshrrev_b32 v69, 3, v65 // (op 1726) 

v_lshrrev_b32 v65, 4, v41 // (op 1726) 

v_lshlrev_b32 v41, 2, v65 // (op 1726) 

v_add_u32 v65, v69, v41 // (op 1726) 


v_and_b32 v41, -16, v1 // (op 1755) 
v_sub_u32 v69, v1, v41 // (op 1755) 

v_and_b32 v41, 4294967280, v1 // (op 1755) 
v_add_u32 v73, v41, v69 // (op 1755) 


v_and_b32 v41, -16, v1 // (op 1779) 
v_sub_u32 v69, v1, v41 // (op 1779) 

v_lshlrev_b32 v41, 2, v69 // (op 1779) 

v_lshrrev_b32 v69, 4, v1 // (op 1779) 
v_lshl_add_u32 v77, v69, 6, v41 // (op 1779) 


s_mov_b32 s0, 0 // (op 404) call()
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 if false)
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403:

s_add_u32 s0, s0, 2 // (op 405) 
s_mul_i32 s19, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 10) least significant
s_mul_hi_u32 s1, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 10) 


v_mov_b32 v41, v17 // (op 10) 
s_lshl_b64 s[18:19], s[30:31], 8 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v69, v41, s1 // (op 10) 

buffer_load_dwordx3 v[26:28], v69, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v69, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v69, s[52:55], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[118:123], v[106:111], a[0:3], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 90) 
buffer_load_dwordx3 v[38:40], v69, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1856) 
v_add_u32 v69, v69, v22 // (op 10) 



buffer_load_dwordx3 v[42:44], v69, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[46:48], v69, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b96_tr_b6 v[124:126], v0 offset:18944 // (op 93) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v0 offset:19200 // (op 93) Transpose load local data 
buffer_load_dwordx3 v[50:52], v69, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1876) 
buffer_load_dwordx3 v[54:56], v69, s[52:55], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[106:108], v5 offset:8192 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:9216 // (op 97) Transpose load local data 
v_mov_b32 v69, v21 // (op 25) 
s_barrier  // (op 1897) 
s_lshl_b32 s1, s40, 4 // (op 25) 


ds_read_u8 v81, v4 offset:21056 // (op 101) Load local data 
v_add_u32 v85, v69, s1 // (op 25) 


s_barrier  // (op 1917) 
buffer_load_dword v33, v85, s[68:71], 0 offen // (op 25) Load value




s_mul_i32 s19, s25, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s24, 6 // (op 4) least significant
s_mul_hi_u32 s1, s24, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 4) 


v_mov_b32 v85, v15 // (op 4) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 4) 
s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v89, v85, s1 // (op 4) 

ds_read_u8 v14, v9 offset:16640 // (op 105) Load local data 
buffer_load_dwordx3 v[58:60], v89, s[20:23], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v22 // (op 4) 
buffer_load_dwordx3 v[62:64], v89, s[20:23], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[106:111], a[0:3], v81, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 107) 


v_mov_b32 v89, v19 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[112:117], a[4:7], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 451) 
s_lshl_b32 s1, s34, 4 // (op 17) 



ds_read_b96_tr_b6 v[112:114], v5 offset:8224 // (op 455) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:9248 // (op 455) Transpose load local data 
v_add_u32 v93, v89, s1 // (op 17) 




ds_read_u8 v18, v9 offset:16672 // (op 459) Load local data 
buffer_load_ubyte v37, v93, s[64:67], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1937) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:129], v[112:117], a[4:7], v81, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 460) 





s_waitcnt vmcnt(23)
ds_write_b96 v65, v[66:68] offset:21120 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v65, v[70:72] offset:21136 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v65, v[74:76] offset:21152 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v65, v[78:80] offset:21168 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v65, v[82:84] offset:21184 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v65, v[86:88] offset:21200 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v65, v[90:92] offset:21216 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v65, v[94:96] offset:21232 // (op 306) Store local data 
s_barrier  // (op 1949) 



s_waitcnt vmcnt(15)
ds_write_b32 v77, v29 offset:37504 // (op 308) Store local data 
s_barrier  // (op 1930) 



s_waitcnt vmcnt(14)
ds_write_b96 v53, v[98:100] offset:38016 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v53, v[102:104] offset:38032 // (op 287) Store local data 
s_barrier  // (op 1943) 



s_waitcnt vmcnt(12)
ds_write_b8 v73, v16 offset:42112 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1933) 



ds_read_b96_tr_b6 v[106:108], v12 offset:21120 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v12 offset:22144 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1946) 


ds_read_u8 v14, v13 offset:37504 // (op 295) Load local data 



ds_read_b96_tr_b6 v[124:126], v12 offset:21152 // (op 474) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v12 offset:22176 // (op 474) Transpose load local data 


ds_read_u8 v81, v13 offset:37536 // (op 478) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1927) 



ds_read_b96_tr_b6 v[118:120], v3 offset:38016 // (op 289) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v3 offset:38272 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1940) 


ds_read_u8 v20, v6 offset:42112 // (op 293) Load local data 

s_mul_i32 s19, s31, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 309) least significant
s_mul_hi_u32 s1, s30, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 309) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 309) 


s_lshl_b64 s[18:19], s[30:31], 8 // (op 309) 
s_mul_i32 s51, s19, 6 // (op 309) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 309) least significant
s_mul_hi_u32 s1, s18, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 309) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 309) 

s_mul_i32 s1, 3, s18 // (op 309) 

v_add_u32 v93, v41, s1 // (op 309) 

buffer_load_dwordx3 v[66:68], v93, s[52:55], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[70:72], v93, s[52:55], 0 offen offset:12 // (op 309) Load value
buffer_load_dwordx3 v[74:76], v93, s[52:55], 0 offen offset:24 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[118:123], v[106:111], a[0:3], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 296) 
buffer_load_dwordx3 v[78:80], v93, s[52:55], 0 offen offset:36 // (op 309) Load value
v_add_u32 v93, v93, v22 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[124:129], a[4:7], v20, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 461) 
buffer_load_dwordx3 v[82:84], v93, s[52:55], 0 offen // (op 309) Load value


buffer_load_dwordx3 v[86:88], v93, s[52:55], 0 offen offset:12 // (op 309) Load value
ds_read_b96_tr_b6 v[124:126], v3 offset:40064 // (op 298) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v3 offset:40320 // (op 298) Transpose load local data 
buffer_load_dwordx3 v[90:92], v93, s[52:55], 0 offen offset:24 // (op 309) Load value
buffer_load_dwordx3 v[94:96], v93, s[52:55], 0 offen offset:36 // (op 309) Load value






s_barrier  // (op 1998) 


s_lshl_b32 s1, s40, 3 // (op 311) 
s_mul_i32 s18, 3, s1 // (op 311) 

v_add_u32 v41, v69, s18 // (op 311) 


buffer_load_dword v29, v41, s[68:71], 0 offen // (op 311) Load value

s_lshl_b64 s[18:19], s[30:31], 8 // (op 1288) 

s_mul_i32 s51, s19, 6 // (op 1288) most significant: low of high * low

s_mul_i32 s50, s18, 6 // (op 1288) least significant
s_mul_hi_u32 s1, s18, 6 // (op 1288) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 1288) most significant: sum


s_lshl_b32 s1, s40, 4 // (op 1416) 
v_add_u32 v21, v21, s1 // (op 1416) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 1288) 

s_lshl_b32 s1, s18, 1 // (op 1288) 

v_add_u32 v17, v17, s1 // (op 1288) 

s_mul_i32 s19, s25, 6 // (op 286) most significant: low of high * low
s_mul_i32 s18, s24, 6 // (op 286) least significant
s_mul_hi_u32 s1, s24, 6 // (op 286) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 286) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 286) 


s_lshl_b64 s[18:19], s[24:25], 8 // (op 286) 
s_mul_i32 s51, s19, 6 // (op 286) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 286) least significant
s_mul_hi_u32 s1, s18, 6 // (op 286) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 286) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 286) 

s_mul_i32 s1, 3, s18 // (op 286) 

v_add_u32 v41, v85, s1 // (op 286) 

ds_read_b96_tr_b6 v[130:132], v12 offset:29312 // (op 300) Transpose load local data 
ds_read_b96_tr_b6 v[133:135], v12 offset:30336 // (op 300) Transpose load local data 
buffer_load_dwordx3 v[98:100], v41, s[20:23], 0 offen // (op 286) Load value


v_add_u32 v41, v41, v22 // (op 286) 
ds_read_u8 v69, v6 offset:42176 // (op 302) Load local data 
buffer_load_dwordx3 v[102:104], v41, s[20:23], 0 offen // (op 286) Load value







s_barrier  // (op 2002) 
ds_read_u8 v41, v13 offset:37760 // (op 304) Load local data 
s_lshl_b32 s1, s34, 3 // (op 310) 



s_mul_i32 s18, 3, s1 // (op 310) 

v_add_u32 v81, v89, s18 // (op 310) 


buffer_load_ubyte v16, v81, s[64:67], 0 offen offset: 0 // (op 310) Load value

s_lshl_b64 s[18:19], s[24:25], 8 // (op 1224) 

s_mul_i32 s51, s19, 6 // (op 1224) most significant: low of high * low
s_barrier  // (op 1869) 
s_mul_i32 s50, s18, 6 // (op 1224) least significant



s_mul_hi_u32 s1, s18, 6 // (op 1224) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 1224) most significant: sum


s_lshl_b32 s1, s34, 4 // (op 1352) 
v_add_u32 v19, v19, s1 // (op 1352) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 1224) 

s_lshl_b32 s1, s18, 1 // (op 1224) 

v_add_u32 v15, v15, s1 // (op 1224) 

ds_read_b96_tr_b6 v[136:138], v12 offset:29344 // (op 465) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[130:135], a[0:3], v69, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 305) 
ds_read_b96_tr_b6 v[139:141], v12 offset:30368 // (op 465) Transpose load local data 


s_waitcnt vmcnt(23)
ds_write_b96 v49, v[26:28] // (op 13) Store local data 


s_barrier  // (op 2038) 
ds_read_u8 v41, v13 offset:37792 // (op 469) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v49, v[30:32] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v49, v[34:36] offset:32 // (op 13) Store local data 
s_barrier  // (op 2042) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:129], v[136:141], a[4:7], v69, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 470) 
s_waitcnt vmcnt(20)
ds_write_b96 v49, v[38:40] offset:48 // (op 13) Store local data 




s_waitcnt vmcnt(19)
ds_write_b96 v49, v[42:44] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v49, v[46:48] offset:80 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v49, v[50:52] offset:96 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v49, v[54:56] offset:112 // (op 13) Store local data 
s_barrier  // (op 1910) 



s_waitcnt vmcnt(15)
ds_write_b32 v61, v33 offset:16384 // (op 30) Store local data 
s_barrier  // (op 1850) 



s_waitcnt vmcnt(14)
ds_write_b96 v45, v[58:60] offset:16896 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v45, v[62:64] offset:16912 // (op 6) Store local data 
s_barrier  // (op 1890) 



s_waitcnt vmcnt(12)
ds_write_b8 v57, v37 offset:20992 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1879) 



ds_read_b96_tr_b6 v[106:108], v5 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:1024 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1920) 


ds_read_u8 v14, v9 offset:16384 // (op 87) Load local data 



ds_read_b96_tr_b6 v[112:114], v5 offset:32 // (op 446) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:1056 // (op 446) Transpose load local data 


ds_read_u8 v18, v9 offset:16416 // (op 450) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1859) 



ds_read_b96_tr_b6 v[118:120], v0 offset:16896 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:17152 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1900) 


ds_read_u8 v20, v4 offset:20992 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 if true)
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403:

















s_mul_hi_i32 s0, s16, s12 // (op 1186) 
v_lshlrev_b32 v15, 1, v1 // (op 1186) 
s_add_i32 s1, s0, s16 // (op 1186) 

s_ashr_i32 s0, s1, 31 // (op 1186) 
s_cmp_eq_i32 s12, 0 // (op 1186) 
s_mov_b32 s18, scc // (op 1186) 
s_cmp_eq_i32 s18, 0 // (op 1186) 
s_cselect_b32 s19, 0, -1 // (op 1186) 

s_and_b32 s18, s15, 31 // (op 1186) 
s_lshl_b32 s49, 1, s18 // (op 1186) 
s_add_i32 s50, s49, s19 // (op 1186) 


s_and_b32 s19, s0, s50 // (op 1186) 


s_add_i32 s0, s1, s19 // (op 1186) 


s_ashr_i32 s1, s0, s18 // (op 1186) 


s_lshl_b32 s0, s15, 24 // (op 1186) 
s_ashr_i32 s18, s0, 31 // (op 1186) 

s_xor_b32 s0, s1, s18 // (op 1186) 

s_sub_i32 s1, s0, s18 // (op 1186) 


s_mul_hi_i32 s0, s1, s44 // (op 1186) 
s_add_i32 s18, s0, s1 // (op 1186) 

s_ashr_i32 s0, s18, 31 // (op 1186) 
s_cmp_eq_i32 s44, 0 // (op 1186) 
s_mov_b32 s19, scc // (op 1186) 
s_cmp_eq_i32 s19, 0 // (op 1186) 
s_cselect_b32 s49, 0, -1 // (op 1186) 

s_and_b32 s19, s45, 31 // (op 1186) 
s_lshl_b32 s50, 1, s19 // (op 1186) 
s_add_i32 s51, s50, s49 // (op 1186) 


s_and_b32 s49, s0, s51 // (op 1186) 


s_add_i32 s0, s18, s49 // (op 1186) 


s_ashr_i32 s18, s0, s19 // (op 1186) 


s_lshl_b32 s0, s45, 24 // (op 1186) 
s_ashr_i32 s19, s0, 31 // (op 1186) 

s_xor_b32 s0, s18, s19 // (op 1186) 

s_sub_i32 s18, s0, s19 // (op 1186) 


s_mul_i32 s0, s1, s13 // (op 1186) 

s_sub_u32 s1, s16, s0 // (op 1186) 

s_mul_i32 s0, s18, s13 // (op 1186) 
s_add_u32 s0, s0, s1 // (op 1186) 


s_mul_hi_i32 s1, s16, s11 // (op 1186) 
s_add_i32 s18, s1, s16 // (op 1186) 

s_ashr_i32 s1, s18, 31 // (op 1186) 
s_cmp_eq_i32 s11, 0 // (op 1186) 
s_mov_b32 s19, scc // (op 1186) 
s_cmp_eq_i32 s19, 0 // (op 1186) 
s_cselect_b32 s49, 0, -1 // (op 1186) 

s_and_b32 s19, s47, 31 // (op 1186) 
s_lshl_b32 s50, 1, s19 // (op 1186) 
s_add_i32 s51, s50, s49 // (op 1186) 


s_and_b32 s49, s1, s51 // (op 1186) 


s_add_i32 s1, s18, s49 // (op 1186) 


s_ashr_i32 s18, s1, s19 // (op 1186) 


s_lshl_b32 s1, s47, 24 // (op 1186) 
s_ashr_i32 s19, s1, 31 // (op 1186) 

s_xor_b32 s1, s18, s19 // (op 1186) 

s_sub_i32 s18, s1, s19 // (op 1186) 


s_add_u32 s50, s8, 64 // (op 1186) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1186) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1186) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1186) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1186) 

s_mul_i32 s1, s13, s50 // (op 1186) 

s_mul_i32 s19, s10, s1 // (op 1186) 

s_mul_i32 s1, s18, s19 // (op 1186) 

s_sub_u32 s19, s16, s1 // (op 1186) 

s_mov_b32 s1, s19 // (op 1186) 
s_mul_hi_i32 s49, s1, s14 // (op 1186) 
s_add_i32 s50, s49, s1 // (op 1186) 


s_ashr_i32 s1, s50, 31 // (op 1186) 
s_cmp_eq_i32 s14, 0 // (op 1186) 
s_mov_b32 s49, scc // (op 1186) 
s_cmp_eq_i32 s49, 0 // (op 1186) 
s_cselect_b32 s51, 0, -1 // (op 1186) 

s_and_b32 s49, s46, 31 // (op 1186) 
s_lshl_b32 s72, 1, s49 // (op 1186) 
s_add_i32 s73, s72, s51 // (op 1186) 


s_and_b32 s51, s1, s73 // (op 1186) 


s_add_i32 s1, s50, s51 // (op 1186) 


s_ashr_i32 s50, s1, s49 // (op 1186) 


s_lshl_b32 s1, s46, 24 // (op 1186) 
s_ashr_i32 s49, s1, 31 // (op 1186) 

s_xor_b32 s1, s50, s49 // (op 1186) 

s_sub_i32 s50, s1, s49 // (op 1186) 


s_mul_i32 s1, s48, s13 // (op 1186) 
s_add_u32 s72, s6, 16 // (op 1186) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 1186) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1186) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1186) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1186) 

s_sub_i32 s49, s72, s1 // (op 1186) 

s_mul_i32 s51, s50, s49 // (op 1186) 


s_sub_u32 s49, s19, s51 // (op 1186) 


s_add_u32 s19, s49, s1 // (op 1186) 


s_cmp_eq_i32 s18, 0 // (op 1186) 
s_mov_b32 s1, scc // (op 1186) 

s_cmp_eq_i32 s1, 0 // (op 1186) 
s_cselect_b32 s18, s19, s0 // (op 1186) 



s_lshl_b32 s19, s18, 4 // (op 1186) 

s_mov_b32 s1, 0 // (op 1186) convert
s_mov_b32 s0, s19 // (op 1186) convert

s_ashr_i64 s[18:19], s[28:29], 8 // (op 1186) 
s_lshr_b32 s49, s18, 1 // (op 1186) 

v_lshl_add_u32 v17, s49, 9, v15 // (op 1186) 


v_mul_lo_u32 v31, v17, s25 // (op 1186) most significant: low of low * high
v_mul_hi_u32 v15, v17, s24 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v30, v17, s24 // (op 1186) least significant: low of low * low
v_add_u32 v31, v31, v15 // (op 1186) most significant: sum


v_mov_b32 v15, s0 // (op 1186) 
v_mov_b32 v17, s1 // (op 1186) 
v_add_co_u32 v32, s[18:19], v15, v30 // (op 1186) least significant half
v_addc_co_u32 v33, s[18:19], v17, v31, s[18:19] // (op 1186) most significant half





v_mul_lo_u32 v31, v33, 6 // (op 1186) most significant: low of high * low
v_mul_hi_u32 v15, v32, 6 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v30, v32, 6 // (op 1186) least significant: low of low * low
v_add_u32 v31, v31, v15 // (op 1186) most significant: sum


v_lshrrev_b64 v[32:33], 3, v[30:31] // (op 1186) 

v_mov_b32 v15, v32 // (op 1186) convert

s_mul_i32 s1, s29, 6 // (op 1188) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1188) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1188) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1188) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1188) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 1188) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1188) least significant
s_mul_hi_u32 s50, s24, s0 // (op 1188) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1188) most significant: sum
s_add_u32 s19, s19, s50 // (op 1188) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1188) 

s_mov_b32 s20, s26 // (op 1188) reg expression
s_mov_b32 s21, s27 // (op 1188) reg expression
s_mov_b32 s22, s0 // (op 1188) reg expression
s_mov_b32 s23, 131072 // (op 1188) call()

s_mul_hi_i32 s0, s16, s12 // (op 1254) 
v_lshlrev_b32 v17, 1, v1 // (op 1254) 
s_add_i32 s1, s0, s16 // (op 1254) 

s_ashr_i32 s0, s1, 31 // (op 1254) 
s_cmp_eq_i32 s12, 0 // (op 1254) 
s_mov_b32 s18, scc // (op 1254) 
s_cmp_eq_i32 s18, 0 // (op 1254) 
s_cselect_b32 s19, 0, -1 // (op 1254) 

s_and_b32 s18, s15, 31 // (op 1254) 
s_lshl_b32 s49, 1, s18 // (op 1254) 
s_add_i32 s50, s49, s19 // (op 1254) 


s_and_b32 s19, s0, s50 // (op 1254) 


s_add_i32 s0, s1, s19 // (op 1254) 


s_ashr_i32 s1, s0, s18 // (op 1254) 


s_lshl_b32 s0, s15, 24 // (op 1254) 
s_ashr_i32 s18, s0, 31 // (op 1254) 

s_xor_b32 s0, s1, s18 // (op 1254) 

s_sub_i32 s1, s0, s18 // (op 1254) 


s_mul_hi_i32 s0, s1, s44 // (op 1254) 
s_add_i32 s18, s0, s1 // (op 1254) 

s_ashr_i32 s0, s18, 31 // (op 1254) 
s_cmp_eq_i32 s44, 0 // (op 1254) 
s_mov_b32 s19, scc // (op 1254) 
s_cmp_eq_i32 s19, 0 // (op 1254) 
s_cselect_b32 s49, 0, -1 // (op 1254) 

s_and_b32 s19, s45, 31 // (op 1254) 
s_lshl_b32 s50, 1, s19 // (op 1254) 
s_add_i32 s51, s50, s49 // (op 1254) 


s_and_b32 s49, s0, s51 // (op 1254) 


s_add_i32 s0, s18, s49 // (op 1254) 


s_ashr_i32 s18, s0, s19 // (op 1254) 


s_lshl_b32 s0, s45, 24 // (op 1254) 
s_ashr_i32 s19, s0, 31 // (op 1254) 

s_xor_b32 s0, s18, s19 // (op 1254) 

s_sub_i32 s18, s0, s19 // (op 1254) 


s_add_u32 s50, s8, 64 // (op 1254) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1254) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1254) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1254) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1254) 

s_mul_i32 s0, s18, s50 // (op 1254) 

s_sub_i32 s49, s1, s0 // (op 1254) 


s_mul_hi_i32 s0, s16, s11 // (op 1254) 
s_add_i32 s1, s0, s16 // (op 1254) 

s_ashr_i32 s0, s1, 31 // (op 1254) 
s_cmp_eq_i32 s11, 0 // (op 1254) 
s_mov_b32 s18, scc // (op 1254) 
s_cmp_eq_i32 s18, 0 // (op 1254) 
s_cselect_b32 s19, 0, -1 // (op 1254) 

s_and_b32 s18, s47, 31 // (op 1254) 
s_lshl_b32 s72, 1, s18 // (op 1254) 
s_add_i32 s73, s72, s19 // (op 1254) 


s_and_b32 s19, s0, s73 // (op 1254) 


s_add_i32 s0, s1, s19 // (op 1254) 


s_ashr_i32 s1, s0, s18 // (op 1254) 


s_lshl_b32 s0, s47, 24 // (op 1254) 
s_ashr_i32 s18, s0, 31 // (op 1254) 

s_xor_b32 s0, s1, s18 // (op 1254) 

s_sub_i32 s1, s0, s18 // (op 1254) 


s_mul_i32 s0, s13, s50 // (op 1254) 

s_mul_i32 s18, s10, s0 // (op 1254) 

s_mul_i32 s0, s1, s18 // (op 1254) 

s_sub_u32 s18, s16, s0 // (op 1254) 

s_mul_hi_i32 s0, s18, s14 // (op 1254) 
s_add_i32 s19, s0, s18 // (op 1254) 


s_ashr_i32 s0, s19, 31 // (op 1254) 
s_cmp_eq_i32 s14, 0 // (op 1254) 
s_mov_b32 s18, scc // (op 1254) 
s_cmp_eq_i32 s18, 0 // (op 1254) 
s_cselect_b32 s50, 0, -1 // (op 1254) 

s_and_b32 s18, s46, 31 // (op 1254) 
s_lshl_b32 s51, 1, s18 // (op 1254) 
s_add_i32 s72, s51, s50 // (op 1254) 


s_and_b32 s50, s0, s72 // (op 1254) 


s_add_i32 s0, s19, s50 // (op 1254) 


s_ashr_i32 s19, s0, s18 // (op 1254) 


s_lshl_b32 s0, s46, 24 // (op 1254) 
s_ashr_i32 s18, s0, 31 // (op 1254) 

s_xor_b32 s0, s19, s18 // (op 1254) 

s_sub_i32 s19, s0, s18 // (op 1254) 


s_cmp_eq_i32 s1, 0 // (op 1254) 
s_mov_b32 s0, scc // (op 1254) 

s_cmp_eq_i32 s0, 0 // (op 1254) 
s_cselect_b32 s1, s19, s49 // (op 1254) 



s_lshl_b32 s0, s1, 6 // (op 1254) 

s_mov_b32 s19, 0 // (op 1254) convert
s_mov_b32 s18, s0 // (op 1254) convert

s_ashr_i64 s[0:1], s[28:29], 8 // (op 1254) 
s_lshr_b32 s49, s0, 1 // (op 1254) 

v_lshl_add_u32 v19, s49, 9, v17 // (op 1254) 


v_mul_lo_u32 v31, v19, s31 // (op 1254) most significant: low of low * high
v_mul_hi_u32 v17, v19, s30 // (op 1254) most significant: high of low * low
v_mul_lo_u32 v30, v19, s30 // (op 1254) least significant: low of low * low
v_add_u32 v31, v31, v17 // (op 1254) most significant: sum

v_mov_b32 v17, s18 // (op 1254) 
v_mov_b32 v69, s19 // (op 1254) 
v_add_co_u32 v30, s[0:1], v30, v17 // (op 1254) least significant half
v_addc_co_u32 v31, s[0:1], v31, v69, s[0:1] // (op 1254) most significant half





v_mul_lo_u32 v33, v31, 6 // (op 1254) most significant: low of high * low
v_mul_hi_u32 v17, v30, 6 // (op 1254) most significant: high of low * low
v_mul_lo_u32 v32, v30, 6 // (op 1254) least significant: low of low * low
v_add_u32 v33, v33, v17 // (op 1254) most significant: sum


v_lshrrev_b64 v[30:31], 3, v[32:33] // (op 1254) 

v_mov_b32 v17, v30 // (op 1254) convert

s_mul_i32 s1, s29, 6 // (op 1256) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1256) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1256) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1256) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1256) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 1256) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1256) least significant
s_mul_hi_u32 s50, s30, s0 // (op 1256) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1256) most significant: sum
s_add_u32 s19, s19, s50 // (op 1256) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1256) 

s_mov_b32 s52, s32 // (op 1256) reg expression
s_mov_b32 s53, s33 // (op 1256) reg expression
s_mov_b32 s54, s0 // (op 1256) reg expression
s_mov_b32 s55, 131072 // (op 1256) call()

s_mul_hi_i32 s0, s16, s12 // (op 1318) 
v_and_b32 v19, -16, v1 // (op 1318) 
s_add_i32 s1, s0, s16 // (op 1318) 

v_sub_u32 v69, v1, v19 // (op 1318) 

s_ashr_i32 s0, s1, 31 // (op 1318) 
v_lshrrev_b32 v19, 4, v1 // (op 1318) 
s_cmp_eq_i32 s12, 0 // (op 1318) 
s_mov_b32 s18, scc // (op 1318) 
s_cmp_eq_i32 s18, 0 // (op 1318) 
s_cselect_b32 s19, 0, -1 // (op 1318) 

s_and_b32 s18, s15, 31 // (op 1318) 
s_lshl_b32 s49, 1, s18 // (op 1318) 
s_add_i32 s50, s49, s19 // (op 1318) 


s_and_b32 s19, s0, s50 // (op 1318) 


s_add_i32 s0, s1, s19 // (op 1318) 


s_ashr_i32 s1, s0, s18 // (op 1318) 


s_lshl_b32 s0, s15, 24 // (op 1318) 
s_ashr_i32 s18, s0, 31 // (op 1318) 

s_xor_b32 s0, s1, s18 // (op 1318) 

s_sub_i32 s1, s0, s18 // (op 1318) 


s_mul_hi_i32 s0, s1, s44 // (op 1318) 
s_add_i32 s18, s0, s1 // (op 1318) 

s_ashr_i32 s0, s18, 31 // (op 1318) 
s_cmp_eq_i32 s44, 0 // (op 1318) 
s_mov_b32 s19, scc // (op 1318) 
s_cmp_eq_i32 s19, 0 // (op 1318) 
s_cselect_b32 s49, 0, -1 // (op 1318) 

s_and_b32 s19, s45, 31 // (op 1318) 
s_lshl_b32 s50, 1, s19 // (op 1318) 
s_add_i32 s51, s50, s49 // (op 1318) 


s_and_b32 s49, s0, s51 // (op 1318) 


s_add_i32 s0, s18, s49 // (op 1318) 


s_ashr_i32 s18, s0, s19 // (op 1318) 


s_lshl_b32 s0, s45, 24 // (op 1318) 
s_ashr_i32 s19, s0, 31 // (op 1318) 

s_xor_b32 s0, s18, s19 // (op 1318) 

s_sub_i32 s18, s0, s19 // (op 1318) 


s_mul_i32 s0, s1, s13 // (op 1318) 

s_sub_u32 s1, s16, s0 // (op 1318) 

s_mul_i32 s0, s18, s13 // (op 1318) 
s_add_u32 s0, s0, s1 // (op 1318) 


s_mul_hi_i32 s1, s16, s11 // (op 1318) 
s_add_i32 s18, s1, s16 // (op 1318) 

s_ashr_i32 s1, s18, 31 // (op 1318) 
s_cmp_eq_i32 s11, 0 // (op 1318) 
s_mov_b32 s19, scc // (op 1318) 
s_cmp_eq_i32 s19, 0 // (op 1318) 
s_cselect_b32 s49, 0, -1 // (op 1318) 

s_and_b32 s19, s47, 31 // (op 1318) 
s_lshl_b32 s50, 1, s19 // (op 1318) 
s_add_i32 s51, s50, s49 // (op 1318) 


s_and_b32 s49, s1, s51 // (op 1318) 


s_add_i32 s1, s18, s49 // (op 1318) 


s_ashr_i32 s18, s1, s19 // (op 1318) 


s_lshl_b32 s1, s47, 24 // (op 1318) 
s_ashr_i32 s19, s1, 31 // (op 1318) 

s_xor_b32 s1, s18, s19 // (op 1318) 

s_sub_i32 s18, s1, s19 // (op 1318) 


s_add_u32 s50, s8, 64 // (op 1318) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1318) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1318) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1318) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1318) 

s_mul_i32 s1, s13, s50 // (op 1318) 

s_mul_i32 s19, s10, s1 // (op 1318) 

s_mul_i32 s1, s18, s19 // (op 1318) 

s_sub_u32 s19, s16, s1 // (op 1318) 

s_mov_b32 s1, s19 // (op 1318) 
s_mul_hi_i32 s49, s1, s14 // (op 1318) 
s_add_i32 s50, s49, s1 // (op 1318) 


s_ashr_i32 s1, s50, 31 // (op 1318) 
s_cmp_eq_i32 s14, 0 // (op 1318) 
s_mov_b32 s49, scc // (op 1318) 
s_cmp_eq_i32 s49, 0 // (op 1318) 
s_cselect_b32 s51, 0, -1 // (op 1318) 

s_and_b32 s49, s46, 31 // (op 1318) 
s_lshl_b32 s72, 1, s49 // (op 1318) 
s_add_i32 s73, s72, s51 // (op 1318) 


s_and_b32 s51, s1, s73 // (op 1318) 


s_add_i32 s1, s50, s51 // (op 1318) 


s_ashr_i32 s50, s1, s49 // (op 1318) 


s_lshl_b32 s1, s46, 24 // (op 1318) 
s_ashr_i32 s49, s1, 31 // (op 1318) 

s_xor_b32 s1, s50, s49 // (op 1318) 

s_sub_i32 s50, s1, s49 // (op 1318) 


s_mul_i32 s1, s48, s13 // (op 1318) 
s_add_u32 s72, s6, 16 // (op 1318) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 1318) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1318) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1318) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1318) 

s_sub_i32 s49, s72, s1 // (op 1318) 

s_mul_i32 s51, s50, s49 // (op 1318) 


s_sub_u32 s49, s19, s51 // (op 1318) 


s_add_u32 s19, s49, s1 // (op 1318) 


s_cmp_eq_i32 s18, 0 // (op 1318) 
s_mov_b32 s1, scc // (op 1318) 

s_cmp_eq_i32 s1, 0 // (op 1318) 
s_cselect_b32 s18, s19, s0 // (op 1318) 



v_lshl_add_u32 v73, s18, 4, v69 // (op 1318) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 1318) 
s_lshr_b32 s18, s0, 1 // (op 1318) 

v_lshl_add_u32 v69, s18, 4, v19 // (op 1318) 


v_mul_lo_u32 v19, v69, s34 // (op 1318) 

v_add_u32 v69, v73, v19 // (op 1318) 


s_mul_i32 s1, s34, s39 // (op 1320) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1320) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1320) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1320) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1320) most significant: sum
s_add_u32 s1, s1, s19 // (op 1320) most significant: sum


s_mov_b32 s64, s36 // (op 1320) reg expression
s_mov_b32 s65, s37 // (op 1320) reg expression
s_mov_b32 s66, s0 // (op 1320) reg expression
s_mov_b32 s67, 131072 // (op 1320) call()

s_mul_hi_i32 s0, s16, s12 // (op 1382) 
v_and_b32 v19, -16, v1 // (op 1382) 
s_add_i32 s1, s0, s16 // (op 1382) 

v_sub_u32 v73, v1, v19 // (op 1382) 

s_ashr_i32 s0, s1, 31 // (op 1382) 
v_lshlrev_b32 v19, 2, v73 // (op 1382) 

s_cmp_eq_i32 s12, 0 // (op 1382) 
s_mov_b32 s18, scc // (op 1382) 
v_lshrrev_b32 v73, 4, v1 // (op 1382) 
s_cmp_eq_i32 s18, 0 // (op 1382) 
s_cselect_b32 s19, 0, -1 // (op 1382) 

s_and_b32 s18, s15, 31 // (op 1382) 
s_lshl_b32 s49, 1, s18 // (op 1382) 
s_add_i32 s50, s49, s19 // (op 1382) 


s_and_b32 s19, s0, s50 // (op 1382) 


s_add_i32 s0, s1, s19 // (op 1382) 


s_ashr_i32 s1, s0, s18 // (op 1382) 


s_lshl_b32 s0, s15, 24 // (op 1382) 
s_ashr_i32 s18, s0, 31 // (op 1382) 

s_xor_b32 s0, s1, s18 // (op 1382) 

s_sub_i32 s1, s0, s18 // (op 1382) 


s_mul_hi_i32 s0, s1, s44 // (op 1382) 
s_add_i32 s18, s0, s1 // (op 1382) 

s_ashr_i32 s0, s18, 31 // (op 1382) 
s_cmp_eq_i32 s44, 0 // (op 1382) 
s_mov_b32 s19, scc // (op 1382) 
s_cmp_eq_i32 s19, 0 // (op 1382) 
s_cselect_b32 s49, 0, -1 // (op 1382) 

s_and_b32 s19, s45, 31 // (op 1382) 
s_lshl_b32 s50, 1, s19 // (op 1382) 
s_add_i32 s51, s50, s49 // (op 1382) 


s_and_b32 s49, s0, s51 // (op 1382) 


s_add_i32 s0, s18, s49 // (op 1382) 


s_ashr_i32 s18, s0, s19 // (op 1382) 


s_lshl_b32 s0, s45, 24 // (op 1382) 
s_ashr_i32 s19, s0, 31 // (op 1382) 

s_xor_b32 s0, s18, s19 // (op 1382) 

s_sub_i32 s18, s0, s19 // (op 1382) 


s_add_u32 s50, s8, 64 // (op 1382) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1382) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1382) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1382) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1382) 

s_mul_i32 s0, s18, s50 // (op 1382) 

s_sub_i32 s49, s1, s0 // (op 1382) 


s_mul_hi_i32 s0, s16, s11 // (op 1382) 
s_add_i32 s1, s0, s16 // (op 1382) 

s_ashr_i32 s0, s1, 31 // (op 1382) 
s_cmp_eq_i32 s11, 0 // (op 1382) 
s_mov_b32 s18, scc // (op 1382) 
s_cmp_eq_i32 s18, 0 // (op 1382) 
s_cselect_b32 s19, 0, -1 // (op 1382) 

s_and_b32 s18, s47, 31 // (op 1382) 
s_lshl_b32 s72, 1, s18 // (op 1382) 
s_add_i32 s73, s72, s19 // (op 1382) 


s_and_b32 s19, s0, s73 // (op 1382) 


s_add_i32 s0, s1, s19 // (op 1382) 


s_ashr_i32 s1, s0, s18 // (op 1382) 


s_lshl_b32 s0, s47, 24 // (op 1382) 
s_ashr_i32 s18, s0, 31 // (op 1382) 

s_xor_b32 s0, s1, s18 // (op 1382) 

s_sub_i32 s1, s0, s18 // (op 1382) 


s_mul_i32 s0, s13, s50 // (op 1382) 

s_mul_i32 s18, s10, s0 // (op 1382) 

s_mul_i32 s0, s1, s18 // (op 1382) 

s_sub_u32 s18, s16, s0 // (op 1382) 

s_mul_hi_i32 s0, s18, s14 // (op 1382) 
s_add_i32 s19, s0, s18 // (op 1382) 


s_ashr_i32 s0, s19, 31 // (op 1382) 
s_cmp_eq_i32 s14, 0 // (op 1382) 
s_mov_b32 s18, scc // (op 1382) 
s_cmp_eq_i32 s18, 0 // (op 1382) 
s_cselect_b32 s50, 0, -1 // (op 1382) 

s_and_b32 s18, s46, 31 // (op 1382) 
s_lshl_b32 s51, 1, s18 // (op 1382) 
s_add_i32 s72, s51, s50 // (op 1382) 


s_and_b32 s50, s0, s72 // (op 1382) 


s_add_i32 s0, s19, s50 // (op 1382) 


s_ashr_i32 s19, s0, s18 // (op 1382) 


s_lshl_b32 s0, s46, 24 // (op 1382) 
s_ashr_i32 s18, s0, 31 // (op 1382) 

s_xor_b32 s0, s19, s18 // (op 1382) 

s_sub_i32 s19, s0, s18 // (op 1382) 


s_cmp_eq_i32 s1, 0 // (op 1382) 
s_mov_b32 s0, scc // (op 1382) 

s_cmp_eq_i32 s0, 0 // (op 1382) 
s_cselect_b32 s1, s19, s49 // (op 1382) 



v_lshl_add_u32 v77, s1, 6, v19 // (op 1382) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 1382) 
s_lshr_b32 s18, s0, 1 // (op 1382) 

v_lshl_add_u32 v19, s18, 4, v73 // (op 1382) 


v_mul_lo_u32 v73, v19, s40 // (op 1382) 
v_add_u32 v73, v73, v77 // (op 1382) 


s_mul_i32 s1, s40, s39 // (op 1384) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1384) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1384) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1384) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1384) most significant: sum
s_add_u32 s1, s1, s19 // (op 1384) most significant: sum


s_mov_b32 s68, s42 // (op 1384) reg expression
s_mov_b32 s69, s43 // (op 1384) reg expression
s_mov_b32 s70, s0 // (op 1384) reg expression
s_mov_b32 s71, 131072 // (op 1384) call()

v_lshlrev_b32 v19, 5, v1 // (op 1475) 
v_mul_lo_u32 v77, v19, 6 // (op 1475) 
v_lshrrev_b32 v81, 3, v77 // (op 1475) 

v_lshrrev_b32 v77, 4, v19 // (op 1475) 

v_lshlrev_b32 v19, 2, v77 // (op 1475) 

v_add_u32 v77, v81, v19 // (op 1475) 


v_lshlrev_b32 v19, 7, v1 // (op 1528) 
v_mul_lo_u32 v81, v19, 6 // (op 1528) 
v_lshrrev_b32 v85, 3, v81 // (op 1528) 

v_lshrrev_b32 v81, 4, v19 // (op 1528) 

v_lshlrev_b32 v19, 2, v81 // (op 1528) 

v_add_u32 v81, v85, v19 // (op 1528) 


v_and_b32 v19, -16, v1 // (op 1586) 
v_sub_u32 v85, v1, v19 // (op 1586) 

v_and_b32 v19, 4294967280, v1 // (op 1586) 
v_add_u32 v89, v19, v85 // (op 1586) 


v_and_b32 v19, -16, v1 // (op 1639) 
v_sub_u32 v85, v1, v19 // (op 1639) 

v_lshlrev_b32 v19, 2, v85 // (op 1639) 

v_lshrrev_b32 v85, 4, v1 // (op 1639) 
v_lshl_add_u32 v93, v85, 6, v19 // (op 1639) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 409) 
s_and_b32 s18, s0, 4294967294 // (op 409) 

s_ashr_i64 s[0:1], s[28:29], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408:

s_add_u32 s18, s18, 1 // (op 410) 
s_mul_i32 s1, s25, 6 // (op 202) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 202) least significant
s_mul_hi_u32 s19, s24, 6 // (op 202) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 202) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v16, v14 // (op 202) reg expression
buffer_load_dwordx3 v[20:22], v16, s[20:23], 0 offen // (op 202) Load value
v_add_u32 v16, v16, v18 // (op 202) 
buffer_load_dwordx3 v[24:26], v16, s[20:23], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[24:25], 8 // (op 1176) 
s_mul_i32 s51, s1, 6 // (op 1176) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 1176) least significant
s_mul_hi_u32 s19, s0, 6 // (op 1176) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 1176) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 1176) 

v_add_u32 v15, v15, s0 // (op 1176) 


s_barrier  // (op 1853) 
s_mul_i32 s1, s31, 6 // (op 225) most significant: low of high * low



s_mul_i32 s0, s30, 6 // (op 225) least significant
s_mul_hi_u32 s19, s30, 6 // (op 225) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 225) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 225) 


v_mov_b32 v14, v17 // (op 225) 
v_mov_b32 v16, v14 // (op 225) reg expression
buffer_load_dwordx3 v[28:30], v16, s[52:55], 0 offen // (op 225) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v77, v[20:22] offset:16896 // (op 203) Store local data 
buffer_load_dwordx3 v[32:34], v16, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[36:38], v16, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[40:42], v16, s[52:55], 0 offen offset:36 // (op 225) Load value
v_add_u32 v16, v16, v18 // (op 225) 
s_waitcnt vmcnt(4)
ds_write_b96 v77, v[24:26] offset:16912 // (op 203) Store local data 
buffer_load_dwordx3 v[44:46], v16, s[52:55], 0 offen // (op 225) Load value

buffer_load_dwordx3 v[48:50], v16, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[52:54], v16, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[56:58], v16, s[52:55], 0 offen offset:36 // (op 225) Load value



s_lshl_b64 s[0:1], s[30:31], 8 // (op 1246) 
s_mul_i32 s51, s1, 6 // (op 1246) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 1246) least significant
s_mul_hi_u32 s19, s0, 6 // (op 1246) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 1246) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 1246) 

v_add_u32 v17, v17, s0 // (op 1246) 



s_barrier  // (op 1872) 
v_mov_b32 v14, v69 // (op 226) 



v_mov_b32 v16, v14 // (op 226) reg expression

buffer_load_ubyte v31, v16, s[64:67], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s19, s34, 3 // (op 1310) 
v_add_u32 v69, v69, s19 // (op 1310) 

s_waitcnt vmcnt(8)
ds_write_b96 v81, v[28:30] // (op 222) Store local data 


v_mov_b32 v14, v73 // (op 227) 
v_mov_b32 v16, v14 // (op 227) reg expression

buffer_load_dword v35, v16, s[68:71], 0 offen // (op 227) Load value


s_lshl_b32 s19, s40, 3 // (op 1374) 
v_add_u32 v73, v73, s19 // (op 1374) 

s_waitcnt vmcnt(8)
ds_write_b96 v81, v[32:34] offset:16 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1862) 
s_waitcnt vmcnt(7)
ds_write_b96 v81, v[36:38] offset:32 // (op 222) Store local data 



ds_read_b96_tr_b6 v[18:20], v0 offset:16896 // (op 205) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:17152 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v81, v[40:42] offset:48 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v81, v[44:46] offset:64 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v81, v[48:50] offset:80 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v81, v[52:54] offset:96 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v81, v[56:58] offset:112 // (op 222) Store local data 

s_barrier  // (op 1893) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1883) 






ds_read_b96_tr_b6 v[24:26], v5 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v5 offset:1024 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v89, v31 offset:20992 // (op 223) Store local data 

s_barrier  // (op 1913) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1903) 





ds_read_u8 v14, v4 offset:20992 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v93, v35 offset:16384 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1923) 


ds_read_u8 v16, v9 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 212) 





ds_read_b96_tr_b6 v[24:26], v0 offset:18944 // (op 214) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:19200 // (op 214) Transpose load local data 



ds_read_b96_tr_b6 v[30:32], v5 offset:8192 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v5 offset:9216 // (op 216) Transpose load local data 


ds_read_u8 v16, v4 offset:21056 // (op 218) Load local data 


ds_read_u8 v36, v9 offset:16640 // (op 220) Load local data 



ds_read_b96_tr_b6 v[82:84], v5 offset:32 // (op 416) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v5 offset:1056 // (op 416) Transpose load local data 


ds_read_u8 v88, v9 offset:16416 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:23], v[82:87], a[4:7], v14, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v16, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 221) 
ds_read_b96_tr_b6 v[18:20], v5 offset:8224 // (op 425) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v5 offset:9248 // (op 425) Transpose load local data 


ds_read_u8 v14, v9 offset:16672 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[18:23], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 430) 




s_ashr_i64 s[0:1], s[28:29], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 if true)
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2024) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157
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
s_branch RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157:


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
RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 152
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 144
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: B6
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
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
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
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
  - .name: RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 142
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_28
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: Subtract_24
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

