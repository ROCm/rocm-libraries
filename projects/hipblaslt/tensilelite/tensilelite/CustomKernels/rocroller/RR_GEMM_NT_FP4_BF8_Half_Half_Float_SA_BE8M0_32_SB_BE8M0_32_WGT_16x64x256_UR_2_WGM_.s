// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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
s_cmp_eq_i32 s12, 0 // (op 1446) 
s_mov_b32 s19, scc // (op 1446) 
v_lshlrev_b32 v9, 2, v8 // (op 1446) 

s_cmp_eq_i32 s19, 0 // (op 1446) 
s_cselect_b32 s51, 0, -1 // (op 1446) 

v_lshl_add_u32 v8, v0, 4, v9 // (op 1446) 


s_and_b32 s19, s15, 31 // (op 1446) 
v_lshrrev_b32 v0, 2, v7 // (op 1446) 

v_and_b32 v7, 3, v0 // (op 1446) 

v_and_b32 v0, 48, v1 // (op 1561) 
v_and_b32 v9, 63, v1 // (op 1561) 
v_sub_u32 v10, v9, v0 // (op 1561) 

v_lshrrev_b32 v9, 1, v10 // (op 1561) 
v_add_lshl_u32 v11, v0, v9, 6 // (op 1561) 


v_and_b32 v0, 1, v10 // (op 1561) 

v_lshlrev_b32 v9, 3, v0 // (op 1561) 

v_lshrrev_b32 v0, 6, v1 // (op 1561) 
v_and_b32 v10, 1, v0 // (op 1561) 

v_lshl_add_u32 v0, v10, 4, v9 // (op 1561) 


v_add_u32 v9, v11, v0 // (op 1561) 


v_lshrrev_b32 v0, 6, v1 // (op 1672) 
v_and_b32 v10, 1, v0 // (op 1672) 

v_and_b32 v0, 15, v1 // (op 1672) 
v_lshl_add_u32 v11, v10, 4, v0 // (op 1672) 


v_and_b32 v0, 63, v1 // (op 1672) 
v_lshrrev_b32 v10, 4, v0 // (op 1672) 

v_lshl_add_u32 v0, v10, 6, v11 // (op 1672) 


v_lshrrev_b32 v10, 7, v1 // (op 1688) 
v_and_b32 v11, 63, v1 // (op 1688) 
v_and_b32 v12, 48, v1 // (op 1743) 
v_and_b32 v13, 63, v1 // (op 1743) 
v_sub_u32 v14, v13, v12 // (op 1743) 

v_lshrrev_b32 v13, 1, v14 // (op 1743) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 1743) 


v_and_b32 v12, 1, v14 // (op 1743) 

v_lshlrev_b32 v13, 3, v12 // (op 1743) 

v_lshrrev_b32 v12, 6, v1 // (op 1743) 
v_and_b32 v14, 1, v12 // (op 1743) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 1743) 


v_add_u32 v13, v15, v12 // (op 1743) 


v_lshrrev_b32 v12, 6, v1 // (op 1796) 
v_and_b32 v14, 1, v12 // (op 1796) 

v_and_b32 v12, 15, v1 // (op 1796) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1796) 


v_and_b32 v12, 63, v1 // (op 1796) 
v_lshrrev_b32 v14, 4, v12 // (op 1796) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 1796) 


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



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 1213) 


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1213) 

v_mov_b32 v16, v14 // (op 1213) convert

s_mul_i32 s1, s24, s29 // (op 1215) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 1215) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1215) least significant
s_mul_hi_u32 s19, s24, s28 // (op 1215) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1215) most significant: sum
s_add_u32 s1, s1, s19 // (op 1215) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1215) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1215) 

s_mov_b32 s20, s26 // (op 1215) reg expression
s_mov_b32 s21, s27 // (op 1215) reg expression
s_mov_b32 s22, s0 // (op 1215) reg expression
s_mov_b32 s23, 131072 // (op 1215) call()

s_mul_hi_i32 s0, s16, s12 // (op 1232) 
v_lshlrev_b32 v14, 1, v1 // (op 1232) 
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

v_mul_lo_u32 v19, v14, s25 // (op 1232) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1232) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 1232) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1232) most significant: sum


v_mov_b32 v17, s0 // (op 1232) 
v_mov_b32 v14, s1 // (op 1232) 
v_add_co_u32 v20, s[18:19], v17, v18 // (op 1232) least significant half
v_addc_co_u32 v21, s[18:19], v14, v19, s[18:19] // (op 1232) most significant half



v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 1232) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 1232) 

v_mov_b32 v17, v14 // (op 1232) convert

s_mul_i32 s1, s24, s29 // (op 1234) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 1234) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1234) least significant
s_mul_hi_u32 s19, s24, s28 // (op 1234) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1234) most significant: sum
s_add_u32 s1, s1, s19 // (op 1234) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1234) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1234) 

s_mov_b32 s20, s26 // (op 1234) reg expression
s_mov_b32 s21, s27 // (op 1234) reg expression
s_mov_b32 s22, s0 // (op 1234) reg expression
s_mov_b32 s23, 131072 // (op 1234) call()

s_mul_hi_i32 s0, s16, s12 // (op 1277) 
v_and_b32 v14, -4, v1 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

v_sub_u32 v15, v1, v14 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
v_lshlrev_b32 v14, 4, v15 // (op 1277) 

s_cmp_eq_i32 s12, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
v_lshrrev_b32 v15, 2, v1 // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

v_lshlrev_b32 v18, 3, v15 // (op 1277) 

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



v_lshl_add_u32 v15, s1, 6, v14 // (op 1277) 


v_mul_lo_u32 v14, v18, s30 // (op 1277) 
v_add_u32 v14, v14, v15 // (op 1277) 


s_mul_i32 s1, s30, s29 // (op 1279) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 1279) least significant
s_mul_hi_u32 s19, s30, s28 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1279) most significant: sum
s_add_u32 s1, s1, s19 // (op 1279) most significant: sum


s_mov_b32 s52, s32 // (op 1279) reg expression
s_mov_b32 s53, s33 // (op 1279) reg expression
s_mov_b32 s54, s0 // (op 1279) reg expression
s_mov_b32 s55, 131072 // (op 1279) call()

s_mul_hi_i32 s0, s16, s12 // (op 1296) 
v_and_b32 v15, -4, v1 // (op 1296) 
s_add_i32 s1, s0, s16 // (op 1296) 

v_sub_u32 v18, v1, v15 // (op 1296) 

s_ashr_i32 s0, s1, 31 // (op 1296) 
v_lshlrev_b32 v15, 4, v18 // (op 1296) 

s_cmp_eq_i32 s12, 0 // (op 1296) 
s_mov_b32 s18, scc // (op 1296) 
v_lshrrev_b32 v18, 2, v1 // (op 1296) 
s_cmp_eq_i32 s18, 0 // (op 1296) 
s_cselect_b32 s19, 0, -1 // (op 1296) 

v_lshlrev_b32 v19, 3, v18 // (op 1296) 

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



v_lshl_add_u32 v18, s1, 6, v15 // (op 1296) 


v_mul_lo_u32 v15, v19, s30 // (op 1296) 
v_add_u32 v15, v15, v18 // (op 1296) 


s_mul_i32 s1, s30, s29 // (op 1298) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 1298) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 1298) least significant
s_mul_hi_u32 s19, s30, s28 // (op 1298) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1298) most significant: sum
s_add_u32 s1, s1, s19 // (op 1298) most significant: sum


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


v_mov_b64 v[22:23], s[30:31] // (op 1037) convert

v_mov_b32 v24, v14 // (op 1037) reg expression
buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[34:37], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[38:41], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[42:45], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[46:49], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[50:53], v24, s[52:55], 0 offen // (op 1037) Load value
v_add_u32 v24, v24, v22 // (op 1037) 
buffer_load_dwordx4 v[54:57], v24, s[52:55], 0 offen // (op 1037) Load value




v_mov_b32 v22, v20 // (op 1040) reg expression

buffer_load_dword v23, v22, s[68:71], 0 offen // (op 1040) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 1043) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1043) 


v_mov_b32 v22, v16 // (op 1043) reg expression
buffer_load_dwordx2 v[58:59], v22, s[20:23], 0 offen // (op 1043) Load value
v_add_u32 v22, v22, v24 // (op 1043) 
buffer_load_dwordx2 v[60:61], v22, s[20:23], 0 offen // (op 1043) Load value




v_mov_b32 v22, v18 // (op 1046) reg expression

buffer_load_ubyte v24, v22, s[64:67], 0 offen offset: 0 // (op 1046) Load value


v_mov_b64 v[62:63], s[30:31] // (op 1049) convert

s_lshl_b32 s0, s30, 8 // (op 1049) 
v_add_u32 v22, v14, s0 // (op 1049) 

buffer_load_dwordx4 v[64:67], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[68:71], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[72:75], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[76:79], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[80:83], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[84:87], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[88:91], v22, s[52:55], 0 offen // (op 1049) Load value
v_add_u32 v22, v22, v62 // (op 1049) 
buffer_load_dwordx4 v[92:95], v22, s[52:55], 0 offen // (op 1049) Load value





s_lshl_b32 s0, s40, 3 // (op 1052) 
v_add_u32 v14, v20, s0 // (op 1052) 


buffer_load_dword v22, v14, s[68:71], 0 offen // (op 1052) Load value



s_lshl_b64 s[0:1], s[24:25], 2 // (op 1055) 
v_lshrrev_b64 v[62:63], 3, s[0:1] // (op 1055) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1055) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 1055) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 1055) 

v_add_u32 v14, v16, s0 // (op 1055) 

buffer_load_dwordx2 v[96:97], v14, s[20:23], 0 offen // (op 1055) Load value
v_add_u32 v14, v14, v62 // (op 1055) 
buffer_load_dwordx2 v[98:99], v14, s[20:23], 0 offen // (op 1055) Load value





s_lshl_b32 s0, s34, 3 // (op 1058) 
v_add_u32 v14, v18, s0 // (op 1058) 


buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 1058) Load value


v_and_b32 v14, -4, v1 // (op 1544) 
v_sub_u32 v18, v1, v14 // (op 1544) 

v_lshlrev_b32 v14, 4, v18 // (op 1544) 

v_lshrrev_b32 v18, 2, v1 // (op 1544) 
v_lshl_add_u32 v20, v18, 9, v14 // (op 1544) 





s_waitcnt vmcnt(23)
ds_write_b128 v20, v[26:29] // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v20, v[30:33] offset:64 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v20, v[34:37] offset:128 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v20, v[38:41] offset:192 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v20, v[42:45] offset:256 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v20, v[46:49] offset:320 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v20, v[50:53] offset:384 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v20, v[54:57] offset:448 // (op 1060) Store local data 

v_and_b32 v14, -16, v1 // (op 1655) 
v_sub_u32 v18, v1, v14 // (op 1655) 

v_lshlrev_b32 v14, 2, v18 // (op 1655) 

v_lshrrev_b32 v18, 4, v1 // (op 1655) 
v_lshl_add_u32 v20, v18, 6, v14 // (op 1655) 





s_waitcnt vmcnt(15)
ds_write_b32 v20, v23 offset:16384 // (op 1061) Store local data 

v_lshlrev_b32 v14, 7, v1 // (op 1491) 
v_lshrrev_b32 v18, 3, v14 // (op 1491) 




s_waitcnt vmcnt(14)
ds_write_b64 v18, v[58:59] offset:16896 // (op 1062) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v18, v[60:61] offset:16904 // (op 1062) Store local data 

v_and_b32 v14, -16, v1 // (op 1602) 
v_sub_u32 v18, v1, v14 // (op 1602) 

v_and_b32 v14, 4294967280, v1 // (op 1602) 
v_add_u32 v20, v14, v18 // (op 1602) 





s_waitcnt vmcnt(12)
ds_write_b8 v20, v24 offset:18944 // (op 1063) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1866) 



ds_read_b64_tr_b8 v[100:101], v9 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v9 offset:512 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v9 offset:4096 // (op 1081) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v9 offset:4608 // (op 1081) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1907) 


ds_read_u8 v14, v0 offset:16384 // (op 1094) Load local data 



ds_read_b64_tr_b8 v[108:109], v9 offset:32 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v9 offset:544 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v9 offset:4128 // (op 1086) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v9 offset:4640 // (op 1086) Transpose load local data 


ds_read_u8 v18, v0 offset:16416 // (op 1099) Load local data 
s_waitcnt lgkmcnt(11)
s_barrier  // (op 1847) 



ds_read_b64_tr_b4 v[116:117], v3 offset:16896 // (op 1077) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v3 offset:17024 // (op 1077) Transpose load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 1887) 


ds_read_u8 v20, v4 offset:18944 // (op 1090) Load local data 
v_lshlrev_b32 v25, 7, v1 // (op 1462) 
v_lshrrev_b32 v62, 3, v25 // (op 1462) 

v_and_b32 v25, -4, v1 // (op 1515) 
v_sub_u32 v63, v1, v25 // (op 1515) 

v_lshlrev_b32 v25, 4, v63 // (op 1515) 

v_lshrrev_b32 v63, 2, v1 // (op 1515) 
v_lshl_add_u32 v120, v63, 9, v25 // (op 1515) 


v_and_b32 v25, -16, v1 // (op 1573) 
v_sub_u32 v63, v1, v25 // (op 1573) 

v_and_b32 v25, 4294967280, v1 // (op 1573) 
v_add_u32 v121, v25, v63 // (op 1573) 


v_and_b32 v25, -16, v1 // (op 1626) 
v_sub_u32 v63, v1, v25 // (op 1626) 

v_lshlrev_b32 v25, 2, v63 // (op 1626) 

v_lshrrev_b32 v63, 4, v1 // (op 1626) 
v_lshl_add_u32 v122, v63, 6, v25 // (op 1626) 


v_lshlrev_b32 v25, 7, v1 // (op 1702) 
v_lshrrev_b32 v63, 3, v25 // (op 1702) 

v_and_b32 v25, -4, v1 // (op 1726) 
v_sub_u32 v123, v1, v25 // (op 1726) 

v_lshlrev_b32 v25, 4, v123 // (op 1726) 

v_lshrrev_b32 v123, 2, v1 // (op 1726) 
v_lshl_add_u32 v124, v123, 9, v25 // (op 1726) 


v_and_b32 v25, -16, v1 // (op 1755) 
v_sub_u32 v123, v1, v25 // (op 1755) 

v_and_b32 v25, 4294967280, v1 // (op 1755) 
v_add_u32 v125, v25, v123 // (op 1755) 


v_and_b32 v25, -16, v1 // (op 1779) 
v_sub_u32 v123, v1, v25 // (op 1779) 

v_lshlrev_b32 v25, 2, v123 // (op 1779) 

v_lshrrev_b32 v123, 4, v1 // (op 1779) 
v_lshl_add_u32 v126, v123, 6, v25 // (op 1779) 


s_mov_b32 s0, 0 // (op 404) call()
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403:

s_add_u32 s0, s0, 2 // (op 405) 
v_mov_b64 v[128:129], s[30:31] // (op 10) convert

v_mov_b32 v25, v15 // (op 10) 
s_lshl_b32 s1, s30, 9 // (op 10) 
v_add_u32 v123, v25, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v123, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[30:33], v123, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[34:37], v123, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[100:107], a[0:3], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 90) 
buffer_load_dwordx4 v[38:41], v123, s[52:55], 0 offen // (op 10) Load value
s_barrier  // (op 1856) 
v_add_u32 v123, v123, v128 // (op 10) 



buffer_load_dwordx4 v[42:45], v123, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
buffer_load_dwordx4 v[46:49], v123, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v123, v123, v128 // (op 10) 
ds_read_b64_tr_b4 v[130:131], v3 offset:17920 // (op 93) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v3 offset:18048 // (op 93) Transpose load local data 
buffer_load_dwordx4 v[50:53], v123, s[52:55], 0 offen // (op 10) Load value
s_barrier  // (op 1876) 
v_add_u32 v123, v123, v128 // (op 10) 



ds_read_b64_tr_b8 v[100:101], v9 offset:8192 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v9 offset:8704 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v9 offset:12288 // (op 97) Transpose load local data 
buffer_load_dwordx4 v[54:57], v123, s[52:55], 0 offen // (op 10) Load value


ds_read_b64_tr_b8 v[106:107], v9 offset:12800 // (op 97) Transpose load local data 


s_barrier  // (op 1897) 
v_mov_b32 v123, v21 // (op 25) 


ds_read_u8 v127, v4 offset:19008 // (op 101) Load local data 
s_lshl_b32 s1, s40, 4 // (op 25) 
s_barrier  // (op 1917) 
v_add_u32 v128, v123, s1 // (op 25) 




ds_read_u8 v14, v0 offset:16640 // (op 105) Load local data 
buffer_load_dword v23, v128, s[68:71], 0 offen // (op 25) Load value


s_lshl_b64 s[18:19], s[24:25], 2 // (op 4) 
v_lshrrev_b64 v[128:129], 3, s[18:19] // (op 4) 


v_mov_b32 v134, v17 // (op 4) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 4) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v135, v134, s1 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[100:107], a[0:3], v127, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 107) 
buffer_load_dwordx2 v[58:59], v135, s[20:23], 0 offen // (op 4) Load value
v_add_u32 v135, v135, v128 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[108:115], a[4:7], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 451) 
buffer_load_dwordx2 v[60:61], v135, s[20:23], 0 offen // (op 4) Load value







ds_read_b64_tr_b8 v[108:109], v9 offset:8224 // (op 455) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v9 offset:8736 // (op 455) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v9 offset:12320 // (op 455) Transpose load local data 
v_mov_b32 v128, v19 // (op 17) 
s_lshl_b32 s1, s34, 4 // (op 17) 
v_add_u32 v129, v128, s1 // (op 17) 


ds_read_b64_tr_b8 v[114:115], v9 offset:12832 // (op 455) Transpose load local data 
buffer_load_ubyte v24, v129, s[64:67], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 1937) 
ds_read_u8 v18, v0 offset:16672 // (op 459) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[108:115], a[4:7], v127, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 460) 
s_waitcnt vmcnt(23)
ds_write_b128 v124, v[64:67] offset:19072 // (op 306) Store local data 


s_waitcnt vmcnt(22)
ds_write_b128 v124, v[68:71] offset:19136 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v124, v[72:75] offset:19200 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v124, v[76:79] offset:19264 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v124, v[80:83] offset:19328 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v124, v[84:87] offset:19392 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v124, v[88:91] offset:19456 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v124, v[92:95] offset:19520 // (op 306) Store local data 
s_barrier  // (op 1949) 



s_waitcnt vmcnt(15)
ds_write_b32 v126, v22 offset:35456 // (op 308) Store local data 
s_barrier  // (op 1930) 



s_waitcnt vmcnt(14)
ds_write_b64 v63, v[96:97] offset:35968 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v63, v[98:99] offset:35976 // (op 287) Store local data 
s_barrier  // (op 1943) 



s_waitcnt vmcnt(12)
ds_write_b8 v125, v16 offset:38016 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1933) 



ds_read_b64_tr_b8 v[100:101], v13 offset:19072 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v13 offset:19584 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v13 offset:23168 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v13 offset:23680 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1946) 


ds_read_u8 v14, v12 offset:35456 // (op 295) Load local data 



ds_read_b64_tr_b8 v[136:137], v13 offset:19104 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v13 offset:19616 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v13 offset:23200 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[142:143], v13 offset:23712 // (op 474) Transpose load local data 


ds_read_u8 v127, v12 offset:35488 // (op 478) Load local data 
s_waitcnt lgkmcnt(11)
s_barrier  // (op 1927) 



ds_read_b64_tr_b4 v[116:117], v5 offset:35968 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v5 offset:36096 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 1940) 


ds_read_u8 v20, v6 offset:38016 // (op 293) Load local data 

v_mov_b64 v[132:133], s[30:31] // (op 309) convert

s_lshl_b32 s1, s30, 8 // (op 309) 
s_mul_i32 s18, 3, s1 // (op 309) 

v_add_u32 v135, v25, s18 // (op 309) 

buffer_load_dwordx4 v[64:67], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[68:71], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[72:75], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[100:107], a[0:3], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 296) 
buffer_load_dwordx4 v[76:79], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[136:143], a[4:7], v20, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 461) 
buffer_load_dwordx4 v[80:83], v135, s[52:55], 0 offen // (op 309) Load value


v_add_u32 v135, v135, v132 // (op 309) 
buffer_load_dwordx4 v[84:87], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 
ds_read_b64_tr_b4 v[136:137], v5 offset:36992 // (op 298) Transpose load local data 
ds_read_b64_tr_b4 v[138:139], v5 offset:37120 // (op 298) Transpose load local data 
buffer_load_dwordx4 v[88:91], v135, s[52:55], 0 offen // (op 309) Load value
v_add_u32 v135, v135, v132 // (op 309) 



s_barrier  // (op 1998) 
buffer_load_dwordx4 v[92:95], v135, s[52:55], 0 offen // (op 309) Load value


ds_read_b64_tr_b8 v[140:141], v13 offset:27264 // (op 300) Transpose load local data 

ds_read_b64_tr_b8 v[142:143], v13 offset:27776 // (op 300) Transpose load local data 


ds_read_b64_tr_b8 v[144:145], v13 offset:31360 // (op 300) Transpose load local data 
s_lshl_b32 s1, s40, 3 // (op 311) 
s_mul_i32 s18, 3, s1 // (op 311) 

s_lshl_b32 s1, s30, 9 // (op 1288) 
v_add_u32 v15, v15, s1 // (op 1288) 

v_add_u32 v25, v123, s18 // (op 311) 


buffer_load_dword v22, v25, s[68:71], 0 offen // (op 311) Load value

ds_read_b64_tr_b8 v[146:147], v13 offset:31872 // (op 300) Transpose load local data 




ds_read_u8 v25, v6 offset:38080 // (op 302) Load local data 
s_lshl_b32 s1, s40, 4 // (op 1416) 
v_add_u32 v21, v21, s1 // (op 1416) 



s_barrier  // (op 2002) 
ds_read_u8 v123, v12 offset:35712 // (op 304) Load local data 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 286) 



v_lshrrev_b64 v[132:133], 3, s[18:19] // (op 286) 


s_lshl_b64 s[18:19], s[24:25], 8 // (op 286) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 286) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 286) 

s_mul_i32 s1, 3, s18 // (op 286) 

v_add_u32 v127, v134, s1 // (op 286) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[140:147], a[0:3], v25, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 305) 
ds_read_b64_tr_b8 v[148:149], v13 offset:27296 // (op 465) Transpose load local data 


ds_read_b64_tr_b8 v[150:151], v13 offset:27808 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[152:153], v13 offset:31392 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[154:155], v13 offset:31904 // (op 465) Transpose load local data 
buffer_load_dwordx2 v[96:97], v127, s[20:23], 0 offen // (op 286) Load value
v_add_u32 v127, v127, v132 // (op 286) 


s_barrier  // (op 2038) 
buffer_load_dwordx2 v[98:99], v127, s[20:23], 0 offen // (op 286) Load value


ds_read_u8 v123, v12 offset:35744 // (op 469) Load local data 



s_barrier  // (op 2042) 
s_lshl_b32 s1, s34, 3 // (op 310) 
s_mul_i32 s18, 3, s1 // (op 310) 

v_add_u32 v127, v128, s18 // (op 310) 


s_lshl_b64 s[18:19], s[24:25], 8 // (op 1224) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 1224) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 1224) 

buffer_load_ubyte v16, v127, s[64:67], 0 offen offset: 0 // (op 310) Load value

s_lshl_b32 s1, s18, 1 // (op 1224) 


v_add_u32 v17, v17, s1 // (op 1224) 

s_barrier  // (op 1869) 
s_lshl_b32 s1, s34, 4 // (op 1352) 



v_add_u32 v19, v19, s1 // (op 1352) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[148:155], a[4:7], v25, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 470) 
s_waitcnt vmcnt(23)
ds_write_b128 v120, v[26:29] // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b128 v120, v[30:33] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v120, v[34:37] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v120, v[38:41] offset:192 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v120, v[42:45] offset:256 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v120, v[46:49] offset:320 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v120, v[50:53] offset:384 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v120, v[54:57] offset:448 // (op 13) Store local data 
s_barrier  // (op 1910) 



s_waitcnt vmcnt(15)
ds_write_b32 v122, v23 offset:16384 // (op 30) Store local data 
s_barrier  // (op 1850) 



s_waitcnt vmcnt(14)
ds_write_b64 v62, v[58:59] offset:16896 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v62, v[60:61] offset:16904 // (op 6) Store local data 
s_barrier  // (op 1890) 



s_waitcnt vmcnt(12)
ds_write_b8 v121, v24 offset:18944 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1879) 



ds_read_b64_tr_b8 v[100:101], v9 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v9 offset:512 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v9 offset:4096 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v9 offset:4608 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1920) 


ds_read_u8 v14, v0 offset:16384 // (op 87) Load local data 



ds_read_b64_tr_b8 v[108:109], v9 offset:32 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v9 offset:544 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v9 offset:4128 // (op 446) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v9 offset:4640 // (op 446) Transpose load local data 


ds_read_u8 v18, v0 offset:16416 // (op 450) Load local data 
s_waitcnt lgkmcnt(11)
s_barrier  // (op 1859) 



ds_read_b64_tr_b4 v[116:117], v3 offset:16896 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v3 offset:17024 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 1900) 


ds_read_u8 v20, v4 offset:18944 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[28:29], 8 // (op 403) 
s_mov_b32 s1, s18 // (op 403) convert

s_and_b32 s18, s1, 4294967294 // (op 403) 

s_cmp_lt_u32 s0, s18 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403:

















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


v_mul_lo_u32 v63, v17, s25 // (op 1186) most significant: low of low * high
v_mul_hi_u32 v15, v17, s24 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v62, v17, s24 // (op 1186) least significant: low of low * low
v_add_u32 v63, v63, v15 // (op 1186) most significant: sum


v_mov_b32 v15, s0 // (op 1186) 
v_mov_b32 v17, s1 // (op 1186) 
v_add_co_u32 v60, s[18:19], v15, v62 // (op 1186) least significant half
v_addc_co_u32 v61, s[18:19], v17, v63, s[18:19] // (op 1186) most significant half



v_lshlrev_b64 v[60:61], 2, v[60:61] // (op 1186) 


v_lshrrev_b64 v[62:63], 3, v[60:61] // (op 1186) 

v_mov_b32 v15, v62 // (op 1186) convert

s_mul_i32 s1, s24, s29 // (op 1188) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 1188) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 1188) least significant
s_mul_hi_u32 s19, s24, s28 // (op 1188) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1188) most significant: sum
s_add_u32 s1, s1, s19 // (op 1188) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1188) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 1188) 

s_mov_b32 s20, s26 // (op 1188) reg expression
s_mov_b32 s21, s27 // (op 1188) reg expression
s_mov_b32 s22, s0 // (op 1188) reg expression
s_mov_b32 s23, 131072 // (op 1188) call()

s_mul_hi_i32 s0, s16, s12 // (op 1254) 
v_and_b32 v17, -4, v1 // (op 1254) 
s_add_i32 s1, s0, s16 // (op 1254) 

v_sub_u32 v19, v1, v17 // (op 1254) 

s_ashr_i32 s0, s1, 31 // (op 1254) 
v_lshlrev_b32 v17, 4, v19 // (op 1254) 

s_cmp_eq_i32 s12, 0 // (op 1254) 
s_mov_b32 s18, scc // (op 1254) 
v_lshrrev_b32 v19, 2, v1 // (op 1254) 
s_cmp_eq_i32 s18, 0 // (op 1254) 
s_cselect_b32 s19, 0, -1 // (op 1254) 

v_lshlrev_b32 v21, 3, v19 // (op 1254) 

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



v_lshl_add_u32 v19, s1, 6, v17 // (op 1254) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 1254) 
s_lshr_b32 s18, s0, 1 // (op 1254) 

v_lshl_add_u32 v17, s18, 9, v21 // (op 1254) 


v_mul_lo_u32 v21, v17, s30 // (op 1254) 
v_add_u32 v21, v21, v19 // (op 1254) 


s_mul_i32 s1, s30, s29 // (op 1256) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 1256) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 1256) least significant
s_mul_hi_u32 s19, s30, s28 // (op 1256) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1256) most significant: sum
s_add_u32 s1, s1, s19 // (op 1256) most significant: sum


s_mov_b32 s52, s32 // (op 1256) reg expression
s_mov_b32 s53, s33 // (op 1256) reg expression
s_mov_b32 s54, s0 // (op 1256) reg expression
s_mov_b32 s55, 131072 // (op 1256) call()

s_mul_hi_i32 s0, s16, s12 // (op 1318) 
v_and_b32 v17, -16, v1 // (op 1318) 
s_add_i32 s1, s0, s16 // (op 1318) 

v_sub_u32 v19, v1, v17 // (op 1318) 

s_ashr_i32 s0, s1, 31 // (op 1318) 
v_lshrrev_b32 v17, 4, v1 // (op 1318) 
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



v_lshl_add_u32 v23, s18, 4, v19 // (op 1318) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 1318) 
s_lshr_b32 s18, s0, 1 // (op 1318) 

v_lshl_add_u32 v19, s18, 4, v17 // (op 1318) 


v_mul_lo_u32 v17, v19, s34 // (op 1318) 

v_add_u32 v19, v23, v17 // (op 1318) 


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
v_and_b32 v17, -16, v1 // (op 1382) 
s_add_i32 s1, s0, s16 // (op 1382) 

v_sub_u32 v23, v1, v17 // (op 1382) 

s_ashr_i32 s0, s1, 31 // (op 1382) 
v_lshlrev_b32 v17, 2, v23 // (op 1382) 

s_cmp_eq_i32 s12, 0 // (op 1382) 
s_mov_b32 s18, scc // (op 1382) 
v_lshrrev_b32 v23, 4, v1 // (op 1382) 
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



v_lshl_add_u32 v24, s1, 6, v17 // (op 1382) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 1382) 
s_lshr_b32 s18, s0, 1 // (op 1382) 

v_lshl_add_u32 v17, s18, 4, v23 // (op 1382) 


v_mul_lo_u32 v23, v17, s40 // (op 1382) 
v_add_u32 v23, v23, v24 // (op 1382) 


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

v_lshlrev_b32 v17, 7, v1 // (op 1475) 
v_lshrrev_b32 v24, 3, v17 // (op 1475) 

v_and_b32 v17, -4, v1 // (op 1528) 
v_sub_u32 v25, v1, v17 // (op 1528) 

v_lshlrev_b32 v17, 4, v25 // (op 1528) 

v_lshrrev_b32 v25, 2, v1 // (op 1528) 
v_lshl_add_u32 v26, v25, 9, v17 // (op 1528) 


v_and_b32 v17, -16, v1 // (op 1586) 
v_sub_u32 v25, v1, v17 // (op 1586) 

v_and_b32 v17, 4294967280, v1 // (op 1586) 
v_add_u32 v27, v17, v25 // (op 1586) 


v_and_b32 v17, -16, v1 // (op 1639) 
v_sub_u32 v25, v1, v17 // (op 1639) 

v_lshlrev_b32 v17, 2, v25 // (op 1639) 

v_lshrrev_b32 v25, 4, v1 // (op 1639) 
v_lshl_add_u32 v28, v25, 6, v17 // (op 1639) 


s_ashr_i64 s[0:1], s[28:29], 8 // (op 409) 
s_and_b32 s18, s0, 4294967294 // (op 409) 

s_ashr_i64 s[0:1], s[28:29], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408:

s_add_u32 s18, s18, 1 // (op 410) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 202) 
v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v18, v14 // (op 202) reg expression
buffer_load_dwordx2 v[30:31], v18, s[20:23], 0 offen // (op 202) Load value
v_add_u32 v18, v18, v16 // (op 202) 
buffer_load_dwordx2 v[32:33], v18, s[20:23], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[24:25], 8 // (op 1176) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 1176) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 1176) 

v_add_u32 v15, v15, s0 // (op 1176) 


s_barrier  // (op 1853) 
v_mov_b64 v[16:17], s[30:31] // (op 225) convert




v_mov_b32 v14, v21 // (op 225) 
v_mov_b32 v18, v14 // (op 225) reg expression
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[30:31] offset:16896 // (op 203) Store local data 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx4 v[42:45], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx4 v[46:49], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
s_waitcnt vmcnt(4)
ds_write_b64 v24, v[32:33] offset:16904 // (op 203) Store local data 
buffer_load_dwordx4 v[50:53], v18, s[52:55], 0 offen // (op 225) Load value

v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx4 v[54:57], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx4 v[58:61], v18, s[52:55], 0 offen // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx4 v[62:65], v18, s[52:55], 0 offen // (op 225) Load value



s_lshl_b32 s19, s30, 8 // (op 1246) 
v_add_u32 v21, v21, s19 // (op 1246) 



s_barrier  // (op 1872) 
v_mov_b32 v14, v19 // (op 226) 



v_mov_b32 v20, v14 // (op 226) reg expression

buffer_load_ubyte v22, v20, s[64:67], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s19, s34, 3 // (op 1310) 
v_add_u32 v19, v19, s19 // (op 1310) 

s_waitcnt vmcnt(8)
ds_write_b128 v26, v[34:37] // (op 222) Store local data 


v_mov_b32 v14, v23 // (op 227) 
v_mov_b32 v20, v14 // (op 227) reg expression

buffer_load_dword v25, v20, s[68:71], 0 offen // (op 227) Load value


s_lshl_b32 s19, s40, 3 // (op 1374) 
v_add_u32 v23, v23, s19 // (op 1374) 

s_waitcnt vmcnt(8)
ds_write_b128 v26, v[38:41] offset:64 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1862) 
s_waitcnt vmcnt(7)
ds_write_b128 v26, v[42:45] offset:128 // (op 222) Store local data 



ds_read_b64_tr_b4 v[30:31], v3 offset:16896 // (op 205) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v3 offset:17024 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b128 v26, v[46:49] offset:192 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v26, v[50:53] offset:256 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v26, v[54:57] offset:320 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v26, v[58:61] offset:384 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v26, v[62:65] offset:448 // (op 222) Store local data 

s_barrier  // (op 1893) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1883) 






ds_read_b64_tr_b8 v[34:35], v9 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v9 offset:512 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:4096 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:4608 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v22 offset:18944 // (op 223) Store local data 

s_barrier  // (op 1913) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1903) 





ds_read_u8 v14, v4 offset:18944 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v28, v25 offset:16384 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1923) 


ds_read_u8 v20, v0 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:41], a[0:3], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 212) 





ds_read_b64_tr_b4 v[34:35], v3 offset:17920 // (op 214) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:18048 // (op 214) Transpose load local data 



ds_read_b64_tr_b8 v[38:39], v9 offset:8192 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:8704 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:12288 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:12800 // (op 216) Transpose load local data 


ds_read_u8 v20, v4 offset:19008 // (op 218) Load local data 


ds_read_u8 v22, v0 offset:16640 // (op 220) Load local data 



ds_read_b64_tr_b8 v[46:47], v9 offset:32 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:544 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:4128 // (op 416) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:4640 // (op 416) Transpose load local data 


ds_read_u8 v25, v0 offset:16416 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[46:53], a[4:7], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:45], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 221) 
ds_read_b64_tr_b8 v[46:47], v9 offset:8224 // (op 425) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v9 offset:8736 // (op 425) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:12320 // (op 425) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:12832 // (op 425) Transpose load local data 


ds_read_u8 v14, v0 offset:16672 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[46:53], a[4:7], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 430) 




s_ashr_i64 s[0:1], s[28:29], 8 // (op 408) 
s_mov_b32 s19, s0 // (op 408) convert

s_cmp_lt_u32 s18, s19 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2024) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157
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
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157:


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
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 38144 // lds bytes
  .amdhsa_accum_offset 156
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
  - .name: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 38144
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 156
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

