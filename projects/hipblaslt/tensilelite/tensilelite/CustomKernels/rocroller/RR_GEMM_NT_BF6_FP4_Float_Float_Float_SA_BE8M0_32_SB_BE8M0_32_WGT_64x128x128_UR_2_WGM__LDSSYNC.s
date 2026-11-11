// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 4478) 
s_mul_hi_i32 s1, s16, s12 // (op 4735) 
v_and_b32 v0, 63, v1 // (op 4540) 
v_and_b32 v3, 48, v1 // (op 4540) 
v_sub_u32 v4, v0, v3 // (op 4540) 

v_lshrrev_b32 v3, 4, v0 // (op 4540) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 4540) 


v_lshrrev_b32 v3, 7, v1 // (op 4540) 
v_lshlrev_b32 v4, 4, v3 // (op 4540) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 4540) 


v_mul_lo_u32 v0, v3, 6 // (op 4540) 
v_lshrrev_b32 v4, 3, v0 // (op 4540) 

v_lshrrev_b32 v0, 4, v3 // (op 4540) 

v_lshlrev_b32 v3, 2, v0 // (op 4540) 

v_add_u32 v0, v4, v3 // (op 4540) 


v_and_b32 v3, 63, v1 // (op 4598) 
v_and_b32 v4, 48, v1 // (op 4598) 
v_sub_u32 v5, v3, v4 // (op 4598) 

v_lshrrev_b32 v4, 4, v3 // (op 4598) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 4598) 


v_lshrrev_b32 v4, 6, v1 // (op 4598) 
v_and_b32 v5, 1, v4 // (op 4598) 

v_lshlrev_b32 v4, 4, v5 // (op 4598) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 4598) 


v_lshlrev_b32 v3, 2, v5 // (op 4598) 

v_lshrrev_b32 v4, 3, v3 // (op 4598) 

v_lshrrev_b32 v3, 7, v1 // (op 4656) 
v_and_b32 v5, 15, v1 // (op 4656) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 4656) 


v_and_b32 v3, 63, v1 // (op 4656) 
v_lshrrev_b32 v5, 4, v3 // (op 4656) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 4656) 


v_lshrrev_b32 v5, 6, v1 // (op 4714) 
v_and_b32 v6, 1, v5 // (op 4714) 

v_and_b32 v5, 15, v1 // (op 4714) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4714) 


v_and_b32 v5, 63, v1 // (op 4714) 
v_lshrrev_b32 v6, 4, v5 // (op 4714) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 4714) 


v_and_b32 v6, 63, v1 // (op 4766) 
v_and_b32 v7, 48, v1 // (op 4766) 
v_sub_u32 v8, v6, v7 // (op 4766) 

v_lshrrev_b32 v7, 4, v6 // (op 4766) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 4766) 


v_lshrrev_b32 v7, 7, v1 // (op 4766) 
v_lshlrev_b32 v8, 4, v7 // (op 4766) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 4766) 


v_mul_lo_u32 v6, v7, 6 // (op 4766) 
v_lshrrev_b32 v8, 3, v6 // (op 4766) 

v_lshrrev_b32 v6, 4, v7 // (op 4766) 

v_lshlrev_b32 v7, 2, v6 // (op 4766) 

v_add_u32 v6, v8, v7 // (op 4766) 


v_and_b32 v7, 63, v1 // (op 4795) 
v_and_b32 v8, 48, v1 // (op 4795) 
v_sub_u32 v9, v7, v8 // (op 4795) 

v_lshrrev_b32 v8, 4, v7 // (op 4795) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 4795) 


v_lshrrev_b32 v8, 6, v1 // (op 4795) 
v_and_b32 v9, 1, v8 // (op 4795) 

v_lshlrev_b32 v8, 4, v9 // (op 4795) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 4795) 


v_lshlrev_b32 v7, 2, v9 // (op 4795) 

v_lshrrev_b32 v8, 3, v7 // (op 4795) 

v_lshrrev_b32 v7, 7, v1 // (op 4824) 
v_and_b32 v9, 15, v1 // (op 4824) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 4824) 


v_and_b32 v7, 63, v1 // (op 4824) 
v_lshrrev_b32 v9, 4, v7 // (op 4824) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 4824) 


v_lshrrev_b32 v9, 6, v1 // (op 4853) 
v_and_b32 v10, 1, v9 // (op 4853) 

v_and_b32 v9, 15, v1 // (op 4853) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4853) 


v_and_b32 v9, 63, v1 // (op 4853) 
v_lshrrev_b32 v10, 4, v9 // (op 4853) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 4853) 


v_lshrrev_b32 v10, 7, v1 // (op 4478) 
s_add_i32 s18, s0, s16 // (op 4478) 

v_and_b32 v11, 63, v1 // (op 4478) 
s_ashr_i32 s0, s18, 31 // (op 4478) 
v_lshrrev_b32 v12, 4, v11 // (op 4478) 
s_cmp_eq_i32 s12, 0 // (op 4478) 
s_mov_b32 s19, scc // (op 4478) 
v_lshlrev_b32 v13, 2, v12 // (op 4478) 

s_cmp_eq_i32 s19, 0 // (op 4478) 
s_cselect_b32 s51, 0, -1 // (op 4478) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4478) 


s_and_b32 s19, s15, 31 // (op 4478) 
v_lshrrev_b32 v10, 2, v11 // (op 4478) 

v_and_b32 v11, 3, v10 // (op 4478) 

v_lshrrev_b32 v10, 7, v1 // (op 4735) 
v_and_b32 v13, 63, v1 // (op 4735) 
s_lshl_b32 s52, 1, s19 // (op 4478) 
v_and_b32 v14, 3, v1 // (op 4478) 
s_add_i32 s53, s52, s51 // (op 4478) 


s_and_b32 s51, s0, s53 // (op 4478) 


s_add_i32 s0, s1, s16 // (op 4735) 

s_ashr_i32 s1, s0, 31 // (op 4735) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4478) 


s_add_i32 s52, s18, s51 // (op 4478) 


v_lshrrev_b32 v11, 6, v1 // (op 4478) 
s_ashr_i32 s18, s52, s19 // (op 4478) 


v_and_b32 v14, 1, v11 // (op 4478) 

s_lshl_b32 s19, s15, 24 // (op 4478) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4478) 


s_ashr_i32 s51, s19, 31 // (op 4478) 

s_xor_b32 s19, s18, s51 // (op 4478) 

s_sub_i32 s18, s19, s51 // (op 4478) 


v_lshrrev_b32 v14, 4, v13 // (op 4735) 
s_cmp_eq_i32 s12, 0 // (op 4735) 
s_mov_b32 s19, scc // (op 4735) 
v_lshlrev_b32 v15, 2, v14 // (op 4735) 

s_cmp_eq_i32 s19, 0 // (op 4735) 
s_cselect_b32 s51, 0, -1 // (op 4735) 

v_lshl_add_u32 v14, v10, 4, v15 // (op 4735) 


s_and_b32 s19, s15, 31 // (op 4735) 
v_lshrrev_b32 v10, 2, v13 // (op 4735) 

s_lshl_b32 s52, 1, s19 // (op 4735) 
v_and_b32 v13, 3, v10 // (op 4735) 

s_add_i32 s53, s52, s51 // (op 4735) 


v_and_b32 v10, 3, v1 // (op 4735) 
s_and_b32 s51, s1, s53 // (op 4735) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4735) 


s_add_i32 s1, s0, s51 // (op 4735) 


v_lshrrev_b32 v10, 6, v1 // (op 4735) 
s_ashr_i32 s0, s1, s19 // (op 4735) 


v_and_b32 v13, 1, v10 // (op 4735) 

s_lshl_b32 s1, s15, 24 // (op 4735) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4735) 


s_ashr_i32 s19, s1, 31 // (op 4735) 

s_xor_b32 s1, s0, s19 // (op 4735) 

s_sub_i32 s0, s1, s19 // (op 4735) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 4478) 
s_add_i32 s19, s1, s18 // (op 4478) 

s_ashr_i32 s1, s19, 31 // (op 4478) 
s_cmp_eq_i32 s44, 0 // (op 4478) 
s_mov_b32 s51, scc // (op 4478) 
s_cmp_eq_i32 s51, 0 // (op 4478) 
s_cselect_b32 s52, 0, -1 // (op 4478) 

s_and_b32 s51, s45, 31 // (op 4478) 
s_lshl_b32 s53, 1, s51 // (op 4478) 
s_add_i32 s54, s53, s52 // (op 4478) 


s_and_b32 s52, s1, s54 // (op 4478) 


s_add_i32 s1, s19, s52 // (op 4478) 


s_ashr_i32 s19, s1, s51 // (op 4478) 


s_lshl_b32 s1, s45, 24 // (op 4478) 
s_ashr_i32 s51, s1, 31 // (op 4478) 

s_xor_b32 s1, s19, s51 // (op 4478) 

s_sub_i32 s19, s1, s51 // (op 4478) 


s_mul_i32 s1, s18, s13 // (op 4478) 
s_sub_u32 s51, s16, s1 // (op 4478) 

s_mul_i32 s1, s19, s13 // (op 4478) 
s_add_u32 s1, s1, s51 // (op 4478) 

s_mul_hi_i32 s51, s16, s11 // (op 4478) 
s_add_i32 s52, s51, s16 // (op 4478) 

s_ashr_i32 s51, s52, 31 // (op 4478) 
s_cmp_eq_i32 s11, 0 // (op 4478) 
s_mov_b32 s53, scc // (op 4478) 
s_cmp_eq_i32 s53, 0 // (op 4478) 
s_cselect_b32 s54, 0, -1 // (op 4478) 

s_and_b32 s53, s47, 31 // (op 4478) 
s_lshl_b32 s55, 1, s53 // (op 4478) 
s_add_i32 s56, s55, s54 // (op 4478) 


s_and_b32 s54, s51, s56 // (op 4478) 


s_add_i32 s51, s52, s54 // (op 4478) 


s_ashr_i32 s52, s51, s53 // (op 4478) 


s_lshl_b32 s51, s47, 24 // (op 4478) 
s_ashr_i32 s53, s51, 31 // (op 4478) 

s_xor_b32 s51, s52, s53 // (op 4478) 

s_sub_i32 s52, s51, s53 // (op 4478) 


s_mul_hi_i32 s51, s0, s44 // (op 4735) 
s_add_i32 s53, s51, s0 // (op 4735) 

s_ashr_i32 s51, s53, 31 // (op 4735) 
s_cmp_eq_i32 s44, 0 // (op 4735) 
s_mov_b32 s54, scc // (op 4735) 
s_cmp_eq_i32 s54, 0 // (op 4735) 
s_cselect_b32 s55, 0, -1 // (op 4735) 

s_and_b32 s54, s45, 31 // (op 4735) 
s_lshl_b32 s56, 1, s54 // (op 4735) 
s_add_i32 s57, s56, s55 // (op 4735) 


s_and_b32 s55, s51, s57 // (op 4735) 


s_add_i32 s51, s53, s55 // (op 4735) 


s_ashr_i32 s53, s51, s54 // (op 4735) 


s_lshl_b32 s51, s45, 24 // (op 4735) 
s_ashr_i32 s54, s51, 31 // (op 4735) 

s_xor_b32 s51, s53, s54 // (op 4735) 

s_sub_i32 s53, s51, s54 // (op 4735) 


s_mul_i32 s51, s0, s13 // (op 4735) 
s_sub_u32 s54, s16, s51 // (op 4735) 

s_mul_i32 s51, s53, s13 // (op 4735) 
s_add_u32 s51, s51, s54 // (op 4735) 

s_mul_hi_i32 s54, s16, s11 // (op 4735) 
s_add_i32 s55, s54, s16 // (op 4735) 

s_ashr_i32 s54, s55, 31 // (op 4735) 
s_cmp_eq_i32 s11, 0 // (op 4735) 
s_mov_b32 s56, scc // (op 4735) 
s_cmp_eq_i32 s56, 0 // (op 4735) 
s_cselect_b32 s57, 0, -1 // (op 4735) 

s_and_b32 s56, s47, 31 // (op 4735) 
s_lshl_b32 s58, 1, s56 // (op 4735) 
s_add_i32 s59, s58, s57 // (op 4735) 


s_and_b32 s57, s54, s59 // (op 4735) 


s_add_i32 s54, s55, s57 // (op 4735) 


s_ashr_i32 s55, s54, s56 // (op 4735) 


s_lshl_b32 s54, s47, 24 // (op 4735) 
s_ashr_i32 s56, s54, 31 // (op 4735) 

s_xor_b32 s54, s55, s56 // (op 4735) 

s_sub_i32 s55, s54, s56 // (op 4735) 


s_add_u32 s56, s8, 128 // (op 4478) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 4478) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4478) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4478) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 4478) 

s_mul_i32 s54, s13, s56 // (op 4478) 
s_mul_i32 s58, s10, s54 // (op 4478) 

s_mul_i32 s54, s52, s58 // (op 4478) 

s_sub_u32 s58, s16, s54 // (op 4478) 

s_mov_b32 s54, s58 // (op 4478) 
s_mul_hi_i32 s59, s54, s14 // (op 4478) 
s_add_i32 s60, s59, s54 // (op 4478) 


s_ashr_i32 s54, s60, 31 // (op 4478) 
s_cmp_eq_i32 s14, 0 // (op 4478) 
s_mov_b32 s59, scc // (op 4478) 
s_cmp_eq_i32 s59, 0 // (op 4478) 
s_cselect_b32 s61, 0, -1 // (op 4478) 

s_and_b32 s59, s46, 31 // (op 4478) 
s_lshl_b32 s62, 1, s59 // (op 4478) 
s_add_i32 s63, s62, s61 // (op 4478) 


s_and_b32 s61, s54, s63 // (op 4478) 


s_add_i32 s54, s60, s61 // (op 4478) 


s_ashr_i32 s60, s54, s59 // (op 4478) 


s_lshl_b32 s54, s46, 24 // (op 4478) 
s_ashr_i32 s59, s54, 31 // (op 4478) 

s_xor_b32 s54, s60, s59 // (op 4478) 

s_sub_i32 s60, s54, s59 // (op 4478) 


s_mul_i32 s54, s48, s13 // (op 4478) 
s_add_u32 s62, s6, 64 // (op 4478) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 4478) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4478) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4478) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 4478) 

s_sub_i32 s59, s62, s54 // (op 4478) 

s_mul_i32 s61, s60, s59 // (op 4478) 

s_sub_u32 s59, s58, s61 // (op 4478) 


s_add_u32 s58, s59, s54 // (op 4478) 


s_cmp_eq_i32 s52, 0 // (op 4478) 
s_mov_b32 s54, scc // (op 4478) 

s_cmp_eq_i32 s54, 0 // (op 4478) 
s_cselect_b32 s52, s58, s1 // (op 4478) 


s_mul_i32 s1, s19, s56 // (op 4478) 


v_lshl_add_u32 v13, s52, 6, v12 // (op 4478) 


s_sub_i32 s19, s18, s1 // (op 4478) 


s_cmp_eq_i32 s54, 0 // (op 4478) 
s_cselect_b32 s1, s60, s19 // (op 4478) 



v_lshl_add_u32 v12, s1, 7, v11 // (op 4478) 


v_mul_lo_u32 v11, v12, s2 // (op 4478) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 4478) 


s_add_u32 s18, s8, 128 // (op 4735) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 4735) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 4735) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 4735) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 7 // (op 4735) 

s_mul_i32 s1, s13, s18 // (op 4735) 
s_mul_i32 s52, s10, s1 // (op 4735) 

s_mul_i32 s1, s55, s52 // (op 4735) 

s_sub_u32 s52, s16, s1 // (op 4735) 

s_mov_b32 s1, s52 // (op 4735) 
s_mul_hi_i32 s54, s1, s14 // (op 4735) 
s_add_i32 s56, s54, s1 // (op 4735) 


s_ashr_i32 s1, s56, 31 // (op 4735) 
s_cmp_eq_i32 s14, 0 // (op 4735) 
s_mov_b32 s54, scc // (op 4735) 
s_cmp_eq_i32 s54, 0 // (op 4735) 
s_cselect_b32 s57, 0, -1 // (op 4735) 

s_and_b32 s54, s46, 31 // (op 4735) 
s_lshl_b32 s58, 1, s54 // (op 4735) 
s_add_i32 s59, s58, s57 // (op 4735) 


s_and_b32 s57, s1, s59 // (op 4735) 


s_add_i32 s1, s56, s57 // (op 4735) 


s_ashr_i32 s56, s1, s54 // (op 4735) 


s_lshl_b32 s1, s46, 24 // (op 4735) 
s_ashr_i32 s54, s1, 31 // (op 4735) 

s_xor_b32 s1, s56, s54 // (op 4735) 

s_sub_i32 s56, s1, s54 // (op 4735) 


s_mul_i32 s1, s48, s13 // (op 4735) 
s_add_u32 s58, s6, 64 // (op 4735) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 4735) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 4735) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 4735) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 4735) 

s_sub_i32 s54, s58, s1 // (op 4735) 

s_mul_i32 s57, s56, s54 // (op 4735) 

s_sub_u32 s54, s52, s57 // (op 4735) 


s_add_u32 s52, s54, s1 // (op 4735) 


s_cmp_eq_i32 s55, 0 // (op 4735) 
s_mov_b32 s1, scc // (op 4735) 

s_cmp_eq_i32 s1, 0 // (op 4735) 
s_cselect_b32 s54, s52, s51 // (op 4735) 


s_mul_i32 s55, s53, s18 // (op 4735) 


v_lshl_add_u32 v11, s54, 6, v14 // (op 4735) 


s_sub_i32 s18, s0, s55 // (op 4735) 


s_cmp_eq_i32 s1, 0 // (op 4735) 
s_cselect_b32 s0, s56, s18 // (op 4735) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 4735) 


v_mul_lo_u32 v10, v13, s4 // (op 4735) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 4735) 


s_mul_i32 s1, s2, s9 // (op 4480) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4480) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4480) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4480) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4480) most significant: sum
s_add_u32 s1, s1, s19 // (op 4480) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4480) 

s_mul_i32 s1, s4, s9 // (op 4737) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 4737) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4737) least significant
s_mul_hi_u32 s52, s4, s8 // (op 4737) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4737) most significant: sum
s_add_u32 s1, s1, s52 // (op 4737) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 4737) 

s_mov_b32 s56, s20 // (op 4480) reg expression
s_mov_b32 s57, s21 // (op 4480) reg expression
s_mov_b32 s58, s18 // (op 4480) reg expression
s_mov_b32 s59, 131072 // (op 4480) call()

s_mov_b32 s60, s22 // (op 4737) reg expression

s_mov_b32 s61, s23 // (op 4737) reg expression
s_mov_b32 s62, s52 // (op 4737) reg expression
s_mov_b32 s63, 131072 // (op 4737) call()


v_mov_b32 v10, s49 // (op 2) Move value
v_mov_b32 v11, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 512) call()
v_accvgpr_write a9, 0.00000 // (op 512) call()
v_accvgpr_write a10, 0.00000 // (op 512) call()
v_accvgpr_write a11, 0.00000 // (op 512) call()
v_accvgpr_write a12, 0.00000 // (op 688) call()
v_accvgpr_write a13, 0.00000 // (op 688) call()
v_accvgpr_write a14, 0.00000 // (op 688) call()
v_accvgpr_write a15, 0.00000 // (op 688) call()
v_accvgpr_write a16, 0.00000 // (op 952) call()
v_accvgpr_write a17, 0.00000 // (op 952) call()
v_accvgpr_write a18, 0.00000 // (op 952) call()
v_accvgpr_write a19, 0.00000 // (op 952) call()
v_accvgpr_write a20, 0.00000 // (op 1261) call()
v_accvgpr_write a21, 0.00000 // (op 1261) call()
v_accvgpr_write a22, 0.00000 // (op 1261) call()
v_accvgpr_write a23, 0.00000 // (op 1261) call()
v_accvgpr_write a24, 0.00000 // (op 1262) call()
v_accvgpr_write a25, 0.00000 // (op 1262) call()
v_accvgpr_write a26, 0.00000 // (op 1262) call()
v_accvgpr_write a27, 0.00000 // (op 1262) call()
v_accvgpr_write a28, 0.00000 // (op 1263) call()
v_accvgpr_write a29, 0.00000 // (op 1263) call()
v_accvgpr_write a30, 0.00000 // (op 1263) call()
v_accvgpr_write a31, 0.00000 // (op 1263) call()
s_mul_hi_i32 s0, s16, s12 // (op 4240) 
v_and_b32 v14, -2, v1 // (op 4240) 
s_add_i32 s1, s0, s16 // (op 4240) 

v_sub_u32 v15, v1, v14 // (op 4240) 

s_ashr_i32 s0, s1, 31 // (op 4240) 
v_lshlrev_b32 v14, 5, v15 // (op 4240) 

s_cmp_eq_i32 s12, 0 // (op 4240) 
s_mov_b32 s18, scc // (op 4240) 
v_lshrrev_b32 v15, 1, v1 // (op 4240) 
s_cmp_eq_i32 s18, 0 // (op 4240) 
s_cselect_b32 s19, 0, -1 // (op 4240) 

s_and_b32 s18, s15, 31 // (op 4240) 
s_lshl_b32 s20, 1, s18 // (op 4240) 
s_add_i32 s21, s20, s19 // (op 4240) 


s_and_b32 s19, s0, s21 // (op 4240) 


s_add_i32 s0, s1, s19 // (op 4240) 


s_ashr_i32 s1, s0, s18 // (op 4240) 


s_lshl_b32 s0, s15, 24 // (op 4240) 
s_ashr_i32 s18, s0, 31 // (op 4240) 

s_xor_b32 s0, s1, s18 // (op 4240) 

s_sub_i32 s1, s0, s18 // (op 4240) 


s_mul_hi_i32 s0, s1, s44 // (op 4240) 
s_add_i32 s18, s0, s1 // (op 4240) 

s_ashr_i32 s0, s18, 31 // (op 4240) 
s_cmp_eq_i32 s44, 0 // (op 4240) 
s_mov_b32 s19, scc // (op 4240) 
s_cmp_eq_i32 s19, 0 // (op 4240) 
s_cselect_b32 s20, 0, -1 // (op 4240) 

s_and_b32 s19, s45, 31 // (op 4240) 
s_lshl_b32 s21, 1, s19 // (op 4240) 
s_add_i32 s22, s21, s20 // (op 4240) 


s_and_b32 s23, s0, s22 // (op 4240) 


s_add_i32 s0, s18, s23 // (op 4240) 


s_ashr_i32 s18, s0, s19 // (op 4240) 


s_lshl_b32 s0, s45, 24 // (op 4240) 
s_ashr_i32 s19, s0, 31 // (op 4240) 

s_xor_b32 s0, s18, s19 // (op 4240) 

s_sub_i32 s18, s0, s19 // (op 4240) 


s_mul_i32 s0, s1, s13 // (op 4240) 

s_sub_u32 s1, s16, s0 // (op 4240) 

s_mul_i32 s0, s18, s13 // (op 4240) 
s_add_u32 s0, s0, s1 // (op 4240) 


s_mul_hi_i32 s1, s16, s11 // (op 4240) 
s_add_i32 s18, s1, s16 // (op 4240) 

s_ashr_i32 s1, s18, 31 // (op 4240) 
s_cmp_eq_i32 s11, 0 // (op 4240) 
s_mov_b32 s19, scc // (op 4240) 
s_cmp_eq_i32 s19, 0 // (op 4240) 
s_cselect_b32 s20, 0, -1 // (op 4240) 

s_and_b32 s19, s47, 31 // (op 4240) 
s_lshl_b32 s21, 1, s19 // (op 4240) 
s_add_i32 s22, s21, s20 // (op 4240) 


s_and_b32 s23, s1, s22 // (op 4240) 


s_add_i32 s1, s18, s23 // (op 4240) 


s_ashr_i32 s18, s1, s19 // (op 4240) 


s_lshl_b32 s1, s47, 24 // (op 4240) 
s_ashr_i32 s19, s1, 31 // (op 4240) 

s_xor_b32 s1, s18, s19 // (op 4240) 

s_sub_i32 s18, s1, s19 // (op 4240) 


s_add_u32 s22, s8, 128 // (op 4240) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 4240) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 4240) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 4240) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 4240) 

s_mul_i32 s1, s13, s22 // (op 4240) 

s_mul_i32 s19, s10, s1 // (op 4240) 

s_mul_i32 s1, s18, s19 // (op 4240) 

s_sub_u32 s19, s16, s1 // (op 4240) 

s_mov_b32 s1, s19 // (op 4240) 
s_mul_hi_i32 s20, s1, s14 // (op 4240) 
s_add_i32 s21, s20, s1 // (op 4240) 


s_ashr_i32 s1, s21, 31 // (op 4240) 
s_cmp_eq_i32 s14, 0 // (op 4240) 
s_mov_b32 s20, scc // (op 4240) 
s_cmp_eq_i32 s20, 0 // (op 4240) 
s_cselect_b32 s22, 0, -1 // (op 4240) 

s_and_b32 s20, s46, 31 // (op 4240) 
s_lshl_b32 s23, 1, s20 // (op 4240) 
s_add_i32 s49, s23, s22 // (op 4240) 


s_and_b32 s22, s1, s49 // (op 4240) 


s_add_i32 s1, s21, s22 // (op 4240) 


s_ashr_i32 s21, s1, s20 // (op 4240) 


s_lshl_b32 s1, s46, 24 // (op 4240) 
s_ashr_i32 s20, s1, 31 // (op 4240) 

s_xor_b32 s1, s21, s20 // (op 4240) 

s_sub_i32 s21, s1, s20 // (op 4240) 


s_mul_i32 s1, s48, s13 // (op 4240) 
s_add_u32 s22, s6, 64 // (op 4240) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4240) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 4240) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 4240) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 6 // (op 4240) 

s_sub_i32 s20, s22, s1 // (op 4240) 

s_mul_i32 s22, s21, s20 // (op 4240) 


s_sub_u32 s23, s19, s22 // (op 4240) 


s_add_u32 s19, s23, s1 // (op 4240) 


s_cmp_eq_i32 s18, 0 // (op 4240) 
s_mov_b32 s1, scc // (op 4240) 

s_cmp_eq_i32 s1, 0 // (op 4240) 
s_cselect_b32 s18, s19, s0 // (op 4240) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 4240) 


v_mov_b32 v19, 0 // (op 4240) convert
v_mov_b32 v18, v16 // (op 4240) convert

v_mul_lo_u32 v17, v15, s25 // (op 4240) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4240) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 4240) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 4240) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 4240) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 4240) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4240) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 4240) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4240) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 4240) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4240) 

v_mov_b32 v16, v14 // (op 4240) convert

s_mul_i32 s1, s29, 6 // (op 4242) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4242) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4242) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4242) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4242) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 4242) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4242) least significant
s_mul_hi_u32 s21, s24, s0 // (op 4242) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 4242) most significant: sum
s_add_u32 s19, s19, s21 // (op 4242) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4242) 

s_mov_b32 s20, s26 // (op 4242) reg expression
s_mov_b32 s21, s27 // (op 4242) reg expression
s_mov_b32 s22, s0 // (op 4242) reg expression
s_mov_b32 s23, 131072 // (op 4242) call()

s_mul_hi_i32 s0, s16, s12 // (op 4259) 
v_and_b32 v14, -2, v1 // (op 4259) 
s_add_i32 s1, s0, s16 // (op 4259) 

v_sub_u32 v15, v1, v14 // (op 4259) 

s_ashr_i32 s0, s1, 31 // (op 4259) 
v_lshlrev_b32 v14, 5, v15 // (op 4259) 

s_cmp_eq_i32 s12, 0 // (op 4259) 
s_mov_b32 s18, scc // (op 4259) 
v_lshrrev_b32 v15, 1, v1 // (op 4259) 
s_cmp_eq_i32 s18, 0 // (op 4259) 
s_cselect_b32 s19, 0, -1 // (op 4259) 

s_and_b32 s18, s15, 31 // (op 4259) 
s_lshl_b32 s49, 1, s18 // (op 4259) 
s_add_i32 s50, s49, s19 // (op 4259) 


s_and_b32 s19, s0, s50 // (op 4259) 


s_add_i32 s0, s1, s19 // (op 4259) 


s_ashr_i32 s1, s0, s18 // (op 4259) 


s_lshl_b32 s0, s15, 24 // (op 4259) 
s_ashr_i32 s18, s0, 31 // (op 4259) 

s_xor_b32 s0, s1, s18 // (op 4259) 

s_sub_i32 s1, s0, s18 // (op 4259) 


s_mul_hi_i32 s0, s1, s44 // (op 4259) 
s_add_i32 s18, s0, s1 // (op 4259) 

s_ashr_i32 s0, s18, 31 // (op 4259) 
s_cmp_eq_i32 s44, 0 // (op 4259) 
s_mov_b32 s19, scc // (op 4259) 
s_cmp_eq_i32 s19, 0 // (op 4259) 
s_cselect_b32 s49, 0, -1 // (op 4259) 

s_and_b32 s19, s45, 31 // (op 4259) 
s_lshl_b32 s50, 1, s19 // (op 4259) 
s_add_i32 s51, s50, s49 // (op 4259) 


s_and_b32 s49, s0, s51 // (op 4259) 


s_add_i32 s0, s18, s49 // (op 4259) 


s_ashr_i32 s18, s0, s19 // (op 4259) 


s_lshl_b32 s0, s45, 24 // (op 4259) 
s_ashr_i32 s19, s0, 31 // (op 4259) 

s_xor_b32 s0, s18, s19 // (op 4259) 

s_sub_i32 s18, s0, s19 // (op 4259) 


s_mul_i32 s0, s1, s13 // (op 4259) 

s_sub_u32 s1, s16, s0 // (op 4259) 

s_mul_i32 s0, s18, s13 // (op 4259) 
s_add_u32 s0, s0, s1 // (op 4259) 


s_mul_hi_i32 s1, s16, s11 // (op 4259) 
s_add_i32 s18, s1, s16 // (op 4259) 

s_ashr_i32 s1, s18, 31 // (op 4259) 
s_cmp_eq_i32 s11, 0 // (op 4259) 
s_mov_b32 s19, scc // (op 4259) 
s_cmp_eq_i32 s19, 0 // (op 4259) 
s_cselect_b32 s49, 0, -1 // (op 4259) 

s_and_b32 s19, s47, 31 // (op 4259) 
s_lshl_b32 s50, 1, s19 // (op 4259) 
s_add_i32 s51, s50, s49 // (op 4259) 


s_and_b32 s49, s1, s51 // (op 4259) 


s_add_i32 s1, s18, s49 // (op 4259) 


s_ashr_i32 s18, s1, s19 // (op 4259) 


s_lshl_b32 s1, s47, 24 // (op 4259) 
s_ashr_i32 s19, s1, 31 // (op 4259) 

s_xor_b32 s1, s18, s19 // (op 4259) 

s_sub_i32 s18, s1, s19 // (op 4259) 


s_add_u32 s54, s8, 128 // (op 4259) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 4259) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4259) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4259) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 4259) 

s_mul_i32 s1, s13, s50 // (op 4259) 

s_mul_i32 s19, s10, s1 // (op 4259) 

s_mul_i32 s1, s18, s19 // (op 4259) 

s_sub_u32 s19, s16, s1 // (op 4259) 

s_mov_b32 s1, s19 // (op 4259) 
s_mul_hi_i32 s49, s1, s14 // (op 4259) 
s_add_i32 s50, s49, s1 // (op 4259) 


s_ashr_i32 s1, s50, 31 // (op 4259) 
s_cmp_eq_i32 s14, 0 // (op 4259) 
s_mov_b32 s49, scc // (op 4259) 
s_cmp_eq_i32 s49, 0 // (op 4259) 
s_cselect_b32 s51, 0, -1 // (op 4259) 

s_and_b32 s49, s46, 31 // (op 4259) 
s_lshl_b32 s52, 1, s49 // (op 4259) 
s_add_i32 s53, s52, s51 // (op 4259) 


s_and_b32 s51, s1, s53 // (op 4259) 


s_add_i32 s1, s50, s51 // (op 4259) 


s_ashr_i32 s50, s1, s49 // (op 4259) 


s_lshl_b32 s1, s46, 24 // (op 4259) 
s_ashr_i32 s49, s1, 31 // (op 4259) 

s_xor_b32 s1, s50, s49 // (op 4259) 

s_sub_i32 s50, s1, s49 // (op 4259) 


s_mul_i32 s1, s48, s13 // (op 4259) 
s_add_u32 s54, s6, 64 // (op 4259) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 4259) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4259) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4259) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 4259) 

s_sub_i32 s49, s54, s1 // (op 4259) 

s_mul_i32 s51, s50, s49 // (op 4259) 


s_sub_u32 s49, s19, s51 // (op 4259) 


s_add_u32 s19, s49, s1 // (op 4259) 


s_cmp_eq_i32 s18, 0 // (op 4259) 
s_mov_b32 s1, scc // (op 4259) 

s_cmp_eq_i32 s1, 0 // (op 4259) 
s_cselect_b32 s18, s19, s0 // (op 4259) 



v_lshl_add_u32 v17, s18, 6, v14 // (op 4259) 


v_mov_b32 v19, 0 // (op 4259) convert
v_mov_b32 v18, v17 // (op 4259) convert

v_mul_lo_u32 v21, v15, s25 // (op 4259) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4259) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 4259) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4259) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 4259) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 4259) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 4259) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 4259) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 4259) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 4259) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 4259) 

v_mov_b32 v17, v14 // (op 4259) convert

s_mul_i32 s1, s29, 6 // (op 4261) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4261) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4261) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4261) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4261) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4261) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4261) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4261) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4261) most significant: sum
s_add_u32 s19, s19, s50 // (op 4261) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4261) 

s_mov_b32 s20, s26 // (op 4261) reg expression
s_mov_b32 s21, s27 // (op 4261) reg expression
s_mov_b32 s22, s0 // (op 4261) reg expression
s_mov_b32 s23, 131072 // (op 4261) call()

s_mul_hi_i32 s0, s16, s12 // (op 4304) 
v_and_b32 v14, -4, v1 // (op 4304) 
s_add_i32 s1, s0, s16 // (op 4304) 

v_sub_u32 v15, v1, v14 // (op 4304) 

s_ashr_i32 s0, s1, 31 // (op 4304) 
v_lshlrev_b32 v14, 5, v15 // (op 4304) 

s_cmp_eq_i32 s12, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
v_lshrrev_b32 v15, 2, v1 // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s19, 0, -1 // (op 4304) 

v_lshlrev_b32 v18, 1, v15 // (op 4304) 

s_and_b32 s18, s15, 31 // (op 4304) 
s_lshl_b32 s49, 1, s18 // (op 4304) 
s_add_i32 s50, s49, s19 // (op 4304) 


s_and_b32 s19, s0, s50 // (op 4304) 


s_add_i32 s0, s1, s19 // (op 4304) 


s_ashr_i32 s1, s0, s18 // (op 4304) 


s_lshl_b32 s0, s15, 24 // (op 4304) 
s_ashr_i32 s18, s0, 31 // (op 4304) 

s_xor_b32 s0, s1, s18 // (op 4304) 

s_sub_i32 s1, s0, s18 // (op 4304) 


s_mul_hi_i32 s0, s1, s44 // (op 4304) 
s_add_i32 s18, s0, s1 // (op 4304) 

s_ashr_i32 s0, s18, 31 // (op 4304) 
s_cmp_eq_i32 s44, 0 // (op 4304) 
s_mov_b32 s19, scc // (op 4304) 
s_cmp_eq_i32 s19, 0 // (op 4304) 
s_cselect_b32 s49, 0, -1 // (op 4304) 

s_and_b32 s19, s45, 31 // (op 4304) 
s_lshl_b32 s50, 1, s19 // (op 4304) 
s_add_i32 s51, s50, s49 // (op 4304) 


s_and_b32 s49, s0, s51 // (op 4304) 


s_add_i32 s0, s18, s49 // (op 4304) 


s_ashr_i32 s18, s0, s19 // (op 4304) 


s_lshl_b32 s0, s45, 24 // (op 4304) 
s_ashr_i32 s19, s0, 31 // (op 4304) 

s_xor_b32 s0, s18, s19 // (op 4304) 

s_sub_i32 s18, s0, s19 // (op 4304) 


s_add_u32 s54, s8, 128 // (op 4304) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 4304) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4304) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4304) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 4304) 

s_mul_i32 s0, s18, s50 // (op 4304) 

s_sub_i32 s49, s1, s0 // (op 4304) 


s_mul_hi_i32 s0, s16, s11 // (op 4304) 
s_add_i32 s1, s0, s16 // (op 4304) 

s_ashr_i32 s0, s1, 31 // (op 4304) 
s_cmp_eq_i32 s11, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s19, 0, -1 // (op 4304) 

s_and_b32 s18, s47, 31 // (op 4304) 
s_lshl_b32 s52, 1, s18 // (op 4304) 
s_add_i32 s53, s52, s19 // (op 4304) 


s_and_b32 s19, s0, s53 // (op 4304) 


s_add_i32 s0, s1, s19 // (op 4304) 


s_ashr_i32 s1, s0, s18 // (op 4304) 


s_lshl_b32 s0, s47, 24 // (op 4304) 
s_ashr_i32 s18, s0, 31 // (op 4304) 

s_xor_b32 s0, s1, s18 // (op 4304) 

s_sub_i32 s1, s0, s18 // (op 4304) 


s_mul_i32 s0, s13, s50 // (op 4304) 

s_mul_i32 s18, s10, s0 // (op 4304) 

s_mul_i32 s0, s1, s18 // (op 4304) 

s_sub_u32 s18, s16, s0 // (op 4304) 

s_mul_hi_i32 s0, s18, s14 // (op 4304) 
s_add_i32 s19, s0, s18 // (op 4304) 


s_ashr_i32 s0, s19, 31 // (op 4304) 
s_cmp_eq_i32 s14, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s50, 0, -1 // (op 4304) 

s_and_b32 s18, s46, 31 // (op 4304) 
s_lshl_b32 s51, 1, s18 // (op 4304) 
s_add_i32 s52, s51, s50 // (op 4304) 


s_and_b32 s50, s0, s52 // (op 4304) 


s_add_i32 s0, s19, s50 // (op 4304) 


s_ashr_i32 s19, s0, s18 // (op 4304) 


s_lshl_b32 s0, s46, 24 // (op 4304) 
s_ashr_i32 s18, s0, 31 // (op 4304) 

s_xor_b32 s0, s19, s18 // (op 4304) 

s_sub_i32 s19, s0, s18 // (op 4304) 


s_cmp_eq_i32 s1, 0 // (op 4304) 
s_mov_b32 s0, scc // (op 4304) 

s_cmp_eq_i32 s0, 0 // (op 4304) 
s_cselect_b32 s1, s19, s49 // (op 4304) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4304) 


v_mov_b32 v21, 0 // (op 4304) convert
v_mov_b32 v20, v15 // (op 4304) convert

v_mul_lo_u32 v15, v18, s31 // (op 4304) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 4304) most significant: high of low * low
v_mul_lo_u32 v14, v18, s30 // (op 4304) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 4304) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 4304) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 4304) most significant half



v_lshlrev_b64 v[18:19], 2, v[14:15] // (op 4304) 

v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 4304) 

v_mov_b32 v18, v14 // (op 4304) convert

s_mul_i32 s1, s30, s29 // (op 4306) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4306) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4306) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4306) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4306) most significant: sum
s_add_u32 s1, s1, s19 // (op 4306) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4306) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4306) 

s_mov_b32 s52, s32 // (op 4306) reg expression
s_mov_b32 s53, s33 // (op 4306) reg expression
s_mov_b32 s54, s0 // (op 4306) reg expression
s_mov_b32 s55, 131072 // (op 4306) call()

s_mul_hi_i32 s0, s16, s12 // (op 4323) 
v_and_b32 v14, -4, v1 // (op 4323) 
s_add_i32 s1, s0, s16 // (op 4323) 

v_sub_u32 v15, v1, v14 // (op 4323) 

s_ashr_i32 s0, s1, 31 // (op 4323) 
v_lshlrev_b32 v14, 5, v15 // (op 4323) 

s_cmp_eq_i32 s12, 0 // (op 4323) 
s_mov_b32 s18, scc // (op 4323) 
v_lshrrev_b32 v15, 2, v1 // (op 4323) 
s_cmp_eq_i32 s18, 0 // (op 4323) 
s_cselect_b32 s19, 0, -1 // (op 4323) 

v_lshlrev_b32 v19, 1, v15 // (op 4323) 

s_and_b32 s18, s15, 31 // (op 4323) 
s_lshl_b32 s49, 1, s18 // (op 4323) 
s_add_i32 s50, s49, s19 // (op 4323) 


s_and_b32 s19, s0, s50 // (op 4323) 


s_add_i32 s0, s1, s19 // (op 4323) 


s_ashr_i32 s1, s0, s18 // (op 4323) 


s_lshl_b32 s0, s15, 24 // (op 4323) 
s_ashr_i32 s18, s0, 31 // (op 4323) 

s_xor_b32 s0, s1, s18 // (op 4323) 

s_sub_i32 s1, s0, s18 // (op 4323) 


s_mul_hi_i32 s0, s1, s44 // (op 4323) 
s_add_i32 s18, s0, s1 // (op 4323) 

s_ashr_i32 s0, s18, 31 // (op 4323) 
s_cmp_eq_i32 s44, 0 // (op 4323) 
s_mov_b32 s19, scc // (op 4323) 
s_cmp_eq_i32 s19, 0 // (op 4323) 
s_cselect_b32 s49, 0, -1 // (op 4323) 

s_and_b32 s19, s45, 31 // (op 4323) 
s_lshl_b32 s50, 1, s19 // (op 4323) 
s_add_i32 s51, s50, s49 // (op 4323) 


s_and_b32 s49, s0, s51 // (op 4323) 


s_add_i32 s0, s18, s49 // (op 4323) 


s_ashr_i32 s18, s0, s19 // (op 4323) 


s_lshl_b32 s0, s45, 24 // (op 4323) 
s_ashr_i32 s19, s0, 31 // (op 4323) 

s_xor_b32 s0, s18, s19 // (op 4323) 

s_sub_i32 s18, s0, s19 // (op 4323) 


s_add_u32 s50, s8, 128 // (op 4323) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4323) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4323) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4323) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 4323) 

s_mul_i32 s0, s18, s50 // (op 4323) 

s_sub_i32 s49, s1, s0 // (op 4323) 


s_mul_hi_i32 s0, s16, s11 // (op 4323) 
s_add_i32 s1, s0, s16 // (op 4323) 

s_ashr_i32 s0, s1, 31 // (op 4323) 
s_cmp_eq_i32 s11, 0 // (op 4323) 
s_mov_b32 s18, scc // (op 4323) 
s_cmp_eq_i32 s18, 0 // (op 4323) 
s_cselect_b32 s19, 0, -1 // (op 4323) 

s_and_b32 s18, s47, 31 // (op 4323) 
s_lshl_b32 s64, 1, s18 // (op 4323) 
s_add_i32 s65, s64, s19 // (op 4323) 


s_and_b32 s19, s0, s65 // (op 4323) 


s_add_i32 s0, s1, s19 // (op 4323) 


s_ashr_i32 s1, s0, s18 // (op 4323) 


s_lshl_b32 s0, s47, 24 // (op 4323) 
s_ashr_i32 s18, s0, 31 // (op 4323) 

s_xor_b32 s0, s1, s18 // (op 4323) 

s_sub_i32 s1, s0, s18 // (op 4323) 


s_mul_i32 s0, s13, s50 // (op 4323) 

s_mul_i32 s18, s10, s0 // (op 4323) 

s_mul_i32 s0, s1, s18 // (op 4323) 

s_sub_u32 s18, s16, s0 // (op 4323) 

s_mul_hi_i32 s0, s18, s14 // (op 4323) 
s_add_i32 s19, s0, s18 // (op 4323) 


s_ashr_i32 s0, s19, 31 // (op 4323) 
s_cmp_eq_i32 s14, 0 // (op 4323) 
s_mov_b32 s18, scc // (op 4323) 
s_cmp_eq_i32 s18, 0 // (op 4323) 
s_cselect_b32 s50, 0, -1 // (op 4323) 

s_and_b32 s18, s46, 31 // (op 4323) 
s_lshl_b32 s51, 1, s18 // (op 4323) 
s_add_i32 s64, s51, s50 // (op 4323) 


s_and_b32 s50, s0, s64 // (op 4323) 


s_add_i32 s0, s19, s50 // (op 4323) 


s_ashr_i32 s19, s0, s18 // (op 4323) 


s_lshl_b32 s0, s46, 24 // (op 4323) 
s_ashr_i32 s18, s0, 31 // (op 4323) 

s_xor_b32 s0, s19, s18 // (op 4323) 

s_sub_i32 s19, s0, s18 // (op 4323) 


s_cmp_eq_i32 s1, 0 // (op 4323) 
s_mov_b32 s0, scc // (op 4323) 

s_cmp_eq_i32 s0, 0 // (op 4323) 
s_cselect_b32 s1, s19, s49 // (op 4323) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4323) 


v_mov_b32 v21, 0 // (op 4323) convert
v_mov_b32 v20, v15 // (op 4323) convert

v_mul_lo_u32 v15, v19, s31 // (op 4323) most significant: low of low * high
v_mul_hi_u32 v22, v19, s30 // (op 4323) most significant: high of low * low
v_mul_lo_u32 v14, v19, s30 // (op 4323) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 4323) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 4323) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 4323) most significant half



v_lshlrev_b64 v[20:21], 2, v[14:15] // (op 4323) 

v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 4323) 

v_mov_b32 v19, v14 // (op 4323) convert

s_mul_i32 s1, s30, s29 // (op 4325) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4325) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4325) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4325) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4325) most significant: sum
s_add_u32 s1, s1, s19 // (op 4325) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4325) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4325) 

s_mov_b32 s52, s32 // (op 4325) reg expression
s_mov_b32 s53, s33 // (op 4325) reg expression
s_mov_b32 s54, s0 // (op 4325) reg expression
s_mov_b32 s55, 131072 // (op 4325) call()

s_mul_hi_i32 s0, s16, s12 // (op 4368) 
v_and_b32 v14, -64, v1 // (op 4368) 
s_add_i32 s1, s0, s16 // (op 4368) 

v_sub_u32 v15, v1, v14 // (op 4368) 

s_ashr_i32 s0, s1, 31 // (op 4368) 
v_lshrrev_b32 v14, 6, v1 // (op 4368) 
s_cmp_eq_i32 s12, 0 // (op 4368) 
s_mov_b32 s18, scc // (op 4368) 
s_cmp_eq_i32 s18, 0 // (op 4368) 
s_cselect_b32 s19, 0, -1 // (op 4368) 

s_and_b32 s18, s15, 31 // (op 4368) 
s_lshl_b32 s49, 1, s18 // (op 4368) 
s_add_i32 s50, s49, s19 // (op 4368) 


s_and_b32 s19, s0, s50 // (op 4368) 


s_add_i32 s0, s1, s19 // (op 4368) 


s_ashr_i32 s1, s0, s18 // (op 4368) 


s_lshl_b32 s0, s15, 24 // (op 4368) 
s_ashr_i32 s18, s0, 31 // (op 4368) 

s_xor_b32 s0, s1, s18 // (op 4368) 

s_sub_i32 s1, s0, s18 // (op 4368) 


s_mul_hi_i32 s0, s1, s44 // (op 4368) 
s_add_i32 s18, s0, s1 // (op 4368) 

s_ashr_i32 s0, s18, 31 // (op 4368) 
s_cmp_eq_i32 s44, 0 // (op 4368) 
s_mov_b32 s19, scc // (op 4368) 
s_cmp_eq_i32 s19, 0 // (op 4368) 
s_cselect_b32 s49, 0, -1 // (op 4368) 

s_and_b32 s19, s45, 31 // (op 4368) 
s_lshl_b32 s50, 1, s19 // (op 4368) 
s_add_i32 s51, s50, s49 // (op 4368) 


s_and_b32 s49, s0, s51 // (op 4368) 


s_add_i32 s0, s18, s49 // (op 4368) 


s_ashr_i32 s18, s0, s19 // (op 4368) 


s_lshl_b32 s0, s45, 24 // (op 4368) 
s_ashr_i32 s19, s0, 31 // (op 4368) 

s_xor_b32 s0, s18, s19 // (op 4368) 

s_sub_i32 s18, s0, s19 // (op 4368) 


s_mul_i32 s0, s1, s13 // (op 4368) 

s_sub_u32 s1, s16, s0 // (op 4368) 

s_mul_i32 s0, s18, s13 // (op 4368) 
s_add_u32 s0, s0, s1 // (op 4368) 


s_mul_hi_i32 s1, s16, s11 // (op 4368) 
s_add_i32 s18, s1, s16 // (op 4368) 

s_ashr_i32 s1, s18, 31 // (op 4368) 
s_cmp_eq_i32 s11, 0 // (op 4368) 
s_mov_b32 s19, scc // (op 4368) 
s_cmp_eq_i32 s19, 0 // (op 4368) 
s_cselect_b32 s49, 0, -1 // (op 4368) 

s_and_b32 s19, s47, 31 // (op 4368) 
s_lshl_b32 s50, 1, s19 // (op 4368) 
s_add_i32 s51, s50, s49 // (op 4368) 


s_and_b32 s49, s1, s51 // (op 4368) 


s_add_i32 s1, s18, s49 // (op 4368) 


s_ashr_i32 s18, s1, s19 // (op 4368) 


s_lshl_b32 s1, s47, 24 // (op 4368) 
s_ashr_i32 s19, s1, 31 // (op 4368) 

s_xor_b32 s1, s18, s19 // (op 4368) 

s_sub_i32 s18, s1, s19 // (op 4368) 


s_add_u32 s50, s8, 128 // (op 4368) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4368) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4368) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4368) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 4368) 

s_mul_i32 s1, s13, s50 // (op 4368) 

s_mul_i32 s19, s10, s1 // (op 4368) 

s_mul_i32 s1, s18, s19 // (op 4368) 

s_sub_u32 s19, s16, s1 // (op 4368) 

s_mov_b32 s1, s19 // (op 4368) 
s_mul_hi_i32 s49, s1, s14 // (op 4368) 
s_add_i32 s50, s49, s1 // (op 4368) 


s_ashr_i32 s1, s50, 31 // (op 4368) 
s_cmp_eq_i32 s14, 0 // (op 4368) 
s_mov_b32 s49, scc // (op 4368) 
s_cmp_eq_i32 s49, 0 // (op 4368) 
s_cselect_b32 s51, 0, -1 // (op 4368) 

s_and_b32 s49, s46, 31 // (op 4368) 
s_lshl_b32 s64, 1, s49 // (op 4368) 
s_add_i32 s65, s64, s51 // (op 4368) 


s_and_b32 s51, s1, s65 // (op 4368) 


s_add_i32 s1, s50, s51 // (op 4368) 


s_ashr_i32 s50, s1, s49 // (op 4368) 


s_lshl_b32 s1, s46, 24 // (op 4368) 
s_ashr_i32 s49, s1, 31 // (op 4368) 

s_xor_b32 s1, s50, s49 // (op 4368) 

s_sub_i32 s50, s1, s49 // (op 4368) 


s_mul_i32 s1, s48, s13 // (op 4368) 
s_add_u32 s64, s6, 64 // (op 4368) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 4368) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 4368) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 4368) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 4368) 

s_sub_i32 s49, s64, s1 // (op 4368) 

s_mul_i32 s51, s50, s49 // (op 4368) 


s_sub_u32 s49, s19, s51 // (op 4368) 


s_add_u32 s19, s49, s1 // (op 4368) 


s_cmp_eq_i32 s18, 0 // (op 4368) 
s_mov_b32 s1, scc // (op 4368) 

s_cmp_eq_i32 s1, 0 // (op 4368) 
s_cselect_b32 s18, s19, s0 // (op 4368) 



v_lshl_add_u32 v20, s18, 6, v15 // (op 4368) 


v_mul_lo_u32 v15, v14, s34 // (op 4368) 

v_add_u32 v14, v20, v15 // (op 4368) 


s_mul_i32 s1, s34, s39 // (op 4370) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4370) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4370) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4370) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4370) most significant: sum
s_add_u32 s1, s1, s19 // (op 4370) most significant: sum


s_mov_b32 s64, s36 // (op 4370) reg expression
s_mov_b32 s65, s37 // (op 4370) reg expression
s_mov_b32 s66, s0 // (op 4370) reg expression
s_mov_b32 s67, 131072 // (op 4370) call()

s_mul_hi_i32 s0, s16, s12 // (op 4387) 
v_and_b32 v15, -64, v1 // (op 4387) 
s_add_i32 s1, s0, s16 // (op 4387) 

v_sub_u32 v20, v1, v15 // (op 4387) 

s_ashr_i32 s0, s1, 31 // (op 4387) 
v_lshrrev_b32 v15, 6, v1 // (op 4387) 
s_cmp_eq_i32 s12, 0 // (op 4387) 
s_mov_b32 s18, scc // (op 4387) 
s_cmp_eq_i32 s18, 0 // (op 4387) 
s_cselect_b32 s19, 0, -1 // (op 4387) 

s_and_b32 s18, s15, 31 // (op 4387) 
s_lshl_b32 s49, 1, s18 // (op 4387) 
s_add_i32 s50, s49, s19 // (op 4387) 


s_and_b32 s19, s0, s50 // (op 4387) 


s_add_i32 s0, s1, s19 // (op 4387) 


s_ashr_i32 s1, s0, s18 // (op 4387) 


s_lshl_b32 s0, s15, 24 // (op 4387) 
s_ashr_i32 s18, s0, 31 // (op 4387) 

s_xor_b32 s0, s1, s18 // (op 4387) 

s_sub_i32 s1, s0, s18 // (op 4387) 


s_mul_hi_i32 s0, s1, s44 // (op 4387) 
s_add_i32 s18, s0, s1 // (op 4387) 

s_ashr_i32 s0, s18, 31 // (op 4387) 
s_cmp_eq_i32 s44, 0 // (op 4387) 
s_mov_b32 s19, scc // (op 4387) 
s_cmp_eq_i32 s19, 0 // (op 4387) 
s_cselect_b32 s49, 0, -1 // (op 4387) 

s_and_b32 s19, s45, 31 // (op 4387) 
s_lshl_b32 s50, 1, s19 // (op 4387) 
s_add_i32 s51, s50, s49 // (op 4387) 


s_and_b32 s49, s0, s51 // (op 4387) 


s_add_i32 s0, s18, s49 // (op 4387) 


s_ashr_i32 s18, s0, s19 // (op 4387) 


s_lshl_b32 s0, s45, 24 // (op 4387) 
s_ashr_i32 s19, s0, 31 // (op 4387) 

s_xor_b32 s0, s18, s19 // (op 4387) 

s_sub_i32 s18, s0, s19 // (op 4387) 


s_mul_i32 s0, s1, s13 // (op 4387) 

s_sub_u32 s1, s16, s0 // (op 4387) 

s_mul_i32 s0, s18, s13 // (op 4387) 
s_add_u32 s0, s0, s1 // (op 4387) 


s_mul_hi_i32 s1, s16, s11 // (op 4387) 
s_add_i32 s18, s1, s16 // (op 4387) 

s_ashr_i32 s1, s18, 31 // (op 4387) 
s_cmp_eq_i32 s11, 0 // (op 4387) 
s_mov_b32 s19, scc // (op 4387) 
s_cmp_eq_i32 s19, 0 // (op 4387) 
s_cselect_b32 s49, 0, -1 // (op 4387) 

s_and_b32 s19, s47, 31 // (op 4387) 
s_lshl_b32 s50, 1, s19 // (op 4387) 
s_add_i32 s51, s50, s49 // (op 4387) 


s_and_b32 s49, s1, s51 // (op 4387) 


s_add_i32 s1, s18, s49 // (op 4387) 


s_ashr_i32 s18, s1, s19 // (op 4387) 


s_lshl_b32 s1, s47, 24 // (op 4387) 
s_ashr_i32 s19, s1, 31 // (op 4387) 

s_xor_b32 s1, s18, s19 // (op 4387) 

s_sub_i32 s18, s1, s19 // (op 4387) 


s_add_u32 s50, s8, 128 // (op 4387) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4387) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4387) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4387) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 4387) 

s_mul_i32 s1, s13, s50 // (op 4387) 

s_mul_i32 s19, s10, s1 // (op 4387) 

s_mul_i32 s1, s18, s19 // (op 4387) 

s_sub_u32 s19, s16, s1 // (op 4387) 

s_mov_b32 s1, s19 // (op 4387) 
s_mul_hi_i32 s49, s1, s14 // (op 4387) 
s_add_i32 s50, s49, s1 // (op 4387) 


s_ashr_i32 s1, s50, 31 // (op 4387) 
s_cmp_eq_i32 s14, 0 // (op 4387) 
s_mov_b32 s49, scc // (op 4387) 
s_cmp_eq_i32 s49, 0 // (op 4387) 
s_cselect_b32 s51, 0, -1 // (op 4387) 

s_and_b32 s49, s46, 31 // (op 4387) 
s_lshl_b32 s68, 1, s49 // (op 4387) 
s_add_i32 s69, s68, s51 // (op 4387) 


s_and_b32 s51, s1, s69 // (op 4387) 


s_add_i32 s1, s50, s51 // (op 4387) 


s_ashr_i32 s50, s1, s49 // (op 4387) 


s_lshl_b32 s1, s46, 24 // (op 4387) 
s_ashr_i32 s49, s1, 31 // (op 4387) 

s_xor_b32 s1, s50, s49 // (op 4387) 

s_sub_i32 s50, s1, s49 // (op 4387) 


s_mul_i32 s1, s48, s13 // (op 4387) 
s_add_u32 s68, s6, 64 // (op 4387) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 4387) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 4387) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 4387) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 4387) 

s_sub_i32 s49, s68, s1 // (op 4387) 

s_mul_i32 s51, s50, s49 // (op 4387) 


s_sub_u32 s49, s19, s51 // (op 4387) 


s_add_u32 s19, s49, s1 // (op 4387) 


s_cmp_eq_i32 s18, 0 // (op 4387) 
s_mov_b32 s1, scc // (op 4387) 

s_cmp_eq_i32 s1, 0 // (op 4387) 
s_cselect_b32 s18, s19, s0 // (op 4387) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 4387) 


v_mul_lo_u32 v20, v15, s34 // (op 4387) 

v_add_u32 v15, v21, v20 // (op 4387) 


s_mul_i32 s1, s34, s39 // (op 4389) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4389) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4389) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4389) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4389) most significant: sum
s_add_u32 s1, s1, s19 // (op 4389) most significant: sum


s_mov_b32 s64, s36 // (op 4389) reg expression
s_mov_b32 s65, s37 // (op 4389) reg expression
s_mov_b32 s66, s0 // (op 4389) reg expression
s_mov_b32 s67, 131072 // (op 4389) call()

s_mul_hi_i32 s0, s16, s12 // (op 4432) 
v_and_b32 v20, -128, v1 // (op 4432) 
s_add_i32 s1, s0, s16 // (op 4432) 

v_sub_u32 v21, v1, v20 // (op 4432) 

s_ashr_i32 s0, s1, 31 // (op 4432) 
v_lshrrev_b32 v20, 7, v1 // (op 4432) 
s_cmp_eq_i32 s12, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
v_lshlrev_b32 v22, 1, v20 // (op 4432) 

s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s19, 0, -1 // (op 4432) 

s_and_b32 s18, s15, 31 // (op 4432) 
s_lshl_b32 s49, 1, s18 // (op 4432) 
s_add_i32 s50, s49, s19 // (op 4432) 


s_and_b32 s19, s0, s50 // (op 4432) 


s_add_i32 s0, s1, s19 // (op 4432) 


s_ashr_i32 s1, s0, s18 // (op 4432) 


s_lshl_b32 s0, s15, 24 // (op 4432) 
s_ashr_i32 s18, s0, 31 // (op 4432) 

s_xor_b32 s0, s1, s18 // (op 4432) 

s_sub_i32 s1, s0, s18 // (op 4432) 


s_mul_hi_i32 s0, s1, s44 // (op 4432) 
s_add_i32 s18, s0, s1 // (op 4432) 

s_ashr_i32 s0, s18, 31 // (op 4432) 
s_cmp_eq_i32 s44, 0 // (op 4432) 
s_mov_b32 s19, scc // (op 4432) 
s_cmp_eq_i32 s19, 0 // (op 4432) 
s_cselect_b32 s49, 0, -1 // (op 4432) 

s_and_b32 s19, s45, 31 // (op 4432) 
s_lshl_b32 s50, 1, s19 // (op 4432) 
s_add_i32 s51, s50, s49 // (op 4432) 


s_and_b32 s49, s0, s51 // (op 4432) 


s_add_i32 s0, s18, s49 // (op 4432) 


s_ashr_i32 s18, s0, s19 // (op 4432) 


s_lshl_b32 s0, s45, 24 // (op 4432) 
s_ashr_i32 s19, s0, 31 // (op 4432) 

s_xor_b32 s0, s18, s19 // (op 4432) 

s_sub_i32 s18, s0, s19 // (op 4432) 


s_add_u32 s50, s8, 128 // (op 4432) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4432) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4432) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4432) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 4432) 

s_mul_i32 s0, s18, s50 // (op 4432) 

s_sub_i32 s49, s1, s0 // (op 4432) 


s_mul_hi_i32 s0, s16, s11 // (op 4432) 
s_add_i32 s1, s0, s16 // (op 4432) 

s_ashr_i32 s0, s1, 31 // (op 4432) 
s_cmp_eq_i32 s11, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s19, 0, -1 // (op 4432) 

s_and_b32 s18, s47, 31 // (op 4432) 
s_lshl_b32 s68, 1, s18 // (op 4432) 
s_add_i32 s69, s68, s19 // (op 4432) 


s_and_b32 s19, s0, s69 // (op 4432) 


s_add_i32 s0, s1, s19 // (op 4432) 


s_ashr_i32 s1, s0, s18 // (op 4432) 


s_lshl_b32 s0, s47, 24 // (op 4432) 
s_ashr_i32 s18, s0, 31 // (op 4432) 

s_xor_b32 s0, s1, s18 // (op 4432) 

s_sub_i32 s1, s0, s18 // (op 4432) 


s_mul_i32 s0, s13, s50 // (op 4432) 

s_mul_i32 s18, s10, s0 // (op 4432) 

s_mul_i32 s0, s1, s18 // (op 4432) 

s_sub_u32 s18, s16, s0 // (op 4432) 

s_mul_hi_i32 s0, s18, s14 // (op 4432) 
s_add_i32 s19, s0, s18 // (op 4432) 


s_ashr_i32 s0, s19, 31 // (op 4432) 
s_cmp_eq_i32 s14, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s50, 0, -1 // (op 4432) 

s_and_b32 s18, s46, 31 // (op 4432) 
s_lshl_b32 s51, 1, s18 // (op 4432) 
s_add_i32 s68, s51, s50 // (op 4432) 


s_and_b32 s50, s0, s68 // (op 4432) 


s_add_i32 s0, s19, s50 // (op 4432) 


s_ashr_i32 s19, s0, s18 // (op 4432) 


s_lshl_b32 s0, s46, 24 // (op 4432) 
s_ashr_i32 s18, s0, 31 // (op 4432) 

s_xor_b32 s0, s19, s18 // (op 4432) 

s_sub_i32 s19, s0, s18 // (op 4432) 


s_cmp_eq_i32 s1, 0 // (op 4432) 
s_mov_b32 s0, scc // (op 4432) 

s_cmp_eq_i32 s0, 0 // (op 4432) 
s_cselect_b32 s1, s19, s49 // (op 4432) 



v_lshl_add_u32 v20, s1, 7, v21 // (op 4432) 


v_mul_lo_u32 v21, v22, s40 // (op 4432) 
v_add_u32 v21, v21, v20 // (op 4432) 


s_mul_i32 s1, s40, s39 // (op 4434) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4434) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4434) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4434) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4434) most significant: sum
s_add_u32 s1, s1, s19 // (op 4434) most significant: sum


s_mov_b32 s68, s42 // (op 4434) reg expression
s_mov_b32 s69, s43 // (op 4434) reg expression
s_mov_b32 s70, s0 // (op 4434) reg expression
s_mov_b32 s71, 131072 // (op 4434) call()

s_mul_hi_i32 s0, s16, s12 // (op 4451) 
v_and_b32 v20, -128, v1 // (op 4451) 
s_add_i32 s1, s0, s16 // (op 4451) 

v_sub_u32 v22, v1, v20 // (op 4451) 

s_ashr_i32 s0, s1, 31 // (op 4451) 
v_lshrrev_b32 v20, 7, v1 // (op 4451) 
s_cmp_eq_i32 s12, 0 // (op 4451) 
s_mov_b32 s18, scc // (op 4451) 
v_lshlrev_b32 v23, 1, v20 // (op 4451) 

s_cmp_eq_i32 s18, 0 // (op 4451) 
s_cselect_b32 s19, 0, -1 // (op 4451) 

s_and_b32 s18, s15, 31 // (op 4451) 
s_lshl_b32 s49, 1, s18 // (op 4451) 
s_add_i32 s50, s49, s19 // (op 4451) 


s_and_b32 s19, s0, s50 // (op 4451) 


s_add_i32 s0, s1, s19 // (op 4451) 


s_ashr_i32 s1, s0, s18 // (op 4451) 


s_lshl_b32 s0, s15, 24 // (op 4451) 
s_ashr_i32 s18, s0, 31 // (op 4451) 

s_xor_b32 s0, s1, s18 // (op 4451) 

s_sub_i32 s1, s0, s18 // (op 4451) 


s_mul_hi_i32 s0, s1, s44 // (op 4451) 
s_add_i32 s18, s0, s1 // (op 4451) 

s_ashr_i32 s0, s18, 31 // (op 4451) 
s_cmp_eq_i32 s44, 0 // (op 4451) 
s_mov_b32 s19, scc // (op 4451) 
s_cmp_eq_i32 s19, 0 // (op 4451) 
s_cselect_b32 s49, 0, -1 // (op 4451) 

s_and_b32 s19, s45, 31 // (op 4451) 
s_lshl_b32 s50, 1, s19 // (op 4451) 
s_add_i32 s51, s50, s49 // (op 4451) 


s_and_b32 s49, s0, s51 // (op 4451) 


s_add_i32 s0, s18, s49 // (op 4451) 


s_ashr_i32 s18, s0, s19 // (op 4451) 


s_lshl_b32 s0, s45, 24 // (op 4451) 
s_ashr_i32 s19, s0, 31 // (op 4451) 

s_xor_b32 s0, s18, s19 // (op 4451) 

s_sub_i32 s18, s0, s19 // (op 4451) 


s_add_u32 s50, s8, 128 // (op 4451) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4451) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4451) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4451) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4451) 

s_mul_i32 s0, s18, s50 // (op 4451) 

s_sub_i32 s49, s1, s0 // (op 4451) 


s_mul_hi_i32 s0, s16, s11 // (op 4451) 
s_add_i32 s1, s0, s16 // (op 4451) 

s_ashr_i32 s0, s1, 31 // (op 4451) 
s_cmp_eq_i32 s11, 0 // (op 4451) 
s_mov_b32 s18, scc // (op 4451) 
s_cmp_eq_i32 s18, 0 // (op 4451) 
s_cselect_b32 s19, 0, -1 // (op 4451) 

s_and_b32 s18, s47, 31 // (op 4451) 
s_lshl_b32 s72, 1, s18 // (op 4451) 
s_add_i32 s73, s72, s19 // (op 4451) 


s_and_b32 s19, s0, s73 // (op 4451) 


s_add_i32 s0, s1, s19 // (op 4451) 


s_ashr_i32 s1, s0, s18 // (op 4451) 


s_lshl_b32 s0, s47, 24 // (op 4451) 
s_ashr_i32 s18, s0, 31 // (op 4451) 

s_xor_b32 s0, s1, s18 // (op 4451) 

s_sub_i32 s1, s0, s18 // (op 4451) 


s_mul_i32 s0, s13, s50 // (op 4451) 

s_mul_i32 s18, s10, s0 // (op 4451) 

s_mul_i32 s0, s1, s18 // (op 4451) 

s_sub_u32 s18, s16, s0 // (op 4451) 

s_mul_hi_i32 s0, s18, s14 // (op 4451) 
s_add_i32 s19, s0, s18 // (op 4451) 


s_ashr_i32 s0, s19, 31 // (op 4451) 
s_cmp_eq_i32 s14, 0 // (op 4451) 
s_mov_b32 s18, scc // (op 4451) 
s_cmp_eq_i32 s18, 0 // (op 4451) 
s_cselect_b32 s50, 0, -1 // (op 4451) 

s_and_b32 s18, s46, 31 // (op 4451) 
s_lshl_b32 s51, 1, s18 // (op 4451) 
s_add_i32 s72, s51, s50 // (op 4451) 


s_and_b32 s50, s0, s72 // (op 4451) 


s_add_i32 s0, s19, s50 // (op 4451) 


s_ashr_i32 s19, s0, s18 // (op 4451) 


s_lshl_b32 s0, s46, 24 // (op 4451) 
s_ashr_i32 s18, s0, 31 // (op 4451) 

s_xor_b32 s0, s19, s18 // (op 4451) 

s_sub_i32 s19, s0, s18 // (op 4451) 


s_cmp_eq_i32 s1, 0 // (op 4451) 
s_mov_b32 s0, scc // (op 4451) 

s_cmp_eq_i32 s0, 0 // (op 4451) 
s_cselect_b32 s1, s19, s49 // (op 4451) 



v_lshl_add_u32 v20, s1, 7, v22 // (op 4451) 


v_mul_lo_u32 v22, v23, s40 // (op 4451) 
v_add_u32 v22, v22, v20 // (op 4451) 


s_mul_i32 s1, s40, s39 // (op 4453) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4453) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4453) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4453) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4453) most significant: sum
s_add_u32 s1, s1, s19 // (op 4453) most significant: sum


s_mov_b32 s68, s42 // (op 4453) reg expression
s_mov_b32 s69, s43 // (op 4453) reg expression
s_mov_b32 s70, s0 // (op 4453) reg expression
s_mov_b32 s71, 131072 // (op 4453) call()


s_lshl_b64 s[0:1], s[30:31], 2 // (op 3984) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3984) 


v_mov_b32 v20, v18 // (op 3984) reg expression
buffer_load_dwordx4 v[26:29], v20, s[52:55], 0 offen // (op 3984) Load value
v_add_u32 v20, v20, v24 // (op 3984) 
buffer_load_dwordx4 v[30:33], v20, s[52:55], 0 offen // (op 3984) Load value



v_mov_b64 v[24:25], s[40:41] // (op 3987) convert

v_mov_b32 v20, v21 // (op 3987) reg expression
buffer_load_ubyte v34, v20, s[68:71], 0 offen offset: 0 // (op 3987) Load value
v_add_u32 v20, v20, v24 // (op 3987) 
buffer_load_ubyte v35, v20, s[68:71], 0 offen offset: 0 // (op 3987) Load value




v_mov_b32 v20, v16 // (op 3990) reg expression

buffer_load_dwordx3 v[36:38], v20, s[20:23], 0 offen // (op 3990) Load value
buffer_load_dwordx3 v[40:42], v20, s[20:23], 0 offen offset:12 // (op 3990) Load value



v_mov_b32 v20, v14 // (op 3993) reg expression

buffer_load_ubyte v39, v20, s[64:67], 0 offen offset: 0 // (op 3993) Load value


s_lshl_b64 s[0:1], s[30:31], 2 // (op 3996) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3996) 


s_lshl_b64 s[0:1], s[30:31], 7 // (op 3996) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 3996) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 3996) 

v_add_u32 v20, v18, s0 // (op 3996) 

buffer_load_dwordx4 v[44:47], v20, s[52:55], 0 offen // (op 3996) Load value
v_add_u32 v20, v20, v24 // (op 3996) 
buffer_load_dwordx4 v[48:51], v20, s[52:55], 0 offen // (op 3996) Load value




v_mov_b64 v[24:25], s[40:41] // (op 3999) convert

s_lshl_b32 s0, s40, 2 // (op 3999) 
v_add_u32 v18, v21, s0 // (op 3999) 

buffer_load_ubyte v52, v18, s[68:71], 0 offen offset: 0 // (op 3999) Load value
v_add_u32 v18, v18, v24 // (op 3999) 
buffer_load_ubyte v53, v18, s[68:71], 0 offen offset: 0 // (op 3999) Load value





s_lshl_b64 s[0:1], s[24:25], 7 // (op 4002) 
s_mul_i32 s19, s1, 6 // (op 4002) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 4002) least significant
s_mul_hi_u32 s49, s0, 6 // (op 4002) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4002) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 4002) 

v_add_u32 v18, v16, s0 // (op 4002) 


buffer_load_dwordx3 v[54:56], v18, s[20:23], 0 offen // (op 4002) Load value
buffer_load_dwordx3 v[58:60], v18, s[20:23], 0 offen offset:12 // (op 4002) Load value




s_lshl_b32 s0, s34, 2 // (op 4005) 
v_add_u32 v16, v14, s0 // (op 4005) 


buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 4005) Load value


v_and_b32 v14, -4, v1 // (op 4581) 
v_sub_u32 v16, v1, v14 // (op 4581) 

v_lshlrev_b32 v14, 5, v16 // (op 4581) 

v_lshrrev_b32 v16, 2, v1 // (op 4581) 
v_lshl_add_u32 v43, v16, 8, v14 // (op 4581) 


v_lshlrev_b32 v14, 2, v43 // (op 4581) 

v_lshrrev_b32 v16, 3, v14 // (op 4581) 




s_waitcnt vmcnt(13)
ds_write_b128 v16, v[26:29] // (op 4007) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[30:33] offset:64 // (op 4007) Store local data 

v_and_b32 v14, -128, v1 // (op 4697) 
v_sub_u32 v16, v1, v14 // (op 4697) 

v_lshrrev_b32 v14, 7, v1 // (op 4697) 
v_lshl_add_u32 v43, v14, 8, v16 // (op 4697) 





s_waitcnt vmcnt(11)
ds_write_b8 v43, v34 offset:8192 // (op 4008) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v43, v35 offset:8320 // (op 4008) Store local data 

v_and_b32 v14, -2, v1 // (op 4523) 
v_sub_u32 v16, v1, v14 // (op 4523) 

v_lshlrev_b32 v14, 5, v16 // (op 4523) 

v_lshrrev_b32 v16, 1, v1 // (op 4523) 
v_lshl_add_u32 v43, v16, 6, v14 // (op 4523) 


v_mul_lo_u32 v14, v43, 6 // (op 4523) 
v_lshrrev_b32 v16, 3, v14 // (op 4523) 

v_lshrrev_b32 v14, 4, v43 // (op 4523) 

v_lshlrev_b32 v43, 2, v14 // (op 4523) 

v_add_u32 v14, v16, v43 // (op 4523) 





s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 4009) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 4009) Store local data 

v_and_b32 v14, -64, v1 // (op 4639) 
v_sub_u32 v16, v1, v14 // (op 4639) 

v_and_b32 v14, 4294967232, v1 // (op 4639) 
v_add_u32 v43, v14, v16 // (op 4639) 





s_waitcnt vmcnt(7)
ds_write_b8 v43, v39 offset:16896 // (op 4010) Store local data 

v_and_b32 v14, -2, v1 // (op 4507) 
v_sub_u32 v16, v1, v14 // (op 4507) 

v_lshlrev_b32 v14, 5, v16 // (op 4507) 

v_lshrrev_b32 v16, 1, v1 // (op 4507) 
v_lshl_add_u32 v43, v16, 6, v14 // (op 4507) 


v_mul_lo_u32 v14, v43, 6 // (op 4507) 
v_lshrrev_b32 v16, 3, v14 // (op 4507) 

v_lshrrev_b32 v14, 4, v43 // (op 4507) 

v_lshlrev_b32 v43, 2, v14 // (op 4507) 

v_add_u32 v14, v16, v43 // (op 4507) 


v_and_b32 v16, -4, v1 // (op 4565) 
v_sub_u32 v43, v1, v16 // (op 4565) 

v_lshlrev_b32 v16, 5, v43 // (op 4565) 

v_lshrrev_b32 v43, 2, v1 // (op 4565) 
v_lshl_add_u32 v57, v43, 8, v16 // (op 4565) 


v_lshlrev_b32 v16, 2, v57 // (op 4565) 

v_lshrrev_b32 v43, 3, v16 // (op 4565) 

v_and_b32 v16, -64, v1 // (op 4623) 
v_sub_u32 v57, v1, v16 // (op 4623) 

v_and_b32 v16, 4294967232, v1 // (op 4623) 
v_add_u32 v20, v16, v57 // (op 4623) 


v_and_b32 v16, -128, v1 // (op 4681) 
v_sub_u32 v21, v1, v16 // (op 4681) 

v_lshrrev_b32 v16, 7, v1 // (op 4681) 
v_lshl_add_u32 v57, v16, 8, v21 // (op 4681) 


v_and_b32 v16, -2, v1 // (op 4749) 
v_sub_u32 v21, v1, v16 // (op 4749) 

v_lshlrev_b32 v16, 5, v21 // (op 4749) 

v_lshrrev_b32 v21, 1, v1 // (op 4749) 
v_lshl_add_u32 v23, v21, 6, v16 // (op 4749) 


v_mul_lo_u32 v16, v23, 6 // (op 4749) 
v_lshrrev_b32 v21, 3, v16 // (op 4749) 

v_lshrrev_b32 v16, 4, v23 // (op 4749) 

v_lshlrev_b32 v23, 2, v16 // (op 4749) 

v_add_u32 v16, v21, v23 // (op 4749) 


v_and_b32 v21, -4, v1 // (op 4778) 
v_sub_u32 v23, v1, v21 // (op 4778) 

v_lshlrev_b32 v21, 5, v23 // (op 4778) 

v_lshrrev_b32 v23, 2, v1 // (op 4778) 
v_lshl_add_u32 v24, v23, 8, v21 // (op 4778) 


v_lshlrev_b32 v21, 2, v24 // (op 4778) 

v_lshrrev_b32 v23, 3, v21 // (op 4778) 

v_and_b32 v21, -64, v1 // (op 4807) 
v_sub_u32 v24, v1, v21 // (op 4807) 

v_and_b32 v21, 4294967232, v1 // (op 4807) 
v_add_u32 v25, v21, v24 // (op 4807) 


v_and_b32 v21, -128, v1 // (op 4836) 
v_sub_u32 v24, v1, v21 // (op 4836) 

v_lshrrev_b32 v21, 7, v1 // (op 4836) 
v_lshl_add_u32 v61, v21, 8, v24 // (op 4836) 


s_mov_b32 s0, 0 // (op 1047) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 1046) 
s_mov_b32 s1, s18 // (op 1046) convert

s_and_b32 s18, s1, 4294967294 // (op 1046) 

s_cmp_lt_u32 s0, s18 // (op 1046) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1046 // (op 1046) Condition: Top (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1046 if false)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1046:

s_barrier  // (op 4916) 
s_add_u32 s0, s0, 2 // (op 1048) 



s_lshl_b64 s[18:19], s[30:31], 2 // (op 10) 
v_lshrrev_b64 v[62:63], 3, s[18:19] // (op 10) 


v_mov_b32 v21, v19 // (op 10) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 10) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 10) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v24, v21, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v24, v24, v62 // (op 10) 
ds_read_b96_tr_b6 v[64:66], v0 offset:8704 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v0 offset:9728 // (op 70) Transpose load local data 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 10) Load value



s_barrier  // (op 4929) 
v_mov_b64 v[62:63], s[40:41] // (op 25) convert




v_mov_b32 v24, v22 // (op 25) 
s_lshl_b32 s1, s40, 3 // (op 25) 
v_add_u32 v70, v24, s1 // (op 25) 

ds_read_b64_tr_b4 v[72:73], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:1024 // (op 75) Transpose load local data 
buffer_load_ubyte v34, v70, s[68:71], 0 offen offset: 0 // (op 25) Load value
s_barrier  // (op 4943) 
v_add_u32 v70, v70, v62 // (op 25) 


ds_read_u8 v71, v3 offset:16896 // (op 80) Load local data 
buffer_load_ubyte v35, v70, s[68:71], 0 offen offset: 0 // (op 25) Load value




s_barrier  // (op 4957) 
v_mov_b32 v70, v17 // (op 4) 


ds_read_u8 v62, v5 offset:8192 // (op 85) Load local data 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 4) 



s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v63, v70, s1 // (op 4) 


ds_read_b64_tr_b4 v[76:77], v4 offset:16 // (op 371) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v4 offset:1040 // (op 371) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[72:75], a[0:3], v71, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 88) 


ds_read_u8 v80, v5 offset:8224 // (op 375) Load local data 
buffer_load_dwordx3 v[36:38], v63, s[20:23], 0 offen // (op 4) Load value



ds_read_b64_tr_b4 v[82:83], v4 offset:32 // (op 547) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v4 offset:1056 // (op 547) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[76:79], a[4:7], v71, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 376) 


ds_read_u8 v81, v5 offset:8256 // (op 551) Load local data 
buffer_load_dwordx3 v[40:42], v63, s[20:23], 0 offen offset:12 // (op 4) Load value






v_mov_b32 v63, v15 // (op 17) 
s_lshl_b32 s1, s34, 3 // (op 17) 
v_add_u32 v86, v63, s1 // (op 17) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[82:85], a[8:11], v71, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 552) 
ds_read_b64_tr_b4 v[88:89], v4 offset:48 // (op 723) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v4 offset:1072 // (op 723) Transpose load local data 
buffer_load_ubyte v39, v86, s[64:67], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 4980) 
ds_read_u8 v86, v5 offset:8288 // (op 727) Load local data 






ds_read_b96_tr_b6 v[92:94], v0 offset:8736 // (op 1080) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v0 offset:9760 // (op 1080) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[88:91], a[12:15], v71, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 728) 




ds_read_u8 v71, v3 offset:16928 // (op 1083) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v23, v[44:47] offset:17152 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[72:75], a[16:19], v71, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1223) 
s_waitcnt vmcnt(12)
ds_write_b128 v23, v[48:51] offset:17216 // (op 253) Store local data 


s_barrier  // (op 4992) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:97], v[76:79], a[20:23], v71, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1177) 





s_waitcnt vmcnt(11)
ds_write_b8 v61, v52 offset:25344 // (op 255) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[82:85], a[24:27], v71, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1131) 
s_waitcnt vmcnt(10)
ds_write_b8 v61, v53 offset:25472 // (op 255) Store local data 


s_barrier  // (op 4974) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[88:91], a[28:31], v71, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1085) 







s_waitcnt vmcnt(9)
ds_write_b96 v16, v[54:56] offset:25856 // (op 243) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v16, v[58:60] offset:25872 // (op 243) Store local data 
s_barrier  // (op 4986) 



s_waitcnt vmcnt(7)
ds_write_b8 v25, v18 offset:34048 // (op 254) Store local data 

s_lshl_b64 s[18:19], s[30:31], 2 // (op 256) 
v_lshrrev_b64 v[64:65], 3, s[18:19] // (op 256) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 256) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 256) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 256) 

s_mul_i32 s1, 3, s18 // (op 256) 

v_add_u32 v62, v21, s1 // (op 256) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4971) 
buffer_load_dwordx4 v[44:47], v62, s[52:55], 0 offen // (op 256) Load value



v_add_u32 v62, v62, v64 // (op 256) 
ds_read_b96_tr_b6 v[72:74], v6 offset:25856 // (op 245) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v6 offset:26880 // (op 245) Transpose load local data 
buffer_load_dwordx4 v[48:51], v62, s[52:55], 0 offen // (op 256) Load value



s_barrier  // (op 4977) 




v_mov_b64 v[64:65], s[40:41] // (op 258) convert

s_lshl_b32 s1, s40, 2 // (op 258) 
s_mul_i32 s18, 3, s1 // (op 258) 

v_add_u32 v21, v24, s18 // (op 258) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 4315) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4315) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4315) 

s_lshl_b32 s1, s18, 1 // (op 4315) 

v_add_u32 v19, v19, s1 // (op 4315) 

ds_read_b64_tr_b4 v[66:67], v8 offset:17152 // (op 247) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v8 offset:18176 // (op 247) Transpose load local data 
buffer_load_ubyte v52, v21, s[68:71], 0 offen offset: 0 // (op 258) Load value
v_add_u32 v21, v21, v64 // (op 258) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4983) 
buffer_load_ubyte v53, v21, s[68:71], 0 offen offset: 0 // (op 258) Load value





ds_read_u8 v21, v7 offset:34048 // (op 249) Load local data 


s_barrier  // (op 4989) 
s_lshl_b32 s1, s40, 3 // (op 4443) 


v_add_u32 v22, v22, s1 // (op 4443) 

ds_read_u8 v24, v9 offset:25344 // (op 251) Load local data 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 242) 



s_mul_i32 s51, s19, 6 // (op 242) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 242) least significant
s_mul_hi_u32 s1, s18, 6 // (op 242) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 242) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 242) 

s_mul_i32 s1, 3, s18 // (op 242) 

v_add_u32 v62, v70, s1 // (op 242) 


buffer_load_dwordx3 v[54:56], v62, s[20:23], 0 offen // (op 242) Load value
ds_read_b64_tr_b4 v[78:79], v8 offset:17168 // (op 381) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[66:69], a[0:3], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 252) 
ds_read_b64_tr_b4 v[80:81], v8 offset:18192 // (op 381) Transpose load local data 
buffer_load_dwordx3 v[58:60], v62, s[20:23], 0 offen offset:12 // (op 242) Load value




ds_read_u8 v62, v9 offset:25376 // (op 385) Load local data 





s_lshl_b32 s1, s34, 2 // (op 257) 
s_mul_i32 s18, 3, s1 // (op 257) 

v_add_u32 v64, v63, s18 // (op 257) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 4251) 
s_mul_i32 s51, s19, 6 // (op 4251) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4251) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4251) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4251) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4251) 

s_lshl_b32 s1, s18, 1 // (op 4251) 

v_add_u32 v17, v17, s1 // (op 4251) 

ds_read_b64_tr_b4 v[82:83], v8 offset:17184 // (op 557) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v8 offset:18208 // (op 557) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[78:81], a[4:7], v21, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 377) 


ds_read_u8 v65, v9 offset:25408 // (op 561) Load local data 
buffer_load_ubyte v18, v64, s[64:67], 0 offen offset: 0 // (op 257) Load value





ds_read_b64_tr_b4 v[86:87], v8 offset:17200 // (op 733) Transpose load local data 
s_barrier  // (op 4932) 
ds_read_b64_tr_b4 v[88:89], v8 offset:18224 // (op 733) Transpose load local data 



s_lshl_b32 s1, s34, 3 // (op 4379) 


s_barrier  // (op 5049) 
v_add_u32 v15, v15, s1 // (op 4379) 

ds_read_u8 v63, v9 offset:25440 // (op 737) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[82:85], a[8:11], v21, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 553) 
s_waitcnt vmcnt(13)
ds_write_b128 v43, v[26:29] // (op 13) Store local data 



s_barrier  // (op 5053) 
ds_read_b96_tr_b6 v[90:92], v6 offset:25888 // (op 1088) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[86:89], a[12:15], v21, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 729) 
ds_read_b96_tr_b6 v[93:95], v6 offset:26912 // (op 1088) Transpose load local data 


s_waitcnt vmcnt(12)
ds_write_b128 v43, v[30:33] offset:64 // (op 13) Store local data 


s_barrier  // (op 5077) 
s_barrier  // (op 4960) 
ds_read_u8 v21, v7 offset:34080 // (op 1091) Load local data 



s_waitcnt vmcnt(11)
ds_write_b8 v57, v34 offset:8192 // (op 30) Store local data 
s_barrier  // (op 5081) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[66:69], a[16:19], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1224) 
s_waitcnt vmcnt(10)
ds_write_b8 v57, v35 offset:8320 // (op 30) Store local data 


s_barrier  // (op 4919) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[78:81], a[20:23], v21, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1178) 





s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[82:85], a[24:27], v21, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1132) 
s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 6) Store local data 


s_barrier  // (op 4946) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[86:89], a[28:31], v21, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1086) 







s_waitcnt vmcnt(7)
ds_write_b8 v20, v39 offset:16896 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[28:29], 7 // (op 1046) 
s_mov_b32 s1, s18 // (op 1046) convert

s_and_b32 s18, s1, 4294967294 // (op 1046) 

s_cmp_lt_u32 s0, s18 // (op 1046) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1046 // (op 1046) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1046 if true)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1046:

















s_mul_hi_i32 s0, s16, s12 // (op 4207) 
v_and_b32 v57, -2, v1 // (op 4207) 
s_add_i32 s1, s0, s16 // (op 4207) 

v_sub_u32 v14, v1, v57 // (op 4207) 

s_ashr_i32 s0, s1, 31 // (op 4207) 
v_lshlrev_b32 v57, 5, v14 // (op 4207) 

s_cmp_eq_i32 s12, 0 // (op 4207) 
s_mov_b32 s18, scc // (op 4207) 
v_lshrrev_b32 v14, 1, v1 // (op 4207) 
s_cmp_eq_i32 s18, 0 // (op 4207) 
s_cselect_b32 s19, 0, -1 // (op 4207) 

s_and_b32 s18, s15, 31 // (op 4207) 
s_lshl_b32 s49, 1, s18 // (op 4207) 
s_add_i32 s50, s49, s19 // (op 4207) 


s_and_b32 s19, s0, s50 // (op 4207) 


s_add_i32 s0, s1, s19 // (op 4207) 


s_ashr_i32 s1, s0, s18 // (op 4207) 


s_lshl_b32 s0, s15, 24 // (op 4207) 
s_ashr_i32 s18, s0, 31 // (op 4207) 

s_xor_b32 s0, s1, s18 // (op 4207) 

s_sub_i32 s1, s0, s18 // (op 4207) 


s_mul_hi_i32 s0, s1, s44 // (op 4207) 
s_add_i32 s18, s0, s1 // (op 4207) 

s_ashr_i32 s0, s18, 31 // (op 4207) 
s_cmp_eq_i32 s44, 0 // (op 4207) 
s_mov_b32 s19, scc // (op 4207) 
s_cmp_eq_i32 s19, 0 // (op 4207) 
s_cselect_b32 s49, 0, -1 // (op 4207) 

s_and_b32 s19, s45, 31 // (op 4207) 
s_lshl_b32 s50, 1, s19 // (op 4207) 
s_add_i32 s51, s50, s49 // (op 4207) 


s_and_b32 s49, s0, s51 // (op 4207) 


s_add_i32 s0, s18, s49 // (op 4207) 


s_ashr_i32 s18, s0, s19 // (op 4207) 


s_lshl_b32 s0, s45, 24 // (op 4207) 
s_ashr_i32 s19, s0, 31 // (op 4207) 

s_xor_b32 s0, s18, s19 // (op 4207) 

s_sub_i32 s18, s0, s19 // (op 4207) 


s_mul_i32 s0, s1, s13 // (op 4207) 

s_sub_u32 s1, s16, s0 // (op 4207) 

s_mul_i32 s0, s18, s13 // (op 4207) 
s_add_u32 s0, s0, s1 // (op 4207) 


s_mul_hi_i32 s1, s16, s11 // (op 4207) 
s_add_i32 s18, s1, s16 // (op 4207) 

s_ashr_i32 s1, s18, 31 // (op 4207) 
s_cmp_eq_i32 s11, 0 // (op 4207) 
s_mov_b32 s19, scc // (op 4207) 
s_cmp_eq_i32 s19, 0 // (op 4207) 
s_cselect_b32 s49, 0, -1 // (op 4207) 

s_and_b32 s19, s47, 31 // (op 4207) 
s_lshl_b32 s50, 1, s19 // (op 4207) 
s_add_i32 s51, s50, s49 // (op 4207) 


s_and_b32 s49, s1, s51 // (op 4207) 


s_add_i32 s1, s18, s49 // (op 4207) 


s_ashr_i32 s18, s1, s19 // (op 4207) 


s_lshl_b32 s1, s47, 24 // (op 4207) 
s_ashr_i32 s19, s1, 31 // (op 4207) 

s_xor_b32 s1, s18, s19 // (op 4207) 

s_sub_i32 s18, s1, s19 // (op 4207) 


s_add_u32 s50, s8, 128 // (op 4207) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4207) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4207) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4207) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4207) 

s_mul_i32 s1, s13, s50 // (op 4207) 

s_mul_i32 s19, s10, s1 // (op 4207) 

s_mul_i32 s1, s18, s19 // (op 4207) 

s_sub_u32 s19, s16, s1 // (op 4207) 

s_mov_b32 s1, s19 // (op 4207) 
s_mul_hi_i32 s49, s1, s14 // (op 4207) 
s_add_i32 s50, s49, s1 // (op 4207) 


s_ashr_i32 s1, s50, 31 // (op 4207) 
s_cmp_eq_i32 s14, 0 // (op 4207) 
s_mov_b32 s49, scc // (op 4207) 
s_cmp_eq_i32 s49, 0 // (op 4207) 
s_cselect_b32 s51, 0, -1 // (op 4207) 

s_and_b32 s49, s46, 31 // (op 4207) 
s_lshl_b32 s72, 1, s49 // (op 4207) 
s_add_i32 s73, s72, s51 // (op 4207) 


s_and_b32 s51, s1, s73 // (op 4207) 


s_add_i32 s1, s50, s51 // (op 4207) 


s_ashr_i32 s50, s1, s49 // (op 4207) 


s_lshl_b32 s1, s46, 24 // (op 4207) 
s_ashr_i32 s49, s1, 31 // (op 4207) 

s_xor_b32 s1, s50, s49 // (op 4207) 

s_sub_i32 s50, s1, s49 // (op 4207) 


s_mul_i32 s1, s48, s13 // (op 4207) 
s_add_u32 s72, s6, 64 // (op 4207) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 4207) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4207) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4207) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 4207) 

s_sub_i32 s49, s72, s1 // (op 4207) 

s_mul_i32 s51, s50, s49 // (op 4207) 


s_sub_u32 s49, s19, s51 // (op 4207) 


s_add_u32 s19, s49, s1 // (op 4207) 


s_cmp_eq_i32 s18, 0 // (op 4207) 
s_mov_b32 s1, scc // (op 4207) 

s_cmp_eq_i32 s1, 0 // (op 4207) 
s_cselect_b32 s18, s19, s0 // (op 4207) 



v_lshl_add_u32 v15, s18, 6, v57 // (op 4207) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4207) 
v_mov_b32 v17, 0 // (op 4207) convert
v_mov_b32 v16, v15 // (op 4207) convert

s_lshr_b32 s18, s0, 1 // (op 4207) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 4207) 


v_mul_lo_u32 v43, v15, s25 // (op 4207) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4207) most significant: high of low * low
v_mul_lo_u32 v42, v15, s24 // (op 4207) least significant: low of low * low
v_add_u32 v43, v43, v14 // (op 4207) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v42 // (op 4207) least significant half
v_addc_co_u32 v15, s[0:1], v17, v43, s[0:1] // (op 4207) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4207) most significant: low of high * low
v_mul_hi_u32 v57, v14, 6 // (op 4207) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4207) least significant: low of low * low
v_add_u32 v17, v17, v57 // (op 4207) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4207) 

v_mov_b32 v57, v14 // (op 4207) convert

s_mul_i32 s1, s29, 6 // (op 4209) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4209) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4209) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4209) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4209) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4209) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4209) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4209) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4209) most significant: sum
s_add_u32 s19, s19, s50 // (op 4209) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4209) 

s_mov_b32 s20, s26 // (op 4209) reg expression
s_mov_b32 s21, s27 // (op 4209) reg expression
s_mov_b32 s22, s0 // (op 4209) reg expression
s_mov_b32 s23, 131072 // (op 4209) call()

s_mul_hi_i32 s0, s16, s12 // (op 4281) 
v_and_b32 v14, -4, v1 // (op 4281) 
s_add_i32 s1, s0, s16 // (op 4281) 

v_sub_u32 v15, v1, v14 // (op 4281) 

s_ashr_i32 s0, s1, 31 // (op 4281) 
v_lshlrev_b32 v14, 5, v15 // (op 4281) 

s_cmp_eq_i32 s12, 0 // (op 4281) 
s_mov_b32 s18, scc // (op 4281) 
v_lshrrev_b32 v15, 2, v1 // (op 4281) 
s_cmp_eq_i32 s18, 0 // (op 4281) 
s_cselect_b32 s19, 0, -1 // (op 4281) 

v_lshlrev_b32 v16, 1, v15 // (op 4281) 

s_and_b32 s18, s15, 31 // (op 4281) 
s_lshl_b32 s49, 1, s18 // (op 4281) 
s_add_i32 s50, s49, s19 // (op 4281) 


s_and_b32 s19, s0, s50 // (op 4281) 


s_add_i32 s0, s1, s19 // (op 4281) 


s_ashr_i32 s1, s0, s18 // (op 4281) 


s_lshl_b32 s0, s15, 24 // (op 4281) 
s_ashr_i32 s18, s0, 31 // (op 4281) 

s_xor_b32 s0, s1, s18 // (op 4281) 

s_sub_i32 s1, s0, s18 // (op 4281) 


s_mul_hi_i32 s0, s1, s44 // (op 4281) 
s_add_i32 s18, s0, s1 // (op 4281) 

s_ashr_i32 s0, s18, 31 // (op 4281) 
s_cmp_eq_i32 s44, 0 // (op 4281) 
s_mov_b32 s19, scc // (op 4281) 
s_cmp_eq_i32 s19, 0 // (op 4281) 
s_cselect_b32 s49, 0, -1 // (op 4281) 

s_and_b32 s19, s45, 31 // (op 4281) 
s_lshl_b32 s50, 1, s19 // (op 4281) 
s_add_i32 s51, s50, s49 // (op 4281) 


s_and_b32 s49, s0, s51 // (op 4281) 


s_add_i32 s0, s18, s49 // (op 4281) 


s_ashr_i32 s18, s0, s19 // (op 4281) 


s_lshl_b32 s0, s45, 24 // (op 4281) 
s_ashr_i32 s19, s0, 31 // (op 4281) 

s_xor_b32 s0, s18, s19 // (op 4281) 

s_sub_i32 s18, s0, s19 // (op 4281) 


s_add_u32 s50, s8, 128 // (op 4281) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4281) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4281) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4281) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4281) 

s_mul_i32 s0, s18, s50 // (op 4281) 

s_sub_i32 s49, s1, s0 // (op 4281) 


s_mul_hi_i32 s0, s16, s11 // (op 4281) 
s_add_i32 s1, s0, s16 // (op 4281) 

s_ashr_i32 s0, s1, 31 // (op 4281) 
s_cmp_eq_i32 s11, 0 // (op 4281) 
s_mov_b32 s18, scc // (op 4281) 
s_cmp_eq_i32 s18, 0 // (op 4281) 
s_cselect_b32 s19, 0, -1 // (op 4281) 

s_and_b32 s18, s47, 31 // (op 4281) 
s_lshl_b32 s72, 1, s18 // (op 4281) 
s_add_i32 s73, s72, s19 // (op 4281) 


s_and_b32 s19, s0, s73 // (op 4281) 


s_add_i32 s0, s1, s19 // (op 4281) 


s_ashr_i32 s1, s0, s18 // (op 4281) 


s_lshl_b32 s0, s47, 24 // (op 4281) 
s_ashr_i32 s18, s0, 31 // (op 4281) 

s_xor_b32 s0, s1, s18 // (op 4281) 

s_sub_i32 s1, s0, s18 // (op 4281) 


s_mul_i32 s0, s13, s50 // (op 4281) 

s_mul_i32 s18, s10, s0 // (op 4281) 

s_mul_i32 s0, s1, s18 // (op 4281) 

s_sub_u32 s18, s16, s0 // (op 4281) 

s_mul_hi_i32 s0, s18, s14 // (op 4281) 
s_add_i32 s19, s0, s18 // (op 4281) 


s_ashr_i32 s0, s19, 31 // (op 4281) 
s_cmp_eq_i32 s14, 0 // (op 4281) 
s_mov_b32 s18, scc // (op 4281) 
s_cmp_eq_i32 s18, 0 // (op 4281) 
s_cselect_b32 s50, 0, -1 // (op 4281) 

s_and_b32 s18, s46, 31 // (op 4281) 
s_lshl_b32 s51, 1, s18 // (op 4281) 
s_add_i32 s72, s51, s50 // (op 4281) 


s_and_b32 s50, s0, s72 // (op 4281) 


s_add_i32 s0, s19, s50 // (op 4281) 


s_ashr_i32 s19, s0, s18 // (op 4281) 


s_lshl_b32 s0, s46, 24 // (op 4281) 
s_ashr_i32 s18, s0, 31 // (op 4281) 

s_xor_b32 s0, s19, s18 // (op 4281) 

s_sub_i32 s19, s0, s18 // (op 4281) 


s_cmp_eq_i32 s1, 0 // (op 4281) 
s_mov_b32 s0, scc // (op 4281) 

s_cmp_eq_i32 s0, 0 // (op 4281) 
s_cselect_b32 s1, s19, s49 // (op 4281) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4281) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4281) 
v_mov_b32 v43, 0 // (op 4281) convert
v_mov_b32 v42, v15 // (op 4281) convert

s_lshr_b32 s18, s0, 1 // (op 4281) 

v_lshl_add_u32 v17, s18, 8, v16 // (op 4281) 


v_mul_lo_u32 v15, v17, s31 // (op 4281) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4281) most significant: high of low * low
v_mul_lo_u32 v14, v17, s30 // (op 4281) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 4281) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v42 // (op 4281) least significant half
v_addc_co_u32 v15, s[0:1], v15, v43, s[0:1] // (op 4281) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 4281) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4281) 

v_mov_b32 v16, v14 // (op 4281) convert

s_mul_i32 s1, s30, s29 // (op 4283) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4283) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4283) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4283) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4283) most significant: sum
s_add_u32 s1, s1, s19 // (op 4283) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4283) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4283) 

s_mov_b32 s52, s32 // (op 4283) reg expression
s_mov_b32 s53, s33 // (op 4283) reg expression
s_mov_b32 s54, s0 // (op 4283) reg expression
s_mov_b32 s55, 131072 // (op 4283) call()

s_mul_hi_i32 s0, s16, s12 // (op 4345) 
v_and_b32 v17, -64, v1 // (op 4345) 
s_add_i32 s1, s0, s16 // (op 4345) 

v_sub_u32 v14, v1, v17 // (op 4345) 

s_ashr_i32 s0, s1, 31 // (op 4345) 
v_lshrrev_b32 v15, 6, v1 // (op 4345) 
s_cmp_eq_i32 s12, 0 // (op 4345) 
s_mov_b32 s18, scc // (op 4345) 
s_cmp_eq_i32 s18, 0 // (op 4345) 
s_cselect_b32 s19, 0, -1 // (op 4345) 

s_and_b32 s18, s15, 31 // (op 4345) 
s_lshl_b32 s49, 1, s18 // (op 4345) 
s_add_i32 s50, s49, s19 // (op 4345) 


s_and_b32 s19, s0, s50 // (op 4345) 


s_add_i32 s0, s1, s19 // (op 4345) 


s_ashr_i32 s1, s0, s18 // (op 4345) 


s_lshl_b32 s0, s15, 24 // (op 4345) 
s_ashr_i32 s18, s0, 31 // (op 4345) 

s_xor_b32 s0, s1, s18 // (op 4345) 

s_sub_i32 s1, s0, s18 // (op 4345) 


s_mul_hi_i32 s0, s1, s44 // (op 4345) 
s_add_i32 s18, s0, s1 // (op 4345) 

s_ashr_i32 s0, s18, 31 // (op 4345) 
s_cmp_eq_i32 s44, 0 // (op 4345) 
s_mov_b32 s19, scc // (op 4345) 
s_cmp_eq_i32 s19, 0 // (op 4345) 
s_cselect_b32 s49, 0, -1 // (op 4345) 

s_and_b32 s19, s45, 31 // (op 4345) 
s_lshl_b32 s50, 1, s19 // (op 4345) 
s_add_i32 s51, s50, s49 // (op 4345) 


s_and_b32 s49, s0, s51 // (op 4345) 


s_add_i32 s0, s18, s49 // (op 4345) 


s_ashr_i32 s18, s0, s19 // (op 4345) 


s_lshl_b32 s0, s45, 24 // (op 4345) 
s_ashr_i32 s19, s0, 31 // (op 4345) 

s_xor_b32 s0, s18, s19 // (op 4345) 

s_sub_i32 s18, s0, s19 // (op 4345) 


s_mul_i32 s0, s1, s13 // (op 4345) 

s_sub_u32 s1, s16, s0 // (op 4345) 

s_mul_i32 s0, s18, s13 // (op 4345) 
s_add_u32 s0, s0, s1 // (op 4345) 


s_mul_hi_i32 s1, s16, s11 // (op 4345) 
s_add_i32 s18, s1, s16 // (op 4345) 

s_ashr_i32 s1, s18, 31 // (op 4345) 
s_cmp_eq_i32 s11, 0 // (op 4345) 
s_mov_b32 s19, scc // (op 4345) 
s_cmp_eq_i32 s19, 0 // (op 4345) 
s_cselect_b32 s49, 0, -1 // (op 4345) 

s_and_b32 s19, s47, 31 // (op 4345) 
s_lshl_b32 s50, 1, s19 // (op 4345) 
s_add_i32 s51, s50, s49 // (op 4345) 


s_and_b32 s49, s1, s51 // (op 4345) 


s_add_i32 s1, s18, s49 // (op 4345) 


s_ashr_i32 s18, s1, s19 // (op 4345) 


s_lshl_b32 s1, s47, 24 // (op 4345) 
s_ashr_i32 s19, s1, 31 // (op 4345) 

s_xor_b32 s1, s18, s19 // (op 4345) 

s_sub_i32 s18, s1, s19 // (op 4345) 


s_add_u32 s50, s8, 128 // (op 4345) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4345) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4345) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4345) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4345) 

s_mul_i32 s1, s13, s50 // (op 4345) 

s_mul_i32 s19, s10, s1 // (op 4345) 

s_mul_i32 s1, s18, s19 // (op 4345) 

s_sub_u32 s19, s16, s1 // (op 4345) 

s_mov_b32 s1, s19 // (op 4345) 
s_mul_hi_i32 s49, s1, s14 // (op 4345) 
s_add_i32 s50, s49, s1 // (op 4345) 


s_ashr_i32 s1, s50, 31 // (op 4345) 
s_cmp_eq_i32 s14, 0 // (op 4345) 
s_mov_b32 s49, scc // (op 4345) 
s_cmp_eq_i32 s49, 0 // (op 4345) 
s_cselect_b32 s51, 0, -1 // (op 4345) 

s_and_b32 s49, s46, 31 // (op 4345) 
s_lshl_b32 s72, 1, s49 // (op 4345) 
s_add_i32 s73, s72, s51 // (op 4345) 


s_and_b32 s51, s1, s73 // (op 4345) 


s_add_i32 s1, s50, s51 // (op 4345) 


s_ashr_i32 s50, s1, s49 // (op 4345) 


s_lshl_b32 s1, s46, 24 // (op 4345) 
s_ashr_i32 s49, s1, 31 // (op 4345) 

s_xor_b32 s1, s50, s49 // (op 4345) 

s_sub_i32 s50, s1, s49 // (op 4345) 


s_mul_i32 s1, s48, s13 // (op 4345) 
s_add_u32 s72, s6, 64 // (op 4345) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 4345) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4345) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4345) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 4345) 

s_sub_i32 s49, s72, s1 // (op 4345) 

s_mul_i32 s51, s50, s49 // (op 4345) 


s_sub_u32 s49, s19, s51 // (op 4345) 


s_add_u32 s19, s49, s1 // (op 4345) 


s_cmp_eq_i32 s18, 0 // (op 4345) 
s_mov_b32 s1, scc // (op 4345) 

s_cmp_eq_i32 s1, 0 // (op 4345) 
s_cselect_b32 s18, s19, s0 // (op 4345) 



v_lshl_add_u32 v17, s18, 6, v14 // (op 4345) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4345) 
s_lshr_b32 s18, s0, 1 // (op 4345) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 4345) 


v_mul_lo_u32 v15, v14, s34 // (op 4345) 

v_add_u32 v14, v17, v15 // (op 4345) 


s_mul_i32 s1, s34, s39 // (op 4347) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4347) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4347) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4347) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4347) most significant: sum
s_add_u32 s1, s1, s19 // (op 4347) most significant: sum


s_mov_b32 s64, s36 // (op 4347) reg expression
s_mov_b32 s65, s37 // (op 4347) reg expression
s_mov_b32 s66, s0 // (op 4347) reg expression
s_mov_b32 s67, 131072 // (op 4347) call()

s_mul_hi_i32 s0, s16, s12 // (op 4409) 
v_and_b32 v15, -128, v1 // (op 4409) 
s_add_i32 s1, s0, s16 // (op 4409) 

v_sub_u32 v17, v1, v15 // (op 4409) 

s_ashr_i32 s0, s1, 31 // (op 4409) 
v_lshrrev_b32 v15, 7, v1 // (op 4409) 
s_cmp_eq_i32 s12, 0 // (op 4409) 
s_mov_b32 s18, scc // (op 4409) 
v_lshlrev_b32 v19, 1, v15 // (op 4409) 

s_cmp_eq_i32 s18, 0 // (op 4409) 
s_cselect_b32 s19, 0, -1 // (op 4409) 

s_and_b32 s18, s15, 31 // (op 4409) 
s_lshl_b32 s49, 1, s18 // (op 4409) 
s_add_i32 s50, s49, s19 // (op 4409) 


s_and_b32 s19, s0, s50 // (op 4409) 


s_add_i32 s0, s1, s19 // (op 4409) 


s_ashr_i32 s1, s0, s18 // (op 4409) 


s_lshl_b32 s0, s15, 24 // (op 4409) 
s_ashr_i32 s18, s0, 31 // (op 4409) 

s_xor_b32 s0, s1, s18 // (op 4409) 

s_sub_i32 s1, s0, s18 // (op 4409) 


s_mul_hi_i32 s0, s1, s44 // (op 4409) 
s_add_i32 s18, s0, s1 // (op 4409) 

s_ashr_i32 s0, s18, 31 // (op 4409) 
s_cmp_eq_i32 s44, 0 // (op 4409) 
s_mov_b32 s19, scc // (op 4409) 
s_cmp_eq_i32 s19, 0 // (op 4409) 
s_cselect_b32 s49, 0, -1 // (op 4409) 

s_and_b32 s19, s45, 31 // (op 4409) 
s_lshl_b32 s50, 1, s19 // (op 4409) 
s_add_i32 s51, s50, s49 // (op 4409) 


s_and_b32 s49, s0, s51 // (op 4409) 


s_add_i32 s0, s18, s49 // (op 4409) 


s_ashr_i32 s18, s0, s19 // (op 4409) 


s_lshl_b32 s0, s45, 24 // (op 4409) 
s_ashr_i32 s19, s0, 31 // (op 4409) 

s_xor_b32 s0, s18, s19 // (op 4409) 

s_sub_i32 s18, s0, s19 // (op 4409) 


s_add_u32 s50, s8, 128 // (op 4409) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4409) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4409) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4409) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4409) 

s_mul_i32 s0, s18, s50 // (op 4409) 

s_sub_i32 s49, s1, s0 // (op 4409) 


s_mul_hi_i32 s0, s16, s11 // (op 4409) 
s_add_i32 s1, s0, s16 // (op 4409) 

s_ashr_i32 s0, s1, 31 // (op 4409) 
s_cmp_eq_i32 s11, 0 // (op 4409) 
s_mov_b32 s18, scc // (op 4409) 
s_cmp_eq_i32 s18, 0 // (op 4409) 
s_cselect_b32 s19, 0, -1 // (op 4409) 

s_and_b32 s18, s47, 31 // (op 4409) 
s_lshl_b32 s72, 1, s18 // (op 4409) 
s_add_i32 s73, s72, s19 // (op 4409) 


s_and_b32 s19, s0, s73 // (op 4409) 


s_add_i32 s0, s1, s19 // (op 4409) 


s_ashr_i32 s1, s0, s18 // (op 4409) 


s_lshl_b32 s0, s47, 24 // (op 4409) 
s_ashr_i32 s18, s0, 31 // (op 4409) 

s_xor_b32 s0, s1, s18 // (op 4409) 

s_sub_i32 s1, s0, s18 // (op 4409) 


s_mul_i32 s0, s13, s50 // (op 4409) 

s_mul_i32 s18, s10, s0 // (op 4409) 

s_mul_i32 s0, s1, s18 // (op 4409) 

s_sub_u32 s18, s16, s0 // (op 4409) 

s_mul_hi_i32 s0, s18, s14 // (op 4409) 
s_add_i32 s19, s0, s18 // (op 4409) 


s_ashr_i32 s0, s19, 31 // (op 4409) 
s_cmp_eq_i32 s14, 0 // (op 4409) 
s_mov_b32 s18, scc // (op 4409) 
s_cmp_eq_i32 s18, 0 // (op 4409) 
s_cselect_b32 s50, 0, -1 // (op 4409) 

s_and_b32 s18, s46, 31 // (op 4409) 
s_lshl_b32 s51, 1, s18 // (op 4409) 
s_add_i32 s72, s51, s50 // (op 4409) 


s_and_b32 s50, s0, s72 // (op 4409) 


s_add_i32 s0, s19, s50 // (op 4409) 


s_ashr_i32 s19, s0, s18 // (op 4409) 


s_lshl_b32 s0, s46, 24 // (op 4409) 
s_ashr_i32 s18, s0, 31 // (op 4409) 

s_xor_b32 s0, s19, s18 // (op 4409) 

s_sub_i32 s19, s0, s18 // (op 4409) 


s_cmp_eq_i32 s1, 0 // (op 4409) 
s_mov_b32 s0, scc // (op 4409) 

s_cmp_eq_i32 s0, 0 // (op 4409) 
s_cselect_b32 s1, s19, s49 // (op 4409) 



v_lshl_add_u32 v15, s1, 7, v17 // (op 4409) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4409) 
s_lshr_b32 s18, s0, 1 // (op 4409) 

v_lshl_add_u32 v17, s18, 3, v19 // (op 4409) 


v_mul_lo_u32 v19, v17, s40 // (op 4409) 
v_add_u32 v19, v19, v15 // (op 4409) 


s_mul_i32 s1, s40, s39 // (op 4411) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4411) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4411) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4411) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4411) most significant: sum
s_add_u32 s1, s1, s19 // (op 4411) most significant: sum


s_mov_b32 s68, s42 // (op 4411) reg expression
s_mov_b32 s69, s43 // (op 4411) reg expression
s_mov_b32 s70, s0 // (op 4411) reg expression
s_mov_b32 s71, 131072 // (op 4411) call()

v_and_b32 v15, -2, v1 // (op 4492) 
v_sub_u32 v17, v1, v15 // (op 4492) 

v_lshlrev_b32 v15, 5, v17 // (op 4492) 

v_lshrrev_b32 v17, 1, v1 // (op 4492) 
v_lshl_add_u32 v20, v17, 6, v15 // (op 4492) 


v_mul_lo_u32 v15, v20, 6 // (op 4492) 
v_lshrrev_b32 v17, 3, v15 // (op 4492) 

v_lshrrev_b32 v15, 4, v20 // (op 4492) 

v_lshlrev_b32 v20, 2, v15 // (op 4492) 

v_add_u32 v15, v17, v20 // (op 4492) 


v_and_b32 v17, -4, v1 // (op 4552) 
v_sub_u32 v20, v1, v17 // (op 4552) 

v_lshlrev_b32 v17, 5, v20 // (op 4552) 

v_lshrrev_b32 v20, 2, v1 // (op 4552) 
v_lshl_add_u32 v21, v20, 8, v17 // (op 4552) 


v_lshlrev_b32 v17, 2, v21 // (op 4552) 

v_lshrrev_b32 v20, 3, v17 // (op 4552) 

v_and_b32 v17, -64, v1 // (op 4610) 
v_sub_u32 v21, v1, v17 // (op 4610) 

v_and_b32 v17, 4294967232, v1 // (op 4610) 
v_add_u32 v22, v17, v21 // (op 4610) 


v_and_b32 v17, -128, v1 // (op 4668) 
v_sub_u32 v21, v1, v17 // (op 4668) 

v_lshrrev_b32 v17, 7, v1 // (op 4668) 
v_lshl_add_u32 v23, v17, 8, v21 // (op 4668) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1021) 
s_and_b32 s18, s0, 4294967294 // (op 1021) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 1020) 
s_mov_b32 s19, s0 // (op 1020) convert

s_cmp_lt_u32 s18, s19 // (op 1020) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1020 // (op 1020) Condition: Top (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1020 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1020:


s_add_u32 s18, s18, 1 // (op 1022) 
v_mov_b32 v21, v57 // (op 176) 
v_mov_b32 v17, v21 // (op 176) reg expression

buffer_load_dwordx3 v[24:26], v17, s[20:23], 0 offen // (op 176) Load value
buffer_load_dwordx3 v[28:30], v17, s[20:23], 0 offen offset:12 // (op 176) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 4197) 
s_mul_i32 s51, s1, 6 // (op 4197) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 4197) least significant
s_mul_hi_u32 s19, s0, 6 // (op 4197) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4197) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 4197) 

v_add_u32 v57, v57, s0 // (op 4197) 


s_barrier  // (op 4922) 
s_lshl_b64 s[0:1], s[30:31], 2 // (op 190) 



v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 190) 


v_mov_b32 v21, v16 // (op 190) 
v_mov_b32 v27, v21 // (op 190) reg expression
buffer_load_dwordx4 v[34:37], v27, s[52:55], 0 offen // (op 190) Load value
v_add_u32 v27, v27, v32 // (op 190) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[24:26] offset:8704 // (op 177) Store local data 
buffer_load_dwordx4 v[38:41], v27, s[52:55], 0 offen // (op 190) Load value



s_lshl_b64 s[0:1], s[30:31], 7 // (op 4273) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 4273) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 4273) 

v_add_u32 v16, v16, s0 // (op 4273) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[28:30] offset:8720 // (op 177) Store local data 



v_mov_b32 v21, v14 // (op 191) 
s_barrier  // (op 4935) 
v_mov_b32 v17, v21 // (op 191) reg expression




buffer_load_ubyte v18, v17, s[64:67], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s19, s34, 2 // (op 4337) 
v_add_u32 v14, v14, s19 // (op 4337) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[34:37] // (op 187) Store local data 

v_mov_b64 v[24:25], s[40:41] // (op 192) convert

v_mov_b32 v17, v19 // (op 192) 
v_mov_b32 v21, v17 // (op 192) reg expression
buffer_load_ubyte v26, v21, s[68:71], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v21, v21, v24 // (op 192) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[38:41] offset:64 // (op 187) Store local data 
buffer_load_ubyte v27, v21, s[68:71], 0 offen offset: 0 // (op 192) Load value




s_lshl_b32 s19, s40, 2 // (op 4401) 
s_barrier  // (op 4949) 
v_add_u32 v19, v19, s19 // (op 4401) 




s_waitcnt lgkmcnt(2)
s_barrier  // (op 4925) 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v18 offset:16896 // (op 188) Store local data 




ds_read_b96_tr_b6 v[28:30], v0 offset:8704 // (op 179) Transpose load local data 
s_barrier  // (op 4963) 
ds_read_b96_tr_b6 v[31:33], v0 offset:9728 // (op 179) Transpose load local data 



s_waitcnt lgkmcnt(3)
s_barrier  // (op 4939) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v26 offset:8192 // (op 189) Store local data 



ds_read_b64_tr_b4 v[34:35], v4 // (op 181) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v4 offset:1024 // (op 181) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v27 offset:8320 // (op 189) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 4953) 


ds_read_u8 v21, v3 offset:16896 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4967) 


ds_read_u8 v17, v5 offset:8192 // (op 185) Load local data 



ds_read_b64_tr_b4 v[24:25], v4 offset:16 // (op 350) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v4 offset:1040 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:37], a[0:3], v21, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 186) 


ds_read_u8 v18, v5 offset:8224 // (op 354) Load local data 



ds_read_b64_tr_b4 v[38:39], v4 offset:32 // (op 526) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:1056 // (op 526) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:33], v[24:27], a[4:7], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 355) 


ds_read_u8 v42, v5 offset:8256 // (op 530) Load local data 



ds_read_b64_tr_b4 v[44:45], v4 offset:48 // (op 702) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v4 offset:1072 // (op 702) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[38:41], a[8:11], v21, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 531) 


ds_read_u8 v43, v5 offset:8288 // (op 706) Load local data 



ds_read_b96_tr_b6 v[48:50], v0 offset:8736 // (op 964) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v0 offset:9760 // (op 964) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:33], v[44:47], a[12:15], v21, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 707) 




ds_read_u8 v21, v3 offset:16928 // (op 967) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[34:37], a[16:19], v21, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 969) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[24:27], a[20:23], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1000) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[38:41], a[24:27], v21, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1031) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[44:47], a[28:31], v21, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1062) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 1020) 
s_mov_b32 s19, s0 // (op 1020) convert

s_cmp_lt_u32 s18, s19 // (op 1020) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1020 // (op 1020) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1020 if true)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1020:























s_barrier  // (op 5061) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1881) 
v_accvgpr_read v7, a9 // (op 1881) 
v_accvgpr_read v8, a10 // (op 1881) 
v_accvgpr_read v9, a11 // (op 1881) 
v_mul_f32 v14, v11, v6 // (op 1881) 
v_mul_f32 v15, v11, v7 // (op 1881) 
v_mul_f32 v16, v11, v8 // (op 1881) 
v_mul_f32 v17, v11, v9 // (op 1881) 



s_lshl_b32 s0, s4, 8 // (op 1883) 
v_add_u32 v0, v13, s0 // (op 1883) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1883) Store value


v_accvgpr_read v6, a12 // (op 1893) 
v_accvgpr_read v7, a13 // (op 1893) 
v_accvgpr_read v8, a14 // (op 1893) 
v_accvgpr_read v9, a15 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



s_lshl_b32 s0, s4, 7 // (op 1895) 
s_mul_i32 s1, 3, s0 // (op 1895) 

v_add_u32 v0, v13, s1 // (op 1895) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1895) Store value


v_accvgpr_read v6, a16 // (op 1927) 
v_accvgpr_read v7, a17 // (op 1927) 
v_accvgpr_read v8, a18 // (op 1927) 
v_accvgpr_read v9, a19 // (op 1927) 
v_mul_f32 v14, v11, v6 // (op 1927) 
v_mul_f32 v15, v11, v7 // (op 1927) 
v_mul_f32 v16, v11, v8 // (op 1927) 
v_mul_f32 v17, v11, v9 // (op 1927) 



v_add_u32 v0, 128, v13 // (op 1930) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1930) Store value


v_accvgpr_read v6, a20 // (op 1940) 
v_accvgpr_read v7, a21 // (op 1940) 
v_accvgpr_read v8, a22 // (op 1940) 
v_accvgpr_read v9, a23 // (op 1940) 
v_mul_f32 v14, v11, v6 // (op 1940) 
v_mul_f32 v15, v11, v7 // (op 1940) 
v_mul_f32 v16, v11, v8 // (op 1940) 
v_mul_f32 v17, v11, v9 // (op 1940) 



s_lshl_b32 s0, s4, 7 // (op 1932) 
s_add_u32 s0, s0, 128 // (op 1932) 
v_add_u32 v0, v13, s0 // (op 1932) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a24 // (op 1945) 
v_accvgpr_read v7, a25 // (op 1945) 
v_accvgpr_read v8, a26 // (op 1945) 
v_accvgpr_read v9, a27 // (op 1945) 
v_mul_f32 v14, v11, v6 // (op 1945) 
v_mul_f32 v15, v11, v7 // (op 1945) 
v_mul_f32 v16, v11, v8 // (op 1945) 
v_mul_f32 v17, v11, v9 // (op 1945) 



s_lshl_b32 s0, s4, 8 // (op 1934) 
s_add_u32 s0, s0, 128 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a28 // (op 1950) 
v_accvgpr_read v7, a29 // (op 1950) 
v_accvgpr_read v8, a30 // (op 1950) 
v_accvgpr_read v9, a31 // (op 1950) 
v_mul_f32 v14, v11, v6 // (op 1950) 
v_mul_f32 v15, v11, v7 // (op 1950) 
v_mul_f32 v16, v11, v8 // (op 1950) 
v_mul_f32 v17, v11, v9 // (op 1950) 



s_lshl_b32 s0, s4, 7 // (op 1936) 
s_mul_i32 s1, 3, s0 // (op 1936) 
s_add_u32 s1, s1, 128 // (op 1936) 

v_add_u32 v0, v13, s1 // (op 1936) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1936) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 2108) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2108) Load value

v_accvgpr_read v14, a16 // (op 2096) 
v_accvgpr_read v15, a17 // (op 2096) 
v_accvgpr_read v16, a18 // (op 2096) 
v_accvgpr_read v17, a19 // (op 2096) 
v_mul_f32 v18, v11, v14 // (op 2096) 
v_mul_f32 v19, v11, v15 // (op 2096) 
v_mul_f32 v20, v11, v16 // (op 2096) 
v_mul_f32 v21, v11, v17 // (op 2096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2107) 
v_mul_f32 v15, v10, v7 // (op 2107) 
v_mul_f32 v16, v10, v8 // (op 2107) 
v_mul_f32 v17, v10, v9 // (op 2107) 

v_add_f32 v6, v14, v18 // (op 2097) 
v_add_f32 v7, v15, v19 // (op 2097) 
v_add_f32 v8, v16, v20 // (op 2097) 
v_add_f32 v9, v17, v21 // (op 2097) 




v_add_u32 v0, 128, v13 // (op 2099) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2099) Store value




s_lshl_b32 s0, s2, 7 // (op 2112) 
s_add_u32 s0, s0, 128 // (op 2112) 
v_add_u32 v0, v12, s0 // (op 2112) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2112) Load value

v_accvgpr_read v14, a20 // (op 2122) 
v_accvgpr_read v15, a21 // (op 2122) 
v_accvgpr_read v16, a22 // (op 2122) 
v_accvgpr_read v17, a23 // (op 2122) 
v_mul_f32 v18, v11, v14 // (op 2122) 
v_mul_f32 v19, v11, v15 // (op 2122) 
v_mul_f32 v20, v11, v16 // (op 2122) 
v_mul_f32 v21, v11, v17 // (op 2122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2111) 
v_add_f32 v7, v15, v19 // (op 2111) 
v_add_f32 v8, v16, v20 // (op 2111) 
v_add_f32 v9, v17, v21 // (op 2111) 




s_lshl_b32 s0, s4, 7 // (op 2101) 
s_add_u32 s0, s0, 128 // (op 2101) 
v_add_u32 v0, v13, s0 // (op 2101) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2101) Store value




s_lshl_b32 s0, s2, 8 // (op 2116) 
s_add_u32 s0, s0, 128 // (op 2116) 
v_add_u32 v0, v12, s0 // (op 2116) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2116) Load value

v_accvgpr_read v14, a24 // (op 2124) 
v_accvgpr_read v15, a25 // (op 2124) 
v_accvgpr_read v16, a26 // (op 2124) 
v_accvgpr_read v17, a27 // (op 2124) 
v_mul_f32 v18, v11, v14 // (op 2124) 
v_mul_f32 v19, v11, v15 // (op 2124) 
v_mul_f32 v20, v11, v16 // (op 2124) 
v_mul_f32 v21, v11, v17 // (op 2124) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2114) 
v_mul_f32 v15, v10, v7 // (op 2114) 
v_mul_f32 v16, v10, v8 // (op 2114) 
v_mul_f32 v17, v10, v9 // (op 2114) 

v_add_f32 v6, v14, v18 // (op 2115) 
v_add_f32 v7, v15, v19 // (op 2115) 
v_add_f32 v8, v16, v20 // (op 2115) 
v_add_f32 v9, v17, v21 // (op 2115) 




s_lshl_b32 s0, s4, 8 // (op 2103) 
s_add_u32 s0, s0, 128 // (op 2103) 
v_add_u32 v0, v13, s0 // (op 2103) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2103) Store value




s_lshl_b32 s0, s2, 7 // (op 2120) 
s_mul_i32 s1, 3, s0 // (op 2120) 
s_add_u32 s1, s1, 128 // (op 2120) 

v_add_u32 v0, v12, s1 // (op 2120) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2120) Load value

v_accvgpr_read v14, a28 // (op 2126) 
v_accvgpr_read v15, a29 // (op 2126) 
v_accvgpr_read v16, a30 // (op 2126) 
v_accvgpr_read v17, a31 // (op 2126) 
v_mul_f32 v18, v11, v14 // (op 2126) 
v_mul_f32 v19, v11, v15 // (op 2126) 
v_mul_f32 v20, v11, v16 // (op 2126) 
v_mul_f32 v21, v11, v17 // (op 2126) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2118) 
v_mul_f32 v15, v10, v7 // (op 2118) 
v_mul_f32 v16, v10, v8 // (op 2118) 
v_mul_f32 v17, v10, v9 // (op 2118) 

v_add_f32 v6, v14, v18 // (op 2119) 
v_add_f32 v7, v15, v19 // (op 2119) 
v_add_f32 v8, v16, v20 // (op 2119) 
v_add_f32 v9, v17, v21 // (op 2119) 




s_lshl_b32 s0, s4, 7 // (op 2105) 
s_mul_i32 s1, 3, s0 // (op 2105) 
s_add_u32 s1, s1, 128 // (op 2105) 

v_add_u32 v0, v13, s1 // (op 2105) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2105) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 2022) 
v_add_u32 v0, v12, s0 // (op 2022) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2022) Load value

v_accvgpr_read v14, a4 // (op 2019) 
v_accvgpr_read v15, a5 // (op 2019) 
v_accvgpr_read v16, a6 // (op 2019) 
v_accvgpr_read v17, a7 // (op 2019) 
v_mul_f32 v18, v11, v14 // (op 2019) 
v_mul_f32 v19, v11, v15 // (op 2019) 
v_mul_f32 v20, v11, v16 // (op 2019) 
v_mul_f32 v21, v11, v17 // (op 2019) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2023) 
v_mul_f32 v15, v10, v7 // (op 2023) 
v_mul_f32 v16, v10, v8 // (op 2023) 
v_mul_f32 v17, v10, v9 // (op 2023) 

v_add_f32 v6, v14, v18 // (op 2020) 
v_add_f32 v7, v15, v19 // (op 2020) 
v_add_f32 v8, v16, v20 // (op 2020) 
v_add_f32 v9, v17, v21 // (op 2020) 




s_lshl_b32 s0, s4, 7 // (op 2021) 
v_add_u32 v0, v13, s0 // (op 2021) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2021) Store value




s_lshl_b32 s0, s2, 8 // (op 2034) 
v_add_u32 v0, v12, s0 // (op 2034) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2034) Load value

v_accvgpr_read v14, a8 // (op 2031) 
v_accvgpr_read v15, a9 // (op 2031) 
v_accvgpr_read v16, a10 // (op 2031) 
v_accvgpr_read v17, a11 // (op 2031) 
v_mul_f32 v18, v11, v14 // (op 2031) 
v_mul_f32 v19, v11, v15 // (op 2031) 
v_mul_f32 v20, v11, v16 // (op 2031) 
v_mul_f32 v21, v11, v17 // (op 2031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2035) 
v_mul_f32 v15, v10, v7 // (op 2035) 
v_mul_f32 v16, v10, v8 // (op 2035) 
v_mul_f32 v17, v10, v9 // (op 2035) 

v_add_f32 v6, v14, v18 // (op 2032) 
v_add_f32 v7, v15, v19 // (op 2032) 
v_add_f32 v8, v16, v20 // (op 2032) 
v_add_f32 v9, v17, v21 // (op 2032) 




s_lshl_b32 s0, s4, 8 // (op 2033) 
v_add_u32 v0, v13, s0 // (op 2033) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2033) Store value




s_lshl_b32 s0, s2, 7 // (op 2046) 
s_mul_i32 s1, 3, s0 // (op 2046) 

v_add_u32 v0, v12, s1 // (op 2046) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2046) Load value

v_accvgpr_read v14, a12 // (op 2043) 
v_accvgpr_read v15, a13 // (op 2043) 
v_accvgpr_read v16, a14 // (op 2043) 
v_accvgpr_read v17, a15 // (op 2043) 
v_mul_f32 v18, v11, v14 // (op 2043) 
v_mul_f32 v19, v11, v15 // (op 2043) 
v_mul_f32 v20, v11, v16 // (op 2043) 
v_mul_f32 v21, v11, v17 // (op 2043) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2047) 
v_mul_f32 v15, v10, v7 // (op 2047) 
v_mul_f32 v16, v10, v8 // (op 2047) 
v_mul_f32 v17, v10, v9 // (op 2047) 

v_add_f32 v6, v14, v18 // (op 2044) 
v_add_f32 v7, v15, v19 // (op 2044) 
v_add_f32 v8, v16, v20 // (op 2044) 
v_add_f32 v9, v17, v21 // (op 2044) 




s_lshl_b32 s0, s4, 7 // (op 2045) 
s_mul_i32 s1, 3, s0 // (op 2045) 

v_add_u32 v0, v13, s1 // (op 2045) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2045) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34304 // lds bytes
  .amdhsa_accum_offset 100
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F4
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 34304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 98
    .agpr_count: 32
    .max_flat_workgroup_size: 256
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

