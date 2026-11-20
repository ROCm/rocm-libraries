// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 8918) 
s_mul_hi_i32 s1, s16, s10 // (op 9175) 
v_and_b32 v0, 63, v1 // (op 8980) 
v_and_b32 v3, 48, v1 // (op 8980) 
v_sub_u32 v4, v0, v3 // (op 8980) 

v_lshrrev_b32 v3, 4, v0 // (op 8980) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 8980) 


v_lshrrev_b32 v3, 7, v1 // (op 8980) 
v_lshlrev_b32 v4, 4, v3 // (op 8980) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 8980) 


v_mul_lo_u32 v0, v3, 6 // (op 8980) 
v_lshrrev_b32 v4, 3, v0 // (op 8980) 

v_lshrrev_b32 v0, 4, v3 // (op 8980) 

v_lshlrev_b32 v3, 2, v0 // (op 8980) 

v_add_u32 v0, v4, v3 // (op 8980) 


v_lshrrev_b32 v3, 6, v1 // (op 9038) 
v_and_b32 v4, 1, v3 // (op 9038) 

v_and_b32 v3, 15, v1 // (op 9038) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 9038) 


v_and_b32 v3, 63, v1 // (op 9038) 
v_lshrrev_b32 v4, 4, v3 // (op 9038) 

v_lshlrev_b32 v3, 5, v4 // (op 9038) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 9038) 


v_mul_lo_u32 v3, v4, 6 // (op 9038) 

v_lshrrev_b32 v4, 3, v3 // (op 9038) 

v_lshrrev_b32 v3, 7, v1 // (op 9096) 
v_and_b32 v5, 15, v1 // (op 9096) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 9096) 


v_and_b32 v3, 63, v1 // (op 9096) 
v_lshrrev_b32 v5, 4, v3 // (op 9096) 

v_lshl_add_u32 v3, v5, 7, v6 // (op 9096) 


v_lshrrev_b32 v5, 6, v1 // (op 9154) 
v_and_b32 v6, 1, v5 // (op 9154) 

v_and_b32 v5, 15, v1 // (op 9154) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9154) 


v_and_b32 v5, 63, v1 // (op 9154) 
v_lshrrev_b32 v6, 4, v5 // (op 9154) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9154) 


v_and_b32 v6, 63, v1 // (op 9206) 
v_and_b32 v7, 48, v1 // (op 9206) 
v_sub_u32 v8, v6, v7 // (op 9206) 

v_lshrrev_b32 v7, 4, v6 // (op 9206) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 9206) 


v_lshrrev_b32 v7, 7, v1 // (op 9206) 
v_lshlrev_b32 v8, 4, v7 // (op 9206) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 9206) 


v_mul_lo_u32 v6, v7, 6 // (op 9206) 
v_lshrrev_b32 v8, 3, v6 // (op 9206) 

v_lshrrev_b32 v6, 4, v7 // (op 9206) 

v_lshlrev_b32 v7, 2, v6 // (op 9206) 

v_add_u32 v6, v8, v7 // (op 9206) 


v_lshrrev_b32 v7, 6, v1 // (op 9235) 
v_and_b32 v8, 1, v7 // (op 9235) 

v_and_b32 v7, 15, v1 // (op 9235) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 9235) 


v_and_b32 v7, 63, v1 // (op 9235) 
v_lshrrev_b32 v8, 4, v7 // (op 9235) 

v_lshlrev_b32 v7, 5, v8 // (op 9235) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 9235) 


v_mul_lo_u32 v7, v8, 6 // (op 9235) 

v_lshrrev_b32 v8, 3, v7 // (op 9235) 

v_lshrrev_b32 v7, 7, v1 // (op 9264) 
v_and_b32 v9, 15, v1 // (op 9264) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 9264) 


v_and_b32 v7, 63, v1 // (op 9264) 
v_lshrrev_b32 v9, 4, v7 // (op 9264) 

v_lshl_add_u32 v7, v9, 7, v10 // (op 9264) 


v_lshrrev_b32 v9, 6, v1 // (op 9293) 
v_and_b32 v10, 1, v9 // (op 9293) 

v_and_b32 v9, 15, v1 // (op 9293) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9293) 


v_and_b32 v9, 63, v1 // (op 9293) 
v_lshrrev_b32 v10, 4, v9 // (op 9293) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9293) 


v_lshrrev_b32 v10, 7, v1 // (op 8918) 
s_add_i32 s18, s0, s16 // (op 8918) 

v_and_b32 v11, 63, v1 // (op 8918) 
s_ashr_i32 s0, s18, 31 // (op 8918) 
v_lshrrev_b32 v12, 4, v11 // (op 8918) 
s_cmp_eq_i32 s10, 0 // (op 8918) 
s_mov_b32 s19, scc // (op 8918) 
v_lshlrev_b32 v13, 2, v12 // (op 8918) 

s_cmp_eq_i32 s19, 0 // (op 8918) 
s_cselect_b32 s51, 0, -1 // (op 8918) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8918) 


s_and_b32 s19, s11, 31 // (op 8918) 
v_lshrrev_b32 v10, 2, v11 // (op 8918) 

v_and_b32 v11, 3, v10 // (op 8918) 

v_lshrrev_b32 v10, 7, v1 // (op 9175) 
v_and_b32 v13, 63, v1 // (op 9175) 
s_lshl_b32 s52, 1, s19 // (op 8918) 
v_and_b32 v14, 3, v1 // (op 8918) 
s_add_i32 s53, s52, s51 // (op 8918) 


s_and_b32 s51, s0, s53 // (op 8918) 


s_add_i32 s0, s1, s16 // (op 9175) 

s_ashr_i32 s1, s0, 31 // (op 9175) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 8918) 


s_add_i32 s52, s18, s51 // (op 8918) 


v_lshrrev_b32 v11, 6, v1 // (op 8918) 
s_ashr_i32 s18, s52, s19 // (op 8918) 


v_and_b32 v14, 1, v11 // (op 8918) 

s_lshl_b32 s19, s11, 24 // (op 8918) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 8918) 


s_ashr_i32 s51, s19, 31 // (op 8918) 

s_xor_b32 s19, s18, s51 // (op 8918) 

s_sub_i32 s18, s19, s51 // (op 8918) 


s_mul_hi_i32 s19, s18, s12 // (op 8918) 
s_add_i32 s51, s19, s18 // (op 8918) 

s_ashr_i32 s19, s51, 31 // (op 8918) 
s_cmp_eq_i32 s12, 0 // (op 8918) 
s_mov_b32 s52, scc // (op 8918) 
s_cmp_eq_i32 s52, 0 // (op 8918) 
s_cselect_b32 s53, 0, -1 // (op 8918) 

s_and_b32 s52, s13, 31 // (op 8918) 
v_lshrrev_b32 v14, 4, v13 // (op 9175) 
s_cmp_eq_i32 s10, 0 // (op 9175) 
s_mov_b32 s54, scc // (op 9175) 
v_lshlrev_b32 v15, 2, v14 // (op 9175) 

s_cmp_eq_i32 s54, 0 // (op 9175) 
s_cselect_b32 s55, 0, -1 // (op 9175) 

s_lshl_b32 s54, 1, s52 // (op 8918) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9175) 


v_lshrrev_b32 v10, 2, v13 // (op 9175) 

s_add_i32 s56, s54, s53 // (op 8918) 


s_and_b32 s53, s19, s56 // (op 8918) 


s_add_i32 s19, s51, s53 // (op 8918) 


s_ashr_i32 s51, s19, s52 // (op 8918) 


s_lshl_b32 s19, s13, 24 // (op 8918) 
s_ashr_i32 s52, s19, 31 // (op 8918) 

s_xor_b32 s19, s51, s52 // (op 8918) 

s_sub_i32 s51, s19, s52 // (op 8918) 


s_mul_i32 s19, s18, s15 // (op 8918) 
s_sub_u32 s52, s16, s19 // (op 8918) 

s_mul_i32 s19, s51, s15 // (op 8918) 
s_add_u32 s19, s19, s52 // (op 8918) 

s_mul_hi_i32 s52, s16, s9 // (op 8918) 
s_add_i32 s53, s52, s16 // (op 8918) 

s_ashr_i32 s52, s53, 31 // (op 8918) 
s_cmp_eq_i32 s9, 0 // (op 8918) 
s_mov_b32 s54, scc // (op 8918) 
s_cmp_eq_i32 s54, 0 // (op 8918) 
s_cselect_b32 s56, 0, -1 // (op 8918) 

s_and_b32 s54, s11, 31 // (op 9175) 
v_and_b32 v13, 3, v10 // (op 9175) 

s_lshl_b32 s57, 1, s54 // (op 9175) 
v_and_b32 v10, 3, v1 // (op 9175) 
s_add_i32 s58, s57, s55 // (op 9175) 


s_and_b32 s55, s1, s58 // (op 9175) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9175) 


s_add_i32 s1, s0, s55 // (op 9175) 


v_lshrrev_b32 v10, 6, v1 // (op 9175) 
s_ashr_i32 s0, s1, s54 // (op 9175) 


v_and_b32 v13, 1, v10 // (op 9175) 

s_lshl_b32 s1, s11, 24 // (op 9175) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9175) 


s_ashr_i32 s54, s1, 31 // (op 9175) 

s_xor_b32 s1, s0, s54 // (op 9175) 

s_sub_i32 s0, s1, s54 // (op 9175) 


s_mul_hi_i32 s1, s0, s12 // (op 9175) 
s_add_i32 s54, s1, s0 // (op 9175) 

s_ashr_i32 s1, s54, 31 // (op 9175) 
s_cmp_eq_i32 s12, 0 // (op 9175) 
s_mov_b32 s55, scc // (op 9175) 
s_cmp_eq_i32 s55, 0 // (op 9175) 
s_cselect_b32 s57, 0, -1 // (op 9175) 

s_and_b32 s55, s13, 31 // (op 9175) 
s_lshl_b32 s58, 1, s55 // (op 9175) 
s_add_i32 s59, s58, s57 // (op 9175) 


s_and_b32 s57, s1, s59 // (op 9175) 


s_add_i32 s1, s54, s57 // (op 9175) 


s_ashr_i32 s54, s1, s55 // (op 9175) 


s_lshl_b32 s1, s13, 24 // (op 9175) 
s_ashr_i32 s55, s1, 31 // (op 9175) 

s_xor_b32 s1, s54, s55 // (op 9175) 

s_sub_i32 s54, s1, s55 // (op 9175) 


s_mul_i32 s1, s0, s15 // (op 9175) 
s_sub_u32 s55, s16, s1 // (op 9175) 

s_mul_i32 s1, s54, s15 // (op 9175) 
s_add_u32 s1, s1, s55 // (op 9175) 

s_mul_hi_i32 s55, s16, s9 // (op 9175) 
s_add_i32 s57, s55, s16 // (op 9175) 

s_ashr_i32 s55, s57, 31 // (op 9175) 
s_cmp_eq_i32 s9, 0 // (op 9175) 
s_mov_b32 s58, scc // (op 9175) 
s_cmp_eq_i32 s58, 0 // (op 9175) 
s_cselect_b32 s59, 0, -1 // (op 9175) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 8918) 
s_lshl_b32 s60, 1, s58 // (op 8918) 
s_add_i32 s61, s60, s56 // (op 8918) 


s_and_b32 s56, s52, s61 // (op 8918) 


s_add_i32 s52, s53, s56 // (op 8918) 


s_ashr_i32 s53, s52, s58 // (op 8918) 


s_lshl_b32 s52, s46, 24 // (op 8918) 
s_ashr_i32 s56, s52, 31 // (op 8918) 

s_xor_b32 s52, s53, s56 // (op 8918) 

s_sub_i32 s53, s52, s56 // (op 8918) 


s_and_b32 s52, s46, 31 // (op 9175) 
s_lshl_b32 s56, 1, s52 // (op 9175) 
s_add_i32 s58, s56, s59 // (op 9175) 


s_and_b32 s56, s55, s58 // (op 9175) 


s_add_i32 s55, s57, s56 // (op 9175) 


s_ashr_i32 s56, s55, s52 // (op 9175) 


s_lshl_b32 s52, s46, 24 // (op 9175) 
s_ashr_i32 s55, s52, 31 // (op 9175) 

s_xor_b32 s52, s56, s55 // (op 9175) 

s_sub_i32 s56, s52, s55 // (op 9175) 


s_add_u32 s58, s6, 128 // (op 8918) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 8918) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 8918) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 8918) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 8918) 

s_mul_i32 s52, s15, s58 // (op 8918) 
s_mul_i32 s55, s8, s52 // (op 8918) 

s_mul_i32 s52, s53, s55 // (op 8918) 

s_sub_u32 s55, s16, s52 // (op 8918) 

s_mov_b32 s52, s55 // (op 8918) 
s_mul_hi_i32 s57, s52, s14 // (op 8918) 
s_add_i32 s60, s57, s52 // (op 8918) 


s_ashr_i32 s52, s60, 31 // (op 8918) 
s_cmp_eq_i32 s14, 0 // (op 8918) 
s_mov_b32 s57, scc // (op 8918) 
s_cmp_eq_i32 s57, 0 // (op 8918) 
s_cselect_b32 s61, 0, -1 // (op 8918) 

s_and_b32 s57, s48, 31 // (op 8918) 
s_lshl_b32 s62, 1, s57 // (op 8918) 
s_add_i32 s63, s62, s61 // (op 8918) 


s_and_b32 s61, s52, s63 // (op 8918) 


s_add_i32 s52, s60, s61 // (op 8918) 


s_ashr_i32 s60, s52, s57 // (op 8918) 


s_lshl_b32 s52, s48, 24 // (op 8918) 
s_ashr_i32 s57, s52, 31 // (op 8918) 

s_xor_b32 s52, s60, s57 // (op 8918) 

s_sub_i32 s60, s52, s57 // (op 8918) 


s_mul_i32 s52, s47, s15 // (op 8918) 
s_add_u32 s62, s20, 128 // (op 8918) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 8918) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 8918) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 8918) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 8918) 

s_sub_i32 s57, s62, s52 // (op 8918) 

s_mul_i32 s61, s60, s57 // (op 8918) 

s_sub_u32 s57, s55, s61 // (op 8918) 


s_add_u32 s55, s57, s52 // (op 8918) 


s_cmp_eq_i32 s53, 0 // (op 8918) 
s_mov_b32 s52, scc // (op 8918) 

s_cmp_eq_i32 s52, 0 // (op 8918) 
s_cselect_b32 s53, s55, s19 // (op 8918) 


s_mul_i32 s19, s51, s58 // (op 8918) 


v_lshl_add_u32 v13, s53, 7, v12 // (op 8918) 


s_sub_i32 s51, s18, s19 // (op 8918) 


s_cmp_eq_i32 s52, 0 // (op 8918) 
s_cselect_b32 s53, s60, s51 // (op 8918) 



v_lshl_add_u32 v12, s53, 7, v11 // (op 8918) 


v_mul_lo_u32 v11, v12, s2 // (op 8918) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 8918) 


s_add_u32 s18, s6, 128 // (op 9175) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 9175) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 9175) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 9175) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 7 // (op 9175) 

s_mul_i32 s55, s15, s18 // (op 9175) 
s_mul_i32 s51, s8, s55 // (op 9175) 

s_mul_i32 s55, s56, s51 // (op 9175) 

s_sub_u32 s51, s16, s55 // (op 9175) 

s_mov_b32 s55, s51 // (op 9175) 
s_mul_hi_i32 s52, s55, s14 // (op 9175) 
s_add_i32 s53, s52, s55 // (op 9175) 


s_ashr_i32 s52, s53, 31 // (op 9175) 
s_cmp_eq_i32 s14, 0 // (op 9175) 
s_mov_b32 s55, scc // (op 9175) 
s_cmp_eq_i32 s55, 0 // (op 9175) 
s_cselect_b32 s57, 0, -1 // (op 9175) 

s_and_b32 s55, s48, 31 // (op 9175) 
s_lshl_b32 s58, 1, s55 // (op 9175) 
s_add_i32 s59, s58, s57 // (op 9175) 


s_and_b32 s57, s52, s59 // (op 9175) 


s_add_i32 s52, s53, s57 // (op 9175) 


s_ashr_i32 s53, s52, s55 // (op 9175) 


s_lshl_b32 s52, s48, 24 // (op 9175) 
s_ashr_i32 s55, s52, 31 // (op 9175) 

s_xor_b32 s52, s53, s55 // (op 9175) 

s_sub_i32 s53, s52, s55 // (op 9175) 


s_mul_i32 s52, s47, s15 // (op 9175) 
s_add_u32 s58, s20, 128 // (op 9175) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 9175) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 9175) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 9175) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 9175) 

s_sub_i32 s55, s58, s52 // (op 9175) 

s_mul_i32 s57, s53, s55 // (op 9175) 

s_sub_u32 s55, s51, s57 // (op 9175) 


s_add_u32 s51, s55, s52 // (op 9175) 


s_cmp_eq_i32 s56, 0 // (op 9175) 
s_mov_b32 s52, scc // (op 9175) 

s_cmp_eq_i32 s52, 0 // (op 9175) 
s_cselect_b32 s55, s51, s1 // (op 9175) 


s_mul_i32 s1, s54, s18 // (op 9175) 


v_lshl_add_u32 v11, s55, 7, v14 // (op 9175) 


s_sub_i32 s51, s0, s1 // (op 9175) 


s_cmp_eq_i32 s52, 0 // (op 9175) 
s_cselect_b32 s0, s53, s51 // (op 9175) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 9175) 


v_mul_lo_u32 v10, v13, s4 // (op 9175) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 9175) 


s_mul_i32 s1, s2, s7 // (op 8920) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8920) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8920) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8920) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8920) most significant: sum
s_add_u32 s1, s1, s19 // (op 8920) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8920) 

s_mul_i32 s1, s4, s7 // (op 9177) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 9177) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9177) least significant
s_mul_hi_u32 s52, s4, s6 // (op 9177) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 9177) most significant: sum
s_add_u32 s1, s1, s52 // (op 9177) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 9177) 

s_mov_b32 s56, s22 // (op 8920) reg expression
s_mov_b32 s57, s23 // (op 8920) reg expression
s_mov_b32 s58, s18 // (op 8920) reg expression
s_mov_b32 s59, 131072 // (op 8920) call()

s_mov_b32 s60, s24 // (op 9177) reg expression

s_mov_b32 s61, s25 // (op 9177) reg expression
s_mov_b32 s62, s52 // (op 9177) reg expression
s_mov_b32 s63, 131072 // (op 9177) call()


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
v_accvgpr_write a32, 0.00000 // (op 1722) call()
v_accvgpr_write a33, 0.00000 // (op 1722) call()
v_accvgpr_write a34, 0.00000 // (op 1722) call()
v_accvgpr_write a35, 0.00000 // (op 1722) call()
v_accvgpr_write a36, 0.00000 // (op 2031) call()
v_accvgpr_write a37, 0.00000 // (op 2031) call()
v_accvgpr_write a38, 0.00000 // (op 2031) call()
v_accvgpr_write a39, 0.00000 // (op 2031) call()
v_accvgpr_write a40, 0.00000 // (op 2032) call()
v_accvgpr_write a41, 0.00000 // (op 2032) call()
v_accvgpr_write a42, 0.00000 // (op 2032) call()
v_accvgpr_write a43, 0.00000 // (op 2032) call()
v_accvgpr_write a44, 0.00000 // (op 2033) call()
v_accvgpr_write a45, 0.00000 // (op 2033) call()
v_accvgpr_write a46, 0.00000 // (op 2033) call()
v_accvgpr_write a47, 0.00000 // (op 2033) call()
v_accvgpr_write a48, 0.00000 // (op 2492) call()
v_accvgpr_write a49, 0.00000 // (op 2492) call()
v_accvgpr_write a50, 0.00000 // (op 2492) call()
v_accvgpr_write a51, 0.00000 // (op 2492) call()
v_accvgpr_write a52, 0.00000 // (op 2801) call()
v_accvgpr_write a53, 0.00000 // (op 2801) call()
v_accvgpr_write a54, 0.00000 // (op 2801) call()
v_accvgpr_write a55, 0.00000 // (op 2801) call()
v_accvgpr_write a56, 0.00000 // (op 2802) call()
v_accvgpr_write a57, 0.00000 // (op 2802) call()
v_accvgpr_write a58, 0.00000 // (op 2802) call()
v_accvgpr_write a59, 0.00000 // (op 2802) call()
v_accvgpr_write a60, 0.00000 // (op 2803) call()
v_accvgpr_write a61, 0.00000 // (op 2803) call()
v_accvgpr_write a62, 0.00000 // (op 2803) call()
v_accvgpr_write a63, 0.00000 // (op 2803) call()
s_mul_hi_i32 s0, s16, s10 // (op 8680) 
v_and_b32 v14, -2, v1 // (op 8680) 
s_add_i32 s1, s0, s16 // (op 8680) 

v_sub_u32 v15, v1, v14 // (op 8680) 

s_ashr_i32 s0, s1, 31 // (op 8680) 
v_lshlrev_b32 v14, 6, v15 // (op 8680) 

s_cmp_eq_i32 s10, 0 // (op 8680) 
s_mov_b32 s18, scc // (op 8680) 
v_lshrrev_b32 v15, 1, v1 // (op 8680) 
s_cmp_eq_i32 s18, 0 // (op 8680) 
s_cselect_b32 s19, 0, -1 // (op 8680) 

s_and_b32 s18, s11, 31 // (op 8680) 
s_lshl_b32 s22, 1, s18 // (op 8680) 
s_add_i32 s23, s22, s19 // (op 8680) 


s_and_b32 s19, s0, s23 // (op 8680) 


s_add_i32 s0, s1, s19 // (op 8680) 


s_ashr_i32 s1, s0, s18 // (op 8680) 


s_lshl_b32 s0, s11, 24 // (op 8680) 
s_ashr_i32 s18, s0, 31 // (op 8680) 

s_xor_b32 s0, s1, s18 // (op 8680) 

s_sub_i32 s1, s0, s18 // (op 8680) 


s_mul_hi_i32 s0, s1, s12 // (op 8680) 
s_add_i32 s18, s0, s1 // (op 8680) 

s_ashr_i32 s0, s18, 31 // (op 8680) 
s_cmp_eq_i32 s12, 0 // (op 8680) 
s_mov_b32 s19, scc // (op 8680) 
s_cmp_eq_i32 s19, 0 // (op 8680) 
s_cselect_b32 s22, 0, -1 // (op 8680) 

s_and_b32 s19, s13, 31 // (op 8680) 
s_lshl_b32 s23, 1, s19 // (op 8680) 
s_add_i32 s24, s23, s22 // (op 8680) 


s_and_b32 s25, s0, s24 // (op 8680) 


s_add_i32 s0, s18, s25 // (op 8680) 


s_ashr_i32 s18, s0, s19 // (op 8680) 


s_lshl_b32 s0, s13, 24 // (op 8680) 
s_ashr_i32 s19, s0, 31 // (op 8680) 

s_xor_b32 s0, s18, s19 // (op 8680) 

s_sub_i32 s18, s0, s19 // (op 8680) 


s_mul_i32 s0, s1, s15 // (op 8680) 

s_sub_u32 s1, s16, s0 // (op 8680) 

s_mul_i32 s0, s18, s15 // (op 8680) 
s_add_u32 s0, s0, s1 // (op 8680) 


s_mul_hi_i32 s1, s16, s9 // (op 8680) 
s_add_i32 s18, s1, s16 // (op 8680) 

s_ashr_i32 s1, s18, 31 // (op 8680) 
s_cmp_eq_i32 s9, 0 // (op 8680) 
s_mov_b32 s19, scc // (op 8680) 
s_cmp_eq_i32 s19, 0 // (op 8680) 
s_cselect_b32 s22, 0, -1 // (op 8680) 

s_and_b32 s19, s46, 31 // (op 8680) 
s_lshl_b32 s23, 1, s19 // (op 8680) 
s_add_i32 s24, s23, s22 // (op 8680) 


s_and_b32 s25, s1, s24 // (op 8680) 


s_add_i32 s1, s18, s25 // (op 8680) 


s_ashr_i32 s18, s1, s19 // (op 8680) 


s_lshl_b32 s1, s46, 24 // (op 8680) 
s_ashr_i32 s19, s1, 31 // (op 8680) 

s_xor_b32 s1, s18, s19 // (op 8680) 

s_sub_i32 s18, s1, s19 // (op 8680) 


s_add_u32 s22, s6, 128 // (op 8680) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8680) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8680) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8680) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 8680) 

s_mul_i32 s1, s15, s22 // (op 8680) 

s_mul_i32 s19, s8, s1 // (op 8680) 

s_mul_i32 s1, s18, s19 // (op 8680) 

s_sub_u32 s19, s16, s1 // (op 8680) 

s_mov_b32 s1, s19 // (op 8680) 
s_mul_hi_i32 s22, s1, s14 // (op 8680) 
s_add_i32 s23, s22, s1 // (op 8680) 


s_ashr_i32 s1, s23, 31 // (op 8680) 
s_cmp_eq_i32 s14, 0 // (op 8680) 
s_mov_b32 s22, scc // (op 8680) 
s_cmp_eq_i32 s22, 0 // (op 8680) 
s_cselect_b32 s24, 0, -1 // (op 8680) 

s_and_b32 s22, s48, 31 // (op 8680) 
s_lshl_b32 s25, 1, s22 // (op 8680) 
s_add_i32 s49, s25, s24 // (op 8680) 


s_and_b32 s24, s1, s49 // (op 8680) 


s_add_i32 s1, s23, s24 // (op 8680) 


s_ashr_i32 s23, s1, s22 // (op 8680) 


s_lshl_b32 s1, s48, 24 // (op 8680) 
s_ashr_i32 s22, s1, 31 // (op 8680) 

s_xor_b32 s1, s23, s22 // (op 8680) 

s_sub_i32 s23, s1, s22 // (op 8680) 


s_mul_i32 s1, s47, s15 // (op 8680) 
s_add_u32 s24, s20, 128 // (op 8680) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8680) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 8680) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 8680) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 7 // (op 8680) 

s_sub_i32 s22, s24, s1 // (op 8680) 

s_mul_i32 s24, s23, s22 // (op 8680) 


s_sub_u32 s25, s19, s24 // (op 8680) 


s_add_u32 s19, s25, s1 // (op 8680) 


s_cmp_eq_i32 s18, 0 // (op 8680) 
s_mov_b32 s1, scc // (op 8680) 

s_cmp_eq_i32 s1, 0 // (op 8680) 
s_cselect_b32 s18, s19, s0 // (op 8680) 



v_lshl_add_u32 v16, s18, 7, v14 // (op 8680) 


v_mov_b32 v19, 0 // (op 8680) convert
v_mov_b32 v18, v16 // (op 8680) convert

v_mul_lo_u32 v17, v15, s27 // (op 8680) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 8680) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 8680) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 8680) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 8680) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 8680) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 8680) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 8680) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 8680) 

v_mov_b32 v16, v14 // (op 8680) convert

s_mul_i32 s1, s31, 6 // (op 8682) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 8682) least significant
s_mul_hi_u32 s18, s30, 6 // (op 8682) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8682) most significant: sum

s_mul_i32 s19, s26, s1 // (op 8682) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 8682) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 8682) least significant
s_mul_hi_u32 s23, s26, s0 // (op 8682) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8682) most significant: sum
s_add_u32 s19, s19, s23 // (op 8682) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8682) 

s_mov_b32 s52, s28 // (op 8682) reg expression
s_mov_b32 s53, s29 // (op 8682) reg expression
s_mov_b32 s54, s0 // (op 8682) reg expression
s_mov_b32 s55, 131072 // (op 8682) call()

s_mul_hi_i32 s0, s16, s10 // (op 8699) 
v_and_b32 v14, -2, v1 // (op 8699) 
s_add_i32 s1, s0, s16 // (op 8699) 

v_sub_u32 v15, v1, v14 // (op 8699) 

s_ashr_i32 s0, s1, 31 // (op 8699) 
v_lshlrev_b32 v14, 6, v15 // (op 8699) 

s_cmp_eq_i32 s10, 0 // (op 8699) 
s_mov_b32 s18, scc // (op 8699) 
v_lshrrev_b32 v15, 1, v1 // (op 8699) 
s_cmp_eq_i32 s18, 0 // (op 8699) 
s_cselect_b32 s19, 0, -1 // (op 8699) 

s_and_b32 s18, s11, 31 // (op 8699) 
s_lshl_b32 s22, 1, s18 // (op 8699) 
s_add_i32 s23, s22, s19 // (op 8699) 


s_and_b32 s19, s0, s23 // (op 8699) 


s_add_i32 s0, s1, s19 // (op 8699) 


s_ashr_i32 s1, s0, s18 // (op 8699) 


s_lshl_b32 s0, s11, 24 // (op 8699) 
s_ashr_i32 s18, s0, 31 // (op 8699) 

s_xor_b32 s0, s1, s18 // (op 8699) 

s_sub_i32 s1, s0, s18 // (op 8699) 


s_mul_hi_i32 s0, s1, s12 // (op 8699) 
s_add_i32 s18, s0, s1 // (op 8699) 

s_ashr_i32 s0, s18, 31 // (op 8699) 
s_cmp_eq_i32 s12, 0 // (op 8699) 
s_mov_b32 s19, scc // (op 8699) 
s_cmp_eq_i32 s19, 0 // (op 8699) 
s_cselect_b32 s22, 0, -1 // (op 8699) 

s_and_b32 s19, s13, 31 // (op 8699) 
s_lshl_b32 s23, 1, s19 // (op 8699) 
s_add_i32 s24, s23, s22 // (op 8699) 


s_and_b32 s25, s0, s24 // (op 8699) 


s_add_i32 s0, s18, s25 // (op 8699) 


s_ashr_i32 s18, s0, s19 // (op 8699) 


s_lshl_b32 s0, s13, 24 // (op 8699) 
s_ashr_i32 s19, s0, 31 // (op 8699) 

s_xor_b32 s0, s18, s19 // (op 8699) 

s_sub_i32 s18, s0, s19 // (op 8699) 


s_mul_i32 s0, s1, s15 // (op 8699) 

s_sub_u32 s1, s16, s0 // (op 8699) 

s_mul_i32 s0, s18, s15 // (op 8699) 
s_add_u32 s0, s0, s1 // (op 8699) 


s_mul_hi_i32 s1, s16, s9 // (op 8699) 
s_add_i32 s18, s1, s16 // (op 8699) 

s_ashr_i32 s1, s18, 31 // (op 8699) 
s_cmp_eq_i32 s9, 0 // (op 8699) 
s_mov_b32 s19, scc // (op 8699) 
s_cmp_eq_i32 s19, 0 // (op 8699) 
s_cselect_b32 s22, 0, -1 // (op 8699) 

s_and_b32 s19, s46, 31 // (op 8699) 
s_lshl_b32 s23, 1, s19 // (op 8699) 
s_add_i32 s24, s23, s22 // (op 8699) 


s_and_b32 s25, s1, s24 // (op 8699) 


s_add_i32 s1, s18, s25 // (op 8699) 


s_ashr_i32 s18, s1, s19 // (op 8699) 


s_lshl_b32 s1, s46, 24 // (op 8699) 
s_ashr_i32 s19, s1, 31 // (op 8699) 

s_xor_b32 s1, s18, s19 // (op 8699) 

s_sub_i32 s18, s1, s19 // (op 8699) 


s_add_u32 s50, s6, 128 // (op 8699) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8699) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8699) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8699) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8699) 

s_mul_i32 s1, s15, s24 // (op 8699) 

s_mul_i32 s19, s8, s1 // (op 8699) 

s_mul_i32 s1, s18, s19 // (op 8699) 

s_sub_u32 s19, s16, s1 // (op 8699) 

s_mov_b32 s1, s19 // (op 8699) 
s_mul_hi_i32 s22, s1, s14 // (op 8699) 
s_add_i32 s23, s22, s1 // (op 8699) 


s_ashr_i32 s1, s23, 31 // (op 8699) 
s_cmp_eq_i32 s14, 0 // (op 8699) 
s_mov_b32 s22, scc // (op 8699) 
s_cmp_eq_i32 s22, 0 // (op 8699) 
s_cselect_b32 s24, 0, -1 // (op 8699) 

s_and_b32 s22, s48, 31 // (op 8699) 
s_lshl_b32 s25, 1, s22 // (op 8699) 
s_add_i32 s49, s25, s24 // (op 8699) 


s_and_b32 s24, s1, s49 // (op 8699) 


s_add_i32 s1, s23, s24 // (op 8699) 


s_ashr_i32 s23, s1, s22 // (op 8699) 


s_lshl_b32 s1, s48, 24 // (op 8699) 
s_ashr_i32 s22, s1, 31 // (op 8699) 

s_xor_b32 s1, s23, s22 // (op 8699) 

s_sub_i32 s23, s1, s22 // (op 8699) 


s_mul_i32 s1, s47, s15 // (op 8699) 
s_add_u32 s24, s20, 128 // (op 8699) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8699) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8699) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8699) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 8699) 

s_sub_i32 s22, s24, s1 // (op 8699) 

s_mul_i32 s24, s23, s22 // (op 8699) 


s_sub_u32 s25, s19, s24 // (op 8699) 


s_add_u32 s19, s25, s1 // (op 8699) 


s_cmp_eq_i32 s18, 0 // (op 8699) 
s_mov_b32 s1, scc // (op 8699) 

s_cmp_eq_i32 s1, 0 // (op 8699) 
s_cselect_b32 s18, s19, s0 // (op 8699) 



v_lshl_add_u32 v17, s18, 7, v14 // (op 8699) 


v_mov_b32 v19, 0 // (op 8699) convert
v_mov_b32 v18, v17 // (op 8699) convert

v_mul_lo_u32 v21, v15, s27 // (op 8699) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 8699) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8699) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 8699) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 8699) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 8699) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 8699) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 8699) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 8699) 

v_mov_b32 v17, v14 // (op 8699) convert

s_mul_i32 s1, s31, 6 // (op 8701) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 8701) least significant
s_mul_hi_u32 s18, s30, 6 // (op 8701) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8701) most significant: sum

s_mul_i32 s19, s26, s1 // (op 8701) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 8701) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 8701) least significant
s_mul_hi_u32 s23, s26, s0 // (op 8701) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8701) most significant: sum
s_add_u32 s19, s19, s23 // (op 8701) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8701) 

s_mov_b32 s52, s28 // (op 8701) reg expression
s_mov_b32 s53, s29 // (op 8701) reg expression
s_mov_b32 s54, s0 // (op 8701) reg expression
s_mov_b32 s55, 131072 // (op 8701) call()

s_mul_hi_i32 s0, s16, s10 // (op 8744) 
v_lshrrev_b32 v14, 1, v1 // (op 8744) 
s_add_i32 s1, s0, s16 // (op 8744) 

v_and_b32 v15, -2, v1 // (op 8744) 
s_ashr_i32 s0, s1, 31 // (op 8744) 
v_sub_u32 v18, v1, v15 // (op 8744) 

s_cmp_eq_i32 s10, 0 // (op 8744) 
s_mov_b32 s18, scc // (op 8744) 
v_lshlrev_b32 v15, 6, v18 // (op 8744) 

s_cmp_eq_i32 s18, 0 // (op 8744) 
s_cselect_b32 s19, 0, -1 // (op 8744) 

v_mov_b32 v19, 0 // (op 8744) convert
v_mov_b32 v18, v15 // (op 8744) convert

s_and_b32 s18, s11, 31 // (op 8744) 
s_lshl_b32 s22, 1, s18 // (op 8744) 
s_add_i32 s23, s22, s19 // (op 8744) 


s_and_b32 s19, s0, s23 // (op 8744) 


s_add_i32 s0, s1, s19 // (op 8744) 


s_ashr_i32 s1, s0, s18 // (op 8744) 


s_lshl_b32 s0, s11, 24 // (op 8744) 
s_ashr_i32 s18, s0, 31 // (op 8744) 

s_xor_b32 s0, s1, s18 // (op 8744) 

s_sub_i32 s1, s0, s18 // (op 8744) 


s_mul_hi_i32 s0, s1, s12 // (op 8744) 
s_add_i32 s18, s0, s1 // (op 8744) 

s_ashr_i32 s0, s18, 31 // (op 8744) 
s_cmp_eq_i32 s12, 0 // (op 8744) 
s_mov_b32 s19, scc // (op 8744) 
s_cmp_eq_i32 s19, 0 // (op 8744) 
s_cselect_b32 s22, 0, -1 // (op 8744) 

s_and_b32 s19, s13, 31 // (op 8744) 
s_lshl_b32 s23, 1, s19 // (op 8744) 
s_add_i32 s24, s23, s22 // (op 8744) 


s_and_b32 s25, s0, s24 // (op 8744) 


s_add_i32 s0, s18, s25 // (op 8744) 


s_ashr_i32 s18, s0, s19 // (op 8744) 


s_lshl_b32 s0, s13, 24 // (op 8744) 
s_ashr_i32 s19, s0, 31 // (op 8744) 

s_xor_b32 s0, s18, s19 // (op 8744) 

s_sub_i32 s18, s0, s19 // (op 8744) 


s_add_u32 s50, s6, 128 // (op 8744) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8744) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8744) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8744) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8744) 

s_mul_i32 s0, s18, s24 // (op 8744) 

s_sub_i32 s18, s1, s0 // (op 8744) 


s_mul_hi_i32 s19, s16, s9 // (op 8744) 
s_add_i32 s0, s19, s16 // (op 8744) 

s_ashr_i32 s1, s0, 31 // (op 8744) 
s_cmp_eq_i32 s9, 0 // (op 8744) 
s_mov_b32 s19, scc // (op 8744) 
s_cmp_eq_i32 s19, 0 // (op 8744) 
s_cselect_b32 s22, 0, -1 // (op 8744) 

s_and_b32 s19, s46, 31 // (op 8744) 
s_lshl_b32 s23, 1, s19 // (op 8744) 
s_add_i32 s49, s23, s22 // (op 8744) 


s_and_b32 s22, s1, s49 // (op 8744) 


s_add_i32 s1, s0, s22 // (op 8744) 


s_ashr_i32 s0, s1, s19 // (op 8744) 


s_lshl_b32 s1, s46, 24 // (op 8744) 
s_ashr_i32 s19, s1, 31 // (op 8744) 

s_xor_b32 s1, s0, s19 // (op 8744) 

s_sub_i32 s0, s1, s19 // (op 8744) 


s_mul_i32 s1, s15, s24 // (op 8744) 

s_mul_i32 s19, s8, s1 // (op 8744) 

s_mul_i32 s1, s0, s19 // (op 8744) 

s_sub_u32 s19, s16, s1 // (op 8744) 

s_mul_hi_i32 s1, s19, s14 // (op 8744) 
s_add_i32 s22, s1, s19 // (op 8744) 


s_ashr_i32 s1, s22, 31 // (op 8744) 
s_cmp_eq_i32 s14, 0 // (op 8744) 
s_mov_b32 s19, scc // (op 8744) 
s_cmp_eq_i32 s19, 0 // (op 8744) 
s_cselect_b32 s23, 0, -1 // (op 8744) 

s_and_b32 s19, s48, 31 // (op 8744) 
s_lshl_b32 s24, 1, s19 // (op 8744) 
s_add_i32 s25, s24, s23 // (op 8744) 


s_and_b32 s23, s1, s25 // (op 8744) 


s_add_i32 s1, s22, s23 // (op 8744) 


s_ashr_i32 s22, s1, s19 // (op 8744) 


s_lshl_b32 s1, s48, 24 // (op 8744) 
s_ashr_i32 s19, s1, 31 // (op 8744) 

s_xor_b32 s1, s22, s19 // (op 8744) 

s_sub_i32 s22, s1, s19 // (op 8744) 


s_cmp_eq_i32 s0, 0 // (op 8744) 
s_mov_b32 s1, scc // (op 8744) 

s_cmp_eq_i32 s1, 0 // (op 8744) 
s_cselect_b32 s0, s22, s18 // (op 8744) 



v_lshl_add_u32 v15, s0, 7, v14 // (op 8744) 


v_mul_lo_u32 v21, v15, s33 // (op 8744) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 8744) most significant: high of low * low
v_mul_lo_u32 v20, v15, s32 // (op 8744) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8744) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 8744) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 8744) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 8744) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 8744) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 8744) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 8744) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 8744) 

v_mov_b32 v18, v14 // (op 8744) convert

s_mul_i32 s1, s7, 6 // (op 8746) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 8746) least significant
s_mul_hi_u32 s18, s6, 6 // (op 8746) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8746) most significant: sum

s_mul_i32 s19, s32, s1 // (op 8746) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 8746) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 8746) least significant
s_mul_hi_u32 s23, s32, s0 // (op 8746) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8746) most significant: sum
s_add_u32 s19, s19, s23 // (op 8746) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8746) 

s_mov_b32 s64, s34 // (op 8746) reg expression
s_mov_b32 s65, s35 // (op 8746) reg expression
s_mov_b32 s66, s0 // (op 8746) reg expression
s_mov_b32 s67, 131072 // (op 8746) call()

s_mul_hi_i32 s0, s16, s10 // (op 8763) 
v_lshrrev_b32 v14, 1, v1 // (op 8763) 
s_add_i32 s1, s0, s16 // (op 8763) 

v_and_b32 v15, -2, v1 // (op 8763) 
s_ashr_i32 s0, s1, 31 // (op 8763) 
v_sub_u32 v19, v1, v15 // (op 8763) 

s_cmp_eq_i32 s10, 0 // (op 8763) 
s_mov_b32 s18, scc // (op 8763) 
v_lshlrev_b32 v15, 6, v19 // (op 8763) 

s_cmp_eq_i32 s18, 0 // (op 8763) 
s_cselect_b32 s19, 0, -1 // (op 8763) 

v_mov_b32 v21, 0 // (op 8763) convert
v_mov_b32 v20, v15 // (op 8763) convert

s_and_b32 s18, s11, 31 // (op 8763) 
s_lshl_b32 s22, 1, s18 // (op 8763) 
s_add_i32 s23, s22, s19 // (op 8763) 


s_and_b32 s19, s0, s23 // (op 8763) 


s_add_i32 s0, s1, s19 // (op 8763) 


s_ashr_i32 s1, s0, s18 // (op 8763) 


s_lshl_b32 s0, s11, 24 // (op 8763) 
s_ashr_i32 s18, s0, 31 // (op 8763) 

s_xor_b32 s0, s1, s18 // (op 8763) 

s_sub_i32 s1, s0, s18 // (op 8763) 


s_mul_hi_i32 s0, s1, s12 // (op 8763) 
s_add_i32 s18, s0, s1 // (op 8763) 

s_ashr_i32 s0, s18, 31 // (op 8763) 
s_cmp_eq_i32 s12, 0 // (op 8763) 
s_mov_b32 s19, scc // (op 8763) 
s_cmp_eq_i32 s19, 0 // (op 8763) 
s_cselect_b32 s22, 0, -1 // (op 8763) 

s_and_b32 s19, s13, 31 // (op 8763) 
s_lshl_b32 s23, 1, s19 // (op 8763) 
s_add_i32 s24, s23, s22 // (op 8763) 


s_and_b32 s25, s0, s24 // (op 8763) 


s_add_i32 s0, s18, s25 // (op 8763) 


s_ashr_i32 s18, s0, s19 // (op 8763) 


s_lshl_b32 s0, s13, 24 // (op 8763) 
s_ashr_i32 s19, s0, 31 // (op 8763) 

s_xor_b32 s0, s18, s19 // (op 8763) 

s_sub_i32 s18, s0, s19 // (op 8763) 


s_add_u32 s50, s6, 128 // (op 8763) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8763) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8763) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8763) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8763) 

s_mul_i32 s0, s18, s24 // (op 8763) 

s_sub_i32 s18, s1, s0 // (op 8763) 


s_mul_hi_i32 s19, s16, s9 // (op 8763) 
s_add_i32 s0, s19, s16 // (op 8763) 

s_ashr_i32 s1, s0, 31 // (op 8763) 
s_cmp_eq_i32 s9, 0 // (op 8763) 
s_mov_b32 s19, scc // (op 8763) 
s_cmp_eq_i32 s19, 0 // (op 8763) 
s_cselect_b32 s22, 0, -1 // (op 8763) 

s_and_b32 s19, s46, 31 // (op 8763) 
s_lshl_b32 s23, 1, s19 // (op 8763) 
s_add_i32 s49, s23, s22 // (op 8763) 


s_and_b32 s22, s1, s49 // (op 8763) 


s_add_i32 s1, s0, s22 // (op 8763) 


s_ashr_i32 s0, s1, s19 // (op 8763) 


s_lshl_b32 s1, s46, 24 // (op 8763) 
s_ashr_i32 s19, s1, 31 // (op 8763) 

s_xor_b32 s1, s0, s19 // (op 8763) 

s_sub_i32 s0, s1, s19 // (op 8763) 


s_mul_i32 s1, s15, s24 // (op 8763) 

s_mul_i32 s19, s8, s1 // (op 8763) 

s_mul_i32 s1, s0, s19 // (op 8763) 

s_sub_u32 s19, s16, s1 // (op 8763) 

s_mul_hi_i32 s1, s19, s14 // (op 8763) 
s_add_i32 s22, s1, s19 // (op 8763) 


s_ashr_i32 s1, s22, 31 // (op 8763) 
s_cmp_eq_i32 s14, 0 // (op 8763) 
s_mov_b32 s19, scc // (op 8763) 
s_cmp_eq_i32 s19, 0 // (op 8763) 
s_cselect_b32 s23, 0, -1 // (op 8763) 

s_and_b32 s19, s48, 31 // (op 8763) 
s_lshl_b32 s24, 1, s19 // (op 8763) 
s_add_i32 s25, s24, s23 // (op 8763) 


s_and_b32 s23, s1, s25 // (op 8763) 


s_add_i32 s1, s22, s23 // (op 8763) 


s_ashr_i32 s22, s1, s19 // (op 8763) 


s_lshl_b32 s1, s48, 24 // (op 8763) 
s_ashr_i32 s19, s1, 31 // (op 8763) 

s_xor_b32 s1, s22, s19 // (op 8763) 

s_sub_i32 s22, s1, s19 // (op 8763) 


s_cmp_eq_i32 s0, 0 // (op 8763) 
s_mov_b32 s1, scc // (op 8763) 

s_cmp_eq_i32 s1, 0 // (op 8763) 
s_cselect_b32 s0, s22, s18 // (op 8763) 



v_lshl_add_u32 v15, s0, 7, v14 // (op 8763) 


v_mul_lo_u32 v23, v15, s33 // (op 8763) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 8763) most significant: high of low * low
v_mul_lo_u32 v22, v15, s32 // (op 8763) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8763) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 8763) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 8763) most significant half



v_mul_lo_u32 v21, v15, 6 // (op 8763) most significant: low of high * low
v_mul_hi_u32 v19, v14, 6 // (op 8763) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 8763) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 8763) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 8763) 

v_mov_b32 v19, v14 // (op 8763) convert

s_mul_i32 s1, s7, 6 // (op 8765) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 8765) least significant
s_mul_hi_u32 s18, s6, 6 // (op 8765) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8765) most significant: sum

s_mul_i32 s19, s32, s1 // (op 8765) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 8765) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 8765) least significant
s_mul_hi_u32 s23, s32, s0 // (op 8765) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8765) most significant: sum
s_add_u32 s19, s19, s23 // (op 8765) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8765) 

s_mov_b32 s64, s34 // (op 8765) reg expression
s_mov_b32 s65, s35 // (op 8765) reg expression
s_mov_b32 s66, s0 // (op 8765) reg expression
s_mov_b32 s67, 131072 // (op 8765) call()

s_mul_hi_i32 s0, s16, s10 // (op 8808) 
v_and_b32 v14, -128, v1 // (op 8808) 
s_add_i32 s1, s0, s16 // (op 8808) 

v_sub_u32 v15, v1, v14 // (op 8808) 

s_ashr_i32 s0, s1, 31 // (op 8808) 
v_lshrrev_b32 v14, 7, v1 // (op 8808) 
s_cmp_eq_i32 s10, 0 // (op 8808) 
s_mov_b32 s18, scc // (op 8808) 
v_lshlrev_b32 v20, 1, v14 // (op 8808) 

s_cmp_eq_i32 s18, 0 // (op 8808) 
s_cselect_b32 s19, 0, -1 // (op 8808) 

s_and_b32 s18, s11, 31 // (op 8808) 
s_lshl_b32 s22, 1, s18 // (op 8808) 
s_add_i32 s23, s22, s19 // (op 8808) 


s_and_b32 s19, s0, s23 // (op 8808) 


s_add_i32 s0, s1, s19 // (op 8808) 


s_ashr_i32 s1, s0, s18 // (op 8808) 


s_lshl_b32 s0, s11, 24 // (op 8808) 
s_ashr_i32 s18, s0, 31 // (op 8808) 

s_xor_b32 s0, s1, s18 // (op 8808) 

s_sub_i32 s1, s0, s18 // (op 8808) 


s_mul_hi_i32 s0, s1, s12 // (op 8808) 
s_add_i32 s18, s0, s1 // (op 8808) 

s_ashr_i32 s0, s18, 31 // (op 8808) 
s_cmp_eq_i32 s12, 0 // (op 8808) 
s_mov_b32 s19, scc // (op 8808) 
s_cmp_eq_i32 s19, 0 // (op 8808) 
s_cselect_b32 s22, 0, -1 // (op 8808) 

s_and_b32 s19, s13, 31 // (op 8808) 
s_lshl_b32 s23, 1, s19 // (op 8808) 
s_add_i32 s24, s23, s22 // (op 8808) 


s_and_b32 s25, s0, s24 // (op 8808) 


s_add_i32 s0, s18, s25 // (op 8808) 


s_ashr_i32 s18, s0, s19 // (op 8808) 


s_lshl_b32 s0, s13, 24 // (op 8808) 
s_ashr_i32 s19, s0, 31 // (op 8808) 

s_xor_b32 s0, s18, s19 // (op 8808) 

s_sub_i32 s18, s0, s19 // (op 8808) 


s_mul_i32 s0, s1, s15 // (op 8808) 

s_sub_u32 s1, s16, s0 // (op 8808) 

s_mul_i32 s0, s18, s15 // (op 8808) 
s_add_u32 s0, s0, s1 // (op 8808) 


s_mul_hi_i32 s1, s16, s9 // (op 8808) 
s_add_i32 s18, s1, s16 // (op 8808) 

s_ashr_i32 s1, s18, 31 // (op 8808) 
s_cmp_eq_i32 s9, 0 // (op 8808) 
s_mov_b32 s19, scc // (op 8808) 
s_cmp_eq_i32 s19, 0 // (op 8808) 
s_cselect_b32 s22, 0, -1 // (op 8808) 

s_and_b32 s19, s46, 31 // (op 8808) 
s_lshl_b32 s23, 1, s19 // (op 8808) 
s_add_i32 s24, s23, s22 // (op 8808) 


s_and_b32 s25, s1, s24 // (op 8808) 


s_add_i32 s1, s18, s25 // (op 8808) 


s_ashr_i32 s18, s1, s19 // (op 8808) 


s_lshl_b32 s1, s46, 24 // (op 8808) 
s_ashr_i32 s19, s1, 31 // (op 8808) 

s_xor_b32 s1, s18, s19 // (op 8808) 

s_sub_i32 s18, s1, s19 // (op 8808) 


s_add_u32 s50, s6, 128 // (op 8808) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8808) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8808) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8808) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8808) 

s_mul_i32 s1, s15, s24 // (op 8808) 

s_mul_i32 s19, s8, s1 // (op 8808) 

s_mul_i32 s1, s18, s19 // (op 8808) 

s_sub_u32 s19, s16, s1 // (op 8808) 

s_mov_b32 s1, s19 // (op 8808) 
s_mul_hi_i32 s22, s1, s14 // (op 8808) 
s_add_i32 s23, s22, s1 // (op 8808) 


s_ashr_i32 s1, s23, 31 // (op 8808) 
s_cmp_eq_i32 s14, 0 // (op 8808) 
s_mov_b32 s22, scc // (op 8808) 
s_cmp_eq_i32 s22, 0 // (op 8808) 
s_cselect_b32 s24, 0, -1 // (op 8808) 

s_and_b32 s22, s48, 31 // (op 8808) 
s_lshl_b32 s25, 1, s22 // (op 8808) 
s_add_i32 s49, s25, s24 // (op 8808) 


s_and_b32 s24, s1, s49 // (op 8808) 


s_add_i32 s1, s23, s24 // (op 8808) 


s_ashr_i32 s23, s1, s22 // (op 8808) 


s_lshl_b32 s1, s48, 24 // (op 8808) 
s_ashr_i32 s22, s1, 31 // (op 8808) 

s_xor_b32 s1, s23, s22 // (op 8808) 

s_sub_i32 s23, s1, s22 // (op 8808) 


s_mul_i32 s1, s47, s15 // (op 8808) 
s_add_u32 s24, s20, 128 // (op 8808) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8808) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8808) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8808) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 8808) 

s_sub_i32 s22, s24, s1 // (op 8808) 

s_mul_i32 s24, s23, s22 // (op 8808) 


s_sub_u32 s25, s19, s24 // (op 8808) 


s_add_u32 s19, s25, s1 // (op 8808) 


s_cmp_eq_i32 s18, 0 // (op 8808) 
s_mov_b32 s1, scc // (op 8808) 

s_cmp_eq_i32 s1, 0 // (op 8808) 
s_cselect_b32 s18, s19, s0 // (op 8808) 



v_lshl_add_u32 v14, s18, 7, v15 // (op 8808) 


v_mul_lo_u32 v15, v20, s36 // (op 8808) 

v_add_u32 v20, v14, v15 // (op 8808) 


s_mul_i32 s1, s36, s41 // (op 8810) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 8810) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 8810) least significant
s_mul_hi_u32 s19, s36, s40 // (op 8810) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8810) most significant: sum
s_add_u32 s1, s1, s19 // (op 8810) most significant: sum


s_mov_b32 s68, s38 // (op 8810) reg expression
s_mov_b32 s69, s39 // (op 8810) reg expression
s_mov_b32 s70, s0 // (op 8810) reg expression
s_mov_b32 s71, 131072 // (op 8810) call()

s_mul_hi_i32 s0, s16, s10 // (op 8827) 
v_and_b32 v14, -128, v1 // (op 8827) 
s_add_i32 s1, s0, s16 // (op 8827) 

v_sub_u32 v15, v1, v14 // (op 8827) 

s_ashr_i32 s0, s1, 31 // (op 8827) 
v_lshrrev_b32 v14, 7, v1 // (op 8827) 
s_cmp_eq_i32 s10, 0 // (op 8827) 
s_mov_b32 s18, scc // (op 8827) 
v_lshlrev_b32 v21, 1, v14 // (op 8827) 

s_cmp_eq_i32 s18, 0 // (op 8827) 
s_cselect_b32 s19, 0, -1 // (op 8827) 

s_and_b32 s18, s11, 31 // (op 8827) 
s_lshl_b32 s22, 1, s18 // (op 8827) 
s_add_i32 s23, s22, s19 // (op 8827) 


s_and_b32 s19, s0, s23 // (op 8827) 


s_add_i32 s0, s1, s19 // (op 8827) 


s_ashr_i32 s1, s0, s18 // (op 8827) 


s_lshl_b32 s0, s11, 24 // (op 8827) 
s_ashr_i32 s18, s0, 31 // (op 8827) 

s_xor_b32 s0, s1, s18 // (op 8827) 

s_sub_i32 s1, s0, s18 // (op 8827) 


s_mul_hi_i32 s0, s1, s12 // (op 8827) 
s_add_i32 s18, s0, s1 // (op 8827) 

s_ashr_i32 s0, s18, 31 // (op 8827) 
s_cmp_eq_i32 s12, 0 // (op 8827) 
s_mov_b32 s19, scc // (op 8827) 
s_cmp_eq_i32 s19, 0 // (op 8827) 
s_cselect_b32 s22, 0, -1 // (op 8827) 

s_and_b32 s19, s13, 31 // (op 8827) 
s_lshl_b32 s23, 1, s19 // (op 8827) 
s_add_i32 s24, s23, s22 // (op 8827) 


s_and_b32 s25, s0, s24 // (op 8827) 


s_add_i32 s0, s18, s25 // (op 8827) 


s_ashr_i32 s18, s0, s19 // (op 8827) 


s_lshl_b32 s0, s13, 24 // (op 8827) 
s_ashr_i32 s19, s0, 31 // (op 8827) 

s_xor_b32 s0, s18, s19 // (op 8827) 

s_sub_i32 s18, s0, s19 // (op 8827) 


s_mul_i32 s0, s1, s15 // (op 8827) 

s_sub_u32 s1, s16, s0 // (op 8827) 

s_mul_i32 s0, s18, s15 // (op 8827) 
s_add_u32 s0, s0, s1 // (op 8827) 


s_mul_hi_i32 s1, s16, s9 // (op 8827) 
s_add_i32 s18, s1, s16 // (op 8827) 

s_ashr_i32 s1, s18, 31 // (op 8827) 
s_cmp_eq_i32 s9, 0 // (op 8827) 
s_mov_b32 s19, scc // (op 8827) 
s_cmp_eq_i32 s19, 0 // (op 8827) 
s_cselect_b32 s22, 0, -1 // (op 8827) 

s_and_b32 s19, s46, 31 // (op 8827) 
s_lshl_b32 s23, 1, s19 // (op 8827) 
s_add_i32 s24, s23, s22 // (op 8827) 


s_and_b32 s25, s1, s24 // (op 8827) 


s_add_i32 s1, s18, s25 // (op 8827) 


s_ashr_i32 s18, s1, s19 // (op 8827) 


s_lshl_b32 s1, s46, 24 // (op 8827) 
s_ashr_i32 s19, s1, 31 // (op 8827) 

s_xor_b32 s1, s18, s19 // (op 8827) 

s_sub_i32 s18, s1, s19 // (op 8827) 


s_add_u32 s50, s6, 128 // (op 8827) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8827) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8827) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8827) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8827) 

s_mul_i32 s1, s15, s24 // (op 8827) 

s_mul_i32 s19, s8, s1 // (op 8827) 

s_mul_i32 s1, s18, s19 // (op 8827) 

s_sub_u32 s19, s16, s1 // (op 8827) 

s_mov_b32 s1, s19 // (op 8827) 
s_mul_hi_i32 s22, s1, s14 // (op 8827) 
s_add_i32 s23, s22, s1 // (op 8827) 


s_ashr_i32 s1, s23, 31 // (op 8827) 
s_cmp_eq_i32 s14, 0 // (op 8827) 
s_mov_b32 s22, scc // (op 8827) 
s_cmp_eq_i32 s22, 0 // (op 8827) 
s_cselect_b32 s24, 0, -1 // (op 8827) 

s_and_b32 s22, s48, 31 // (op 8827) 
s_lshl_b32 s25, 1, s22 // (op 8827) 
s_add_i32 s49, s25, s24 // (op 8827) 


s_and_b32 s24, s1, s49 // (op 8827) 


s_add_i32 s1, s23, s24 // (op 8827) 


s_ashr_i32 s23, s1, s22 // (op 8827) 


s_lshl_b32 s1, s48, 24 // (op 8827) 
s_ashr_i32 s22, s1, 31 // (op 8827) 

s_xor_b32 s1, s23, s22 // (op 8827) 

s_sub_i32 s23, s1, s22 // (op 8827) 


s_mul_i32 s1, s47, s15 // (op 8827) 
s_add_u32 s24, s20, 128 // (op 8827) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8827) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8827) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8827) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 8827) 

s_sub_i32 s22, s24, s1 // (op 8827) 

s_mul_i32 s24, s23, s22 // (op 8827) 


s_sub_u32 s25, s19, s24 // (op 8827) 


s_add_u32 s19, s25, s1 // (op 8827) 


s_cmp_eq_i32 s18, 0 // (op 8827) 
s_mov_b32 s1, scc // (op 8827) 

s_cmp_eq_i32 s1, 0 // (op 8827) 
s_cselect_b32 s18, s19, s0 // (op 8827) 



v_lshl_add_u32 v14, s18, 7, v15 // (op 8827) 


v_mul_lo_u32 v15, v21, s36 // (op 8827) 

v_add_u32 v21, v14, v15 // (op 8827) 


s_mul_i32 s1, s36, s41 // (op 8829) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 8829) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 8829) least significant
s_mul_hi_u32 s19, s36, s40 // (op 8829) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8829) most significant: sum
s_add_u32 s1, s1, s19 // (op 8829) most significant: sum


s_mov_b32 s68, s38 // (op 8829) reg expression
s_mov_b32 s69, s39 // (op 8829) reg expression
s_mov_b32 s70, s0 // (op 8829) reg expression
s_mov_b32 s71, 131072 // (op 8829) call()

s_mul_hi_i32 s0, s16, s10 // (op 8872) 
v_lshrrev_b32 v14, 2, v1 // (op 8872) 
s_add_i32 s1, s0, s16 // (op 8872) 

v_lshlrev_b32 v15, 1, v14 // (op 8872) 

s_ashr_i32 s0, s1, 31 // (op 8872) 
v_and_b32 v14, -4, v1 // (op 8872) 
s_cmp_eq_i32 s10, 0 // (op 8872) 
s_mov_b32 s18, scc // (op 8872) 
v_sub_u32 v22, v1, v14 // (op 8872) 

s_cmp_eq_i32 s18, 0 // (op 8872) 
s_cselect_b32 s19, 0, -1 // (op 8872) 

s_and_b32 s18, s11, 31 // (op 8872) 
s_lshl_b32 s22, 1, s18 // (op 8872) 
s_add_i32 s23, s22, s19 // (op 8872) 


s_and_b32 s19, s0, s23 // (op 8872) 


s_add_i32 s0, s1, s19 // (op 8872) 


s_ashr_i32 s1, s0, s18 // (op 8872) 


s_lshl_b32 s0, s11, 24 // (op 8872) 
s_ashr_i32 s18, s0, 31 // (op 8872) 

s_xor_b32 s0, s1, s18 // (op 8872) 

s_sub_i32 s1, s0, s18 // (op 8872) 


s_mul_hi_i32 s0, s1, s12 // (op 8872) 
s_add_i32 s18, s0, s1 // (op 8872) 

s_ashr_i32 s0, s18, 31 // (op 8872) 
s_cmp_eq_i32 s12, 0 // (op 8872) 
s_mov_b32 s19, scc // (op 8872) 
s_cmp_eq_i32 s19, 0 // (op 8872) 
s_cselect_b32 s22, 0, -1 // (op 8872) 

s_and_b32 s19, s13, 31 // (op 8872) 
s_lshl_b32 s23, 1, s19 // (op 8872) 
s_add_i32 s24, s23, s22 // (op 8872) 


s_and_b32 s25, s0, s24 // (op 8872) 


s_add_i32 s0, s18, s25 // (op 8872) 


s_ashr_i32 s18, s0, s19 // (op 8872) 


s_lshl_b32 s0, s13, 24 // (op 8872) 
s_ashr_i32 s19, s0, 31 // (op 8872) 

s_xor_b32 s0, s18, s19 // (op 8872) 

s_sub_i32 s18, s0, s19 // (op 8872) 


s_add_u32 s50, s6, 128 // (op 8872) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8872) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8872) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8872) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8872) 

s_mul_i32 s0, s18, s24 // (op 8872) 

s_sub_i32 s18, s1, s0 // (op 8872) 


s_mul_hi_i32 s19, s16, s9 // (op 8872) 
s_add_i32 s0, s19, s16 // (op 8872) 

s_ashr_i32 s1, s0, 31 // (op 8872) 
s_cmp_eq_i32 s9, 0 // (op 8872) 
s_mov_b32 s19, scc // (op 8872) 
s_cmp_eq_i32 s19, 0 // (op 8872) 
s_cselect_b32 s22, 0, -1 // (op 8872) 

s_and_b32 s19, s46, 31 // (op 8872) 
s_lshl_b32 s23, 1, s19 // (op 8872) 
s_add_i32 s49, s23, s22 // (op 8872) 


s_and_b32 s22, s1, s49 // (op 8872) 


s_add_i32 s1, s0, s22 // (op 8872) 


s_ashr_i32 s0, s1, s19 // (op 8872) 


s_lshl_b32 s1, s46, 24 // (op 8872) 
s_ashr_i32 s19, s1, 31 // (op 8872) 

s_xor_b32 s1, s0, s19 // (op 8872) 

s_sub_i32 s0, s1, s19 // (op 8872) 


s_mul_i32 s1, s15, s24 // (op 8872) 

s_mul_i32 s19, s8, s1 // (op 8872) 

s_mul_i32 s1, s0, s19 // (op 8872) 

s_sub_u32 s19, s16, s1 // (op 8872) 

s_mul_hi_i32 s1, s19, s14 // (op 8872) 
s_add_i32 s22, s1, s19 // (op 8872) 


s_ashr_i32 s1, s22, 31 // (op 8872) 
s_cmp_eq_i32 s14, 0 // (op 8872) 
s_mov_b32 s19, scc // (op 8872) 
s_cmp_eq_i32 s19, 0 // (op 8872) 
s_cselect_b32 s23, 0, -1 // (op 8872) 

s_and_b32 s19, s48, 31 // (op 8872) 
s_lshl_b32 s24, 1, s19 // (op 8872) 
s_add_i32 s25, s24, s23 // (op 8872) 


s_and_b32 s23, s1, s25 // (op 8872) 


s_add_i32 s1, s22, s23 // (op 8872) 


s_ashr_i32 s22, s1, s19 // (op 8872) 


s_lshl_b32 s1, s48, 24 // (op 8872) 
s_ashr_i32 s19, s1, 31 // (op 8872) 

s_xor_b32 s1, s22, s19 // (op 8872) 

s_sub_i32 s22, s1, s19 // (op 8872) 


s_cmp_eq_i32 s0, 0 // (op 8872) 
s_mov_b32 s1, scc // (op 8872) 

s_cmp_eq_i32 s1, 0 // (op 8872) 
s_cselect_b32 s0, s22, s18 // (op 8872) 



v_lshl_add_u32 v14, s0, 7, v15 // (op 8872) 


v_mul_lo_u32 v15, v14, s42 // (op 8872) 

v_add_u32 v14, v22, v15 // (op 8872) 


s_mul_i32 s1, s42, s7 // (op 8874) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 8874) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 8874) least significant
s_mul_hi_u32 s19, s42, s6 // (op 8874) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8874) most significant: sum
s_add_u32 s1, s1, s19 // (op 8874) most significant: sum


s_mov_b32 s72, s44 // (op 8874) reg expression
s_mov_b32 s73, s45 // (op 8874) reg expression
s_mov_b32 s74, s0 // (op 8874) reg expression
s_mov_b32 s75, 131072 // (op 8874) call()

s_mul_hi_i32 s0, s16, s10 // (op 8891) 
v_lshrrev_b32 v15, 2, v1 // (op 8891) 
s_add_i32 s1, s0, s16 // (op 8891) 

v_lshlrev_b32 v22, 1, v15 // (op 8891) 

s_ashr_i32 s0, s1, 31 // (op 8891) 
v_and_b32 v15, -4, v1 // (op 8891) 
s_cmp_eq_i32 s10, 0 // (op 8891) 
s_mov_b32 s18, scc // (op 8891) 
v_sub_u32 v23, v1, v15 // (op 8891) 

s_cmp_eq_i32 s18, 0 // (op 8891) 
s_cselect_b32 s19, 0, -1 // (op 8891) 

s_and_b32 s18, s11, 31 // (op 8891) 
s_lshl_b32 s22, 1, s18 // (op 8891) 
s_add_i32 s23, s22, s19 // (op 8891) 


s_and_b32 s19, s0, s23 // (op 8891) 


s_add_i32 s0, s1, s19 // (op 8891) 


s_ashr_i32 s1, s0, s18 // (op 8891) 


s_lshl_b32 s0, s11, 24 // (op 8891) 
s_ashr_i32 s18, s0, 31 // (op 8891) 

s_xor_b32 s0, s1, s18 // (op 8891) 

s_sub_i32 s1, s0, s18 // (op 8891) 


s_mul_hi_i32 s0, s1, s12 // (op 8891) 
s_add_i32 s18, s0, s1 // (op 8891) 

s_ashr_i32 s0, s18, 31 // (op 8891) 
s_cmp_eq_i32 s12, 0 // (op 8891) 
s_mov_b32 s19, scc // (op 8891) 
s_cmp_eq_i32 s19, 0 // (op 8891) 
s_cselect_b32 s22, 0, -1 // (op 8891) 

s_and_b32 s19, s13, 31 // (op 8891) 
s_lshl_b32 s23, 1, s19 // (op 8891) 
s_add_i32 s24, s23, s22 // (op 8891) 


s_and_b32 s25, s0, s24 // (op 8891) 


s_add_i32 s0, s18, s25 // (op 8891) 


s_ashr_i32 s18, s0, s19 // (op 8891) 


s_lshl_b32 s0, s13, 24 // (op 8891) 
s_ashr_i32 s19, s0, 31 // (op 8891) 

s_xor_b32 s0, s18, s19 // (op 8891) 

s_sub_i32 s18, s0, s19 // (op 8891) 


s_add_u32 s50, s6, 128 // (op 8891) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8891) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8891) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8891) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8891) 

s_mul_i32 s0, s18, s24 // (op 8891) 

s_sub_i32 s18, s1, s0 // (op 8891) 


s_mul_hi_i32 s19, s16, s9 // (op 8891) 
s_add_i32 s0, s19, s16 // (op 8891) 

s_ashr_i32 s1, s0, 31 // (op 8891) 
s_cmp_eq_i32 s9, 0 // (op 8891) 
s_mov_b32 s19, scc // (op 8891) 
s_cmp_eq_i32 s19, 0 // (op 8891) 
s_cselect_b32 s22, 0, -1 // (op 8891) 

s_and_b32 s19, s46, 31 // (op 8891) 
s_lshl_b32 s23, 1, s19 // (op 8891) 
s_add_i32 s49, s23, s22 // (op 8891) 


s_and_b32 s22, s1, s49 // (op 8891) 


s_add_i32 s1, s0, s22 // (op 8891) 


s_ashr_i32 s0, s1, s19 // (op 8891) 


s_lshl_b32 s1, s46, 24 // (op 8891) 
s_ashr_i32 s19, s1, 31 // (op 8891) 

s_xor_b32 s1, s0, s19 // (op 8891) 

s_sub_i32 s0, s1, s19 // (op 8891) 


s_mul_i32 s1, s15, s24 // (op 8891) 

s_mul_i32 s19, s8, s1 // (op 8891) 

s_mul_i32 s1, s0, s19 // (op 8891) 

s_sub_u32 s19, s16, s1 // (op 8891) 

s_mul_hi_i32 s1, s19, s14 // (op 8891) 
s_add_i32 s22, s1, s19 // (op 8891) 


s_ashr_i32 s1, s22, 31 // (op 8891) 
s_cmp_eq_i32 s14, 0 // (op 8891) 
s_mov_b32 s19, scc // (op 8891) 
s_cmp_eq_i32 s19, 0 // (op 8891) 
s_cselect_b32 s23, 0, -1 // (op 8891) 

s_and_b32 s19, s48, 31 // (op 8891) 
s_lshl_b32 s24, 1, s19 // (op 8891) 
s_add_i32 s25, s24, s23 // (op 8891) 


s_and_b32 s23, s1, s25 // (op 8891) 


s_add_i32 s1, s22, s23 // (op 8891) 


s_ashr_i32 s22, s1, s19 // (op 8891) 


s_lshl_b32 s1, s48, 24 // (op 8891) 
s_ashr_i32 s19, s1, 31 // (op 8891) 

s_xor_b32 s1, s22, s19 // (op 8891) 

s_sub_i32 s22, s1, s19 // (op 8891) 


s_cmp_eq_i32 s0, 0 // (op 8891) 
s_mov_b32 s1, scc // (op 8891) 

s_cmp_eq_i32 s1, 0 // (op 8891) 
s_cselect_b32 s0, s22, s18 // (op 8891) 



v_lshl_add_u32 v15, s0, 7, v22 // (op 8891) 


v_mul_lo_u32 v22, v15, s42 // (op 8891) 

v_add_u32 v15, v23, v22 // (op 8891) 


s_mul_i32 s1, s42, s7 // (op 8893) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 8893) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 8893) least significant
s_mul_hi_u32 s19, s42, s6 // (op 8893) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8893) most significant: sum
s_add_u32 s1, s1, s19 // (op 8893) most significant: sum


s_mov_b32 s72, s44 // (op 8893) reg expression
s_mov_b32 s73, s45 // (op 8893) reg expression
s_mov_b32 s74, s0 // (op 8893) reg expression
s_mov_b32 s75, 131072 // (op 8893) call()




buffer_load_dwordx4 v[22:25], v18, s[64:67], 0 offen // (op 8268) Load value
buffer_load_dwordx2 v[26:27], v18, s[64:67], 0 offen offset:16 // (op 8268) Load value
buffer_load_dwordx4 v[28:31], v18, s[64:67], 0 offen offset:24 // (op 8268) Load value
buffer_load_dwordx2 v[32:33], v18, s[64:67], 0 offen offset:40 // (op 8268) Load value

v_mov_b64 v[34:35], s[42:43] // (op 8271) convert

v_mov_b32 v36, v14 // (op 8271) 
buffer_load_ubyte v38, v36, s[72:75], 0 offen offset: 0 // (op 8271) Load value
v_add_u32 v36, v36, v34 // (op 8271) 
buffer_load_ubyte v39, v36, s[72:75], 0 offen offset: 0 // (op 8271) Load value




v_mov_b32 v34, v16 // (op 8274) reg expression

buffer_load_dwordx3 v[40:42], v34, s[52:55], 0 offen // (op 8274) Load value
buffer_load_dwordx3 v[44:46], v34, s[52:55], 0 offen offset:12 // (op 8274) Load value
buffer_load_dwordx3 v[48:50], v34, s[52:55], 0 offen offset:24 // (op 8274) Load value
buffer_load_dwordx3 v[52:54], v34, s[52:55], 0 offen offset:36 // (op 8274) Load value


v_mov_b64 v[34:35], s[36:37] // (op 8277) convert

v_mov_b32 v43, v20 // (op 8277) reg expression
buffer_load_ubyte v36, v43, s[68:71], 0 offen offset: 0 // (op 8277) Load value
v_add_u32 v43, v43, v34 // (op 8277) 
buffer_load_ubyte v37, v43, s[68:71], 0 offen offset: 0 // (op 8277) Load value





buffer_load_dwordx4 v[56:59], v18, s[64:67], 0 offen offset:96 // (op 8280) Load value
buffer_load_dwordx2 v[60:61], v18, s[64:67], 0 offen offset:112 // (op 8280) Load value
buffer_load_dwordx4 v[62:65], v18, s[64:67], 0 offen offset:120 // (op 8280) Load value
buffer_load_dwordx2 v[66:67], v18, s[64:67], 0 offen offset:136 // (op 8280) Load value


v_mov_b64 v[34:35], s[42:43] // (op 8283) convert

v_mov_b32 v18, v14 // (op 8283) 
buffer_load_ubyte v68, v18, s[72:75], 0 offen offset:4 // (op 8283) Load value
v_add_u32 v18, v18, v34 // (op 8283) 
buffer_load_ubyte v69, v18, s[72:75], 0 offen offset:4 // (op 8283) Load value





s_lshl_b64 s[0:1], s[26:27], 7 // (op 8286) 
s_mul_i32 s19, s1, 6 // (op 8286) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 8286) least significant
s_mul_hi_u32 s22, s0, 6 // (op 8286) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8286) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 8286) 

v_add_u32 v14, v16, s0 // (op 8286) 


buffer_load_dwordx3 v[70:72], v14, s[52:55], 0 offen // (op 8286) Load value
buffer_load_dwordx3 v[74:76], v14, s[52:55], 0 offen offset:12 // (op 8286) Load value
buffer_load_dwordx3 v[78:80], v14, s[52:55], 0 offen offset:24 // (op 8286) Load value
buffer_load_dwordx3 v[82:84], v14, s[52:55], 0 offen offset:36 // (op 8286) Load value



v_mov_b64 v[34:35], s[36:37] // (op 8289) convert

s_lshl_b32 s0, s36, 2 // (op 8289) 
v_add_u32 v14, v20, s0 // (op 8289) 

buffer_load_ubyte v86, v14, s[68:71], 0 offen offset: 0 // (op 8289) Load value
v_add_u32 v14, v14, v34 // (op 8289) 
buffer_load_ubyte v87, v14, s[68:71], 0 offen offset: 0 // (op 8289) Load value



v_and_b32 v14, -2, v1 // (op 9021) 
v_sub_u32 v16, v1, v14 // (op 9021) 

v_lshlrev_b32 v14, 6, v16 // (op 9021) 

v_lshrrev_b32 v16, 1, v1 // (op 9021) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 9021) 


v_mul_lo_u32 v14, v18, 6 // (op 9021) 

v_lshrrev_b32 v16, 3, v14 // (op 9021) 




s_waitcnt vmcnt(23)
ds_write_b128 v16, v[22:25] // (op 8291) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v16, v[26:27] offset:16 // (op 8291) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v16, v[28:31] offset:24 // (op 8291) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v16, v[32:33] offset:40 // (op 8291) Store local data 

v_and_b32 v14, -4, v1 // (op 9137) 
v_sub_u32 v16, v1, v14 // (op 9137) 

v_lshrrev_b32 v14, 2, v1 // (op 9137) 
v_lshl_add_u32 v18, v14, 3, v16 // (op 9137) 





s_waitcnt vmcnt(19)
ds_write_b8 v18, v38 offset:12288 // (op 8292) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v18, v39 offset:12292 // (op 8292) Store local data 

v_and_b32 v14, -2, v1 // (op 8963) 
v_sub_u32 v16, v1, v14 // (op 8963) 

v_lshlrev_b32 v14, 6, v16 // (op 8963) 

v_lshrrev_b32 v16, 1, v1 // (op 8963) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 8963) 


v_mul_lo_u32 v14, v18, 6 // (op 8963) 
v_lshrrev_b32 v16, 3, v14 // (op 8963) 

v_lshrrev_b32 v14, 4, v18 // (op 8963) 

v_lshlrev_b32 v18, 2, v14 // (op 8963) 

v_add_u32 v14, v16, v18 // (op 8963) 





s_waitcnt vmcnt(17)
ds_write_b96 v14, v[40:42] offset:12800 // (op 8293) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[44:46] offset:12816 // (op 8293) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v14, v[48:50] offset:12832 // (op 8293) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[52:54] offset:12848 // (op 8293) Store local data 

v_and_b32 v14, -128, v1 // (op 9079) 
v_sub_u32 v16, v1, v14 // (op 9079) 

v_lshrrev_b32 v14, 7, v1 // (op 9079) 
v_lshl_add_u32 v18, v14, 8, v16 // (op 9079) 





s_waitcnt vmcnt(13)
ds_write_b8 v18, v36 offset:29184 // (op 8294) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v18, v37 offset:29312 // (op 8294) Store local data 

v_and_b32 v14, -2, v1 // (op 8934) 
v_sub_u32 v16, v1, v14 // (op 8934) 

v_lshlrev_b32 v14, 6, v16 // (op 8934) 

v_lshrrev_b32 v16, 1, v1 // (op 8934) 
v_lshl_add_u32 v18, v16, 7, v14 // (op 8934) 


v_mul_lo_u32 v14, v18, 6 // (op 8934) 
v_lshrrev_b32 v16, 3, v14 // (op 8934) 

v_lshrrev_b32 v14, 4, v18 // (op 8934) 

v_lshlrev_b32 v18, 2, v14 // (op 8934) 

v_add_u32 v14, v16, v18 // (op 8934) 


v_and_b32 v16, -2, v1 // (op 8992) 
v_sub_u32 v18, v1, v16 // (op 8992) 

v_lshlrev_b32 v16, 6, v18 // (op 8992) 

v_lshrrev_b32 v18, 1, v1 // (op 8992) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 8992) 


v_mul_lo_u32 v16, v20, 6 // (op 8992) 

v_lshrrev_b32 v18, 3, v16 // (op 8992) 

v_and_b32 v16, -128, v1 // (op 9050) 
v_sub_u32 v20, v1, v16 // (op 9050) 

v_lshrrev_b32 v16, 7, v1 // (op 9050) 
v_lshl_add_u32 v43, v16, 8, v20 // (op 9050) 


v_and_b32 v16, -4, v1 // (op 9108) 
v_sub_u32 v20, v1, v16 // (op 9108) 

v_lshrrev_b32 v16, 2, v1 // (op 9108) 
v_lshl_add_u32 v47, v16, 3, v20 // (op 9108) 


v_and_b32 v16, -2, v1 // (op 9189) 
v_sub_u32 v20, v1, v16 // (op 9189) 

v_lshlrev_b32 v16, 6, v20 // (op 9189) 

v_lshrrev_b32 v20, 1, v1 // (op 9189) 
v_lshl_add_u32 v51, v20, 7, v16 // (op 9189) 


v_mul_lo_u32 v16, v51, 6 // (op 9189) 
v_lshrrev_b32 v20, 3, v16 // (op 9189) 

v_lshrrev_b32 v16, 4, v51 // (op 9189) 

v_lshlrev_b32 v51, 2, v16 // (op 9189) 

v_add_u32 v16, v20, v51 // (op 9189) 


v_and_b32 v20, -2, v1 // (op 9218) 
v_sub_u32 v51, v1, v20 // (op 9218) 

v_lshlrev_b32 v20, 6, v51 // (op 9218) 

v_lshrrev_b32 v51, 1, v1 // (op 9218) 
v_lshl_add_u32 v55, v51, 7, v20 // (op 9218) 


v_mul_lo_u32 v20, v55, 6 // (op 9218) 

v_lshrrev_b32 v51, 3, v20 // (op 9218) 

v_and_b32 v20, -128, v1 // (op 9247) 
v_sub_u32 v55, v1, v20 // (op 9247) 

v_lshrrev_b32 v20, 7, v1 // (op 9247) 
v_lshl_add_u32 v73, v20, 8, v55 // (op 9247) 


v_and_b32 v20, -4, v1 // (op 9276) 
v_sub_u32 v55, v1, v20 // (op 9276) 

v_lshrrev_b32 v20, 2, v1 // (op 9276) 
v_lshl_add_u32 v77, v20, 3, v55 // (op 9276) 


s_mov_b32 s0, 0 // (op 2587) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 2586) 
s_mov_b32 s1, s18 // (op 2586) convert

s_and_b32 s18, s1, 4294967294 // (op 2586) 

s_cmp_lt_u32 s0, s18 // (op 2586) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2586 // (op 2586) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2586 if false)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2586:


s_barrier  // (op 9380) 
s_add_u32 s0, s0, 2 // (op 2588) 



v_mov_b32 v20, v19 // (op 10) 

ds_read_b96_tr_b6 v[88:90], v0 offset:12800 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[91:93], v0 offset:14848 // (op 70) Transpose load local data 
buffer_load_dwordx4 v[22:25], v20, s[64:67], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 9393) 
buffer_load_dwordx2 v[26:27], v20, s[64:67], 0 offen offset:208 // (op 10) Load value



ds_read_b128 v[94:97], v4 // (op 75) Load local data 
ds_read_b64 v[98:99], v4 offset:16 // (op 75) Load local data 
buffer_load_dwordx4 v[28:31], v20, s[64:67], 0 offen offset:216 // (op 10) Load value
s_barrier  // (op 9407) 
buffer_load_dwordx2 v[32:33], v20, s[64:67], 0 offen offset:232 // (op 10) Load value



ds_read_u8 v55, v3 offset:29184 // (op 80) Load local data 
v_mov_b64 v[34:35], s[42:43] // (op 25) convert

s_barrier  // (op 9421) 
v_mov_b32 v81, v15 // (op 25) 



ds_read_u8 v85, v5 offset:12288 // (op 85) Load local data 
v_mov_b32 v100, v81 // (op 25) 



buffer_load_ubyte v38, v100, s[72:75], 0 offen offset:8 // (op 25) Load value
v_add_u32 v100, v100, v34 // (op 25) 
buffer_load_ubyte v39, v100, s[72:75], 0 offen offset:8 // (op 25) Load value


ds_read_b128 v[100:103], v4 offset:3072 // (op 371) Load local data 


v_mov_b32 v34, v17 // (op 4) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 4) 
s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v35, v34, s1 // (op 4) 


s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:93], v[94:99], a[0:3], v55, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 88) 
ds_read_b64 v[104:105], v4 offset:3088 // (op 371) Load local data 
buffer_load_dwordx3 v[40:42], v35, s[52:55], 0 offen // (op 4) Load value



ds_read_u8 v106, v5 offset:12416 // (op 375) Load local data 
buffer_load_dwordx3 v[44:46], v35, s[52:55], 0 offen offset:12 // (op 4) Load value



buffer_load_dwordx3 v[48:50], v35, s[52:55], 0 offen offset:24 // (op 4) Load value
ds_read_b128 v[108:111], v4 offset:6144 // (op 547) Load local data 
ds_read_b64 v[112:113], v4 offset:6160 // (op 547) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:93], v[100:105], a[4:7], v55, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 376) 



ds_read_u8 v107, v5 offset:12544 // (op 551) Load local data 
buffer_load_dwordx3 v[52:54], v35, s[52:55], 0 offen offset:36 // (op 4) Load value





v_mov_b64 v[114:115], s[36:37] // (op 17) convert

v_mov_b32 v35, v21 // (op 17) 
s_lshl_b32 s1, s36, 3 // (op 17) 
v_add_u32 v116, v35, s1 // (op 17) 

ds_read_b128 v[118:121], v4 offset:9216 // (op 723) Load local data 
ds_read_b64 v[122:123], v4 offset:9232 // (op 723) Load local data 
buffer_load_ubyte v36, v116, s[68:71], 0 offen offset: 0 // (op 17) Load value



v_add_u32 v116, v116, v114 // (op 17) 
buffer_load_ubyte v37, v116, s[68:71], 0 offen offset: 0 // (op 17) Load value


ds_read_u8 v114, v5 offset:12672 // (op 727) Load local data 



s_barrier  // (op 9444) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[108:113], a[8:11], v55, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 552) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:93], v[118:123], a[12:15], v55, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 728) 
ds_read_b96_tr_b6 v[124:126], v0 offset:12832 // (op 1080) Transpose load local data 


ds_read_b96_tr_b6 v[127:129], v0 offset:14880 // (op 1080) Transpose load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v51, v[56:59] offset:29696 // (op 253) Store local data 


ds_read_u8 v55, v3 offset:29216 // (op 1083) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v51, v[60:61] offset:29712 // (op 253) Store local data 



ds_read_b96_tr_b6 v[88:90], v0 offset:12864 // (op 1850) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[94:99], a[16:19], v55, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1223) 
ds_read_b96_tr_b6 v[91:93], v0 offset:14912 // (op 1850) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:129], v[100:105], a[20:23], v55, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1177) 
ds_read_u8 v115, v3 offset:29248 // (op 1853) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:129], v[108:113], a[24:27], v55, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1131) 
ds_read_b96_tr_b6 v[130:132], v0 offset:12896 // (op 2620) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:129], v[118:123], a[28:31], v55, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1085) 
ds_read_b96_tr_b6 v[133:135], v0 offset:14944 // (op 2620) Transpose load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:93], v[94:99], a[32:35], v115, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1993) 
ds_read_u8 v55, v3 offset:29280 // (op 2623) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:93], v[100:105], a[36:39], v115, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1947) 
s_waitcnt vmcnt(21)
ds_write_b128 v51, v[62:65] offset:29720 // (op 253) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[108:113], a[40:43], v115, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1901) 
s_waitcnt vmcnt(20)
ds_write_b64 v51, v[66:67] offset:29736 // (op 253) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:93], v[118:123], a[44:47], v115, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1855) 


s_barrier  // (op 9456) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[94:99], a[48:51], v55, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2763) 
s_waitcnt vmcnt(19)
ds_write_b8 v77, v68 offset:41984 // (op 255) Store local data 


s_waitcnt vmcnt(18)
ds_write_b8 v77, v69 offset:41988 // (op 255) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[100:105], a[52:55], v55, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2717) 


s_barrier  // (op 9438) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[108:113], a[56:59], v55, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2671) 
s_waitcnt vmcnt(17)
ds_write_b96 v16, v[70:72] offset:42496 // (op 243) Store local data 


s_waitcnt vmcnt(16)
ds_write_b96 v16, v[74:76] offset:42512 // (op 243) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[118:123], a[60:63], v55, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2625) 
s_waitcnt vmcnt(15)
ds_write_b96 v16, v[78:80] offset:42528 // (op 243) Store local data 




s_waitcnt vmcnt(14)
ds_write_b96 v16, v[82:84] offset:42544 // (op 243) Store local data 
s_barrier  // (op 9450) 



s_waitcnt vmcnt(13)
ds_write_b8 v73, v86 offset:58880 // (op 254) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v73, v87 offset:59008 // (op 254) Store local data 



buffer_load_dwordx4 v[56:59], v20, s[64:67], 0 offen offset:288 // (op 256) Load value
buffer_load_dwordx2 v[60:61], v20, s[64:67], 0 offen offset:304 // (op 256) Load value
buffer_load_dwordx4 v[62:65], v20, s[64:67], 0 offen offset:312 // (op 256) Load value
s_waitcnt lgkmcnt(2)
s_barrier  // (op 9435) 
buffer_load_dwordx2 v[66:67], v20, s[64:67], 0 offen offset:328 // (op 256) Load value




ds_read_b96_tr_b6 v[88:90], v6 offset:42496 // (op 245) Transpose load local data 

ds_read_b96_tr_b6 v[91:93], v6 offset:44544 // (op 245) Transpose load local data 
v_add_u32 v19, 192, v19 // (op 8755) 
s_barrier  // (op 9441) 
v_mov_b64 v[94:95], s[42:43] // (op 258) convert




v_mov_b32 v20, v81 // (op 258) 
buffer_load_ubyte v68, v20, s[72:75], 0 offen offset:12 // (op 258) Load value
v_add_u32 v20, v20, v94 // (op 258) 
buffer_load_ubyte v69, v20, s[72:75], 0 offen offset:12 // (op 258) Load value


ds_read_b128 v[94:97], v8 offset:29696 // (op 247) Load local data 

ds_read_b64 v[98:99], v8 offset:29712 // (op 247) Load local data 


v_add_u32 v15, 8, v15 // (op 8883) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 242) 
s_mul_i32 s51, s19, 6 // (op 242) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 242) least significant
s_mul_hi_u32 s1, s18, 6 // (op 242) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 242) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 242) 

s_mul_i32 s1, 3, s18 // (op 242) 

v_add_u32 v20, v34, s1 // (op 242) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 9447) 
buffer_load_dwordx3 v[70:72], v20, s[52:55], 0 offen // (op 242) Load value


ds_read_u8 v55, v7 offset:58880 // (op 249) Load local data 
buffer_load_dwordx3 v[74:76], v20, s[52:55], 0 offen offset:12 // (op 242) Load value
s_barrier  // (op 9453) 
buffer_load_dwordx3 v[78:80], v20, s[52:55], 0 offen offset:24 // (op 242) Load value



ds_read_u8 v81, v9 offset:41984 // (op 251) Load local data 
buffer_load_dwordx3 v[82:84], v20, s[52:55], 0 offen offset:36 // (op 242) Load value





ds_read_b128 v[100:103], v8 offset:32768 // (op 381) Load local data 

ds_read_b64 v[104:105], v8 offset:32784 // (op 381) Load local data 
v_mov_b64 v[106:107], s[36:37] // (op 257) convert




ds_read_u8 v20, v9 offset:42112 // (op 385) Load local data 
s_lshl_b32 s1, s36, 2 // (op 257) 



s_mul_i32 s18, 3, s1 // (op 257) 

v_add_u32 v34, v35, s18 // (op 257) 

buffer_load_ubyte v86, v34, s[68:71], 0 offen offset: 0 // (op 257) Load value
v_add_u32 v34, v34, v106 // (op 257) 
buffer_load_ubyte v87, v34, s[68:71], 0 offen offset: 0 // (op 257) Load value


s_lshl_b64 s[18:19], s[26:27], 7 // (op 8691) 

s_mul_i32 s51, s19, 6 // (op 8691) most significant: low of high * low
s_barrier  // (op 9396) 
s_mul_i32 s50, s18, 6 // (op 8691) least significant



s_mul_hi_u32 s1, s18, 6 // (op 8691) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 8691) most significant: sum


s_lshl_b32 s1, s36, 3 // (op 8819) 
v_add_u32 v21, v21, s1 // (op 8819) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 8691) 

s_lshl_b32 s1, s18, 1 // (op 8691) 

v_add_u32 v17, v17, s1 // (op 8691) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:93], v[94:99], a[0:3], v55, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 252) 
ds_read_b128 v[106:109], v8 offset:35840 // (op 557) Load local data 
ds_read_b64 v[110:111], v8 offset:35856 // (op 557) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:93], v[100:105], a[4:7], v55, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 377) 



ds_read_u8 v85, v9 offset:42240 // (op 561) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v18, v[22:25] // (op 13) Store local data 



ds_read_b128 v[112:115], v8 offset:38912 // (op 733) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[106:111], a[8:11], v55, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 553) 
ds_read_b64 v[116:117], v8 offset:38928 // (op 733) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v18, v[26:27] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v18, v[28:31] offset:24 // (op 13) Store local data 



s_barrier  // (op 9513) 
ds_read_u8 v34, v9 offset:42368 // (op 737) Load local data 
s_waitcnt vmcnt(20)
ds_write_b64 v18, v[32:33] offset:40 // (op 13) Store local data 



s_barrier  // (op 9517) 
s_barrier  // (op 9424) 
ds_read_b96_tr_b6 v[118:120], v6 offset:42528 // (op 1088) Transpose load local data 



ds_read_b96_tr_b6 v[121:123], v6 offset:44576 // (op 1088) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:93], v[112:117], a[12:15], v55, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 729) 




ds_read_u8 v35, v7 offset:58912 // (op 1091) Load local data 
s_waitcnt vmcnt(19)
ds_write_b8 v47, v38 offset:12288 // (op 30) Store local data 



ds_read_b96_tr_b6 v[88:90], v6 offset:42560 // (op 1858) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:123], v[94:99], a[16:19], v35, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1224) 
ds_read_b96_tr_b6 v[91:93], v6 offset:44608 // (op 1858) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:123], v[100:105], a[20:23], v35, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1178) 
ds_read_u8 v55, v7 offset:58944 // (op 1861) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:123], v[106:111], a[24:27], v35, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1132) 
ds_read_b96_tr_b6 v[124:126], v6 offset:42592 // (op 2628) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[118:123], v[112:117], a[28:31], v35, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1086) 
ds_read_b96_tr_b6 v[127:129], v6 offset:44640 // (op 2628) Transpose load local data 




s_barrier  // (op 9553) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:93], v[94:99], a[32:35], v55, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1994) 
ds_read_u8 v35, v7 offset:58976 // (op 2631) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:93], v[100:105], a[36:39], v55, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1948) 
s_barrier  // (op 9557) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[106:111], a[40:43], v55, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1902) 
s_waitcnt vmcnt(18)
ds_write_b8 v47, v39 offset:12292 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:93], v[112:117], a[44:47], v55, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1856) 


s_barrier  // (op 9383) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[94:99], a[48:51], v35, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2764) 
s_waitcnt vmcnt(17)
ds_write_b96 v14, v[40:42] offset:12800 // (op 6) Store local data 


s_waitcnt vmcnt(16)
ds_write_b96 v14, v[44:46] offset:12816 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:129], v[100:105], a[52:55], v35, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2718) 
s_waitcnt vmcnt(15)
ds_write_b96 v14, v[48:50] offset:12832 // (op 6) Store local data 


s_waitcnt vmcnt(14)
ds_write_b96 v14, v[52:54] offset:12848 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:129], v[106:111], a[56:59], v35, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2672) 


s_barrier  // (op 9410) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:129], v[112:117], a[60:63], v35, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2626) 
s_waitcnt vmcnt(13)
ds_write_b8 v43, v36 offset:29184 // (op 21) Store local data 




s_waitcnt vmcnt(12)
ds_write_b8 v43, v37 offset:29312 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 2586) 
s_mov_b32 s1, s18 // (op 2586) convert

s_and_b32 s18, s1, 4294967294 // (op 2586) 

s_cmp_lt_u32 s0, s18 // (op 2586) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2586 // (op 2586) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2586 if true)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2586:

















s_mul_hi_i32 s0, s16, s10 // (op 8639) 
v_and_b32 v73, -2, v1 // (op 8639) 
s_add_i32 s1, s0, s16 // (op 8639) 

v_sub_u32 v77, v1, v73 // (op 8639) 

s_ashr_i32 s0, s1, 31 // (op 8639) 
v_lshlrev_b32 v73, 6, v77 // (op 8639) 

s_cmp_eq_i32 s10, 0 // (op 8639) 
s_mov_b32 s18, scc // (op 8639) 
v_lshrrev_b32 v77, 1, v1 // (op 8639) 
s_cmp_eq_i32 s18, 0 // (op 8639) 
s_cselect_b32 s19, 0, -1 // (op 8639) 

s_and_b32 s18, s11, 31 // (op 8639) 
s_lshl_b32 s22, 1, s18 // (op 8639) 
s_add_i32 s23, s22, s19 // (op 8639) 


s_and_b32 s19, s0, s23 // (op 8639) 


s_add_i32 s0, s1, s19 // (op 8639) 


s_ashr_i32 s1, s0, s18 // (op 8639) 


s_lshl_b32 s0, s11, 24 // (op 8639) 
s_ashr_i32 s18, s0, 31 // (op 8639) 

s_xor_b32 s0, s1, s18 // (op 8639) 

s_sub_i32 s1, s0, s18 // (op 8639) 


s_mul_hi_i32 s0, s1, s12 // (op 8639) 
s_add_i32 s18, s0, s1 // (op 8639) 

s_ashr_i32 s0, s18, 31 // (op 8639) 
s_cmp_eq_i32 s12, 0 // (op 8639) 
s_mov_b32 s19, scc // (op 8639) 
s_cmp_eq_i32 s19, 0 // (op 8639) 
s_cselect_b32 s22, 0, -1 // (op 8639) 

s_and_b32 s19, s13, 31 // (op 8639) 
s_lshl_b32 s23, 1, s19 // (op 8639) 
s_add_i32 s24, s23, s22 // (op 8639) 


s_and_b32 s25, s0, s24 // (op 8639) 


s_add_i32 s0, s18, s25 // (op 8639) 


s_ashr_i32 s18, s0, s19 // (op 8639) 


s_lshl_b32 s0, s13, 24 // (op 8639) 
s_ashr_i32 s19, s0, 31 // (op 8639) 

s_xor_b32 s0, s18, s19 // (op 8639) 

s_sub_i32 s18, s0, s19 // (op 8639) 


s_mul_i32 s0, s1, s15 // (op 8639) 

s_sub_u32 s1, s16, s0 // (op 8639) 

s_mul_i32 s0, s18, s15 // (op 8639) 
s_add_u32 s0, s0, s1 // (op 8639) 


s_mul_hi_i32 s1, s16, s9 // (op 8639) 
s_add_i32 s18, s1, s16 // (op 8639) 

s_ashr_i32 s1, s18, 31 // (op 8639) 
s_cmp_eq_i32 s9, 0 // (op 8639) 
s_mov_b32 s19, scc // (op 8639) 
s_cmp_eq_i32 s19, 0 // (op 8639) 
s_cselect_b32 s22, 0, -1 // (op 8639) 

s_and_b32 s19, s46, 31 // (op 8639) 
s_lshl_b32 s23, 1, s19 // (op 8639) 
s_add_i32 s24, s23, s22 // (op 8639) 


s_and_b32 s25, s1, s24 // (op 8639) 


s_add_i32 s1, s18, s25 // (op 8639) 


s_ashr_i32 s18, s1, s19 // (op 8639) 


s_lshl_b32 s1, s46, 24 // (op 8639) 
s_ashr_i32 s19, s1, 31 // (op 8639) 

s_xor_b32 s1, s18, s19 // (op 8639) 

s_sub_i32 s18, s1, s19 // (op 8639) 


s_add_u32 s50, s6, 128 // (op 8639) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8639) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8639) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8639) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8639) 

s_mul_i32 s1, s15, s24 // (op 8639) 

s_mul_i32 s19, s8, s1 // (op 8639) 

s_mul_i32 s1, s18, s19 // (op 8639) 

s_sub_u32 s19, s16, s1 // (op 8639) 

s_mov_b32 s1, s19 // (op 8639) 
s_mul_hi_i32 s22, s1, s14 // (op 8639) 
s_add_i32 s23, s22, s1 // (op 8639) 


s_ashr_i32 s1, s23, 31 // (op 8639) 
s_cmp_eq_i32 s14, 0 // (op 8639) 
s_mov_b32 s22, scc // (op 8639) 
s_cmp_eq_i32 s22, 0 // (op 8639) 
s_cselect_b32 s24, 0, -1 // (op 8639) 

s_and_b32 s22, s48, 31 // (op 8639) 
s_lshl_b32 s25, 1, s22 // (op 8639) 
s_add_i32 s49, s25, s24 // (op 8639) 


s_and_b32 s24, s1, s49 // (op 8639) 


s_add_i32 s1, s23, s24 // (op 8639) 


s_ashr_i32 s23, s1, s22 // (op 8639) 


s_lshl_b32 s1, s48, 24 // (op 8639) 
s_ashr_i32 s22, s1, 31 // (op 8639) 

s_xor_b32 s1, s23, s22 // (op 8639) 

s_sub_i32 s23, s1, s22 // (op 8639) 


s_mul_i32 s1, s47, s15 // (op 8639) 
s_add_u32 s24, s20, 128 // (op 8639) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8639) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8639) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8639) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 8639) 

s_sub_i32 s22, s24, s1 // (op 8639) 

s_mul_i32 s24, s23, s22 // (op 8639) 


s_sub_u32 s25, s19, s24 // (op 8639) 


s_add_u32 s19, s25, s1 // (op 8639) 


s_cmp_eq_i32 s18, 0 // (op 8639) 
s_mov_b32 s1, scc // (op 8639) 

s_cmp_eq_i32 s1, 0 // (op 8639) 
s_cselect_b32 s18, s19, s0 // (op 8639) 



v_lshl_add_u32 v81, s18, 7, v73 // (op 8639) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8639) 
v_mov_b32 v15, 0 // (op 8639) convert
v_mov_b32 v14, v81 // (op 8639) convert

s_lshr_b32 s18, s0, 1 // (op 8639) 

v_lshl_add_u32 v73, s18, 8, v77 // (op 8639) 


v_mul_lo_u32 v17, v73, s27 // (op 8639) most significant: low of low * high
v_mul_hi_u32 v77, v73, s26 // (op 8639) most significant: high of low * low
v_mul_lo_u32 v16, v73, s26 // (op 8639) least significant: low of low * low
v_add_u32 v17, v17, v77 // (op 8639) most significant: sum


v_add_co_u32 v18, s[0:1], v14, v16 // (op 8639) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 8639) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 8639) most significant: low of high * low
v_mul_hi_u32 v73, v18, 6 // (op 8639) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 8639) least significant: low of low * low
v_add_u32 v15, v15, v73 // (op 8639) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8639) 

v_mov_b32 v73, v16 // (op 8639) convert

s_mul_i32 s1, s31, 6 // (op 8641) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 8641) least significant
s_mul_hi_u32 s18, s30, 6 // (op 8641) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8641) most significant: sum

s_mul_i32 s19, s26, s1 // (op 8641) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 8641) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 8641) least significant
s_mul_hi_u32 s23, s26, s0 // (op 8641) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8641) most significant: sum
s_add_u32 s19, s19, s23 // (op 8641) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8641) 

s_mov_b32 s52, s28 // (op 8641) reg expression
s_mov_b32 s53, s29 // (op 8641) reg expression
s_mov_b32 s54, s0 // (op 8641) reg expression
s_mov_b32 s55, 131072 // (op 8641) call()

s_mul_hi_i32 s0, s16, s10 // (op 8721) 
v_lshrrev_b32 v77, 1, v1 // (op 8721) 
s_add_i32 s1, s0, s16 // (op 8721) 

v_and_b32 v81, -2, v1 // (op 8721) 
s_ashr_i32 s0, s1, 31 // (op 8721) 
v_sub_u32 v85, v1, v81 // (op 8721) 

s_cmp_eq_i32 s10, 0 // (op 8721) 
s_mov_b32 s18, scc // (op 8721) 
v_lshlrev_b32 v81, 6, v85 // (op 8721) 

s_cmp_eq_i32 s18, 0 // (op 8721) 
s_cselect_b32 s19, 0, -1 // (op 8721) 

s_and_b32 s18, s11, 31 // (op 8721) 
s_lshl_b32 s22, 1, s18 // (op 8721) 
s_add_i32 s23, s22, s19 // (op 8721) 


s_and_b32 s19, s0, s23 // (op 8721) 


s_add_i32 s0, s1, s19 // (op 8721) 


s_ashr_i32 s1, s0, s18 // (op 8721) 


s_lshl_b32 s0, s11, 24 // (op 8721) 
s_ashr_i32 s18, s0, 31 // (op 8721) 

s_xor_b32 s0, s1, s18 // (op 8721) 

s_sub_i32 s1, s0, s18 // (op 8721) 


s_mul_hi_i32 s0, s1, s12 // (op 8721) 
s_add_i32 s18, s0, s1 // (op 8721) 

s_ashr_i32 s0, s18, 31 // (op 8721) 
s_cmp_eq_i32 s12, 0 // (op 8721) 
s_mov_b32 s19, scc // (op 8721) 
s_cmp_eq_i32 s19, 0 // (op 8721) 
s_cselect_b32 s22, 0, -1 // (op 8721) 

s_and_b32 s19, s13, 31 // (op 8721) 
s_lshl_b32 s23, 1, s19 // (op 8721) 
s_add_i32 s24, s23, s22 // (op 8721) 


s_and_b32 s25, s0, s24 // (op 8721) 


s_add_i32 s0, s18, s25 // (op 8721) 


s_ashr_i32 s18, s0, s19 // (op 8721) 


s_lshl_b32 s0, s13, 24 // (op 8721) 
s_ashr_i32 s19, s0, 31 // (op 8721) 

s_xor_b32 s0, s18, s19 // (op 8721) 

s_sub_i32 s18, s0, s19 // (op 8721) 


s_add_u32 s50, s6, 128 // (op 8721) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8721) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8721) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8721) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8721) 

s_mul_i32 s0, s18, s24 // (op 8721) 

s_sub_i32 s18, s1, s0 // (op 8721) 


s_mul_hi_i32 s19, s16, s9 // (op 8721) 
s_add_i32 s0, s19, s16 // (op 8721) 

s_ashr_i32 s1, s0, 31 // (op 8721) 
s_cmp_eq_i32 s9, 0 // (op 8721) 
s_mov_b32 s19, scc // (op 8721) 
s_cmp_eq_i32 s19, 0 // (op 8721) 
s_cselect_b32 s22, 0, -1 // (op 8721) 

s_and_b32 s19, s46, 31 // (op 8721) 
s_lshl_b32 s23, 1, s19 // (op 8721) 
s_add_i32 s49, s23, s22 // (op 8721) 


s_and_b32 s22, s1, s49 // (op 8721) 


s_add_i32 s1, s0, s22 // (op 8721) 


s_ashr_i32 s0, s1, s19 // (op 8721) 


s_lshl_b32 s1, s46, 24 // (op 8721) 
s_ashr_i32 s19, s1, 31 // (op 8721) 

s_xor_b32 s1, s0, s19 // (op 8721) 

s_sub_i32 s0, s1, s19 // (op 8721) 


s_mul_i32 s1, s15, s24 // (op 8721) 

s_mul_i32 s19, s8, s1 // (op 8721) 

s_mul_i32 s1, s0, s19 // (op 8721) 

s_sub_u32 s19, s16, s1 // (op 8721) 

s_mul_hi_i32 s1, s19, s14 // (op 8721) 
s_add_i32 s22, s1, s19 // (op 8721) 


s_ashr_i32 s1, s22, 31 // (op 8721) 
s_cmp_eq_i32 s14, 0 // (op 8721) 
s_mov_b32 s19, scc // (op 8721) 
s_cmp_eq_i32 s19, 0 // (op 8721) 
s_cselect_b32 s23, 0, -1 // (op 8721) 

s_and_b32 s19, s48, 31 // (op 8721) 
s_lshl_b32 s24, 1, s19 // (op 8721) 
s_add_i32 s25, s24, s23 // (op 8721) 


s_and_b32 s23, s1, s25 // (op 8721) 


s_add_i32 s1, s22, s23 // (op 8721) 


s_ashr_i32 s22, s1, s19 // (op 8721) 


s_lshl_b32 s1, s48, 24 // (op 8721) 
s_ashr_i32 s19, s1, 31 // (op 8721) 

s_xor_b32 s1, s22, s19 // (op 8721) 

s_sub_i32 s22, s1, s19 // (op 8721) 


s_cmp_eq_i32 s0, 0 // (op 8721) 
s_mov_b32 s1, scc // (op 8721) 

s_cmp_eq_i32 s1, 0 // (op 8721) 
s_cselect_b32 s0, s22, s18 // (op 8721) 



v_lshl_add_u32 v85, s0, 7, v77 // (op 8721) 


v_mul_lo_u32 v15, v85, s33 // (op 8721) most significant: low of low * high
v_mul_hi_u32 v77, v85, s32 // (op 8721) most significant: high of low * low
v_mul_lo_u32 v14, v85, s32 // (op 8721) least significant: low of low * low
v_add_u32 v15, v15, v77 // (op 8721) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8721) 
s_lshr_b32 s18, s0, 1 // (op 8721) 

v_lshl_add_u32 v77, s18, 8, v81 // (op 8721) 


v_mov_b32 v17, 0 // (op 8721) convert
v_mov_b32 v16, v77 // (op 8721) convert

v_add_co_u32 v18, s[0:1], v16, v14 // (op 8721) least significant half
v_addc_co_u32 v19, s[0:1], v17, v15, s[0:1] // (op 8721) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 8721) most significant: low of high * low
v_mul_hi_u32 v77, v18, 6 // (op 8721) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 8721) least significant: low of low * low
v_add_u32 v15, v15, v77 // (op 8721) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8721) 

v_mov_b32 v77, v16 // (op 8721) convert

s_mul_i32 s1, s7, 6 // (op 8723) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 8723) least significant
s_mul_hi_u32 s18, s6, 6 // (op 8723) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8723) most significant: sum

s_mul_i32 s19, s32, s1 // (op 8723) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 8723) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 8723) least significant
s_mul_hi_u32 s23, s32, s0 // (op 8723) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 8723) most significant: sum
s_add_u32 s19, s19, s23 // (op 8723) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8723) 

s_mov_b32 s64, s34 // (op 8723) reg expression
s_mov_b32 s65, s35 // (op 8723) reg expression
s_mov_b32 s66, s0 // (op 8723) reg expression
s_mov_b32 s67, 131072 // (op 8723) call()

s_mul_hi_i32 s0, s16, s10 // (op 8785) 
v_and_b32 v81, -128, v1 // (op 8785) 
s_add_i32 s1, s0, s16 // (op 8785) 

v_sub_u32 v85, v1, v81 // (op 8785) 

s_ashr_i32 s0, s1, 31 // (op 8785) 
v_lshrrev_b32 v81, 7, v1 // (op 8785) 
s_cmp_eq_i32 s10, 0 // (op 8785) 
s_mov_b32 s18, scc // (op 8785) 
v_lshlrev_b32 v14, 1, v81 // (op 8785) 

s_cmp_eq_i32 s18, 0 // (op 8785) 
s_cselect_b32 s19, 0, -1 // (op 8785) 

s_and_b32 s18, s11, 31 // (op 8785) 
s_lshl_b32 s22, 1, s18 // (op 8785) 
s_add_i32 s23, s22, s19 // (op 8785) 


s_and_b32 s19, s0, s23 // (op 8785) 


s_add_i32 s0, s1, s19 // (op 8785) 


s_ashr_i32 s1, s0, s18 // (op 8785) 


s_lshl_b32 s0, s11, 24 // (op 8785) 
s_ashr_i32 s18, s0, 31 // (op 8785) 

s_xor_b32 s0, s1, s18 // (op 8785) 

s_sub_i32 s1, s0, s18 // (op 8785) 


s_mul_hi_i32 s0, s1, s12 // (op 8785) 
s_add_i32 s18, s0, s1 // (op 8785) 

s_ashr_i32 s0, s18, 31 // (op 8785) 
s_cmp_eq_i32 s12, 0 // (op 8785) 
s_mov_b32 s19, scc // (op 8785) 
s_cmp_eq_i32 s19, 0 // (op 8785) 
s_cselect_b32 s22, 0, -1 // (op 8785) 

s_and_b32 s19, s13, 31 // (op 8785) 
s_lshl_b32 s23, 1, s19 // (op 8785) 
s_add_i32 s24, s23, s22 // (op 8785) 


s_and_b32 s25, s0, s24 // (op 8785) 


s_add_i32 s0, s18, s25 // (op 8785) 


s_ashr_i32 s18, s0, s19 // (op 8785) 


s_lshl_b32 s0, s13, 24 // (op 8785) 
s_ashr_i32 s19, s0, 31 // (op 8785) 

s_xor_b32 s0, s18, s19 // (op 8785) 

s_sub_i32 s18, s0, s19 // (op 8785) 


s_mul_i32 s0, s1, s15 // (op 8785) 

s_sub_u32 s1, s16, s0 // (op 8785) 

s_mul_i32 s0, s18, s15 // (op 8785) 
s_add_u32 s0, s0, s1 // (op 8785) 


s_mul_hi_i32 s1, s16, s9 // (op 8785) 
s_add_i32 s18, s1, s16 // (op 8785) 

s_ashr_i32 s1, s18, 31 // (op 8785) 
s_cmp_eq_i32 s9, 0 // (op 8785) 
s_mov_b32 s19, scc // (op 8785) 
s_cmp_eq_i32 s19, 0 // (op 8785) 
s_cselect_b32 s22, 0, -1 // (op 8785) 

s_and_b32 s19, s46, 31 // (op 8785) 
s_lshl_b32 s23, 1, s19 // (op 8785) 
s_add_i32 s24, s23, s22 // (op 8785) 


s_and_b32 s25, s1, s24 // (op 8785) 


s_add_i32 s1, s18, s25 // (op 8785) 


s_ashr_i32 s18, s1, s19 // (op 8785) 


s_lshl_b32 s1, s46, 24 // (op 8785) 
s_ashr_i32 s19, s1, 31 // (op 8785) 

s_xor_b32 s1, s18, s19 // (op 8785) 

s_sub_i32 s18, s1, s19 // (op 8785) 


s_add_u32 s50, s6, 128 // (op 8785) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8785) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8785) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8785) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8785) 

s_mul_i32 s1, s15, s24 // (op 8785) 

s_mul_i32 s19, s8, s1 // (op 8785) 

s_mul_i32 s1, s18, s19 // (op 8785) 

s_sub_u32 s19, s16, s1 // (op 8785) 

s_mov_b32 s1, s19 // (op 8785) 
s_mul_hi_i32 s22, s1, s14 // (op 8785) 
s_add_i32 s23, s22, s1 // (op 8785) 


s_ashr_i32 s1, s23, 31 // (op 8785) 
s_cmp_eq_i32 s14, 0 // (op 8785) 
s_mov_b32 s22, scc // (op 8785) 
s_cmp_eq_i32 s22, 0 // (op 8785) 
s_cselect_b32 s24, 0, -1 // (op 8785) 

s_and_b32 s22, s48, 31 // (op 8785) 
s_lshl_b32 s25, 1, s22 // (op 8785) 
s_add_i32 s49, s25, s24 // (op 8785) 


s_and_b32 s24, s1, s49 // (op 8785) 


s_add_i32 s1, s23, s24 // (op 8785) 


s_ashr_i32 s23, s1, s22 // (op 8785) 


s_lshl_b32 s1, s48, 24 // (op 8785) 
s_ashr_i32 s22, s1, 31 // (op 8785) 

s_xor_b32 s1, s23, s22 // (op 8785) 

s_sub_i32 s23, s1, s22 // (op 8785) 


s_mul_i32 s1, s47, s15 // (op 8785) 
s_add_u32 s24, s20, 128 // (op 8785) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8785) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8785) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8785) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 8785) 

s_sub_i32 s22, s24, s1 // (op 8785) 

s_mul_i32 s24, s23, s22 // (op 8785) 


s_sub_u32 s25, s19, s24 // (op 8785) 


s_add_u32 s19, s25, s1 // (op 8785) 


s_cmp_eq_i32 s18, 0 // (op 8785) 
s_mov_b32 s1, scc // (op 8785) 

s_cmp_eq_i32 s1, 0 // (op 8785) 
s_cselect_b32 s18, s19, s0 // (op 8785) 



v_lshl_add_u32 v81, s18, 7, v85 // (op 8785) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8785) 
s_lshr_b32 s18, s0, 1 // (op 8785) 

v_lshl_add_u32 v85, s18, 3, v14 // (op 8785) 


v_mul_lo_u32 v14, v85, s36 // (op 8785) 

v_add_u32 v85, v81, v14 // (op 8785) 


s_mul_i32 s1, s36, s41 // (op 8787) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 8787) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 8787) least significant
s_mul_hi_u32 s19, s36, s40 // (op 8787) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8787) most significant: sum
s_add_u32 s1, s1, s19 // (op 8787) most significant: sum


s_mov_b32 s68, s38 // (op 8787) reg expression
s_mov_b32 s69, s39 // (op 8787) reg expression
s_mov_b32 s70, s0 // (op 8787) reg expression
s_mov_b32 s71, 131072 // (op 8787) call()

s_mul_hi_i32 s0, s16, s10 // (op 8849) 
v_lshrrev_b32 v81, 2, v1 // (op 8849) 
s_add_i32 s1, s0, s16 // (op 8849) 

v_lshlrev_b32 v14, 1, v81 // (op 8849) 

s_ashr_i32 s0, s1, 31 // (op 8849) 
v_and_b32 v81, -4, v1 // (op 8849) 
s_cmp_eq_i32 s10, 0 // (op 8849) 
s_mov_b32 s18, scc // (op 8849) 
v_sub_u32 v15, v1, v81 // (op 8849) 

s_cmp_eq_i32 s18, 0 // (op 8849) 
s_cselect_b32 s19, 0, -1 // (op 8849) 

s_and_b32 s18, s11, 31 // (op 8849) 
s_lshl_b32 s22, 1, s18 // (op 8849) 
s_add_i32 s23, s22, s19 // (op 8849) 


s_and_b32 s19, s0, s23 // (op 8849) 


s_add_i32 s0, s1, s19 // (op 8849) 


s_ashr_i32 s1, s0, s18 // (op 8849) 


s_lshl_b32 s0, s11, 24 // (op 8849) 
s_ashr_i32 s18, s0, 31 // (op 8849) 

s_xor_b32 s0, s1, s18 // (op 8849) 

s_sub_i32 s1, s0, s18 // (op 8849) 


s_mul_hi_i32 s0, s1, s12 // (op 8849) 
s_add_i32 s18, s0, s1 // (op 8849) 

s_ashr_i32 s0, s18, 31 // (op 8849) 
s_cmp_eq_i32 s12, 0 // (op 8849) 
s_mov_b32 s19, scc // (op 8849) 
s_cmp_eq_i32 s19, 0 // (op 8849) 
s_cselect_b32 s22, 0, -1 // (op 8849) 

s_and_b32 s19, s13, 31 // (op 8849) 
s_lshl_b32 s23, 1, s19 // (op 8849) 
s_add_i32 s24, s23, s22 // (op 8849) 


s_and_b32 s25, s0, s24 // (op 8849) 


s_add_i32 s0, s18, s25 // (op 8849) 


s_ashr_i32 s18, s0, s19 // (op 8849) 


s_lshl_b32 s0, s13, 24 // (op 8849) 
s_ashr_i32 s19, s0, 31 // (op 8849) 

s_xor_b32 s0, s18, s19 // (op 8849) 

s_sub_i32 s18, s0, s19 // (op 8849) 


s_add_u32 s50, s6, 128 // (op 8849) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8849) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8849) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8849) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 8849) 

s_mul_i32 s0, s18, s24 // (op 8849) 

s_sub_i32 s18, s1, s0 // (op 8849) 


s_mul_hi_i32 s19, s16, s9 // (op 8849) 
s_add_i32 s0, s19, s16 // (op 8849) 

s_ashr_i32 s1, s0, 31 // (op 8849) 
s_cmp_eq_i32 s9, 0 // (op 8849) 
s_mov_b32 s19, scc // (op 8849) 
s_cmp_eq_i32 s19, 0 // (op 8849) 
s_cselect_b32 s22, 0, -1 // (op 8849) 

s_and_b32 s19, s46, 31 // (op 8849) 
s_lshl_b32 s23, 1, s19 // (op 8849) 
s_add_i32 s49, s23, s22 // (op 8849) 


s_and_b32 s22, s1, s49 // (op 8849) 


s_add_i32 s1, s0, s22 // (op 8849) 


s_ashr_i32 s0, s1, s19 // (op 8849) 


s_lshl_b32 s1, s46, 24 // (op 8849) 
s_ashr_i32 s19, s1, 31 // (op 8849) 

s_xor_b32 s1, s0, s19 // (op 8849) 

s_sub_i32 s0, s1, s19 // (op 8849) 


s_mul_i32 s1, s15, s24 // (op 8849) 

s_mul_i32 s19, s8, s1 // (op 8849) 

s_mul_i32 s1, s0, s19 // (op 8849) 

s_sub_u32 s19, s16, s1 // (op 8849) 

s_mul_hi_i32 s1, s19, s14 // (op 8849) 
s_add_i32 s22, s1, s19 // (op 8849) 


s_ashr_i32 s1, s22, 31 // (op 8849) 
s_cmp_eq_i32 s14, 0 // (op 8849) 
s_mov_b32 s19, scc // (op 8849) 
s_cmp_eq_i32 s19, 0 // (op 8849) 
s_cselect_b32 s23, 0, -1 // (op 8849) 

s_and_b32 s19, s48, 31 // (op 8849) 
s_lshl_b32 s24, 1, s19 // (op 8849) 
s_add_i32 s25, s24, s23 // (op 8849) 


s_and_b32 s23, s1, s25 // (op 8849) 


s_add_i32 s1, s22, s23 // (op 8849) 


s_ashr_i32 s22, s1, s19 // (op 8849) 


s_lshl_b32 s1, s48, 24 // (op 8849) 
s_ashr_i32 s19, s1, 31 // (op 8849) 

s_xor_b32 s1, s22, s19 // (op 8849) 

s_sub_i32 s22, s1, s19 // (op 8849) 


s_cmp_eq_i32 s0, 0 // (op 8849) 
s_mov_b32 s1, scc // (op 8849) 

s_cmp_eq_i32 s1, 0 // (op 8849) 
s_cselect_b32 s0, s22, s18 // (op 8849) 



v_lshl_add_u32 v81, s0, 7, v14 // (op 8849) 


v_mul_lo_u32 v14, v81, s42 // (op 8849) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 8849) 
s_lshr_b32 s18, s0, 1 // (op 8849) 

v_lshl_add_u32 v81, s18, 3, v15 // (op 8849) 


v_add_u32 v15, v81, v14 // (op 8849) 


s_mul_i32 s1, s42, s7 // (op 8851) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 8851) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 8851) least significant
s_mul_hi_u32 s19, s42, s6 // (op 8851) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8851) most significant: sum
s_add_u32 s1, s1, s19 // (op 8851) most significant: sum


s_mov_b32 s72, s44 // (op 8851) reg expression
s_mov_b32 s73, s45 // (op 8851) reg expression
s_mov_b32 s74, s0 // (op 8851) reg expression
s_mov_b32 s75, 131072 // (op 8851) call()

v_and_b32 v14, -2, v1 // (op 8947) 
v_sub_u32 v81, v1, v14 // (op 8947) 

v_lshlrev_b32 v14, 6, v81 // (op 8947) 

v_lshrrev_b32 v81, 1, v1 // (op 8947) 
v_lshl_add_u32 v16, v81, 7, v14 // (op 8947) 


v_mul_lo_u32 v14, v16, 6 // (op 8947) 
v_lshrrev_b32 v81, 3, v14 // (op 8947) 

v_lshrrev_b32 v14, 4, v16 // (op 8947) 

v_lshlrev_b32 v16, 2, v14 // (op 8947) 

v_add_u32 v14, v81, v16 // (op 8947) 


v_and_b32 v81, -2, v1 // (op 9005) 
v_sub_u32 v16, v1, v81 // (op 9005) 

v_lshlrev_b32 v81, 6, v16 // (op 9005) 

v_lshrrev_b32 v16, 1, v1 // (op 9005) 
v_lshl_add_u32 v17, v16, 7, v81 // (op 9005) 


v_mul_lo_u32 v16, v17, 6 // (op 9005) 

v_lshrrev_b32 v81, 3, v16 // (op 9005) 

v_and_b32 v16, -128, v1 // (op 9063) 
v_sub_u32 v17, v1, v16 // (op 9063) 

v_lshrrev_b32 v16, 7, v1 // (op 9063) 
v_lshl_add_u32 v18, v16, 8, v17 // (op 9063) 


v_and_b32 v16, -4, v1 // (op 9121) 
v_sub_u32 v17, v1, v16 // (op 9121) 

v_lshrrev_b32 v16, 2, v1 // (op 9121) 
v_lshl_add_u32 v19, v16, 3, v17 // (op 9121) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2592) 
s_and_b32 s18, s0, 4294967294 // (op 2592) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 2591) 
s_mov_b32 s19, s0 // (op 2591) convert

s_cmp_lt_u32 s18, s19 // (op 2591) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2591 // (op 2591) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2591 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2591:


s_add_u32 s18, s18, 1 // (op 2593) 
v_mov_b32 v16, v73 // (op 176) 
v_mov_b32 v17, v16 // (op 176) reg expression

buffer_load_dwordx3 v[20:22], v17, s[52:55], 0 offen // (op 176) Load value
buffer_load_dwordx3 v[24:26], v17, s[52:55], 0 offen offset:12 // (op 176) Load value
buffer_load_dwordx3 v[28:30], v17, s[52:55], 0 offen offset:24 // (op 176) Load value
buffer_load_dwordx3 v[32:34], v17, s[52:55], 0 offen offset:36 // (op 176) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 8629) 
s_mul_i32 s51, s1, 6 // (op 8629) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 8629) least significant
s_mul_hi_u32 s19, s0, 6 // (op 8629) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 8629) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 8629) 

v_add_u32 v73, v73, s0 // (op 8629) 



s_barrier  // (op 9386) 
v_mov_b32 v23, v77 // (op 190) 




buffer_load_dwordx4 v[36:39], v23, s[64:67], 0 offen // (op 190) Load value
buffer_load_dwordx2 v[40:41], v23, s[64:67], 0 offen offset:16 // (op 190) Load value
s_waitcnt vmcnt(5)
ds_write_b96 v14, v[20:22] offset:12800 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v14, v[24:26] offset:12816 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v14, v[28:30] offset:12832 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[32:34] offset:12848 // (op 177) Store local data 
buffer_load_dwordx4 v[42:45], v23, s[64:67], 0 offen offset:24 // (op 190) Load value

buffer_load_dwordx2 v[46:47], v23, s[64:67], 0 offen offset:40 // (op 190) Load value

v_add_u32 v77, 96, v77 // (op 8713) 

s_barrier  // (op 9399) 
v_mov_b64 v[16:17], s[36:37] // (op 191) convert




v_mov_b32 v20, v85 // (op 191) 
v_mov_b32 v21, v20 // (op 191) reg expression
buffer_load_ubyte v22, v21, s[68:71], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v21, v21, v16 // (op 191) 
s_waitcnt vmcnt(4)
ds_write_b128 v81, v[36:39] // (op 187) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v81, v[40:41] offset:16 // (op 187) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v81, v[42:45] offset:24 // (op 187) Store local data 
buffer_load_ubyte v23, v21, s[68:71], 0 offen offset: 0 // (op 191) Load value



s_lshl_b32 s19, s36, 2 // (op 8777) 
v_add_u32 v85, v85, s19 // (op 8777) 

s_waitcnt vmcnt(2)
ds_write_b64 v81, v[46:47] offset:40 // (op 187) Store local data 


v_mov_b64 v[16:17], s[42:43] // (op 192) convert

s_barrier  // (op 9413) 
v_mov_b32 v20, v15 // (op 192) 



v_mov_b32 v21, v20 // (op 192) 
buffer_load_ubyte v24, v21, s[72:75], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v21, v21, v16 // (op 192) 
buffer_load_ubyte v25, v21, s[72:75], 0 offen offset: 0 // (op 192) Load value



v_add_u32 v15, 4, v15 // (op 8841) 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v22 offset:29184 // (op 188) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9389) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:29312 // (op 188) Store local data 




ds_read_b96_tr_b6 v[26:28], v0 offset:12800 // (op 179) Transpose load local data 
s_barrier  // (op 9427) 
ds_read_b96_tr_b6 v[29:31], v0 offset:14848 // (op 179) Transpose load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 9403) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v24 offset:12288 // (op 189) Store local data 



ds_read_b128 v[32:35], v4 // (op 181) Load local data 
ds_read_b64 v[36:37], v4 offset:16 // (op 181) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v25 offset:12292 // (op 189) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 9417) 


ds_read_u8 v16, v3 offset:29184 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9431) 



ds_read_u8 v17, v5 offset:12288 // (op 185) Load local data 



ds_read_b128 v[20:23], v4 offset:3072 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 186) 
ds_read_b64 v[24:25], v4 offset:3088 // (op 350) Load local data 



ds_read_u8 v38, v5 offset:12416 // (op 354) Load local data 



ds_read_b128 v[40:43], v4 offset:6144 // (op 526) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[20:25], a[4:7], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 355) 
ds_read_b64 v[44:45], v4 offset:6160 // (op 526) Load local data 



ds_read_u8 v39, v5 offset:12544 // (op 530) Load local data 



ds_read_b128 v[46:49], v4 offset:9216 // (op 702) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[40:45], a[8:11], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 531) 
ds_read_b64 v[50:51], v4 offset:9232 // (op 702) Load local data 



ds_read_u8 v52, v5 offset:12672 // (op 706) Load local data 



ds_read_b96_tr_b6 v[54:56], v0 offset:12832 // (op 964) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[46:51], a[12:15], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 707) 
ds_read_b96_tr_b6 v[57:59], v0 offset:14880 // (op 964) Transpose load local data 




ds_read_u8 v16, v3 offset:29216 // (op 967) Load local data 



ds_read_b96_tr_b6 v[26:28], v0 offset:12864 // (op 1734) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[32:37], a[16:19], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 969) 
ds_read_b96_tr_b6 v[29:31], v0 offset:14912 // (op 1734) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:59], v[20:25], a[20:23], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1000) 
ds_read_u8 v53, v3 offset:29248 // (op 1737) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:59], v[40:45], a[24:27], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1031) 
ds_read_b96_tr_b6 v[60:62], v0 offset:12896 // (op 2504) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:59], v[46:51], a[28:31], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1062) 
ds_read_b96_tr_b6 v[63:65], v0 offset:14944 // (op 2504) Transpose load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:31], v[32:37], a[32:35], v53, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1739) 
ds_read_u8 v16, v3 offset:29280 // (op 2507) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:31], v[20:25], a[36:39], v53, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[26:31], v[40:45], a[40:43], v53, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[26:31], v[46:51], a[44:47], v53, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1832) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:65], v[32:37], a[48:51], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2509) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[20:25], a[52:55], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2540) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:65], v[40:45], a[56:59], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2571) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:65], v[46:51], a[60:63], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2602) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 2591) 
s_mov_b32 s19, s0 // (op 2591) convert

s_cmp_lt_u32 s18, s19 // (op 2591) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2591 // (op 2591) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2591 if true)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2591:























s_barrier  // (op 9541) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 3561) 
v_accvgpr_read v7, a5 // (op 3561) 
v_accvgpr_read v8, a6 // (op 3561) 
v_accvgpr_read v9, a7 // (op 3561) 
v_mul_f32 v14, v11, v6 // (op 3561) 
v_mul_f32 v15, v11, v7 // (op 3561) 
v_mul_f32 v16, v11, v8 // (op 3561) 
v_mul_f32 v17, v11, v9 // (op 3561) 



s_lshl_b32 s0, s4, 7 // (op 3563) 
v_add_u32 v0, v13, s0 // (op 3563) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3563) Store value


v_accvgpr_read v6, a8 // (op 3573) 
v_accvgpr_read v7, a9 // (op 3573) 
v_accvgpr_read v8, a10 // (op 3573) 
v_accvgpr_read v9, a11 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 8 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a12 // (op 3585) 
v_accvgpr_read v7, a13 // (op 3585) 
v_accvgpr_read v8, a14 // (op 3585) 
v_accvgpr_read v9, a15 // (op 3585) 
v_mul_f32 v14, v11, v6 // (op 3585) 
v_mul_f32 v15, v11, v7 // (op 3585) 
v_mul_f32 v16, v11, v8 // (op 3585) 
v_mul_f32 v17, v11, v9 // (op 3585) 



s_lshl_b32 s0, s4, 7 // (op 3587) 
s_mul_i32 s1, 3, s0 // (op 3587) 

v_add_u32 v0, v13, s1 // (op 3587) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3587) Store value


v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 



v_add_u32 v0, 128, v13 // (op 3622) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3622) Store value


v_accvgpr_read v6, a20 // (op 3632) 
v_accvgpr_read v7, a21 // (op 3632) 
v_accvgpr_read v8, a22 // (op 3632) 
v_accvgpr_read v9, a23 // (op 3632) 
v_mul_f32 v14, v11, v6 // (op 3632) 
v_mul_f32 v15, v11, v7 // (op 3632) 
v_mul_f32 v16, v11, v8 // (op 3632) 
v_mul_f32 v17, v11, v9 // (op 3632) 



s_lshl_b32 s0, s4, 7 // (op 3624) 
s_add_u32 s0, s0, 128 // (op 3624) 
v_add_u32 v0, v13, s0 // (op 3624) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3624) Store value


v_accvgpr_read v6, a24 // (op 3637) 
v_accvgpr_read v7, a25 // (op 3637) 
v_accvgpr_read v8, a26 // (op 3637) 
v_accvgpr_read v9, a27 // (op 3637) 
v_mul_f32 v14, v11, v6 // (op 3637) 
v_mul_f32 v15, v11, v7 // (op 3637) 
v_mul_f32 v16, v11, v8 // (op 3637) 
v_mul_f32 v17, v11, v9 // (op 3637) 



s_lshl_b32 s0, s4, 8 // (op 3626) 
s_add_u32 s0, s0, 128 // (op 3626) 
v_add_u32 v0, v13, s0 // (op 3626) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3626) Store value


v_accvgpr_read v6, a28 // (op 3642) 
v_accvgpr_read v7, a29 // (op 3642) 
v_accvgpr_read v8, a30 // (op 3642) 
v_accvgpr_read v9, a31 // (op 3642) 
v_mul_f32 v14, v11, v6 // (op 3642) 
v_mul_f32 v15, v11, v7 // (op 3642) 
v_mul_f32 v16, v11, v8 // (op 3642) 
v_mul_f32 v17, v11, v9 // (op 3642) 



s_lshl_b32 s0, s4, 7 // (op 3628) 
s_mul_i32 s1, 3, s0 // (op 3628) 
s_add_u32 s1, s1, 128 // (op 3628) 

v_add_u32 v0, v13, s1 // (op 3628) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3628) Store value


v_accvgpr_read v6, a32 // (op 3687) 
v_accvgpr_read v7, a33 // (op 3687) 
v_accvgpr_read v8, a34 // (op 3687) 
v_accvgpr_read v9, a35 // (op 3687) 
v_mul_f32 v14, v11, v6 // (op 3687) 
v_mul_f32 v15, v11, v7 // (op 3687) 
v_mul_f32 v16, v11, v8 // (op 3687) 
v_mul_f32 v17, v11, v9 // (op 3687) 



v_add_u32 v0, 256, v13 // (op 3690) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3690) Store value


v_accvgpr_read v6, a36 // (op 3700) 
v_accvgpr_read v7, a37 // (op 3700) 
v_accvgpr_read v8, a38 // (op 3700) 
v_accvgpr_read v9, a39 // (op 3700) 
v_mul_f32 v14, v11, v6 // (op 3700) 
v_mul_f32 v15, v11, v7 // (op 3700) 
v_mul_f32 v16, v11, v8 // (op 3700) 
v_mul_f32 v17, v11, v9 // (op 3700) 



s_lshl_b32 s0, s4, 7 // (op 3692) 
s_add_u32 s0, s0, 256 // (op 3692) 
v_add_u32 v0, v13, s0 // (op 3692) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3692) Store value


v_accvgpr_read v6, a40 // (op 3705) 
v_accvgpr_read v7, a41 // (op 3705) 
v_accvgpr_read v8, a42 // (op 3705) 
v_accvgpr_read v9, a43 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 



s_lshl_b32 s0, s4, 8 // (op 3694) 
s_add_u32 s0, s0, 256 // (op 3694) 
v_add_u32 v0, v13, s0 // (op 3694) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3694) Store value


v_accvgpr_read v6, a44 // (op 3710) 
v_accvgpr_read v7, a45 // (op 3710) 
v_accvgpr_read v8, a46 // (op 3710) 
v_accvgpr_read v9, a47 // (op 3710) 
v_mul_f32 v14, v11, v6 // (op 3710) 
v_mul_f32 v15, v11, v7 // (op 3710) 
v_mul_f32 v16, v11, v8 // (op 3710) 
v_mul_f32 v17, v11, v9 // (op 3710) 



s_lshl_b32 s0, s4, 7 // (op 3696) 
s_mul_i32 s1, 3, s0 // (op 3696) 
s_add_u32 s1, s1, 256 // (op 3696) 

v_add_u32 v0, v13, s1 // (op 3696) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3696) Store value


v_accvgpr_read v6, a48 // (op 3755) 
v_accvgpr_read v7, a49 // (op 3755) 
v_accvgpr_read v8, a50 // (op 3755) 
v_accvgpr_read v9, a51 // (op 3755) 
v_mul_f32 v14, v11, v6 // (op 3755) 
v_mul_f32 v15, v11, v7 // (op 3755) 
v_mul_f32 v16, v11, v8 // (op 3755) 
v_mul_f32 v17, v11, v9 // (op 3755) 



v_add_u32 v0, 384, v13 // (op 3758) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3758) Store value


v_accvgpr_read v6, a52 // (op 3768) 
v_accvgpr_read v7, a53 // (op 3768) 
v_accvgpr_read v8, a54 // (op 3768) 
v_accvgpr_read v9, a55 // (op 3768) 
v_mul_f32 v14, v11, v6 // (op 3768) 
v_mul_f32 v15, v11, v7 // (op 3768) 
v_mul_f32 v16, v11, v8 // (op 3768) 
v_mul_f32 v17, v11, v9 // (op 3768) 



s_lshl_b32 s0, s4, 7 // (op 3760) 
s_add_u32 s0, s0, 384 // (op 3760) 
v_add_u32 v0, v13, s0 // (op 3760) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3760) Store value


v_accvgpr_read v6, a56 // (op 3773) 
v_accvgpr_read v7, a57 // (op 3773) 
v_accvgpr_read v8, a58 // (op 3773) 
v_accvgpr_read v9, a59 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 



s_lshl_b32 s0, s4, 8 // (op 3762) 
s_add_u32 s0, s0, 384 // (op 3762) 
v_add_u32 v0, v13, s0 // (op 3762) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3762) Store value


v_accvgpr_read v6, a60 // (op 3778) 
v_accvgpr_read v7, a61 // (op 3778) 
v_accvgpr_read v8, a62 // (op 3778) 
v_accvgpr_read v9, a63 // (op 3778) 
v_mul_f32 v14, v11, v6 // (op 3778) 
v_mul_f32 v15, v11, v7 // (op 3778) 
v_mul_f32 v16, v11, v8 // (op 3778) 
v_mul_f32 v17, v11, v9 // (op 3778) 



s_lshl_b32 s0, s4, 7 // (op 3764) 
s_mul_i32 s1, 3, s0 // (op 3764) 
s_add_u32 s1, s1, 384 // (op 3764) 

v_add_u32 v0, v13, s1 // (op 3764) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3764) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 384, v12 // (op 4122) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4122) Load value

v_accvgpr_read v14, a48 // (op 4110) 
v_accvgpr_read v15, a49 // (op 4110) 
v_accvgpr_read v16, a50 // (op 4110) 
v_accvgpr_read v17, a51 // (op 4110) 
v_mul_f32 v18, v11, v14 // (op 4110) 
v_mul_f32 v19, v11, v15 // (op 4110) 
v_mul_f32 v20, v11, v16 // (op 4110) 
v_mul_f32 v21, v11, v17 // (op 4110) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4121) 
v_mul_f32 v15, v10, v7 // (op 4121) 
v_mul_f32 v16, v10, v8 // (op 4121) 
v_mul_f32 v17, v10, v9 // (op 4121) 

v_add_f32 v6, v14, v18 // (op 4111) 
v_add_f32 v7, v15, v19 // (op 4111) 
v_add_f32 v8, v16, v20 // (op 4111) 
v_add_f32 v9, v17, v21 // (op 4111) 




v_add_u32 v0, 384, v13 // (op 4113) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4113) Store value




s_lshl_b32 s0, s2, 7 // (op 4126) 
s_add_u32 s0, s0, 384 // (op 4126) 
v_add_u32 v0, v12, s0 // (op 4126) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4126) Load value

v_accvgpr_read v14, a52 // (op 4136) 
v_accvgpr_read v15, a53 // (op 4136) 
v_accvgpr_read v16, a54 // (op 4136) 
v_accvgpr_read v17, a55 // (op 4136) 
v_mul_f32 v18, v11, v14 // (op 4136) 
v_mul_f32 v19, v11, v15 // (op 4136) 
v_mul_f32 v20, v11, v16 // (op 4136) 
v_mul_f32 v21, v11, v17 // (op 4136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4124) 
v_mul_f32 v15, v10, v7 // (op 4124) 
v_mul_f32 v16, v10, v8 // (op 4124) 
v_mul_f32 v17, v10, v9 // (op 4124) 

v_add_f32 v6, v14, v18 // (op 4125) 
v_add_f32 v7, v15, v19 // (op 4125) 
v_add_f32 v8, v16, v20 // (op 4125) 
v_add_f32 v9, v17, v21 // (op 4125) 




s_lshl_b32 s0, s4, 7 // (op 4115) 
s_add_u32 s0, s0, 384 // (op 4115) 
v_add_u32 v0, v13, s0 // (op 4115) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4115) Store value




s_lshl_b32 s0, s2, 8 // (op 4130) 
s_add_u32 s0, s0, 384 // (op 4130) 
v_add_u32 v0, v12, s0 // (op 4130) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4130) Load value

v_accvgpr_read v14, a56 // (op 4138) 
v_accvgpr_read v15, a57 // (op 4138) 
v_accvgpr_read v16, a58 // (op 4138) 
v_accvgpr_read v17, a59 // (op 4138) 
v_mul_f32 v18, v11, v14 // (op 4138) 
v_mul_f32 v19, v11, v15 // (op 4138) 
v_mul_f32 v20, v11, v16 // (op 4138) 
v_mul_f32 v21, v11, v17 // (op 4138) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4128) 
v_mul_f32 v15, v10, v7 // (op 4128) 
v_mul_f32 v16, v10, v8 // (op 4128) 
v_mul_f32 v17, v10, v9 // (op 4128) 

v_add_f32 v6, v14, v18 // (op 4129) 
v_add_f32 v7, v15, v19 // (op 4129) 
v_add_f32 v8, v16, v20 // (op 4129) 
v_add_f32 v9, v17, v21 // (op 4129) 




s_lshl_b32 s0, s4, 8 // (op 4117) 
s_add_u32 s0, s0, 384 // (op 4117) 
v_add_u32 v0, v13, s0 // (op 4117) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4117) Store value




s_lshl_b32 s0, s2, 7 // (op 4134) 
s_mul_i32 s1, 3, s0 // (op 4134) 
s_add_u32 s1, s1, 384 // (op 4134) 

v_add_u32 v0, v12, s1 // (op 4134) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4134) Load value

v_accvgpr_read v14, a60 // (op 4140) 
v_accvgpr_read v15, a61 // (op 4140) 
v_accvgpr_read v16, a62 // (op 4140) 
v_accvgpr_read v17, a63 // (op 4140) 
v_mul_f32 v18, v11, v14 // (op 4140) 
v_mul_f32 v19, v11, v15 // (op 4140) 
v_mul_f32 v20, v11, v16 // (op 4140) 
v_mul_f32 v21, v11, v17 // (op 4140) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4132) 
v_mul_f32 v15, v10, v7 // (op 4132) 
v_mul_f32 v16, v10, v8 // (op 4132) 
v_mul_f32 v17, v10, v9 // (op 4132) 

v_add_f32 v6, v14, v18 // (op 4133) 
v_add_f32 v7, v15, v19 // (op 4133) 
v_add_f32 v8, v16, v20 // (op 4133) 
v_add_f32 v9, v17, v21 // (op 4133) 




s_lshl_b32 s0, s4, 7 // (op 4119) 
s_mul_i32 s1, 3, s0 // (op 4119) 
s_add_u32 s1, s1, 384 // (op 4119) 

v_add_u32 v0, v13, s1 // (op 4119) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4119) Store value




v_add_u32 v0, 256, v12 // (op 4043) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4043) Load value

v_accvgpr_read v14, a32 // (op 4031) 
v_accvgpr_read v15, a33 // (op 4031) 
v_accvgpr_read v16, a34 // (op 4031) 
v_accvgpr_read v17, a35 // (op 4031) 
v_mul_f32 v18, v11, v14 // (op 4031) 
v_mul_f32 v19, v11, v15 // (op 4031) 
v_mul_f32 v20, v11, v16 // (op 4031) 
v_mul_f32 v21, v11, v17 // (op 4031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4042) 
v_mul_f32 v15, v10, v7 // (op 4042) 
v_mul_f32 v16, v10, v8 // (op 4042) 
v_mul_f32 v17, v10, v9 // (op 4042) 

v_add_f32 v6, v14, v18 // (op 4032) 
v_add_f32 v7, v15, v19 // (op 4032) 
v_add_f32 v8, v16, v20 // (op 4032) 
v_add_f32 v9, v17, v21 // (op 4032) 




v_add_u32 v0, 256, v13 // (op 4034) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4034) Store value




s_lshl_b32 s0, s2, 7 // (op 4047) 
s_add_u32 s0, s0, 256 // (op 4047) 
v_add_u32 v0, v12, s0 // (op 4047) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4047) Load value

v_accvgpr_read v14, a36 // (op 4057) 
v_accvgpr_read v15, a37 // (op 4057) 
v_accvgpr_read v16, a38 // (op 4057) 
v_accvgpr_read v17, a39 // (op 4057) 
v_mul_f32 v18, v11, v14 // (op 4057) 
v_mul_f32 v19, v11, v15 // (op 4057) 
v_mul_f32 v20, v11, v16 // (op 4057) 
v_mul_f32 v21, v11, v17 // (op 4057) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4045) 
v_mul_f32 v15, v10, v7 // (op 4045) 
v_mul_f32 v16, v10, v8 // (op 4045) 
v_mul_f32 v17, v10, v9 // (op 4045) 

v_add_f32 v6, v14, v18 // (op 4046) 
v_add_f32 v7, v15, v19 // (op 4046) 
v_add_f32 v8, v16, v20 // (op 4046) 
v_add_f32 v9, v17, v21 // (op 4046) 




s_lshl_b32 s0, s4, 7 // (op 4036) 
s_add_u32 s0, s0, 256 // (op 4036) 
v_add_u32 v0, v13, s0 // (op 4036) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4036) Store value




s_lshl_b32 s0, s2, 8 // (op 4051) 
s_add_u32 s0, s0, 256 // (op 4051) 
v_add_u32 v0, v12, s0 // (op 4051) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4051) Load value

v_accvgpr_read v14, a40 // (op 4059) 
v_accvgpr_read v15, a41 // (op 4059) 
v_accvgpr_read v16, a42 // (op 4059) 
v_accvgpr_read v17, a43 // (op 4059) 
v_mul_f32 v18, v11, v14 // (op 4059) 
v_mul_f32 v19, v11, v15 // (op 4059) 
v_mul_f32 v20, v11, v16 // (op 4059) 
v_mul_f32 v21, v11, v17 // (op 4059) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4049) 
v_mul_f32 v15, v10, v7 // (op 4049) 
v_mul_f32 v16, v10, v8 // (op 4049) 
v_mul_f32 v17, v10, v9 // (op 4049) 

v_add_f32 v6, v14, v18 // (op 4050) 
v_add_f32 v7, v15, v19 // (op 4050) 
v_add_f32 v8, v16, v20 // (op 4050) 
v_add_f32 v9, v17, v21 // (op 4050) 




s_lshl_b32 s0, s4, 8 // (op 4038) 
s_add_u32 s0, s0, 256 // (op 4038) 
v_add_u32 v0, v13, s0 // (op 4038) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4038) Store value




s_lshl_b32 s0, s2, 7 // (op 4055) 
s_mul_i32 s1, 3, s0 // (op 4055) 
s_add_u32 s1, s1, 256 // (op 4055) 

v_add_u32 v0, v12, s1 // (op 4055) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4055) Load value

v_accvgpr_read v14, a44 // (op 4061) 
v_accvgpr_read v15, a45 // (op 4061) 
v_accvgpr_read v16, a46 // (op 4061) 
v_accvgpr_read v17, a47 // (op 4061) 
v_mul_f32 v18, v11, v14 // (op 4061) 
v_mul_f32 v19, v11, v15 // (op 4061) 
v_mul_f32 v20, v11, v16 // (op 4061) 
v_mul_f32 v21, v11, v17 // (op 4061) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4053) 
v_mul_f32 v15, v10, v7 // (op 4053) 
v_mul_f32 v16, v10, v8 // (op 4053) 
v_mul_f32 v17, v10, v9 // (op 4053) 

v_add_f32 v6, v14, v18 // (op 4054) 
v_add_f32 v7, v15, v19 // (op 4054) 
v_add_f32 v8, v16, v20 // (op 4054) 
v_add_f32 v9, v17, v21 // (op 4054) 




s_lshl_b32 s0, s4, 7 // (op 4040) 
s_mul_i32 s1, 3, s0 // (op 4040) 
s_add_u32 s1, s1, 256 // (op 4040) 

v_add_u32 v0, v13, s1 // (op 4040) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4040) Store value




v_add_u32 v0, 128, v12 // (op 3964) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3964) Load value

v_accvgpr_read v14, a16 // (op 3952) 
v_accvgpr_read v15, a17 // (op 3952) 
v_accvgpr_read v16, a18 // (op 3952) 
v_accvgpr_read v17, a19 // (op 3952) 
v_mul_f32 v18, v11, v14 // (op 3952) 
v_mul_f32 v19, v11, v15 // (op 3952) 
v_mul_f32 v20, v11, v16 // (op 3952) 
v_mul_f32 v21, v11, v17 // (op 3952) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3963) 
v_mul_f32 v15, v10, v7 // (op 3963) 
v_mul_f32 v16, v10, v8 // (op 3963) 
v_mul_f32 v17, v10, v9 // (op 3963) 

v_add_f32 v6, v14, v18 // (op 3953) 
v_add_f32 v7, v15, v19 // (op 3953) 
v_add_f32 v8, v16, v20 // (op 3953) 
v_add_f32 v9, v17, v21 // (op 3953) 




v_add_u32 v0, 128, v13 // (op 3955) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3955) Store value




s_lshl_b32 s0, s2, 7 // (op 3968) 
s_add_u32 s0, s0, 128 // (op 3968) 
v_add_u32 v0, v12, s0 // (op 3968) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3968) Load value

v_accvgpr_read v14, a20 // (op 3978) 
v_accvgpr_read v15, a21 // (op 3978) 
v_accvgpr_read v16, a22 // (op 3978) 
v_accvgpr_read v17, a23 // (op 3978) 
v_mul_f32 v18, v11, v14 // (op 3978) 
v_mul_f32 v19, v11, v15 // (op 3978) 
v_mul_f32 v20, v11, v16 // (op 3978) 
v_mul_f32 v21, v11, v17 // (op 3978) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3966) 
v_mul_f32 v15, v10, v7 // (op 3966) 
v_mul_f32 v16, v10, v8 // (op 3966) 
v_mul_f32 v17, v10, v9 // (op 3966) 

v_add_f32 v6, v14, v18 // (op 3967) 
v_add_f32 v7, v15, v19 // (op 3967) 
v_add_f32 v8, v16, v20 // (op 3967) 
v_add_f32 v9, v17, v21 // (op 3967) 




s_lshl_b32 s0, s4, 7 // (op 3957) 
s_add_u32 s0, s0, 128 // (op 3957) 
v_add_u32 v0, v13, s0 // (op 3957) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3957) Store value




s_lshl_b32 s0, s2, 8 // (op 3972) 
s_add_u32 s0, s0, 128 // (op 3972) 
v_add_u32 v0, v12, s0 // (op 3972) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3972) Load value

v_accvgpr_read v14, a24 // (op 3980) 
v_accvgpr_read v15, a25 // (op 3980) 
v_accvgpr_read v16, a26 // (op 3980) 
v_accvgpr_read v17, a27 // (op 3980) 
v_mul_f32 v18, v11, v14 // (op 3980) 
v_mul_f32 v19, v11, v15 // (op 3980) 
v_mul_f32 v20, v11, v16 // (op 3980) 
v_mul_f32 v21, v11, v17 // (op 3980) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3970) 
v_mul_f32 v15, v10, v7 // (op 3970) 
v_mul_f32 v16, v10, v8 // (op 3970) 
v_mul_f32 v17, v10, v9 // (op 3970) 

v_add_f32 v6, v14, v18 // (op 3971) 
v_add_f32 v7, v15, v19 // (op 3971) 
v_add_f32 v8, v16, v20 // (op 3971) 
v_add_f32 v9, v17, v21 // (op 3971) 




s_lshl_b32 s0, s4, 8 // (op 3959) 
s_add_u32 s0, s0, 128 // (op 3959) 
v_add_u32 v0, v13, s0 // (op 3959) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3959) Store value




s_lshl_b32 s0, s2, 7 // (op 3976) 
s_mul_i32 s1, 3, s0 // (op 3976) 
s_add_u32 s1, s1, 128 // (op 3976) 

v_add_u32 v0, v12, s1 // (op 3976) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3976) Load value

v_accvgpr_read v14, a28 // (op 3982) 
v_accvgpr_read v15, a29 // (op 3982) 
v_accvgpr_read v16, a30 // (op 3982) 
v_accvgpr_read v17, a31 // (op 3982) 
v_mul_f32 v18, v11, v14 // (op 3982) 
v_mul_f32 v19, v11, v15 // (op 3982) 
v_mul_f32 v20, v11, v16 // (op 3982) 
v_mul_f32 v21, v11, v17 // (op 3982) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3974) 
v_mul_f32 v15, v10, v7 // (op 3974) 
v_mul_f32 v16, v10, v8 // (op 3974) 
v_mul_f32 v17, v10, v9 // (op 3974) 

v_add_f32 v6, v14, v18 // (op 3975) 
v_add_f32 v7, v15, v19 // (op 3975) 
v_add_f32 v8, v16, v20 // (op 3975) 
v_add_f32 v9, v17, v21 // (op 3975) 




s_lshl_b32 s0, s4, 7 // (op 3961) 
s_mul_i32 s1, 3, s0 // (op 3961) 
s_add_u32 s1, s1, 128 // (op 3961) 

v_add_u32 v0, v13, s1 // (op 3961) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3961) Store value




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




s_lshl_b32 s0, s2, 7 // (op 3878) 
v_add_u32 v0, v12, s0 // (op 3878) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3878) Load value

v_accvgpr_read v14, a4 // (op 3875) 
v_accvgpr_read v15, a5 // (op 3875) 
v_accvgpr_read v16, a6 // (op 3875) 
v_accvgpr_read v17, a7 // (op 3875) 
v_mul_f32 v18, v11, v14 // (op 3875) 
v_mul_f32 v19, v11, v15 // (op 3875) 
v_mul_f32 v20, v11, v16 // (op 3875) 
v_mul_f32 v21, v11, v17 // (op 3875) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3879) 
v_mul_f32 v15, v10, v7 // (op 3879) 
v_mul_f32 v16, v10, v8 // (op 3879) 
v_mul_f32 v17, v10, v9 // (op 3879) 

v_add_f32 v6, v14, v18 // (op 3876) 
v_add_f32 v7, v15, v19 // (op 3876) 
v_add_f32 v8, v16, v20 // (op 3876) 
v_add_f32 v9, v17, v21 // (op 3876) 




s_lshl_b32 s0, s4, 7 // (op 3877) 
v_add_u32 v0, v13, s0 // (op 3877) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3877) Store value




s_lshl_b32 s0, s2, 8 // (op 3890) 
v_add_u32 v0, v12, s0 // (op 3890) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3890) Load value

v_accvgpr_read v14, a8 // (op 3887) 
v_accvgpr_read v15, a9 // (op 3887) 
v_accvgpr_read v16, a10 // (op 3887) 
v_accvgpr_read v17, a11 // (op 3887) 
v_mul_f32 v18, v11, v14 // (op 3887) 
v_mul_f32 v19, v11, v15 // (op 3887) 
v_mul_f32 v20, v11, v16 // (op 3887) 
v_mul_f32 v21, v11, v17 // (op 3887) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3891) 
v_mul_f32 v15, v10, v7 // (op 3891) 
v_mul_f32 v16, v10, v8 // (op 3891) 
v_mul_f32 v17, v10, v9 // (op 3891) 

v_add_f32 v6, v14, v18 // (op 3888) 
v_add_f32 v7, v15, v19 // (op 3888) 
v_add_f32 v8, v16, v20 // (op 3888) 
v_add_f32 v9, v17, v21 // (op 3888) 




s_lshl_b32 s0, s4, 8 // (op 3889) 
v_add_u32 v0, v13, s0 // (op 3889) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3889) Store value




s_lshl_b32 s0, s2, 7 // (op 3902) 
s_mul_i32 s1, 3, s0 // (op 3902) 

v_add_u32 v0, v12, s1 // (op 3902) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3902) Load value

v_accvgpr_read v14, a12 // (op 3899) 
v_accvgpr_read v15, a13 // (op 3899) 
v_accvgpr_read v16, a14 // (op 3899) 
v_accvgpr_read v17, a15 // (op 3899) 
v_mul_f32 v18, v11, v14 // (op 3899) 
v_mul_f32 v19, v11, v15 // (op 3899) 
v_mul_f32 v20, v11, v16 // (op 3899) 
v_mul_f32 v21, v11, v17 // (op 3899) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3903) 
v_mul_f32 v15, v10, v7 // (op 3903) 
v_mul_f32 v16, v10, v8 // (op 3903) 
v_mul_f32 v17, v10, v9 // (op 3903) 

v_add_f32 v6, v14, v18 // (op 3900) 
v_add_f32 v7, v15, v19 // (op 3900) 
v_add_f32 v8, v16, v20 // (op 3900) 
v_add_f32 v9, v17, v21 // (op 3900) 




s_lshl_b32 s0, s4, 7 // (op 3901) 
s_mul_i32 s1, 3, s0 // (op 3901) 

v_add_u32 v0, v13, s1 // (op 3901) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3901) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 200
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 59392 // lds bytes
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F6
    DestDataType: s
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 59392
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 136
    .agpr_count: 64
    .max_flat_workgroup_size: 256
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

