// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_exec_begin:


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
v_and_b32 v0, 48, v1 // (op 8980) 
v_and_b32 v3, 63, v1 // (op 8980) 
v_sub_u32 v4, v3, v0 // (op 8980) 

v_lshrrev_b32 v3, 1, v4 // (op 8980) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 8980) 


v_and_b32 v0, 1, v4 // (op 8980) 

v_lshlrev_b32 v3, 3, v0 // (op 8980) 

v_lshrrev_b32 v0, 7, v1 // (op 8980) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8980) 


v_add_u32 v0, v5, v4 // (op 8980) 


v_lshrrev_b32 v3, 6, v1 // (op 9038) 
v_and_b32 v4, 1, v3 // (op 9038) 

v_and_b32 v3, 15, v1 // (op 9038) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 9038) 


v_and_b32 v3, 63, v1 // (op 9038) 
v_lshrrev_b32 v4, 4, v3 // (op 9038) 

v_lshlrev_b32 v3, 5, v4 // (op 9038) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 9038) 


v_lshlrev_b32 v3, 2, v4 // (op 9038) 

v_lshrrev_b32 v4, 3, v3 // (op 9038) 

v_lshrrev_b32 v3, 7, v1 // (op 9096) 
v_and_b32 v5, 15, v1 // (op 9096) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 9096) 


v_and_b32 v3, 63, v1 // (op 9096) 
v_lshrrev_b32 v5, 4, v3 // (op 9096) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 9096) 


v_lshrrev_b32 v5, 6, v1 // (op 9154) 
v_and_b32 v6, 1, v5 // (op 9154) 

v_and_b32 v5, 15, v1 // (op 9154) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9154) 


v_and_b32 v5, 63, v1 // (op 9154) 
v_lshrrev_b32 v6, 4, v5 // (op 9154) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9154) 


v_and_b32 v6, 48, v1 // (op 9206) 
v_and_b32 v7, 63, v1 // (op 9206) 
v_sub_u32 v8, v7, v6 // (op 9206) 

v_lshrrev_b32 v7, 1, v8 // (op 9206) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 9206) 


v_and_b32 v6, 1, v8 // (op 9206) 

v_lshlrev_b32 v7, 3, v6 // (op 9206) 

v_lshrrev_b32 v6, 7, v1 // (op 9206) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9206) 


v_add_u32 v6, v9, v8 // (op 9206) 


v_lshrrev_b32 v7, 6, v1 // (op 9235) 
v_and_b32 v8, 1, v7 // (op 9235) 

v_and_b32 v7, 15, v1 // (op 9235) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 9235) 


v_and_b32 v7, 63, v1 // (op 9235) 
v_lshrrev_b32 v8, 4, v7 // (op 9235) 

v_lshlrev_b32 v7, 5, v8 // (op 9235) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 9235) 


v_lshlrev_b32 v7, 2, v8 // (op 9235) 

v_lshrrev_b32 v8, 3, v7 // (op 9235) 

v_lshrrev_b32 v7, 7, v1 // (op 9264) 
v_and_b32 v9, 15, v1 // (op 9264) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 9264) 


v_and_b32 v7, 63, v1 // (op 9264) 
v_lshrrev_b32 v9, 4, v7 // (op 9264) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 9264) 


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


s_add_u32 s58, s6, 256 // (op 8918) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 8918) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 8918) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 8918) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 8918) 

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
s_add_u32 s62, s20, 64 // (op 8918) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 8918) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 8918) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 8918) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 8918) 

s_sub_i32 s57, s62, s52 // (op 8918) 

s_mul_i32 s61, s60, s57 // (op 8918) 

s_sub_u32 s57, s55, s61 // (op 8918) 


s_add_u32 s55, s57, s52 // (op 8918) 


s_cmp_eq_i32 s53, 0 // (op 8918) 
s_mov_b32 s52, scc // (op 8918) 

s_cmp_eq_i32 s52, 0 // (op 8918) 
s_cselect_b32 s53, s55, s19 // (op 8918) 


s_mul_i32 s19, s51, s58 // (op 8918) 


v_lshl_add_u32 v13, s53, 6, v12 // (op 8918) 


s_sub_i32 s51, s18, s19 // (op 8918) 


s_cmp_eq_i32 s52, 0 // (op 8918) 
s_cselect_b32 s53, s60, s51 // (op 8918) 



v_lshl_add_u32 v12, s53, 8, v11 // (op 8918) 


v_mul_lo_u32 v11, v12, s2 // (op 8918) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 8918) 


s_add_u32 s18, s6, 256 // (op 9175) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 9175) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 9175) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 9175) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 8 // (op 9175) 

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
s_add_u32 s58, s20, 64 // (op 9175) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 9175) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 9175) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 9175) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 9175) 

s_sub_i32 s55, s58, s52 // (op 9175) 

s_mul_i32 s57, s53, s55 // (op 9175) 

s_sub_u32 s55, s51, s57 // (op 9175) 


s_add_u32 s51, s55, s52 // (op 9175) 


s_cmp_eq_i32 s56, 0 // (op 9175) 
s_mov_b32 s52, scc // (op 9175) 

s_cmp_eq_i32 s52, 0 // (op 9175) 
s_cselect_b32 s55, s51, s1 // (op 9175) 


s_mul_i32 s1, s54, s18 // (op 9175) 


v_lshl_add_u32 v11, s55, 6, v14 // (op 9175) 


s_sub_i32 s51, s0, s1 // (op 9175) 


s_cmp_eq_i32 s52, 0 // (op 9175) 
s_cselect_b32 s0, s53, s51 // (op 9175) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 9175) 


v_mul_lo_u32 v10, v13, s4 // (op 9175) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 9175) 


s_mul_i32 s1, s2, s7 // (op 8920) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8920) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8920) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8920) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8920) most significant: sum
s_add_u32 s1, s1, s19 // (op 8920) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8920) 

s_mul_i32 s1, s4, s7 // (op 9177) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 9177) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9177) least significant
s_mul_hi_u32 s52, s4, s6 // (op 9177) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 9177) most significant: sum
s_add_u32 s1, s1, s52 // (op 9177) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 9177) 

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


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 516) call()
v_accvgpr_write a9, 0.00000 // (op 516) call()
v_accvgpr_write a10, 0.00000 // (op 516) call()
v_accvgpr_write a11, 0.00000 // (op 516) call()
v_accvgpr_write a12, 0.00000 // (op 692) call()
v_accvgpr_write a13, 0.00000 // (op 692) call()
v_accvgpr_write a14, 0.00000 // (op 692) call()
v_accvgpr_write a15, 0.00000 // (op 692) call()
v_accvgpr_write a16, 0.00000 // (op 868) call()
v_accvgpr_write a17, 0.00000 // (op 868) call()
v_accvgpr_write a18, 0.00000 // (op 868) call()
v_accvgpr_write a19, 0.00000 // (op 868) call()
v_accvgpr_write a20, 0.00000 // (op 1044) call()
v_accvgpr_write a21, 0.00000 // (op 1044) call()
v_accvgpr_write a22, 0.00000 // (op 1044) call()
v_accvgpr_write a23, 0.00000 // (op 1044) call()
v_accvgpr_write a24, 0.00000 // (op 1220) call()
v_accvgpr_write a25, 0.00000 // (op 1220) call()
v_accvgpr_write a26, 0.00000 // (op 1220) call()
v_accvgpr_write a27, 0.00000 // (op 1220) call()
v_accvgpr_write a28, 0.00000 // (op 1396) call()
v_accvgpr_write a29, 0.00000 // (op 1396) call()
v_accvgpr_write a30, 0.00000 // (op 1396) call()
v_accvgpr_write a31, 0.00000 // (op 1396) call()
v_accvgpr_write a32, 0.00000 // (op 1748) call()
v_accvgpr_write a33, 0.00000 // (op 1748) call()
v_accvgpr_write a34, 0.00000 // (op 1748) call()
v_accvgpr_write a35, 0.00000 // (op 1748) call()
v_accvgpr_write a36, 0.00000 // (op 2365) call()
v_accvgpr_write a37, 0.00000 // (op 2365) call()
v_accvgpr_write a38, 0.00000 // (op 2365) call()
v_accvgpr_write a39, 0.00000 // (op 2365) call()
v_accvgpr_write a40, 0.00000 // (op 2366) call()
v_accvgpr_write a41, 0.00000 // (op 2366) call()
v_accvgpr_write a42, 0.00000 // (op 2366) call()
v_accvgpr_write a43, 0.00000 // (op 2366) call()
v_accvgpr_write a44, 0.00000 // (op 2367) call()
v_accvgpr_write a45, 0.00000 // (op 2367) call()
v_accvgpr_write a46, 0.00000 // (op 2367) call()
v_accvgpr_write a47, 0.00000 // (op 2367) call()
v_accvgpr_write a48, 0.00000 // (op 2368) call()
v_accvgpr_write a49, 0.00000 // (op 2368) call()
v_accvgpr_write a50, 0.00000 // (op 2368) call()
v_accvgpr_write a51, 0.00000 // (op 2368) call()
v_accvgpr_write a52, 0.00000 // (op 2369) call()
v_accvgpr_write a53, 0.00000 // (op 2369) call()
v_accvgpr_write a54, 0.00000 // (op 2369) call()
v_accvgpr_write a55, 0.00000 // (op 2369) call()
v_accvgpr_write a56, 0.00000 // (op 2370) call()
v_accvgpr_write a57, 0.00000 // (op 2370) call()
v_accvgpr_write a58, 0.00000 // (op 2370) call()
v_accvgpr_write a59, 0.00000 // (op 2370) call()
v_accvgpr_write a60, 0.00000 // (op 2371) call()
v_accvgpr_write a61, 0.00000 // (op 2371) call()
v_accvgpr_write a62, 0.00000 // (op 2371) call()
v_accvgpr_write a63, 0.00000 // (op 2371) call()
s_mul_hi_i32 s0, s16, s10 // (op 8680) 
v_and_b32 v14, -4, v1 // (op 8680) 
s_add_i32 s1, s0, s16 // (op 8680) 

v_sub_u32 v15, v1, v14 // (op 8680) 

s_ashr_i32 s0, s1, 31 // (op 8680) 
v_lshlrev_b32 v14, 4, v15 // (op 8680) 

s_cmp_eq_i32 s10, 0 // (op 8680) 
s_mov_b32 s18, scc // (op 8680) 
v_lshrrev_b32 v15, 2, v1 // (op 8680) 
s_cmp_eq_i32 s18, 0 // (op 8680) 
s_cselect_b32 s19, 0, -1 // (op 8680) 

v_lshlrev_b32 v16, 1, v15 // (op 8680) 

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


s_add_u32 s22, s6, 256 // (op 8680) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8680) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8680) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8680) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8680) 

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
s_add_u32 s24, s20, 64 // (op 8680) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8680) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 8680) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 8680) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 8680) 

s_sub_i32 s22, s24, s1 // (op 8680) 

s_mul_i32 s24, s23, s22 // (op 8680) 


s_sub_u32 s25, s19, s24 // (op 8680) 


s_add_u32 s19, s25, s1 // (op 8680) 


s_cmp_eq_i32 s18, 0 // (op 8680) 
s_mov_b32 s1, scc // (op 8680) 

s_cmp_eq_i32 s1, 0 // (op 8680) 
s_cselect_b32 s18, s19, s0 // (op 8680) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 8680) 


v_mul_lo_u32 v14, v16, s26 // (op 8680) 

v_add_u32 v16, v15, v14 // (op 8680) 


s_mul_i32 s1, s26, s31 // (op 8682) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 8682) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 8682) least significant
s_mul_hi_u32 s19, s26, s30 // (op 8682) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8682) most significant: sum
s_add_u32 s1, s1, s19 // (op 8682) most significant: sum


s_mov_b32 s52, s28 // (op 8682) reg expression
s_mov_b32 s53, s29 // (op 8682) reg expression
s_mov_b32 s54, s0 // (op 8682) reg expression
s_mov_b32 s55, 131072 // (op 8682) call()

s_mul_hi_i32 s0, s16, s10 // (op 8699) 
v_and_b32 v14, -4, v1 // (op 8699) 
s_add_i32 s1, s0, s16 // (op 8699) 

v_sub_u32 v15, v1, v14 // (op 8699) 

s_ashr_i32 s0, s1, 31 // (op 8699) 
v_lshlrev_b32 v14, 4, v15 // (op 8699) 

s_cmp_eq_i32 s10, 0 // (op 8699) 
s_mov_b32 s18, scc // (op 8699) 
v_lshrrev_b32 v15, 2, v1 // (op 8699) 
s_cmp_eq_i32 s18, 0 // (op 8699) 
s_cselect_b32 s19, 0, -1 // (op 8699) 

v_lshlrev_b32 v17, 1, v15 // (op 8699) 

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


s_add_u32 s50, s6, 256 // (op 8699) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8699) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8699) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8699) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8699) 

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
s_add_u32 s24, s20, 64 // (op 8699) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8699) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8699) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8699) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8699) 

s_sub_i32 s22, s24, s1 // (op 8699) 

s_mul_i32 s24, s23, s22 // (op 8699) 


s_sub_u32 s25, s19, s24 // (op 8699) 


s_add_u32 s19, s25, s1 // (op 8699) 


s_cmp_eq_i32 s18, 0 // (op 8699) 
s_mov_b32 s1, scc // (op 8699) 

s_cmp_eq_i32 s1, 0 // (op 8699) 
s_cselect_b32 s18, s19, s0 // (op 8699) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 8699) 


v_mul_lo_u32 v14, v17, s26 // (op 8699) 

v_add_u32 v17, v15, v14 // (op 8699) 


s_mul_i32 s1, s26, s31 // (op 8701) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 8701) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 8701) least significant
s_mul_hi_u32 s19, s26, s30 // (op 8701) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8701) most significant: sum
s_add_u32 s1, s1, s19 // (op 8701) most significant: sum


s_mov_b32 s52, s28 // (op 8701) reg expression
s_mov_b32 s53, s29 // (op 8701) reg expression
s_mov_b32 s54, s0 // (op 8701) reg expression
s_mov_b32 s55, 131072 // (op 8701) call()

s_mul_hi_i32 s0, s16, s10 // (op 8744) 
v_and_b32 v14, 4294967292, v1 // (op 8744) 
s_add_i32 s1, s0, s16 // (op 8744) 

v_and_b32 v15, -4, v1 // (op 8744) 
s_ashr_i32 s0, s1, 31 // (op 8744) 
v_sub_u32 v18, v1, v15 // (op 8744) 

s_cmp_eq_i32 s10, 0 // (op 8744) 
s_mov_b32 s18, scc // (op 8744) 
v_lshlrev_b32 v15, 5, v18 // (op 8744) 

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


s_add_u32 s50, s6, 256 // (op 8744) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8744) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8744) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8744) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8744) 

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



v_lshl_add_u32 v15, s0, 8, v14 // (op 8744) 


v_mul_lo_u32 v21, v15, s33 // (op 8744) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 8744) most significant: high of low * low
v_mul_lo_u32 v20, v15, s32 // (op 8744) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8744) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 8744) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 8744) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 8744) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 8744) 

v_mov_b32 v14, v18 // (op 8744) convert

s_mul_i32 s1, s32, s7 // (op 8746) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 8746) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 8746) least significant
s_mul_hi_u32 s19, s32, s6 // (op 8746) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8746) most significant: sum
s_add_u32 s1, s1, s19 // (op 8746) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8746) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8746) 

s_mov_b32 s64, s34 // (op 8746) reg expression
s_mov_b32 s65, s35 // (op 8746) reg expression
s_mov_b32 s66, s0 // (op 8746) reg expression
s_mov_b32 s67, 131072 // (op 8746) call()

s_mul_hi_i32 s0, s16, s10 // (op 8763) 
v_and_b32 v15, 4294967292, v1 // (op 8763) 
s_add_i32 s1, s0, s16 // (op 8763) 

v_and_b32 v18, -4, v1 // (op 8763) 
s_ashr_i32 s0, s1, 31 // (op 8763) 
v_sub_u32 v19, v1, v18 // (op 8763) 

s_cmp_eq_i32 s10, 0 // (op 8763) 
s_mov_b32 s18, scc // (op 8763) 
v_lshlrev_b32 v18, 5, v19 // (op 8763) 

s_cmp_eq_i32 s18, 0 // (op 8763) 
s_cselect_b32 s19, 0, -1 // (op 8763) 

v_mov_b32 v21, 0 // (op 8763) convert
v_mov_b32 v20, v18 // (op 8763) convert

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


s_add_u32 s50, s6, 256 // (op 8763) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8763) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8763) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8763) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8763) 

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



v_lshl_add_u32 v18, s0, 8, v15 // (op 8763) 


v_mul_lo_u32 v23, v18, s33 // (op 8763) most significant: low of low * high
v_mul_hi_u32 v15, v18, s32 // (op 8763) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 8763) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 8763) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 8763) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 8763) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 8763) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 8763) 

v_mov_b32 v15, v20 // (op 8763) convert

s_mul_i32 s1, s32, s7 // (op 8765) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 8765) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 8765) least significant
s_mul_hi_u32 s19, s32, s6 // (op 8765) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8765) most significant: sum
s_add_u32 s1, s1, s19 // (op 8765) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8765) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8765) 

s_mov_b32 s64, s34 // (op 8765) reg expression
s_mov_b32 s65, s35 // (op 8765) reg expression
s_mov_b32 s66, s0 // (op 8765) reg expression
s_mov_b32 s67, 131072 // (op 8765) call()

s_mul_hi_i32 s0, s16, s10 // (op 8808) 
v_and_b32 v18, -64, v1 // (op 8808) 
s_add_i32 s1, s0, s16 // (op 8808) 

v_sub_u32 v19, v1, v18 // (op 8808) 

s_ashr_i32 s0, s1, 31 // (op 8808) 
v_lshrrev_b32 v18, 6, v1 // (op 8808) 
s_cmp_eq_i32 s10, 0 // (op 8808) 
s_mov_b32 s18, scc // (op 8808) 
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


s_add_u32 s50, s6, 256 // (op 8808) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8808) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8808) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8808) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8808) 

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
s_add_u32 s24, s20, 64 // (op 8808) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8808) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8808) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8808) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8808) 

s_sub_i32 s22, s24, s1 // (op 8808) 

s_mul_i32 s24, s23, s22 // (op 8808) 


s_sub_u32 s25, s19, s24 // (op 8808) 


s_add_u32 s19, s25, s1 // (op 8808) 


s_cmp_eq_i32 s18, 0 // (op 8808) 
s_mov_b32 s1, scc // (op 8808) 

s_cmp_eq_i32 s1, 0 // (op 8808) 
s_cselect_b32 s18, s19, s0 // (op 8808) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 8808) 


v_mul_lo_u32 v19, v18, s36 // (op 8808) 

v_add_u32 v18, v20, v19 // (op 8808) 


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
v_and_b32 v19, -64, v1 // (op 8827) 
s_add_i32 s1, s0, s16 // (op 8827) 

v_sub_u32 v20, v1, v19 // (op 8827) 

s_ashr_i32 s0, s1, 31 // (op 8827) 
v_lshrrev_b32 v19, 6, v1 // (op 8827) 
s_cmp_eq_i32 s10, 0 // (op 8827) 
s_mov_b32 s18, scc // (op 8827) 
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


s_add_u32 s50, s6, 256 // (op 8827) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8827) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8827) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8827) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8827) 

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
s_add_u32 s24, s20, 64 // (op 8827) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8827) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8827) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8827) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8827) 

s_sub_i32 s22, s24, s1 // (op 8827) 

s_mul_i32 s24, s23, s22 // (op 8827) 


s_sub_u32 s25, s19, s24 // (op 8827) 


s_add_u32 s19, s25, s1 // (op 8827) 


s_cmp_eq_i32 s18, 0 // (op 8827) 
s_mov_b32 s1, scc // (op 8827) 

s_cmp_eq_i32 s1, 0 // (op 8827) 
s_cselect_b32 s18, s19, s0 // (op 8827) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 8827) 


v_mul_lo_u32 v20, v19, s36 // (op 8827) 

v_add_u32 v19, v21, v20 // (op 8827) 


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
s_add_i32 s1, s0, s16 // (op 8872) 

s_ashr_i32 s0, s1, 31 // (op 8872) 
s_cmp_eq_i32 s10, 0 // (op 8872) 
s_mov_b32 s18, scc // (op 8872) 
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


s_add_u32 s50, s6, 256 // (op 8872) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8872) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8872) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8872) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8872) 

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



v_lshl_add_u32 v20, s0, 8, v1 // (op 8872) 

v_mul_lo_u32 v21, v20, s42 // (op 8872) 

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
s_add_i32 s1, s0, s16 // (op 8891) 

s_ashr_i32 s0, s1, 31 // (op 8891) 
s_cmp_eq_i32 s10, 0 // (op 8891) 
s_mov_b32 s18, scc // (op 8891) 
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


s_add_u32 s50, s6, 256 // (op 8891) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8891) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8891) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8891) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8891) 

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



v_lshl_add_u32 v20, s0, 8, v1 // (op 8891) 

v_mul_lo_u32 v22, v20, s42 // (op 8891) 

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


s_lshl_b64 s[0:1], s[32:33], 2 // (op 8200) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 8200) 


v_mov_b32 v20, v14 // (op 8200) 
buffer_load_dwordx4 v[26:29], v20, s[64:67], 0 offen // (op 8200) Load value
v_add_u32 v20, v20, v24 // (op 8200) 
buffer_load_dwordx4 v[30:33], v20, s[64:67], 0 offen // (op 8200) Load value
v_add_u32 v20, v20, v24 // (op 8200) 
buffer_load_dwordx4 v[34:37], v20, s[64:67], 0 offen // (op 8200) Load value
v_add_u32 v20, v20, v24 // (op 8200) 
buffer_load_dwordx4 v[38:41], v20, s[64:67], 0 offen // (op 8200) Load value





buffer_load_dword v20, v21, s[72:75], 0 offen // (op 8203) Load value

v_mov_b64 v[24:25], s[26:27] // (op 8206) convert

v_mov_b32 v23, v16 // (op 8206) reg expression
buffer_load_dwordx4 v[42:45], v23, s[52:55], 0 offen // (op 8206) Load value
v_add_u32 v23, v23, v24 // (op 8206) 
buffer_load_dwordx4 v[46:49], v23, s[52:55], 0 offen // (op 8206) Load value




v_mov_b32 v23, v18 // (op 8209) reg expression

buffer_load_ubyte v24, v23, s[68:71], 0 offen offset: 0 // (op 8209) Load value


s_lshl_b64 s[0:1], s[32:33], 2 // (op 8212) 
v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 8212) 


v_mov_b32 v23, v14 // (op 8212) 
buffer_load_dwordx4 v[52:55], v23, s[64:67], 0 offen offset:64 // (op 8212) Load value
v_add_u32 v23, v23, v50 // (op 8212) 
buffer_load_dwordx4 v[56:59], v23, s[64:67], 0 offen offset:64 // (op 8212) Load value
v_add_u32 v23, v23, v50 // (op 8212) 
buffer_load_dwordx4 v[60:63], v23, s[64:67], 0 offen offset:64 // (op 8212) Load value
v_add_u32 v23, v23, v50 // (op 8212) 
buffer_load_dwordx4 v[64:67], v23, s[64:67], 0 offen offset:64 // (op 8212) Load value






buffer_load_dword v14, v21, s[72:75], 0 offen offset:4 // (op 8215) Load value


v_mov_b64 v[50:51], s[26:27] // (op 8218) convert

s_lshl_b32 s0, s26, 7 // (op 8218) 
v_add_u32 v21, v16, s0 // (op 8218) 

buffer_load_dwordx4 v[68:71], v21, s[52:55], 0 offen // (op 8218) Load value
v_add_u32 v21, v21, v50 // (op 8218) 
buffer_load_dwordx4 v[72:75], v21, s[52:55], 0 offen // (op 8218) Load value





s_lshl_b32 s0, s36, 2 // (op 8221) 
v_add_u32 v16, v18, s0 // (op 8221) 


buffer_load_ubyte v21, v16, s[68:71], 0 offen offset: 0 // (op 8221) Load value


v_and_b32 v16, -4, v1 // (op 9021) 
v_sub_u32 v18, v1, v16 // (op 9021) 

v_lshlrev_b32 v16, 5, v18 // (op 9021) 

v_lshrrev_b32 v18, 2, v1 // (op 9021) 
v_lshl_add_u32 v23, v18, 9, v16 // (op 9021) 


v_lshlrev_b32 v16, 2, v23 // (op 9021) 

v_lshrrev_b32 v18, 3, v16 // (op 9021) 




s_waitcnt vmcnt(15)
ds_write_b128 v18, v[26:29] // (op 8223) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v18, v[30:33] offset:64 // (op 8223) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[34:37] offset:128 // (op 8223) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[38:41] offset:192 // (op 8223) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 9137) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v20 offset:16384 // (op 8224) Store local data 

v_and_b32 v16, -4, v1 // (op 8963) 
v_sub_u32 v18, v1, v16 // (op 8963) 

v_lshlrev_b32 v16, 4, v18 // (op 8963) 

v_lshrrev_b32 v18, 2, v1 // (op 8963) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 8963) 





s_waitcnt vmcnt(10)
ds_write_b128 v23, v[42:45] offset:17408 // (op 8225) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v23, v[46:49] offset:17472 // (op 8225) Store local data 

v_and_b32 v16, -64, v1 // (op 9079) 
v_sub_u32 v18, v1, v16 // (op 9079) 

v_and_b32 v16, 4294967232, v1 // (op 9079) 
v_add_u32 v23, v16, v18 // (op 9079) 





s_waitcnt vmcnt(8)
ds_write_b8 v23, v24 offset:25600 // (op 8226) Store local data 

v_and_b32 v16, -4, v1 // (op 8934) 
v_sub_u32 v18, v1, v16 // (op 8934) 

v_lshlrev_b32 v16, 4, v18 // (op 8934) 

v_lshrrev_b32 v18, 2, v1 // (op 8934) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 8934) 


v_and_b32 v16, -4, v1 // (op 8992) 
v_sub_u32 v18, v1, v16 // (op 8992) 

v_lshlrev_b32 v16, 5, v18 // (op 8992) 

v_lshrrev_b32 v18, 2, v1 // (op 8992) 
v_lshl_add_u32 v25, v18, 9, v16 // (op 8992) 


v_lshlrev_b32 v16, 2, v25 // (op 8992) 

v_lshrrev_b32 v18, 3, v16 // (op 8992) 

v_and_b32 v16, -64, v1 // (op 9050) 
v_sub_u32 v25, v1, v16 // (op 9050) 

v_and_b32 v16, 4294967232, v1 // (op 9050) 
v_add_u32 v50, v16, v25 // (op 9050) 


v_lshlrev_b32 v16, 2, v1 // (op 9108) 
v_and_b32 v25, -4, v1 // (op 9189) 
v_sub_u32 v51, v1, v25 // (op 9189) 

v_lshlrev_b32 v25, 4, v51 // (op 9189) 

v_lshrrev_b32 v51, 2, v1 // (op 9189) 
v_lshl_add_u32 v76, v51, 7, v25 // (op 9189) 


v_and_b32 v25, -4, v1 // (op 9218) 
v_sub_u32 v51, v1, v25 // (op 9218) 

v_lshlrev_b32 v25, 5, v51 // (op 9218) 

v_lshrrev_b32 v51, 2, v1 // (op 9218) 
v_lshl_add_u32 v77, v51, 9, v25 // (op 9218) 


v_lshlrev_b32 v25, 2, v77 // (op 9218) 

v_lshrrev_b32 v51, 3, v25 // (op 9218) 

v_and_b32 v25, -64, v1 // (op 9247) 
v_sub_u32 v77, v1, v25 // (op 9247) 

v_and_b32 v25, 4294967232, v1 // (op 9247) 
v_add_u32 v78, v25, v77 // (op 9247) 


v_lshlrev_b32 v25, 2, v1 // (op 9276) 
s_mov_b32 s0, 0 // (op 1967) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 1966) 
s_mov_b32 s1, s18 // (op 1966) convert

s_and_b32 s18, s1, 4294967294 // (op 1966) 

s_cmp_lt_u32 s0, s18 // (op 1966) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1966 // (op 1966) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1966 if false)
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1966:

s_barrier  // (op 9380) 
s_add_u32 s0, s0, 2 // (op 1968) 



s_lshl_b64 s[18:19], s[32:33], 2 // (op 10) 
v_lshrrev_b64 v[80:81], 3, s[18:19] // (op 10) 


v_mov_b32 v77, v15 // (op 10) 
v_mov_b32 v79, v77 // (op 10) 
buffer_load_dwordx4 v[26:29], v79, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v79, v79, v80 // (op 10) 
buffer_load_dwordx4 v[30:33], v79, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v79, v79, v80 // (op 10) 
ds_read_b64_tr_b8 v[82:83], v0 offset:17408 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:17920 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:21504 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:22016 // (op 72) Transpose load local data 
buffer_load_dwordx4 v[34:37], v79, s[64:67], 0 offen offset:128 // (op 10) Load value
s_barrier  // (op 9393) 
v_add_u32 v79, v79, v80 // (op 10) 



ds_read_b128 v[90:93], v4 // (op 77) Load local data 
buffer_load_dwordx4 v[38:41], v79, s[64:67], 0 offen offset:128 // (op 10) Load value




s_barrier  // (op 9407) 
v_mov_b32 v79, v22 // (op 25) 



buffer_load_dword v20, v79, s[72:75], 0 offen offset:8 // (op 25) Load value
ds_read_u8 v80, v3 offset:25600 // (op 82) Load local data 

s_barrier  // (op 9421) 
v_mov_b64 v[94:95], s[26:27] // (op 4) convert




ds_read_u8 v81, v5 offset:16384 // (op 87) Load local data 
v_mov_b32 v96, v17 // (op 4) 



s_lshl_b32 s1, s26, 8 // (op 4) 
v_add_u32 v97, v96, s1 // (op 4) 

ds_read_b128 v[98:101], v4 offset:2048 // (op 375) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[90:93], a[0:3], v80, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 90) 



ds_read_u8 v102, v5 offset:16512 // (op 379) Load local data 
buffer_load_dwordx4 v[42:45], v97, s[52:55], 0 offen // (op 4) Load value



v_add_u32 v97, v97, v94 // (op 4) 
ds_read_b128 v[104:107], v4 offset:4096 // (op 551) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[98:101], a[4:7], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 380) 



ds_read_u8 v103, v5 offset:16640 // (op 555) Load local data 
buffer_load_dwordx4 v[46:49], v97, s[52:55], 0 offen // (op 4) Load value







v_mov_b32 v97, v19 // (op 17) 
s_lshl_b32 s1, s36, 3 // (op 17) 
v_add_u32 v94, v97, s1 // (op 17) 


buffer_load_ubyte v24, v94, s[68:71], 0 offen offset: 0 // (op 17) Load value

ds_read_b128 v[108:111], v4 offset:6144 // (op 727) Load local data 



s_barrier  // (op 9444) 
ds_read_u8 v94, v5 offset:16768 // (op 731) Load local data 






ds_read_b128 v[112:115], v4 offset:8192 // (op 903) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:89], v[104:107], a[8:11], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 556) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:89], v[108:111], a[12:15], v80, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 732) 
ds_read_u8 v95, v5 offset:16896 // (op 907) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v51, v[52:55] offset:25856 // (op 257) Store local data 



ds_read_b128 v[116:119], v4 offset:10240 // (op 1079) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:89], v[112:115], a[16:19], v80, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 908) 



ds_read_u8 v120, v5 offset:17024 // (op 1083) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v51, v[56:59] offset:25920 // (op 257) Store local data 



ds_read_b128 v[122:125], v4 offset:12288 // (op 1255) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[82:89], v[116:119], a[20:23], v80, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1084) 



ds_read_u8 v121, v5 offset:17152 // (op 1259) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v51, v[60:63] offset:25984 // (op 257) Store local data 



ds_read_b128 v[126:129], v4 offset:14336 // (op 1431) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:89], v[122:125], a[24:27], v80, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1260) 



ds_read_u8 v130, v5 offset:17280 // (op 1435) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v51, v[64:67] offset:26048 // (op 257) Store local data 



ds_read_b64_tr_b8 v[132:133], v0 offset:17440 // (op 2000) Transpose load local data 
s_barrier  // (op 9456) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:89], v[126:129], a[28:31], v80, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1436) 





ds_read_b64_tr_b8 v[134:135], v0 offset:17952 // (op 2000) Transpose load local data 
ds_read_b64_tr_b8 v[136:137], v0 offset:21536 // (op 2000) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v0 offset:22048 // (op 2000) Transpose load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v25, v14 offset:42240 // (op 259) Store local data 


ds_read_u8 v80, v3 offset:25632 // (op 2003) Load local data 
s_barrier  // (op 9438) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:139], v[90:93], a[32:35], v80, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2327) 





s_waitcnt vmcnt(10)
ds_write_b128 v76, v[68:71] offset:43264 // (op 247) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:139], v[98:101], a[36:39], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2281) 
s_waitcnt vmcnt(9)
ds_write_b128 v76, v[72:75] offset:43328 // (op 247) Store local data 


s_barrier  // (op 9450) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:139], v[104:107], a[40:43], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2235) 





s_waitcnt vmcnt(8)
ds_write_b8 v78, v21 offset:51456 // (op 258) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:139], v[108:111], a[44:47], v80, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2189) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:139], v[112:115], a[48:51], v80, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2143) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:139], v[116:119], a[52:55], v80, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2097) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:139], v[122:125], a[56:59], v80, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2051) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:139], v[126:129], a[60:63], v80, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2005) 





s_lshl_b64 s[18:19], s[32:33], 2 // (op 260) 
v_lshrrev_b64 v[80:81], 3, s[18:19] // (op 260) 


v_mov_b32 v82, v77 // (op 260) 
buffer_load_dwordx4 v[52:55], v82, s[64:67], 0 offen offset:192 // (op 260) Load value
v_add_u32 v82, v82, v80 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9435) 
buffer_load_dwordx4 v[56:59], v82, s[64:67], 0 offen offset:192 // (op 260) Load value



v_add_u32 v82, v82, v80 // (op 260) 
ds_read_b64_tr_b8 v[88:89], v6 offset:43264 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v6 offset:43776 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v6 offset:47360 // (op 249) Transpose load local data 
buffer_load_dwordx4 v[60:63], v82, s[64:67], 0 offen offset:192 // (op 260) Load value
v_add_u32 v82, v82, v80 // (op 260) 
ds_read_b64_tr_b8 v[94:95], v6 offset:47872 // (op 249) Transpose load local data 
buffer_load_dwordx4 v[64:67], v82, s[64:67], 0 offen offset:192 // (op 260) Load value



s_barrier  // (op 9441) 






v_add_u32 v15, 128, v15 // (op 8755) 
ds_read_b128 v[80:83], v8 offset:25856 // (op 251) Load local data 
buffer_load_dword v14, v79, s[72:75], 0 offen offset:12 // (op 262) Load value

s_waitcnt lgkmcnt(5)
s_barrier  // (op 9447) 



v_add_u32 v22, 8, v22 // (op 8883) 
ds_read_u8 v77, v7 offset:51456 // (op 253) Load local data 
v_mov_b64 v[84:85], s[26:27] // (op 246) convert

s_barrier  // (op 9453) 
s_lshl_b32 s1, s26, 7 // (op 246) 



ds_read_u8 v79, v9 offset:42240 // (op 255) Load local data 
s_mul_i32 s18, 3, s1 // (op 246) 




v_add_u32 v86, v96, s18 // (op 246) 

buffer_load_dwordx4 v[68:71], v86, s[52:55], 0 offen // (op 246) Load value
v_add_u32 v86, v86, v84 // (op 246) 
ds_read_b128 v[98:101], v8 offset:27904 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:95], v[80:83], a[0:3], v77, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 256) 



ds_read_u8 v87, v9 offset:42368 // (op 389) Load local data 
buffer_load_dwordx4 v[72:75], v86, s[52:55], 0 offen // (op 246) Load value






ds_read_b128 v[102:105], v8 offset:29952 // (op 561) Load local data 





ds_read_u8 v96, v9 offset:42496 // (op 565) Load local data 
s_lshl_b32 s1, s36, 2 // (op 261) 



s_mul_i32 s18, 3, s1 // (op 261) 

s_lshl_b32 s1, s26, 8 // (op 8691) 
v_add_u32 v17, v17, s1 // (op 8691) 

v_add_u32 v84, v97, s18 // (op 261) 


ds_read_b128 v[106:109], v8 offset:32000 // (op 737) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:95], v[98:101], a[4:7], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 381) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:95], v[102:105], a[8:11], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 557) 
ds_read_u8 v85, v9 offset:42624 // (op 741) Load local data 
buffer_load_ubyte v21, v84, s[68:71], 0 offen offset: 0 // (op 261) Load value





ds_read_b128 v[110:113], v8 offset:34048 // (op 913) Load local data 



s_barrier  // (op 9396) 
ds_read_u8 v84, v9 offset:42752 // (op 917) Load local data 






s_lshl_b32 s1, s36, 3 // (op 8819) 
v_add_u32 v19, v19, s1 // (op 8819) 

ds_read_b128 v[114:117], v8 offset:36096 // (op 1089) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:95], v[106:109], a[12:15], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 733) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:95], v[110:113], a[16:19], v77, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 909) 
ds_read_u8 v86, v9 offset:42880 // (op 1093) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v18, v[26:29] // (op 13) Store local data 



ds_read_b128 v[118:121], v8 offset:38144 // (op 1265) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:95], v[114:117], a[20:23], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1085) 



ds_read_u8 v97, v9 offset:43008 // (op 1269) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v18, v[30:33] offset:64 // (op 13) Store local data 



ds_read_b128 v[122:125], v8 offset:40192 // (op 1441) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[118:121], a[24:27], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1261) 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[34:37] offset:128 // (op 13) Store local data 



s_barrier  // (op 9517) 
ds_read_u8 v126, v9 offset:43136 // (op 1445) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[38:41] offset:192 // (op 13) Store local data 



s_barrier  // (op 9521) 
s_barrier  // (op 9424) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:95], v[122:125], a[28:31], v77, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1437) 





ds_read_b64_tr_b8 v[88:89], v6 offset:43296 // (op 2008) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v6 offset:43808 // (op 2008) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v6 offset:47392 // (op 2008) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v6 offset:47904 // (op 2008) Transpose load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v16, v20 offset:16384 // (op 30) Store local data 


s_barrier  // (op 9553) 
s_barrier  // (op 9383) 
ds_read_u8 v77, v7 offset:51488 // (op 2011) Load local data 



s_waitcnt vmcnt(10)
ds_write_b128 v23, v[42:45] offset:17408 // (op 6) Store local data 
s_barrier  // (op 9557) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:95], v[80:83], a[32:35], v77, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2328) 
s_waitcnt vmcnt(9)
ds_write_b128 v23, v[46:49] offset:17472 // (op 6) Store local data 


s_barrier  // (op 9410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:95], v[98:101], a[36:39], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2282) 





s_waitcnt vmcnt(8)
ds_write_b8 v50, v24 offset:25600 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[102:105], a[40:43], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2236) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[106:109], a[44:47], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2190) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[88:95], v[110:113], a[48:51], v77, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[88:95], v[114:117], a[52:55], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2098) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[88:95], v[118:121], a[56:59], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2052) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[88:95], v[122:125], a[60:63], v77, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2006) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 1966) 
s_mov_b32 s1, s18 // (op 1966) convert

s_and_b32 s18, s1, 4294967294 // (op 1966) 

s_cmp_lt_u32 s0, s18 // (op 1966) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1966 // (op 1966) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1966 if true)
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1966:

















s_mul_hi_i32 s0, s16, s10 // (op 8639) 
v_and_b32 v15, -4, v1 // (op 8639) 
s_add_i32 s1, s0, s16 // (op 8639) 

v_sub_u32 v16, v1, v15 // (op 8639) 

s_ashr_i32 s0, s1, 31 // (op 8639) 
v_lshlrev_b32 v15, 4, v16 // (op 8639) 

s_cmp_eq_i32 s10, 0 // (op 8639) 
s_mov_b32 s18, scc // (op 8639) 
v_lshrrev_b32 v16, 2, v1 // (op 8639) 
s_cmp_eq_i32 s18, 0 // (op 8639) 
s_cselect_b32 s19, 0, -1 // (op 8639) 

v_lshlrev_b32 v17, 1, v16 // (op 8639) 

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


s_add_u32 s50, s6, 256 // (op 8639) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8639) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8639) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8639) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8639) 

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
s_add_u32 s24, s20, 64 // (op 8639) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8639) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8639) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8639) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8639) 

s_sub_i32 s22, s24, s1 // (op 8639) 

s_mul_i32 s24, s23, s22 // (op 8639) 


s_sub_u32 s25, s19, s24 // (op 8639) 


s_add_u32 s19, s25, s1 // (op 8639) 


s_cmp_eq_i32 s18, 0 // (op 8639) 
s_mov_b32 s1, scc // (op 8639) 

s_cmp_eq_i32 s1, 0 // (op 8639) 
s_cselect_b32 s18, s19, s0 // (op 8639) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 8639) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8639) 
s_lshr_b32 s18, s0, 1 // (op 8639) 

v_lshl_add_u32 v15, s18, 8, v17 // (op 8639) 


v_mul_lo_u32 v17, v15, s26 // (op 8639) 

v_add_u32 v15, v16, v17 // (op 8639) 


s_mul_i32 s1, s26, s31 // (op 8641) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 8641) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 8641) least significant
s_mul_hi_u32 s19, s26, s30 // (op 8641) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8641) most significant: sum
s_add_u32 s1, s1, s19 // (op 8641) most significant: sum


s_mov_b32 s52, s28 // (op 8641) reg expression
s_mov_b32 s53, s29 // (op 8641) reg expression
s_mov_b32 s54, s0 // (op 8641) reg expression
s_mov_b32 s55, 131072 // (op 8641) call()

s_mul_hi_i32 s0, s16, s10 // (op 8721) 
v_and_b32 v16, 4294967292, v1 // (op 8721) 
s_add_i32 s1, s0, s16 // (op 8721) 

v_and_b32 v17, -4, v1 // (op 8721) 
s_ashr_i32 s0, s1, 31 // (op 8721) 
v_sub_u32 v18, v1, v17 // (op 8721) 

s_cmp_eq_i32 s10, 0 // (op 8721) 
s_mov_b32 s18, scc // (op 8721) 
v_lshlrev_b32 v17, 5, v18 // (op 8721) 

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


s_add_u32 s50, s6, 256 // (op 8721) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8721) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8721) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8721) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8721) 

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



v_lshl_add_u32 v18, s0, 8, v16 // (op 8721) 


v_mul_lo_u32 v23, v18, s33 // (op 8721) most significant: low of low * high
v_mul_hi_u32 v16, v18, s32 // (op 8721) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 8721) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 8721) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8721) 
s_lshr_b32 s18, s0, 1 // (op 8721) 

v_lshl_add_u32 v16, s18, 8, v17 // (op 8721) 


v_mov_b32 v25, 0 // (op 8721) convert
v_mov_b32 v24, v16 // (op 8721) convert

v_add_co_u32 v16, s[0:1], v24, v22 // (op 8721) least significant half
v_addc_co_u32 v17, s[0:1], v25, v23, s[0:1] // (op 8721) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 8721) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 8721) 

v_mov_b32 v20, v18 // (op 8721) convert

s_mul_i32 s1, s32, s7 // (op 8723) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 8723) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 8723) least significant
s_mul_hi_u32 s19, s32, s6 // (op 8723) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8723) most significant: sum
s_add_u32 s1, s1, s19 // (op 8723) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8723) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8723) 

s_mov_b32 s64, s34 // (op 8723) reg expression
s_mov_b32 s65, s35 // (op 8723) reg expression
s_mov_b32 s66, s0 // (op 8723) reg expression
s_mov_b32 s67, 131072 // (op 8723) call()

s_mul_hi_i32 s0, s16, s10 // (op 8785) 
v_and_b32 v16, -64, v1 // (op 8785) 
s_add_i32 s1, s0, s16 // (op 8785) 

v_sub_u32 v17, v1, v16 // (op 8785) 

s_ashr_i32 s0, s1, 31 // (op 8785) 
v_lshrrev_b32 v16, 6, v1 // (op 8785) 
s_cmp_eq_i32 s10, 0 // (op 8785) 
s_mov_b32 s18, scc // (op 8785) 
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


s_add_u32 s50, s6, 256 // (op 8785) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8785) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8785) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8785) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8785) 

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
s_add_u32 s24, s20, 64 // (op 8785) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8785) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8785) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8785) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8785) 

s_sub_i32 s22, s24, s1 // (op 8785) 

s_mul_i32 s24, s23, s22 // (op 8785) 


s_sub_u32 s25, s19, s24 // (op 8785) 


s_add_u32 s19, s25, s1 // (op 8785) 


s_cmp_eq_i32 s18, 0 // (op 8785) 
s_mov_b32 s1, scc // (op 8785) 

s_cmp_eq_i32 s1, 0 // (op 8785) 
s_cselect_b32 s18, s19, s0 // (op 8785) 



v_lshl_add_u32 v18, s18, 6, v17 // (op 8785) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 8785) 
s_lshr_b32 s18, s0, 1 // (op 8785) 

v_lshl_add_u32 v17, s18, 3, v16 // (op 8785) 


v_mul_lo_u32 v16, v17, s36 // (op 8785) 

v_add_u32 v17, v18, v16 // (op 8785) 


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
s_add_i32 s1, s0, s16 // (op 8849) 

s_ashr_i32 s0, s1, 31 // (op 8849) 
s_cmp_eq_i32 s10, 0 // (op 8849) 
s_mov_b32 s18, scc // (op 8849) 
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


s_add_u32 s50, s6, 256 // (op 8849) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8849) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 8849) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 8849) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 8849) 

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



v_lshl_add_u32 v16, s0, 8, v1 // (op 8849) 

v_mul_lo_u32 v18, v16, s42 // (op 8849) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 8849) 
s_lshr_b32 s18, s0, 1 // (op 8849) 

v_lshl_add_u32 v16, s18, 3, v18 // (op 8849) 


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

v_and_b32 v18, -4, v1 // (op 8947) 
v_sub_u32 v19, v1, v18 // (op 8947) 

v_lshlrev_b32 v18, 4, v19 // (op 8947) 

v_lshrrev_b32 v19, 2, v1 // (op 8947) 
v_lshl_add_u32 v22, v19, 7, v18 // (op 8947) 


v_and_b32 v18, -4, v1 // (op 9005) 
v_sub_u32 v19, v1, v18 // (op 9005) 

v_lshlrev_b32 v18, 5, v19 // (op 9005) 

v_lshrrev_b32 v19, 2, v1 // (op 9005) 
v_lshl_add_u32 v23, v19, 9, v18 // (op 9005) 


v_lshlrev_b32 v18, 2, v23 // (op 9005) 

v_lshrrev_b32 v19, 3, v18 // (op 9005) 

v_and_b32 v18, -64, v1 // (op 9063) 
v_sub_u32 v23, v1, v18 // (op 9063) 

v_and_b32 v18, 4294967232, v1 // (op 9063) 
v_add_u32 v24, v18, v23 // (op 9063) 


v_lshlrev_b32 v18, 2, v1 // (op 9121) 
s_ashr_i64 s[0:1], s[30:31], 7 // (op 1972) 
s_and_b32 s18, s0, 4294967294 // (op 1972) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 1971) 
s_mov_b32 s19, s0 // (op 1971) convert

s_cmp_lt_u32 s18, s19 // (op 1971) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1971 // (op 1971) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1971 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1971:

s_add_u32 s18, s18, 1 // (op 1973) 
v_mov_b64 v[26:27], s[26:27] // (op 180) convert

v_mov_b32 v14, v15 // (op 180) 
v_mov_b32 v21, v14 // (op 180) reg expression
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 180) Load value
v_add_u32 v21, v21, v26 // (op 180) 
buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 180) Load value



s_lshl_b32 s19, s26, 7 // (op 8629) 
v_add_u32 v15, v15, s19 // (op 8629) 


s_barrier  // (op 9386) 
s_lshl_b64 s[0:1], s[32:33], 2 // (op 194) 



v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 194) 


v_mov_b32 v14, v20 // (op 194) 
v_mov_b32 v21, v14 // (op 194) 
buffer_load_dwordx4 v[36:39], v21, s[64:67], 0 offen // (op 194) Load value
v_add_u32 v21, v21, v26 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[28:31] offset:17408 // (op 181) Store local data 
buffer_load_dwordx4 v[40:43], v21, s[64:67], 0 offen // (op 194) Load value
v_add_u32 v21, v21, v26 // (op 194) 
buffer_load_dwordx4 v[44:47], v21, s[64:67], 0 offen // (op 194) Load value
v_add_u32 v21, v21, v26 // (op 194) 
buffer_load_dwordx4 v[48:51], v21, s[64:67], 0 offen // (op 194) Load value



v_add_u32 v20, 64, v20 // (op 8713) 
s_waitcnt vmcnt(4)
ds_write_b128 v22, v[32:35] offset:17472 // (op 181) Store local data 



v_mov_b32 v14, v17 // (op 195) 
s_barrier  // (op 9399) 
v_mov_b32 v21, v14 // (op 195) reg expression




s_waitcnt vmcnt(3)
ds_write_b128 v19, v[36:39] // (op 191) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[40:43] offset:64 // (op 191) Store local data 
buffer_load_ubyte v23, v21, s[68:71], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s19, s36, 2 // (op 8777) 
v_add_u32 v17, v17, s19 // (op 8777) 

s_waitcnt vmcnt(2)
ds_write_b128 v19, v[44:47] offset:128 // (op 191) Store local data 


v_mov_b32 v14, v16 // (op 196) 

buffer_load_dword v21, v14, s[72:75], 0 offen // (op 196) Load value

v_add_u32 v16, 4, v16 // (op 8841) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[48:51] offset:192 // (op 191) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 9389) 



s_barrier  // (op 9413) 
ds_read_b64_tr_b8 v[26:27], v0 offset:17408 // (op 183) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:17920 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:21504 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:22016 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v23 offset:25600 // (op 192) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 9403) 



s_barrier  // (op 9427) 
ds_read_b128 v[34:37], v4 // (op 185) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 9417) 
s_waitcnt vmcnt(0)
ds_write_b32 v18, v21 offset:16384 // (op 193) Store local data 



ds_read_u8 v14, v3 offset:25600 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9431) 



ds_read_u8 v21, v5 offset:16384 // (op 189) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 190) 



ds_read_u8 v23, v5 offset:16512 // (op 358) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 530) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 359) 



ds_read_u8 v25, v5 offset:16640 // (op 534) Load local data 



ds_read_b128 v[46:49], v4 offset:6144 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[42:45], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 535) 



ds_read_u8 v50, v5 offset:16768 // (op 710) Load local data 



ds_read_b128 v[52:55], v4 offset:8192 // (op 882) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[46:49], a[12:15], v14, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 711) 



ds_read_u8 v51, v5 offset:16896 // (op 886) Load local data 



ds_read_b128 v[56:59], v4 offset:10240 // (op 1058) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[52:55], a[16:19], v14, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 887) 



ds_read_u8 v60, v5 offset:17024 // (op 1062) Load local data 



ds_read_b128 v[62:65], v4 offset:12288 // (op 1234) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[56:59], a[20:23], v14, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1063) 



ds_read_u8 v61, v5 offset:17152 // (op 1238) Load local data 



ds_read_b128 v[66:69], v4 offset:14336 // (op 1410) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:33], v[62:65], a[24:27], v14, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1239) 



ds_read_u8 v70, v5 offset:17280 // (op 1414) Load local data 



ds_read_b64_tr_b8 v[72:73], v0 offset:17440 // (op 1760) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:17952 // (op 1760) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:33], v[66:69], a[28:31], v14, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1415) 
ds_read_b64_tr_b8 v[76:77], v0 offset:21536 // (op 1760) Transpose load local data 


ds_read_b64_tr_b8 v[78:79], v0 offset:22048 // (op 1760) Transpose load local data 


ds_read_u8 v14, v3 offset:25632 // (op 1763) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:79], v[34:37], a[32:35], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1765) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:79], v[38:41], a[36:39], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1796) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:79], v[42:45], a[40:43], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1827) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:79], v[46:49], a[44:47], v14, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1858) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:79], v[52:55], a[48:51], v14, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1889) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:79], v[56:59], a[52:55], v14, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1920) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:79], v[62:65], a[56:59], v14, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1951) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:79], v[66:69], a[60:63], v14, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1982) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 1971) 
s_mov_b32 s19, s0 // (op 1971) convert

s_cmp_lt_u32 s18, s19 // (op 1971) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1971 // (op 1971) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1971 if true)
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1971:























s_barrier  // (op 9541) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 3577) 
v_accvgpr_read v7, a5 // (op 3577) 
v_accvgpr_read v8, a6 // (op 3577) 
v_accvgpr_read v9, a7 // (op 3577) 
v_mul_f32 v14, v11, v6 // (op 3577) 
v_mul_f32 v15, v11, v7 // (op 3577) 
v_mul_f32 v16, v11, v8 // (op 3577) 
v_mul_f32 v17, v11, v9 // (op 3577) 

v_cvt_f16_f32 v6, v14 // (op 3579) 
v_cvt_f16_f32 v7, v15 // (op 3579) 
v_cvt_f16_f32 v8, v16 // (op 3579) 
v_cvt_f16_f32 v9, v17 // (op 3579) 


v_pack_b32_f16 v4, v6, v7 // (op 3580) 
v_pack_b32_f16 v5, v8, v9 // (op 3580) 

s_lshl_b32 s0, s4, 6 // (op 3580) 
v_add_u32 v0, v13, s0 // (op 3580) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3580) Store value



v_accvgpr_read v6, a8 // (op 3591) 
v_accvgpr_read v7, a9 // (op 3591) 
v_accvgpr_read v8, a10 // (op 3591) 
v_accvgpr_read v9, a11 // (op 3591) 
v_mul_f32 v14, v11, v6 // (op 3591) 
v_mul_f32 v15, v11, v7 // (op 3591) 
v_mul_f32 v16, v11, v8 // (op 3591) 
v_mul_f32 v17, v11, v9 // (op 3591) 

v_cvt_f16_f32 v6, v14 // (op 3593) 
v_cvt_f16_f32 v7, v15 // (op 3593) 
v_cvt_f16_f32 v8, v16 // (op 3593) 
v_cvt_f16_f32 v9, v17 // (op 3593) 


v_pack_b32_f16 v4, v6, v7 // (op 3594) 
v_pack_b32_f16 v5, v8, v9 // (op 3594) 

s_lshl_b32 s0, s4, 7 // (op 3594) 
v_add_u32 v0, v13, s0 // (op 3594) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3594) Store value



v_accvgpr_read v6, a12 // (op 3605) 
v_accvgpr_read v7, a13 // (op 3605) 
v_accvgpr_read v8, a14 // (op 3605) 
v_accvgpr_read v9, a15 // (op 3605) 
v_mul_f32 v14, v11, v6 // (op 3605) 
v_mul_f32 v15, v11, v7 // (op 3605) 
v_mul_f32 v16, v11, v8 // (op 3605) 
v_mul_f32 v17, v11, v9 // (op 3605) 

v_cvt_f16_f32 v6, v14 // (op 3607) 
v_cvt_f16_f32 v7, v15 // (op 3607) 
v_cvt_f16_f32 v8, v16 // (op 3607) 
v_cvt_f16_f32 v9, v17 // (op 3607) 


v_pack_b32_f16 v4, v6, v7 // (op 3608) 
v_pack_b32_f16 v5, v8, v9 // (op 3608) 

s_lshl_b32 s0, s4, 6 // (op 3608) 
s_mul_i32 s1, 3, s0 // (op 3608) 

v_add_u32 v0, v13, s1 // (op 3608) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3608) Store value



v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 

v_cvt_f16_f32 v6, v14 // (op 3621) 
v_cvt_f16_f32 v7, v15 // (op 3621) 
v_cvt_f16_f32 v8, v16 // (op 3621) 
v_cvt_f16_f32 v9, v17 // (op 3621) 


v_pack_b32_f16 v4, v6, v7 // (op 3622) 
v_pack_b32_f16 v5, v8, v9 // (op 3622) 

s_lshl_b32 s0, s4, 8 // (op 3622) 
v_add_u32 v0, v13, s0 // (op 3622) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3622) Store value



v_accvgpr_read v6, a20 // (op 3633) 
v_accvgpr_read v7, a21 // (op 3633) 
v_accvgpr_read v8, a22 // (op 3633) 
v_accvgpr_read v9, a23 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 

v_cvt_f16_f32 v6, v14 // (op 3635) 
v_cvt_f16_f32 v7, v15 // (op 3635) 
v_cvt_f16_f32 v8, v16 // (op 3635) 
v_cvt_f16_f32 v9, v17 // (op 3635) 


v_pack_b32_f16 v4, v6, v7 // (op 3636) 
v_pack_b32_f16 v5, v8, v9 // (op 3636) 

s_lshl_b32 s0, s4, 6 // (op 3636) 
s_mul_i32 s1, 5, s0 // (op 3636) 

v_add_u32 v0, v13, s1 // (op 3636) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3636) Store value



v_accvgpr_read v6, a24 // (op 3647) 
v_accvgpr_read v7, a25 // (op 3647) 
v_accvgpr_read v8, a26 // (op 3647) 
v_accvgpr_read v9, a27 // (op 3647) 
v_mul_f32 v14, v11, v6 // (op 3647) 
v_mul_f32 v15, v11, v7 // (op 3647) 
v_mul_f32 v16, v11, v8 // (op 3647) 
v_mul_f32 v17, v11, v9 // (op 3647) 

v_cvt_f16_f32 v6, v14 // (op 3649) 
v_cvt_f16_f32 v7, v15 // (op 3649) 
v_cvt_f16_f32 v8, v16 // (op 3649) 
v_cvt_f16_f32 v9, v17 // (op 3649) 


v_pack_b32_f16 v4, v6, v7 // (op 3650) 
v_pack_b32_f16 v5, v8, v9 // (op 3650) 

s_lshl_b32 s0, s4, 6 // (op 3650) 
s_mul_i32 s1, 6, s0 // (op 3650) 

v_add_u32 v0, v13, s1 // (op 3650) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3650) Store value



v_accvgpr_read v6, a28 // (op 3661) 
v_accvgpr_read v7, a29 // (op 3661) 
v_accvgpr_read v8, a30 // (op 3661) 
v_accvgpr_read v9, a31 // (op 3661) 
v_mul_f32 v14, v11, v6 // (op 3661) 
v_mul_f32 v15, v11, v7 // (op 3661) 
v_mul_f32 v16, v11, v8 // (op 3661) 
v_mul_f32 v17, v11, v9 // (op 3661) 

v_cvt_f16_f32 v6, v14 // (op 3663) 
v_cvt_f16_f32 v7, v15 // (op 3663) 
v_cvt_f16_f32 v8, v16 // (op 3663) 
v_cvt_f16_f32 v9, v17 // (op 3663) 


v_pack_b32_f16 v4, v6, v7 // (op 3664) 
v_pack_b32_f16 v5, v8, v9 // (op 3664) 

s_lshl_b32 s0, s4, 6 // (op 3664) 
s_mul_i32 s1, 7, s0 // (op 3664) 

v_add_u32 v0, v13, s1 // (op 3664) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3664) Store value



v_accvgpr_read v6, a32 // (op 3717) 
v_accvgpr_read v7, a33 // (op 3717) 
v_accvgpr_read v8, a34 // (op 3717) 
v_accvgpr_read v9, a35 // (op 3717) 
v_mul_f32 v14, v11, v6 // (op 3717) 
v_mul_f32 v15, v11, v7 // (op 3717) 
v_mul_f32 v16, v11, v8 // (op 3717) 
v_mul_f32 v17, v11, v9 // (op 3717) 

v_cvt_f16_f32 v6, v14 // (op 3719) 
v_cvt_f16_f32 v7, v15 // (op 3719) 
v_cvt_f16_f32 v8, v16 // (op 3719) 
v_cvt_f16_f32 v9, v17 // (op 3719) 


v_pack_b32_f16 v4, v6, v7 // (op 3721) 
v_pack_b32_f16 v5, v8, v9 // (op 3721) 

v_add_u32 v0, 64, v13 // (op 3721) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3721) Store value



v_accvgpr_read v6, a36 // (op 3739) 
v_accvgpr_read v7, a37 // (op 3739) 
v_accvgpr_read v8, a38 // (op 3739) 
v_accvgpr_read v9, a39 // (op 3739) 
v_mul_f32 v14, v11, v6 // (op 3739) 
v_mul_f32 v15, v11, v7 // (op 3739) 
v_mul_f32 v16, v11, v8 // (op 3739) 
v_mul_f32 v17, v11, v9 // (op 3739) 

v_cvt_f16_f32 v6, v14 // (op 3741) 
v_cvt_f16_f32 v7, v15 // (op 3741) 
v_cvt_f16_f32 v8, v16 // (op 3741) 
v_cvt_f16_f32 v9, v17 // (op 3741) 


v_pack_b32_f16 v4, v6, v7 // (op 3723) 
v_pack_b32_f16 v5, v8, v9 // (op 3723) 

s_lshl_b32 s0, s4, 6 // (op 3723) 
s_add_u32 s0, s0, 64 // (op 3723) 
v_add_u32 v0, v13, s0 // (op 3723) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3723) Store value



v_accvgpr_read v6, a40 // (op 3745) 
v_accvgpr_read v7, a41 // (op 3745) 
v_accvgpr_read v8, a42 // (op 3745) 
v_accvgpr_read v9, a43 // (op 3745) 
v_mul_f32 v14, v11, v6 // (op 3745) 
v_mul_f32 v15, v11, v7 // (op 3745) 
v_mul_f32 v16, v11, v8 // (op 3745) 
v_mul_f32 v17, v11, v9 // (op 3745) 

v_cvt_f16_f32 v6, v14 // (op 3747) 
v_cvt_f16_f32 v7, v15 // (op 3747) 
v_cvt_f16_f32 v8, v16 // (op 3747) 
v_cvt_f16_f32 v9, v17 // (op 3747) 


v_pack_b32_f16 v4, v6, v7 // (op 3725) 
v_pack_b32_f16 v5, v8, v9 // (op 3725) 

s_lshl_b32 s0, s4, 7 // (op 3725) 
s_add_u32 s0, s0, 64 // (op 3725) 
v_add_u32 v0, v13, s0 // (op 3725) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3725) Store value



v_accvgpr_read v6, a44 // (op 3751) 
v_accvgpr_read v7, a45 // (op 3751) 
v_accvgpr_read v8, a46 // (op 3751) 
v_accvgpr_read v9, a47 // (op 3751) 
v_mul_f32 v14, v11, v6 // (op 3751) 
v_mul_f32 v15, v11, v7 // (op 3751) 
v_mul_f32 v16, v11, v8 // (op 3751) 
v_mul_f32 v17, v11, v9 // (op 3751) 

v_cvt_f16_f32 v6, v14 // (op 3753) 
v_cvt_f16_f32 v7, v15 // (op 3753) 
v_cvt_f16_f32 v8, v16 // (op 3753) 
v_cvt_f16_f32 v9, v17 // (op 3753) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

s_lshl_b32 s0, s4, 6 // (op 3727) 
s_mul_i32 s1, 3, s0 // (op 3727) 
s_add_u32 s1, s1, 64 // (op 3727) 

v_add_u32 v0, v13, s1 // (op 3727) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a48 // (op 3757) 
v_accvgpr_read v7, a49 // (op 3757) 
v_accvgpr_read v8, a50 // (op 3757) 
v_accvgpr_read v9, a51 // (op 3757) 
v_mul_f32 v14, v11, v6 // (op 3757) 
v_mul_f32 v15, v11, v7 // (op 3757) 
v_mul_f32 v16, v11, v8 // (op 3757) 
v_mul_f32 v17, v11, v9 // (op 3757) 

v_cvt_f16_f32 v6, v14 // (op 3759) 
v_cvt_f16_f32 v7, v15 // (op 3759) 
v_cvt_f16_f32 v8, v16 // (op 3759) 
v_cvt_f16_f32 v9, v17 // (op 3759) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 8 // (op 3729) 
s_add_u32 s0, s0, 64 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a52 // (op 3763) 
v_accvgpr_read v7, a53 // (op 3763) 
v_accvgpr_read v8, a54 // (op 3763) 
v_accvgpr_read v9, a55 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_cvt_f16_f32 v6, v14 // (op 3765) 
v_cvt_f16_f32 v7, v15 // (op 3765) 
v_cvt_f16_f32 v8, v16 // (op 3765) 
v_cvt_f16_f32 v9, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3731) 
v_pack_b32_f16 v5, v8, v9 // (op 3731) 

s_lshl_b32 s0, s4, 6 // (op 3731) 
s_mul_i32 s1, 5, s0 // (op 3731) 
s_add_u32 s1, s1, 64 // (op 3731) 

v_add_u32 v0, v13, s1 // (op 3731) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3731) Store value



v_accvgpr_read v6, a56 // (op 3769) 
v_accvgpr_read v7, a57 // (op 3769) 
v_accvgpr_read v8, a58 // (op 3769) 
v_accvgpr_read v9, a59 // (op 3769) 
v_mul_f32 v14, v11, v6 // (op 3769) 
v_mul_f32 v15, v11, v7 // (op 3769) 
v_mul_f32 v16, v11, v8 // (op 3769) 
v_mul_f32 v17, v11, v9 // (op 3769) 

v_cvt_f16_f32 v6, v14 // (op 3771) 
v_cvt_f16_f32 v7, v15 // (op 3771) 
v_cvt_f16_f32 v8, v16 // (op 3771) 
v_cvt_f16_f32 v9, v17 // (op 3771) 


v_pack_b32_f16 v4, v6, v7 // (op 3733) 
v_pack_b32_f16 v5, v8, v9 // (op 3733) 

s_lshl_b32 s0, s4, 6 // (op 3733) 
s_mul_i32 s1, 6, s0 // (op 3733) 
s_add_u32 s1, s1, 64 // (op 3733) 

v_add_u32 v0, v13, s1 // (op 3733) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3733) Store value



v_accvgpr_read v6, a60 // (op 3775) 
v_accvgpr_read v7, a61 // (op 3775) 
v_accvgpr_read v8, a62 // (op 3775) 
v_accvgpr_read v9, a63 // (op 3775) 
v_mul_f32 v14, v11, v6 // (op 3775) 
v_mul_f32 v15, v11, v7 // (op 3775) 
v_mul_f32 v16, v11, v8 // (op 3775) 
v_mul_f32 v17, v11, v9 // (op 3775) 

v_cvt_f16_f32 v6, v14 // (op 3777) 
v_cvt_f16_f32 v7, v15 // (op 3777) 
v_cvt_f16_f32 v8, v16 // (op 3777) 
v_cvt_f16_f32 v9, v17 // (op 3777) 


v_pack_b32_f16 v4, v6, v7 // (op 3735) 
v_pack_b32_f16 v5, v8, v9 // (op 3735) 

s_lshl_b32 s0, s4, 6 // (op 3735) 
s_mul_i32 s1, 7, s0 // (op 3735) 
s_add_u32 s1, s1, 64 // (op 3735) 

v_add_u32 v0, v13, s1 // (op 3735) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3735) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 4113) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4113) Load value

v_accvgpr_read v4, a32 // (op 4092) 
v_accvgpr_read v5, a33 // (op 4092) 
v_accvgpr_read v6, a34 // (op 4092) 
v_accvgpr_read v7, a35 // (op 4092) 
v_mul_f32 v14, v11, v4 // (op 4092) 
v_mul_f32 v15, v11, v5 // (op 4092) 
v_mul_f32 v16, v11, v6 // (op 4092) 
v_mul_f32 v17, v11, v7 // (op 4092) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4112) 
v_lshrrev_b32 v7, 16, v8 // (op 4112) 
v_cvt_f32_f16 v7, v7 // (op 4112) 
v_mul_f32 v18, v10, v6 // (op 4112) 
v_mul_f32 v19, v10, v7 // (op 4112) 
v_cvt_f32_f16 v6, v9 // (op 4112) 
v_lshrrev_b32 v7, 16, v9 // (op 4112) 
v_cvt_f32_f16 v7, v7 // (op 4112) 
v_mul_f32 v20, v10, v6 // (op 4112) 
v_mul_f32 v21, v10, v7 // (op 4112) 


v_add_f32 v6, v18, v14 // (op 4093) 
v_add_f32 v7, v19, v15 // (op 4093) 
v_add_f32 v8, v20, v16 // (op 4093) 
v_add_f32 v9, v21, v17 // (op 4093) 


v_cvt_f16_f32 v14, v6 // (op 4094) 
v_cvt_f16_f32 v15, v7 // (op 4094) 
v_cvt_f16_f32 v16, v8 // (op 4094) 
v_cvt_f16_f32 v17, v9 // (op 4094) 


v_pack_b32_f16 v8, v14, v15 // (op 4096) 
v_pack_b32_f16 v9, v16, v17 // (op 4096) 

v_add_u32 v0, 64, v13 // (op 4096) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4096) Store value





s_lshl_b32 s0, s2, 6 // (op 4118) 
s_add_u32 s0, s0, 64 // (op 4118) 
v_add_u32 v0, v12, s0 // (op 4118) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4118) Load value

v_accvgpr_read v4, a36 // (op 4150) 
v_accvgpr_read v5, a37 // (op 4150) 
v_accvgpr_read v6, a38 // (op 4150) 
v_accvgpr_read v7, a39 // (op 4150) 
v_mul_f32 v14, v11, v4 // (op 4150) 
v_mul_f32 v15, v11, v5 // (op 4150) 
v_mul_f32 v16, v11, v6 // (op 4150) 
v_mul_f32 v17, v11, v7 // (op 4150) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4115) 
v_lshrrev_b32 v7, 16, v8 // (op 4115) 
v_cvt_f32_f16 v7, v7 // (op 4115) 
v_mul_f32 v18, v10, v6 // (op 4115) 
v_mul_f32 v19, v10, v7 // (op 4115) 
v_cvt_f32_f16 v6, v9 // (op 4115) 
v_lshrrev_b32 v7, 16, v9 // (op 4115) 
v_cvt_f32_f16 v7, v7 // (op 4115) 
v_mul_f32 v20, v10, v6 // (op 4115) 
v_mul_f32 v21, v10, v7 // (op 4115) 


v_add_f32 v6, v18, v14 // (op 4116) 
v_add_f32 v7, v19, v15 // (op 4116) 
v_add_f32 v8, v20, v16 // (op 4116) 
v_add_f32 v9, v21, v17 // (op 4116) 


v_cvt_f16_f32 v14, v6 // (op 4117) 
v_cvt_f16_f32 v15, v7 // (op 4117) 
v_cvt_f16_f32 v16, v8 // (op 4117) 
v_cvt_f16_f32 v17, v9 // (op 4117) 


v_pack_b32_f16 v8, v14, v15 // (op 4098) 
v_pack_b32_f16 v9, v16, v17 // (op 4098) 

s_lshl_b32 s0, s4, 6 // (op 4098) 
s_add_u32 s0, s0, 64 // (op 4098) 
v_add_u32 v0, v13, s0 // (op 4098) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4098) Store value





s_lshl_b32 s0, s2, 7 // (op 4123) 
s_add_u32 s0, s0, 64 // (op 4123) 
v_add_u32 v0, v12, s0 // (op 4123) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4123) Load value

v_accvgpr_read v4, a40 // (op 4152) 
v_accvgpr_read v5, a41 // (op 4152) 
v_accvgpr_read v6, a42 // (op 4152) 
v_accvgpr_read v7, a43 // (op 4152) 
v_mul_f32 v14, v11, v4 // (op 4152) 
v_mul_f32 v15, v11, v5 // (op 4152) 
v_mul_f32 v16, v11, v6 // (op 4152) 
v_mul_f32 v17, v11, v7 // (op 4152) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4120) 
v_lshrrev_b32 v7, 16, v8 // (op 4120) 
v_cvt_f32_f16 v7, v7 // (op 4120) 
v_mul_f32 v18, v10, v6 // (op 4120) 
v_mul_f32 v19, v10, v7 // (op 4120) 
v_cvt_f32_f16 v6, v9 // (op 4120) 
v_lshrrev_b32 v7, 16, v9 // (op 4120) 
v_cvt_f32_f16 v7, v7 // (op 4120) 
v_mul_f32 v20, v10, v6 // (op 4120) 
v_mul_f32 v21, v10, v7 // (op 4120) 


v_add_f32 v6, v18, v14 // (op 4121) 
v_add_f32 v7, v19, v15 // (op 4121) 
v_add_f32 v8, v20, v16 // (op 4121) 
v_add_f32 v9, v21, v17 // (op 4121) 


v_cvt_f16_f32 v14, v6 // (op 4122) 
v_cvt_f16_f32 v15, v7 // (op 4122) 
v_cvt_f16_f32 v16, v8 // (op 4122) 
v_cvt_f16_f32 v17, v9 // (op 4122) 


v_pack_b32_f16 v8, v14, v15 // (op 4100) 
v_pack_b32_f16 v9, v16, v17 // (op 4100) 

s_lshl_b32 s0, s4, 7 // (op 4100) 
s_add_u32 s0, s0, 64 // (op 4100) 
v_add_u32 v0, v13, s0 // (op 4100) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4100) Store value





s_lshl_b32 s0, s2, 6 // (op 4128) 
s_mul_i32 s1, 3, s0 // (op 4128) 
s_add_u32 s1, s1, 64 // (op 4128) 

v_add_u32 v0, v12, s1 // (op 4128) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4128) Load value

v_accvgpr_read v4, a44 // (op 4154) 
v_accvgpr_read v5, a45 // (op 4154) 
v_accvgpr_read v6, a46 // (op 4154) 
v_accvgpr_read v7, a47 // (op 4154) 
v_mul_f32 v14, v11, v4 // (op 4154) 
v_mul_f32 v15, v11, v5 // (op 4154) 
v_mul_f32 v16, v11, v6 // (op 4154) 
v_mul_f32 v17, v11, v7 // (op 4154) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4125) 
v_lshrrev_b32 v7, 16, v8 // (op 4125) 
v_cvt_f32_f16 v7, v7 // (op 4125) 
v_mul_f32 v18, v10, v6 // (op 4125) 
v_mul_f32 v19, v10, v7 // (op 4125) 
v_cvt_f32_f16 v6, v9 // (op 4125) 
v_lshrrev_b32 v7, 16, v9 // (op 4125) 
v_cvt_f32_f16 v7, v7 // (op 4125) 
v_mul_f32 v20, v10, v6 // (op 4125) 
v_mul_f32 v21, v10, v7 // (op 4125) 


v_add_f32 v6, v18, v14 // (op 4126) 
v_add_f32 v7, v19, v15 // (op 4126) 
v_add_f32 v8, v20, v16 // (op 4126) 
v_add_f32 v9, v21, v17 // (op 4126) 


v_cvt_f16_f32 v14, v6 // (op 4127) 
v_cvt_f16_f32 v15, v7 // (op 4127) 
v_cvt_f16_f32 v16, v8 // (op 4127) 
v_cvt_f16_f32 v17, v9 // (op 4127) 


v_pack_b32_f16 v8, v14, v15 // (op 4102) 
v_pack_b32_f16 v9, v16, v17 // (op 4102) 

s_lshl_b32 s0, s4, 6 // (op 4102) 
s_mul_i32 s1, 3, s0 // (op 4102) 
s_add_u32 s1, s1, 64 // (op 4102) 

v_add_u32 v0, v13, s1 // (op 4102) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4102) Store value





s_lshl_b32 s0, s2, 8 // (op 4133) 
s_add_u32 s0, s0, 64 // (op 4133) 
v_add_u32 v0, v12, s0 // (op 4133) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4133) Load value

v_accvgpr_read v4, a48 // (op 4156) 
v_accvgpr_read v5, a49 // (op 4156) 
v_accvgpr_read v6, a50 // (op 4156) 
v_accvgpr_read v7, a51 // (op 4156) 
v_mul_f32 v14, v11, v4 // (op 4156) 
v_mul_f32 v15, v11, v5 // (op 4156) 
v_mul_f32 v16, v11, v6 // (op 4156) 
v_mul_f32 v17, v11, v7 // (op 4156) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4130) 
v_lshrrev_b32 v7, 16, v8 // (op 4130) 
v_cvt_f32_f16 v7, v7 // (op 4130) 
v_mul_f32 v18, v10, v6 // (op 4130) 
v_mul_f32 v19, v10, v7 // (op 4130) 
v_cvt_f32_f16 v6, v9 // (op 4130) 
v_lshrrev_b32 v7, 16, v9 // (op 4130) 
v_cvt_f32_f16 v7, v7 // (op 4130) 
v_mul_f32 v20, v10, v6 // (op 4130) 
v_mul_f32 v21, v10, v7 // (op 4130) 


v_add_f32 v6, v18, v14 // (op 4131) 
v_add_f32 v7, v19, v15 // (op 4131) 
v_add_f32 v8, v20, v16 // (op 4131) 
v_add_f32 v9, v21, v17 // (op 4131) 


v_cvt_f16_f32 v14, v6 // (op 4132) 
v_cvt_f16_f32 v15, v7 // (op 4132) 
v_cvt_f16_f32 v16, v8 // (op 4132) 
v_cvt_f16_f32 v17, v9 // (op 4132) 


v_pack_b32_f16 v8, v14, v15 // (op 4104) 
v_pack_b32_f16 v9, v16, v17 // (op 4104) 

s_lshl_b32 s0, s4, 8 // (op 4104) 
s_add_u32 s0, s0, 64 // (op 4104) 
v_add_u32 v0, v13, s0 // (op 4104) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4104) Store value





s_lshl_b32 s0, s2, 6 // (op 4138) 
s_mul_i32 s1, 5, s0 // (op 4138) 
s_add_u32 s1, s1, 64 // (op 4138) 

v_add_u32 v0, v12, s1 // (op 4138) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4138) Load value

v_accvgpr_read v4, a52 // (op 4158) 
v_accvgpr_read v5, a53 // (op 4158) 
v_accvgpr_read v6, a54 // (op 4158) 
v_accvgpr_read v7, a55 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4135) 
v_lshrrev_b32 v7, 16, v8 // (op 4135) 
v_cvt_f32_f16 v7, v7 // (op 4135) 
v_mul_f32 v18, v10, v6 // (op 4135) 
v_mul_f32 v19, v10, v7 // (op 4135) 
v_cvt_f32_f16 v6, v9 // (op 4135) 
v_lshrrev_b32 v7, 16, v9 // (op 4135) 
v_cvt_f32_f16 v7, v7 // (op 4135) 
v_mul_f32 v20, v10, v6 // (op 4135) 
v_mul_f32 v21, v10, v7 // (op 4135) 


v_add_f32 v6, v18, v14 // (op 4136) 
v_add_f32 v7, v19, v15 // (op 4136) 
v_add_f32 v8, v20, v16 // (op 4136) 
v_add_f32 v9, v21, v17 // (op 4136) 


v_cvt_f16_f32 v14, v6 // (op 4137) 
v_cvt_f16_f32 v15, v7 // (op 4137) 
v_cvt_f16_f32 v16, v8 // (op 4137) 
v_cvt_f16_f32 v17, v9 // (op 4137) 


v_pack_b32_f16 v8, v14, v15 // (op 4106) 
v_pack_b32_f16 v9, v16, v17 // (op 4106) 

s_lshl_b32 s0, s4, 6 // (op 4106) 
s_mul_i32 s1, 5, s0 // (op 4106) 
s_add_u32 s1, s1, 64 // (op 4106) 

v_add_u32 v0, v13, s1 // (op 4106) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4106) Store value





s_lshl_b32 s0, s2, 6 // (op 4143) 
s_mul_i32 s1, 6, s0 // (op 4143) 
s_add_u32 s1, s1, 64 // (op 4143) 

v_add_u32 v0, v12, s1 // (op 4143) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4143) Load value

v_accvgpr_read v4, a56 // (op 4160) 
v_accvgpr_read v5, a57 // (op 4160) 
v_accvgpr_read v6, a58 // (op 4160) 
v_accvgpr_read v7, a59 // (op 4160) 
v_mul_f32 v14, v11, v4 // (op 4160) 
v_mul_f32 v15, v11, v5 // (op 4160) 
v_mul_f32 v16, v11, v6 // (op 4160) 
v_mul_f32 v17, v11, v7 // (op 4160) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4140) 
v_lshrrev_b32 v7, 16, v8 // (op 4140) 
v_cvt_f32_f16 v7, v7 // (op 4140) 
v_mul_f32 v18, v10, v6 // (op 4140) 
v_mul_f32 v19, v10, v7 // (op 4140) 
v_cvt_f32_f16 v6, v9 // (op 4140) 
v_lshrrev_b32 v7, 16, v9 // (op 4140) 
v_cvt_f32_f16 v7, v7 // (op 4140) 
v_mul_f32 v20, v10, v6 // (op 4140) 
v_mul_f32 v21, v10, v7 // (op 4140) 


v_add_f32 v6, v18, v14 // (op 4141) 
v_add_f32 v7, v19, v15 // (op 4141) 
v_add_f32 v8, v20, v16 // (op 4141) 
v_add_f32 v9, v21, v17 // (op 4141) 


v_cvt_f16_f32 v14, v6 // (op 4142) 
v_cvt_f16_f32 v15, v7 // (op 4142) 
v_cvt_f16_f32 v16, v8 // (op 4142) 
v_cvt_f16_f32 v17, v9 // (op 4142) 


v_pack_b32_f16 v8, v14, v15 // (op 4108) 
v_pack_b32_f16 v9, v16, v17 // (op 4108) 

s_lshl_b32 s0, s4, 6 // (op 4108) 
s_mul_i32 s1, 6, s0 // (op 4108) 
s_add_u32 s1, s1, 64 // (op 4108) 

v_add_u32 v0, v13, s1 // (op 4108) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4108) Store value





s_lshl_b32 s0, s2, 6 // (op 4148) 
s_mul_i32 s1, 7, s0 // (op 4148) 
s_add_u32 s1, s1, 64 // (op 4148) 

v_add_u32 v0, v12, s1 // (op 4148) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4148) Load value

v_accvgpr_read v4, a60 // (op 4162) 
v_accvgpr_read v5, a61 // (op 4162) 
v_accvgpr_read v6, a62 // (op 4162) 
v_accvgpr_read v7, a63 // (op 4162) 
v_mul_f32 v14, v11, v4 // (op 4162) 
v_mul_f32 v15, v11, v5 // (op 4162) 
v_mul_f32 v16, v11, v6 // (op 4162) 
v_mul_f32 v17, v11, v7 // (op 4162) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4145) 
v_lshrrev_b32 v7, 16, v8 // (op 4145) 
v_cvt_f32_f16 v7, v7 // (op 4145) 
v_mul_f32 v18, v10, v6 // (op 4145) 
v_mul_f32 v19, v10, v7 // (op 4145) 
v_cvt_f32_f16 v6, v9 // (op 4145) 
v_lshrrev_b32 v7, 16, v9 // (op 4145) 
v_cvt_f32_f16 v7, v7 // (op 4145) 
v_mul_f32 v20, v10, v6 // (op 4145) 
v_mul_f32 v21, v10, v7 // (op 4145) 


v_add_f32 v6, v18, v14 // (op 4146) 
v_add_f32 v7, v19, v15 // (op 4146) 
v_add_f32 v8, v20, v16 // (op 4146) 
v_add_f32 v9, v21, v17 // (op 4146) 


v_cvt_f16_f32 v14, v6 // (op 4147) 
v_cvt_f16_f32 v15, v7 // (op 4147) 
v_cvt_f16_f32 v16, v8 // (op 4147) 
v_cvt_f16_f32 v17, v9 // (op 4147) 


v_pack_b32_f16 v8, v14, v15 // (op 4110) 
v_pack_b32_f16 v9, v16, v17 // (op 4110) 

s_lshl_b32 s0, s4, 6 // (op 4110) 
s_mul_i32 s1, 7, s0 // (op 4110) 
s_add_u32 s1, s1, 64 // (op 4110) 

v_add_u32 v0, v13, s1 // (op 4110) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4110) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 3917) 
v_add_u32 v0, v12, s0 // (op 3917) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3917) Load value

v_accvgpr_read v4, a4 // (op 3913) 
v_accvgpr_read v5, a5 // (op 3913) 
v_accvgpr_read v6, a6 // (op 3913) 
v_accvgpr_read v7, a7 // (op 3913) 
v_mul_f32 v14, v11, v4 // (op 3913) 
v_mul_f32 v15, v11, v5 // (op 3913) 
v_mul_f32 v16, v11, v6 // (op 3913) 
v_mul_f32 v17, v11, v7 // (op 3913) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3918) 
v_lshrrev_b32 v7, 16, v8 // (op 3918) 
v_cvt_f32_f16 v7, v7 // (op 3918) 
v_mul_f32 v18, v10, v6 // (op 3918) 
v_mul_f32 v19, v10, v7 // (op 3918) 
v_cvt_f32_f16 v6, v9 // (op 3918) 
v_lshrrev_b32 v7, 16, v9 // (op 3918) 
v_cvt_f32_f16 v7, v7 // (op 3918) 
v_mul_f32 v20, v10, v6 // (op 3918) 
v_mul_f32 v21, v10, v7 // (op 3918) 


v_add_f32 v6, v18, v14 // (op 3914) 
v_add_f32 v7, v19, v15 // (op 3914) 
v_add_f32 v8, v20, v16 // (op 3914) 
v_add_f32 v9, v21, v17 // (op 3914) 


v_cvt_f16_f32 v14, v6 // (op 3915) 
v_cvt_f16_f32 v15, v7 // (op 3915) 
v_cvt_f16_f32 v16, v8 // (op 3915) 
v_cvt_f16_f32 v17, v9 // (op 3915) 


v_pack_b32_f16 v8, v14, v15 // (op 3916) 
v_pack_b32_f16 v9, v16, v17 // (op 3916) 

s_lshl_b32 s0, s4, 6 // (op 3916) 
v_add_u32 v0, v13, s0 // (op 3916) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3916) Store value





s_lshl_b32 s0, s2, 7 // (op 3931) 
v_add_u32 v0, v12, s0 // (op 3931) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3931) Load value

v_accvgpr_read v4, a8 // (op 3927) 
v_accvgpr_read v5, a9 // (op 3927) 
v_accvgpr_read v6, a10 // (op 3927) 
v_accvgpr_read v7, a11 // (op 3927) 
v_mul_f32 v14, v11, v4 // (op 3927) 
v_mul_f32 v15, v11, v5 // (op 3927) 
v_mul_f32 v16, v11, v6 // (op 3927) 
v_mul_f32 v17, v11, v7 // (op 3927) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3932) 
v_lshrrev_b32 v7, 16, v8 // (op 3932) 
v_cvt_f32_f16 v7, v7 // (op 3932) 
v_mul_f32 v18, v10, v6 // (op 3932) 
v_mul_f32 v19, v10, v7 // (op 3932) 
v_cvt_f32_f16 v6, v9 // (op 3932) 
v_lshrrev_b32 v7, 16, v9 // (op 3932) 
v_cvt_f32_f16 v7, v7 // (op 3932) 
v_mul_f32 v20, v10, v6 // (op 3932) 
v_mul_f32 v21, v10, v7 // (op 3932) 


v_add_f32 v6, v18, v14 // (op 3928) 
v_add_f32 v7, v19, v15 // (op 3928) 
v_add_f32 v8, v20, v16 // (op 3928) 
v_add_f32 v9, v21, v17 // (op 3928) 


v_cvt_f16_f32 v14, v6 // (op 3929) 
v_cvt_f16_f32 v15, v7 // (op 3929) 
v_cvt_f16_f32 v16, v8 // (op 3929) 
v_cvt_f16_f32 v17, v9 // (op 3929) 


v_pack_b32_f16 v8, v14, v15 // (op 3930) 
v_pack_b32_f16 v9, v16, v17 // (op 3930) 

s_lshl_b32 s0, s4, 7 // (op 3930) 
v_add_u32 v0, v13, s0 // (op 3930) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3930) Store value





s_lshl_b32 s0, s2, 6 // (op 3945) 
s_mul_i32 s1, 3, s0 // (op 3945) 

v_add_u32 v0, v12, s1 // (op 3945) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3945) Load value

v_accvgpr_read v4, a12 // (op 3941) 
v_accvgpr_read v5, a13 // (op 3941) 
v_accvgpr_read v6, a14 // (op 3941) 
v_accvgpr_read v7, a15 // (op 3941) 
v_mul_f32 v14, v11, v4 // (op 3941) 
v_mul_f32 v15, v11, v5 // (op 3941) 
v_mul_f32 v16, v11, v6 // (op 3941) 
v_mul_f32 v17, v11, v7 // (op 3941) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3946) 
v_lshrrev_b32 v7, 16, v8 // (op 3946) 
v_cvt_f32_f16 v7, v7 // (op 3946) 
v_mul_f32 v18, v10, v6 // (op 3946) 
v_mul_f32 v19, v10, v7 // (op 3946) 
v_cvt_f32_f16 v6, v9 // (op 3946) 
v_lshrrev_b32 v7, 16, v9 // (op 3946) 
v_cvt_f32_f16 v7, v7 // (op 3946) 
v_mul_f32 v20, v10, v6 // (op 3946) 
v_mul_f32 v21, v10, v7 // (op 3946) 


v_add_f32 v6, v18, v14 // (op 3942) 
v_add_f32 v7, v19, v15 // (op 3942) 
v_add_f32 v8, v20, v16 // (op 3942) 
v_add_f32 v9, v21, v17 // (op 3942) 


v_cvt_f16_f32 v14, v6 // (op 3943) 
v_cvt_f16_f32 v15, v7 // (op 3943) 
v_cvt_f16_f32 v16, v8 // (op 3943) 
v_cvt_f16_f32 v17, v9 // (op 3943) 


v_pack_b32_f16 v8, v14, v15 // (op 3944) 
v_pack_b32_f16 v9, v16, v17 // (op 3944) 

s_lshl_b32 s0, s4, 6 // (op 3944) 
s_mul_i32 s1, 3, s0 // (op 3944) 

v_add_u32 v0, v13, s1 // (op 3944) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3944) Store value





s_lshl_b32 s0, s2, 8 // (op 3959) 
v_add_u32 v0, v12, s0 // (op 3959) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3959) Load value

v_accvgpr_read v4, a16 // (op 3955) 
v_accvgpr_read v5, a17 // (op 3955) 
v_accvgpr_read v6, a18 // (op 3955) 
v_accvgpr_read v7, a19 // (op 3955) 
v_mul_f32 v14, v11, v4 // (op 3955) 
v_mul_f32 v15, v11, v5 // (op 3955) 
v_mul_f32 v16, v11, v6 // (op 3955) 
v_mul_f32 v17, v11, v7 // (op 3955) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3960) 
v_lshrrev_b32 v7, 16, v8 // (op 3960) 
v_cvt_f32_f16 v7, v7 // (op 3960) 
v_mul_f32 v18, v10, v6 // (op 3960) 
v_mul_f32 v19, v10, v7 // (op 3960) 
v_cvt_f32_f16 v6, v9 // (op 3960) 
v_lshrrev_b32 v7, 16, v9 // (op 3960) 
v_cvt_f32_f16 v7, v7 // (op 3960) 
v_mul_f32 v20, v10, v6 // (op 3960) 
v_mul_f32 v21, v10, v7 // (op 3960) 


v_add_f32 v6, v18, v14 // (op 3956) 
v_add_f32 v7, v19, v15 // (op 3956) 
v_add_f32 v8, v20, v16 // (op 3956) 
v_add_f32 v9, v21, v17 // (op 3956) 


v_cvt_f16_f32 v14, v6 // (op 3957) 
v_cvt_f16_f32 v15, v7 // (op 3957) 
v_cvt_f16_f32 v16, v8 // (op 3957) 
v_cvt_f16_f32 v17, v9 // (op 3957) 


v_pack_b32_f16 v8, v14, v15 // (op 3958) 
v_pack_b32_f16 v9, v16, v17 // (op 3958) 

s_lshl_b32 s0, s4, 8 // (op 3958) 
v_add_u32 v0, v13, s0 // (op 3958) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3958) Store value





s_lshl_b32 s0, s2, 6 // (op 3973) 
s_mul_i32 s1, 5, s0 // (op 3973) 

v_add_u32 v0, v12, s1 // (op 3973) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3973) Load value

v_accvgpr_read v4, a20 // (op 3969) 
v_accvgpr_read v5, a21 // (op 3969) 
v_accvgpr_read v6, a22 // (op 3969) 
v_accvgpr_read v7, a23 // (op 3969) 
v_mul_f32 v14, v11, v4 // (op 3969) 
v_mul_f32 v15, v11, v5 // (op 3969) 
v_mul_f32 v16, v11, v6 // (op 3969) 
v_mul_f32 v17, v11, v7 // (op 3969) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3974) 
v_lshrrev_b32 v7, 16, v8 // (op 3974) 
v_cvt_f32_f16 v7, v7 // (op 3974) 
v_mul_f32 v18, v10, v6 // (op 3974) 
v_mul_f32 v19, v10, v7 // (op 3974) 
v_cvt_f32_f16 v6, v9 // (op 3974) 
v_lshrrev_b32 v7, 16, v9 // (op 3974) 
v_cvt_f32_f16 v7, v7 // (op 3974) 
v_mul_f32 v20, v10, v6 // (op 3974) 
v_mul_f32 v21, v10, v7 // (op 3974) 


v_add_f32 v6, v18, v14 // (op 3970) 
v_add_f32 v7, v19, v15 // (op 3970) 
v_add_f32 v8, v20, v16 // (op 3970) 
v_add_f32 v9, v21, v17 // (op 3970) 


v_cvt_f16_f32 v14, v6 // (op 3971) 
v_cvt_f16_f32 v15, v7 // (op 3971) 
v_cvt_f16_f32 v16, v8 // (op 3971) 
v_cvt_f16_f32 v17, v9 // (op 3971) 


v_pack_b32_f16 v8, v14, v15 // (op 3972) 
v_pack_b32_f16 v9, v16, v17 // (op 3972) 

s_lshl_b32 s0, s4, 6 // (op 3972) 
s_mul_i32 s1, 5, s0 // (op 3972) 

v_add_u32 v0, v13, s1 // (op 3972) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3972) Store value





s_lshl_b32 s0, s2, 6 // (op 3987) 
s_mul_i32 s1, 6, s0 // (op 3987) 

v_add_u32 v0, v12, s1 // (op 3987) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a24 // (op 3983) 
v_accvgpr_read v5, a25 // (op 3983) 
v_accvgpr_read v6, a26 // (op 3983) 
v_accvgpr_read v7, a27 // (op 3983) 
v_mul_f32 v14, v11, v4 // (op 3983) 
v_mul_f32 v15, v11, v5 // (op 3983) 
v_mul_f32 v16, v11, v6 // (op 3983) 
v_mul_f32 v17, v11, v7 // (op 3983) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3988) 
v_lshrrev_b32 v7, 16, v8 // (op 3988) 
v_cvt_f32_f16 v7, v7 // (op 3988) 
v_mul_f32 v18, v10, v6 // (op 3988) 
v_mul_f32 v19, v10, v7 // (op 3988) 
v_cvt_f32_f16 v6, v9 // (op 3988) 
v_lshrrev_b32 v7, 16, v9 // (op 3988) 
v_cvt_f32_f16 v7, v7 // (op 3988) 
v_mul_f32 v20, v10, v6 // (op 3988) 
v_mul_f32 v21, v10, v7 // (op 3988) 


v_add_f32 v6, v18, v14 // (op 3984) 
v_add_f32 v7, v19, v15 // (op 3984) 
v_add_f32 v8, v20, v16 // (op 3984) 
v_add_f32 v9, v21, v17 // (op 3984) 


v_cvt_f16_f32 v14, v6 // (op 3985) 
v_cvt_f16_f32 v15, v7 // (op 3985) 
v_cvt_f16_f32 v16, v8 // (op 3985) 
v_cvt_f16_f32 v17, v9 // (op 3985) 


v_pack_b32_f16 v8, v14, v15 // (op 3986) 
v_pack_b32_f16 v9, v16, v17 // (op 3986) 

s_lshl_b32 s0, s4, 6 // (op 3986) 
s_mul_i32 s1, 6, s0 // (op 3986) 

v_add_u32 v0, v13, s1 // (op 3986) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3986) Store value





s_lshl_b32 s0, s2, 6 // (op 4001) 
s_mul_i32 s1, 7, s0 // (op 4001) 

v_add_u32 v0, v12, s1 // (op 4001) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4001) Load value

v_accvgpr_read v4, a28 // (op 3997) 
v_accvgpr_read v5, a29 // (op 3997) 
v_accvgpr_read v6, a30 // (op 3997) 
v_accvgpr_read v7, a31 // (op 3997) 
v_mul_f32 v14, v11, v4 // (op 3997) 
v_mul_f32 v15, v11, v5 // (op 3997) 
v_mul_f32 v16, v11, v6 // (op 3997) 
v_mul_f32 v17, v11, v7 // (op 3997) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4002) 
v_lshrrev_b32 v7, 16, v8 // (op 4002) 
v_cvt_f32_f16 v7, v7 // (op 4002) 
v_mul_f32 v18, v10, v6 // (op 4002) 
v_mul_f32 v19, v10, v7 // (op 4002) 
v_cvt_f32_f16 v6, v9 // (op 4002) 
v_lshrrev_b32 v7, 16, v9 // (op 4002) 
v_cvt_f32_f16 v7, v7 // (op 4002) 
v_mul_f32 v20, v10, v6 // (op 4002) 
v_mul_f32 v21, v10, v7 // (op 4002) 


v_add_f32 v6, v18, v14 // (op 3998) 
v_add_f32 v7, v19, v15 // (op 3998) 
v_add_f32 v8, v20, v16 // (op 3998) 
v_add_f32 v9, v21, v17 // (op 3998) 


v_cvt_f16_f32 v14, v6 // (op 3999) 
v_cvt_f16_f32 v15, v7 // (op 3999) 
v_cvt_f16_f32 v16, v8 // (op 3999) 
v_cvt_f16_f32 v17, v9 // (op 3999) 


v_pack_b32_f16 v8, v14, v15 // (op 4000) 
v_pack_b32_f16 v9, v16, v17 // (op 4000) 

s_lshl_b32 s0, s4, 6 // (op 4000) 
s_mul_i32 s1, 7, s0 // (op 4000) 

v_add_u32 v0, v13, s1 // (op 4000) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4000) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 204
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51712 // lds bytes
  .amdhsa_accum_offset 140
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
    DataTypeB: F4
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 51712
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 140
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

