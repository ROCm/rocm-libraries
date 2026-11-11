// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_,@function
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 35234) 
s_mul_hi_i32 s1, s16, s10 // (op 35489) 
v_and_b32 v0, 63, v1 // (op 35295) 
v_and_b32 v3, 48, v1 // (op 35295) 
v_sub_u32 v4, v0, v3 // (op 35295) 

v_lshrrev_b32 v3, 4, v0 // (op 35295) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 35295) 


v_lshrrev_b32 v3, 7, v1 // (op 35295) 
v_lshlrev_b32 v4, 4, v3 // (op 35295) 

v_lshl_add_u32 v3, v0, 8, v4 // (op 35295) 


v_lshlrev_b32 v0, 2, v3 // (op 35295) 

v_lshrrev_b32 v3, 3, v0 // (op 35295) 

v_lshrrev_b32 v0, 6, v1 // (op 35352) 
v_and_b32 v4, 1, v0 // (op 35352) 

v_and_b32 v0, 15, v1 // (op 35352) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 35352) 


v_and_b32 v0, 48, v1 // (op 35352) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 35352) 


v_lshrrev_b32 v0, 7, v1 // (op 35410) 
v_and_b32 v5, 15, v1 // (op 35410) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 35410) 


v_and_b32 v0, 63, v1 // (op 35410) 
v_lshrrev_b32 v5, 4, v0 // (op 35410) 

v_lshl_add_u32 v0, v5, 8, v6 // (op 35410) 


v_lshrrev_b32 v5, 6, v1 // (op 35468) 
v_and_b32 v6, 1, v5 // (op 35468) 

v_and_b32 v5, 15, v1 // (op 35468) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 35468) 


v_and_b32 v5, 63, v1 // (op 35468) 
v_lshrrev_b32 v6, 4, v5 // (op 35468) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 35468) 


v_and_b32 v6, 63, v1 // (op 35535) 
v_and_b32 v7, 48, v1 // (op 35535) 
v_sub_u32 v8, v6, v7 // (op 35535) 

v_lshrrev_b32 v7, 4, v6 // (op 35535) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 35535) 


v_lshrrev_b32 v7, 7, v1 // (op 35535) 
v_lshlrev_b32 v8, 4, v7 // (op 35535) 

v_lshl_add_u32 v7, v6, 8, v8 // (op 35535) 


v_lshlrev_b32 v6, 2, v7 // (op 35535) 

v_lshrrev_b32 v7, 3, v6 // (op 35535) 

v_lshrrev_b32 v6, 6, v1 // (op 35579) 
v_and_b32 v8, 1, v6 // (op 35579) 

v_and_b32 v6, 15, v1 // (op 35579) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 35579) 


v_and_b32 v6, 48, v1 // (op 35579) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 35579) 


v_lshrrev_b32 v6, 7, v1 // (op 35608) 
v_and_b32 v9, 15, v1 // (op 35608) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 35608) 


v_and_b32 v6, 63, v1 // (op 35608) 
v_lshrrev_b32 v9, 4, v6 // (op 35608) 

v_lshl_add_u32 v6, v9, 8, v10 // (op 35608) 


v_lshrrev_b32 v9, 6, v1 // (op 35637) 
v_and_b32 v10, 1, v9 // (op 35637) 

v_and_b32 v9, 15, v1 // (op 35637) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 35637) 


v_and_b32 v9, 63, v1 // (op 35637) 
v_lshrrev_b32 v10, 4, v9 // (op 35637) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 35637) 


v_lshrrev_b32 v10, 7, v1 // (op 35234) 
s_add_i32 s18, s0, s16 // (op 35234) 

v_and_b32 v11, 63, v1 // (op 35234) 
s_ashr_i32 s0, s18, 31 // (op 35234) 
v_lshrrev_b32 v12, 4, v11 // (op 35234) 
s_cmp_eq_i32 s10, 0 // (op 35234) 
s_mov_b32 s19, scc // (op 35234) 
v_lshlrev_b32 v13, 2, v12 // (op 35234) 

s_cmp_eq_i32 s19, 0 // (op 35234) 
s_cselect_b32 s51, 0, -1 // (op 35234) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 35234) 


s_and_b32 s19, s11, 31 // (op 35234) 
v_lshrrev_b32 v10, 2, v11 // (op 35234) 

v_and_b32 v11, 3, v10 // (op 35234) 

v_lshrrev_b32 v10, 7, v1 // (op 35489) 
v_and_b32 v13, 63, v1 // (op 35489) 
s_lshl_b32 s52, 1, s19 // (op 35234) 
v_and_b32 v14, 3, v1 // (op 35234) 
s_add_i32 s53, s52, s51 // (op 35234) 


s_and_b32 s51, s0, s53 // (op 35234) 


s_add_i32 s0, s1, s16 // (op 35489) 

s_ashr_i32 s1, s0, 31 // (op 35489) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 35234) 


s_add_i32 s52, s18, s51 // (op 35234) 


v_lshrrev_b32 v11, 6, v1 // (op 35234) 
s_ashr_i32 s18, s52, s19 // (op 35234) 


v_and_b32 v14, 1, v11 // (op 35234) 

s_lshl_b32 s19, s11, 24 // (op 35234) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 35234) 


s_ashr_i32 s51, s19, 31 // (op 35234) 

s_xor_b32 s19, s18, s51 // (op 35234) 

s_sub_i32 s18, s19, s51 // (op 35234) 


s_mul_hi_i32 s19, s18, s12 // (op 35234) 
s_add_i32 s51, s19, s18 // (op 35234) 

s_ashr_i32 s19, s51, 31 // (op 35234) 
s_cmp_eq_i32 s12, 0 // (op 35234) 
s_mov_b32 s52, scc // (op 35234) 
s_cmp_eq_i32 s52, 0 // (op 35234) 
s_cselect_b32 s53, 0, -1 // (op 35234) 

s_and_b32 s52, s13, 31 // (op 35234) 
v_lshrrev_b32 v14, 4, v13 // (op 35489) 
s_cmp_eq_i32 s10, 0 // (op 35489) 
s_mov_b32 s54, scc // (op 35489) 
v_lshlrev_b32 v15, 2, v14 // (op 35489) 

s_cmp_eq_i32 s54, 0 // (op 35489) 
s_cselect_b32 s55, 0, -1 // (op 35489) 

s_lshl_b32 s54, 1, s52 // (op 35234) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 35489) 


v_lshrrev_b32 v10, 2, v13 // (op 35489) 

s_add_i32 s56, s54, s53 // (op 35234) 


s_and_b32 s53, s19, s56 // (op 35234) 


s_add_i32 s19, s51, s53 // (op 35234) 


s_ashr_i32 s51, s19, s52 // (op 35234) 


s_lshl_b32 s19, s13, 24 // (op 35234) 
s_ashr_i32 s52, s19, 31 // (op 35234) 

s_xor_b32 s19, s51, s52 // (op 35234) 

s_sub_i32 s51, s19, s52 // (op 35234) 


s_mul_i32 s19, s18, s15 // (op 35234) 
s_sub_u32 s52, s16, s19 // (op 35234) 

s_mul_i32 s19, s51, s15 // (op 35234) 
s_add_u32 s19, s19, s52 // (op 35234) 

s_mul_hi_i32 s52, s16, s9 // (op 35234) 
s_add_i32 s53, s52, s16 // (op 35234) 

s_ashr_i32 s52, s53, 31 // (op 35234) 
s_cmp_eq_i32 s9, 0 // (op 35234) 
s_mov_b32 s54, scc // (op 35234) 
s_cmp_eq_i32 s54, 0 // (op 35234) 
s_cselect_b32 s56, 0, -1 // (op 35234) 

s_and_b32 s54, s11, 31 // (op 35489) 
v_and_b32 v13, 3, v10 // (op 35489) 

s_lshl_b32 s57, 1, s54 // (op 35489) 
v_and_b32 v10, 3, v1 // (op 35489) 
s_add_i32 s58, s57, s55 // (op 35489) 


s_and_b32 s55, s1, s58 // (op 35489) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 35489) 


s_add_i32 s1, s0, s55 // (op 35489) 


v_lshrrev_b32 v10, 6, v1 // (op 35489) 
s_ashr_i32 s0, s1, s54 // (op 35489) 


v_and_b32 v13, 1, v10 // (op 35489) 

s_lshl_b32 s1, s11, 24 // (op 35489) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 35489) 


s_ashr_i32 s54, s1, 31 // (op 35489) 

s_xor_b32 s1, s0, s54 // (op 35489) 

s_sub_i32 s0, s1, s54 // (op 35489) 


s_mul_hi_i32 s1, s0, s12 // (op 35489) 
s_add_i32 s54, s1, s0 // (op 35489) 

s_ashr_i32 s1, s54, 31 // (op 35489) 
s_cmp_eq_i32 s12, 0 // (op 35489) 
s_mov_b32 s55, scc // (op 35489) 
s_cmp_eq_i32 s55, 0 // (op 35489) 
s_cselect_b32 s57, 0, -1 // (op 35489) 

s_and_b32 s55, s13, 31 // (op 35489) 
s_lshl_b32 s58, 1, s55 // (op 35489) 
s_add_i32 s59, s58, s57 // (op 35489) 


s_and_b32 s57, s1, s59 // (op 35489) 


s_add_i32 s1, s54, s57 // (op 35489) 


s_ashr_i32 s54, s1, s55 // (op 35489) 


s_lshl_b32 s1, s13, 24 // (op 35489) 
s_ashr_i32 s55, s1, 31 // (op 35489) 

s_xor_b32 s1, s54, s55 // (op 35489) 

s_sub_i32 s54, s1, s55 // (op 35489) 


s_mul_i32 s1, s0, s15 // (op 35489) 
s_sub_u32 s55, s16, s1 // (op 35489) 

s_mul_i32 s1, s54, s15 // (op 35489) 
s_add_u32 s1, s1, s55 // (op 35489) 

s_mul_hi_i32 s55, s16, s9 // (op 35489) 
s_add_i32 s57, s55, s16 // (op 35489) 

s_ashr_i32 s55, s57, 31 // (op 35489) 
s_cmp_eq_i32 s9, 0 // (op 35489) 
s_mov_b32 s58, scc // (op 35489) 
s_cmp_eq_i32 s58, 0 // (op 35489) 
s_cselect_b32 s59, 0, -1 // (op 35489) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 35234) 
s_lshl_b32 s60, 1, s58 // (op 35234) 
s_add_i32 s61, s60, s56 // (op 35234) 


s_and_b32 s56, s52, s61 // (op 35234) 


s_add_i32 s52, s53, s56 // (op 35234) 


s_ashr_i32 s53, s52, s58 // (op 35234) 


s_lshl_b32 s52, s46, 24 // (op 35234) 
s_ashr_i32 s56, s52, 31 // (op 35234) 

s_xor_b32 s52, s53, s56 // (op 35234) 

s_sub_i32 s53, s52, s56 // (op 35234) 


s_and_b32 s52, s46, 31 // (op 35489) 
s_lshl_b32 s56, 1, s52 // (op 35489) 
s_add_i32 s58, s56, s59 // (op 35489) 


s_and_b32 s56, s55, s58 // (op 35489) 


s_add_i32 s55, s57, s56 // (op 35489) 


s_ashr_i32 s56, s55, s52 // (op 35489) 


s_lshl_b32 s52, s46, 24 // (op 35489) 
s_ashr_i32 s55, s52, 31 // (op 35489) 

s_xor_b32 s52, s56, s55 // (op 35489) 

s_sub_i32 s56, s52, s55 // (op 35489) 


s_add_u32 s58, s6, 256 // (op 35234) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 35234) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 35234) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 35234) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 35234) 

s_mul_i32 s52, s15, s58 // (op 35234) 
s_mul_i32 s55, s8, s52 // (op 35234) 

s_mul_i32 s52, s53, s55 // (op 35234) 

s_sub_u32 s55, s16, s52 // (op 35234) 

s_mov_b32 s52, s55 // (op 35234) 
s_mul_hi_i32 s57, s52, s14 // (op 35234) 
s_add_i32 s60, s57, s52 // (op 35234) 


s_ashr_i32 s52, s60, 31 // (op 35234) 
s_cmp_eq_i32 s14, 0 // (op 35234) 
s_mov_b32 s57, scc // (op 35234) 
s_cmp_eq_i32 s57, 0 // (op 35234) 
s_cselect_b32 s61, 0, -1 // (op 35234) 

s_and_b32 s57, s48, 31 // (op 35234) 
s_lshl_b32 s62, 1, s57 // (op 35234) 
s_add_i32 s63, s62, s61 // (op 35234) 


s_and_b32 s61, s52, s63 // (op 35234) 


s_add_i32 s52, s60, s61 // (op 35234) 


s_ashr_i32 s60, s52, s57 // (op 35234) 


s_lshl_b32 s52, s48, 24 // (op 35234) 
s_ashr_i32 s57, s52, 31 // (op 35234) 

s_xor_b32 s52, s60, s57 // (op 35234) 

s_sub_i32 s60, s52, s57 // (op 35234) 


s_mul_i32 s52, s47, s15 // (op 35234) 
s_add_u32 s62, s20, 256 // (op 35234) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 35234) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 35234) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 35234) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 8 // (op 35234) 

s_sub_i32 s57, s62, s52 // (op 35234) 

s_mul_i32 s61, s60, s57 // (op 35234) 

s_sub_u32 s57, s55, s61 // (op 35234) 


s_add_u32 s55, s57, s52 // (op 35234) 


s_cmp_eq_i32 s53, 0 // (op 35234) 
s_mov_b32 s52, scc // (op 35234) 

s_cmp_eq_i32 s52, 0 // (op 35234) 
s_cselect_b32 s53, s55, s19 // (op 35234) 


s_mul_i32 s19, s51, s58 // (op 35234) 


v_lshl_add_u32 v13, s53, 8, v12 // (op 35234) 


s_sub_i32 s51, s18, s19 // (op 35234) 


s_cmp_eq_i32 s52, 0 // (op 35234) 
s_cselect_b32 s53, s60, s51 // (op 35234) 



v_lshl_add_u32 v12, s53, 8, v11 // (op 35234) 


v_mul_lo_u32 v11, v12, s2 // (op 35234) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 35234) 


s_add_u32 s18, s6, 256 // (op 35489) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 35489) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 35489) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 35489) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 8 // (op 35489) 

s_mul_i32 s55, s15, s18 // (op 35489) 
s_mul_i32 s51, s8, s55 // (op 35489) 

s_mul_i32 s55, s56, s51 // (op 35489) 

s_sub_u32 s51, s16, s55 // (op 35489) 

s_mov_b32 s55, s51 // (op 35489) 
s_mul_hi_i32 s52, s55, s14 // (op 35489) 
s_add_i32 s53, s52, s55 // (op 35489) 


s_ashr_i32 s52, s53, 31 // (op 35489) 
s_cmp_eq_i32 s14, 0 // (op 35489) 
s_mov_b32 s55, scc // (op 35489) 
s_cmp_eq_i32 s55, 0 // (op 35489) 
s_cselect_b32 s57, 0, -1 // (op 35489) 

s_and_b32 s55, s48, 31 // (op 35489) 
s_lshl_b32 s58, 1, s55 // (op 35489) 
s_add_i32 s59, s58, s57 // (op 35489) 


s_and_b32 s57, s52, s59 // (op 35489) 


s_add_i32 s52, s53, s57 // (op 35489) 


s_ashr_i32 s53, s52, s55 // (op 35489) 


s_lshl_b32 s52, s48, 24 // (op 35489) 
s_ashr_i32 s55, s52, 31 // (op 35489) 

s_xor_b32 s52, s53, s55 // (op 35489) 

s_sub_i32 s53, s52, s55 // (op 35489) 


s_mul_i32 s52, s47, s15 // (op 35489) 
s_add_u32 s58, s20, 256 // (op 35489) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 35489) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 35489) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 35489) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 35489) 

s_sub_i32 s55, s58, s52 // (op 35489) 

s_mul_i32 s57, s53, s55 // (op 35489) 

s_sub_u32 s55, s51, s57 // (op 35489) 


s_add_u32 s51, s55, s52 // (op 35489) 


s_cmp_eq_i32 s56, 0 // (op 35489) 
s_mov_b32 s52, scc // (op 35489) 

s_cmp_eq_i32 s52, 0 // (op 35489) 
s_cselect_b32 s55, s51, s1 // (op 35489) 


s_mul_i32 s1, s54, s18 // (op 35489) 


v_lshl_add_u32 v11, s55, 8, v14 // (op 35489) 


s_sub_i32 s51, s0, s1 // (op 35489) 


s_cmp_eq_i32 s52, 0 // (op 35489) 
s_cselect_b32 s0, s53, s51 // (op 35489) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 35489) 


v_mul_lo_u32 v10, v13, s4 // (op 35489) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 35489) 


s_mul_i32 s1, s2, s7 // (op 35236) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 35236) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 35236) least significant
s_mul_hi_u32 s19, s2, s6 // (op 35236) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35236) most significant: sum
s_add_u32 s1, s1, s19 // (op 35236) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 35236) 

s_mul_i32 s1, s4, s7 // (op 35491) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 35491) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 35491) least significant
s_mul_hi_u32 s52, s4, s6 // (op 35491) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 35491) most significant: sum
s_add_u32 s1, s1, s52 // (op 35491) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 35491) 

s_mov_b32 s56, s22 // (op 35236) reg expression
s_mov_b32 s57, s23 // (op 35236) reg expression
s_mov_b32 s58, s18 // (op 35236) reg expression
s_mov_b32 s59, 131072 // (op 35236) call()

s_mov_b32 s60, s24 // (op 35491) reg expression

s_mov_b32 s61, s25 // (op 35491) reg expression
s_mov_b32 s62, s52 // (op 35491) reg expression
s_mov_b32 s63, 131072 // (op 35491) call()


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
v_accvgpr_write a64, 0.00000 // (op 3286) call()
v_accvgpr_write a65, 0.00000 // (op 3286) call()
v_accvgpr_write a66, 0.00000 // (op 3286) call()
v_accvgpr_write a67, 0.00000 // (op 3286) call()
v_accvgpr_write a68, 0.00000 // (op 3903) call()
v_accvgpr_write a69, 0.00000 // (op 3903) call()
v_accvgpr_write a70, 0.00000 // (op 3903) call()
v_accvgpr_write a71, 0.00000 // (op 3903) call()
v_accvgpr_write a72, 0.00000 // (op 3904) call()
v_accvgpr_write a73, 0.00000 // (op 3904) call()
v_accvgpr_write a74, 0.00000 // (op 3904) call()
v_accvgpr_write a75, 0.00000 // (op 3904) call()
v_accvgpr_write a76, 0.00000 // (op 3905) call()
v_accvgpr_write a77, 0.00000 // (op 3905) call()
v_accvgpr_write a78, 0.00000 // (op 3905) call()
v_accvgpr_write a79, 0.00000 // (op 3905) call()
v_accvgpr_write a80, 0.00000 // (op 3906) call()
v_accvgpr_write a81, 0.00000 // (op 3906) call()
v_accvgpr_write a82, 0.00000 // (op 3906) call()
v_accvgpr_write a83, 0.00000 // (op 3906) call()
v_accvgpr_write a84, 0.00000 // (op 3907) call()
v_accvgpr_write a85, 0.00000 // (op 3907) call()
v_accvgpr_write a86, 0.00000 // (op 3907) call()
v_accvgpr_write a87, 0.00000 // (op 3907) call()
v_accvgpr_write a88, 0.00000 // (op 3908) call()
v_accvgpr_write a89, 0.00000 // (op 3908) call()
v_accvgpr_write a90, 0.00000 // (op 3908) call()
v_accvgpr_write a91, 0.00000 // (op 3908) call()
v_accvgpr_write a92, 0.00000 // (op 3909) call()
v_accvgpr_write a93, 0.00000 // (op 3909) call()
v_accvgpr_write a94, 0.00000 // (op 3909) call()
v_accvgpr_write a95, 0.00000 // (op 3909) call()
v_accvgpr_write a96, 0.00000 // (op 4824) call()
v_accvgpr_write a97, 0.00000 // (op 4824) call()
v_accvgpr_write a98, 0.00000 // (op 4824) call()
v_accvgpr_write a99, 0.00000 // (op 4824) call()
v_accvgpr_write a100, 0.00000 // (op 5441) call()
v_accvgpr_write a101, 0.00000 // (op 5441) call()
v_accvgpr_write a102, 0.00000 // (op 5441) call()
v_accvgpr_write a103, 0.00000 // (op 5441) call()
v_accvgpr_write a104, 0.00000 // (op 5442) call()
v_accvgpr_write a105, 0.00000 // (op 5442) call()
v_accvgpr_write a106, 0.00000 // (op 5442) call()
v_accvgpr_write a107, 0.00000 // (op 5442) call()
v_accvgpr_write a108, 0.00000 // (op 5443) call()
v_accvgpr_write a109, 0.00000 // (op 5443) call()
v_accvgpr_write a110, 0.00000 // (op 5443) call()
v_accvgpr_write a111, 0.00000 // (op 5443) call()
v_accvgpr_write a112, 0.00000 // (op 5444) call()
v_accvgpr_write a113, 0.00000 // (op 5444) call()
v_accvgpr_write a114, 0.00000 // (op 5444) call()
v_accvgpr_write a115, 0.00000 // (op 5444) call()
v_accvgpr_write a116, 0.00000 // (op 5445) call()
v_accvgpr_write a117, 0.00000 // (op 5445) call()
v_accvgpr_write a118, 0.00000 // (op 5445) call()
v_accvgpr_write a119, 0.00000 // (op 5445) call()
v_accvgpr_write a120, 0.00000 // (op 5446) call()
v_accvgpr_write a121, 0.00000 // (op 5446) call()
v_accvgpr_write a122, 0.00000 // (op 5446) call()
v_accvgpr_write a123, 0.00000 // (op 5446) call()
v_accvgpr_write a124, 0.00000 // (op 5447) call()
v_accvgpr_write a125, 0.00000 // (op 5447) call()
v_accvgpr_write a126, 0.00000 // (op 5447) call()
v_accvgpr_write a127, 0.00000 // (op 5447) call()
v_accvgpr_write a128, 0.00000 // (op 6362) call()
v_accvgpr_write a129, 0.00000 // (op 6362) call()
v_accvgpr_write a130, 0.00000 // (op 6362) call()
v_accvgpr_write a131, 0.00000 // (op 6362) call()
v_accvgpr_write a132, 0.00000 // (op 6979) call()
v_accvgpr_write a133, 0.00000 // (op 6979) call()
v_accvgpr_write a134, 0.00000 // (op 6979) call()
v_accvgpr_write a135, 0.00000 // (op 6979) call()
v_accvgpr_write a136, 0.00000 // (op 6980) call()
v_accvgpr_write a137, 0.00000 // (op 6980) call()
v_accvgpr_write a138, 0.00000 // (op 6980) call()
v_accvgpr_write a139, 0.00000 // (op 6980) call()
v_accvgpr_write a140, 0.00000 // (op 6981) call()
v_accvgpr_write a141, 0.00000 // (op 6981) call()
v_accvgpr_write a142, 0.00000 // (op 6981) call()
v_accvgpr_write a143, 0.00000 // (op 6981) call()
v_accvgpr_write a144, 0.00000 // (op 6982) call()
v_accvgpr_write a145, 0.00000 // (op 6982) call()
v_accvgpr_write a146, 0.00000 // (op 6982) call()
v_accvgpr_write a147, 0.00000 // (op 6982) call()
v_accvgpr_write a148, 0.00000 // (op 6983) call()
v_accvgpr_write a149, 0.00000 // (op 6983) call()
v_accvgpr_write a150, 0.00000 // (op 6983) call()
v_accvgpr_write a151, 0.00000 // (op 6983) call()
v_accvgpr_write a152, 0.00000 // (op 6984) call()
v_accvgpr_write a153, 0.00000 // (op 6984) call()
v_accvgpr_write a154, 0.00000 // (op 6984) call()
v_accvgpr_write a155, 0.00000 // (op 6984) call()
v_accvgpr_write a156, 0.00000 // (op 6985) call()
v_accvgpr_write a157, 0.00000 // (op 6985) call()
v_accvgpr_write a158, 0.00000 // (op 6985) call()
v_accvgpr_write a159, 0.00000 // (op 6985) call()
v_accvgpr_write a160, 0.00000 // (op 7900) call()
v_accvgpr_write a161, 0.00000 // (op 7900) call()
v_accvgpr_write a162, 0.00000 // (op 7900) call()
v_accvgpr_write a163, 0.00000 // (op 7900) call()
v_accvgpr_write a164, 0.00000 // (op 8517) call()
v_accvgpr_write a165, 0.00000 // (op 8517) call()
v_accvgpr_write a166, 0.00000 // (op 8517) call()
v_accvgpr_write a167, 0.00000 // (op 8517) call()
v_accvgpr_write a168, 0.00000 // (op 8518) call()
v_accvgpr_write a169, 0.00000 // (op 8518) call()
v_accvgpr_write a170, 0.00000 // (op 8518) call()
v_accvgpr_write a171, 0.00000 // (op 8518) call()
v_accvgpr_write a172, 0.00000 // (op 8519) call()
v_accvgpr_write a173, 0.00000 // (op 8519) call()
v_accvgpr_write a174, 0.00000 // (op 8519) call()
v_accvgpr_write a175, 0.00000 // (op 8519) call()
v_accvgpr_write a176, 0.00000 // (op 8520) call()
v_accvgpr_write a177, 0.00000 // (op 8520) call()
v_accvgpr_write a178, 0.00000 // (op 8520) call()
v_accvgpr_write a179, 0.00000 // (op 8520) call()
v_accvgpr_write a180, 0.00000 // (op 8521) call()
v_accvgpr_write a181, 0.00000 // (op 8521) call()
v_accvgpr_write a182, 0.00000 // (op 8521) call()
v_accvgpr_write a183, 0.00000 // (op 8521) call()
v_accvgpr_write a184, 0.00000 // (op 8522) call()
v_accvgpr_write a185, 0.00000 // (op 8522) call()
v_accvgpr_write a186, 0.00000 // (op 8522) call()
v_accvgpr_write a187, 0.00000 // (op 8522) call()
v_accvgpr_write a188, 0.00000 // (op 8523) call()
v_accvgpr_write a189, 0.00000 // (op 8523) call()
v_accvgpr_write a190, 0.00000 // (op 8523) call()
v_accvgpr_write a191, 0.00000 // (op 8523) call()
v_accvgpr_write a192, 0.00000 // (op 9438) call()
v_accvgpr_write a193, 0.00000 // (op 9438) call()
v_accvgpr_write a194, 0.00000 // (op 9438) call()
v_accvgpr_write a195, 0.00000 // (op 9438) call()
v_accvgpr_write a196, 0.00000 // (op 10055) call()
v_accvgpr_write a197, 0.00000 // (op 10055) call()
v_accvgpr_write a198, 0.00000 // (op 10055) call()
v_accvgpr_write a199, 0.00000 // (op 10055) call()
v_accvgpr_write a200, 0.00000 // (op 10056) call()
v_accvgpr_write a201, 0.00000 // (op 10056) call()
v_accvgpr_write a202, 0.00000 // (op 10056) call()
v_accvgpr_write a203, 0.00000 // (op 10056) call()
v_accvgpr_write a204, 0.00000 // (op 10057) call()
v_accvgpr_write a205, 0.00000 // (op 10057) call()
v_accvgpr_write a206, 0.00000 // (op 10057) call()
v_accvgpr_write a207, 0.00000 // (op 10057) call()
v_accvgpr_write a208, 0.00000 // (op 10058) call()
v_accvgpr_write a209, 0.00000 // (op 10058) call()
v_accvgpr_write a210, 0.00000 // (op 10058) call()
v_accvgpr_write a211, 0.00000 // (op 10058) call()
v_accvgpr_write a212, 0.00000 // (op 10059) call()
v_accvgpr_write a213, 0.00000 // (op 10059) call()
v_accvgpr_write a214, 0.00000 // (op 10059) call()
v_accvgpr_write a215, 0.00000 // (op 10059) call()
v_accvgpr_write a216, 0.00000 // (op 10060) call()
v_accvgpr_write a217, 0.00000 // (op 10060) call()
v_accvgpr_write a218, 0.00000 // (op 10060) call()
v_accvgpr_write a219, 0.00000 // (op 10060) call()
v_accvgpr_write a220, 0.00000 // (op 10061) call()
v_accvgpr_write a221, 0.00000 // (op 10061) call()
v_accvgpr_write a222, 0.00000 // (op 10061) call()
v_accvgpr_write a223, 0.00000 // (op 10061) call()
v_accvgpr_write a224, 0.00000 // (op 10976) call()
v_accvgpr_write a225, 0.00000 // (op 10976) call()
v_accvgpr_write a226, 0.00000 // (op 10976) call()
v_accvgpr_write a227, 0.00000 // (op 10976) call()
v_accvgpr_write a228, 0.00000 // (op 11593) call()
v_accvgpr_write a229, 0.00000 // (op 11593) call()
v_accvgpr_write a230, 0.00000 // (op 11593) call()
v_accvgpr_write a231, 0.00000 // (op 11593) call()
v_accvgpr_write a232, 0.00000 // (op 11594) call()
v_accvgpr_write a233, 0.00000 // (op 11594) call()
v_accvgpr_write a234, 0.00000 // (op 11594) call()
v_accvgpr_write a235, 0.00000 // (op 11594) call()
v_accvgpr_write a236, 0.00000 // (op 11595) call()
v_accvgpr_write a237, 0.00000 // (op 11595) call()
v_accvgpr_write a238, 0.00000 // (op 11595) call()
v_accvgpr_write a239, 0.00000 // (op 11595) call()
v_accvgpr_write a240, 0.00000 // (op 11596) call()
v_accvgpr_write a241, 0.00000 // (op 11596) call()
v_accvgpr_write a242, 0.00000 // (op 11596) call()
v_accvgpr_write a243, 0.00000 // (op 11596) call()
v_accvgpr_write a244, 0.00000 // (op 11597) call()
v_accvgpr_write a245, 0.00000 // (op 11597) call()
v_accvgpr_write a246, 0.00000 // (op 11597) call()
v_accvgpr_write a247, 0.00000 // (op 11597) call()
v_accvgpr_write a248, 0.00000 // (op 11598) call()
v_accvgpr_write a249, 0.00000 // (op 11598) call()
v_accvgpr_write a250, 0.00000 // (op 11598) call()
v_accvgpr_write a251, 0.00000 // (op 11598) call()
v_accvgpr_write a252, 0.00000 // (op 11599) call()
v_accvgpr_write a253, 0.00000 // (op 11599) call()
v_accvgpr_write a254, 0.00000 // (op 11599) call()
v_accvgpr_write a255, 0.00000 // (op 11599) call()
s_mul_hi_i32 s0, s16, s10 // (op 34996) 
v_and_b32 v14, -8, v1 // (op 34996) 
s_add_i32 s1, s0, s16 // (op 34996) 

v_sub_u32 v15, v1, v14 // (op 34996) 

s_ashr_i32 s0, s1, 31 // (op 34996) 
v_lshlrev_b32 v14, 5, v15 // (op 34996) 

s_cmp_eq_i32 s10, 0 // (op 34996) 
s_mov_b32 s18, scc // (op 34996) 
v_lshrrev_b32 v15, 3, v1 // (op 34996) 
s_cmp_eq_i32 s18, 0 // (op 34996) 
s_cselect_b32 s19, 0, -1 // (op 34996) 

s_and_b32 s18, s11, 31 // (op 34996) 
s_lshl_b32 s22, 1, s18 // (op 34996) 
s_add_i32 s23, s22, s19 // (op 34996) 


s_and_b32 s19, s0, s23 // (op 34996) 


s_add_i32 s0, s1, s19 // (op 34996) 


s_ashr_i32 s1, s0, s18 // (op 34996) 


s_lshl_b32 s0, s11, 24 // (op 34996) 
s_ashr_i32 s18, s0, 31 // (op 34996) 

s_xor_b32 s0, s1, s18 // (op 34996) 

s_sub_i32 s1, s0, s18 // (op 34996) 


s_mul_hi_i32 s0, s1, s12 // (op 34996) 
s_add_i32 s18, s0, s1 // (op 34996) 

s_ashr_i32 s0, s18, 31 // (op 34996) 
s_cmp_eq_i32 s12, 0 // (op 34996) 
s_mov_b32 s19, scc // (op 34996) 
s_cmp_eq_i32 s19, 0 // (op 34996) 
s_cselect_b32 s22, 0, -1 // (op 34996) 

s_and_b32 s19, s13, 31 // (op 34996) 
s_lshl_b32 s23, 1, s19 // (op 34996) 
s_add_i32 s24, s23, s22 // (op 34996) 


s_and_b32 s25, s0, s24 // (op 34996) 


s_add_i32 s0, s18, s25 // (op 34996) 


s_ashr_i32 s18, s0, s19 // (op 34996) 


s_lshl_b32 s0, s13, 24 // (op 34996) 
s_ashr_i32 s19, s0, 31 // (op 34996) 

s_xor_b32 s0, s18, s19 // (op 34996) 

s_sub_i32 s18, s0, s19 // (op 34996) 


s_mul_i32 s0, s1, s15 // (op 34996) 

s_sub_u32 s1, s16, s0 // (op 34996) 

s_mul_i32 s0, s18, s15 // (op 34996) 
s_add_u32 s0, s0, s1 // (op 34996) 


s_mul_hi_i32 s1, s16, s9 // (op 34996) 
s_add_i32 s18, s1, s16 // (op 34996) 

s_ashr_i32 s1, s18, 31 // (op 34996) 
s_cmp_eq_i32 s9, 0 // (op 34996) 
s_mov_b32 s19, scc // (op 34996) 
s_cmp_eq_i32 s19, 0 // (op 34996) 
s_cselect_b32 s22, 0, -1 // (op 34996) 

s_and_b32 s19, s46, 31 // (op 34996) 
s_lshl_b32 s23, 1, s19 // (op 34996) 
s_add_i32 s24, s23, s22 // (op 34996) 


s_and_b32 s25, s1, s24 // (op 34996) 


s_add_i32 s1, s18, s25 // (op 34996) 


s_ashr_i32 s18, s1, s19 // (op 34996) 


s_lshl_b32 s1, s46, 24 // (op 34996) 
s_ashr_i32 s19, s1, 31 // (op 34996) 

s_xor_b32 s1, s18, s19 // (op 34996) 

s_sub_i32 s18, s1, s19 // (op 34996) 


s_add_u32 s22, s6, 256 // (op 34996) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 34996) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 34996) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 34996) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 34996) 

s_mul_i32 s1, s15, s22 // (op 34996) 

s_mul_i32 s19, s8, s1 // (op 34996) 

s_mul_i32 s1, s18, s19 // (op 34996) 

s_sub_u32 s19, s16, s1 // (op 34996) 

s_mov_b32 s1, s19 // (op 34996) 
s_mul_hi_i32 s22, s1, s14 // (op 34996) 
s_add_i32 s23, s22, s1 // (op 34996) 


s_ashr_i32 s1, s23, 31 // (op 34996) 
s_cmp_eq_i32 s14, 0 // (op 34996) 
s_mov_b32 s22, scc // (op 34996) 
s_cmp_eq_i32 s22, 0 // (op 34996) 
s_cselect_b32 s24, 0, -1 // (op 34996) 

s_and_b32 s22, s48, 31 // (op 34996) 
s_lshl_b32 s25, 1, s22 // (op 34996) 
s_add_i32 s49, s25, s24 // (op 34996) 


s_and_b32 s24, s1, s49 // (op 34996) 


s_add_i32 s1, s23, s24 // (op 34996) 


s_ashr_i32 s23, s1, s22 // (op 34996) 


s_lshl_b32 s1, s48, 24 // (op 34996) 
s_ashr_i32 s22, s1, 31 // (op 34996) 

s_xor_b32 s1, s23, s22 // (op 34996) 

s_sub_i32 s23, s1, s22 // (op 34996) 


s_mul_i32 s1, s47, s15 // (op 34996) 
s_add_u32 s24, s20, 256 // (op 34996) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34996) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 34996) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 34996) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 8 // (op 34996) 

s_sub_i32 s22, s24, s1 // (op 34996) 

s_mul_i32 s24, s23, s22 // (op 34996) 


s_sub_u32 s25, s19, s24 // (op 34996) 


s_add_u32 s19, s25, s1 // (op 34996) 


s_cmp_eq_i32 s18, 0 // (op 34996) 
s_mov_b32 s1, scc // (op 34996) 

s_cmp_eq_i32 s1, 0 // (op 34996) 
s_cselect_b32 s18, s19, s0 // (op 34996) 



v_lshl_add_u32 v16, s18, 8, v14 // (op 34996) 


v_mov_b32 v19, 0 // (op 34996) convert
v_mov_b32 v18, v16 // (op 34996) convert

v_mul_lo_u32 v17, v15, s27 // (op 34996) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 34996) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 34996) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 34996) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 34996) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 34996) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 34996) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 34996) 

s_mul_i32 s1, s26, s31 // (op 34998) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 34998) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 34998) least significant
s_mul_hi_u32 s19, s26, s30 // (op 34998) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34998) most significant: sum
s_add_u32 s1, s1, s19 // (op 34998) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 34998) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 34998) 

s_mov_b32 s52, s28 // (op 34998) reg expression
s_mov_b32 s53, s29 // (op 34998) reg expression
s_mov_b32 s54, s0 // (op 34998) reg expression
s_mov_b32 s55, 131072 // (op 34998) call()

s_mul_hi_i32 s0, s16, s10 // (op 35015) 
v_and_b32 v14, -8, v1 // (op 35015) 
s_add_i32 s1, s0, s16 // (op 35015) 

v_sub_u32 v15, v1, v14 // (op 35015) 

s_ashr_i32 s0, s1, 31 // (op 35015) 
v_lshlrev_b32 v14, 5, v15 // (op 35015) 

s_cmp_eq_i32 s10, 0 // (op 35015) 
s_mov_b32 s18, scc // (op 35015) 
v_lshrrev_b32 v15, 3, v1 // (op 35015) 
s_cmp_eq_i32 s18, 0 // (op 35015) 
s_cselect_b32 s19, 0, -1 // (op 35015) 

s_and_b32 s18, s11, 31 // (op 35015) 
s_lshl_b32 s22, 1, s18 // (op 35015) 
s_add_i32 s23, s22, s19 // (op 35015) 


s_and_b32 s19, s0, s23 // (op 35015) 


s_add_i32 s0, s1, s19 // (op 35015) 


s_ashr_i32 s1, s0, s18 // (op 35015) 


s_lshl_b32 s0, s11, 24 // (op 35015) 
s_ashr_i32 s18, s0, 31 // (op 35015) 

s_xor_b32 s0, s1, s18 // (op 35015) 

s_sub_i32 s1, s0, s18 // (op 35015) 


s_mul_hi_i32 s0, s1, s12 // (op 35015) 
s_add_i32 s18, s0, s1 // (op 35015) 

s_ashr_i32 s0, s18, 31 // (op 35015) 
s_cmp_eq_i32 s12, 0 // (op 35015) 
s_mov_b32 s19, scc // (op 35015) 
s_cmp_eq_i32 s19, 0 // (op 35015) 
s_cselect_b32 s22, 0, -1 // (op 35015) 

s_and_b32 s19, s13, 31 // (op 35015) 
s_lshl_b32 s23, 1, s19 // (op 35015) 
s_add_i32 s24, s23, s22 // (op 35015) 


s_and_b32 s25, s0, s24 // (op 35015) 


s_add_i32 s0, s18, s25 // (op 35015) 


s_ashr_i32 s18, s0, s19 // (op 35015) 


s_lshl_b32 s0, s13, 24 // (op 35015) 
s_ashr_i32 s19, s0, 31 // (op 35015) 

s_xor_b32 s0, s18, s19 // (op 35015) 

s_sub_i32 s18, s0, s19 // (op 35015) 


s_mul_i32 s0, s1, s15 // (op 35015) 

s_sub_u32 s1, s16, s0 // (op 35015) 

s_mul_i32 s0, s18, s15 // (op 35015) 
s_add_u32 s0, s0, s1 // (op 35015) 


s_mul_hi_i32 s1, s16, s9 // (op 35015) 
s_add_i32 s18, s1, s16 // (op 35015) 

s_ashr_i32 s1, s18, 31 // (op 35015) 
s_cmp_eq_i32 s9, 0 // (op 35015) 
s_mov_b32 s19, scc // (op 35015) 
s_cmp_eq_i32 s19, 0 // (op 35015) 
s_cselect_b32 s22, 0, -1 // (op 35015) 

s_and_b32 s19, s46, 31 // (op 35015) 
s_lshl_b32 s23, 1, s19 // (op 35015) 
s_add_i32 s24, s23, s22 // (op 35015) 


s_and_b32 s25, s1, s24 // (op 35015) 


s_add_i32 s1, s18, s25 // (op 35015) 


s_ashr_i32 s18, s1, s19 // (op 35015) 


s_lshl_b32 s1, s46, 24 // (op 35015) 
s_ashr_i32 s19, s1, 31 // (op 35015) 

s_xor_b32 s1, s18, s19 // (op 35015) 

s_sub_i32 s18, s1, s19 // (op 35015) 


s_add_u32 s50, s6, 256 // (op 35015) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35015) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35015) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35015) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35015) 

s_mul_i32 s1, s15, s24 // (op 35015) 

s_mul_i32 s19, s8, s1 // (op 35015) 

s_mul_i32 s1, s18, s19 // (op 35015) 

s_sub_u32 s19, s16, s1 // (op 35015) 

s_mov_b32 s1, s19 // (op 35015) 
s_mul_hi_i32 s22, s1, s14 // (op 35015) 
s_add_i32 s23, s22, s1 // (op 35015) 


s_ashr_i32 s1, s23, 31 // (op 35015) 
s_cmp_eq_i32 s14, 0 // (op 35015) 
s_mov_b32 s22, scc // (op 35015) 
s_cmp_eq_i32 s22, 0 // (op 35015) 
s_cselect_b32 s24, 0, -1 // (op 35015) 

s_and_b32 s22, s48, 31 // (op 35015) 
s_lshl_b32 s25, 1, s22 // (op 35015) 
s_add_i32 s49, s25, s24 // (op 35015) 


s_and_b32 s24, s1, s49 // (op 35015) 


s_add_i32 s1, s23, s24 // (op 35015) 


s_ashr_i32 s23, s1, s22 // (op 35015) 


s_lshl_b32 s1, s48, 24 // (op 35015) 
s_ashr_i32 s22, s1, 31 // (op 35015) 

s_xor_b32 s1, s23, s22 // (op 35015) 

s_sub_i32 s23, s1, s22 // (op 35015) 


s_mul_i32 s1, s47, s15 // (op 35015) 
s_add_u32 s24, s20, 256 // (op 35015) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 35015) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 35015) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 35015) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 35015) 

s_sub_i32 s22, s24, s1 // (op 35015) 

s_mul_i32 s24, s23, s22 // (op 35015) 


s_sub_u32 s25, s19, s24 // (op 35015) 


s_add_u32 s19, s25, s1 // (op 35015) 


s_cmp_eq_i32 s18, 0 // (op 35015) 
s_mov_b32 s1, scc // (op 35015) 

s_cmp_eq_i32 s1, 0 // (op 35015) 
s_cselect_b32 s18, s19, s0 // (op 35015) 



v_lshl_add_u32 v18, s18, 8, v14 // (op 35015) 


v_mov_b32 v21, 0 // (op 35015) convert
v_mov_b32 v20, v18 // (op 35015) convert

v_mul_lo_u32 v19, v15, s27 // (op 35015) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 35015) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 35015) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 35015) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v18 // (op 35015) least significant half
v_addc_co_u32 v15, s[0:1], v21, v19, s[0:1] // (op 35015) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 35015) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 35015) 

s_mul_i32 s1, s26, s31 // (op 35017) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 35017) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 35017) least significant
s_mul_hi_u32 s19, s26, s30 // (op 35017) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35017) most significant: sum
s_add_u32 s1, s1, s19 // (op 35017) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 35017) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 35017) 

s_mov_b32 s52, s28 // (op 35017) reg expression
s_mov_b32 s53, s29 // (op 35017) reg expression
s_mov_b32 s54, s0 // (op 35017) reg expression
s_mov_b32 s55, 131072 // (op 35017) call()

s_mul_hi_i32 s0, s16, s10 // (op 35060) 
v_lshrrev_b32 v14, 3, v1 // (op 35060) 
s_add_i32 s1, s0, s16 // (op 35060) 

v_and_b32 v15, -8, v1 // (op 35060) 
s_ashr_i32 s0, s1, 31 // (op 35060) 
v_sub_u32 v20, v1, v15 // (op 35060) 

s_cmp_eq_i32 s10, 0 // (op 35060) 
s_mov_b32 s18, scc // (op 35060) 
v_lshlrev_b32 v15, 4, v20 // (op 35060) 

s_cmp_eq_i32 s18, 0 // (op 35060) 
s_cselect_b32 s19, 0, -1 // (op 35060) 

v_mov_b32 v21, 0 // (op 35060) convert
v_mov_b32 v20, v15 // (op 35060) convert

s_and_b32 s18, s11, 31 // (op 35060) 
s_lshl_b32 s22, 1, s18 // (op 35060) 
s_add_i32 s23, s22, s19 // (op 35060) 


s_and_b32 s19, s0, s23 // (op 35060) 


s_add_i32 s0, s1, s19 // (op 35060) 


s_ashr_i32 s1, s0, s18 // (op 35060) 


s_lshl_b32 s0, s11, 24 // (op 35060) 
s_ashr_i32 s18, s0, 31 // (op 35060) 

s_xor_b32 s0, s1, s18 // (op 35060) 

s_sub_i32 s1, s0, s18 // (op 35060) 


s_mul_hi_i32 s0, s1, s12 // (op 35060) 
s_add_i32 s18, s0, s1 // (op 35060) 

s_ashr_i32 s0, s18, 31 // (op 35060) 
s_cmp_eq_i32 s12, 0 // (op 35060) 
s_mov_b32 s19, scc // (op 35060) 
s_cmp_eq_i32 s19, 0 // (op 35060) 
s_cselect_b32 s22, 0, -1 // (op 35060) 

s_and_b32 s19, s13, 31 // (op 35060) 
s_lshl_b32 s23, 1, s19 // (op 35060) 
s_add_i32 s24, s23, s22 // (op 35060) 


s_and_b32 s25, s0, s24 // (op 35060) 


s_add_i32 s0, s18, s25 // (op 35060) 


s_ashr_i32 s18, s0, s19 // (op 35060) 


s_lshl_b32 s0, s13, 24 // (op 35060) 
s_ashr_i32 s19, s0, 31 // (op 35060) 

s_xor_b32 s0, s18, s19 // (op 35060) 

s_sub_i32 s18, s0, s19 // (op 35060) 


s_add_u32 s50, s6, 256 // (op 35060) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35060) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35060) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35060) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35060) 

s_mul_i32 s0, s18, s24 // (op 35060) 

s_sub_i32 s18, s1, s0 // (op 35060) 


s_mul_hi_i32 s19, s16, s9 // (op 35060) 
s_add_i32 s0, s19, s16 // (op 35060) 

s_ashr_i32 s1, s0, 31 // (op 35060) 
s_cmp_eq_i32 s9, 0 // (op 35060) 
s_mov_b32 s19, scc // (op 35060) 
s_cmp_eq_i32 s19, 0 // (op 35060) 
s_cselect_b32 s22, 0, -1 // (op 35060) 

s_and_b32 s19, s46, 31 // (op 35060) 
s_lshl_b32 s23, 1, s19 // (op 35060) 
s_add_i32 s49, s23, s22 // (op 35060) 


s_and_b32 s22, s1, s49 // (op 35060) 


s_add_i32 s1, s0, s22 // (op 35060) 


s_ashr_i32 s0, s1, s19 // (op 35060) 


s_lshl_b32 s1, s46, 24 // (op 35060) 
s_ashr_i32 s19, s1, 31 // (op 35060) 

s_xor_b32 s1, s0, s19 // (op 35060) 

s_sub_i32 s0, s1, s19 // (op 35060) 


s_mul_i32 s1, s15, s24 // (op 35060) 

s_mul_i32 s19, s8, s1 // (op 35060) 

s_mul_i32 s1, s0, s19 // (op 35060) 

s_sub_u32 s19, s16, s1 // (op 35060) 

s_mul_hi_i32 s1, s19, s14 // (op 35060) 
s_add_i32 s22, s1, s19 // (op 35060) 


s_ashr_i32 s1, s22, 31 // (op 35060) 
s_cmp_eq_i32 s14, 0 // (op 35060) 
s_mov_b32 s19, scc // (op 35060) 
s_cmp_eq_i32 s19, 0 // (op 35060) 
s_cselect_b32 s23, 0, -1 // (op 35060) 

s_and_b32 s19, s48, 31 // (op 35060) 
s_lshl_b32 s24, 1, s19 // (op 35060) 
s_add_i32 s25, s24, s23 // (op 35060) 


s_and_b32 s23, s1, s25 // (op 35060) 


s_add_i32 s1, s22, s23 // (op 35060) 


s_ashr_i32 s22, s1, s19 // (op 35060) 


s_lshl_b32 s1, s48, 24 // (op 35060) 
s_ashr_i32 s19, s1, 31 // (op 35060) 

s_xor_b32 s1, s22, s19 // (op 35060) 

s_sub_i32 s22, s1, s19 // (op 35060) 


s_cmp_eq_i32 s0, 0 // (op 35060) 
s_mov_b32 s1, scc // (op 35060) 

s_cmp_eq_i32 s1, 0 // (op 35060) 
s_cselect_b32 s0, s22, s18 // (op 35060) 



v_lshl_add_u32 v15, s0, 8, v14 // (op 35060) 


v_mul_lo_u32 v23, v15, s33 // (op 35060) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 35060) most significant: high of low * low
v_mul_lo_u32 v22, v15, s32 // (op 35060) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 35060) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 35060) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 35060) most significant half



s_mul_i32 s1, s32, s7 // (op 35062) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 35062) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 35062) least significant
s_mul_hi_u32 s19, s32, s6 // (op 35062) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35062) most significant: sum
s_add_u32 s1, s1, s19 // (op 35062) most significant: sum


s_mov_b32 s64, s34 // (op 35062) reg expression
s_mov_b32 s65, s35 // (op 35062) reg expression
s_mov_b32 s66, s0 // (op 35062) reg expression
s_mov_b32 s67, 131072 // (op 35062) call()

s_mul_hi_i32 s0, s16, s10 // (op 35079) 
v_lshrrev_b32 v20, 3, v1 // (op 35079) 
s_add_i32 s1, s0, s16 // (op 35079) 

v_and_b32 v21, -8, v1 // (op 35079) 
s_ashr_i32 s0, s1, 31 // (op 35079) 
v_sub_u32 v22, v1, v21 // (op 35079) 

s_cmp_eq_i32 s10, 0 // (op 35079) 
s_mov_b32 s18, scc // (op 35079) 
v_lshlrev_b32 v21, 4, v22 // (op 35079) 

s_cmp_eq_i32 s18, 0 // (op 35079) 
s_cselect_b32 s19, 0, -1 // (op 35079) 

v_mov_b32 v23, 0 // (op 35079) convert
v_mov_b32 v22, v21 // (op 35079) convert

s_and_b32 s18, s11, 31 // (op 35079) 
s_lshl_b32 s22, 1, s18 // (op 35079) 
s_add_i32 s23, s22, s19 // (op 35079) 


s_and_b32 s19, s0, s23 // (op 35079) 


s_add_i32 s0, s1, s19 // (op 35079) 


s_ashr_i32 s1, s0, s18 // (op 35079) 


s_lshl_b32 s0, s11, 24 // (op 35079) 
s_ashr_i32 s18, s0, 31 // (op 35079) 

s_xor_b32 s0, s1, s18 // (op 35079) 

s_sub_i32 s1, s0, s18 // (op 35079) 


s_mul_hi_i32 s0, s1, s12 // (op 35079) 
s_add_i32 s18, s0, s1 // (op 35079) 

s_ashr_i32 s0, s18, 31 // (op 35079) 
s_cmp_eq_i32 s12, 0 // (op 35079) 
s_mov_b32 s19, scc // (op 35079) 
s_cmp_eq_i32 s19, 0 // (op 35079) 
s_cselect_b32 s22, 0, -1 // (op 35079) 

s_and_b32 s19, s13, 31 // (op 35079) 
s_lshl_b32 s23, 1, s19 // (op 35079) 
s_add_i32 s24, s23, s22 // (op 35079) 


s_and_b32 s25, s0, s24 // (op 35079) 


s_add_i32 s0, s18, s25 // (op 35079) 


s_ashr_i32 s18, s0, s19 // (op 35079) 


s_lshl_b32 s0, s13, 24 // (op 35079) 
s_ashr_i32 s19, s0, 31 // (op 35079) 

s_xor_b32 s0, s18, s19 // (op 35079) 

s_sub_i32 s18, s0, s19 // (op 35079) 


s_add_u32 s50, s6, 256 // (op 35079) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35079) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35079) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35079) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35079) 

s_mul_i32 s0, s18, s24 // (op 35079) 

s_sub_i32 s18, s1, s0 // (op 35079) 


s_mul_hi_i32 s19, s16, s9 // (op 35079) 
s_add_i32 s0, s19, s16 // (op 35079) 

s_ashr_i32 s1, s0, 31 // (op 35079) 
s_cmp_eq_i32 s9, 0 // (op 35079) 
s_mov_b32 s19, scc // (op 35079) 
s_cmp_eq_i32 s19, 0 // (op 35079) 
s_cselect_b32 s22, 0, -1 // (op 35079) 

s_and_b32 s19, s46, 31 // (op 35079) 
s_lshl_b32 s23, 1, s19 // (op 35079) 
s_add_i32 s49, s23, s22 // (op 35079) 


s_and_b32 s22, s1, s49 // (op 35079) 


s_add_i32 s1, s0, s22 // (op 35079) 


s_ashr_i32 s0, s1, s19 // (op 35079) 


s_lshl_b32 s1, s46, 24 // (op 35079) 
s_ashr_i32 s19, s1, 31 // (op 35079) 

s_xor_b32 s1, s0, s19 // (op 35079) 

s_sub_i32 s0, s1, s19 // (op 35079) 


s_mul_i32 s1, s15, s24 // (op 35079) 

s_mul_i32 s19, s8, s1 // (op 35079) 

s_mul_i32 s1, s0, s19 // (op 35079) 

s_sub_u32 s19, s16, s1 // (op 35079) 

s_mul_hi_i32 s1, s19, s14 // (op 35079) 
s_add_i32 s22, s1, s19 // (op 35079) 


s_ashr_i32 s1, s22, 31 // (op 35079) 
s_cmp_eq_i32 s14, 0 // (op 35079) 
s_mov_b32 s19, scc // (op 35079) 
s_cmp_eq_i32 s19, 0 // (op 35079) 
s_cselect_b32 s23, 0, -1 // (op 35079) 

s_and_b32 s19, s48, 31 // (op 35079) 
s_lshl_b32 s24, 1, s19 // (op 35079) 
s_add_i32 s25, s24, s23 // (op 35079) 


s_and_b32 s23, s1, s25 // (op 35079) 


s_add_i32 s1, s22, s23 // (op 35079) 


s_ashr_i32 s22, s1, s19 // (op 35079) 


s_lshl_b32 s1, s48, 24 // (op 35079) 
s_ashr_i32 s19, s1, 31 // (op 35079) 

s_xor_b32 s1, s22, s19 // (op 35079) 

s_sub_i32 s22, s1, s19 // (op 35079) 


s_cmp_eq_i32 s0, 0 // (op 35079) 
s_mov_b32 s1, scc // (op 35079) 

s_cmp_eq_i32 s1, 0 // (op 35079) 
s_cselect_b32 s0, s22, s18 // (op 35079) 



v_lshl_add_u32 v21, s0, 8, v20 // (op 35079) 


v_mul_lo_u32 v25, v21, s33 // (op 35079) most significant: low of low * high
v_mul_hi_u32 v20, v21, s32 // (op 35079) most significant: high of low * low
v_mul_lo_u32 v24, v21, s32 // (op 35079) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 35079) most significant: sum


v_add_co_u32 v20, s[0:1], v22, v24 // (op 35079) least significant half
v_addc_co_u32 v21, s[0:1], v23, v25, s[0:1] // (op 35079) most significant half



s_mul_i32 s1, s32, s7 // (op 35081) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 35081) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 35081) least significant
s_mul_hi_u32 s19, s32, s6 // (op 35081) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35081) most significant: sum
s_add_u32 s1, s1, s19 // (op 35081) most significant: sum


s_mov_b32 s64, s34 // (op 35081) reg expression
s_mov_b32 s65, s35 // (op 35081) reg expression
s_mov_b32 s66, s0 // (op 35081) reg expression
s_mov_b32 s67, 131072 // (op 35081) call()

s_mul_hi_i32 s0, s16, s10 // (op 35124) 
v_and_b32 v22, -64, v1 // (op 35124) 
s_add_i32 s1, s0, s16 // (op 35124) 

v_sub_u32 v23, v1, v22 // (op 35124) 

s_ashr_i32 s0, s1, 31 // (op 35124) 
v_lshlrev_b32 v22, 2, v23 // (op 35124) 

s_cmp_eq_i32 s10, 0 // (op 35124) 
s_mov_b32 s18, scc // (op 35124) 
v_lshrrev_b32 v23, 6, v1 // (op 35124) 
s_cmp_eq_i32 s18, 0 // (op 35124) 
s_cselect_b32 s19, 0, -1 // (op 35124) 

s_and_b32 s18, s11, 31 // (op 35124) 
s_lshl_b32 s22, 1, s18 // (op 35124) 
s_add_i32 s23, s22, s19 // (op 35124) 


s_and_b32 s19, s0, s23 // (op 35124) 


s_add_i32 s0, s1, s19 // (op 35124) 


s_ashr_i32 s1, s0, s18 // (op 35124) 


s_lshl_b32 s0, s11, 24 // (op 35124) 
s_ashr_i32 s18, s0, 31 // (op 35124) 

s_xor_b32 s0, s1, s18 // (op 35124) 

s_sub_i32 s1, s0, s18 // (op 35124) 


s_mul_hi_i32 s0, s1, s12 // (op 35124) 
s_add_i32 s18, s0, s1 // (op 35124) 

s_ashr_i32 s0, s18, 31 // (op 35124) 
s_cmp_eq_i32 s12, 0 // (op 35124) 
s_mov_b32 s19, scc // (op 35124) 
s_cmp_eq_i32 s19, 0 // (op 35124) 
s_cselect_b32 s22, 0, -1 // (op 35124) 

s_and_b32 s19, s13, 31 // (op 35124) 
s_lshl_b32 s23, 1, s19 // (op 35124) 
s_add_i32 s24, s23, s22 // (op 35124) 


s_and_b32 s25, s0, s24 // (op 35124) 


s_add_i32 s0, s18, s25 // (op 35124) 


s_ashr_i32 s18, s0, s19 // (op 35124) 


s_lshl_b32 s0, s13, 24 // (op 35124) 
s_ashr_i32 s19, s0, 31 // (op 35124) 

s_xor_b32 s0, s18, s19 // (op 35124) 

s_sub_i32 s18, s0, s19 // (op 35124) 


s_mul_i32 s0, s1, s15 // (op 35124) 

s_sub_u32 s1, s16, s0 // (op 35124) 

s_mul_i32 s0, s18, s15 // (op 35124) 
s_add_u32 s0, s0, s1 // (op 35124) 


s_mul_hi_i32 s1, s16, s9 // (op 35124) 
s_add_i32 s18, s1, s16 // (op 35124) 

s_ashr_i32 s1, s18, 31 // (op 35124) 
s_cmp_eq_i32 s9, 0 // (op 35124) 
s_mov_b32 s19, scc // (op 35124) 
s_cmp_eq_i32 s19, 0 // (op 35124) 
s_cselect_b32 s22, 0, -1 // (op 35124) 

s_and_b32 s19, s46, 31 // (op 35124) 
s_lshl_b32 s23, 1, s19 // (op 35124) 
s_add_i32 s24, s23, s22 // (op 35124) 


s_and_b32 s25, s1, s24 // (op 35124) 


s_add_i32 s1, s18, s25 // (op 35124) 


s_ashr_i32 s18, s1, s19 // (op 35124) 


s_lshl_b32 s1, s46, 24 // (op 35124) 
s_ashr_i32 s19, s1, 31 // (op 35124) 

s_xor_b32 s1, s18, s19 // (op 35124) 

s_sub_i32 s18, s1, s19 // (op 35124) 


s_add_u32 s50, s6, 256 // (op 35124) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35124) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35124) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35124) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35124) 

s_mul_i32 s1, s15, s24 // (op 35124) 

s_mul_i32 s19, s8, s1 // (op 35124) 

s_mul_i32 s1, s18, s19 // (op 35124) 

s_sub_u32 s19, s16, s1 // (op 35124) 

s_mov_b32 s1, s19 // (op 35124) 
s_mul_hi_i32 s22, s1, s14 // (op 35124) 
s_add_i32 s23, s22, s1 // (op 35124) 


s_ashr_i32 s1, s23, 31 // (op 35124) 
s_cmp_eq_i32 s14, 0 // (op 35124) 
s_mov_b32 s22, scc // (op 35124) 
s_cmp_eq_i32 s22, 0 // (op 35124) 
s_cselect_b32 s24, 0, -1 // (op 35124) 

s_and_b32 s22, s48, 31 // (op 35124) 
s_lshl_b32 s25, 1, s22 // (op 35124) 
s_add_i32 s49, s25, s24 // (op 35124) 


s_and_b32 s24, s1, s49 // (op 35124) 


s_add_i32 s1, s23, s24 // (op 35124) 


s_ashr_i32 s23, s1, s22 // (op 35124) 


s_lshl_b32 s1, s48, 24 // (op 35124) 
s_ashr_i32 s22, s1, 31 // (op 35124) 

s_xor_b32 s1, s23, s22 // (op 35124) 

s_sub_i32 s23, s1, s22 // (op 35124) 


s_mul_i32 s1, s47, s15 // (op 35124) 
s_add_u32 s24, s20, 256 // (op 35124) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 35124) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 35124) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 35124) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 35124) 

s_sub_i32 s22, s24, s1 // (op 35124) 

s_mul_i32 s24, s23, s22 // (op 35124) 


s_sub_u32 s25, s19, s24 // (op 35124) 


s_add_u32 s19, s25, s1 // (op 35124) 


s_cmp_eq_i32 s18, 0 // (op 35124) 
s_mov_b32 s1, scc // (op 35124) 

s_cmp_eq_i32 s1, 0 // (op 35124) 
s_cselect_b32 s18, s19, s0 // (op 35124) 



v_lshl_add_u32 v24, s18, 8, v22 // (op 35124) 


v_mul_lo_u32 v22, v23, s36 // (op 35124) 

v_add_u32 v23, v24, v22 // (op 35124) 


s_mul_i32 s1, s36, s41 // (op 35126) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 35126) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 35126) least significant
s_mul_hi_u32 s19, s36, s40 // (op 35126) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35126) most significant: sum
s_add_u32 s1, s1, s19 // (op 35126) most significant: sum


s_mov_b32 s68, s38 // (op 35126) reg expression
s_mov_b32 s69, s39 // (op 35126) reg expression
s_mov_b32 s70, s0 // (op 35126) reg expression
s_mov_b32 s71, 131072 // (op 35126) call()

s_mul_hi_i32 s0, s16, s10 // (op 35143) 
v_and_b32 v22, -64, v1 // (op 35143) 
s_add_i32 s1, s0, s16 // (op 35143) 

v_sub_u32 v24, v1, v22 // (op 35143) 

s_ashr_i32 s0, s1, 31 // (op 35143) 
v_lshlrev_b32 v22, 2, v24 // (op 35143) 

s_cmp_eq_i32 s10, 0 // (op 35143) 
s_mov_b32 s18, scc // (op 35143) 
v_lshrrev_b32 v24, 6, v1 // (op 35143) 
s_cmp_eq_i32 s18, 0 // (op 35143) 
s_cselect_b32 s19, 0, -1 // (op 35143) 

s_and_b32 s18, s11, 31 // (op 35143) 
s_lshl_b32 s22, 1, s18 // (op 35143) 
s_add_i32 s23, s22, s19 // (op 35143) 


s_and_b32 s19, s0, s23 // (op 35143) 


s_add_i32 s0, s1, s19 // (op 35143) 


s_ashr_i32 s1, s0, s18 // (op 35143) 


s_lshl_b32 s0, s11, 24 // (op 35143) 
s_ashr_i32 s18, s0, 31 // (op 35143) 

s_xor_b32 s0, s1, s18 // (op 35143) 

s_sub_i32 s1, s0, s18 // (op 35143) 


s_mul_hi_i32 s0, s1, s12 // (op 35143) 
s_add_i32 s18, s0, s1 // (op 35143) 

s_ashr_i32 s0, s18, 31 // (op 35143) 
s_cmp_eq_i32 s12, 0 // (op 35143) 
s_mov_b32 s19, scc // (op 35143) 
s_cmp_eq_i32 s19, 0 // (op 35143) 
s_cselect_b32 s22, 0, -1 // (op 35143) 

s_and_b32 s19, s13, 31 // (op 35143) 
s_lshl_b32 s23, 1, s19 // (op 35143) 
s_add_i32 s24, s23, s22 // (op 35143) 


s_and_b32 s25, s0, s24 // (op 35143) 


s_add_i32 s0, s18, s25 // (op 35143) 


s_ashr_i32 s18, s0, s19 // (op 35143) 


s_lshl_b32 s0, s13, 24 // (op 35143) 
s_ashr_i32 s19, s0, 31 // (op 35143) 

s_xor_b32 s0, s18, s19 // (op 35143) 

s_sub_i32 s18, s0, s19 // (op 35143) 


s_mul_i32 s0, s1, s15 // (op 35143) 

s_sub_u32 s1, s16, s0 // (op 35143) 

s_mul_i32 s0, s18, s15 // (op 35143) 
s_add_u32 s0, s0, s1 // (op 35143) 


s_mul_hi_i32 s1, s16, s9 // (op 35143) 
s_add_i32 s18, s1, s16 // (op 35143) 

s_ashr_i32 s1, s18, 31 // (op 35143) 
s_cmp_eq_i32 s9, 0 // (op 35143) 
s_mov_b32 s19, scc // (op 35143) 
s_cmp_eq_i32 s19, 0 // (op 35143) 
s_cselect_b32 s22, 0, -1 // (op 35143) 

s_and_b32 s19, s46, 31 // (op 35143) 
s_lshl_b32 s23, 1, s19 // (op 35143) 
s_add_i32 s24, s23, s22 // (op 35143) 


s_and_b32 s25, s1, s24 // (op 35143) 


s_add_i32 s1, s18, s25 // (op 35143) 


s_ashr_i32 s18, s1, s19 // (op 35143) 


s_lshl_b32 s1, s46, 24 // (op 35143) 
s_ashr_i32 s19, s1, 31 // (op 35143) 

s_xor_b32 s1, s18, s19 // (op 35143) 

s_sub_i32 s18, s1, s19 // (op 35143) 


s_add_u32 s50, s6, 256 // (op 35143) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35143) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35143) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35143) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35143) 

s_mul_i32 s1, s15, s24 // (op 35143) 

s_mul_i32 s19, s8, s1 // (op 35143) 

s_mul_i32 s1, s18, s19 // (op 35143) 

s_sub_u32 s19, s16, s1 // (op 35143) 

s_mov_b32 s1, s19 // (op 35143) 
s_mul_hi_i32 s22, s1, s14 // (op 35143) 
s_add_i32 s23, s22, s1 // (op 35143) 


s_ashr_i32 s1, s23, 31 // (op 35143) 
s_cmp_eq_i32 s14, 0 // (op 35143) 
s_mov_b32 s22, scc // (op 35143) 
s_cmp_eq_i32 s22, 0 // (op 35143) 
s_cselect_b32 s24, 0, -1 // (op 35143) 

s_and_b32 s22, s48, 31 // (op 35143) 
s_lshl_b32 s25, 1, s22 // (op 35143) 
s_add_i32 s49, s25, s24 // (op 35143) 


s_and_b32 s24, s1, s49 // (op 35143) 


s_add_i32 s1, s23, s24 // (op 35143) 


s_ashr_i32 s23, s1, s22 // (op 35143) 


s_lshl_b32 s1, s48, 24 // (op 35143) 
s_ashr_i32 s22, s1, 31 // (op 35143) 

s_xor_b32 s1, s23, s22 // (op 35143) 

s_sub_i32 s23, s1, s22 // (op 35143) 


s_mul_i32 s1, s47, s15 // (op 35143) 
s_add_u32 s24, s20, 256 // (op 35143) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 35143) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 35143) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 35143) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 35143) 

s_sub_i32 s22, s24, s1 // (op 35143) 

s_mul_i32 s24, s23, s22 // (op 35143) 


s_sub_u32 s25, s19, s24 // (op 35143) 


s_add_u32 s19, s25, s1 // (op 35143) 


s_cmp_eq_i32 s18, 0 // (op 35143) 
s_mov_b32 s1, scc // (op 35143) 

s_cmp_eq_i32 s1, 0 // (op 35143) 
s_cselect_b32 s18, s19, s0 // (op 35143) 



v_lshl_add_u32 v25, s18, 8, v22 // (op 35143) 


v_mul_lo_u32 v22, v24, s36 // (op 35143) 

v_add_u32 v24, v25, v22 // (op 35143) 


s_mul_i32 s1, s36, s41 // (op 35145) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 35145) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 35145) least significant
s_mul_hi_u32 s19, s36, s40 // (op 35145) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35145) most significant: sum
s_add_u32 s1, s1, s19 // (op 35145) most significant: sum


s_mov_b32 s68, s38 // (op 35145) reg expression
s_mov_b32 s69, s39 // (op 35145) reg expression
s_mov_b32 s70, s0 // (op 35145) reg expression
s_mov_b32 s71, 131072 // (op 35145) call()

s_mul_hi_i32 s0, s16, s10 // (op 35188) 
s_add_i32 s1, s0, s16 // (op 35188) 

s_ashr_i32 s0, s1, 31 // (op 35188) 
s_cmp_eq_i32 s10, 0 // (op 35188) 
s_mov_b32 s18, scc // (op 35188) 
s_cmp_eq_i32 s18, 0 // (op 35188) 
s_cselect_b32 s19, 0, -1 // (op 35188) 

s_and_b32 s18, s11, 31 // (op 35188) 
s_lshl_b32 s22, 1, s18 // (op 35188) 
s_add_i32 s23, s22, s19 // (op 35188) 


s_and_b32 s19, s0, s23 // (op 35188) 


s_add_i32 s0, s1, s19 // (op 35188) 


s_ashr_i32 s1, s0, s18 // (op 35188) 


s_lshl_b32 s0, s11, 24 // (op 35188) 
s_ashr_i32 s18, s0, 31 // (op 35188) 

s_xor_b32 s0, s1, s18 // (op 35188) 

s_sub_i32 s1, s0, s18 // (op 35188) 


s_mul_hi_i32 s0, s1, s12 // (op 35188) 
s_add_i32 s18, s0, s1 // (op 35188) 

s_ashr_i32 s0, s18, 31 // (op 35188) 
s_cmp_eq_i32 s12, 0 // (op 35188) 
s_mov_b32 s19, scc // (op 35188) 
s_cmp_eq_i32 s19, 0 // (op 35188) 
s_cselect_b32 s22, 0, -1 // (op 35188) 

s_and_b32 s19, s13, 31 // (op 35188) 
s_lshl_b32 s23, 1, s19 // (op 35188) 
s_add_i32 s24, s23, s22 // (op 35188) 


s_and_b32 s25, s0, s24 // (op 35188) 


s_add_i32 s0, s18, s25 // (op 35188) 


s_ashr_i32 s18, s0, s19 // (op 35188) 


s_lshl_b32 s0, s13, 24 // (op 35188) 
s_ashr_i32 s19, s0, 31 // (op 35188) 

s_xor_b32 s0, s18, s19 // (op 35188) 

s_sub_i32 s18, s0, s19 // (op 35188) 


s_add_u32 s50, s6, 256 // (op 35188) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35188) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35188) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35188) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35188) 

s_mul_i32 s0, s18, s24 // (op 35188) 

s_sub_i32 s18, s1, s0 // (op 35188) 


s_mul_hi_i32 s19, s16, s9 // (op 35188) 
s_add_i32 s0, s19, s16 // (op 35188) 

s_ashr_i32 s1, s0, 31 // (op 35188) 
s_cmp_eq_i32 s9, 0 // (op 35188) 
s_mov_b32 s19, scc // (op 35188) 
s_cmp_eq_i32 s19, 0 // (op 35188) 
s_cselect_b32 s22, 0, -1 // (op 35188) 

s_and_b32 s19, s46, 31 // (op 35188) 
s_lshl_b32 s23, 1, s19 // (op 35188) 
s_add_i32 s49, s23, s22 // (op 35188) 


s_and_b32 s22, s1, s49 // (op 35188) 


s_add_i32 s1, s0, s22 // (op 35188) 


s_ashr_i32 s0, s1, s19 // (op 35188) 


s_lshl_b32 s1, s46, 24 // (op 35188) 
s_ashr_i32 s19, s1, 31 // (op 35188) 

s_xor_b32 s1, s0, s19 // (op 35188) 

s_sub_i32 s0, s1, s19 // (op 35188) 


s_mul_i32 s1, s15, s24 // (op 35188) 

s_mul_i32 s19, s8, s1 // (op 35188) 

s_mul_i32 s1, s0, s19 // (op 35188) 

s_sub_u32 s19, s16, s1 // (op 35188) 

s_mul_hi_i32 s1, s19, s14 // (op 35188) 
s_add_i32 s22, s1, s19 // (op 35188) 


s_ashr_i32 s1, s22, 31 // (op 35188) 
s_cmp_eq_i32 s14, 0 // (op 35188) 
s_mov_b32 s19, scc // (op 35188) 
s_cmp_eq_i32 s19, 0 // (op 35188) 
s_cselect_b32 s23, 0, -1 // (op 35188) 

s_and_b32 s19, s48, 31 // (op 35188) 
s_lshl_b32 s24, 1, s19 // (op 35188) 
s_add_i32 s25, s24, s23 // (op 35188) 


s_and_b32 s23, s1, s25 // (op 35188) 


s_add_i32 s1, s22, s23 // (op 35188) 


s_ashr_i32 s22, s1, s19 // (op 35188) 


s_lshl_b32 s1, s48, 24 // (op 35188) 
s_ashr_i32 s19, s1, 31 // (op 35188) 

s_xor_b32 s1, s22, s19 // (op 35188) 

s_sub_i32 s22, s1, s19 // (op 35188) 


s_cmp_eq_i32 s0, 0 // (op 35188) 
s_mov_b32 s1, scc // (op 35188) 

s_cmp_eq_i32 s1, 0 // (op 35188) 
s_cselect_b32 s0, s22, s18 // (op 35188) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 35188) 

v_mul_lo_u32 v25, v22, s42 // (op 35188) 

s_mul_i32 s1, s42, s7 // (op 35190) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 35190) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 35190) least significant
s_mul_hi_u32 s19, s42, s6 // (op 35190) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35190) most significant: sum
s_add_u32 s1, s1, s19 // (op 35190) most significant: sum


s_mov_b32 s72, s44 // (op 35190) reg expression
s_mov_b32 s73, s45 // (op 35190) reg expression
s_mov_b32 s74, s0 // (op 35190) reg expression
s_mov_b32 s75, 131072 // (op 35190) call()

s_mul_hi_i32 s0, s16, s10 // (op 35207) 
s_add_i32 s1, s0, s16 // (op 35207) 

s_ashr_i32 s0, s1, 31 // (op 35207) 
s_cmp_eq_i32 s10, 0 // (op 35207) 
s_mov_b32 s18, scc // (op 35207) 
s_cmp_eq_i32 s18, 0 // (op 35207) 
s_cselect_b32 s19, 0, -1 // (op 35207) 

s_and_b32 s18, s11, 31 // (op 35207) 
s_lshl_b32 s22, 1, s18 // (op 35207) 
s_add_i32 s23, s22, s19 // (op 35207) 


s_and_b32 s19, s0, s23 // (op 35207) 


s_add_i32 s0, s1, s19 // (op 35207) 


s_ashr_i32 s1, s0, s18 // (op 35207) 


s_lshl_b32 s0, s11, 24 // (op 35207) 
s_ashr_i32 s18, s0, 31 // (op 35207) 

s_xor_b32 s0, s1, s18 // (op 35207) 

s_sub_i32 s1, s0, s18 // (op 35207) 


s_mul_hi_i32 s0, s1, s12 // (op 35207) 
s_add_i32 s18, s0, s1 // (op 35207) 

s_ashr_i32 s0, s18, 31 // (op 35207) 
s_cmp_eq_i32 s12, 0 // (op 35207) 
s_mov_b32 s19, scc // (op 35207) 
s_cmp_eq_i32 s19, 0 // (op 35207) 
s_cselect_b32 s22, 0, -1 // (op 35207) 

s_and_b32 s19, s13, 31 // (op 35207) 
s_lshl_b32 s23, 1, s19 // (op 35207) 
s_add_i32 s24, s23, s22 // (op 35207) 


s_and_b32 s25, s0, s24 // (op 35207) 


s_add_i32 s0, s18, s25 // (op 35207) 


s_ashr_i32 s18, s0, s19 // (op 35207) 


s_lshl_b32 s0, s13, 24 // (op 35207) 
s_ashr_i32 s19, s0, 31 // (op 35207) 

s_xor_b32 s0, s18, s19 // (op 35207) 

s_sub_i32 s18, s0, s19 // (op 35207) 


s_add_u32 s50, s6, 256 // (op 35207) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35207) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35207) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35207) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35207) 

s_mul_i32 s0, s18, s24 // (op 35207) 

s_sub_i32 s18, s1, s0 // (op 35207) 


s_mul_hi_i32 s19, s16, s9 // (op 35207) 
s_add_i32 s0, s19, s16 // (op 35207) 

s_ashr_i32 s1, s0, 31 // (op 35207) 
s_cmp_eq_i32 s9, 0 // (op 35207) 
s_mov_b32 s19, scc // (op 35207) 
s_cmp_eq_i32 s19, 0 // (op 35207) 
s_cselect_b32 s22, 0, -1 // (op 35207) 

s_and_b32 s19, s46, 31 // (op 35207) 
s_lshl_b32 s23, 1, s19 // (op 35207) 
s_add_i32 s49, s23, s22 // (op 35207) 


s_and_b32 s22, s1, s49 // (op 35207) 


s_add_i32 s1, s0, s22 // (op 35207) 


s_ashr_i32 s0, s1, s19 // (op 35207) 


s_lshl_b32 s1, s46, 24 // (op 35207) 
s_ashr_i32 s19, s1, 31 // (op 35207) 

s_xor_b32 s1, s0, s19 // (op 35207) 

s_sub_i32 s0, s1, s19 // (op 35207) 


s_mul_i32 s1, s15, s24 // (op 35207) 

s_mul_i32 s19, s8, s1 // (op 35207) 

s_mul_i32 s1, s0, s19 // (op 35207) 

s_sub_u32 s19, s16, s1 // (op 35207) 

s_mul_hi_i32 s1, s19, s14 // (op 35207) 
s_add_i32 s22, s1, s19 // (op 35207) 


s_ashr_i32 s1, s22, 31 // (op 35207) 
s_cmp_eq_i32 s14, 0 // (op 35207) 
s_mov_b32 s19, scc // (op 35207) 
s_cmp_eq_i32 s19, 0 // (op 35207) 
s_cselect_b32 s23, 0, -1 // (op 35207) 

s_and_b32 s19, s48, 31 // (op 35207) 
s_lshl_b32 s24, 1, s19 // (op 35207) 
s_add_i32 s25, s24, s23 // (op 35207) 


s_and_b32 s23, s1, s25 // (op 35207) 


s_add_i32 s1, s22, s23 // (op 35207) 


s_ashr_i32 s22, s1, s19 // (op 35207) 


s_lshl_b32 s1, s48, 24 // (op 35207) 
s_ashr_i32 s19, s1, 31 // (op 35207) 

s_xor_b32 s1, s22, s19 // (op 35207) 

s_sub_i32 s22, s1, s19 // (op 35207) 


s_cmp_eq_i32 s0, 0 // (op 35207) 
s_mov_b32 s1, scc // (op 35207) 

s_cmp_eq_i32 s1, 0 // (op 35207) 
s_cselect_b32 s0, s22, s18 // (op 35207) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 35207) 

v_mul_lo_u32 v26, v22, s42 // (op 35207) 

s_mul_i32 s1, s42, s7 // (op 35209) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 35209) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 35209) least significant
s_mul_hi_u32 s19, s42, s6 // (op 35209) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35209) most significant: sum
s_add_u32 s1, s1, s19 // (op 35209) most significant: sum


s_mov_b32 s72, s44 // (op 35209) reg expression
s_mov_b32 s73, s45 // (op 35209) reg expression
s_mov_b32 s74, s0 // (op 35209) reg expression
s_mov_b32 s75, 131072 // (op 35209) call()

v_and_b32 v22, -8, v1 // (op 35335) 
v_sub_u32 v27, v1, v22 // (op 35335) 

v_lshlrev_b32 v22, 4, v27 // (op 35335) 

v_lshrrev_b32 v27, 3, v1 // (op 35335) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 35335) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 35335) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 34882) 
v_mov_b64 v[28:29], s[18:19] // (op 34882) convert


v_mov_b64 v[30:31], v[14:15] // (op 34882) 
s_mov_b32 m0, s0 // (op 34882) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34882) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34882) most significant half

s_add_u32 m0, m0, 4096 // (op 34882) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34882) Load value direct to lds






buffer_load_dword v22, v25, s[72:75], 0 offen // (op 34280) Load value
v_and_b32 v27, -8, v1 // (op 35278) 
v_sub_u32 v28, v1, v27 // (op 35278) 

v_lshlrev_b32 v27, 5, v28 // (op 35278) 

v_lshrrev_b32 v28, 3, v1 // (op 35278) 
v_lshl_add_u32 v29, v28, 8, v27 // (op 35278) 


v_lshlrev_b32 v27, 2, v29 // (op 35278) 

v_lshrrev_b32 v28, 3, v27 // (op 35278) 

s_nop 0
v_readfirstlane_b32 s0, v28 // (op 35278) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 5 // (op 34887) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 34887) 

v_lshrrev_b64 v[28:29], 3, s[50:51] // (op 34887) 


v_mov_b64 v[30:31], v[16:17] // (op 34887) reg expression
s_add_u32 s1, 32768, s0 // (op 34887) 
s_mov_b32 m0, s1 // (op 34887) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34887) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34887) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34887) most significant half

s_add_u32 m0, m0, 4096 // (op 34887) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34887) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34887) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34887) most significant half

s_add_u32 m0, m0, 4096 // (op 34887) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34887) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34887) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34887) most significant half

s_add_u32 m0, m0, 4096 // (op 34887) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34887) Load value direct to lds






v_mov_b32 v27, v23 // (op 34286) reg expression

buffer_load_dword v28, v27, s[68:71], 0 offen // (op 34286) Load value

v_and_b32 v27, -8, v1 // (op 35562) 
v_sub_u32 v29, v1, v27 // (op 35562) 

v_lshlrev_b32 v27, 4, v29 // (op 35562) 

v_lshrrev_b32 v29, 3, v1 // (op 35562) 
v_lshl_add_u32 v30, v29, 7, v27 // (op 35562) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 35562) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 34892) 
v_mov_b64 v[30:31], s[18:19] // (op 34892) convert


v_mov_b64 v[32:33], v[14:15] // (op 34892) 
s_add_u32 s1, 49152, s0 // (op 34892) 
s_mov_b32 m0, s1 // (op 34892) reg expression
s_mov_b32 s18, 128 // (op 34892) call()
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34892) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34892) most significant half

s_add_u32 m0, m0, 4096 // (op 34892) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34892) Load value direct to lds









buffer_load_dword v27, v25, s[72:75], 0 offen offset:4 // (op 34292) Load value

v_and_b32 v25, -8, v1 // (op 35518) 
v_sub_u32 v14, v1, v25 // (op 35518) 

v_lshlrev_b32 v15, 5, v14 // (op 35518) 

v_lshrrev_b32 v14, 3, v1 // (op 35518) 
v_lshl_add_u32 v25, v14, 8, v15 // (op 35518) 


v_lshlrev_b32 v14, 2, v25 // (op 35518) 

v_lshrrev_b32 v15, 3, v14 // (op 35518) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 35518) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 5 // (op 34894) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 34894) 

v_lshrrev_b64 v[14:15], 3, s[50:51] // (op 34894) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 34894) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 34894) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 34894) 

v_mov_b32 v25, s18 // (op 34894) 
v_mov_b32 v29, s19 // (op 34894) 
v_add_co_u32 v30, s[50:51], v16, v25 // (op 34894) least significant half
v_addc_co_u32 v31, s[50:51], v17, v29, s[50:51] // (op 34894) most significant half




s_add_u32 s1, 81920, s0 // (op 34894) 
s_mov_b32 m0, s1 // (op 34894) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34894) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v14 // (op 34894) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34894) most significant half

s_add_u32 m0, m0, 4096 // (op 34894) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34894) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v14 // (op 34894) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34894) most significant half

s_add_u32 m0, m0, 4096 // (op 34894) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34894) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v14 // (op 34894) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34894) most significant half

s_add_u32 m0, m0, 4096 // (op 34894) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34894) Load value direct to lds







s_lshl_b32 s0, s36, 2 // (op 34298) 
v_add_u32 v25, v23, s0 // (op 34298) 


buffer_load_dword v14, v25, s[68:71], 0 offen // (op 34298) Load value


v_lshlrev_b32 v23, 2, v1 // (op 35451) 



v_add_u32 v25, 98304, v23 // (op 34301) 
s_waitcnt vmcnt(19)
ds_write_b32 v25, v22 // (op 34301) Store local data 


v_and_b32 v23, -64, v1 // (op 35393) 
v_sub_u32 v25, v1, v23 // (op 35393) 

v_lshlrev_b32 v23, 2, v25 // (op 35393) 

v_lshrrev_b32 v25, 6, v1 // (op 35393) 
v_lshl_add_u32 v15, v25, 8, v23 // (op 35393) 





v_add_u32 v23, 99328, v15 // (op 34303) 
s_waitcnt vmcnt(14)
ds_write_b32 v23, v28 // (op 34303) Store local data 


s_barrier  // (op 35946) 



ds_read_b128 v[30:33], v4 // (op 34358) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 34358) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 35980) 



v_add_u32 v23, 98304, v5 // (op 34438) 
ds_read_u8 v25, v23 // (op 34438) Load local data 




ds_read_b128 v[38:41], v4 offset:4096 // (op 34363) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 34363) Load local data 



v_add_u32 v23, 98432, v5 // (op 34443) 
ds_read_u8 v29, v23 // (op 34443) Load local data 




ds_read_b128 v[46:49], v4 offset:8192 // (op 34368) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 34368) Load local data 



v_add_u32 v23, 98560, v5 // (op 34448) 
ds_read_u8 v15, v23 // (op 34448) Load local data 




ds_read_b128 v[54:57], v4 offset:12288 // (op 34373) Load local data 
ds_read_b128 v[58:61], v4 offset:12352 // (op 34373) Load local data 



v_add_u32 v23, 98688, v5 // (op 34453) 
ds_read_u8 v16, v23 // (op 34453) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 34378) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 34378) Load local data 



v_add_u32 v17, 98816, v5 // (op 34458) 
ds_read_u8 v23, v17 // (op 34458) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 34383) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 34383) Load local data 



v_add_u32 v17, 98944, v5 // (op 34463) 
ds_read_u8 v78, v17 // (op 34463) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 34388) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 34388) Load local data 



v_add_u32 v17, 99072, v5 // (op 34468) 
ds_read_u8 v79, v17 // (op 34468) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 34393) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 34393) Load local data 



v_add_u32 v17, 99200, v5 // (op 34473) 
ds_read_u8 v96, v17 // (op 34473) Load local data 

s_barrier  // (op 35930) 



ds_read_b64_tr_b4 v[98:99], v3 offset:32768 // (op 34318) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:34816 // (op 34318) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 35963) 


v_add_u32 v17, 99328, v0 // (op 34398) 
ds_read_u8 v97, v17 // (op 34398) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:32784 // (op 34323) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:34832 // (op 34323) Transpose load local data 


v_add_u32 v17, 99360, v0 // (op 34403) 
ds_read_u8 v106, v17 // (op 34403) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:32800 // (op 34328) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:34848 // (op 34328) Transpose load local data 


v_add_u32 v17, 99392, v0 // (op 34408) 
ds_read_u8 v107, v17 // (op 34408) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:32816 // (op 34333) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:34864 // (op 34333) Transpose load local data 


v_add_u32 v17, 99424, v0 // (op 34413) 
ds_read_u8 v116, v17 // (op 34413) Load local data 




ds_read_b64_tr_b4 v[118:119], v3 offset:32832 // (op 34338) Transpose load local data 
ds_read_b64_tr_b4 v[120:121], v3 offset:34880 // (op 34338) Transpose load local data 


v_add_u32 v17, 99456, v0 // (op 34418) 
ds_read_u8 v117, v17 // (op 34418) Load local data 




ds_read_b64_tr_b4 v[122:123], v3 offset:32848 // (op 34343) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v3 offset:34896 // (op 34343) Transpose load local data 


v_add_u32 v17, 99488, v0 // (op 34423) 
ds_read_u8 v126, v17 // (op 34423) Load local data 




ds_read_b64_tr_b4 v[128:129], v3 offset:32864 // (op 34348) Transpose load local data 
ds_read_b64_tr_b4 v[130:131], v3 offset:34912 // (op 34348) Transpose load local data 


v_add_u32 v17, 99520, v0 // (op 34428) 
ds_read_u8 v127, v17 // (op 34428) Load local data 




ds_read_b64_tr_b4 v[132:133], v3 offset:32880 // (op 34353) Transpose load local data 
ds_read_b64_tr_b4 v[134:135], v3 offset:34928 // (op 34353) Transpose load local data 


v_add_u32 v17, 99552, v0 // (op 34433) 
ds_read_u8 v136, v17 // (op 34433) Load local data 

v_and_b32 v17, -8, v1 // (op 35250) 
v_sub_u32 v137, v1, v17 // (op 35250) 

v_lshlrev_b32 v17, 5, v137 // (op 35250) 

v_lshrrev_b32 v137, 3, v1 // (op 35250) 
v_lshl_add_u32 v138, v137, 8, v17 // (op 35250) 


v_lshlrev_b32 v17, 2, v138 // (op 35250) 

v_lshrrev_b32 v137, 3, v17 // (op 35250) 

s_nop 0
v_readfirstlane_b32 s0, v137 // (op 35250) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 35307) 
v_sub_u32 v137, v1, v17 // (op 35307) 

v_lshlrev_b32 v17, 4, v137 // (op 35307) 

v_lshrrev_b32 v137, 3, v1 // (op 35307) 
v_lshl_add_u32 v138, v137, 7, v17 // (op 35307) 


s_nop 0
v_readfirstlane_b32 s1, v138 // (op 35307) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 35364) 
v_sub_u32 v137, v1, v17 // (op 35364) 

v_lshlrev_b32 v17, 2, v137 // (op 35364) 

v_lshrrev_b32 v137, 6, v1 // (op 35364) 
v_lshl_add_u32 v138, v137, 8, v17 // (op 35364) 


v_lshlrev_b32 v17, 2, v1 // (op 35422) 
v_and_b32 v137, -8, v1 // (op 35503) 
v_sub_u32 v139, v1, v137 // (op 35503) 

v_lshlrev_b32 v137, 5, v139 // (op 35503) 

v_lshrrev_b32 v139, 3, v1 // (op 35503) 
v_lshl_add_u32 v140, v139, 8, v137 // (op 35503) 


v_lshlrev_b32 v137, 2, v140 // (op 35503) 

v_lshrrev_b32 v139, 3, v137 // (op 35503) 

s_nop 0
v_readfirstlane_b32 s18, v139 // (op 35503) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v137, -8, v1 // (op 35547) 
v_sub_u32 v139, v1, v137 // (op 35547) 

v_lshlrev_b32 v137, 4, v139 // (op 35547) 

v_lshrrev_b32 v139, 3, v1 // (op 35547) 
v_lshl_add_u32 v140, v139, 7, v137 // (op 35547) 


s_nop 0
v_readfirstlane_b32 s19, v140 // (op 35547) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v137, -64, v1 // (op 35591) 
v_sub_u32 v139, v1, v137 // (op 35591) 

v_lshlrev_b32 v137, 2, v139 // (op 35591) 

v_lshrrev_b32 v139, 6, v1 // (op 35591) 
v_lshl_add_u32 v140, v139, 8, v137 // (op 35591) 


v_lshlrev_b32 v137, 2, v1 // (op 35620) 
s_mov_b32 s22, 0 // (op 9657) call()
s_ashr_i64 s[24:25], s[30:31], 7 // (op 9656) 
s_mov_b32 s23, s24 // (op 9656) convert

s_and_b32 s24, s23, 4294967294 // (op 9656) 

s_cmp_lt_u32 s22, s24 // (op 9656) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9656 // (op 9656) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9656 if false)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9656:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:101], v[30:37], a[0:3], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 90) 
s_barrier  // (op 35949) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:101], v[38:45], a[4:7], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 380) 
s_add_u32 s22, s22, 2 // (op 9658) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:101], v[46:53], a[8:11], v97, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 556) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:101], v[54:61], a[12:15], v97, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 732) 
v_mov_b64 v[142:143], s[24:25] // (op 34854) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:101], v[62:69], a[16:19], v97, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 908) 
v_mov_b64 v[144:145], v[20:21] // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:101], v[70:77], a[20:23], v97, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1084) 
v_mov_b64 v[146:147], v[144:145] // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:101], v[80:87], a[24:27], v97, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1260) 
s_mov_b32 m0, s1 // (op 34854) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:101], v[88:95], a[28:31], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1436) 
s_mov_b32 s23, 256 // (op 34854) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[102:105], v[30:37], a[32:35], v106, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2327) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:105], v[38:45], a[36:39], v106, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2281) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:105], v[46:53], a[40:43], v106, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2235) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[102:105], v[54:61], a[44:47], v106, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[62:69], a[48:51], v106, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2143) 
s_add_u32 m0, m0, 4096 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[70:77], a[52:55], v106, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2097) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[80:87], a[56:59], v106, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2051) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[88:95], a[60:63], v106, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2005) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:111], v[30:37], a[64:67], v107, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3865) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[38:45], a[68:71], v107, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[46:53], a[72:75], v107, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3773) 
s_add_u32 m0, m0, 4096 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[54:61], a[76:79], v107, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3727) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[62:69], a[80:83], v107, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3681) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:111], v[70:77], a[84:87], v107, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3635) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[80:87], a[88:91], v107, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3589) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[88:95], a[92:95], v107, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3543) 
s_add_u32 m0, m0, 4096 // (op 34854) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[30:37], a[96:99], v116, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5403) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[38:45], a[100:103], v116, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5357) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[46:53], a[104:107], v116, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5311) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[54:61], a[108:111], v116, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5265) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[62:69], a[112:115], v116, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5219) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[70:77], a[116:119], v116, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5173) 
s_add_u32 m0, m0, 4096 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:115], v[80:87], a[120:123], v116, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5127) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[112:115], v[88:95], a[124:127], v116, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5081) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[118:121], v[30:37], a[128:131], v117, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6941) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:121], v[38:45], a[132:135], v117, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6895) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[118:121], v[46:53], a[136:139], v117, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[118:121], v[54:61], a[140:143], v117, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6803) 
s_add_u32 m0, m0, 4096 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[118:121], v[62:69], a[144:147], v117, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6757) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[118:121], v[70:77], a[148:151], v117, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6711) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[118:121], v[80:87], a[152:155], v117, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6665) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[118:121], v[88:95], a[156:159], v117, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6619) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

s_add_u32 m0, m0, 4096 // (op 34854) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:125], v[30:37], a[160:163], v126, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8479) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[122:125], v[38:45], a[164:167], v126, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8433) 
v_add_co_u32 v146, s[24:25], v146, v142 // (op 34854) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[122:125], v[46:53], a[168:171], v126, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8387) 
v_addc_co_u32 v147, s[24:25], 0, v147, s[24:25] // (op 34854) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[122:125], v[54:61], a[172:175], v126, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8341) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[122:125], v[62:69], a[176:179], v126, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8295) 
s_add_u32 m0, m0, 4096 // (op 34854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[122:125], v[70:77], a[180:183], v126, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8249) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[122:125], v[80:87], a[184:187], v126, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8203) 
buffer_load_dwordx4 v146, s[64:67], s23 offen offset: 0 lds // (op 34854) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[122:125], v[88:95], a[188:191], v126, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8157) 
v_mov_b32 v139, v26 // (op 25) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[128:131], v[30:37], a[192:195], v127, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 10017) 
buffer_load_dword v22, v139, s[72:75], 0 offen offset:8 // (op 25) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[38:45], a[196:199], v127, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9971) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 35933) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[128:131], v[46:53], a[200:203], v127, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9925) 
s_lshl_b64 s[24:25], s[26:27], 5 // (op 34849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[128:131], v[54:61], a[204:207], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9879) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 34849) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[62:69], a[208:211], v127, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9833) 
v_lshrrev_b64 v[142:143], 3, s[50:51] // (op 34849) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[128:131], v[70:77], a[212:215], v127, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9787) 
v_mov_b64 v[146:147], v[18:19] // (op 34849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[128:131], v[80:87], a[216:219], v127, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9741) 
s_lshl_b64 s[24:25], s[26:27], 7 // (op 34849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[128:131], v[88:95], a[220:223], v127, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9695) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 34849) 

s_lshr_b64 s[24:25], s[50:51], 3 // (op 34849) 

s_lshl_b64 s[50:51], s[24:25], 1 // (op 34849) 

v_mov_b32 v141, s50 // (op 34849) 
v_mov_b32 v148, s51 // (op 34849) 
v_add_co_u32 v150, s[24:25], v146, v141 // (op 34849) least significant half
v_addc_co_u32 v151, s[24:25], v147, v148, s[24:25] // (op 34849) most significant half




s_add_u32 s23, 32768, s0 // (op 34849) 
s_mov_b32 m0, s23 // (op 34849) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[30:37], a[224:227], v136, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11555) 
buffer_load_dwordx4 v150, s[52:55], 0 offen offset: 0 lds // (op 34849) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[132:135], v[38:45], a[228:231], v136, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11509) 
v_add_co_u32 v150, s[24:25], v150, v142 // (op 34849) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[46:53], a[232:235], v136, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11463) 
v_addc_co_u32 v151, s[24:25], 0, v151, s[24:25] // (op 34849) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[54:61], a[236:239], v136, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[132:135], v[62:69], a[240:243], v136, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11371) 
s_add_u32 m0, m0, 4096 // (op 34849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[132:135], v[70:77], a[244:247], v136, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11325) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[80:87], a[248:251], v136, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11279) 
buffer_load_dwordx4 v150, s[52:55], 0 offen offset: 0 lds // (op 34849) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[88:95], a[252:255], v136, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11233) 
v_add_co_u32 v150, s[24:25], v150, v142 // (op 34849) least significant half
v_addc_co_u32 v151, s[24:25], 0, v151, s[24:25] // (op 34849) most significant half

s_add_u32 m0, m0, 4096 // (op 34849) 
buffer_load_dwordx4 v150, s[52:55], 0 offen offset: 0 lds // (op 34849) Load value direct to lds
v_add_co_u32 v150, s[24:25], v150, v142 // (op 34849) least significant half
v_addc_co_u32 v151, s[24:25], 0, v151, s[24:25] // (op 34849) most significant half

s_add_u32 m0, m0, 4096 // (op 34849) 
buffer_load_dwordx4 v150, s[52:55], 0 offen offset: 0 lds // (op 34849) Load value direct to lds





v_mov_b32 v141, v24 // (op 17) 
s_lshl_b32 s23, s36, 3 // (op 17) 
v_add_u32 v142, v141, s23 // (op 17) 


buffer_load_dword v28, v142, s[68:71], 0 offen // (op 17) Load value

s_barrier  // (op 36019) 



v_add_u32 v142, 100352, v137 // (op 259) 
s_waitcnt vmcnt(19)
ds_write_b32 v142, v27 // (op 259) Store local data 

s_barrier  // (op 36013) 



v_add_u32 v142, 101376, v140 // (op 258) 
s_waitcnt vmcnt(14)
ds_write_b32 v142, v14 // (op 258) Store local data 

s_barrier  // (op 36003) 



ds_read_b128 v[148:151], v8 offset:49152 // (op 251) Load local data 
ds_read_b128 v[152:155], v8 offset:49216 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 36016) 



v_add_u32 v142, 100352, v9 // (op 255) 
ds_read_u8 v143, v142 // (op 255) Load local data 




ds_read_b128 v[156:159], v8 offset:53248 // (op 385) Load local data 
ds_read_b128 v[160:163], v8 offset:53312 // (op 385) Load local data 



v_add_u32 v142, 100480, v9 // (op 389) 
ds_read_u8 v106, v142 // (op 389) Load local data 




ds_read_b128 v[164:167], v8 offset:57344 // (op 561) Load local data 
ds_read_b128 v[168:171], v8 offset:57408 // (op 561) Load local data 



v_add_u32 v142, 100608, v9 // (op 565) 
ds_read_u8 v107, v142 // (op 565) Load local data 




ds_read_b128 v[172:175], v8 offset:61440 // (op 737) Load local data 
ds_read_b128 v[176:179], v8 offset:61504 // (op 737) Load local data 



v_add_u32 v142, 100736, v9 // (op 741) 
ds_read_u8 v116, v142 // (op 741) Load local data 




v_add_u32 v142, 65536, v8 // (op 913) 
ds_read_b128 v[180:183], v142 // (op 913) Load local data 

v_add_u32 v142, 65600, v8 // (op 913) 
ds_read_b128 v[184:187], v142 // (op 913) Load local data 




v_add_u32 v142, 100864, v9 // (op 917) 
ds_read_u8 v117, v142 // (op 917) Load local data 




v_add_u32 v142, 69632, v8 // (op 1089) 
ds_read_b128 v[188:191], v142 // (op 1089) Load local data 

v_add_u32 v142, 69696, v8 // (op 1089) 
ds_read_b128 v[192:195], v142 // (op 1089) Load local data 




v_add_u32 v142, 100992, v9 // (op 1093) 
ds_read_u8 v126, v142 // (op 1093) Load local data 




v_add_u32 v142, 73728, v8 // (op 1265) 
ds_read_b128 v[196:199], v142 // (op 1265) Load local data 

v_add_u32 v142, 73792, v8 // (op 1265) 
ds_read_b128 v[200:203], v142 // (op 1265) Load local data 




v_add_u32 v142, 101120, v9 // (op 1269) 
ds_read_u8 v127, v142 // (op 1269) Load local data 




v_add_u32 v142, 77824, v8 // (op 1441) 
ds_read_b128 v[204:207], v142 // (op 1441) Load local data 

v_add_u32 v142, 77888, v8 // (op 1441) 
ds_read_b128 v[208:211], v142 // (op 1441) Load local data 




v_add_u32 v142, 101248, v9 // (op 1445) 
ds_read_u8 v212, v142 // (op 1445) Load local data 

s_barrier  // (op 35997) 
s_barrier  // (op 36064) 



v_add_u32 v142, 81920, v7 // (op 249) 
ds_read_b64_tr_b4 v[98:99], v142 // (op 249) Transpose load local data 

v_add_u32 v142, 83968, v7 // (op 249) 
ds_read_b64_tr_b4 v[100:101], v142 // (op 249) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 36010) 


v_add_u32 v142, 101376, v6 // (op 253) 
ds_read_u8 v97, v142 // (op 253) Load local data 




v_add_u32 v142, 81936, v7 // (op 2008) 
ds_read_b64_tr_b4 v[102:103], v142 // (op 2008) Transpose load local data 

v_add_u32 v142, 83984, v7 // (op 2008) 
ds_read_b64_tr_b4 v[104:105], v142 // (op 2008) Transpose load local data 



v_add_u32 v142, 101408, v6 // (op 2011) 
ds_read_u8 v29, v142 // (op 2011) Load local data 




v_add_u32 v142, 81952, v7 // (op 3546) 
ds_read_b64_tr_b4 v[108:109], v142 // (op 3546) Transpose load local data 

v_add_u32 v142, 84000, v7 // (op 3546) 
ds_read_b64_tr_b4 v[110:111], v142 // (op 3546) Transpose load local data 



v_add_u32 v142, 101440, v6 // (op 3549) 
ds_read_u8 v15, v142 // (op 3549) Load local data 




v_add_u32 v142, 81968, v7 // (op 5084) 
ds_read_b64_tr_b4 v[112:113], v142 // (op 5084) Transpose load local data 

v_add_u32 v142, 84016, v7 // (op 5084) 
ds_read_b64_tr_b4 v[114:115], v142 // (op 5084) Transpose load local data 



v_add_u32 v142, 101472, v6 // (op 5087) 
ds_read_u8 v16, v142 // (op 5087) Load local data 




v_add_u32 v142, 81984, v7 // (op 6622) 
ds_read_b64_tr_b4 v[118:119], v142 // (op 6622) Transpose load local data 

v_add_u32 v142, 84032, v7 // (op 6622) 
ds_read_b64_tr_b4 v[120:121], v142 // (op 6622) Transpose load local data 



v_add_u32 v142, 101504, v6 // (op 6625) 
ds_read_u8 v23, v142 // (op 6625) Load local data 




v_add_u32 v142, 82000, v7 // (op 8160) 
ds_read_b64_tr_b4 v[122:123], v142 // (op 8160) Transpose load local data 

v_add_u32 v142, 84048, v7 // (op 8160) 
ds_read_b64_tr_b4 v[124:125], v142 // (op 8160) Transpose load local data 



v_add_u32 v142, 101536, v6 // (op 8163) 
ds_read_u8 v78, v142 // (op 8163) Load local data 




v_add_u32 v142, 82016, v7 // (op 9698) 
ds_read_b64_tr_b4 v[128:129], v142 // (op 9698) Transpose load local data 

v_add_u32 v142, 84064, v7 // (op 9698) 
ds_read_b64_tr_b4 v[130:131], v142 // (op 9698) Transpose load local data 



v_add_u32 v142, 101568, v6 // (op 9701) 
ds_read_u8 v79, v142 // (op 9701) Load local data 




v_add_u32 v142, 82032, v7 // (op 11236) 
ds_read_b64_tr_b4 v[132:133], v142 // (op 11236) Transpose load local data 

v_add_u32 v142, 84080, v7 // (op 11236) 
ds_read_b64_tr_b4 v[134:135], v142 // (op 11236) Transpose load local data 



v_add_u32 v142, 101600, v6 // (op 11239) 
ds_read_u8 v136, v142 // (op 11239) Load local data 

s_barrier  // (op 36122) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:101], v[148:155], a[0:3], v97, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 256) 
s_barrier  // (op 36007) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:101], v[156:163], a[4:7], v97, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 381) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:101], v[164:171], a[8:11], v97, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 557) 
v_mov_b64 v[214:215], s[24:25] // (op 34877) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:101], v[172:179], a[12:15], v97, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 733) 
v_mov_b64 v[216:217], v[144:145] // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:101], v[180:187], a[16:19], v97, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 909) 
s_add_u32 s23, 49152, s19 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:101], v[188:195], a[20:23], v97, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1085) 
s_mov_b32 m0, s23 // (op 34877) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:101], v[196:203], a[24:27], v97, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1261) 
s_mov_b32 s24, 384 // (op 34877) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:101], v[204:211], a[28:31], v97, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1437) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[102:105], v[148:155], a[32:35], v29, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2328) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:105], v[156:163], a[36:39], v29, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2282) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:105], v[164:171], a[40:43], v29, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2236) 
s_add_u32 m0, m0, 4096 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[102:105], v[172:179], a[44:47], v29, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2190) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[180:187], a[48:51], v29, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2144) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[188:195], a[52:55], v29, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2098) 
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[196:203], a[56:59], v29, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2052) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[204:211], a[60:63], v29, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2006) 
s_add_u32 m0, m0, 4096 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:111], v[148:155], a[64:67], v15, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3866) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[156:163], a[68:71], v15, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3820) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[164:171], a[72:75], v15, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3774) 
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[172:179], a[76:79], v15, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3728) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[180:187], a[80:83], v15, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3682) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:111], v[188:195], a[84:87], v15, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3636) 
s_add_u32 m0, m0, 4096 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[196:203], a[88:91], v15, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3590) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[204:211], a[92:95], v15, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3544) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[148:155], a[96:99], v16, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5404) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[156:163], a[100:103], v16, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5358) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[164:171], a[104:107], v16, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5312) 
s_add_u32 m0, m0, 4096 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[172:179], a[108:111], v16, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5266) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[180:187], a[112:115], v16, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5220) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[188:195], a[116:119], v16, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5174) 
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:115], v[196:203], a[120:123], v16, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5128) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[112:115], v[204:211], a[124:127], v16, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5082) 
s_add_u32 m0, m0, 4096 // (op 34877) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[118:121], v[148:155], a[128:131], v23, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6942) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:121], v[156:163], a[132:135], v23, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6896) 
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[118:121], v[164:171], a[136:139], v23, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6850) 
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[118:121], v[172:179], a[140:143], v23, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[118:121], v[180:187], a[144:147], v23, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6758) 
s_add_u32 m0, m0, 4096 // (op 34877) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[118:121], v[188:195], a[148:151], v23, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6712) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[118:121], v[196:203], a[152:155], v23, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6666) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[118:121], v[204:211], a[156:159], v23, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6620) 
v_add_co_u32 v216, s[50:51], v216, v214 // (op 34877) least significant half
v_addc_co_u32 v217, s[50:51], 0, v217, s[50:51] // (op 34877) most significant half

s_add_u32 m0, m0, 4096 // (op 34877) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:125], v[148:155], a[160:163], v78, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8480) 
buffer_load_dwordx4 v216, s[64:67], s24 offen offset: 0 lds // (op 34877) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[122:125], v[156:163], a[164:167], v78, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8434) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[122:125], v[164:171], a[168:171], v78, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8388) 
v_add_co_u32 v20, vcc, 256, v20 // (op 35071) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[122:125], v[172:179], a[172:175], v78, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8342) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 35071) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[122:125], v[180:187], a[176:179], v78, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8296) 
buffer_load_dword v27, v139, s[72:75], 0 offen offset:12 // (op 262) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[122:125], v[188:195], a[180:183], v78, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[122:125], v[196:203], a[184:187], v78, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8204) 
v_add_u32 v26, 8, v26 // (op 35199) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[122:125], v[204:211], a[188:191], v78, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8158) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 36000) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[128:131], v[148:155], a[192:195], v79, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 10018) 
s_lshl_b64 s[24:25], s[26:27], 5 // (op 34872) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[156:163], a[196:199], v79, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9972) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 34872) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[128:131], v[164:171], a[200:203], v79, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9926) 
v_lshrrev_b64 v[144:145], 3, s[50:51] // (op 34872) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[128:131], v[172:179], a[204:207], v79, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9880) 
s_lshl_b64 s[24:25], s[26:27], 7 // (op 34872) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[180:187], a[208:211], v79, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9834) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 34872) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[128:131], v[188:195], a[212:215], v79, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9788) 
s_lshr_b64 s[24:25], s[50:51], 3 // (op 34872) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[128:131], v[196:203], a[216:219], v79, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9742) 
s_mul_i32 s51, 3, s25 // (op 34872) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[128:131], v[204:211], a[220:223], v79, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9696) 
s_mul_i32 s50, 3, s24 // (op 34872) least significant
s_mul_hi_u32 s23, 3, s24 // (op 34872) most significant: high of low * low
s_add_u32 s51, s51, s23 // (op 34872) most significant: sum


v_mov_b32 v139, s50 // (op 34872) 
v_mov_b32 v142, s51 // (op 34872) 
v_add_co_u32 v214, s[24:25], v146, v139 // (op 34872) least significant half
v_addc_co_u32 v215, s[24:25], v147, v142, s[24:25] // (op 34872) most significant half




s_add_u32 s23, 81920, s18 // (op 34872) 
s_mov_b32 m0, s23 // (op 34872) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[148:155], a[224:227], v136, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11556) 
buffer_load_dwordx4 v214, s[52:55], 0 offen offset: 0 lds // (op 34872) Load value direct to lds


v_add_co_u32 v214, s[24:25], v214, v144 // (op 34872) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[132:135], v[156:163], a[228:231], v136, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11510) 
v_addc_co_u32 v215, s[24:25], 0, v215, s[24:25] // (op 34872) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[164:171], a[232:235], v136, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11464) 
s_add_u32 m0, m0, 4096 // (op 34872) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[172:179], a[236:239], v136, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11418) 
buffer_load_dwordx4 v214, s[52:55], 0 offen offset: 0 lds // (op 34872) Load value direct to lds

v_add_co_u32 v214, s[24:25], v214, v144 // (op 34872) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[132:135], v[180:187], a[240:243], v136, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11372) 
v_addc_co_u32 v215, s[24:25], 0, v215, s[24:25] // (op 34872) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[132:135], v[188:195], a[244:247], v136, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11326) 
s_add_u32 m0, m0, 4096 // (op 34872) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[196:203], a[248:251], v136, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11280) 
buffer_load_dwordx4 v214, s[52:55], 0 offen offset: 0 lds // (op 34872) Load value direct to lds

v_add_co_u32 v214, s[24:25], v214, v144 // (op 34872) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[204:211], a[252:255], v136, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11234) 
v_addc_co_u32 v215, s[24:25], 0, v215, s[24:25] // (op 34872) most significant half



s_add_u32 m0, m0, 4096 // (op 34872) 
buffer_load_dwordx4 v214, s[52:55], 0 offen offset: 0 lds // (op 34872) Load value direct to lds






s_lshl_b32 s23, s36, 2 // (op 261) 
s_mul_i32 s24, 3, s23 // (op 261) 

v_add_u32 v139, v141, s24 // (op 261) 


s_lshl_b64 s[24:25], s[26:27], 7 // (op 35007) 
s_lshl_b64 s[50:51], s[24:25], 2 // (op 35007) 

s_lshr_b64 s[24:25], s[50:51], 3 // (op 35007) 

s_lshl_b64 s[50:51], s[24:25], 1 // (op 35007) 

v_mov_b32 v142, s50 // (op 35007) 
v_mov_b32 v143, s51 // (op 35007) 
v_add_co_u32 v18, s[24:25], v18, v142 // (op 35007) least significant half
v_addc_co_u32 v19, s[24:25], v19, v143, s[24:25] // (op 35007) most significant half




buffer_load_dword v14, v139, s[68:71], 0 offen // (op 261) Load value


s_barrier  // (op 35983) 
s_lshl_b32 s23, s36, 3 // (op 35135) 



v_add_u32 v24, v24, s23 // (op 35135) 

v_add_u32 v139, 98304, v17 // (op 30) 
s_waitcnt vmcnt(19)
ds_write_b32 v139, v22 // (op 30) Store local data 

s_barrier  // (op 35966) 



v_add_u32 v139, 99328, v138 // (op 21) 
s_waitcnt vmcnt(14)
ds_write_b32 v139, v28 // (op 21) Store local data 

s_barrier  // (op 35955) 



ds_read_b128 v[30:33], v4 // (op 77) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 35990) 



v_add_u32 v139, 98304, v5 // (op 87) 
ds_read_u8 v25, v139 // (op 87) Load local data 




ds_read_b128 v[38:41], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 375) Load local data 



v_add_u32 v139, 98432, v5 // (op 379) 
ds_read_u8 v29, v139 // (op 379) Load local data 




ds_read_b128 v[46:49], v4 offset:8192 // (op 551) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 551) Load local data 



v_add_u32 v139, 98560, v5 // (op 555) 
ds_read_u8 v15, v139 // (op 555) Load local data 




ds_read_b128 v[54:57], v4 offset:12288 // (op 727) Load local data 
ds_read_b128 v[58:61], v4 offset:12352 // (op 727) Load local data 



v_add_u32 v139, 98688, v5 // (op 731) 
ds_read_u8 v16, v139 // (op 731) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 903) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 903) Load local data 



v_add_u32 v139, 98816, v5 // (op 907) 
ds_read_u8 v23, v139 // (op 907) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 1079) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 1079) Load local data 



v_add_u32 v139, 98944, v5 // (op 1083) 
ds_read_u8 v78, v139 // (op 1083) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 1255) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 1255) Load local data 



v_add_u32 v139, 99072, v5 // (op 1259) 
ds_read_u8 v79, v139 // (op 1259) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 1431) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 1431) Load local data 



v_add_u32 v139, 99200, v5 // (op 1435) 
ds_read_u8 v96, v139 // (op 1435) Load local data 

s_barrier  // (op 35939) 



ds_read_b64_tr_b4 v[98:99], v3 offset:32768 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:34816 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 35973) 


v_add_u32 v139, 99328, v0 // (op 82) 
ds_read_u8 v97, v139 // (op 82) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:32784 // (op 2000) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:34832 // (op 2000) Transpose load local data 


v_add_u32 v139, 99360, v0 // (op 2003) 
ds_read_u8 v106, v139 // (op 2003) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:32800 // (op 3538) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:34848 // (op 3538) Transpose load local data 


v_add_u32 v139, 99392, v0 // (op 3541) 
ds_read_u8 v107, v139 // (op 3541) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:32816 // (op 5076) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:34864 // (op 5076) Transpose load local data 


v_add_u32 v139, 99424, v0 // (op 5079) 
ds_read_u8 v116, v139 // (op 5079) Load local data 




ds_read_b64_tr_b4 v[118:119], v3 offset:32832 // (op 6614) Transpose load local data 
ds_read_b64_tr_b4 v[120:121], v3 offset:34880 // (op 6614) Transpose load local data 


v_add_u32 v139, 99456, v0 // (op 6617) 
ds_read_u8 v117, v139 // (op 6617) Load local data 




ds_read_b64_tr_b4 v[122:123], v3 offset:32848 // (op 8152) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v3 offset:34896 // (op 8152) Transpose load local data 


v_add_u32 v139, 99488, v0 // (op 8155) 
ds_read_u8 v126, v139 // (op 8155) Load local data 




ds_read_b64_tr_b4 v[128:129], v3 offset:32864 // (op 9690) Transpose load local data 
ds_read_b64_tr_b4 v[130:131], v3 offset:34912 // (op 9690) Transpose load local data 


v_add_u32 v139, 99520, v0 // (op 9693) 
ds_read_u8 v127, v139 // (op 9693) Load local data 




ds_read_b64_tr_b4 v[132:133], v3 offset:32880 // (op 11228) Transpose load local data 
ds_read_b64_tr_b4 v[134:135], v3 offset:34928 // (op 11228) Transpose load local data 


v_add_u32 v139, 99552, v0 // (op 11231) 
ds_read_u8 v136, v139 // (op 11231) Load local data 

s_ashr_i64 s[24:25], s[30:31], 7 // (op 9656) 
s_mov_b32 s23, s24 // (op 9656) convert

s_and_b32 s24, s23, 4294967294 // (op 9656) 

s_cmp_lt_u32 s22, s24 // (op 9656) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9656 // (op 9656) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9656 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9656:





































s_barrier  // (op 36088) 
s_mul_hi_i32 s0, s16, s10 // (op 34907) 
v_and_b32 v15, -8, v1 // (op 34907) 
s_add_i32 s1, s0, s16 // (op 34907) 

v_sub_u32 v16, v1, v15 // (op 34907) 

s_ashr_i32 s0, s1, 31 // (op 34907) 
v_lshlrev_b32 v15, 5, v16 // (op 34907) 

s_cmp_eq_i32 s10, 0 // (op 34907) 
s_mov_b32 s18, scc // (op 34907) 
v_lshrrev_b32 v16, 3, v1 // (op 34907) 
s_cmp_eq_i32 s18, 0 // (op 34907) 
s_cselect_b32 s19, 0, -1 // (op 34907) 

s_and_b32 s18, s11, 31 // (op 34907) 
s_lshl_b32 s22, 1, s18 // (op 34907) 
s_add_i32 s23, s22, s19 // (op 34907) 


s_and_b32 s19, s0, s23 // (op 34907) 


s_add_i32 s0, s1, s19 // (op 34907) 


s_ashr_i32 s1, s0, s18 // (op 34907) 


s_lshl_b32 s0, s11, 24 // (op 34907) 
s_ashr_i32 s18, s0, 31 // (op 34907) 

s_xor_b32 s0, s1, s18 // (op 34907) 

s_sub_i32 s1, s0, s18 // (op 34907) 


s_mul_hi_i32 s0, s1, s12 // (op 34907) 
s_add_i32 s18, s0, s1 // (op 34907) 

s_ashr_i32 s0, s18, 31 // (op 34907) 
s_cmp_eq_i32 s12, 0 // (op 34907) 
s_mov_b32 s19, scc // (op 34907) 
s_cmp_eq_i32 s19, 0 // (op 34907) 
s_cselect_b32 s22, 0, -1 // (op 34907) 

s_and_b32 s19, s13, 31 // (op 34907) 
s_lshl_b32 s23, 1, s19 // (op 34907) 
s_add_i32 s24, s23, s22 // (op 34907) 


s_and_b32 s25, s0, s24 // (op 34907) 


s_add_i32 s0, s18, s25 // (op 34907) 


s_ashr_i32 s18, s0, s19 // (op 34907) 


s_lshl_b32 s0, s13, 24 // (op 34907) 
s_ashr_i32 s19, s0, 31 // (op 34907) 

s_xor_b32 s0, s18, s19 // (op 34907) 

s_sub_i32 s18, s0, s19 // (op 34907) 


s_mul_i32 s0, s1, s15 // (op 34907) 

s_sub_u32 s1, s16, s0 // (op 34907) 

s_mul_i32 s0, s18, s15 // (op 34907) 
s_add_u32 s0, s0, s1 // (op 34907) 


s_mul_hi_i32 s1, s16, s9 // (op 34907) 
s_add_i32 s18, s1, s16 // (op 34907) 

s_ashr_i32 s1, s18, 31 // (op 34907) 
s_cmp_eq_i32 s9, 0 // (op 34907) 
s_mov_b32 s19, scc // (op 34907) 
s_cmp_eq_i32 s19, 0 // (op 34907) 
s_cselect_b32 s22, 0, -1 // (op 34907) 

s_and_b32 s19, s46, 31 // (op 34907) 
s_lshl_b32 s23, 1, s19 // (op 34907) 
s_add_i32 s24, s23, s22 // (op 34907) 


s_and_b32 s25, s1, s24 // (op 34907) 


s_add_i32 s1, s18, s25 // (op 34907) 


s_ashr_i32 s18, s1, s19 // (op 34907) 


s_lshl_b32 s1, s46, 24 // (op 34907) 
s_ashr_i32 s19, s1, 31 // (op 34907) 

s_xor_b32 s1, s18, s19 // (op 34907) 

s_sub_i32 s18, s1, s19 // (op 34907) 


s_add_u32 s50, s6, 256 // (op 34907) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34907) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34907) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34907) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34907) 

s_mul_i32 s1, s15, s24 // (op 34907) 

s_mul_i32 s19, s8, s1 // (op 34907) 

s_mul_i32 s1, s18, s19 // (op 34907) 

s_sub_u32 s19, s16, s1 // (op 34907) 

s_mov_b32 s1, s19 // (op 34907) 
s_mul_hi_i32 s22, s1, s14 // (op 34907) 
s_add_i32 s23, s22, s1 // (op 34907) 


s_ashr_i32 s1, s23, 31 // (op 34907) 
s_cmp_eq_i32 s14, 0 // (op 34907) 
s_mov_b32 s22, scc // (op 34907) 
s_cmp_eq_i32 s22, 0 // (op 34907) 
s_cselect_b32 s24, 0, -1 // (op 34907) 

s_and_b32 s22, s48, 31 // (op 34907) 
s_lshl_b32 s25, 1, s22 // (op 34907) 
s_add_i32 s49, s25, s24 // (op 34907) 


s_and_b32 s24, s1, s49 // (op 34907) 


s_add_i32 s1, s23, s24 // (op 34907) 


s_ashr_i32 s23, s1, s22 // (op 34907) 


s_lshl_b32 s1, s48, 24 // (op 34907) 
s_ashr_i32 s22, s1, 31 // (op 34907) 

s_xor_b32 s1, s23, s22 // (op 34907) 

s_sub_i32 s23, s1, s22 // (op 34907) 


s_mul_i32 s1, s47, s15 // (op 34907) 
s_add_u32 s24, s20, 256 // (op 34907) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34907) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34907) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34907) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34907) 

s_sub_i32 s22, s24, s1 // (op 34907) 

s_mul_i32 s24, s23, s22 // (op 34907) 


s_sub_u32 s25, s19, s24 // (op 34907) 


s_add_u32 s19, s25, s1 // (op 34907) 


s_cmp_eq_i32 s18, 0 // (op 34907) 
s_mov_b32 s1, scc // (op 34907) 

s_cmp_eq_i32 s1, 0 // (op 34907) 
s_cselect_b32 s18, s19, s0 // (op 34907) 



v_lshl_add_u32 v17, s18, 8, v15 // (op 34907) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 34907) 
v_mov_b32 v19, 0 // (op 34907) convert
v_mov_b32 v18, v17 // (op 34907) convert

s_lshr_b32 s18, s0, 1 // (op 34907) 

v_lshl_add_u32 v15, s18, 8, v16 // (op 34907) 


v_mul_lo_u32 v17, v15, s27 // (op 34907) most significant: low of low * high
v_mul_hi_u32 v20, v15, s26 // (op 34907) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 34907) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 34907) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 34907) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 34907) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 34907) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 34907) 

s_mul_i32 s1, s26, s31 // (op 34909) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 34909) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 34909) least significant
s_mul_hi_u32 s19, s26, s30 // (op 34909) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34909) most significant: sum
s_add_u32 s1, s1, s19 // (op 34909) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 34909) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 34909) 

s_mov_b32 s52, s28 // (op 34909) reg expression
s_mov_b32 s53, s29 // (op 34909) reg expression
s_mov_b32 s54, s0 // (op 34909) reg expression
s_mov_b32 s55, 131072 // (op 34909) call()

s_mul_hi_i32 s0, s16, s10 // (op 35037) 
v_lshrrev_b32 v15, 3, v1 // (op 35037) 
s_add_i32 s1, s0, s16 // (op 35037) 

v_and_b32 v16, -8, v1 // (op 35037) 
s_ashr_i32 s0, s1, 31 // (op 35037) 
v_sub_u32 v17, v1, v16 // (op 35037) 

s_cmp_eq_i32 s10, 0 // (op 35037) 
s_mov_b32 s18, scc // (op 35037) 
v_lshlrev_b32 v16, 4, v17 // (op 35037) 

s_cmp_eq_i32 s18, 0 // (op 35037) 
s_cselect_b32 s19, 0, -1 // (op 35037) 

s_and_b32 s18, s11, 31 // (op 35037) 
s_lshl_b32 s22, 1, s18 // (op 35037) 
s_add_i32 s23, s22, s19 // (op 35037) 


s_and_b32 s19, s0, s23 // (op 35037) 


s_add_i32 s0, s1, s19 // (op 35037) 


s_ashr_i32 s1, s0, s18 // (op 35037) 


s_lshl_b32 s0, s11, 24 // (op 35037) 
s_ashr_i32 s18, s0, 31 // (op 35037) 

s_xor_b32 s0, s1, s18 // (op 35037) 

s_sub_i32 s1, s0, s18 // (op 35037) 


s_mul_hi_i32 s0, s1, s12 // (op 35037) 
s_add_i32 s18, s0, s1 // (op 35037) 

s_ashr_i32 s0, s18, 31 // (op 35037) 
s_cmp_eq_i32 s12, 0 // (op 35037) 
s_mov_b32 s19, scc // (op 35037) 
s_cmp_eq_i32 s19, 0 // (op 35037) 
s_cselect_b32 s22, 0, -1 // (op 35037) 

s_and_b32 s19, s13, 31 // (op 35037) 
s_lshl_b32 s23, 1, s19 // (op 35037) 
s_add_i32 s24, s23, s22 // (op 35037) 


s_and_b32 s25, s0, s24 // (op 35037) 


s_add_i32 s0, s18, s25 // (op 35037) 


s_ashr_i32 s18, s0, s19 // (op 35037) 


s_lshl_b32 s0, s13, 24 // (op 35037) 
s_ashr_i32 s19, s0, 31 // (op 35037) 

s_xor_b32 s0, s18, s19 // (op 35037) 

s_sub_i32 s18, s0, s19 // (op 35037) 


s_add_u32 s50, s6, 256 // (op 35037) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35037) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35037) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35037) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35037) 

s_mul_i32 s0, s18, s24 // (op 35037) 

s_sub_i32 s18, s1, s0 // (op 35037) 


s_mul_hi_i32 s19, s16, s9 // (op 35037) 
s_add_i32 s0, s19, s16 // (op 35037) 

s_ashr_i32 s1, s0, 31 // (op 35037) 
s_cmp_eq_i32 s9, 0 // (op 35037) 
s_mov_b32 s19, scc // (op 35037) 
s_cmp_eq_i32 s19, 0 // (op 35037) 
s_cselect_b32 s22, 0, -1 // (op 35037) 

s_and_b32 s19, s46, 31 // (op 35037) 
s_lshl_b32 s23, 1, s19 // (op 35037) 
s_add_i32 s49, s23, s22 // (op 35037) 


s_and_b32 s22, s1, s49 // (op 35037) 


s_add_i32 s1, s0, s22 // (op 35037) 


s_ashr_i32 s0, s1, s19 // (op 35037) 


s_lshl_b32 s1, s46, 24 // (op 35037) 
s_ashr_i32 s19, s1, 31 // (op 35037) 

s_xor_b32 s1, s0, s19 // (op 35037) 

s_sub_i32 s0, s1, s19 // (op 35037) 


s_mul_i32 s1, s15, s24 // (op 35037) 

s_mul_i32 s19, s8, s1 // (op 35037) 

s_mul_i32 s1, s0, s19 // (op 35037) 

s_sub_u32 s19, s16, s1 // (op 35037) 

s_mul_hi_i32 s1, s19, s14 // (op 35037) 
s_add_i32 s22, s1, s19 // (op 35037) 


s_ashr_i32 s1, s22, 31 // (op 35037) 
s_cmp_eq_i32 s14, 0 // (op 35037) 
s_mov_b32 s19, scc // (op 35037) 
s_cmp_eq_i32 s19, 0 // (op 35037) 
s_cselect_b32 s23, 0, -1 // (op 35037) 

s_and_b32 s19, s48, 31 // (op 35037) 
s_lshl_b32 s24, 1, s19 // (op 35037) 
s_add_i32 s25, s24, s23 // (op 35037) 


s_and_b32 s23, s1, s25 // (op 35037) 


s_add_i32 s1, s22, s23 // (op 35037) 


s_ashr_i32 s22, s1, s19 // (op 35037) 


s_lshl_b32 s1, s48, 24 // (op 35037) 
s_ashr_i32 s19, s1, 31 // (op 35037) 

s_xor_b32 s1, s22, s19 // (op 35037) 

s_sub_i32 s22, s1, s19 // (op 35037) 


s_cmp_eq_i32 s0, 0 // (op 35037) 
s_mov_b32 s1, scc // (op 35037) 

s_cmp_eq_i32 s1, 0 // (op 35037) 
s_cselect_b32 s0, s22, s18 // (op 35037) 



v_lshl_add_u32 v17, s0, 8, v15 // (op 35037) 


v_mul_lo_u32 v21, v17, s33 // (op 35037) most significant: low of low * high
v_mul_hi_u32 v15, v17, s32 // (op 35037) most significant: high of low * low
v_mul_lo_u32 v20, v17, s32 // (op 35037) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 35037) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 35037) 
s_lshr_b32 s18, s0, 1 // (op 35037) 

v_lshl_add_u32 v15, s18, 8, v16 // (op 35037) 


v_mov_b32 v17, 0 // (op 35037) convert
v_mov_b32 v16, v15 // (op 35037) convert

v_add_co_u32 v22, s[0:1], v16, v20 // (op 35037) least significant half
v_addc_co_u32 v23, s[0:1], v17, v21, s[0:1] // (op 35037) most significant half



s_mul_i32 s1, s32, s7 // (op 35039) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 35039) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 35039) least significant
s_mul_hi_u32 s19, s32, s6 // (op 35039) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35039) most significant: sum
s_add_u32 s1, s1, s19 // (op 35039) most significant: sum


s_mov_b32 s64, s34 // (op 35039) reg expression
s_mov_b32 s65, s35 // (op 35039) reg expression
s_mov_b32 s66, s0 // (op 35039) reg expression
s_mov_b32 s67, 131072 // (op 35039) call()

s_mul_hi_i32 s0, s16, s10 // (op 35101) 
v_and_b32 v15, -64, v1 // (op 35101) 
s_add_i32 s1, s0, s16 // (op 35101) 

v_sub_u32 v16, v1, v15 // (op 35101) 

s_ashr_i32 s0, s1, 31 // (op 35101) 
v_lshlrev_b32 v15, 2, v16 // (op 35101) 

s_cmp_eq_i32 s10, 0 // (op 35101) 
s_mov_b32 s18, scc // (op 35101) 
v_lshrrev_b32 v16, 6, v1 // (op 35101) 
s_cmp_eq_i32 s18, 0 // (op 35101) 
s_cselect_b32 s19, 0, -1 // (op 35101) 

s_and_b32 s18, s11, 31 // (op 35101) 
s_lshl_b32 s22, 1, s18 // (op 35101) 
s_add_i32 s23, s22, s19 // (op 35101) 


s_and_b32 s19, s0, s23 // (op 35101) 


s_add_i32 s0, s1, s19 // (op 35101) 


s_ashr_i32 s1, s0, s18 // (op 35101) 


s_lshl_b32 s0, s11, 24 // (op 35101) 
s_ashr_i32 s18, s0, 31 // (op 35101) 

s_xor_b32 s0, s1, s18 // (op 35101) 

s_sub_i32 s1, s0, s18 // (op 35101) 


s_mul_hi_i32 s0, s1, s12 // (op 35101) 
s_add_i32 s18, s0, s1 // (op 35101) 

s_ashr_i32 s0, s18, 31 // (op 35101) 
s_cmp_eq_i32 s12, 0 // (op 35101) 
s_mov_b32 s19, scc // (op 35101) 
s_cmp_eq_i32 s19, 0 // (op 35101) 
s_cselect_b32 s22, 0, -1 // (op 35101) 

s_and_b32 s19, s13, 31 // (op 35101) 
s_lshl_b32 s23, 1, s19 // (op 35101) 
s_add_i32 s24, s23, s22 // (op 35101) 


s_and_b32 s25, s0, s24 // (op 35101) 


s_add_i32 s0, s18, s25 // (op 35101) 


s_ashr_i32 s18, s0, s19 // (op 35101) 


s_lshl_b32 s0, s13, 24 // (op 35101) 
s_ashr_i32 s19, s0, 31 // (op 35101) 

s_xor_b32 s0, s18, s19 // (op 35101) 

s_sub_i32 s18, s0, s19 // (op 35101) 


s_mul_i32 s0, s1, s15 // (op 35101) 

s_sub_u32 s1, s16, s0 // (op 35101) 

s_mul_i32 s0, s18, s15 // (op 35101) 
s_add_u32 s0, s0, s1 // (op 35101) 


s_mul_hi_i32 s1, s16, s9 // (op 35101) 
s_add_i32 s18, s1, s16 // (op 35101) 

s_ashr_i32 s1, s18, 31 // (op 35101) 
s_cmp_eq_i32 s9, 0 // (op 35101) 
s_mov_b32 s19, scc // (op 35101) 
s_cmp_eq_i32 s19, 0 // (op 35101) 
s_cselect_b32 s22, 0, -1 // (op 35101) 

s_and_b32 s19, s46, 31 // (op 35101) 
s_lshl_b32 s23, 1, s19 // (op 35101) 
s_add_i32 s24, s23, s22 // (op 35101) 


s_and_b32 s25, s1, s24 // (op 35101) 


s_add_i32 s1, s18, s25 // (op 35101) 


s_ashr_i32 s18, s1, s19 // (op 35101) 


s_lshl_b32 s1, s46, 24 // (op 35101) 
s_ashr_i32 s19, s1, 31 // (op 35101) 

s_xor_b32 s1, s18, s19 // (op 35101) 

s_sub_i32 s18, s1, s19 // (op 35101) 


s_add_u32 s50, s6, 256 // (op 35101) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35101) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35101) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35101) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35101) 

s_mul_i32 s1, s15, s24 // (op 35101) 

s_mul_i32 s19, s8, s1 // (op 35101) 

s_mul_i32 s1, s18, s19 // (op 35101) 

s_sub_u32 s19, s16, s1 // (op 35101) 

s_mov_b32 s1, s19 // (op 35101) 
s_mul_hi_i32 s22, s1, s14 // (op 35101) 
s_add_i32 s23, s22, s1 // (op 35101) 


s_ashr_i32 s1, s23, 31 // (op 35101) 
s_cmp_eq_i32 s14, 0 // (op 35101) 
s_mov_b32 s22, scc // (op 35101) 
s_cmp_eq_i32 s22, 0 // (op 35101) 
s_cselect_b32 s24, 0, -1 // (op 35101) 

s_and_b32 s22, s48, 31 // (op 35101) 
s_lshl_b32 s25, 1, s22 // (op 35101) 
s_add_i32 s49, s25, s24 // (op 35101) 


s_and_b32 s24, s1, s49 // (op 35101) 


s_add_i32 s1, s23, s24 // (op 35101) 


s_ashr_i32 s23, s1, s22 // (op 35101) 


s_lshl_b32 s1, s48, 24 // (op 35101) 
s_ashr_i32 s22, s1, 31 // (op 35101) 

s_xor_b32 s1, s23, s22 // (op 35101) 

s_sub_i32 s23, s1, s22 // (op 35101) 


s_mul_i32 s1, s47, s15 // (op 35101) 
s_add_u32 s24, s20, 256 // (op 35101) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 35101) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 35101) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 35101) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 35101) 

s_sub_i32 s22, s24, s1 // (op 35101) 

s_mul_i32 s24, s23, s22 // (op 35101) 


s_sub_u32 s25, s19, s24 // (op 35101) 


s_add_u32 s19, s25, s1 // (op 35101) 


s_cmp_eq_i32 s18, 0 // (op 35101) 
s_mov_b32 s1, scc // (op 35101) 

s_cmp_eq_i32 s1, 0 // (op 35101) 
s_cselect_b32 s18, s19, s0 // (op 35101) 



v_lshl_add_u32 v17, s18, 8, v15 // (op 35101) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 35101) 
s_lshr_b32 s18, s0, 1 // (op 35101) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 35101) 


v_mul_lo_u32 v16, v15, s36 // (op 35101) 

v_add_u32 v15, v17, v16 // (op 35101) 


s_mul_i32 s1, s36, s41 // (op 35103) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 35103) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 35103) least significant
s_mul_hi_u32 s19, s36, s40 // (op 35103) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35103) most significant: sum
s_add_u32 s1, s1, s19 // (op 35103) most significant: sum


s_mov_b32 s68, s38 // (op 35103) reg expression
s_mov_b32 s69, s39 // (op 35103) reg expression
s_mov_b32 s70, s0 // (op 35103) reg expression
s_mov_b32 s71, 131072 // (op 35103) call()

s_mul_hi_i32 s0, s16, s10 // (op 35165) 
s_add_i32 s1, s0, s16 // (op 35165) 

s_ashr_i32 s0, s1, 31 // (op 35165) 
s_cmp_eq_i32 s10, 0 // (op 35165) 
s_mov_b32 s18, scc // (op 35165) 
s_cmp_eq_i32 s18, 0 // (op 35165) 
s_cselect_b32 s19, 0, -1 // (op 35165) 

s_and_b32 s18, s11, 31 // (op 35165) 
s_lshl_b32 s22, 1, s18 // (op 35165) 
s_add_i32 s23, s22, s19 // (op 35165) 


s_and_b32 s19, s0, s23 // (op 35165) 


s_add_i32 s0, s1, s19 // (op 35165) 


s_ashr_i32 s1, s0, s18 // (op 35165) 


s_lshl_b32 s0, s11, 24 // (op 35165) 
s_ashr_i32 s18, s0, 31 // (op 35165) 

s_xor_b32 s0, s1, s18 // (op 35165) 

s_sub_i32 s1, s0, s18 // (op 35165) 


s_mul_hi_i32 s0, s1, s12 // (op 35165) 
s_add_i32 s18, s0, s1 // (op 35165) 

s_ashr_i32 s0, s18, 31 // (op 35165) 
s_cmp_eq_i32 s12, 0 // (op 35165) 
s_mov_b32 s19, scc // (op 35165) 
s_cmp_eq_i32 s19, 0 // (op 35165) 
s_cselect_b32 s22, 0, -1 // (op 35165) 

s_and_b32 s19, s13, 31 // (op 35165) 
s_lshl_b32 s23, 1, s19 // (op 35165) 
s_add_i32 s24, s23, s22 // (op 35165) 


s_and_b32 s25, s0, s24 // (op 35165) 


s_add_i32 s0, s18, s25 // (op 35165) 


s_ashr_i32 s18, s0, s19 // (op 35165) 


s_lshl_b32 s0, s13, 24 // (op 35165) 
s_ashr_i32 s19, s0, 31 // (op 35165) 

s_xor_b32 s0, s18, s19 // (op 35165) 

s_sub_i32 s18, s0, s19 // (op 35165) 


s_add_u32 s50, s6, 256 // (op 35165) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 35165) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 35165) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 35165) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 35165) 

s_mul_i32 s0, s18, s24 // (op 35165) 

s_sub_i32 s18, s1, s0 // (op 35165) 


s_mul_hi_i32 s19, s16, s9 // (op 35165) 
s_add_i32 s0, s19, s16 // (op 35165) 

s_ashr_i32 s1, s0, 31 // (op 35165) 
s_cmp_eq_i32 s9, 0 // (op 35165) 
s_mov_b32 s19, scc // (op 35165) 
s_cmp_eq_i32 s19, 0 // (op 35165) 
s_cselect_b32 s22, 0, -1 // (op 35165) 

s_and_b32 s19, s46, 31 // (op 35165) 
s_lshl_b32 s23, 1, s19 // (op 35165) 
s_add_i32 s49, s23, s22 // (op 35165) 


s_and_b32 s22, s1, s49 // (op 35165) 


s_add_i32 s1, s0, s22 // (op 35165) 


s_ashr_i32 s0, s1, s19 // (op 35165) 


s_lshl_b32 s1, s46, 24 // (op 35165) 
s_ashr_i32 s19, s1, 31 // (op 35165) 

s_xor_b32 s1, s0, s19 // (op 35165) 

s_sub_i32 s0, s1, s19 // (op 35165) 


s_mul_i32 s1, s15, s24 // (op 35165) 

s_mul_i32 s19, s8, s1 // (op 35165) 

s_mul_i32 s1, s0, s19 // (op 35165) 

s_sub_u32 s19, s16, s1 // (op 35165) 

s_mul_hi_i32 s1, s19, s14 // (op 35165) 
s_add_i32 s22, s1, s19 // (op 35165) 


s_ashr_i32 s1, s22, 31 // (op 35165) 
s_cmp_eq_i32 s14, 0 // (op 35165) 
s_mov_b32 s19, scc // (op 35165) 
s_cmp_eq_i32 s19, 0 // (op 35165) 
s_cselect_b32 s23, 0, -1 // (op 35165) 

s_and_b32 s19, s48, 31 // (op 35165) 
s_lshl_b32 s24, 1, s19 // (op 35165) 
s_add_i32 s25, s24, s23 // (op 35165) 


s_and_b32 s23, s1, s25 // (op 35165) 


s_add_i32 s1, s22, s23 // (op 35165) 


s_ashr_i32 s22, s1, s19 // (op 35165) 


s_lshl_b32 s1, s48, 24 // (op 35165) 
s_ashr_i32 s19, s1, 31 // (op 35165) 

s_xor_b32 s1, s22, s19 // (op 35165) 

s_sub_i32 s22, s1, s19 // (op 35165) 


s_cmp_eq_i32 s0, 0 // (op 35165) 
s_mov_b32 s1, scc // (op 35165) 

s_cmp_eq_i32 s1, 0 // (op 35165) 
s_cselect_b32 s0, s22, s18 // (op 35165) 



v_lshl_add_u32 v16, s0, 8, v1 // (op 35165) 

v_mul_lo_u32 v17, v16, s42 // (op 35165) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 35165) 
s_lshr_b32 s18, s0, 1 // (op 35165) 

v_lshl_add_u32 v16, s18, 3, v17 // (op 35165) 


s_mul_i32 s1, s42, s7 // (op 35167) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 35167) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 35167) least significant
s_mul_hi_u32 s19, s42, s6 // (op 35167) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35167) most significant: sum
s_add_u32 s1, s1, s19 // (op 35167) most significant: sum


s_mov_b32 s72, s44 // (op 35167) reg expression
s_mov_b32 s73, s45 // (op 35167) reg expression
s_mov_b32 s74, s0 // (op 35167) reg expression
s_mov_b32 s75, 131072 // (op 35167) call()

v_and_b32 v17, -8, v1 // (op 35263) 
v_sub_u32 v20, v1, v17 // (op 35263) 

v_lshlrev_b32 v17, 5, v20 // (op 35263) 

v_lshrrev_b32 v20, 3, v1 // (op 35263) 
v_lshl_add_u32 v21, v20, 8, v17 // (op 35263) 


v_lshlrev_b32 v17, 2, v21 // (op 35263) 

v_lshrrev_b32 v20, 3, v17 // (op 35263) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 35263) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 35320) 
v_sub_u32 v20, v1, v17 // (op 35320) 

v_lshlrev_b32 v17, 4, v20 // (op 35320) 

v_lshrrev_b32 v20, 3, v1 // (op 35320) 
v_lshl_add_u32 v21, v20, 7, v17 // (op 35320) 


s_nop 0
v_readfirstlane_b32 s1, v21 // (op 35320) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 35377) 
v_sub_u32 v20, v1, v17 // (op 35377) 

v_lshlrev_b32 v17, 2, v20 // (op 35377) 

v_lshrrev_b32 v20, 6, v1 // (op 35377) 
v_lshl_add_u32 v21, v20, 8, v17 // (op 35377) 


v_lshlrev_b32 v17, 2, v1 // (op 35435) 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 11200) 
s_and_b32 s22, s18, 4294967294 // (op 11200) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 11199) 
s_mov_b32 s23, s18 // (op 11199) convert

s_cmp_lt_u32 s22, s23 // (op 11199) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11199 // (op 11199) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11199 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11199:
s_barrier  // (op 35936) 
s_add_u32 s22, s22, 1 // (op 11201) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 34859) 
s_lshl_b64 s[24:25], s[18:19], 2 // (op 34859) 

v_lshrrev_b64 v[24:25], 3, s[24:25] // (op 34859) 


v_mov_b64 v[26:27], v[18:19] // (op 34859) 
v_mov_b64 v[28:29], v[26:27] // (op 34859) reg expression
s_add_u32 s18, 32768, s0 // (op 34859) 
s_mov_b32 m0, s18 // (op 34859) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34859) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34859) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34859) most significant half

s_add_u32 m0, m0, 4096 // (op 34859) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34859) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34859) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34859) most significant half

s_add_u32 m0, m0, 4096 // (op 34859) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34859) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34859) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34859) most significant half

s_add_u32 m0, m0, 4096 // (op 34859) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34859) Load value direct to lds




s_lshl_b64 s[18:19], s[26:27], 7 // (op 34897) 
s_lshl_b64 s[24:25], s[18:19], 2 // (op 34897) 

s_lshr_b64 s[18:19], s[24:25], 3 // (op 34897) 

v_mov_b32 v14, s18 // (op 34897) 
v_mov_b32 v20, s19 // (op 34897) 
v_add_co_u32 v18, s[24:25], v18, v14 // (op 34897) least significant half
v_addc_co_u32 v19, s[24:25], v19, v20, s[24:25] // (op 34897) most significant half




s_barrier  // (op 35952) 

s_lshl_b64 s[18:19], s[32:33], 5 // (op 34865) 
v_mov_b64 v[24:25], s[18:19] // (op 34865) convert


v_mov_b64 v[26:27], v[22:23] // (op 34865) 
v_mov_b64 v[28:29], v[26:27] // (op 34865) 
s_mov_b32 m0, s1 // (op 34865) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34865) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34865) most significant half

s_add_u32 m0, m0, 4096 // (op 34865) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34865) Load value direct to lds



v_add_co_u32 v22, vcc, 128, v22 // (op 35029) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 35029) most significant half


v_mov_b32 v14, v15 // (op 195) 
v_mov_b32 v20, v14 // (op 195) reg expression

buffer_load_dword v24, v20, s[68:71], 0 offen // (op 195) Load value


s_lshl_b32 s18, s36, 2 // (op 35093) 
v_add_u32 v15, v15, s18 // (op 35093) 



s_barrier  // (op 35969) 
v_mov_b32 v14, v16 // (op 196) 




v_add_u32 v20, 99328, v21 // (op 192) 
buffer_load_dword v25, v14, s[72:75], 0 offen // (op 196) Load value

v_add_u32 v16, 4, v16 // (op 35157) 
s_waitcnt vmcnt(1)
ds_write_b32 v20, v24 // (op 192) Store local data 


s_barrier  // (op 35942) 



s_barrier  // (op 35986) 
ds_read_b64_tr_b4 v[26:27], v3 offset:32768 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v3 offset:34816 // (op 183) Transpose load local data 
v_add_u32 v14, 98304, v17 // (op 193) 
s_barrier  // (op 35959) 
s_waitcnt vmcnt(0)
ds_write_b32 v14, v25 // (op 193) Store local data 





ds_read_b128 v[30:33], v4 // (op 185) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 35976) 


v_add_u32 v14, 99328, v0 // (op 187) 
ds_read_u8 v20, v14 // (op 187) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 35993) 



v_add_u32 v14, 98304, v5 // (op 189) 
ds_read_u8 v24, v14 // (op 189) Load local data 




ds_read_b128 v[38:41], v4 offset:4096 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 190) 
ds_read_b128 v[42:45], v4 offset:4160 // (op 354) Load local data 



v_add_u32 v14, 98432, v5 // (op 358) 
ds_read_u8 v25, v14 // (op 358) Load local data 




ds_read_b128 v[46:49], v4 offset:8192 // (op 530) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 359) 
ds_read_b128 v[50:53], v4 offset:8256 // (op 530) Load local data 



v_add_u32 v14, 98560, v5 // (op 534) 
ds_read_u8 v54, v14 // (op 534) Load local data 




ds_read_b128 v[56:59], v4 offset:12288 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[46:53], a[8:11], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 535) 
ds_read_b128 v[60:63], v4 offset:12352 // (op 706) Load local data 



v_add_u32 v14, 98688, v5 // (op 710) 
ds_read_u8 v55, v14 // (op 710) Load local data 




ds_read_b128 v[64:67], v4 offset:16384 // (op 882) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[56:63], a[12:15], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 711) 
ds_read_b128 v[68:71], v4 offset:16448 // (op 882) Load local data 



v_add_u32 v14, 98816, v5 // (op 886) 
ds_read_u8 v72, v14 // (op 886) Load local data 




ds_read_b128 v[74:77], v4 offset:20480 // (op 1058) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[64:71], a[16:19], v20, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 887) 
ds_read_b128 v[78:81], v4 offset:20544 // (op 1058) Load local data 



v_add_u32 v14, 98944, v5 // (op 1062) 
ds_read_u8 v73, v14 // (op 1062) Load local data 




ds_read_b128 v[82:85], v4 offset:24576 // (op 1234) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[74:81], a[20:23], v20, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1063) 
ds_read_b128 v[86:89], v4 offset:24640 // (op 1234) Load local data 



v_add_u32 v14, 99072, v5 // (op 1238) 
ds_read_u8 v90, v14 // (op 1238) Load local data 




ds_read_b128 v[92:95], v4 offset:28672 // (op 1410) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[82:89], a[24:27], v20, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1239) 
ds_read_b128 v[96:99], v4 offset:28736 // (op 1410) Load local data 



v_add_u32 v14, 99200, v5 // (op 1414) 
ds_read_u8 v91, v14 // (op 1414) Load local data 




ds_read_b64_tr_b4 v[100:101], v3 offset:32784 // (op 1760) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:29], v[92:99], a[28:31], v20, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1415) 
ds_read_b64_tr_b4 v[102:103], v3 offset:34832 // (op 1760) Transpose load local data 




v_add_u32 v14, 99360, v0 // (op 1763) 
ds_read_u8 v20, v14 // (op 1763) Load local data 




ds_read_b64_tr_b4 v[26:27], v3 offset:32800 // (op 3298) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:103], v[30:37], a[32:35], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1765) 
ds_read_b64_tr_b4 v[28:29], v3 offset:34848 // (op 3298) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:103], v[38:45], a[36:39], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1796) 
v_add_u32 v14, 99392, v0 // (op 3301) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[46:53], a[40:43], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1827) 
ds_read_u8 v104, v14 // (op 3301) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[56:63], a[44:47], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1858) 
ds_read_b64_tr_b4 v[106:107], v3 offset:32816 // (op 4836) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:103], v[64:71], a[48:51], v20, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1889) 
ds_read_b64_tr_b4 v[108:109], v3 offset:34864 // (op 4836) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:103], v[74:81], a[52:55], v20, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1920) 
v_add_u32 v14, 99424, v0 // (op 4839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[82:89], a[56:59], v20, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1951) 
ds_read_u8 v105, v14 // (op 4839) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[92:99], a[60:63], v20, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1982) 
ds_read_b64_tr_b4 v[110:111], v3 offset:32832 // (op 6374) Transpose load local data 


ds_read_b64_tr_b4 v[112:113], v3 offset:34880 // (op 6374) Transpose load local data 


s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[26:29], v[30:37], a[64:67], v104, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3303) 
v_add_u32 v14, 99456, v0 // (op 6377) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[38:45], a[68:71], v104, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3334) 
ds_read_u8 v20, v14 // (op 6377) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[46:53], a[72:75], v104, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3365) 
ds_read_b64_tr_b4 v[100:101], v3 offset:32848 // (op 7912) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[56:63], a[76:79], v104, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3396) 
ds_read_b64_tr_b4 v[102:103], v3 offset:34896 // (op 7912) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[26:29], v[64:71], a[80:83], v104, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3427) 
v_add_u32 v14, 99488, v0 // (op 7915) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[26:29], v[74:81], a[84:87], v104, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3458) 
ds_read_u8 v114, v14 // (op 7915) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[82:89], a[88:91], v104, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3489) 
ds_read_b64_tr_b4 v[116:117], v3 offset:32864 // (op 9450) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[92:99], a[92:95], v104, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3520) 
ds_read_b64_tr_b4 v[118:119], v3 offset:34912 // (op 9450) Transpose load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[106:109], v[30:37], a[96:99], v105, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4841) 
v_add_u32 v14, 99520, v0 // (op 9453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[106:109], v[38:45], a[100:103], v105, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4872) 
ds_read_u8 v104, v14 // (op 9453) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[106:109], v[46:53], a[104:107], v105, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4903) 
ds_read_b64_tr_b4 v[26:27], v3 offset:32880 // (op 10988) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[106:109], v[56:63], a[108:111], v105, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4934) 
ds_read_b64_tr_b4 v[28:29], v3 offset:34928 // (op 10988) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[106:109], v[64:71], a[112:115], v105, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4965) 
v_add_u32 v14, 99552, v0 // (op 10991) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[106:109], v[74:81], a[116:119], v105, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4996) 
ds_read_u8 v115, v14 // (op 10991) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[106:109], v[82:89], a[120:123], v105, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5027) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[106:109], v[92:99], a[124:127], v105, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5058) 


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[110:113], v[30:37], a[128:131], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6379) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[110:113], v[38:45], a[132:135], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[110:113], v[46:53], a[136:139], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6441) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[110:113], v[56:63], a[140:143], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6472) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[110:113], v[64:71], a[144:147], v20, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[110:113], v[74:81], a[148:151], v20, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6534) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:113], v[82:89], a[152:155], v20, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6565) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[110:113], v[92:99], a[156:159], v20, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6596) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[30:37], a[160:163], v114, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7917) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[38:45], a[164:167], v114, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7948) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[46:53], a[168:171], v114, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7979) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[56:63], a[172:175], v114, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8010) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[64:71], a[176:179], v114, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8041) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[74:81], a[180:183], v114, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8072) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[82:89], a[184:187], v114, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8103) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[92:99], a[188:191], v114, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8134) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[30:37], a[192:195], v104, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9455) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[38:45], a[196:199], v104, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9486) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[46:53], a[200:203], v104, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9517) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[56:63], a[204:207], v104, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[64:71], a[208:211], v104, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9579) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:119], v[74:81], a[212:215], v104, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9610) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:119], v[82:89], a[216:219], v104, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9641) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:119], v[92:99], a[220:223], v104, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9672) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[26:29], v[30:37], a[224:227], v115, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 10993) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[26:29], v[38:45], a[228:231], v115, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11024) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[46:53], a[232:235], v115, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11055) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[56:63], a[236:239], v115, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11086) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[26:29], v[64:71], a[240:243], v115, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11117) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[26:29], v[74:81], a[244:247], v115, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11148) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[26:29], v[82:89], a[248:251], v115, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11179) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[26:29], v[92:99], a[252:255], v115, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 11210) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 11199) 
s_mov_b32 s23, s18 // (op 11199) convert

s_cmp_lt_u32 s22, s23 // (op 11199) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11199 // (op 11199) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11199 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11199:























s_barrier  // (op 36110) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 13693) 
v_accvgpr_read v7, a5 // (op 13693) 
v_accvgpr_read v8, a6 // (op 13693) 
v_accvgpr_read v9, a7 // (op 13693) 
v_mul_f32 v14, v11, v6 // (op 13693) 
v_mul_f32 v15, v11, v7 // (op 13693) 
v_mul_f32 v16, v11, v8 // (op 13693) 
v_mul_f32 v17, v11, v9 // (op 13693) 

v_lshrrev_b32 v6, 16, v14 // (op 13695) 
v_lshrrev_b32 v7, 16, v15 // (op 13695) 
v_lshrrev_b32 v8, 16, v16 // (op 13695) 
v_lshrrev_b32 v9, 16, v17 // (op 13695) 


v_pack_b32_f16 v4, v6, v7 // (op 13696) 
v_pack_b32_f16 v5, v8, v9 // (op 13696) 

s_lshl_b32 s0, s4, 6 // (op 13696) 
v_add_u32 v0, v13, s0 // (op 13696) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13696) Store value



v_accvgpr_read v6, a8 // (op 13707) 
v_accvgpr_read v7, a9 // (op 13707) 
v_accvgpr_read v8, a10 // (op 13707) 
v_accvgpr_read v9, a11 // (op 13707) 
v_mul_f32 v14, v11, v6 // (op 13707) 
v_mul_f32 v15, v11, v7 // (op 13707) 
v_mul_f32 v16, v11, v8 // (op 13707) 
v_mul_f32 v17, v11, v9 // (op 13707) 

v_lshrrev_b32 v6, 16, v14 // (op 13709) 
v_lshrrev_b32 v7, 16, v15 // (op 13709) 
v_lshrrev_b32 v8, 16, v16 // (op 13709) 
v_lshrrev_b32 v9, 16, v17 // (op 13709) 


v_pack_b32_f16 v4, v6, v7 // (op 13710) 
v_pack_b32_f16 v5, v8, v9 // (op 13710) 

s_lshl_b32 s0, s4, 7 // (op 13710) 
v_add_u32 v0, v13, s0 // (op 13710) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13710) Store value



v_accvgpr_read v6, a12 // (op 13721) 
v_accvgpr_read v7, a13 // (op 13721) 
v_accvgpr_read v8, a14 // (op 13721) 
v_accvgpr_read v9, a15 // (op 13721) 
v_mul_f32 v14, v11, v6 // (op 13721) 
v_mul_f32 v15, v11, v7 // (op 13721) 
v_mul_f32 v16, v11, v8 // (op 13721) 
v_mul_f32 v17, v11, v9 // (op 13721) 

v_lshrrev_b32 v6, 16, v14 // (op 13723) 
v_lshrrev_b32 v7, 16, v15 // (op 13723) 
v_lshrrev_b32 v8, 16, v16 // (op 13723) 
v_lshrrev_b32 v9, 16, v17 // (op 13723) 


v_pack_b32_f16 v4, v6, v7 // (op 13724) 
v_pack_b32_f16 v5, v8, v9 // (op 13724) 

s_lshl_b32 s0, s4, 6 // (op 13724) 
s_mul_i32 s1, 3, s0 // (op 13724) 

v_add_u32 v0, v13, s1 // (op 13724) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13724) Store value



v_accvgpr_read v6, a16 // (op 13735) 
v_accvgpr_read v7, a17 // (op 13735) 
v_accvgpr_read v8, a18 // (op 13735) 
v_accvgpr_read v9, a19 // (op 13735) 
v_mul_f32 v14, v11, v6 // (op 13735) 
v_mul_f32 v15, v11, v7 // (op 13735) 
v_mul_f32 v16, v11, v8 // (op 13735) 
v_mul_f32 v17, v11, v9 // (op 13735) 

v_lshrrev_b32 v6, 16, v14 // (op 13737) 
v_lshrrev_b32 v7, 16, v15 // (op 13737) 
v_lshrrev_b32 v8, 16, v16 // (op 13737) 
v_lshrrev_b32 v9, 16, v17 // (op 13737) 


v_pack_b32_f16 v4, v6, v7 // (op 13738) 
v_pack_b32_f16 v5, v8, v9 // (op 13738) 

s_lshl_b32 s0, s4, 8 // (op 13738) 
v_add_u32 v0, v13, s0 // (op 13738) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13738) Store value



v_accvgpr_read v6, a20 // (op 13749) 
v_accvgpr_read v7, a21 // (op 13749) 
v_accvgpr_read v8, a22 // (op 13749) 
v_accvgpr_read v9, a23 // (op 13749) 
v_mul_f32 v14, v11, v6 // (op 13749) 
v_mul_f32 v15, v11, v7 // (op 13749) 
v_mul_f32 v16, v11, v8 // (op 13749) 
v_mul_f32 v17, v11, v9 // (op 13749) 

v_lshrrev_b32 v6, 16, v14 // (op 13751) 
v_lshrrev_b32 v7, 16, v15 // (op 13751) 
v_lshrrev_b32 v8, 16, v16 // (op 13751) 
v_lshrrev_b32 v9, 16, v17 // (op 13751) 


v_pack_b32_f16 v4, v6, v7 // (op 13752) 
v_pack_b32_f16 v5, v8, v9 // (op 13752) 

s_lshl_b32 s0, s4, 6 // (op 13752) 
s_mul_i32 s1, 5, s0 // (op 13752) 

v_add_u32 v0, v13, s1 // (op 13752) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13752) Store value



v_accvgpr_read v6, a24 // (op 13763) 
v_accvgpr_read v7, a25 // (op 13763) 
v_accvgpr_read v8, a26 // (op 13763) 
v_accvgpr_read v9, a27 // (op 13763) 
v_mul_f32 v14, v11, v6 // (op 13763) 
v_mul_f32 v15, v11, v7 // (op 13763) 
v_mul_f32 v16, v11, v8 // (op 13763) 
v_mul_f32 v17, v11, v9 // (op 13763) 

v_lshrrev_b32 v6, 16, v14 // (op 13765) 
v_lshrrev_b32 v7, 16, v15 // (op 13765) 
v_lshrrev_b32 v8, 16, v16 // (op 13765) 
v_lshrrev_b32 v9, 16, v17 // (op 13765) 


v_pack_b32_f16 v4, v6, v7 // (op 13766) 
v_pack_b32_f16 v5, v8, v9 // (op 13766) 

s_lshl_b32 s0, s4, 6 // (op 13766) 
s_mul_i32 s1, 6, s0 // (op 13766) 

v_add_u32 v0, v13, s1 // (op 13766) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13766) Store value



v_accvgpr_read v6, a28 // (op 13777) 
v_accvgpr_read v7, a29 // (op 13777) 
v_accvgpr_read v8, a30 // (op 13777) 
v_accvgpr_read v9, a31 // (op 13777) 
v_mul_f32 v14, v11, v6 // (op 13777) 
v_mul_f32 v15, v11, v7 // (op 13777) 
v_mul_f32 v16, v11, v8 // (op 13777) 
v_mul_f32 v17, v11, v9 // (op 13777) 

v_lshrrev_b32 v6, 16, v14 // (op 13779) 
v_lshrrev_b32 v7, 16, v15 // (op 13779) 
v_lshrrev_b32 v8, 16, v16 // (op 13779) 
v_lshrrev_b32 v9, 16, v17 // (op 13779) 


v_pack_b32_f16 v4, v6, v7 // (op 13780) 
v_pack_b32_f16 v5, v8, v9 // (op 13780) 

s_lshl_b32 s0, s4, 6 // (op 13780) 
s_mul_i32 s1, 7, s0 // (op 13780) 

v_add_u32 v0, v13, s1 // (op 13780) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13780) Store value



v_accvgpr_read v6, a32 // (op 13833) 
v_accvgpr_read v7, a33 // (op 13833) 
v_accvgpr_read v8, a34 // (op 13833) 
v_accvgpr_read v9, a35 // (op 13833) 
v_mul_f32 v14, v11, v6 // (op 13833) 
v_mul_f32 v15, v11, v7 // (op 13833) 
v_mul_f32 v16, v11, v8 // (op 13833) 
v_mul_f32 v17, v11, v9 // (op 13833) 

v_lshrrev_b32 v6, 16, v14 // (op 13835) 
v_lshrrev_b32 v7, 16, v15 // (op 13835) 
v_lshrrev_b32 v8, 16, v16 // (op 13835) 
v_lshrrev_b32 v9, 16, v17 // (op 13835) 


v_pack_b32_f16 v4, v6, v7 // (op 13837) 
v_pack_b32_f16 v5, v8, v9 // (op 13837) 

v_add_u32 v0, 64, v13 // (op 13837) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13837) Store value



v_accvgpr_read v6, a36 // (op 13855) 
v_accvgpr_read v7, a37 // (op 13855) 
v_accvgpr_read v8, a38 // (op 13855) 
v_accvgpr_read v9, a39 // (op 13855) 
v_mul_f32 v14, v11, v6 // (op 13855) 
v_mul_f32 v15, v11, v7 // (op 13855) 
v_mul_f32 v16, v11, v8 // (op 13855) 
v_mul_f32 v17, v11, v9 // (op 13855) 

v_lshrrev_b32 v6, 16, v14 // (op 13857) 
v_lshrrev_b32 v7, 16, v15 // (op 13857) 
v_lshrrev_b32 v8, 16, v16 // (op 13857) 
v_lshrrev_b32 v9, 16, v17 // (op 13857) 


v_pack_b32_f16 v4, v6, v7 // (op 13839) 
v_pack_b32_f16 v5, v8, v9 // (op 13839) 

s_lshl_b32 s0, s4, 6 // (op 13839) 
s_add_u32 s0, s0, 64 // (op 13839) 
v_add_u32 v0, v13, s0 // (op 13839) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13839) Store value



v_accvgpr_read v6, a40 // (op 13861) 
v_accvgpr_read v7, a41 // (op 13861) 
v_accvgpr_read v8, a42 // (op 13861) 
v_accvgpr_read v9, a43 // (op 13861) 
v_mul_f32 v14, v11, v6 // (op 13861) 
v_mul_f32 v15, v11, v7 // (op 13861) 
v_mul_f32 v16, v11, v8 // (op 13861) 
v_mul_f32 v17, v11, v9 // (op 13861) 

v_lshrrev_b32 v6, 16, v14 // (op 13863) 
v_lshrrev_b32 v7, 16, v15 // (op 13863) 
v_lshrrev_b32 v8, 16, v16 // (op 13863) 
v_lshrrev_b32 v9, 16, v17 // (op 13863) 


v_pack_b32_f16 v4, v6, v7 // (op 13841) 
v_pack_b32_f16 v5, v8, v9 // (op 13841) 

s_lshl_b32 s0, s4, 7 // (op 13841) 
s_add_u32 s0, s0, 64 // (op 13841) 
v_add_u32 v0, v13, s0 // (op 13841) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13841) Store value



v_accvgpr_read v6, a44 // (op 13867) 
v_accvgpr_read v7, a45 // (op 13867) 
v_accvgpr_read v8, a46 // (op 13867) 
v_accvgpr_read v9, a47 // (op 13867) 
v_mul_f32 v14, v11, v6 // (op 13867) 
v_mul_f32 v15, v11, v7 // (op 13867) 
v_mul_f32 v16, v11, v8 // (op 13867) 
v_mul_f32 v17, v11, v9 // (op 13867) 

v_lshrrev_b32 v6, 16, v14 // (op 13869) 
v_lshrrev_b32 v7, 16, v15 // (op 13869) 
v_lshrrev_b32 v8, 16, v16 // (op 13869) 
v_lshrrev_b32 v9, 16, v17 // (op 13869) 


v_pack_b32_f16 v4, v6, v7 // (op 13843) 
v_pack_b32_f16 v5, v8, v9 // (op 13843) 

s_lshl_b32 s0, s4, 6 // (op 13843) 
s_mul_i32 s1, 3, s0 // (op 13843) 
s_add_u32 s1, s1, 64 // (op 13843) 

v_add_u32 v0, v13, s1 // (op 13843) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13843) Store value



v_accvgpr_read v6, a48 // (op 13873) 
v_accvgpr_read v7, a49 // (op 13873) 
v_accvgpr_read v8, a50 // (op 13873) 
v_accvgpr_read v9, a51 // (op 13873) 
v_mul_f32 v14, v11, v6 // (op 13873) 
v_mul_f32 v15, v11, v7 // (op 13873) 
v_mul_f32 v16, v11, v8 // (op 13873) 
v_mul_f32 v17, v11, v9 // (op 13873) 

v_lshrrev_b32 v6, 16, v14 // (op 13875) 
v_lshrrev_b32 v7, 16, v15 // (op 13875) 
v_lshrrev_b32 v8, 16, v16 // (op 13875) 
v_lshrrev_b32 v9, 16, v17 // (op 13875) 


v_pack_b32_f16 v4, v6, v7 // (op 13845) 
v_pack_b32_f16 v5, v8, v9 // (op 13845) 

s_lshl_b32 s0, s4, 8 // (op 13845) 
s_add_u32 s0, s0, 64 // (op 13845) 
v_add_u32 v0, v13, s0 // (op 13845) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13845) Store value



v_accvgpr_read v6, a52 // (op 13879) 
v_accvgpr_read v7, a53 // (op 13879) 
v_accvgpr_read v8, a54 // (op 13879) 
v_accvgpr_read v9, a55 // (op 13879) 
v_mul_f32 v14, v11, v6 // (op 13879) 
v_mul_f32 v15, v11, v7 // (op 13879) 
v_mul_f32 v16, v11, v8 // (op 13879) 
v_mul_f32 v17, v11, v9 // (op 13879) 

v_lshrrev_b32 v6, 16, v14 // (op 13881) 
v_lshrrev_b32 v7, 16, v15 // (op 13881) 
v_lshrrev_b32 v8, 16, v16 // (op 13881) 
v_lshrrev_b32 v9, 16, v17 // (op 13881) 


v_pack_b32_f16 v4, v6, v7 // (op 13847) 
v_pack_b32_f16 v5, v8, v9 // (op 13847) 

s_lshl_b32 s0, s4, 6 // (op 13847) 
s_mul_i32 s1, 5, s0 // (op 13847) 
s_add_u32 s1, s1, 64 // (op 13847) 

v_add_u32 v0, v13, s1 // (op 13847) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13847) Store value



v_accvgpr_read v6, a56 // (op 13885) 
v_accvgpr_read v7, a57 // (op 13885) 
v_accvgpr_read v8, a58 // (op 13885) 
v_accvgpr_read v9, a59 // (op 13885) 
v_mul_f32 v14, v11, v6 // (op 13885) 
v_mul_f32 v15, v11, v7 // (op 13885) 
v_mul_f32 v16, v11, v8 // (op 13885) 
v_mul_f32 v17, v11, v9 // (op 13885) 

v_lshrrev_b32 v6, 16, v14 // (op 13887) 
v_lshrrev_b32 v7, 16, v15 // (op 13887) 
v_lshrrev_b32 v8, 16, v16 // (op 13887) 
v_lshrrev_b32 v9, 16, v17 // (op 13887) 


v_pack_b32_f16 v4, v6, v7 // (op 13849) 
v_pack_b32_f16 v5, v8, v9 // (op 13849) 

s_lshl_b32 s0, s4, 6 // (op 13849) 
s_mul_i32 s1, 6, s0 // (op 13849) 
s_add_u32 s1, s1, 64 // (op 13849) 

v_add_u32 v0, v13, s1 // (op 13849) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13849) Store value



v_accvgpr_read v6, a60 // (op 13891) 
v_accvgpr_read v7, a61 // (op 13891) 
v_accvgpr_read v8, a62 // (op 13891) 
v_accvgpr_read v9, a63 // (op 13891) 
v_mul_f32 v14, v11, v6 // (op 13891) 
v_mul_f32 v15, v11, v7 // (op 13891) 
v_mul_f32 v16, v11, v8 // (op 13891) 
v_mul_f32 v17, v11, v9 // (op 13891) 

v_lshrrev_b32 v6, 16, v14 // (op 13893) 
v_lshrrev_b32 v7, 16, v15 // (op 13893) 
v_lshrrev_b32 v8, 16, v16 // (op 13893) 
v_lshrrev_b32 v9, 16, v17 // (op 13893) 


v_pack_b32_f16 v4, v6, v7 // (op 13851) 
v_pack_b32_f16 v5, v8, v9 // (op 13851) 

s_lshl_b32 s0, s4, 6 // (op 13851) 
s_mul_i32 s1, 7, s0 // (op 13851) 
s_add_u32 s1, s1, 64 // (op 13851) 

v_add_u32 v0, v13, s1 // (op 13851) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13851) Store value



v_accvgpr_read v6, a64 // (op 13981) 
v_accvgpr_read v7, a65 // (op 13981) 
v_accvgpr_read v8, a66 // (op 13981) 
v_accvgpr_read v9, a67 // (op 13981) 
v_mul_f32 v14, v11, v6 // (op 13981) 
v_mul_f32 v15, v11, v7 // (op 13981) 
v_mul_f32 v16, v11, v8 // (op 13981) 
v_mul_f32 v17, v11, v9 // (op 13981) 

v_lshrrev_b32 v6, 16, v14 // (op 13983) 
v_lshrrev_b32 v7, 16, v15 // (op 13983) 
v_lshrrev_b32 v8, 16, v16 // (op 13983) 
v_lshrrev_b32 v9, 16, v17 // (op 13983) 


v_pack_b32_f16 v4, v6, v7 // (op 13985) 
v_pack_b32_f16 v5, v8, v9 // (op 13985) 

v_add_u32 v0, 128, v13 // (op 13985) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13985) Store value



v_accvgpr_read v6, a68 // (op 14003) 
v_accvgpr_read v7, a69 // (op 14003) 
v_accvgpr_read v8, a70 // (op 14003) 
v_accvgpr_read v9, a71 // (op 14003) 
v_mul_f32 v14, v11, v6 // (op 14003) 
v_mul_f32 v15, v11, v7 // (op 14003) 
v_mul_f32 v16, v11, v8 // (op 14003) 
v_mul_f32 v17, v11, v9 // (op 14003) 

v_lshrrev_b32 v6, 16, v14 // (op 14005) 
v_lshrrev_b32 v7, 16, v15 // (op 14005) 
v_lshrrev_b32 v8, 16, v16 // (op 14005) 
v_lshrrev_b32 v9, 16, v17 // (op 14005) 


v_pack_b32_f16 v4, v6, v7 // (op 13987) 
v_pack_b32_f16 v5, v8, v9 // (op 13987) 

s_lshl_b32 s0, s4, 6 // (op 13987) 
s_add_u32 s0, s0, 128 // (op 13987) 
v_add_u32 v0, v13, s0 // (op 13987) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13987) Store value



v_accvgpr_read v6, a72 // (op 14009) 
v_accvgpr_read v7, a73 // (op 14009) 
v_accvgpr_read v8, a74 // (op 14009) 
v_accvgpr_read v9, a75 // (op 14009) 
v_mul_f32 v14, v11, v6 // (op 14009) 
v_mul_f32 v15, v11, v7 // (op 14009) 
v_mul_f32 v16, v11, v8 // (op 14009) 
v_mul_f32 v17, v11, v9 // (op 14009) 

v_lshrrev_b32 v6, 16, v14 // (op 14011) 
v_lshrrev_b32 v7, 16, v15 // (op 14011) 
v_lshrrev_b32 v8, 16, v16 // (op 14011) 
v_lshrrev_b32 v9, 16, v17 // (op 14011) 


v_pack_b32_f16 v4, v6, v7 // (op 13989) 
v_pack_b32_f16 v5, v8, v9 // (op 13989) 

s_lshl_b32 s0, s4, 7 // (op 13989) 
s_add_u32 s0, s0, 128 // (op 13989) 
v_add_u32 v0, v13, s0 // (op 13989) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13989) Store value



v_accvgpr_read v6, a76 // (op 14015) 
v_accvgpr_read v7, a77 // (op 14015) 
v_accvgpr_read v8, a78 // (op 14015) 
v_accvgpr_read v9, a79 // (op 14015) 
v_mul_f32 v14, v11, v6 // (op 14015) 
v_mul_f32 v15, v11, v7 // (op 14015) 
v_mul_f32 v16, v11, v8 // (op 14015) 
v_mul_f32 v17, v11, v9 // (op 14015) 

v_lshrrev_b32 v6, 16, v14 // (op 14017) 
v_lshrrev_b32 v7, 16, v15 // (op 14017) 
v_lshrrev_b32 v8, 16, v16 // (op 14017) 
v_lshrrev_b32 v9, 16, v17 // (op 14017) 


v_pack_b32_f16 v4, v6, v7 // (op 13991) 
v_pack_b32_f16 v5, v8, v9 // (op 13991) 

s_lshl_b32 s0, s4, 6 // (op 13991) 
s_mul_i32 s1, 3, s0 // (op 13991) 
s_add_u32 s1, s1, 128 // (op 13991) 

v_add_u32 v0, v13, s1 // (op 13991) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13991) Store value



v_accvgpr_read v6, a80 // (op 14021) 
v_accvgpr_read v7, a81 // (op 14021) 
v_accvgpr_read v8, a82 // (op 14021) 
v_accvgpr_read v9, a83 // (op 14021) 
v_mul_f32 v14, v11, v6 // (op 14021) 
v_mul_f32 v15, v11, v7 // (op 14021) 
v_mul_f32 v16, v11, v8 // (op 14021) 
v_mul_f32 v17, v11, v9 // (op 14021) 

v_lshrrev_b32 v6, 16, v14 // (op 14023) 
v_lshrrev_b32 v7, 16, v15 // (op 14023) 
v_lshrrev_b32 v8, 16, v16 // (op 14023) 
v_lshrrev_b32 v9, 16, v17 // (op 14023) 


v_pack_b32_f16 v4, v6, v7 // (op 13993) 
v_pack_b32_f16 v5, v8, v9 // (op 13993) 

s_lshl_b32 s0, s4, 8 // (op 13993) 
s_add_u32 s0, s0, 128 // (op 13993) 
v_add_u32 v0, v13, s0 // (op 13993) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13993) Store value



v_accvgpr_read v6, a84 // (op 14027) 
v_accvgpr_read v7, a85 // (op 14027) 
v_accvgpr_read v8, a86 // (op 14027) 
v_accvgpr_read v9, a87 // (op 14027) 
v_mul_f32 v14, v11, v6 // (op 14027) 
v_mul_f32 v15, v11, v7 // (op 14027) 
v_mul_f32 v16, v11, v8 // (op 14027) 
v_mul_f32 v17, v11, v9 // (op 14027) 

v_lshrrev_b32 v6, 16, v14 // (op 14029) 
v_lshrrev_b32 v7, 16, v15 // (op 14029) 
v_lshrrev_b32 v8, 16, v16 // (op 14029) 
v_lshrrev_b32 v9, 16, v17 // (op 14029) 


v_pack_b32_f16 v4, v6, v7 // (op 13995) 
v_pack_b32_f16 v5, v8, v9 // (op 13995) 

s_lshl_b32 s0, s4, 6 // (op 13995) 
s_mul_i32 s1, 5, s0 // (op 13995) 
s_add_u32 s1, s1, 128 // (op 13995) 

v_add_u32 v0, v13, s1 // (op 13995) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13995) Store value



v_accvgpr_read v6, a88 // (op 14033) 
v_accvgpr_read v7, a89 // (op 14033) 
v_accvgpr_read v8, a90 // (op 14033) 
v_accvgpr_read v9, a91 // (op 14033) 
v_mul_f32 v14, v11, v6 // (op 14033) 
v_mul_f32 v15, v11, v7 // (op 14033) 
v_mul_f32 v16, v11, v8 // (op 14033) 
v_mul_f32 v17, v11, v9 // (op 14033) 

v_lshrrev_b32 v6, 16, v14 // (op 14035) 
v_lshrrev_b32 v7, 16, v15 // (op 14035) 
v_lshrrev_b32 v8, 16, v16 // (op 14035) 
v_lshrrev_b32 v9, 16, v17 // (op 14035) 


v_pack_b32_f16 v4, v6, v7 // (op 13997) 
v_pack_b32_f16 v5, v8, v9 // (op 13997) 

s_lshl_b32 s0, s4, 6 // (op 13997) 
s_mul_i32 s1, 6, s0 // (op 13997) 
s_add_u32 s1, s1, 128 // (op 13997) 

v_add_u32 v0, v13, s1 // (op 13997) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13997) Store value



v_accvgpr_read v6, a92 // (op 14039) 
v_accvgpr_read v7, a93 // (op 14039) 
v_accvgpr_read v8, a94 // (op 14039) 
v_accvgpr_read v9, a95 // (op 14039) 
v_mul_f32 v14, v11, v6 // (op 14039) 
v_mul_f32 v15, v11, v7 // (op 14039) 
v_mul_f32 v16, v11, v8 // (op 14039) 
v_mul_f32 v17, v11, v9 // (op 14039) 

v_lshrrev_b32 v6, 16, v14 // (op 14041) 
v_lshrrev_b32 v7, 16, v15 // (op 14041) 
v_lshrrev_b32 v8, 16, v16 // (op 14041) 
v_lshrrev_b32 v9, 16, v17 // (op 14041) 


v_pack_b32_f16 v4, v6, v7 // (op 13999) 
v_pack_b32_f16 v5, v8, v9 // (op 13999) 

s_lshl_b32 s0, s4, 6 // (op 13999) 
s_mul_i32 s1, 7, s0 // (op 13999) 
s_add_u32 s1, s1, 128 // (op 13999) 

v_add_u32 v0, v13, s1 // (op 13999) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 13999) Store value



v_accvgpr_read v6, a96 // (op 14129) 
v_accvgpr_read v7, a97 // (op 14129) 
v_accvgpr_read v8, a98 // (op 14129) 
v_accvgpr_read v9, a99 // (op 14129) 
v_mul_f32 v14, v11, v6 // (op 14129) 
v_mul_f32 v15, v11, v7 // (op 14129) 
v_mul_f32 v16, v11, v8 // (op 14129) 
v_mul_f32 v17, v11, v9 // (op 14129) 

v_lshrrev_b32 v6, 16, v14 // (op 14131) 
v_lshrrev_b32 v7, 16, v15 // (op 14131) 
v_lshrrev_b32 v8, 16, v16 // (op 14131) 
v_lshrrev_b32 v9, 16, v17 // (op 14131) 


v_pack_b32_f16 v4, v6, v7 // (op 14133) 
v_pack_b32_f16 v5, v8, v9 // (op 14133) 

v_add_u32 v0, 192, v13 // (op 14133) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14133) Store value



v_accvgpr_read v6, a100 // (op 14151) 
v_accvgpr_read v7, a101 // (op 14151) 
v_accvgpr_read v8, a102 // (op 14151) 
v_accvgpr_read v9, a103 // (op 14151) 
v_mul_f32 v14, v11, v6 // (op 14151) 
v_mul_f32 v15, v11, v7 // (op 14151) 
v_mul_f32 v16, v11, v8 // (op 14151) 
v_mul_f32 v17, v11, v9 // (op 14151) 

v_lshrrev_b32 v6, 16, v14 // (op 14153) 
v_lshrrev_b32 v7, 16, v15 // (op 14153) 
v_lshrrev_b32 v8, 16, v16 // (op 14153) 
v_lshrrev_b32 v9, 16, v17 // (op 14153) 


v_pack_b32_f16 v4, v6, v7 // (op 14135) 
v_pack_b32_f16 v5, v8, v9 // (op 14135) 

s_lshl_b32 s0, s4, 6 // (op 14135) 
s_add_u32 s0, s0, 192 // (op 14135) 
v_add_u32 v0, v13, s0 // (op 14135) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14135) Store value



v_accvgpr_read v6, a104 // (op 14157) 
v_accvgpr_read v7, a105 // (op 14157) 
v_accvgpr_read v8, a106 // (op 14157) 
v_accvgpr_read v9, a107 // (op 14157) 
v_mul_f32 v14, v11, v6 // (op 14157) 
v_mul_f32 v15, v11, v7 // (op 14157) 
v_mul_f32 v16, v11, v8 // (op 14157) 
v_mul_f32 v17, v11, v9 // (op 14157) 

v_lshrrev_b32 v6, 16, v14 // (op 14159) 
v_lshrrev_b32 v7, 16, v15 // (op 14159) 
v_lshrrev_b32 v8, 16, v16 // (op 14159) 
v_lshrrev_b32 v9, 16, v17 // (op 14159) 


v_pack_b32_f16 v4, v6, v7 // (op 14137) 
v_pack_b32_f16 v5, v8, v9 // (op 14137) 

s_lshl_b32 s0, s4, 7 // (op 14137) 
s_add_u32 s0, s0, 192 // (op 14137) 
v_add_u32 v0, v13, s0 // (op 14137) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14137) Store value



v_accvgpr_read v6, a108 // (op 14163) 
v_accvgpr_read v7, a109 // (op 14163) 
v_accvgpr_read v8, a110 // (op 14163) 
v_accvgpr_read v9, a111 // (op 14163) 
v_mul_f32 v14, v11, v6 // (op 14163) 
v_mul_f32 v15, v11, v7 // (op 14163) 
v_mul_f32 v16, v11, v8 // (op 14163) 
v_mul_f32 v17, v11, v9 // (op 14163) 

v_lshrrev_b32 v6, 16, v14 // (op 14165) 
v_lshrrev_b32 v7, 16, v15 // (op 14165) 
v_lshrrev_b32 v8, 16, v16 // (op 14165) 
v_lshrrev_b32 v9, 16, v17 // (op 14165) 


v_pack_b32_f16 v4, v6, v7 // (op 14139) 
v_pack_b32_f16 v5, v8, v9 // (op 14139) 

s_lshl_b32 s0, s4, 6 // (op 14139) 
s_mul_i32 s1, 3, s0 // (op 14139) 
s_add_u32 s1, s1, 192 // (op 14139) 

v_add_u32 v0, v13, s1 // (op 14139) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14139) Store value



v_accvgpr_read v6, a112 // (op 14169) 
v_accvgpr_read v7, a113 // (op 14169) 
v_accvgpr_read v8, a114 // (op 14169) 
v_accvgpr_read v9, a115 // (op 14169) 
v_mul_f32 v14, v11, v6 // (op 14169) 
v_mul_f32 v15, v11, v7 // (op 14169) 
v_mul_f32 v16, v11, v8 // (op 14169) 
v_mul_f32 v17, v11, v9 // (op 14169) 

v_lshrrev_b32 v6, 16, v14 // (op 14171) 
v_lshrrev_b32 v7, 16, v15 // (op 14171) 
v_lshrrev_b32 v8, 16, v16 // (op 14171) 
v_lshrrev_b32 v9, 16, v17 // (op 14171) 


v_pack_b32_f16 v4, v6, v7 // (op 14141) 
v_pack_b32_f16 v5, v8, v9 // (op 14141) 

s_lshl_b32 s0, s4, 8 // (op 14141) 
s_add_u32 s0, s0, 192 // (op 14141) 
v_add_u32 v0, v13, s0 // (op 14141) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14141) Store value



v_accvgpr_read v6, a116 // (op 14175) 
v_accvgpr_read v7, a117 // (op 14175) 
v_accvgpr_read v8, a118 // (op 14175) 
v_accvgpr_read v9, a119 // (op 14175) 
v_mul_f32 v14, v11, v6 // (op 14175) 
v_mul_f32 v15, v11, v7 // (op 14175) 
v_mul_f32 v16, v11, v8 // (op 14175) 
v_mul_f32 v17, v11, v9 // (op 14175) 

v_lshrrev_b32 v6, 16, v14 // (op 14177) 
v_lshrrev_b32 v7, 16, v15 // (op 14177) 
v_lshrrev_b32 v8, 16, v16 // (op 14177) 
v_lshrrev_b32 v9, 16, v17 // (op 14177) 


v_pack_b32_f16 v4, v6, v7 // (op 14143) 
v_pack_b32_f16 v5, v8, v9 // (op 14143) 

s_lshl_b32 s0, s4, 6 // (op 14143) 
s_mul_i32 s1, 5, s0 // (op 14143) 
s_add_u32 s1, s1, 192 // (op 14143) 

v_add_u32 v0, v13, s1 // (op 14143) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14143) Store value



v_accvgpr_read v6, a120 // (op 14181) 
v_accvgpr_read v7, a121 // (op 14181) 
v_accvgpr_read v8, a122 // (op 14181) 
v_accvgpr_read v9, a123 // (op 14181) 
v_mul_f32 v14, v11, v6 // (op 14181) 
v_mul_f32 v15, v11, v7 // (op 14181) 
v_mul_f32 v16, v11, v8 // (op 14181) 
v_mul_f32 v17, v11, v9 // (op 14181) 

v_lshrrev_b32 v6, 16, v14 // (op 14183) 
v_lshrrev_b32 v7, 16, v15 // (op 14183) 
v_lshrrev_b32 v8, 16, v16 // (op 14183) 
v_lshrrev_b32 v9, 16, v17 // (op 14183) 


v_pack_b32_f16 v4, v6, v7 // (op 14145) 
v_pack_b32_f16 v5, v8, v9 // (op 14145) 

s_lshl_b32 s0, s4, 6 // (op 14145) 
s_mul_i32 s1, 6, s0 // (op 14145) 
s_add_u32 s1, s1, 192 // (op 14145) 

v_add_u32 v0, v13, s1 // (op 14145) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14145) Store value



v_accvgpr_read v6, a124 // (op 14187) 
v_accvgpr_read v7, a125 // (op 14187) 
v_accvgpr_read v8, a126 // (op 14187) 
v_accvgpr_read v9, a127 // (op 14187) 
v_mul_f32 v14, v11, v6 // (op 14187) 
v_mul_f32 v15, v11, v7 // (op 14187) 
v_mul_f32 v16, v11, v8 // (op 14187) 
v_mul_f32 v17, v11, v9 // (op 14187) 

v_lshrrev_b32 v6, 16, v14 // (op 14189) 
v_lshrrev_b32 v7, 16, v15 // (op 14189) 
v_lshrrev_b32 v8, 16, v16 // (op 14189) 
v_lshrrev_b32 v9, 16, v17 // (op 14189) 


v_pack_b32_f16 v4, v6, v7 // (op 14147) 
v_pack_b32_f16 v5, v8, v9 // (op 14147) 

s_lshl_b32 s0, s4, 6 // (op 14147) 
s_mul_i32 s1, 7, s0 // (op 14147) 
s_add_u32 s1, s1, 192 // (op 14147) 

v_add_u32 v0, v13, s1 // (op 14147) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14147) Store value



v_accvgpr_read v6, a128 // (op 14277) 
v_accvgpr_read v7, a129 // (op 14277) 
v_accvgpr_read v8, a130 // (op 14277) 
v_accvgpr_read v9, a131 // (op 14277) 
v_mul_f32 v14, v11, v6 // (op 14277) 
v_mul_f32 v15, v11, v7 // (op 14277) 
v_mul_f32 v16, v11, v8 // (op 14277) 
v_mul_f32 v17, v11, v9 // (op 14277) 

v_lshrrev_b32 v6, 16, v14 // (op 14279) 
v_lshrrev_b32 v7, 16, v15 // (op 14279) 
v_lshrrev_b32 v8, 16, v16 // (op 14279) 
v_lshrrev_b32 v9, 16, v17 // (op 14279) 


v_pack_b32_f16 v4, v6, v7 // (op 14281) 
v_pack_b32_f16 v5, v8, v9 // (op 14281) 

v_add_u32 v0, 256, v13 // (op 14281) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14281) Store value



v_accvgpr_read v6, a132 // (op 14299) 
v_accvgpr_read v7, a133 // (op 14299) 
v_accvgpr_read v8, a134 // (op 14299) 
v_accvgpr_read v9, a135 // (op 14299) 
v_mul_f32 v14, v11, v6 // (op 14299) 
v_mul_f32 v15, v11, v7 // (op 14299) 
v_mul_f32 v16, v11, v8 // (op 14299) 
v_mul_f32 v17, v11, v9 // (op 14299) 

v_lshrrev_b32 v6, 16, v14 // (op 14301) 
v_lshrrev_b32 v7, 16, v15 // (op 14301) 
v_lshrrev_b32 v8, 16, v16 // (op 14301) 
v_lshrrev_b32 v9, 16, v17 // (op 14301) 


v_pack_b32_f16 v4, v6, v7 // (op 14283) 
v_pack_b32_f16 v5, v8, v9 // (op 14283) 

s_lshl_b32 s0, s4, 6 // (op 14283) 
s_add_u32 s0, s0, 256 // (op 14283) 
v_add_u32 v0, v13, s0 // (op 14283) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14283) Store value



v_accvgpr_read v6, a136 // (op 14305) 
v_accvgpr_read v7, a137 // (op 14305) 
v_accvgpr_read v8, a138 // (op 14305) 
v_accvgpr_read v9, a139 // (op 14305) 
v_mul_f32 v14, v11, v6 // (op 14305) 
v_mul_f32 v15, v11, v7 // (op 14305) 
v_mul_f32 v16, v11, v8 // (op 14305) 
v_mul_f32 v17, v11, v9 // (op 14305) 

v_lshrrev_b32 v6, 16, v14 // (op 14307) 
v_lshrrev_b32 v7, 16, v15 // (op 14307) 
v_lshrrev_b32 v8, 16, v16 // (op 14307) 
v_lshrrev_b32 v9, 16, v17 // (op 14307) 


v_pack_b32_f16 v4, v6, v7 // (op 14285) 
v_pack_b32_f16 v5, v8, v9 // (op 14285) 

s_lshl_b32 s0, s4, 7 // (op 14285) 
s_add_u32 s0, s0, 256 // (op 14285) 
v_add_u32 v0, v13, s0 // (op 14285) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14285) Store value



v_accvgpr_read v6, a140 // (op 14311) 
v_accvgpr_read v7, a141 // (op 14311) 
v_accvgpr_read v8, a142 // (op 14311) 
v_accvgpr_read v9, a143 // (op 14311) 
v_mul_f32 v14, v11, v6 // (op 14311) 
v_mul_f32 v15, v11, v7 // (op 14311) 
v_mul_f32 v16, v11, v8 // (op 14311) 
v_mul_f32 v17, v11, v9 // (op 14311) 

v_lshrrev_b32 v6, 16, v14 // (op 14313) 
v_lshrrev_b32 v7, 16, v15 // (op 14313) 
v_lshrrev_b32 v8, 16, v16 // (op 14313) 
v_lshrrev_b32 v9, 16, v17 // (op 14313) 


v_pack_b32_f16 v4, v6, v7 // (op 14287) 
v_pack_b32_f16 v5, v8, v9 // (op 14287) 

s_lshl_b32 s0, s4, 6 // (op 14287) 
s_mul_i32 s1, 3, s0 // (op 14287) 
s_add_u32 s1, s1, 256 // (op 14287) 

v_add_u32 v0, v13, s1 // (op 14287) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14287) Store value



v_accvgpr_read v6, a144 // (op 14317) 
v_accvgpr_read v7, a145 // (op 14317) 
v_accvgpr_read v8, a146 // (op 14317) 
v_accvgpr_read v9, a147 // (op 14317) 
v_mul_f32 v14, v11, v6 // (op 14317) 
v_mul_f32 v15, v11, v7 // (op 14317) 
v_mul_f32 v16, v11, v8 // (op 14317) 
v_mul_f32 v17, v11, v9 // (op 14317) 

v_lshrrev_b32 v6, 16, v14 // (op 14319) 
v_lshrrev_b32 v7, 16, v15 // (op 14319) 
v_lshrrev_b32 v8, 16, v16 // (op 14319) 
v_lshrrev_b32 v9, 16, v17 // (op 14319) 


v_pack_b32_f16 v4, v6, v7 // (op 14289) 
v_pack_b32_f16 v5, v8, v9 // (op 14289) 

s_lshl_b32 s0, s4, 8 // (op 14289) 
s_add_u32 s0, s0, 256 // (op 14289) 
v_add_u32 v0, v13, s0 // (op 14289) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14289) Store value



v_accvgpr_read v6, a148 // (op 14323) 
v_accvgpr_read v7, a149 // (op 14323) 
v_accvgpr_read v8, a150 // (op 14323) 
v_accvgpr_read v9, a151 // (op 14323) 
v_mul_f32 v14, v11, v6 // (op 14323) 
v_mul_f32 v15, v11, v7 // (op 14323) 
v_mul_f32 v16, v11, v8 // (op 14323) 
v_mul_f32 v17, v11, v9 // (op 14323) 

v_lshrrev_b32 v6, 16, v14 // (op 14325) 
v_lshrrev_b32 v7, 16, v15 // (op 14325) 
v_lshrrev_b32 v8, 16, v16 // (op 14325) 
v_lshrrev_b32 v9, 16, v17 // (op 14325) 


v_pack_b32_f16 v4, v6, v7 // (op 14291) 
v_pack_b32_f16 v5, v8, v9 // (op 14291) 

s_lshl_b32 s0, s4, 6 // (op 14291) 
s_mul_i32 s1, 5, s0 // (op 14291) 
s_add_u32 s1, s1, 256 // (op 14291) 

v_add_u32 v0, v13, s1 // (op 14291) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14291) Store value



v_accvgpr_read v6, a152 // (op 14329) 
v_accvgpr_read v7, a153 // (op 14329) 
v_accvgpr_read v8, a154 // (op 14329) 
v_accvgpr_read v9, a155 // (op 14329) 
v_mul_f32 v14, v11, v6 // (op 14329) 
v_mul_f32 v15, v11, v7 // (op 14329) 
v_mul_f32 v16, v11, v8 // (op 14329) 
v_mul_f32 v17, v11, v9 // (op 14329) 

v_lshrrev_b32 v6, 16, v14 // (op 14331) 
v_lshrrev_b32 v7, 16, v15 // (op 14331) 
v_lshrrev_b32 v8, 16, v16 // (op 14331) 
v_lshrrev_b32 v9, 16, v17 // (op 14331) 


v_pack_b32_f16 v4, v6, v7 // (op 14293) 
v_pack_b32_f16 v5, v8, v9 // (op 14293) 

s_lshl_b32 s0, s4, 6 // (op 14293) 
s_mul_i32 s1, 6, s0 // (op 14293) 
s_add_u32 s1, s1, 256 // (op 14293) 

v_add_u32 v0, v13, s1 // (op 14293) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14293) Store value



v_accvgpr_read v6, a156 // (op 14335) 
v_accvgpr_read v7, a157 // (op 14335) 
v_accvgpr_read v8, a158 // (op 14335) 
v_accvgpr_read v9, a159 // (op 14335) 
v_mul_f32 v14, v11, v6 // (op 14335) 
v_mul_f32 v15, v11, v7 // (op 14335) 
v_mul_f32 v16, v11, v8 // (op 14335) 
v_mul_f32 v17, v11, v9 // (op 14335) 

v_lshrrev_b32 v6, 16, v14 // (op 14337) 
v_lshrrev_b32 v7, 16, v15 // (op 14337) 
v_lshrrev_b32 v8, 16, v16 // (op 14337) 
v_lshrrev_b32 v9, 16, v17 // (op 14337) 


v_pack_b32_f16 v4, v6, v7 // (op 14295) 
v_pack_b32_f16 v5, v8, v9 // (op 14295) 

s_lshl_b32 s0, s4, 6 // (op 14295) 
s_mul_i32 s1, 7, s0 // (op 14295) 
s_add_u32 s1, s1, 256 // (op 14295) 

v_add_u32 v0, v13, s1 // (op 14295) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14295) Store value



v_accvgpr_read v6, a160 // (op 14425) 
v_accvgpr_read v7, a161 // (op 14425) 
v_accvgpr_read v8, a162 // (op 14425) 
v_accvgpr_read v9, a163 // (op 14425) 
v_mul_f32 v14, v11, v6 // (op 14425) 
v_mul_f32 v15, v11, v7 // (op 14425) 
v_mul_f32 v16, v11, v8 // (op 14425) 
v_mul_f32 v17, v11, v9 // (op 14425) 

v_lshrrev_b32 v6, 16, v14 // (op 14427) 
v_lshrrev_b32 v7, 16, v15 // (op 14427) 
v_lshrrev_b32 v8, 16, v16 // (op 14427) 
v_lshrrev_b32 v9, 16, v17 // (op 14427) 


v_pack_b32_f16 v4, v6, v7 // (op 14429) 
v_pack_b32_f16 v5, v8, v9 // (op 14429) 

v_add_u32 v0, 320, v13 // (op 14429) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14429) Store value



v_accvgpr_read v6, a164 // (op 14447) 
v_accvgpr_read v7, a165 // (op 14447) 
v_accvgpr_read v8, a166 // (op 14447) 
v_accvgpr_read v9, a167 // (op 14447) 
v_mul_f32 v14, v11, v6 // (op 14447) 
v_mul_f32 v15, v11, v7 // (op 14447) 
v_mul_f32 v16, v11, v8 // (op 14447) 
v_mul_f32 v17, v11, v9 // (op 14447) 

v_lshrrev_b32 v6, 16, v14 // (op 14449) 
v_lshrrev_b32 v7, 16, v15 // (op 14449) 
v_lshrrev_b32 v8, 16, v16 // (op 14449) 
v_lshrrev_b32 v9, 16, v17 // (op 14449) 


v_pack_b32_f16 v4, v6, v7 // (op 14431) 
v_pack_b32_f16 v5, v8, v9 // (op 14431) 

s_lshl_b32 s0, s4, 6 // (op 14431) 
s_add_u32 s0, s0, 320 // (op 14431) 
v_add_u32 v0, v13, s0 // (op 14431) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14431) Store value



v_accvgpr_read v6, a168 // (op 14453) 
v_accvgpr_read v7, a169 // (op 14453) 
v_accvgpr_read v8, a170 // (op 14453) 
v_accvgpr_read v9, a171 // (op 14453) 
v_mul_f32 v14, v11, v6 // (op 14453) 
v_mul_f32 v15, v11, v7 // (op 14453) 
v_mul_f32 v16, v11, v8 // (op 14453) 
v_mul_f32 v17, v11, v9 // (op 14453) 

v_lshrrev_b32 v6, 16, v14 // (op 14455) 
v_lshrrev_b32 v7, 16, v15 // (op 14455) 
v_lshrrev_b32 v8, 16, v16 // (op 14455) 
v_lshrrev_b32 v9, 16, v17 // (op 14455) 


v_pack_b32_f16 v4, v6, v7 // (op 14433) 
v_pack_b32_f16 v5, v8, v9 // (op 14433) 

s_lshl_b32 s0, s4, 7 // (op 14433) 
s_add_u32 s0, s0, 320 // (op 14433) 
v_add_u32 v0, v13, s0 // (op 14433) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14433) Store value



v_accvgpr_read v6, a172 // (op 14459) 
v_accvgpr_read v7, a173 // (op 14459) 
v_accvgpr_read v8, a174 // (op 14459) 
v_accvgpr_read v9, a175 // (op 14459) 
v_mul_f32 v14, v11, v6 // (op 14459) 
v_mul_f32 v15, v11, v7 // (op 14459) 
v_mul_f32 v16, v11, v8 // (op 14459) 
v_mul_f32 v17, v11, v9 // (op 14459) 

v_lshrrev_b32 v6, 16, v14 // (op 14461) 
v_lshrrev_b32 v7, 16, v15 // (op 14461) 
v_lshrrev_b32 v8, 16, v16 // (op 14461) 
v_lshrrev_b32 v9, 16, v17 // (op 14461) 


v_pack_b32_f16 v4, v6, v7 // (op 14435) 
v_pack_b32_f16 v5, v8, v9 // (op 14435) 

s_lshl_b32 s0, s4, 6 // (op 14435) 
s_mul_i32 s1, 3, s0 // (op 14435) 
s_add_u32 s1, s1, 320 // (op 14435) 

v_add_u32 v0, v13, s1 // (op 14435) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14435) Store value



v_accvgpr_read v6, a176 // (op 14465) 
v_accvgpr_read v7, a177 // (op 14465) 
v_accvgpr_read v8, a178 // (op 14465) 
v_accvgpr_read v9, a179 // (op 14465) 
v_mul_f32 v14, v11, v6 // (op 14465) 
v_mul_f32 v15, v11, v7 // (op 14465) 
v_mul_f32 v16, v11, v8 // (op 14465) 
v_mul_f32 v17, v11, v9 // (op 14465) 

v_lshrrev_b32 v6, 16, v14 // (op 14467) 
v_lshrrev_b32 v7, 16, v15 // (op 14467) 
v_lshrrev_b32 v8, 16, v16 // (op 14467) 
v_lshrrev_b32 v9, 16, v17 // (op 14467) 


v_pack_b32_f16 v4, v6, v7 // (op 14437) 
v_pack_b32_f16 v5, v8, v9 // (op 14437) 

s_lshl_b32 s0, s4, 8 // (op 14437) 
s_add_u32 s0, s0, 320 // (op 14437) 
v_add_u32 v0, v13, s0 // (op 14437) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14437) Store value



v_accvgpr_read v6, a180 // (op 14471) 
v_accvgpr_read v7, a181 // (op 14471) 
v_accvgpr_read v8, a182 // (op 14471) 
v_accvgpr_read v9, a183 // (op 14471) 
v_mul_f32 v14, v11, v6 // (op 14471) 
v_mul_f32 v15, v11, v7 // (op 14471) 
v_mul_f32 v16, v11, v8 // (op 14471) 
v_mul_f32 v17, v11, v9 // (op 14471) 

v_lshrrev_b32 v6, 16, v14 // (op 14473) 
v_lshrrev_b32 v7, 16, v15 // (op 14473) 
v_lshrrev_b32 v8, 16, v16 // (op 14473) 
v_lshrrev_b32 v9, 16, v17 // (op 14473) 


v_pack_b32_f16 v4, v6, v7 // (op 14439) 
v_pack_b32_f16 v5, v8, v9 // (op 14439) 

s_lshl_b32 s0, s4, 6 // (op 14439) 
s_mul_i32 s1, 5, s0 // (op 14439) 
s_add_u32 s1, s1, 320 // (op 14439) 

v_add_u32 v0, v13, s1 // (op 14439) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14439) Store value



v_accvgpr_read v6, a184 // (op 14477) 
v_accvgpr_read v7, a185 // (op 14477) 
v_accvgpr_read v8, a186 // (op 14477) 
v_accvgpr_read v9, a187 // (op 14477) 
v_mul_f32 v14, v11, v6 // (op 14477) 
v_mul_f32 v15, v11, v7 // (op 14477) 
v_mul_f32 v16, v11, v8 // (op 14477) 
v_mul_f32 v17, v11, v9 // (op 14477) 

v_lshrrev_b32 v6, 16, v14 // (op 14479) 
v_lshrrev_b32 v7, 16, v15 // (op 14479) 
v_lshrrev_b32 v8, 16, v16 // (op 14479) 
v_lshrrev_b32 v9, 16, v17 // (op 14479) 


v_pack_b32_f16 v4, v6, v7 // (op 14441) 
v_pack_b32_f16 v5, v8, v9 // (op 14441) 

s_lshl_b32 s0, s4, 6 // (op 14441) 
s_mul_i32 s1, 6, s0 // (op 14441) 
s_add_u32 s1, s1, 320 // (op 14441) 

v_add_u32 v0, v13, s1 // (op 14441) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14441) Store value



v_accvgpr_read v6, a188 // (op 14483) 
v_accvgpr_read v7, a189 // (op 14483) 
v_accvgpr_read v8, a190 // (op 14483) 
v_accvgpr_read v9, a191 // (op 14483) 
v_mul_f32 v14, v11, v6 // (op 14483) 
v_mul_f32 v15, v11, v7 // (op 14483) 
v_mul_f32 v16, v11, v8 // (op 14483) 
v_mul_f32 v17, v11, v9 // (op 14483) 

v_lshrrev_b32 v6, 16, v14 // (op 14485) 
v_lshrrev_b32 v7, 16, v15 // (op 14485) 
v_lshrrev_b32 v8, 16, v16 // (op 14485) 
v_lshrrev_b32 v9, 16, v17 // (op 14485) 


v_pack_b32_f16 v4, v6, v7 // (op 14443) 
v_pack_b32_f16 v5, v8, v9 // (op 14443) 

s_lshl_b32 s0, s4, 6 // (op 14443) 
s_mul_i32 s1, 7, s0 // (op 14443) 
s_add_u32 s1, s1, 320 // (op 14443) 

v_add_u32 v0, v13, s1 // (op 14443) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14443) Store value



v_accvgpr_read v6, a192 // (op 14573) 
v_accvgpr_read v7, a193 // (op 14573) 
v_accvgpr_read v8, a194 // (op 14573) 
v_accvgpr_read v9, a195 // (op 14573) 
v_mul_f32 v14, v11, v6 // (op 14573) 
v_mul_f32 v15, v11, v7 // (op 14573) 
v_mul_f32 v16, v11, v8 // (op 14573) 
v_mul_f32 v17, v11, v9 // (op 14573) 

v_lshrrev_b32 v6, 16, v14 // (op 14575) 
v_lshrrev_b32 v7, 16, v15 // (op 14575) 
v_lshrrev_b32 v8, 16, v16 // (op 14575) 
v_lshrrev_b32 v9, 16, v17 // (op 14575) 


v_pack_b32_f16 v4, v6, v7 // (op 14577) 
v_pack_b32_f16 v5, v8, v9 // (op 14577) 

v_add_u32 v0, 384, v13 // (op 14577) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14577) Store value



v_accvgpr_read v6, a196 // (op 14595) 
v_accvgpr_read v7, a197 // (op 14595) 
v_accvgpr_read v8, a198 // (op 14595) 
v_accvgpr_read v9, a199 // (op 14595) 
v_mul_f32 v14, v11, v6 // (op 14595) 
v_mul_f32 v15, v11, v7 // (op 14595) 
v_mul_f32 v16, v11, v8 // (op 14595) 
v_mul_f32 v17, v11, v9 // (op 14595) 

v_lshrrev_b32 v6, 16, v14 // (op 14597) 
v_lshrrev_b32 v7, 16, v15 // (op 14597) 
v_lshrrev_b32 v8, 16, v16 // (op 14597) 
v_lshrrev_b32 v9, 16, v17 // (op 14597) 


v_pack_b32_f16 v4, v6, v7 // (op 14579) 
v_pack_b32_f16 v5, v8, v9 // (op 14579) 

s_lshl_b32 s0, s4, 6 // (op 14579) 
s_add_u32 s0, s0, 384 // (op 14579) 
v_add_u32 v0, v13, s0 // (op 14579) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14579) Store value



v_accvgpr_read v6, a200 // (op 14601) 
v_accvgpr_read v7, a201 // (op 14601) 
v_accvgpr_read v8, a202 // (op 14601) 
v_accvgpr_read v9, a203 // (op 14601) 
v_mul_f32 v14, v11, v6 // (op 14601) 
v_mul_f32 v15, v11, v7 // (op 14601) 
v_mul_f32 v16, v11, v8 // (op 14601) 
v_mul_f32 v17, v11, v9 // (op 14601) 

v_lshrrev_b32 v6, 16, v14 // (op 14603) 
v_lshrrev_b32 v7, 16, v15 // (op 14603) 
v_lshrrev_b32 v8, 16, v16 // (op 14603) 
v_lshrrev_b32 v9, 16, v17 // (op 14603) 


v_pack_b32_f16 v4, v6, v7 // (op 14581) 
v_pack_b32_f16 v5, v8, v9 // (op 14581) 

s_lshl_b32 s0, s4, 7 // (op 14581) 
s_add_u32 s0, s0, 384 // (op 14581) 
v_add_u32 v0, v13, s0 // (op 14581) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14581) Store value



v_accvgpr_read v6, a204 // (op 14607) 
v_accvgpr_read v7, a205 // (op 14607) 
v_accvgpr_read v8, a206 // (op 14607) 
v_accvgpr_read v9, a207 // (op 14607) 
v_mul_f32 v14, v11, v6 // (op 14607) 
v_mul_f32 v15, v11, v7 // (op 14607) 
v_mul_f32 v16, v11, v8 // (op 14607) 
v_mul_f32 v17, v11, v9 // (op 14607) 

v_lshrrev_b32 v6, 16, v14 // (op 14609) 
v_lshrrev_b32 v7, 16, v15 // (op 14609) 
v_lshrrev_b32 v8, 16, v16 // (op 14609) 
v_lshrrev_b32 v9, 16, v17 // (op 14609) 


v_pack_b32_f16 v4, v6, v7 // (op 14583) 
v_pack_b32_f16 v5, v8, v9 // (op 14583) 

s_lshl_b32 s0, s4, 6 // (op 14583) 
s_mul_i32 s1, 3, s0 // (op 14583) 
s_add_u32 s1, s1, 384 // (op 14583) 

v_add_u32 v0, v13, s1 // (op 14583) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14583) Store value



v_accvgpr_read v6, a208 // (op 14613) 
v_accvgpr_read v7, a209 // (op 14613) 
v_accvgpr_read v8, a210 // (op 14613) 
v_accvgpr_read v9, a211 // (op 14613) 
v_mul_f32 v14, v11, v6 // (op 14613) 
v_mul_f32 v15, v11, v7 // (op 14613) 
v_mul_f32 v16, v11, v8 // (op 14613) 
v_mul_f32 v17, v11, v9 // (op 14613) 

v_lshrrev_b32 v6, 16, v14 // (op 14615) 
v_lshrrev_b32 v7, 16, v15 // (op 14615) 
v_lshrrev_b32 v8, 16, v16 // (op 14615) 
v_lshrrev_b32 v9, 16, v17 // (op 14615) 


v_pack_b32_f16 v4, v6, v7 // (op 14585) 
v_pack_b32_f16 v5, v8, v9 // (op 14585) 

s_lshl_b32 s0, s4, 8 // (op 14585) 
s_add_u32 s0, s0, 384 // (op 14585) 
v_add_u32 v0, v13, s0 // (op 14585) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14585) Store value



v_accvgpr_read v6, a212 // (op 14619) 
v_accvgpr_read v7, a213 // (op 14619) 
v_accvgpr_read v8, a214 // (op 14619) 
v_accvgpr_read v9, a215 // (op 14619) 
v_mul_f32 v14, v11, v6 // (op 14619) 
v_mul_f32 v15, v11, v7 // (op 14619) 
v_mul_f32 v16, v11, v8 // (op 14619) 
v_mul_f32 v17, v11, v9 // (op 14619) 

v_lshrrev_b32 v6, 16, v14 // (op 14621) 
v_lshrrev_b32 v7, 16, v15 // (op 14621) 
v_lshrrev_b32 v8, 16, v16 // (op 14621) 
v_lshrrev_b32 v9, 16, v17 // (op 14621) 


v_pack_b32_f16 v4, v6, v7 // (op 14587) 
v_pack_b32_f16 v5, v8, v9 // (op 14587) 

s_lshl_b32 s0, s4, 6 // (op 14587) 
s_mul_i32 s1, 5, s0 // (op 14587) 
s_add_u32 s1, s1, 384 // (op 14587) 

v_add_u32 v0, v13, s1 // (op 14587) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14587) Store value



v_accvgpr_read v6, a216 // (op 14625) 
v_accvgpr_read v7, a217 // (op 14625) 
v_accvgpr_read v8, a218 // (op 14625) 
v_accvgpr_read v9, a219 // (op 14625) 
v_mul_f32 v14, v11, v6 // (op 14625) 
v_mul_f32 v15, v11, v7 // (op 14625) 
v_mul_f32 v16, v11, v8 // (op 14625) 
v_mul_f32 v17, v11, v9 // (op 14625) 

v_lshrrev_b32 v6, 16, v14 // (op 14627) 
v_lshrrev_b32 v7, 16, v15 // (op 14627) 
v_lshrrev_b32 v8, 16, v16 // (op 14627) 
v_lshrrev_b32 v9, 16, v17 // (op 14627) 


v_pack_b32_f16 v4, v6, v7 // (op 14589) 
v_pack_b32_f16 v5, v8, v9 // (op 14589) 

s_lshl_b32 s0, s4, 6 // (op 14589) 
s_mul_i32 s1, 6, s0 // (op 14589) 
s_add_u32 s1, s1, 384 // (op 14589) 

v_add_u32 v0, v13, s1 // (op 14589) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14589) Store value



v_accvgpr_read v6, a220 // (op 14631) 
v_accvgpr_read v7, a221 // (op 14631) 
v_accvgpr_read v8, a222 // (op 14631) 
v_accvgpr_read v9, a223 // (op 14631) 
v_mul_f32 v14, v11, v6 // (op 14631) 
v_mul_f32 v15, v11, v7 // (op 14631) 
v_mul_f32 v16, v11, v8 // (op 14631) 
v_mul_f32 v17, v11, v9 // (op 14631) 

v_lshrrev_b32 v6, 16, v14 // (op 14633) 
v_lshrrev_b32 v7, 16, v15 // (op 14633) 
v_lshrrev_b32 v8, 16, v16 // (op 14633) 
v_lshrrev_b32 v9, 16, v17 // (op 14633) 


v_pack_b32_f16 v4, v6, v7 // (op 14591) 
v_pack_b32_f16 v5, v8, v9 // (op 14591) 

s_lshl_b32 s0, s4, 6 // (op 14591) 
s_mul_i32 s1, 7, s0 // (op 14591) 
s_add_u32 s1, s1, 384 // (op 14591) 

v_add_u32 v0, v13, s1 // (op 14591) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14591) Store value



v_accvgpr_read v6, a224 // (op 14721) 
v_accvgpr_read v7, a225 // (op 14721) 
v_accvgpr_read v8, a226 // (op 14721) 
v_accvgpr_read v9, a227 // (op 14721) 
v_mul_f32 v14, v11, v6 // (op 14721) 
v_mul_f32 v15, v11, v7 // (op 14721) 
v_mul_f32 v16, v11, v8 // (op 14721) 
v_mul_f32 v17, v11, v9 // (op 14721) 

v_lshrrev_b32 v6, 16, v14 // (op 14723) 
v_lshrrev_b32 v7, 16, v15 // (op 14723) 
v_lshrrev_b32 v8, 16, v16 // (op 14723) 
v_lshrrev_b32 v9, 16, v17 // (op 14723) 


v_pack_b32_f16 v4, v6, v7 // (op 14725) 
v_pack_b32_f16 v5, v8, v9 // (op 14725) 

v_add_u32 v0, 448, v13 // (op 14725) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14725) Store value



v_accvgpr_read v6, a228 // (op 14743) 
v_accvgpr_read v7, a229 // (op 14743) 
v_accvgpr_read v8, a230 // (op 14743) 
v_accvgpr_read v9, a231 // (op 14743) 
v_mul_f32 v14, v11, v6 // (op 14743) 
v_mul_f32 v15, v11, v7 // (op 14743) 
v_mul_f32 v16, v11, v8 // (op 14743) 
v_mul_f32 v17, v11, v9 // (op 14743) 

v_lshrrev_b32 v6, 16, v14 // (op 14745) 
v_lshrrev_b32 v7, 16, v15 // (op 14745) 
v_lshrrev_b32 v8, 16, v16 // (op 14745) 
v_lshrrev_b32 v9, 16, v17 // (op 14745) 


v_pack_b32_f16 v4, v6, v7 // (op 14727) 
v_pack_b32_f16 v5, v8, v9 // (op 14727) 

s_lshl_b32 s0, s4, 6 // (op 14727) 
s_add_u32 s0, s0, 448 // (op 14727) 
v_add_u32 v0, v13, s0 // (op 14727) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14727) Store value



v_accvgpr_read v6, a232 // (op 14749) 
v_accvgpr_read v7, a233 // (op 14749) 
v_accvgpr_read v8, a234 // (op 14749) 
v_accvgpr_read v9, a235 // (op 14749) 
v_mul_f32 v14, v11, v6 // (op 14749) 
v_mul_f32 v15, v11, v7 // (op 14749) 
v_mul_f32 v16, v11, v8 // (op 14749) 
v_mul_f32 v17, v11, v9 // (op 14749) 

v_lshrrev_b32 v6, 16, v14 // (op 14751) 
v_lshrrev_b32 v7, 16, v15 // (op 14751) 
v_lshrrev_b32 v8, 16, v16 // (op 14751) 
v_lshrrev_b32 v9, 16, v17 // (op 14751) 


v_pack_b32_f16 v4, v6, v7 // (op 14729) 
v_pack_b32_f16 v5, v8, v9 // (op 14729) 

s_lshl_b32 s0, s4, 7 // (op 14729) 
s_add_u32 s0, s0, 448 // (op 14729) 
v_add_u32 v0, v13, s0 // (op 14729) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14729) Store value



v_accvgpr_read v6, a236 // (op 14755) 
v_accvgpr_read v7, a237 // (op 14755) 
v_accvgpr_read v8, a238 // (op 14755) 
v_accvgpr_read v9, a239 // (op 14755) 
v_mul_f32 v14, v11, v6 // (op 14755) 
v_mul_f32 v15, v11, v7 // (op 14755) 
v_mul_f32 v16, v11, v8 // (op 14755) 
v_mul_f32 v17, v11, v9 // (op 14755) 

v_lshrrev_b32 v6, 16, v14 // (op 14757) 
v_lshrrev_b32 v7, 16, v15 // (op 14757) 
v_lshrrev_b32 v8, 16, v16 // (op 14757) 
v_lshrrev_b32 v9, 16, v17 // (op 14757) 


v_pack_b32_f16 v4, v6, v7 // (op 14731) 
v_pack_b32_f16 v5, v8, v9 // (op 14731) 

s_lshl_b32 s0, s4, 6 // (op 14731) 
s_mul_i32 s1, 3, s0 // (op 14731) 
s_add_u32 s1, s1, 448 // (op 14731) 

v_add_u32 v0, v13, s1 // (op 14731) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14731) Store value



v_accvgpr_read v6, a240 // (op 14761) 
v_accvgpr_read v7, a241 // (op 14761) 
v_accvgpr_read v8, a242 // (op 14761) 
v_accvgpr_read v9, a243 // (op 14761) 
v_mul_f32 v14, v11, v6 // (op 14761) 
v_mul_f32 v15, v11, v7 // (op 14761) 
v_mul_f32 v16, v11, v8 // (op 14761) 
v_mul_f32 v17, v11, v9 // (op 14761) 

v_lshrrev_b32 v6, 16, v14 // (op 14763) 
v_lshrrev_b32 v7, 16, v15 // (op 14763) 
v_lshrrev_b32 v8, 16, v16 // (op 14763) 
v_lshrrev_b32 v9, 16, v17 // (op 14763) 


v_pack_b32_f16 v4, v6, v7 // (op 14733) 
v_pack_b32_f16 v5, v8, v9 // (op 14733) 

s_lshl_b32 s0, s4, 8 // (op 14733) 
s_add_u32 s0, s0, 448 // (op 14733) 
v_add_u32 v0, v13, s0 // (op 14733) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14733) Store value



v_accvgpr_read v6, a244 // (op 14767) 
v_accvgpr_read v7, a245 // (op 14767) 
v_accvgpr_read v8, a246 // (op 14767) 
v_accvgpr_read v9, a247 // (op 14767) 
v_mul_f32 v14, v11, v6 // (op 14767) 
v_mul_f32 v15, v11, v7 // (op 14767) 
v_mul_f32 v16, v11, v8 // (op 14767) 
v_mul_f32 v17, v11, v9 // (op 14767) 

v_lshrrev_b32 v6, 16, v14 // (op 14769) 
v_lshrrev_b32 v7, 16, v15 // (op 14769) 
v_lshrrev_b32 v8, 16, v16 // (op 14769) 
v_lshrrev_b32 v9, 16, v17 // (op 14769) 


v_pack_b32_f16 v4, v6, v7 // (op 14735) 
v_pack_b32_f16 v5, v8, v9 // (op 14735) 

s_lshl_b32 s0, s4, 6 // (op 14735) 
s_mul_i32 s1, 5, s0 // (op 14735) 
s_add_u32 s1, s1, 448 // (op 14735) 

v_add_u32 v0, v13, s1 // (op 14735) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14735) Store value



v_accvgpr_read v6, a248 // (op 14773) 
v_accvgpr_read v7, a249 // (op 14773) 
v_accvgpr_read v8, a250 // (op 14773) 
v_accvgpr_read v9, a251 // (op 14773) 
v_mul_f32 v14, v11, v6 // (op 14773) 
v_mul_f32 v15, v11, v7 // (op 14773) 
v_mul_f32 v16, v11, v8 // (op 14773) 
v_mul_f32 v17, v11, v9 // (op 14773) 

v_lshrrev_b32 v6, 16, v14 // (op 14775) 
v_lshrrev_b32 v7, 16, v15 // (op 14775) 
v_lshrrev_b32 v8, 16, v16 // (op 14775) 
v_lshrrev_b32 v9, 16, v17 // (op 14775) 


v_pack_b32_f16 v4, v6, v7 // (op 14737) 
v_pack_b32_f16 v5, v8, v9 // (op 14737) 

s_lshl_b32 s0, s4, 6 // (op 14737) 
s_mul_i32 s1, 6, s0 // (op 14737) 
s_add_u32 s1, s1, 448 // (op 14737) 

v_add_u32 v0, v13, s1 // (op 14737) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14737) Store value



v_accvgpr_read v6, a252 // (op 14779) 
v_accvgpr_read v7, a253 // (op 14779) 
v_accvgpr_read v8, a254 // (op 14779) 
v_accvgpr_read v9, a255 // (op 14779) 
v_mul_f32 v14, v11, v6 // (op 14779) 
v_mul_f32 v15, v11, v7 // (op 14779) 
v_mul_f32 v16, v11, v8 // (op 14779) 
v_mul_f32 v17, v11, v9 // (op 14779) 

v_lshrrev_b32 v6, 16, v14 // (op 14781) 
v_lshrrev_b32 v7, 16, v15 // (op 14781) 
v_lshrrev_b32 v8, 16, v16 // (op 14781) 
v_lshrrev_b32 v9, 16, v17 // (op 14781) 


v_pack_b32_f16 v4, v6, v7 // (op 14739) 
v_pack_b32_f16 v5, v8, v9 // (op 14739) 

s_lshl_b32 s0, s4, 6 // (op 14739) 
s_mul_i32 s1, 7, s0 // (op 14739) 
s_add_u32 s1, s1, 448 // (op 14739) 

v_add_u32 v0, v13, s1 // (op 14739) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 14739) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_135:


v_add_u32 v0, 448, v12 // (op 16299) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16299) Load value

v_accvgpr_read v4, a224 // (op 16278) 
v_accvgpr_read v5, a225 // (op 16278) 
v_accvgpr_read v6, a226 // (op 16278) 
v_accvgpr_read v7, a227 // (op 16278) 
v_mul_f32 v14, v11, v4 // (op 16278) 
v_mul_f32 v15, v11, v5 // (op 16278) 
v_mul_f32 v16, v11, v6 // (op 16278) 
v_mul_f32 v17, v11, v7 // (op 16278) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16298) 
v_lshrrev_b32 v7, 16, v8 // (op 16298) 
v_lshlrev_b32 v6, 16, v6 // (op 16298) 
v_lshlrev_b32 v7, 16, v7 // (op 16298) 
v_mul_f32 v18, v10, v6 // (op 16298) 
v_mul_f32 v19, v10, v7 // (op 16298) 
v_and_b32 v6, 65535, v9 // (op 16298) 
v_lshrrev_b32 v7, 16, v9 // (op 16298) 
v_lshlrev_b32 v6, 16, v6 // (op 16298) 
v_lshlrev_b32 v7, 16, v7 // (op 16298) 
v_mul_f32 v20, v10, v6 // (op 16298) 
v_mul_f32 v21, v10, v7 // (op 16298) 


v_add_f32 v6, v18, v14 // (op 16279) 
v_add_f32 v7, v19, v15 // (op 16279) 
v_add_f32 v8, v20, v16 // (op 16279) 
v_add_f32 v9, v21, v17 // (op 16279) 


v_lshrrev_b32 v14, 16, v6 // (op 16280) 
v_lshrrev_b32 v15, 16, v7 // (op 16280) 
v_lshrrev_b32 v16, 16, v8 // (op 16280) 
v_lshrrev_b32 v17, 16, v9 // (op 16280) 


v_pack_b32_f16 v8, v14, v15 // (op 16282) 
v_pack_b32_f16 v9, v16, v17 // (op 16282) 

v_add_u32 v0, 448, v13 // (op 16282) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16282) Store value





s_lshl_b32 s0, s2, 6 // (op 16304) 
s_add_u32 s0, s0, 448 // (op 16304) 
v_add_u32 v0, v12, s0 // (op 16304) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16304) Load value

v_accvgpr_read v4, a228 // (op 16336) 
v_accvgpr_read v5, a229 // (op 16336) 
v_accvgpr_read v6, a230 // (op 16336) 
v_accvgpr_read v7, a231 // (op 16336) 
v_mul_f32 v14, v11, v4 // (op 16336) 
v_mul_f32 v15, v11, v5 // (op 16336) 
v_mul_f32 v16, v11, v6 // (op 16336) 
v_mul_f32 v17, v11, v7 // (op 16336) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16301) 
v_lshrrev_b32 v7, 16, v8 // (op 16301) 
v_lshlrev_b32 v6, 16, v6 // (op 16301) 
v_lshlrev_b32 v7, 16, v7 // (op 16301) 
v_mul_f32 v18, v10, v6 // (op 16301) 
v_mul_f32 v19, v10, v7 // (op 16301) 
v_and_b32 v6, 65535, v9 // (op 16301) 
v_lshrrev_b32 v7, 16, v9 // (op 16301) 
v_lshlrev_b32 v6, 16, v6 // (op 16301) 
v_lshlrev_b32 v7, 16, v7 // (op 16301) 
v_mul_f32 v20, v10, v6 // (op 16301) 
v_mul_f32 v21, v10, v7 // (op 16301) 


v_add_f32 v6, v18, v14 // (op 16302) 
v_add_f32 v7, v19, v15 // (op 16302) 
v_add_f32 v8, v20, v16 // (op 16302) 
v_add_f32 v9, v21, v17 // (op 16302) 


v_lshrrev_b32 v14, 16, v6 // (op 16303) 
v_lshrrev_b32 v15, 16, v7 // (op 16303) 
v_lshrrev_b32 v16, 16, v8 // (op 16303) 
v_lshrrev_b32 v17, 16, v9 // (op 16303) 


v_pack_b32_f16 v8, v14, v15 // (op 16284) 
v_pack_b32_f16 v9, v16, v17 // (op 16284) 

s_lshl_b32 s0, s4, 6 // (op 16284) 
s_add_u32 s0, s0, 448 // (op 16284) 
v_add_u32 v0, v13, s0 // (op 16284) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16284) Store value





s_lshl_b32 s0, s2, 7 // (op 16309) 
s_add_u32 s0, s0, 448 // (op 16309) 
v_add_u32 v0, v12, s0 // (op 16309) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16309) Load value

v_accvgpr_read v4, a232 // (op 16338) 
v_accvgpr_read v5, a233 // (op 16338) 
v_accvgpr_read v6, a234 // (op 16338) 
v_accvgpr_read v7, a235 // (op 16338) 
v_mul_f32 v14, v11, v4 // (op 16338) 
v_mul_f32 v15, v11, v5 // (op 16338) 
v_mul_f32 v16, v11, v6 // (op 16338) 
v_mul_f32 v17, v11, v7 // (op 16338) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16306) 
v_lshrrev_b32 v7, 16, v8 // (op 16306) 
v_lshlrev_b32 v6, 16, v6 // (op 16306) 
v_lshlrev_b32 v7, 16, v7 // (op 16306) 
v_mul_f32 v18, v10, v6 // (op 16306) 
v_mul_f32 v19, v10, v7 // (op 16306) 
v_and_b32 v6, 65535, v9 // (op 16306) 
v_lshrrev_b32 v7, 16, v9 // (op 16306) 
v_lshlrev_b32 v6, 16, v6 // (op 16306) 
v_lshlrev_b32 v7, 16, v7 // (op 16306) 
v_mul_f32 v20, v10, v6 // (op 16306) 
v_mul_f32 v21, v10, v7 // (op 16306) 


v_add_f32 v6, v18, v14 // (op 16307) 
v_add_f32 v7, v19, v15 // (op 16307) 
v_add_f32 v8, v20, v16 // (op 16307) 
v_add_f32 v9, v21, v17 // (op 16307) 


v_lshrrev_b32 v14, 16, v6 // (op 16308) 
v_lshrrev_b32 v15, 16, v7 // (op 16308) 
v_lshrrev_b32 v16, 16, v8 // (op 16308) 
v_lshrrev_b32 v17, 16, v9 // (op 16308) 


v_pack_b32_f16 v8, v14, v15 // (op 16286) 
v_pack_b32_f16 v9, v16, v17 // (op 16286) 

s_lshl_b32 s0, s4, 7 // (op 16286) 
s_add_u32 s0, s0, 448 // (op 16286) 
v_add_u32 v0, v13, s0 // (op 16286) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16286) Store value





s_lshl_b32 s0, s2, 6 // (op 16314) 
s_mul_i32 s1, 3, s0 // (op 16314) 
s_add_u32 s1, s1, 448 // (op 16314) 

v_add_u32 v0, v12, s1 // (op 16314) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16314) Load value

v_accvgpr_read v4, a236 // (op 16340) 
v_accvgpr_read v5, a237 // (op 16340) 
v_accvgpr_read v6, a238 // (op 16340) 
v_accvgpr_read v7, a239 // (op 16340) 
v_mul_f32 v14, v11, v4 // (op 16340) 
v_mul_f32 v15, v11, v5 // (op 16340) 
v_mul_f32 v16, v11, v6 // (op 16340) 
v_mul_f32 v17, v11, v7 // (op 16340) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16311) 
v_lshrrev_b32 v7, 16, v8 // (op 16311) 
v_lshlrev_b32 v6, 16, v6 // (op 16311) 
v_lshlrev_b32 v7, 16, v7 // (op 16311) 
v_mul_f32 v18, v10, v6 // (op 16311) 
v_mul_f32 v19, v10, v7 // (op 16311) 
v_and_b32 v6, 65535, v9 // (op 16311) 
v_lshrrev_b32 v7, 16, v9 // (op 16311) 
v_lshlrev_b32 v6, 16, v6 // (op 16311) 
v_lshlrev_b32 v7, 16, v7 // (op 16311) 
v_mul_f32 v20, v10, v6 // (op 16311) 
v_mul_f32 v21, v10, v7 // (op 16311) 


v_add_f32 v6, v18, v14 // (op 16312) 
v_add_f32 v7, v19, v15 // (op 16312) 
v_add_f32 v8, v20, v16 // (op 16312) 
v_add_f32 v9, v21, v17 // (op 16312) 


v_lshrrev_b32 v14, 16, v6 // (op 16313) 
v_lshrrev_b32 v15, 16, v7 // (op 16313) 
v_lshrrev_b32 v16, 16, v8 // (op 16313) 
v_lshrrev_b32 v17, 16, v9 // (op 16313) 


v_pack_b32_f16 v8, v14, v15 // (op 16288) 
v_pack_b32_f16 v9, v16, v17 // (op 16288) 

s_lshl_b32 s0, s4, 6 // (op 16288) 
s_mul_i32 s1, 3, s0 // (op 16288) 
s_add_u32 s1, s1, 448 // (op 16288) 

v_add_u32 v0, v13, s1 // (op 16288) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16288) Store value





s_lshl_b32 s0, s2, 8 // (op 16319) 
s_add_u32 s0, s0, 448 // (op 16319) 
v_add_u32 v0, v12, s0 // (op 16319) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16319) Load value

v_accvgpr_read v4, a240 // (op 16342) 
v_accvgpr_read v5, a241 // (op 16342) 
v_accvgpr_read v6, a242 // (op 16342) 
v_accvgpr_read v7, a243 // (op 16342) 
v_mul_f32 v14, v11, v4 // (op 16342) 
v_mul_f32 v15, v11, v5 // (op 16342) 
v_mul_f32 v16, v11, v6 // (op 16342) 
v_mul_f32 v17, v11, v7 // (op 16342) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16316) 
v_lshrrev_b32 v7, 16, v8 // (op 16316) 
v_lshlrev_b32 v6, 16, v6 // (op 16316) 
v_lshlrev_b32 v7, 16, v7 // (op 16316) 
v_mul_f32 v18, v10, v6 // (op 16316) 
v_mul_f32 v19, v10, v7 // (op 16316) 
v_and_b32 v6, 65535, v9 // (op 16316) 
v_lshrrev_b32 v7, 16, v9 // (op 16316) 
v_lshlrev_b32 v6, 16, v6 // (op 16316) 
v_lshlrev_b32 v7, 16, v7 // (op 16316) 
v_mul_f32 v20, v10, v6 // (op 16316) 
v_mul_f32 v21, v10, v7 // (op 16316) 


v_add_f32 v6, v18, v14 // (op 16317) 
v_add_f32 v7, v19, v15 // (op 16317) 
v_add_f32 v8, v20, v16 // (op 16317) 
v_add_f32 v9, v21, v17 // (op 16317) 


v_lshrrev_b32 v14, 16, v6 // (op 16318) 
v_lshrrev_b32 v15, 16, v7 // (op 16318) 
v_lshrrev_b32 v16, 16, v8 // (op 16318) 
v_lshrrev_b32 v17, 16, v9 // (op 16318) 


v_pack_b32_f16 v8, v14, v15 // (op 16290) 
v_pack_b32_f16 v9, v16, v17 // (op 16290) 

s_lshl_b32 s0, s4, 8 // (op 16290) 
s_add_u32 s0, s0, 448 // (op 16290) 
v_add_u32 v0, v13, s0 // (op 16290) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16290) Store value





s_lshl_b32 s0, s2, 6 // (op 16324) 
s_mul_i32 s1, 5, s0 // (op 16324) 
s_add_u32 s1, s1, 448 // (op 16324) 

v_add_u32 v0, v12, s1 // (op 16324) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16324) Load value

v_accvgpr_read v4, a244 // (op 16344) 
v_accvgpr_read v5, a245 // (op 16344) 
v_accvgpr_read v6, a246 // (op 16344) 
v_accvgpr_read v7, a247 // (op 16344) 
v_mul_f32 v14, v11, v4 // (op 16344) 
v_mul_f32 v15, v11, v5 // (op 16344) 
v_mul_f32 v16, v11, v6 // (op 16344) 
v_mul_f32 v17, v11, v7 // (op 16344) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16321) 
v_lshrrev_b32 v7, 16, v8 // (op 16321) 
v_lshlrev_b32 v6, 16, v6 // (op 16321) 
v_lshlrev_b32 v7, 16, v7 // (op 16321) 
v_mul_f32 v18, v10, v6 // (op 16321) 
v_mul_f32 v19, v10, v7 // (op 16321) 
v_and_b32 v6, 65535, v9 // (op 16321) 
v_lshrrev_b32 v7, 16, v9 // (op 16321) 
v_lshlrev_b32 v6, 16, v6 // (op 16321) 
v_lshlrev_b32 v7, 16, v7 // (op 16321) 
v_mul_f32 v20, v10, v6 // (op 16321) 
v_mul_f32 v21, v10, v7 // (op 16321) 


v_add_f32 v6, v18, v14 // (op 16322) 
v_add_f32 v7, v19, v15 // (op 16322) 
v_add_f32 v8, v20, v16 // (op 16322) 
v_add_f32 v9, v21, v17 // (op 16322) 


v_lshrrev_b32 v14, 16, v6 // (op 16323) 
v_lshrrev_b32 v15, 16, v7 // (op 16323) 
v_lshrrev_b32 v16, 16, v8 // (op 16323) 
v_lshrrev_b32 v17, 16, v9 // (op 16323) 


v_pack_b32_f16 v8, v14, v15 // (op 16292) 
v_pack_b32_f16 v9, v16, v17 // (op 16292) 

s_lshl_b32 s0, s4, 6 // (op 16292) 
s_mul_i32 s1, 5, s0 // (op 16292) 
s_add_u32 s1, s1, 448 // (op 16292) 

v_add_u32 v0, v13, s1 // (op 16292) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16292) Store value





s_lshl_b32 s0, s2, 6 // (op 16329) 
s_mul_i32 s1, 6, s0 // (op 16329) 
s_add_u32 s1, s1, 448 // (op 16329) 

v_add_u32 v0, v12, s1 // (op 16329) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16329) Load value

v_accvgpr_read v4, a248 // (op 16346) 
v_accvgpr_read v5, a249 // (op 16346) 
v_accvgpr_read v6, a250 // (op 16346) 
v_accvgpr_read v7, a251 // (op 16346) 
v_mul_f32 v14, v11, v4 // (op 16346) 
v_mul_f32 v15, v11, v5 // (op 16346) 
v_mul_f32 v16, v11, v6 // (op 16346) 
v_mul_f32 v17, v11, v7 // (op 16346) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16326) 
v_lshrrev_b32 v7, 16, v8 // (op 16326) 
v_lshlrev_b32 v6, 16, v6 // (op 16326) 
v_lshlrev_b32 v7, 16, v7 // (op 16326) 
v_mul_f32 v18, v10, v6 // (op 16326) 
v_mul_f32 v19, v10, v7 // (op 16326) 
v_and_b32 v6, 65535, v9 // (op 16326) 
v_lshrrev_b32 v7, 16, v9 // (op 16326) 
v_lshlrev_b32 v6, 16, v6 // (op 16326) 
v_lshlrev_b32 v7, 16, v7 // (op 16326) 
v_mul_f32 v20, v10, v6 // (op 16326) 
v_mul_f32 v21, v10, v7 // (op 16326) 


v_add_f32 v6, v18, v14 // (op 16327) 
v_add_f32 v7, v19, v15 // (op 16327) 
v_add_f32 v8, v20, v16 // (op 16327) 
v_add_f32 v9, v21, v17 // (op 16327) 


v_lshrrev_b32 v14, 16, v6 // (op 16328) 
v_lshrrev_b32 v15, 16, v7 // (op 16328) 
v_lshrrev_b32 v16, 16, v8 // (op 16328) 
v_lshrrev_b32 v17, 16, v9 // (op 16328) 


v_pack_b32_f16 v8, v14, v15 // (op 16294) 
v_pack_b32_f16 v9, v16, v17 // (op 16294) 

s_lshl_b32 s0, s4, 6 // (op 16294) 
s_mul_i32 s1, 6, s0 // (op 16294) 
s_add_u32 s1, s1, 448 // (op 16294) 

v_add_u32 v0, v13, s1 // (op 16294) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16294) Store value





s_lshl_b32 s0, s2, 6 // (op 16334) 
s_mul_i32 s1, 7, s0 // (op 16334) 
s_add_u32 s1, s1, 448 // (op 16334) 

v_add_u32 v0, v12, s1 // (op 16334) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16334) Load value

v_accvgpr_read v4, a252 // (op 16348) 
v_accvgpr_read v5, a253 // (op 16348) 
v_accvgpr_read v6, a254 // (op 16348) 
v_accvgpr_read v7, a255 // (op 16348) 
v_mul_f32 v14, v11, v4 // (op 16348) 
v_mul_f32 v15, v11, v5 // (op 16348) 
v_mul_f32 v16, v11, v6 // (op 16348) 
v_mul_f32 v17, v11, v7 // (op 16348) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16331) 
v_lshrrev_b32 v7, 16, v8 // (op 16331) 
v_lshlrev_b32 v6, 16, v6 // (op 16331) 
v_lshlrev_b32 v7, 16, v7 // (op 16331) 
v_mul_f32 v18, v10, v6 // (op 16331) 
v_mul_f32 v19, v10, v7 // (op 16331) 
v_and_b32 v6, 65535, v9 // (op 16331) 
v_lshrrev_b32 v7, 16, v9 // (op 16331) 
v_lshlrev_b32 v6, 16, v6 // (op 16331) 
v_lshlrev_b32 v7, 16, v7 // (op 16331) 
v_mul_f32 v20, v10, v6 // (op 16331) 
v_mul_f32 v21, v10, v7 // (op 16331) 


v_add_f32 v6, v18, v14 // (op 16332) 
v_add_f32 v7, v19, v15 // (op 16332) 
v_add_f32 v8, v20, v16 // (op 16332) 
v_add_f32 v9, v21, v17 // (op 16332) 


v_lshrrev_b32 v14, 16, v6 // (op 16333) 
v_lshrrev_b32 v15, 16, v7 // (op 16333) 
v_lshrrev_b32 v16, 16, v8 // (op 16333) 
v_lshrrev_b32 v17, 16, v9 // (op 16333) 


v_pack_b32_f16 v8, v14, v15 // (op 16296) 
v_pack_b32_f16 v9, v16, v17 // (op 16296) 

s_lshl_b32 s0, s4, 6 // (op 16296) 
s_mul_i32 s1, 7, s0 // (op 16296) 
s_add_u32 s1, s1, 448 // (op 16296) 

v_add_u32 v0, v13, s1 // (op 16296) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16296) Store value





v_add_u32 v0, 384, v12 // (op 16128) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16128) Load value

v_accvgpr_read v4, a192 // (op 16107) 
v_accvgpr_read v5, a193 // (op 16107) 
v_accvgpr_read v6, a194 // (op 16107) 
v_accvgpr_read v7, a195 // (op 16107) 
v_mul_f32 v14, v11, v4 // (op 16107) 
v_mul_f32 v15, v11, v5 // (op 16107) 
v_mul_f32 v16, v11, v6 // (op 16107) 
v_mul_f32 v17, v11, v7 // (op 16107) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16127) 
v_lshrrev_b32 v7, 16, v8 // (op 16127) 
v_lshlrev_b32 v6, 16, v6 // (op 16127) 
v_lshlrev_b32 v7, 16, v7 // (op 16127) 
v_mul_f32 v18, v10, v6 // (op 16127) 
v_mul_f32 v19, v10, v7 // (op 16127) 
v_and_b32 v6, 65535, v9 // (op 16127) 
v_lshrrev_b32 v7, 16, v9 // (op 16127) 
v_lshlrev_b32 v6, 16, v6 // (op 16127) 
v_lshlrev_b32 v7, 16, v7 // (op 16127) 
v_mul_f32 v20, v10, v6 // (op 16127) 
v_mul_f32 v21, v10, v7 // (op 16127) 


v_add_f32 v6, v18, v14 // (op 16108) 
v_add_f32 v7, v19, v15 // (op 16108) 
v_add_f32 v8, v20, v16 // (op 16108) 
v_add_f32 v9, v21, v17 // (op 16108) 


v_lshrrev_b32 v14, 16, v6 // (op 16109) 
v_lshrrev_b32 v15, 16, v7 // (op 16109) 
v_lshrrev_b32 v16, 16, v8 // (op 16109) 
v_lshrrev_b32 v17, 16, v9 // (op 16109) 


v_pack_b32_f16 v8, v14, v15 // (op 16111) 
v_pack_b32_f16 v9, v16, v17 // (op 16111) 

v_add_u32 v0, 384, v13 // (op 16111) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16111) Store value





s_lshl_b32 s0, s2, 6 // (op 16133) 
s_add_u32 s0, s0, 384 // (op 16133) 
v_add_u32 v0, v12, s0 // (op 16133) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16133) Load value

v_accvgpr_read v4, a196 // (op 16165) 
v_accvgpr_read v5, a197 // (op 16165) 
v_accvgpr_read v6, a198 // (op 16165) 
v_accvgpr_read v7, a199 // (op 16165) 
v_mul_f32 v14, v11, v4 // (op 16165) 
v_mul_f32 v15, v11, v5 // (op 16165) 
v_mul_f32 v16, v11, v6 // (op 16165) 
v_mul_f32 v17, v11, v7 // (op 16165) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16130) 
v_lshrrev_b32 v7, 16, v8 // (op 16130) 
v_lshlrev_b32 v6, 16, v6 // (op 16130) 
v_lshlrev_b32 v7, 16, v7 // (op 16130) 
v_mul_f32 v18, v10, v6 // (op 16130) 
v_mul_f32 v19, v10, v7 // (op 16130) 
v_and_b32 v6, 65535, v9 // (op 16130) 
v_lshrrev_b32 v7, 16, v9 // (op 16130) 
v_lshlrev_b32 v6, 16, v6 // (op 16130) 
v_lshlrev_b32 v7, 16, v7 // (op 16130) 
v_mul_f32 v20, v10, v6 // (op 16130) 
v_mul_f32 v21, v10, v7 // (op 16130) 


v_add_f32 v6, v18, v14 // (op 16131) 
v_add_f32 v7, v19, v15 // (op 16131) 
v_add_f32 v8, v20, v16 // (op 16131) 
v_add_f32 v9, v21, v17 // (op 16131) 


v_lshrrev_b32 v14, 16, v6 // (op 16132) 
v_lshrrev_b32 v15, 16, v7 // (op 16132) 
v_lshrrev_b32 v16, 16, v8 // (op 16132) 
v_lshrrev_b32 v17, 16, v9 // (op 16132) 


v_pack_b32_f16 v8, v14, v15 // (op 16113) 
v_pack_b32_f16 v9, v16, v17 // (op 16113) 

s_lshl_b32 s0, s4, 6 // (op 16113) 
s_add_u32 s0, s0, 384 // (op 16113) 
v_add_u32 v0, v13, s0 // (op 16113) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16113) Store value





s_lshl_b32 s0, s2, 7 // (op 16138) 
s_add_u32 s0, s0, 384 // (op 16138) 
v_add_u32 v0, v12, s0 // (op 16138) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16138) Load value

v_accvgpr_read v4, a200 // (op 16167) 
v_accvgpr_read v5, a201 // (op 16167) 
v_accvgpr_read v6, a202 // (op 16167) 
v_accvgpr_read v7, a203 // (op 16167) 
v_mul_f32 v14, v11, v4 // (op 16167) 
v_mul_f32 v15, v11, v5 // (op 16167) 
v_mul_f32 v16, v11, v6 // (op 16167) 
v_mul_f32 v17, v11, v7 // (op 16167) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16135) 
v_lshrrev_b32 v7, 16, v8 // (op 16135) 
v_lshlrev_b32 v6, 16, v6 // (op 16135) 
v_lshlrev_b32 v7, 16, v7 // (op 16135) 
v_mul_f32 v18, v10, v6 // (op 16135) 
v_mul_f32 v19, v10, v7 // (op 16135) 
v_and_b32 v6, 65535, v9 // (op 16135) 
v_lshrrev_b32 v7, 16, v9 // (op 16135) 
v_lshlrev_b32 v6, 16, v6 // (op 16135) 
v_lshlrev_b32 v7, 16, v7 // (op 16135) 
v_mul_f32 v20, v10, v6 // (op 16135) 
v_mul_f32 v21, v10, v7 // (op 16135) 


v_add_f32 v6, v18, v14 // (op 16136) 
v_add_f32 v7, v19, v15 // (op 16136) 
v_add_f32 v8, v20, v16 // (op 16136) 
v_add_f32 v9, v21, v17 // (op 16136) 


v_lshrrev_b32 v14, 16, v6 // (op 16137) 
v_lshrrev_b32 v15, 16, v7 // (op 16137) 
v_lshrrev_b32 v16, 16, v8 // (op 16137) 
v_lshrrev_b32 v17, 16, v9 // (op 16137) 


v_pack_b32_f16 v8, v14, v15 // (op 16115) 
v_pack_b32_f16 v9, v16, v17 // (op 16115) 

s_lshl_b32 s0, s4, 7 // (op 16115) 
s_add_u32 s0, s0, 384 // (op 16115) 
v_add_u32 v0, v13, s0 // (op 16115) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16115) Store value





s_lshl_b32 s0, s2, 6 // (op 16143) 
s_mul_i32 s1, 3, s0 // (op 16143) 
s_add_u32 s1, s1, 384 // (op 16143) 

v_add_u32 v0, v12, s1 // (op 16143) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16143) Load value

v_accvgpr_read v4, a204 // (op 16169) 
v_accvgpr_read v5, a205 // (op 16169) 
v_accvgpr_read v6, a206 // (op 16169) 
v_accvgpr_read v7, a207 // (op 16169) 
v_mul_f32 v14, v11, v4 // (op 16169) 
v_mul_f32 v15, v11, v5 // (op 16169) 
v_mul_f32 v16, v11, v6 // (op 16169) 
v_mul_f32 v17, v11, v7 // (op 16169) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16140) 
v_lshrrev_b32 v7, 16, v8 // (op 16140) 
v_lshlrev_b32 v6, 16, v6 // (op 16140) 
v_lshlrev_b32 v7, 16, v7 // (op 16140) 
v_mul_f32 v18, v10, v6 // (op 16140) 
v_mul_f32 v19, v10, v7 // (op 16140) 
v_and_b32 v6, 65535, v9 // (op 16140) 
v_lshrrev_b32 v7, 16, v9 // (op 16140) 
v_lshlrev_b32 v6, 16, v6 // (op 16140) 
v_lshlrev_b32 v7, 16, v7 // (op 16140) 
v_mul_f32 v20, v10, v6 // (op 16140) 
v_mul_f32 v21, v10, v7 // (op 16140) 


v_add_f32 v6, v18, v14 // (op 16141) 
v_add_f32 v7, v19, v15 // (op 16141) 
v_add_f32 v8, v20, v16 // (op 16141) 
v_add_f32 v9, v21, v17 // (op 16141) 


v_lshrrev_b32 v14, 16, v6 // (op 16142) 
v_lshrrev_b32 v15, 16, v7 // (op 16142) 
v_lshrrev_b32 v16, 16, v8 // (op 16142) 
v_lshrrev_b32 v17, 16, v9 // (op 16142) 


v_pack_b32_f16 v8, v14, v15 // (op 16117) 
v_pack_b32_f16 v9, v16, v17 // (op 16117) 

s_lshl_b32 s0, s4, 6 // (op 16117) 
s_mul_i32 s1, 3, s0 // (op 16117) 
s_add_u32 s1, s1, 384 // (op 16117) 

v_add_u32 v0, v13, s1 // (op 16117) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16117) Store value





s_lshl_b32 s0, s2, 8 // (op 16148) 
s_add_u32 s0, s0, 384 // (op 16148) 
v_add_u32 v0, v12, s0 // (op 16148) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16148) Load value

v_accvgpr_read v4, a208 // (op 16171) 
v_accvgpr_read v5, a209 // (op 16171) 
v_accvgpr_read v6, a210 // (op 16171) 
v_accvgpr_read v7, a211 // (op 16171) 
v_mul_f32 v14, v11, v4 // (op 16171) 
v_mul_f32 v15, v11, v5 // (op 16171) 
v_mul_f32 v16, v11, v6 // (op 16171) 
v_mul_f32 v17, v11, v7 // (op 16171) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16145) 
v_lshrrev_b32 v7, 16, v8 // (op 16145) 
v_lshlrev_b32 v6, 16, v6 // (op 16145) 
v_lshlrev_b32 v7, 16, v7 // (op 16145) 
v_mul_f32 v18, v10, v6 // (op 16145) 
v_mul_f32 v19, v10, v7 // (op 16145) 
v_and_b32 v6, 65535, v9 // (op 16145) 
v_lshrrev_b32 v7, 16, v9 // (op 16145) 
v_lshlrev_b32 v6, 16, v6 // (op 16145) 
v_lshlrev_b32 v7, 16, v7 // (op 16145) 
v_mul_f32 v20, v10, v6 // (op 16145) 
v_mul_f32 v21, v10, v7 // (op 16145) 


v_add_f32 v6, v18, v14 // (op 16146) 
v_add_f32 v7, v19, v15 // (op 16146) 
v_add_f32 v8, v20, v16 // (op 16146) 
v_add_f32 v9, v21, v17 // (op 16146) 


v_lshrrev_b32 v14, 16, v6 // (op 16147) 
v_lshrrev_b32 v15, 16, v7 // (op 16147) 
v_lshrrev_b32 v16, 16, v8 // (op 16147) 
v_lshrrev_b32 v17, 16, v9 // (op 16147) 


v_pack_b32_f16 v8, v14, v15 // (op 16119) 
v_pack_b32_f16 v9, v16, v17 // (op 16119) 

s_lshl_b32 s0, s4, 8 // (op 16119) 
s_add_u32 s0, s0, 384 // (op 16119) 
v_add_u32 v0, v13, s0 // (op 16119) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16119) Store value





s_lshl_b32 s0, s2, 6 // (op 16153) 
s_mul_i32 s1, 5, s0 // (op 16153) 
s_add_u32 s1, s1, 384 // (op 16153) 

v_add_u32 v0, v12, s1 // (op 16153) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16153) Load value

v_accvgpr_read v4, a212 // (op 16173) 
v_accvgpr_read v5, a213 // (op 16173) 
v_accvgpr_read v6, a214 // (op 16173) 
v_accvgpr_read v7, a215 // (op 16173) 
v_mul_f32 v14, v11, v4 // (op 16173) 
v_mul_f32 v15, v11, v5 // (op 16173) 
v_mul_f32 v16, v11, v6 // (op 16173) 
v_mul_f32 v17, v11, v7 // (op 16173) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16150) 
v_lshrrev_b32 v7, 16, v8 // (op 16150) 
v_lshlrev_b32 v6, 16, v6 // (op 16150) 
v_lshlrev_b32 v7, 16, v7 // (op 16150) 
v_mul_f32 v18, v10, v6 // (op 16150) 
v_mul_f32 v19, v10, v7 // (op 16150) 
v_and_b32 v6, 65535, v9 // (op 16150) 
v_lshrrev_b32 v7, 16, v9 // (op 16150) 
v_lshlrev_b32 v6, 16, v6 // (op 16150) 
v_lshlrev_b32 v7, 16, v7 // (op 16150) 
v_mul_f32 v20, v10, v6 // (op 16150) 
v_mul_f32 v21, v10, v7 // (op 16150) 


v_add_f32 v6, v18, v14 // (op 16151) 
v_add_f32 v7, v19, v15 // (op 16151) 
v_add_f32 v8, v20, v16 // (op 16151) 
v_add_f32 v9, v21, v17 // (op 16151) 


v_lshrrev_b32 v14, 16, v6 // (op 16152) 
v_lshrrev_b32 v15, 16, v7 // (op 16152) 
v_lshrrev_b32 v16, 16, v8 // (op 16152) 
v_lshrrev_b32 v17, 16, v9 // (op 16152) 


v_pack_b32_f16 v8, v14, v15 // (op 16121) 
v_pack_b32_f16 v9, v16, v17 // (op 16121) 

s_lshl_b32 s0, s4, 6 // (op 16121) 
s_mul_i32 s1, 5, s0 // (op 16121) 
s_add_u32 s1, s1, 384 // (op 16121) 

v_add_u32 v0, v13, s1 // (op 16121) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16121) Store value





s_lshl_b32 s0, s2, 6 // (op 16158) 
s_mul_i32 s1, 6, s0 // (op 16158) 
s_add_u32 s1, s1, 384 // (op 16158) 

v_add_u32 v0, v12, s1 // (op 16158) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16158) Load value

v_accvgpr_read v4, a216 // (op 16175) 
v_accvgpr_read v5, a217 // (op 16175) 
v_accvgpr_read v6, a218 // (op 16175) 
v_accvgpr_read v7, a219 // (op 16175) 
v_mul_f32 v14, v11, v4 // (op 16175) 
v_mul_f32 v15, v11, v5 // (op 16175) 
v_mul_f32 v16, v11, v6 // (op 16175) 
v_mul_f32 v17, v11, v7 // (op 16175) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16155) 
v_lshrrev_b32 v7, 16, v8 // (op 16155) 
v_lshlrev_b32 v6, 16, v6 // (op 16155) 
v_lshlrev_b32 v7, 16, v7 // (op 16155) 
v_mul_f32 v18, v10, v6 // (op 16155) 
v_mul_f32 v19, v10, v7 // (op 16155) 
v_and_b32 v6, 65535, v9 // (op 16155) 
v_lshrrev_b32 v7, 16, v9 // (op 16155) 
v_lshlrev_b32 v6, 16, v6 // (op 16155) 
v_lshlrev_b32 v7, 16, v7 // (op 16155) 
v_mul_f32 v20, v10, v6 // (op 16155) 
v_mul_f32 v21, v10, v7 // (op 16155) 


v_add_f32 v6, v18, v14 // (op 16156) 
v_add_f32 v7, v19, v15 // (op 16156) 
v_add_f32 v8, v20, v16 // (op 16156) 
v_add_f32 v9, v21, v17 // (op 16156) 


v_lshrrev_b32 v14, 16, v6 // (op 16157) 
v_lshrrev_b32 v15, 16, v7 // (op 16157) 
v_lshrrev_b32 v16, 16, v8 // (op 16157) 
v_lshrrev_b32 v17, 16, v9 // (op 16157) 


v_pack_b32_f16 v8, v14, v15 // (op 16123) 
v_pack_b32_f16 v9, v16, v17 // (op 16123) 

s_lshl_b32 s0, s4, 6 // (op 16123) 
s_mul_i32 s1, 6, s0 // (op 16123) 
s_add_u32 s1, s1, 384 // (op 16123) 

v_add_u32 v0, v13, s1 // (op 16123) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16123) Store value





s_lshl_b32 s0, s2, 6 // (op 16163) 
s_mul_i32 s1, 7, s0 // (op 16163) 
s_add_u32 s1, s1, 384 // (op 16163) 

v_add_u32 v0, v12, s1 // (op 16163) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 16163) Load value

v_accvgpr_read v4, a220 // (op 16177) 
v_accvgpr_read v5, a221 // (op 16177) 
v_accvgpr_read v6, a222 // (op 16177) 
v_accvgpr_read v7, a223 // (op 16177) 
v_mul_f32 v14, v11, v4 // (op 16177) 
v_mul_f32 v15, v11, v5 // (op 16177) 
v_mul_f32 v16, v11, v6 // (op 16177) 
v_mul_f32 v17, v11, v7 // (op 16177) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 16160) 
v_lshrrev_b32 v7, 16, v8 // (op 16160) 
v_lshlrev_b32 v6, 16, v6 // (op 16160) 
v_lshlrev_b32 v7, 16, v7 // (op 16160) 
v_mul_f32 v18, v10, v6 // (op 16160) 
v_mul_f32 v19, v10, v7 // (op 16160) 
v_and_b32 v6, 65535, v9 // (op 16160) 
v_lshrrev_b32 v7, 16, v9 // (op 16160) 
v_lshlrev_b32 v6, 16, v6 // (op 16160) 
v_lshlrev_b32 v7, 16, v7 // (op 16160) 
v_mul_f32 v20, v10, v6 // (op 16160) 
v_mul_f32 v21, v10, v7 // (op 16160) 


v_add_f32 v6, v18, v14 // (op 16161) 
v_add_f32 v7, v19, v15 // (op 16161) 
v_add_f32 v8, v20, v16 // (op 16161) 
v_add_f32 v9, v21, v17 // (op 16161) 


v_lshrrev_b32 v14, 16, v6 // (op 16162) 
v_lshrrev_b32 v15, 16, v7 // (op 16162) 
v_lshrrev_b32 v16, 16, v8 // (op 16162) 
v_lshrrev_b32 v17, 16, v9 // (op 16162) 


v_pack_b32_f16 v8, v14, v15 // (op 16125) 
v_pack_b32_f16 v9, v16, v17 // (op 16125) 

s_lshl_b32 s0, s4, 6 // (op 16125) 
s_mul_i32 s1, 7, s0 // (op 16125) 
s_add_u32 s1, s1, 384 // (op 16125) 

v_add_u32 v0, v13, s1 // (op 16125) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 16125) Store value





v_add_u32 v0, 320, v12 // (op 15957) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15957) Load value

v_accvgpr_read v4, a160 // (op 15936) 
v_accvgpr_read v5, a161 // (op 15936) 
v_accvgpr_read v6, a162 // (op 15936) 
v_accvgpr_read v7, a163 // (op 15936) 
v_mul_f32 v14, v11, v4 // (op 15936) 
v_mul_f32 v15, v11, v5 // (op 15936) 
v_mul_f32 v16, v11, v6 // (op 15936) 
v_mul_f32 v17, v11, v7 // (op 15936) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15956) 
v_lshrrev_b32 v7, 16, v8 // (op 15956) 
v_lshlrev_b32 v6, 16, v6 // (op 15956) 
v_lshlrev_b32 v7, 16, v7 // (op 15956) 
v_mul_f32 v18, v10, v6 // (op 15956) 
v_mul_f32 v19, v10, v7 // (op 15956) 
v_and_b32 v6, 65535, v9 // (op 15956) 
v_lshrrev_b32 v7, 16, v9 // (op 15956) 
v_lshlrev_b32 v6, 16, v6 // (op 15956) 
v_lshlrev_b32 v7, 16, v7 // (op 15956) 
v_mul_f32 v20, v10, v6 // (op 15956) 
v_mul_f32 v21, v10, v7 // (op 15956) 


v_add_f32 v6, v18, v14 // (op 15937) 
v_add_f32 v7, v19, v15 // (op 15937) 
v_add_f32 v8, v20, v16 // (op 15937) 
v_add_f32 v9, v21, v17 // (op 15937) 


v_lshrrev_b32 v14, 16, v6 // (op 15938) 
v_lshrrev_b32 v15, 16, v7 // (op 15938) 
v_lshrrev_b32 v16, 16, v8 // (op 15938) 
v_lshrrev_b32 v17, 16, v9 // (op 15938) 


v_pack_b32_f16 v8, v14, v15 // (op 15940) 
v_pack_b32_f16 v9, v16, v17 // (op 15940) 

v_add_u32 v0, 320, v13 // (op 15940) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15940) Store value





s_lshl_b32 s0, s2, 6 // (op 15962) 
s_add_u32 s0, s0, 320 // (op 15962) 
v_add_u32 v0, v12, s0 // (op 15962) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15962) Load value

v_accvgpr_read v4, a164 // (op 15994) 
v_accvgpr_read v5, a165 // (op 15994) 
v_accvgpr_read v6, a166 // (op 15994) 
v_accvgpr_read v7, a167 // (op 15994) 
v_mul_f32 v14, v11, v4 // (op 15994) 
v_mul_f32 v15, v11, v5 // (op 15994) 
v_mul_f32 v16, v11, v6 // (op 15994) 
v_mul_f32 v17, v11, v7 // (op 15994) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15959) 
v_lshrrev_b32 v7, 16, v8 // (op 15959) 
v_lshlrev_b32 v6, 16, v6 // (op 15959) 
v_lshlrev_b32 v7, 16, v7 // (op 15959) 
v_mul_f32 v18, v10, v6 // (op 15959) 
v_mul_f32 v19, v10, v7 // (op 15959) 
v_and_b32 v6, 65535, v9 // (op 15959) 
v_lshrrev_b32 v7, 16, v9 // (op 15959) 
v_lshlrev_b32 v6, 16, v6 // (op 15959) 
v_lshlrev_b32 v7, 16, v7 // (op 15959) 
v_mul_f32 v20, v10, v6 // (op 15959) 
v_mul_f32 v21, v10, v7 // (op 15959) 


v_add_f32 v6, v18, v14 // (op 15960) 
v_add_f32 v7, v19, v15 // (op 15960) 
v_add_f32 v8, v20, v16 // (op 15960) 
v_add_f32 v9, v21, v17 // (op 15960) 


v_lshrrev_b32 v14, 16, v6 // (op 15961) 
v_lshrrev_b32 v15, 16, v7 // (op 15961) 
v_lshrrev_b32 v16, 16, v8 // (op 15961) 
v_lshrrev_b32 v17, 16, v9 // (op 15961) 


v_pack_b32_f16 v8, v14, v15 // (op 15942) 
v_pack_b32_f16 v9, v16, v17 // (op 15942) 

s_lshl_b32 s0, s4, 6 // (op 15942) 
s_add_u32 s0, s0, 320 // (op 15942) 
v_add_u32 v0, v13, s0 // (op 15942) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15942) Store value





s_lshl_b32 s0, s2, 7 // (op 15967) 
s_add_u32 s0, s0, 320 // (op 15967) 
v_add_u32 v0, v12, s0 // (op 15967) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15967) Load value

v_accvgpr_read v4, a168 // (op 15996) 
v_accvgpr_read v5, a169 // (op 15996) 
v_accvgpr_read v6, a170 // (op 15996) 
v_accvgpr_read v7, a171 // (op 15996) 
v_mul_f32 v14, v11, v4 // (op 15996) 
v_mul_f32 v15, v11, v5 // (op 15996) 
v_mul_f32 v16, v11, v6 // (op 15996) 
v_mul_f32 v17, v11, v7 // (op 15996) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15964) 
v_lshrrev_b32 v7, 16, v8 // (op 15964) 
v_lshlrev_b32 v6, 16, v6 // (op 15964) 
v_lshlrev_b32 v7, 16, v7 // (op 15964) 
v_mul_f32 v18, v10, v6 // (op 15964) 
v_mul_f32 v19, v10, v7 // (op 15964) 
v_and_b32 v6, 65535, v9 // (op 15964) 
v_lshrrev_b32 v7, 16, v9 // (op 15964) 
v_lshlrev_b32 v6, 16, v6 // (op 15964) 
v_lshlrev_b32 v7, 16, v7 // (op 15964) 
v_mul_f32 v20, v10, v6 // (op 15964) 
v_mul_f32 v21, v10, v7 // (op 15964) 


v_add_f32 v6, v18, v14 // (op 15965) 
v_add_f32 v7, v19, v15 // (op 15965) 
v_add_f32 v8, v20, v16 // (op 15965) 
v_add_f32 v9, v21, v17 // (op 15965) 


v_lshrrev_b32 v14, 16, v6 // (op 15966) 
v_lshrrev_b32 v15, 16, v7 // (op 15966) 
v_lshrrev_b32 v16, 16, v8 // (op 15966) 
v_lshrrev_b32 v17, 16, v9 // (op 15966) 


v_pack_b32_f16 v8, v14, v15 // (op 15944) 
v_pack_b32_f16 v9, v16, v17 // (op 15944) 

s_lshl_b32 s0, s4, 7 // (op 15944) 
s_add_u32 s0, s0, 320 // (op 15944) 
v_add_u32 v0, v13, s0 // (op 15944) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15944) Store value





s_lshl_b32 s0, s2, 6 // (op 15972) 
s_mul_i32 s1, 3, s0 // (op 15972) 
s_add_u32 s1, s1, 320 // (op 15972) 

v_add_u32 v0, v12, s1 // (op 15972) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15972) Load value

v_accvgpr_read v4, a172 // (op 15998) 
v_accvgpr_read v5, a173 // (op 15998) 
v_accvgpr_read v6, a174 // (op 15998) 
v_accvgpr_read v7, a175 // (op 15998) 
v_mul_f32 v14, v11, v4 // (op 15998) 
v_mul_f32 v15, v11, v5 // (op 15998) 
v_mul_f32 v16, v11, v6 // (op 15998) 
v_mul_f32 v17, v11, v7 // (op 15998) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15969) 
v_lshrrev_b32 v7, 16, v8 // (op 15969) 
v_lshlrev_b32 v6, 16, v6 // (op 15969) 
v_lshlrev_b32 v7, 16, v7 // (op 15969) 
v_mul_f32 v18, v10, v6 // (op 15969) 
v_mul_f32 v19, v10, v7 // (op 15969) 
v_and_b32 v6, 65535, v9 // (op 15969) 
v_lshrrev_b32 v7, 16, v9 // (op 15969) 
v_lshlrev_b32 v6, 16, v6 // (op 15969) 
v_lshlrev_b32 v7, 16, v7 // (op 15969) 
v_mul_f32 v20, v10, v6 // (op 15969) 
v_mul_f32 v21, v10, v7 // (op 15969) 


v_add_f32 v6, v18, v14 // (op 15970) 
v_add_f32 v7, v19, v15 // (op 15970) 
v_add_f32 v8, v20, v16 // (op 15970) 
v_add_f32 v9, v21, v17 // (op 15970) 


v_lshrrev_b32 v14, 16, v6 // (op 15971) 
v_lshrrev_b32 v15, 16, v7 // (op 15971) 
v_lshrrev_b32 v16, 16, v8 // (op 15971) 
v_lshrrev_b32 v17, 16, v9 // (op 15971) 


v_pack_b32_f16 v8, v14, v15 // (op 15946) 
v_pack_b32_f16 v9, v16, v17 // (op 15946) 

s_lshl_b32 s0, s4, 6 // (op 15946) 
s_mul_i32 s1, 3, s0 // (op 15946) 
s_add_u32 s1, s1, 320 // (op 15946) 

v_add_u32 v0, v13, s1 // (op 15946) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15946) Store value





s_lshl_b32 s0, s2, 8 // (op 15977) 
s_add_u32 s0, s0, 320 // (op 15977) 
v_add_u32 v0, v12, s0 // (op 15977) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15977) Load value

v_accvgpr_read v4, a176 // (op 16000) 
v_accvgpr_read v5, a177 // (op 16000) 
v_accvgpr_read v6, a178 // (op 16000) 
v_accvgpr_read v7, a179 // (op 16000) 
v_mul_f32 v14, v11, v4 // (op 16000) 
v_mul_f32 v15, v11, v5 // (op 16000) 
v_mul_f32 v16, v11, v6 // (op 16000) 
v_mul_f32 v17, v11, v7 // (op 16000) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15974) 
v_lshrrev_b32 v7, 16, v8 // (op 15974) 
v_lshlrev_b32 v6, 16, v6 // (op 15974) 
v_lshlrev_b32 v7, 16, v7 // (op 15974) 
v_mul_f32 v18, v10, v6 // (op 15974) 
v_mul_f32 v19, v10, v7 // (op 15974) 
v_and_b32 v6, 65535, v9 // (op 15974) 
v_lshrrev_b32 v7, 16, v9 // (op 15974) 
v_lshlrev_b32 v6, 16, v6 // (op 15974) 
v_lshlrev_b32 v7, 16, v7 // (op 15974) 
v_mul_f32 v20, v10, v6 // (op 15974) 
v_mul_f32 v21, v10, v7 // (op 15974) 


v_add_f32 v6, v18, v14 // (op 15975) 
v_add_f32 v7, v19, v15 // (op 15975) 
v_add_f32 v8, v20, v16 // (op 15975) 
v_add_f32 v9, v21, v17 // (op 15975) 


v_lshrrev_b32 v14, 16, v6 // (op 15976) 
v_lshrrev_b32 v15, 16, v7 // (op 15976) 
v_lshrrev_b32 v16, 16, v8 // (op 15976) 
v_lshrrev_b32 v17, 16, v9 // (op 15976) 


v_pack_b32_f16 v8, v14, v15 // (op 15948) 
v_pack_b32_f16 v9, v16, v17 // (op 15948) 

s_lshl_b32 s0, s4, 8 // (op 15948) 
s_add_u32 s0, s0, 320 // (op 15948) 
v_add_u32 v0, v13, s0 // (op 15948) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15948) Store value





s_lshl_b32 s0, s2, 6 // (op 15982) 
s_mul_i32 s1, 5, s0 // (op 15982) 
s_add_u32 s1, s1, 320 // (op 15982) 

v_add_u32 v0, v12, s1 // (op 15982) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15982) Load value

v_accvgpr_read v4, a180 // (op 16002) 
v_accvgpr_read v5, a181 // (op 16002) 
v_accvgpr_read v6, a182 // (op 16002) 
v_accvgpr_read v7, a183 // (op 16002) 
v_mul_f32 v14, v11, v4 // (op 16002) 
v_mul_f32 v15, v11, v5 // (op 16002) 
v_mul_f32 v16, v11, v6 // (op 16002) 
v_mul_f32 v17, v11, v7 // (op 16002) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15979) 
v_lshrrev_b32 v7, 16, v8 // (op 15979) 
v_lshlrev_b32 v6, 16, v6 // (op 15979) 
v_lshlrev_b32 v7, 16, v7 // (op 15979) 
v_mul_f32 v18, v10, v6 // (op 15979) 
v_mul_f32 v19, v10, v7 // (op 15979) 
v_and_b32 v6, 65535, v9 // (op 15979) 
v_lshrrev_b32 v7, 16, v9 // (op 15979) 
v_lshlrev_b32 v6, 16, v6 // (op 15979) 
v_lshlrev_b32 v7, 16, v7 // (op 15979) 
v_mul_f32 v20, v10, v6 // (op 15979) 
v_mul_f32 v21, v10, v7 // (op 15979) 


v_add_f32 v6, v18, v14 // (op 15980) 
v_add_f32 v7, v19, v15 // (op 15980) 
v_add_f32 v8, v20, v16 // (op 15980) 
v_add_f32 v9, v21, v17 // (op 15980) 


v_lshrrev_b32 v14, 16, v6 // (op 15981) 
v_lshrrev_b32 v15, 16, v7 // (op 15981) 
v_lshrrev_b32 v16, 16, v8 // (op 15981) 
v_lshrrev_b32 v17, 16, v9 // (op 15981) 


v_pack_b32_f16 v8, v14, v15 // (op 15950) 
v_pack_b32_f16 v9, v16, v17 // (op 15950) 

s_lshl_b32 s0, s4, 6 // (op 15950) 
s_mul_i32 s1, 5, s0 // (op 15950) 
s_add_u32 s1, s1, 320 // (op 15950) 

v_add_u32 v0, v13, s1 // (op 15950) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15950) Store value





s_lshl_b32 s0, s2, 6 // (op 15987) 
s_mul_i32 s1, 6, s0 // (op 15987) 
s_add_u32 s1, s1, 320 // (op 15987) 

v_add_u32 v0, v12, s1 // (op 15987) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15987) Load value

v_accvgpr_read v4, a184 // (op 16004) 
v_accvgpr_read v5, a185 // (op 16004) 
v_accvgpr_read v6, a186 // (op 16004) 
v_accvgpr_read v7, a187 // (op 16004) 
v_mul_f32 v14, v11, v4 // (op 16004) 
v_mul_f32 v15, v11, v5 // (op 16004) 
v_mul_f32 v16, v11, v6 // (op 16004) 
v_mul_f32 v17, v11, v7 // (op 16004) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15984) 
v_lshrrev_b32 v7, 16, v8 // (op 15984) 
v_lshlrev_b32 v6, 16, v6 // (op 15984) 
v_lshlrev_b32 v7, 16, v7 // (op 15984) 
v_mul_f32 v18, v10, v6 // (op 15984) 
v_mul_f32 v19, v10, v7 // (op 15984) 
v_and_b32 v6, 65535, v9 // (op 15984) 
v_lshrrev_b32 v7, 16, v9 // (op 15984) 
v_lshlrev_b32 v6, 16, v6 // (op 15984) 
v_lshlrev_b32 v7, 16, v7 // (op 15984) 
v_mul_f32 v20, v10, v6 // (op 15984) 
v_mul_f32 v21, v10, v7 // (op 15984) 


v_add_f32 v6, v18, v14 // (op 15985) 
v_add_f32 v7, v19, v15 // (op 15985) 
v_add_f32 v8, v20, v16 // (op 15985) 
v_add_f32 v9, v21, v17 // (op 15985) 


v_lshrrev_b32 v14, 16, v6 // (op 15986) 
v_lshrrev_b32 v15, 16, v7 // (op 15986) 
v_lshrrev_b32 v16, 16, v8 // (op 15986) 
v_lshrrev_b32 v17, 16, v9 // (op 15986) 


v_pack_b32_f16 v8, v14, v15 // (op 15952) 
v_pack_b32_f16 v9, v16, v17 // (op 15952) 

s_lshl_b32 s0, s4, 6 // (op 15952) 
s_mul_i32 s1, 6, s0 // (op 15952) 
s_add_u32 s1, s1, 320 // (op 15952) 

v_add_u32 v0, v13, s1 // (op 15952) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15952) Store value





s_lshl_b32 s0, s2, 6 // (op 15992) 
s_mul_i32 s1, 7, s0 // (op 15992) 
s_add_u32 s1, s1, 320 // (op 15992) 

v_add_u32 v0, v12, s1 // (op 15992) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15992) Load value

v_accvgpr_read v4, a188 // (op 16006) 
v_accvgpr_read v5, a189 // (op 16006) 
v_accvgpr_read v6, a190 // (op 16006) 
v_accvgpr_read v7, a191 // (op 16006) 
v_mul_f32 v14, v11, v4 // (op 16006) 
v_mul_f32 v15, v11, v5 // (op 16006) 
v_mul_f32 v16, v11, v6 // (op 16006) 
v_mul_f32 v17, v11, v7 // (op 16006) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15989) 
v_lshrrev_b32 v7, 16, v8 // (op 15989) 
v_lshlrev_b32 v6, 16, v6 // (op 15989) 
v_lshlrev_b32 v7, 16, v7 // (op 15989) 
v_mul_f32 v18, v10, v6 // (op 15989) 
v_mul_f32 v19, v10, v7 // (op 15989) 
v_and_b32 v6, 65535, v9 // (op 15989) 
v_lshrrev_b32 v7, 16, v9 // (op 15989) 
v_lshlrev_b32 v6, 16, v6 // (op 15989) 
v_lshlrev_b32 v7, 16, v7 // (op 15989) 
v_mul_f32 v20, v10, v6 // (op 15989) 
v_mul_f32 v21, v10, v7 // (op 15989) 


v_add_f32 v6, v18, v14 // (op 15990) 
v_add_f32 v7, v19, v15 // (op 15990) 
v_add_f32 v8, v20, v16 // (op 15990) 
v_add_f32 v9, v21, v17 // (op 15990) 


v_lshrrev_b32 v14, 16, v6 // (op 15991) 
v_lshrrev_b32 v15, 16, v7 // (op 15991) 
v_lshrrev_b32 v16, 16, v8 // (op 15991) 
v_lshrrev_b32 v17, 16, v9 // (op 15991) 


v_pack_b32_f16 v8, v14, v15 // (op 15954) 
v_pack_b32_f16 v9, v16, v17 // (op 15954) 

s_lshl_b32 s0, s4, 6 // (op 15954) 
s_mul_i32 s1, 7, s0 // (op 15954) 
s_add_u32 s1, s1, 320 // (op 15954) 

v_add_u32 v0, v13, s1 // (op 15954) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15954) Store value





v_add_u32 v0, 256, v12 // (op 15786) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15786) Load value

v_accvgpr_read v4, a128 // (op 15765) 
v_accvgpr_read v5, a129 // (op 15765) 
v_accvgpr_read v6, a130 // (op 15765) 
v_accvgpr_read v7, a131 // (op 15765) 
v_mul_f32 v14, v11, v4 // (op 15765) 
v_mul_f32 v15, v11, v5 // (op 15765) 
v_mul_f32 v16, v11, v6 // (op 15765) 
v_mul_f32 v17, v11, v7 // (op 15765) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15785) 
v_lshrrev_b32 v7, 16, v8 // (op 15785) 
v_lshlrev_b32 v6, 16, v6 // (op 15785) 
v_lshlrev_b32 v7, 16, v7 // (op 15785) 
v_mul_f32 v18, v10, v6 // (op 15785) 
v_mul_f32 v19, v10, v7 // (op 15785) 
v_and_b32 v6, 65535, v9 // (op 15785) 
v_lshrrev_b32 v7, 16, v9 // (op 15785) 
v_lshlrev_b32 v6, 16, v6 // (op 15785) 
v_lshlrev_b32 v7, 16, v7 // (op 15785) 
v_mul_f32 v20, v10, v6 // (op 15785) 
v_mul_f32 v21, v10, v7 // (op 15785) 


v_add_f32 v6, v18, v14 // (op 15766) 
v_add_f32 v7, v19, v15 // (op 15766) 
v_add_f32 v8, v20, v16 // (op 15766) 
v_add_f32 v9, v21, v17 // (op 15766) 


v_lshrrev_b32 v14, 16, v6 // (op 15767) 
v_lshrrev_b32 v15, 16, v7 // (op 15767) 
v_lshrrev_b32 v16, 16, v8 // (op 15767) 
v_lshrrev_b32 v17, 16, v9 // (op 15767) 


v_pack_b32_f16 v8, v14, v15 // (op 15769) 
v_pack_b32_f16 v9, v16, v17 // (op 15769) 

v_add_u32 v0, 256, v13 // (op 15769) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15769) Store value





s_lshl_b32 s0, s2, 6 // (op 15791) 
s_add_u32 s0, s0, 256 // (op 15791) 
v_add_u32 v0, v12, s0 // (op 15791) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15791) Load value

v_accvgpr_read v4, a132 // (op 15823) 
v_accvgpr_read v5, a133 // (op 15823) 
v_accvgpr_read v6, a134 // (op 15823) 
v_accvgpr_read v7, a135 // (op 15823) 
v_mul_f32 v14, v11, v4 // (op 15823) 
v_mul_f32 v15, v11, v5 // (op 15823) 
v_mul_f32 v16, v11, v6 // (op 15823) 
v_mul_f32 v17, v11, v7 // (op 15823) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15788) 
v_lshrrev_b32 v7, 16, v8 // (op 15788) 
v_lshlrev_b32 v6, 16, v6 // (op 15788) 
v_lshlrev_b32 v7, 16, v7 // (op 15788) 
v_mul_f32 v18, v10, v6 // (op 15788) 
v_mul_f32 v19, v10, v7 // (op 15788) 
v_and_b32 v6, 65535, v9 // (op 15788) 
v_lshrrev_b32 v7, 16, v9 // (op 15788) 
v_lshlrev_b32 v6, 16, v6 // (op 15788) 
v_lshlrev_b32 v7, 16, v7 // (op 15788) 
v_mul_f32 v20, v10, v6 // (op 15788) 
v_mul_f32 v21, v10, v7 // (op 15788) 


v_add_f32 v6, v18, v14 // (op 15789) 
v_add_f32 v7, v19, v15 // (op 15789) 
v_add_f32 v8, v20, v16 // (op 15789) 
v_add_f32 v9, v21, v17 // (op 15789) 


v_lshrrev_b32 v14, 16, v6 // (op 15790) 
v_lshrrev_b32 v15, 16, v7 // (op 15790) 
v_lshrrev_b32 v16, 16, v8 // (op 15790) 
v_lshrrev_b32 v17, 16, v9 // (op 15790) 


v_pack_b32_f16 v8, v14, v15 // (op 15771) 
v_pack_b32_f16 v9, v16, v17 // (op 15771) 

s_lshl_b32 s0, s4, 6 // (op 15771) 
s_add_u32 s0, s0, 256 // (op 15771) 
v_add_u32 v0, v13, s0 // (op 15771) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15771) Store value





s_lshl_b32 s0, s2, 7 // (op 15796) 
s_add_u32 s0, s0, 256 // (op 15796) 
v_add_u32 v0, v12, s0 // (op 15796) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15796) Load value

v_accvgpr_read v4, a136 // (op 15825) 
v_accvgpr_read v5, a137 // (op 15825) 
v_accvgpr_read v6, a138 // (op 15825) 
v_accvgpr_read v7, a139 // (op 15825) 
v_mul_f32 v14, v11, v4 // (op 15825) 
v_mul_f32 v15, v11, v5 // (op 15825) 
v_mul_f32 v16, v11, v6 // (op 15825) 
v_mul_f32 v17, v11, v7 // (op 15825) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15793) 
v_lshrrev_b32 v7, 16, v8 // (op 15793) 
v_lshlrev_b32 v6, 16, v6 // (op 15793) 
v_lshlrev_b32 v7, 16, v7 // (op 15793) 
v_mul_f32 v18, v10, v6 // (op 15793) 
v_mul_f32 v19, v10, v7 // (op 15793) 
v_and_b32 v6, 65535, v9 // (op 15793) 
v_lshrrev_b32 v7, 16, v9 // (op 15793) 
v_lshlrev_b32 v6, 16, v6 // (op 15793) 
v_lshlrev_b32 v7, 16, v7 // (op 15793) 
v_mul_f32 v20, v10, v6 // (op 15793) 
v_mul_f32 v21, v10, v7 // (op 15793) 


v_add_f32 v6, v18, v14 // (op 15794) 
v_add_f32 v7, v19, v15 // (op 15794) 
v_add_f32 v8, v20, v16 // (op 15794) 
v_add_f32 v9, v21, v17 // (op 15794) 


v_lshrrev_b32 v14, 16, v6 // (op 15795) 
v_lshrrev_b32 v15, 16, v7 // (op 15795) 
v_lshrrev_b32 v16, 16, v8 // (op 15795) 
v_lshrrev_b32 v17, 16, v9 // (op 15795) 


v_pack_b32_f16 v8, v14, v15 // (op 15773) 
v_pack_b32_f16 v9, v16, v17 // (op 15773) 

s_lshl_b32 s0, s4, 7 // (op 15773) 
s_add_u32 s0, s0, 256 // (op 15773) 
v_add_u32 v0, v13, s0 // (op 15773) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15773) Store value





s_lshl_b32 s0, s2, 6 // (op 15801) 
s_mul_i32 s1, 3, s0 // (op 15801) 
s_add_u32 s1, s1, 256 // (op 15801) 

v_add_u32 v0, v12, s1 // (op 15801) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15801) Load value

v_accvgpr_read v4, a140 // (op 15827) 
v_accvgpr_read v5, a141 // (op 15827) 
v_accvgpr_read v6, a142 // (op 15827) 
v_accvgpr_read v7, a143 // (op 15827) 
v_mul_f32 v14, v11, v4 // (op 15827) 
v_mul_f32 v15, v11, v5 // (op 15827) 
v_mul_f32 v16, v11, v6 // (op 15827) 
v_mul_f32 v17, v11, v7 // (op 15827) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15798) 
v_lshrrev_b32 v7, 16, v8 // (op 15798) 
v_lshlrev_b32 v6, 16, v6 // (op 15798) 
v_lshlrev_b32 v7, 16, v7 // (op 15798) 
v_mul_f32 v18, v10, v6 // (op 15798) 
v_mul_f32 v19, v10, v7 // (op 15798) 
v_and_b32 v6, 65535, v9 // (op 15798) 
v_lshrrev_b32 v7, 16, v9 // (op 15798) 
v_lshlrev_b32 v6, 16, v6 // (op 15798) 
v_lshlrev_b32 v7, 16, v7 // (op 15798) 
v_mul_f32 v20, v10, v6 // (op 15798) 
v_mul_f32 v21, v10, v7 // (op 15798) 


v_add_f32 v6, v18, v14 // (op 15799) 
v_add_f32 v7, v19, v15 // (op 15799) 
v_add_f32 v8, v20, v16 // (op 15799) 
v_add_f32 v9, v21, v17 // (op 15799) 


v_lshrrev_b32 v14, 16, v6 // (op 15800) 
v_lshrrev_b32 v15, 16, v7 // (op 15800) 
v_lshrrev_b32 v16, 16, v8 // (op 15800) 
v_lshrrev_b32 v17, 16, v9 // (op 15800) 


v_pack_b32_f16 v8, v14, v15 // (op 15775) 
v_pack_b32_f16 v9, v16, v17 // (op 15775) 

s_lshl_b32 s0, s4, 6 // (op 15775) 
s_mul_i32 s1, 3, s0 // (op 15775) 
s_add_u32 s1, s1, 256 // (op 15775) 

v_add_u32 v0, v13, s1 // (op 15775) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15775) Store value





s_lshl_b32 s0, s2, 8 // (op 15806) 
s_add_u32 s0, s0, 256 // (op 15806) 
v_add_u32 v0, v12, s0 // (op 15806) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15806) Load value

v_accvgpr_read v4, a144 // (op 15829) 
v_accvgpr_read v5, a145 // (op 15829) 
v_accvgpr_read v6, a146 // (op 15829) 
v_accvgpr_read v7, a147 // (op 15829) 
v_mul_f32 v14, v11, v4 // (op 15829) 
v_mul_f32 v15, v11, v5 // (op 15829) 
v_mul_f32 v16, v11, v6 // (op 15829) 
v_mul_f32 v17, v11, v7 // (op 15829) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15803) 
v_lshrrev_b32 v7, 16, v8 // (op 15803) 
v_lshlrev_b32 v6, 16, v6 // (op 15803) 
v_lshlrev_b32 v7, 16, v7 // (op 15803) 
v_mul_f32 v18, v10, v6 // (op 15803) 
v_mul_f32 v19, v10, v7 // (op 15803) 
v_and_b32 v6, 65535, v9 // (op 15803) 
v_lshrrev_b32 v7, 16, v9 // (op 15803) 
v_lshlrev_b32 v6, 16, v6 // (op 15803) 
v_lshlrev_b32 v7, 16, v7 // (op 15803) 
v_mul_f32 v20, v10, v6 // (op 15803) 
v_mul_f32 v21, v10, v7 // (op 15803) 


v_add_f32 v6, v18, v14 // (op 15804) 
v_add_f32 v7, v19, v15 // (op 15804) 
v_add_f32 v8, v20, v16 // (op 15804) 
v_add_f32 v9, v21, v17 // (op 15804) 


v_lshrrev_b32 v14, 16, v6 // (op 15805) 
v_lshrrev_b32 v15, 16, v7 // (op 15805) 
v_lshrrev_b32 v16, 16, v8 // (op 15805) 
v_lshrrev_b32 v17, 16, v9 // (op 15805) 


v_pack_b32_f16 v8, v14, v15 // (op 15777) 
v_pack_b32_f16 v9, v16, v17 // (op 15777) 

s_lshl_b32 s0, s4, 8 // (op 15777) 
s_add_u32 s0, s0, 256 // (op 15777) 
v_add_u32 v0, v13, s0 // (op 15777) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15777) Store value





s_lshl_b32 s0, s2, 6 // (op 15811) 
s_mul_i32 s1, 5, s0 // (op 15811) 
s_add_u32 s1, s1, 256 // (op 15811) 

v_add_u32 v0, v12, s1 // (op 15811) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15811) Load value

v_accvgpr_read v4, a148 // (op 15831) 
v_accvgpr_read v5, a149 // (op 15831) 
v_accvgpr_read v6, a150 // (op 15831) 
v_accvgpr_read v7, a151 // (op 15831) 
v_mul_f32 v14, v11, v4 // (op 15831) 
v_mul_f32 v15, v11, v5 // (op 15831) 
v_mul_f32 v16, v11, v6 // (op 15831) 
v_mul_f32 v17, v11, v7 // (op 15831) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15808) 
v_lshrrev_b32 v7, 16, v8 // (op 15808) 
v_lshlrev_b32 v6, 16, v6 // (op 15808) 
v_lshlrev_b32 v7, 16, v7 // (op 15808) 
v_mul_f32 v18, v10, v6 // (op 15808) 
v_mul_f32 v19, v10, v7 // (op 15808) 
v_and_b32 v6, 65535, v9 // (op 15808) 
v_lshrrev_b32 v7, 16, v9 // (op 15808) 
v_lshlrev_b32 v6, 16, v6 // (op 15808) 
v_lshlrev_b32 v7, 16, v7 // (op 15808) 
v_mul_f32 v20, v10, v6 // (op 15808) 
v_mul_f32 v21, v10, v7 // (op 15808) 


v_add_f32 v6, v18, v14 // (op 15809) 
v_add_f32 v7, v19, v15 // (op 15809) 
v_add_f32 v8, v20, v16 // (op 15809) 
v_add_f32 v9, v21, v17 // (op 15809) 


v_lshrrev_b32 v14, 16, v6 // (op 15810) 
v_lshrrev_b32 v15, 16, v7 // (op 15810) 
v_lshrrev_b32 v16, 16, v8 // (op 15810) 
v_lshrrev_b32 v17, 16, v9 // (op 15810) 


v_pack_b32_f16 v8, v14, v15 // (op 15779) 
v_pack_b32_f16 v9, v16, v17 // (op 15779) 

s_lshl_b32 s0, s4, 6 // (op 15779) 
s_mul_i32 s1, 5, s0 // (op 15779) 
s_add_u32 s1, s1, 256 // (op 15779) 

v_add_u32 v0, v13, s1 // (op 15779) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15779) Store value





s_lshl_b32 s0, s2, 6 // (op 15816) 
s_mul_i32 s1, 6, s0 // (op 15816) 
s_add_u32 s1, s1, 256 // (op 15816) 

v_add_u32 v0, v12, s1 // (op 15816) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15816) Load value

v_accvgpr_read v4, a152 // (op 15833) 
v_accvgpr_read v5, a153 // (op 15833) 
v_accvgpr_read v6, a154 // (op 15833) 
v_accvgpr_read v7, a155 // (op 15833) 
v_mul_f32 v14, v11, v4 // (op 15833) 
v_mul_f32 v15, v11, v5 // (op 15833) 
v_mul_f32 v16, v11, v6 // (op 15833) 
v_mul_f32 v17, v11, v7 // (op 15833) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15813) 
v_lshrrev_b32 v7, 16, v8 // (op 15813) 
v_lshlrev_b32 v6, 16, v6 // (op 15813) 
v_lshlrev_b32 v7, 16, v7 // (op 15813) 
v_mul_f32 v18, v10, v6 // (op 15813) 
v_mul_f32 v19, v10, v7 // (op 15813) 
v_and_b32 v6, 65535, v9 // (op 15813) 
v_lshrrev_b32 v7, 16, v9 // (op 15813) 
v_lshlrev_b32 v6, 16, v6 // (op 15813) 
v_lshlrev_b32 v7, 16, v7 // (op 15813) 
v_mul_f32 v20, v10, v6 // (op 15813) 
v_mul_f32 v21, v10, v7 // (op 15813) 


v_add_f32 v6, v18, v14 // (op 15814) 
v_add_f32 v7, v19, v15 // (op 15814) 
v_add_f32 v8, v20, v16 // (op 15814) 
v_add_f32 v9, v21, v17 // (op 15814) 


v_lshrrev_b32 v14, 16, v6 // (op 15815) 
v_lshrrev_b32 v15, 16, v7 // (op 15815) 
v_lshrrev_b32 v16, 16, v8 // (op 15815) 
v_lshrrev_b32 v17, 16, v9 // (op 15815) 


v_pack_b32_f16 v8, v14, v15 // (op 15781) 
v_pack_b32_f16 v9, v16, v17 // (op 15781) 

s_lshl_b32 s0, s4, 6 // (op 15781) 
s_mul_i32 s1, 6, s0 // (op 15781) 
s_add_u32 s1, s1, 256 // (op 15781) 

v_add_u32 v0, v13, s1 // (op 15781) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15781) Store value





s_lshl_b32 s0, s2, 6 // (op 15821) 
s_mul_i32 s1, 7, s0 // (op 15821) 
s_add_u32 s1, s1, 256 // (op 15821) 

v_add_u32 v0, v12, s1 // (op 15821) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15821) Load value

v_accvgpr_read v4, a156 // (op 15835) 
v_accvgpr_read v5, a157 // (op 15835) 
v_accvgpr_read v6, a158 // (op 15835) 
v_accvgpr_read v7, a159 // (op 15835) 
v_mul_f32 v14, v11, v4 // (op 15835) 
v_mul_f32 v15, v11, v5 // (op 15835) 
v_mul_f32 v16, v11, v6 // (op 15835) 
v_mul_f32 v17, v11, v7 // (op 15835) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15818) 
v_lshrrev_b32 v7, 16, v8 // (op 15818) 
v_lshlrev_b32 v6, 16, v6 // (op 15818) 
v_lshlrev_b32 v7, 16, v7 // (op 15818) 
v_mul_f32 v18, v10, v6 // (op 15818) 
v_mul_f32 v19, v10, v7 // (op 15818) 
v_and_b32 v6, 65535, v9 // (op 15818) 
v_lshrrev_b32 v7, 16, v9 // (op 15818) 
v_lshlrev_b32 v6, 16, v6 // (op 15818) 
v_lshlrev_b32 v7, 16, v7 // (op 15818) 
v_mul_f32 v20, v10, v6 // (op 15818) 
v_mul_f32 v21, v10, v7 // (op 15818) 


v_add_f32 v6, v18, v14 // (op 15819) 
v_add_f32 v7, v19, v15 // (op 15819) 
v_add_f32 v8, v20, v16 // (op 15819) 
v_add_f32 v9, v21, v17 // (op 15819) 


v_lshrrev_b32 v14, 16, v6 // (op 15820) 
v_lshrrev_b32 v15, 16, v7 // (op 15820) 
v_lshrrev_b32 v16, 16, v8 // (op 15820) 
v_lshrrev_b32 v17, 16, v9 // (op 15820) 


v_pack_b32_f16 v8, v14, v15 // (op 15783) 
v_pack_b32_f16 v9, v16, v17 // (op 15783) 

s_lshl_b32 s0, s4, 6 // (op 15783) 
s_mul_i32 s1, 7, s0 // (op 15783) 
s_add_u32 s1, s1, 256 // (op 15783) 

v_add_u32 v0, v13, s1 // (op 15783) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15783) Store value





v_add_u32 v0, 192, v12 // (op 15615) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15615) Load value

v_accvgpr_read v4, a96 // (op 15594) 
v_accvgpr_read v5, a97 // (op 15594) 
v_accvgpr_read v6, a98 // (op 15594) 
v_accvgpr_read v7, a99 // (op 15594) 
v_mul_f32 v14, v11, v4 // (op 15594) 
v_mul_f32 v15, v11, v5 // (op 15594) 
v_mul_f32 v16, v11, v6 // (op 15594) 
v_mul_f32 v17, v11, v7 // (op 15594) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15614) 
v_lshrrev_b32 v7, 16, v8 // (op 15614) 
v_lshlrev_b32 v6, 16, v6 // (op 15614) 
v_lshlrev_b32 v7, 16, v7 // (op 15614) 
v_mul_f32 v18, v10, v6 // (op 15614) 
v_mul_f32 v19, v10, v7 // (op 15614) 
v_and_b32 v6, 65535, v9 // (op 15614) 
v_lshrrev_b32 v7, 16, v9 // (op 15614) 
v_lshlrev_b32 v6, 16, v6 // (op 15614) 
v_lshlrev_b32 v7, 16, v7 // (op 15614) 
v_mul_f32 v20, v10, v6 // (op 15614) 
v_mul_f32 v21, v10, v7 // (op 15614) 


v_add_f32 v6, v18, v14 // (op 15595) 
v_add_f32 v7, v19, v15 // (op 15595) 
v_add_f32 v8, v20, v16 // (op 15595) 
v_add_f32 v9, v21, v17 // (op 15595) 


v_lshrrev_b32 v14, 16, v6 // (op 15596) 
v_lshrrev_b32 v15, 16, v7 // (op 15596) 
v_lshrrev_b32 v16, 16, v8 // (op 15596) 
v_lshrrev_b32 v17, 16, v9 // (op 15596) 


v_pack_b32_f16 v8, v14, v15 // (op 15598) 
v_pack_b32_f16 v9, v16, v17 // (op 15598) 

v_add_u32 v0, 192, v13 // (op 15598) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15598) Store value





s_lshl_b32 s0, s2, 6 // (op 15620) 
s_add_u32 s0, s0, 192 // (op 15620) 
v_add_u32 v0, v12, s0 // (op 15620) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15620) Load value

v_accvgpr_read v4, a100 // (op 15652) 
v_accvgpr_read v5, a101 // (op 15652) 
v_accvgpr_read v6, a102 // (op 15652) 
v_accvgpr_read v7, a103 // (op 15652) 
v_mul_f32 v14, v11, v4 // (op 15652) 
v_mul_f32 v15, v11, v5 // (op 15652) 
v_mul_f32 v16, v11, v6 // (op 15652) 
v_mul_f32 v17, v11, v7 // (op 15652) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15617) 
v_lshrrev_b32 v7, 16, v8 // (op 15617) 
v_lshlrev_b32 v6, 16, v6 // (op 15617) 
v_lshlrev_b32 v7, 16, v7 // (op 15617) 
v_mul_f32 v18, v10, v6 // (op 15617) 
v_mul_f32 v19, v10, v7 // (op 15617) 
v_and_b32 v6, 65535, v9 // (op 15617) 
v_lshrrev_b32 v7, 16, v9 // (op 15617) 
v_lshlrev_b32 v6, 16, v6 // (op 15617) 
v_lshlrev_b32 v7, 16, v7 // (op 15617) 
v_mul_f32 v20, v10, v6 // (op 15617) 
v_mul_f32 v21, v10, v7 // (op 15617) 


v_add_f32 v6, v18, v14 // (op 15618) 
v_add_f32 v7, v19, v15 // (op 15618) 
v_add_f32 v8, v20, v16 // (op 15618) 
v_add_f32 v9, v21, v17 // (op 15618) 


v_lshrrev_b32 v14, 16, v6 // (op 15619) 
v_lshrrev_b32 v15, 16, v7 // (op 15619) 
v_lshrrev_b32 v16, 16, v8 // (op 15619) 
v_lshrrev_b32 v17, 16, v9 // (op 15619) 


v_pack_b32_f16 v8, v14, v15 // (op 15600) 
v_pack_b32_f16 v9, v16, v17 // (op 15600) 

s_lshl_b32 s0, s4, 6 // (op 15600) 
s_add_u32 s0, s0, 192 // (op 15600) 
v_add_u32 v0, v13, s0 // (op 15600) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15600) Store value





s_lshl_b32 s0, s2, 7 // (op 15625) 
s_add_u32 s0, s0, 192 // (op 15625) 
v_add_u32 v0, v12, s0 // (op 15625) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15625) Load value

v_accvgpr_read v4, a104 // (op 15654) 
v_accvgpr_read v5, a105 // (op 15654) 
v_accvgpr_read v6, a106 // (op 15654) 
v_accvgpr_read v7, a107 // (op 15654) 
v_mul_f32 v14, v11, v4 // (op 15654) 
v_mul_f32 v15, v11, v5 // (op 15654) 
v_mul_f32 v16, v11, v6 // (op 15654) 
v_mul_f32 v17, v11, v7 // (op 15654) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15622) 
v_lshrrev_b32 v7, 16, v8 // (op 15622) 
v_lshlrev_b32 v6, 16, v6 // (op 15622) 
v_lshlrev_b32 v7, 16, v7 // (op 15622) 
v_mul_f32 v18, v10, v6 // (op 15622) 
v_mul_f32 v19, v10, v7 // (op 15622) 
v_and_b32 v6, 65535, v9 // (op 15622) 
v_lshrrev_b32 v7, 16, v9 // (op 15622) 
v_lshlrev_b32 v6, 16, v6 // (op 15622) 
v_lshlrev_b32 v7, 16, v7 // (op 15622) 
v_mul_f32 v20, v10, v6 // (op 15622) 
v_mul_f32 v21, v10, v7 // (op 15622) 


v_add_f32 v6, v18, v14 // (op 15623) 
v_add_f32 v7, v19, v15 // (op 15623) 
v_add_f32 v8, v20, v16 // (op 15623) 
v_add_f32 v9, v21, v17 // (op 15623) 


v_lshrrev_b32 v14, 16, v6 // (op 15624) 
v_lshrrev_b32 v15, 16, v7 // (op 15624) 
v_lshrrev_b32 v16, 16, v8 // (op 15624) 
v_lshrrev_b32 v17, 16, v9 // (op 15624) 


v_pack_b32_f16 v8, v14, v15 // (op 15602) 
v_pack_b32_f16 v9, v16, v17 // (op 15602) 

s_lshl_b32 s0, s4, 7 // (op 15602) 
s_add_u32 s0, s0, 192 // (op 15602) 
v_add_u32 v0, v13, s0 // (op 15602) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15602) Store value





s_lshl_b32 s0, s2, 6 // (op 15630) 
s_mul_i32 s1, 3, s0 // (op 15630) 
s_add_u32 s1, s1, 192 // (op 15630) 

v_add_u32 v0, v12, s1 // (op 15630) 


v_accvgpr_read v6, a108 // (op 15656) 
v_accvgpr_read v7, a109 // (op 15656) 
v_accvgpr_read v8, a110 // (op 15656) 
v_accvgpr_read v9, a111 // (op 15656) 
v_mul_f32 v14, v11, v6 // (op 15656) 
v_mul_f32 v15, v11, v7 // (op 15656) 
v_mul_f32 v16, v11, v8 // (op 15656) 
v_mul_f32 v17, v11, v9 // (op 15656) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15630) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15627) 
v_lshrrev_b32 v7, 16, v8 // (op 15627) 
v_lshlrev_b32 v6, 16, v6 // (op 15627) 
v_lshlrev_b32 v7, 16, v7 // (op 15627) 
v_mul_f32 v18, v10, v6 // (op 15627) 
v_mul_f32 v19, v10, v7 // (op 15627) 
v_and_b32 v6, 65535, v9 // (op 15627) 
v_lshrrev_b32 v7, 16, v9 // (op 15627) 
v_lshlrev_b32 v6, 16, v6 // (op 15627) 
v_lshlrev_b32 v7, 16, v7 // (op 15627) 
v_mul_f32 v20, v10, v6 // (op 15627) 
v_mul_f32 v21, v10, v7 // (op 15627) 


v_add_f32 v6, v18, v14 // (op 15628) 
v_add_f32 v7, v19, v15 // (op 15628) 
v_add_f32 v8, v20, v16 // (op 15628) 
v_add_f32 v9, v21, v17 // (op 15628) 


v_lshrrev_b32 v14, 16, v6 // (op 15629) 
v_lshrrev_b32 v15, 16, v7 // (op 15629) 
v_lshrrev_b32 v16, 16, v8 // (op 15629) 
v_lshrrev_b32 v17, 16, v9 // (op 15629) 


v_pack_b32_f16 v8, v14, v15 // (op 15604) 
v_pack_b32_f16 v9, v16, v17 // (op 15604) 

s_lshl_b32 s0, s4, 6 // (op 15604) 
s_mul_i32 s1, 3, s0 // (op 15604) 
s_add_u32 s1, s1, 192 // (op 15604) 

v_add_u32 v0, v13, s1 // (op 15604) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15604) Store value





s_lshl_b32 s0, s2, 8 // (op 15635) 
s_add_u32 s0, s0, 192 // (op 15635) 
v_add_u32 v0, v12, s0 // (op 15635) 


v_accvgpr_read v6, a112 // (op 15658) 
v_accvgpr_read v7, a113 // (op 15658) 
v_accvgpr_read v8, a114 // (op 15658) 
v_accvgpr_read v9, a115 // (op 15658) 
v_mul_f32 v14, v11, v6 // (op 15658) 
v_mul_f32 v15, v11, v7 // (op 15658) 
v_mul_f32 v16, v11, v8 // (op 15658) 
v_mul_f32 v17, v11, v9 // (op 15658) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15635) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15632) 
v_lshrrev_b32 v7, 16, v8 // (op 15632) 
v_lshlrev_b32 v6, 16, v6 // (op 15632) 
v_lshlrev_b32 v7, 16, v7 // (op 15632) 
v_mul_f32 v18, v10, v6 // (op 15632) 
v_mul_f32 v19, v10, v7 // (op 15632) 
v_and_b32 v6, 65535, v9 // (op 15632) 
v_lshrrev_b32 v7, 16, v9 // (op 15632) 
v_lshlrev_b32 v6, 16, v6 // (op 15632) 
v_lshlrev_b32 v7, 16, v7 // (op 15632) 
v_mul_f32 v20, v10, v6 // (op 15632) 
v_mul_f32 v21, v10, v7 // (op 15632) 


v_add_f32 v6, v18, v14 // (op 15633) 
v_add_f32 v7, v19, v15 // (op 15633) 
v_add_f32 v8, v20, v16 // (op 15633) 
v_add_f32 v9, v21, v17 // (op 15633) 


v_lshrrev_b32 v14, 16, v6 // (op 15634) 
v_lshrrev_b32 v15, 16, v7 // (op 15634) 
v_lshrrev_b32 v16, 16, v8 // (op 15634) 
v_lshrrev_b32 v17, 16, v9 // (op 15634) 


v_pack_b32_f16 v8, v14, v15 // (op 15606) 
v_pack_b32_f16 v9, v16, v17 // (op 15606) 

s_lshl_b32 s0, s4, 8 // (op 15606) 
s_add_u32 s0, s0, 192 // (op 15606) 
v_add_u32 v0, v13, s0 // (op 15606) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15606) Store value





s_lshl_b32 s0, s2, 6 // (op 15640) 
s_mul_i32 s1, 5, s0 // (op 15640) 
s_add_u32 s1, s1, 192 // (op 15640) 

v_add_u32 v0, v12, s1 // (op 15640) 


v_accvgpr_read v6, a116 // (op 15660) 
v_accvgpr_read v7, a117 // (op 15660) 
v_accvgpr_read v8, a118 // (op 15660) 
v_accvgpr_read v9, a119 // (op 15660) 
v_mul_f32 v14, v11, v6 // (op 15660) 
v_mul_f32 v15, v11, v7 // (op 15660) 
v_mul_f32 v16, v11, v8 // (op 15660) 
v_mul_f32 v17, v11, v9 // (op 15660) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15640) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15637) 
v_lshrrev_b32 v7, 16, v8 // (op 15637) 
v_lshlrev_b32 v6, 16, v6 // (op 15637) 
v_lshlrev_b32 v7, 16, v7 // (op 15637) 
v_mul_f32 v18, v10, v6 // (op 15637) 
v_mul_f32 v19, v10, v7 // (op 15637) 
v_and_b32 v6, 65535, v9 // (op 15637) 
v_lshrrev_b32 v7, 16, v9 // (op 15637) 
v_lshlrev_b32 v6, 16, v6 // (op 15637) 
v_lshlrev_b32 v7, 16, v7 // (op 15637) 
v_mul_f32 v20, v10, v6 // (op 15637) 
v_mul_f32 v21, v10, v7 // (op 15637) 


v_add_f32 v6, v18, v14 // (op 15638) 
v_add_f32 v7, v19, v15 // (op 15638) 
v_add_f32 v8, v20, v16 // (op 15638) 
v_add_f32 v9, v21, v17 // (op 15638) 


v_lshrrev_b32 v14, 16, v6 // (op 15639) 
v_lshrrev_b32 v15, 16, v7 // (op 15639) 
v_lshrrev_b32 v16, 16, v8 // (op 15639) 
v_lshrrev_b32 v17, 16, v9 // (op 15639) 


v_pack_b32_f16 v8, v14, v15 // (op 15608) 
v_pack_b32_f16 v9, v16, v17 // (op 15608) 

s_lshl_b32 s0, s4, 6 // (op 15608) 
s_mul_i32 s1, 5, s0 // (op 15608) 
s_add_u32 s1, s1, 192 // (op 15608) 

v_add_u32 v0, v13, s1 // (op 15608) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15608) Store value





s_lshl_b32 s0, s2, 6 // (op 15645) 
s_mul_i32 s1, 6, s0 // (op 15645) 
s_add_u32 s1, s1, 192 // (op 15645) 

v_add_u32 v0, v12, s1 // (op 15645) 


v_accvgpr_read v6, a120 // (op 15662) 
v_accvgpr_read v7, a121 // (op 15662) 
v_accvgpr_read v8, a122 // (op 15662) 
v_accvgpr_read v9, a123 // (op 15662) 
v_mul_f32 v14, v11, v6 // (op 15662) 
v_mul_f32 v15, v11, v7 // (op 15662) 
v_mul_f32 v16, v11, v8 // (op 15662) 
v_mul_f32 v17, v11, v9 // (op 15662) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15645) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15642) 
v_lshrrev_b32 v7, 16, v8 // (op 15642) 
v_lshlrev_b32 v6, 16, v6 // (op 15642) 
v_lshlrev_b32 v7, 16, v7 // (op 15642) 
v_mul_f32 v18, v10, v6 // (op 15642) 
v_mul_f32 v19, v10, v7 // (op 15642) 
v_and_b32 v6, 65535, v9 // (op 15642) 
v_lshrrev_b32 v7, 16, v9 // (op 15642) 
v_lshlrev_b32 v6, 16, v6 // (op 15642) 
v_lshlrev_b32 v7, 16, v7 // (op 15642) 
v_mul_f32 v20, v10, v6 // (op 15642) 
v_mul_f32 v21, v10, v7 // (op 15642) 


v_add_f32 v6, v18, v14 // (op 15643) 
v_add_f32 v7, v19, v15 // (op 15643) 
v_add_f32 v8, v20, v16 // (op 15643) 
v_add_f32 v9, v21, v17 // (op 15643) 


v_lshrrev_b32 v14, 16, v6 // (op 15644) 
v_lshrrev_b32 v15, 16, v7 // (op 15644) 
v_lshrrev_b32 v16, 16, v8 // (op 15644) 
v_lshrrev_b32 v17, 16, v9 // (op 15644) 


v_pack_b32_f16 v8, v14, v15 // (op 15610) 
v_pack_b32_f16 v9, v16, v17 // (op 15610) 

s_lshl_b32 s0, s4, 6 // (op 15610) 
s_mul_i32 s1, 6, s0 // (op 15610) 
s_add_u32 s1, s1, 192 // (op 15610) 

v_add_u32 v0, v13, s1 // (op 15610) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15610) Store value





s_lshl_b32 s0, s2, 6 // (op 15650) 
s_mul_i32 s1, 7, s0 // (op 15650) 
s_add_u32 s1, s1, 192 // (op 15650) 

v_add_u32 v0, v12, s1 // (op 15650) 


v_accvgpr_read v6, a124 // (op 15664) 
v_accvgpr_read v7, a125 // (op 15664) 
v_accvgpr_read v8, a126 // (op 15664) 
v_accvgpr_read v9, a127 // (op 15664) 
v_mul_f32 v14, v11, v6 // (op 15664) 
v_mul_f32 v15, v11, v7 // (op 15664) 
v_mul_f32 v16, v11, v8 // (op 15664) 
v_mul_f32 v17, v11, v9 // (op 15664) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15650) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15647) 
v_lshrrev_b32 v7, 16, v8 // (op 15647) 
v_lshlrev_b32 v6, 16, v6 // (op 15647) 
v_lshlrev_b32 v7, 16, v7 // (op 15647) 
v_mul_f32 v18, v10, v6 // (op 15647) 
v_mul_f32 v19, v10, v7 // (op 15647) 
v_and_b32 v6, 65535, v9 // (op 15647) 
v_lshrrev_b32 v7, 16, v9 // (op 15647) 
v_lshlrev_b32 v6, 16, v6 // (op 15647) 
v_lshlrev_b32 v7, 16, v7 // (op 15647) 
v_mul_f32 v20, v10, v6 // (op 15647) 
v_mul_f32 v21, v10, v7 // (op 15647) 


v_add_f32 v6, v18, v14 // (op 15648) 
v_add_f32 v7, v19, v15 // (op 15648) 
v_add_f32 v8, v20, v16 // (op 15648) 
v_add_f32 v9, v21, v17 // (op 15648) 


v_lshrrev_b32 v14, 16, v6 // (op 15649) 
v_lshrrev_b32 v15, 16, v7 // (op 15649) 
v_lshrrev_b32 v16, 16, v8 // (op 15649) 
v_lshrrev_b32 v17, 16, v9 // (op 15649) 


v_pack_b32_f16 v8, v14, v15 // (op 15612) 
v_pack_b32_f16 v9, v16, v17 // (op 15612) 

s_lshl_b32 s0, s4, 6 // (op 15612) 
s_mul_i32 s1, 7, s0 // (op 15612) 
s_add_u32 s1, s1, 192 // (op 15612) 

v_add_u32 v0, v13, s1 // (op 15612) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15612) Store value





v_add_u32 v0, 128, v12 // (op 15444) 

v_accvgpr_read v6, a64 // (op 15423) 
v_accvgpr_read v7, a65 // (op 15423) 
v_accvgpr_read v8, a66 // (op 15423) 
v_accvgpr_read v9, a67 // (op 15423) 
v_mul_f32 v14, v11, v6 // (op 15423) 
v_mul_f32 v15, v11, v7 // (op 15423) 
v_mul_f32 v16, v11, v8 // (op 15423) 
v_mul_f32 v17, v11, v9 // (op 15423) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15444) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15443) 
v_lshrrev_b32 v7, 16, v8 // (op 15443) 
v_lshlrev_b32 v6, 16, v6 // (op 15443) 
v_lshlrev_b32 v7, 16, v7 // (op 15443) 
v_mul_f32 v18, v10, v6 // (op 15443) 
v_mul_f32 v19, v10, v7 // (op 15443) 
v_and_b32 v6, 65535, v9 // (op 15443) 
v_lshrrev_b32 v7, 16, v9 // (op 15443) 
v_lshlrev_b32 v6, 16, v6 // (op 15443) 
v_lshlrev_b32 v7, 16, v7 // (op 15443) 
v_mul_f32 v20, v10, v6 // (op 15443) 
v_mul_f32 v21, v10, v7 // (op 15443) 


v_add_f32 v6, v18, v14 // (op 15424) 
v_add_f32 v7, v19, v15 // (op 15424) 
v_add_f32 v8, v20, v16 // (op 15424) 
v_add_f32 v9, v21, v17 // (op 15424) 


v_lshrrev_b32 v14, 16, v6 // (op 15425) 
v_lshrrev_b32 v15, 16, v7 // (op 15425) 
v_lshrrev_b32 v16, 16, v8 // (op 15425) 
v_lshrrev_b32 v17, 16, v9 // (op 15425) 


v_pack_b32_f16 v8, v14, v15 // (op 15427) 
v_pack_b32_f16 v9, v16, v17 // (op 15427) 

v_add_u32 v0, 128, v13 // (op 15427) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15427) Store value





s_lshl_b32 s0, s2, 6 // (op 15449) 
s_add_u32 s0, s0, 128 // (op 15449) 
v_add_u32 v0, v12, s0 // (op 15449) 


v_accvgpr_read v6, a68 // (op 15481) 
v_accvgpr_read v7, a69 // (op 15481) 
v_accvgpr_read v8, a70 // (op 15481) 
v_accvgpr_read v9, a71 // (op 15481) 
v_mul_f32 v14, v11, v6 // (op 15481) 
v_mul_f32 v15, v11, v7 // (op 15481) 
v_mul_f32 v16, v11, v8 // (op 15481) 
v_mul_f32 v17, v11, v9 // (op 15481) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15449) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15446) 
v_lshrrev_b32 v7, 16, v8 // (op 15446) 
v_lshlrev_b32 v6, 16, v6 // (op 15446) 
v_lshlrev_b32 v7, 16, v7 // (op 15446) 
v_mul_f32 v18, v10, v6 // (op 15446) 
v_mul_f32 v19, v10, v7 // (op 15446) 
v_and_b32 v6, 65535, v9 // (op 15446) 
v_lshrrev_b32 v7, 16, v9 // (op 15446) 
v_lshlrev_b32 v6, 16, v6 // (op 15446) 
v_lshlrev_b32 v7, 16, v7 // (op 15446) 
v_mul_f32 v20, v10, v6 // (op 15446) 
v_mul_f32 v21, v10, v7 // (op 15446) 


v_add_f32 v6, v18, v14 // (op 15447) 
v_add_f32 v7, v19, v15 // (op 15447) 
v_add_f32 v8, v20, v16 // (op 15447) 
v_add_f32 v9, v21, v17 // (op 15447) 


v_lshrrev_b32 v14, 16, v6 // (op 15448) 
v_lshrrev_b32 v15, 16, v7 // (op 15448) 
v_lshrrev_b32 v16, 16, v8 // (op 15448) 
v_lshrrev_b32 v17, 16, v9 // (op 15448) 


v_pack_b32_f16 v8, v14, v15 // (op 15429) 
v_pack_b32_f16 v9, v16, v17 // (op 15429) 

s_lshl_b32 s0, s4, 6 // (op 15429) 
s_add_u32 s0, s0, 128 // (op 15429) 
v_add_u32 v0, v13, s0 // (op 15429) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15429) Store value





s_lshl_b32 s0, s2, 7 // (op 15454) 
s_add_u32 s0, s0, 128 // (op 15454) 
v_add_u32 v0, v12, s0 // (op 15454) 


v_accvgpr_read v6, a72 // (op 15483) 
v_accvgpr_read v7, a73 // (op 15483) 
v_accvgpr_read v8, a74 // (op 15483) 
v_accvgpr_read v9, a75 // (op 15483) 
v_mul_f32 v14, v11, v6 // (op 15483) 
v_mul_f32 v15, v11, v7 // (op 15483) 
v_mul_f32 v16, v11, v8 // (op 15483) 
v_mul_f32 v17, v11, v9 // (op 15483) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15454) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15451) 
v_lshrrev_b32 v7, 16, v8 // (op 15451) 
v_lshlrev_b32 v6, 16, v6 // (op 15451) 
v_lshlrev_b32 v7, 16, v7 // (op 15451) 
v_mul_f32 v18, v10, v6 // (op 15451) 
v_mul_f32 v19, v10, v7 // (op 15451) 
v_and_b32 v6, 65535, v9 // (op 15451) 
v_lshrrev_b32 v7, 16, v9 // (op 15451) 
v_lshlrev_b32 v6, 16, v6 // (op 15451) 
v_lshlrev_b32 v7, 16, v7 // (op 15451) 
v_mul_f32 v20, v10, v6 // (op 15451) 
v_mul_f32 v21, v10, v7 // (op 15451) 


v_add_f32 v6, v18, v14 // (op 15452) 
v_add_f32 v7, v19, v15 // (op 15452) 
v_add_f32 v8, v20, v16 // (op 15452) 
v_add_f32 v9, v21, v17 // (op 15452) 


v_lshrrev_b32 v14, 16, v6 // (op 15453) 
v_lshrrev_b32 v15, 16, v7 // (op 15453) 
v_lshrrev_b32 v16, 16, v8 // (op 15453) 
v_lshrrev_b32 v17, 16, v9 // (op 15453) 


v_pack_b32_f16 v8, v14, v15 // (op 15431) 
v_pack_b32_f16 v9, v16, v17 // (op 15431) 

s_lshl_b32 s0, s4, 7 // (op 15431) 
s_add_u32 s0, s0, 128 // (op 15431) 
v_add_u32 v0, v13, s0 // (op 15431) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15431) Store value





s_lshl_b32 s0, s2, 6 // (op 15459) 
s_mul_i32 s1, 3, s0 // (op 15459) 
s_add_u32 s1, s1, 128 // (op 15459) 

v_add_u32 v0, v12, s1 // (op 15459) 


v_accvgpr_read v6, a76 // (op 15485) 
v_accvgpr_read v7, a77 // (op 15485) 
v_accvgpr_read v8, a78 // (op 15485) 
v_accvgpr_read v9, a79 // (op 15485) 
v_mul_f32 v14, v11, v6 // (op 15485) 
v_mul_f32 v15, v11, v7 // (op 15485) 
v_mul_f32 v16, v11, v8 // (op 15485) 
v_mul_f32 v17, v11, v9 // (op 15485) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15459) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15456) 
v_lshrrev_b32 v7, 16, v8 // (op 15456) 
v_lshlrev_b32 v6, 16, v6 // (op 15456) 
v_lshlrev_b32 v7, 16, v7 // (op 15456) 
v_mul_f32 v18, v10, v6 // (op 15456) 
v_mul_f32 v19, v10, v7 // (op 15456) 
v_and_b32 v6, 65535, v9 // (op 15456) 
v_lshrrev_b32 v7, 16, v9 // (op 15456) 
v_lshlrev_b32 v6, 16, v6 // (op 15456) 
v_lshlrev_b32 v7, 16, v7 // (op 15456) 
v_mul_f32 v20, v10, v6 // (op 15456) 
v_mul_f32 v21, v10, v7 // (op 15456) 


v_add_f32 v6, v18, v14 // (op 15457) 
v_add_f32 v7, v19, v15 // (op 15457) 
v_add_f32 v8, v20, v16 // (op 15457) 
v_add_f32 v9, v21, v17 // (op 15457) 


v_lshrrev_b32 v14, 16, v6 // (op 15458) 
v_lshrrev_b32 v15, 16, v7 // (op 15458) 
v_lshrrev_b32 v16, 16, v8 // (op 15458) 
v_lshrrev_b32 v17, 16, v9 // (op 15458) 


v_pack_b32_f16 v8, v14, v15 // (op 15433) 
v_pack_b32_f16 v9, v16, v17 // (op 15433) 

s_lshl_b32 s0, s4, 6 // (op 15433) 
s_mul_i32 s1, 3, s0 // (op 15433) 
s_add_u32 s1, s1, 128 // (op 15433) 

v_add_u32 v0, v13, s1 // (op 15433) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15433) Store value





s_lshl_b32 s0, s2, 8 // (op 15464) 
s_add_u32 s0, s0, 128 // (op 15464) 
v_add_u32 v0, v12, s0 // (op 15464) 


v_accvgpr_read v6, a80 // (op 15487) 
v_accvgpr_read v7, a81 // (op 15487) 
v_accvgpr_read v8, a82 // (op 15487) 
v_accvgpr_read v9, a83 // (op 15487) 
v_mul_f32 v14, v11, v6 // (op 15487) 
v_mul_f32 v15, v11, v7 // (op 15487) 
v_mul_f32 v16, v11, v8 // (op 15487) 
v_mul_f32 v17, v11, v9 // (op 15487) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15464) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15461) 
v_lshrrev_b32 v7, 16, v8 // (op 15461) 
v_lshlrev_b32 v6, 16, v6 // (op 15461) 
v_lshlrev_b32 v7, 16, v7 // (op 15461) 
v_mul_f32 v18, v10, v6 // (op 15461) 
v_mul_f32 v19, v10, v7 // (op 15461) 
v_and_b32 v6, 65535, v9 // (op 15461) 
v_lshrrev_b32 v7, 16, v9 // (op 15461) 
v_lshlrev_b32 v6, 16, v6 // (op 15461) 
v_lshlrev_b32 v7, 16, v7 // (op 15461) 
v_mul_f32 v20, v10, v6 // (op 15461) 
v_mul_f32 v21, v10, v7 // (op 15461) 


v_add_f32 v6, v18, v14 // (op 15462) 
v_add_f32 v7, v19, v15 // (op 15462) 
v_add_f32 v8, v20, v16 // (op 15462) 
v_add_f32 v9, v21, v17 // (op 15462) 


v_lshrrev_b32 v14, 16, v6 // (op 15463) 
v_lshrrev_b32 v15, 16, v7 // (op 15463) 
v_lshrrev_b32 v16, 16, v8 // (op 15463) 
v_lshrrev_b32 v17, 16, v9 // (op 15463) 


v_pack_b32_f16 v8, v14, v15 // (op 15435) 
v_pack_b32_f16 v9, v16, v17 // (op 15435) 

s_lshl_b32 s0, s4, 8 // (op 15435) 
s_add_u32 s0, s0, 128 // (op 15435) 
v_add_u32 v0, v13, s0 // (op 15435) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15435) Store value





s_lshl_b32 s0, s2, 6 // (op 15469) 
s_mul_i32 s1, 5, s0 // (op 15469) 
s_add_u32 s1, s1, 128 // (op 15469) 

v_add_u32 v0, v12, s1 // (op 15469) 


v_accvgpr_read v6, a84 // (op 15489) 
v_accvgpr_read v7, a85 // (op 15489) 
v_accvgpr_read v8, a86 // (op 15489) 
v_accvgpr_read v9, a87 // (op 15489) 
v_mul_f32 v14, v11, v6 // (op 15489) 
v_mul_f32 v15, v11, v7 // (op 15489) 
v_mul_f32 v16, v11, v8 // (op 15489) 
v_mul_f32 v17, v11, v9 // (op 15489) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15469) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15466) 
v_lshrrev_b32 v7, 16, v8 // (op 15466) 
v_lshlrev_b32 v6, 16, v6 // (op 15466) 
v_lshlrev_b32 v7, 16, v7 // (op 15466) 
v_mul_f32 v18, v10, v6 // (op 15466) 
v_mul_f32 v19, v10, v7 // (op 15466) 
v_and_b32 v6, 65535, v9 // (op 15466) 
v_lshrrev_b32 v7, 16, v9 // (op 15466) 
v_lshlrev_b32 v6, 16, v6 // (op 15466) 
v_lshlrev_b32 v7, 16, v7 // (op 15466) 
v_mul_f32 v20, v10, v6 // (op 15466) 
v_mul_f32 v21, v10, v7 // (op 15466) 


v_add_f32 v6, v18, v14 // (op 15467) 
v_add_f32 v7, v19, v15 // (op 15467) 
v_add_f32 v8, v20, v16 // (op 15467) 
v_add_f32 v9, v21, v17 // (op 15467) 


v_lshrrev_b32 v14, 16, v6 // (op 15468) 
v_lshrrev_b32 v15, 16, v7 // (op 15468) 
v_lshrrev_b32 v16, 16, v8 // (op 15468) 
v_lshrrev_b32 v17, 16, v9 // (op 15468) 


v_pack_b32_f16 v8, v14, v15 // (op 15437) 
v_pack_b32_f16 v9, v16, v17 // (op 15437) 

s_lshl_b32 s0, s4, 6 // (op 15437) 
s_mul_i32 s1, 5, s0 // (op 15437) 
s_add_u32 s1, s1, 128 // (op 15437) 

v_add_u32 v0, v13, s1 // (op 15437) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15437) Store value





s_lshl_b32 s0, s2, 6 // (op 15474) 
s_mul_i32 s1, 6, s0 // (op 15474) 
s_add_u32 s1, s1, 128 // (op 15474) 

v_add_u32 v0, v12, s1 // (op 15474) 


v_accvgpr_read v6, a88 // (op 15491) 
v_accvgpr_read v7, a89 // (op 15491) 
v_accvgpr_read v8, a90 // (op 15491) 
v_accvgpr_read v9, a91 // (op 15491) 
v_mul_f32 v14, v11, v6 // (op 15491) 
v_mul_f32 v15, v11, v7 // (op 15491) 
v_mul_f32 v16, v11, v8 // (op 15491) 
v_mul_f32 v17, v11, v9 // (op 15491) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15474) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15471) 
v_lshrrev_b32 v7, 16, v8 // (op 15471) 
v_lshlrev_b32 v6, 16, v6 // (op 15471) 
v_lshlrev_b32 v7, 16, v7 // (op 15471) 
v_mul_f32 v18, v10, v6 // (op 15471) 
v_mul_f32 v19, v10, v7 // (op 15471) 
v_and_b32 v6, 65535, v9 // (op 15471) 
v_lshrrev_b32 v7, 16, v9 // (op 15471) 
v_lshlrev_b32 v6, 16, v6 // (op 15471) 
v_lshlrev_b32 v7, 16, v7 // (op 15471) 
v_mul_f32 v20, v10, v6 // (op 15471) 
v_mul_f32 v21, v10, v7 // (op 15471) 


v_add_f32 v6, v18, v14 // (op 15472) 
v_add_f32 v7, v19, v15 // (op 15472) 
v_add_f32 v8, v20, v16 // (op 15472) 
v_add_f32 v9, v21, v17 // (op 15472) 


v_lshrrev_b32 v14, 16, v6 // (op 15473) 
v_lshrrev_b32 v15, 16, v7 // (op 15473) 
v_lshrrev_b32 v16, 16, v8 // (op 15473) 
v_lshrrev_b32 v17, 16, v9 // (op 15473) 


v_pack_b32_f16 v8, v14, v15 // (op 15439) 
v_pack_b32_f16 v9, v16, v17 // (op 15439) 

s_lshl_b32 s0, s4, 6 // (op 15439) 
s_mul_i32 s1, 6, s0 // (op 15439) 
s_add_u32 s1, s1, 128 // (op 15439) 

v_add_u32 v0, v13, s1 // (op 15439) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15439) Store value





s_lshl_b32 s0, s2, 6 // (op 15479) 
s_mul_i32 s1, 7, s0 // (op 15479) 
s_add_u32 s1, s1, 128 // (op 15479) 

v_add_u32 v0, v12, s1 // (op 15479) 


v_accvgpr_read v6, a92 // (op 15493) 
v_accvgpr_read v7, a93 // (op 15493) 
v_accvgpr_read v8, a94 // (op 15493) 
v_accvgpr_read v9, a95 // (op 15493) 
v_mul_f32 v14, v11, v6 // (op 15493) 
v_mul_f32 v15, v11, v7 // (op 15493) 
v_mul_f32 v16, v11, v8 // (op 15493) 
v_mul_f32 v17, v11, v9 // (op 15493) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15479) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15476) 
v_lshrrev_b32 v7, 16, v8 // (op 15476) 
v_lshlrev_b32 v6, 16, v6 // (op 15476) 
v_lshlrev_b32 v7, 16, v7 // (op 15476) 
v_mul_f32 v18, v10, v6 // (op 15476) 
v_mul_f32 v19, v10, v7 // (op 15476) 
v_and_b32 v6, 65535, v9 // (op 15476) 
v_lshrrev_b32 v7, 16, v9 // (op 15476) 
v_lshlrev_b32 v6, 16, v6 // (op 15476) 
v_lshlrev_b32 v7, 16, v7 // (op 15476) 
v_mul_f32 v20, v10, v6 // (op 15476) 
v_mul_f32 v21, v10, v7 // (op 15476) 


v_add_f32 v6, v18, v14 // (op 15477) 
v_add_f32 v7, v19, v15 // (op 15477) 
v_add_f32 v8, v20, v16 // (op 15477) 
v_add_f32 v9, v21, v17 // (op 15477) 


v_lshrrev_b32 v14, 16, v6 // (op 15478) 
v_lshrrev_b32 v15, 16, v7 // (op 15478) 
v_lshrrev_b32 v16, 16, v8 // (op 15478) 
v_lshrrev_b32 v17, 16, v9 // (op 15478) 


v_pack_b32_f16 v8, v14, v15 // (op 15441) 
v_pack_b32_f16 v9, v16, v17 // (op 15441) 

s_lshl_b32 s0, s4, 6 // (op 15441) 
s_mul_i32 s1, 7, s0 // (op 15441) 
s_add_u32 s1, s1, 128 // (op 15441) 

v_add_u32 v0, v13, s1 // (op 15441) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15441) Store value





v_add_u32 v0, 64, v12 // (op 15273) 

v_accvgpr_read v6, a32 // (op 15252) 
v_accvgpr_read v7, a33 // (op 15252) 
v_accvgpr_read v8, a34 // (op 15252) 
v_accvgpr_read v9, a35 // (op 15252) 
v_mul_f32 v14, v11, v6 // (op 15252) 
v_mul_f32 v15, v11, v7 // (op 15252) 
v_mul_f32 v16, v11, v8 // (op 15252) 
v_mul_f32 v17, v11, v9 // (op 15252) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15273) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15272) 
v_lshrrev_b32 v7, 16, v8 // (op 15272) 
v_lshlrev_b32 v6, 16, v6 // (op 15272) 
v_lshlrev_b32 v7, 16, v7 // (op 15272) 
v_mul_f32 v18, v10, v6 // (op 15272) 
v_mul_f32 v19, v10, v7 // (op 15272) 
v_and_b32 v6, 65535, v9 // (op 15272) 
v_lshrrev_b32 v7, 16, v9 // (op 15272) 
v_lshlrev_b32 v6, 16, v6 // (op 15272) 
v_lshlrev_b32 v7, 16, v7 // (op 15272) 
v_mul_f32 v20, v10, v6 // (op 15272) 
v_mul_f32 v21, v10, v7 // (op 15272) 


v_add_f32 v6, v18, v14 // (op 15253) 
v_add_f32 v7, v19, v15 // (op 15253) 
v_add_f32 v8, v20, v16 // (op 15253) 
v_add_f32 v9, v21, v17 // (op 15253) 


v_lshrrev_b32 v14, 16, v6 // (op 15254) 
v_lshrrev_b32 v15, 16, v7 // (op 15254) 
v_lshrrev_b32 v16, 16, v8 // (op 15254) 
v_lshrrev_b32 v17, 16, v9 // (op 15254) 


v_pack_b32_f16 v8, v14, v15 // (op 15256) 
v_pack_b32_f16 v9, v16, v17 // (op 15256) 

v_add_u32 v0, 64, v13 // (op 15256) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15256) Store value





s_lshl_b32 s0, s2, 6 // (op 15278) 
s_add_u32 s0, s0, 64 // (op 15278) 
v_add_u32 v0, v12, s0 // (op 15278) 


v_accvgpr_read v6, a36 // (op 15310) 
v_accvgpr_read v7, a37 // (op 15310) 
v_accvgpr_read v8, a38 // (op 15310) 
v_accvgpr_read v9, a39 // (op 15310) 
v_mul_f32 v14, v11, v6 // (op 15310) 
v_mul_f32 v15, v11, v7 // (op 15310) 
v_mul_f32 v16, v11, v8 // (op 15310) 
v_mul_f32 v17, v11, v9 // (op 15310) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15278) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15275) 
v_lshrrev_b32 v7, 16, v8 // (op 15275) 
v_lshlrev_b32 v6, 16, v6 // (op 15275) 
v_lshlrev_b32 v7, 16, v7 // (op 15275) 
v_mul_f32 v18, v10, v6 // (op 15275) 
v_mul_f32 v19, v10, v7 // (op 15275) 
v_and_b32 v6, 65535, v9 // (op 15275) 
v_lshrrev_b32 v7, 16, v9 // (op 15275) 
v_lshlrev_b32 v6, 16, v6 // (op 15275) 
v_lshlrev_b32 v7, 16, v7 // (op 15275) 
v_mul_f32 v20, v10, v6 // (op 15275) 
v_mul_f32 v21, v10, v7 // (op 15275) 


v_add_f32 v6, v18, v14 // (op 15276) 
v_add_f32 v7, v19, v15 // (op 15276) 
v_add_f32 v8, v20, v16 // (op 15276) 
v_add_f32 v9, v21, v17 // (op 15276) 


v_lshrrev_b32 v14, 16, v6 // (op 15277) 
v_lshrrev_b32 v15, 16, v7 // (op 15277) 
v_lshrrev_b32 v16, 16, v8 // (op 15277) 
v_lshrrev_b32 v17, 16, v9 // (op 15277) 


v_pack_b32_f16 v8, v14, v15 // (op 15258) 
v_pack_b32_f16 v9, v16, v17 // (op 15258) 

s_lshl_b32 s0, s4, 6 // (op 15258) 
s_add_u32 s0, s0, 64 // (op 15258) 
v_add_u32 v0, v13, s0 // (op 15258) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15258) Store value





s_lshl_b32 s0, s2, 7 // (op 15283) 
s_add_u32 s0, s0, 64 // (op 15283) 
v_add_u32 v0, v12, s0 // (op 15283) 


v_accvgpr_read v6, a40 // (op 15312) 
v_accvgpr_read v7, a41 // (op 15312) 
v_accvgpr_read v8, a42 // (op 15312) 
v_accvgpr_read v9, a43 // (op 15312) 
v_mul_f32 v14, v11, v6 // (op 15312) 
v_mul_f32 v15, v11, v7 // (op 15312) 
v_mul_f32 v16, v11, v8 // (op 15312) 
v_mul_f32 v17, v11, v9 // (op 15312) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15283) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15280) 
v_lshrrev_b32 v7, 16, v8 // (op 15280) 
v_lshlrev_b32 v6, 16, v6 // (op 15280) 
v_lshlrev_b32 v7, 16, v7 // (op 15280) 
v_mul_f32 v18, v10, v6 // (op 15280) 
v_mul_f32 v19, v10, v7 // (op 15280) 
v_and_b32 v6, 65535, v9 // (op 15280) 
v_lshrrev_b32 v7, 16, v9 // (op 15280) 
v_lshlrev_b32 v6, 16, v6 // (op 15280) 
v_lshlrev_b32 v7, 16, v7 // (op 15280) 
v_mul_f32 v20, v10, v6 // (op 15280) 
v_mul_f32 v21, v10, v7 // (op 15280) 


v_add_f32 v6, v18, v14 // (op 15281) 
v_add_f32 v7, v19, v15 // (op 15281) 
v_add_f32 v8, v20, v16 // (op 15281) 
v_add_f32 v9, v21, v17 // (op 15281) 


v_lshrrev_b32 v14, 16, v6 // (op 15282) 
v_lshrrev_b32 v15, 16, v7 // (op 15282) 
v_lshrrev_b32 v16, 16, v8 // (op 15282) 
v_lshrrev_b32 v17, 16, v9 // (op 15282) 


v_pack_b32_f16 v8, v14, v15 // (op 15260) 
v_pack_b32_f16 v9, v16, v17 // (op 15260) 

s_lshl_b32 s0, s4, 7 // (op 15260) 
s_add_u32 s0, s0, 64 // (op 15260) 
v_add_u32 v0, v13, s0 // (op 15260) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15260) Store value





s_lshl_b32 s0, s2, 6 // (op 15288) 
s_mul_i32 s1, 3, s0 // (op 15288) 
s_add_u32 s1, s1, 64 // (op 15288) 

v_add_u32 v0, v12, s1 // (op 15288) 


v_accvgpr_read v6, a44 // (op 15314) 
v_accvgpr_read v7, a45 // (op 15314) 
v_accvgpr_read v8, a46 // (op 15314) 
v_accvgpr_read v9, a47 // (op 15314) 
v_mul_f32 v14, v11, v6 // (op 15314) 
v_mul_f32 v15, v11, v7 // (op 15314) 
v_mul_f32 v16, v11, v8 // (op 15314) 
v_mul_f32 v17, v11, v9 // (op 15314) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15288) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15285) 
v_lshrrev_b32 v7, 16, v8 // (op 15285) 
v_lshlrev_b32 v6, 16, v6 // (op 15285) 
v_lshlrev_b32 v7, 16, v7 // (op 15285) 
v_mul_f32 v18, v10, v6 // (op 15285) 
v_mul_f32 v19, v10, v7 // (op 15285) 
v_and_b32 v6, 65535, v9 // (op 15285) 
v_lshrrev_b32 v7, 16, v9 // (op 15285) 
v_lshlrev_b32 v6, 16, v6 // (op 15285) 
v_lshlrev_b32 v7, 16, v7 // (op 15285) 
v_mul_f32 v20, v10, v6 // (op 15285) 
v_mul_f32 v21, v10, v7 // (op 15285) 


v_add_f32 v6, v18, v14 // (op 15286) 
v_add_f32 v7, v19, v15 // (op 15286) 
v_add_f32 v8, v20, v16 // (op 15286) 
v_add_f32 v9, v21, v17 // (op 15286) 


v_lshrrev_b32 v14, 16, v6 // (op 15287) 
v_lshrrev_b32 v15, 16, v7 // (op 15287) 
v_lshrrev_b32 v16, 16, v8 // (op 15287) 
v_lshrrev_b32 v17, 16, v9 // (op 15287) 


v_pack_b32_f16 v8, v14, v15 // (op 15262) 
v_pack_b32_f16 v9, v16, v17 // (op 15262) 

s_lshl_b32 s0, s4, 6 // (op 15262) 
s_mul_i32 s1, 3, s0 // (op 15262) 
s_add_u32 s1, s1, 64 // (op 15262) 

v_add_u32 v0, v13, s1 // (op 15262) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15262) Store value





s_lshl_b32 s0, s2, 8 // (op 15293) 
s_add_u32 s0, s0, 64 // (op 15293) 
v_add_u32 v0, v12, s0 // (op 15293) 


v_accvgpr_read v6, a48 // (op 15316) 
v_accvgpr_read v7, a49 // (op 15316) 
v_accvgpr_read v8, a50 // (op 15316) 
v_accvgpr_read v9, a51 // (op 15316) 
v_mul_f32 v14, v11, v6 // (op 15316) 
v_mul_f32 v15, v11, v7 // (op 15316) 
v_mul_f32 v16, v11, v8 // (op 15316) 
v_mul_f32 v17, v11, v9 // (op 15316) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15293) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15290) 
v_lshrrev_b32 v7, 16, v8 // (op 15290) 
v_lshlrev_b32 v6, 16, v6 // (op 15290) 
v_lshlrev_b32 v7, 16, v7 // (op 15290) 
v_mul_f32 v18, v10, v6 // (op 15290) 
v_mul_f32 v19, v10, v7 // (op 15290) 
v_and_b32 v6, 65535, v9 // (op 15290) 
v_lshrrev_b32 v7, 16, v9 // (op 15290) 
v_lshlrev_b32 v6, 16, v6 // (op 15290) 
v_lshlrev_b32 v7, 16, v7 // (op 15290) 
v_mul_f32 v20, v10, v6 // (op 15290) 
v_mul_f32 v21, v10, v7 // (op 15290) 


v_add_f32 v6, v18, v14 // (op 15291) 
v_add_f32 v7, v19, v15 // (op 15291) 
v_add_f32 v8, v20, v16 // (op 15291) 
v_add_f32 v9, v21, v17 // (op 15291) 


v_lshrrev_b32 v14, 16, v6 // (op 15292) 
v_lshrrev_b32 v15, 16, v7 // (op 15292) 
v_lshrrev_b32 v16, 16, v8 // (op 15292) 
v_lshrrev_b32 v17, 16, v9 // (op 15292) 


v_pack_b32_f16 v8, v14, v15 // (op 15264) 
v_pack_b32_f16 v9, v16, v17 // (op 15264) 

s_lshl_b32 s0, s4, 8 // (op 15264) 
s_add_u32 s0, s0, 64 // (op 15264) 
v_add_u32 v0, v13, s0 // (op 15264) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15264) Store value





s_lshl_b32 s0, s2, 6 // (op 15298) 
s_mul_i32 s1, 5, s0 // (op 15298) 
s_add_u32 s1, s1, 64 // (op 15298) 

v_add_u32 v0, v12, s1 // (op 15298) 


v_accvgpr_read v6, a52 // (op 15318) 
v_accvgpr_read v7, a53 // (op 15318) 
v_accvgpr_read v8, a54 // (op 15318) 
v_accvgpr_read v9, a55 // (op 15318) 
v_mul_f32 v14, v11, v6 // (op 15318) 
v_mul_f32 v15, v11, v7 // (op 15318) 
v_mul_f32 v16, v11, v8 // (op 15318) 
v_mul_f32 v17, v11, v9 // (op 15318) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15298) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15295) 
v_lshrrev_b32 v7, 16, v8 // (op 15295) 
v_lshlrev_b32 v6, 16, v6 // (op 15295) 
v_lshlrev_b32 v7, 16, v7 // (op 15295) 
v_mul_f32 v18, v10, v6 // (op 15295) 
v_mul_f32 v19, v10, v7 // (op 15295) 
v_and_b32 v6, 65535, v9 // (op 15295) 
v_lshrrev_b32 v7, 16, v9 // (op 15295) 
v_lshlrev_b32 v6, 16, v6 // (op 15295) 
v_lshlrev_b32 v7, 16, v7 // (op 15295) 
v_mul_f32 v20, v10, v6 // (op 15295) 
v_mul_f32 v21, v10, v7 // (op 15295) 


v_add_f32 v6, v18, v14 // (op 15296) 
v_add_f32 v7, v19, v15 // (op 15296) 
v_add_f32 v8, v20, v16 // (op 15296) 
v_add_f32 v9, v21, v17 // (op 15296) 


v_lshrrev_b32 v14, 16, v6 // (op 15297) 
v_lshrrev_b32 v15, 16, v7 // (op 15297) 
v_lshrrev_b32 v16, 16, v8 // (op 15297) 
v_lshrrev_b32 v17, 16, v9 // (op 15297) 


v_pack_b32_f16 v8, v14, v15 // (op 15266) 
v_pack_b32_f16 v9, v16, v17 // (op 15266) 

s_lshl_b32 s0, s4, 6 // (op 15266) 
s_mul_i32 s1, 5, s0 // (op 15266) 
s_add_u32 s1, s1, 64 // (op 15266) 

v_add_u32 v0, v13, s1 // (op 15266) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15266) Store value





s_lshl_b32 s0, s2, 6 // (op 15303) 
s_mul_i32 s1, 6, s0 // (op 15303) 
s_add_u32 s1, s1, 64 // (op 15303) 

v_add_u32 v0, v12, s1 // (op 15303) 


v_accvgpr_read v6, a56 // (op 15320) 
v_accvgpr_read v7, a57 // (op 15320) 
v_accvgpr_read v8, a58 // (op 15320) 
v_accvgpr_read v9, a59 // (op 15320) 
v_mul_f32 v14, v11, v6 // (op 15320) 
v_mul_f32 v15, v11, v7 // (op 15320) 
v_mul_f32 v16, v11, v8 // (op 15320) 
v_mul_f32 v17, v11, v9 // (op 15320) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15303) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15300) 
v_lshrrev_b32 v7, 16, v8 // (op 15300) 
v_lshlrev_b32 v6, 16, v6 // (op 15300) 
v_lshlrev_b32 v7, 16, v7 // (op 15300) 
v_mul_f32 v18, v10, v6 // (op 15300) 
v_mul_f32 v19, v10, v7 // (op 15300) 
v_and_b32 v6, 65535, v9 // (op 15300) 
v_lshrrev_b32 v7, 16, v9 // (op 15300) 
v_lshlrev_b32 v6, 16, v6 // (op 15300) 
v_lshlrev_b32 v7, 16, v7 // (op 15300) 
v_mul_f32 v20, v10, v6 // (op 15300) 
v_mul_f32 v21, v10, v7 // (op 15300) 


v_add_f32 v6, v18, v14 // (op 15301) 
v_add_f32 v7, v19, v15 // (op 15301) 
v_add_f32 v8, v20, v16 // (op 15301) 
v_add_f32 v9, v21, v17 // (op 15301) 


v_lshrrev_b32 v14, 16, v6 // (op 15302) 
v_lshrrev_b32 v15, 16, v7 // (op 15302) 
v_lshrrev_b32 v16, 16, v8 // (op 15302) 
v_lshrrev_b32 v17, 16, v9 // (op 15302) 


v_pack_b32_f16 v8, v14, v15 // (op 15268) 
v_pack_b32_f16 v9, v16, v17 // (op 15268) 

s_lshl_b32 s0, s4, 6 // (op 15268) 
s_mul_i32 s1, 6, s0 // (op 15268) 
s_add_u32 s1, s1, 64 // (op 15268) 

v_add_u32 v0, v13, s1 // (op 15268) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15268) Store value





s_lshl_b32 s0, s2, 6 // (op 15308) 
s_mul_i32 s1, 7, s0 // (op 15308) 
s_add_u32 s1, s1, 64 // (op 15308) 

v_add_u32 v0, v12, s1 // (op 15308) 


v_accvgpr_read v6, a60 // (op 15322) 
v_accvgpr_read v7, a61 // (op 15322) 
v_accvgpr_read v8, a62 // (op 15322) 
v_accvgpr_read v9, a63 // (op 15322) 
v_mul_f32 v14, v11, v6 // (op 15322) 
v_mul_f32 v15, v11, v7 // (op 15322) 
v_mul_f32 v16, v11, v8 // (op 15322) 
v_mul_f32 v17, v11, v9 // (op 15322) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15308) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15305) 
v_lshrrev_b32 v7, 16, v8 // (op 15305) 
v_lshlrev_b32 v6, 16, v6 // (op 15305) 
v_lshlrev_b32 v7, 16, v7 // (op 15305) 
v_mul_f32 v18, v10, v6 // (op 15305) 
v_mul_f32 v19, v10, v7 // (op 15305) 
v_and_b32 v6, 65535, v9 // (op 15305) 
v_lshrrev_b32 v7, 16, v9 // (op 15305) 
v_lshlrev_b32 v6, 16, v6 // (op 15305) 
v_lshlrev_b32 v7, 16, v7 // (op 15305) 
v_mul_f32 v20, v10, v6 // (op 15305) 
v_mul_f32 v21, v10, v7 // (op 15305) 


v_add_f32 v6, v18, v14 // (op 15306) 
v_add_f32 v7, v19, v15 // (op 15306) 
v_add_f32 v8, v20, v16 // (op 15306) 
v_add_f32 v9, v21, v17 // (op 15306) 


v_lshrrev_b32 v14, 16, v6 // (op 15307) 
v_lshrrev_b32 v15, 16, v7 // (op 15307) 
v_lshrrev_b32 v16, 16, v8 // (op 15307) 
v_lshrrev_b32 v17, 16, v9 // (op 15307) 


v_pack_b32_f16 v8, v14, v15 // (op 15270) 
v_pack_b32_f16 v9, v16, v17 // (op 15270) 

s_lshl_b32 s0, s4, 6 // (op 15270) 
s_mul_i32 s1, 7, s0 // (op 15270) 
s_add_u32 s1, s1, 64 // (op 15270) 

v_add_u32 v0, v13, s1 // (op 15270) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15270) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

v_accvgpr_read v6, a0 // (op 144) 
v_accvgpr_read v7, a1 // (op 144) 
v_accvgpr_read v8, a2 // (op 144) 
v_accvgpr_read v9, a3 // (op 144) 
v_mul_f32 v14, v11, v6 // (op 144) 
v_mul_f32 v15, v11, v7 // (op 144) 
v_mul_f32 v16, v11, v8 // (op 144) 
v_mul_f32 v17, v11, v9 // (op 144) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 148) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 15077) 
v_add_u32 v0, v12, s0 // (op 15077) 


v_accvgpr_read v6, a4 // (op 15073) 
v_accvgpr_read v7, a5 // (op 15073) 
v_accvgpr_read v8, a6 // (op 15073) 
v_accvgpr_read v9, a7 // (op 15073) 
v_mul_f32 v14, v11, v6 // (op 15073) 
v_mul_f32 v15, v11, v7 // (op 15073) 
v_mul_f32 v16, v11, v8 // (op 15073) 
v_mul_f32 v17, v11, v9 // (op 15073) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15077) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15078) 
v_lshrrev_b32 v7, 16, v8 // (op 15078) 
v_lshlrev_b32 v6, 16, v6 // (op 15078) 
v_lshlrev_b32 v7, 16, v7 // (op 15078) 
v_mul_f32 v18, v10, v6 // (op 15078) 
v_mul_f32 v19, v10, v7 // (op 15078) 
v_and_b32 v6, 65535, v9 // (op 15078) 
v_lshrrev_b32 v7, 16, v9 // (op 15078) 
v_lshlrev_b32 v6, 16, v6 // (op 15078) 
v_lshlrev_b32 v7, 16, v7 // (op 15078) 
v_mul_f32 v20, v10, v6 // (op 15078) 
v_mul_f32 v21, v10, v7 // (op 15078) 


v_add_f32 v6, v18, v14 // (op 15074) 
v_add_f32 v7, v19, v15 // (op 15074) 
v_add_f32 v8, v20, v16 // (op 15074) 
v_add_f32 v9, v21, v17 // (op 15074) 


v_lshrrev_b32 v14, 16, v6 // (op 15075) 
v_lshrrev_b32 v15, 16, v7 // (op 15075) 
v_lshrrev_b32 v16, 16, v8 // (op 15075) 
v_lshrrev_b32 v17, 16, v9 // (op 15075) 


v_pack_b32_f16 v8, v14, v15 // (op 15076) 
v_pack_b32_f16 v9, v16, v17 // (op 15076) 

s_lshl_b32 s0, s4, 6 // (op 15076) 
v_add_u32 v0, v13, s0 // (op 15076) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15076) Store value





s_lshl_b32 s0, s2, 7 // (op 15091) 
v_add_u32 v0, v12, s0 // (op 15091) 


v_accvgpr_read v6, a8 // (op 15087) 
v_accvgpr_read v7, a9 // (op 15087) 
v_accvgpr_read v8, a10 // (op 15087) 
v_accvgpr_read v9, a11 // (op 15087) 
v_mul_f32 v14, v11, v6 // (op 15087) 
v_mul_f32 v15, v11, v7 // (op 15087) 
v_mul_f32 v16, v11, v8 // (op 15087) 
v_mul_f32 v17, v11, v9 // (op 15087) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15091) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15092) 
v_lshrrev_b32 v7, 16, v8 // (op 15092) 
v_lshlrev_b32 v6, 16, v6 // (op 15092) 
v_lshlrev_b32 v7, 16, v7 // (op 15092) 
v_mul_f32 v18, v10, v6 // (op 15092) 
v_mul_f32 v19, v10, v7 // (op 15092) 
v_and_b32 v6, 65535, v9 // (op 15092) 
v_lshrrev_b32 v7, 16, v9 // (op 15092) 
v_lshlrev_b32 v6, 16, v6 // (op 15092) 
v_lshlrev_b32 v7, 16, v7 // (op 15092) 
v_mul_f32 v20, v10, v6 // (op 15092) 
v_mul_f32 v21, v10, v7 // (op 15092) 


v_add_f32 v6, v18, v14 // (op 15088) 
v_add_f32 v7, v19, v15 // (op 15088) 
v_add_f32 v8, v20, v16 // (op 15088) 
v_add_f32 v9, v21, v17 // (op 15088) 


v_lshrrev_b32 v14, 16, v6 // (op 15089) 
v_lshrrev_b32 v15, 16, v7 // (op 15089) 
v_lshrrev_b32 v16, 16, v8 // (op 15089) 
v_lshrrev_b32 v17, 16, v9 // (op 15089) 


v_pack_b32_f16 v8, v14, v15 // (op 15090) 
v_pack_b32_f16 v9, v16, v17 // (op 15090) 

s_lshl_b32 s0, s4, 7 // (op 15090) 
v_add_u32 v0, v13, s0 // (op 15090) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15090) Store value





s_lshl_b32 s0, s2, 6 // (op 15105) 
s_mul_i32 s1, 3, s0 // (op 15105) 

v_add_u32 v0, v12, s1 // (op 15105) 


v_accvgpr_read v6, a12 // (op 15101) 
v_accvgpr_read v7, a13 // (op 15101) 
v_accvgpr_read v8, a14 // (op 15101) 
v_accvgpr_read v9, a15 // (op 15101) 
v_mul_f32 v14, v11, v6 // (op 15101) 
v_mul_f32 v15, v11, v7 // (op 15101) 
v_mul_f32 v16, v11, v8 // (op 15101) 
v_mul_f32 v17, v11, v9 // (op 15101) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15105) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15106) 
v_lshrrev_b32 v7, 16, v8 // (op 15106) 
v_lshlrev_b32 v6, 16, v6 // (op 15106) 
v_lshlrev_b32 v7, 16, v7 // (op 15106) 
v_mul_f32 v18, v10, v6 // (op 15106) 
v_mul_f32 v19, v10, v7 // (op 15106) 
v_and_b32 v6, 65535, v9 // (op 15106) 
v_lshrrev_b32 v7, 16, v9 // (op 15106) 
v_lshlrev_b32 v6, 16, v6 // (op 15106) 
v_lshlrev_b32 v7, 16, v7 // (op 15106) 
v_mul_f32 v20, v10, v6 // (op 15106) 
v_mul_f32 v21, v10, v7 // (op 15106) 


v_add_f32 v6, v18, v14 // (op 15102) 
v_add_f32 v7, v19, v15 // (op 15102) 
v_add_f32 v8, v20, v16 // (op 15102) 
v_add_f32 v9, v21, v17 // (op 15102) 


v_lshrrev_b32 v14, 16, v6 // (op 15103) 
v_lshrrev_b32 v15, 16, v7 // (op 15103) 
v_lshrrev_b32 v16, 16, v8 // (op 15103) 
v_lshrrev_b32 v17, 16, v9 // (op 15103) 


v_pack_b32_f16 v8, v14, v15 // (op 15104) 
v_pack_b32_f16 v9, v16, v17 // (op 15104) 

s_lshl_b32 s0, s4, 6 // (op 15104) 
s_mul_i32 s1, 3, s0 // (op 15104) 

v_add_u32 v0, v13, s1 // (op 15104) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15104) Store value





s_lshl_b32 s0, s2, 8 // (op 15119) 
v_add_u32 v0, v12, s0 // (op 15119) 


v_accvgpr_read v6, a16 // (op 15115) 
v_accvgpr_read v7, a17 // (op 15115) 
v_accvgpr_read v8, a18 // (op 15115) 
v_accvgpr_read v9, a19 // (op 15115) 
v_mul_f32 v14, v11, v6 // (op 15115) 
v_mul_f32 v15, v11, v7 // (op 15115) 
v_mul_f32 v16, v11, v8 // (op 15115) 
v_mul_f32 v17, v11, v9 // (op 15115) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15119) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15120) 
v_lshrrev_b32 v7, 16, v8 // (op 15120) 
v_lshlrev_b32 v6, 16, v6 // (op 15120) 
v_lshlrev_b32 v7, 16, v7 // (op 15120) 
v_mul_f32 v18, v10, v6 // (op 15120) 
v_mul_f32 v19, v10, v7 // (op 15120) 
v_and_b32 v6, 65535, v9 // (op 15120) 
v_lshrrev_b32 v7, 16, v9 // (op 15120) 
v_lshlrev_b32 v6, 16, v6 // (op 15120) 
v_lshlrev_b32 v7, 16, v7 // (op 15120) 
v_mul_f32 v20, v10, v6 // (op 15120) 
v_mul_f32 v21, v10, v7 // (op 15120) 


v_add_f32 v6, v18, v14 // (op 15116) 
v_add_f32 v7, v19, v15 // (op 15116) 
v_add_f32 v8, v20, v16 // (op 15116) 
v_add_f32 v9, v21, v17 // (op 15116) 


v_lshrrev_b32 v14, 16, v6 // (op 15117) 
v_lshrrev_b32 v15, 16, v7 // (op 15117) 
v_lshrrev_b32 v16, 16, v8 // (op 15117) 
v_lshrrev_b32 v17, 16, v9 // (op 15117) 


v_pack_b32_f16 v8, v14, v15 // (op 15118) 
v_pack_b32_f16 v9, v16, v17 // (op 15118) 

s_lshl_b32 s0, s4, 8 // (op 15118) 
v_add_u32 v0, v13, s0 // (op 15118) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15118) Store value





s_lshl_b32 s0, s2, 6 // (op 15133) 
s_mul_i32 s1, 5, s0 // (op 15133) 

v_add_u32 v0, v12, s1 // (op 15133) 


v_accvgpr_read v6, a20 // (op 15129) 
v_accvgpr_read v7, a21 // (op 15129) 
v_accvgpr_read v8, a22 // (op 15129) 
v_accvgpr_read v9, a23 // (op 15129) 
v_mul_f32 v14, v11, v6 // (op 15129) 
v_mul_f32 v15, v11, v7 // (op 15129) 
v_mul_f32 v16, v11, v8 // (op 15129) 
v_mul_f32 v17, v11, v9 // (op 15129) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15133) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15134) 
v_lshrrev_b32 v7, 16, v8 // (op 15134) 
v_lshlrev_b32 v6, 16, v6 // (op 15134) 
v_lshlrev_b32 v7, 16, v7 // (op 15134) 
v_mul_f32 v18, v10, v6 // (op 15134) 
v_mul_f32 v19, v10, v7 // (op 15134) 
v_and_b32 v6, 65535, v9 // (op 15134) 
v_lshrrev_b32 v7, 16, v9 // (op 15134) 
v_lshlrev_b32 v6, 16, v6 // (op 15134) 
v_lshlrev_b32 v7, 16, v7 // (op 15134) 
v_mul_f32 v20, v10, v6 // (op 15134) 
v_mul_f32 v21, v10, v7 // (op 15134) 


v_add_f32 v6, v18, v14 // (op 15130) 
v_add_f32 v7, v19, v15 // (op 15130) 
v_add_f32 v8, v20, v16 // (op 15130) 
v_add_f32 v9, v21, v17 // (op 15130) 


v_lshrrev_b32 v14, 16, v6 // (op 15131) 
v_lshrrev_b32 v15, 16, v7 // (op 15131) 
v_lshrrev_b32 v16, 16, v8 // (op 15131) 
v_lshrrev_b32 v17, 16, v9 // (op 15131) 


v_pack_b32_f16 v8, v14, v15 // (op 15132) 
v_pack_b32_f16 v9, v16, v17 // (op 15132) 

s_lshl_b32 s0, s4, 6 // (op 15132) 
s_mul_i32 s1, 5, s0 // (op 15132) 

v_add_u32 v0, v13, s1 // (op 15132) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15132) Store value





s_lshl_b32 s0, s2, 6 // (op 15147) 
s_mul_i32 s1, 6, s0 // (op 15147) 

v_add_u32 v0, v12, s1 // (op 15147) 


v_accvgpr_read v6, a24 // (op 15143) 
v_accvgpr_read v7, a25 // (op 15143) 
v_accvgpr_read v8, a26 // (op 15143) 
v_accvgpr_read v9, a27 // (op 15143) 
v_mul_f32 v14, v11, v6 // (op 15143) 
v_mul_f32 v15, v11, v7 // (op 15143) 
v_mul_f32 v16, v11, v8 // (op 15143) 
v_mul_f32 v17, v11, v9 // (op 15143) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15147) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15148) 
v_lshrrev_b32 v7, 16, v8 // (op 15148) 
v_lshlrev_b32 v6, 16, v6 // (op 15148) 
v_lshlrev_b32 v7, 16, v7 // (op 15148) 
v_mul_f32 v18, v10, v6 // (op 15148) 
v_mul_f32 v19, v10, v7 // (op 15148) 
v_and_b32 v6, 65535, v9 // (op 15148) 
v_lshrrev_b32 v7, 16, v9 // (op 15148) 
v_lshlrev_b32 v6, 16, v6 // (op 15148) 
v_lshlrev_b32 v7, 16, v7 // (op 15148) 
v_mul_f32 v20, v10, v6 // (op 15148) 
v_mul_f32 v21, v10, v7 // (op 15148) 


v_add_f32 v6, v18, v14 // (op 15144) 
v_add_f32 v7, v19, v15 // (op 15144) 
v_add_f32 v8, v20, v16 // (op 15144) 
v_add_f32 v9, v21, v17 // (op 15144) 


v_lshrrev_b32 v14, 16, v6 // (op 15145) 
v_lshrrev_b32 v15, 16, v7 // (op 15145) 
v_lshrrev_b32 v16, 16, v8 // (op 15145) 
v_lshrrev_b32 v17, 16, v9 // (op 15145) 


v_pack_b32_f16 v8, v14, v15 // (op 15146) 
v_pack_b32_f16 v9, v16, v17 // (op 15146) 

s_lshl_b32 s0, s4, 6 // (op 15146) 
s_mul_i32 s1, 6, s0 // (op 15146) 

v_add_u32 v0, v13, s1 // (op 15146) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15146) Store value





s_lshl_b32 s0, s2, 6 // (op 15161) 
s_mul_i32 s1, 7, s0 // (op 15161) 

v_add_u32 v0, v12, s1 // (op 15161) 


v_accvgpr_read v6, a28 // (op 15157) 
v_accvgpr_read v7, a29 // (op 15157) 
v_accvgpr_read v8, a30 // (op 15157) 
v_accvgpr_read v9, a31 // (op 15157) 
v_mul_f32 v14, v11, v6 // (op 15157) 
v_mul_f32 v15, v11, v7 // (op 15157) 
v_mul_f32 v16, v11, v8 // (op 15157) 
v_mul_f32 v17, v11, v9 // (op 15157) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 15161) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 15162) 
v_lshrrev_b32 v7, 16, v8 // (op 15162) 
v_lshlrev_b32 v6, 16, v6 // (op 15162) 
v_lshlrev_b32 v7, 16, v7 // (op 15162) 
v_mul_f32 v18, v10, v6 // (op 15162) 
v_mul_f32 v19, v10, v7 // (op 15162) 
v_and_b32 v6, 65535, v9 // (op 15162) 
v_lshrrev_b32 v7, 16, v9 // (op 15162) 
v_lshlrev_b32 v6, 16, v6 // (op 15162) 
v_lshlrev_b32 v7, 16, v7 // (op 15162) 
v_mul_f32 v20, v10, v6 // (op 15162) 
v_mul_f32 v21, v10, v7 // (op 15162) 


v_add_f32 v6, v18, v14 // (op 15158) 
v_add_f32 v7, v19, v15 // (op 15158) 
v_add_f32 v8, v20, v16 // (op 15158) 
v_add_f32 v9, v21, v17 // (op 15158) 


v_lshrrev_b32 v14, 16, v6 // (op 15159) 
v_lshrrev_b32 v15, 16, v7 // (op 15159) 
v_lshrrev_b32 v16, 16, v8 // (op 15159) 
v_lshrrev_b32 v17, 16, v9 // (op 15159) 


v_pack_b32_f16 v8, v14, v15 // (op 15160) 
v_pack_b32_f16 v9, v16, v17 // (op 15160) 

s_lshl_b32 s0, s4, 6 // (op 15160) 
s_mul_i32 s1, 7, s0 // (op 15160) 

v_add_u32 v0, v13, s1 // (op 15160) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 15160) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_135:








































































s_endpgm  // End of RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_, .LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end-RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 476
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 102400 // lds bytes
  .amdhsa_accum_offset 220
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
    DataTypeB: F8
    DestDataType: b
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 102400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 218
    .agpr_count: 256
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

