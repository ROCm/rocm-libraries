// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 8965) 
s_mul_hi_i32 s1, s16, s11 // (op 9222) 
v_lshrrev_b32 v0, 7, v1 // (op 9027) 
v_and_b32 v3, 15, v1 // (op 9027) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9027) 


v_and_b32 v0, 63, v1 // (op 9027) 
v_lshrrev_b32 v3, 4, v0 // (op 9027) 

v_lshlrev_b32 v0, 5, v3 // (op 9027) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 9027) 


v_mul_lo_u32 v0, v3, 6 // (op 9027) 

v_lshrrev_b32 v3, 3, v0 // (op 9027) 

v_and_b32 v0, 63, v1 // (op 9085) 
v_and_b32 v4, 48, v1 // (op 9085) 
v_sub_u32 v5, v0, v4 // (op 9085) 

v_lshrrev_b32 v4, 4, v0 // (op 9085) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 9085) 


v_lshrrev_b32 v4, 6, v1 // (op 9085) 
v_and_b32 v5, 1, v4 // (op 9085) 

v_lshlrev_b32 v4, 4, v5 // (op 9085) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 9085) 


v_mul_lo_u32 v0, v5, 6 // (op 9085) 
v_lshrrev_b32 v4, 3, v0 // (op 9085) 

v_lshrrev_b32 v0, 4, v5 // (op 9085) 

v_lshlrev_b32 v5, 2, v0 // (op 9085) 

v_add_u32 v0, v4, v5 // (op 9085) 


v_lshrrev_b32 v4, 7, v1 // (op 9143) 
v_and_b32 v5, 15, v1 // (op 9143) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 9143) 


v_and_b32 v4, 63, v1 // (op 9143) 
v_lshrrev_b32 v5, 4, v4 // (op 9143) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 9143) 


v_lshrrev_b32 v5, 6, v1 // (op 9201) 
v_and_b32 v6, 1, v5 // (op 9201) 

v_and_b32 v5, 15, v1 // (op 9201) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9201) 


v_and_b32 v5, 63, v1 // (op 9201) 
v_lshrrev_b32 v6, 4, v5 // (op 9201) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 9201) 


v_lshrrev_b32 v6, 7, v1 // (op 9253) 
v_and_b32 v7, 15, v1 // (op 9253) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9253) 


v_and_b32 v6, 63, v1 // (op 9253) 
v_lshrrev_b32 v7, 4, v6 // (op 9253) 

v_lshlrev_b32 v6, 5, v7 // (op 9253) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 9253) 


v_mul_lo_u32 v6, v7, 6 // (op 9253) 

v_lshrrev_b32 v7, 3, v6 // (op 9253) 

v_and_b32 v6, 63, v1 // (op 9282) 
v_and_b32 v8, 48, v1 // (op 9282) 
v_sub_u32 v9, v6, v8 // (op 9282) 

v_lshrrev_b32 v8, 4, v6 // (op 9282) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 9282) 


v_lshrrev_b32 v8, 6, v1 // (op 9282) 
v_and_b32 v9, 1, v8 // (op 9282) 

v_lshlrev_b32 v8, 4, v9 // (op 9282) 

v_lshl_add_u32 v9, v6, 7, v8 // (op 9282) 


v_mul_lo_u32 v6, v9, 6 // (op 9282) 
v_lshrrev_b32 v8, 3, v6 // (op 9282) 

v_lshrrev_b32 v6, 4, v9 // (op 9282) 

v_lshlrev_b32 v9, 2, v6 // (op 9282) 

v_add_u32 v6, v8, v9 // (op 9282) 


v_lshrrev_b32 v8, 7, v1 // (op 9311) 
v_and_b32 v9, 15, v1 // (op 9311) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 9311) 


v_and_b32 v8, 63, v1 // (op 9311) 
v_lshrrev_b32 v9, 4, v8 // (op 9311) 

v_lshl_add_u32 v8, v10, 2, v9 // (op 9311) 


v_lshrrev_b32 v9, 6, v1 // (op 9340) 
v_and_b32 v10, 1, v9 // (op 9340) 

v_and_b32 v9, 15, v1 // (op 9340) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9340) 


v_and_b32 v9, 63, v1 // (op 9340) 
v_lshrrev_b32 v10, 4, v9 // (op 9340) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 9340) 


v_lshrrev_b32 v10, 7, v1 // (op 8965) 
s_add_i32 s18, s0, s16 // (op 8965) 

v_and_b32 v11, 63, v1 // (op 8965) 
s_ashr_i32 s0, s18, 31 // (op 8965) 
v_lshrrev_b32 v12, 4, v11 // (op 8965) 
s_cmp_eq_i32 s11, 0 // (op 8965) 
s_mov_b32 s19, scc // (op 8965) 
v_lshlrev_b32 v13, 2, v12 // (op 8965) 

s_cmp_eq_i32 s19, 0 // (op 8965) 
s_cselect_b32 s51, 0, -1 // (op 8965) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8965) 


s_and_b32 s19, s12, 31 // (op 8965) 
v_lshrrev_b32 v10, 2, v11 // (op 8965) 

v_and_b32 v11, 3, v10 // (op 8965) 

v_lshrrev_b32 v10, 7, v1 // (op 9222) 
v_and_b32 v13, 63, v1 // (op 9222) 
s_lshl_b32 s52, 1, s19 // (op 8965) 
v_and_b32 v14, 3, v1 // (op 8965) 
s_add_i32 s53, s52, s51 // (op 8965) 


s_and_b32 s51, s0, s53 // (op 8965) 


s_add_i32 s0, s1, s16 // (op 9222) 

s_ashr_i32 s1, s0, 31 // (op 9222) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 8965) 


s_add_i32 s52, s18, s51 // (op 8965) 


v_lshrrev_b32 v11, 6, v1 // (op 8965) 
s_ashr_i32 s18, s52, s19 // (op 8965) 


v_and_b32 v14, 1, v11 // (op 8965) 

s_lshl_b32 s19, s12, 24 // (op 8965) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 8965) 


s_ashr_i32 s51, s19, 31 // (op 8965) 

s_xor_b32 s19, s18, s51 // (op 8965) 

s_sub_i32 s18, s19, s51 // (op 8965) 


s_mul_hi_i32 s19, s18, s13 // (op 8965) 
s_add_i32 s51, s19, s18 // (op 8965) 

s_ashr_i32 s19, s51, 31 // (op 8965) 
s_cmp_eq_i32 s13, 0 // (op 8965) 
s_mov_b32 s52, scc // (op 8965) 
s_cmp_eq_i32 s52, 0 // (op 8965) 
s_cselect_b32 s53, 0, -1 // (op 8965) 

s_and_b32 s52, s14, 31 // (op 8965) 
v_lshrrev_b32 v14, 4, v13 // (op 9222) 
s_cmp_eq_i32 s11, 0 // (op 9222) 
s_mov_b32 s54, scc // (op 9222) 
v_lshlrev_b32 v15, 2, v14 // (op 9222) 

s_cmp_eq_i32 s54, 0 // (op 9222) 
s_cselect_b32 s55, 0, -1 // (op 9222) 

s_lshl_b32 s54, 1, s52 // (op 8965) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9222) 


v_lshrrev_b32 v10, 2, v13 // (op 9222) 

s_add_i32 s56, s54, s53 // (op 8965) 


s_and_b32 s53, s19, s56 // (op 8965) 


s_add_i32 s19, s51, s53 // (op 8965) 


s_ashr_i32 s51, s19, s52 // (op 8965) 


s_lshl_b32 s19, s14, 24 // (op 8965) 
s_ashr_i32 s52, s19, 31 // (op 8965) 

s_xor_b32 s19, s51, s52 // (op 8965) 

s_sub_i32 s51, s19, s52 // (op 8965) 


s_and_b32 s19, s12, 31 // (op 9222) 
s_lshl_b32 s52, 1, s19 // (op 9222) 
v_and_b32 v13, 3, v10 // (op 9222) 

s_add_i32 s53, s52, s55 // (op 9222) 


v_and_b32 v10, 3, v1 // (op 9222) 
s_and_b32 s52, s1, s53 // (op 9222) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9222) 


s_add_i32 s1, s0, s52 // (op 9222) 


v_lshrrev_b32 v10, 6, v1 // (op 9222) 
s_ashr_i32 s0, s1, s19 // (op 9222) 


v_and_b32 v13, 1, v10 // (op 9222) 

s_lshl_b32 s1, s12, 24 // (op 9222) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9222) 


s_ashr_i32 s19, s1, 31 // (op 9222) 

s_xor_b32 s1, s0, s19 // (op 9222) 

s_sub_i32 s0, s1, s19 // (op 9222) 


s_mul_hi_i32 s1, s0, s13 // (op 9222) 
s_add_i32 s19, s1, s0 // (op 9222) 

s_ashr_i32 s1, s19, 31 // (op 9222) 
s_cmp_eq_i32 s13, 0 // (op 9222) 
s_mov_b32 s52, scc // (op 9222) 
s_cmp_eq_i32 s52, 0 // (op 9222) 
s_cselect_b32 s53, 0, -1 // (op 9222) 

s_and_b32 s52, s14, 31 // (op 9222) 
s_lshl_b32 s54, 1, s52 // (op 9222) 
s_add_i32 s55, s54, s53 // (op 9222) 


s_and_b32 s53, s1, s55 // (op 9222) 


s_add_i32 s1, s19, s53 // (op 9222) 


s_ashr_i32 s19, s1, s52 // (op 9222) 


s_lshl_b32 s1, s14, 24 // (op 9222) 
s_ashr_i32 s52, s1, 31 // (op 9222) 

s_xor_b32 s1, s19, s52 // (op 9222) 

s_sub_i32 s19, s1, s52 // (op 9222) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 8965) 
s_sub_u32 s52, s16, s1 // (op 8965) 

s_mul_i32 s1, s51, s44 // (op 8965) 
s_add_u32 s1, s1, s52 // (op 8965) 

s_mul_hi_i32 s52, s16, s10 // (op 8965) 
s_add_i32 s53, s52, s16 // (op 8965) 

s_ashr_i32 s52, s53, 31 // (op 8965) 
s_cmp_eq_i32 s10, 0 // (op 8965) 
s_mov_b32 s54, scc // (op 8965) 
s_cmp_eq_i32 s54, 0 // (op 8965) 
s_cselect_b32 s55, 0, -1 // (op 8965) 

s_and_b32 s54, s47, 31 // (op 8965) 
s_lshl_b32 s56, 1, s54 // (op 8965) 
s_add_i32 s57, s56, s55 // (op 8965) 


s_and_b32 s55, s52, s57 // (op 8965) 


s_add_i32 s52, s53, s55 // (op 8965) 


s_ashr_i32 s53, s52, s54 // (op 8965) 


s_lshl_b32 s52, s47, 24 // (op 8965) 
s_ashr_i32 s54, s52, 31 // (op 8965) 

s_xor_b32 s52, s53, s54 // (op 8965) 

s_sub_i32 s53, s52, s54 // (op 8965) 


s_mul_i32 s52, s0, s44 // (op 9222) 
s_sub_u32 s54, s16, s52 // (op 9222) 

s_mul_i32 s52, s19, s44 // (op 9222) 
s_add_u32 s52, s52, s54 // (op 9222) 

s_mul_hi_i32 s54, s16, s10 // (op 9222) 
s_add_i32 s55, s54, s16 // (op 9222) 

s_ashr_i32 s54, s55, 31 // (op 9222) 
s_cmp_eq_i32 s10, 0 // (op 9222) 
s_mov_b32 s56, scc // (op 9222) 
s_cmp_eq_i32 s56, 0 // (op 9222) 
s_cselect_b32 s57, 0, -1 // (op 9222) 

s_and_b32 s56, s47, 31 // (op 9222) 
s_lshl_b32 s58, 1, s56 // (op 9222) 
s_add_i32 s59, s58, s57 // (op 9222) 


s_and_b32 s57, s54, s59 // (op 9222) 


s_add_i32 s54, s55, s57 // (op 9222) 


s_ashr_i32 s55, s54, s56 // (op 9222) 


s_lshl_b32 s54, s47, 24 // (op 9222) 
s_ashr_i32 s56, s54, 31 // (op 9222) 

s_xor_b32 s54, s55, s56 // (op 9222) 

s_sub_i32 s55, s54, s56 // (op 9222) 


s_add_u32 s56, s6, 128 // (op 8965) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 8965) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 8965) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 8965) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 8965) 

s_mul_i32 s54, s44, s56 // (op 8965) 
s_mul_i32 s58, s48, s54 // (op 8965) 

s_mul_i32 s54, s53, s58 // (op 8965) 

s_sub_u32 s58, s16, s54 // (op 8965) 

s_mov_b32 s54, s58 // (op 8965) 
s_mul_hi_i32 s59, s54, s15 // (op 8965) 
s_add_i32 s60, s59, s54 // (op 8965) 


s_ashr_i32 s54, s60, 31 // (op 8965) 
s_cmp_eq_i32 s15, 0 // (op 8965) 
s_mov_b32 s59, scc // (op 8965) 
s_cmp_eq_i32 s59, 0 // (op 8965) 
s_cselect_b32 s61, 0, -1 // (op 8965) 

s_and_b32 s59, s46, 31 // (op 8965) 
s_lshl_b32 s62, 1, s59 // (op 8965) 
s_add_i32 s63, s62, s61 // (op 8965) 


s_and_b32 s61, s54, s63 // (op 8965) 


s_add_i32 s54, s60, s61 // (op 8965) 


s_ashr_i32 s60, s54, s59 // (op 8965) 


s_lshl_b32 s54, s46, 24 // (op 8965) 
s_ashr_i32 s59, s54, 31 // (op 8965) 

s_xor_b32 s54, s60, s59 // (op 8965) 

s_sub_i32 s60, s54, s59 // (op 8965) 


s_mul_i32 s54, s45, s44 // (op 8965) 
s_add_u32 s62, s8, 128 // (op 8965) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 8965) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 8965) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 8965) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 8965) 

s_sub_i32 s59, s62, s54 // (op 8965) 

s_mul_i32 s61, s60, s59 // (op 8965) 

s_sub_u32 s59, s58, s61 // (op 8965) 


s_add_u32 s58, s59, s54 // (op 8965) 


s_cmp_eq_i32 s53, 0 // (op 8965) 
s_mov_b32 s54, scc // (op 8965) 

s_cmp_eq_i32 s54, 0 // (op 8965) 
s_cselect_b32 s53, s58, s1 // (op 8965) 


s_mul_i32 s1, s51, s56 // (op 8965) 


v_lshl_add_u32 v13, s53, 7, v12 // (op 8965) 


s_sub_i32 s51, s18, s1 // (op 8965) 


s_cmp_eq_i32 s54, 0 // (op 8965) 
s_cselect_b32 s1, s60, s51 // (op 8965) 



v_lshl_add_u32 v12, s1, 7, v11 // (op 8965) 


v_mul_lo_u32 v11, v12, s2 // (op 8965) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 8965) 


s_add_u32 s56, s6, 128 // (op 9222) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 9222) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9222) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9222) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 9222) 

s_mul_i32 s1, s44, s56 // (op 9222) 
s_mul_i32 s18, s48, s1 // (op 9222) 

s_mul_i32 s1, s55, s18 // (op 9222) 

s_sub_u32 s18, s16, s1 // (op 9222) 

s_mov_b32 s1, s18 // (op 9222) 
s_mul_hi_i32 s51, s1, s15 // (op 9222) 
s_add_i32 s53, s51, s1 // (op 9222) 


s_ashr_i32 s1, s53, 31 // (op 9222) 
s_cmp_eq_i32 s15, 0 // (op 9222) 
s_mov_b32 s51, scc // (op 9222) 
s_cmp_eq_i32 s51, 0 // (op 9222) 
s_cselect_b32 s54, 0, -1 // (op 9222) 

s_and_b32 s51, s46, 31 // (op 9222) 
s_lshl_b32 s58, 1, s51 // (op 9222) 
s_add_i32 s59, s58, s54 // (op 9222) 


s_and_b32 s54, s1, s59 // (op 9222) 


s_add_i32 s1, s53, s54 // (op 9222) 


s_ashr_i32 s53, s1, s51 // (op 9222) 


s_lshl_b32 s1, s46, 24 // (op 9222) 
s_ashr_i32 s51, s1, 31 // (op 9222) 

s_xor_b32 s1, s53, s51 // (op 9222) 

s_sub_i32 s53, s1, s51 // (op 9222) 


s_mul_i32 s1, s45, s44 // (op 9222) 
s_add_u32 s58, s8, 128 // (op 9222) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 9222) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 9222) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 9222) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 9222) 

s_sub_i32 s51, s58, s1 // (op 9222) 

s_mul_i32 s54, s53, s51 // (op 9222) 

s_sub_u32 s51, s18, s54 // (op 9222) 


s_add_u32 s18, s51, s1 // (op 9222) 


s_cmp_eq_i32 s55, 0 // (op 9222) 
s_mov_b32 s1, scc // (op 9222) 

s_cmp_eq_i32 s1, 0 // (op 9222) 
s_cselect_b32 s51, s18, s52 // (op 9222) 


s_mul_i32 s18, s19, s56 // (op 9222) 


v_lshl_add_u32 v11, s51, 7, v14 // (op 9222) 


s_sub_i32 s19, s0, s18 // (op 9222) 


s_cmp_eq_i32 s1, 0 // (op 9222) 
s_cselect_b32 s0, s53, s19 // (op 9222) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 9222) 


v_mul_lo_u32 v10, v13, s4 // (op 9222) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 9222) 


s_mul_i32 s1, s2, s7 // (op 8967) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8967) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8967) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8967) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8967) most significant: sum
s_add_u32 s1, s1, s19 // (op 8967) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8967) 

s_mul_i32 s1, s4, s7 // (op 9224) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 9224) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9224) least significant
s_mul_hi_u32 s52, s4, s6 // (op 9224) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 9224) most significant: sum
s_add_u32 s1, s1, s52 // (op 9224) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 9224) 

s_mov_b32 s56, s20 // (op 8967) reg expression
s_mov_b32 s57, s21 // (op 8967) reg expression
s_mov_b32 s58, s18 // (op 8967) reg expression
s_mov_b32 s59, 131072 // (op 8967) call()

s_mov_b32 s60, s22 // (op 9224) reg expression

s_mov_b32 s61, s23 // (op 9224) reg expression
s_mov_b32 s62, s52 // (op 9224) reg expression
s_mov_b32 s63, 131072 // (op 9224) call()


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
v_accvgpr_write a16, 0.00000 // (op 956) call()
v_accvgpr_write a17, 0.00000 // (op 956) call()
v_accvgpr_write a18, 0.00000 // (op 956) call()
v_accvgpr_write a19, 0.00000 // (op 956) call()
v_accvgpr_write a20, 0.00000 // (op 1265) call()
v_accvgpr_write a21, 0.00000 // (op 1265) call()
v_accvgpr_write a22, 0.00000 // (op 1265) call()
v_accvgpr_write a23, 0.00000 // (op 1265) call()
v_accvgpr_write a24, 0.00000 // (op 1266) call()
v_accvgpr_write a25, 0.00000 // (op 1266) call()
v_accvgpr_write a26, 0.00000 // (op 1266) call()
v_accvgpr_write a27, 0.00000 // (op 1266) call()
v_accvgpr_write a28, 0.00000 // (op 1267) call()
v_accvgpr_write a29, 0.00000 // (op 1267) call()
v_accvgpr_write a30, 0.00000 // (op 1267) call()
v_accvgpr_write a31, 0.00000 // (op 1267) call()
v_accvgpr_write a32, 0.00000 // (op 1726) call()
v_accvgpr_write a33, 0.00000 // (op 1726) call()
v_accvgpr_write a34, 0.00000 // (op 1726) call()
v_accvgpr_write a35, 0.00000 // (op 1726) call()
v_accvgpr_write a36, 0.00000 // (op 2035) call()
v_accvgpr_write a37, 0.00000 // (op 2035) call()
v_accvgpr_write a38, 0.00000 // (op 2035) call()
v_accvgpr_write a39, 0.00000 // (op 2035) call()
v_accvgpr_write a40, 0.00000 // (op 2036) call()
v_accvgpr_write a41, 0.00000 // (op 2036) call()
v_accvgpr_write a42, 0.00000 // (op 2036) call()
v_accvgpr_write a43, 0.00000 // (op 2036) call()
v_accvgpr_write a44, 0.00000 // (op 2037) call()
v_accvgpr_write a45, 0.00000 // (op 2037) call()
v_accvgpr_write a46, 0.00000 // (op 2037) call()
v_accvgpr_write a47, 0.00000 // (op 2037) call()
v_accvgpr_write a48, 0.00000 // (op 2496) call()
v_accvgpr_write a49, 0.00000 // (op 2496) call()
v_accvgpr_write a50, 0.00000 // (op 2496) call()
v_accvgpr_write a51, 0.00000 // (op 2496) call()
v_accvgpr_write a52, 0.00000 // (op 2805) call()
v_accvgpr_write a53, 0.00000 // (op 2805) call()
v_accvgpr_write a54, 0.00000 // (op 2805) call()
v_accvgpr_write a55, 0.00000 // (op 2805) call()
v_accvgpr_write a56, 0.00000 // (op 2806) call()
v_accvgpr_write a57, 0.00000 // (op 2806) call()
v_accvgpr_write a58, 0.00000 // (op 2806) call()
v_accvgpr_write a59, 0.00000 // (op 2806) call()
v_accvgpr_write a60, 0.00000 // (op 2807) call()
v_accvgpr_write a61, 0.00000 // (op 2807) call()
v_accvgpr_write a62, 0.00000 // (op 2807) call()
v_accvgpr_write a63, 0.00000 // (op 2807) call()
s_mul_hi_i32 s0, s16, s11 // (op 8727) 
v_lshrrev_b32 v14, 1, v1 // (op 8727) 
s_add_i32 s1, s0, s16 // (op 8727) 

v_and_b32 v15, -2, v1 // (op 8727) 
s_ashr_i32 s0, s1, 31 // (op 8727) 
v_sub_u32 v16, v1, v15 // (op 8727) 

s_cmp_eq_i32 s11, 0 // (op 8727) 
s_mov_b32 s18, scc // (op 8727) 
v_lshlrev_b32 v15, 6, v16 // (op 8727) 

s_cmp_eq_i32 s18, 0 // (op 8727) 
s_cselect_b32 s19, 0, -1 // (op 8727) 

v_mov_b32 v17, 0 // (op 8727) convert
v_mov_b32 v16, v15 // (op 8727) convert

s_and_b32 s18, s12, 31 // (op 8727) 
s_lshl_b32 s20, 1, s18 // (op 8727) 
s_add_i32 s21, s20, s19 // (op 8727) 


s_and_b32 s19, s0, s21 // (op 8727) 


s_add_i32 s0, s1, s19 // (op 8727) 


s_ashr_i32 s1, s0, s18 // (op 8727) 


s_lshl_b32 s0, s12, 24 // (op 8727) 
s_ashr_i32 s18, s0, 31 // (op 8727) 

s_xor_b32 s0, s1, s18 // (op 8727) 

s_sub_i32 s1, s0, s18 // (op 8727) 


s_mul_hi_i32 s0, s1, s13 // (op 8727) 
s_add_i32 s18, s0, s1 // (op 8727) 

s_ashr_i32 s0, s18, 31 // (op 8727) 
s_cmp_eq_i32 s13, 0 // (op 8727) 
s_mov_b32 s19, scc // (op 8727) 
s_cmp_eq_i32 s19, 0 // (op 8727) 
s_cselect_b32 s20, 0, -1 // (op 8727) 

s_and_b32 s19, s14, 31 // (op 8727) 
s_lshl_b32 s21, 1, s19 // (op 8727) 
s_add_i32 s22, s21, s20 // (op 8727) 


s_and_b32 s23, s0, s22 // (op 8727) 


s_add_i32 s0, s18, s23 // (op 8727) 


s_ashr_i32 s18, s0, s19 // (op 8727) 


s_lshl_b32 s0, s14, 24 // (op 8727) 
s_ashr_i32 s19, s0, 31 // (op 8727) 

s_xor_b32 s0, s18, s19 // (op 8727) 

s_sub_i32 s18, s0, s19 // (op 8727) 


s_mul_i32 s0, s1, s44 // (op 8727) 

s_sub_u32 s1, s16, s0 // (op 8727) 

s_mul_i32 s0, s18, s44 // (op 8727) 
s_add_u32 s0, s0, s1 // (op 8727) 


s_mul_hi_i32 s1, s16, s10 // (op 8727) 
s_add_i32 s18, s1, s16 // (op 8727) 

s_ashr_i32 s1, s18, 31 // (op 8727) 
s_cmp_eq_i32 s10, 0 // (op 8727) 
s_mov_b32 s19, scc // (op 8727) 
s_cmp_eq_i32 s19, 0 // (op 8727) 
s_cselect_b32 s20, 0, -1 // (op 8727) 

s_and_b32 s19, s47, 31 // (op 8727) 
s_lshl_b32 s21, 1, s19 // (op 8727) 
s_add_i32 s22, s21, s20 // (op 8727) 


s_and_b32 s23, s1, s22 // (op 8727) 


s_add_i32 s1, s18, s23 // (op 8727) 


s_ashr_i32 s18, s1, s19 // (op 8727) 


s_lshl_b32 s1, s47, 24 // (op 8727) 
s_ashr_i32 s19, s1, 31 // (op 8727) 

s_xor_b32 s1, s18, s19 // (op 8727) 

s_sub_i32 s18, s1, s19 // (op 8727) 


s_add_u32 s22, s6, 128 // (op 8727) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8727) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 8727) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 8727) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 8727) 

s_mul_i32 s1, s44, s22 // (op 8727) 

s_mul_i32 s19, s48, s1 // (op 8727) 

s_mul_i32 s1, s18, s19 // (op 8727) 

s_sub_u32 s19, s16, s1 // (op 8727) 

s_mov_b32 s1, s19 // (op 8727) 
s_mul_hi_i32 s20, s1, s15 // (op 8727) 
s_add_i32 s21, s20, s1 // (op 8727) 


s_ashr_i32 s1, s21, 31 // (op 8727) 
s_cmp_eq_i32 s15, 0 // (op 8727) 
s_mov_b32 s20, scc // (op 8727) 
s_cmp_eq_i32 s20, 0 // (op 8727) 
s_cselect_b32 s22, 0, -1 // (op 8727) 

s_and_b32 s20, s46, 31 // (op 8727) 
s_lshl_b32 s23, 1, s20 // (op 8727) 
s_add_i32 s49, s23, s22 // (op 8727) 


s_and_b32 s22, s1, s49 // (op 8727) 


s_add_i32 s1, s21, s22 // (op 8727) 


s_ashr_i32 s21, s1, s20 // (op 8727) 


s_lshl_b32 s1, s46, 24 // (op 8727) 
s_ashr_i32 s20, s1, 31 // (op 8727) 

s_xor_b32 s1, s21, s20 // (op 8727) 

s_sub_i32 s21, s1, s20 // (op 8727) 


s_mul_i32 s1, s45, s44 // (op 8727) 
s_add_u32 s22, s8, 128 // (op 8727) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 8727) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 8727) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 8727) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 7 // (op 8727) 

s_sub_i32 s20, s22, s1 // (op 8727) 

s_mul_i32 s22, s21, s20 // (op 8727) 


s_sub_u32 s23, s19, s22 // (op 8727) 


s_add_u32 s19, s23, s1 // (op 8727) 


s_cmp_eq_i32 s18, 0 // (op 8727) 
s_mov_b32 s1, scc // (op 8727) 

s_cmp_eq_i32 s1, 0 // (op 8727) 
s_cselect_b32 s18, s19, s0 // (op 8727) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 8727) 


v_mul_lo_u32 v19, v15, s25 // (op 8727) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8727) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 8727) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8727) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8727) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8727) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 8727) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 8727) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 8727) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 8727) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8727) 

v_mov_b32 v14, v16 // (op 8727) convert

s_mul_i32 s1, s9, 6 // (op 8729) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 8729) least significant
s_mul_hi_u32 s18, s8, 6 // (op 8729) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8729) most significant: sum

s_mul_i32 s19, s24, s1 // (op 8729) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 8729) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 8729) least significant
s_mul_hi_u32 s21, s24, s0 // (op 8729) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 8729) most significant: sum
s_add_u32 s19, s19, s21 // (op 8729) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8729) 

s_mov_b32 s20, s26 // (op 8729) reg expression
s_mov_b32 s21, s27 // (op 8729) reg expression
s_mov_b32 s22, s0 // (op 8729) reg expression
s_mov_b32 s23, 131072 // (op 8729) call()

s_mul_hi_i32 s0, s16, s11 // (op 8746) 
v_lshrrev_b32 v15, 1, v1 // (op 8746) 
s_add_i32 s1, s0, s16 // (op 8746) 

v_and_b32 v16, -2, v1 // (op 8746) 
s_ashr_i32 s0, s1, 31 // (op 8746) 
v_sub_u32 v17, v1, v16 // (op 8746) 

s_cmp_eq_i32 s11, 0 // (op 8746) 
s_mov_b32 s18, scc // (op 8746) 
v_lshlrev_b32 v16, 6, v17 // (op 8746) 

s_cmp_eq_i32 s18, 0 // (op 8746) 
s_cselect_b32 s19, 0, -1 // (op 8746) 

v_mov_b32 v19, 0 // (op 8746) convert
v_mov_b32 v18, v16 // (op 8746) convert

s_and_b32 s18, s12, 31 // (op 8746) 
s_lshl_b32 s49, 1, s18 // (op 8746) 
s_add_i32 s50, s49, s19 // (op 8746) 


s_and_b32 s19, s0, s50 // (op 8746) 


s_add_i32 s0, s1, s19 // (op 8746) 


s_ashr_i32 s1, s0, s18 // (op 8746) 


s_lshl_b32 s0, s12, 24 // (op 8746) 
s_ashr_i32 s18, s0, 31 // (op 8746) 

s_xor_b32 s0, s1, s18 // (op 8746) 

s_sub_i32 s1, s0, s18 // (op 8746) 


s_mul_hi_i32 s0, s1, s13 // (op 8746) 
s_add_i32 s18, s0, s1 // (op 8746) 

s_ashr_i32 s0, s18, 31 // (op 8746) 
s_cmp_eq_i32 s13, 0 // (op 8746) 
s_mov_b32 s19, scc // (op 8746) 
s_cmp_eq_i32 s19, 0 // (op 8746) 
s_cselect_b32 s49, 0, -1 // (op 8746) 

s_and_b32 s19, s14, 31 // (op 8746) 
s_lshl_b32 s50, 1, s19 // (op 8746) 
s_add_i32 s51, s50, s49 // (op 8746) 


s_and_b32 s49, s0, s51 // (op 8746) 


s_add_i32 s0, s18, s49 // (op 8746) 


s_ashr_i32 s18, s0, s19 // (op 8746) 


s_lshl_b32 s0, s14, 24 // (op 8746) 
s_ashr_i32 s19, s0, 31 // (op 8746) 

s_xor_b32 s0, s18, s19 // (op 8746) 

s_sub_i32 s18, s0, s19 // (op 8746) 


s_mul_i32 s0, s1, s44 // (op 8746) 

s_sub_u32 s1, s16, s0 // (op 8746) 

s_mul_i32 s0, s18, s44 // (op 8746) 
s_add_u32 s0, s0, s1 // (op 8746) 


s_mul_hi_i32 s1, s16, s10 // (op 8746) 
s_add_i32 s18, s1, s16 // (op 8746) 

s_ashr_i32 s1, s18, 31 // (op 8746) 
s_cmp_eq_i32 s10, 0 // (op 8746) 
s_mov_b32 s19, scc // (op 8746) 
s_cmp_eq_i32 s19, 0 // (op 8746) 
s_cselect_b32 s49, 0, -1 // (op 8746) 

s_and_b32 s19, s47, 31 // (op 8746) 
s_lshl_b32 s50, 1, s19 // (op 8746) 
s_add_i32 s51, s50, s49 // (op 8746) 


s_and_b32 s49, s1, s51 // (op 8746) 


s_add_i32 s1, s18, s49 // (op 8746) 


s_ashr_i32 s18, s1, s19 // (op 8746) 


s_lshl_b32 s1, s47, 24 // (op 8746) 
s_ashr_i32 s19, s1, 31 // (op 8746) 

s_xor_b32 s1, s18, s19 // (op 8746) 

s_sub_i32 s18, s1, s19 // (op 8746) 


s_add_u32 s54, s6, 128 // (op 8746) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 8746) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8746) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8746) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 8746) 

s_mul_i32 s1, s44, s50 // (op 8746) 

s_mul_i32 s19, s48, s1 // (op 8746) 

s_mul_i32 s1, s18, s19 // (op 8746) 

s_sub_u32 s19, s16, s1 // (op 8746) 

s_mov_b32 s1, s19 // (op 8746) 
s_mul_hi_i32 s49, s1, s15 // (op 8746) 
s_add_i32 s50, s49, s1 // (op 8746) 


s_ashr_i32 s1, s50, 31 // (op 8746) 
s_cmp_eq_i32 s15, 0 // (op 8746) 
s_mov_b32 s49, scc // (op 8746) 
s_cmp_eq_i32 s49, 0 // (op 8746) 
s_cselect_b32 s51, 0, -1 // (op 8746) 

s_and_b32 s49, s46, 31 // (op 8746) 
s_lshl_b32 s52, 1, s49 // (op 8746) 
s_add_i32 s53, s52, s51 // (op 8746) 


s_and_b32 s51, s1, s53 // (op 8746) 


s_add_i32 s1, s50, s51 // (op 8746) 


s_ashr_i32 s50, s1, s49 // (op 8746) 


s_lshl_b32 s1, s46, 24 // (op 8746) 
s_ashr_i32 s49, s1, 31 // (op 8746) 

s_xor_b32 s1, s50, s49 // (op 8746) 

s_sub_i32 s50, s1, s49 // (op 8746) 


s_mul_i32 s1, s45, s44 // (op 8746) 
s_add_u32 s54, s8, 128 // (op 8746) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 8746) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8746) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8746) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 8746) 

s_sub_i32 s49, s54, s1 // (op 8746) 

s_mul_i32 s51, s50, s49 // (op 8746) 


s_sub_u32 s49, s19, s51 // (op 8746) 


s_add_u32 s19, s49, s1 // (op 8746) 


s_cmp_eq_i32 s18, 0 // (op 8746) 
s_mov_b32 s1, scc // (op 8746) 

s_cmp_eq_i32 s1, 0 // (op 8746) 
s_cselect_b32 s18, s19, s0 // (op 8746) 



v_lshl_add_u32 v16, s18, 7, v15 // (op 8746) 


v_mul_lo_u32 v21, v16, s25 // (op 8746) most significant: low of low * high
v_mul_hi_u32 v15, v16, s24 // (op 8746) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 8746) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8746) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8746) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8746) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 8746) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 8746) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 8746) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 8746) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 8746) 

v_mov_b32 v15, v16 // (op 8746) convert

s_mul_i32 s1, s9, 6 // (op 8748) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 8748) least significant
s_mul_hi_u32 s18, s8, 6 // (op 8748) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8748) most significant: sum

s_mul_i32 s19, s24, s1 // (op 8748) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 8748) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 8748) least significant
s_mul_hi_u32 s50, s24, s0 // (op 8748) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8748) most significant: sum
s_add_u32 s19, s19, s50 // (op 8748) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8748) 

s_mov_b32 s20, s26 // (op 8748) reg expression
s_mov_b32 s21, s27 // (op 8748) reg expression
s_mov_b32 s22, s0 // (op 8748) reg expression
s_mov_b32 s23, 131072 // (op 8748) call()

s_mul_hi_i32 s0, s16, s11 // (op 8791) 
v_and_b32 v16, -2, v1 // (op 8791) 
s_add_i32 s1, s0, s16 // (op 8791) 

v_sub_u32 v17, v1, v16 // (op 8791) 

s_ashr_i32 s0, s1, 31 // (op 8791) 
v_lshlrev_b32 v16, 6, v17 // (op 8791) 

s_cmp_eq_i32 s11, 0 // (op 8791) 
s_mov_b32 s18, scc // (op 8791) 
v_lshrrev_b32 v17, 1, v1 // (op 8791) 
s_cmp_eq_i32 s18, 0 // (op 8791) 
s_cselect_b32 s19, 0, -1 // (op 8791) 

s_and_b32 s18, s12, 31 // (op 8791) 
s_lshl_b32 s49, 1, s18 // (op 8791) 
s_add_i32 s50, s49, s19 // (op 8791) 


s_and_b32 s19, s0, s50 // (op 8791) 


s_add_i32 s0, s1, s19 // (op 8791) 


s_ashr_i32 s1, s0, s18 // (op 8791) 


s_lshl_b32 s0, s12, 24 // (op 8791) 
s_ashr_i32 s18, s0, 31 // (op 8791) 

s_xor_b32 s0, s1, s18 // (op 8791) 

s_sub_i32 s1, s0, s18 // (op 8791) 


s_mul_hi_i32 s0, s1, s13 // (op 8791) 
s_add_i32 s18, s0, s1 // (op 8791) 

s_ashr_i32 s0, s18, 31 // (op 8791) 
s_cmp_eq_i32 s13, 0 // (op 8791) 
s_mov_b32 s19, scc // (op 8791) 
s_cmp_eq_i32 s19, 0 // (op 8791) 
s_cselect_b32 s49, 0, -1 // (op 8791) 

s_and_b32 s19, s14, 31 // (op 8791) 
s_lshl_b32 s50, 1, s19 // (op 8791) 
s_add_i32 s51, s50, s49 // (op 8791) 


s_and_b32 s49, s0, s51 // (op 8791) 


s_add_i32 s0, s18, s49 // (op 8791) 


s_ashr_i32 s18, s0, s19 // (op 8791) 


s_lshl_b32 s0, s14, 24 // (op 8791) 
s_ashr_i32 s19, s0, 31 // (op 8791) 

s_xor_b32 s0, s18, s19 // (op 8791) 

s_sub_i32 s18, s0, s19 // (op 8791) 


s_add_u32 s54, s6, 128 // (op 8791) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 8791) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8791) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8791) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 8791) 

s_mul_i32 s0, s18, s50 // (op 8791) 

s_sub_i32 s49, s1, s0 // (op 8791) 


s_mul_hi_i32 s0, s16, s10 // (op 8791) 
s_add_i32 s1, s0, s16 // (op 8791) 

s_ashr_i32 s0, s1, 31 // (op 8791) 
s_cmp_eq_i32 s10, 0 // (op 8791) 
s_mov_b32 s18, scc // (op 8791) 
s_cmp_eq_i32 s18, 0 // (op 8791) 
s_cselect_b32 s19, 0, -1 // (op 8791) 

s_and_b32 s18, s47, 31 // (op 8791) 
s_lshl_b32 s52, 1, s18 // (op 8791) 
s_add_i32 s53, s52, s19 // (op 8791) 


s_and_b32 s19, s0, s53 // (op 8791) 


s_add_i32 s0, s1, s19 // (op 8791) 


s_ashr_i32 s1, s0, s18 // (op 8791) 


s_lshl_b32 s0, s47, 24 // (op 8791) 
s_ashr_i32 s18, s0, 31 // (op 8791) 

s_xor_b32 s0, s1, s18 // (op 8791) 

s_sub_i32 s1, s0, s18 // (op 8791) 


s_mul_i32 s0, s44, s50 // (op 8791) 

s_mul_i32 s18, s48, s0 // (op 8791) 

s_mul_i32 s0, s1, s18 // (op 8791) 

s_sub_u32 s18, s16, s0 // (op 8791) 

s_mul_hi_i32 s0, s18, s15 // (op 8791) 
s_add_i32 s19, s0, s18 // (op 8791) 


s_ashr_i32 s0, s19, 31 // (op 8791) 
s_cmp_eq_i32 s15, 0 // (op 8791) 
s_mov_b32 s18, scc // (op 8791) 
s_cmp_eq_i32 s18, 0 // (op 8791) 
s_cselect_b32 s50, 0, -1 // (op 8791) 

s_and_b32 s18, s46, 31 // (op 8791) 
s_lshl_b32 s51, 1, s18 // (op 8791) 
s_add_i32 s52, s51, s50 // (op 8791) 


s_and_b32 s50, s0, s52 // (op 8791) 


s_add_i32 s0, s19, s50 // (op 8791) 


s_ashr_i32 s19, s0, s18 // (op 8791) 


s_lshl_b32 s0, s46, 24 // (op 8791) 
s_ashr_i32 s18, s0, 31 // (op 8791) 

s_xor_b32 s0, s19, s18 // (op 8791) 

s_sub_i32 s19, s0, s18 // (op 8791) 


s_cmp_eq_i32 s1, 0 // (op 8791) 
s_mov_b32 s0, scc // (op 8791) 

s_cmp_eq_i32 s0, 0 // (op 8791) 
s_cselect_b32 s1, s19, s49 // (op 8791) 



v_lshl_add_u32 v18, s1, 7, v16 // (op 8791) 


v_mov_b32 v21, 0 // (op 8791) convert
v_mov_b32 v20, v18 // (op 8791) convert

v_mul_lo_u32 v19, v17, s29 // (op 8791) most significant: low of low * high
v_mul_hi_u32 v16, v17, s28 // (op 8791) most significant: high of low * low
v_mul_lo_u32 v18, v17, s28 // (op 8791) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8791) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 8791) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 8791) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 8791) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 8791) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 8791) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 8791) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 8791) 

v_mov_b32 v16, v18 // (op 8791) convert

s_mul_i32 s1, s33, 6 // (op 8793) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 8793) least significant
s_mul_hi_u32 s18, s32, 6 // (op 8793) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8793) most significant: sum

s_mul_i32 s19, s28, s1 // (op 8793) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 8793) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 8793) least significant
s_mul_hi_u32 s50, s28, s0 // (op 8793) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8793) most significant: sum
s_add_u32 s19, s19, s50 // (op 8793) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8793) 

s_mov_b32 s52, s30 // (op 8793) reg expression
s_mov_b32 s53, s31 // (op 8793) reg expression
s_mov_b32 s54, s0 // (op 8793) reg expression
s_mov_b32 s55, 131072 // (op 8793) call()

s_mul_hi_i32 s0, s16, s11 // (op 8810) 
v_and_b32 v17, -2, v1 // (op 8810) 
s_add_i32 s1, s0, s16 // (op 8810) 

v_sub_u32 v18, v1, v17 // (op 8810) 

s_ashr_i32 s0, s1, 31 // (op 8810) 
v_lshlrev_b32 v17, 6, v18 // (op 8810) 

s_cmp_eq_i32 s11, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
v_lshrrev_b32 v18, 1, v1 // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s19, 0, -1 // (op 8810) 

s_and_b32 s18, s12, 31 // (op 8810) 
s_lshl_b32 s49, 1, s18 // (op 8810) 
s_add_i32 s50, s49, s19 // (op 8810) 


s_and_b32 s19, s0, s50 // (op 8810) 


s_add_i32 s0, s1, s19 // (op 8810) 


s_ashr_i32 s1, s0, s18 // (op 8810) 


s_lshl_b32 s0, s12, 24 // (op 8810) 
s_ashr_i32 s18, s0, 31 // (op 8810) 

s_xor_b32 s0, s1, s18 // (op 8810) 

s_sub_i32 s1, s0, s18 // (op 8810) 


s_mul_hi_i32 s0, s1, s13 // (op 8810) 
s_add_i32 s18, s0, s1 // (op 8810) 

s_ashr_i32 s0, s18, 31 // (op 8810) 
s_cmp_eq_i32 s13, 0 // (op 8810) 
s_mov_b32 s19, scc // (op 8810) 
s_cmp_eq_i32 s19, 0 // (op 8810) 
s_cselect_b32 s49, 0, -1 // (op 8810) 

s_and_b32 s19, s14, 31 // (op 8810) 
s_lshl_b32 s50, 1, s19 // (op 8810) 
s_add_i32 s51, s50, s49 // (op 8810) 


s_and_b32 s49, s0, s51 // (op 8810) 


s_add_i32 s0, s18, s49 // (op 8810) 


s_ashr_i32 s18, s0, s19 // (op 8810) 


s_lshl_b32 s0, s14, 24 // (op 8810) 
s_ashr_i32 s19, s0, 31 // (op 8810) 

s_xor_b32 s0, s18, s19 // (op 8810) 

s_sub_i32 s18, s0, s19 // (op 8810) 


s_add_u32 s50, s6, 128 // (op 8810) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8810) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 8810) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 8810) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 8810) 

s_mul_i32 s0, s18, s50 // (op 8810) 

s_sub_i32 s49, s1, s0 // (op 8810) 


s_mul_hi_i32 s0, s16, s10 // (op 8810) 
s_add_i32 s1, s0, s16 // (op 8810) 

s_ashr_i32 s0, s1, 31 // (op 8810) 
s_cmp_eq_i32 s10, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s19, 0, -1 // (op 8810) 

s_and_b32 s18, s47, 31 // (op 8810) 
s_lshl_b32 s64, 1, s18 // (op 8810) 
s_add_i32 s65, s64, s19 // (op 8810) 


s_and_b32 s19, s0, s65 // (op 8810) 


s_add_i32 s0, s1, s19 // (op 8810) 


s_ashr_i32 s1, s0, s18 // (op 8810) 


s_lshl_b32 s0, s47, 24 // (op 8810) 
s_ashr_i32 s18, s0, 31 // (op 8810) 

s_xor_b32 s0, s1, s18 // (op 8810) 

s_sub_i32 s1, s0, s18 // (op 8810) 


s_mul_i32 s0, s44, s50 // (op 8810) 

s_mul_i32 s18, s48, s0 // (op 8810) 

s_mul_i32 s0, s1, s18 // (op 8810) 

s_sub_u32 s18, s16, s0 // (op 8810) 

s_mul_hi_i32 s0, s18, s15 // (op 8810) 
s_add_i32 s19, s0, s18 // (op 8810) 


s_ashr_i32 s0, s19, 31 // (op 8810) 
s_cmp_eq_i32 s15, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s50, 0, -1 // (op 8810) 

s_and_b32 s18, s46, 31 // (op 8810) 
s_lshl_b32 s51, 1, s18 // (op 8810) 
s_add_i32 s64, s51, s50 // (op 8810) 


s_and_b32 s50, s0, s64 // (op 8810) 


s_add_i32 s0, s19, s50 // (op 8810) 


s_ashr_i32 s19, s0, s18 // (op 8810) 


s_lshl_b32 s0, s46, 24 // (op 8810) 
s_ashr_i32 s18, s0, 31 // (op 8810) 

s_xor_b32 s0, s19, s18 // (op 8810) 

s_sub_i32 s19, s0, s18 // (op 8810) 


s_cmp_eq_i32 s1, 0 // (op 8810) 
s_mov_b32 s0, scc // (op 8810) 

s_cmp_eq_i32 s0, 0 // (op 8810) 
s_cselect_b32 s1, s19, s49 // (op 8810) 



v_lshl_add_u32 v19, s1, 7, v17 // (op 8810) 


v_mov_b32 v21, 0 // (op 8810) convert
v_mov_b32 v20, v19 // (op 8810) convert

v_mul_lo_u32 v23, v18, s29 // (op 8810) most significant: low of low * high
v_mul_hi_u32 v17, v18, s28 // (op 8810) most significant: high of low * low
v_mul_lo_u32 v22, v18, s28 // (op 8810) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 8810) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 8810) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 8810) most significant half



v_mul_lo_u32 v21, v23, 6 // (op 8810) most significant: low of high * low
v_mul_hi_u32 v17, v22, 6 // (op 8810) most significant: high of low * low
v_mul_lo_u32 v20, v22, 6 // (op 8810) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8810) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 8810) 

v_mov_b32 v17, v18 // (op 8810) convert

s_mul_i32 s1, s33, 6 // (op 8812) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 8812) least significant
s_mul_hi_u32 s18, s32, 6 // (op 8812) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8812) most significant: sum

s_mul_i32 s19, s28, s1 // (op 8812) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 8812) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 8812) least significant
s_mul_hi_u32 s50, s28, s0 // (op 8812) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8812) most significant: sum
s_add_u32 s19, s19, s50 // (op 8812) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8812) 

s_mov_b32 s52, s30 // (op 8812) reg expression
s_mov_b32 s53, s31 // (op 8812) reg expression
s_mov_b32 s54, s0 // (op 8812) reg expression
s_mov_b32 s55, 131072 // (op 8812) call()

s_mul_hi_i32 s0, s16, s11 // (op 8855) 
v_lshrrev_b32 v18, 2, v1 // (op 8855) 
s_add_i32 s1, s0, s16 // (op 8855) 

v_lshlrev_b32 v19, 1, v18 // (op 8855) 

s_ashr_i32 s0, s1, 31 // (op 8855) 
v_and_b32 v18, -4, v1 // (op 8855) 
s_cmp_eq_i32 s11, 0 // (op 8855) 
s_mov_b32 s18, scc // (op 8855) 
v_sub_u32 v20, v1, v18 // (op 8855) 

s_cmp_eq_i32 s18, 0 // (op 8855) 
s_cselect_b32 s19, 0, -1 // (op 8855) 

s_and_b32 s18, s12, 31 // (op 8855) 
s_lshl_b32 s49, 1, s18 // (op 8855) 
s_add_i32 s50, s49, s19 // (op 8855) 


s_and_b32 s19, s0, s50 // (op 8855) 


s_add_i32 s0, s1, s19 // (op 8855) 


s_ashr_i32 s1, s0, s18 // (op 8855) 


s_lshl_b32 s0, s12, 24 // (op 8855) 
s_ashr_i32 s18, s0, 31 // (op 8855) 

s_xor_b32 s0, s1, s18 // (op 8855) 

s_sub_i32 s1, s0, s18 // (op 8855) 


s_mul_hi_i32 s0, s1, s13 // (op 8855) 
s_add_i32 s18, s0, s1 // (op 8855) 

s_ashr_i32 s0, s18, 31 // (op 8855) 
s_cmp_eq_i32 s13, 0 // (op 8855) 
s_mov_b32 s19, scc // (op 8855) 
s_cmp_eq_i32 s19, 0 // (op 8855) 
s_cselect_b32 s49, 0, -1 // (op 8855) 

s_and_b32 s19, s14, 31 // (op 8855) 
s_lshl_b32 s50, 1, s19 // (op 8855) 
s_add_i32 s51, s50, s49 // (op 8855) 


s_and_b32 s49, s0, s51 // (op 8855) 


s_add_i32 s0, s18, s49 // (op 8855) 


s_ashr_i32 s18, s0, s19 // (op 8855) 


s_lshl_b32 s0, s14, 24 // (op 8855) 
s_ashr_i32 s19, s0, 31 // (op 8855) 

s_xor_b32 s0, s18, s19 // (op 8855) 

s_sub_i32 s18, s0, s19 // (op 8855) 


s_mul_i32 s0, s1, s44 // (op 8855) 

s_sub_u32 s1, s16, s0 // (op 8855) 

s_mul_i32 s0, s18, s44 // (op 8855) 
s_add_u32 s0, s0, s1 // (op 8855) 


s_mul_hi_i32 s1, s16, s10 // (op 8855) 
s_add_i32 s18, s1, s16 // (op 8855) 

s_ashr_i32 s1, s18, 31 // (op 8855) 
s_cmp_eq_i32 s10, 0 // (op 8855) 
s_mov_b32 s19, scc // (op 8855) 
s_cmp_eq_i32 s19, 0 // (op 8855) 
s_cselect_b32 s49, 0, -1 // (op 8855) 

s_and_b32 s19, s47, 31 // (op 8855) 
s_lshl_b32 s50, 1, s19 // (op 8855) 
s_add_i32 s51, s50, s49 // (op 8855) 


s_and_b32 s49, s1, s51 // (op 8855) 


s_add_i32 s1, s18, s49 // (op 8855) 


s_ashr_i32 s18, s1, s19 // (op 8855) 


s_lshl_b32 s1, s47, 24 // (op 8855) 
s_ashr_i32 s19, s1, 31 // (op 8855) 

s_xor_b32 s1, s18, s19 // (op 8855) 

s_sub_i32 s18, s1, s19 // (op 8855) 


s_add_u32 s50, s6, 128 // (op 8855) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8855) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 8855) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 8855) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 8855) 

s_mul_i32 s1, s44, s50 // (op 8855) 

s_mul_i32 s19, s48, s1 // (op 8855) 

s_mul_i32 s1, s18, s19 // (op 8855) 

s_sub_u32 s19, s16, s1 // (op 8855) 

s_mov_b32 s1, s19 // (op 8855) 
s_mul_hi_i32 s49, s1, s15 // (op 8855) 
s_add_i32 s50, s49, s1 // (op 8855) 


s_ashr_i32 s1, s50, 31 // (op 8855) 
s_cmp_eq_i32 s15, 0 // (op 8855) 
s_mov_b32 s49, scc // (op 8855) 
s_cmp_eq_i32 s49, 0 // (op 8855) 
s_cselect_b32 s51, 0, -1 // (op 8855) 

s_and_b32 s49, s46, 31 // (op 8855) 
s_lshl_b32 s64, 1, s49 // (op 8855) 
s_add_i32 s65, s64, s51 // (op 8855) 


s_and_b32 s51, s1, s65 // (op 8855) 


s_add_i32 s1, s50, s51 // (op 8855) 


s_ashr_i32 s50, s1, s49 // (op 8855) 


s_lshl_b32 s1, s46, 24 // (op 8855) 
s_ashr_i32 s49, s1, 31 // (op 8855) 

s_xor_b32 s1, s50, s49 // (op 8855) 

s_sub_i32 s50, s1, s49 // (op 8855) 


s_mul_i32 s1, s45, s44 // (op 8855) 
s_add_u32 s64, s8, 128 // (op 8855) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 8855) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 8855) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 8855) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 8855) 

s_sub_i32 s49, s64, s1 // (op 8855) 

s_mul_i32 s51, s50, s49 // (op 8855) 


s_sub_u32 s49, s19, s51 // (op 8855) 


s_add_u32 s19, s49, s1 // (op 8855) 


s_cmp_eq_i32 s18, 0 // (op 8855) 
s_mov_b32 s1, scc // (op 8855) 

s_cmp_eq_i32 s1, 0 // (op 8855) 
s_cselect_b32 s18, s19, s0 // (op 8855) 



v_lshl_add_u32 v18, s18, 7, v19 // (op 8855) 


v_mul_lo_u32 v19, v18, s34 // (op 8855) 
v_add_u32 v19, v19, v20 // (op 8855) 


s_mul_i32 s1, s34, s9 // (op 8857) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8857) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8857) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8857) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8857) most significant: sum
s_add_u32 s1, s1, s19 // (op 8857) most significant: sum


s_mov_b32 s64, s36 // (op 8857) reg expression
s_mov_b32 s65, s37 // (op 8857) reg expression
s_mov_b32 s66, s0 // (op 8857) reg expression
s_mov_b32 s67, 131072 // (op 8857) call()

s_mul_hi_i32 s0, s16, s11 // (op 8874) 
v_lshrrev_b32 v18, 2, v1 // (op 8874) 
s_add_i32 s1, s0, s16 // (op 8874) 

v_lshlrev_b32 v20, 1, v18 // (op 8874) 

s_ashr_i32 s0, s1, 31 // (op 8874) 
v_and_b32 v18, -4, v1 // (op 8874) 
s_cmp_eq_i32 s11, 0 // (op 8874) 
s_mov_b32 s18, scc // (op 8874) 
v_sub_u32 v21, v1, v18 // (op 8874) 

s_cmp_eq_i32 s18, 0 // (op 8874) 
s_cselect_b32 s19, 0, -1 // (op 8874) 

s_and_b32 s18, s12, 31 // (op 8874) 
s_lshl_b32 s49, 1, s18 // (op 8874) 
s_add_i32 s50, s49, s19 // (op 8874) 


s_and_b32 s19, s0, s50 // (op 8874) 


s_add_i32 s0, s1, s19 // (op 8874) 


s_ashr_i32 s1, s0, s18 // (op 8874) 


s_lshl_b32 s0, s12, 24 // (op 8874) 
s_ashr_i32 s18, s0, 31 // (op 8874) 

s_xor_b32 s0, s1, s18 // (op 8874) 

s_sub_i32 s1, s0, s18 // (op 8874) 


s_mul_hi_i32 s0, s1, s13 // (op 8874) 
s_add_i32 s18, s0, s1 // (op 8874) 

s_ashr_i32 s0, s18, 31 // (op 8874) 
s_cmp_eq_i32 s13, 0 // (op 8874) 
s_mov_b32 s19, scc // (op 8874) 
s_cmp_eq_i32 s19, 0 // (op 8874) 
s_cselect_b32 s49, 0, -1 // (op 8874) 

s_and_b32 s19, s14, 31 // (op 8874) 
s_lshl_b32 s50, 1, s19 // (op 8874) 
s_add_i32 s51, s50, s49 // (op 8874) 


s_and_b32 s49, s0, s51 // (op 8874) 


s_add_i32 s0, s18, s49 // (op 8874) 


s_ashr_i32 s18, s0, s19 // (op 8874) 


s_lshl_b32 s0, s14, 24 // (op 8874) 
s_ashr_i32 s19, s0, 31 // (op 8874) 

s_xor_b32 s0, s18, s19 // (op 8874) 

s_sub_i32 s18, s0, s19 // (op 8874) 


s_mul_i32 s0, s1, s44 // (op 8874) 

s_sub_u32 s1, s16, s0 // (op 8874) 

s_mul_i32 s0, s18, s44 // (op 8874) 
s_add_u32 s0, s0, s1 // (op 8874) 


s_mul_hi_i32 s1, s16, s10 // (op 8874) 
s_add_i32 s18, s1, s16 // (op 8874) 

s_ashr_i32 s1, s18, 31 // (op 8874) 
s_cmp_eq_i32 s10, 0 // (op 8874) 
s_mov_b32 s19, scc // (op 8874) 
s_cmp_eq_i32 s19, 0 // (op 8874) 
s_cselect_b32 s49, 0, -1 // (op 8874) 

s_and_b32 s19, s47, 31 // (op 8874) 
s_lshl_b32 s50, 1, s19 // (op 8874) 
s_add_i32 s51, s50, s49 // (op 8874) 


s_and_b32 s49, s1, s51 // (op 8874) 


s_add_i32 s1, s18, s49 // (op 8874) 


s_ashr_i32 s18, s1, s19 // (op 8874) 


s_lshl_b32 s1, s47, 24 // (op 8874) 
s_ashr_i32 s19, s1, 31 // (op 8874) 

s_xor_b32 s1, s18, s19 // (op 8874) 

s_sub_i32 s18, s1, s19 // (op 8874) 


s_add_u32 s50, s6, 128 // (op 8874) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8874) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 8874) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 8874) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 8874) 

s_mul_i32 s1, s44, s50 // (op 8874) 

s_mul_i32 s19, s48, s1 // (op 8874) 

s_mul_i32 s1, s18, s19 // (op 8874) 

s_sub_u32 s19, s16, s1 // (op 8874) 

s_mov_b32 s1, s19 // (op 8874) 
s_mul_hi_i32 s49, s1, s15 // (op 8874) 
s_add_i32 s50, s49, s1 // (op 8874) 


s_ashr_i32 s1, s50, 31 // (op 8874) 
s_cmp_eq_i32 s15, 0 // (op 8874) 
s_mov_b32 s49, scc // (op 8874) 
s_cmp_eq_i32 s49, 0 // (op 8874) 
s_cselect_b32 s51, 0, -1 // (op 8874) 

s_and_b32 s49, s46, 31 // (op 8874) 
s_lshl_b32 s68, 1, s49 // (op 8874) 
s_add_i32 s69, s68, s51 // (op 8874) 


s_and_b32 s51, s1, s69 // (op 8874) 


s_add_i32 s1, s50, s51 // (op 8874) 


s_ashr_i32 s50, s1, s49 // (op 8874) 


s_lshl_b32 s1, s46, 24 // (op 8874) 
s_ashr_i32 s49, s1, 31 // (op 8874) 

s_xor_b32 s1, s50, s49 // (op 8874) 

s_sub_i32 s50, s1, s49 // (op 8874) 


s_mul_i32 s1, s45, s44 // (op 8874) 
s_add_u32 s68, s8, 128 // (op 8874) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 8874) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 8874) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 8874) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 8874) 

s_sub_i32 s49, s68, s1 // (op 8874) 

s_mul_i32 s51, s50, s49 // (op 8874) 


s_sub_u32 s49, s19, s51 // (op 8874) 


s_add_u32 s19, s49, s1 // (op 8874) 


s_cmp_eq_i32 s18, 0 // (op 8874) 
s_mov_b32 s1, scc // (op 8874) 

s_cmp_eq_i32 s1, 0 // (op 8874) 
s_cselect_b32 s18, s19, s0 // (op 8874) 



v_lshl_add_u32 v18, s18, 7, v20 // (op 8874) 


v_mul_lo_u32 v20, v18, s34 // (op 8874) 
v_add_u32 v20, v20, v21 // (op 8874) 


s_mul_i32 s1, s34, s9 // (op 8876) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8876) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8876) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8876) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8876) most significant: sum
s_add_u32 s1, s1, s19 // (op 8876) most significant: sum


s_mov_b32 s64, s36 // (op 8876) reg expression
s_mov_b32 s65, s37 // (op 8876) reg expression
s_mov_b32 s66, s0 // (op 8876) reg expression
s_mov_b32 s67, 131072 // (op 8876) call()

s_mul_hi_i32 s0, s16, s11 // (op 8919) 
v_and_b32 v18, -128, v1 // (op 8919) 
s_add_i32 s1, s0, s16 // (op 8919) 

v_sub_u32 v21, v1, v18 // (op 8919) 

s_ashr_i32 s0, s1, 31 // (op 8919) 
v_lshrrev_b32 v18, 7, v1 // (op 8919) 
s_cmp_eq_i32 s11, 0 // (op 8919) 
s_mov_b32 s18, scc // (op 8919) 
v_lshlrev_b32 v22, 1, v18 // (op 8919) 

s_cmp_eq_i32 s18, 0 // (op 8919) 
s_cselect_b32 s19, 0, -1 // (op 8919) 

s_and_b32 s18, s12, 31 // (op 8919) 
s_lshl_b32 s49, 1, s18 // (op 8919) 
s_add_i32 s50, s49, s19 // (op 8919) 


s_and_b32 s19, s0, s50 // (op 8919) 


s_add_i32 s0, s1, s19 // (op 8919) 


s_ashr_i32 s1, s0, s18 // (op 8919) 


s_lshl_b32 s0, s12, 24 // (op 8919) 
s_ashr_i32 s18, s0, 31 // (op 8919) 

s_xor_b32 s0, s1, s18 // (op 8919) 

s_sub_i32 s1, s0, s18 // (op 8919) 


s_mul_hi_i32 s0, s1, s13 // (op 8919) 
s_add_i32 s18, s0, s1 // (op 8919) 

s_ashr_i32 s0, s18, 31 // (op 8919) 
s_cmp_eq_i32 s13, 0 // (op 8919) 
s_mov_b32 s19, scc // (op 8919) 
s_cmp_eq_i32 s19, 0 // (op 8919) 
s_cselect_b32 s49, 0, -1 // (op 8919) 

s_and_b32 s19, s14, 31 // (op 8919) 
s_lshl_b32 s50, 1, s19 // (op 8919) 
s_add_i32 s51, s50, s49 // (op 8919) 


s_and_b32 s49, s0, s51 // (op 8919) 


s_add_i32 s0, s18, s49 // (op 8919) 


s_ashr_i32 s18, s0, s19 // (op 8919) 


s_lshl_b32 s0, s14, 24 // (op 8919) 
s_ashr_i32 s19, s0, 31 // (op 8919) 

s_xor_b32 s0, s18, s19 // (op 8919) 

s_sub_i32 s18, s0, s19 // (op 8919) 


s_add_u32 s50, s6, 128 // (op 8919) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8919) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 8919) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 8919) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 8919) 

s_mul_i32 s0, s18, s50 // (op 8919) 

s_sub_i32 s49, s1, s0 // (op 8919) 


s_mul_hi_i32 s0, s16, s10 // (op 8919) 
s_add_i32 s1, s0, s16 // (op 8919) 

s_ashr_i32 s0, s1, 31 // (op 8919) 
s_cmp_eq_i32 s10, 0 // (op 8919) 
s_mov_b32 s18, scc // (op 8919) 
s_cmp_eq_i32 s18, 0 // (op 8919) 
s_cselect_b32 s19, 0, -1 // (op 8919) 

s_and_b32 s18, s47, 31 // (op 8919) 
s_lshl_b32 s68, 1, s18 // (op 8919) 
s_add_i32 s69, s68, s19 // (op 8919) 


s_and_b32 s19, s0, s69 // (op 8919) 


s_add_i32 s0, s1, s19 // (op 8919) 


s_ashr_i32 s1, s0, s18 // (op 8919) 


s_lshl_b32 s0, s47, 24 // (op 8919) 
s_ashr_i32 s18, s0, 31 // (op 8919) 

s_xor_b32 s0, s1, s18 // (op 8919) 

s_sub_i32 s1, s0, s18 // (op 8919) 


s_mul_i32 s0, s44, s50 // (op 8919) 

s_mul_i32 s18, s48, s0 // (op 8919) 

s_mul_i32 s0, s1, s18 // (op 8919) 

s_sub_u32 s18, s16, s0 // (op 8919) 

s_mul_hi_i32 s0, s18, s15 // (op 8919) 
s_add_i32 s19, s0, s18 // (op 8919) 


s_ashr_i32 s0, s19, 31 // (op 8919) 
s_cmp_eq_i32 s15, 0 // (op 8919) 
s_mov_b32 s18, scc // (op 8919) 
s_cmp_eq_i32 s18, 0 // (op 8919) 
s_cselect_b32 s50, 0, -1 // (op 8919) 

s_and_b32 s18, s46, 31 // (op 8919) 
s_lshl_b32 s51, 1, s18 // (op 8919) 
s_add_i32 s68, s51, s50 // (op 8919) 


s_and_b32 s50, s0, s68 // (op 8919) 


s_add_i32 s0, s19, s50 // (op 8919) 


s_ashr_i32 s19, s0, s18 // (op 8919) 


s_lshl_b32 s0, s46, 24 // (op 8919) 
s_ashr_i32 s18, s0, 31 // (op 8919) 

s_xor_b32 s0, s19, s18 // (op 8919) 

s_sub_i32 s19, s0, s18 // (op 8919) 


s_cmp_eq_i32 s1, 0 // (op 8919) 
s_mov_b32 s0, scc // (op 8919) 

s_cmp_eq_i32 s0, 0 // (op 8919) 
s_cselect_b32 s1, s19, s49 // (op 8919) 



v_lshl_add_u32 v18, s1, 7, v21 // (op 8919) 


v_mul_lo_u32 v21, v22, s38 // (op 8919) 
v_add_u32 v21, v21, v18 // (op 8919) 


s_mul_i32 s1, s38, s43 // (op 8921) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8921) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8921) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8921) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8921) most significant: sum
s_add_u32 s1, s1, s19 // (op 8921) most significant: sum


s_mov_b32 s68, s40 // (op 8921) reg expression
s_mov_b32 s69, s41 // (op 8921) reg expression
s_mov_b32 s70, s0 // (op 8921) reg expression
s_mov_b32 s71, 131072 // (op 8921) call()

s_mul_hi_i32 s0, s16, s11 // (op 8938) 
v_and_b32 v18, -128, v1 // (op 8938) 
s_add_i32 s1, s0, s16 // (op 8938) 

v_sub_u32 v22, v1, v18 // (op 8938) 

s_ashr_i32 s0, s1, 31 // (op 8938) 
v_lshrrev_b32 v18, 7, v1 // (op 8938) 
s_cmp_eq_i32 s11, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
v_lshlrev_b32 v23, 1, v18 // (op 8938) 

s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s19, 0, -1 // (op 8938) 

s_and_b32 s18, s12, 31 // (op 8938) 
s_lshl_b32 s49, 1, s18 // (op 8938) 
s_add_i32 s50, s49, s19 // (op 8938) 


s_and_b32 s19, s0, s50 // (op 8938) 


s_add_i32 s0, s1, s19 // (op 8938) 


s_ashr_i32 s1, s0, s18 // (op 8938) 


s_lshl_b32 s0, s12, 24 // (op 8938) 
s_ashr_i32 s18, s0, 31 // (op 8938) 

s_xor_b32 s0, s1, s18 // (op 8938) 

s_sub_i32 s1, s0, s18 // (op 8938) 


s_mul_hi_i32 s0, s1, s13 // (op 8938) 
s_add_i32 s18, s0, s1 // (op 8938) 

s_ashr_i32 s0, s18, 31 // (op 8938) 
s_cmp_eq_i32 s13, 0 // (op 8938) 
s_mov_b32 s19, scc // (op 8938) 
s_cmp_eq_i32 s19, 0 // (op 8938) 
s_cselect_b32 s49, 0, -1 // (op 8938) 

s_and_b32 s19, s14, 31 // (op 8938) 
s_lshl_b32 s50, 1, s19 // (op 8938) 
s_add_i32 s51, s50, s49 // (op 8938) 


s_and_b32 s49, s0, s51 // (op 8938) 


s_add_i32 s0, s18, s49 // (op 8938) 


s_ashr_i32 s18, s0, s19 // (op 8938) 


s_lshl_b32 s0, s14, 24 // (op 8938) 
s_ashr_i32 s19, s0, 31 // (op 8938) 

s_xor_b32 s0, s18, s19 // (op 8938) 

s_sub_i32 s18, s0, s19 // (op 8938) 


s_add_u32 s50, s6, 128 // (op 8938) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8938) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8938) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8938) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8938) 

s_mul_i32 s0, s18, s50 // (op 8938) 

s_sub_i32 s49, s1, s0 // (op 8938) 


s_mul_hi_i32 s0, s16, s10 // (op 8938) 
s_add_i32 s1, s0, s16 // (op 8938) 

s_ashr_i32 s0, s1, 31 // (op 8938) 
s_cmp_eq_i32 s10, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s19, 0, -1 // (op 8938) 

s_and_b32 s18, s47, 31 // (op 8938) 
s_lshl_b32 s72, 1, s18 // (op 8938) 
s_add_i32 s73, s72, s19 // (op 8938) 


s_and_b32 s19, s0, s73 // (op 8938) 


s_add_i32 s0, s1, s19 // (op 8938) 


s_ashr_i32 s1, s0, s18 // (op 8938) 


s_lshl_b32 s0, s47, 24 // (op 8938) 
s_ashr_i32 s18, s0, 31 // (op 8938) 

s_xor_b32 s0, s1, s18 // (op 8938) 

s_sub_i32 s1, s0, s18 // (op 8938) 


s_mul_i32 s0, s44, s50 // (op 8938) 

s_mul_i32 s18, s48, s0 // (op 8938) 

s_mul_i32 s0, s1, s18 // (op 8938) 

s_sub_u32 s18, s16, s0 // (op 8938) 

s_mul_hi_i32 s0, s18, s15 // (op 8938) 
s_add_i32 s19, s0, s18 // (op 8938) 


s_ashr_i32 s0, s19, 31 // (op 8938) 
s_cmp_eq_i32 s15, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s50, 0, -1 // (op 8938) 

s_and_b32 s18, s46, 31 // (op 8938) 
s_lshl_b32 s51, 1, s18 // (op 8938) 
s_add_i32 s72, s51, s50 // (op 8938) 


s_and_b32 s50, s0, s72 // (op 8938) 


s_add_i32 s0, s19, s50 // (op 8938) 


s_ashr_i32 s19, s0, s18 // (op 8938) 


s_lshl_b32 s0, s46, 24 // (op 8938) 
s_ashr_i32 s18, s0, 31 // (op 8938) 

s_xor_b32 s0, s19, s18 // (op 8938) 

s_sub_i32 s19, s0, s18 // (op 8938) 


s_cmp_eq_i32 s1, 0 // (op 8938) 
s_mov_b32 s0, scc // (op 8938) 

s_cmp_eq_i32 s0, 0 // (op 8938) 
s_cselect_b32 s1, s19, s49 // (op 8938) 



v_lshl_add_u32 v18, s1, 7, v22 // (op 8938) 


v_mul_lo_u32 v22, v23, s38 // (op 8938) 
v_add_u32 v22, v22, v18 // (op 8938) 


s_mul_i32 s1, s38, s43 // (op 8940) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8940) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8940) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8940) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8940) most significant: sum
s_add_u32 s1, s1, s19 // (op 8940) most significant: sum


s_mov_b32 s68, s40 // (op 8940) reg expression
s_mov_b32 s69, s41 // (op 8940) reg expression
s_mov_b32 s70, s0 // (op 8940) reg expression
s_mov_b32 s71, 131072 // (op 8940) call()



v_mov_b32 v18, v16 // (op 8315) reg expression

buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 8315) Load value
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen offset:12 // (op 8315) Load value
buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen offset:24 // (op 8315) Load value
buffer_load_dwordx3 v[36:38], v18, s[52:55], 0 offen offset:36 // (op 8315) Load value


v_mov_b64 v[40:41], s[38:39] // (op 8318) convert

v_mov_b32 v18, v21 // (op 8318) reg expression
buffer_load_ubyte v42, v18, s[68:71], 0 offen offset: 0 // (op 8318) Load value
v_add_u32 v18, v18, v40 // (op 8318) 
buffer_load_ubyte v43, v18, s[68:71], 0 offen offset: 0 // (op 8318) Load value





buffer_load_dwordx4 v[44:47], v14, s[20:23], 0 offen // (op 8321) Load value
buffer_load_dwordx2 v[48:49], v14, s[20:23], 0 offen offset:16 // (op 8321) Load value
buffer_load_dwordx4 v[50:53], v14, s[20:23], 0 offen offset:24 // (op 8321) Load value
buffer_load_dwordx2 v[54:55], v14, s[20:23], 0 offen offset:40 // (op 8321) Load value

v_mov_b64 v[40:41], s[34:35] // (op 8324) convert

v_mov_b32 v18, v19 // (op 8324) 
buffer_load_ubyte v56, v18, s[64:67], 0 offen offset: 0 // (op 8324) Load value
v_add_u32 v18, v18, v40 // (op 8324) 
buffer_load_ubyte v57, v18, s[64:67], 0 offen offset: 0 // (op 8324) Load value




s_lshl_b64 s[0:1], s[28:29], 7 // (op 8327) 
s_mul_i32 s19, s1, 6 // (op 8327) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 8327) least significant
s_mul_hi_u32 s49, s0, 6 // (op 8327) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8327) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 8327) 

v_add_u32 v18, v16, s0 // (op 8327) 


buffer_load_dwordx3 v[58:60], v18, s[52:55], 0 offen // (op 8327) Load value
buffer_load_dwordx3 v[62:64], v18, s[52:55], 0 offen offset:12 // (op 8327) Load value
buffer_load_dwordx3 v[66:68], v18, s[52:55], 0 offen offset:24 // (op 8327) Load value
buffer_load_dwordx3 v[70:72], v18, s[52:55], 0 offen offset:36 // (op 8327) Load value



v_mov_b64 v[40:41], s[38:39] // (op 8330) convert

s_lshl_b32 s0, s38, 2 // (op 8330) 
v_add_u32 v16, v21, s0 // (op 8330) 

buffer_load_ubyte v74, v16, s[68:71], 0 offen offset: 0 // (op 8330) Load value
v_add_u32 v16, v16, v40 // (op 8330) 
buffer_load_ubyte v75, v16, s[68:71], 0 offen offset: 0 // (op 8330) Load value






buffer_load_dwordx4 v[76:79], v14, s[20:23], 0 offen offset:96 // (op 8333) Load value
buffer_load_dwordx2 v[80:81], v14, s[20:23], 0 offen offset:112 // (op 8333) Load value
buffer_load_dwordx4 v[82:85], v14, s[20:23], 0 offen offset:120 // (op 8333) Load value
buffer_load_dwordx2 v[86:87], v14, s[20:23], 0 offen offset:136 // (op 8333) Load value


v_mov_b64 v[40:41], s[34:35] // (op 8336) convert

v_mov_b32 v14, v19 // (op 8336) 
buffer_load_ubyte v88, v14, s[64:67], 0 offen offset:4 // (op 8336) Load value
v_add_u32 v14, v14, v40 // (op 8336) 
buffer_load_ubyte v89, v14, s[64:67], 0 offen offset:4 // (op 8336) Load value



v_and_b32 v14, -2, v1 // (op 9068) 
v_sub_u32 v16, v1, v14 // (op 9068) 

v_lshlrev_b32 v14, 6, v16 // (op 9068) 

v_lshrrev_b32 v16, 1, v1 // (op 9068) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 9068) 


v_mul_lo_u32 v14, v21, 6 // (op 9068) 
v_lshrrev_b32 v16, 3, v14 // (op 9068) 

v_lshrrev_b32 v14, 4, v21 // (op 9068) 

v_lshlrev_b32 v21, 2, v14 // (op 9068) 

v_add_u32 v14, v16, v21 // (op 9068) 





s_waitcnt vmcnt(23)
ds_write_b96 v14, v[24:26] // (op 8338) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[28:30] offset:16 // (op 8338) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[32:34] offset:32 // (op 8338) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[36:38] offset:48 // (op 8338) Store local data 

v_and_b32 v14, -128, v1 // (op 9184) 
v_sub_u32 v16, v1, v14 // (op 9184) 

v_lshrrev_b32 v14, 7, v1 // (op 9184) 
v_lshl_add_u32 v21, v14, 8, v16 // (op 9184) 





s_waitcnt vmcnt(19)
ds_write_b8 v21, v42 offset:16384 // (op 8339) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v21, v43 offset:16512 // (op 8339) Store local data 

v_and_b32 v14, -2, v1 // (op 9010) 
v_sub_u32 v16, v1, v14 // (op 9010) 

v_lshlrev_b32 v14, 6, v16 // (op 9010) 

v_lshrrev_b32 v16, 1, v1 // (op 9010) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 9010) 


v_mul_lo_u32 v14, v21, 6 // (op 9010) 

v_lshrrev_b32 v16, 3, v14 // (op 9010) 




s_waitcnt vmcnt(17)
ds_write_b128 v16, v[44:47] offset:16896 // (op 8340) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v16, v[48:49] offset:16912 // (op 8340) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[50:53] offset:16920 // (op 8340) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v16, v[54:55] offset:16936 // (op 8340) Store local data 

v_and_b32 v14, -4, v1 // (op 9126) 
v_sub_u32 v16, v1, v14 // (op 9126) 

v_lshrrev_b32 v14, 2, v1 // (op 9126) 
v_lshl_add_u32 v21, v14, 3, v16 // (op 9126) 





s_waitcnt vmcnt(13)
ds_write_b8 v21, v56 offset:29184 // (op 8341) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v21, v57 offset:29188 // (op 8341) Store local data 

v_and_b32 v14, -2, v1 // (op 8981) 
v_sub_u32 v16, v1, v14 // (op 8981) 

v_lshlrev_b32 v14, 6, v16 // (op 8981) 

v_lshrrev_b32 v16, 1, v1 // (op 8981) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 8981) 


v_mul_lo_u32 v14, v21, 6 // (op 8981) 

v_lshrrev_b32 v16, 3, v14 // (op 8981) 

v_and_b32 v14, -2, v1 // (op 9039) 
v_sub_u32 v21, v1, v14 // (op 9039) 

v_lshlrev_b32 v14, 6, v21 // (op 9039) 

v_lshrrev_b32 v21, 1, v1 // (op 9039) 
v_lshl_add_u32 v23, v21, 7, v14 // (op 9039) 


v_mul_lo_u32 v14, v23, 6 // (op 9039) 
v_lshrrev_b32 v21, 3, v14 // (op 9039) 

v_lshrrev_b32 v14, 4, v23 // (op 9039) 

v_lshlrev_b32 v23, 2, v14 // (op 9039) 

v_add_u32 v14, v21, v23 // (op 9039) 


v_and_b32 v21, -4, v1 // (op 9097) 
v_sub_u32 v23, v1, v21 // (op 9097) 

v_lshrrev_b32 v21, 2, v1 // (op 9097) 
v_lshl_add_u32 v27, v21, 3, v23 // (op 9097) 


v_and_b32 v21, -128, v1 // (op 9155) 
v_sub_u32 v23, v1, v21 // (op 9155) 

v_lshrrev_b32 v21, 7, v1 // (op 9155) 
v_lshl_add_u32 v31, v21, 8, v23 // (op 9155) 


v_and_b32 v21, -2, v1 // (op 9236) 
v_sub_u32 v23, v1, v21 // (op 9236) 

v_lshlrev_b32 v21, 6, v23 // (op 9236) 

v_lshrrev_b32 v23, 1, v1 // (op 9236) 
v_lshl_add_u32 v35, v23, 7, v21 // (op 9236) 


v_mul_lo_u32 v21, v35, 6 // (op 9236) 

v_lshrrev_b32 v23, 3, v21 // (op 9236) 

v_and_b32 v21, -2, v1 // (op 9265) 
v_sub_u32 v35, v1, v21 // (op 9265) 

v_lshlrev_b32 v21, 6, v35 // (op 9265) 

v_lshrrev_b32 v35, 1, v1 // (op 9265) 
v_lshl_add_u32 v61, v35, 7, v21 // (op 9265) 


v_mul_lo_u32 v21, v61, 6 // (op 9265) 
v_lshrrev_b32 v35, 3, v21 // (op 9265) 

v_lshrrev_b32 v21, 4, v61 // (op 9265) 

v_lshlrev_b32 v61, 2, v21 // (op 9265) 

v_add_u32 v21, v35, v61 // (op 9265) 


v_and_b32 v35, -4, v1 // (op 9294) 
v_sub_u32 v61, v1, v35 // (op 9294) 

v_lshrrev_b32 v35, 2, v1 // (op 9294) 
v_lshl_add_u32 v65, v35, 3, v61 // (op 9294) 


v_and_b32 v35, -128, v1 // (op 9323) 
v_sub_u32 v61, v1, v35 // (op 9323) 

v_lshrrev_b32 v35, 7, v1 // (op 9323) 
v_lshl_add_u32 v69, v35, 8, v61 // (op 9323) 


s_mov_b32 s0, 0 // (op 2529) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 2528) 
s_mov_b32 s1, s18 // (op 2528) convert

s_and_b32 s18, s1, 4294967294 // (op 2528) 

s_cmp_lt_u32 s0, s18 // (op 2528) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2528 // (op 2528) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2528 if false)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2528:


s_barrier  // (op 9427) 
s_add_u32 s0, s0, 2 // (op 2530) 



v_mov_b32 v35, v17 // (op 10) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v61, v35, s1 // (op 10) 


ds_read_b128 v[90:93], v3 offset:16896 // (op 72) Load local data 
ds_read_b64 v[94:95], v3 offset:16912 // (op 72) Load local data 
buffer_load_dwordx3 v[24:26], v61, s[52:55], 0 offen // (op 10) Load value
s_barrier  // (op 9440) 
buffer_load_dwordx3 v[28:30], v61, s[52:55], 0 offen offset:12 // (op 10) Load value



ds_read_b96_tr_b6 v[96:98], v0 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[99:101], v0 offset:2048 // (op 77) Transpose load local data 
buffer_load_dwordx3 v[32:34], v61, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 9454) 
buffer_load_dwordx3 v[36:38], v61, s[52:55], 0 offen offset:36 // (op 10) Load value





ds_read_u8 v61, v4 offset:29184 // (op 82) Load local data 
v_mov_b64 v[18:19], s[38:39] // (op 25) convert

s_barrier  // (op 9468) 
v_mov_b32 v73, v22 // (op 25) 


ds_read_u8 v39, v5 offset:16384 // (op 87) Load local data 
s_lshl_b32 s1, s38, 3 // (op 25) 



v_add_u32 v40, v73, s1 // (op 25) 

buffer_load_ubyte v42, v40, s[68:71], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v40, v40, v18 // (op 25) 
buffer_load_ubyte v43, v40, s[68:71], 0 offen offset: 0 // (op 25) Load value


ds_read_b96_tr_b6 v[102:104], v0 offset:32 // (op 375) Transpose load local data 


v_mov_b32 v18, v15 // (op 4) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:101], a[0:3], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 90) 
ds_read_b96_tr_b6 v[105:107], v0 offset:2080 // (op 375) Transpose load local data 
buffer_load_dwordx4 v[44:47], v18, s[20:23], 0 offen offset:192 // (op 4) Load value


ds_read_u8 v19, v5 offset:16416 // (op 379) Load local data 
buffer_load_dwordx2 v[48:49], v18, s[20:23], 0 offen offset:208 // (op 4) Load value



ds_read_b96_tr_b6 v[108:110], v0 offset:64 // (op 551) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v0 offset:2112 // (op 551) Transpose load local data 
buffer_load_dwordx4 v[50:53], v18, s[20:23], 0 offen offset:216 // (op 4) Load value


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[102:107], a[4:7], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 380) 
ds_read_u8 v40, v5 offset:16448 // (op 555) Load local data 
buffer_load_dwordx2 v[54:55], v18, s[20:23], 0 offen offset:232 // (op 4) Load value




v_mov_b64 v[114:115], s[34:35] // (op 17) convert

v_mov_b32 v41, v20 // (op 17) 
v_mov_b32 v116, v41 // (op 17) 
ds_read_b96_tr_b6 v[118:120], v0 offset:96 // (op 727) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:2144 // (op 727) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[108:113], a[8:11], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 556) 


buffer_load_ubyte v56, v116, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v116, v116, v114 // (op 17) 
ds_read_u8 v117, v5 offset:16480 // (op 731) Load local data 
buffer_load_ubyte v57, v116, s[64:67], 0 offen offset:8 // (op 17) Load value





s_barrier  // (op 9491) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[118:123], a[12:15], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 732) 





ds_read_b128 v[90:93], v3 offset:19968 // (op 1084) Load local data 
ds_read_b64 v[94:95], v3 offset:19984 // (op 1084) Load local data 
s_waitcnt vmcnt(23)
ds_write_b96 v21, v[58:60] offset:29696 // (op 257) Store local data 



ds_read_u8 v61, v4 offset:29312 // (op 1087) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v21, v[62:64] offset:29712 // (op 257) Store local data 



ds_read_b128 v[124:127], v3 offset:23040 // (op 1854) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:101], a[16:19], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1227) 
ds_read_b64 v[128:129], v3 offset:23056 // (op 1854) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[102:107], a[20:23], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1181) 
ds_read_u8 v114, v4 offset:29440 // (op 1857) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[108:113], a[24:27], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1135) 
ds_read_b128 v[130:133], v3 offset:26112 // (op 2624) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[118:123], a[28:31], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1089) 
ds_read_b64 v[134:135], v3 offset:26128 // (op 2624) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:129], v[96:101], a[32:35], v114, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1997) 
ds_read_u8 v61, v4 offset:29568 // (op 2627) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[124:129], v[102:107], a[36:39], v114, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1951) 
s_waitcnt vmcnt(21)
ds_write_b96 v21, v[66:68] offset:29728 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:129], v[108:113], a[40:43], v114, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1905) 
s_waitcnt vmcnt(20)
ds_write_b96 v21, v[70:72] offset:29744 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:129], v[118:123], a[44:47], v114, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1859) 


s_barrier  // (op 9503) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[96:101], a[48:51], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2767) 
s_waitcnt vmcnt(19)
ds_write_b8 v69, v74 offset:46080 // (op 259) Store local data 


s_waitcnt vmcnt(18)
ds_write_b8 v69, v75 offset:46208 // (op 259) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[102:107], a[52:55], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2721) 


s_barrier  // (op 9485) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[108:113], a[56:59], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2675) 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[76:79] offset:46592 // (op 247) Store local data 


s_waitcnt vmcnt(16)
ds_write_b64 v23, v[80:81] offset:46608 // (op 247) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[118:123], a[60:63], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2629) 
s_waitcnt vmcnt(15)
ds_write_b128 v23, v[82:85] offset:46616 // (op 247) Store local data 




s_waitcnt vmcnt(14)
ds_write_b64 v23, v[86:87] offset:46632 // (op 247) Store local data 
s_barrier  // (op 9497) 



s_waitcnt vmcnt(13)
ds_write_b8 v65, v88 offset:58880 // (op 258) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v65, v89 offset:58884 // (op 258) Store local data 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 260) 
s_mul_i32 s51, s19, 6 // (op 260) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 260) least significant
s_mul_hi_u32 s1, s18, 6 // (op 260) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 260) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 260) 

s_mul_i32 s1, 3, s18 // (op 260) 

v_add_u32 v19, v35, s1 // (op 260) 


buffer_load_dwordx3 v[58:60], v19, s[52:55], 0 offen // (op 260) Load value
buffer_load_dwordx3 v[62:64], v19, s[52:55], 0 offen offset:12 // (op 260) Load value
buffer_load_dwordx3 v[66:68], v19, s[52:55], 0 offen offset:24 // (op 260) Load value
s_waitcnt lgkmcnt(2)
s_barrier  // (op 9482) 
buffer_load_dwordx3 v[70:72], v19, s[52:55], 0 offen offset:36 // (op 260) Load value





ds_read_b128 v[90:93], v7 offset:46592 // (op 249) Load local data 

ds_read_b64 v[94:95], v7 offset:46608 // (op 249) Load local data 
v_mov_b64 v[96:97], s[38:39] // (op 262) convert

s_barrier  // (op 9488) 
s_lshl_b32 s1, s38, 2 // (op 262) 



s_mul_i32 s18, 3, s1 // (op 262) 

v_add_u32 v19, v73, s18 // (op 262) 

buffer_load_ubyte v74, v19, s[68:71], 0 offen offset: 0 // (op 262) Load value
v_add_u32 v19, v19, v96 // (op 262) 
buffer_load_ubyte v75, v19, s[68:71], 0 offen offset: 0 // (op 262) Load value


s_lshl_b64 s[18:19], s[28:29], 7 // (op 8802) 

s_mul_i32 s51, s19, 6 // (op 8802) most significant: low of high * low



s_mul_i32 s50, s18, 6 // (op 8802) least significant
s_mul_hi_u32 s1, s18, 6 // (op 8802) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 8802) most significant: sum


s_lshl_b32 s1, s38, 3 // (op 8930) 
v_add_u32 v22, v22, s1 // (op 8930) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 8802) 

s_lshl_b32 s1, s18, 1 // (op 8802) 

v_add_u32 v17, v17, s1 // (op 8802) 

ds_read_b96_tr_b6 v[96:98], v6 offset:29696 // (op 251) Transpose load local data 
ds_read_b96_tr_b6 v[99:101], v6 offset:31744 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[76:79], v18, s[20:23], 0 offen offset:288 // (op 246) Load value
buffer_load_dwordx2 v[80:81], v18, s[20:23], 0 offen offset:304 // (op 246) Load value
buffer_load_dwordx4 v[82:85], v18, s[20:23], 0 offen offset:312 // (op 246) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9494) 
buffer_load_dwordx2 v[86:87], v18, s[20:23], 0 offen offset:328 // (op 246) Load value




ds_read_u8 v35, v8 offset:58880 // (op 253) Load local data 

s_barrier  // (op 9500) 
v_add_u32 v15, 192, v15 // (op 8738) 


ds_read_u8 v61, v9 offset:46080 // (op 255) Load local data 
v_mov_b64 v[18:19], s[34:35] // (op 261) convert




v_mov_b32 v73, v41 // (op 261) 
ds_read_b96_tr_b6 v[102:104], v6 offset:29728 // (op 385) Transpose load local data 
buffer_load_ubyte v88, v73, s[64:67], 0 offen offset:12 // (op 261) Load value
v_add_u32 v73, v73, v18 // (op 261) 
buffer_load_ubyte v89, v73, s[64:67], 0 offen offset:12 // (op 261) Load value


s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:101], a[0:3], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 256) 

ds_read_b96_tr_b6 v[105:107], v6 offset:31776 // (op 385) Transpose load local data 


s_barrier  // (op 9443) 
v_add_u32 v20, 8, v20 // (op 8866) 



ds_read_u8 v73, v9 offset:46112 // (op 389) Load local data 
s_waitcnt vmcnt(23)
ds_write_b96 v14, v[24:26] // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[102:107], a[4:7], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 381) 
ds_read_b96_tr_b6 v[108:110], v6 offset:29760 // (op 561) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v6 offset:31808 // (op 561) Transpose load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[28:30] offset:16 // (op 13) Store local data 


ds_read_u8 v18, v9 offset:46144 // (op 565) Load local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[32:34] offset:32 // (op 13) Store local data 



ds_read_b96_tr_b6 v[114:116], v6 offset:29792 // (op 737) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[108:113], a[8:11], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 557) 
ds_read_b96_tr_b6 v[117:119], v6 offset:31840 // (op 737) Transpose load local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[36:38] offset:48 // (op 13) Store local data 


s_barrier  // (op 9564) 
s_barrier  // (op 9471) 
ds_read_u8 v19, v9 offset:46176 // (op 741) Load local data 



s_waitcnt vmcnt(19)
ds_write_b8 v31, v42 offset:16384 // (op 30) Store local data 



s_barrier  // (op 9568) 
ds_read_b128 v[120:123], v7 offset:49664 // (op 1092) Load local data 
ds_read_b64 v[124:125], v7 offset:49680 // (op 1092) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[114:119], a[12:15], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 733) 





ds_read_u8 v35, v8 offset:59008 // (op 1095) Load local data 
s_waitcnt vmcnt(18)
ds_write_b8 v31, v43 offset:16512 // (op 30) Store local data 



ds_read_b128 v[90:93], v7 offset:52736 // (op 1862) Load local data 
s_barrier  // (op 9430) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[96:101], a[16:19], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1228) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[102:107], a[20:23], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1182) 
ds_read_b64 v[94:95], v7 offset:52752 // (op 1862) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[120:125], v[108:113], a[24:27], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1136) 
ds_read_u8 v39, v8 offset:59136 // (op 1865) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[120:125], v[114:119], a[28:31], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1090) 
ds_read_b128 v[126:129], v7 offset:55808 // (op 2632) Load local data 


ds_read_b64 v[130:131], v7 offset:55824 // (op 2632) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[96:101], a[32:35], v39, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1998) 



s_barrier  // (op 9604) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:95], v[102:107], a[36:39], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1952) 
ds_read_u8 v35, v8 offset:59264 // (op 2635) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:95], v[108:113], a[40:43], v39, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1906) 
s_barrier  // (op 9608) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:95], v[114:119], a[44:47], v39, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1860) 
s_waitcnt vmcnt(17)
ds_write_b128 v16, v[44:47] offset:16896 // (op 6) Store local data 


s_waitcnt vmcnt(16)
ds_write_b64 v16, v[48:49] offset:16912 // (op 6) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[126:131], v[96:101], a[48:51], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2768) 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[50:53] offset:16920 // (op 6) Store local data 


s_waitcnt vmcnt(14)
ds_write_b64 v16, v[54:55] offset:16936 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[102:107], a[52:55], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2722) 


s_barrier  // (op 9457) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:131], v[108:113], a[56:59], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2676) 
s_waitcnt vmcnt(13)
ds_write_b8 v27, v56 offset:29184 // (op 21) Store local data 


s_waitcnt vmcnt(12)
ds_write_b8 v27, v57 offset:29188 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:131], v[114:119], a[60:63], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2630) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 2528) 
s_mov_b32 s1, s18 // (op 2528) convert

s_and_b32 s18, s1, 4294967294 // (op 2528) 

s_cmp_lt_u32 s0, s18 // (op 2528) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2528 // (op 2528) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2528 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2528:

















s_mul_hi_i32 s0, s16, s11 // (op 8686) 
v_lshrrev_b32 v61, 1, v1 // (op 8686) 
s_add_i32 s1, s0, s16 // (op 8686) 

v_and_b32 v65, -2, v1 // (op 8686) 
s_ashr_i32 s0, s1, 31 // (op 8686) 
v_sub_u32 v69, v1, v65 // (op 8686) 

s_cmp_eq_i32 s11, 0 // (op 8686) 
s_mov_b32 s18, scc // (op 8686) 
v_lshlrev_b32 v65, 6, v69 // (op 8686) 

s_cmp_eq_i32 s18, 0 // (op 8686) 
s_cselect_b32 s19, 0, -1 // (op 8686) 

s_and_b32 s18, s12, 31 // (op 8686) 
s_lshl_b32 s49, 1, s18 // (op 8686) 
s_add_i32 s50, s49, s19 // (op 8686) 


s_and_b32 s19, s0, s50 // (op 8686) 


s_add_i32 s0, s1, s19 // (op 8686) 


s_ashr_i32 s1, s0, s18 // (op 8686) 


s_lshl_b32 s0, s12, 24 // (op 8686) 
s_ashr_i32 s18, s0, 31 // (op 8686) 

s_xor_b32 s0, s1, s18 // (op 8686) 

s_sub_i32 s1, s0, s18 // (op 8686) 


s_mul_hi_i32 s0, s1, s13 // (op 8686) 
s_add_i32 s18, s0, s1 // (op 8686) 

s_ashr_i32 s0, s18, 31 // (op 8686) 
s_cmp_eq_i32 s13, 0 // (op 8686) 
s_mov_b32 s19, scc // (op 8686) 
s_cmp_eq_i32 s19, 0 // (op 8686) 
s_cselect_b32 s49, 0, -1 // (op 8686) 

s_and_b32 s19, s14, 31 // (op 8686) 
s_lshl_b32 s50, 1, s19 // (op 8686) 
s_add_i32 s51, s50, s49 // (op 8686) 


s_and_b32 s49, s0, s51 // (op 8686) 


s_add_i32 s0, s18, s49 // (op 8686) 


s_ashr_i32 s18, s0, s19 // (op 8686) 


s_lshl_b32 s0, s14, 24 // (op 8686) 
s_ashr_i32 s19, s0, 31 // (op 8686) 

s_xor_b32 s0, s18, s19 // (op 8686) 

s_sub_i32 s18, s0, s19 // (op 8686) 


s_mul_i32 s0, s1, s44 // (op 8686) 

s_sub_u32 s1, s16, s0 // (op 8686) 

s_mul_i32 s0, s18, s44 // (op 8686) 
s_add_u32 s0, s0, s1 // (op 8686) 


s_mul_hi_i32 s1, s16, s10 // (op 8686) 
s_add_i32 s18, s1, s16 // (op 8686) 

s_ashr_i32 s1, s18, 31 // (op 8686) 
s_cmp_eq_i32 s10, 0 // (op 8686) 
s_mov_b32 s19, scc // (op 8686) 
s_cmp_eq_i32 s19, 0 // (op 8686) 
s_cselect_b32 s49, 0, -1 // (op 8686) 

s_and_b32 s19, s47, 31 // (op 8686) 
s_lshl_b32 s50, 1, s19 // (op 8686) 
s_add_i32 s51, s50, s49 // (op 8686) 


s_and_b32 s49, s1, s51 // (op 8686) 


s_add_i32 s1, s18, s49 // (op 8686) 


s_ashr_i32 s18, s1, s19 // (op 8686) 


s_lshl_b32 s1, s47, 24 // (op 8686) 
s_ashr_i32 s19, s1, 31 // (op 8686) 

s_xor_b32 s1, s18, s19 // (op 8686) 

s_sub_i32 s18, s1, s19 // (op 8686) 


s_add_u32 s50, s6, 128 // (op 8686) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8686) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8686) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8686) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8686) 

s_mul_i32 s1, s44, s50 // (op 8686) 

s_mul_i32 s19, s48, s1 // (op 8686) 

s_mul_i32 s1, s18, s19 // (op 8686) 

s_sub_u32 s19, s16, s1 // (op 8686) 

s_mov_b32 s1, s19 // (op 8686) 
s_mul_hi_i32 s49, s1, s15 // (op 8686) 
s_add_i32 s50, s49, s1 // (op 8686) 


s_ashr_i32 s1, s50, 31 // (op 8686) 
s_cmp_eq_i32 s15, 0 // (op 8686) 
s_mov_b32 s49, scc // (op 8686) 
s_cmp_eq_i32 s49, 0 // (op 8686) 
s_cselect_b32 s51, 0, -1 // (op 8686) 

s_and_b32 s49, s46, 31 // (op 8686) 
s_lshl_b32 s72, 1, s49 // (op 8686) 
s_add_i32 s73, s72, s51 // (op 8686) 


s_and_b32 s51, s1, s73 // (op 8686) 


s_add_i32 s1, s50, s51 // (op 8686) 


s_ashr_i32 s50, s1, s49 // (op 8686) 


s_lshl_b32 s1, s46, 24 // (op 8686) 
s_ashr_i32 s49, s1, 31 // (op 8686) 

s_xor_b32 s1, s50, s49 // (op 8686) 

s_sub_i32 s50, s1, s49 // (op 8686) 


s_mul_i32 s1, s45, s44 // (op 8686) 
s_add_u32 s72, s8, 128 // (op 8686) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 8686) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 8686) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 8686) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 8686) 

s_sub_i32 s49, s72, s1 // (op 8686) 

s_mul_i32 s51, s50, s49 // (op 8686) 


s_sub_u32 s49, s19, s51 // (op 8686) 


s_add_u32 s19, s49, s1 // (op 8686) 


s_cmp_eq_i32 s18, 0 // (op 8686) 
s_mov_b32 s1, scc // (op 8686) 

s_cmp_eq_i32 s1, 0 // (op 8686) 
s_cselect_b32 s18, s19, s0 // (op 8686) 



v_lshl_add_u32 v69, s18, 7, v61 // (op 8686) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8686) 
s_lshr_b32 s18, s0, 1 // (op 8686) 

v_lshl_add_u32 v61, s18, 8, v65 // (op 8686) 


v_mov_b32 v15, 0 // (op 8686) convert
v_mov_b32 v14, v61 // (op 8686) convert

v_mul_lo_u32 v17, v69, s25 // (op 8686) most significant: low of low * high
v_mul_hi_u32 v61, v69, s24 // (op 8686) most significant: high of low * low
v_mul_lo_u32 v16, v69, s24 // (op 8686) least significant: low of low * low
v_add_u32 v17, v17, v61 // (op 8686) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 8686) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 8686) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 8686) most significant: low of high * low
v_mul_hi_u32 v61, v16, 6 // (op 8686) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 8686) least significant: low of low * low
v_add_u32 v15, v15, v61 // (op 8686) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8686) 

v_mov_b32 v61, v16 // (op 8686) convert

s_mul_i32 s1, s9, 6 // (op 8688) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 8688) least significant
s_mul_hi_u32 s18, s8, 6 // (op 8688) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8688) most significant: sum

s_mul_i32 s19, s24, s1 // (op 8688) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 8688) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 8688) least significant
s_mul_hi_u32 s50, s24, s0 // (op 8688) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8688) most significant: sum
s_add_u32 s19, s19, s50 // (op 8688) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8688) 

s_mov_b32 s20, s26 // (op 8688) reg expression
s_mov_b32 s21, s27 // (op 8688) reg expression
s_mov_b32 s22, s0 // (op 8688) reg expression
s_mov_b32 s23, 131072 // (op 8688) call()

s_mul_hi_i32 s0, s16, s11 // (op 8768) 
v_and_b32 v65, -2, v1 // (op 8768) 
s_add_i32 s1, s0, s16 // (op 8768) 

v_sub_u32 v69, v1, v65 // (op 8768) 

s_ashr_i32 s0, s1, 31 // (op 8768) 
v_lshlrev_b32 v65, 6, v69 // (op 8768) 

s_cmp_eq_i32 s11, 0 // (op 8768) 
s_mov_b32 s18, scc // (op 8768) 
v_lshrrev_b32 v69, 1, v1 // (op 8768) 
s_cmp_eq_i32 s18, 0 // (op 8768) 
s_cselect_b32 s19, 0, -1 // (op 8768) 

s_and_b32 s18, s12, 31 // (op 8768) 
s_lshl_b32 s49, 1, s18 // (op 8768) 
s_add_i32 s50, s49, s19 // (op 8768) 


s_and_b32 s19, s0, s50 // (op 8768) 


s_add_i32 s0, s1, s19 // (op 8768) 


s_ashr_i32 s1, s0, s18 // (op 8768) 


s_lshl_b32 s0, s12, 24 // (op 8768) 
s_ashr_i32 s18, s0, 31 // (op 8768) 

s_xor_b32 s0, s1, s18 // (op 8768) 

s_sub_i32 s1, s0, s18 // (op 8768) 


s_mul_hi_i32 s0, s1, s13 // (op 8768) 
s_add_i32 s18, s0, s1 // (op 8768) 

s_ashr_i32 s0, s18, 31 // (op 8768) 
s_cmp_eq_i32 s13, 0 // (op 8768) 
s_mov_b32 s19, scc // (op 8768) 
s_cmp_eq_i32 s19, 0 // (op 8768) 
s_cselect_b32 s49, 0, -1 // (op 8768) 

s_and_b32 s19, s14, 31 // (op 8768) 
s_lshl_b32 s50, 1, s19 // (op 8768) 
s_add_i32 s51, s50, s49 // (op 8768) 


s_and_b32 s49, s0, s51 // (op 8768) 


s_add_i32 s0, s18, s49 // (op 8768) 


s_ashr_i32 s18, s0, s19 // (op 8768) 


s_lshl_b32 s0, s14, 24 // (op 8768) 
s_ashr_i32 s19, s0, 31 // (op 8768) 

s_xor_b32 s0, s18, s19 // (op 8768) 

s_sub_i32 s18, s0, s19 // (op 8768) 


s_add_u32 s50, s6, 128 // (op 8768) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8768) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8768) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8768) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8768) 

s_mul_i32 s0, s18, s50 // (op 8768) 

s_sub_i32 s49, s1, s0 // (op 8768) 


s_mul_hi_i32 s0, s16, s10 // (op 8768) 
s_add_i32 s1, s0, s16 // (op 8768) 

s_ashr_i32 s0, s1, 31 // (op 8768) 
s_cmp_eq_i32 s10, 0 // (op 8768) 
s_mov_b32 s18, scc // (op 8768) 
s_cmp_eq_i32 s18, 0 // (op 8768) 
s_cselect_b32 s19, 0, -1 // (op 8768) 

s_and_b32 s18, s47, 31 // (op 8768) 
s_lshl_b32 s72, 1, s18 // (op 8768) 
s_add_i32 s73, s72, s19 // (op 8768) 


s_and_b32 s19, s0, s73 // (op 8768) 


s_add_i32 s0, s1, s19 // (op 8768) 


s_ashr_i32 s1, s0, s18 // (op 8768) 


s_lshl_b32 s0, s47, 24 // (op 8768) 
s_ashr_i32 s18, s0, 31 // (op 8768) 

s_xor_b32 s0, s1, s18 // (op 8768) 

s_sub_i32 s1, s0, s18 // (op 8768) 


s_mul_i32 s0, s44, s50 // (op 8768) 

s_mul_i32 s18, s48, s0 // (op 8768) 

s_mul_i32 s0, s1, s18 // (op 8768) 

s_sub_u32 s18, s16, s0 // (op 8768) 

s_mul_hi_i32 s0, s18, s15 // (op 8768) 
s_add_i32 s19, s0, s18 // (op 8768) 


s_ashr_i32 s0, s19, 31 // (op 8768) 
s_cmp_eq_i32 s15, 0 // (op 8768) 
s_mov_b32 s18, scc // (op 8768) 
s_cmp_eq_i32 s18, 0 // (op 8768) 
s_cselect_b32 s50, 0, -1 // (op 8768) 

s_and_b32 s18, s46, 31 // (op 8768) 
s_lshl_b32 s51, 1, s18 // (op 8768) 
s_add_i32 s72, s51, s50 // (op 8768) 


s_and_b32 s50, s0, s72 // (op 8768) 


s_add_i32 s0, s19, s50 // (op 8768) 


s_ashr_i32 s19, s0, s18 // (op 8768) 


s_lshl_b32 s0, s46, 24 // (op 8768) 
s_ashr_i32 s18, s0, 31 // (op 8768) 

s_xor_b32 s0, s19, s18 // (op 8768) 

s_sub_i32 s19, s0, s18 // (op 8768) 


s_cmp_eq_i32 s1, 0 // (op 8768) 
s_mov_b32 s0, scc // (op 8768) 

s_cmp_eq_i32 s0, 0 // (op 8768) 
s_cselect_b32 s1, s19, s49 // (op 8768) 



v_lshl_add_u32 v73, s1, 7, v65 // (op 8768) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8768) 
v_mov_b32 v15, 0 // (op 8768) convert
v_mov_b32 v14, v73 // (op 8768) convert

s_lshr_b32 s18, s0, 1 // (op 8768) 

v_lshl_add_u32 v65, s18, 8, v69 // (op 8768) 


v_mul_lo_u32 v17, v65, s29 // (op 8768) most significant: low of low * high
v_mul_hi_u32 v69, v65, s28 // (op 8768) most significant: high of low * low
v_mul_lo_u32 v16, v65, s28 // (op 8768) least significant: low of low * low
v_add_u32 v17, v17, v69 // (op 8768) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 8768) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 8768) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 8768) most significant: low of high * low
v_mul_hi_u32 v65, v16, 6 // (op 8768) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 8768) least significant: low of low * low
v_add_u32 v15, v15, v65 // (op 8768) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8768) 

v_mov_b32 v65, v16 // (op 8768) convert

s_mul_i32 s1, s33, 6 // (op 8770) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 8770) least significant
s_mul_hi_u32 s18, s32, 6 // (op 8770) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8770) most significant: sum

s_mul_i32 s19, s28, s1 // (op 8770) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 8770) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 8770) least significant
s_mul_hi_u32 s50, s28, s0 // (op 8770) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 8770) most significant: sum
s_add_u32 s19, s19, s50 // (op 8770) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 8770) 

s_mov_b32 s52, s30 // (op 8770) reg expression
s_mov_b32 s53, s31 // (op 8770) reg expression
s_mov_b32 s54, s0 // (op 8770) reg expression
s_mov_b32 s55, 131072 // (op 8770) call()

s_mul_hi_i32 s0, s16, s11 // (op 8832) 
v_lshrrev_b32 v69, 2, v1 // (op 8832) 
s_add_i32 s1, s0, s16 // (op 8832) 

v_lshlrev_b32 v73, 1, v69 // (op 8832) 

s_ashr_i32 s0, s1, 31 // (op 8832) 
v_and_b32 v69, -4, v1 // (op 8832) 
s_cmp_eq_i32 s11, 0 // (op 8832) 
s_mov_b32 s18, scc // (op 8832) 
v_sub_u32 v14, v1, v69 // (op 8832) 

s_cmp_eq_i32 s18, 0 // (op 8832) 
s_cselect_b32 s19, 0, -1 // (op 8832) 

s_and_b32 s18, s12, 31 // (op 8832) 
s_lshl_b32 s49, 1, s18 // (op 8832) 
s_add_i32 s50, s49, s19 // (op 8832) 


s_and_b32 s19, s0, s50 // (op 8832) 


s_add_i32 s0, s1, s19 // (op 8832) 


s_ashr_i32 s1, s0, s18 // (op 8832) 


s_lshl_b32 s0, s12, 24 // (op 8832) 
s_ashr_i32 s18, s0, 31 // (op 8832) 

s_xor_b32 s0, s1, s18 // (op 8832) 

s_sub_i32 s1, s0, s18 // (op 8832) 


s_mul_hi_i32 s0, s1, s13 // (op 8832) 
s_add_i32 s18, s0, s1 // (op 8832) 

s_ashr_i32 s0, s18, 31 // (op 8832) 
s_cmp_eq_i32 s13, 0 // (op 8832) 
s_mov_b32 s19, scc // (op 8832) 
s_cmp_eq_i32 s19, 0 // (op 8832) 
s_cselect_b32 s49, 0, -1 // (op 8832) 

s_and_b32 s19, s14, 31 // (op 8832) 
s_lshl_b32 s50, 1, s19 // (op 8832) 
s_add_i32 s51, s50, s49 // (op 8832) 


s_and_b32 s49, s0, s51 // (op 8832) 


s_add_i32 s0, s18, s49 // (op 8832) 


s_ashr_i32 s18, s0, s19 // (op 8832) 


s_lshl_b32 s0, s14, 24 // (op 8832) 
s_ashr_i32 s19, s0, 31 // (op 8832) 

s_xor_b32 s0, s18, s19 // (op 8832) 

s_sub_i32 s18, s0, s19 // (op 8832) 


s_mul_i32 s0, s1, s44 // (op 8832) 

s_sub_u32 s1, s16, s0 // (op 8832) 

s_mul_i32 s0, s18, s44 // (op 8832) 
s_add_u32 s0, s0, s1 // (op 8832) 


s_mul_hi_i32 s1, s16, s10 // (op 8832) 
s_add_i32 s18, s1, s16 // (op 8832) 

s_ashr_i32 s1, s18, 31 // (op 8832) 
s_cmp_eq_i32 s10, 0 // (op 8832) 
s_mov_b32 s19, scc // (op 8832) 
s_cmp_eq_i32 s19, 0 // (op 8832) 
s_cselect_b32 s49, 0, -1 // (op 8832) 

s_and_b32 s19, s47, 31 // (op 8832) 
s_lshl_b32 s50, 1, s19 // (op 8832) 
s_add_i32 s51, s50, s49 // (op 8832) 


s_and_b32 s49, s1, s51 // (op 8832) 


s_add_i32 s1, s18, s49 // (op 8832) 


s_ashr_i32 s18, s1, s19 // (op 8832) 


s_lshl_b32 s1, s47, 24 // (op 8832) 
s_ashr_i32 s19, s1, 31 // (op 8832) 

s_xor_b32 s1, s18, s19 // (op 8832) 

s_sub_i32 s18, s1, s19 // (op 8832) 


s_add_u32 s50, s6, 128 // (op 8832) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8832) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8832) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8832) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8832) 

s_mul_i32 s1, s44, s50 // (op 8832) 

s_mul_i32 s19, s48, s1 // (op 8832) 

s_mul_i32 s1, s18, s19 // (op 8832) 

s_sub_u32 s19, s16, s1 // (op 8832) 

s_mov_b32 s1, s19 // (op 8832) 
s_mul_hi_i32 s49, s1, s15 // (op 8832) 
s_add_i32 s50, s49, s1 // (op 8832) 


s_ashr_i32 s1, s50, 31 // (op 8832) 
s_cmp_eq_i32 s15, 0 // (op 8832) 
s_mov_b32 s49, scc // (op 8832) 
s_cmp_eq_i32 s49, 0 // (op 8832) 
s_cselect_b32 s51, 0, -1 // (op 8832) 

s_and_b32 s49, s46, 31 // (op 8832) 
s_lshl_b32 s72, 1, s49 // (op 8832) 
s_add_i32 s73, s72, s51 // (op 8832) 


s_and_b32 s51, s1, s73 // (op 8832) 


s_add_i32 s1, s50, s51 // (op 8832) 


s_ashr_i32 s50, s1, s49 // (op 8832) 


s_lshl_b32 s1, s46, 24 // (op 8832) 
s_ashr_i32 s49, s1, 31 // (op 8832) 

s_xor_b32 s1, s50, s49 // (op 8832) 

s_sub_i32 s50, s1, s49 // (op 8832) 


s_mul_i32 s1, s45, s44 // (op 8832) 
s_add_u32 s72, s8, 128 // (op 8832) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 8832) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 8832) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 8832) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 8832) 

s_sub_i32 s49, s72, s1 // (op 8832) 

s_mul_i32 s51, s50, s49 // (op 8832) 


s_sub_u32 s49, s19, s51 // (op 8832) 


s_add_u32 s19, s49, s1 // (op 8832) 


s_cmp_eq_i32 s18, 0 // (op 8832) 
s_mov_b32 s1, scc // (op 8832) 

s_cmp_eq_i32 s1, 0 // (op 8832) 
s_cselect_b32 s18, s19, s0 // (op 8832) 



v_lshl_add_u32 v69, s18, 7, v73 // (op 8832) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8832) 
s_lshr_b32 s18, s0, 1 // (op 8832) 

v_lshl_add_u32 v73, s18, 3, v14 // (op 8832) 


v_mul_lo_u32 v14, v69, s34 // (op 8832) 
v_add_u32 v14, v14, v73 // (op 8832) 


s_mul_i32 s1, s34, s9 // (op 8834) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8834) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8834) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8834) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8834) most significant: sum
s_add_u32 s1, s1, s19 // (op 8834) most significant: sum


s_mov_b32 s64, s36 // (op 8834) reg expression
s_mov_b32 s65, s37 // (op 8834) reg expression
s_mov_b32 s66, s0 // (op 8834) reg expression
s_mov_b32 s67, 131072 // (op 8834) call()

s_mul_hi_i32 s0, s16, s11 // (op 8896) 
v_and_b32 v69, -128, v1 // (op 8896) 
s_add_i32 s1, s0, s16 // (op 8896) 

v_sub_u32 v73, v1, v69 // (op 8896) 

s_ashr_i32 s0, s1, 31 // (op 8896) 
v_lshrrev_b32 v69, 7, v1 // (op 8896) 
s_cmp_eq_i32 s11, 0 // (op 8896) 
s_mov_b32 s18, scc // (op 8896) 
v_lshlrev_b32 v15, 1, v69 // (op 8896) 

s_cmp_eq_i32 s18, 0 // (op 8896) 
s_cselect_b32 s19, 0, -1 // (op 8896) 

s_and_b32 s18, s12, 31 // (op 8896) 
s_lshl_b32 s49, 1, s18 // (op 8896) 
s_add_i32 s50, s49, s19 // (op 8896) 


s_and_b32 s19, s0, s50 // (op 8896) 


s_add_i32 s0, s1, s19 // (op 8896) 


s_ashr_i32 s1, s0, s18 // (op 8896) 


s_lshl_b32 s0, s12, 24 // (op 8896) 
s_ashr_i32 s18, s0, 31 // (op 8896) 

s_xor_b32 s0, s1, s18 // (op 8896) 

s_sub_i32 s1, s0, s18 // (op 8896) 


s_mul_hi_i32 s0, s1, s13 // (op 8896) 
s_add_i32 s18, s0, s1 // (op 8896) 

s_ashr_i32 s0, s18, 31 // (op 8896) 
s_cmp_eq_i32 s13, 0 // (op 8896) 
s_mov_b32 s19, scc // (op 8896) 
s_cmp_eq_i32 s19, 0 // (op 8896) 
s_cselect_b32 s49, 0, -1 // (op 8896) 

s_and_b32 s19, s14, 31 // (op 8896) 
s_lshl_b32 s50, 1, s19 // (op 8896) 
s_add_i32 s51, s50, s49 // (op 8896) 


s_and_b32 s49, s0, s51 // (op 8896) 


s_add_i32 s0, s18, s49 // (op 8896) 


s_ashr_i32 s18, s0, s19 // (op 8896) 


s_lshl_b32 s0, s14, 24 // (op 8896) 
s_ashr_i32 s19, s0, 31 // (op 8896) 

s_xor_b32 s0, s18, s19 // (op 8896) 

s_sub_i32 s18, s0, s19 // (op 8896) 


s_add_u32 s50, s6, 128 // (op 8896) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8896) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8896) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8896) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8896) 

s_mul_i32 s0, s18, s50 // (op 8896) 

s_sub_i32 s49, s1, s0 // (op 8896) 


s_mul_hi_i32 s0, s16, s10 // (op 8896) 
s_add_i32 s1, s0, s16 // (op 8896) 

s_ashr_i32 s0, s1, 31 // (op 8896) 
s_cmp_eq_i32 s10, 0 // (op 8896) 
s_mov_b32 s18, scc // (op 8896) 
s_cmp_eq_i32 s18, 0 // (op 8896) 
s_cselect_b32 s19, 0, -1 // (op 8896) 

s_and_b32 s18, s47, 31 // (op 8896) 
s_lshl_b32 s72, 1, s18 // (op 8896) 
s_add_i32 s73, s72, s19 // (op 8896) 


s_and_b32 s19, s0, s73 // (op 8896) 


s_add_i32 s0, s1, s19 // (op 8896) 


s_ashr_i32 s1, s0, s18 // (op 8896) 


s_lshl_b32 s0, s47, 24 // (op 8896) 
s_ashr_i32 s18, s0, 31 // (op 8896) 

s_xor_b32 s0, s1, s18 // (op 8896) 

s_sub_i32 s1, s0, s18 // (op 8896) 


s_mul_i32 s0, s44, s50 // (op 8896) 

s_mul_i32 s18, s48, s0 // (op 8896) 

s_mul_i32 s0, s1, s18 // (op 8896) 

s_sub_u32 s18, s16, s0 // (op 8896) 

s_mul_hi_i32 s0, s18, s15 // (op 8896) 
s_add_i32 s19, s0, s18 // (op 8896) 


s_ashr_i32 s0, s19, 31 // (op 8896) 
s_cmp_eq_i32 s15, 0 // (op 8896) 
s_mov_b32 s18, scc // (op 8896) 
s_cmp_eq_i32 s18, 0 // (op 8896) 
s_cselect_b32 s50, 0, -1 // (op 8896) 

s_and_b32 s18, s46, 31 // (op 8896) 
s_lshl_b32 s51, 1, s18 // (op 8896) 
s_add_i32 s72, s51, s50 // (op 8896) 


s_and_b32 s50, s0, s72 // (op 8896) 


s_add_i32 s0, s19, s50 // (op 8896) 


s_ashr_i32 s19, s0, s18 // (op 8896) 


s_lshl_b32 s0, s46, 24 // (op 8896) 
s_ashr_i32 s18, s0, 31 // (op 8896) 

s_xor_b32 s0, s19, s18 // (op 8896) 

s_sub_i32 s19, s0, s18 // (op 8896) 


s_cmp_eq_i32 s1, 0 // (op 8896) 
s_mov_b32 s0, scc // (op 8896) 

s_cmp_eq_i32 s0, 0 // (op 8896) 
s_cselect_b32 s1, s19, s49 // (op 8896) 



v_lshl_add_u32 v69, s1, 7, v73 // (op 8896) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8896) 
s_lshr_b32 s18, s0, 1 // (op 8896) 

v_lshl_add_u32 v73, s18, 3, v15 // (op 8896) 


v_mul_lo_u32 v15, v73, s38 // (op 8896) 
v_add_u32 v15, v15, v69 // (op 8896) 


s_mul_i32 s1, s38, s43 // (op 8898) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8898) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8898) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8898) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8898) most significant: sum
s_add_u32 s1, s1, s19 // (op 8898) most significant: sum


s_mov_b32 s68, s40 // (op 8898) reg expression
s_mov_b32 s69, s41 // (op 8898) reg expression
s_mov_b32 s70, s0 // (op 8898) reg expression
s_mov_b32 s71, 131072 // (op 8898) call()

v_and_b32 v69, -2, v1 // (op 8994) 
v_sub_u32 v73, v1, v69 // (op 8994) 

v_lshlrev_b32 v69, 6, v73 // (op 8994) 

v_lshrrev_b32 v73, 1, v1 // (op 8994) 
v_lshl_add_u32 v16, v73, 7, v69 // (op 8994) 


v_mul_lo_u32 v69, v16, 6 // (op 8994) 

v_lshrrev_b32 v73, 3, v69 // (op 8994) 

v_and_b32 v69, -2, v1 // (op 9052) 
v_sub_u32 v16, v1, v69 // (op 9052) 

v_lshlrev_b32 v69, 6, v16 // (op 9052) 

v_lshrrev_b32 v16, 1, v1 // (op 9052) 
v_lshl_add_u32 v17, v16, 7, v69 // (op 9052) 


v_mul_lo_u32 v16, v17, 6 // (op 9052) 
v_lshrrev_b32 v69, 3, v16 // (op 9052) 

v_lshrrev_b32 v16, 4, v17 // (op 9052) 

v_lshlrev_b32 v17, 2, v16 // (op 9052) 

v_add_u32 v16, v69, v17 // (op 9052) 


v_and_b32 v69, -4, v1 // (op 9110) 
v_sub_u32 v17, v1, v69 // (op 9110) 

v_lshrrev_b32 v69, 2, v1 // (op 9110) 
v_lshl_add_u32 v18, v69, 3, v17 // (op 9110) 


v_and_b32 v17, -128, v1 // (op 9168) 
v_sub_u32 v69, v1, v17 // (op 9168) 

v_lshrrev_b32 v17, 7, v1 // (op 9168) 
v_lshl_add_u32 v19, v17, 8, v69 // (op 9168) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 2596) 
s_and_b32 s18, s0, 4294967294 // (op 2596) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 2595) 
s_mov_b32 s19, s0 // (op 2595) convert

s_cmp_lt_u32 s18, s19 // (op 2595) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2595 // (op 2595) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2595 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2595:


s_add_u32 s18, s18, 1 // (op 2597) 
v_mov_b32 v17, v61 // (op 180) 

buffer_load_dwordx4 v[20:23], v17, s[20:23], 0 offen // (op 180) Load value
buffer_load_dwordx2 v[24:25], v17, s[20:23], 0 offen offset:16 // (op 180) Load value
buffer_load_dwordx4 v[26:29], v17, s[20:23], 0 offen offset:24 // (op 180) Load value
buffer_load_dwordx2 v[30:31], v17, s[20:23], 0 offen offset:40 // (op 180) Load value

v_add_u32 v61, 96, v61 // (op 8676) 


s_barrier  // (op 9433) 
v_mov_b32 v17, v65 // (op 194) 



v_mov_b32 v32, v17 // (op 194) reg expression

buffer_load_dwordx3 v[34:36], v32, s[52:55], 0 offen // (op 194) Load value
buffer_load_dwordx3 v[38:40], v32, s[52:55], 0 offen offset:12 // (op 194) Load value
s_waitcnt vmcnt(5)
ds_write_b128 v73, v[20:23] offset:16896 // (op 181) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v73, v[24:25] offset:16912 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v73, v[26:29] offset:16920 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v73, v[30:31] offset:16936 // (op 181) Store local data 
buffer_load_dwordx3 v[42:44], v32, s[52:55], 0 offen offset:24 // (op 194) Load value

buffer_load_dwordx3 v[46:48], v32, s[52:55], 0 offen offset:36 // (op 194) Load value


s_lshl_b64 s[0:1], s[28:29], 7 // (op 8760) 
s_mul_i32 s51, s1, 6 // (op 8760) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 8760) least significant
s_mul_hi_u32 s19, s0, 6 // (op 8760) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 8760) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 8760) 

v_add_u32 v65, v65, s0 // (op 8760) 


s_barrier  // (op 9446) 
v_mov_b64 v[20:21], s[34:35] // (op 195) convert




v_mov_b32 v17, v14 // (op 195) 
v_mov_b32 v37, v17 // (op 195) 
buffer_load_ubyte v22, v37, s[64:67], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v37, v37, v20 // (op 195) 
s_waitcnt vmcnt(4)
ds_write_b96 v16, v[34:36] // (op 191) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v16, v[38:40] offset:16 // (op 191) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[42:44] offset:32 // (op 191) Store local data 
buffer_load_ubyte v23, v37, s[64:67], 0 offen offset: 0 // (op 195) Load value



v_add_u32 v14, 4, v14 // (op 8824) 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[46:48] offset:48 // (op 191) Store local data 


v_mov_b64 v[20:21], s[38:39] // (op 196) convert

s_barrier  // (op 9460) 
v_mov_b32 v17, v15 // (op 196) 



v_mov_b32 v24, v17 // (op 196) reg expression
buffer_load_ubyte v62, v24, s[68:71], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v24, v24, v20 // (op 196) 
buffer_load_ubyte v63, v24, s[68:71], 0 offen offset: 0 // (op 196) Load value



s_lshl_b32 s19, s38, 2 // (op 8888) 
v_add_u32 v15, v15, s19 // (op 8888) 

s_waitcnt vmcnt(3)
ds_write_b8 v18, v22 offset:29184 // (op 192) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9436) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:29188 // (op 192) Store local data 




ds_read_b128 v[20:23], v3 offset:16896 // (op 183) Load local data 
s_barrier  // (op 9474) 
ds_read_b64 v[24:25], v3 offset:16912 // (op 183) Load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 9450) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v62 offset:16384 // (op 193) Store local data 



ds_read_b96_tr_b6 v[26:28], v0 // (op 185) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v0 offset:2048 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v63 offset:16512 // (op 193) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 9464) 



ds_read_u8 v17, v4 offset:29184 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9478) 


ds_read_u8 v32, v5 offset:16384 // (op 189) Load local data 



ds_read_b96_tr_b6 v[66:68], v0 offset:32 // (op 354) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v17, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 190) 
ds_read_b96_tr_b6 v[69:71], v0 offset:2080 // (op 354) Transpose load local data 


ds_read_u8 v72, v5 offset:16416 // (op 358) Load local data 



ds_read_b96_tr_b6 v[34:36], v0 offset:64 // (op 530) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[66:71], a[4:7], v17, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 359) 
ds_read_b96_tr_b6 v[37:39], v0 offset:2112 // (op 530) Transpose load local data 


ds_read_u8 v33, v5 offset:16448 // (op 534) Load local data 



ds_read_b96_tr_b6 v[40:42], v0 offset:96 // (op 706) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[34:39], a[8:11], v17, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 535) 
ds_read_b96_tr_b6 v[43:45], v0 offset:2144 // (op 706) Transpose load local data 


ds_read_u8 v46, v5 offset:16480 // (op 710) Load local data 



ds_read_b128 v[48:51], v3 offset:19968 // (op 968) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[40:45], a[12:15], v17, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 711) 
ds_read_b64 v[52:53], v3 offset:19984 // (op 968) Load local data 





ds_read_u8 v17, v4 offset:29312 // (op 971) Load local data 



ds_read_b128 v[20:23], v3 offset:23040 // (op 1738) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[26:31], a[16:19], v17, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 973) 
ds_read_b64 v[24:25], v3 offset:23056 // (op 1738) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[66:71], a[20:23], v17, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1004) 
ds_read_u8 v47, v4 offset:29440 // (op 1741) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[34:39], a[24:27], v17, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1035) 
ds_read_b128 v[54:57], v3 offset:26112 // (op 2508) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[40:45], a[28:31], v17, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1066) 
ds_read_b64 v[58:59], v3 offset:26128 // (op 2508) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[20:25], v[26:31], a[32:35], v47, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1743) 
ds_read_u8 v17, v4 offset:29568 // (op 2511) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[20:25], v[66:71], a[36:39], v47, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1774) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[20:25], v[34:39], a[40:43], v47, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[20:25], v[40:45], a[44:47], v47, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1836) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[26:31], a[48:51], v17, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2513) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[66:71], a[52:55], v17, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2544) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[34:39], a[56:59], v17, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2575) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[40:45], a[60:63], v17, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2606) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 2595) 
s_mov_b32 s19, s0 // (op 2595) convert

s_cmp_lt_u32 s18, s19 // (op 2595) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2595 // (op 2595) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2595 if true)
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2595:























s_barrier  // (op 9592) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
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



v_accvgpr_read v6, a4 // (op 3565) 
v_accvgpr_read v7, a5 // (op 3565) 
v_accvgpr_read v8, a6 // (op 3565) 
v_accvgpr_read v9, a7 // (op 3565) 
v_mul_f32 v14, v11, v6 // (op 3565) 
v_mul_f32 v15, v11, v7 // (op 3565) 
v_mul_f32 v16, v11, v8 // (op 3565) 
v_mul_f32 v17, v11, v9 // (op 3565) 

v_lshrrev_b32 v6, 16, v14 // (op 3567) 
v_lshrrev_b32 v7, 16, v15 // (op 3567) 
v_lshrrev_b32 v8, 16, v16 // (op 3567) 
v_lshrrev_b32 v9, 16, v17 // (op 3567) 


v_pack_b32_f16 v4, v6, v7 // (op 3568) 
v_pack_b32_f16 v5, v8, v9 // (op 3568) 

s_lshl_b32 s0, s4, 6 // (op 3568) 
v_add_u32 v0, v13, s0 // (op 3568) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3568) Store value



v_accvgpr_read v6, a8 // (op 3579) 
v_accvgpr_read v7, a9 // (op 3579) 
v_accvgpr_read v8, a10 // (op 3579) 
v_accvgpr_read v9, a11 // (op 3579) 
v_mul_f32 v14, v11, v6 // (op 3579) 
v_mul_f32 v15, v11, v7 // (op 3579) 
v_mul_f32 v16, v11, v8 // (op 3579) 
v_mul_f32 v17, v11, v9 // (op 3579) 

v_lshrrev_b32 v6, 16, v14 // (op 3581) 
v_lshrrev_b32 v7, 16, v15 // (op 3581) 
v_lshrrev_b32 v8, 16, v16 // (op 3581) 
v_lshrrev_b32 v9, 16, v17 // (op 3581) 


v_pack_b32_f16 v4, v6, v7 // (op 3582) 
v_pack_b32_f16 v5, v8, v9 // (op 3582) 

s_lshl_b32 s0, s4, 7 // (op 3582) 
v_add_u32 v0, v13, s0 // (op 3582) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3582) Store value



v_accvgpr_read v6, a12 // (op 3593) 
v_accvgpr_read v7, a13 // (op 3593) 
v_accvgpr_read v8, a14 // (op 3593) 
v_accvgpr_read v9, a15 // (op 3593) 
v_mul_f32 v14, v11, v6 // (op 3593) 
v_mul_f32 v15, v11, v7 // (op 3593) 
v_mul_f32 v16, v11, v8 // (op 3593) 
v_mul_f32 v17, v11, v9 // (op 3593) 

v_lshrrev_b32 v6, 16, v14 // (op 3595) 
v_lshrrev_b32 v7, 16, v15 // (op 3595) 
v_lshrrev_b32 v8, 16, v16 // (op 3595) 
v_lshrrev_b32 v9, 16, v17 // (op 3595) 


v_pack_b32_f16 v4, v6, v7 // (op 3596) 
v_pack_b32_f16 v5, v8, v9 // (op 3596) 

s_lshl_b32 s0, s4, 6 // (op 3596) 
s_mul_i32 s1, 3, s0 // (op 3596) 

v_add_u32 v0, v13, s1 // (op 3596) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3596) Store value



v_accvgpr_read v6, a16 // (op 3629) 
v_accvgpr_read v7, a17 // (op 3629) 
v_accvgpr_read v8, a18 // (op 3629) 
v_accvgpr_read v9, a19 // (op 3629) 
v_mul_f32 v14, v11, v6 // (op 3629) 
v_mul_f32 v15, v11, v7 // (op 3629) 
v_mul_f32 v16, v11, v8 // (op 3629) 
v_mul_f32 v17, v11, v9 // (op 3629) 

v_lshrrev_b32 v6, 16, v14 // (op 3631) 
v_lshrrev_b32 v7, 16, v15 // (op 3631) 
v_lshrrev_b32 v8, 16, v16 // (op 3631) 
v_lshrrev_b32 v9, 16, v17 // (op 3631) 


v_pack_b32_f16 v4, v6, v7 // (op 3633) 
v_pack_b32_f16 v5, v8, v9 // (op 3633) 

v_add_u32 v0, 64, v13 // (op 3633) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3633) Store value



v_accvgpr_read v6, a20 // (op 3643) 
v_accvgpr_read v7, a21 // (op 3643) 
v_accvgpr_read v8, a22 // (op 3643) 
v_accvgpr_read v9, a23 // (op 3643) 
v_mul_f32 v14, v11, v6 // (op 3643) 
v_mul_f32 v15, v11, v7 // (op 3643) 
v_mul_f32 v16, v11, v8 // (op 3643) 
v_mul_f32 v17, v11, v9 // (op 3643) 

v_lshrrev_b32 v6, 16, v14 // (op 3645) 
v_lshrrev_b32 v7, 16, v15 // (op 3645) 
v_lshrrev_b32 v8, 16, v16 // (op 3645) 
v_lshrrev_b32 v9, 16, v17 // (op 3645) 


v_pack_b32_f16 v4, v6, v7 // (op 3635) 
v_pack_b32_f16 v5, v8, v9 // (op 3635) 

s_lshl_b32 s0, s4, 6 // (op 3635) 
s_add_u32 s0, s0, 64 // (op 3635) 
v_add_u32 v0, v13, s0 // (op 3635) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3635) Store value



v_accvgpr_read v6, a24 // (op 3649) 
v_accvgpr_read v7, a25 // (op 3649) 
v_accvgpr_read v8, a26 // (op 3649) 
v_accvgpr_read v9, a27 // (op 3649) 
v_mul_f32 v14, v11, v6 // (op 3649) 
v_mul_f32 v15, v11, v7 // (op 3649) 
v_mul_f32 v16, v11, v8 // (op 3649) 
v_mul_f32 v17, v11, v9 // (op 3649) 

v_lshrrev_b32 v6, 16, v14 // (op 3651) 
v_lshrrev_b32 v7, 16, v15 // (op 3651) 
v_lshrrev_b32 v8, 16, v16 // (op 3651) 
v_lshrrev_b32 v9, 16, v17 // (op 3651) 


v_pack_b32_f16 v4, v6, v7 // (op 3637) 
v_pack_b32_f16 v5, v8, v9 // (op 3637) 

s_lshl_b32 s0, s4, 7 // (op 3637) 
s_add_u32 s0, s0, 64 // (op 3637) 
v_add_u32 v0, v13, s0 // (op 3637) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3637) Store value



v_accvgpr_read v6, a28 // (op 3655) 
v_accvgpr_read v7, a29 // (op 3655) 
v_accvgpr_read v8, a30 // (op 3655) 
v_accvgpr_read v9, a31 // (op 3655) 
v_mul_f32 v14, v11, v6 // (op 3655) 
v_mul_f32 v15, v11, v7 // (op 3655) 
v_mul_f32 v16, v11, v8 // (op 3655) 
v_mul_f32 v17, v11, v9 // (op 3655) 

v_lshrrev_b32 v6, 16, v14 // (op 3657) 
v_lshrrev_b32 v7, 16, v15 // (op 3657) 
v_lshrrev_b32 v8, 16, v16 // (op 3657) 
v_lshrrev_b32 v9, 16, v17 // (op 3657) 


v_pack_b32_f16 v4, v6, v7 // (op 3639) 
v_pack_b32_f16 v5, v8, v9 // (op 3639) 

s_lshl_b32 s0, s4, 6 // (op 3639) 
s_mul_i32 s1, 3, s0 // (op 3639) 
s_add_u32 s1, s1, 64 // (op 3639) 

v_add_u32 v0, v13, s1 // (op 3639) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3639) Store value



v_accvgpr_read v6, a32 // (op 3705) 
v_accvgpr_read v7, a33 // (op 3705) 
v_accvgpr_read v8, a34 // (op 3705) 
v_accvgpr_read v9, a35 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 

v_lshrrev_b32 v6, 16, v14 // (op 3707) 
v_lshrrev_b32 v7, 16, v15 // (op 3707) 
v_lshrrev_b32 v8, 16, v16 // (op 3707) 
v_lshrrev_b32 v9, 16, v17 // (op 3707) 


v_pack_b32_f16 v4, v6, v7 // (op 3709) 
v_pack_b32_f16 v5, v8, v9 // (op 3709) 

v_add_u32 v0, 128, v13 // (op 3709) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3709) Store value



v_accvgpr_read v6, a36 // (op 3719) 
v_accvgpr_read v7, a37 // (op 3719) 
v_accvgpr_read v8, a38 // (op 3719) 
v_accvgpr_read v9, a39 // (op 3719) 
v_mul_f32 v14, v11, v6 // (op 3719) 
v_mul_f32 v15, v11, v7 // (op 3719) 
v_mul_f32 v16, v11, v8 // (op 3719) 
v_mul_f32 v17, v11, v9 // (op 3719) 

v_lshrrev_b32 v6, 16, v14 // (op 3721) 
v_lshrrev_b32 v7, 16, v15 // (op 3721) 
v_lshrrev_b32 v8, 16, v16 // (op 3721) 
v_lshrrev_b32 v9, 16, v17 // (op 3721) 


v_pack_b32_f16 v4, v6, v7 // (op 3711) 
v_pack_b32_f16 v5, v8, v9 // (op 3711) 

s_lshl_b32 s0, s4, 6 // (op 3711) 
s_add_u32 s0, s0, 128 // (op 3711) 
v_add_u32 v0, v13, s0 // (op 3711) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3711) Store value



v_accvgpr_read v6, a40 // (op 3725) 
v_accvgpr_read v7, a41 // (op 3725) 
v_accvgpr_read v8, a42 // (op 3725) 
v_accvgpr_read v9, a43 // (op 3725) 
v_mul_f32 v14, v11, v6 // (op 3725) 
v_mul_f32 v15, v11, v7 // (op 3725) 
v_mul_f32 v16, v11, v8 // (op 3725) 
v_mul_f32 v17, v11, v9 // (op 3725) 

v_lshrrev_b32 v6, 16, v14 // (op 3727) 
v_lshrrev_b32 v7, 16, v15 // (op 3727) 
v_lshrrev_b32 v8, 16, v16 // (op 3727) 
v_lshrrev_b32 v9, 16, v17 // (op 3727) 


v_pack_b32_f16 v4, v6, v7 // (op 3713) 
v_pack_b32_f16 v5, v8, v9 // (op 3713) 

s_lshl_b32 s0, s4, 7 // (op 3713) 
s_add_u32 s0, s0, 128 // (op 3713) 
v_add_u32 v0, v13, s0 // (op 3713) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3713) Store value



v_accvgpr_read v6, a44 // (op 3731) 
v_accvgpr_read v7, a45 // (op 3731) 
v_accvgpr_read v8, a46 // (op 3731) 
v_accvgpr_read v9, a47 // (op 3731) 
v_mul_f32 v14, v11, v6 // (op 3731) 
v_mul_f32 v15, v11, v7 // (op 3731) 
v_mul_f32 v16, v11, v8 // (op 3731) 
v_mul_f32 v17, v11, v9 // (op 3731) 

v_lshrrev_b32 v6, 16, v14 // (op 3733) 
v_lshrrev_b32 v7, 16, v15 // (op 3733) 
v_lshrrev_b32 v8, 16, v16 // (op 3733) 
v_lshrrev_b32 v9, 16, v17 // (op 3733) 


v_pack_b32_f16 v4, v6, v7 // (op 3715) 
v_pack_b32_f16 v5, v8, v9 // (op 3715) 

s_lshl_b32 s0, s4, 6 // (op 3715) 
s_mul_i32 s1, 3, s0 // (op 3715) 
s_add_u32 s1, s1, 128 // (op 3715) 

v_add_u32 v0, v13, s1 // (op 3715) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3715) Store value



v_accvgpr_read v6, a48 // (op 3781) 
v_accvgpr_read v7, a49 // (op 3781) 
v_accvgpr_read v8, a50 // (op 3781) 
v_accvgpr_read v9, a51 // (op 3781) 
v_mul_f32 v14, v11, v6 // (op 3781) 
v_mul_f32 v15, v11, v7 // (op 3781) 
v_mul_f32 v16, v11, v8 // (op 3781) 
v_mul_f32 v17, v11, v9 // (op 3781) 

v_lshrrev_b32 v6, 16, v14 // (op 3783) 
v_lshrrev_b32 v7, 16, v15 // (op 3783) 
v_lshrrev_b32 v8, 16, v16 // (op 3783) 
v_lshrrev_b32 v9, 16, v17 // (op 3783) 


v_pack_b32_f16 v4, v6, v7 // (op 3785) 
v_pack_b32_f16 v5, v8, v9 // (op 3785) 

v_add_u32 v0, 192, v13 // (op 3785) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3785) Store value



v_accvgpr_read v6, a52 // (op 3795) 
v_accvgpr_read v7, a53 // (op 3795) 
v_accvgpr_read v8, a54 // (op 3795) 
v_accvgpr_read v9, a55 // (op 3795) 
v_mul_f32 v14, v11, v6 // (op 3795) 
v_mul_f32 v15, v11, v7 // (op 3795) 
v_mul_f32 v16, v11, v8 // (op 3795) 
v_mul_f32 v17, v11, v9 // (op 3795) 

v_lshrrev_b32 v6, 16, v14 // (op 3797) 
v_lshrrev_b32 v7, 16, v15 // (op 3797) 
v_lshrrev_b32 v8, 16, v16 // (op 3797) 
v_lshrrev_b32 v9, 16, v17 // (op 3797) 


v_pack_b32_f16 v4, v6, v7 // (op 3787) 
v_pack_b32_f16 v5, v8, v9 // (op 3787) 

s_lshl_b32 s0, s4, 6 // (op 3787) 
s_add_u32 s0, s0, 192 // (op 3787) 
v_add_u32 v0, v13, s0 // (op 3787) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3787) Store value



v_accvgpr_read v6, a56 // (op 3801) 
v_accvgpr_read v7, a57 // (op 3801) 
v_accvgpr_read v8, a58 // (op 3801) 
v_accvgpr_read v9, a59 // (op 3801) 
v_mul_f32 v14, v11, v6 // (op 3801) 
v_mul_f32 v15, v11, v7 // (op 3801) 
v_mul_f32 v16, v11, v8 // (op 3801) 
v_mul_f32 v17, v11, v9 // (op 3801) 

v_lshrrev_b32 v6, 16, v14 // (op 3803) 
v_lshrrev_b32 v7, 16, v15 // (op 3803) 
v_lshrrev_b32 v8, 16, v16 // (op 3803) 
v_lshrrev_b32 v9, 16, v17 // (op 3803) 


v_pack_b32_f16 v4, v6, v7 // (op 3789) 
v_pack_b32_f16 v5, v8, v9 // (op 3789) 

s_lshl_b32 s0, s4, 7 // (op 3789) 
s_add_u32 s0, s0, 192 // (op 3789) 
v_add_u32 v0, v13, s0 // (op 3789) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3789) Store value



v_accvgpr_read v6, a60 // (op 3807) 
v_accvgpr_read v7, a61 // (op 3807) 
v_accvgpr_read v8, a62 // (op 3807) 
v_accvgpr_read v9, a63 // (op 3807) 
v_mul_f32 v14, v11, v6 // (op 3807) 
v_mul_f32 v15, v11, v7 // (op 3807) 
v_mul_f32 v16, v11, v8 // (op 3807) 
v_mul_f32 v17, v11, v9 // (op 3807) 

v_lshrrev_b32 v6, 16, v14 // (op 3809) 
v_lshrrev_b32 v7, 16, v15 // (op 3809) 
v_lshrrev_b32 v8, 16, v16 // (op 3809) 
v_lshrrev_b32 v9, 16, v17 // (op 3809) 


v_pack_b32_f16 v4, v6, v7 // (op 3791) 
v_pack_b32_f16 v5, v8, v9 // (op 3791) 

s_lshl_b32 s0, s4, 6 // (op 3791) 
s_mul_i32 s1, 3, s0 // (op 3791) 
s_add_u32 s1, s1, 192 // (op 3791) 

v_add_u32 v0, v13, s1 // (op 3791) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 3791) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 192, v12 // (op 4179) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4179) Load value

v_accvgpr_read v4, a48 // (op 4166) 
v_accvgpr_read v5, a49 // (op 4166) 
v_accvgpr_read v6, a50 // (op 4166) 
v_accvgpr_read v7, a51 // (op 4166) 
v_mul_f32 v14, v11, v4 // (op 4166) 
v_mul_f32 v15, v11, v5 // (op 4166) 
v_mul_f32 v16, v11, v6 // (op 4166) 
v_mul_f32 v17, v11, v7 // (op 4166) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4178) 
v_lshrrev_b32 v7, 16, v8 // (op 4178) 
v_lshlrev_b32 v6, 16, v6 // (op 4178) 
v_lshlrev_b32 v7, 16, v7 // (op 4178) 
v_mul_f32 v18, v10, v6 // (op 4178) 
v_mul_f32 v19, v10, v7 // (op 4178) 
v_and_b32 v6, 65535, v9 // (op 4178) 
v_lshrrev_b32 v7, 16, v9 // (op 4178) 
v_lshlrev_b32 v6, 16, v6 // (op 4178) 
v_lshlrev_b32 v7, 16, v7 // (op 4178) 
v_mul_f32 v20, v10, v6 // (op 4178) 
v_mul_f32 v21, v10, v7 // (op 4178) 


v_add_f32 v6, v18, v14 // (op 4167) 
v_add_f32 v7, v19, v15 // (op 4167) 
v_add_f32 v8, v20, v16 // (op 4167) 
v_add_f32 v9, v21, v17 // (op 4167) 


v_lshrrev_b32 v14, 16, v6 // (op 4168) 
v_lshrrev_b32 v15, 16, v7 // (op 4168) 
v_lshrrev_b32 v16, 16, v8 // (op 4168) 
v_lshrrev_b32 v17, 16, v9 // (op 4168) 


v_pack_b32_f16 v8, v14, v15 // (op 4170) 
v_pack_b32_f16 v9, v16, v17 // (op 4170) 

v_add_u32 v0, 192, v13 // (op 4170) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4170) Store value





s_lshl_b32 s0, s2, 6 // (op 4184) 
s_add_u32 s0, s0, 192 // (op 4184) 
v_add_u32 v0, v12, s0 // (op 4184) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4184) Load value

v_accvgpr_read v4, a52 // (op 4196) 
v_accvgpr_read v5, a53 // (op 4196) 
v_accvgpr_read v6, a54 // (op 4196) 
v_accvgpr_read v7, a55 // (op 4196) 
v_mul_f32 v14, v11, v4 // (op 4196) 
v_mul_f32 v15, v11, v5 // (op 4196) 
v_mul_f32 v16, v11, v6 // (op 4196) 
v_mul_f32 v17, v11, v7 // (op 4196) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4181) 
v_lshrrev_b32 v7, 16, v8 // (op 4181) 
v_lshlrev_b32 v6, 16, v6 // (op 4181) 
v_lshlrev_b32 v7, 16, v7 // (op 4181) 
v_mul_f32 v18, v10, v6 // (op 4181) 
v_mul_f32 v19, v10, v7 // (op 4181) 
v_and_b32 v6, 65535, v9 // (op 4181) 
v_lshrrev_b32 v7, 16, v9 // (op 4181) 
v_lshlrev_b32 v6, 16, v6 // (op 4181) 
v_lshlrev_b32 v7, 16, v7 // (op 4181) 
v_mul_f32 v20, v10, v6 // (op 4181) 
v_mul_f32 v21, v10, v7 // (op 4181) 


v_add_f32 v6, v18, v14 // (op 4182) 
v_add_f32 v7, v19, v15 // (op 4182) 
v_add_f32 v8, v20, v16 // (op 4182) 
v_add_f32 v9, v21, v17 // (op 4182) 


v_lshrrev_b32 v14, 16, v6 // (op 4183) 
v_lshrrev_b32 v15, 16, v7 // (op 4183) 
v_lshrrev_b32 v16, 16, v8 // (op 4183) 
v_lshrrev_b32 v17, 16, v9 // (op 4183) 


v_pack_b32_f16 v8, v14, v15 // (op 4172) 
v_pack_b32_f16 v9, v16, v17 // (op 4172) 

s_lshl_b32 s0, s4, 6 // (op 4172) 
s_add_u32 s0, s0, 192 // (op 4172) 
v_add_u32 v0, v13, s0 // (op 4172) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4172) Store value





s_lshl_b32 s0, s2, 7 // (op 4189) 
s_add_u32 s0, s0, 192 // (op 4189) 
v_add_u32 v0, v12, s0 // (op 4189) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4189) Load value

v_accvgpr_read v4, a56 // (op 4198) 
v_accvgpr_read v5, a57 // (op 4198) 
v_accvgpr_read v6, a58 // (op 4198) 
v_accvgpr_read v7, a59 // (op 4198) 
v_mul_f32 v14, v11, v4 // (op 4198) 
v_mul_f32 v15, v11, v5 // (op 4198) 
v_mul_f32 v16, v11, v6 // (op 4198) 
v_mul_f32 v17, v11, v7 // (op 4198) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4186) 
v_lshrrev_b32 v7, 16, v8 // (op 4186) 
v_lshlrev_b32 v6, 16, v6 // (op 4186) 
v_lshlrev_b32 v7, 16, v7 // (op 4186) 
v_mul_f32 v18, v10, v6 // (op 4186) 
v_mul_f32 v19, v10, v7 // (op 4186) 
v_and_b32 v6, 65535, v9 // (op 4186) 
v_lshrrev_b32 v7, 16, v9 // (op 4186) 
v_lshlrev_b32 v6, 16, v6 // (op 4186) 
v_lshlrev_b32 v7, 16, v7 // (op 4186) 
v_mul_f32 v20, v10, v6 // (op 4186) 
v_mul_f32 v21, v10, v7 // (op 4186) 


v_add_f32 v6, v18, v14 // (op 4187) 
v_add_f32 v7, v19, v15 // (op 4187) 
v_add_f32 v8, v20, v16 // (op 4187) 
v_add_f32 v9, v21, v17 // (op 4187) 


v_lshrrev_b32 v14, 16, v6 // (op 4188) 
v_lshrrev_b32 v15, 16, v7 // (op 4188) 
v_lshrrev_b32 v16, 16, v8 // (op 4188) 
v_lshrrev_b32 v17, 16, v9 // (op 4188) 


v_pack_b32_f16 v8, v14, v15 // (op 4174) 
v_pack_b32_f16 v9, v16, v17 // (op 4174) 

s_lshl_b32 s0, s4, 7 // (op 4174) 
s_add_u32 s0, s0, 192 // (op 4174) 
v_add_u32 v0, v13, s0 // (op 4174) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4174) Store value





s_lshl_b32 s0, s2, 6 // (op 4194) 
s_mul_i32 s1, 3, s0 // (op 4194) 
s_add_u32 s1, s1, 192 // (op 4194) 

v_add_u32 v0, v12, s1 // (op 4194) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4194) Load value

v_accvgpr_read v4, a60 // (op 4200) 
v_accvgpr_read v5, a61 // (op 4200) 
v_accvgpr_read v6, a62 // (op 4200) 
v_accvgpr_read v7, a63 // (op 4200) 
v_mul_f32 v14, v11, v4 // (op 4200) 
v_mul_f32 v15, v11, v5 // (op 4200) 
v_mul_f32 v16, v11, v6 // (op 4200) 
v_mul_f32 v17, v11, v7 // (op 4200) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4191) 
v_lshrrev_b32 v7, 16, v8 // (op 4191) 
v_lshlrev_b32 v6, 16, v6 // (op 4191) 
v_lshlrev_b32 v7, 16, v7 // (op 4191) 
v_mul_f32 v18, v10, v6 // (op 4191) 
v_mul_f32 v19, v10, v7 // (op 4191) 
v_and_b32 v6, 65535, v9 // (op 4191) 
v_lshrrev_b32 v7, 16, v9 // (op 4191) 
v_lshlrev_b32 v6, 16, v6 // (op 4191) 
v_lshlrev_b32 v7, 16, v7 // (op 4191) 
v_mul_f32 v20, v10, v6 // (op 4191) 
v_mul_f32 v21, v10, v7 // (op 4191) 


v_add_f32 v6, v18, v14 // (op 4192) 
v_add_f32 v7, v19, v15 // (op 4192) 
v_add_f32 v8, v20, v16 // (op 4192) 
v_add_f32 v9, v21, v17 // (op 4192) 


v_lshrrev_b32 v14, 16, v6 // (op 4193) 
v_lshrrev_b32 v15, 16, v7 // (op 4193) 
v_lshrrev_b32 v16, 16, v8 // (op 4193) 
v_lshrrev_b32 v17, 16, v9 // (op 4193) 


v_pack_b32_f16 v8, v14, v15 // (op 4176) 
v_pack_b32_f16 v9, v16, v17 // (op 4176) 

s_lshl_b32 s0, s4, 6 // (op 4176) 
s_mul_i32 s1, 3, s0 // (op 4176) 
s_add_u32 s1, s1, 192 // (op 4176) 

v_add_u32 v0, v13, s1 // (op 4176) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4176) Store value





v_add_u32 v0, 128, v12 // (op 4092) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4092) Load value

v_accvgpr_read v4, a32 // (op 4079) 
v_accvgpr_read v5, a33 // (op 4079) 
v_accvgpr_read v6, a34 // (op 4079) 
v_accvgpr_read v7, a35 // (op 4079) 
v_mul_f32 v14, v11, v4 // (op 4079) 
v_mul_f32 v15, v11, v5 // (op 4079) 
v_mul_f32 v16, v11, v6 // (op 4079) 
v_mul_f32 v17, v11, v7 // (op 4079) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4091) 
v_lshrrev_b32 v7, 16, v8 // (op 4091) 
v_lshlrev_b32 v6, 16, v6 // (op 4091) 
v_lshlrev_b32 v7, 16, v7 // (op 4091) 
v_mul_f32 v18, v10, v6 // (op 4091) 
v_mul_f32 v19, v10, v7 // (op 4091) 
v_and_b32 v6, 65535, v9 // (op 4091) 
v_lshrrev_b32 v7, 16, v9 // (op 4091) 
v_lshlrev_b32 v6, 16, v6 // (op 4091) 
v_lshlrev_b32 v7, 16, v7 // (op 4091) 
v_mul_f32 v20, v10, v6 // (op 4091) 
v_mul_f32 v21, v10, v7 // (op 4091) 


v_add_f32 v6, v18, v14 // (op 4080) 
v_add_f32 v7, v19, v15 // (op 4080) 
v_add_f32 v8, v20, v16 // (op 4080) 
v_add_f32 v9, v21, v17 // (op 4080) 


v_lshrrev_b32 v14, 16, v6 // (op 4081) 
v_lshrrev_b32 v15, 16, v7 // (op 4081) 
v_lshrrev_b32 v16, 16, v8 // (op 4081) 
v_lshrrev_b32 v17, 16, v9 // (op 4081) 


v_pack_b32_f16 v8, v14, v15 // (op 4083) 
v_pack_b32_f16 v9, v16, v17 // (op 4083) 

v_add_u32 v0, 128, v13 // (op 4083) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4083) Store value





s_lshl_b32 s0, s2, 6 // (op 4097) 
s_add_u32 s0, s0, 128 // (op 4097) 
v_add_u32 v0, v12, s0 // (op 4097) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4097) Load value

v_accvgpr_read v4, a36 // (op 4109) 
v_accvgpr_read v5, a37 // (op 4109) 
v_accvgpr_read v6, a38 // (op 4109) 
v_accvgpr_read v7, a39 // (op 4109) 
v_mul_f32 v14, v11, v4 // (op 4109) 
v_mul_f32 v15, v11, v5 // (op 4109) 
v_mul_f32 v16, v11, v6 // (op 4109) 
v_mul_f32 v17, v11, v7 // (op 4109) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4094) 
v_lshrrev_b32 v7, 16, v8 // (op 4094) 
v_lshlrev_b32 v6, 16, v6 // (op 4094) 
v_lshlrev_b32 v7, 16, v7 // (op 4094) 
v_mul_f32 v18, v10, v6 // (op 4094) 
v_mul_f32 v19, v10, v7 // (op 4094) 
v_and_b32 v6, 65535, v9 // (op 4094) 
v_lshrrev_b32 v7, 16, v9 // (op 4094) 
v_lshlrev_b32 v6, 16, v6 // (op 4094) 
v_lshlrev_b32 v7, 16, v7 // (op 4094) 
v_mul_f32 v20, v10, v6 // (op 4094) 
v_mul_f32 v21, v10, v7 // (op 4094) 


v_add_f32 v6, v18, v14 // (op 4095) 
v_add_f32 v7, v19, v15 // (op 4095) 
v_add_f32 v8, v20, v16 // (op 4095) 
v_add_f32 v9, v21, v17 // (op 4095) 


v_lshrrev_b32 v14, 16, v6 // (op 4096) 
v_lshrrev_b32 v15, 16, v7 // (op 4096) 
v_lshrrev_b32 v16, 16, v8 // (op 4096) 
v_lshrrev_b32 v17, 16, v9 // (op 4096) 


v_pack_b32_f16 v8, v14, v15 // (op 4085) 
v_pack_b32_f16 v9, v16, v17 // (op 4085) 

s_lshl_b32 s0, s4, 6 // (op 4085) 
s_add_u32 s0, s0, 128 // (op 4085) 
v_add_u32 v0, v13, s0 // (op 4085) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4085) Store value





s_lshl_b32 s0, s2, 7 // (op 4102) 
s_add_u32 s0, s0, 128 // (op 4102) 
v_add_u32 v0, v12, s0 // (op 4102) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4102) Load value

v_accvgpr_read v4, a40 // (op 4111) 
v_accvgpr_read v5, a41 // (op 4111) 
v_accvgpr_read v6, a42 // (op 4111) 
v_accvgpr_read v7, a43 // (op 4111) 
v_mul_f32 v14, v11, v4 // (op 4111) 
v_mul_f32 v15, v11, v5 // (op 4111) 
v_mul_f32 v16, v11, v6 // (op 4111) 
v_mul_f32 v17, v11, v7 // (op 4111) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4099) 
v_lshrrev_b32 v7, 16, v8 // (op 4099) 
v_lshlrev_b32 v6, 16, v6 // (op 4099) 
v_lshlrev_b32 v7, 16, v7 // (op 4099) 
v_mul_f32 v18, v10, v6 // (op 4099) 
v_mul_f32 v19, v10, v7 // (op 4099) 
v_and_b32 v6, 65535, v9 // (op 4099) 
v_lshrrev_b32 v7, 16, v9 // (op 4099) 
v_lshlrev_b32 v6, 16, v6 // (op 4099) 
v_lshlrev_b32 v7, 16, v7 // (op 4099) 
v_mul_f32 v20, v10, v6 // (op 4099) 
v_mul_f32 v21, v10, v7 // (op 4099) 


v_add_f32 v6, v18, v14 // (op 4100) 
v_add_f32 v7, v19, v15 // (op 4100) 
v_add_f32 v8, v20, v16 // (op 4100) 
v_add_f32 v9, v21, v17 // (op 4100) 


v_lshrrev_b32 v14, 16, v6 // (op 4101) 
v_lshrrev_b32 v15, 16, v7 // (op 4101) 
v_lshrrev_b32 v16, 16, v8 // (op 4101) 
v_lshrrev_b32 v17, 16, v9 // (op 4101) 


v_pack_b32_f16 v8, v14, v15 // (op 4087) 
v_pack_b32_f16 v9, v16, v17 // (op 4087) 

s_lshl_b32 s0, s4, 7 // (op 4087) 
s_add_u32 s0, s0, 128 // (op 4087) 
v_add_u32 v0, v13, s0 // (op 4087) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4087) Store value





s_lshl_b32 s0, s2, 6 // (op 4107) 
s_mul_i32 s1, 3, s0 // (op 4107) 
s_add_u32 s1, s1, 128 // (op 4107) 

v_add_u32 v0, v12, s1 // (op 4107) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4107) Load value

v_accvgpr_read v4, a44 // (op 4113) 
v_accvgpr_read v5, a45 // (op 4113) 
v_accvgpr_read v6, a46 // (op 4113) 
v_accvgpr_read v7, a47 // (op 4113) 
v_mul_f32 v14, v11, v4 // (op 4113) 
v_mul_f32 v15, v11, v5 // (op 4113) 
v_mul_f32 v16, v11, v6 // (op 4113) 
v_mul_f32 v17, v11, v7 // (op 4113) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4104) 
v_lshrrev_b32 v7, 16, v8 // (op 4104) 
v_lshlrev_b32 v6, 16, v6 // (op 4104) 
v_lshlrev_b32 v7, 16, v7 // (op 4104) 
v_mul_f32 v18, v10, v6 // (op 4104) 
v_mul_f32 v19, v10, v7 // (op 4104) 
v_and_b32 v6, 65535, v9 // (op 4104) 
v_lshrrev_b32 v7, 16, v9 // (op 4104) 
v_lshlrev_b32 v6, 16, v6 // (op 4104) 
v_lshlrev_b32 v7, 16, v7 // (op 4104) 
v_mul_f32 v20, v10, v6 // (op 4104) 
v_mul_f32 v21, v10, v7 // (op 4104) 


v_add_f32 v6, v18, v14 // (op 4105) 
v_add_f32 v7, v19, v15 // (op 4105) 
v_add_f32 v8, v20, v16 // (op 4105) 
v_add_f32 v9, v21, v17 // (op 4105) 


v_lshrrev_b32 v14, 16, v6 // (op 4106) 
v_lshrrev_b32 v15, 16, v7 // (op 4106) 
v_lshrrev_b32 v16, 16, v8 // (op 4106) 
v_lshrrev_b32 v17, 16, v9 // (op 4106) 


v_pack_b32_f16 v8, v14, v15 // (op 4089) 
v_pack_b32_f16 v9, v16, v17 // (op 4089) 

s_lshl_b32 s0, s4, 6 // (op 4089) 
s_mul_i32 s1, 3, s0 // (op 4089) 
s_add_u32 s1, s1, 128 // (op 4089) 

v_add_u32 v0, v13, s1 // (op 4089) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4089) Store value





v_add_u32 v0, 64, v12 // (op 4005) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4005) Load value

v_accvgpr_read v4, a16 // (op 3992) 
v_accvgpr_read v5, a17 // (op 3992) 
v_accvgpr_read v6, a18 // (op 3992) 
v_accvgpr_read v7, a19 // (op 3992) 
v_mul_f32 v14, v11, v4 // (op 3992) 
v_mul_f32 v15, v11, v5 // (op 3992) 
v_mul_f32 v16, v11, v6 // (op 3992) 
v_mul_f32 v17, v11, v7 // (op 3992) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4004) 
v_lshrrev_b32 v7, 16, v8 // (op 4004) 
v_lshlrev_b32 v6, 16, v6 // (op 4004) 
v_lshlrev_b32 v7, 16, v7 // (op 4004) 
v_mul_f32 v18, v10, v6 // (op 4004) 
v_mul_f32 v19, v10, v7 // (op 4004) 
v_and_b32 v6, 65535, v9 // (op 4004) 
v_lshrrev_b32 v7, 16, v9 // (op 4004) 
v_lshlrev_b32 v6, 16, v6 // (op 4004) 
v_lshlrev_b32 v7, 16, v7 // (op 4004) 
v_mul_f32 v20, v10, v6 // (op 4004) 
v_mul_f32 v21, v10, v7 // (op 4004) 


v_add_f32 v6, v18, v14 // (op 3993) 
v_add_f32 v7, v19, v15 // (op 3993) 
v_add_f32 v8, v20, v16 // (op 3993) 
v_add_f32 v9, v21, v17 // (op 3993) 


v_lshrrev_b32 v14, 16, v6 // (op 3994) 
v_lshrrev_b32 v15, 16, v7 // (op 3994) 
v_lshrrev_b32 v16, 16, v8 // (op 3994) 
v_lshrrev_b32 v17, 16, v9 // (op 3994) 


v_pack_b32_f16 v8, v14, v15 // (op 3996) 
v_pack_b32_f16 v9, v16, v17 // (op 3996) 

v_add_u32 v0, 64, v13 // (op 3996) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3996) Store value





s_lshl_b32 s0, s2, 6 // (op 4010) 
s_add_u32 s0, s0, 64 // (op 4010) 
v_add_u32 v0, v12, s0 // (op 4010) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4010) Load value

v_accvgpr_read v4, a20 // (op 4022) 
v_accvgpr_read v5, a21 // (op 4022) 
v_accvgpr_read v6, a22 // (op 4022) 
v_accvgpr_read v7, a23 // (op 4022) 
v_mul_f32 v14, v11, v4 // (op 4022) 
v_mul_f32 v15, v11, v5 // (op 4022) 
v_mul_f32 v16, v11, v6 // (op 4022) 
v_mul_f32 v17, v11, v7 // (op 4022) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4007) 
v_lshrrev_b32 v7, 16, v8 // (op 4007) 
v_lshlrev_b32 v6, 16, v6 // (op 4007) 
v_lshlrev_b32 v7, 16, v7 // (op 4007) 
v_mul_f32 v18, v10, v6 // (op 4007) 
v_mul_f32 v19, v10, v7 // (op 4007) 
v_and_b32 v6, 65535, v9 // (op 4007) 
v_lshrrev_b32 v7, 16, v9 // (op 4007) 
v_lshlrev_b32 v6, 16, v6 // (op 4007) 
v_lshlrev_b32 v7, 16, v7 // (op 4007) 
v_mul_f32 v20, v10, v6 // (op 4007) 
v_mul_f32 v21, v10, v7 // (op 4007) 


v_add_f32 v6, v18, v14 // (op 4008) 
v_add_f32 v7, v19, v15 // (op 4008) 
v_add_f32 v8, v20, v16 // (op 4008) 
v_add_f32 v9, v21, v17 // (op 4008) 


v_lshrrev_b32 v14, 16, v6 // (op 4009) 
v_lshrrev_b32 v15, 16, v7 // (op 4009) 
v_lshrrev_b32 v16, 16, v8 // (op 4009) 
v_lshrrev_b32 v17, 16, v9 // (op 4009) 


v_pack_b32_f16 v8, v14, v15 // (op 3998) 
v_pack_b32_f16 v9, v16, v17 // (op 3998) 

s_lshl_b32 s0, s4, 6 // (op 3998) 
s_add_u32 s0, s0, 64 // (op 3998) 
v_add_u32 v0, v13, s0 // (op 3998) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3998) Store value





s_lshl_b32 s0, s2, 7 // (op 4015) 
s_add_u32 s0, s0, 64 // (op 4015) 
v_add_u32 v0, v12, s0 // (op 4015) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4015) Load value

v_accvgpr_read v4, a24 // (op 4024) 
v_accvgpr_read v5, a25 // (op 4024) 
v_accvgpr_read v6, a26 // (op 4024) 
v_accvgpr_read v7, a27 // (op 4024) 
v_mul_f32 v14, v11, v4 // (op 4024) 
v_mul_f32 v15, v11, v5 // (op 4024) 
v_mul_f32 v16, v11, v6 // (op 4024) 
v_mul_f32 v17, v11, v7 // (op 4024) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4012) 
v_lshrrev_b32 v7, 16, v8 // (op 4012) 
v_lshlrev_b32 v6, 16, v6 // (op 4012) 
v_lshlrev_b32 v7, 16, v7 // (op 4012) 
v_mul_f32 v18, v10, v6 // (op 4012) 
v_mul_f32 v19, v10, v7 // (op 4012) 
v_and_b32 v6, 65535, v9 // (op 4012) 
v_lshrrev_b32 v7, 16, v9 // (op 4012) 
v_lshlrev_b32 v6, 16, v6 // (op 4012) 
v_lshlrev_b32 v7, 16, v7 // (op 4012) 
v_mul_f32 v20, v10, v6 // (op 4012) 
v_mul_f32 v21, v10, v7 // (op 4012) 


v_add_f32 v6, v18, v14 // (op 4013) 
v_add_f32 v7, v19, v15 // (op 4013) 
v_add_f32 v8, v20, v16 // (op 4013) 
v_add_f32 v9, v21, v17 // (op 4013) 


v_lshrrev_b32 v14, 16, v6 // (op 4014) 
v_lshrrev_b32 v15, 16, v7 // (op 4014) 
v_lshrrev_b32 v16, 16, v8 // (op 4014) 
v_lshrrev_b32 v17, 16, v9 // (op 4014) 


v_pack_b32_f16 v8, v14, v15 // (op 4000) 
v_pack_b32_f16 v9, v16, v17 // (op 4000) 

s_lshl_b32 s0, s4, 7 // (op 4000) 
s_add_u32 s0, s0, 64 // (op 4000) 
v_add_u32 v0, v13, s0 // (op 4000) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4000) Store value





s_lshl_b32 s0, s2, 6 // (op 4020) 
s_mul_i32 s1, 3, s0 // (op 4020) 
s_add_u32 s1, s1, 64 // (op 4020) 

v_add_u32 v0, v12, s1 // (op 4020) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4020) Load value

v_accvgpr_read v4, a28 // (op 4026) 
v_accvgpr_read v5, a29 // (op 4026) 
v_accvgpr_read v6, a30 // (op 4026) 
v_accvgpr_read v7, a31 // (op 4026) 
v_mul_f32 v14, v11, v4 // (op 4026) 
v_mul_f32 v15, v11, v5 // (op 4026) 
v_mul_f32 v16, v11, v6 // (op 4026) 
v_mul_f32 v17, v11, v7 // (op 4026) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4017) 
v_lshrrev_b32 v7, 16, v8 // (op 4017) 
v_lshlrev_b32 v6, 16, v6 // (op 4017) 
v_lshlrev_b32 v7, 16, v7 // (op 4017) 
v_mul_f32 v18, v10, v6 // (op 4017) 
v_mul_f32 v19, v10, v7 // (op 4017) 
v_and_b32 v6, 65535, v9 // (op 4017) 
v_lshrrev_b32 v7, 16, v9 // (op 4017) 
v_lshlrev_b32 v6, 16, v6 // (op 4017) 
v_lshlrev_b32 v7, 16, v7 // (op 4017) 
v_mul_f32 v20, v10, v6 // (op 4017) 
v_mul_f32 v21, v10, v7 // (op 4017) 


v_add_f32 v6, v18, v14 // (op 4018) 
v_add_f32 v7, v19, v15 // (op 4018) 
v_add_f32 v8, v20, v16 // (op 4018) 
v_add_f32 v9, v21, v17 // (op 4018) 


v_lshrrev_b32 v14, 16, v6 // (op 4019) 
v_lshrrev_b32 v15, 16, v7 // (op 4019) 
v_lshrrev_b32 v16, 16, v8 // (op 4019) 
v_lshrrev_b32 v17, 16, v9 // (op 4019) 


v_pack_b32_f16 v8, v14, v15 // (op 4002) 
v_pack_b32_f16 v9, v16, v17 // (op 4002) 

s_lshl_b32 s0, s4, 6 // (op 4002) 
s_mul_i32 s1, 3, s0 // (op 4002) 
s_add_u32 s1, s1, 64 // (op 4002) 

v_add_u32 v0, v13, s1 // (op 4002) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 4002) Store value





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





s_lshl_b32 s0, s2, 6 // (op 3913) 
v_add_u32 v0, v12, s0 // (op 3913) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3913) Load value

v_accvgpr_read v4, a4 // (op 3909) 
v_accvgpr_read v5, a5 // (op 3909) 
v_accvgpr_read v6, a6 // (op 3909) 
v_accvgpr_read v7, a7 // (op 3909) 
v_mul_f32 v14, v11, v4 // (op 3909) 
v_mul_f32 v15, v11, v5 // (op 3909) 
v_mul_f32 v16, v11, v6 // (op 3909) 
v_mul_f32 v17, v11, v7 // (op 3909) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3914) 
v_lshrrev_b32 v7, 16, v8 // (op 3914) 
v_lshlrev_b32 v6, 16, v6 // (op 3914) 
v_lshlrev_b32 v7, 16, v7 // (op 3914) 
v_mul_f32 v18, v10, v6 // (op 3914) 
v_mul_f32 v19, v10, v7 // (op 3914) 
v_and_b32 v6, 65535, v9 // (op 3914) 
v_lshrrev_b32 v7, 16, v9 // (op 3914) 
v_lshlrev_b32 v6, 16, v6 // (op 3914) 
v_lshlrev_b32 v7, 16, v7 // (op 3914) 
v_mul_f32 v20, v10, v6 // (op 3914) 
v_mul_f32 v21, v10, v7 // (op 3914) 


v_add_f32 v6, v18, v14 // (op 3910) 
v_add_f32 v7, v19, v15 // (op 3910) 
v_add_f32 v8, v20, v16 // (op 3910) 
v_add_f32 v9, v21, v17 // (op 3910) 


v_lshrrev_b32 v14, 16, v6 // (op 3911) 
v_lshrrev_b32 v15, 16, v7 // (op 3911) 
v_lshrrev_b32 v16, 16, v8 // (op 3911) 
v_lshrrev_b32 v17, 16, v9 // (op 3911) 


v_pack_b32_f16 v8, v14, v15 // (op 3912) 
v_pack_b32_f16 v9, v16, v17 // (op 3912) 

s_lshl_b32 s0, s4, 6 // (op 3912) 
v_add_u32 v0, v13, s0 // (op 3912) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3912) Store value





s_lshl_b32 s0, s2, 7 // (op 3927) 
v_add_u32 v0, v12, s0 // (op 3927) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3927) Load value

v_accvgpr_read v4, a8 // (op 3923) 
v_accvgpr_read v5, a9 // (op 3923) 
v_accvgpr_read v6, a10 // (op 3923) 
v_accvgpr_read v7, a11 // (op 3923) 
v_mul_f32 v14, v11, v4 // (op 3923) 
v_mul_f32 v15, v11, v5 // (op 3923) 
v_mul_f32 v16, v11, v6 // (op 3923) 
v_mul_f32 v17, v11, v7 // (op 3923) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3928) 
v_lshrrev_b32 v7, 16, v8 // (op 3928) 
v_lshlrev_b32 v6, 16, v6 // (op 3928) 
v_lshlrev_b32 v7, 16, v7 // (op 3928) 
v_mul_f32 v18, v10, v6 // (op 3928) 
v_mul_f32 v19, v10, v7 // (op 3928) 
v_and_b32 v6, 65535, v9 // (op 3928) 
v_lshrrev_b32 v7, 16, v9 // (op 3928) 
v_lshlrev_b32 v6, 16, v6 // (op 3928) 
v_lshlrev_b32 v7, 16, v7 // (op 3928) 
v_mul_f32 v20, v10, v6 // (op 3928) 
v_mul_f32 v21, v10, v7 // (op 3928) 


v_add_f32 v6, v18, v14 // (op 3924) 
v_add_f32 v7, v19, v15 // (op 3924) 
v_add_f32 v8, v20, v16 // (op 3924) 
v_add_f32 v9, v21, v17 // (op 3924) 


v_lshrrev_b32 v14, 16, v6 // (op 3925) 
v_lshrrev_b32 v15, 16, v7 // (op 3925) 
v_lshrrev_b32 v16, 16, v8 // (op 3925) 
v_lshrrev_b32 v17, 16, v9 // (op 3925) 


v_pack_b32_f16 v8, v14, v15 // (op 3926) 
v_pack_b32_f16 v9, v16, v17 // (op 3926) 

s_lshl_b32 s0, s4, 7 // (op 3926) 
v_add_u32 v0, v13, s0 // (op 3926) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3926) Store value





s_lshl_b32 s0, s2, 6 // (op 3941) 
s_mul_i32 s1, 3, s0 // (op 3941) 

v_add_u32 v0, v12, s1 // (op 3941) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3941) Load value

v_accvgpr_read v4, a12 // (op 3937) 
v_accvgpr_read v5, a13 // (op 3937) 
v_accvgpr_read v6, a14 // (op 3937) 
v_accvgpr_read v7, a15 // (op 3937) 
v_mul_f32 v14, v11, v4 // (op 3937) 
v_mul_f32 v15, v11, v5 // (op 3937) 
v_mul_f32 v16, v11, v6 // (op 3937) 
v_mul_f32 v17, v11, v7 // (op 3937) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3942) 
v_lshrrev_b32 v7, 16, v8 // (op 3942) 
v_lshlrev_b32 v6, 16, v6 // (op 3942) 
v_lshlrev_b32 v7, 16, v7 // (op 3942) 
v_mul_f32 v18, v10, v6 // (op 3942) 
v_mul_f32 v19, v10, v7 // (op 3942) 
v_and_b32 v6, 65535, v9 // (op 3942) 
v_lshrrev_b32 v7, 16, v9 // (op 3942) 
v_lshlrev_b32 v6, 16, v6 // (op 3942) 
v_lshlrev_b32 v7, 16, v7 // (op 3942) 
v_mul_f32 v20, v10, v6 // (op 3942) 
v_mul_f32 v21, v10, v7 // (op 3942) 


v_add_f32 v6, v18, v14 // (op 3938) 
v_add_f32 v7, v19, v15 // (op 3938) 
v_add_f32 v8, v20, v16 // (op 3938) 
v_add_f32 v9, v21, v17 // (op 3938) 


v_lshrrev_b32 v14, 16, v6 // (op 3939) 
v_lshrrev_b32 v15, 16, v7 // (op 3939) 
v_lshrrev_b32 v16, 16, v8 // (op 3939) 
v_lshrrev_b32 v17, 16, v9 // (op 3939) 


v_pack_b32_f16 v8, v14, v15 // (op 3940) 
v_pack_b32_f16 v9, v16, v17 // (op 3940) 

s_lshl_b32 s0, s4, 6 // (op 3940) 
s_mul_i32 s1, 3, s0 // (op 3940) 

v_add_u32 v0, v13, s1 // (op 3940) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 3940) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
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
    DestDataType: b
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock },
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
  - .name: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_TT_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__LDSSYNC.kd
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicMultiple_30
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
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: Subtract_24
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
      - .name: Subtract_21
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

