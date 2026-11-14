// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_,@function
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 3692) 
s_mul_hi_i32 s1, s16, s11 // (op 3881) 
v_lshrrev_b32 v0, 7, v1 // (op 3749) 
v_and_b32 v3, 15, v1 // (op 3749) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3749) 


v_and_b32 v0, 63, v1 // (op 3749) 
v_lshrrev_b32 v3, 4, v0 // (op 3749) 

v_lshlrev_b32 v0, 5, v3 // (op 3749) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3749) 


v_lshlrev_b32 v0, 2, v3 // (op 3749) 

v_lshrrev_b32 v3, 3, v0 // (op 3749) 

v_lshrrev_b32 v0, 7, v1 // (op 3907) 
v_and_b32 v4, 15, v1 // (op 3907) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3907) 


v_and_b32 v0, 63, v1 // (op 3907) 
v_lshrrev_b32 v4, 4, v0 // (op 3907) 

v_lshlrev_b32 v0, 5, v4 // (op 3907) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3907) 


v_lshlrev_b32 v0, 2, v4 // (op 3907) 

v_lshrrev_b32 v4, 3, v0 // (op 3907) 

v_lshrrev_b32 v0, 7, v1 // (op 3692) 
s_add_i32 s18, s0, s16 // (op 3692) 

v_and_b32 v5, 63, v1 // (op 3692) 
s_ashr_i32 s0, s18, 31 // (op 3692) 
v_lshrrev_b32 v6, 4, v5 // (op 3692) 
s_cmp_eq_i32 s11, 0 // (op 3692) 
s_mov_b32 s19, scc // (op 3692) 
v_lshlrev_b32 v7, 2, v6 // (op 3692) 

s_cmp_eq_i32 s19, 0 // (op 3692) 
s_cselect_b32 s51, 0, -1 // (op 3692) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3692) 


s_and_b32 s19, s12, 31 // (op 3692) 
v_lshrrev_b32 v0, 2, v5 // (op 3692) 

v_and_b32 v5, 3, v0 // (op 3692) 

v_and_b32 v0, 48, v1 // (op 3807) 
v_and_b32 v7, 63, v1 // (op 3807) 
v_sub_u32 v8, v7, v0 // (op 3807) 

v_lshrrev_b32 v7, 1, v8 // (op 3807) 
v_add_lshl_u32 v9, v0, v7, 8 // (op 3807) 


v_and_b32 v0, 1, v8 // (op 3807) 

v_lshlrev_b32 v7, 3, v0 // (op 3807) 

v_lshrrev_b32 v0, 6, v1 // (op 3807) 
v_and_b32 v8, 1, v0 // (op 3807) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 3807) 


v_add_u32 v7, v9, v0 // (op 3807) 


v_lshrrev_b32 v0, 6, v1 // (op 3865) 
v_and_b32 v8, 1, v0 // (op 3865) 

v_and_b32 v0, 15, v1 // (op 3865) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 3865) 


v_and_b32 v0, 63, v1 // (op 3865) 
v_lshrrev_b32 v8, 4, v0 // (op 3865) 

v_lshl_add_u32 v0, v8, 8, v9 // (op 3865) 


v_lshrrev_b32 v8, 7, v1 // (op 3881) 
v_and_b32 v9, 63, v1 // (op 3881) 
v_and_b32 v10, 48, v1 // (op 3936) 
v_and_b32 v11, 63, v1 // (op 3936) 
v_sub_u32 v12, v11, v10 // (op 3936) 

v_lshrrev_b32 v11, 1, v12 // (op 3936) 
v_add_lshl_u32 v13, v10, v11, 8 // (op 3936) 


v_and_b32 v10, 1, v12 // (op 3936) 

v_lshlrev_b32 v11, 3, v10 // (op 3936) 

v_lshrrev_b32 v10, 6, v1 // (op 3936) 
v_and_b32 v12, 1, v10 // (op 3936) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 3936) 


v_add_u32 v11, v13, v10 // (op 3936) 


v_lshrrev_b32 v10, 6, v1 // (op 3965) 
v_and_b32 v12, 1, v10 // (op 3965) 

v_and_b32 v10, 15, v1 // (op 3965) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3965) 


v_and_b32 v10, 63, v1 // (op 3965) 
v_lshrrev_b32 v12, 4, v10 // (op 3965) 

v_lshl_add_u32 v10, v12, 8, v13 // (op 3965) 


s_lshl_b32 s52, 1, s19 // (op 3692) 
v_and_b32 v12, 3, v1 // (op 3692) 
s_add_i32 s53, s52, s51 // (op 3692) 


s_and_b32 s51, s0, s53 // (op 3692) 


s_add_i32 s0, s1, s16 // (op 3881) 

s_ashr_i32 s1, s0, 31 // (op 3881) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3692) 


s_add_i32 s52, s18, s51 // (op 3692) 


v_lshrrev_b32 v5, 6, v1 // (op 3692) 
s_ashr_i32 s18, s52, s19 // (op 3692) 


v_and_b32 v12, 1, v5 // (op 3692) 

s_lshl_b32 s19, s12, 24 // (op 3692) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3692) 


s_ashr_i32 s51, s19, 31 // (op 3692) 

s_xor_b32 s19, s18, s51 // (op 3692) 

s_sub_i32 s18, s19, s51 // (op 3692) 


s_mul_hi_i32 s19, s18, s13 // (op 3692) 
s_add_i32 s51, s19, s18 // (op 3692) 

s_ashr_i32 s19, s51, 31 // (op 3692) 
s_cmp_eq_i32 s13, 0 // (op 3692) 
s_mov_b32 s52, scc // (op 3692) 
s_cmp_eq_i32 s52, 0 // (op 3692) 
s_cselect_b32 s53, 0, -1 // (op 3692) 

s_and_b32 s52, s14, 31 // (op 3692) 
v_lshrrev_b32 v12, 4, v9 // (op 3881) 
s_cmp_eq_i32 s11, 0 // (op 3881) 
s_mov_b32 s54, scc // (op 3881) 
v_lshlrev_b32 v13, 2, v12 // (op 3881) 

s_cmp_eq_i32 s54, 0 // (op 3881) 
s_cselect_b32 s55, 0, -1 // (op 3881) 

s_lshl_b32 s54, 1, s52 // (op 3692) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 3881) 


v_lshrrev_b32 v8, 2, v9 // (op 3881) 

s_add_i32 s56, s54, s53 // (op 3692) 


s_and_b32 s53, s19, s56 // (op 3692) 


s_add_i32 s19, s51, s53 // (op 3692) 


s_ashr_i32 s51, s19, s52 // (op 3692) 


s_lshl_b32 s19, s14, 24 // (op 3692) 
s_ashr_i32 s52, s19, 31 // (op 3692) 

s_xor_b32 s19, s51, s52 // (op 3692) 

s_sub_i32 s51, s19, s52 // (op 3692) 


s_and_b32 s19, s12, 31 // (op 3881) 
s_lshl_b32 s52, 1, s19 // (op 3881) 
v_and_b32 v9, 3, v8 // (op 3881) 

s_add_i32 s53, s52, s55 // (op 3881) 


v_and_b32 v8, 3, v1 // (op 3881) 
s_and_b32 s52, s1, s53 // (op 3881) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 3881) 


s_add_i32 s1, s0, s52 // (op 3881) 


v_lshrrev_b32 v8, 6, v1 // (op 3881) 
s_ashr_i32 s0, s1, s19 // (op 3881) 


v_and_b32 v9, 1, v8 // (op 3881) 

s_lshl_b32 s1, s12, 24 // (op 3881) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 3881) 


s_ashr_i32 s19, s1, 31 // (op 3881) 

s_xor_b32 s1, s0, s19 // (op 3881) 

s_sub_i32 s0, s1, s19 // (op 3881) 


s_mul_hi_i32 s1, s0, s13 // (op 3881) 
s_add_i32 s19, s1, s0 // (op 3881) 

s_ashr_i32 s1, s19, 31 // (op 3881) 
s_cmp_eq_i32 s13, 0 // (op 3881) 
s_mov_b32 s52, scc // (op 3881) 
s_cmp_eq_i32 s52, 0 // (op 3881) 
s_cselect_b32 s53, 0, -1 // (op 3881) 

s_and_b32 s52, s14, 31 // (op 3881) 
s_lshl_b32 s54, 1, s52 // (op 3881) 
s_add_i32 s55, s54, s53 // (op 3881) 


s_and_b32 s53, s1, s55 // (op 3881) 


s_add_i32 s1, s19, s53 // (op 3881) 


s_ashr_i32 s19, s1, s52 // (op 3881) 


s_lshl_b32 s1, s14, 24 // (op 3881) 
s_ashr_i32 s52, s1, 31 // (op 3881) 

s_xor_b32 s1, s19, s52 // (op 3881) 

s_sub_i32 s19, s1, s52 // (op 3881) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 3692) 
s_sub_u32 s52, s16, s1 // (op 3692) 

s_mul_i32 s1, s51, s44 // (op 3692) 
s_add_u32 s1, s1, s52 // (op 3692) 

s_mul_hi_i32 s52, s16, s10 // (op 3692) 
s_add_i32 s53, s52, s16 // (op 3692) 

s_ashr_i32 s52, s53, 31 // (op 3692) 
s_cmp_eq_i32 s10, 0 // (op 3692) 
s_mov_b32 s54, scc // (op 3692) 
s_cmp_eq_i32 s54, 0 // (op 3692) 
s_cselect_b32 s55, 0, -1 // (op 3692) 

s_and_b32 s54, s47, 31 // (op 3692) 
s_lshl_b32 s56, 1, s54 // (op 3692) 
s_add_i32 s57, s56, s55 // (op 3692) 


s_and_b32 s55, s52, s57 // (op 3692) 


s_add_i32 s52, s53, s55 // (op 3692) 


s_ashr_i32 s53, s52, s54 // (op 3692) 


s_lshl_b32 s52, s47, 24 // (op 3692) 
s_ashr_i32 s54, s52, 31 // (op 3692) 

s_xor_b32 s52, s53, s54 // (op 3692) 

s_sub_i32 s53, s52, s54 // (op 3692) 


s_mul_i32 s52, s0, s44 // (op 3881) 
s_sub_u32 s54, s16, s52 // (op 3881) 

s_mul_i32 s52, s19, s44 // (op 3881) 
s_add_u32 s52, s52, s54 // (op 3881) 

s_mul_hi_i32 s54, s16, s10 // (op 3881) 
s_add_i32 s55, s54, s16 // (op 3881) 

s_ashr_i32 s54, s55, 31 // (op 3881) 
s_cmp_eq_i32 s10, 0 // (op 3881) 
s_mov_b32 s56, scc // (op 3881) 
s_cmp_eq_i32 s56, 0 // (op 3881) 
s_cselect_b32 s57, 0, -1 // (op 3881) 

s_and_b32 s56, s47, 31 // (op 3881) 
s_lshl_b32 s58, 1, s56 // (op 3881) 
s_add_i32 s59, s58, s57 // (op 3881) 


s_and_b32 s57, s54, s59 // (op 3881) 


s_add_i32 s54, s55, s57 // (op 3881) 


s_ashr_i32 s55, s54, s56 // (op 3881) 


s_lshl_b32 s54, s47, 24 // (op 3881) 
s_ashr_i32 s56, s54, 31 // (op 3881) 

s_xor_b32 s54, s55, s56 // (op 3881) 

s_sub_i32 s55, s54, s56 // (op 3881) 


s_add_u32 s56, s6, 256 // (op 3692) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3692) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3692) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3692) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3692) 

s_mul_i32 s54, s44, s56 // (op 3692) 
s_mul_i32 s58, s48, s54 // (op 3692) 

s_mul_i32 s54, s53, s58 // (op 3692) 

s_sub_u32 s58, s16, s54 // (op 3692) 

s_mov_b32 s54, s58 // (op 3692) 
s_mul_hi_i32 s59, s54, s15 // (op 3692) 
s_add_i32 s60, s59, s54 // (op 3692) 


s_ashr_i32 s54, s60, 31 // (op 3692) 
s_cmp_eq_i32 s15, 0 // (op 3692) 
s_mov_b32 s59, scc // (op 3692) 
s_cmp_eq_i32 s59, 0 // (op 3692) 
s_cselect_b32 s61, 0, -1 // (op 3692) 

s_and_b32 s59, s46, 31 // (op 3692) 
s_lshl_b32 s62, 1, s59 // (op 3692) 
s_add_i32 s63, s62, s61 // (op 3692) 


s_and_b32 s61, s54, s63 // (op 3692) 


s_add_i32 s54, s60, s61 // (op 3692) 


s_ashr_i32 s60, s54, s59 // (op 3692) 


s_lshl_b32 s54, s46, 24 // (op 3692) 
s_ashr_i32 s59, s54, 31 // (op 3692) 

s_xor_b32 s54, s60, s59 // (op 3692) 

s_sub_i32 s60, s54, s59 // (op 3692) 


s_mul_i32 s54, s45, s44 // (op 3692) 
s_add_u32 s62, s8, 16 // (op 3692) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 3692) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3692) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3692) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 3692) 

s_sub_i32 s59, s62, s54 // (op 3692) 

s_mul_i32 s61, s60, s59 // (op 3692) 

s_sub_u32 s59, s58, s61 // (op 3692) 


s_add_u32 s58, s59, s54 // (op 3692) 


s_cmp_eq_i32 s53, 0 // (op 3692) 
s_mov_b32 s54, scc // (op 3692) 

s_cmp_eq_i32 s54, 0 // (op 3692) 
s_cselect_b32 s53, s58, s1 // (op 3692) 


s_mul_i32 s1, s51, s56 // (op 3692) 


v_lshl_add_u32 v9, s53, 4, v6 // (op 3692) 


s_sub_i32 s51, s18, s1 // (op 3692) 


s_cmp_eq_i32 s54, 0 // (op 3692) 
s_cselect_b32 s1, s60, s51 // (op 3692) 



v_lshl_add_u32 v6, s1, 8, v5 // (op 3692) 


v_mul_lo_u32 v5, v6, s2 // (op 3692) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 3692) 


s_add_u32 s56, s6, 256 // (op 3881) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3881) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3881) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3881) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3881) 

s_mul_i32 s1, s44, s56 // (op 3881) 
s_mul_i32 s18, s48, s1 // (op 3881) 

s_mul_i32 s1, s55, s18 // (op 3881) 

s_sub_u32 s18, s16, s1 // (op 3881) 

s_mov_b32 s1, s18 // (op 3881) 
s_mul_hi_i32 s51, s1, s15 // (op 3881) 
s_add_i32 s53, s51, s1 // (op 3881) 


s_ashr_i32 s1, s53, 31 // (op 3881) 
s_cmp_eq_i32 s15, 0 // (op 3881) 
s_mov_b32 s51, scc // (op 3881) 
s_cmp_eq_i32 s51, 0 // (op 3881) 
s_cselect_b32 s54, 0, -1 // (op 3881) 

s_and_b32 s51, s46, 31 // (op 3881) 
s_lshl_b32 s58, 1, s51 // (op 3881) 
s_add_i32 s59, s58, s54 // (op 3881) 


s_and_b32 s54, s1, s59 // (op 3881) 


s_add_i32 s1, s53, s54 // (op 3881) 


s_ashr_i32 s53, s1, s51 // (op 3881) 


s_lshl_b32 s1, s46, 24 // (op 3881) 
s_ashr_i32 s51, s1, 31 // (op 3881) 

s_xor_b32 s1, s53, s51 // (op 3881) 

s_sub_i32 s53, s1, s51 // (op 3881) 


s_mul_i32 s1, s45, s44 // (op 3881) 
s_add_u32 s58, s8, 16 // (op 3881) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 3881) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3881) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3881) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 3881) 

s_sub_i32 s51, s58, s1 // (op 3881) 

s_mul_i32 s54, s53, s51 // (op 3881) 

s_sub_u32 s51, s18, s54 // (op 3881) 


s_add_u32 s18, s51, s1 // (op 3881) 


s_cmp_eq_i32 s55, 0 // (op 3881) 
s_mov_b32 s1, scc // (op 3881) 

s_cmp_eq_i32 s1, 0 // (op 3881) 
s_cselect_b32 s51, s18, s52 // (op 3881) 


s_mul_i32 s18, s19, s56 // (op 3881) 


v_lshl_add_u32 v5, s51, 4, v12 // (op 3881) 


s_sub_i32 s19, s0, s18 // (op 3881) 


s_cmp_eq_i32 s1, 0 // (op 3881) 
s_cselect_b32 s0, s53, s19 // (op 3881) 



v_lshl_add_u32 v9, s0, 8, v8 // (op 3881) 


v_mul_lo_u32 v8, v9, s4 // (op 3881) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 3881) 


s_mul_i32 s1, s2, s7 // (op 3694) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 3694) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 3694) least significant
s_mul_hi_u32 s19, s2, s6 // (op 3694) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3694) most significant: sum
s_add_u32 s1, s1, s19 // (op 3694) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3694) 

s_mul_i32 s1, s4, s7 // (op 3883) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 3883) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 3883) least significant
s_mul_hi_u32 s52, s4, s6 // (op 3883) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3883) most significant: sum
s_add_u32 s1, s1, s52 // (op 3883) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 3883) 

s_mov_b32 s56, s20 // (op 3694) reg expression
s_mov_b32 s57, s21 // (op 3694) reg expression
s_mov_b32 s58, s18 // (op 3694) reg expression
s_mov_b32 s59, 131072 // (op 3694) call()

s_mov_b32 s60, s22 // (op 3883) reg expression

s_mov_b32 s61, s23 // (op 3883) reg expression
s_mov_b32 s62, s52 // (op 3883) reg expression
s_mov_b32 s63, 131072 // (op 3883) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
v_accvgpr_write a16, 0.00000 // (op 778) call()
v_accvgpr_write a17, 0.00000 // (op 778) call()
v_accvgpr_write a18, 0.00000 // (op 778) call()
v_accvgpr_write a19, 0.00000 // (op 778) call()
v_accvgpr_write a20, 0.00000 // (op 934) call()
v_accvgpr_write a21, 0.00000 // (op 934) call()
v_accvgpr_write a22, 0.00000 // (op 934) call()
v_accvgpr_write a23, 0.00000 // (op 934) call()
v_accvgpr_write a24, 0.00000 // (op 1090) call()
v_accvgpr_write a25, 0.00000 // (op 1090) call()
v_accvgpr_write a26, 0.00000 // (op 1090) call()
v_accvgpr_write a27, 0.00000 // (op 1090) call()
v_accvgpr_write a28, 0.00000 // (op 1246) call()
v_accvgpr_write a29, 0.00000 // (op 1246) call()
v_accvgpr_write a30, 0.00000 // (op 1246) call()
v_accvgpr_write a31, 0.00000 // (op 1246) call()
s_mul_hi_i32 s0, s16, s11 // (op 3459) 
v_lshrrev_b32 v12, 3, v1 // (op 3459) 
s_add_i32 s1, s0, s16 // (op 3459) 

v_and_b32 v13, -8, v1 // (op 3459) 
s_ashr_i32 s0, s1, 31 // (op 3459) 
v_sub_u32 v14, v1, v13 // (op 3459) 

s_cmp_eq_i32 s11, 0 // (op 3459) 
s_mov_b32 s18, scc // (op 3459) 
v_lshlrev_b32 v13, 4, v14 // (op 3459) 

s_cmp_eq_i32 s18, 0 // (op 3459) 
s_cselect_b32 s19, 0, -1 // (op 3459) 

v_mov_b32 v15, 0 // (op 3459) convert
v_mov_b32 v14, v13 // (op 3459) convert

s_and_b32 s18, s12, 31 // (op 3459) 
s_lshl_b32 s20, 1, s18 // (op 3459) 
s_add_i32 s21, s20, s19 // (op 3459) 


s_and_b32 s19, s0, s21 // (op 3459) 


s_add_i32 s0, s1, s19 // (op 3459) 


s_ashr_i32 s1, s0, s18 // (op 3459) 


s_lshl_b32 s0, s12, 24 // (op 3459) 
s_ashr_i32 s18, s0, 31 // (op 3459) 

s_xor_b32 s0, s1, s18 // (op 3459) 

s_sub_i32 s1, s0, s18 // (op 3459) 


s_mul_hi_i32 s0, s1, s13 // (op 3459) 
s_add_i32 s18, s0, s1 // (op 3459) 

s_ashr_i32 s0, s18, 31 // (op 3459) 
s_cmp_eq_i32 s13, 0 // (op 3459) 
s_mov_b32 s19, scc // (op 3459) 
s_cmp_eq_i32 s19, 0 // (op 3459) 
s_cselect_b32 s20, 0, -1 // (op 3459) 

s_and_b32 s19, s14, 31 // (op 3459) 
s_lshl_b32 s21, 1, s19 // (op 3459) 
s_add_i32 s22, s21, s20 // (op 3459) 


s_and_b32 s23, s0, s22 // (op 3459) 


s_add_i32 s0, s18, s23 // (op 3459) 


s_ashr_i32 s18, s0, s19 // (op 3459) 


s_lshl_b32 s0, s14, 24 // (op 3459) 
s_ashr_i32 s19, s0, 31 // (op 3459) 

s_xor_b32 s0, s18, s19 // (op 3459) 

s_sub_i32 s18, s0, s19 // (op 3459) 


s_mul_i32 s0, s1, s44 // (op 3459) 

s_sub_u32 s1, s16, s0 // (op 3459) 

s_mul_i32 s0, s18, s44 // (op 3459) 
s_add_u32 s0, s0, s1 // (op 3459) 


s_mul_hi_i32 s1, s16, s10 // (op 3459) 
s_add_i32 s18, s1, s16 // (op 3459) 

s_ashr_i32 s1, s18, 31 // (op 3459) 
s_cmp_eq_i32 s10, 0 // (op 3459) 
s_mov_b32 s19, scc // (op 3459) 
s_cmp_eq_i32 s19, 0 // (op 3459) 
s_cselect_b32 s20, 0, -1 // (op 3459) 

s_and_b32 s19, s47, 31 // (op 3459) 
s_lshl_b32 s21, 1, s19 // (op 3459) 
s_add_i32 s22, s21, s20 // (op 3459) 


s_and_b32 s23, s1, s22 // (op 3459) 


s_add_i32 s1, s18, s23 // (op 3459) 


s_ashr_i32 s18, s1, s19 // (op 3459) 


s_lshl_b32 s1, s47, 24 // (op 3459) 
s_ashr_i32 s19, s1, 31 // (op 3459) 

s_xor_b32 s1, s18, s19 // (op 3459) 

s_sub_i32 s18, s1, s19 // (op 3459) 


s_add_u32 s22, s6, 256 // (op 3459) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 3459) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 3459) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 3459) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 3459) 

s_mul_i32 s1, s44, s22 // (op 3459) 

s_mul_i32 s19, s48, s1 // (op 3459) 

s_mul_i32 s1, s18, s19 // (op 3459) 

s_sub_u32 s19, s16, s1 // (op 3459) 

s_mov_b32 s1, s19 // (op 3459) 
s_mul_hi_i32 s20, s1, s15 // (op 3459) 
s_add_i32 s21, s20, s1 // (op 3459) 


s_ashr_i32 s1, s21, 31 // (op 3459) 
s_cmp_eq_i32 s15, 0 // (op 3459) 
s_mov_b32 s20, scc // (op 3459) 
s_cmp_eq_i32 s20, 0 // (op 3459) 
s_cselect_b32 s22, 0, -1 // (op 3459) 

s_and_b32 s20, s46, 31 // (op 3459) 
s_lshl_b32 s23, 1, s20 // (op 3459) 
s_add_i32 s49, s23, s22 // (op 3459) 


s_and_b32 s22, s1, s49 // (op 3459) 


s_add_i32 s1, s21, s22 // (op 3459) 


s_ashr_i32 s21, s1, s20 // (op 3459) 


s_lshl_b32 s1, s46, 24 // (op 3459) 
s_ashr_i32 s20, s1, 31 // (op 3459) 

s_xor_b32 s1, s21, s20 // (op 3459) 

s_sub_i32 s21, s1, s20 // (op 3459) 


s_mul_i32 s1, s45, s44 // (op 3459) 
s_add_u32 s22, s8, 16 // (op 3459) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 3459) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 3459) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 3459) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 3459) 

s_sub_i32 s20, s22, s1 // (op 3459) 

s_mul_i32 s22, s21, s20 // (op 3459) 


s_sub_u32 s23, s19, s22 // (op 3459) 


s_add_u32 s19, s23, s1 // (op 3459) 


s_cmp_eq_i32 s18, 0 // (op 3459) 
s_mov_b32 s1, scc // (op 3459) 

s_cmp_eq_i32 s1, 0 // (op 3459) 
s_cselect_b32 s18, s19, s0 // (op 3459) 



v_lshl_add_u32 v13, s18, 4, v12 // (op 3459) 


v_mul_lo_u32 v17, v13, s25 // (op 3459) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3459) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 3459) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3459) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3459) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3459) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3459) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3459) 

v_mov_b32 v12, v14 // (op 3459) convert

s_mul_i32 s1, s24, s9 // (op 3461) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3461) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3461) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3461) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3461) most significant: sum
s_add_u32 s1, s1, s19 // (op 3461) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3461) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3461) 

s_mov_b32 s20, s26 // (op 3461) reg expression
s_mov_b32 s21, s27 // (op 3461) reg expression
s_mov_b32 s22, s0 // (op 3461) reg expression
s_mov_b32 s23, 131072 // (op 3461) call()

s_mul_hi_i32 s0, s16, s11 // (op 3478) 
v_lshrrev_b32 v13, 3, v1 // (op 3478) 
s_add_i32 s1, s0, s16 // (op 3478) 

v_and_b32 v14, -8, v1 // (op 3478) 
s_ashr_i32 s0, s1, 31 // (op 3478) 
v_sub_u32 v15, v1, v14 // (op 3478) 

s_cmp_eq_i32 s11, 0 // (op 3478) 
s_mov_b32 s18, scc // (op 3478) 
v_lshlrev_b32 v14, 4, v15 // (op 3478) 

s_cmp_eq_i32 s18, 0 // (op 3478) 
s_cselect_b32 s19, 0, -1 // (op 3478) 

v_mov_b32 v17, 0 // (op 3478) convert
v_mov_b32 v16, v14 // (op 3478) convert

s_and_b32 s18, s12, 31 // (op 3478) 
s_lshl_b32 s49, 1, s18 // (op 3478) 
s_add_i32 s50, s49, s19 // (op 3478) 


s_and_b32 s19, s0, s50 // (op 3478) 


s_add_i32 s0, s1, s19 // (op 3478) 


s_ashr_i32 s1, s0, s18 // (op 3478) 


s_lshl_b32 s0, s12, 24 // (op 3478) 
s_ashr_i32 s18, s0, 31 // (op 3478) 

s_xor_b32 s0, s1, s18 // (op 3478) 

s_sub_i32 s1, s0, s18 // (op 3478) 


s_mul_hi_i32 s0, s1, s13 // (op 3478) 
s_add_i32 s18, s0, s1 // (op 3478) 

s_ashr_i32 s0, s18, 31 // (op 3478) 
s_cmp_eq_i32 s13, 0 // (op 3478) 
s_mov_b32 s19, scc // (op 3478) 
s_cmp_eq_i32 s19, 0 // (op 3478) 
s_cselect_b32 s49, 0, -1 // (op 3478) 

s_and_b32 s19, s14, 31 // (op 3478) 
s_lshl_b32 s50, 1, s19 // (op 3478) 
s_add_i32 s51, s50, s49 // (op 3478) 


s_and_b32 s49, s0, s51 // (op 3478) 


s_add_i32 s0, s18, s49 // (op 3478) 


s_ashr_i32 s18, s0, s19 // (op 3478) 


s_lshl_b32 s0, s14, 24 // (op 3478) 
s_ashr_i32 s19, s0, 31 // (op 3478) 

s_xor_b32 s0, s18, s19 // (op 3478) 

s_sub_i32 s18, s0, s19 // (op 3478) 


s_mul_i32 s0, s1, s44 // (op 3478) 

s_sub_u32 s1, s16, s0 // (op 3478) 

s_mul_i32 s0, s18, s44 // (op 3478) 
s_add_u32 s0, s0, s1 // (op 3478) 


s_mul_hi_i32 s1, s16, s10 // (op 3478) 
s_add_i32 s18, s1, s16 // (op 3478) 

s_ashr_i32 s1, s18, 31 // (op 3478) 
s_cmp_eq_i32 s10, 0 // (op 3478) 
s_mov_b32 s19, scc // (op 3478) 
s_cmp_eq_i32 s19, 0 // (op 3478) 
s_cselect_b32 s49, 0, -1 // (op 3478) 

s_and_b32 s19, s47, 31 // (op 3478) 
s_lshl_b32 s50, 1, s19 // (op 3478) 
s_add_i32 s51, s50, s49 // (op 3478) 


s_and_b32 s49, s1, s51 // (op 3478) 


s_add_i32 s1, s18, s49 // (op 3478) 


s_ashr_i32 s18, s1, s19 // (op 3478) 


s_lshl_b32 s1, s47, 24 // (op 3478) 
s_ashr_i32 s19, s1, 31 // (op 3478) 

s_xor_b32 s1, s18, s19 // (op 3478) 

s_sub_i32 s18, s1, s19 // (op 3478) 


s_add_u32 s54, s6, 256 // (op 3478) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3478) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3478) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3478) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3478) 

s_mul_i32 s1, s44, s50 // (op 3478) 

s_mul_i32 s19, s48, s1 // (op 3478) 

s_mul_i32 s1, s18, s19 // (op 3478) 

s_sub_u32 s19, s16, s1 // (op 3478) 

s_mov_b32 s1, s19 // (op 3478) 
s_mul_hi_i32 s49, s1, s15 // (op 3478) 
s_add_i32 s50, s49, s1 // (op 3478) 


s_ashr_i32 s1, s50, 31 // (op 3478) 
s_cmp_eq_i32 s15, 0 // (op 3478) 
s_mov_b32 s49, scc // (op 3478) 
s_cmp_eq_i32 s49, 0 // (op 3478) 
s_cselect_b32 s51, 0, -1 // (op 3478) 

s_and_b32 s49, s46, 31 // (op 3478) 
s_lshl_b32 s52, 1, s49 // (op 3478) 
s_add_i32 s53, s52, s51 // (op 3478) 


s_and_b32 s51, s1, s53 // (op 3478) 


s_add_i32 s1, s50, s51 // (op 3478) 


s_ashr_i32 s50, s1, s49 // (op 3478) 


s_lshl_b32 s1, s46, 24 // (op 3478) 
s_ashr_i32 s49, s1, 31 // (op 3478) 

s_xor_b32 s1, s50, s49 // (op 3478) 

s_sub_i32 s50, s1, s49 // (op 3478) 


s_mul_i32 s1, s45, s44 // (op 3478) 
s_add_u32 s54, s8, 16 // (op 3478) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 3478) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3478) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3478) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 3478) 

s_sub_i32 s49, s54, s1 // (op 3478) 

s_mul_i32 s51, s50, s49 // (op 3478) 


s_sub_u32 s49, s19, s51 // (op 3478) 


s_add_u32 s19, s49, s1 // (op 3478) 


s_cmp_eq_i32 s18, 0 // (op 3478) 
s_mov_b32 s1, scc // (op 3478) 

s_cmp_eq_i32 s1, 0 // (op 3478) 
s_cselect_b32 s18, s19, s0 // (op 3478) 



v_lshl_add_u32 v14, s18, 4, v13 // (op 3478) 


v_mul_lo_u32 v19, v14, s25 // (op 3478) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 3478) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 3478) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3478) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3478) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3478) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 3478) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3478) 

v_mov_b32 v13, v14 // (op 3478) convert

s_mul_i32 s1, s24, s9 // (op 3480) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3480) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3480) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3480) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3480) most significant: sum
s_add_u32 s1, s1, s19 // (op 3480) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3480) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3480) 

s_mov_b32 s20, s26 // (op 3480) reg expression
s_mov_b32 s21, s27 // (op 3480) reg expression
s_mov_b32 s22, s0 // (op 3480) reg expression
s_mov_b32 s23, 131072 // (op 3480) call()

s_mul_hi_i32 s0, s16, s11 // (op 3523) 
v_and_b32 v14, -16, v1 // (op 3523) 
s_add_i32 s1, s0, s16 // (op 3523) 

v_sub_u32 v15, v1, v14 // (op 3523) 

s_ashr_i32 s0, s1, 31 // (op 3523) 
v_lshlrev_b32 v14, 4, v15 // (op 3523) 

s_cmp_eq_i32 s11, 0 // (op 3523) 
s_mov_b32 s18, scc // (op 3523) 
v_and_b32 v15, 4294967280, v1 // (op 3523) 
s_cmp_eq_i32 s18, 0 // (op 3523) 
s_cselect_b32 s19, 0, -1 // (op 3523) 

s_and_b32 s18, s12, 31 // (op 3523) 
s_lshl_b32 s49, 1, s18 // (op 3523) 
s_add_i32 s50, s49, s19 // (op 3523) 


s_and_b32 s19, s0, s50 // (op 3523) 


s_add_i32 s0, s1, s19 // (op 3523) 


s_ashr_i32 s1, s0, s18 // (op 3523) 


s_lshl_b32 s0, s12, 24 // (op 3523) 
s_ashr_i32 s18, s0, 31 // (op 3523) 

s_xor_b32 s0, s1, s18 // (op 3523) 

s_sub_i32 s1, s0, s18 // (op 3523) 


s_mul_hi_i32 s0, s1, s13 // (op 3523) 
s_add_i32 s18, s0, s1 // (op 3523) 

s_ashr_i32 s0, s18, 31 // (op 3523) 
s_cmp_eq_i32 s13, 0 // (op 3523) 
s_mov_b32 s19, scc // (op 3523) 
s_cmp_eq_i32 s19, 0 // (op 3523) 
s_cselect_b32 s49, 0, -1 // (op 3523) 

s_and_b32 s19, s14, 31 // (op 3523) 
s_lshl_b32 s50, 1, s19 // (op 3523) 
s_add_i32 s51, s50, s49 // (op 3523) 


s_and_b32 s49, s0, s51 // (op 3523) 


s_add_i32 s0, s18, s49 // (op 3523) 


s_ashr_i32 s18, s0, s19 // (op 3523) 


s_lshl_b32 s0, s14, 24 // (op 3523) 
s_ashr_i32 s19, s0, 31 // (op 3523) 

s_xor_b32 s0, s18, s19 // (op 3523) 

s_sub_i32 s18, s0, s19 // (op 3523) 


s_add_u32 s54, s6, 256 // (op 3523) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3523) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3523) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3523) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3523) 

s_mul_i32 s0, s18, s50 // (op 3523) 

s_sub_i32 s49, s1, s0 // (op 3523) 


s_mul_hi_i32 s0, s16, s10 // (op 3523) 
s_add_i32 s1, s0, s16 // (op 3523) 

s_ashr_i32 s0, s1, 31 // (op 3523) 
s_cmp_eq_i32 s10, 0 // (op 3523) 
s_mov_b32 s18, scc // (op 3523) 
s_cmp_eq_i32 s18, 0 // (op 3523) 
s_cselect_b32 s19, 0, -1 // (op 3523) 

s_and_b32 s18, s47, 31 // (op 3523) 
s_lshl_b32 s52, 1, s18 // (op 3523) 
s_add_i32 s53, s52, s19 // (op 3523) 


s_and_b32 s19, s0, s53 // (op 3523) 


s_add_i32 s0, s1, s19 // (op 3523) 


s_ashr_i32 s1, s0, s18 // (op 3523) 


s_lshl_b32 s0, s47, 24 // (op 3523) 
s_ashr_i32 s18, s0, 31 // (op 3523) 

s_xor_b32 s0, s1, s18 // (op 3523) 

s_sub_i32 s1, s0, s18 // (op 3523) 


s_mul_i32 s0, s44, s50 // (op 3523) 

s_mul_i32 s18, s48, s0 // (op 3523) 

s_mul_i32 s0, s1, s18 // (op 3523) 

s_sub_u32 s18, s16, s0 // (op 3523) 

s_mul_hi_i32 s0, s18, s15 // (op 3523) 
s_add_i32 s19, s0, s18 // (op 3523) 


s_ashr_i32 s0, s19, 31 // (op 3523) 
s_cmp_eq_i32 s15, 0 // (op 3523) 
s_mov_b32 s18, scc // (op 3523) 
s_cmp_eq_i32 s18, 0 // (op 3523) 
s_cselect_b32 s50, 0, -1 // (op 3523) 

s_and_b32 s18, s46, 31 // (op 3523) 
s_lshl_b32 s51, 1, s18 // (op 3523) 
s_add_i32 s52, s51, s50 // (op 3523) 


s_and_b32 s50, s0, s52 // (op 3523) 


s_add_i32 s0, s19, s50 // (op 3523) 


s_ashr_i32 s19, s0, s18 // (op 3523) 


s_lshl_b32 s0, s46, 24 // (op 3523) 
s_ashr_i32 s18, s0, 31 // (op 3523) 

s_xor_b32 s0, s19, s18 // (op 3523) 

s_sub_i32 s19, s0, s18 // (op 3523) 


s_cmp_eq_i32 s1, 0 // (op 3523) 
s_mov_b32 s0, scc // (op 3523) 

s_cmp_eq_i32 s0, 0 // (op 3523) 
s_cselect_b32 s1, s19, s49 // (op 3523) 



v_lshl_add_u32 v16, s1, 8, v14 // (op 3523) 


v_mul_lo_u32 v14, v15, s28 // (op 3523) 
v_add_u32 v14, v14, v16 // (op 3523) 


s_mul_i32 s1, s28, s33 // (op 3525) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3525) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3525) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3525) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3525) most significant: sum
s_add_u32 s1, s1, s19 // (op 3525) most significant: sum


s_mov_b32 s52, s30 // (op 3525) reg expression
s_mov_b32 s53, s31 // (op 3525) reg expression
s_mov_b32 s54, s0 // (op 3525) reg expression
s_mov_b32 s55, 131072 // (op 3525) call()

s_mul_hi_i32 s0, s16, s11 // (op 3542) 
v_and_b32 v15, -16, v1 // (op 3542) 
s_add_i32 s1, s0, s16 // (op 3542) 

v_sub_u32 v16, v1, v15 // (op 3542) 

s_ashr_i32 s0, s1, 31 // (op 3542) 
v_lshlrev_b32 v15, 4, v16 // (op 3542) 

s_cmp_eq_i32 s11, 0 // (op 3542) 
s_mov_b32 s18, scc // (op 3542) 
v_and_b32 v16, 4294967280, v1 // (op 3542) 
s_cmp_eq_i32 s18, 0 // (op 3542) 
s_cselect_b32 s19, 0, -1 // (op 3542) 

s_and_b32 s18, s12, 31 // (op 3542) 
s_lshl_b32 s49, 1, s18 // (op 3542) 
s_add_i32 s50, s49, s19 // (op 3542) 


s_and_b32 s19, s0, s50 // (op 3542) 


s_add_i32 s0, s1, s19 // (op 3542) 


s_ashr_i32 s1, s0, s18 // (op 3542) 


s_lshl_b32 s0, s12, 24 // (op 3542) 
s_ashr_i32 s18, s0, 31 // (op 3542) 

s_xor_b32 s0, s1, s18 // (op 3542) 

s_sub_i32 s1, s0, s18 // (op 3542) 


s_mul_hi_i32 s0, s1, s13 // (op 3542) 
s_add_i32 s18, s0, s1 // (op 3542) 

s_ashr_i32 s0, s18, 31 // (op 3542) 
s_cmp_eq_i32 s13, 0 // (op 3542) 
s_mov_b32 s19, scc // (op 3542) 
s_cmp_eq_i32 s19, 0 // (op 3542) 
s_cselect_b32 s49, 0, -1 // (op 3542) 

s_and_b32 s19, s14, 31 // (op 3542) 
s_lshl_b32 s50, 1, s19 // (op 3542) 
s_add_i32 s51, s50, s49 // (op 3542) 


s_and_b32 s49, s0, s51 // (op 3542) 


s_add_i32 s0, s18, s49 // (op 3542) 


s_ashr_i32 s18, s0, s19 // (op 3542) 


s_lshl_b32 s0, s14, 24 // (op 3542) 
s_ashr_i32 s19, s0, 31 // (op 3542) 

s_xor_b32 s0, s18, s19 // (op 3542) 

s_sub_i32 s18, s0, s19 // (op 3542) 


s_add_u32 s50, s6, 256 // (op 3542) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3542) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3542) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3542) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3542) 

s_mul_i32 s0, s18, s50 // (op 3542) 

s_sub_i32 s49, s1, s0 // (op 3542) 


s_mul_hi_i32 s0, s16, s10 // (op 3542) 
s_add_i32 s1, s0, s16 // (op 3542) 

s_ashr_i32 s0, s1, 31 // (op 3542) 
s_cmp_eq_i32 s10, 0 // (op 3542) 
s_mov_b32 s18, scc // (op 3542) 
s_cmp_eq_i32 s18, 0 // (op 3542) 
s_cselect_b32 s19, 0, -1 // (op 3542) 

s_and_b32 s18, s47, 31 // (op 3542) 
s_lshl_b32 s64, 1, s18 // (op 3542) 
s_add_i32 s65, s64, s19 // (op 3542) 


s_and_b32 s19, s0, s65 // (op 3542) 


s_add_i32 s0, s1, s19 // (op 3542) 


s_ashr_i32 s1, s0, s18 // (op 3542) 


s_lshl_b32 s0, s47, 24 // (op 3542) 
s_ashr_i32 s18, s0, 31 // (op 3542) 

s_xor_b32 s0, s1, s18 // (op 3542) 

s_sub_i32 s1, s0, s18 // (op 3542) 


s_mul_i32 s0, s44, s50 // (op 3542) 

s_mul_i32 s18, s48, s0 // (op 3542) 

s_mul_i32 s0, s1, s18 // (op 3542) 

s_sub_u32 s18, s16, s0 // (op 3542) 

s_mul_hi_i32 s0, s18, s15 // (op 3542) 
s_add_i32 s19, s0, s18 // (op 3542) 


s_ashr_i32 s0, s19, 31 // (op 3542) 
s_cmp_eq_i32 s15, 0 // (op 3542) 
s_mov_b32 s18, scc // (op 3542) 
s_cmp_eq_i32 s18, 0 // (op 3542) 
s_cselect_b32 s50, 0, -1 // (op 3542) 

s_and_b32 s18, s46, 31 // (op 3542) 
s_lshl_b32 s51, 1, s18 // (op 3542) 
s_add_i32 s64, s51, s50 // (op 3542) 


s_and_b32 s50, s0, s64 // (op 3542) 


s_add_i32 s0, s19, s50 // (op 3542) 


s_ashr_i32 s19, s0, s18 // (op 3542) 


s_lshl_b32 s0, s46, 24 // (op 3542) 
s_ashr_i32 s18, s0, 31 // (op 3542) 

s_xor_b32 s0, s19, s18 // (op 3542) 

s_sub_i32 s19, s0, s18 // (op 3542) 


s_cmp_eq_i32 s1, 0 // (op 3542) 
s_mov_b32 s0, scc // (op 3542) 

s_cmp_eq_i32 s0, 0 // (op 3542) 
s_cselect_b32 s1, s19, s49 // (op 3542) 



v_lshl_add_u32 v17, s1, 8, v15 // (op 3542) 


v_mul_lo_u32 v15, v16, s28 // (op 3542) 
v_add_u32 v15, v15, v17 // (op 3542) 


s_mul_i32 s1, s28, s33 // (op 3544) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3544) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3544) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3544) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3544) most significant: sum
s_add_u32 s1, s1, s19 // (op 3544) most significant: sum


s_mov_b32 s52, s30 // (op 3544) reg expression
s_mov_b32 s53, s31 // (op 3544) reg expression
s_mov_b32 s54, s0 // (op 3544) reg expression
s_mov_b32 s55, 131072 // (op 3544) call()

s_mul_hi_i32 s0, s16, s11 // (op 3587) 
v_lshrrev_b32 v16, 7, v1 // (op 3587) 
s_add_i32 s1, s0, s16 // (op 3587) 

v_and_b32 v17, 15, v1 // (op 3587) 
s_ashr_i32 s0, s1, 31 // (op 3587) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 3587) 


s_cmp_eq_i32 s11, 0 // (op 3587) 
s_mov_b32 s18, scc // (op 3587) 
v_and_b32 v16, 63, v1 // (op 3587) 
s_cmp_eq_i32 s18, 0 // (op 3587) 
s_cselect_b32 s19, 0, -1 // (op 3587) 

v_lshrrev_b32 v17, 4, v16 // (op 3587) 

s_and_b32 s18, s12, 31 // (op 3587) 
s_lshl_b32 s49, 1, s18 // (op 3587) 
s_add_i32 s50, s49, s19 // (op 3587) 


s_and_b32 s19, s0, s50 // (op 3587) 


s_add_i32 s0, s1, s19 // (op 3587) 


s_ashr_i32 s1, s0, s18 // (op 3587) 


s_lshl_b32 s0, s12, 24 // (op 3587) 
s_ashr_i32 s18, s0, 31 // (op 3587) 

s_xor_b32 s0, s1, s18 // (op 3587) 

s_sub_i32 s1, s0, s18 // (op 3587) 


s_mul_hi_i32 s0, s1, s13 // (op 3587) 
s_add_i32 s18, s0, s1 // (op 3587) 

s_ashr_i32 s0, s18, 31 // (op 3587) 
s_cmp_eq_i32 s13, 0 // (op 3587) 
s_mov_b32 s19, scc // (op 3587) 
s_cmp_eq_i32 s19, 0 // (op 3587) 
s_cselect_b32 s49, 0, -1 // (op 3587) 

s_and_b32 s19, s14, 31 // (op 3587) 
s_lshl_b32 s50, 1, s19 // (op 3587) 
s_add_i32 s51, s50, s49 // (op 3587) 


s_and_b32 s49, s0, s51 // (op 3587) 


s_add_i32 s0, s18, s49 // (op 3587) 


s_ashr_i32 s18, s0, s19 // (op 3587) 


s_lshl_b32 s0, s14, 24 // (op 3587) 
s_ashr_i32 s19, s0, 31 // (op 3587) 

s_xor_b32 s0, s18, s19 // (op 3587) 

s_sub_i32 s18, s0, s19 // (op 3587) 


s_mul_i32 s0, s1, s44 // (op 3587) 

s_sub_u32 s1, s16, s0 // (op 3587) 

s_mul_i32 s0, s18, s44 // (op 3587) 
s_add_u32 s0, s0, s1 // (op 3587) 


s_mul_hi_i32 s1, s16, s10 // (op 3587) 
s_add_i32 s18, s1, s16 // (op 3587) 

s_ashr_i32 s1, s18, 31 // (op 3587) 
s_cmp_eq_i32 s10, 0 // (op 3587) 
s_mov_b32 s19, scc // (op 3587) 
s_cmp_eq_i32 s19, 0 // (op 3587) 
s_cselect_b32 s49, 0, -1 // (op 3587) 

s_and_b32 s19, s47, 31 // (op 3587) 
s_lshl_b32 s50, 1, s19 // (op 3587) 
s_add_i32 s51, s50, s49 // (op 3587) 


s_and_b32 s49, s1, s51 // (op 3587) 


s_add_i32 s1, s18, s49 // (op 3587) 


s_ashr_i32 s18, s1, s19 // (op 3587) 


s_lshl_b32 s1, s47, 24 // (op 3587) 
s_ashr_i32 s19, s1, 31 // (op 3587) 

s_xor_b32 s1, s18, s19 // (op 3587) 

s_sub_i32 s18, s1, s19 // (op 3587) 


s_add_u32 s50, s6, 256 // (op 3587) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3587) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3587) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3587) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3587) 

s_mul_i32 s1, s44, s50 // (op 3587) 

s_mul_i32 s19, s48, s1 // (op 3587) 

s_mul_i32 s1, s18, s19 // (op 3587) 

s_sub_u32 s19, s16, s1 // (op 3587) 

s_mov_b32 s1, s19 // (op 3587) 
s_mul_hi_i32 s49, s1, s15 // (op 3587) 
s_add_i32 s50, s49, s1 // (op 3587) 


s_ashr_i32 s1, s50, 31 // (op 3587) 
s_cmp_eq_i32 s15, 0 // (op 3587) 
s_mov_b32 s49, scc // (op 3587) 
s_cmp_eq_i32 s49, 0 // (op 3587) 
s_cselect_b32 s51, 0, -1 // (op 3587) 

s_and_b32 s49, s46, 31 // (op 3587) 
s_lshl_b32 s64, 1, s49 // (op 3587) 
s_add_i32 s65, s64, s51 // (op 3587) 


s_and_b32 s51, s1, s65 // (op 3587) 


s_add_i32 s1, s50, s51 // (op 3587) 


s_ashr_i32 s50, s1, s49 // (op 3587) 


s_lshl_b32 s1, s46, 24 // (op 3587) 
s_ashr_i32 s49, s1, 31 // (op 3587) 

s_xor_b32 s1, s50, s49 // (op 3587) 

s_sub_i32 s50, s1, s49 // (op 3587) 


s_mul_i32 s1, s45, s44 // (op 3587) 
s_add_u32 s64, s8, 16 // (op 3587) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 3587) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3587) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3587) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 3587) 

s_sub_i32 s49, s64, s1 // (op 3587) 

s_mul_i32 s51, s50, s49 // (op 3587) 


s_sub_u32 s49, s19, s51 // (op 3587) 


s_add_u32 s19, s49, s1 // (op 3587) 


s_cmp_eq_i32 s18, 0 // (op 3587) 
s_mov_b32 s1, scc // (op 3587) 

s_cmp_eq_i32 s1, 0 // (op 3587) 
s_cselect_b32 s18, s19, s0 // (op 3587) 



v_lshl_add_u32 v16, s18, 4, v18 // (op 3587) 


v_mul_lo_u32 v18, v16, s34 // (op 3587) 
v_add_u32 v18, v18, v17 // (op 3587) 


s_mul_i32 s1, s34, s9 // (op 3589) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3589) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3589) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3589) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3589) most significant: sum
s_add_u32 s1, s1, s19 // (op 3589) most significant: sum


s_mov_b32 s64, s36 // (op 3589) reg expression
s_mov_b32 s65, s37 // (op 3589) reg expression
s_mov_b32 s66, s0 // (op 3589) reg expression
s_mov_b32 s67, 131072 // (op 3589) call()

s_mul_hi_i32 s0, s16, s11 // (op 3606) 
v_lshrrev_b32 v16, 7, v1 // (op 3606) 
s_add_i32 s1, s0, s16 // (op 3606) 

v_and_b32 v17, 15, v1 // (op 3606) 
s_ashr_i32 s0, s1, 31 // (op 3606) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 3606) 


s_cmp_eq_i32 s11, 0 // (op 3606) 
s_mov_b32 s18, scc // (op 3606) 
v_and_b32 v16, 63, v1 // (op 3606) 
s_cmp_eq_i32 s18, 0 // (op 3606) 
s_cselect_b32 s19, 0, -1 // (op 3606) 

v_lshrrev_b32 v17, 4, v16 // (op 3606) 

s_and_b32 s18, s12, 31 // (op 3606) 
s_lshl_b32 s49, 1, s18 // (op 3606) 
s_add_i32 s50, s49, s19 // (op 3606) 


s_and_b32 s19, s0, s50 // (op 3606) 


s_add_i32 s0, s1, s19 // (op 3606) 


s_ashr_i32 s1, s0, s18 // (op 3606) 


s_lshl_b32 s0, s12, 24 // (op 3606) 
s_ashr_i32 s18, s0, 31 // (op 3606) 

s_xor_b32 s0, s1, s18 // (op 3606) 

s_sub_i32 s1, s0, s18 // (op 3606) 


s_mul_hi_i32 s0, s1, s13 // (op 3606) 
s_add_i32 s18, s0, s1 // (op 3606) 

s_ashr_i32 s0, s18, 31 // (op 3606) 
s_cmp_eq_i32 s13, 0 // (op 3606) 
s_mov_b32 s19, scc // (op 3606) 
s_cmp_eq_i32 s19, 0 // (op 3606) 
s_cselect_b32 s49, 0, -1 // (op 3606) 

s_and_b32 s19, s14, 31 // (op 3606) 
s_lshl_b32 s50, 1, s19 // (op 3606) 
s_add_i32 s51, s50, s49 // (op 3606) 


s_and_b32 s49, s0, s51 // (op 3606) 


s_add_i32 s0, s18, s49 // (op 3606) 


s_ashr_i32 s18, s0, s19 // (op 3606) 


s_lshl_b32 s0, s14, 24 // (op 3606) 
s_ashr_i32 s19, s0, 31 // (op 3606) 

s_xor_b32 s0, s18, s19 // (op 3606) 

s_sub_i32 s18, s0, s19 // (op 3606) 


s_mul_i32 s0, s1, s44 // (op 3606) 

s_sub_u32 s1, s16, s0 // (op 3606) 

s_mul_i32 s0, s18, s44 // (op 3606) 
s_add_u32 s0, s0, s1 // (op 3606) 


s_mul_hi_i32 s1, s16, s10 // (op 3606) 
s_add_i32 s18, s1, s16 // (op 3606) 

s_ashr_i32 s1, s18, 31 // (op 3606) 
s_cmp_eq_i32 s10, 0 // (op 3606) 
s_mov_b32 s19, scc // (op 3606) 
s_cmp_eq_i32 s19, 0 // (op 3606) 
s_cselect_b32 s49, 0, -1 // (op 3606) 

s_and_b32 s19, s47, 31 // (op 3606) 
s_lshl_b32 s50, 1, s19 // (op 3606) 
s_add_i32 s51, s50, s49 // (op 3606) 


s_and_b32 s49, s1, s51 // (op 3606) 


s_add_i32 s1, s18, s49 // (op 3606) 


s_ashr_i32 s18, s1, s19 // (op 3606) 


s_lshl_b32 s1, s47, 24 // (op 3606) 
s_ashr_i32 s19, s1, 31 // (op 3606) 

s_xor_b32 s1, s18, s19 // (op 3606) 

s_sub_i32 s18, s1, s19 // (op 3606) 


s_add_u32 s50, s6, 256 // (op 3606) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3606) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3606) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3606) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3606) 

s_mul_i32 s1, s44, s50 // (op 3606) 

s_mul_i32 s19, s48, s1 // (op 3606) 

s_mul_i32 s1, s18, s19 // (op 3606) 

s_sub_u32 s19, s16, s1 // (op 3606) 

s_mov_b32 s1, s19 // (op 3606) 
s_mul_hi_i32 s49, s1, s15 // (op 3606) 
s_add_i32 s50, s49, s1 // (op 3606) 


s_ashr_i32 s1, s50, 31 // (op 3606) 
s_cmp_eq_i32 s15, 0 // (op 3606) 
s_mov_b32 s49, scc // (op 3606) 
s_cmp_eq_i32 s49, 0 // (op 3606) 
s_cselect_b32 s51, 0, -1 // (op 3606) 

s_and_b32 s49, s46, 31 // (op 3606) 
s_lshl_b32 s68, 1, s49 // (op 3606) 
s_add_i32 s69, s68, s51 // (op 3606) 


s_and_b32 s51, s1, s69 // (op 3606) 


s_add_i32 s1, s50, s51 // (op 3606) 


s_ashr_i32 s50, s1, s49 // (op 3606) 


s_lshl_b32 s1, s46, 24 // (op 3606) 
s_ashr_i32 s49, s1, 31 // (op 3606) 

s_xor_b32 s1, s50, s49 // (op 3606) 

s_sub_i32 s50, s1, s49 // (op 3606) 


s_mul_i32 s1, s45, s44 // (op 3606) 
s_add_u32 s68, s8, 16 // (op 3606) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 3606) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3606) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3606) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 3606) 

s_sub_i32 s49, s68, s1 // (op 3606) 

s_mul_i32 s51, s50, s49 // (op 3606) 


s_sub_u32 s49, s19, s51 // (op 3606) 


s_add_u32 s19, s49, s1 // (op 3606) 


s_cmp_eq_i32 s18, 0 // (op 3606) 
s_mov_b32 s1, scc // (op 3606) 

s_cmp_eq_i32 s1, 0 // (op 3606) 
s_cselect_b32 s18, s19, s0 // (op 3606) 



v_lshl_add_u32 v16, s18, 4, v19 // (op 3606) 


v_mul_lo_u32 v19, v16, s34 // (op 3606) 
v_add_u32 v19, v19, v17 // (op 3606) 


s_mul_i32 s1, s34, s9 // (op 3608) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3608) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3608) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3608) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3608) most significant: sum
s_add_u32 s1, s1, s19 // (op 3608) most significant: sum


s_mov_b32 s64, s36 // (op 3608) reg expression
s_mov_b32 s65, s37 // (op 3608) reg expression
s_mov_b32 s66, s0 // (op 3608) reg expression
s_mov_b32 s67, 131072 // (op 3608) call()

s_mul_hi_i32 s0, s16, s11 // (op 3651) 
v_and_b32 v16, -32, v1 // (op 3651) 
s_add_i32 s1, s0, s16 // (op 3651) 

v_sub_u32 v17, v1, v16 // (op 3651) 

s_ashr_i32 s0, s1, 31 // (op 3651) 
v_lshlrev_b32 v16, 3, v17 // (op 3651) 

s_cmp_eq_i32 s11, 0 // (op 3651) 
s_mov_b32 s18, scc // (op 3651) 
v_lshrrev_b32 v17, 5, v1 // (op 3651) 
s_cmp_eq_i32 s18, 0 // (op 3651) 
s_cselect_b32 s19, 0, -1 // (op 3651) 

s_and_b32 s18, s12, 31 // (op 3651) 
s_lshl_b32 s49, 1, s18 // (op 3651) 
s_add_i32 s50, s49, s19 // (op 3651) 


s_and_b32 s19, s0, s50 // (op 3651) 


s_add_i32 s0, s1, s19 // (op 3651) 


s_ashr_i32 s1, s0, s18 // (op 3651) 


s_lshl_b32 s0, s12, 24 // (op 3651) 
s_ashr_i32 s18, s0, 31 // (op 3651) 

s_xor_b32 s0, s1, s18 // (op 3651) 

s_sub_i32 s1, s0, s18 // (op 3651) 


s_mul_hi_i32 s0, s1, s13 // (op 3651) 
s_add_i32 s18, s0, s1 // (op 3651) 

s_ashr_i32 s0, s18, 31 // (op 3651) 
s_cmp_eq_i32 s13, 0 // (op 3651) 
s_mov_b32 s19, scc // (op 3651) 
s_cmp_eq_i32 s19, 0 // (op 3651) 
s_cselect_b32 s49, 0, -1 // (op 3651) 

s_and_b32 s19, s14, 31 // (op 3651) 
s_lshl_b32 s50, 1, s19 // (op 3651) 
s_add_i32 s51, s50, s49 // (op 3651) 


s_and_b32 s49, s0, s51 // (op 3651) 


s_add_i32 s0, s18, s49 // (op 3651) 


s_ashr_i32 s18, s0, s19 // (op 3651) 


s_lshl_b32 s0, s14, 24 // (op 3651) 
s_ashr_i32 s19, s0, 31 // (op 3651) 

s_xor_b32 s0, s18, s19 // (op 3651) 

s_sub_i32 s18, s0, s19 // (op 3651) 


s_add_u32 s50, s6, 256 // (op 3651) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3651) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3651) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3651) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3651) 

s_mul_i32 s0, s18, s50 // (op 3651) 

s_sub_i32 s49, s1, s0 // (op 3651) 


s_mul_hi_i32 s0, s16, s10 // (op 3651) 
s_add_i32 s1, s0, s16 // (op 3651) 

s_ashr_i32 s0, s1, 31 // (op 3651) 
s_cmp_eq_i32 s10, 0 // (op 3651) 
s_mov_b32 s18, scc // (op 3651) 
s_cmp_eq_i32 s18, 0 // (op 3651) 
s_cselect_b32 s19, 0, -1 // (op 3651) 

s_and_b32 s18, s47, 31 // (op 3651) 
s_lshl_b32 s68, 1, s18 // (op 3651) 
s_add_i32 s69, s68, s19 // (op 3651) 


s_and_b32 s19, s0, s69 // (op 3651) 


s_add_i32 s0, s1, s19 // (op 3651) 


s_ashr_i32 s1, s0, s18 // (op 3651) 


s_lshl_b32 s0, s47, 24 // (op 3651) 
s_ashr_i32 s18, s0, 31 // (op 3651) 

s_xor_b32 s0, s1, s18 // (op 3651) 

s_sub_i32 s1, s0, s18 // (op 3651) 


s_mul_i32 s0, s44, s50 // (op 3651) 

s_mul_i32 s18, s48, s0 // (op 3651) 

s_mul_i32 s0, s1, s18 // (op 3651) 

s_sub_u32 s18, s16, s0 // (op 3651) 

s_mul_hi_i32 s0, s18, s15 // (op 3651) 
s_add_i32 s19, s0, s18 // (op 3651) 


s_ashr_i32 s0, s19, 31 // (op 3651) 
s_cmp_eq_i32 s15, 0 // (op 3651) 
s_mov_b32 s18, scc // (op 3651) 
s_cmp_eq_i32 s18, 0 // (op 3651) 
s_cselect_b32 s50, 0, -1 // (op 3651) 

s_and_b32 s18, s46, 31 // (op 3651) 
s_lshl_b32 s51, 1, s18 // (op 3651) 
s_add_i32 s68, s51, s50 // (op 3651) 


s_and_b32 s50, s0, s68 // (op 3651) 


s_add_i32 s0, s19, s50 // (op 3651) 


s_ashr_i32 s19, s0, s18 // (op 3651) 


s_lshl_b32 s0, s46, 24 // (op 3651) 
s_ashr_i32 s18, s0, 31 // (op 3651) 

s_xor_b32 s0, s19, s18 // (op 3651) 

s_sub_i32 s19, s0, s18 // (op 3651) 


s_cmp_eq_i32 s1, 0 // (op 3651) 
s_mov_b32 s0, scc // (op 3651) 

s_cmp_eq_i32 s0, 0 // (op 3651) 
s_cselect_b32 s1, s19, s49 // (op 3651) 



v_lshl_add_u32 v20, s1, 8, v16 // (op 3651) 


v_mul_lo_u32 v16, v17, s38 // (op 3651) 
v_add_u32 v16, v16, v20 // (op 3651) 


s_mul_i32 s1, s38, s43 // (op 3653) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3653) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3653) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3653) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3653) most significant: sum
s_add_u32 s1, s1, s19 // (op 3653) most significant: sum


s_mov_b32 s68, s40 // (op 3653) reg expression
s_mov_b32 s69, s41 // (op 3653) reg expression
s_mov_b32 s70, s0 // (op 3653) reg expression
s_mov_b32 s71, 131072 // (op 3653) call()

s_mul_hi_i32 s0, s16, s11 // (op 3670) 
v_and_b32 v17, -32, v1 // (op 3670) 
s_add_i32 s1, s0, s16 // (op 3670) 

v_sub_u32 v20, v1, v17 // (op 3670) 

s_ashr_i32 s0, s1, 31 // (op 3670) 
v_lshlrev_b32 v17, 3, v20 // (op 3670) 

s_cmp_eq_i32 s11, 0 // (op 3670) 
s_mov_b32 s18, scc // (op 3670) 
v_lshrrev_b32 v20, 5, v1 // (op 3670) 
s_cmp_eq_i32 s18, 0 // (op 3670) 
s_cselect_b32 s19, 0, -1 // (op 3670) 

s_and_b32 s18, s12, 31 // (op 3670) 
s_lshl_b32 s49, 1, s18 // (op 3670) 
s_add_i32 s50, s49, s19 // (op 3670) 


s_and_b32 s19, s0, s50 // (op 3670) 


s_add_i32 s0, s1, s19 // (op 3670) 


s_ashr_i32 s1, s0, s18 // (op 3670) 


s_lshl_b32 s0, s12, 24 // (op 3670) 
s_ashr_i32 s18, s0, 31 // (op 3670) 

s_xor_b32 s0, s1, s18 // (op 3670) 

s_sub_i32 s1, s0, s18 // (op 3670) 


s_mul_hi_i32 s0, s1, s13 // (op 3670) 
s_add_i32 s18, s0, s1 // (op 3670) 

s_ashr_i32 s0, s18, 31 // (op 3670) 
s_cmp_eq_i32 s13, 0 // (op 3670) 
s_mov_b32 s19, scc // (op 3670) 
s_cmp_eq_i32 s19, 0 // (op 3670) 
s_cselect_b32 s49, 0, -1 // (op 3670) 

s_and_b32 s19, s14, 31 // (op 3670) 
s_lshl_b32 s50, 1, s19 // (op 3670) 
s_add_i32 s51, s50, s49 // (op 3670) 


s_and_b32 s49, s0, s51 // (op 3670) 


s_add_i32 s0, s18, s49 // (op 3670) 


s_ashr_i32 s18, s0, s19 // (op 3670) 


s_lshl_b32 s0, s14, 24 // (op 3670) 
s_ashr_i32 s19, s0, 31 // (op 3670) 

s_xor_b32 s0, s18, s19 // (op 3670) 

s_sub_i32 s18, s0, s19 // (op 3670) 


s_add_u32 s50, s6, 256 // (op 3670) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3670) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3670) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3670) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3670) 

s_mul_i32 s0, s18, s50 // (op 3670) 

s_sub_i32 s49, s1, s0 // (op 3670) 


s_mul_hi_i32 s0, s16, s10 // (op 3670) 
s_add_i32 s1, s0, s16 // (op 3670) 

s_ashr_i32 s0, s1, 31 // (op 3670) 
s_cmp_eq_i32 s10, 0 // (op 3670) 
s_mov_b32 s18, scc // (op 3670) 
s_cmp_eq_i32 s18, 0 // (op 3670) 
s_cselect_b32 s19, 0, -1 // (op 3670) 

s_and_b32 s18, s47, 31 // (op 3670) 
s_lshl_b32 s72, 1, s18 // (op 3670) 
s_add_i32 s73, s72, s19 // (op 3670) 


s_and_b32 s19, s0, s73 // (op 3670) 


s_add_i32 s0, s1, s19 // (op 3670) 


s_ashr_i32 s1, s0, s18 // (op 3670) 


s_lshl_b32 s0, s47, 24 // (op 3670) 
s_ashr_i32 s18, s0, 31 // (op 3670) 

s_xor_b32 s0, s1, s18 // (op 3670) 

s_sub_i32 s1, s0, s18 // (op 3670) 


s_mul_i32 s0, s44, s50 // (op 3670) 

s_mul_i32 s18, s48, s0 // (op 3670) 

s_mul_i32 s0, s1, s18 // (op 3670) 

s_sub_u32 s18, s16, s0 // (op 3670) 

s_mul_hi_i32 s0, s18, s15 // (op 3670) 
s_add_i32 s19, s0, s18 // (op 3670) 


s_ashr_i32 s0, s19, 31 // (op 3670) 
s_cmp_eq_i32 s15, 0 // (op 3670) 
s_mov_b32 s18, scc // (op 3670) 
s_cmp_eq_i32 s18, 0 // (op 3670) 
s_cselect_b32 s50, 0, -1 // (op 3670) 

s_and_b32 s18, s46, 31 // (op 3670) 
s_lshl_b32 s51, 1, s18 // (op 3670) 
s_add_i32 s72, s51, s50 // (op 3670) 


s_and_b32 s50, s0, s72 // (op 3670) 


s_add_i32 s0, s19, s50 // (op 3670) 


s_ashr_i32 s19, s0, s18 // (op 3670) 


s_lshl_b32 s0, s46, 24 // (op 3670) 
s_ashr_i32 s18, s0, 31 // (op 3670) 

s_xor_b32 s0, s19, s18 // (op 3670) 

s_sub_i32 s19, s0, s18 // (op 3670) 


s_cmp_eq_i32 s1, 0 // (op 3670) 
s_mov_b32 s0, scc // (op 3670) 

s_cmp_eq_i32 s0, 0 // (op 3670) 
s_cselect_b32 s1, s19, s49 // (op 3670) 



v_lshl_add_u32 v21, s1, 8, v17 // (op 3670) 


v_mul_lo_u32 v17, v20, s38 // (op 3670) 
v_add_u32 v17, v17, v21 // (op 3670) 


s_mul_i32 s1, s38, s43 // (op 3672) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3672) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3672) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3672) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3672) most significant: sum
s_add_u32 s1, s1, s19 // (op 3672) most significant: sum


s_mov_b32 s68, s40 // (op 3672) reg expression
s_mov_b32 s69, s41 // (op 3672) reg expression
s_mov_b32 s70, s0 // (op 3672) reg expression
s_mov_b32 s71, 131072 // (op 3672) call()




buffer_load_ubyte v20, v18, s[64:67], 0 offen offset: 0 // (op 3140) Load value

v_mov_b64 v[22:23], s[28:29] // (op 3144) convert

v_mov_b32 v21, v14 // (op 3144) reg expression
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[36:39], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[40:43], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[44:47], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[48:51], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[52:55], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[56:59], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[60:63], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[64:67], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[68:71], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[72:75], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[76:79], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[80:83], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[84:87], v21, s[52:55], 0 offen // (op 3144) Load value




v_mov_b32 v21, v16 // (op 3147) reg expression

buffer_load_dwordx2 v[22:23], v21, s[68:71], 0 offen // (op 3147) Load value




buffer_load_dwordx2 v[88:89], v12, s[20:23], 0 offen // (op 3150) Load value



buffer_load_ubyte v21, v18, s[64:67], 0 offen offset:4 // (op 3154) Load value


v_mov_b64 v[90:91], s[28:29] // (op 3158) convert

s_lshl_b32 s0, s28, 7 // (op 3158) 
v_add_u32 v18, v14, s0 // (op 3158) 

buffer_load_dwordx4 v[92:95], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[96:99], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[100:103], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[104:107], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[108:111], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[112:115], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[116:119], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[120:123], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[124:127], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[128:131], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[132:135], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[136:139], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[140:143], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[144:147], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[148:151], v18, s[52:55], 0 offen // (op 3158) Load value
v_add_u32 v18, v18, v90 // (op 3158) 
buffer_load_dwordx4 v[152:155], v18, s[52:55], 0 offen // (op 3158) Load value





s_lshl_b32 s0, s38, 2 // (op 3161) 
v_add_u32 v14, v16, s0 // (op 3161) 


buffer_load_dwordx2 v[90:91], v14, s[68:71], 0 offen // (op 3161) Load value





buffer_load_dwordx2 v[156:157], v12, s[20:23], 0 offen offset:64 // (op 3164) Load value

v_and_b32 v12, -16, v1 // (op 3790) 
v_sub_u32 v14, v1, v12 // (op 3790) 

v_lshlrev_b32 v12, 4, v14 // (op 3790) 

v_lshrrev_b32 v14, 4, v1 // (op 3790) 
v_lshl_add_u32 v16, v14, 12, v12 // (op 3790) 





s_waitcnt vmcnt(36)
ds_write_b128 v16, v[24:27] // (op 3166) Store local data 
s_waitcnt vmcnt(35)
ds_write_b128 v16, v[28:31] offset:256 // (op 3166) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v16, v[32:35] offset:512 // (op 3166) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v16, v[36:39] offset:768 // (op 3166) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v16, v[40:43] offset:1024 // (op 3166) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v16, v[44:47] offset:1280 // (op 3166) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v16, v[48:51] offset:1536 // (op 3166) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v16, v[52:55] offset:1792 // (op 3166) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v16, v[56:59] offset:2048 // (op 3166) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v16, v[60:63] offset:2304 // (op 3166) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v16, v[64:67] offset:2560 // (op 3166) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v16, v[68:71] offset:2816 // (op 3166) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v16, v[72:75] offset:3072 // (op 3166) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v16, v[76:79] offset:3328 // (op 3166) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v16, v[80:83] offset:3584 // (op 3166) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v16, v[84:87] offset:3840 // (op 3166) Store local data 

v_and_b32 v12, -32, v1 // (op 3848) 
v_sub_u32 v14, v1, v12 // (op 3848) 

v_lshlrev_b32 v12, 3, v14 // (op 3848) 

v_lshrrev_b32 v14, 5, v1 // (op 3848) 
v_lshl_add_u32 v16, v14, 8, v12 // (op 3848) 





s_waitcnt vmcnt(20)
ds_write_b64 v16, v[22:23] offset:32768 // (op 3167) Store local data 

v_and_b32 v12, -8, v1 // (op 3737) 
v_sub_u32 v14, v1, v12 // (op 3737) 

v_lshlrev_b32 v12, 4, v14 // (op 3737) 

v_lshrrev_b32 v14, 3, v1 // (op 3737) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 3737) 


v_lshlrev_b32 v12, 2, v16 // (op 3737) 

v_lshrrev_b32 v14, 3, v12 // (op 3737) 




s_waitcnt vmcnt(19)
ds_write_b64 v14, v[88:89] offset:33792 // (op 3168) Store local data 

v_and_b32 v12, -8, v1 // (op 3708) 
v_sub_u32 v14, v1, v12 // (op 3708) 

v_lshlrev_b32 v12, 4, v14 // (op 3708) 

v_lshrrev_b32 v14, 3, v1 // (op 3708) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 3708) 


v_lshlrev_b32 v12, 2, v16 // (op 3708) 

v_lshrrev_b32 v14, 3, v12 // (op 3708) 

v_and_b32 v12, -16, v1 // (op 3761) 
v_sub_u32 v16, v1, v12 // (op 3761) 

v_lshlrev_b32 v12, 4, v16 // (op 3761) 

v_lshrrev_b32 v16, 4, v1 // (op 3761) 
v_lshl_add_u32 v18, v16, 12, v12 // (op 3761) 


v_and_b32 v12, -32, v1 // (op 3819) 
v_sub_u32 v16, v1, v12 // (op 3819) 

v_lshlrev_b32 v12, 3, v16 // (op 3819) 

v_lshrrev_b32 v16, 5, v1 // (op 3819) 
v_lshl_add_u32 v158, v16, 8, v12 // (op 3819) 


v_and_b32 v12, -8, v1 // (op 3895) 
v_sub_u32 v16, v1, v12 // (op 3895) 

v_lshlrev_b32 v12, 4, v16 // (op 3895) 

v_lshrrev_b32 v16, 3, v1 // (op 3895) 
v_lshl_add_u32 v159, v16, 7, v12 // (op 3895) 


v_lshlrev_b32 v12, 2, v159 // (op 3895) 

v_lshrrev_b32 v16, 3, v12 // (op 3895) 

v_and_b32 v12, -16, v1 // (op 3919) 
v_sub_u32 v159, v1, v12 // (op 3919) 

v_lshlrev_b32 v12, 4, v159 // (op 3919) 

v_lshrrev_b32 v159, 4, v1 // (op 3919) 
v_lshl_add_u32 v160, v159, 12, v12 // (op 3919) 


v_and_b32 v12, -32, v1 // (op 3948) 
v_sub_u32 v159, v1, v12 // (op 3948) 

v_lshlrev_b32 v12, 3, v159 // (op 3948) 

v_lshrrev_b32 v159, 5, v1 // (op 3948) 
v_lshl_add_u32 v161, v159, 8, v12 // (op 3948) 


s_mov_b32 s0, 0 // (op 1248) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1247) 
s_mov_b32 s1, s18 // (op 1247) convert

s_and_b32 s18, s1, 4294967294 // (op 1247) 

s_cmp_lt_u32 s0, s18 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1247 // (op 1247) Condition: Top (jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1247 if false)
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1247:
s_add_u32 s0, s0, 2 // (op 1249) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4024) 



ds_read_b128 v[162:165], v3 offset:33792 // (op 67) Load local data 
s_barrier  // (op 4037) 



ds_read_b64_tr_b8 v[166:167], v7 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v7 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[170:171], v7 offset:16384 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v7 offset:18432 // (op 72) Transpose load local data 
s_barrier  // (op 4051) 


ds_read_u8 v12, v0 offset:32768 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[162:165], v[166:173], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 84) 
ds_read_b64_tr_b8 v[174:175], v7 offset:32 // (op 343) Transpose load local data 


ds_read_b64_tr_b8 v[176:177], v7 offset:2080 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[178:179], v7 offset:16416 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v7 offset:18464 // (op 343) Transpose load local data 


ds_read_u8 v12, v0 offset:32800 // (op 346) Load local data 



ds_read_b64_tr_b8 v[166:167], v7 offset:64 // (op 499) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[162:165], v[174:181], a[4:7], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 347) 
ds_read_b64_tr_b8 v[168:169], v7 offset:2112 // (op 499) Transpose load local data 


ds_read_b64_tr_b8 v[170:171], v7 offset:16448 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v7 offset:18496 // (op 499) Transpose load local data 


ds_read_u8 v12, v0 offset:32832 // (op 502) Load local data 



ds_read_b64_tr_b8 v[174:175], v7 offset:96 // (op 655) Transpose load local data 
ds_read_b64_tr_b8 v[176:177], v7 offset:2144 // (op 655) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[162:165], v[166:173], a[8:11], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 503) 
ds_read_b64_tr_b8 v[178:179], v7 offset:16480 // (op 655) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v7 offset:18528 // (op 655) Transpose load local data 


ds_read_u8 v12, v0 offset:32864 // (op 658) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[162:165], v[174:181], a[12:15], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 659) 
ds_read_b64_tr_b8 v[166:167], v7 offset:128 // (op 811) Transpose load local data 


ds_read_b64_tr_b8 v[168:169], v7 offset:2176 // (op 811) Transpose load local data 
ds_read_b64_tr_b8 v[170:171], v7 offset:16512 // (op 811) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v7 offset:18560 // (op 811) Transpose load local data 


ds_read_u8 v12, v0 offset:32896 // (op 814) Load local data 



ds_read_b64_tr_b8 v[174:175], v7 offset:160 // (op 967) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[162:165], v[166:173], a[16:19], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 815) 
ds_read_b64_tr_b8 v[176:177], v7 offset:2208 // (op 967) Transpose load local data 


ds_read_b64_tr_b8 v[178:179], v7 offset:16544 // (op 967) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v7 offset:18592 // (op 967) Transpose load local data 


ds_read_u8 v12, v0 offset:32928 // (op 970) Load local data 



ds_read_b64_tr_b8 v[166:167], v7 offset:192 // (op 1123) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v7 offset:2240 // (op 1123) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[162:165], v[174:181], a[20:23], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 971) 
ds_read_b64_tr_b8 v[170:171], v7 offset:16576 // (op 1123) Transpose load local data 


ds_read_b64_tr_b8 v[172:173], v7 offset:18624 // (op 1123) Transpose load local data 


ds_read_u8 v12, v0 offset:32960 // (op 1126) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[162:165], v[166:173], a[24:27], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1127) 
ds_read_b64_tr_b8 v[174:175], v7 offset:224 // (op 1279) Transpose load local data 


ds_read_b64_tr_b8 v[176:177], v7 offset:2272 // (op 1279) Transpose load local data 
ds_read_b64_tr_b8 v[178:179], v7 offset:16608 // (op 1279) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v7 offset:18656 // (op 1279) Transpose load local data 


ds_read_u8 v12, v0 offset:32992 // (op 1282) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[162:165], v[174:181], a[28:31], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1283) 





v_mov_b32 v12, v19 // (op 77) 

buffer_load_ubyte v20, v12, s[64:67], 0 offen offset:8 // (op 77) Load value

v_mov_b64 v[162:163], s[28:29] // (op 10) convert

v_mov_b32 v159, v15 // (op 10) 
s_lshl_b32 s1, s28, 8 // (op 10) 
v_add_u32 v164, v159, s1 // (op 10) 

buffer_load_dwordx4 v[24:27], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[28:31], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[32:35], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[36:39], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[40:43], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[44:47], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[48:51], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[52:55], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[56:59], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[60:63], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[64:67], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[68:71], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[72:75], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[76:79], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[80:83], v164, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v164, v164, v162 // (op 10) 
buffer_load_dwordx4 v[84:87], v164, s[52:55], 0 offen // (op 10) Load value




v_mov_b32 v162, v17 // (op 21) 
s_lshl_b32 s1, s38, 3 // (op 21) 
v_add_u32 v163, v162, s1 // (op 21) 


buffer_load_dwordx2 v[22:23], v163, s[68:71], 0 offen // (op 21) Load value



v_mov_b32 v163, v13 // (op 4) 

buffer_load_dwordx2 v[88:89], v163, s[20:23], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 4074) 



s_waitcnt vmcnt(36)
ds_write_b128 v160, v[92:95] offset:34816 // (op 239) Store local data 
s_waitcnt vmcnt(35)
ds_write_b128 v160, v[96:99] offset:35072 // (op 239) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v160, v[100:103] offset:35328 // (op 239) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v160, v[104:107] offset:35584 // (op 239) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v160, v[108:111] offset:35840 // (op 239) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v160, v[112:115] offset:36096 // (op 239) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v160, v[116:119] offset:36352 // (op 239) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v160, v[120:123] offset:36608 // (op 239) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v160, v[124:127] offset:36864 // (op 239) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v160, v[128:131] offset:37120 // (op 239) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v160, v[132:135] offset:37376 // (op 239) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v160, v[136:139] offset:37632 // (op 239) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v160, v[140:143] offset:37888 // (op 239) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v160, v[144:147] offset:38144 // (op 239) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v160, v[148:151] offset:38400 // (op 239) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v160, v[152:155] offset:38656 // (op 239) Store local data 
s_barrier  // (op 4080) 



v_add_u32 v164, 67584, v161 // (op 240) 
s_waitcnt vmcnt(20)
ds_write_b64 v164, v[90:91] // (op 240) Store local data 

s_barrier  // (op 4068) 



v_add_u32 v164, 68608, v16 // (op 229) 
s_waitcnt vmcnt(19)
ds_write_b64 v164, v[156:157] // (op 229) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4065) 



v_add_u32 v164, 68608, v4 // (op 231) 
ds_read_b128 v[166:169], v164 // (op 231) Load local data 

s_barrier  // (op 4071) 
s_barrier  // (op 4121) 



ds_read_b64_tr_b8 v[170:171], v11 offset:34816 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v11 offset:36864 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[174:175], v11 offset:51200 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[176:177], v11 offset:53248 // (op 233) Transpose load local data 
s_barrier  // (op 4077) 


v_add_u32 v164, 67584, v10 // (op 237) 
ds_read_u8 v165, v164 // (op 237) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[166:169], v[170:177], a[0:3], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 238) 
ds_read_b64_tr_b8 v[178:179], v11 offset:34848 // (op 352) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v11 offset:36896 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[182:183], v11 offset:51232 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v11 offset:53280 // (op 352) Transpose load local data 


v_add_u32 v164, 67616, v10 // (op 355) 
ds_read_u8 v165, v164 // (op 355) Load local data 




ds_read_b64_tr_b8 v[170:171], v11 offset:34880 // (op 508) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[166:169], v[178:185], a[4:7], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 348) 
ds_read_b64_tr_b8 v[172:173], v11 offset:36928 // (op 508) Transpose load local data 


ds_read_b64_tr_b8 v[174:175], v11 offset:51264 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[176:177], v11 offset:53312 // (op 508) Transpose load local data 


v_add_u32 v164, 67648, v10 // (op 511) 
ds_read_u8 v165, v164 // (op 511) Load local data 




ds_read_b64_tr_b8 v[178:179], v11 offset:34912 // (op 664) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v11 offset:36960 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[166:169], v[170:177], a[8:11], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 504) 
ds_read_b64_tr_b8 v[182:183], v11 offset:51296 // (op 664) Transpose load local data 


ds_read_b64_tr_b8 v[184:185], v11 offset:53344 // (op 664) Transpose load local data 


v_add_u32 v164, 67680, v10 // (op 667) 
ds_read_u8 v165, v164 // (op 667) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[166:169], v[178:185], a[12:15], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 660) 
ds_read_b64_tr_b8 v[170:171], v11 offset:34944 // (op 820) Transpose load local data 


ds_read_b64_tr_b8 v[172:173], v11 offset:36992 // (op 820) Transpose load local data 
ds_read_b64_tr_b8 v[174:175], v11 offset:51328 // (op 820) Transpose load local data 
ds_read_b64_tr_b8 v[176:177], v11 offset:53376 // (op 820) Transpose load local data 


v_add_u32 v164, 67712, v10 // (op 823) 
ds_read_u8 v165, v164 // (op 823) Load local data 




ds_read_b64_tr_b8 v[178:179], v11 offset:34976 // (op 976) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[166:169], v[170:177], a[16:19], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 816) 
ds_read_b64_tr_b8 v[180:181], v11 offset:37024 // (op 976) Transpose load local data 


ds_read_b64_tr_b8 v[182:183], v11 offset:51360 // (op 976) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v11 offset:53408 // (op 976) Transpose load local data 


v_add_u32 v164, 67744, v10 // (op 979) 
ds_read_u8 v165, v164 // (op 979) Load local data 




ds_read_b64_tr_b8 v[170:171], v11 offset:35008 // (op 1132) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v11 offset:37056 // (op 1132) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:169], v[178:185], a[20:23], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 972) 
ds_read_b64_tr_b8 v[174:175], v11 offset:51392 // (op 1132) Transpose load local data 


ds_read_b64_tr_b8 v[176:177], v11 offset:53440 // (op 1132) Transpose load local data 


v_add_u32 v164, 67776, v10 // (op 1135) 
ds_read_u8 v165, v164 // (op 1135) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[166:169], v[170:177], a[24:27], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1128) 
ds_read_b64_tr_b8 v[178:179], v11 offset:35040 // (op 1288) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v11 offset:37088 // (op 1288) Transpose load local data 
ds_read_b64_tr_b8 v[182:183], v11 offset:51424 // (op 1288) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v11 offset:53472 // (op 1288) Transpose load local data 


s_barrier  // (op 4191) 
v_add_u32 v164, 67808, v10 // (op 1291) 
ds_read_u8 v165, v164 // (op 1291) Load local data 

s_barrier  // (op 4195) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[166:169], v[178:185], a[28:31], v21, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1284) 






buffer_load_ubyte v21, v12, s[64:67], 0 offen offset:12 // (op 235) Load value


v_add_u32 v19, 8, v19 // (op 3598) 
v_mov_b64 v[164:165], s[28:29] // (op 241) convert

s_lshl_b32 s1, s28, 7 // (op 241) 
s_mul_i32 s18, 3, s1 // (op 241) 

v_add_u32 v12, v159, s18 // (op 241) 

buffer_load_dwordx4 v[92:95], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[96:99], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[100:103], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[104:107], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[108:111], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[112:115], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[116:119], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[120:123], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[124:127], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[128:131], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[132:135], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[136:139], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[140:143], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[144:147], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[148:151], v12, s[52:55], 0 offen // (op 241) Load value
v_add_u32 v12, v12, v164 // (op 241) 
buffer_load_dwordx4 v[152:155], v12, s[52:55], 0 offen // (op 241) Load value





s_lshl_b32 s1, s38, 2 // (op 242) 
s_mul_i32 s18, 3, s1 // (op 242) 

s_lshl_b32 s1, s28, 8 // (op 3534) 
v_add_u32 v15, v15, s1 // (op 3534) 

v_add_u32 v12, v162, s18 // (op 242) 


buffer_load_dwordx2 v[90:91], v12, s[68:71], 0 offen // (op 242) Load value





s_lshl_b32 s1, s38, 3 // (op 3662) 
v_add_u32 v17, v17, s1 // (op 3662) 

buffer_load_dwordx2 v[156:157], v163, s[20:23], 0 offen offset:192 // (op 228) Load value

s_barrier  // (op 4040) 
v_add_u32 v13, 128, v13 // (op 3470) 



s_waitcnt vmcnt(36)
ds_write_b128 v18, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(35)
ds_write_b128 v18, v[28:31] offset:256 // (op 13) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v18, v[32:35] offset:512 // (op 13) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v18, v[36:39] offset:768 // (op 13) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v18, v[40:43] offset:1024 // (op 13) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v18, v[44:47] offset:1280 // (op 13) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v18, v[48:51] offset:1536 // (op 13) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v18, v[52:55] offset:1792 // (op 13) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v18, v[56:59] offset:2048 // (op 13) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v18, v[60:63] offset:2304 // (op 13) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v18, v[64:67] offset:2560 // (op 13) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v18, v[68:71] offset:2816 // (op 13) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v18, v[72:75] offset:3072 // (op 13) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v18, v[76:79] offset:3328 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v18, v[80:83] offset:3584 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v18, v[84:87] offset:3840 // (op 13) Store local data 
s_barrier  // (op 4054) 



s_waitcnt vmcnt(20)
ds_write_b64 v158, v[22:23] offset:32768 // (op 26) Store local data 
s_barrier  // (op 4027) 



s_waitcnt vmcnt(19)
ds_write_b64 v14, v[88:89] offset:33792 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1247) 
s_mov_b32 s1, s18 // (op 1247) convert

s_and_b32 s18, s1, 4294967294 // (op 1247) 

s_cmp_lt_u32 s0, s18 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1247 // (op 1247) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1247 if true)
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1247:















s_mul_hi_i32 s0, s16, s11 // (op 3426) 
v_lshrrev_b32 v12, 3, v1 // (op 3426) 
s_add_i32 s1, s0, s16 // (op 3426) 

v_and_b32 v13, -8, v1 // (op 3426) 
s_ashr_i32 s0, s1, 31 // (op 3426) 
v_sub_u32 v14, v1, v13 // (op 3426) 

s_cmp_eq_i32 s11, 0 // (op 3426) 
s_mov_b32 s18, scc // (op 3426) 
v_lshlrev_b32 v13, 4, v14 // (op 3426) 

s_cmp_eq_i32 s18, 0 // (op 3426) 
s_cselect_b32 s19, 0, -1 // (op 3426) 

s_and_b32 s18, s12, 31 // (op 3426) 
s_lshl_b32 s49, 1, s18 // (op 3426) 
s_add_i32 s50, s49, s19 // (op 3426) 


s_and_b32 s19, s0, s50 // (op 3426) 


s_add_i32 s0, s1, s19 // (op 3426) 


s_ashr_i32 s1, s0, s18 // (op 3426) 


s_lshl_b32 s0, s12, 24 // (op 3426) 
s_ashr_i32 s18, s0, 31 // (op 3426) 

s_xor_b32 s0, s1, s18 // (op 3426) 

s_sub_i32 s1, s0, s18 // (op 3426) 


s_mul_hi_i32 s0, s1, s13 // (op 3426) 
s_add_i32 s18, s0, s1 // (op 3426) 

s_ashr_i32 s0, s18, 31 // (op 3426) 
s_cmp_eq_i32 s13, 0 // (op 3426) 
s_mov_b32 s19, scc // (op 3426) 
s_cmp_eq_i32 s19, 0 // (op 3426) 
s_cselect_b32 s49, 0, -1 // (op 3426) 

s_and_b32 s19, s14, 31 // (op 3426) 
s_lshl_b32 s50, 1, s19 // (op 3426) 
s_add_i32 s51, s50, s49 // (op 3426) 


s_and_b32 s49, s0, s51 // (op 3426) 


s_add_i32 s0, s18, s49 // (op 3426) 


s_ashr_i32 s18, s0, s19 // (op 3426) 


s_lshl_b32 s0, s14, 24 // (op 3426) 
s_ashr_i32 s19, s0, 31 // (op 3426) 

s_xor_b32 s0, s18, s19 // (op 3426) 

s_sub_i32 s18, s0, s19 // (op 3426) 


s_mul_i32 s0, s1, s44 // (op 3426) 

s_sub_u32 s1, s16, s0 // (op 3426) 

s_mul_i32 s0, s18, s44 // (op 3426) 
s_add_u32 s0, s0, s1 // (op 3426) 


s_mul_hi_i32 s1, s16, s10 // (op 3426) 
s_add_i32 s18, s1, s16 // (op 3426) 

s_ashr_i32 s1, s18, 31 // (op 3426) 
s_cmp_eq_i32 s10, 0 // (op 3426) 
s_mov_b32 s19, scc // (op 3426) 
s_cmp_eq_i32 s19, 0 // (op 3426) 
s_cselect_b32 s49, 0, -1 // (op 3426) 

s_and_b32 s19, s47, 31 // (op 3426) 
s_lshl_b32 s50, 1, s19 // (op 3426) 
s_add_i32 s51, s50, s49 // (op 3426) 


s_and_b32 s49, s1, s51 // (op 3426) 


s_add_i32 s1, s18, s49 // (op 3426) 


s_ashr_i32 s18, s1, s19 // (op 3426) 


s_lshl_b32 s1, s47, 24 // (op 3426) 
s_ashr_i32 s19, s1, 31 // (op 3426) 

s_xor_b32 s1, s18, s19 // (op 3426) 

s_sub_i32 s18, s1, s19 // (op 3426) 


s_add_u32 s50, s6, 256 // (op 3426) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3426) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3426) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3426) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3426) 

s_mul_i32 s1, s44, s50 // (op 3426) 

s_mul_i32 s19, s48, s1 // (op 3426) 

s_mul_i32 s1, s18, s19 // (op 3426) 

s_sub_u32 s19, s16, s1 // (op 3426) 

s_mov_b32 s1, s19 // (op 3426) 
s_mul_hi_i32 s49, s1, s15 // (op 3426) 
s_add_i32 s50, s49, s1 // (op 3426) 


s_ashr_i32 s1, s50, 31 // (op 3426) 
s_cmp_eq_i32 s15, 0 // (op 3426) 
s_mov_b32 s49, scc // (op 3426) 
s_cmp_eq_i32 s49, 0 // (op 3426) 
s_cselect_b32 s51, 0, -1 // (op 3426) 

s_and_b32 s49, s46, 31 // (op 3426) 
s_lshl_b32 s72, 1, s49 // (op 3426) 
s_add_i32 s73, s72, s51 // (op 3426) 


s_and_b32 s51, s1, s73 // (op 3426) 


s_add_i32 s1, s50, s51 // (op 3426) 


s_ashr_i32 s50, s1, s49 // (op 3426) 


s_lshl_b32 s1, s46, 24 // (op 3426) 
s_ashr_i32 s49, s1, 31 // (op 3426) 

s_xor_b32 s1, s50, s49 // (op 3426) 

s_sub_i32 s50, s1, s49 // (op 3426) 


s_mul_i32 s1, s45, s44 // (op 3426) 
s_add_u32 s72, s8, 16 // (op 3426) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3426) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3426) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3426) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 3426) 

s_sub_i32 s49, s72, s1 // (op 3426) 

s_mul_i32 s51, s50, s49 // (op 3426) 


s_sub_u32 s49, s19, s51 // (op 3426) 


s_add_u32 s19, s49, s1 // (op 3426) 


s_cmp_eq_i32 s18, 0 // (op 3426) 
s_mov_b32 s1, scc // (op 3426) 

s_cmp_eq_i32 s1, 0 // (op 3426) 
s_cselect_b32 s18, s19, s0 // (op 3426) 



v_lshl_add_u32 v14, s18, 4, v12 // (op 3426) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3426) 
s_lshr_b32 s18, s0, 1 // (op 3426) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 3426) 


v_mov_b32 v23, 0 // (op 3426) convert
v_mov_b32 v22, v12 // (op 3426) convert

v_mul_lo_u32 v13, v14, s25 // (op 3426) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 3426) most significant: high of low * low
v_mul_lo_u32 v12, v14, s24 // (op 3426) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3426) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 3426) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 3426) most significant half



v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 3426) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3426) 

v_mov_b32 v14, v12 // (op 3426) convert

s_mul_i32 s1, s24, s9 // (op 3428) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3428) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3428) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3428) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3428) most significant: sum
s_add_u32 s1, s1, s19 // (op 3428) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3428) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3428) 

s_mov_b32 s20, s26 // (op 3428) reg expression
s_mov_b32 s21, s27 // (op 3428) reg expression
s_mov_b32 s22, s0 // (op 3428) reg expression
s_mov_b32 s23, 131072 // (op 3428) call()

s_mul_hi_i32 s0, s16, s11 // (op 3500) 
v_and_b32 v12, -16, v1 // (op 3500) 
s_add_i32 s1, s0, s16 // (op 3500) 

v_sub_u32 v13, v1, v12 // (op 3500) 

s_ashr_i32 s0, s1, 31 // (op 3500) 
v_lshlrev_b32 v12, 4, v13 // (op 3500) 

s_cmp_eq_i32 s11, 0 // (op 3500) 
s_mov_b32 s18, scc // (op 3500) 
v_and_b32 v13, 4294967280, v1 // (op 3500) 
s_cmp_eq_i32 s18, 0 // (op 3500) 
s_cselect_b32 s19, 0, -1 // (op 3500) 

s_and_b32 s18, s12, 31 // (op 3500) 
s_lshl_b32 s49, 1, s18 // (op 3500) 
s_add_i32 s50, s49, s19 // (op 3500) 


s_and_b32 s19, s0, s50 // (op 3500) 


s_add_i32 s0, s1, s19 // (op 3500) 


s_ashr_i32 s1, s0, s18 // (op 3500) 


s_lshl_b32 s0, s12, 24 // (op 3500) 
s_ashr_i32 s18, s0, 31 // (op 3500) 

s_xor_b32 s0, s1, s18 // (op 3500) 

s_sub_i32 s1, s0, s18 // (op 3500) 


s_mul_hi_i32 s0, s1, s13 // (op 3500) 
s_add_i32 s18, s0, s1 // (op 3500) 

s_ashr_i32 s0, s18, 31 // (op 3500) 
s_cmp_eq_i32 s13, 0 // (op 3500) 
s_mov_b32 s19, scc // (op 3500) 
s_cmp_eq_i32 s19, 0 // (op 3500) 
s_cselect_b32 s49, 0, -1 // (op 3500) 

s_and_b32 s19, s14, 31 // (op 3500) 
s_lshl_b32 s50, 1, s19 // (op 3500) 
s_add_i32 s51, s50, s49 // (op 3500) 


s_and_b32 s49, s0, s51 // (op 3500) 


s_add_i32 s0, s18, s49 // (op 3500) 


s_ashr_i32 s18, s0, s19 // (op 3500) 


s_lshl_b32 s0, s14, 24 // (op 3500) 
s_ashr_i32 s19, s0, 31 // (op 3500) 

s_xor_b32 s0, s18, s19 // (op 3500) 

s_sub_i32 s18, s0, s19 // (op 3500) 


s_add_u32 s50, s6, 256 // (op 3500) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3500) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3500) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3500) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3500) 

s_mul_i32 s0, s18, s50 // (op 3500) 

s_sub_i32 s49, s1, s0 // (op 3500) 


s_mul_hi_i32 s0, s16, s10 // (op 3500) 
s_add_i32 s1, s0, s16 // (op 3500) 

s_ashr_i32 s0, s1, 31 // (op 3500) 
s_cmp_eq_i32 s10, 0 // (op 3500) 
s_mov_b32 s18, scc // (op 3500) 
s_cmp_eq_i32 s18, 0 // (op 3500) 
s_cselect_b32 s19, 0, -1 // (op 3500) 

s_and_b32 s18, s47, 31 // (op 3500) 
s_lshl_b32 s72, 1, s18 // (op 3500) 
s_add_i32 s73, s72, s19 // (op 3500) 


s_and_b32 s19, s0, s73 // (op 3500) 


s_add_i32 s0, s1, s19 // (op 3500) 


s_ashr_i32 s1, s0, s18 // (op 3500) 


s_lshl_b32 s0, s47, 24 // (op 3500) 
s_ashr_i32 s18, s0, 31 // (op 3500) 

s_xor_b32 s0, s1, s18 // (op 3500) 

s_sub_i32 s1, s0, s18 // (op 3500) 


s_mul_i32 s0, s44, s50 // (op 3500) 

s_mul_i32 s18, s48, s0 // (op 3500) 

s_mul_i32 s0, s1, s18 // (op 3500) 

s_sub_u32 s18, s16, s0 // (op 3500) 

s_mul_hi_i32 s0, s18, s15 // (op 3500) 
s_add_i32 s19, s0, s18 // (op 3500) 


s_ashr_i32 s0, s19, 31 // (op 3500) 
s_cmp_eq_i32 s15, 0 // (op 3500) 
s_mov_b32 s18, scc // (op 3500) 
s_cmp_eq_i32 s18, 0 // (op 3500) 
s_cselect_b32 s50, 0, -1 // (op 3500) 

s_and_b32 s18, s46, 31 // (op 3500) 
s_lshl_b32 s51, 1, s18 // (op 3500) 
s_add_i32 s72, s51, s50 // (op 3500) 


s_and_b32 s50, s0, s72 // (op 3500) 


s_add_i32 s0, s19, s50 // (op 3500) 


s_ashr_i32 s19, s0, s18 // (op 3500) 


s_lshl_b32 s0, s46, 24 // (op 3500) 
s_ashr_i32 s18, s0, 31 // (op 3500) 

s_xor_b32 s0, s19, s18 // (op 3500) 

s_sub_i32 s19, s0, s18 // (op 3500) 


s_cmp_eq_i32 s1, 0 // (op 3500) 
s_mov_b32 s0, scc // (op 3500) 

s_cmp_eq_i32 s0, 0 // (op 3500) 
s_cselect_b32 s1, s19, s49 // (op 3500) 



v_lshl_add_u32 v15, s1, 8, v12 // (op 3500) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3500) 
s_lshr_b32 s18, s0, 1 // (op 3500) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 3500) 


v_mul_lo_u32 v13, v12, s28 // (op 3500) 
v_add_u32 v13, v13, v15 // (op 3500) 


s_mul_i32 s1, s28, s33 // (op 3502) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3502) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3502) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3502) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3502) most significant: sum
s_add_u32 s1, s1, s19 // (op 3502) most significant: sum


s_mov_b32 s52, s30 // (op 3502) reg expression
s_mov_b32 s53, s31 // (op 3502) reg expression
s_mov_b32 s54, s0 // (op 3502) reg expression
s_mov_b32 s55, 131072 // (op 3502) call()

s_mul_hi_i32 s0, s16, s11 // (op 3564) 
v_lshrrev_b32 v12, 7, v1 // (op 3564) 
s_add_i32 s1, s0, s16 // (op 3564) 

v_and_b32 v15, 15, v1 // (op 3564) 
s_ashr_i32 s0, s1, 31 // (op 3564) 
v_lshl_add_u32 v16, v12, 4, v15 // (op 3564) 


s_cmp_eq_i32 s11, 0 // (op 3564) 
s_mov_b32 s18, scc // (op 3564) 
v_and_b32 v12, 63, v1 // (op 3564) 
s_cmp_eq_i32 s18, 0 // (op 3564) 
s_cselect_b32 s19, 0, -1 // (op 3564) 

v_lshrrev_b32 v15, 4, v12 // (op 3564) 

s_and_b32 s18, s12, 31 // (op 3564) 
s_lshl_b32 s49, 1, s18 // (op 3564) 
s_add_i32 s50, s49, s19 // (op 3564) 


s_and_b32 s19, s0, s50 // (op 3564) 


s_add_i32 s0, s1, s19 // (op 3564) 


s_ashr_i32 s1, s0, s18 // (op 3564) 


s_lshl_b32 s0, s12, 24 // (op 3564) 
s_ashr_i32 s18, s0, 31 // (op 3564) 

s_xor_b32 s0, s1, s18 // (op 3564) 

s_sub_i32 s1, s0, s18 // (op 3564) 


s_mul_hi_i32 s0, s1, s13 // (op 3564) 
s_add_i32 s18, s0, s1 // (op 3564) 

s_ashr_i32 s0, s18, 31 // (op 3564) 
s_cmp_eq_i32 s13, 0 // (op 3564) 
s_mov_b32 s19, scc // (op 3564) 
s_cmp_eq_i32 s19, 0 // (op 3564) 
s_cselect_b32 s49, 0, -1 // (op 3564) 

s_and_b32 s19, s14, 31 // (op 3564) 
s_lshl_b32 s50, 1, s19 // (op 3564) 
s_add_i32 s51, s50, s49 // (op 3564) 


s_and_b32 s49, s0, s51 // (op 3564) 


s_add_i32 s0, s18, s49 // (op 3564) 


s_ashr_i32 s18, s0, s19 // (op 3564) 


s_lshl_b32 s0, s14, 24 // (op 3564) 
s_ashr_i32 s19, s0, 31 // (op 3564) 

s_xor_b32 s0, s18, s19 // (op 3564) 

s_sub_i32 s18, s0, s19 // (op 3564) 


s_mul_i32 s0, s1, s44 // (op 3564) 

s_sub_u32 s1, s16, s0 // (op 3564) 

s_mul_i32 s0, s18, s44 // (op 3564) 
s_add_u32 s0, s0, s1 // (op 3564) 


s_mul_hi_i32 s1, s16, s10 // (op 3564) 
s_add_i32 s18, s1, s16 // (op 3564) 

s_ashr_i32 s1, s18, 31 // (op 3564) 
s_cmp_eq_i32 s10, 0 // (op 3564) 
s_mov_b32 s19, scc // (op 3564) 
s_cmp_eq_i32 s19, 0 // (op 3564) 
s_cselect_b32 s49, 0, -1 // (op 3564) 

s_and_b32 s19, s47, 31 // (op 3564) 
s_lshl_b32 s50, 1, s19 // (op 3564) 
s_add_i32 s51, s50, s49 // (op 3564) 


s_and_b32 s49, s1, s51 // (op 3564) 


s_add_i32 s1, s18, s49 // (op 3564) 


s_ashr_i32 s18, s1, s19 // (op 3564) 


s_lshl_b32 s1, s47, 24 // (op 3564) 
s_ashr_i32 s19, s1, 31 // (op 3564) 

s_xor_b32 s1, s18, s19 // (op 3564) 

s_sub_i32 s18, s1, s19 // (op 3564) 


s_add_u32 s50, s6, 256 // (op 3564) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3564) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3564) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3564) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3564) 

s_mul_i32 s1, s44, s50 // (op 3564) 

s_mul_i32 s19, s48, s1 // (op 3564) 

s_mul_i32 s1, s18, s19 // (op 3564) 

s_sub_u32 s19, s16, s1 // (op 3564) 

s_mov_b32 s1, s19 // (op 3564) 
s_mul_hi_i32 s49, s1, s15 // (op 3564) 
s_add_i32 s50, s49, s1 // (op 3564) 


s_ashr_i32 s1, s50, 31 // (op 3564) 
s_cmp_eq_i32 s15, 0 // (op 3564) 
s_mov_b32 s49, scc // (op 3564) 
s_cmp_eq_i32 s49, 0 // (op 3564) 
s_cselect_b32 s51, 0, -1 // (op 3564) 

s_and_b32 s49, s46, 31 // (op 3564) 
s_lshl_b32 s72, 1, s49 // (op 3564) 
s_add_i32 s73, s72, s51 // (op 3564) 


s_and_b32 s51, s1, s73 // (op 3564) 


s_add_i32 s1, s50, s51 // (op 3564) 


s_ashr_i32 s50, s1, s49 // (op 3564) 


s_lshl_b32 s1, s46, 24 // (op 3564) 
s_ashr_i32 s49, s1, 31 // (op 3564) 

s_xor_b32 s1, s50, s49 // (op 3564) 

s_sub_i32 s50, s1, s49 // (op 3564) 


s_mul_i32 s1, s45, s44 // (op 3564) 
s_add_u32 s72, s8, 16 // (op 3564) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3564) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3564) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3564) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 3564) 

s_sub_i32 s49, s72, s1 // (op 3564) 

s_mul_i32 s51, s50, s49 // (op 3564) 


s_sub_u32 s49, s19, s51 // (op 3564) 


s_add_u32 s19, s49, s1 // (op 3564) 


s_cmp_eq_i32 s18, 0 // (op 3564) 
s_mov_b32 s1, scc // (op 3564) 

s_cmp_eq_i32 s1, 0 // (op 3564) 
s_cselect_b32 s18, s19, s0 // (op 3564) 



v_lshl_add_u32 v12, s18, 4, v16 // (op 3564) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3564) 
s_lshr_b32 s18, s0, 1 // (op 3564) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 3564) 


v_mul_lo_u32 v15, v12, s34 // (op 3564) 
v_add_u32 v15, v15, v16 // (op 3564) 


s_mul_i32 s1, s34, s9 // (op 3566) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3566) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3566) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3566) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3566) most significant: sum
s_add_u32 s1, s1, s19 // (op 3566) most significant: sum


s_mov_b32 s64, s36 // (op 3566) reg expression
s_mov_b32 s65, s37 // (op 3566) reg expression
s_mov_b32 s66, s0 // (op 3566) reg expression
s_mov_b32 s67, 131072 // (op 3566) call()

s_mul_hi_i32 s0, s16, s11 // (op 3628) 
v_and_b32 v12, -32, v1 // (op 3628) 
s_add_i32 s1, s0, s16 // (op 3628) 

v_sub_u32 v16, v1, v12 // (op 3628) 

s_ashr_i32 s0, s1, 31 // (op 3628) 
v_lshlrev_b32 v12, 3, v16 // (op 3628) 

s_cmp_eq_i32 s11, 0 // (op 3628) 
s_mov_b32 s18, scc // (op 3628) 
v_lshrrev_b32 v16, 5, v1 // (op 3628) 
s_cmp_eq_i32 s18, 0 // (op 3628) 
s_cselect_b32 s19, 0, -1 // (op 3628) 

s_and_b32 s18, s12, 31 // (op 3628) 
s_lshl_b32 s49, 1, s18 // (op 3628) 
s_add_i32 s50, s49, s19 // (op 3628) 


s_and_b32 s19, s0, s50 // (op 3628) 


s_add_i32 s0, s1, s19 // (op 3628) 


s_ashr_i32 s1, s0, s18 // (op 3628) 


s_lshl_b32 s0, s12, 24 // (op 3628) 
s_ashr_i32 s18, s0, 31 // (op 3628) 

s_xor_b32 s0, s1, s18 // (op 3628) 

s_sub_i32 s1, s0, s18 // (op 3628) 


s_mul_hi_i32 s0, s1, s13 // (op 3628) 
s_add_i32 s18, s0, s1 // (op 3628) 

s_ashr_i32 s0, s18, 31 // (op 3628) 
s_cmp_eq_i32 s13, 0 // (op 3628) 
s_mov_b32 s19, scc // (op 3628) 
s_cmp_eq_i32 s19, 0 // (op 3628) 
s_cselect_b32 s49, 0, -1 // (op 3628) 

s_and_b32 s19, s14, 31 // (op 3628) 
s_lshl_b32 s50, 1, s19 // (op 3628) 
s_add_i32 s51, s50, s49 // (op 3628) 


s_and_b32 s49, s0, s51 // (op 3628) 


s_add_i32 s0, s18, s49 // (op 3628) 


s_ashr_i32 s18, s0, s19 // (op 3628) 


s_lshl_b32 s0, s14, 24 // (op 3628) 
s_ashr_i32 s19, s0, 31 // (op 3628) 

s_xor_b32 s0, s18, s19 // (op 3628) 

s_sub_i32 s18, s0, s19 // (op 3628) 


s_add_u32 s50, s6, 256 // (op 3628) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3628) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3628) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3628) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3628) 

s_mul_i32 s0, s18, s50 // (op 3628) 

s_sub_i32 s49, s1, s0 // (op 3628) 


s_mul_hi_i32 s0, s16, s10 // (op 3628) 
s_add_i32 s1, s0, s16 // (op 3628) 

s_ashr_i32 s0, s1, 31 // (op 3628) 
s_cmp_eq_i32 s10, 0 // (op 3628) 
s_mov_b32 s18, scc // (op 3628) 
s_cmp_eq_i32 s18, 0 // (op 3628) 
s_cselect_b32 s19, 0, -1 // (op 3628) 

s_and_b32 s18, s47, 31 // (op 3628) 
s_lshl_b32 s72, 1, s18 // (op 3628) 
s_add_i32 s73, s72, s19 // (op 3628) 


s_and_b32 s19, s0, s73 // (op 3628) 


s_add_i32 s0, s1, s19 // (op 3628) 


s_ashr_i32 s1, s0, s18 // (op 3628) 


s_lshl_b32 s0, s47, 24 // (op 3628) 
s_ashr_i32 s18, s0, 31 // (op 3628) 

s_xor_b32 s0, s1, s18 // (op 3628) 

s_sub_i32 s1, s0, s18 // (op 3628) 


s_mul_i32 s0, s44, s50 // (op 3628) 

s_mul_i32 s18, s48, s0 // (op 3628) 

s_mul_i32 s0, s1, s18 // (op 3628) 

s_sub_u32 s18, s16, s0 // (op 3628) 

s_mul_hi_i32 s0, s18, s15 // (op 3628) 
s_add_i32 s19, s0, s18 // (op 3628) 


s_ashr_i32 s0, s19, 31 // (op 3628) 
s_cmp_eq_i32 s15, 0 // (op 3628) 
s_mov_b32 s18, scc // (op 3628) 
s_cmp_eq_i32 s18, 0 // (op 3628) 
s_cselect_b32 s50, 0, -1 // (op 3628) 

s_and_b32 s18, s46, 31 // (op 3628) 
s_lshl_b32 s51, 1, s18 // (op 3628) 
s_add_i32 s72, s51, s50 // (op 3628) 


s_and_b32 s50, s0, s72 // (op 3628) 


s_add_i32 s0, s19, s50 // (op 3628) 


s_ashr_i32 s19, s0, s18 // (op 3628) 


s_lshl_b32 s0, s46, 24 // (op 3628) 
s_ashr_i32 s18, s0, 31 // (op 3628) 

s_xor_b32 s0, s19, s18 // (op 3628) 

s_sub_i32 s19, s0, s18 // (op 3628) 


s_cmp_eq_i32 s1, 0 // (op 3628) 
s_mov_b32 s0, scc // (op 3628) 

s_cmp_eq_i32 s0, 0 // (op 3628) 
s_cselect_b32 s1, s19, s49 // (op 3628) 



v_lshl_add_u32 v17, s1, 8, v12 // (op 3628) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3628) 
s_lshr_b32 s18, s0, 1 // (op 3628) 

v_lshl_add_u32 v12, s18, 3, v16 // (op 3628) 


v_mul_lo_u32 v16, v12, s38 // (op 3628) 
v_add_u32 v16, v16, v17 // (op 3628) 


s_mul_i32 s1, s38, s43 // (op 3630) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3630) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3630) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3630) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3630) most significant: sum
s_add_u32 s1, s1, s19 // (op 3630) most significant: sum


s_mov_b32 s68, s40 // (op 3630) reg expression
s_mov_b32 s69, s41 // (op 3630) reg expression
s_mov_b32 s70, s0 // (op 3630) reg expression
s_mov_b32 s71, 131072 // (op 3630) call()

v_and_b32 v12, -8, v1 // (op 3721) 
v_sub_u32 v17, v1, v12 // (op 3721) 

v_lshlrev_b32 v12, 4, v17 // (op 3721) 

v_lshrrev_b32 v17, 3, v1 // (op 3721) 
v_lshl_add_u32 v18, v17, 7, v12 // (op 3721) 


v_lshlrev_b32 v12, 2, v18 // (op 3721) 

v_lshrrev_b32 v17, 3, v12 // (op 3721) 

v_and_b32 v12, -16, v1 // (op 3774) 
v_sub_u32 v18, v1, v12 // (op 3774) 

v_lshlrev_b32 v12, 4, v18 // (op 3774) 

v_lshrrev_b32 v18, 4, v1 // (op 3774) 
v_lshl_add_u32 v19, v18, 12, v12 // (op 3774) 


v_and_b32 v12, -32, v1 // (op 3832) 
v_sub_u32 v18, v1, v12 // (op 3832) 

v_lshlrev_b32 v12, 3, v18 // (op 3832) 

v_lshrrev_b32 v18, 5, v1 // (op 3832) 
v_lshl_add_u32 v20, v18, 8, v12 // (op 3832) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1253) 
s_and_b32 s18, s0, 4294967294 // (op 1253) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1252) 
s_mov_b32 s19, s0 // (op 1252) convert

s_cmp_lt_u32 s18, s19 // (op 1252) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1252 // (op 1252) Condition: Top (jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1252 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1252:


s_add_u32 s18, s18, 1 // (op 1254) 
v_mov_b32 v12, v14 // (op 170) 

buffer_load_dwordx2 v[22:23], v12, s[20:23], 0 offen // (op 170) Load value

v_add_u32 v14, 64, v14 // (op 3416) 

s_barrier  // (op 4030) 
v_mov_b64 v[24:25], s[28:29] // (op 183) convert




v_mov_b32 v12, v13 // (op 183) 
v_mov_b32 v18, v12 // (op 183) reg expression
buffer_load_dwordx4 v[26:29], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[30:33], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[22:23] offset:33792 // (op 171) Store local data 
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 183) Load value

v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[42:45], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[46:49], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[50:53], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[54:57], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[58:61], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[62:65], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[66:69], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[70:73], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[74:77], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[78:81], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[82:85], v18, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v18, v18, v24 // (op 183) 
buffer_load_dwordx4 v[86:89], v18, s[52:55], 0 offen // (op 183) Load value



s_lshl_b32 s19, s28, 7 // (op 3492) 
v_add_u32 v13, v13, s19 // (op 3492) 



s_barrier  // (op 4043) 
v_mov_b32 v12, v16 // (op 184) 



v_mov_b32 v18, v12 // (op 184) reg expression

buffer_load_dwordx2 v[24:25], v18, s[68:71], 0 offen // (op 184) Load value


s_lshl_b32 s19, s38, 2 // (op 3620) 
v_add_u32 v16, v16, s19 // (op 3620) 

s_waitcnt vmcnt(16)
ds_write_b128 v19, v[26:29] // (op 181) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4033) 
s_waitcnt vmcnt(15)
ds_write_b128 v19, v[30:33] offset:256 // (op 181) Store local data 



ds_read_b128 v[90:93], v3 offset:33792 // (op 173) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v19, v[34:37] offset:512 // (op 181) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v19, v[38:41] offset:768 // (op 181) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v19, v[42:45] offset:1024 // (op 181) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v19, v[46:49] offset:1280 // (op 181) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v19, v[50:53] offset:1536 // (op 181) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v19, v[54:57] offset:1792 // (op 181) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v19, v[58:61] offset:2048 // (op 181) Store local data 
s_waitcnt vmcnt(7)
ds_write_b128 v19, v[62:65] offset:2304 // (op 181) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v19, v[66:69] offset:2560 // (op 181) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v19, v[70:73] offset:2816 // (op 181) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v19, v[74:77] offset:3072 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v19, v[78:81] offset:3328 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[82:85] offset:3584 // (op 181) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v19, v[86:89] offset:3840 // (op 181) Store local data 

s_barrier  // (op 4057) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4047) 






ds_read_b64_tr_b8 v[26:27], v7 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v7 offset:2048 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:16384 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:18432 // (op 175) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b64 v20, v[24:25] offset:32768 // (op 182) Store local data 



v_mov_b32 v12, v15 // (op 177) 

buffer_load_ubyte v18, v12, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v15, 4, v15 // (op 3556) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4061) 


ds_read_u8 v12, v0 offset:32768 // (op 179) Load local data 



ds_read_b64_tr_b8 v[34:35], v7 offset:32 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:2080 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[26:33], a[0:3], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 180) 
ds_read_b64_tr_b8 v[38:39], v7 offset:16416 // (op 324) Transpose load local data 


ds_read_b64_tr_b8 v[40:41], v7 offset:18464 // (op 324) Transpose load local data 


ds_read_u8 v12, v0 offset:32800 // (op 327) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[34:41], a[4:7], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 328) 
ds_read_b64_tr_b8 v[26:27], v7 offset:64 // (op 480) Transpose load local data 


ds_read_b64_tr_b8 v[28:29], v7 offset:2112 // (op 480) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:16448 // (op 480) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:18496 // (op 480) Transpose load local data 


ds_read_u8 v12, v0 offset:32832 // (op 483) Load local data 



ds_read_b64_tr_b8 v[34:35], v7 offset:96 // (op 636) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[26:33], a[8:11], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 484) 
ds_read_b64_tr_b8 v[36:37], v7 offset:2144 // (op 636) Transpose load local data 


ds_read_b64_tr_b8 v[38:39], v7 offset:16480 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:18528 // (op 636) Transpose load local data 


ds_read_u8 v12, v0 offset:32864 // (op 639) Load local data 



ds_read_b64_tr_b8 v[26:27], v7 offset:128 // (op 792) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v7 offset:2176 // (op 792) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[34:41], a[12:15], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 640) 
ds_read_b64_tr_b8 v[30:31], v7 offset:16512 // (op 792) Transpose load local data 


ds_read_b64_tr_b8 v[32:33], v7 offset:18560 // (op 792) Transpose load local data 


ds_read_u8 v12, v0 offset:32896 // (op 795) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:93], v[26:33], a[16:19], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 796) 
ds_read_b64_tr_b8 v[34:35], v7 offset:160 // (op 948) Transpose load local data 


ds_read_b64_tr_b8 v[36:37], v7 offset:2208 // (op 948) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:16544 // (op 948) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:18592 // (op 948) Transpose load local data 


ds_read_u8 v12, v0 offset:32928 // (op 951) Load local data 



ds_read_b64_tr_b8 v[26:27], v7 offset:192 // (op 1104) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:93], v[34:41], a[20:23], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 952) 
ds_read_b64_tr_b8 v[28:29], v7 offset:2240 // (op 1104) Transpose load local data 


ds_read_b64_tr_b8 v[30:31], v7 offset:16576 // (op 1104) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:18624 // (op 1104) Transpose load local data 


ds_read_u8 v12, v0 offset:32960 // (op 1107) Load local data 



ds_read_b64_tr_b8 v[34:35], v7 offset:224 // (op 1260) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:2272 // (op 1260) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:93], v[26:33], a[24:27], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1108) 
ds_read_b64_tr_b8 v[38:39], v7 offset:16608 // (op 1260) Transpose load local data 


ds_read_b64_tr_b8 v[40:41], v7 offset:18656 // (op 1260) Transpose load local data 


ds_read_u8 v12, v0 offset:32992 // (op 1263) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:93], v[34:41], a[28:31], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1264) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 1252) 
s_mov_b32 s19, s0 // (op 1252) convert

s_cmp_lt_u32 s18, s19 // (op 1252) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1252 // (op 1252) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1252 if true)
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1252:






















s_barrier  // (op 4179) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v10, a4 // (op 1576) 
v_accvgpr_read v11, a5 // (op 1576) 
v_accvgpr_read v12, a6 // (op 1576) 
v_accvgpr_read v13, a7 // (op 1576) 
v_mul_f32 v14, v8, v10 // (op 1576) 
v_mul_f32 v15, v8, v11 // (op 1576) 
v_mul_f32 v16, v8, v12 // (op 1576) 
v_mul_f32 v17, v8, v13 // (op 1576) 

v_lshrrev_b32 v10, 16, v14 // (op 1578) 
v_lshrrev_b32 v11, 16, v15 // (op 1578) 
v_lshrrev_b32 v12, 16, v16 // (op 1578) 
v_lshrrev_b32 v13, 16, v17 // (op 1578) 


v_pack_b32_f16 v14, v10, v11 // (op 1579) 
v_pack_b32_f16 v15, v12, v13 // (op 1579) 

s_lshl_b32 s0, s4, 6 // (op 1579) 
v_add_u32 v0, v9, s0 // (op 1579) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1579) Store value



v_accvgpr_read v10, a8 // (op 1590) 
v_accvgpr_read v11, a9 // (op 1590) 
v_accvgpr_read v12, a10 // (op 1590) 
v_accvgpr_read v13, a11 // (op 1590) 
v_mul_f32 v14, v8, v10 // (op 1590) 
v_mul_f32 v15, v8, v11 // (op 1590) 
v_mul_f32 v16, v8, v12 // (op 1590) 
v_mul_f32 v17, v8, v13 // (op 1590) 

v_lshrrev_b32 v10, 16, v14 // (op 1592) 
v_lshrrev_b32 v11, 16, v15 // (op 1592) 
v_lshrrev_b32 v12, 16, v16 // (op 1592) 
v_lshrrev_b32 v13, 16, v17 // (op 1592) 


v_pack_b32_f16 v14, v10, v11 // (op 1593) 
v_pack_b32_f16 v15, v12, v13 // (op 1593) 

s_lshl_b32 s0, s4, 7 // (op 1593) 
v_add_u32 v0, v9, s0 // (op 1593) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1593) Store value



v_accvgpr_read v10, a12 // (op 1604) 
v_accvgpr_read v11, a13 // (op 1604) 
v_accvgpr_read v12, a14 // (op 1604) 
v_accvgpr_read v13, a15 // (op 1604) 
v_mul_f32 v14, v8, v10 // (op 1604) 
v_mul_f32 v15, v8, v11 // (op 1604) 
v_mul_f32 v16, v8, v12 // (op 1604) 
v_mul_f32 v17, v8, v13 // (op 1604) 

v_lshrrev_b32 v10, 16, v14 // (op 1606) 
v_lshrrev_b32 v11, 16, v15 // (op 1606) 
v_lshrrev_b32 v12, 16, v16 // (op 1606) 
v_lshrrev_b32 v13, 16, v17 // (op 1606) 


v_pack_b32_f16 v14, v10, v11 // (op 1607) 
v_pack_b32_f16 v15, v12, v13 // (op 1607) 

s_lshl_b32 s0, s4, 6 // (op 1607) 
s_mul_i32 s1, 3, s0 // (op 1607) 

v_add_u32 v0, v9, s1 // (op 1607) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1607) Store value



v_accvgpr_read v10, a16 // (op 1618) 
v_accvgpr_read v11, a17 // (op 1618) 
v_accvgpr_read v12, a18 // (op 1618) 
v_accvgpr_read v13, a19 // (op 1618) 
v_mul_f32 v14, v8, v10 // (op 1618) 
v_mul_f32 v15, v8, v11 // (op 1618) 
v_mul_f32 v16, v8, v12 // (op 1618) 
v_mul_f32 v17, v8, v13 // (op 1618) 

v_lshrrev_b32 v10, 16, v14 // (op 1620) 
v_lshrrev_b32 v11, 16, v15 // (op 1620) 
v_lshrrev_b32 v12, 16, v16 // (op 1620) 
v_lshrrev_b32 v13, 16, v17 // (op 1620) 


v_pack_b32_f16 v14, v10, v11 // (op 1621) 
v_pack_b32_f16 v15, v12, v13 // (op 1621) 

s_lshl_b32 s0, s4, 8 // (op 1621) 
v_add_u32 v0, v9, s0 // (op 1621) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1621) Store value



v_accvgpr_read v10, a20 // (op 1632) 
v_accvgpr_read v11, a21 // (op 1632) 
v_accvgpr_read v12, a22 // (op 1632) 
v_accvgpr_read v13, a23 // (op 1632) 
v_mul_f32 v14, v8, v10 // (op 1632) 
v_mul_f32 v15, v8, v11 // (op 1632) 
v_mul_f32 v16, v8, v12 // (op 1632) 
v_mul_f32 v17, v8, v13 // (op 1632) 

v_lshrrev_b32 v10, 16, v14 // (op 1634) 
v_lshrrev_b32 v11, 16, v15 // (op 1634) 
v_lshrrev_b32 v12, 16, v16 // (op 1634) 
v_lshrrev_b32 v13, 16, v17 // (op 1634) 


v_pack_b32_f16 v14, v10, v11 // (op 1635) 
v_pack_b32_f16 v15, v12, v13 // (op 1635) 

s_lshl_b32 s0, s4, 6 // (op 1635) 
s_mul_i32 s1, 5, s0 // (op 1635) 

v_add_u32 v0, v9, s1 // (op 1635) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1635) Store value



v_accvgpr_read v10, a24 // (op 1646) 
v_accvgpr_read v11, a25 // (op 1646) 
v_accvgpr_read v12, a26 // (op 1646) 
v_accvgpr_read v13, a27 // (op 1646) 
v_mul_f32 v14, v8, v10 // (op 1646) 
v_mul_f32 v15, v8, v11 // (op 1646) 
v_mul_f32 v16, v8, v12 // (op 1646) 
v_mul_f32 v17, v8, v13 // (op 1646) 

v_lshrrev_b32 v10, 16, v14 // (op 1648) 
v_lshrrev_b32 v11, 16, v15 // (op 1648) 
v_lshrrev_b32 v12, 16, v16 // (op 1648) 
v_lshrrev_b32 v13, 16, v17 // (op 1648) 


v_pack_b32_f16 v14, v10, v11 // (op 1649) 
v_pack_b32_f16 v15, v12, v13 // (op 1649) 

s_lshl_b32 s0, s4, 6 // (op 1649) 
s_mul_i32 s1, 6, s0 // (op 1649) 

v_add_u32 v0, v9, s1 // (op 1649) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1649) Store value



v_accvgpr_read v10, a28 // (op 1660) 
v_accvgpr_read v11, a29 // (op 1660) 
v_accvgpr_read v12, a30 // (op 1660) 
v_accvgpr_read v13, a31 // (op 1660) 
v_mul_f32 v14, v8, v10 // (op 1660) 
v_mul_f32 v15, v8, v11 // (op 1660) 
v_mul_f32 v16, v8, v12 // (op 1660) 
v_mul_f32 v17, v8, v13 // (op 1660) 

v_lshrrev_b32 v10, 16, v14 // (op 1662) 
v_lshrrev_b32 v11, 16, v15 // (op 1662) 
v_lshrrev_b32 v12, 16, v16 // (op 1662) 
v_lshrrev_b32 v13, 16, v17 // (op 1662) 


v_pack_b32_f16 v14, v10, v11 // (op 1663) 
v_pack_b32_f16 v15, v12, v13 // (op 1663) 

s_lshl_b32 s0, s4, 6 // (op 1663) 
s_mul_i32 s1, 7, s0 // (op 1663) 

v_add_u32 v0, v9, s1 // (op 1663) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1663) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_and_b32 v12, 65535, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 1717) 
v_add_u32 v0, v6, s0 // (op 1717) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1717) Load value

v_accvgpr_read v12, a4 // (op 1713) 
v_accvgpr_read v13, a5 // (op 1713) 
v_accvgpr_read v14, a6 // (op 1713) 
v_accvgpr_read v15, a7 // (op 1713) 
v_mul_f32 v16, v8, v12 // (op 1713) 
v_mul_f32 v17, v8, v13 // (op 1713) 
v_mul_f32 v18, v8, v14 // (op 1713) 
v_mul_f32 v19, v8, v15 // (op 1713) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1718) 
v_lshrrev_b32 v13, 16, v10 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v20, v5, v12 // (op 1718) 
v_mul_f32 v21, v5, v13 // (op 1718) 
v_and_b32 v12, 65535, v11 // (op 1718) 
v_lshrrev_b32 v13, 16, v11 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v22, v5, v12 // (op 1718) 
v_mul_f32 v23, v5, v13 // (op 1718) 


v_add_f32 v10, v20, v16 // (op 1714) 
v_add_f32 v11, v21, v17 // (op 1714) 
v_add_f32 v12, v22, v18 // (op 1714) 
v_add_f32 v13, v23, v19 // (op 1714) 


v_lshrrev_b32 v14, 16, v10 // (op 1715) 
v_lshrrev_b32 v15, 16, v11 // (op 1715) 
v_lshrrev_b32 v16, 16, v12 // (op 1715) 
v_lshrrev_b32 v17, 16, v13 // (op 1715) 


v_pack_b32_f16 v10, v14, v15 // (op 1716) 
v_pack_b32_f16 v11, v16, v17 // (op 1716) 

s_lshl_b32 s0, s4, 6 // (op 1716) 
v_add_u32 v0, v9, s0 // (op 1716) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1716) Store value





s_lshl_b32 s0, s2, 7 // (op 1731) 
v_add_u32 v0, v6, s0 // (op 1731) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1731) Load value

v_accvgpr_read v12, a8 // (op 1727) 
v_accvgpr_read v13, a9 // (op 1727) 
v_accvgpr_read v14, a10 // (op 1727) 
v_accvgpr_read v15, a11 // (op 1727) 
v_mul_f32 v16, v8, v12 // (op 1727) 
v_mul_f32 v17, v8, v13 // (op 1727) 
v_mul_f32 v18, v8, v14 // (op 1727) 
v_mul_f32 v19, v8, v15 // (op 1727) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1732) 
v_lshrrev_b32 v13, 16, v10 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v20, v5, v12 // (op 1732) 
v_mul_f32 v21, v5, v13 // (op 1732) 
v_and_b32 v12, 65535, v11 // (op 1732) 
v_lshrrev_b32 v13, 16, v11 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v22, v5, v12 // (op 1732) 
v_mul_f32 v23, v5, v13 // (op 1732) 


v_add_f32 v10, v20, v16 // (op 1728) 
v_add_f32 v11, v21, v17 // (op 1728) 
v_add_f32 v12, v22, v18 // (op 1728) 
v_add_f32 v13, v23, v19 // (op 1728) 


v_lshrrev_b32 v14, 16, v10 // (op 1729) 
v_lshrrev_b32 v15, 16, v11 // (op 1729) 
v_lshrrev_b32 v16, 16, v12 // (op 1729) 
v_lshrrev_b32 v17, 16, v13 // (op 1729) 


v_pack_b32_f16 v10, v14, v15 // (op 1730) 
v_pack_b32_f16 v11, v16, v17 // (op 1730) 

s_lshl_b32 s0, s4, 7 // (op 1730) 
v_add_u32 v0, v9, s0 // (op 1730) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1730) Store value





s_lshl_b32 s0, s2, 6 // (op 1745) 
s_mul_i32 s1, 3, s0 // (op 1745) 

v_add_u32 v0, v6, s1 // (op 1745) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1745) Load value

v_accvgpr_read v12, a12 // (op 1741) 
v_accvgpr_read v13, a13 // (op 1741) 
v_accvgpr_read v14, a14 // (op 1741) 
v_accvgpr_read v15, a15 // (op 1741) 
v_mul_f32 v16, v8, v12 // (op 1741) 
v_mul_f32 v17, v8, v13 // (op 1741) 
v_mul_f32 v18, v8, v14 // (op 1741) 
v_mul_f32 v19, v8, v15 // (op 1741) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1746) 
v_lshrrev_b32 v13, 16, v10 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v20, v5, v12 // (op 1746) 
v_mul_f32 v21, v5, v13 // (op 1746) 
v_and_b32 v12, 65535, v11 // (op 1746) 
v_lshrrev_b32 v13, 16, v11 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v22, v5, v12 // (op 1746) 
v_mul_f32 v23, v5, v13 // (op 1746) 


v_add_f32 v10, v20, v16 // (op 1742) 
v_add_f32 v11, v21, v17 // (op 1742) 
v_add_f32 v12, v22, v18 // (op 1742) 
v_add_f32 v13, v23, v19 // (op 1742) 


v_lshrrev_b32 v14, 16, v10 // (op 1743) 
v_lshrrev_b32 v15, 16, v11 // (op 1743) 
v_lshrrev_b32 v16, 16, v12 // (op 1743) 
v_lshrrev_b32 v17, 16, v13 // (op 1743) 


v_pack_b32_f16 v10, v14, v15 // (op 1744) 
v_pack_b32_f16 v11, v16, v17 // (op 1744) 

s_lshl_b32 s0, s4, 6 // (op 1744) 
s_mul_i32 s1, 3, s0 // (op 1744) 

v_add_u32 v0, v9, s1 // (op 1744) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1744) Store value





s_lshl_b32 s0, s2, 8 // (op 1759) 
v_add_u32 v0, v6, s0 // (op 1759) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1759) Load value

v_accvgpr_read v12, a16 // (op 1755) 
v_accvgpr_read v13, a17 // (op 1755) 
v_accvgpr_read v14, a18 // (op 1755) 
v_accvgpr_read v15, a19 // (op 1755) 
v_mul_f32 v16, v8, v12 // (op 1755) 
v_mul_f32 v17, v8, v13 // (op 1755) 
v_mul_f32 v18, v8, v14 // (op 1755) 
v_mul_f32 v19, v8, v15 // (op 1755) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1760) 
v_lshrrev_b32 v13, 16, v10 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v20, v5, v12 // (op 1760) 
v_mul_f32 v21, v5, v13 // (op 1760) 
v_and_b32 v12, 65535, v11 // (op 1760) 
v_lshrrev_b32 v13, 16, v11 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v22, v5, v12 // (op 1760) 
v_mul_f32 v23, v5, v13 // (op 1760) 


v_add_f32 v10, v20, v16 // (op 1756) 
v_add_f32 v11, v21, v17 // (op 1756) 
v_add_f32 v12, v22, v18 // (op 1756) 
v_add_f32 v13, v23, v19 // (op 1756) 


v_lshrrev_b32 v14, 16, v10 // (op 1757) 
v_lshrrev_b32 v15, 16, v11 // (op 1757) 
v_lshrrev_b32 v16, 16, v12 // (op 1757) 
v_lshrrev_b32 v17, 16, v13 // (op 1757) 


v_pack_b32_f16 v10, v14, v15 // (op 1758) 
v_pack_b32_f16 v11, v16, v17 // (op 1758) 

s_lshl_b32 s0, s4, 8 // (op 1758) 
v_add_u32 v0, v9, s0 // (op 1758) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1758) Store value





s_lshl_b32 s0, s2, 6 // (op 1773) 
s_mul_i32 s1, 5, s0 // (op 1773) 

v_add_u32 v0, v6, s1 // (op 1773) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1773) Load value

v_accvgpr_read v12, a20 // (op 1769) 
v_accvgpr_read v13, a21 // (op 1769) 
v_accvgpr_read v14, a22 // (op 1769) 
v_accvgpr_read v15, a23 // (op 1769) 
v_mul_f32 v16, v8, v12 // (op 1769) 
v_mul_f32 v17, v8, v13 // (op 1769) 
v_mul_f32 v18, v8, v14 // (op 1769) 
v_mul_f32 v19, v8, v15 // (op 1769) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1774) 
v_lshrrev_b32 v13, 16, v10 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v20, v5, v12 // (op 1774) 
v_mul_f32 v21, v5, v13 // (op 1774) 
v_and_b32 v12, 65535, v11 // (op 1774) 
v_lshrrev_b32 v13, 16, v11 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v22, v5, v12 // (op 1774) 
v_mul_f32 v23, v5, v13 // (op 1774) 


v_add_f32 v10, v20, v16 // (op 1770) 
v_add_f32 v11, v21, v17 // (op 1770) 
v_add_f32 v12, v22, v18 // (op 1770) 
v_add_f32 v13, v23, v19 // (op 1770) 


v_lshrrev_b32 v14, 16, v10 // (op 1771) 
v_lshrrev_b32 v15, 16, v11 // (op 1771) 
v_lshrrev_b32 v16, 16, v12 // (op 1771) 
v_lshrrev_b32 v17, 16, v13 // (op 1771) 


v_pack_b32_f16 v10, v14, v15 // (op 1772) 
v_pack_b32_f16 v11, v16, v17 // (op 1772) 

s_lshl_b32 s0, s4, 6 // (op 1772) 
s_mul_i32 s1, 5, s0 // (op 1772) 

v_add_u32 v0, v9, s1 // (op 1772) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1772) Store value





s_lshl_b32 s0, s2, 6 // (op 1787) 
s_mul_i32 s1, 6, s0 // (op 1787) 

v_add_u32 v0, v6, s1 // (op 1787) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1787) Load value

v_accvgpr_read v12, a24 // (op 1783) 
v_accvgpr_read v13, a25 // (op 1783) 
v_accvgpr_read v14, a26 // (op 1783) 
v_accvgpr_read v15, a27 // (op 1783) 
v_mul_f32 v16, v8, v12 // (op 1783) 
v_mul_f32 v17, v8, v13 // (op 1783) 
v_mul_f32 v18, v8, v14 // (op 1783) 
v_mul_f32 v19, v8, v15 // (op 1783) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1788) 
v_lshrrev_b32 v13, 16, v10 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v20, v5, v12 // (op 1788) 
v_mul_f32 v21, v5, v13 // (op 1788) 
v_and_b32 v12, 65535, v11 // (op 1788) 
v_lshrrev_b32 v13, 16, v11 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v22, v5, v12 // (op 1788) 
v_mul_f32 v23, v5, v13 // (op 1788) 


v_add_f32 v10, v20, v16 // (op 1784) 
v_add_f32 v11, v21, v17 // (op 1784) 
v_add_f32 v12, v22, v18 // (op 1784) 
v_add_f32 v13, v23, v19 // (op 1784) 


v_lshrrev_b32 v14, 16, v10 // (op 1785) 
v_lshrrev_b32 v15, 16, v11 // (op 1785) 
v_lshrrev_b32 v16, 16, v12 // (op 1785) 
v_lshrrev_b32 v17, 16, v13 // (op 1785) 


v_pack_b32_f16 v10, v14, v15 // (op 1786) 
v_pack_b32_f16 v11, v16, v17 // (op 1786) 

s_lshl_b32 s0, s4, 6 // (op 1786) 
s_mul_i32 s1, 6, s0 // (op 1786) 

v_add_u32 v0, v9, s1 // (op 1786) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1786) Store value





s_lshl_b32 s0, s2, 6 // (op 1801) 
s_mul_i32 s1, 7, s0 // (op 1801) 

v_add_u32 v0, v6, s1 // (op 1801) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1801) Load value

v_accvgpr_read v12, a28 // (op 1797) 
v_accvgpr_read v13, a29 // (op 1797) 
v_accvgpr_read v14, a30 // (op 1797) 
v_accvgpr_read v15, a31 // (op 1797) 
v_mul_f32 v16, v8, v12 // (op 1797) 
v_mul_f32 v17, v8, v13 // (op 1797) 
v_mul_f32 v18, v8, v14 // (op 1797) 
v_mul_f32 v19, v8, v15 // (op 1797) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1802) 
v_lshrrev_b32 v13, 16, v10 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v20, v5, v12 // (op 1802) 
v_mul_f32 v21, v5, v13 // (op 1802) 
v_and_b32 v12, 65535, v11 // (op 1802) 
v_lshrrev_b32 v13, 16, v11 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v22, v5, v12 // (op 1802) 
v_mul_f32 v23, v5, v13 // (op 1802) 


v_add_f32 v10, v20, v16 // (op 1798) 
v_add_f32 v11, v21, v17 // (op 1798) 
v_add_f32 v12, v22, v18 // (op 1798) 
v_add_f32 v13, v23, v19 // (op 1798) 


v_lshrrev_b32 v14, 16, v10 // (op 1799) 
v_lshrrev_b32 v15, 16, v11 // (op 1799) 
v_lshrrev_b32 v16, 16, v12 // (op 1799) 
v_lshrrev_b32 v17, 16, v13 // (op 1799) 


v_pack_b32_f16 v10, v14, v15 // (op 1800) 
v_pack_b32_f16 v11, v16, v17 // (op 1800) 

s_lshl_b32 s0, s4, 6 // (op 1800) 
s_mul_i32 s1, 7, s0 // (op 1800) 

v_add_u32 v0, v9, s1 // (op 1800) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1800) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.LRR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__end:
.size RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_, .LRR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__end-RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 220
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 69632 // lds bytes
  .amdhsa_accum_offset 188
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
    macrotile: [16, 256, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 69632
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 186
    .agpr_count: 32
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

