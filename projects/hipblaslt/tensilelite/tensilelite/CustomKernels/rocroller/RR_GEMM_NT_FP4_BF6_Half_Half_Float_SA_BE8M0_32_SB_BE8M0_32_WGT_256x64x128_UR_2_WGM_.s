// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_,@function
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2744) 
s_mul_hi_i32 s1, s16, s10 // (op 2986) 
v_and_b32 v0, 63, v1 // (op 2859) 
v_and_b32 v3, 48, v1 // (op 2859) 
v_sub_u32 v4, v0, v3 // (op 2859) 

v_lshrrev_b32 v3, 5, v0 // (op 2859) 
v_lshl_add_u32 v5, v3, 5, v4 // (op 2859) 


v_lshrrev_b32 v3, 4, v0 // (op 2859) 

v_and_b32 v0, -2, v3 // (op 2859) 
v_sub_u32 v4, v3, v0 // (op 2859) 


v_lshlrev_b32 v0, 4, v4 // (op 2859) 

v_lshrrev_b32 v3, 6, v1 // (op 2859) 
v_and_b32 v4, 1, v3 // (op 2859) 

v_lshl_add_u32 v3, v4, 5, v0 // (op 2859) 


v_lshl_add_u32 v0, v5, 6, v3 // (op 2859) 


v_mul_lo_u32 v3, v0, 6 // (op 2859) 
v_lshrrev_b32 v4, 3, v3 // (op 2859) 

v_lshrrev_b32 v3, 4, v0 // (op 2859) 

v_lshlrev_b32 v0, 2, v3 // (op 2859) 

v_add_u32 v3, v4, v0 // (op 2859) 


v_lshrrev_b32 v0, 6, v1 // (op 2970) 
v_and_b32 v4, 1, v0 // (op 2970) 

v_and_b32 v0, 31, v1 // (op 2970) 
v_lshl_add_u32 v5, v4, 5, v0 // (op 2970) 


v_and_b32 v0, 63, v1 // (op 2970) 
v_lshrrev_b32 v4, 5, v0 // (op 2970) 

v_lshl_add_u32 v0, v4, 6, v5 // (op 2970) 


v_and_b32 v4, 63, v1 // (op 3041) 
v_and_b32 v5, 48, v1 // (op 3041) 
v_sub_u32 v6, v4, v5 // (op 3041) 

v_lshrrev_b32 v5, 5, v4 // (op 3041) 
v_lshl_add_u32 v7, v5, 5, v6 // (op 3041) 


v_lshrrev_b32 v5, 4, v4 // (op 3041) 

v_and_b32 v4, -2, v5 // (op 3041) 
v_sub_u32 v6, v5, v4 // (op 3041) 


v_lshlrev_b32 v4, 4, v6 // (op 3041) 

v_lshrrev_b32 v5, 6, v1 // (op 3041) 
v_and_b32 v6, 1, v5 // (op 3041) 

v_lshl_add_u32 v5, v6, 5, v4 // (op 3041) 


v_lshl_add_u32 v4, v7, 6, v5 // (op 3041) 


v_mul_lo_u32 v5, v4, 6 // (op 3041) 
v_lshrrev_b32 v6, 3, v5 // (op 3041) 

v_lshrrev_b32 v5, 4, v4 // (op 3041) 

v_lshlrev_b32 v4, 2, v5 // (op 3041) 

v_add_u32 v5, v6, v4 // (op 3041) 


v_lshrrev_b32 v4, 6, v1 // (op 3094) 
v_and_b32 v6, 1, v4 // (op 3094) 

v_and_b32 v4, 31, v1 // (op 3094) 
v_lshl_add_u32 v7, v6, 5, v4 // (op 3094) 


v_and_b32 v4, 63, v1 // (op 3094) 
v_lshrrev_b32 v6, 5, v4 // (op 3094) 

v_lshl_add_u32 v4, v6, 6, v7 // (op 3094) 


v_lshrrev_b32 v6, 7, v1 // (op 2744) 
s_add_i32 s18, s0, s16 // (op 2744) 

v_and_b32 v7, 63, v1 // (op 2744) 
s_ashr_i32 s0, s18, 31 // (op 2744) 
v_lshrrev_b32 v8, 5, v7 // (op 2744) 
s_cmp_eq_i32 s10, 0 // (op 2744) 
s_mov_b32 s19, scc // (op 2744) 
v_lshlrev_b32 v9, 2, v8 // (op 2744) 

s_cmp_eq_i32 s19, 0 // (op 2744) 
s_cselect_b32 s51, 0, -1 // (op 2744) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2744) 


s_and_b32 s19, s13, 31 // (op 2744) 
v_lshrrev_b32 v6, 2, v7 // (op 2744) 

v_and_b32 v7, 7, v6 // (op 2744) 

v_and_b32 v6, 63, v1 // (op 2806) 
v_and_b32 v9, 48, v1 // (op 2806) 
v_sub_u32 v10, v6, v9 // (op 2806) 

v_lshrrev_b32 v9, 5, v6 // (op 2806) 
v_lshl_add_u32 v11, v9, 5, v10 // (op 2806) 


v_lshrrev_b32 v9, 4, v6 // (op 2806) 

v_and_b32 v6, -2, v9 // (op 2806) 
v_sub_u32 v10, v9, v6 // (op 2806) 


v_lshlrev_b32 v6, 4, v10 // (op 2806) 

v_lshrrev_b32 v9, 7, v1 // (op 2806) 
v_lshl_add_u32 v10, v9, 5, v6 // (op 2806) 


v_lshl_add_u32 v6, v11, 8, v10 // (op 2806) 


v_lshlrev_b32 v9, 2, v6 // (op 2806) 

v_lshrrev_b32 v6, 3, v9 // (op 2806) 

v_lshrrev_b32 v9, 7, v1 // (op 2917) 
v_and_b32 v10, 31, v1 // (op 2917) 
v_lshl_add_u32 v11, v9, 5, v10 // (op 2917) 


v_and_b32 v9, 63, v1 // (op 2917) 
v_lshrrev_b32 v10, 5, v9 // (op 2917) 

v_lshl_add_u32 v9, v10, 8, v11 // (op 2917) 


v_lshrrev_b32 v10, 7, v1 // (op 2986) 
v_and_b32 v11, 63, v1 // (op 2986) 
v_and_b32 v12, 63, v1 // (op 3017) 
v_and_b32 v13, 48, v1 // (op 3017) 
v_sub_u32 v14, v12, v13 // (op 3017) 

v_lshrrev_b32 v13, 5, v12 // (op 3017) 
v_lshl_add_u32 v15, v13, 5, v14 // (op 3017) 


v_lshrrev_b32 v13, 4, v12 // (op 3017) 

v_and_b32 v12, -2, v13 // (op 3017) 
v_sub_u32 v14, v13, v12 // (op 3017) 


v_lshlrev_b32 v12, 4, v14 // (op 3017) 

v_lshrrev_b32 v13, 7, v1 // (op 3017) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 3017) 


v_lshl_add_u32 v12, v15, 8, v14 // (op 3017) 


v_lshlrev_b32 v13, 2, v12 // (op 3017) 

v_lshrrev_b32 v12, 3, v13 // (op 3017) 

v_lshrrev_b32 v13, 7, v1 // (op 3070) 
v_and_b32 v14, 31, v1 // (op 3070) 
v_lshl_add_u32 v15, v13, 5, v14 // (op 3070) 


v_and_b32 v13, 63, v1 // (op 3070) 
v_lshrrev_b32 v14, 5, v13 // (op 3070) 

v_lshl_add_u32 v13, v14, 8, v15 // (op 3070) 


s_lshl_b32 s52, 1, s19 // (op 2744) 
v_and_b32 v14, 3, v1 // (op 2744) 
s_add_i32 s53, s52, s51 // (op 2744) 


s_and_b32 s51, s0, s53 // (op 2744) 


s_add_i32 s0, s1, s16 // (op 2986) 

s_ashr_i32 s1, s0, 31 // (op 2986) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 2744) 


s_add_i32 s52, s18, s51 // (op 2744) 


v_lshrrev_b32 v7, 6, v1 // (op 2744) 
s_ashr_i32 s18, s52, s19 // (op 2744) 


v_and_b32 v14, 1, v7 // (op 2744) 

s_lshl_b32 s19, s13, 24 // (op 2744) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2744) 


s_ashr_i32 s51, s19, 31 // (op 2744) 

s_xor_b32 s19, s18, s51 // (op 2744) 

s_sub_i32 s18, s19, s51 // (op 2744) 


s_mul_hi_i32 s19, s18, s14 // (op 2744) 
s_add_i32 s51, s19, s18 // (op 2744) 

s_ashr_i32 s19, s51, 31 // (op 2744) 
s_cmp_eq_i32 s14, 0 // (op 2744) 
s_mov_b32 s52, scc // (op 2744) 
s_cmp_eq_i32 s52, 0 // (op 2744) 
s_cselect_b32 s53, 0, -1 // (op 2744) 

s_and_b32 s52, s15, 31 // (op 2744) 
v_lshrrev_b32 v14, 5, v11 // (op 2986) 
s_cmp_eq_i32 s10, 0 // (op 2986) 
s_mov_b32 s54, scc // (op 2986) 
v_lshlrev_b32 v15, 2, v14 // (op 2986) 

s_cmp_eq_i32 s54, 0 // (op 2986) 
s_cselect_b32 s55, 0, -1 // (op 2986) 

s_lshl_b32 s54, 1, s52 // (op 2744) 
v_lshl_add_u32 v14, v10, 5, v15 // (op 2986) 


v_lshrrev_b32 v10, 2, v11 // (op 2986) 

s_add_i32 s56, s54, s53 // (op 2744) 


s_and_b32 s53, s19, s56 // (op 2744) 


s_add_i32 s19, s51, s53 // (op 2744) 


s_ashr_i32 s51, s19, s52 // (op 2744) 


s_lshl_b32 s19, s15, 24 // (op 2744) 
s_ashr_i32 s52, s19, 31 // (op 2744) 

s_xor_b32 s19, s51, s52 // (op 2744) 

s_sub_i32 s51, s19, s52 // (op 2744) 


s_mul_i32 s19, s18, s11 // (op 2744) 
s_sub_u32 s52, s16, s19 // (op 2744) 

s_mul_i32 s19, s51, s11 // (op 2744) 
s_add_u32 s19, s19, s52 // (op 2744) 

s_mul_hi_i32 s52, s16, s9 // (op 2744) 
s_add_i32 s53, s52, s16 // (op 2744) 

s_ashr_i32 s52, s53, 31 // (op 2744) 
s_cmp_eq_i32 s9, 0 // (op 2744) 
s_mov_b32 s54, scc // (op 2744) 
s_cmp_eq_i32 s54, 0 // (op 2744) 
s_cselect_b32 s56, 0, -1 // (op 2744) 

s_and_b32 s54, s13, 31 // (op 2986) 
v_and_b32 v11, 7, v10 // (op 2986) 

s_lshl_b32 s57, 1, s54 // (op 2986) 
v_and_b32 v10, 3, v1 // (op 2986) 
s_add_i32 s58, s57, s55 // (op 2986) 


s_and_b32 s55, s1, s58 // (op 2986) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2986) 


s_add_i32 s1, s0, s55 // (op 2986) 


v_lshrrev_b32 v10, 6, v1 // (op 2986) 
s_ashr_i32 s0, s1, s54 // (op 2986) 


v_and_b32 v11, 1, v10 // (op 2986) 

s_lshl_b32 s1, s13, 24 // (op 2986) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2986) 


s_ashr_i32 s54, s1, 31 // (op 2986) 

s_xor_b32 s1, s0, s54 // (op 2986) 

s_sub_i32 s0, s1, s54 // (op 2986) 


s_mul_hi_i32 s1, s0, s14 // (op 2986) 
s_add_i32 s54, s1, s0 // (op 2986) 

s_ashr_i32 s1, s54, 31 // (op 2986) 
s_cmp_eq_i32 s14, 0 // (op 2986) 
s_mov_b32 s55, scc // (op 2986) 
s_cmp_eq_i32 s55, 0 // (op 2986) 
s_cselect_b32 s57, 0, -1 // (op 2986) 

s_and_b32 s55, s15, 31 // (op 2986) 
s_lshl_b32 s58, 1, s55 // (op 2986) 
s_add_i32 s59, s58, s57 // (op 2986) 


s_and_b32 s57, s1, s59 // (op 2986) 


s_add_i32 s1, s54, s57 // (op 2986) 


s_ashr_i32 s54, s1, s55 // (op 2986) 


s_lshl_b32 s1, s15, 24 // (op 2986) 
s_ashr_i32 s55, s1, 31 // (op 2986) 

s_xor_b32 s1, s54, s55 // (op 2986) 

s_sub_i32 s54, s1, s55 // (op 2986) 


s_mul_i32 s1, s0, s11 // (op 2986) 
s_sub_u32 s55, s16, s1 // (op 2986) 

s_mul_i32 s1, s54, s11 // (op 2986) 
s_add_u32 s1, s1, s55 // (op 2986) 

s_mul_hi_i32 s55, s16, s9 // (op 2986) 
s_add_i32 s57, s55, s16 // (op 2986) 

s_ashr_i32 s55, s57, 31 // (op 2986) 
s_cmp_eq_i32 s9, 0 // (op 2986) 
s_mov_b32 s58, scc // (op 2986) 
s_cmp_eq_i32 s58, 0 // (op 2986) 
s_cselect_b32 s59, 0, -1 // (op 2986) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 2744) 
s_lshl_b32 s60, 1, s58 // (op 2744) 
s_add_i32 s61, s60, s56 // (op 2744) 


s_and_b32 s56, s52, s61 // (op 2744) 


s_add_i32 s52, s53, s56 // (op 2744) 


s_ashr_i32 s53, s52, s58 // (op 2744) 


s_lshl_b32 s52, s47, 24 // (op 2744) 
s_ashr_i32 s56, s52, 31 // (op 2744) 

s_xor_b32 s52, s53, s56 // (op 2744) 

s_sub_i32 s53, s52, s56 // (op 2744) 


s_and_b32 s52, s47, 31 // (op 2986) 
s_lshl_b32 s56, 1, s52 // (op 2986) 
s_add_i32 s58, s56, s59 // (op 2986) 


s_and_b32 s56, s55, s58 // (op 2986) 


s_add_i32 s55, s57, s56 // (op 2986) 


s_ashr_i32 s56, s55, s52 // (op 2986) 


s_lshl_b32 s52, s47, 24 // (op 2986) 
s_ashr_i32 s55, s52, 31 // (op 2986) 

s_xor_b32 s52, s56, s55 // (op 2986) 

s_sub_i32 s56, s52, s55 // (op 2986) 


s_add_u32 s58, s6, 64 // (op 2744) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2744) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2744) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2744) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2744) 

s_mul_i32 s52, s11, s58 // (op 2744) 
s_mul_i32 s55, s8, s52 // (op 2744) 

s_mul_i32 s52, s53, s55 // (op 2744) 

s_sub_u32 s55, s16, s52 // (op 2744) 

s_mov_b32 s52, s55 // (op 2744) 
s_mul_hi_i32 s57, s52, s12 // (op 2744) 
s_add_i32 s60, s57, s52 // (op 2744) 


s_ashr_i32 s52, s60, 31 // (op 2744) 
s_cmp_eq_i32 s12, 0 // (op 2744) 
s_mov_b32 s57, scc // (op 2744) 
s_cmp_eq_i32 s57, 0 // (op 2744) 
s_cselect_b32 s61, 0, -1 // (op 2744) 

s_and_b32 s57, s46, 31 // (op 2744) 
s_lshl_b32 s62, 1, s57 // (op 2744) 
s_add_i32 s63, s62, s61 // (op 2744) 


s_and_b32 s61, s52, s63 // (op 2744) 


s_add_i32 s52, s60, s61 // (op 2744) 


s_ashr_i32 s60, s52, s57 // (op 2744) 


s_lshl_b32 s52, s46, 24 // (op 2744) 
s_ashr_i32 s57, s52, 31 // (op 2744) 

s_xor_b32 s52, s60, s57 // (op 2744) 

s_sub_i32 s60, s52, s57 // (op 2744) 


s_mul_i32 s52, s48, s11 // (op 2744) 
s_add_u32 s62, s4, 256 // (op 2744) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 2744) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2744) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2744) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 8 // (op 2744) 

s_sub_i32 s57, s62, s52 // (op 2744) 

s_mul_i32 s61, s60, s57 // (op 2744) 

s_sub_u32 s57, s55, s61 // (op 2744) 


s_add_u32 s55, s57, s52 // (op 2744) 


s_cmp_eq_i32 s53, 0 // (op 2744) 
s_mov_b32 s52, scc // (op 2744) 

s_cmp_eq_i32 s52, 0 // (op 2744) 
s_cselect_b32 s53, s55, s19 // (op 2744) 


s_mul_i32 s19, s51, s58 // (op 2744) 


v_lshl_add_u32 v11, s53, 8, v8 // (op 2744) 


s_sub_i32 s51, s18, s19 // (op 2744) 


s_cmp_eq_i32 s52, 0 // (op 2744) 
s_cselect_b32 s53, s60, s51 // (op 2744) 



v_lshl_add_u32 v8, s53, 6, v7 // (op 2744) 


v_mul_lo_u32 v7, v8, s2 // (op 2744) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 2744) 


s_add_u32 s18, s6, 64 // (op 2986) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2986) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2986) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2986) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 2986) 

s_mul_i32 s55, s11, s18 // (op 2986) 
s_mul_i32 s51, s8, s55 // (op 2986) 

s_mul_i32 s55, s56, s51 // (op 2986) 

s_sub_u32 s51, s16, s55 // (op 2986) 

s_mov_b32 s55, s51 // (op 2986) 
s_mul_hi_i32 s52, s55, s12 // (op 2986) 
s_add_i32 s53, s52, s55 // (op 2986) 


s_ashr_i32 s52, s53, 31 // (op 2986) 
s_cmp_eq_i32 s12, 0 // (op 2986) 
s_mov_b32 s55, scc // (op 2986) 
s_cmp_eq_i32 s55, 0 // (op 2986) 
s_cselect_b32 s57, 0, -1 // (op 2986) 

s_and_b32 s55, s46, 31 // (op 2986) 
s_lshl_b32 s58, 1, s55 // (op 2986) 
s_add_i32 s59, s58, s57 // (op 2986) 


s_and_b32 s57, s52, s59 // (op 2986) 


s_add_i32 s52, s53, s57 // (op 2986) 


s_ashr_i32 s53, s52, s55 // (op 2986) 


s_lshl_b32 s52, s46, 24 // (op 2986) 
s_ashr_i32 s55, s52, 31 // (op 2986) 

s_xor_b32 s52, s53, s55 // (op 2986) 

s_sub_i32 s53, s52, s55 // (op 2986) 


s_mul_i32 s52, s48, s11 // (op 2986) 
s_add_u32 s58, s4, 256 // (op 2986) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 2986) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2986) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2986) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 2986) 

s_sub_i32 s55, s58, s52 // (op 2986) 

s_mul_i32 s57, s53, s55 // (op 2986) 

s_sub_u32 s55, s51, s57 // (op 2986) 


s_add_u32 s51, s55, s52 // (op 2986) 


s_cmp_eq_i32 s56, 0 // (op 2986) 
s_mov_b32 s52, scc // (op 2986) 

s_cmp_eq_i32 s52, 0 // (op 2986) 
s_cselect_b32 s55, s51, s1 // (op 2986) 


s_mul_i32 s1, s54, s18 // (op 2986) 


v_lshl_add_u32 v7, s55, 8, v14 // (op 2986) 


s_sub_i32 s51, s0, s1 // (op 2986) 


s_cmp_eq_i32 s52, 0 // (op 2986) 
s_cselect_b32 s0, s53, s51 // (op 2986) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 2986) 


v_mul_lo_u32 v10, v11, s20 // (op 2986) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 2986) 


s_mul_i32 s1, s2, s7 // (op 2746) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2746) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2746) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2746) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2746) most significant: sum
s_add_u32 s1, s1, s19 // (op 2746) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2746) 

s_mul_i32 s1, s20, s7 // (op 2988) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 2988) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 2988) least significant
s_mul_hi_u32 s52, s20, s6 // (op 2988) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2988) most significant: sum
s_add_u32 s1, s1, s52 // (op 2988) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2988) 

s_mov_b32 s56, s22 // (op 2746) reg expression
s_mov_b32 s57, s23 // (op 2746) reg expression
s_mov_b32 s58, s18 // (op 2746) reg expression
s_mov_b32 s59, 131072 // (op 2746) call()

s_mov_b32 s60, s24 // (op 2988) reg expression

s_mov_b32 s61, s25 // (op 2988) reg expression
s_mov_b32 s62, s52 // (op 2988) reg expression
s_mov_b32 s63, 131072 // (op 2988) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_accvgpr_write a16, 0.00000 // (op 405) call()
v_accvgpr_write a17, 0.00000 // (op 405) call()
v_accvgpr_write a18, 0.00000 // (op 405) call()
v_accvgpr_write a19, 0.00000 // (op 405) call()
v_accvgpr_write a20, 0.00000 // (op 405) call()
v_accvgpr_write a21, 0.00000 // (op 405) call()
v_accvgpr_write a22, 0.00000 // (op 405) call()
v_accvgpr_write a23, 0.00000 // (op 405) call()
v_accvgpr_write a24, 0.00000 // (op 405) call()
v_accvgpr_write a25, 0.00000 // (op 405) call()
v_accvgpr_write a26, 0.00000 // (op 405) call()
v_accvgpr_write a27, 0.00000 // (op 405) call()
v_accvgpr_write a28, 0.00000 // (op 405) call()
v_accvgpr_write a29, 0.00000 // (op 405) call()
v_accvgpr_write a30, 0.00000 // (op 405) call()
v_accvgpr_write a31, 0.00000 // (op 405) call()
v_accvgpr_write a32, 0.00000 // (op 645) call()
v_accvgpr_write a33, 0.00000 // (op 645) call()
v_accvgpr_write a34, 0.00000 // (op 645) call()
v_accvgpr_write a35, 0.00000 // (op 645) call()
v_accvgpr_write a36, 0.00000 // (op 645) call()
v_accvgpr_write a37, 0.00000 // (op 645) call()
v_accvgpr_write a38, 0.00000 // (op 645) call()
v_accvgpr_write a39, 0.00000 // (op 645) call()
v_accvgpr_write a40, 0.00000 // (op 645) call()
v_accvgpr_write a41, 0.00000 // (op 645) call()
v_accvgpr_write a42, 0.00000 // (op 645) call()
v_accvgpr_write a43, 0.00000 // (op 645) call()
v_accvgpr_write a44, 0.00000 // (op 645) call()
v_accvgpr_write a45, 0.00000 // (op 645) call()
v_accvgpr_write a46, 0.00000 // (op 645) call()
v_accvgpr_write a47, 0.00000 // (op 645) call()
v_accvgpr_write a48, 0.00000 // (op 885) call()
v_accvgpr_write a49, 0.00000 // (op 885) call()
v_accvgpr_write a50, 0.00000 // (op 885) call()
v_accvgpr_write a51, 0.00000 // (op 885) call()
v_accvgpr_write a52, 0.00000 // (op 885) call()
v_accvgpr_write a53, 0.00000 // (op 885) call()
v_accvgpr_write a54, 0.00000 // (op 885) call()
v_accvgpr_write a55, 0.00000 // (op 885) call()
v_accvgpr_write a56, 0.00000 // (op 885) call()
v_accvgpr_write a57, 0.00000 // (op 885) call()
v_accvgpr_write a58, 0.00000 // (op 885) call()
v_accvgpr_write a59, 0.00000 // (op 885) call()
v_accvgpr_write a60, 0.00000 // (op 885) call()
v_accvgpr_write a61, 0.00000 // (op 885) call()
v_accvgpr_write a62, 0.00000 // (op 885) call()
v_accvgpr_write a63, 0.00000 // (op 885) call()
s_mul_hi_i32 s0, s16, s10 // (op 2511) 
v_and_b32 v14, -8, v1 // (op 2511) 
s_add_i32 s1, s0, s16 // (op 2511) 

v_sub_u32 v15, v1, v14 // (op 2511) 

s_ashr_i32 s0, s1, 31 // (op 2511) 
v_lshlrev_b32 v14, 5, v15 // (op 2511) 

s_cmp_eq_i32 s10, 0 // (op 2511) 
s_mov_b32 s18, scc // (op 2511) 
v_lshrrev_b32 v15, 3, v1 // (op 2511) 
s_cmp_eq_i32 s18, 0 // (op 2511) 
s_cselect_b32 s19, 0, -1 // (op 2511) 

v_lshlrev_b32 v16, 2, v15 // (op 2511) 

s_and_b32 s18, s13, 31 // (op 2511) 
s_lshl_b32 s22, 1, s18 // (op 2511) 
s_add_i32 s23, s22, s19 // (op 2511) 


s_and_b32 s19, s0, s23 // (op 2511) 


s_add_i32 s0, s1, s19 // (op 2511) 


s_ashr_i32 s1, s0, s18 // (op 2511) 


s_lshl_b32 s0, s13, 24 // (op 2511) 
s_ashr_i32 s18, s0, 31 // (op 2511) 

s_xor_b32 s0, s1, s18 // (op 2511) 

s_sub_i32 s1, s0, s18 // (op 2511) 


s_mul_hi_i32 s0, s1, s14 // (op 2511) 
s_add_i32 s18, s0, s1 // (op 2511) 

s_ashr_i32 s0, s18, 31 // (op 2511) 
s_cmp_eq_i32 s14, 0 // (op 2511) 
s_mov_b32 s19, scc // (op 2511) 
s_cmp_eq_i32 s19, 0 // (op 2511) 
s_cselect_b32 s22, 0, -1 // (op 2511) 

s_and_b32 s19, s15, 31 // (op 2511) 
s_lshl_b32 s23, 1, s19 // (op 2511) 
s_add_i32 s24, s23, s22 // (op 2511) 


s_and_b32 s25, s0, s24 // (op 2511) 


s_add_i32 s0, s18, s25 // (op 2511) 


s_ashr_i32 s18, s0, s19 // (op 2511) 


s_lshl_b32 s0, s15, 24 // (op 2511) 
s_ashr_i32 s19, s0, 31 // (op 2511) 

s_xor_b32 s0, s18, s19 // (op 2511) 

s_sub_i32 s18, s0, s19 // (op 2511) 


s_mul_i32 s0, s1, s11 // (op 2511) 

s_sub_u32 s1, s16, s0 // (op 2511) 

s_mul_i32 s0, s18, s11 // (op 2511) 
s_add_u32 s0, s0, s1 // (op 2511) 


s_mul_hi_i32 s1, s16, s9 // (op 2511) 
s_add_i32 s18, s1, s16 // (op 2511) 

s_ashr_i32 s1, s18, 31 // (op 2511) 
s_cmp_eq_i32 s9, 0 // (op 2511) 
s_mov_b32 s19, scc // (op 2511) 
s_cmp_eq_i32 s19, 0 // (op 2511) 
s_cselect_b32 s22, 0, -1 // (op 2511) 

s_and_b32 s19, s47, 31 // (op 2511) 
s_lshl_b32 s23, 1, s19 // (op 2511) 
s_add_i32 s24, s23, s22 // (op 2511) 


s_and_b32 s25, s1, s24 // (op 2511) 


s_add_i32 s1, s18, s25 // (op 2511) 


s_ashr_i32 s18, s1, s19 // (op 2511) 


s_lshl_b32 s1, s47, 24 // (op 2511) 
s_ashr_i32 s19, s1, 31 // (op 2511) 

s_xor_b32 s1, s18, s19 // (op 2511) 

s_sub_i32 s18, s1, s19 // (op 2511) 


s_add_u32 s22, s6, 64 // (op 2511) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2511) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2511) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2511) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2511) 

s_mul_i32 s1, s11, s22 // (op 2511) 

s_mul_i32 s19, s8, s1 // (op 2511) 

s_mul_i32 s1, s18, s19 // (op 2511) 

s_sub_u32 s19, s16, s1 // (op 2511) 

s_mov_b32 s1, s19 // (op 2511) 
s_mul_hi_i32 s22, s1, s12 // (op 2511) 
s_add_i32 s23, s22, s1 // (op 2511) 


s_ashr_i32 s1, s23, 31 // (op 2511) 
s_cmp_eq_i32 s12, 0 // (op 2511) 
s_mov_b32 s22, scc // (op 2511) 
s_cmp_eq_i32 s22, 0 // (op 2511) 
s_cselect_b32 s24, 0, -1 // (op 2511) 

s_and_b32 s22, s46, 31 // (op 2511) 
s_lshl_b32 s25, 1, s22 // (op 2511) 
s_add_i32 s49, s25, s24 // (op 2511) 


s_and_b32 s24, s1, s49 // (op 2511) 


s_add_i32 s1, s23, s24 // (op 2511) 


s_ashr_i32 s23, s1, s22 // (op 2511) 


s_lshl_b32 s1, s46, 24 // (op 2511) 
s_ashr_i32 s22, s1, 31 // (op 2511) 

s_xor_b32 s1, s23, s22 // (op 2511) 

s_sub_i32 s23, s1, s22 // (op 2511) 


s_mul_i32 s1, s48, s11 // (op 2511) 
s_add_u32 s24, s4, 256 // (op 2511) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2511) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2511) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2511) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 8 // (op 2511) 

s_sub_i32 s22, s24, s1 // (op 2511) 

s_mul_i32 s24, s23, s22 // (op 2511) 


s_sub_u32 s25, s19, s24 // (op 2511) 


s_add_u32 s19, s25, s1 // (op 2511) 


s_cmp_eq_i32 s18, 0 // (op 2511) 
s_mov_b32 s1, scc // (op 2511) 

s_cmp_eq_i32 s1, 0 // (op 2511) 
s_cselect_b32 s18, s19, s0 // (op 2511) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 2511) 


v_mov_b32 v19, 0 // (op 2511) convert
v_mov_b32 v18, v15 // (op 2511) convert

v_mul_lo_u32 v15, v16, s27 // (op 2511) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 2511) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 2511) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 2511) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v14 // (op 2511) least significant half
v_addc_co_u32 v17, s[0:1], v19, v15, s[0:1] // (op 2511) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 2511) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2511) 

v_mov_b32 v16, v14 // (op 2511) convert

s_mul_i32 s1, s26, s31 // (op 2513) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2513) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2513) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2513) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2513) most significant: sum
s_add_u32 s1, s1, s19 // (op 2513) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2513) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2513) 

s_mov_b32 s52, s28 // (op 2513) reg expression
s_mov_b32 s53, s29 // (op 2513) reg expression
s_mov_b32 s54, s0 // (op 2513) reg expression
s_mov_b32 s55, 131072 // (op 2513) call()

s_mul_hi_i32 s0, s16, s10 // (op 2530) 
v_and_b32 v14, -8, v1 // (op 2530) 
s_add_i32 s1, s0, s16 // (op 2530) 

v_sub_u32 v15, v1, v14 // (op 2530) 

s_ashr_i32 s0, s1, 31 // (op 2530) 
v_lshlrev_b32 v14, 5, v15 // (op 2530) 

s_cmp_eq_i32 s10, 0 // (op 2530) 
s_mov_b32 s18, scc // (op 2530) 
v_lshrrev_b32 v15, 3, v1 // (op 2530) 
s_cmp_eq_i32 s18, 0 // (op 2530) 
s_cselect_b32 s19, 0, -1 // (op 2530) 

v_lshlrev_b32 v17, 2, v15 // (op 2530) 

s_and_b32 s18, s13, 31 // (op 2530) 
s_lshl_b32 s22, 1, s18 // (op 2530) 
s_add_i32 s23, s22, s19 // (op 2530) 


s_and_b32 s19, s0, s23 // (op 2530) 


s_add_i32 s0, s1, s19 // (op 2530) 


s_ashr_i32 s1, s0, s18 // (op 2530) 


s_lshl_b32 s0, s13, 24 // (op 2530) 
s_ashr_i32 s18, s0, 31 // (op 2530) 

s_xor_b32 s0, s1, s18 // (op 2530) 

s_sub_i32 s1, s0, s18 // (op 2530) 


s_mul_hi_i32 s0, s1, s14 // (op 2530) 
s_add_i32 s18, s0, s1 // (op 2530) 

s_ashr_i32 s0, s18, 31 // (op 2530) 
s_cmp_eq_i32 s14, 0 // (op 2530) 
s_mov_b32 s19, scc // (op 2530) 
s_cmp_eq_i32 s19, 0 // (op 2530) 
s_cselect_b32 s22, 0, -1 // (op 2530) 

s_and_b32 s19, s15, 31 // (op 2530) 
s_lshl_b32 s23, 1, s19 // (op 2530) 
s_add_i32 s24, s23, s22 // (op 2530) 


s_and_b32 s25, s0, s24 // (op 2530) 


s_add_i32 s0, s18, s25 // (op 2530) 


s_ashr_i32 s18, s0, s19 // (op 2530) 


s_lshl_b32 s0, s15, 24 // (op 2530) 
s_ashr_i32 s19, s0, 31 // (op 2530) 

s_xor_b32 s0, s18, s19 // (op 2530) 

s_sub_i32 s18, s0, s19 // (op 2530) 


s_mul_i32 s0, s1, s11 // (op 2530) 

s_sub_u32 s1, s16, s0 // (op 2530) 

s_mul_i32 s0, s18, s11 // (op 2530) 
s_add_u32 s0, s0, s1 // (op 2530) 


s_mul_hi_i32 s1, s16, s9 // (op 2530) 
s_add_i32 s18, s1, s16 // (op 2530) 

s_ashr_i32 s1, s18, 31 // (op 2530) 
s_cmp_eq_i32 s9, 0 // (op 2530) 
s_mov_b32 s19, scc // (op 2530) 
s_cmp_eq_i32 s19, 0 // (op 2530) 
s_cselect_b32 s22, 0, -1 // (op 2530) 

s_and_b32 s19, s47, 31 // (op 2530) 
s_lshl_b32 s23, 1, s19 // (op 2530) 
s_add_i32 s24, s23, s22 // (op 2530) 


s_and_b32 s25, s1, s24 // (op 2530) 


s_add_i32 s1, s18, s25 // (op 2530) 


s_ashr_i32 s18, s1, s19 // (op 2530) 


s_lshl_b32 s1, s47, 24 // (op 2530) 
s_ashr_i32 s19, s1, 31 // (op 2530) 

s_xor_b32 s1, s18, s19 // (op 2530) 

s_sub_i32 s18, s1, s19 // (op 2530) 


s_add_u32 s50, s6, 64 // (op 2530) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2530) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2530) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2530) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2530) 

s_mul_i32 s1, s11, s24 // (op 2530) 

s_mul_i32 s19, s8, s1 // (op 2530) 

s_mul_i32 s1, s18, s19 // (op 2530) 

s_sub_u32 s19, s16, s1 // (op 2530) 

s_mov_b32 s1, s19 // (op 2530) 
s_mul_hi_i32 s22, s1, s12 // (op 2530) 
s_add_i32 s23, s22, s1 // (op 2530) 


s_ashr_i32 s1, s23, 31 // (op 2530) 
s_cmp_eq_i32 s12, 0 // (op 2530) 
s_mov_b32 s22, scc // (op 2530) 
s_cmp_eq_i32 s22, 0 // (op 2530) 
s_cselect_b32 s24, 0, -1 // (op 2530) 

s_and_b32 s22, s46, 31 // (op 2530) 
s_lshl_b32 s25, 1, s22 // (op 2530) 
s_add_i32 s49, s25, s24 // (op 2530) 


s_and_b32 s24, s1, s49 // (op 2530) 


s_add_i32 s1, s23, s24 // (op 2530) 


s_ashr_i32 s23, s1, s22 // (op 2530) 


s_lshl_b32 s1, s46, 24 // (op 2530) 
s_ashr_i32 s22, s1, 31 // (op 2530) 

s_xor_b32 s1, s23, s22 // (op 2530) 

s_sub_i32 s23, s1, s22 // (op 2530) 


s_mul_i32 s1, s48, s11 // (op 2530) 
s_add_u32 s24, s4, 256 // (op 2530) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2530) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2530) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2530) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 2530) 

s_sub_i32 s22, s24, s1 // (op 2530) 

s_mul_i32 s24, s23, s22 // (op 2530) 


s_sub_u32 s25, s19, s24 // (op 2530) 


s_add_u32 s19, s25, s1 // (op 2530) 


s_cmp_eq_i32 s18, 0 // (op 2530) 
s_mov_b32 s1, scc // (op 2530) 

s_cmp_eq_i32 s1, 0 // (op 2530) 
s_cselect_b32 s18, s19, s0 // (op 2530) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 2530) 


v_mov_b32 v19, 0 // (op 2530) convert
v_mov_b32 v18, v15 // (op 2530) convert

v_mul_lo_u32 v15, v17, s27 // (op 2530) most significant: low of low * high
v_mul_hi_u32 v20, v17, s26 // (op 2530) most significant: high of low * low
v_mul_lo_u32 v14, v17, s26 // (op 2530) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2530) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 2530) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 2530) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 2530) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 2530) 

v_mov_b32 v17, v14 // (op 2530) convert

s_mul_i32 s1, s26, s31 // (op 2532) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2532) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2532) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2532) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2532) most significant: sum
s_add_u32 s1, s1, s19 // (op 2532) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2532) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2532) 

s_mov_b32 s52, s28 // (op 2532) reg expression
s_mov_b32 s53, s29 // (op 2532) reg expression
s_mov_b32 s54, s0 // (op 2532) reg expression
s_mov_b32 s55, 131072 // (op 2532) call()

s_mul_hi_i32 s0, s16, s10 // (op 2575) 
v_and_b32 v14, -2, v1 // (op 2575) 
s_add_i32 s1, s0, s16 // (op 2575) 

v_sub_u32 v15, v1, v14 // (op 2575) 

s_ashr_i32 s0, s1, 31 // (op 2575) 
v_lshlrev_b32 v14, 5, v15 // (op 2575) 

s_cmp_eq_i32 s10, 0 // (op 2575) 
s_mov_b32 s18, scc // (op 2575) 
v_lshrrev_b32 v15, 1, v1 // (op 2575) 
s_cmp_eq_i32 s18, 0 // (op 2575) 
s_cselect_b32 s19, 0, -1 // (op 2575) 

s_and_b32 s18, s13, 31 // (op 2575) 
s_lshl_b32 s22, 1, s18 // (op 2575) 
s_add_i32 s23, s22, s19 // (op 2575) 


s_and_b32 s19, s0, s23 // (op 2575) 


s_add_i32 s0, s1, s19 // (op 2575) 


s_ashr_i32 s1, s0, s18 // (op 2575) 


s_lshl_b32 s0, s13, 24 // (op 2575) 
s_ashr_i32 s18, s0, 31 // (op 2575) 

s_xor_b32 s0, s1, s18 // (op 2575) 

s_sub_i32 s1, s0, s18 // (op 2575) 


s_mul_hi_i32 s0, s1, s14 // (op 2575) 
s_add_i32 s18, s0, s1 // (op 2575) 

s_ashr_i32 s0, s18, 31 // (op 2575) 
s_cmp_eq_i32 s14, 0 // (op 2575) 
s_mov_b32 s19, scc // (op 2575) 
s_cmp_eq_i32 s19, 0 // (op 2575) 
s_cselect_b32 s22, 0, -1 // (op 2575) 

s_and_b32 s19, s15, 31 // (op 2575) 
s_lshl_b32 s23, 1, s19 // (op 2575) 
s_add_i32 s24, s23, s22 // (op 2575) 


s_and_b32 s25, s0, s24 // (op 2575) 


s_add_i32 s0, s18, s25 // (op 2575) 


s_ashr_i32 s18, s0, s19 // (op 2575) 


s_lshl_b32 s0, s15, 24 // (op 2575) 
s_ashr_i32 s19, s0, 31 // (op 2575) 

s_xor_b32 s0, s18, s19 // (op 2575) 

s_sub_i32 s18, s0, s19 // (op 2575) 


s_add_u32 s50, s6, 64 // (op 2575) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2575) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2575) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2575) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2575) 

s_mul_i32 s0, s18, s24 // (op 2575) 

s_sub_i32 s18, s1, s0 // (op 2575) 


s_mul_hi_i32 s19, s16, s9 // (op 2575) 
s_add_i32 s0, s19, s16 // (op 2575) 

s_ashr_i32 s1, s0, 31 // (op 2575) 
s_cmp_eq_i32 s9, 0 // (op 2575) 
s_mov_b32 s19, scc // (op 2575) 
s_cmp_eq_i32 s19, 0 // (op 2575) 
s_cselect_b32 s22, 0, -1 // (op 2575) 

s_and_b32 s19, s47, 31 // (op 2575) 
s_lshl_b32 s23, 1, s19 // (op 2575) 
s_add_i32 s49, s23, s22 // (op 2575) 


s_and_b32 s22, s1, s49 // (op 2575) 


s_add_i32 s1, s0, s22 // (op 2575) 


s_ashr_i32 s0, s1, s19 // (op 2575) 


s_lshl_b32 s1, s47, 24 // (op 2575) 
s_ashr_i32 s19, s1, 31 // (op 2575) 

s_xor_b32 s1, s0, s19 // (op 2575) 

s_sub_i32 s0, s1, s19 // (op 2575) 


s_mul_i32 s1, s11, s24 // (op 2575) 

s_mul_i32 s19, s8, s1 // (op 2575) 

s_mul_i32 s1, s0, s19 // (op 2575) 

s_sub_u32 s19, s16, s1 // (op 2575) 

s_mul_hi_i32 s1, s19, s12 // (op 2575) 
s_add_i32 s22, s1, s19 // (op 2575) 


s_ashr_i32 s1, s22, 31 // (op 2575) 
s_cmp_eq_i32 s12, 0 // (op 2575) 
s_mov_b32 s19, scc // (op 2575) 
s_cmp_eq_i32 s19, 0 // (op 2575) 
s_cselect_b32 s23, 0, -1 // (op 2575) 

s_and_b32 s19, s46, 31 // (op 2575) 
s_lshl_b32 s24, 1, s19 // (op 2575) 
s_add_i32 s25, s24, s23 // (op 2575) 


s_and_b32 s23, s1, s25 // (op 2575) 


s_add_i32 s1, s22, s23 // (op 2575) 


s_ashr_i32 s22, s1, s19 // (op 2575) 


s_lshl_b32 s1, s46, 24 // (op 2575) 
s_ashr_i32 s19, s1, 31 // (op 2575) 

s_xor_b32 s1, s22, s19 // (op 2575) 

s_sub_i32 s22, s1, s19 // (op 2575) 


s_cmp_eq_i32 s0, 0 // (op 2575) 
s_mov_b32 s1, scc // (op 2575) 

s_cmp_eq_i32 s1, 0 // (op 2575) 
s_cselect_b32 s0, s22, s18 // (op 2575) 



v_lshl_add_u32 v18, s0, 6, v14 // (op 2575) 


v_mov_b32 v21, 0 // (op 2575) convert
v_mov_b32 v20, v18 // (op 2575) convert

v_mul_lo_u32 v19, v15, s33 // (op 2575) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 2575) most significant: high of low * low
v_mul_lo_u32 v18, v15, s32 // (op 2575) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2575) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 2575) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 2575) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 2575) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 2575) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 2575) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2575) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2575) 

v_mov_b32 v14, v18 // (op 2575) convert

s_mul_i32 s1, s31, 6 // (op 2577) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2577) least significant
s_mul_hi_u32 s18, s30, 6 // (op 2577) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2577) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2577) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2577) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2577) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2577) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2577) most significant: sum
s_add_u32 s19, s19, s23 // (op 2577) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2577) 

s_mov_b32 s64, s34 // (op 2577) reg expression
s_mov_b32 s65, s35 // (op 2577) reg expression
s_mov_b32 s66, s0 // (op 2577) reg expression
s_mov_b32 s67, 131072 // (op 2577) call()

s_mul_hi_i32 s0, s16, s10 // (op 2594) 
v_and_b32 v15, -2, v1 // (op 2594) 
s_add_i32 s1, s0, s16 // (op 2594) 

v_sub_u32 v18, v1, v15 // (op 2594) 

s_ashr_i32 s0, s1, 31 // (op 2594) 
v_lshlrev_b32 v15, 5, v18 // (op 2594) 

s_cmp_eq_i32 s10, 0 // (op 2594) 
s_mov_b32 s18, scc // (op 2594) 
v_lshrrev_b32 v18, 1, v1 // (op 2594) 
s_cmp_eq_i32 s18, 0 // (op 2594) 
s_cselect_b32 s19, 0, -1 // (op 2594) 

s_and_b32 s18, s13, 31 // (op 2594) 
s_lshl_b32 s22, 1, s18 // (op 2594) 
s_add_i32 s23, s22, s19 // (op 2594) 


s_and_b32 s19, s0, s23 // (op 2594) 


s_add_i32 s0, s1, s19 // (op 2594) 


s_ashr_i32 s1, s0, s18 // (op 2594) 


s_lshl_b32 s0, s13, 24 // (op 2594) 
s_ashr_i32 s18, s0, 31 // (op 2594) 

s_xor_b32 s0, s1, s18 // (op 2594) 

s_sub_i32 s1, s0, s18 // (op 2594) 


s_mul_hi_i32 s0, s1, s14 // (op 2594) 
s_add_i32 s18, s0, s1 // (op 2594) 

s_ashr_i32 s0, s18, 31 // (op 2594) 
s_cmp_eq_i32 s14, 0 // (op 2594) 
s_mov_b32 s19, scc // (op 2594) 
s_cmp_eq_i32 s19, 0 // (op 2594) 
s_cselect_b32 s22, 0, -1 // (op 2594) 

s_and_b32 s19, s15, 31 // (op 2594) 
s_lshl_b32 s23, 1, s19 // (op 2594) 
s_add_i32 s24, s23, s22 // (op 2594) 


s_and_b32 s25, s0, s24 // (op 2594) 


s_add_i32 s0, s18, s25 // (op 2594) 


s_ashr_i32 s18, s0, s19 // (op 2594) 


s_lshl_b32 s0, s15, 24 // (op 2594) 
s_ashr_i32 s19, s0, 31 // (op 2594) 

s_xor_b32 s0, s18, s19 // (op 2594) 

s_sub_i32 s18, s0, s19 // (op 2594) 


s_add_u32 s50, s6, 64 // (op 2594) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2594) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2594) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2594) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2594) 

s_mul_i32 s0, s18, s24 // (op 2594) 

s_sub_i32 s18, s1, s0 // (op 2594) 


s_mul_hi_i32 s19, s16, s9 // (op 2594) 
s_add_i32 s0, s19, s16 // (op 2594) 

s_ashr_i32 s1, s0, 31 // (op 2594) 
s_cmp_eq_i32 s9, 0 // (op 2594) 
s_mov_b32 s19, scc // (op 2594) 
s_cmp_eq_i32 s19, 0 // (op 2594) 
s_cselect_b32 s22, 0, -1 // (op 2594) 

s_and_b32 s19, s47, 31 // (op 2594) 
s_lshl_b32 s23, 1, s19 // (op 2594) 
s_add_i32 s49, s23, s22 // (op 2594) 


s_and_b32 s22, s1, s49 // (op 2594) 


s_add_i32 s1, s0, s22 // (op 2594) 


s_ashr_i32 s0, s1, s19 // (op 2594) 


s_lshl_b32 s1, s47, 24 // (op 2594) 
s_ashr_i32 s19, s1, 31 // (op 2594) 

s_xor_b32 s1, s0, s19 // (op 2594) 

s_sub_i32 s0, s1, s19 // (op 2594) 


s_mul_i32 s1, s11, s24 // (op 2594) 

s_mul_i32 s19, s8, s1 // (op 2594) 

s_mul_i32 s1, s0, s19 // (op 2594) 

s_sub_u32 s19, s16, s1 // (op 2594) 

s_mul_hi_i32 s1, s19, s12 // (op 2594) 
s_add_i32 s22, s1, s19 // (op 2594) 


s_ashr_i32 s1, s22, 31 // (op 2594) 
s_cmp_eq_i32 s12, 0 // (op 2594) 
s_mov_b32 s19, scc // (op 2594) 
s_cmp_eq_i32 s19, 0 // (op 2594) 
s_cselect_b32 s23, 0, -1 // (op 2594) 

s_and_b32 s19, s46, 31 // (op 2594) 
s_lshl_b32 s24, 1, s19 // (op 2594) 
s_add_i32 s25, s24, s23 // (op 2594) 


s_and_b32 s23, s1, s25 // (op 2594) 


s_add_i32 s1, s22, s23 // (op 2594) 


s_ashr_i32 s22, s1, s19 // (op 2594) 


s_lshl_b32 s1, s46, 24 // (op 2594) 
s_ashr_i32 s19, s1, 31 // (op 2594) 

s_xor_b32 s1, s22, s19 // (op 2594) 

s_sub_i32 s22, s1, s19 // (op 2594) 


s_cmp_eq_i32 s0, 0 // (op 2594) 
s_mov_b32 s1, scc // (op 2594) 

s_cmp_eq_i32 s1, 0 // (op 2594) 
s_cselect_b32 s0, s22, s18 // (op 2594) 



v_lshl_add_u32 v19, s0, 6, v15 // (op 2594) 


v_mov_b32 v21, 0 // (op 2594) convert
v_mov_b32 v20, v19 // (op 2594) convert

v_mul_lo_u32 v23, v18, s33 // (op 2594) most significant: low of low * high
v_mul_hi_u32 v15, v18, s32 // (op 2594) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 2594) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2594) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 2594) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 2594) most significant half



v_mul_lo_u32 v19, v23, 6 // (op 2594) most significant: low of high * low
v_mul_hi_u32 v15, v22, 6 // (op 2594) most significant: high of low * low
v_mul_lo_u32 v18, v22, 6 // (op 2594) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2594) most significant: sum


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 2594) 

v_mov_b32 v15, v20 // (op 2594) convert

s_mul_i32 s1, s31, 6 // (op 2596) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2596) least significant
s_mul_hi_u32 s18, s30, 6 // (op 2596) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2596) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2596) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2596) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2596) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2596) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2596) most significant: sum
s_add_u32 s19, s19, s23 // (op 2596) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2596) 

s_mov_b32 s64, s34 // (op 2596) reg expression
s_mov_b32 s65, s35 // (op 2596) reg expression
s_mov_b32 s66, s0 // (op 2596) reg expression
s_mov_b32 s67, 131072 // (op 2596) call()

s_mul_hi_i32 s0, s16, s10 // (op 2639) 
v_and_b32 v18, -64, v1 // (op 2639) 
s_add_i32 s1, s0, s16 // (op 2639) 

v_sub_u32 v19, v1, v18 // (op 2639) 

s_ashr_i32 s0, s1, 31 // (op 2639) 
v_lshlrev_b32 v18, 2, v19 // (op 2639) 

s_cmp_eq_i32 s10, 0 // (op 2639) 
s_mov_b32 s18, scc // (op 2639) 
v_lshrrev_b32 v19, 6, v1 // (op 2639) 
s_cmp_eq_i32 s18, 0 // (op 2639) 
s_cselect_b32 s19, 0, -1 // (op 2639) 

s_and_b32 s18, s13, 31 // (op 2639) 
s_lshl_b32 s22, 1, s18 // (op 2639) 
s_add_i32 s23, s22, s19 // (op 2639) 


s_and_b32 s19, s0, s23 // (op 2639) 


s_add_i32 s0, s1, s19 // (op 2639) 


s_ashr_i32 s1, s0, s18 // (op 2639) 


s_lshl_b32 s0, s13, 24 // (op 2639) 
s_ashr_i32 s18, s0, 31 // (op 2639) 

s_xor_b32 s0, s1, s18 // (op 2639) 

s_sub_i32 s1, s0, s18 // (op 2639) 


s_mul_hi_i32 s0, s1, s14 // (op 2639) 
s_add_i32 s18, s0, s1 // (op 2639) 

s_ashr_i32 s0, s18, 31 // (op 2639) 
s_cmp_eq_i32 s14, 0 // (op 2639) 
s_mov_b32 s19, scc // (op 2639) 
s_cmp_eq_i32 s19, 0 // (op 2639) 
s_cselect_b32 s22, 0, -1 // (op 2639) 

s_and_b32 s19, s15, 31 // (op 2639) 
s_lshl_b32 s23, 1, s19 // (op 2639) 
s_add_i32 s24, s23, s22 // (op 2639) 


s_and_b32 s25, s0, s24 // (op 2639) 


s_add_i32 s0, s18, s25 // (op 2639) 


s_ashr_i32 s18, s0, s19 // (op 2639) 


s_lshl_b32 s0, s15, 24 // (op 2639) 
s_ashr_i32 s19, s0, 31 // (op 2639) 

s_xor_b32 s0, s18, s19 // (op 2639) 

s_sub_i32 s18, s0, s19 // (op 2639) 


s_mul_i32 s0, s1, s11 // (op 2639) 

s_sub_u32 s1, s16, s0 // (op 2639) 

s_mul_i32 s0, s18, s11 // (op 2639) 
s_add_u32 s0, s0, s1 // (op 2639) 


s_mul_hi_i32 s1, s16, s9 // (op 2639) 
s_add_i32 s18, s1, s16 // (op 2639) 

s_ashr_i32 s1, s18, 31 // (op 2639) 
s_cmp_eq_i32 s9, 0 // (op 2639) 
s_mov_b32 s19, scc // (op 2639) 
s_cmp_eq_i32 s19, 0 // (op 2639) 
s_cselect_b32 s22, 0, -1 // (op 2639) 

s_and_b32 s19, s47, 31 // (op 2639) 
s_lshl_b32 s23, 1, s19 // (op 2639) 
s_add_i32 s24, s23, s22 // (op 2639) 


s_and_b32 s25, s1, s24 // (op 2639) 


s_add_i32 s1, s18, s25 // (op 2639) 


s_ashr_i32 s18, s1, s19 // (op 2639) 


s_lshl_b32 s1, s47, 24 // (op 2639) 
s_ashr_i32 s19, s1, 31 // (op 2639) 

s_xor_b32 s1, s18, s19 // (op 2639) 

s_sub_i32 s18, s1, s19 // (op 2639) 


s_add_u32 s50, s6, 64 // (op 2639) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2639) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2639) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2639) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2639) 

s_mul_i32 s1, s11, s24 // (op 2639) 

s_mul_i32 s19, s8, s1 // (op 2639) 

s_mul_i32 s1, s18, s19 // (op 2639) 

s_sub_u32 s19, s16, s1 // (op 2639) 

s_mov_b32 s1, s19 // (op 2639) 
s_mul_hi_i32 s22, s1, s12 // (op 2639) 
s_add_i32 s23, s22, s1 // (op 2639) 


s_ashr_i32 s1, s23, 31 // (op 2639) 
s_cmp_eq_i32 s12, 0 // (op 2639) 
s_mov_b32 s22, scc // (op 2639) 
s_cmp_eq_i32 s22, 0 // (op 2639) 
s_cselect_b32 s24, 0, -1 // (op 2639) 

s_and_b32 s22, s46, 31 // (op 2639) 
s_lshl_b32 s25, 1, s22 // (op 2639) 
s_add_i32 s49, s25, s24 // (op 2639) 


s_and_b32 s24, s1, s49 // (op 2639) 


s_add_i32 s1, s23, s24 // (op 2639) 


s_ashr_i32 s23, s1, s22 // (op 2639) 


s_lshl_b32 s1, s46, 24 // (op 2639) 
s_ashr_i32 s22, s1, 31 // (op 2639) 

s_xor_b32 s1, s23, s22 // (op 2639) 

s_sub_i32 s23, s1, s22 // (op 2639) 


s_mul_i32 s1, s48, s11 // (op 2639) 
s_add_u32 s24, s4, 256 // (op 2639) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2639) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2639) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2639) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 2639) 

s_sub_i32 s22, s24, s1 // (op 2639) 

s_mul_i32 s24, s23, s22 // (op 2639) 


s_sub_u32 s25, s19, s24 // (op 2639) 


s_add_u32 s19, s25, s1 // (op 2639) 


s_cmp_eq_i32 s18, 0 // (op 2639) 
s_mov_b32 s1, scc // (op 2639) 

s_cmp_eq_i32 s1, 0 // (op 2639) 
s_cselect_b32 s18, s19, s0 // (op 2639) 



v_lshl_add_u32 v20, s18, 8, v18 // (op 2639) 


v_mul_lo_u32 v18, v19, s36 // (op 2639) 

v_add_u32 v19, v20, v18 // (op 2639) 


s_mul_i32 s1, s36, s41 // (op 2641) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2641) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2641) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2641) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2641) most significant: sum
s_add_u32 s1, s1, s19 // (op 2641) most significant: sum


s_mov_b32 s68, s38 // (op 2641) reg expression
s_mov_b32 s69, s39 // (op 2641) reg expression
s_mov_b32 s70, s0 // (op 2641) reg expression
s_mov_b32 s71, 131072 // (op 2641) call()

s_mul_hi_i32 s0, s16, s10 // (op 2658) 
v_and_b32 v18, -64, v1 // (op 2658) 
s_add_i32 s1, s0, s16 // (op 2658) 

v_sub_u32 v20, v1, v18 // (op 2658) 

s_ashr_i32 s0, s1, 31 // (op 2658) 
v_lshlrev_b32 v18, 2, v20 // (op 2658) 

s_cmp_eq_i32 s10, 0 // (op 2658) 
s_mov_b32 s18, scc // (op 2658) 
v_lshrrev_b32 v20, 6, v1 // (op 2658) 
s_cmp_eq_i32 s18, 0 // (op 2658) 
s_cselect_b32 s19, 0, -1 // (op 2658) 

s_and_b32 s18, s13, 31 // (op 2658) 
s_lshl_b32 s22, 1, s18 // (op 2658) 
s_add_i32 s23, s22, s19 // (op 2658) 


s_and_b32 s19, s0, s23 // (op 2658) 


s_add_i32 s0, s1, s19 // (op 2658) 


s_ashr_i32 s1, s0, s18 // (op 2658) 


s_lshl_b32 s0, s13, 24 // (op 2658) 
s_ashr_i32 s18, s0, 31 // (op 2658) 

s_xor_b32 s0, s1, s18 // (op 2658) 

s_sub_i32 s1, s0, s18 // (op 2658) 


s_mul_hi_i32 s0, s1, s14 // (op 2658) 
s_add_i32 s18, s0, s1 // (op 2658) 

s_ashr_i32 s0, s18, 31 // (op 2658) 
s_cmp_eq_i32 s14, 0 // (op 2658) 
s_mov_b32 s19, scc // (op 2658) 
s_cmp_eq_i32 s19, 0 // (op 2658) 
s_cselect_b32 s22, 0, -1 // (op 2658) 

s_and_b32 s19, s15, 31 // (op 2658) 
s_lshl_b32 s23, 1, s19 // (op 2658) 
s_add_i32 s24, s23, s22 // (op 2658) 


s_and_b32 s25, s0, s24 // (op 2658) 


s_add_i32 s0, s18, s25 // (op 2658) 


s_ashr_i32 s18, s0, s19 // (op 2658) 


s_lshl_b32 s0, s15, 24 // (op 2658) 
s_ashr_i32 s19, s0, 31 // (op 2658) 

s_xor_b32 s0, s18, s19 // (op 2658) 

s_sub_i32 s18, s0, s19 // (op 2658) 


s_mul_i32 s0, s1, s11 // (op 2658) 

s_sub_u32 s1, s16, s0 // (op 2658) 

s_mul_i32 s0, s18, s11 // (op 2658) 
s_add_u32 s0, s0, s1 // (op 2658) 


s_mul_hi_i32 s1, s16, s9 // (op 2658) 
s_add_i32 s18, s1, s16 // (op 2658) 

s_ashr_i32 s1, s18, 31 // (op 2658) 
s_cmp_eq_i32 s9, 0 // (op 2658) 
s_mov_b32 s19, scc // (op 2658) 
s_cmp_eq_i32 s19, 0 // (op 2658) 
s_cselect_b32 s22, 0, -1 // (op 2658) 

s_and_b32 s19, s47, 31 // (op 2658) 
s_lshl_b32 s23, 1, s19 // (op 2658) 
s_add_i32 s24, s23, s22 // (op 2658) 


s_and_b32 s25, s1, s24 // (op 2658) 


s_add_i32 s1, s18, s25 // (op 2658) 


s_ashr_i32 s18, s1, s19 // (op 2658) 


s_lshl_b32 s1, s47, 24 // (op 2658) 
s_ashr_i32 s19, s1, 31 // (op 2658) 

s_xor_b32 s1, s18, s19 // (op 2658) 

s_sub_i32 s18, s1, s19 // (op 2658) 


s_add_u32 s50, s6, 64 // (op 2658) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2658) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2658) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2658) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2658) 

s_mul_i32 s1, s11, s24 // (op 2658) 

s_mul_i32 s19, s8, s1 // (op 2658) 

s_mul_i32 s1, s18, s19 // (op 2658) 

s_sub_u32 s19, s16, s1 // (op 2658) 

s_mov_b32 s1, s19 // (op 2658) 
s_mul_hi_i32 s22, s1, s12 // (op 2658) 
s_add_i32 s23, s22, s1 // (op 2658) 


s_ashr_i32 s1, s23, 31 // (op 2658) 
s_cmp_eq_i32 s12, 0 // (op 2658) 
s_mov_b32 s22, scc // (op 2658) 
s_cmp_eq_i32 s22, 0 // (op 2658) 
s_cselect_b32 s24, 0, -1 // (op 2658) 

s_and_b32 s22, s46, 31 // (op 2658) 
s_lshl_b32 s25, 1, s22 // (op 2658) 
s_add_i32 s49, s25, s24 // (op 2658) 


s_and_b32 s24, s1, s49 // (op 2658) 


s_add_i32 s1, s23, s24 // (op 2658) 


s_ashr_i32 s23, s1, s22 // (op 2658) 


s_lshl_b32 s1, s46, 24 // (op 2658) 
s_ashr_i32 s22, s1, 31 // (op 2658) 

s_xor_b32 s1, s23, s22 // (op 2658) 

s_sub_i32 s23, s1, s22 // (op 2658) 


s_mul_i32 s1, s48, s11 // (op 2658) 
s_add_u32 s24, s4, 256 // (op 2658) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2658) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2658) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2658) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 2658) 

s_sub_i32 s22, s24, s1 // (op 2658) 

s_mul_i32 s24, s23, s22 // (op 2658) 


s_sub_u32 s25, s19, s24 // (op 2658) 


s_add_u32 s19, s25, s1 // (op 2658) 


s_cmp_eq_i32 s18, 0 // (op 2658) 
s_mov_b32 s1, scc // (op 2658) 

s_cmp_eq_i32 s1, 0 // (op 2658) 
s_cselect_b32 s18, s19, s0 // (op 2658) 



v_lshl_add_u32 v21, s18, 8, v18 // (op 2658) 


v_mul_lo_u32 v18, v20, s36 // (op 2658) 

v_add_u32 v20, v21, v18 // (op 2658) 


s_mul_i32 s1, s36, s41 // (op 2660) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2660) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2660) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2660) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2660) most significant: sum
s_add_u32 s1, s1, s19 // (op 2660) most significant: sum


s_mov_b32 s68, s38 // (op 2660) reg expression
s_mov_b32 s69, s39 // (op 2660) reg expression
s_mov_b32 s70, s0 // (op 2660) reg expression
s_mov_b32 s71, 131072 // (op 2660) call()

s_mul_hi_i32 s0, s16, s10 // (op 2703) 
v_and_b32 v18, -64, v1 // (op 2703) 
s_add_i32 s1, s0, s16 // (op 2703) 

v_sub_u32 v21, v1, v18 // (op 2703) 

s_ashr_i32 s0, s1, 31 // (op 2703) 
v_lshrrev_b32 v18, 6, v1 // (op 2703) 
s_cmp_eq_i32 s10, 0 // (op 2703) 
s_mov_b32 s18, scc // (op 2703) 
s_cmp_eq_i32 s18, 0 // (op 2703) 
s_cselect_b32 s19, 0, -1 // (op 2703) 

s_and_b32 s18, s13, 31 // (op 2703) 
s_lshl_b32 s22, 1, s18 // (op 2703) 
s_add_i32 s23, s22, s19 // (op 2703) 


s_and_b32 s19, s0, s23 // (op 2703) 


s_add_i32 s0, s1, s19 // (op 2703) 


s_ashr_i32 s1, s0, s18 // (op 2703) 


s_lshl_b32 s0, s13, 24 // (op 2703) 
s_ashr_i32 s18, s0, 31 // (op 2703) 

s_xor_b32 s0, s1, s18 // (op 2703) 

s_sub_i32 s1, s0, s18 // (op 2703) 


s_mul_hi_i32 s0, s1, s14 // (op 2703) 
s_add_i32 s18, s0, s1 // (op 2703) 

s_ashr_i32 s0, s18, 31 // (op 2703) 
s_cmp_eq_i32 s14, 0 // (op 2703) 
s_mov_b32 s19, scc // (op 2703) 
s_cmp_eq_i32 s19, 0 // (op 2703) 
s_cselect_b32 s22, 0, -1 // (op 2703) 

s_and_b32 s19, s15, 31 // (op 2703) 
s_lshl_b32 s23, 1, s19 // (op 2703) 
s_add_i32 s24, s23, s22 // (op 2703) 


s_and_b32 s25, s0, s24 // (op 2703) 


s_add_i32 s0, s18, s25 // (op 2703) 


s_ashr_i32 s18, s0, s19 // (op 2703) 


s_lshl_b32 s0, s15, 24 // (op 2703) 
s_ashr_i32 s19, s0, 31 // (op 2703) 

s_xor_b32 s0, s18, s19 // (op 2703) 

s_sub_i32 s18, s0, s19 // (op 2703) 


s_add_u32 s50, s6, 64 // (op 2703) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2703) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2703) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2703) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2703) 

s_mul_i32 s0, s18, s24 // (op 2703) 

s_sub_i32 s18, s1, s0 // (op 2703) 


s_mul_hi_i32 s19, s16, s9 // (op 2703) 
s_add_i32 s0, s19, s16 // (op 2703) 

s_ashr_i32 s1, s0, 31 // (op 2703) 
s_cmp_eq_i32 s9, 0 // (op 2703) 
s_mov_b32 s19, scc // (op 2703) 
s_cmp_eq_i32 s19, 0 // (op 2703) 
s_cselect_b32 s22, 0, -1 // (op 2703) 

s_and_b32 s19, s47, 31 // (op 2703) 
s_lshl_b32 s23, 1, s19 // (op 2703) 
s_add_i32 s49, s23, s22 // (op 2703) 


s_and_b32 s22, s1, s49 // (op 2703) 


s_add_i32 s1, s0, s22 // (op 2703) 


s_ashr_i32 s0, s1, s19 // (op 2703) 


s_lshl_b32 s1, s47, 24 // (op 2703) 
s_ashr_i32 s19, s1, 31 // (op 2703) 

s_xor_b32 s1, s0, s19 // (op 2703) 

s_sub_i32 s0, s1, s19 // (op 2703) 


s_mul_i32 s1, s11, s24 // (op 2703) 

s_mul_i32 s19, s8, s1 // (op 2703) 

s_mul_i32 s1, s0, s19 // (op 2703) 

s_sub_u32 s19, s16, s1 // (op 2703) 

s_mul_hi_i32 s1, s19, s12 // (op 2703) 
s_add_i32 s22, s1, s19 // (op 2703) 


s_ashr_i32 s1, s22, 31 // (op 2703) 
s_cmp_eq_i32 s12, 0 // (op 2703) 
s_mov_b32 s19, scc // (op 2703) 
s_cmp_eq_i32 s19, 0 // (op 2703) 
s_cselect_b32 s23, 0, -1 // (op 2703) 

s_and_b32 s19, s46, 31 // (op 2703) 
s_lshl_b32 s24, 1, s19 // (op 2703) 
s_add_i32 s25, s24, s23 // (op 2703) 


s_and_b32 s23, s1, s25 // (op 2703) 


s_add_i32 s1, s22, s23 // (op 2703) 


s_ashr_i32 s22, s1, s19 // (op 2703) 


s_lshl_b32 s1, s46, 24 // (op 2703) 
s_ashr_i32 s19, s1, 31 // (op 2703) 

s_xor_b32 s1, s22, s19 // (op 2703) 

s_sub_i32 s22, s1, s19 // (op 2703) 


s_cmp_eq_i32 s0, 0 // (op 2703) 
s_mov_b32 s1, scc // (op 2703) 

s_cmp_eq_i32 s1, 0 // (op 2703) 
s_cselect_b32 s0, s22, s18 // (op 2703) 



v_lshl_add_u32 v22, s0, 6, v21 // (op 2703) 


v_mul_lo_u32 v21, v18, s42 // (op 2703) 
v_add_u32 v21, v21, v22 // (op 2703) 


s_mul_i32 s1, s42, s41 // (op 2705) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 2705) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 2705) least significant
s_mul_hi_u32 s19, s42, s40 // (op 2705) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2705) most significant: sum
s_add_u32 s1, s1, s19 // (op 2705) most significant: sum


s_mov_b32 s72, s44 // (op 2705) reg expression
s_mov_b32 s73, s45 // (op 2705) reg expression
s_mov_b32 s74, s0 // (op 2705) reg expression
s_mov_b32 s75, 131072 // (op 2705) call()

s_mul_hi_i32 s0, s16, s10 // (op 2722) 
v_and_b32 v18, -64, v1 // (op 2722) 
s_add_i32 s1, s0, s16 // (op 2722) 

v_sub_u32 v22, v1, v18 // (op 2722) 

s_ashr_i32 s0, s1, 31 // (op 2722) 
v_lshrrev_b32 v18, 6, v1 // (op 2722) 
s_cmp_eq_i32 s10, 0 // (op 2722) 
s_mov_b32 s18, scc // (op 2722) 
s_cmp_eq_i32 s18, 0 // (op 2722) 
s_cselect_b32 s19, 0, -1 // (op 2722) 

s_and_b32 s18, s13, 31 // (op 2722) 
s_lshl_b32 s22, 1, s18 // (op 2722) 
s_add_i32 s23, s22, s19 // (op 2722) 


s_and_b32 s19, s0, s23 // (op 2722) 


s_add_i32 s0, s1, s19 // (op 2722) 


s_ashr_i32 s1, s0, s18 // (op 2722) 


s_lshl_b32 s0, s13, 24 // (op 2722) 
s_ashr_i32 s18, s0, 31 // (op 2722) 

s_xor_b32 s0, s1, s18 // (op 2722) 

s_sub_i32 s1, s0, s18 // (op 2722) 


s_mul_hi_i32 s0, s1, s14 // (op 2722) 
s_add_i32 s18, s0, s1 // (op 2722) 

s_ashr_i32 s0, s18, 31 // (op 2722) 
s_cmp_eq_i32 s14, 0 // (op 2722) 
s_mov_b32 s19, scc // (op 2722) 
s_cmp_eq_i32 s19, 0 // (op 2722) 
s_cselect_b32 s22, 0, -1 // (op 2722) 

s_and_b32 s19, s15, 31 // (op 2722) 
s_lshl_b32 s23, 1, s19 // (op 2722) 
s_add_i32 s24, s23, s22 // (op 2722) 


s_and_b32 s25, s0, s24 // (op 2722) 


s_add_i32 s0, s18, s25 // (op 2722) 


s_ashr_i32 s18, s0, s19 // (op 2722) 


s_lshl_b32 s0, s15, 24 // (op 2722) 
s_ashr_i32 s19, s0, 31 // (op 2722) 

s_xor_b32 s0, s18, s19 // (op 2722) 

s_sub_i32 s18, s0, s19 // (op 2722) 


s_add_u32 s50, s6, 64 // (op 2722) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2722) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2722) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2722) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2722) 

s_mul_i32 s0, s18, s24 // (op 2722) 

s_sub_i32 s18, s1, s0 // (op 2722) 


s_mul_hi_i32 s19, s16, s9 // (op 2722) 
s_add_i32 s0, s19, s16 // (op 2722) 

s_ashr_i32 s1, s0, 31 // (op 2722) 
s_cmp_eq_i32 s9, 0 // (op 2722) 
s_mov_b32 s19, scc // (op 2722) 
s_cmp_eq_i32 s19, 0 // (op 2722) 
s_cselect_b32 s22, 0, -1 // (op 2722) 

s_and_b32 s19, s47, 31 // (op 2722) 
s_lshl_b32 s23, 1, s19 // (op 2722) 
s_add_i32 s49, s23, s22 // (op 2722) 


s_and_b32 s22, s1, s49 // (op 2722) 


s_add_i32 s1, s0, s22 // (op 2722) 


s_ashr_i32 s0, s1, s19 // (op 2722) 


s_lshl_b32 s1, s47, 24 // (op 2722) 
s_ashr_i32 s19, s1, 31 // (op 2722) 

s_xor_b32 s1, s0, s19 // (op 2722) 

s_sub_i32 s0, s1, s19 // (op 2722) 


s_mul_i32 s1, s11, s24 // (op 2722) 

s_mul_i32 s19, s8, s1 // (op 2722) 

s_mul_i32 s1, s0, s19 // (op 2722) 

s_sub_u32 s19, s16, s1 // (op 2722) 

s_mul_hi_i32 s1, s19, s12 // (op 2722) 
s_add_i32 s22, s1, s19 // (op 2722) 


s_ashr_i32 s1, s22, 31 // (op 2722) 
s_cmp_eq_i32 s12, 0 // (op 2722) 
s_mov_b32 s19, scc // (op 2722) 
s_cmp_eq_i32 s19, 0 // (op 2722) 
s_cselect_b32 s23, 0, -1 // (op 2722) 

s_and_b32 s19, s46, 31 // (op 2722) 
s_lshl_b32 s24, 1, s19 // (op 2722) 
s_add_i32 s25, s24, s23 // (op 2722) 


s_and_b32 s23, s1, s25 // (op 2722) 


s_add_i32 s1, s22, s23 // (op 2722) 


s_ashr_i32 s22, s1, s19 // (op 2722) 


s_lshl_b32 s1, s46, 24 // (op 2722) 
s_ashr_i32 s19, s1, 31 // (op 2722) 

s_xor_b32 s1, s22, s19 // (op 2722) 

s_sub_i32 s22, s1, s19 // (op 2722) 


s_cmp_eq_i32 s0, 0 // (op 2722) 
s_mov_b32 s1, scc // (op 2722) 

s_cmp_eq_i32 s1, 0 // (op 2722) 
s_cselect_b32 s0, s22, s18 // (op 2722) 



v_lshl_add_u32 v23, s0, 6, v22 // (op 2722) 


v_mul_lo_u32 v22, v18, s42 // (op 2722) 
v_add_u32 v22, v22, v23 // (op 2722) 


s_mul_i32 s1, s42, s41 // (op 2724) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 2724) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 2724) least significant
s_mul_hi_u32 s19, s42, s40 // (op 2724) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2724) most significant: sum
s_add_u32 s1, s1, s19 // (op 2724) most significant: sum


s_mov_b32 s72, s44 // (op 2724) reg expression
s_mov_b32 s73, s45 // (op 2724) reg expression
s_mov_b32 s74, s0 // (op 2724) reg expression
s_mov_b32 s75, 131072 // (op 2724) call()



v_mov_b32 v18, v14 // (op 2249) reg expression

buffer_load_dwordx3 v[24:26], v18, s[64:67], 0 offen // (op 2249) Load value
buffer_load_dwordx3 v[28:30], v18, s[64:67], 0 offen offset:12 // (op 2249) Load value



v_mov_b32 v18, v21 // (op 2252) reg expression

buffer_load_ubyte v23, v18, s[72:75], 0 offen offset: 0 // (op 2252) Load value


s_lshl_b64 s[0:1], s[26:27], 2 // (op 2255) 
v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 2255) 


v_mov_b32 v18, v16 // (op 2255) reg expression
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 2255) Load value
v_add_u32 v18, v18, v32 // (op 2255) 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 2255) Load value
v_add_u32 v18, v18, v32 // (op 2255) 
buffer_load_dwordx4 v[42:45], v18, s[52:55], 0 offen // (op 2255) Load value
v_add_u32 v18, v18, v32 // (op 2255) 
buffer_load_dwordx4 v[46:49], v18, s[52:55], 0 offen // (op 2255) Load value




v_mov_b32 v18, v19 // (op 2258) reg expression

buffer_load_dword v27, v18, s[68:71], 0 offen // (op 2258) Load value



s_lshl_b64 s[0:1], s[32:33], 7 // (op 2261) 
s_mul_i32 s19, s1, 6 // (op 2261) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 2261) least significant
s_mul_hi_u32 s22, s0, 6 // (op 2261) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2261) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 2261) 

v_add_u32 v18, v14, s0 // (op 2261) 


buffer_load_dwordx3 v[50:52], v18, s[64:67], 0 offen // (op 2261) Load value
buffer_load_dwordx3 v[54:56], v18, s[64:67], 0 offen offset:12 // (op 2261) Load value




s_lshl_b32 s0, s42, 2 // (op 2264) 
v_add_u32 v14, v21, s0 // (op 2264) 


buffer_load_ubyte v18, v14, s[72:75], 0 offen offset: 0 // (op 2264) Load value



s_lshl_b64 s[0:1], s[26:27], 2 // (op 2267) 
v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 2267) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 2267) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 2267) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 2267) 

v_add_u32 v14, v16, s0 // (op 2267) 

buffer_load_dwordx4 v[58:61], v14, s[52:55], 0 offen // (op 2267) Load value
v_add_u32 v14, v14, v32 // (op 2267) 
buffer_load_dwordx4 v[62:65], v14, s[52:55], 0 offen // (op 2267) Load value
v_add_u32 v14, v14, v32 // (op 2267) 
buffer_load_dwordx4 v[66:69], v14, s[52:55], 0 offen // (op 2267) Load value
v_add_u32 v14, v14, v32 // (op 2267) 
buffer_load_dwordx4 v[70:73], v14, s[52:55], 0 offen // (op 2267) Load value





s_lshl_b32 s0, s36, 2 // (op 2270) 
v_add_u32 v14, v19, s0 // (op 2270) 


buffer_load_dword v16, v14, s[68:71], 0 offen // (op 2270) Load value


v_and_b32 v14, -2, v1 // (op 2847) 
v_sub_u32 v19, v1, v14 // (op 2847) 

v_lshlrev_b32 v14, 5, v19 // (op 2847) 

v_lshrrev_b32 v19, 1, v1 // (op 2847) 
v_lshl_add_u32 v21, v19, 6, v14 // (op 2847) 


v_mul_lo_u32 v14, v21, 6 // (op 2847) 
v_lshrrev_b32 v19, 3, v14 // (op 2847) 

v_lshrrev_b32 v14, 4, v21 // (op 2847) 

v_lshlrev_b32 v21, 2, v14 // (op 2847) 

v_add_u32 v14, v19, v21 // (op 2847) 





s_waitcnt vmcnt(15)
ds_write_b96 v14, v[24:26] // (op 2272) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[28:30] offset:16 // (op 2272) Store local data 

v_and_b32 v14, -64, v1 // (op 2958) 
v_sub_u32 v19, v1, v14 // (op 2958) 

v_and_b32 v14, 4294967232, v1 // (op 2958) 
v_add_u32 v21, v14, v19 // (op 2958) 





s_waitcnt vmcnt(13)
ds_write_b8 v21, v23 offset:8192 // (op 2273) Store local data 

v_and_b32 v14, -8, v1 // (op 2789) 
v_sub_u32 v19, v1, v14 // (op 2789) 

v_lshlrev_b32 v14, 5, v19 // (op 2789) 

v_lshrrev_b32 v19, 3, v1 // (op 2789) 
v_lshl_add_u32 v21, v19, 10, v14 // (op 2789) 


v_lshlrev_b32 v14, 2, v21 // (op 2789) 

v_lshrrev_b32 v19, 3, v14 // (op 2789) 




s_waitcnt vmcnt(12)
ds_write_b128 v19, v[34:37] offset:8448 // (op 2274) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v19, v[38:41] offset:8576 // (op 2274) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v19, v[42:45] offset:8704 // (op 2274) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v19, v[46:49] offset:8832 // (op 2274) Store local data 

v_and_b32 v14, -64, v1 // (op 2900) 
v_sub_u32 v19, v1, v14 // (op 2900) 

v_lshlrev_b32 v14, 2, v19 // (op 2900) 

v_lshrrev_b32 v19, 6, v1 // (op 2900) 
v_lshl_add_u32 v21, v19, 8, v14 // (op 2900) 





s_waitcnt vmcnt(8)
ds_write_b32 v21, v27 offset:24832 // (op 2275) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 3176) 



ds_read_b96_tr_b6 v[74:76], v3 // (op 2309) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v3 offset:1024 // (op 2309) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 3217) 


ds_read_u8 v14, v0 offset:8192 // (op 2332) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3157) 



ds_read_b64_tr_b4 v[80:81], v6 offset:8448 // (op 2290) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v6 offset:10496 // (op 2290) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3197) 


ds_read_u8 v19, v9 offset:24832 // (op 2313) Load local data 



ds_read_b64_tr_b4 v[84:85], v6 offset:8480 // (op 2295) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v6 offset:10528 // (op 2295) Transpose load local data 


ds_read_u8 v21, v9 offset:24896 // (op 2318) Load local data 



ds_read_b64_tr_b4 v[88:89], v6 offset:8512 // (op 2300) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v6 offset:10560 // (op 2300) Transpose load local data 


ds_read_u8 v53, v9 offset:24960 // (op 2323) Load local data 



ds_read_b64_tr_b4 v[92:93], v6 offset:8544 // (op 2305) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v6 offset:10592 // (op 2305) Transpose load local data 


ds_read_u8 v57, v9 offset:25024 // (op 2328) Load local data 
v_and_b32 v31, -8, v1 // (op 2760) 
v_sub_u32 v32, v1, v31 // (op 2760) 

v_lshlrev_b32 v31, 5, v32 // (op 2760) 

v_lshrrev_b32 v32, 3, v1 // (op 2760) 
v_lshl_add_u32 v33, v32, 10, v31 // (op 2760) 


v_lshlrev_b32 v31, 2, v33 // (op 2760) 

v_lshrrev_b32 v32, 3, v31 // (op 2760) 

v_and_b32 v31, -2, v1 // (op 2818) 
v_sub_u32 v33, v1, v31 // (op 2818) 

v_lshlrev_b32 v31, 5, v33 // (op 2818) 

v_lshrrev_b32 v33, 1, v1 // (op 2818) 
v_lshl_add_u32 v96, v33, 6, v31 // (op 2818) 


v_mul_lo_u32 v31, v96, 6 // (op 2818) 
v_lshrrev_b32 v33, 3, v31 // (op 2818) 

v_lshrrev_b32 v31, 4, v96 // (op 2818) 

v_lshlrev_b32 v96, 2, v31 // (op 2818) 

v_add_u32 v31, v33, v96 // (op 2818) 


v_and_b32 v33, -64, v1 // (op 2871) 
v_sub_u32 v96, v1, v33 // (op 2871) 

v_lshlrev_b32 v33, 2, v96 // (op 2871) 

v_lshrrev_b32 v96, 6, v1 // (op 2871) 
v_lshl_add_u32 v97, v96, 8, v33 // (op 2871) 


v_and_b32 v33, -64, v1 // (op 2929) 
v_sub_u32 v96, v1, v33 // (op 2929) 

v_and_b32 v33, 4294967232, v1 // (op 2929) 
v_add_u32 v98, v33, v96 // (op 2929) 


v_and_b32 v33, -8, v1 // (op 3000) 
v_sub_u32 v96, v1, v33 // (op 3000) 

v_lshlrev_b32 v33, 5, v96 // (op 3000) 

v_lshrrev_b32 v96, 3, v1 // (op 3000) 
v_lshl_add_u32 v99, v96, 10, v33 // (op 3000) 


v_lshlrev_b32 v33, 2, v99 // (op 3000) 

v_lshrrev_b32 v96, 3, v33 // (op 3000) 

v_and_b32 v33, -2, v1 // (op 3029) 
v_sub_u32 v99, v1, v33 // (op 3029) 

v_lshlrev_b32 v33, 5, v99 // (op 3029) 

v_lshrrev_b32 v99, 1, v1 // (op 3029) 
v_lshl_add_u32 v100, v99, 6, v33 // (op 3029) 


v_mul_lo_u32 v33, v100, 6 // (op 3029) 
v_lshrrev_b32 v99, 3, v33 // (op 3029) 

v_lshrrev_b32 v33, 4, v100 // (op 3029) 

v_lshlrev_b32 v100, 2, v33 // (op 3029) 

v_add_u32 v33, v99, v100 // (op 3029) 


v_and_b32 v99, -64, v1 // (op 3053) 
v_sub_u32 v100, v1, v99 // (op 3053) 

v_lshlrev_b32 v99, 2, v100 // (op 3053) 

v_lshrrev_b32 v100, 6, v1 // (op 3053) 
v_lshl_add_u32 v101, v100, 8, v99 // (op 3053) 


v_and_b32 v99, -64, v1 // (op 3082) 
v_sub_u32 v100, v1, v99 // (op 3082) 

v_and_b32 v99, 4294967232, v1 // (op 3082) 
v_add_u32 v102, v99, v100 // (op 3082) 


s_mov_b32 s0, 0 // (op 887) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 886) 
s_mov_b32 s1, s18 // (op 886) convert

s_and_b32 s18, s1, 4294967294 // (op 886) 

s_cmp_lt_u32 s0, s18 // (op 886) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoop_886 // (op 886) Condition: Top (jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoop_886 if false)
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoop_886:


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[80:83], v[74:79], a[0:15], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 90) 
s_add_u32 s0, s0, 2 // (op 888) 
v_mov_b32 v99, v15 // (op 10) 
s_lshl_b64 s[18:19], s[32:33], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v100, v99, s1 // (op 10) 


buffer_load_dwordx3 v[24:26], v100, s[64:67], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[28:30], v100, s[64:67], 0 offen offset:12 // (op 10) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 3166) 





ds_read_b64_tr_b4 v[80:81], v6 offset:16640 // (op 93) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v6 offset:18688 // (op 93) Transpose load local data 
v_mov_b32 v100, v22 // (op 25) 
s_barrier  // (op 3186) 
s_lshl_b32 s1, s42, 3 // (op 25) 



v_add_u32 v103, v100, s1 // (op 25) 


buffer_load_ubyte v23, v103, s[72:75], 0 offen offset: 0 // (op 25) Load value

ds_read_b96_tr_b6 v[104:106], v3 offset:4096 // (op 97) Transpose load local data 

s_lshl_b64 s[18:19], s[26:27], 2 // (op 4) 
v_lshrrev_b64 v[110:111], 3, s[18:19] // (op 4) 


v_mov_b32 v103, v17 // (op 4) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 4) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v112, v103, s1 // (op 4) 

ds_read_b96_tr_b6 v[107:109], v3 offset:5120 // (op 97) Transpose load local data 
buffer_load_dwordx4 v[34:37], v112, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v112, v112, v110 // (op 4) 
buffer_load_dwordx4 v[38:41], v112, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v112, v112, v110 // (op 4) 
buffer_load_dwordx4 v[42:45], v112, s[52:55], 0 offen // (op 4) Load value
v_add_u32 v112, v112, v110 // (op 4) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3207) 
buffer_load_dwordx4 v[46:49], v112, s[52:55], 0 offen // (op 4) Load value






ds_read_u8 v19, v9 offset:25344 // (op 101) Load local data 
v_mov_b32 v110, v20 // (op 17) 
s_barrier  // (op 3227) 
s_lshl_b32 s1, s36, 3 // (op 17) 


ds_read_u8 v111, v0 offset:8320 // (op 105) Load local data 
v_add_u32 v112, v110, s1 // (op 17) 


buffer_load_dword v27, v112, s[68:71], 0 offen // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[80:83], v[104:109], a[0:15], v19, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 107) 
s_barrier  // (op 3247) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[84:87], v[74:79], a[16:31], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 454) 
s_waitcnt vmcnt(15)
ds_write_b96 v33, v[50:52] offset:25856 // (op 306) Store local data 



ds_read_b64_tr_b4 v[80:81], v6 offset:16672 // (op 456) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v6 offset:18720 // (op 456) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b96 v33, v[54:56] offset:25872 // (op 306) Store local data 


ds_read_u8 v19, v9 offset:25408 // (op 460) Load local data 
s_barrier  // (op 3259) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[80:83], v[104:109], a[16:31], v19, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 463) 



s_waitcnt vmcnt(13)
ds_write_b8 v102, v18 offset:34048 // (op 308) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[88:91], v[74:79], a[32:47], v53, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 694) 



s_barrier  // (op 3240) 
ds_read_b64_tr_b4 v[80:81], v6 offset:16704 // (op 696) Transpose load local data 



ds_read_b64_tr_b4 v[82:83], v6 offset:18752 // (op 696) Transpose load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v96, v[58:61] offset:34304 // (op 287) Store local data 


ds_read_u8 v19, v9 offset:25472 // (op 700) Load local data 
s_waitcnt vmcnt(11)
ds_write_b128 v96, v[62:65] offset:34432 // (op 287) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[80:83], v[104:109], a[32:47], v19, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 703) 
s_waitcnt vmcnt(10)
ds_write_b128 v96, v[66:69] offset:34560 // (op 287) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v96, v[70:73] offset:34688 // (op 287) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[92:95], v[74:79], a[48:63], v57, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 934) 



s_barrier  // (op 3253) 
ds_read_b64_tr_b4 v[92:93], v6 offset:16736 // (op 936) Transpose load local data 



ds_read_b64_tr_b4 v[94:95], v6 offset:18784 // (op 936) Transpose load local data 
s_waitcnt vmcnt(8)
ds_write_b32 v101, v16 offset:50688 // (op 307) Store local data 


ds_read_u8 v57, v9 offset:25536 // (op 940) Load local data 
s_barrier  // (op 3243) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[92:95], v[104:109], a[48:63], v57, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 943) 





ds_read_b96_tr_b6 v[74:76], v5 offset:25856 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v5 offset:26880 // (op 291) Transpose load local data 
s_barrier  // (op 3256) 


ds_read_u8 v14, v4 offset:34048 // (op 295) Load local data 
s_barrier  // (op 3237) 



ds_read_b64_tr_b4 v[80:81], v12 offset:34304 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v12 offset:36352 // (op 289) Transpose load local data 
s_barrier  // (op 3250) 


ds_read_u8 v19, v13 offset:50688 // (op 293) Load local data 



ds_read_b64_tr_b4 v[84:85], v12 offset:34336 // (op 475) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v12 offset:36384 // (op 475) Transpose load local data 


ds_read_u8 v21, v13 offset:50752 // (op 479) Load local data 



ds_read_b64_tr_b4 v[88:89], v12 offset:34368 // (op 715) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v12 offset:36416 // (op 715) Transpose load local data 


ds_read_u8 v53, v13 offset:50816 // (op 719) Load local data 



ds_read_b64_tr_b4 v[104:105], v12 offset:34400 // (op 955) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v12 offset:36448 // (op 955) Transpose load local data 


ds_read_u8 v108, v13 offset:50880 // (op 959) Load local data 


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[80:83], v[74:79], a[0:15], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 296) 
s_lshl_b64 s[18:19], s[32:33], 7 // (op 309) 
s_mul_i32 s51, s19, 6 // (op 309) most significant: low of high * low



s_mul_i32 s50, s18, 6 // (op 309) least significant
s_mul_hi_u32 s1, s18, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 309) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 309) 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[84:87], v[74:79], a[16:31], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 464) 
s_mul_i32 s1, 3, s18 // (op 309) 

v_add_u32 v109, v99, s1 // (op 309) 


buffer_load_dwordx3 v[50:52], v109, s[64:67], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[54:56], v109, s[64:67], 0 offen offset:12 // (op 309) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[88:91], v[74:79], a[32:47], v53, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 704) 



s_lshl_b32 s1, s42, 2 // (op 311) 
s_mul_i32 s18, 3, s1 // (op 311) 

v_add_u32 v99, v100, s18 // (op 311) 


buffer_load_ubyte v18, v99, s[72:75], 0 offen offset: 0 // (op 311) Load value

s_lshl_b64 s[18:19], s[32:33], 7 // (op 2586) 

s_mul_i32 s51, s19, 6 // (op 2586) most significant: low of high * low

s_mul_i32 s50, s18, 6 // (op 2586) least significant
s_mul_hi_u32 s1, s18, 6 // (op 2586) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 2586) most significant: sum


s_lshl_b32 s1, s42, 3 // (op 2714) 
v_add_u32 v22, v22, s1 // (op 2714) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 2586) 

s_lshl_b32 s1, s18, 1 // (op 2586) 

v_add_u32 v15, v15, s1 // (op 2586) 

s_lshl_b64 s[18:19], s[26:27], 2 // (op 286) 
v_lshrrev_b64 v[112:113], 3, s[18:19] // (op 286) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 286) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 286) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 286) 

s_mul_i32 s1, 3, s18 // (op 286) 

v_add_u32 v109, v103, s1 // (op 286) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[104:107], v[74:79], a[48:63], v108, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 944) 
ds_read_b64_tr_b4 v[114:115], v12 offset:42496 // (op 298) Transpose load local data 


ds_read_b64_tr_b4 v[116:117], v12 offset:44544 // (op 298) Transpose load local data 
buffer_load_dwordx4 v[58:61], v109, s[52:55], 0 offen // (op 286) Load value



v_add_u32 v109, v109, v112 // (op 286) 
buffer_load_dwordx4 v[62:65], v109, s[52:55], 0 offen // (op 286) Load value
v_add_u32 v109, v109, v112 // (op 286) 
ds_read_b96_tr_b6 v[118:120], v5 offset:29952 // (op 300) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v5 offset:30976 // (op 300) Transpose load local data 
buffer_load_dwordx4 v[66:69], v109, s[52:55], 0 offen // (op 286) Load value
v_add_u32 v109, v109, v112 // (op 286) 


s_barrier  // (op 3308) 
ds_read_u8 v111, v13 offset:51200 // (op 302) Load local data 
buffer_load_dwordx4 v[70:73], v109, s[52:55], 0 offen // (op 286) Load value





ds_read_u8 v99, v4 offset:34176 // (op 304) Load local data 


s_lshl_b32 s1, s36, 2 // (op 310) 



s_barrier  // (op 3312) 
s_mul_i32 s18, 3, s1 // (op 310) 

v_add_u32 v100, v110, s18 // (op 310) 


buffer_load_dword v16, v100, s[68:71], 0 offen // (op 310) Load value

s_lshl_b64 s[18:19], s[26:27], 7 // (op 2522) 

s_lshl_b64 s[50:51], s[18:19], 2 // (op 2522) 

s_barrier  // (op 3179) 
s_lshl_b32 s1, s36, 3 // (op 2650) 



v_add_u32 v20, v20, s1 // (op 2650) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 2522) 

s_lshl_b32 s1, s18, 1 // (op 2522) 

v_add_u32 v17, v17, s1 // (op 2522) 

ds_read_b64_tr_b4 v[104:105], v12 offset:42528 // (op 466) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v12 offset:44576 // (op 466) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[114:117], v[118:123], a[0:15], v111, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 305) 




ds_read_u8 v100, v13 offset:51264 // (op 470) Load local data 
s_waitcnt vmcnt(15)
ds_write_b96 v31, v[24:26] // (op 13) Store local data 



ds_read_b64_tr_b4 v[108:109], v12 offset:42560 // (op 706) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[104:107], v[118:123], a[16:31], v100, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 473) 
ds_read_b64_tr_b4 v[110:111], v12 offset:44608 // (op 706) Transpose load local data 




ds_read_u8 v100, v13 offset:51328 // (op 710) Load local data 
s_waitcnt vmcnt(14)
ds_write_b96 v31, v[28:30] offset:16 // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[108:111], v[118:123], a[32:47], v100, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 713) 


s_barrier  // (op 3220) 
ds_read_b64_tr_b4 v[114:115], v12 offset:42592 // (op 946) Transpose load local data 



ds_read_b64_tr_b4 v[116:117], v12 offset:44640 // (op 946) Transpose load local data 
s_waitcnt vmcnt(13)
ds_write_b8 v98, v23 offset:8192 // (op 30) Store local data 


s_barrier  // (op 3368) 
s_barrier  // (op 3160) 
ds_read_u8 v100, v13 offset:51392 // (op 950) Load local data 



s_waitcnt vmcnt(12)
ds_write_b128 v32, v[34:37] offset:8448 // (op 6) Store local data 
s_barrier  // (op 3372) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[114:117], v[118:123], a[48:63], v100, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 953) 
s_waitcnt vmcnt(11)
ds_write_b128 v32, v[38:41] offset:8576 // (op 6) Store local data 




s_waitcnt vmcnt(10)
ds_write_b128 v32, v[42:45] offset:8704 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v32, v[46:49] offset:8832 // (op 6) Store local data 
s_barrier  // (op 3200) 



s_waitcnt vmcnt(8)
ds_write_b32 v97, v27 offset:24832 // (op 21) Store local data 
s_barrier  // (op 3189) 



ds_read_b96_tr_b6 v[74:76], v3 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v3 offset:1024 // (op 77) Transpose load local data 
s_barrier  // (op 3230) 


ds_read_u8 v14, v0 offset:8192 // (op 87) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3169) 



ds_read_b64_tr_b4 v[80:81], v6 offset:8448 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v6 offset:10496 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3210) 


ds_read_u8 v19, v9 offset:24832 // (op 82) Load local data 



ds_read_b64_tr_b4 v[84:85], v6 offset:8480 // (op 447) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v6 offset:10528 // (op 447) Transpose load local data 


ds_read_u8 v21, v9 offset:24896 // (op 451) Load local data 



ds_read_b64_tr_b4 v[88:89], v6 offset:8512 // (op 687) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v6 offset:10560 // (op 687) Transpose load local data 


ds_read_u8 v53, v9 offset:24960 // (op 691) Load local data 



ds_read_b64_tr_b4 v[92:93], v6 offset:8544 // (op 927) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v6 offset:10592 // (op 927) Transpose load local data 


ds_read_u8 v57, v9 offset:25024 // (op 931) Load local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 886) 
s_mov_b32 s1, s18 // (op 886) convert

s_and_b32 s18, s1, 4294967294 // (op 886) 

s_cmp_lt_u32 s0, s18 // (op 886) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoop_886 // (op 886) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoop_886 if true)
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoop_886:

















s_mul_hi_i32 s0, s16, s10 // (op 2482) 
v_and_b32 v15, -8, v1 // (op 2482) 
s_add_i32 s1, s0, s16 // (op 2482) 

v_sub_u32 v17, v1, v15 // (op 2482) 

s_ashr_i32 s0, s1, 31 // (op 2482) 
v_lshlrev_b32 v15, 5, v17 // (op 2482) 

s_cmp_eq_i32 s10, 0 // (op 2482) 
s_mov_b32 s18, scc // (op 2482) 
v_lshrrev_b32 v17, 3, v1 // (op 2482) 
s_cmp_eq_i32 s18, 0 // (op 2482) 
s_cselect_b32 s19, 0, -1 // (op 2482) 

v_lshlrev_b32 v20, 2, v17 // (op 2482) 

s_and_b32 s18, s13, 31 // (op 2482) 
s_lshl_b32 s22, 1, s18 // (op 2482) 
s_add_i32 s23, s22, s19 // (op 2482) 


s_and_b32 s19, s0, s23 // (op 2482) 


s_add_i32 s0, s1, s19 // (op 2482) 


s_ashr_i32 s1, s0, s18 // (op 2482) 


s_lshl_b32 s0, s13, 24 // (op 2482) 
s_ashr_i32 s18, s0, 31 // (op 2482) 

s_xor_b32 s0, s1, s18 // (op 2482) 

s_sub_i32 s1, s0, s18 // (op 2482) 


s_mul_hi_i32 s0, s1, s14 // (op 2482) 
s_add_i32 s18, s0, s1 // (op 2482) 

s_ashr_i32 s0, s18, 31 // (op 2482) 
s_cmp_eq_i32 s14, 0 // (op 2482) 
s_mov_b32 s19, scc // (op 2482) 
s_cmp_eq_i32 s19, 0 // (op 2482) 
s_cselect_b32 s22, 0, -1 // (op 2482) 

s_and_b32 s19, s15, 31 // (op 2482) 
s_lshl_b32 s23, 1, s19 // (op 2482) 
s_add_i32 s24, s23, s22 // (op 2482) 


s_and_b32 s25, s0, s24 // (op 2482) 


s_add_i32 s0, s18, s25 // (op 2482) 


s_ashr_i32 s18, s0, s19 // (op 2482) 


s_lshl_b32 s0, s15, 24 // (op 2482) 
s_ashr_i32 s19, s0, 31 // (op 2482) 

s_xor_b32 s0, s18, s19 // (op 2482) 

s_sub_i32 s18, s0, s19 // (op 2482) 


s_mul_i32 s0, s1, s11 // (op 2482) 

s_sub_u32 s1, s16, s0 // (op 2482) 

s_mul_i32 s0, s18, s11 // (op 2482) 
s_add_u32 s0, s0, s1 // (op 2482) 


s_mul_hi_i32 s1, s16, s9 // (op 2482) 
s_add_i32 s18, s1, s16 // (op 2482) 

s_ashr_i32 s1, s18, 31 // (op 2482) 
s_cmp_eq_i32 s9, 0 // (op 2482) 
s_mov_b32 s19, scc // (op 2482) 
s_cmp_eq_i32 s19, 0 // (op 2482) 
s_cselect_b32 s22, 0, -1 // (op 2482) 

s_and_b32 s19, s47, 31 // (op 2482) 
s_lshl_b32 s23, 1, s19 // (op 2482) 
s_add_i32 s24, s23, s22 // (op 2482) 


s_and_b32 s25, s1, s24 // (op 2482) 


s_add_i32 s1, s18, s25 // (op 2482) 


s_ashr_i32 s18, s1, s19 // (op 2482) 


s_lshl_b32 s1, s47, 24 // (op 2482) 
s_ashr_i32 s19, s1, 31 // (op 2482) 

s_xor_b32 s1, s18, s19 // (op 2482) 

s_sub_i32 s18, s1, s19 // (op 2482) 


s_add_u32 s50, s6, 64 // (op 2482) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2482) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2482) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2482) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2482) 

s_mul_i32 s1, s11, s24 // (op 2482) 

s_mul_i32 s19, s8, s1 // (op 2482) 

s_mul_i32 s1, s18, s19 // (op 2482) 

s_sub_u32 s19, s16, s1 // (op 2482) 

s_mov_b32 s1, s19 // (op 2482) 
s_mul_hi_i32 s22, s1, s12 // (op 2482) 
s_add_i32 s23, s22, s1 // (op 2482) 


s_ashr_i32 s1, s23, 31 // (op 2482) 
s_cmp_eq_i32 s12, 0 // (op 2482) 
s_mov_b32 s22, scc // (op 2482) 
s_cmp_eq_i32 s22, 0 // (op 2482) 
s_cselect_b32 s24, 0, -1 // (op 2482) 

s_and_b32 s22, s46, 31 // (op 2482) 
s_lshl_b32 s25, 1, s22 // (op 2482) 
s_add_i32 s49, s25, s24 // (op 2482) 


s_and_b32 s24, s1, s49 // (op 2482) 


s_add_i32 s1, s23, s24 // (op 2482) 


s_ashr_i32 s23, s1, s22 // (op 2482) 


s_lshl_b32 s1, s46, 24 // (op 2482) 
s_ashr_i32 s22, s1, 31 // (op 2482) 

s_xor_b32 s1, s23, s22 // (op 2482) 

s_sub_i32 s23, s1, s22 // (op 2482) 


s_mul_i32 s1, s48, s11 // (op 2482) 
s_add_u32 s24, s4, 256 // (op 2482) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2482) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2482) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2482) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 2482) 

s_sub_i32 s22, s24, s1 // (op 2482) 

s_mul_i32 s24, s23, s22 // (op 2482) 


s_sub_u32 s25, s19, s24 // (op 2482) 


s_add_u32 s19, s25, s1 // (op 2482) 


s_cmp_eq_i32 s18, 0 // (op 2482) 
s_mov_b32 s1, scc // (op 2482) 

s_cmp_eq_i32 s1, 0 // (op 2482) 
s_cselect_b32 s18, s19, s0 // (op 2482) 



v_lshl_add_u32 v17, s18, 8, v15 // (op 2482) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2482) 
v_mov_b32 v23, 0 // (op 2482) convert
v_mov_b32 v22, v17 // (op 2482) convert

s_lshr_b32 s18, s0, 1 // (op 2482) 

v_lshl_add_u32 v15, s18, 8, v20 // (op 2482) 


v_mul_lo_u32 v25, v15, s27 // (op 2482) most significant: low of low * high
v_mul_hi_u32 v17, v15, s26 // (op 2482) most significant: high of low * low
v_mul_lo_u32 v24, v15, s26 // (op 2482) least significant: low of low * low
v_add_u32 v25, v25, v17 // (op 2482) most significant: sum


v_add_co_u32 v26, s[0:1], v22, v24 // (op 2482) least significant half
v_addc_co_u32 v27, s[0:1], v23, v25, s[0:1] // (op 2482) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 2482) 


v_lshrrev_b64 v[22:23], 3, v[26:27] // (op 2482) 

v_mov_b32 v15, v22 // (op 2482) convert

s_mul_i32 s1, s26, s31 // (op 2484) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2484) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2484) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2484) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2484) most significant: sum
s_add_u32 s1, s1, s19 // (op 2484) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2484) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2484) 

s_mov_b32 s52, s28 // (op 2484) reg expression
s_mov_b32 s53, s29 // (op 2484) reg expression
s_mov_b32 s54, s0 // (op 2484) reg expression
s_mov_b32 s55, 131072 // (op 2484) call()

s_mul_hi_i32 s0, s16, s10 // (op 2552) 
v_and_b32 v17, -2, v1 // (op 2552) 
s_add_i32 s1, s0, s16 // (op 2552) 

v_sub_u32 v20, v1, v17 // (op 2552) 

s_ashr_i32 s0, s1, 31 // (op 2552) 
v_lshlrev_b32 v17, 5, v20 // (op 2552) 

s_cmp_eq_i32 s10, 0 // (op 2552) 
s_mov_b32 s18, scc // (op 2552) 
v_lshrrev_b32 v20, 1, v1 // (op 2552) 
s_cmp_eq_i32 s18, 0 // (op 2552) 
s_cselect_b32 s19, 0, -1 // (op 2552) 

s_and_b32 s18, s13, 31 // (op 2552) 
s_lshl_b32 s22, 1, s18 // (op 2552) 
s_add_i32 s23, s22, s19 // (op 2552) 


s_and_b32 s19, s0, s23 // (op 2552) 


s_add_i32 s0, s1, s19 // (op 2552) 


s_ashr_i32 s1, s0, s18 // (op 2552) 


s_lshl_b32 s0, s13, 24 // (op 2552) 
s_ashr_i32 s18, s0, 31 // (op 2552) 

s_xor_b32 s0, s1, s18 // (op 2552) 

s_sub_i32 s1, s0, s18 // (op 2552) 


s_mul_hi_i32 s0, s1, s14 // (op 2552) 
s_add_i32 s18, s0, s1 // (op 2552) 

s_ashr_i32 s0, s18, 31 // (op 2552) 
s_cmp_eq_i32 s14, 0 // (op 2552) 
s_mov_b32 s19, scc // (op 2552) 
s_cmp_eq_i32 s19, 0 // (op 2552) 
s_cselect_b32 s22, 0, -1 // (op 2552) 

s_and_b32 s19, s15, 31 // (op 2552) 
s_lshl_b32 s23, 1, s19 // (op 2552) 
s_add_i32 s24, s23, s22 // (op 2552) 


s_and_b32 s25, s0, s24 // (op 2552) 


s_add_i32 s0, s18, s25 // (op 2552) 


s_ashr_i32 s18, s0, s19 // (op 2552) 


s_lshl_b32 s0, s15, 24 // (op 2552) 
s_ashr_i32 s19, s0, 31 // (op 2552) 

s_xor_b32 s0, s18, s19 // (op 2552) 

s_sub_i32 s18, s0, s19 // (op 2552) 


s_add_u32 s50, s6, 64 // (op 2552) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2552) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2552) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2552) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2552) 

s_mul_i32 s0, s18, s24 // (op 2552) 

s_sub_i32 s18, s1, s0 // (op 2552) 


s_mul_hi_i32 s19, s16, s9 // (op 2552) 
s_add_i32 s0, s19, s16 // (op 2552) 

s_ashr_i32 s1, s0, 31 // (op 2552) 
s_cmp_eq_i32 s9, 0 // (op 2552) 
s_mov_b32 s19, scc // (op 2552) 
s_cmp_eq_i32 s19, 0 // (op 2552) 
s_cselect_b32 s22, 0, -1 // (op 2552) 

s_and_b32 s19, s47, 31 // (op 2552) 
s_lshl_b32 s23, 1, s19 // (op 2552) 
s_add_i32 s49, s23, s22 // (op 2552) 


s_and_b32 s22, s1, s49 // (op 2552) 


s_add_i32 s1, s0, s22 // (op 2552) 


s_ashr_i32 s0, s1, s19 // (op 2552) 


s_lshl_b32 s1, s47, 24 // (op 2552) 
s_ashr_i32 s19, s1, 31 // (op 2552) 

s_xor_b32 s1, s0, s19 // (op 2552) 

s_sub_i32 s0, s1, s19 // (op 2552) 


s_mul_i32 s1, s11, s24 // (op 2552) 

s_mul_i32 s19, s8, s1 // (op 2552) 

s_mul_i32 s1, s0, s19 // (op 2552) 

s_sub_u32 s19, s16, s1 // (op 2552) 

s_mul_hi_i32 s1, s19, s12 // (op 2552) 
s_add_i32 s22, s1, s19 // (op 2552) 


s_ashr_i32 s1, s22, 31 // (op 2552) 
s_cmp_eq_i32 s12, 0 // (op 2552) 
s_mov_b32 s19, scc // (op 2552) 
s_cmp_eq_i32 s19, 0 // (op 2552) 
s_cselect_b32 s23, 0, -1 // (op 2552) 

s_and_b32 s19, s46, 31 // (op 2552) 
s_lshl_b32 s24, 1, s19 // (op 2552) 
s_add_i32 s25, s24, s23 // (op 2552) 


s_and_b32 s23, s1, s25 // (op 2552) 


s_add_i32 s1, s22, s23 // (op 2552) 


s_ashr_i32 s22, s1, s19 // (op 2552) 


s_lshl_b32 s1, s46, 24 // (op 2552) 
s_ashr_i32 s19, s1, 31 // (op 2552) 

s_xor_b32 s1, s22, s19 // (op 2552) 

s_sub_i32 s22, s1, s19 // (op 2552) 


s_cmp_eq_i32 s0, 0 // (op 2552) 
s_mov_b32 s1, scc // (op 2552) 

s_cmp_eq_i32 s1, 0 // (op 2552) 
s_cselect_b32 s0, s22, s18 // (op 2552) 



v_lshl_add_u32 v22, s0, 6, v17 // (op 2552) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2552) 
v_mov_b32 v49, 0 // (op 2552) convert
v_mov_b32 v48, v22 // (op 2552) convert

s_lshr_b32 s18, s0, 1 // (op 2552) 

v_lshl_add_u32 v17, s18, 8, v20 // (op 2552) 


v_mul_lo_u32 v23, v17, s33 // (op 2552) most significant: low of low * high
v_mul_hi_u32 v20, v17, s32 // (op 2552) most significant: high of low * low
v_mul_lo_u32 v22, v17, s32 // (op 2552) least significant: low of low * low
v_add_u32 v23, v23, v20 // (op 2552) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v48 // (op 2552) least significant half
v_addc_co_u32 v23, s[0:1], v23, v49, s[0:1] // (op 2552) most significant half



v_mul_lo_u32 v25, v23, 6 // (op 2552) most significant: low of high * low
v_mul_hi_u32 v17, v22, 6 // (op 2552) most significant: high of low * low
v_mul_lo_u32 v24, v22, 6 // (op 2552) least significant: low of low * low
v_add_u32 v25, v25, v17 // (op 2552) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 2552) 

v_mov_b32 v17, v22 // (op 2552) convert

s_mul_i32 s1, s31, 6 // (op 2554) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2554) least significant
s_mul_hi_u32 s18, s30, 6 // (op 2554) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2554) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2554) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2554) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2554) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2554) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2554) most significant: sum
s_add_u32 s19, s19, s23 // (op 2554) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2554) 

s_mov_b32 s64, s34 // (op 2554) reg expression
s_mov_b32 s65, s35 // (op 2554) reg expression
s_mov_b32 s66, s0 // (op 2554) reg expression
s_mov_b32 s67, 131072 // (op 2554) call()

s_mul_hi_i32 s0, s16, s10 // (op 2616) 
v_and_b32 v20, -64, v1 // (op 2616) 
s_add_i32 s1, s0, s16 // (op 2616) 

v_sub_u32 v22, v1, v20 // (op 2616) 

s_ashr_i32 s0, s1, 31 // (op 2616) 
v_lshlrev_b32 v20, 2, v22 // (op 2616) 

s_cmp_eq_i32 s10, 0 // (op 2616) 
s_mov_b32 s18, scc // (op 2616) 
v_lshrrev_b32 v22, 6, v1 // (op 2616) 
s_cmp_eq_i32 s18, 0 // (op 2616) 
s_cselect_b32 s19, 0, -1 // (op 2616) 

s_and_b32 s18, s13, 31 // (op 2616) 
s_lshl_b32 s22, 1, s18 // (op 2616) 
s_add_i32 s23, s22, s19 // (op 2616) 


s_and_b32 s19, s0, s23 // (op 2616) 


s_add_i32 s0, s1, s19 // (op 2616) 


s_ashr_i32 s1, s0, s18 // (op 2616) 


s_lshl_b32 s0, s13, 24 // (op 2616) 
s_ashr_i32 s18, s0, 31 // (op 2616) 

s_xor_b32 s0, s1, s18 // (op 2616) 

s_sub_i32 s1, s0, s18 // (op 2616) 


s_mul_hi_i32 s0, s1, s14 // (op 2616) 
s_add_i32 s18, s0, s1 // (op 2616) 

s_ashr_i32 s0, s18, 31 // (op 2616) 
s_cmp_eq_i32 s14, 0 // (op 2616) 
s_mov_b32 s19, scc // (op 2616) 
s_cmp_eq_i32 s19, 0 // (op 2616) 
s_cselect_b32 s22, 0, -1 // (op 2616) 

s_and_b32 s19, s15, 31 // (op 2616) 
s_lshl_b32 s23, 1, s19 // (op 2616) 
s_add_i32 s24, s23, s22 // (op 2616) 


s_and_b32 s25, s0, s24 // (op 2616) 


s_add_i32 s0, s18, s25 // (op 2616) 


s_ashr_i32 s18, s0, s19 // (op 2616) 


s_lshl_b32 s0, s15, 24 // (op 2616) 
s_ashr_i32 s19, s0, 31 // (op 2616) 

s_xor_b32 s0, s18, s19 // (op 2616) 

s_sub_i32 s18, s0, s19 // (op 2616) 


s_mul_i32 s0, s1, s11 // (op 2616) 

s_sub_u32 s1, s16, s0 // (op 2616) 

s_mul_i32 s0, s18, s11 // (op 2616) 
s_add_u32 s0, s0, s1 // (op 2616) 


s_mul_hi_i32 s1, s16, s9 // (op 2616) 
s_add_i32 s18, s1, s16 // (op 2616) 

s_ashr_i32 s1, s18, 31 // (op 2616) 
s_cmp_eq_i32 s9, 0 // (op 2616) 
s_mov_b32 s19, scc // (op 2616) 
s_cmp_eq_i32 s19, 0 // (op 2616) 
s_cselect_b32 s22, 0, -1 // (op 2616) 

s_and_b32 s19, s47, 31 // (op 2616) 
s_lshl_b32 s23, 1, s19 // (op 2616) 
s_add_i32 s24, s23, s22 // (op 2616) 


s_and_b32 s25, s1, s24 // (op 2616) 


s_add_i32 s1, s18, s25 // (op 2616) 


s_ashr_i32 s18, s1, s19 // (op 2616) 


s_lshl_b32 s1, s47, 24 // (op 2616) 
s_ashr_i32 s19, s1, 31 // (op 2616) 

s_xor_b32 s1, s18, s19 // (op 2616) 

s_sub_i32 s18, s1, s19 // (op 2616) 


s_add_u32 s50, s6, 64 // (op 2616) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2616) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2616) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2616) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2616) 

s_mul_i32 s1, s11, s24 // (op 2616) 

s_mul_i32 s19, s8, s1 // (op 2616) 

s_mul_i32 s1, s18, s19 // (op 2616) 

s_sub_u32 s19, s16, s1 // (op 2616) 

s_mov_b32 s1, s19 // (op 2616) 
s_mul_hi_i32 s22, s1, s12 // (op 2616) 
s_add_i32 s23, s22, s1 // (op 2616) 


s_ashr_i32 s1, s23, 31 // (op 2616) 
s_cmp_eq_i32 s12, 0 // (op 2616) 
s_mov_b32 s22, scc // (op 2616) 
s_cmp_eq_i32 s22, 0 // (op 2616) 
s_cselect_b32 s24, 0, -1 // (op 2616) 

s_and_b32 s22, s46, 31 // (op 2616) 
s_lshl_b32 s25, 1, s22 // (op 2616) 
s_add_i32 s49, s25, s24 // (op 2616) 


s_and_b32 s24, s1, s49 // (op 2616) 


s_add_i32 s1, s23, s24 // (op 2616) 


s_ashr_i32 s23, s1, s22 // (op 2616) 


s_lshl_b32 s1, s46, 24 // (op 2616) 
s_ashr_i32 s22, s1, 31 // (op 2616) 

s_xor_b32 s1, s23, s22 // (op 2616) 

s_sub_i32 s23, s1, s22 // (op 2616) 


s_mul_i32 s1, s48, s11 // (op 2616) 
s_add_u32 s24, s4, 256 // (op 2616) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 2616) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2616) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2616) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 2616) 

s_sub_i32 s22, s24, s1 // (op 2616) 

s_mul_i32 s24, s23, s22 // (op 2616) 


s_sub_u32 s25, s19, s24 // (op 2616) 


s_add_u32 s19, s25, s1 // (op 2616) 


s_cmp_eq_i32 s18, 0 // (op 2616) 
s_mov_b32 s1, scc // (op 2616) 

s_cmp_eq_i32 s1, 0 // (op 2616) 
s_cselect_b32 s18, s19, s0 // (op 2616) 



v_lshl_add_u32 v23, s18, 8, v20 // (op 2616) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2616) 
s_lshr_b32 s18, s0, 1 // (op 2616) 

v_lshl_add_u32 v20, s18, 3, v22 // (op 2616) 


v_mul_lo_u32 v22, v20, s36 // (op 2616) 

v_add_u32 v20, v23, v22 // (op 2616) 


s_mul_i32 s1, s36, s41 // (op 2618) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2618) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2618) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2618) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2618) most significant: sum
s_add_u32 s1, s1, s19 // (op 2618) most significant: sum


s_mov_b32 s68, s38 // (op 2618) reg expression
s_mov_b32 s69, s39 // (op 2618) reg expression
s_mov_b32 s70, s0 // (op 2618) reg expression
s_mov_b32 s71, 131072 // (op 2618) call()

s_mul_hi_i32 s0, s16, s10 // (op 2680) 
v_and_b32 v22, -64, v1 // (op 2680) 
s_add_i32 s1, s0, s16 // (op 2680) 

v_sub_u32 v23, v1, v22 // (op 2680) 

s_ashr_i32 s0, s1, 31 // (op 2680) 
v_lshrrev_b32 v22, 6, v1 // (op 2680) 
s_cmp_eq_i32 s10, 0 // (op 2680) 
s_mov_b32 s18, scc // (op 2680) 
s_cmp_eq_i32 s18, 0 // (op 2680) 
s_cselect_b32 s19, 0, -1 // (op 2680) 

s_and_b32 s18, s13, 31 // (op 2680) 
s_lshl_b32 s22, 1, s18 // (op 2680) 
s_add_i32 s23, s22, s19 // (op 2680) 


s_and_b32 s19, s0, s23 // (op 2680) 


s_add_i32 s0, s1, s19 // (op 2680) 


s_ashr_i32 s1, s0, s18 // (op 2680) 


s_lshl_b32 s0, s13, 24 // (op 2680) 
s_ashr_i32 s18, s0, 31 // (op 2680) 

s_xor_b32 s0, s1, s18 // (op 2680) 

s_sub_i32 s1, s0, s18 // (op 2680) 


s_mul_hi_i32 s0, s1, s14 // (op 2680) 
s_add_i32 s18, s0, s1 // (op 2680) 

s_ashr_i32 s0, s18, 31 // (op 2680) 
s_cmp_eq_i32 s14, 0 // (op 2680) 
s_mov_b32 s19, scc // (op 2680) 
s_cmp_eq_i32 s19, 0 // (op 2680) 
s_cselect_b32 s22, 0, -1 // (op 2680) 

s_and_b32 s19, s15, 31 // (op 2680) 
s_lshl_b32 s23, 1, s19 // (op 2680) 
s_add_i32 s24, s23, s22 // (op 2680) 


s_and_b32 s25, s0, s24 // (op 2680) 


s_add_i32 s0, s18, s25 // (op 2680) 


s_ashr_i32 s18, s0, s19 // (op 2680) 


s_lshl_b32 s0, s15, 24 // (op 2680) 
s_ashr_i32 s19, s0, 31 // (op 2680) 

s_xor_b32 s0, s18, s19 // (op 2680) 

s_sub_i32 s18, s0, s19 // (op 2680) 


s_add_u32 s50, s6, 64 // (op 2680) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2680) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2680) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2680) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2680) 

s_mul_i32 s0, s18, s24 // (op 2680) 

s_sub_i32 s18, s1, s0 // (op 2680) 


s_mul_hi_i32 s19, s16, s9 // (op 2680) 
s_add_i32 s0, s19, s16 // (op 2680) 

s_ashr_i32 s1, s0, 31 // (op 2680) 
s_cmp_eq_i32 s9, 0 // (op 2680) 
s_mov_b32 s19, scc // (op 2680) 
s_cmp_eq_i32 s19, 0 // (op 2680) 
s_cselect_b32 s22, 0, -1 // (op 2680) 

s_and_b32 s19, s47, 31 // (op 2680) 
s_lshl_b32 s23, 1, s19 // (op 2680) 
s_add_i32 s49, s23, s22 // (op 2680) 


s_and_b32 s22, s1, s49 // (op 2680) 


s_add_i32 s1, s0, s22 // (op 2680) 


s_ashr_i32 s0, s1, s19 // (op 2680) 


s_lshl_b32 s1, s47, 24 // (op 2680) 
s_ashr_i32 s19, s1, 31 // (op 2680) 

s_xor_b32 s1, s0, s19 // (op 2680) 

s_sub_i32 s0, s1, s19 // (op 2680) 


s_mul_i32 s1, s11, s24 // (op 2680) 

s_mul_i32 s19, s8, s1 // (op 2680) 

s_mul_i32 s1, s0, s19 // (op 2680) 

s_sub_u32 s19, s16, s1 // (op 2680) 

s_mul_hi_i32 s1, s19, s12 // (op 2680) 
s_add_i32 s22, s1, s19 // (op 2680) 


s_ashr_i32 s1, s22, 31 // (op 2680) 
s_cmp_eq_i32 s12, 0 // (op 2680) 
s_mov_b32 s19, scc // (op 2680) 
s_cmp_eq_i32 s19, 0 // (op 2680) 
s_cselect_b32 s23, 0, -1 // (op 2680) 

s_and_b32 s19, s46, 31 // (op 2680) 
s_lshl_b32 s24, 1, s19 // (op 2680) 
s_add_i32 s25, s24, s23 // (op 2680) 


s_and_b32 s23, s1, s25 // (op 2680) 


s_add_i32 s1, s22, s23 // (op 2680) 


s_ashr_i32 s22, s1, s19 // (op 2680) 


s_lshl_b32 s1, s46, 24 // (op 2680) 
s_ashr_i32 s19, s1, 31 // (op 2680) 

s_xor_b32 s1, s22, s19 // (op 2680) 

s_sub_i32 s22, s1, s19 // (op 2680) 


s_cmp_eq_i32 s0, 0 // (op 2680) 
s_mov_b32 s1, scc // (op 2680) 

s_cmp_eq_i32 s1, 0 // (op 2680) 
s_cselect_b32 s0, s22, s18 // (op 2680) 



v_lshl_add_u32 v24, s0, 6, v23 // (op 2680) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2680) 
s_lshr_b32 s18, s0, 1 // (op 2680) 

v_lshl_add_u32 v23, s18, 3, v22 // (op 2680) 


v_mul_lo_u32 v22, v23, s42 // (op 2680) 
v_add_u32 v22, v22, v24 // (op 2680) 


s_mul_i32 s1, s42, s41 // (op 2682) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 2682) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 2682) least significant
s_mul_hi_u32 s19, s42, s40 // (op 2682) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2682) most significant: sum
s_add_u32 s1, s1, s19 // (op 2682) most significant: sum


s_mov_b32 s72, s44 // (op 2682) reg expression
s_mov_b32 s73, s45 // (op 2682) reg expression
s_mov_b32 s74, s0 // (op 2682) reg expression
s_mov_b32 s75, 131072 // (op 2682) call()

v_and_b32 v23, -8, v1 // (op 2773) 
v_sub_u32 v24, v1, v23 // (op 2773) 

v_lshlrev_b32 v23, 5, v24 // (op 2773) 

v_lshrrev_b32 v24, 3, v1 // (op 2773) 
v_lshl_add_u32 v25, v24, 10, v23 // (op 2773) 


v_lshlrev_b32 v23, 2, v25 // (op 2773) 

v_lshrrev_b32 v24, 3, v23 // (op 2773) 

v_and_b32 v23, -2, v1 // (op 2831) 
v_sub_u32 v25, v1, v23 // (op 2831) 

v_lshlrev_b32 v23, 5, v25 // (op 2831) 

v_lshrrev_b32 v25, 1, v1 // (op 2831) 
v_lshl_add_u32 v26, v25, 6, v23 // (op 2831) 


v_mul_lo_u32 v23, v26, 6 // (op 2831) 
v_lshrrev_b32 v25, 3, v23 // (op 2831) 

v_lshrrev_b32 v23, 4, v26 // (op 2831) 

v_lshlrev_b32 v26, 2, v23 // (op 2831) 

v_add_u32 v23, v25, v26 // (op 2831) 


v_and_b32 v25, -64, v1 // (op 2884) 
v_sub_u32 v26, v1, v25 // (op 2884) 

v_lshlrev_b32 v25, 2, v26 // (op 2884) 

v_lshrrev_b32 v26, 6, v1 // (op 2884) 
v_lshl_add_u32 v27, v26, 8, v25 // (op 2884) 


v_and_b32 v25, -64, v1 // (op 2942) 
v_sub_u32 v26, v1, v25 // (op 2942) 

v_and_b32 v25, 4294967232, v1 // (op 2942) 
v_add_u32 v28, v25, v26 // (op 2942) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 892) 
s_and_b32 s18, s0, 4294967294 // (op 892) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 891) 
s_mov_b32 s19, s0 // (op 891) convert

s_cmp_lt_u32 s18, s19 // (op 891) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_891 // (op 891) Condition: Top (jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_891 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoopTail_891:

s_add_u32 s18, s18, 1 // (op 893) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 202) 
v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v16, v14 // (op 202) reg expression
buffer_load_dwordx4 v[30:33], v16, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v16, v16, v18 // (op 202) 
buffer_load_dwordx4 v[34:37], v16, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v16, v16, v18 // (op 202) 
buffer_load_dwordx4 v[38:41], v16, s[52:55], 0 offen // (op 202) Load value
v_add_u32 v16, v16, v18 // (op 202) 
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen // (op 202) Load value



s_lshl_b64 s[0:1], s[26:27], 7 // (op 2472) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 2472) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 2472) 

v_add_u32 v15, v15, s0 // (op 2472) 



s_barrier  // (op 3163) 
v_mov_b32 v14, v17 // (op 225) 



v_mov_b32 v16, v14 // (op 225) reg expression

buffer_load_dwordx3 v[46:48], v16, s[64:67], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[50:52], v16, s[64:67], 0 offen offset:12 // (op 225) Load value


s_lshl_b64 s[0:1], s[32:33], 7 // (op 2544) 
s_mul_i32 s51, s1, 6 // (op 2544) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 2544) least significant
s_mul_hi_u32 s19, s0, 6 // (op 2544) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 2544) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 2544) 

v_add_u32 v17, v17, s0 // (op 2544) 

s_waitcnt vmcnt(5)
ds_write_b128 v24, v[30:33] offset:8448 // (op 203) Store local data 


v_mov_b32 v14, v20 // (op 226) 
v_mov_b32 v16, v14 // (op 226) reg expression

s_waitcnt vmcnt(4)
ds_write_b128 v24, v[34:37] offset:8576 // (op 203) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v24, v[38:41] offset:8704 // (op 203) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v24, v[42:45] offset:8832 // (op 203) Store local data 
buffer_load_dword v21, v16, s[68:71], 0 offen // (op 226) Load value



s_lshl_b32 s19, s36, 2 // (op 2608) 
s_barrier  // (op 3182) 
v_add_u32 v20, v20, s19 // (op 2608) 






v_mov_b32 v14, v22 // (op 227) 
v_mov_b32 v16, v14 // (op 227) reg expression

buffer_load_ubyte v49, v16, s[72:75], 0 offen offset: 0 // (op 227) Load value


s_lshl_b32 s19, s42, 2 // (op 2672) 
v_add_u32 v22, v22, s19 // (op 2672) 

s_waitcnt vmcnt(3)
ds_write_b96 v23, v[46:48] // (op 222) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3172) 
s_waitcnt vmcnt(2)
ds_write_b96 v23, v[50:52] offset:16 // (op 222) Store local data 




ds_read_b64_tr_b4 v[30:31], v6 offset:8448 // (op 205) Transpose load local data 
s_barrier  // (op 3203) 
ds_read_b64_tr_b4 v[32:33], v6 offset:10496 // (op 205) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 3193) 
s_waitcnt vmcnt(1)
ds_write_b32 v27, v21 offset:24832 // (op 223) Store local data 




ds_read_b96_tr_b6 v[34:36], v3 // (op 207) Transpose load local data 
s_barrier  // (op 3223) 
ds_read_b96_tr_b6 v[37:39], v3 offset:1024 // (op 207) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 3213) 
s_waitcnt vmcnt(0)
ds_write_b8 v28, v49 offset:8192 // (op 224) Store local data 



ds_read_u8 v14, v9 offset:24832 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3233) 


ds_read_u8 v16, v0 offset:8192 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[30:33], v[34:39], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 212) 





ds_read_b64_tr_b4 v[30:31], v6 offset:16640 // (op 214) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v6 offset:18688 // (op 214) Transpose load local data 



ds_read_b96_tr_b6 v[40:42], v3 offset:4096 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v3 offset:5120 // (op 216) Transpose load local data 


ds_read_u8 v14, v9 offset:25344 // (op 218) Load local data 


ds_read_u8 v21, v0 offset:8320 // (op 220) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[30:33], v[40:45], a[0:15], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 221) 
ds_read_b64_tr_b4 v[46:47], v6 offset:8480 // (op 417) Transpose load local data 


ds_read_b64_tr_b4 v[48:49], v6 offset:10528 // (op 417) Transpose load local data 


ds_read_u8 v14, v9 offset:24896 // (op 421) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[46:49], v[34:39], a[16:31], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 424) 





ds_read_b64_tr_b4 v[30:31], v6 offset:16672 // (op 426) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v6 offset:18720 // (op 426) Transpose load local data 


ds_read_u8 v14, v9 offset:25408 // (op 430) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[30:33], v[40:45], a[16:31], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 433) 
ds_read_b64_tr_b4 v[46:47], v6 offset:8512 // (op 657) Transpose load local data 


ds_read_b64_tr_b4 v[48:49], v6 offset:10560 // (op 657) Transpose load local data 


ds_read_u8 v14, v9 offset:24960 // (op 661) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[46:49], v[34:39], a[32:47], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 664) 





ds_read_b64_tr_b4 v[30:31], v6 offset:16704 // (op 666) Transpose load local data 
ds_read_b64_tr_b4 v[32:33], v6 offset:18752 // (op 666) Transpose load local data 


ds_read_u8 v14, v9 offset:25472 // (op 670) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[30:33], v[40:45], a[32:47], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 673) 
ds_read_b64_tr_b4 v[46:47], v6 offset:8544 // (op 897) Transpose load local data 


ds_read_b64_tr_b4 v[48:49], v6 offset:10592 // (op 897) Transpose load local data 


ds_read_u8 v14, v9 offset:25024 // (op 901) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[46:49], v[34:39], a[48:63], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 904) 







ds_read_b64_tr_b4 v[36:37], v6 offset:16736 // (op 906) Transpose load local data 
ds_read_b64_tr_b4 v[38:39], v6 offset:18784 // (op 906) Transpose load local data 


ds_read_u8 v14, v9 offset:25536 // (op 910) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[36:39], v[40:45], a[48:63], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 913) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 891) 
s_mov_b32 s19, s0 // (op 891) convert

s_cmp_lt_u32 s18, s19 // (op 891) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoopTail_891 // (op 891) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopTop_KLoopTail_891 if true)
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_891:























s_barrier  // (op 3354) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalFalse_0_beta_157
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 

v_cvt_f16_f32 v12, v28 // (op 56) 
v_cvt_f16_f32 v13, v29 // (op 56) 
v_cvt_f16_f32 v14, v30 // (op 56) 
v_cvt_f16_f32 v15, v31 // (op 56) 
v_cvt_f16_f32 v16, v32 // (op 56) 
v_cvt_f16_f32 v17, v33 // (op 56) 
v_cvt_f16_f32 v18, v34 // (op 56) 
v_cvt_f16_f32 v19, v35 // (op 56) 
v_cvt_f16_f32 v20, v36 // (op 56) 
v_cvt_f16_f32 v21, v37 // (op 56) 
v_cvt_f16_f32 v22, v38 // (op 56) 
v_cvt_f16_f32 v23, v39 // (op 56) 
v_cvt_f16_f32 v24, v40 // (op 56) 
v_cvt_f16_f32 v25, v41 // (op 56) 
v_cvt_f16_f32 v26, v42 // (op 56) 
v_cvt_f16_f32 v27, v43 // (op 56) 


v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 


buffer_store_dwordx2 v[28:29], v11, s[60:63], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[30:31], v11, s[60:63], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[32:33], v11, s[60:63], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v11, s[60:63], 0 offen offset:48 // (op 58) Store value


v_accvgpr_read v12, a16 // (op 1283) 
v_accvgpr_read v13, a17 // (op 1283) 
v_accvgpr_read v14, a18 // (op 1283) 
v_accvgpr_read v15, a19 // (op 1283) 
v_accvgpr_read v16, a20 // (op 1283) 
v_accvgpr_read v17, a21 // (op 1283) 
v_accvgpr_read v18, a22 // (op 1283) 
v_accvgpr_read v19, a23 // (op 1283) 
v_accvgpr_read v20, a24 // (op 1283) 
v_accvgpr_read v21, a25 // (op 1283) 
v_accvgpr_read v22, a26 // (op 1283) 
v_accvgpr_read v23, a27 // (op 1283) 
v_accvgpr_read v24, a28 // (op 1283) 
v_accvgpr_read v25, a29 // (op 1283) 
v_accvgpr_read v26, a30 // (op 1283) 
v_accvgpr_read v27, a31 // (op 1283) 
v_mul_f32 v28, v10, v12 // (op 1283) 
v_mul_f32 v29, v10, v13 // (op 1283) 
v_mul_f32 v30, v10, v14 // (op 1283) 
v_mul_f32 v31, v10, v15 // (op 1283) 
v_mul_f32 v32, v10, v16 // (op 1283) 
v_mul_f32 v33, v10, v17 // (op 1283) 
v_mul_f32 v34, v10, v18 // (op 1283) 
v_mul_f32 v35, v10, v19 // (op 1283) 
v_mul_f32 v36, v10, v20 // (op 1283) 
v_mul_f32 v37, v10, v21 // (op 1283) 
v_mul_f32 v38, v10, v22 // (op 1283) 
v_mul_f32 v39, v10, v23 // (op 1283) 
v_mul_f32 v40, v10, v24 // (op 1283) 
v_mul_f32 v41, v10, v25 // (op 1283) 
v_mul_f32 v42, v10, v26 // (op 1283) 
v_mul_f32 v43, v10, v27 // (op 1283) 

v_cvt_f16_f32 v12, v28 // (op 1285) 
v_cvt_f16_f32 v13, v29 // (op 1285) 
v_cvt_f16_f32 v14, v30 // (op 1285) 
v_cvt_f16_f32 v15, v31 // (op 1285) 
v_cvt_f16_f32 v16, v32 // (op 1285) 
v_cvt_f16_f32 v17, v33 // (op 1285) 
v_cvt_f16_f32 v18, v34 // (op 1285) 
v_cvt_f16_f32 v19, v35 // (op 1285) 
v_cvt_f16_f32 v20, v36 // (op 1285) 
v_cvt_f16_f32 v21, v37 // (op 1285) 
v_cvt_f16_f32 v22, v38 // (op 1285) 
v_cvt_f16_f32 v23, v39 // (op 1285) 
v_cvt_f16_f32 v24, v40 // (op 1285) 
v_cvt_f16_f32 v25, v41 // (op 1285) 
v_cvt_f16_f32 v26, v42 // (op 1285) 
v_cvt_f16_f32 v27, v43 // (op 1285) 


v_pack_b32_f16 v28, v12, v13 // (op 1286) 
v_pack_b32_f16 v29, v14, v15 // (op 1286) 
v_pack_b32_f16 v30, v16, v17 // (op 1286) 
v_pack_b32_f16 v31, v18, v19 // (op 1286) 
v_pack_b32_f16 v32, v20, v21 // (op 1286) 
v_pack_b32_f16 v33, v22, v23 // (op 1286) 
v_pack_b32_f16 v34, v24, v25 // (op 1286) 
v_pack_b32_f16 v35, v26, v27 // (op 1286) 


buffer_store_dwordx2 v[28:29], v11, s[60:63], 0 offen offset:128 // (op 1286) Store value
buffer_store_dwordx2 v[30:31], v11, s[60:63], 0 offen offset:144 // (op 1286) Store value
buffer_store_dwordx2 v[32:33], v11, s[60:63], 0 offen offset:160 // (op 1286) Store value
buffer_store_dwordx2 v[34:35], v11, s[60:63], 0 offen offset:176 // (op 1286) Store value


v_accvgpr_read v12, a32 // (op 1303) 
v_accvgpr_read v13, a33 // (op 1303) 
v_accvgpr_read v14, a34 // (op 1303) 
v_accvgpr_read v15, a35 // (op 1303) 
v_accvgpr_read v16, a36 // (op 1303) 
v_accvgpr_read v17, a37 // (op 1303) 
v_accvgpr_read v18, a38 // (op 1303) 
v_accvgpr_read v19, a39 // (op 1303) 
v_accvgpr_read v20, a40 // (op 1303) 
v_accvgpr_read v21, a41 // (op 1303) 
v_accvgpr_read v22, a42 // (op 1303) 
v_accvgpr_read v23, a43 // (op 1303) 
v_accvgpr_read v24, a44 // (op 1303) 
v_accvgpr_read v25, a45 // (op 1303) 
v_accvgpr_read v26, a46 // (op 1303) 
v_accvgpr_read v27, a47 // (op 1303) 
v_mul_f32 v28, v10, v12 // (op 1303) 
v_mul_f32 v29, v10, v13 // (op 1303) 
v_mul_f32 v30, v10, v14 // (op 1303) 
v_mul_f32 v31, v10, v15 // (op 1303) 
v_mul_f32 v32, v10, v16 // (op 1303) 
v_mul_f32 v33, v10, v17 // (op 1303) 
v_mul_f32 v34, v10, v18 // (op 1303) 
v_mul_f32 v35, v10, v19 // (op 1303) 
v_mul_f32 v36, v10, v20 // (op 1303) 
v_mul_f32 v37, v10, v21 // (op 1303) 
v_mul_f32 v38, v10, v22 // (op 1303) 
v_mul_f32 v39, v10, v23 // (op 1303) 
v_mul_f32 v40, v10, v24 // (op 1303) 
v_mul_f32 v41, v10, v25 // (op 1303) 
v_mul_f32 v42, v10, v26 // (op 1303) 
v_mul_f32 v43, v10, v27 // (op 1303) 

v_cvt_f16_f32 v12, v28 // (op 1305) 
v_cvt_f16_f32 v13, v29 // (op 1305) 
v_cvt_f16_f32 v14, v30 // (op 1305) 
v_cvt_f16_f32 v15, v31 // (op 1305) 
v_cvt_f16_f32 v16, v32 // (op 1305) 
v_cvt_f16_f32 v17, v33 // (op 1305) 
v_cvt_f16_f32 v18, v34 // (op 1305) 
v_cvt_f16_f32 v19, v35 // (op 1305) 
v_cvt_f16_f32 v20, v36 // (op 1305) 
v_cvt_f16_f32 v21, v37 // (op 1305) 
v_cvt_f16_f32 v22, v38 // (op 1305) 
v_cvt_f16_f32 v23, v39 // (op 1305) 
v_cvt_f16_f32 v24, v40 // (op 1305) 
v_cvt_f16_f32 v25, v41 // (op 1305) 
v_cvt_f16_f32 v26, v42 // (op 1305) 
v_cvt_f16_f32 v27, v43 // (op 1305) 


v_pack_b32_f16 v28, v12, v13 // (op 1306) 
v_pack_b32_f16 v29, v14, v15 // (op 1306) 
v_pack_b32_f16 v30, v16, v17 // (op 1306) 
v_pack_b32_f16 v31, v18, v19 // (op 1306) 
v_pack_b32_f16 v32, v20, v21 // (op 1306) 
v_pack_b32_f16 v33, v22, v23 // (op 1306) 
v_pack_b32_f16 v34, v24, v25 // (op 1306) 
v_pack_b32_f16 v35, v26, v27 // (op 1306) 


buffer_store_dwordx2 v[28:29], v11, s[60:63], 0 offen offset:256 // (op 1306) Store value
buffer_store_dwordx2 v[30:31], v11, s[60:63], 0 offen offset:272 // (op 1306) Store value
buffer_store_dwordx2 v[32:33], v11, s[60:63], 0 offen offset:288 // (op 1306) Store value
buffer_store_dwordx2 v[34:35], v11, s[60:63], 0 offen offset:304 // (op 1306) Store value


v_accvgpr_read v12, a48 // (op 1323) 
v_accvgpr_read v13, a49 // (op 1323) 
v_accvgpr_read v14, a50 // (op 1323) 
v_accvgpr_read v15, a51 // (op 1323) 
v_accvgpr_read v16, a52 // (op 1323) 
v_accvgpr_read v17, a53 // (op 1323) 
v_accvgpr_read v18, a54 // (op 1323) 
v_accvgpr_read v19, a55 // (op 1323) 
v_accvgpr_read v20, a56 // (op 1323) 
v_accvgpr_read v21, a57 // (op 1323) 
v_accvgpr_read v22, a58 // (op 1323) 
v_accvgpr_read v23, a59 // (op 1323) 
v_accvgpr_read v24, a60 // (op 1323) 
v_accvgpr_read v25, a61 // (op 1323) 
v_accvgpr_read v26, a62 // (op 1323) 
v_accvgpr_read v27, a63 // (op 1323) 
v_mul_f32 v28, v10, v12 // (op 1323) 
v_mul_f32 v29, v10, v13 // (op 1323) 
v_mul_f32 v30, v10, v14 // (op 1323) 
v_mul_f32 v31, v10, v15 // (op 1323) 
v_mul_f32 v32, v10, v16 // (op 1323) 
v_mul_f32 v33, v10, v17 // (op 1323) 
v_mul_f32 v34, v10, v18 // (op 1323) 
v_mul_f32 v35, v10, v19 // (op 1323) 
v_mul_f32 v36, v10, v20 // (op 1323) 
v_mul_f32 v37, v10, v21 // (op 1323) 
v_mul_f32 v38, v10, v22 // (op 1323) 
v_mul_f32 v39, v10, v23 // (op 1323) 
v_mul_f32 v40, v10, v24 // (op 1323) 
v_mul_f32 v41, v10, v25 // (op 1323) 
v_mul_f32 v42, v10, v26 // (op 1323) 
v_mul_f32 v43, v10, v27 // (op 1323) 

v_cvt_f16_f32 v12, v28 // (op 1325) 
v_cvt_f16_f32 v13, v29 // (op 1325) 
v_cvt_f16_f32 v14, v30 // (op 1325) 
v_cvt_f16_f32 v15, v31 // (op 1325) 
v_cvt_f16_f32 v16, v32 // (op 1325) 
v_cvt_f16_f32 v17, v33 // (op 1325) 
v_cvt_f16_f32 v18, v34 // (op 1325) 
v_cvt_f16_f32 v19, v35 // (op 1325) 
v_cvt_f16_f32 v20, v36 // (op 1325) 
v_cvt_f16_f32 v21, v37 // (op 1325) 
v_cvt_f16_f32 v22, v38 // (op 1325) 
v_cvt_f16_f32 v23, v39 // (op 1325) 
v_cvt_f16_f32 v24, v40 // (op 1325) 
v_cvt_f16_f32 v25, v41 // (op 1325) 
v_cvt_f16_f32 v26, v42 // (op 1325) 
v_cvt_f16_f32 v27, v43 // (op 1325) 


v_pack_b32_f16 v28, v12, v13 // (op 1326) 
v_pack_b32_f16 v29, v14, v15 // (op 1326) 
v_pack_b32_f16 v30, v16, v17 // (op 1326) 
v_pack_b32_f16 v31, v18, v19 // (op 1326) 
v_pack_b32_f16 v32, v20, v21 // (op 1326) 
v_pack_b32_f16 v33, v22, v23 // (op 1326) 
v_pack_b32_f16 v34, v24, v25 // (op 1326) 
v_pack_b32_f16 v35, v26, v27 // (op 1326) 


buffer_store_dwordx2 v[28:29], v11, s[60:63], 0 offen offset:384 // (op 1326) Store value
buffer_store_dwordx2 v[30:31], v11, s[60:63], 0 offen offset:400 // (op 1326) Store value
buffer_store_dwordx2 v[32:33], v11, s[60:63], 0 offen offset:416 // (op 1326) Store value
buffer_store_dwordx2 v[34:35], v11, s[60:63], 0 offen offset:432 // (op 1326) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalFalse_0_beta_157:



buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen offset:384 // (op 1406) Load value
buffer_load_dwordx2 v[14:15], v8, s[56:59], 0 offen offset:400 // (op 1406) Load value
buffer_load_dwordx2 v[16:17], v8, s[56:59], 0 offen offset:416 // (op 1406) Load value
v_accvgpr_read v20, a48 // (op 1402) 
v_accvgpr_read v21, a49 // (op 1402) 
v_accvgpr_read v22, a50 // (op 1402) 
v_accvgpr_read v23, a51 // (op 1402) 
v_accvgpr_read v24, a52 // (op 1402) 
v_accvgpr_read v25, a53 // (op 1402) 
v_accvgpr_read v26, a54 // (op 1402) 
v_accvgpr_read v27, a55 // (op 1402) 
v_accvgpr_read v28, a56 // (op 1402) 
v_accvgpr_read v29, a57 // (op 1402) 
v_accvgpr_read v30, a58 // (op 1402) 
v_accvgpr_read v31, a59 // (op 1402) 
v_accvgpr_read v32, a60 // (op 1402) 
v_accvgpr_read v33, a61 // (op 1402) 
v_accvgpr_read v34, a62 // (op 1402) 
v_accvgpr_read v35, a63 // (op 1402) 
v_mul_f32 v36, v10, v20 // (op 1402) 
v_mul_f32 v37, v10, v21 // (op 1402) 
v_mul_f32 v38, v10, v22 // (op 1402) 
v_mul_f32 v39, v10, v23 // (op 1402) 
v_mul_f32 v40, v10, v24 // (op 1402) 
v_mul_f32 v41, v10, v25 // (op 1402) 
v_mul_f32 v42, v10, v26 // (op 1402) 
v_mul_f32 v43, v10, v27 // (op 1402) 
v_mul_f32 v44, v10, v28 // (op 1402) 
v_mul_f32 v45, v10, v29 // (op 1402) 
v_mul_f32 v46, v10, v30 // (op 1402) 
v_mul_f32 v47, v10, v31 // (op 1402) 
v_mul_f32 v48, v10, v32 // (op 1402) 
v_mul_f32 v49, v10, v33 // (op 1402) 
v_mul_f32 v50, v10, v34 // (op 1402) 
v_mul_f32 v51, v10, v35 // (op 1402) 

buffer_load_dwordx2 v[18:19], v8, s[56:59], 0 offen offset:432 // (op 1406) Load value
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1407) 
v_lshrrev_b32 v21, 16, v12 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v52, v7, v20 // (op 1407) 
v_mul_f32 v53, v7, v21 // (op 1407) 
v_cvt_f32_f16 v20, v13 // (op 1407) 
v_lshrrev_b32 v21, 16, v13 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v54, v7, v20 // (op 1407) 
v_mul_f32 v55, v7, v21 // (op 1407) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1407) 
v_lshrrev_b32 v21, 16, v14 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v56, v7, v20 // (op 1407) 
v_mul_f32 v57, v7, v21 // (op 1407) 
v_cvt_f32_f16 v20, v15 // (op 1407) 
v_lshrrev_b32 v21, 16, v15 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v58, v7, v20 // (op 1407) 
v_mul_f32 v59, v7, v21 // (op 1407) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1407) 
v_lshrrev_b32 v21, 16, v16 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v60, v7, v20 // (op 1407) 
v_mul_f32 v61, v7, v21 // (op 1407) 
v_cvt_f32_f16 v20, v17 // (op 1407) 
v_lshrrev_b32 v21, 16, v17 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v62, v7, v20 // (op 1407) 
v_mul_f32 v63, v7, v21 // (op 1407) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1407) 
v_lshrrev_b32 v21, 16, v18 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v64, v7, v20 // (op 1407) 
v_mul_f32 v65, v7, v21 // (op 1407) 
v_cvt_f32_f16 v20, v19 // (op 1407) 
v_lshrrev_b32 v21, 16, v19 // (op 1407) 
v_cvt_f32_f16 v21, v21 // (op 1407) 
v_mul_f32 v66, v7, v20 // (op 1407) 
v_mul_f32 v67, v7, v21 // (op 1407) 


v_add_f32 v12, v52, v36 // (op 1403) 
v_add_f32 v13, v53, v37 // (op 1403) 
v_add_f32 v14, v54, v38 // (op 1403) 
v_add_f32 v15, v55, v39 // (op 1403) 
v_add_f32 v16, v56, v40 // (op 1403) 
v_add_f32 v17, v57, v41 // (op 1403) 
v_add_f32 v18, v58, v42 // (op 1403) 
v_add_f32 v19, v59, v43 // (op 1403) 
v_add_f32 v20, v60, v44 // (op 1403) 
v_add_f32 v21, v61, v45 // (op 1403) 
v_add_f32 v22, v62, v46 // (op 1403) 
v_add_f32 v23, v63, v47 // (op 1403) 
v_add_f32 v24, v64, v48 // (op 1403) 
v_add_f32 v25, v65, v49 // (op 1403) 
v_add_f32 v26, v66, v50 // (op 1403) 
v_add_f32 v27, v67, v51 // (op 1403) 


v_cvt_f16_f32 v28, v12 // (op 1404) 
v_cvt_f16_f32 v29, v13 // (op 1404) 
v_cvt_f16_f32 v30, v14 // (op 1404) 
v_cvt_f16_f32 v31, v15 // (op 1404) 
v_cvt_f16_f32 v32, v16 // (op 1404) 
v_cvt_f16_f32 v33, v17 // (op 1404) 
v_cvt_f16_f32 v34, v18 // (op 1404) 
v_cvt_f16_f32 v35, v19 // (op 1404) 
v_cvt_f16_f32 v36, v20 // (op 1404) 
v_cvt_f16_f32 v37, v21 // (op 1404) 
v_cvt_f16_f32 v38, v22 // (op 1404) 
v_cvt_f16_f32 v39, v23 // (op 1404) 
v_cvt_f16_f32 v40, v24 // (op 1404) 
v_cvt_f16_f32 v41, v25 // (op 1404) 
v_cvt_f16_f32 v42, v26 // (op 1404) 
v_cvt_f16_f32 v43, v27 // (op 1404) 


v_pack_b32_f16 v12, v28, v29 // (op 1405) 
v_pack_b32_f16 v13, v30, v31 // (op 1405) 
v_pack_b32_f16 v14, v32, v33 // (op 1405) 
v_pack_b32_f16 v15, v34, v35 // (op 1405) 
v_pack_b32_f16 v16, v36, v37 // (op 1405) 
v_pack_b32_f16 v17, v38, v39 // (op 1405) 
v_pack_b32_f16 v18, v40, v41 // (op 1405) 
v_pack_b32_f16 v19, v42, v43 // (op 1405) 


buffer_store_dwordx2 v[12:13], v11, s[60:63], 0 offen offset:384 // (op 1405) Store value
buffer_store_dwordx2 v[14:15], v11, s[60:63], 0 offen offset:400 // (op 1405) Store value
buffer_store_dwordx2 v[16:17], v11, s[60:63], 0 offen offset:416 // (op 1405) Store value
buffer_store_dwordx2 v[18:19], v11, s[60:63], 0 offen offset:432 // (op 1405) Store value





buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen offset:256 // (op 1386) Load value
buffer_load_dwordx2 v[14:15], v8, s[56:59], 0 offen offset:272 // (op 1386) Load value
v_accvgpr_read v20, a32 // (op 1382) 
v_accvgpr_read v21, a33 // (op 1382) 
v_accvgpr_read v22, a34 // (op 1382) 
v_accvgpr_read v23, a35 // (op 1382) 
v_accvgpr_read v24, a36 // (op 1382) 
v_accvgpr_read v25, a37 // (op 1382) 
v_accvgpr_read v26, a38 // (op 1382) 
v_accvgpr_read v27, a39 // (op 1382) 
v_accvgpr_read v28, a40 // (op 1382) 
v_accvgpr_read v29, a41 // (op 1382) 
v_accvgpr_read v30, a42 // (op 1382) 
v_accvgpr_read v31, a43 // (op 1382) 
v_accvgpr_read v32, a44 // (op 1382) 
v_accvgpr_read v33, a45 // (op 1382) 
v_accvgpr_read v34, a46 // (op 1382) 
v_accvgpr_read v35, a47 // (op 1382) 
v_mul_f32 v36, v10, v20 // (op 1382) 
v_mul_f32 v37, v10, v21 // (op 1382) 
v_mul_f32 v38, v10, v22 // (op 1382) 
v_mul_f32 v39, v10, v23 // (op 1382) 
v_mul_f32 v40, v10, v24 // (op 1382) 
v_mul_f32 v41, v10, v25 // (op 1382) 
v_mul_f32 v42, v10, v26 // (op 1382) 
v_mul_f32 v43, v10, v27 // (op 1382) 
v_mul_f32 v44, v10, v28 // (op 1382) 
v_mul_f32 v45, v10, v29 // (op 1382) 
v_mul_f32 v46, v10, v30 // (op 1382) 
v_mul_f32 v47, v10, v31 // (op 1382) 
v_mul_f32 v48, v10, v32 // (op 1382) 
v_mul_f32 v49, v10, v33 // (op 1382) 
v_mul_f32 v50, v10, v34 // (op 1382) 
v_mul_f32 v51, v10, v35 // (op 1382) 

buffer_load_dwordx2 v[16:17], v8, s[56:59], 0 offen offset:288 // (op 1386) Load value
buffer_load_dwordx2 v[18:19], v8, s[56:59], 0 offen offset:304 // (op 1386) Load value
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1387) 
v_lshrrev_b32 v21, 16, v12 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v52, v7, v20 // (op 1387) 
v_mul_f32 v53, v7, v21 // (op 1387) 
v_cvt_f32_f16 v20, v13 // (op 1387) 
v_lshrrev_b32 v21, 16, v13 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v54, v7, v20 // (op 1387) 
v_mul_f32 v55, v7, v21 // (op 1387) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1387) 
v_lshrrev_b32 v21, 16, v14 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v56, v7, v20 // (op 1387) 
v_mul_f32 v57, v7, v21 // (op 1387) 
v_cvt_f32_f16 v20, v15 // (op 1387) 
v_lshrrev_b32 v21, 16, v15 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v58, v7, v20 // (op 1387) 
v_mul_f32 v59, v7, v21 // (op 1387) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1387) 
v_lshrrev_b32 v21, 16, v16 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v60, v7, v20 // (op 1387) 
v_mul_f32 v61, v7, v21 // (op 1387) 
v_cvt_f32_f16 v20, v17 // (op 1387) 
v_lshrrev_b32 v21, 16, v17 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v62, v7, v20 // (op 1387) 
v_mul_f32 v63, v7, v21 // (op 1387) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1387) 
v_lshrrev_b32 v21, 16, v18 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v64, v7, v20 // (op 1387) 
v_mul_f32 v65, v7, v21 // (op 1387) 
v_cvt_f32_f16 v20, v19 // (op 1387) 
v_lshrrev_b32 v21, 16, v19 // (op 1387) 
v_cvt_f32_f16 v21, v21 // (op 1387) 
v_mul_f32 v66, v7, v20 // (op 1387) 
v_mul_f32 v67, v7, v21 // (op 1387) 


v_add_f32 v12, v52, v36 // (op 1383) 
v_add_f32 v13, v53, v37 // (op 1383) 
v_add_f32 v14, v54, v38 // (op 1383) 
v_add_f32 v15, v55, v39 // (op 1383) 
v_add_f32 v16, v56, v40 // (op 1383) 
v_add_f32 v17, v57, v41 // (op 1383) 
v_add_f32 v18, v58, v42 // (op 1383) 
v_add_f32 v19, v59, v43 // (op 1383) 
v_add_f32 v20, v60, v44 // (op 1383) 
v_add_f32 v21, v61, v45 // (op 1383) 
v_add_f32 v22, v62, v46 // (op 1383) 
v_add_f32 v23, v63, v47 // (op 1383) 
v_add_f32 v24, v64, v48 // (op 1383) 
v_add_f32 v25, v65, v49 // (op 1383) 
v_add_f32 v26, v66, v50 // (op 1383) 
v_add_f32 v27, v67, v51 // (op 1383) 


v_cvt_f16_f32 v28, v12 // (op 1384) 
v_cvt_f16_f32 v29, v13 // (op 1384) 
v_cvt_f16_f32 v30, v14 // (op 1384) 
v_cvt_f16_f32 v31, v15 // (op 1384) 
v_cvt_f16_f32 v32, v16 // (op 1384) 
v_cvt_f16_f32 v33, v17 // (op 1384) 
v_cvt_f16_f32 v34, v18 // (op 1384) 
v_cvt_f16_f32 v35, v19 // (op 1384) 
v_cvt_f16_f32 v36, v20 // (op 1384) 
v_cvt_f16_f32 v37, v21 // (op 1384) 
v_cvt_f16_f32 v38, v22 // (op 1384) 
v_cvt_f16_f32 v39, v23 // (op 1384) 
v_cvt_f16_f32 v40, v24 // (op 1384) 
v_cvt_f16_f32 v41, v25 // (op 1384) 
v_cvt_f16_f32 v42, v26 // (op 1384) 
v_cvt_f16_f32 v43, v27 // (op 1384) 


v_pack_b32_f16 v12, v28, v29 // (op 1385) 
v_pack_b32_f16 v13, v30, v31 // (op 1385) 
v_pack_b32_f16 v14, v32, v33 // (op 1385) 
v_pack_b32_f16 v15, v34, v35 // (op 1385) 
v_pack_b32_f16 v16, v36, v37 // (op 1385) 
v_pack_b32_f16 v17, v38, v39 // (op 1385) 
v_pack_b32_f16 v18, v40, v41 // (op 1385) 
v_pack_b32_f16 v19, v42, v43 // (op 1385) 


buffer_store_dwordx2 v[12:13], v11, s[60:63], 0 offen offset:256 // (op 1385) Store value
buffer_store_dwordx2 v[14:15], v11, s[60:63], 0 offen offset:272 // (op 1385) Store value
buffer_store_dwordx2 v[16:17], v11, s[60:63], 0 offen offset:288 // (op 1385) Store value
buffer_store_dwordx2 v[18:19], v11, s[60:63], 0 offen offset:304 // (op 1385) Store value





buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen offset:128 // (op 1366) Load value
buffer_load_dwordx2 v[14:15], v8, s[56:59], 0 offen offset:144 // (op 1366) Load value
v_accvgpr_read v20, a16 // (op 1362) 
v_accvgpr_read v21, a17 // (op 1362) 
v_accvgpr_read v22, a18 // (op 1362) 
v_accvgpr_read v23, a19 // (op 1362) 
v_accvgpr_read v24, a20 // (op 1362) 
v_accvgpr_read v25, a21 // (op 1362) 
v_accvgpr_read v26, a22 // (op 1362) 
v_accvgpr_read v27, a23 // (op 1362) 
v_accvgpr_read v28, a24 // (op 1362) 
v_accvgpr_read v29, a25 // (op 1362) 
v_accvgpr_read v30, a26 // (op 1362) 
v_accvgpr_read v31, a27 // (op 1362) 
v_accvgpr_read v32, a28 // (op 1362) 
v_accvgpr_read v33, a29 // (op 1362) 
v_accvgpr_read v34, a30 // (op 1362) 
v_accvgpr_read v35, a31 // (op 1362) 
v_mul_f32 v36, v10, v20 // (op 1362) 
v_mul_f32 v37, v10, v21 // (op 1362) 
v_mul_f32 v38, v10, v22 // (op 1362) 
v_mul_f32 v39, v10, v23 // (op 1362) 
v_mul_f32 v40, v10, v24 // (op 1362) 
v_mul_f32 v41, v10, v25 // (op 1362) 
v_mul_f32 v42, v10, v26 // (op 1362) 
v_mul_f32 v43, v10, v27 // (op 1362) 
v_mul_f32 v44, v10, v28 // (op 1362) 
v_mul_f32 v45, v10, v29 // (op 1362) 
v_mul_f32 v46, v10, v30 // (op 1362) 
v_mul_f32 v47, v10, v31 // (op 1362) 
v_mul_f32 v48, v10, v32 // (op 1362) 
v_mul_f32 v49, v10, v33 // (op 1362) 
v_mul_f32 v50, v10, v34 // (op 1362) 
v_mul_f32 v51, v10, v35 // (op 1362) 

buffer_load_dwordx2 v[16:17], v8, s[56:59], 0 offen offset:160 // (op 1366) Load value
buffer_load_dwordx2 v[18:19], v8, s[56:59], 0 offen offset:176 // (op 1366) Load value
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1367) 
v_lshrrev_b32 v21, 16, v12 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v52, v7, v20 // (op 1367) 
v_mul_f32 v53, v7, v21 // (op 1367) 
v_cvt_f32_f16 v20, v13 // (op 1367) 
v_lshrrev_b32 v21, 16, v13 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v54, v7, v20 // (op 1367) 
v_mul_f32 v55, v7, v21 // (op 1367) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1367) 
v_lshrrev_b32 v21, 16, v14 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v56, v7, v20 // (op 1367) 
v_mul_f32 v57, v7, v21 // (op 1367) 
v_cvt_f32_f16 v20, v15 // (op 1367) 
v_lshrrev_b32 v21, 16, v15 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v58, v7, v20 // (op 1367) 
v_mul_f32 v59, v7, v21 // (op 1367) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1367) 
v_lshrrev_b32 v21, 16, v16 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v60, v7, v20 // (op 1367) 
v_mul_f32 v61, v7, v21 // (op 1367) 
v_cvt_f32_f16 v20, v17 // (op 1367) 
v_lshrrev_b32 v21, 16, v17 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v62, v7, v20 // (op 1367) 
v_mul_f32 v63, v7, v21 // (op 1367) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1367) 
v_lshrrev_b32 v21, 16, v18 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v64, v7, v20 // (op 1367) 
v_mul_f32 v65, v7, v21 // (op 1367) 
v_cvt_f32_f16 v20, v19 // (op 1367) 
v_lshrrev_b32 v21, 16, v19 // (op 1367) 
v_cvt_f32_f16 v21, v21 // (op 1367) 
v_mul_f32 v66, v7, v20 // (op 1367) 
v_mul_f32 v67, v7, v21 // (op 1367) 


v_add_f32 v12, v52, v36 // (op 1363) 
v_add_f32 v13, v53, v37 // (op 1363) 
v_add_f32 v14, v54, v38 // (op 1363) 
v_add_f32 v15, v55, v39 // (op 1363) 
v_add_f32 v16, v56, v40 // (op 1363) 
v_add_f32 v17, v57, v41 // (op 1363) 
v_add_f32 v18, v58, v42 // (op 1363) 
v_add_f32 v19, v59, v43 // (op 1363) 
v_add_f32 v20, v60, v44 // (op 1363) 
v_add_f32 v21, v61, v45 // (op 1363) 
v_add_f32 v22, v62, v46 // (op 1363) 
v_add_f32 v23, v63, v47 // (op 1363) 
v_add_f32 v24, v64, v48 // (op 1363) 
v_add_f32 v25, v65, v49 // (op 1363) 
v_add_f32 v26, v66, v50 // (op 1363) 
v_add_f32 v27, v67, v51 // (op 1363) 


v_cvt_f16_f32 v28, v12 // (op 1364) 
v_cvt_f16_f32 v29, v13 // (op 1364) 
v_cvt_f16_f32 v30, v14 // (op 1364) 
v_cvt_f16_f32 v31, v15 // (op 1364) 
v_cvt_f16_f32 v32, v16 // (op 1364) 
v_cvt_f16_f32 v33, v17 // (op 1364) 
v_cvt_f16_f32 v34, v18 // (op 1364) 
v_cvt_f16_f32 v35, v19 // (op 1364) 
v_cvt_f16_f32 v36, v20 // (op 1364) 
v_cvt_f16_f32 v37, v21 // (op 1364) 
v_cvt_f16_f32 v38, v22 // (op 1364) 
v_cvt_f16_f32 v39, v23 // (op 1364) 
v_cvt_f16_f32 v40, v24 // (op 1364) 
v_cvt_f16_f32 v41, v25 // (op 1364) 
v_cvt_f16_f32 v42, v26 // (op 1364) 
v_cvt_f16_f32 v43, v27 // (op 1364) 


v_pack_b32_f16 v12, v28, v29 // (op 1365) 
v_pack_b32_f16 v13, v30, v31 // (op 1365) 
v_pack_b32_f16 v14, v32, v33 // (op 1365) 
v_pack_b32_f16 v15, v34, v35 // (op 1365) 
v_pack_b32_f16 v16, v36, v37 // (op 1365) 
v_pack_b32_f16 v17, v38, v39 // (op 1365) 
v_pack_b32_f16 v18, v40, v41 // (op 1365) 
v_pack_b32_f16 v19, v42, v43 // (op 1365) 


buffer_store_dwordx2 v[12:13], v11, s[60:63], 0 offen offset:128 // (op 1365) Store value
buffer_store_dwordx2 v[14:15], v11, s[60:63], 0 offen offset:144 // (op 1365) Store value
buffer_store_dwordx2 v[16:17], v11, s[60:63], 0 offen offset:160 // (op 1365) Store value
buffer_store_dwordx2 v[18:19], v11, s[60:63], 0 offen offset:176 // (op 1365) Store value





buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[14:15], v8, s[56:59], 0 offen offset:16 // (op 170) Load value
v_accvgpr_read v20, a0 // (op 166) 
v_accvgpr_read v21, a1 // (op 166) 
v_accvgpr_read v22, a2 // (op 166) 
v_accvgpr_read v23, a3 // (op 166) 
v_accvgpr_read v24, a4 // (op 166) 
v_accvgpr_read v25, a5 // (op 166) 
v_accvgpr_read v26, a6 // (op 166) 
v_accvgpr_read v27, a7 // (op 166) 
v_accvgpr_read v28, a8 // (op 166) 
v_accvgpr_read v29, a9 // (op 166) 
v_accvgpr_read v30, a10 // (op 166) 
v_accvgpr_read v31, a11 // (op 166) 
v_accvgpr_read v32, a12 // (op 166) 
v_accvgpr_read v33, a13 // (op 166) 
v_accvgpr_read v34, a14 // (op 166) 
v_accvgpr_read v35, a15 // (op 166) 
v_mul_f32 v36, v10, v20 // (op 166) 
v_mul_f32 v37, v10, v21 // (op 166) 
v_mul_f32 v38, v10, v22 // (op 166) 
v_mul_f32 v39, v10, v23 // (op 166) 
v_mul_f32 v40, v10, v24 // (op 166) 
v_mul_f32 v41, v10, v25 // (op 166) 
v_mul_f32 v42, v10, v26 // (op 166) 
v_mul_f32 v43, v10, v27 // (op 166) 
v_mul_f32 v44, v10, v28 // (op 166) 
v_mul_f32 v45, v10, v29 // (op 166) 
v_mul_f32 v46, v10, v30 // (op 166) 
v_mul_f32 v47, v10, v31 // (op 166) 
v_mul_f32 v48, v10, v32 // (op 166) 
v_mul_f32 v49, v10, v33 // (op 166) 
v_mul_f32 v50, v10, v34 // (op 166) 
v_mul_f32 v51, v10, v35 // (op 166) 

buffer_load_dwordx2 v[16:17], v8, s[56:59], 0 offen offset:32 // (op 170) Load value
buffer_load_dwordx2 v[18:19], v8, s[56:59], 0 offen offset:48 // (op 170) Load value
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 171) 
v_lshrrev_b32 v21, 16, v12 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v52, v7, v20 // (op 171) 
v_mul_f32 v53, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v13 // (op 171) 
v_lshrrev_b32 v21, 16, v13 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v54, v7, v20 // (op 171) 
v_mul_f32 v55, v7, v21 // (op 171) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 171) 
v_lshrrev_b32 v21, 16, v14 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v56, v7, v20 // (op 171) 
v_mul_f32 v57, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v15 // (op 171) 
v_lshrrev_b32 v21, 16, v15 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v58, v7, v20 // (op 171) 
v_mul_f32 v59, v7, v21 // (op 171) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 171) 
v_lshrrev_b32 v21, 16, v16 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v60, v7, v20 // (op 171) 
v_mul_f32 v61, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v17 // (op 171) 
v_lshrrev_b32 v21, 16, v17 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v62, v7, v20 // (op 171) 
v_mul_f32 v63, v7, v21 // (op 171) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 171) 
v_lshrrev_b32 v21, 16, v18 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v64, v7, v20 // (op 171) 
v_mul_f32 v65, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v19 // (op 171) 
v_lshrrev_b32 v21, 16, v19 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v66, v7, v20 // (op 171) 
v_mul_f32 v67, v7, v21 // (op 171) 


v_add_f32 v12, v52, v36 // (op 167) 
v_add_f32 v13, v53, v37 // (op 167) 
v_add_f32 v14, v54, v38 // (op 167) 
v_add_f32 v15, v55, v39 // (op 167) 
v_add_f32 v16, v56, v40 // (op 167) 
v_add_f32 v17, v57, v41 // (op 167) 
v_add_f32 v18, v58, v42 // (op 167) 
v_add_f32 v19, v59, v43 // (op 167) 
v_add_f32 v20, v60, v44 // (op 167) 
v_add_f32 v21, v61, v45 // (op 167) 
v_add_f32 v22, v62, v46 // (op 167) 
v_add_f32 v23, v63, v47 // (op 167) 
v_add_f32 v24, v64, v48 // (op 167) 
v_add_f32 v25, v65, v49 // (op 167) 
v_add_f32 v26, v66, v50 // (op 167) 
v_add_f32 v27, v67, v51 // (op 167) 


v_cvt_f16_f32 v28, v12 // (op 168) 
v_cvt_f16_f32 v29, v13 // (op 168) 
v_cvt_f16_f32 v30, v14 // (op 168) 
v_cvt_f16_f32 v31, v15 // (op 168) 
v_cvt_f16_f32 v32, v16 // (op 168) 
v_cvt_f16_f32 v33, v17 // (op 168) 
v_cvt_f16_f32 v34, v18 // (op 168) 
v_cvt_f16_f32 v35, v19 // (op 168) 
v_cvt_f16_f32 v36, v20 // (op 168) 
v_cvt_f16_f32 v37, v21 // (op 168) 
v_cvt_f16_f32 v38, v22 // (op 168) 
v_cvt_f16_f32 v39, v23 // (op 168) 
v_cvt_f16_f32 v40, v24 // (op 168) 
v_cvt_f16_f32 v41, v25 // (op 168) 
v_cvt_f16_f32 v42, v26 // (op 168) 
v_cvt_f16_f32 v43, v27 // (op 168) 


v_pack_b32_f16 v12, v28, v29 // (op 169) 
v_pack_b32_f16 v13, v30, v31 // (op 169) 
v_pack_b32_f16 v14, v32, v33 // (op 169) 
v_pack_b32_f16 v15, v34, v35 // (op 169) 
v_pack_b32_f16 v16, v36, v37 // (op 169) 
v_pack_b32_f16 v17, v38, v39 // (op 169) 
v_pack_b32_f16 v18, v40, v41 // (op 169) 
v_pack_b32_f16 v19, v42, v43 // (op 169) 


buffer_store_dwordx2 v[12:13], v11, s[60:63], 0 offen // (op 169) Store value
buffer_store_dwordx2 v[14:15], v11, s[60:63], 0 offen offset:16 // (op 169) Store value
buffer_store_dwordx2 v[16:17], v11, s[60:63], 0 offen offset:32 // (op 169) Store value
buffer_store_dwordx2 v[18:19], v11, s[60:63], 0 offen offset:48 // (op 169) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_
.LRR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_, .LRR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__end-RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 188
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51712 // lds bytes
  .amdhsa_accum_offset 124
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: StrideD0 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 51712
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 124
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
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
      - .name: WGM_18
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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

