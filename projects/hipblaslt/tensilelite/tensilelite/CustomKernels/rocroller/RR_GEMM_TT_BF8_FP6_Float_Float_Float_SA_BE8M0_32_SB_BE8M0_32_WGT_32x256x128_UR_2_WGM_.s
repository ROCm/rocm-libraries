// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 3683) 
s_mul_hi_i32 s1, s16, s11 // (op 3871) 
v_lshrrev_b32 v0, 7, v1 // (op 3739) 
v_and_b32 v3, 15, v1 // (op 3739) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3739) 


v_and_b32 v0, 48, v1 // (op 3739) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3739) 


v_lshrrev_b32 v0, 7, v1 // (op 3912) 
v_and_b32 v4, 15, v1 // (op 3912) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3912) 


v_and_b32 v0, 48, v1 // (op 3912) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3912) 


v_lshrrev_b32 v0, 7, v1 // (op 3683) 
s_add_i32 s18, s0, s16 // (op 3683) 

v_and_b32 v5, 63, v1 // (op 3683) 
s_ashr_i32 s0, s18, 31 // (op 3683) 
v_lshrrev_b32 v6, 4, v5 // (op 3683) 
s_cmp_eq_i32 s11, 0 // (op 3683) 
s_mov_b32 s19, scc // (op 3683) 
v_lshlrev_b32 v7, 2, v6 // (op 3683) 

s_cmp_eq_i32 s19, 0 // (op 3683) 
s_cselect_b32 s51, 0, -1 // (op 3683) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3683) 


s_and_b32 s19, s12, 31 // (op 3683) 
v_lshrrev_b32 v0, 2, v5 // (op 3683) 

v_and_b32 v5, 3, v0 // (op 3683) 

v_and_b32 v0, 63, v1 // (op 3797) 
v_and_b32 v7, 48, v1 // (op 3797) 
v_sub_u32 v8, v0, v7 // (op 3797) 

v_lshrrev_b32 v7, 4, v0 // (op 3797) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 3797) 


v_lshrrev_b32 v7, 6, v1 // (op 3797) 
v_and_b32 v8, 1, v7 // (op 3797) 

v_lshlrev_b32 v7, 4, v8 // (op 3797) 

v_lshl_add_u32 v8, v0, 8, v7 // (op 3797) 


v_mul_lo_u32 v0, v8, 6 // (op 3797) 
v_lshrrev_b32 v7, 3, v0 // (op 3797) 

v_lshrrev_b32 v0, 4, v8 // (op 3797) 

v_lshlrev_b32 v8, 2, v0 // (op 3797) 

v_add_u32 v0, v7, v8 // (op 3797) 


v_lshrrev_b32 v7, 6, v1 // (op 3855) 
v_and_b32 v8, 1, v7 // (op 3855) 

v_and_b32 v7, 15, v1 // (op 3855) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 3855) 


v_and_b32 v7, 63, v1 // (op 3855) 
v_lshrrev_b32 v8, 4, v7 // (op 3855) 

v_lshl_add_u32 v7, v8, 8, v9 // (op 3855) 


v_lshrrev_b32 v8, 7, v1 // (op 3871) 
v_and_b32 v9, 63, v1 // (op 3871) 
v_and_b32 v10, 63, v1 // (op 3941) 
v_and_b32 v11, 48, v1 // (op 3941) 
v_sub_u32 v12, v10, v11 // (op 3941) 

v_lshrrev_b32 v11, 4, v10 // (op 3941) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 3941) 


v_lshrrev_b32 v11, 6, v1 // (op 3941) 
v_and_b32 v12, 1, v11 // (op 3941) 

v_lshlrev_b32 v11, 4, v12 // (op 3941) 

v_lshl_add_u32 v12, v10, 8, v11 // (op 3941) 


v_mul_lo_u32 v10, v12, 6 // (op 3941) 
v_lshrrev_b32 v11, 3, v10 // (op 3941) 

v_lshrrev_b32 v10, 4, v12 // (op 3941) 

v_lshlrev_b32 v12, 2, v10 // (op 3941) 

v_add_u32 v10, v11, v12 // (op 3941) 


v_lshrrev_b32 v11, 6, v1 // (op 3970) 
v_and_b32 v12, 1, v11 // (op 3970) 

v_and_b32 v11, 15, v1 // (op 3970) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 3970) 


v_and_b32 v11, 63, v1 // (op 3970) 
v_lshrrev_b32 v12, 4, v11 // (op 3970) 

v_lshl_add_u32 v11, v12, 8, v13 // (op 3970) 


s_lshl_b32 s52, 1, s19 // (op 3683) 
v_and_b32 v12, 3, v1 // (op 3683) 
s_add_i32 s53, s52, s51 // (op 3683) 


s_and_b32 s51, s0, s53 // (op 3683) 


s_add_i32 s0, s1, s16 // (op 3871) 

s_ashr_i32 s1, s0, 31 // (op 3871) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3683) 


s_add_i32 s52, s18, s51 // (op 3683) 


v_lshrrev_b32 v5, 6, v1 // (op 3683) 
s_ashr_i32 s18, s52, s19 // (op 3683) 


v_and_b32 v12, 1, v5 // (op 3683) 

s_lshl_b32 s19, s12, 24 // (op 3683) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3683) 


s_ashr_i32 s51, s19, 31 // (op 3683) 

s_xor_b32 s19, s18, s51 // (op 3683) 

s_sub_i32 s18, s19, s51 // (op 3683) 


s_mul_hi_i32 s19, s18, s13 // (op 3683) 
s_add_i32 s51, s19, s18 // (op 3683) 

s_ashr_i32 s19, s51, 31 // (op 3683) 
s_cmp_eq_i32 s13, 0 // (op 3683) 
s_mov_b32 s52, scc // (op 3683) 
s_cmp_eq_i32 s52, 0 // (op 3683) 
s_cselect_b32 s53, 0, -1 // (op 3683) 

s_and_b32 s52, s14, 31 // (op 3683) 
v_lshrrev_b32 v12, 4, v9 // (op 3871) 
s_cmp_eq_i32 s11, 0 // (op 3871) 
s_mov_b32 s54, scc // (op 3871) 
v_lshlrev_b32 v13, 2, v12 // (op 3871) 

s_cmp_eq_i32 s54, 0 // (op 3871) 
s_cselect_b32 s55, 0, -1 // (op 3871) 

s_lshl_b32 s54, 1, s52 // (op 3683) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 3871) 


v_lshrrev_b32 v8, 2, v9 // (op 3871) 

s_add_i32 s56, s54, s53 // (op 3683) 


s_and_b32 s53, s19, s56 // (op 3683) 


s_add_i32 s19, s51, s53 // (op 3683) 


s_ashr_i32 s51, s19, s52 // (op 3683) 


s_lshl_b32 s19, s14, 24 // (op 3683) 
s_ashr_i32 s52, s19, 31 // (op 3683) 

s_xor_b32 s19, s51, s52 // (op 3683) 

s_sub_i32 s51, s19, s52 // (op 3683) 


s_and_b32 s19, s12, 31 // (op 3871) 
s_lshl_b32 s52, 1, s19 // (op 3871) 
v_and_b32 v9, 3, v8 // (op 3871) 

s_add_i32 s53, s52, s55 // (op 3871) 


v_and_b32 v8, 3, v1 // (op 3871) 
s_and_b32 s52, s1, s53 // (op 3871) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 3871) 


s_add_i32 s1, s0, s52 // (op 3871) 


v_lshrrev_b32 v8, 6, v1 // (op 3871) 
s_ashr_i32 s0, s1, s19 // (op 3871) 


v_and_b32 v9, 1, v8 // (op 3871) 

s_lshl_b32 s1, s12, 24 // (op 3871) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 3871) 


s_ashr_i32 s19, s1, 31 // (op 3871) 

s_xor_b32 s1, s0, s19 // (op 3871) 

s_sub_i32 s0, s1, s19 // (op 3871) 


s_mul_hi_i32 s1, s0, s13 // (op 3871) 
s_add_i32 s19, s1, s0 // (op 3871) 

s_ashr_i32 s1, s19, 31 // (op 3871) 
s_cmp_eq_i32 s13, 0 // (op 3871) 
s_mov_b32 s52, scc // (op 3871) 
s_cmp_eq_i32 s52, 0 // (op 3871) 
s_cselect_b32 s53, 0, -1 // (op 3871) 

s_and_b32 s52, s14, 31 // (op 3871) 
s_lshl_b32 s54, 1, s52 // (op 3871) 
s_add_i32 s55, s54, s53 // (op 3871) 


s_and_b32 s53, s1, s55 // (op 3871) 


s_add_i32 s1, s19, s53 // (op 3871) 


s_ashr_i32 s19, s1, s52 // (op 3871) 


s_lshl_b32 s1, s14, 24 // (op 3871) 
s_ashr_i32 s52, s1, 31 // (op 3871) 

s_xor_b32 s1, s19, s52 // (op 3871) 

s_sub_i32 s19, s1, s52 // (op 3871) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 3683) 
s_sub_u32 s52, s16, s1 // (op 3683) 

s_mul_i32 s1, s51, s44 // (op 3683) 
s_add_u32 s1, s1, s52 // (op 3683) 

s_mul_hi_i32 s52, s16, s10 // (op 3683) 
s_add_i32 s53, s52, s16 // (op 3683) 

s_ashr_i32 s52, s53, 31 // (op 3683) 
s_cmp_eq_i32 s10, 0 // (op 3683) 
s_mov_b32 s54, scc // (op 3683) 
s_cmp_eq_i32 s54, 0 // (op 3683) 
s_cselect_b32 s55, 0, -1 // (op 3683) 

s_and_b32 s54, s47, 31 // (op 3683) 
s_lshl_b32 s56, 1, s54 // (op 3683) 
s_add_i32 s57, s56, s55 // (op 3683) 


s_and_b32 s55, s52, s57 // (op 3683) 


s_add_i32 s52, s53, s55 // (op 3683) 


s_ashr_i32 s53, s52, s54 // (op 3683) 


s_lshl_b32 s52, s47, 24 // (op 3683) 
s_ashr_i32 s54, s52, 31 // (op 3683) 

s_xor_b32 s52, s53, s54 // (op 3683) 

s_sub_i32 s53, s52, s54 // (op 3683) 


s_mul_i32 s52, s0, s44 // (op 3871) 
s_sub_u32 s54, s16, s52 // (op 3871) 

s_mul_i32 s52, s19, s44 // (op 3871) 
s_add_u32 s52, s52, s54 // (op 3871) 

s_mul_hi_i32 s54, s16, s10 // (op 3871) 
s_add_i32 s55, s54, s16 // (op 3871) 

s_ashr_i32 s54, s55, 31 // (op 3871) 
s_cmp_eq_i32 s10, 0 // (op 3871) 
s_mov_b32 s56, scc // (op 3871) 
s_cmp_eq_i32 s56, 0 // (op 3871) 
s_cselect_b32 s57, 0, -1 // (op 3871) 

s_and_b32 s56, s47, 31 // (op 3871) 
s_lshl_b32 s58, 1, s56 // (op 3871) 
s_add_i32 s59, s58, s57 // (op 3871) 


s_and_b32 s57, s54, s59 // (op 3871) 


s_add_i32 s54, s55, s57 // (op 3871) 


s_ashr_i32 s55, s54, s56 // (op 3871) 


s_lshl_b32 s54, s47, 24 // (op 3871) 
s_ashr_i32 s56, s54, 31 // (op 3871) 

s_xor_b32 s54, s55, s56 // (op 3871) 

s_sub_i32 s55, s54, s56 // (op 3871) 


s_add_u32 s56, s6, 256 // (op 3683) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3683) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3683) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3683) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3683) 

s_mul_i32 s54, s44, s56 // (op 3683) 
s_mul_i32 s58, s48, s54 // (op 3683) 

s_mul_i32 s54, s53, s58 // (op 3683) 

s_sub_u32 s58, s16, s54 // (op 3683) 

s_mov_b32 s54, s58 // (op 3683) 
s_mul_hi_i32 s59, s54, s15 // (op 3683) 
s_add_i32 s60, s59, s54 // (op 3683) 


s_ashr_i32 s54, s60, 31 // (op 3683) 
s_cmp_eq_i32 s15, 0 // (op 3683) 
s_mov_b32 s59, scc // (op 3683) 
s_cmp_eq_i32 s59, 0 // (op 3683) 
s_cselect_b32 s61, 0, -1 // (op 3683) 

s_and_b32 s59, s46, 31 // (op 3683) 
s_lshl_b32 s62, 1, s59 // (op 3683) 
s_add_i32 s63, s62, s61 // (op 3683) 


s_and_b32 s61, s54, s63 // (op 3683) 


s_add_i32 s54, s60, s61 // (op 3683) 


s_ashr_i32 s60, s54, s59 // (op 3683) 


s_lshl_b32 s54, s46, 24 // (op 3683) 
s_ashr_i32 s59, s54, 31 // (op 3683) 

s_xor_b32 s54, s60, s59 // (op 3683) 

s_sub_i32 s60, s54, s59 // (op 3683) 


s_mul_i32 s54, s45, s44 // (op 3683) 
s_add_u32 s62, s8, 32 // (op 3683) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 3683) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3683) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3683) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 3683) 

s_sub_i32 s59, s62, s54 // (op 3683) 

s_mul_i32 s61, s60, s59 // (op 3683) 

s_sub_u32 s59, s58, s61 // (op 3683) 


s_add_u32 s58, s59, s54 // (op 3683) 


s_cmp_eq_i32 s53, 0 // (op 3683) 
s_mov_b32 s54, scc // (op 3683) 

s_cmp_eq_i32 s54, 0 // (op 3683) 
s_cselect_b32 s53, s58, s1 // (op 3683) 


s_mul_i32 s1, s51, s56 // (op 3683) 


v_lshl_add_u32 v9, s53, 5, v6 // (op 3683) 


s_sub_i32 s51, s18, s1 // (op 3683) 


s_cmp_eq_i32 s54, 0 // (op 3683) 
s_cselect_b32 s1, s60, s51 // (op 3683) 



v_lshl_add_u32 v6, s1, 8, v5 // (op 3683) 


v_mul_lo_u32 v5, v6, s2 // (op 3683) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 3683) 


s_add_u32 s56, s6, 256 // (op 3871) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3871) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3871) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3871) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3871) 

s_mul_i32 s1, s44, s56 // (op 3871) 
s_mul_i32 s18, s48, s1 // (op 3871) 

s_mul_i32 s1, s55, s18 // (op 3871) 

s_sub_u32 s18, s16, s1 // (op 3871) 

s_mov_b32 s1, s18 // (op 3871) 
s_mul_hi_i32 s51, s1, s15 // (op 3871) 
s_add_i32 s53, s51, s1 // (op 3871) 


s_ashr_i32 s1, s53, 31 // (op 3871) 
s_cmp_eq_i32 s15, 0 // (op 3871) 
s_mov_b32 s51, scc // (op 3871) 
s_cmp_eq_i32 s51, 0 // (op 3871) 
s_cselect_b32 s54, 0, -1 // (op 3871) 

s_and_b32 s51, s46, 31 // (op 3871) 
s_lshl_b32 s58, 1, s51 // (op 3871) 
s_add_i32 s59, s58, s54 // (op 3871) 


s_and_b32 s54, s1, s59 // (op 3871) 


s_add_i32 s1, s53, s54 // (op 3871) 


s_ashr_i32 s53, s1, s51 // (op 3871) 


s_lshl_b32 s1, s46, 24 // (op 3871) 
s_ashr_i32 s51, s1, 31 // (op 3871) 

s_xor_b32 s1, s53, s51 // (op 3871) 

s_sub_i32 s53, s1, s51 // (op 3871) 


s_mul_i32 s1, s45, s44 // (op 3871) 
s_add_u32 s58, s8, 32 // (op 3871) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 3871) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3871) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3871) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 3871) 

s_sub_i32 s51, s58, s1 // (op 3871) 

s_mul_i32 s54, s53, s51 // (op 3871) 

s_sub_u32 s51, s18, s54 // (op 3871) 


s_add_u32 s18, s51, s1 // (op 3871) 


s_cmp_eq_i32 s55, 0 // (op 3871) 
s_mov_b32 s1, scc // (op 3871) 

s_cmp_eq_i32 s1, 0 // (op 3871) 
s_cselect_b32 s51, s18, s52 // (op 3871) 


s_mul_i32 s18, s19, s56 // (op 3871) 


v_lshl_add_u32 v5, s51, 5, v12 // (op 3871) 


s_sub_i32 s19, s0, s18 // (op 3871) 


s_cmp_eq_i32 s1, 0 // (op 3871) 
s_cselect_b32 s0, s53, s19 // (op 3871) 



v_lshl_add_u32 v9, s0, 8, v8 // (op 3871) 


v_mul_lo_u32 v8, v9, s4 // (op 3871) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 3871) 


s_mul_i32 s1, s2, s7 // (op 3685) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 3685) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 3685) least significant
s_mul_hi_u32 s19, s2, s6 // (op 3685) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3685) most significant: sum
s_add_u32 s1, s1, s19 // (op 3685) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3685) 

s_mul_i32 s1, s4, s7 // (op 3873) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 3873) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 3873) least significant
s_mul_hi_u32 s52, s4, s6 // (op 3873) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3873) most significant: sum
s_add_u32 s1, s1, s52 // (op 3873) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 3873) 

s_mov_b32 s56, s20 // (op 3685) reg expression
s_mov_b32 s57, s21 // (op 3685) reg expression
s_mov_b32 s58, s18 // (op 3685) reg expression
s_mov_b32 s59, 131072 // (op 3685) call()

s_mov_b32 s60, s22 // (op 3873) reg expression

s_mov_b32 s61, s23 // (op 3873) reg expression
s_mov_b32 s62, s52 // (op 3873) reg expression
s_mov_b32 s63, 131072 // (op 3873) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_accvgpr_write a16, 0.00000 // (op 774) call()
v_accvgpr_write a17, 0.00000 // (op 774) call()
v_accvgpr_write a18, 0.00000 // (op 774) call()
v_accvgpr_write a19, 0.00000 // (op 774) call()
v_accvgpr_write a20, 0.00000 // (op 930) call()
v_accvgpr_write a21, 0.00000 // (op 930) call()
v_accvgpr_write a22, 0.00000 // (op 930) call()
v_accvgpr_write a23, 0.00000 // (op 930) call()
v_accvgpr_write a24, 0.00000 // (op 1086) call()
v_accvgpr_write a25, 0.00000 // (op 1086) call()
v_accvgpr_write a26, 0.00000 // (op 1086) call()
v_accvgpr_write a27, 0.00000 // (op 1086) call()
v_accvgpr_write a28, 0.00000 // (op 1242) call()
v_accvgpr_write a29, 0.00000 // (op 1242) call()
v_accvgpr_write a30, 0.00000 // (op 1242) call()
v_accvgpr_write a31, 0.00000 // (op 1242) call()
s_mul_hi_i32 s0, s16, s11 // (op 3450) 
v_lshrrev_b32 v12, 3, v1 // (op 3450) 
s_add_i32 s1, s0, s16 // (op 3450) 

v_and_b32 v13, -8, v1 // (op 3450) 
s_ashr_i32 s0, s1, 31 // (op 3450) 
v_sub_u32 v14, v1, v13 // (op 3450) 

s_cmp_eq_i32 s11, 0 // (op 3450) 
s_mov_b32 s18, scc // (op 3450) 
v_lshlrev_b32 v13, 4, v14 // (op 3450) 

s_cmp_eq_i32 s18, 0 // (op 3450) 
s_cselect_b32 s19, 0, -1 // (op 3450) 

v_mov_b32 v15, 0 // (op 3450) convert
v_mov_b32 v14, v13 // (op 3450) convert

s_and_b32 s18, s12, 31 // (op 3450) 
s_lshl_b32 s20, 1, s18 // (op 3450) 
s_add_i32 s21, s20, s19 // (op 3450) 


s_and_b32 s19, s0, s21 // (op 3450) 


s_add_i32 s0, s1, s19 // (op 3450) 


s_ashr_i32 s1, s0, s18 // (op 3450) 


s_lshl_b32 s0, s12, 24 // (op 3450) 
s_ashr_i32 s18, s0, 31 // (op 3450) 

s_xor_b32 s0, s1, s18 // (op 3450) 

s_sub_i32 s1, s0, s18 // (op 3450) 


s_mul_hi_i32 s0, s1, s13 // (op 3450) 
s_add_i32 s18, s0, s1 // (op 3450) 

s_ashr_i32 s0, s18, 31 // (op 3450) 
s_cmp_eq_i32 s13, 0 // (op 3450) 
s_mov_b32 s19, scc // (op 3450) 
s_cmp_eq_i32 s19, 0 // (op 3450) 
s_cselect_b32 s20, 0, -1 // (op 3450) 

s_and_b32 s19, s14, 31 // (op 3450) 
s_lshl_b32 s21, 1, s19 // (op 3450) 
s_add_i32 s22, s21, s20 // (op 3450) 


s_and_b32 s23, s0, s22 // (op 3450) 


s_add_i32 s0, s18, s23 // (op 3450) 


s_ashr_i32 s18, s0, s19 // (op 3450) 


s_lshl_b32 s0, s14, 24 // (op 3450) 
s_ashr_i32 s19, s0, 31 // (op 3450) 

s_xor_b32 s0, s18, s19 // (op 3450) 

s_sub_i32 s18, s0, s19 // (op 3450) 


s_mul_i32 s0, s1, s44 // (op 3450) 

s_sub_u32 s1, s16, s0 // (op 3450) 

s_mul_i32 s0, s18, s44 // (op 3450) 
s_add_u32 s0, s0, s1 // (op 3450) 


s_mul_hi_i32 s1, s16, s10 // (op 3450) 
s_add_i32 s18, s1, s16 // (op 3450) 

s_ashr_i32 s1, s18, 31 // (op 3450) 
s_cmp_eq_i32 s10, 0 // (op 3450) 
s_mov_b32 s19, scc // (op 3450) 
s_cmp_eq_i32 s19, 0 // (op 3450) 
s_cselect_b32 s20, 0, -1 // (op 3450) 

s_and_b32 s19, s47, 31 // (op 3450) 
s_lshl_b32 s21, 1, s19 // (op 3450) 
s_add_i32 s22, s21, s20 // (op 3450) 


s_and_b32 s23, s1, s22 // (op 3450) 


s_add_i32 s1, s18, s23 // (op 3450) 


s_ashr_i32 s18, s1, s19 // (op 3450) 


s_lshl_b32 s1, s47, 24 // (op 3450) 
s_ashr_i32 s19, s1, 31 // (op 3450) 

s_xor_b32 s1, s18, s19 // (op 3450) 

s_sub_i32 s18, s1, s19 // (op 3450) 


s_add_u32 s22, s6, 256 // (op 3450) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 3450) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 3450) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 3450) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 3450) 

s_mul_i32 s1, s44, s22 // (op 3450) 

s_mul_i32 s19, s48, s1 // (op 3450) 

s_mul_i32 s1, s18, s19 // (op 3450) 

s_sub_u32 s19, s16, s1 // (op 3450) 

s_mov_b32 s1, s19 // (op 3450) 
s_mul_hi_i32 s20, s1, s15 // (op 3450) 
s_add_i32 s21, s20, s1 // (op 3450) 


s_ashr_i32 s1, s21, 31 // (op 3450) 
s_cmp_eq_i32 s15, 0 // (op 3450) 
s_mov_b32 s20, scc // (op 3450) 
s_cmp_eq_i32 s20, 0 // (op 3450) 
s_cselect_b32 s22, 0, -1 // (op 3450) 

s_and_b32 s20, s46, 31 // (op 3450) 
s_lshl_b32 s23, 1, s20 // (op 3450) 
s_add_i32 s49, s23, s22 // (op 3450) 


s_and_b32 s22, s1, s49 // (op 3450) 


s_add_i32 s1, s21, s22 // (op 3450) 


s_ashr_i32 s21, s1, s20 // (op 3450) 


s_lshl_b32 s1, s46, 24 // (op 3450) 
s_ashr_i32 s20, s1, 31 // (op 3450) 

s_xor_b32 s1, s21, s20 // (op 3450) 

s_sub_i32 s21, s1, s20 // (op 3450) 


s_mul_i32 s1, s45, s44 // (op 3450) 
s_add_u32 s22, s8, 32 // (op 3450) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 3450) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 3450) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 3450) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 5 // (op 3450) 

s_sub_i32 s20, s22, s1 // (op 3450) 

s_mul_i32 s22, s21, s20 // (op 3450) 


s_sub_u32 s23, s19, s22 // (op 3450) 


s_add_u32 s19, s23, s1 // (op 3450) 


s_cmp_eq_i32 s18, 0 // (op 3450) 
s_mov_b32 s1, scc // (op 3450) 

s_cmp_eq_i32 s1, 0 // (op 3450) 
s_cselect_b32 s18, s19, s0 // (op 3450) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 3450) 


v_mul_lo_u32 v17, v13, s25 // (op 3450) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3450) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 3450) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3450) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3450) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3450) most significant half



s_mul_i32 s1, s24, s9 // (op 3452) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3452) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3452) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3452) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3452) most significant: sum
s_add_u32 s1, s1, s19 // (op 3452) most significant: sum


s_mov_b32 s20, s26 // (op 3452) reg expression
s_mov_b32 s21, s27 // (op 3452) reg expression
s_mov_b32 s22, s0 // (op 3452) reg expression
s_mov_b32 s23, 131072 // (op 3452) call()

s_mul_hi_i32 s0, s16, s11 // (op 3469) 
v_lshrrev_b32 v12, 3, v1 // (op 3469) 
s_add_i32 s1, s0, s16 // (op 3469) 

v_and_b32 v13, -8, v1 // (op 3469) 
s_ashr_i32 s0, s1, 31 // (op 3469) 
v_sub_u32 v14, v1, v13 // (op 3469) 

s_cmp_eq_i32 s11, 0 // (op 3469) 
s_mov_b32 s18, scc // (op 3469) 
v_lshlrev_b32 v13, 4, v14 // (op 3469) 

s_cmp_eq_i32 s18, 0 // (op 3469) 
s_cselect_b32 s19, 0, -1 // (op 3469) 

v_mov_b32 v15, 0 // (op 3469) convert
v_mov_b32 v14, v13 // (op 3469) convert

s_and_b32 s18, s12, 31 // (op 3469) 
s_lshl_b32 s49, 1, s18 // (op 3469) 
s_add_i32 s50, s49, s19 // (op 3469) 


s_and_b32 s19, s0, s50 // (op 3469) 


s_add_i32 s0, s1, s19 // (op 3469) 


s_ashr_i32 s1, s0, s18 // (op 3469) 


s_lshl_b32 s0, s12, 24 // (op 3469) 
s_ashr_i32 s18, s0, 31 // (op 3469) 

s_xor_b32 s0, s1, s18 // (op 3469) 

s_sub_i32 s1, s0, s18 // (op 3469) 


s_mul_hi_i32 s0, s1, s13 // (op 3469) 
s_add_i32 s18, s0, s1 // (op 3469) 

s_ashr_i32 s0, s18, 31 // (op 3469) 
s_cmp_eq_i32 s13, 0 // (op 3469) 
s_mov_b32 s19, scc // (op 3469) 
s_cmp_eq_i32 s19, 0 // (op 3469) 
s_cselect_b32 s49, 0, -1 // (op 3469) 

s_and_b32 s19, s14, 31 // (op 3469) 
s_lshl_b32 s50, 1, s19 // (op 3469) 
s_add_i32 s51, s50, s49 // (op 3469) 


s_and_b32 s49, s0, s51 // (op 3469) 


s_add_i32 s0, s18, s49 // (op 3469) 


s_ashr_i32 s18, s0, s19 // (op 3469) 


s_lshl_b32 s0, s14, 24 // (op 3469) 
s_ashr_i32 s19, s0, 31 // (op 3469) 

s_xor_b32 s0, s18, s19 // (op 3469) 

s_sub_i32 s18, s0, s19 // (op 3469) 


s_mul_i32 s0, s1, s44 // (op 3469) 

s_sub_u32 s1, s16, s0 // (op 3469) 

s_mul_i32 s0, s18, s44 // (op 3469) 
s_add_u32 s0, s0, s1 // (op 3469) 


s_mul_hi_i32 s1, s16, s10 // (op 3469) 
s_add_i32 s18, s1, s16 // (op 3469) 

s_ashr_i32 s1, s18, 31 // (op 3469) 
s_cmp_eq_i32 s10, 0 // (op 3469) 
s_mov_b32 s19, scc // (op 3469) 
s_cmp_eq_i32 s19, 0 // (op 3469) 
s_cselect_b32 s49, 0, -1 // (op 3469) 

s_and_b32 s19, s47, 31 // (op 3469) 
s_lshl_b32 s50, 1, s19 // (op 3469) 
s_add_i32 s51, s50, s49 // (op 3469) 


s_and_b32 s49, s1, s51 // (op 3469) 


s_add_i32 s1, s18, s49 // (op 3469) 


s_ashr_i32 s18, s1, s19 // (op 3469) 


s_lshl_b32 s1, s47, 24 // (op 3469) 
s_ashr_i32 s19, s1, 31 // (op 3469) 

s_xor_b32 s1, s18, s19 // (op 3469) 

s_sub_i32 s18, s1, s19 // (op 3469) 


s_add_u32 s54, s6, 256 // (op 3469) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3469) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3469) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3469) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3469) 

s_mul_i32 s1, s44, s50 // (op 3469) 

s_mul_i32 s19, s48, s1 // (op 3469) 

s_mul_i32 s1, s18, s19 // (op 3469) 

s_sub_u32 s19, s16, s1 // (op 3469) 

s_mov_b32 s1, s19 // (op 3469) 
s_mul_hi_i32 s49, s1, s15 // (op 3469) 
s_add_i32 s50, s49, s1 // (op 3469) 


s_ashr_i32 s1, s50, 31 // (op 3469) 
s_cmp_eq_i32 s15, 0 // (op 3469) 
s_mov_b32 s49, scc // (op 3469) 
s_cmp_eq_i32 s49, 0 // (op 3469) 
s_cselect_b32 s51, 0, -1 // (op 3469) 

s_and_b32 s49, s46, 31 // (op 3469) 
s_lshl_b32 s52, 1, s49 // (op 3469) 
s_add_i32 s53, s52, s51 // (op 3469) 


s_and_b32 s51, s1, s53 // (op 3469) 


s_add_i32 s1, s50, s51 // (op 3469) 


s_ashr_i32 s50, s1, s49 // (op 3469) 


s_lshl_b32 s1, s46, 24 // (op 3469) 
s_ashr_i32 s49, s1, 31 // (op 3469) 

s_xor_b32 s1, s50, s49 // (op 3469) 

s_sub_i32 s50, s1, s49 // (op 3469) 


s_mul_i32 s1, s45, s44 // (op 3469) 
s_add_u32 s54, s8, 32 // (op 3469) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 3469) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3469) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3469) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 5 // (op 3469) 

s_sub_i32 s49, s54, s1 // (op 3469) 

s_mul_i32 s51, s50, s49 // (op 3469) 


s_sub_u32 s49, s19, s51 // (op 3469) 


s_add_u32 s19, s49, s1 // (op 3469) 


s_cmp_eq_i32 s18, 0 // (op 3469) 
s_mov_b32 s1, scc // (op 3469) 

s_cmp_eq_i32 s1, 0 // (op 3469) 
s_cselect_b32 s18, s19, s0 // (op 3469) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 3469) 


v_mul_lo_u32 v19, v13, s25 // (op 3469) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3469) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 3469) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3469) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3469) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3469) most significant half



s_mul_i32 s1, s24, s9 // (op 3471) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3471) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3471) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3471) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3471) most significant: sum
s_add_u32 s1, s1, s19 // (op 3471) most significant: sum


s_mov_b32 s20, s26 // (op 3471) reg expression
s_mov_b32 s21, s27 // (op 3471) reg expression
s_mov_b32 s22, s0 // (op 3471) reg expression
s_mov_b32 s23, 131072 // (op 3471) call()

s_mul_hi_i32 s0, s16, s11 // (op 3514) 
v_and_b32 v12, -4, v1 // (op 3514) 
s_add_i32 s1, s0, s16 // (op 3514) 

v_sub_u32 v13, v1, v12 // (op 3514) 

s_ashr_i32 s0, s1, 31 // (op 3514) 
v_lshlrev_b32 v12, 6, v13 // (op 3514) 

s_cmp_eq_i32 s11, 0 // (op 3514) 
s_mov_b32 s18, scc // (op 3514) 
v_lshrrev_b32 v13, 2, v1 // (op 3514) 
s_cmp_eq_i32 s18, 0 // (op 3514) 
s_cselect_b32 s19, 0, -1 // (op 3514) 

v_lshlrev_b32 v14, 1, v13 // (op 3514) 

s_and_b32 s18, s12, 31 // (op 3514) 
s_lshl_b32 s49, 1, s18 // (op 3514) 
s_add_i32 s50, s49, s19 // (op 3514) 


s_and_b32 s19, s0, s50 // (op 3514) 


s_add_i32 s0, s1, s19 // (op 3514) 


s_ashr_i32 s1, s0, s18 // (op 3514) 


s_lshl_b32 s0, s12, 24 // (op 3514) 
s_ashr_i32 s18, s0, 31 // (op 3514) 

s_xor_b32 s0, s1, s18 // (op 3514) 

s_sub_i32 s1, s0, s18 // (op 3514) 


s_mul_hi_i32 s0, s1, s13 // (op 3514) 
s_add_i32 s18, s0, s1 // (op 3514) 

s_ashr_i32 s0, s18, 31 // (op 3514) 
s_cmp_eq_i32 s13, 0 // (op 3514) 
s_mov_b32 s19, scc // (op 3514) 
s_cmp_eq_i32 s19, 0 // (op 3514) 
s_cselect_b32 s49, 0, -1 // (op 3514) 

s_and_b32 s19, s14, 31 // (op 3514) 
s_lshl_b32 s50, 1, s19 // (op 3514) 
s_add_i32 s51, s50, s49 // (op 3514) 


s_and_b32 s49, s0, s51 // (op 3514) 


s_add_i32 s0, s18, s49 // (op 3514) 


s_ashr_i32 s18, s0, s19 // (op 3514) 


s_lshl_b32 s0, s14, 24 // (op 3514) 
s_ashr_i32 s19, s0, 31 // (op 3514) 

s_xor_b32 s0, s18, s19 // (op 3514) 

s_sub_i32 s18, s0, s19 // (op 3514) 


s_add_u32 s54, s6, 256 // (op 3514) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3514) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3514) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3514) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3514) 

s_mul_i32 s0, s18, s50 // (op 3514) 

s_sub_i32 s49, s1, s0 // (op 3514) 


s_mul_hi_i32 s0, s16, s10 // (op 3514) 
s_add_i32 s1, s0, s16 // (op 3514) 

s_ashr_i32 s0, s1, 31 // (op 3514) 
s_cmp_eq_i32 s10, 0 // (op 3514) 
s_mov_b32 s18, scc // (op 3514) 
s_cmp_eq_i32 s18, 0 // (op 3514) 
s_cselect_b32 s19, 0, -1 // (op 3514) 

s_and_b32 s18, s47, 31 // (op 3514) 
s_lshl_b32 s52, 1, s18 // (op 3514) 
s_add_i32 s53, s52, s19 // (op 3514) 


s_and_b32 s19, s0, s53 // (op 3514) 


s_add_i32 s0, s1, s19 // (op 3514) 


s_ashr_i32 s1, s0, s18 // (op 3514) 


s_lshl_b32 s0, s47, 24 // (op 3514) 
s_ashr_i32 s18, s0, 31 // (op 3514) 

s_xor_b32 s0, s1, s18 // (op 3514) 

s_sub_i32 s1, s0, s18 // (op 3514) 


s_mul_i32 s0, s44, s50 // (op 3514) 

s_mul_i32 s18, s48, s0 // (op 3514) 

s_mul_i32 s0, s1, s18 // (op 3514) 

s_sub_u32 s18, s16, s0 // (op 3514) 

s_mul_hi_i32 s0, s18, s15 // (op 3514) 
s_add_i32 s19, s0, s18 // (op 3514) 


s_ashr_i32 s0, s19, 31 // (op 3514) 
s_cmp_eq_i32 s15, 0 // (op 3514) 
s_mov_b32 s18, scc // (op 3514) 
s_cmp_eq_i32 s18, 0 // (op 3514) 
s_cselect_b32 s50, 0, -1 // (op 3514) 

s_and_b32 s18, s46, 31 // (op 3514) 
s_lshl_b32 s51, 1, s18 // (op 3514) 
s_add_i32 s52, s51, s50 // (op 3514) 


s_and_b32 s50, s0, s52 // (op 3514) 


s_add_i32 s0, s19, s50 // (op 3514) 


s_ashr_i32 s19, s0, s18 // (op 3514) 


s_lshl_b32 s0, s46, 24 // (op 3514) 
s_ashr_i32 s18, s0, 31 // (op 3514) 

s_xor_b32 s0, s19, s18 // (op 3514) 

s_sub_i32 s19, s0, s18 // (op 3514) 


s_cmp_eq_i32 s1, 0 // (op 3514) 
s_mov_b32 s0, scc // (op 3514) 

s_cmp_eq_i32 s0, 0 // (op 3514) 
s_cselect_b32 s1, s19, s49 // (op 3514) 



v_lshl_add_u32 v13, s1, 8, v12 // (op 3514) 


v_mov_b32 v21, 0 // (op 3514) convert
v_mov_b32 v20, v13 // (op 3514) convert

v_mul_lo_u32 v13, v14, s29 // (op 3514) most significant: low of low * high
v_mul_hi_u32 v15, v14, s28 // (op 3514) most significant: high of low * low
v_mul_lo_u32 v12, v14, s28 // (op 3514) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3514) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v20 // (op 3514) least significant half
v_addc_co_u32 v13, s[0:1], v13, v21, s[0:1] // (op 3514) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3514) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 3514) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3514) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 3514) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3514) 

v_mov_b32 v14, v12 // (op 3514) convert

s_mul_i32 s1, s33, 6 // (op 3516) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3516) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3516) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3516) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3516) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3516) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3516) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3516) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3516) most significant: sum
s_add_u32 s19, s19, s50 // (op 3516) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3516) 

s_mov_b32 s52, s30 // (op 3516) reg expression
s_mov_b32 s53, s31 // (op 3516) reg expression
s_mov_b32 s54, s0 // (op 3516) reg expression
s_mov_b32 s55, 131072 // (op 3516) call()

s_mul_hi_i32 s0, s16, s11 // (op 3533) 
v_and_b32 v15, -4, v1 // (op 3533) 
s_add_i32 s1, s0, s16 // (op 3533) 

v_sub_u32 v12, v1, v15 // (op 3533) 

s_ashr_i32 s0, s1, 31 // (op 3533) 
v_lshlrev_b32 v13, 6, v12 // (op 3533) 

s_cmp_eq_i32 s11, 0 // (op 3533) 
s_mov_b32 s18, scc // (op 3533) 
v_lshrrev_b32 v12, 2, v1 // (op 3533) 
s_cmp_eq_i32 s18, 0 // (op 3533) 
s_cselect_b32 s19, 0, -1 // (op 3533) 

v_lshlrev_b32 v15, 1, v12 // (op 3533) 

s_and_b32 s18, s12, 31 // (op 3533) 
s_lshl_b32 s49, 1, s18 // (op 3533) 
s_add_i32 s50, s49, s19 // (op 3533) 


s_and_b32 s19, s0, s50 // (op 3533) 


s_add_i32 s0, s1, s19 // (op 3533) 


s_ashr_i32 s1, s0, s18 // (op 3533) 


s_lshl_b32 s0, s12, 24 // (op 3533) 
s_ashr_i32 s18, s0, 31 // (op 3533) 

s_xor_b32 s0, s1, s18 // (op 3533) 

s_sub_i32 s1, s0, s18 // (op 3533) 


s_mul_hi_i32 s0, s1, s13 // (op 3533) 
s_add_i32 s18, s0, s1 // (op 3533) 

s_ashr_i32 s0, s18, 31 // (op 3533) 
s_cmp_eq_i32 s13, 0 // (op 3533) 
s_mov_b32 s19, scc // (op 3533) 
s_cmp_eq_i32 s19, 0 // (op 3533) 
s_cselect_b32 s49, 0, -1 // (op 3533) 

s_and_b32 s19, s14, 31 // (op 3533) 
s_lshl_b32 s50, 1, s19 // (op 3533) 
s_add_i32 s51, s50, s49 // (op 3533) 


s_and_b32 s49, s0, s51 // (op 3533) 


s_add_i32 s0, s18, s49 // (op 3533) 


s_ashr_i32 s18, s0, s19 // (op 3533) 


s_lshl_b32 s0, s14, 24 // (op 3533) 
s_ashr_i32 s19, s0, 31 // (op 3533) 

s_xor_b32 s0, s18, s19 // (op 3533) 

s_sub_i32 s18, s0, s19 // (op 3533) 


s_add_u32 s50, s6, 256 // (op 3533) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3533) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3533) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3533) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3533) 

s_mul_i32 s0, s18, s50 // (op 3533) 

s_sub_i32 s49, s1, s0 // (op 3533) 


s_mul_hi_i32 s0, s16, s10 // (op 3533) 
s_add_i32 s1, s0, s16 // (op 3533) 

s_ashr_i32 s0, s1, 31 // (op 3533) 
s_cmp_eq_i32 s10, 0 // (op 3533) 
s_mov_b32 s18, scc // (op 3533) 
s_cmp_eq_i32 s18, 0 // (op 3533) 
s_cselect_b32 s19, 0, -1 // (op 3533) 

s_and_b32 s18, s47, 31 // (op 3533) 
s_lshl_b32 s64, 1, s18 // (op 3533) 
s_add_i32 s65, s64, s19 // (op 3533) 


s_and_b32 s19, s0, s65 // (op 3533) 


s_add_i32 s0, s1, s19 // (op 3533) 


s_ashr_i32 s1, s0, s18 // (op 3533) 


s_lshl_b32 s0, s47, 24 // (op 3533) 
s_ashr_i32 s18, s0, 31 // (op 3533) 

s_xor_b32 s0, s1, s18 // (op 3533) 

s_sub_i32 s1, s0, s18 // (op 3533) 


s_mul_i32 s0, s44, s50 // (op 3533) 

s_mul_i32 s18, s48, s0 // (op 3533) 

s_mul_i32 s0, s1, s18 // (op 3533) 

s_sub_u32 s18, s16, s0 // (op 3533) 

s_mul_hi_i32 s0, s18, s15 // (op 3533) 
s_add_i32 s19, s0, s18 // (op 3533) 


s_ashr_i32 s0, s19, 31 // (op 3533) 
s_cmp_eq_i32 s15, 0 // (op 3533) 
s_mov_b32 s18, scc // (op 3533) 
s_cmp_eq_i32 s18, 0 // (op 3533) 
s_cselect_b32 s50, 0, -1 // (op 3533) 

s_and_b32 s18, s46, 31 // (op 3533) 
s_lshl_b32 s51, 1, s18 // (op 3533) 
s_add_i32 s64, s51, s50 // (op 3533) 


s_and_b32 s50, s0, s64 // (op 3533) 


s_add_i32 s0, s19, s50 // (op 3533) 


s_ashr_i32 s19, s0, s18 // (op 3533) 


s_lshl_b32 s0, s46, 24 // (op 3533) 
s_ashr_i32 s18, s0, 31 // (op 3533) 

s_xor_b32 s0, s19, s18 // (op 3533) 

s_sub_i32 s19, s0, s18 // (op 3533) 


s_cmp_eq_i32 s1, 0 // (op 3533) 
s_mov_b32 s0, scc // (op 3533) 

s_cmp_eq_i32 s0, 0 // (op 3533) 
s_cselect_b32 s1, s19, s49 // (op 3533) 



v_lshl_add_u32 v12, s1, 8, v13 // (op 3533) 


v_mov_b32 v21, 0 // (op 3533) convert
v_mov_b32 v20, v12 // (op 3533) convert

v_mul_lo_u32 v13, v15, s29 // (op 3533) most significant: low of low * high
v_mul_hi_u32 v22, v15, s28 // (op 3533) most significant: high of low * low
v_mul_lo_u32 v12, v15, s28 // (op 3533) least significant: low of low * low
v_add_u32 v13, v13, v22 // (op 3533) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v20 // (op 3533) least significant half
v_addc_co_u32 v13, s[0:1], v13, v21, s[0:1] // (op 3533) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 3533) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 3533) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 3533) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3533) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 3533) 

v_mov_b32 v15, v12 // (op 3533) convert

s_mul_i32 s1, s33, 6 // (op 3535) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3535) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3535) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3535) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3535) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3535) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3535) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3535) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3535) most significant: sum
s_add_u32 s19, s19, s50 // (op 3535) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3535) 

s_mov_b32 s52, s30 // (op 3535) reg expression
s_mov_b32 s53, s31 // (op 3535) reg expression
s_mov_b32 s54, s0 // (op 3535) reg expression
s_mov_b32 s55, 131072 // (op 3535) call()

s_mul_hi_i32 s0, s16, s11 // (op 3578) 
v_lshrrev_b32 v12, 7, v1 // (op 3578) 
s_add_i32 s1, s0, s16 // (op 3578) 

v_and_b32 v13, 15, v1 // (op 3578) 
s_ashr_i32 s0, s1, 31 // (op 3578) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 3578) 


s_cmp_eq_i32 s11, 0 // (op 3578) 
s_mov_b32 s18, scc // (op 3578) 
v_and_b32 v12, 63, v1 // (op 3578) 
s_cmp_eq_i32 s18, 0 // (op 3578) 
s_cselect_b32 s19, 0, -1 // (op 3578) 

v_lshrrev_b32 v13, 4, v12 // (op 3578) 

s_and_b32 s18, s12, 31 // (op 3578) 
s_lshl_b32 s49, 1, s18 // (op 3578) 
s_add_i32 s50, s49, s19 // (op 3578) 


s_and_b32 s19, s0, s50 // (op 3578) 


s_add_i32 s0, s1, s19 // (op 3578) 


s_ashr_i32 s1, s0, s18 // (op 3578) 


s_lshl_b32 s0, s12, 24 // (op 3578) 
s_ashr_i32 s18, s0, 31 // (op 3578) 

s_xor_b32 s0, s1, s18 // (op 3578) 

s_sub_i32 s1, s0, s18 // (op 3578) 


s_mul_hi_i32 s0, s1, s13 // (op 3578) 
s_add_i32 s18, s0, s1 // (op 3578) 

s_ashr_i32 s0, s18, 31 // (op 3578) 
s_cmp_eq_i32 s13, 0 // (op 3578) 
s_mov_b32 s19, scc // (op 3578) 
s_cmp_eq_i32 s19, 0 // (op 3578) 
s_cselect_b32 s49, 0, -1 // (op 3578) 

s_and_b32 s19, s14, 31 // (op 3578) 
s_lshl_b32 s50, 1, s19 // (op 3578) 
s_add_i32 s51, s50, s49 // (op 3578) 


s_and_b32 s49, s0, s51 // (op 3578) 


s_add_i32 s0, s18, s49 // (op 3578) 


s_ashr_i32 s18, s0, s19 // (op 3578) 


s_lshl_b32 s0, s14, 24 // (op 3578) 
s_ashr_i32 s19, s0, 31 // (op 3578) 

s_xor_b32 s0, s18, s19 // (op 3578) 

s_sub_i32 s18, s0, s19 // (op 3578) 


s_mul_i32 s0, s1, s44 // (op 3578) 

s_sub_u32 s1, s16, s0 // (op 3578) 

s_mul_i32 s0, s18, s44 // (op 3578) 
s_add_u32 s0, s0, s1 // (op 3578) 


s_mul_hi_i32 s1, s16, s10 // (op 3578) 
s_add_i32 s18, s1, s16 // (op 3578) 

s_ashr_i32 s1, s18, 31 // (op 3578) 
s_cmp_eq_i32 s10, 0 // (op 3578) 
s_mov_b32 s19, scc // (op 3578) 
s_cmp_eq_i32 s19, 0 // (op 3578) 
s_cselect_b32 s49, 0, -1 // (op 3578) 

s_and_b32 s19, s47, 31 // (op 3578) 
s_lshl_b32 s50, 1, s19 // (op 3578) 
s_add_i32 s51, s50, s49 // (op 3578) 


s_and_b32 s49, s1, s51 // (op 3578) 


s_add_i32 s1, s18, s49 // (op 3578) 


s_ashr_i32 s18, s1, s19 // (op 3578) 


s_lshl_b32 s1, s47, 24 // (op 3578) 
s_ashr_i32 s19, s1, 31 // (op 3578) 

s_xor_b32 s1, s18, s19 // (op 3578) 

s_sub_i32 s18, s1, s19 // (op 3578) 


s_add_u32 s50, s6, 256 // (op 3578) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3578) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3578) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3578) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3578) 

s_mul_i32 s1, s44, s50 // (op 3578) 

s_mul_i32 s19, s48, s1 // (op 3578) 

s_mul_i32 s1, s18, s19 // (op 3578) 

s_sub_u32 s19, s16, s1 // (op 3578) 

s_mov_b32 s1, s19 // (op 3578) 
s_mul_hi_i32 s49, s1, s15 // (op 3578) 
s_add_i32 s50, s49, s1 // (op 3578) 


s_ashr_i32 s1, s50, 31 // (op 3578) 
s_cmp_eq_i32 s15, 0 // (op 3578) 
s_mov_b32 s49, scc // (op 3578) 
s_cmp_eq_i32 s49, 0 // (op 3578) 
s_cselect_b32 s51, 0, -1 // (op 3578) 

s_and_b32 s49, s46, 31 // (op 3578) 
s_lshl_b32 s64, 1, s49 // (op 3578) 
s_add_i32 s65, s64, s51 // (op 3578) 


s_and_b32 s51, s1, s65 // (op 3578) 


s_add_i32 s1, s50, s51 // (op 3578) 


s_ashr_i32 s50, s1, s49 // (op 3578) 


s_lshl_b32 s1, s46, 24 // (op 3578) 
s_ashr_i32 s49, s1, 31 // (op 3578) 

s_xor_b32 s1, s50, s49 // (op 3578) 

s_sub_i32 s50, s1, s49 // (op 3578) 


s_mul_i32 s1, s45, s44 // (op 3578) 
s_add_u32 s64, s8, 32 // (op 3578) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 3578) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3578) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3578) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 3578) 

s_sub_i32 s49, s64, s1 // (op 3578) 

s_mul_i32 s51, s50, s49 // (op 3578) 


s_sub_u32 s49, s19, s51 // (op 3578) 


s_add_u32 s19, s49, s1 // (op 3578) 


s_cmp_eq_i32 s18, 0 // (op 3578) 
s_mov_b32 s1, scc // (op 3578) 

s_cmp_eq_i32 s1, 0 // (op 3578) 
s_cselect_b32 s18, s19, s0 // (op 3578) 



v_lshl_add_u32 v12, s18, 5, v20 // (op 3578) 


v_mul_lo_u32 v20, v12, s34 // (op 3578) 
v_add_u32 v20, v20, v13 // (op 3578) 


s_mul_i32 s1, s34, s9 // (op 3580) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3580) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3580) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3580) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3580) most significant: sum
s_add_u32 s1, s1, s19 // (op 3580) most significant: sum


s_mov_b32 s64, s36 // (op 3580) reg expression
s_mov_b32 s65, s37 // (op 3580) reg expression
s_mov_b32 s66, s0 // (op 3580) reg expression
s_mov_b32 s67, 131072 // (op 3580) call()

s_mul_hi_i32 s0, s16, s11 // (op 3597) 
v_lshrrev_b32 v12, 7, v1 // (op 3597) 
s_add_i32 s1, s0, s16 // (op 3597) 

v_and_b32 v13, 15, v1 // (op 3597) 
s_ashr_i32 s0, s1, 31 // (op 3597) 
v_lshl_add_u32 v21, v12, 4, v13 // (op 3597) 


s_cmp_eq_i32 s11, 0 // (op 3597) 
s_mov_b32 s18, scc // (op 3597) 
v_and_b32 v12, 63, v1 // (op 3597) 
s_cmp_eq_i32 s18, 0 // (op 3597) 
s_cselect_b32 s19, 0, -1 // (op 3597) 

v_lshrrev_b32 v13, 4, v12 // (op 3597) 

s_and_b32 s18, s12, 31 // (op 3597) 
s_lshl_b32 s49, 1, s18 // (op 3597) 
s_add_i32 s50, s49, s19 // (op 3597) 


s_and_b32 s19, s0, s50 // (op 3597) 


s_add_i32 s0, s1, s19 // (op 3597) 


s_ashr_i32 s1, s0, s18 // (op 3597) 


s_lshl_b32 s0, s12, 24 // (op 3597) 
s_ashr_i32 s18, s0, 31 // (op 3597) 

s_xor_b32 s0, s1, s18 // (op 3597) 

s_sub_i32 s1, s0, s18 // (op 3597) 


s_mul_hi_i32 s0, s1, s13 // (op 3597) 
s_add_i32 s18, s0, s1 // (op 3597) 

s_ashr_i32 s0, s18, 31 // (op 3597) 
s_cmp_eq_i32 s13, 0 // (op 3597) 
s_mov_b32 s19, scc // (op 3597) 
s_cmp_eq_i32 s19, 0 // (op 3597) 
s_cselect_b32 s49, 0, -1 // (op 3597) 

s_and_b32 s19, s14, 31 // (op 3597) 
s_lshl_b32 s50, 1, s19 // (op 3597) 
s_add_i32 s51, s50, s49 // (op 3597) 


s_and_b32 s49, s0, s51 // (op 3597) 


s_add_i32 s0, s18, s49 // (op 3597) 


s_ashr_i32 s18, s0, s19 // (op 3597) 


s_lshl_b32 s0, s14, 24 // (op 3597) 
s_ashr_i32 s19, s0, 31 // (op 3597) 

s_xor_b32 s0, s18, s19 // (op 3597) 

s_sub_i32 s18, s0, s19 // (op 3597) 


s_mul_i32 s0, s1, s44 // (op 3597) 

s_sub_u32 s1, s16, s0 // (op 3597) 

s_mul_i32 s0, s18, s44 // (op 3597) 
s_add_u32 s0, s0, s1 // (op 3597) 


s_mul_hi_i32 s1, s16, s10 // (op 3597) 
s_add_i32 s18, s1, s16 // (op 3597) 

s_ashr_i32 s1, s18, 31 // (op 3597) 
s_cmp_eq_i32 s10, 0 // (op 3597) 
s_mov_b32 s19, scc // (op 3597) 
s_cmp_eq_i32 s19, 0 // (op 3597) 
s_cselect_b32 s49, 0, -1 // (op 3597) 

s_and_b32 s19, s47, 31 // (op 3597) 
s_lshl_b32 s50, 1, s19 // (op 3597) 
s_add_i32 s51, s50, s49 // (op 3597) 


s_and_b32 s49, s1, s51 // (op 3597) 


s_add_i32 s1, s18, s49 // (op 3597) 


s_ashr_i32 s18, s1, s19 // (op 3597) 


s_lshl_b32 s1, s47, 24 // (op 3597) 
s_ashr_i32 s19, s1, 31 // (op 3597) 

s_xor_b32 s1, s18, s19 // (op 3597) 

s_sub_i32 s18, s1, s19 // (op 3597) 


s_add_u32 s50, s6, 256 // (op 3597) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3597) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3597) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3597) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3597) 

s_mul_i32 s1, s44, s50 // (op 3597) 

s_mul_i32 s19, s48, s1 // (op 3597) 

s_mul_i32 s1, s18, s19 // (op 3597) 

s_sub_u32 s19, s16, s1 // (op 3597) 

s_mov_b32 s1, s19 // (op 3597) 
s_mul_hi_i32 s49, s1, s15 // (op 3597) 
s_add_i32 s50, s49, s1 // (op 3597) 


s_ashr_i32 s1, s50, 31 // (op 3597) 
s_cmp_eq_i32 s15, 0 // (op 3597) 
s_mov_b32 s49, scc // (op 3597) 
s_cmp_eq_i32 s49, 0 // (op 3597) 
s_cselect_b32 s51, 0, -1 // (op 3597) 

s_and_b32 s49, s46, 31 // (op 3597) 
s_lshl_b32 s68, 1, s49 // (op 3597) 
s_add_i32 s69, s68, s51 // (op 3597) 


s_and_b32 s51, s1, s69 // (op 3597) 


s_add_i32 s1, s50, s51 // (op 3597) 


s_ashr_i32 s50, s1, s49 // (op 3597) 


s_lshl_b32 s1, s46, 24 // (op 3597) 
s_ashr_i32 s49, s1, 31 // (op 3597) 

s_xor_b32 s1, s50, s49 // (op 3597) 

s_sub_i32 s50, s1, s49 // (op 3597) 


s_mul_i32 s1, s45, s44 // (op 3597) 
s_add_u32 s68, s8, 32 // (op 3597) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 3597) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3597) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3597) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 3597) 

s_sub_i32 s49, s68, s1 // (op 3597) 

s_mul_i32 s51, s50, s49 // (op 3597) 


s_sub_u32 s49, s19, s51 // (op 3597) 


s_add_u32 s19, s49, s1 // (op 3597) 


s_cmp_eq_i32 s18, 0 // (op 3597) 
s_mov_b32 s1, scc // (op 3597) 

s_cmp_eq_i32 s1, 0 // (op 3597) 
s_cselect_b32 s18, s19, s0 // (op 3597) 



v_lshl_add_u32 v12, s18, 5, v21 // (op 3597) 


v_mul_lo_u32 v21, v12, s34 // (op 3597) 
v_add_u32 v21, v21, v13 // (op 3597) 


s_mul_i32 s1, s34, s9 // (op 3599) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3599) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3599) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3599) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3599) most significant: sum
s_add_u32 s1, s1, s19 // (op 3599) most significant: sum


s_mov_b32 s64, s36 // (op 3599) reg expression
s_mov_b32 s65, s37 // (op 3599) reg expression
s_mov_b32 s66, s0 // (op 3599) reg expression
s_mov_b32 s67, 131072 // (op 3599) call()

s_mul_hi_i32 s0, s16, s11 // (op 3642) 
v_and_b32 v12, -64, v1 // (op 3642) 
s_add_i32 s1, s0, s16 // (op 3642) 

v_sub_u32 v13, v1, v12 // (op 3642) 

s_ashr_i32 s0, s1, 31 // (op 3642) 
v_lshlrev_b32 v12, 2, v13 // (op 3642) 

s_cmp_eq_i32 s11, 0 // (op 3642) 
s_mov_b32 s18, scc // (op 3642) 
v_lshrrev_b32 v13, 6, v1 // (op 3642) 
s_cmp_eq_i32 s18, 0 // (op 3642) 
s_cselect_b32 s19, 0, -1 // (op 3642) 

s_and_b32 s18, s12, 31 // (op 3642) 
s_lshl_b32 s49, 1, s18 // (op 3642) 
s_add_i32 s50, s49, s19 // (op 3642) 


s_and_b32 s19, s0, s50 // (op 3642) 


s_add_i32 s0, s1, s19 // (op 3642) 


s_ashr_i32 s1, s0, s18 // (op 3642) 


s_lshl_b32 s0, s12, 24 // (op 3642) 
s_ashr_i32 s18, s0, 31 // (op 3642) 

s_xor_b32 s0, s1, s18 // (op 3642) 

s_sub_i32 s1, s0, s18 // (op 3642) 


s_mul_hi_i32 s0, s1, s13 // (op 3642) 
s_add_i32 s18, s0, s1 // (op 3642) 

s_ashr_i32 s0, s18, 31 // (op 3642) 
s_cmp_eq_i32 s13, 0 // (op 3642) 
s_mov_b32 s19, scc // (op 3642) 
s_cmp_eq_i32 s19, 0 // (op 3642) 
s_cselect_b32 s49, 0, -1 // (op 3642) 

s_and_b32 s19, s14, 31 // (op 3642) 
s_lshl_b32 s50, 1, s19 // (op 3642) 
s_add_i32 s51, s50, s49 // (op 3642) 


s_and_b32 s49, s0, s51 // (op 3642) 


s_add_i32 s0, s18, s49 // (op 3642) 


s_ashr_i32 s18, s0, s19 // (op 3642) 


s_lshl_b32 s0, s14, 24 // (op 3642) 
s_ashr_i32 s19, s0, 31 // (op 3642) 

s_xor_b32 s0, s18, s19 // (op 3642) 

s_sub_i32 s18, s0, s19 // (op 3642) 


s_add_u32 s50, s6, 256 // (op 3642) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3642) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3642) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3642) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3642) 

s_mul_i32 s0, s18, s50 // (op 3642) 

s_sub_i32 s49, s1, s0 // (op 3642) 


s_mul_hi_i32 s0, s16, s10 // (op 3642) 
s_add_i32 s1, s0, s16 // (op 3642) 

s_ashr_i32 s0, s1, 31 // (op 3642) 
s_cmp_eq_i32 s10, 0 // (op 3642) 
s_mov_b32 s18, scc // (op 3642) 
s_cmp_eq_i32 s18, 0 // (op 3642) 
s_cselect_b32 s19, 0, -1 // (op 3642) 

s_and_b32 s18, s47, 31 // (op 3642) 
s_lshl_b32 s68, 1, s18 // (op 3642) 
s_add_i32 s69, s68, s19 // (op 3642) 


s_and_b32 s19, s0, s69 // (op 3642) 


s_add_i32 s0, s1, s19 // (op 3642) 


s_ashr_i32 s1, s0, s18 // (op 3642) 


s_lshl_b32 s0, s47, 24 // (op 3642) 
s_ashr_i32 s18, s0, 31 // (op 3642) 

s_xor_b32 s0, s1, s18 // (op 3642) 

s_sub_i32 s1, s0, s18 // (op 3642) 


s_mul_i32 s0, s44, s50 // (op 3642) 

s_mul_i32 s18, s48, s0 // (op 3642) 

s_mul_i32 s0, s1, s18 // (op 3642) 

s_sub_u32 s18, s16, s0 // (op 3642) 

s_mul_hi_i32 s0, s18, s15 // (op 3642) 
s_add_i32 s19, s0, s18 // (op 3642) 


s_ashr_i32 s0, s19, 31 // (op 3642) 
s_cmp_eq_i32 s15, 0 // (op 3642) 
s_mov_b32 s18, scc // (op 3642) 
s_cmp_eq_i32 s18, 0 // (op 3642) 
s_cselect_b32 s50, 0, -1 // (op 3642) 

s_and_b32 s18, s46, 31 // (op 3642) 
s_lshl_b32 s51, 1, s18 // (op 3642) 
s_add_i32 s68, s51, s50 // (op 3642) 


s_and_b32 s50, s0, s68 // (op 3642) 


s_add_i32 s0, s19, s50 // (op 3642) 


s_ashr_i32 s19, s0, s18 // (op 3642) 


s_lshl_b32 s0, s46, 24 // (op 3642) 
s_ashr_i32 s18, s0, 31 // (op 3642) 

s_xor_b32 s0, s19, s18 // (op 3642) 

s_sub_i32 s19, s0, s18 // (op 3642) 


s_cmp_eq_i32 s1, 0 // (op 3642) 
s_mov_b32 s0, scc // (op 3642) 

s_cmp_eq_i32 s0, 0 // (op 3642) 
s_cselect_b32 s1, s19, s49 // (op 3642) 



v_lshl_add_u32 v22, s1, 8, v12 // (op 3642) 


v_mul_lo_u32 v12, v13, s38 // (op 3642) 
v_add_u32 v12, v12, v22 // (op 3642) 


s_mul_i32 s1, s38, s43 // (op 3644) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3644) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3644) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3644) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3644) most significant: sum
s_add_u32 s1, s1, s19 // (op 3644) most significant: sum


s_mov_b32 s68, s40 // (op 3644) reg expression
s_mov_b32 s69, s41 // (op 3644) reg expression
s_mov_b32 s70, s0 // (op 3644) reg expression
s_mov_b32 s71, 131072 // (op 3644) call()

s_mul_hi_i32 s0, s16, s11 // (op 3661) 
v_and_b32 v13, -64, v1 // (op 3661) 
s_add_i32 s1, s0, s16 // (op 3661) 

v_sub_u32 v22, v1, v13 // (op 3661) 

s_ashr_i32 s0, s1, 31 // (op 3661) 
v_lshlrev_b32 v13, 2, v22 // (op 3661) 

s_cmp_eq_i32 s11, 0 // (op 3661) 
s_mov_b32 s18, scc // (op 3661) 
v_lshrrev_b32 v22, 6, v1 // (op 3661) 
s_cmp_eq_i32 s18, 0 // (op 3661) 
s_cselect_b32 s19, 0, -1 // (op 3661) 

s_and_b32 s18, s12, 31 // (op 3661) 
s_lshl_b32 s49, 1, s18 // (op 3661) 
s_add_i32 s50, s49, s19 // (op 3661) 


s_and_b32 s19, s0, s50 // (op 3661) 


s_add_i32 s0, s1, s19 // (op 3661) 


s_ashr_i32 s1, s0, s18 // (op 3661) 


s_lshl_b32 s0, s12, 24 // (op 3661) 
s_ashr_i32 s18, s0, 31 // (op 3661) 

s_xor_b32 s0, s1, s18 // (op 3661) 

s_sub_i32 s1, s0, s18 // (op 3661) 


s_mul_hi_i32 s0, s1, s13 // (op 3661) 
s_add_i32 s18, s0, s1 // (op 3661) 

s_ashr_i32 s0, s18, 31 // (op 3661) 
s_cmp_eq_i32 s13, 0 // (op 3661) 
s_mov_b32 s19, scc // (op 3661) 
s_cmp_eq_i32 s19, 0 // (op 3661) 
s_cselect_b32 s49, 0, -1 // (op 3661) 

s_and_b32 s19, s14, 31 // (op 3661) 
s_lshl_b32 s50, 1, s19 // (op 3661) 
s_add_i32 s51, s50, s49 // (op 3661) 


s_and_b32 s49, s0, s51 // (op 3661) 


s_add_i32 s0, s18, s49 // (op 3661) 


s_ashr_i32 s18, s0, s19 // (op 3661) 


s_lshl_b32 s0, s14, 24 // (op 3661) 
s_ashr_i32 s19, s0, 31 // (op 3661) 

s_xor_b32 s0, s18, s19 // (op 3661) 

s_sub_i32 s18, s0, s19 // (op 3661) 


s_add_u32 s50, s6, 256 // (op 3661) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3661) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3661) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3661) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3661) 

s_mul_i32 s0, s18, s50 // (op 3661) 

s_sub_i32 s49, s1, s0 // (op 3661) 


s_mul_hi_i32 s0, s16, s10 // (op 3661) 
s_add_i32 s1, s0, s16 // (op 3661) 

s_ashr_i32 s0, s1, 31 // (op 3661) 
s_cmp_eq_i32 s10, 0 // (op 3661) 
s_mov_b32 s18, scc // (op 3661) 
s_cmp_eq_i32 s18, 0 // (op 3661) 
s_cselect_b32 s19, 0, -1 // (op 3661) 

s_and_b32 s18, s47, 31 // (op 3661) 
s_lshl_b32 s72, 1, s18 // (op 3661) 
s_add_i32 s73, s72, s19 // (op 3661) 


s_and_b32 s19, s0, s73 // (op 3661) 


s_add_i32 s0, s1, s19 // (op 3661) 


s_ashr_i32 s1, s0, s18 // (op 3661) 


s_lshl_b32 s0, s47, 24 // (op 3661) 
s_ashr_i32 s18, s0, 31 // (op 3661) 

s_xor_b32 s0, s1, s18 // (op 3661) 

s_sub_i32 s1, s0, s18 // (op 3661) 


s_mul_i32 s0, s44, s50 // (op 3661) 

s_mul_i32 s18, s48, s0 // (op 3661) 

s_mul_i32 s0, s1, s18 // (op 3661) 

s_sub_u32 s18, s16, s0 // (op 3661) 

s_mul_hi_i32 s0, s18, s15 // (op 3661) 
s_add_i32 s19, s0, s18 // (op 3661) 


s_ashr_i32 s0, s19, 31 // (op 3661) 
s_cmp_eq_i32 s15, 0 // (op 3661) 
s_mov_b32 s18, scc // (op 3661) 
s_cmp_eq_i32 s18, 0 // (op 3661) 
s_cselect_b32 s50, 0, -1 // (op 3661) 

s_and_b32 s18, s46, 31 // (op 3661) 
s_lshl_b32 s51, 1, s18 // (op 3661) 
s_add_i32 s72, s51, s50 // (op 3661) 


s_and_b32 s50, s0, s72 // (op 3661) 


s_add_i32 s0, s19, s50 // (op 3661) 


s_ashr_i32 s19, s0, s18 // (op 3661) 


s_lshl_b32 s0, s46, 24 // (op 3661) 
s_ashr_i32 s18, s0, 31 // (op 3661) 

s_xor_b32 s0, s19, s18 // (op 3661) 

s_sub_i32 s19, s0, s18 // (op 3661) 


s_cmp_eq_i32 s1, 0 // (op 3661) 
s_mov_b32 s0, scc // (op 3661) 

s_cmp_eq_i32 s0, 0 // (op 3661) 
s_cselect_b32 s1, s19, s49 // (op 3661) 



v_lshl_add_u32 v23, s1, 8, v13 // (op 3661) 


v_mul_lo_u32 v13, v22, s38 // (op 3661) 
v_add_u32 v13, v13, v23 // (op 3661) 


s_mul_i32 s1, s38, s43 // (op 3663) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3663) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3663) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3663) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3663) most significant: sum
s_add_u32 s1, s1, s19 // (op 3663) most significant: sum


s_mov_b32 s68, s40 // (op 3663) reg expression
s_mov_b32 s69, s41 // (op 3663) reg expression
s_mov_b32 s70, s0 // (op 3663) reg expression
s_mov_b32 s71, 131072 // (op 3663) call()




buffer_load_ubyte v22, v20, s[64:67], 0 offen offset: 0 // (op 3108) Load value

s_mul_i32 s1, s29, 6 // (op 3112) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3112) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3112) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3112) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3112) 


v_mov_b32 v23, v14 // (op 3112) reg expression
buffer_load_dwordx3 v[26:28], v23, s[52:55], 0 offen // (op 3112) Load value
buffer_load_dwordx3 v[30:32], v23, s[52:55], 0 offen offset:12 // (op 3112) Load value
buffer_load_dwordx3 v[34:36], v23, s[52:55], 0 offen offset:24 // (op 3112) Load value
buffer_load_dwordx3 v[38:40], v23, s[52:55], 0 offen offset:36 // (op 3112) Load value
v_add_u32 v23, v23, v24 // (op 3112) 
buffer_load_dwordx3 v[42:44], v23, s[52:55], 0 offen // (op 3112) Load value
buffer_load_dwordx3 v[46:48], v23, s[52:55], 0 offen offset:12 // (op 3112) Load value
buffer_load_dwordx3 v[50:52], v23, s[52:55], 0 offen offset:24 // (op 3112) Load value
buffer_load_dwordx3 v[54:56], v23, s[52:55], 0 offen offset:36 // (op 3112) Load value




v_mov_b32 v29, v12 // (op 3115) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 3115) Load value

v_and_b32 v29, -8, v1 // (op 3727) 
v_sub_u32 v37, v1, v29 // (op 3727) 

v_lshlrev_b32 v29, 4, v37 // (op 3727) 

v_lshrrev_b32 v37, 3, v1 // (op 3727) 
v_lshl_add_u32 v41, v37, 7, v29 // (op 3727) 


s_nop 0
v_readfirstlane_b32 s0, v41 // (op 3727) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3399) reg expression
buffer_load_dwordx4 v16, s[20:23], 0 offen offset: 0 lds // (op 3399) Load value direct to lds




buffer_load_ubyte v29, v20, s[64:67], 0 offen offset:4 // (op 3122) Load value


s_mul_i32 s1, s29, 6 // (op 3126) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3126) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3126) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3126) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3126) 


s_lshl_b64 s[0:1], s[28:29], 7 // (op 3126) 
s_mul_i32 s19, s1, 6 // (op 3126) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 3126) least significant
s_mul_hi_u32 s49, s0, 6 // (op 3126) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3126) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 3126) 

v_add_u32 v20, v14, s0 // (op 3126) 

buffer_load_dwordx3 v[58:60], v20, s[52:55], 0 offen // (op 3126) Load value
buffer_load_dwordx3 v[62:64], v20, s[52:55], 0 offen offset:12 // (op 3126) Load value
buffer_load_dwordx3 v[66:68], v20, s[52:55], 0 offen offset:24 // (op 3126) Load value
buffer_load_dwordx3 v[70:72], v20, s[52:55], 0 offen offset:36 // (op 3126) Load value
v_add_u32 v20, v20, v24 // (op 3126) 
buffer_load_dwordx3 v[74:76], v20, s[52:55], 0 offen // (op 3126) Load value
buffer_load_dwordx3 v[78:80], v20, s[52:55], 0 offen offset:12 // (op 3126) Load value
buffer_load_dwordx3 v[82:84], v20, s[52:55], 0 offen offset:24 // (op 3126) Load value
buffer_load_dwordx3 v[86:88], v20, s[52:55], 0 offen offset:36 // (op 3126) Load value





s_lshl_b32 s0, s38, 2 // (op 3129) 
v_add_u32 v14, v12, s0 // (op 3129) 


buffer_load_dword v20, v14, s[68:71], 0 offen // (op 3129) Load value


v_and_b32 v12, -8, v1 // (op 3900) 
v_sub_u32 v14, v1, v12 // (op 3900) 

v_lshlrev_b32 v12, 4, v14 // (op 3900) 

v_lshrrev_b32 v14, 3, v1 // (op 3900) 
v_lshl_add_u32 v37, v14, 7, v12 // (op 3900) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 3900) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3404) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3404) reg expression
s_mov_b32 s18, 128 // (op 3404) call()
buffer_load_dwordx4 v16, s[20:23], s18 offen offset: 0 lds // (op 3404) Load value direct to lds




v_and_b32 v12, -4, v1 // (op 3780) 
v_sub_u32 v14, v1, v12 // (op 3780) 

v_lshlrev_b32 v12, 6, v14 // (op 3780) 

v_lshrrev_b32 v14, 2, v1 // (op 3780) 
v_lshl_add_u32 v37, v14, 9, v12 // (op 3780) 


v_mul_lo_u32 v12, v37, 6 // (op 3780) 
v_lshrrev_b32 v14, 3, v12 // (op 3780) 

v_lshrrev_b32 v12, 4, v37 // (op 3780) 

v_lshlrev_b32 v37, 2, v12 // (op 3780) 

v_add_u32 v12, v14, v37 // (op 3780) 





s_waitcnt vmcnt(20)
ds_write_b96 v12, v[26:28] offset:8192 // (op 3134) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v12, v[30:32] offset:8208 // (op 3134) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[34:36] offset:8224 // (op 3134) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[38:40] offset:8240 // (op 3134) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[42:44] offset:8448 // (op 3134) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[46:48] offset:8464 // (op 3134) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[50:52] offset:8480 // (op 3134) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[54:56] offset:8496 // (op 3134) Store local data 

v_and_b32 v12, -64, v1 // (op 3838) 
v_sub_u32 v14, v1, v12 // (op 3838) 

v_lshlrev_b32 v12, 2, v14 // (op 3838) 

v_lshrrev_b32 v14, 6, v1 // (op 3838) 
v_lshl_add_u32 v37, v14, 8, v12 // (op 3838) 





s_waitcnt vmcnt(12)
ds_write_b32 v37, v33 offset:40960 // (op 3135) Store local data 

v_and_b32 v12, -8, v1 // (op 3699) 
v_sub_u32 v14, v1, v12 // (op 3699) 

v_lshlrev_b32 v12, 4, v14 // (op 3699) 

v_lshrrev_b32 v14, 3, v1 // (op 3699) 
v_lshl_add_u32 v37, v14, 7, v12 // (op 3699) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 3699) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 3751) 
v_sub_u32 v14, v1, v12 // (op 3751) 

v_lshlrev_b32 v12, 6, v14 // (op 3751) 

v_lshrrev_b32 v14, 2, v1 // (op 3751) 
v_lshl_add_u32 v37, v14, 9, v12 // (op 3751) 


v_mul_lo_u32 v12, v37, 6 // (op 3751) 
v_lshrrev_b32 v14, 3, v12 // (op 3751) 

v_lshrrev_b32 v12, 4, v37 // (op 3751) 

v_lshlrev_b32 v37, 2, v12 // (op 3751) 

v_add_u32 v12, v14, v37 // (op 3751) 


v_and_b32 v14, -64, v1 // (op 3809) 
v_sub_u32 v37, v1, v14 // (op 3809) 

v_lshlrev_b32 v14, 2, v37 // (op 3809) 

v_lshrrev_b32 v37, 6, v1 // (op 3809) 
v_lshl_add_u32 v41, v37, 8, v14 // (op 3809) 


v_and_b32 v14, -8, v1 // (op 3885) 
v_sub_u32 v37, v1, v14 // (op 3885) 

v_lshlrev_b32 v14, 4, v37 // (op 3885) 

v_lshrrev_b32 v37, 3, v1 // (op 3885) 
v_lshl_add_u32 v45, v37, 7, v14 // (op 3885) 


s_nop 0
v_readfirstlane_b32 s1, v45 // (op 3885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 3924) 
v_sub_u32 v37, v1, v14 // (op 3924) 

v_lshlrev_b32 v14, 6, v37 // (op 3924) 

v_lshrrev_b32 v37, 2, v1 // (op 3924) 
v_lshl_add_u32 v45, v37, 9, v14 // (op 3924) 


v_mul_lo_u32 v14, v45, 6 // (op 3924) 
v_lshrrev_b32 v37, 3, v14 // (op 3924) 

v_lshrrev_b32 v14, 4, v45 // (op 3924) 

v_lshlrev_b32 v45, 2, v14 // (op 3924) 

v_add_u32 v14, v37, v45 // (op 3924) 


v_and_b32 v37, -64, v1 // (op 3953) 
v_sub_u32 v45, v1, v37 // (op 3953) 

v_lshlrev_b32 v37, 2, v45 // (op 3953) 

v_lshrrev_b32 v45, 6, v1 // (op 3953) 
v_lshl_add_u32 v49, v45, 8, v37 // (op 3953) 


s_mov_b32 s18, 0 // (op 1244) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1243) 
s_mov_b32 s19, s50 // (op 1243) convert

s_and_b32 s49, s19, 4294967294 // (op 1243) 

s_cmp_lt_u32 s18, s49 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243:
s_add_u32 s18, s18, 2 // (op 1245) 
s_waitcnt vmcnt(11)
s_barrier  // (op 4029) 



ds_read_b128 v[90:93], v3 // (op 65) Load local data 
ds_read_b128 v[94:97], v3 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4042) 



ds_read_b96_tr_b6 v[98:100], v0 offset:8192 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v0 offset:12288 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4056) 


ds_read_u8 v37, v7 offset:40960 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[98:103], a[0:3], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 82) 
ds_read_b96_tr_b6 v[104:106], v0 offset:8224 // (op 339) Transpose load local data 


ds_read_b96_tr_b6 v[107:109], v0 offset:12320 // (op 339) Transpose load local data 


ds_read_u8 v37, v7 offset:40992 // (op 342) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[104:109], a[4:7], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 343) 
ds_read_b96_tr_b6 v[98:100], v0 offset:8256 // (op 495) Transpose load local data 


ds_read_b96_tr_b6 v[101:103], v0 offset:12352 // (op 495) Transpose load local data 


ds_read_u8 v37, v7 offset:41024 // (op 498) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[98:103], a[8:11], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 499) 
ds_read_b96_tr_b6 v[104:106], v0 offset:8288 // (op 651) Transpose load local data 


ds_read_b96_tr_b6 v[107:109], v0 offset:12384 // (op 651) Transpose load local data 


ds_read_u8 v37, v7 offset:41056 // (op 654) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:97], v[104:109], a[12:15], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 655) 
ds_read_b96_tr_b6 v[98:100], v0 offset:8320 // (op 807) Transpose load local data 


ds_read_b96_tr_b6 v[101:103], v0 offset:12416 // (op 807) Transpose load local data 


ds_read_u8 v37, v7 offset:41088 // (op 810) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:97], v[98:103], a[16:19], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 811) 
ds_read_b96_tr_b6 v[104:106], v0 offset:8352 // (op 963) Transpose load local data 


ds_read_b96_tr_b6 v[107:109], v0 offset:12448 // (op 963) Transpose load local data 


ds_read_u8 v37, v7 offset:41120 // (op 966) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:97], v[104:109], a[20:23], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 967) 
ds_read_b96_tr_b6 v[98:100], v0 offset:8384 // (op 1119) Transpose load local data 


ds_read_b96_tr_b6 v[101:103], v0 offset:12480 // (op 1119) Transpose load local data 


ds_read_u8 v37, v7 offset:41152 // (op 1122) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:97], v[98:103], a[24:27], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1123) 
ds_read_b96_tr_b6 v[104:106], v0 offset:8416 // (op 1275) Transpose load local data 


ds_read_b96_tr_b6 v[107:109], v0 offset:12512 // (op 1275) Transpose load local data 


ds_read_u8 v37, v7 offset:41184 // (op 1278) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:97], v[104:109], a[28:31], v22, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1279) 





v_mov_b32 v37, v21 // (op 75) 

buffer_load_ubyte v22, v37, s[64:67], 0 offen offset:8 // (op 75) Load value

s_mul_i32 s51, s29, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 10) least significant
s_mul_hi_u32 s19, s28, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[50:51] // (op 10) 


v_mov_b32 v45, v15 // (op 10) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v53, v45, s19 // (op 10) 

buffer_load_dwordx3 v[26:28], v53, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v53, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v53, s[52:55], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[38:40], v53, s[52:55], 0 offen offset:36 // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
buffer_load_dwordx3 v[42:44], v53, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[46:48], v53, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[50:52], v53, s[52:55], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[54:56], v53, s[52:55], 0 offen offset:36 // (op 10) Load value




v_mov_b32 v53, v13 // (op 21) 
s_lshl_b32 s19, s38, 3 // (op 21) 
v_add_u32 v57, v53, s19 // (op 21) 


buffer_load_dword v33, v57, s[68:71], 0 offen // (op 21) Load value

s_barrier  // (op 4032) 


v_mov_b64 v[16:17], v[18:19] // (op 3383) 

s_mov_b32 m0, s0 // (op 3383) reg expression
s_mov_b32 s19, 256 // (op 3383) call()
buffer_load_dwordx4 v16, s[20:23], s19 offen offset: 0 lds // (op 3383) Load value direct to lds

s_barrier  // (op 4079) 



s_waitcnt vmcnt(20)
ds_write_b96 v14, v[58:60] offset:41984 // (op 235) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v14, v[62:64] offset:42000 // (op 235) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v14, v[66:68] offset:42016 // (op 235) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v14, v[70:72] offset:42032 // (op 235) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[74:76] offset:42240 // (op 235) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v14, v[78:80] offset:42256 // (op 235) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[82:84] offset:42272 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v14, v[86:88] offset:42288 // (op 235) Store local data 
s_barrier  // (op 4085) 



v_add_u32 v57, 74752, v49 // (op 236) 
s_waitcnt vmcnt(12)
ds_write_b32 v57, v20 // (op 236) Store local data 

s_waitcnt vmcnt(11)
s_barrier  // (op 4070) 



ds_read_b128 v[90:93], v4 offset:4096 // (op 227) Load local data 
ds_read_b128 v[94:97], v4 offset:4160 // (op 227) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4076) 



ds_read_b96_tr_b6 v[98:100], v10 offset:41984 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v10 offset:46080 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4082) 


v_add_u32 v57, 74752, v11 // (op 233) 
ds_read_u8 v61, v57 // (op 233) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42016 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[98:103], a[0:3], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 234) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46112 // (op 348) Transpose load local data 




v_add_u32 v57, 74784, v11 // (op 351) 
ds_read_u8 v61, v57 // (op 351) Load local data 




ds_read_b96_tr_b6 v[98:100], v10 offset:42048 // (op 504) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[104:109], a[4:7], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 344) 
ds_read_b96_tr_b6 v[101:103], v10 offset:46144 // (op 504) Transpose load local data 




v_add_u32 v57, 74816, v11 // (op 507) 
ds_read_u8 v61, v57 // (op 507) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42080 // (op 660) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[98:103], a[8:11], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 500) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46176 // (op 660) Transpose load local data 




v_add_u32 v57, 74848, v11 // (op 663) 
ds_read_u8 v61, v57 // (op 663) Load local data 




ds_read_b96_tr_b6 v[98:100], v10 offset:42112 // (op 816) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:97], v[104:109], a[12:15], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 656) 
ds_read_b96_tr_b6 v[101:103], v10 offset:46208 // (op 816) Transpose load local data 




v_add_u32 v57, 74880, v11 // (op 819) 
ds_read_u8 v61, v57 // (op 819) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42144 // (op 972) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:97], v[98:103], a[16:19], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 812) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46240 // (op 972) Transpose load local data 




v_add_u32 v57, 74912, v11 // (op 975) 
ds_read_u8 v61, v57 // (op 975) Load local data 




ds_read_b96_tr_b6 v[98:100], v10 offset:42176 // (op 1128) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:97], v[104:109], a[20:23], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 968) 
ds_read_b96_tr_b6 v[101:103], v10 offset:46272 // (op 1128) Transpose load local data 




v_add_u32 v57, 74944, v11 // (op 1131) 
ds_read_u8 v61, v57 // (op 1131) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42208 // (op 1284) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:97], v[98:103], a[24:27], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1124) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46304 // (op 1284) Transpose load local data 




s_barrier  // (op 4182) 
v_add_u32 v57, 74976, v11 // (op 1287) 
ds_read_u8 v61, v57 // (op 1287) Load local data 

s_barrier  // (op 4186) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:97], v[104:109], a[28:31], v29, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1280) 






buffer_load_ubyte v29, v37, s[64:67], 0 offen offset:12 // (op 231) Load value


v_add_u32 v21, 8, v21 // (op 3589) 
s_mul_i32 s51, s29, 6 // (op 237) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 237) least significant
s_mul_hi_u32 s19, s28, 6 // (op 237) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 237) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 237) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 237) 
s_mul_i32 s73, s51, 6 // (op 237) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 237) least significant
s_mul_hi_u32 s19, s50, 6 // (op 237) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 237) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 237) 

s_mul_i32 s19, 3, s50 // (op 237) 

v_add_u32 v23, v45, s19 // (op 237) 

buffer_load_dwordx3 v[58:60], v23, s[52:55], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[62:64], v23, s[52:55], 0 offen offset:12 // (op 237) Load value
buffer_load_dwordx3 v[66:68], v23, s[52:55], 0 offen offset:24 // (op 237) Load value
buffer_load_dwordx3 v[70:72], v23, s[52:55], 0 offen offset:36 // (op 237) Load value
v_add_u32 v23, v23, v24 // (op 237) 
buffer_load_dwordx3 v[74:76], v23, s[52:55], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[78:80], v23, s[52:55], 0 offen offset:12 // (op 237) Load value
buffer_load_dwordx3 v[82:84], v23, s[52:55], 0 offen offset:24 // (op 237) Load value
buffer_load_dwordx3 v[86:88], v23, s[52:55], 0 offen offset:36 // (op 237) Load value





s_lshl_b32 s19, s38, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v37, v53, s49 // (op 238) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 3525) 
s_mul_i32 s73, s51, 6 // (op 3525) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 3525) least significant
s_mul_hi_u32 s19, s50, 6 // (op 3525) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 3525) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 3525) 

s_lshl_b32 s19, s50, 1 // (op 3525) 

v_add_u32 v15, v15, s19 // (op 3525) 

buffer_load_dword v20, v37, s[68:71], 0 offen // (op 238) Load value


s_barrier  // (op 4073) 
s_lshl_b32 s19, s38, 3 // (op 3653) 


v_add_u32 v13, v13, s19 // (op 3653) 

s_add_u32 s19, 4096, s1 // (op 3394) 

s_mov_b32 m0, s19 // (op 3394) reg expression
s_mov_b32 s49, 384 // (op 3394) call()
buffer_load_dwordx4 v16, s[20:23], s49 offen offset: 0 lds // (op 3394) Load value direct to lds



s_barrier  // (op 4045) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3461) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3461) most significant half
s_waitcnt vmcnt(20)
ds_write_b96 v12, v[26:28] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v12, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[38:40] offset:8240 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[42:44] offset:8448 // (op 13) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[46:48] offset:8464 // (op 13) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[50:52] offset:8480 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[54:56] offset:8496 // (op 13) Store local data 
s_barrier  // (op 4059) 



s_waitcnt vmcnt(12)
ds_write_b32 v41, v33 offset:40960 // (op 26) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1243) 
s_mov_b32 s19, s50 // (op 1243) convert

s_and_b32 s49, s19, 4294967294 // (op 1243) 

s_cmp_lt_u32 s18, s49 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243:














s_barrier  // (op 4166) 
s_mul_hi_i32 s0, s16, s11 // (op 3417) 
v_lshrrev_b32 v61, 3, v1 // (op 3417) 
s_add_i32 s1, s0, s16 // (op 3417) 

v_and_b32 v65, -8, v1 // (op 3417) 
s_ashr_i32 s0, s1, 31 // (op 3417) 
v_sub_u32 v69, v1, v65 // (op 3417) 

s_cmp_eq_i32 s11, 0 // (op 3417) 
s_mov_b32 s18, scc // (op 3417) 
v_lshlrev_b32 v65, 4, v69 // (op 3417) 

s_cmp_eq_i32 s18, 0 // (op 3417) 
s_cselect_b32 s19, 0, -1 // (op 3417) 

s_and_b32 s18, s12, 31 // (op 3417) 
s_lshl_b32 s49, 1, s18 // (op 3417) 
s_add_i32 s50, s49, s19 // (op 3417) 


s_and_b32 s19, s0, s50 // (op 3417) 


s_add_i32 s0, s1, s19 // (op 3417) 


s_ashr_i32 s1, s0, s18 // (op 3417) 


s_lshl_b32 s0, s12, 24 // (op 3417) 
s_ashr_i32 s18, s0, 31 // (op 3417) 

s_xor_b32 s0, s1, s18 // (op 3417) 

s_sub_i32 s1, s0, s18 // (op 3417) 


s_mul_hi_i32 s0, s1, s13 // (op 3417) 
s_add_i32 s18, s0, s1 // (op 3417) 

s_ashr_i32 s0, s18, 31 // (op 3417) 
s_cmp_eq_i32 s13, 0 // (op 3417) 
s_mov_b32 s19, scc // (op 3417) 
s_cmp_eq_i32 s19, 0 // (op 3417) 
s_cselect_b32 s49, 0, -1 // (op 3417) 

s_and_b32 s19, s14, 31 // (op 3417) 
s_lshl_b32 s50, 1, s19 // (op 3417) 
s_add_i32 s51, s50, s49 // (op 3417) 


s_and_b32 s49, s0, s51 // (op 3417) 


s_add_i32 s0, s18, s49 // (op 3417) 


s_ashr_i32 s18, s0, s19 // (op 3417) 


s_lshl_b32 s0, s14, 24 // (op 3417) 
s_ashr_i32 s19, s0, 31 // (op 3417) 

s_xor_b32 s0, s18, s19 // (op 3417) 

s_sub_i32 s18, s0, s19 // (op 3417) 


s_mul_i32 s0, s1, s44 // (op 3417) 

s_sub_u32 s1, s16, s0 // (op 3417) 

s_mul_i32 s0, s18, s44 // (op 3417) 
s_add_u32 s0, s0, s1 // (op 3417) 


s_mul_hi_i32 s1, s16, s10 // (op 3417) 
s_add_i32 s18, s1, s16 // (op 3417) 

s_ashr_i32 s1, s18, 31 // (op 3417) 
s_cmp_eq_i32 s10, 0 // (op 3417) 
s_mov_b32 s19, scc // (op 3417) 
s_cmp_eq_i32 s19, 0 // (op 3417) 
s_cselect_b32 s49, 0, -1 // (op 3417) 

s_and_b32 s19, s47, 31 // (op 3417) 
s_lshl_b32 s50, 1, s19 // (op 3417) 
s_add_i32 s51, s50, s49 // (op 3417) 


s_and_b32 s49, s1, s51 // (op 3417) 


s_add_i32 s1, s18, s49 // (op 3417) 


s_ashr_i32 s18, s1, s19 // (op 3417) 


s_lshl_b32 s1, s47, 24 // (op 3417) 
s_ashr_i32 s19, s1, 31 // (op 3417) 

s_xor_b32 s1, s18, s19 // (op 3417) 

s_sub_i32 s18, s1, s19 // (op 3417) 


s_add_u32 s50, s6, 256 // (op 3417) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3417) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3417) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3417) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3417) 

s_mul_i32 s1, s44, s50 // (op 3417) 

s_mul_i32 s19, s48, s1 // (op 3417) 

s_mul_i32 s1, s18, s19 // (op 3417) 

s_sub_u32 s19, s16, s1 // (op 3417) 

s_mov_b32 s1, s19 // (op 3417) 
s_mul_hi_i32 s49, s1, s15 // (op 3417) 
s_add_i32 s50, s49, s1 // (op 3417) 


s_ashr_i32 s1, s50, 31 // (op 3417) 
s_cmp_eq_i32 s15, 0 // (op 3417) 
s_mov_b32 s49, scc // (op 3417) 
s_cmp_eq_i32 s49, 0 // (op 3417) 
s_cselect_b32 s51, 0, -1 // (op 3417) 

s_and_b32 s49, s46, 31 // (op 3417) 
s_lshl_b32 s72, 1, s49 // (op 3417) 
s_add_i32 s73, s72, s51 // (op 3417) 


s_and_b32 s51, s1, s73 // (op 3417) 


s_add_i32 s1, s50, s51 // (op 3417) 


s_ashr_i32 s50, s1, s49 // (op 3417) 


s_lshl_b32 s1, s46, 24 // (op 3417) 
s_ashr_i32 s49, s1, 31 // (op 3417) 

s_xor_b32 s1, s50, s49 // (op 3417) 

s_sub_i32 s50, s1, s49 // (op 3417) 


s_mul_i32 s1, s45, s44 // (op 3417) 
s_add_u32 s72, s8, 32 // (op 3417) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3417) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3417) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3417) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 3417) 

s_sub_i32 s49, s72, s1 // (op 3417) 

s_mul_i32 s51, s50, s49 // (op 3417) 


s_sub_u32 s49, s19, s51 // (op 3417) 


s_add_u32 s19, s49, s1 // (op 3417) 


s_cmp_eq_i32 s18, 0 // (op 3417) 
s_mov_b32 s1, scc // (op 3417) 

s_cmp_eq_i32 s1, 0 // (op 3417) 
s_cselect_b32 s18, s19, s0 // (op 3417) 



v_lshl_add_u32 v69, s18, 5, v61 // (op 3417) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3417) 
s_lshr_b32 s18, s0, 1 // (op 3417) 

v_lshl_add_u32 v61, s18, 8, v65 // (op 3417) 


v_mov_b32 v13, 0 // (op 3417) convert
v_mov_b32 v12, v61 // (op 3417) convert

v_mul_lo_u32 v15, v69, s25 // (op 3417) most significant: low of low * high
v_mul_hi_u32 v61, v69, s24 // (op 3417) most significant: high of low * low
v_mul_lo_u32 v14, v69, s24 // (op 3417) least significant: low of low * low
v_add_u32 v15, v15, v61 // (op 3417) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 3417) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 3417) most significant half



s_mul_i32 s1, s24, s9 // (op 3419) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3419) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3419) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3419) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3419) most significant: sum
s_add_u32 s1, s1, s19 // (op 3419) most significant: sum


s_mov_b32 s20, s26 // (op 3419) reg expression
s_mov_b32 s21, s27 // (op 3419) reg expression
s_mov_b32 s22, s0 // (op 3419) reg expression
s_mov_b32 s23, 131072 // (op 3419) call()

s_mul_hi_i32 s0, s16, s11 // (op 3491) 
v_and_b32 v61, -4, v1 // (op 3491) 
s_add_i32 s1, s0, s16 // (op 3491) 

v_sub_u32 v65, v1, v61 // (op 3491) 

s_ashr_i32 s0, s1, 31 // (op 3491) 
v_lshlrev_b32 v61, 6, v65 // (op 3491) 

s_cmp_eq_i32 s11, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
v_lshrrev_b32 v65, 2, v1 // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s19, 0, -1 // (op 3491) 

v_lshlrev_b32 v69, 1, v65 // (op 3491) 

s_and_b32 s18, s12, 31 // (op 3491) 
s_lshl_b32 s49, 1, s18 // (op 3491) 
s_add_i32 s50, s49, s19 // (op 3491) 


s_and_b32 s19, s0, s50 // (op 3491) 


s_add_i32 s0, s1, s19 // (op 3491) 


s_ashr_i32 s1, s0, s18 // (op 3491) 


s_lshl_b32 s0, s12, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s1, s18 // (op 3491) 

s_sub_i32 s1, s0, s18 // (op 3491) 


s_mul_hi_i32 s0, s1, s13 // (op 3491) 
s_add_i32 s18, s0, s1 // (op 3491) 

s_ashr_i32 s0, s18, 31 // (op 3491) 
s_cmp_eq_i32 s13, 0 // (op 3491) 
s_mov_b32 s19, scc // (op 3491) 
s_cmp_eq_i32 s19, 0 // (op 3491) 
s_cselect_b32 s49, 0, -1 // (op 3491) 

s_and_b32 s19, s14, 31 // (op 3491) 
s_lshl_b32 s50, 1, s19 // (op 3491) 
s_add_i32 s51, s50, s49 // (op 3491) 


s_and_b32 s49, s0, s51 // (op 3491) 


s_add_i32 s0, s18, s49 // (op 3491) 


s_ashr_i32 s18, s0, s19 // (op 3491) 


s_lshl_b32 s0, s14, 24 // (op 3491) 
s_ashr_i32 s19, s0, 31 // (op 3491) 

s_xor_b32 s0, s18, s19 // (op 3491) 

s_sub_i32 s18, s0, s19 // (op 3491) 


s_add_u32 s50, s6, 256 // (op 3491) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3491) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3491) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3491) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3491) 

s_mul_i32 s0, s18, s50 // (op 3491) 

s_sub_i32 s49, s1, s0 // (op 3491) 


s_mul_hi_i32 s0, s16, s10 // (op 3491) 
s_add_i32 s1, s0, s16 // (op 3491) 

s_ashr_i32 s0, s1, 31 // (op 3491) 
s_cmp_eq_i32 s10, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s19, 0, -1 // (op 3491) 

s_and_b32 s18, s47, 31 // (op 3491) 
s_lshl_b32 s72, 1, s18 // (op 3491) 
s_add_i32 s73, s72, s19 // (op 3491) 


s_and_b32 s19, s0, s73 // (op 3491) 


s_add_i32 s0, s1, s19 // (op 3491) 


s_ashr_i32 s1, s0, s18 // (op 3491) 


s_lshl_b32 s0, s47, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s1, s18 // (op 3491) 

s_sub_i32 s1, s0, s18 // (op 3491) 


s_mul_i32 s0, s44, s50 // (op 3491) 

s_mul_i32 s18, s48, s0 // (op 3491) 

s_mul_i32 s0, s1, s18 // (op 3491) 

s_sub_u32 s18, s16, s0 // (op 3491) 

s_mul_hi_i32 s0, s18, s15 // (op 3491) 
s_add_i32 s19, s0, s18 // (op 3491) 


s_ashr_i32 s0, s19, 31 // (op 3491) 
s_cmp_eq_i32 s15, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s50, 0, -1 // (op 3491) 

s_and_b32 s18, s46, 31 // (op 3491) 
s_lshl_b32 s51, 1, s18 // (op 3491) 
s_add_i32 s72, s51, s50 // (op 3491) 


s_and_b32 s50, s0, s72 // (op 3491) 


s_add_i32 s0, s19, s50 // (op 3491) 


s_ashr_i32 s19, s0, s18 // (op 3491) 


s_lshl_b32 s0, s46, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s19, s18 // (op 3491) 

s_sub_i32 s19, s0, s18 // (op 3491) 


s_cmp_eq_i32 s1, 0 // (op 3491) 
s_mov_b32 s0, scc // (op 3491) 

s_cmp_eq_i32 s0, 0 // (op 3491) 
s_cselect_b32 s1, s19, s49 // (op 3491) 



v_lshl_add_u32 v65, s1, 8, v61 // (op 3491) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3491) 
v_mov_b32 v13, 0 // (op 3491) convert
v_mov_b32 v12, v65 // (op 3491) convert

s_lshr_b32 s18, s0, 1 // (op 3491) 

v_lshl_add_u32 v61, s18, 8, v69 // (op 3491) 


v_mul_lo_u32 v17, v61, s29 // (op 3491) most significant: low of low * high
v_mul_hi_u32 v65, v61, s28 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v16, v61, s28 // (op 3491) least significant: low of low * low
v_add_u32 v17, v17, v65 // (op 3491) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 3491) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 3491) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 3491) most significant: low of high * low
v_mul_hi_u32 v61, v16, 6 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 3491) least significant: low of low * low
v_add_u32 v13, v13, v61 // (op 3491) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 3491) 

v_mov_b32 v61, v16 // (op 3491) convert

s_mul_i32 s1, s33, 6 // (op 3493) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3493) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3493) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3493) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3493) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3493) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3493) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3493) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3493) most significant: sum
s_add_u32 s19, s19, s50 // (op 3493) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3493) 

s_mov_b32 s52, s30 // (op 3493) reg expression
s_mov_b32 s53, s31 // (op 3493) reg expression
s_mov_b32 s54, s0 // (op 3493) reg expression
s_mov_b32 s55, 131072 // (op 3493) call()

s_mul_hi_i32 s0, s16, s11 // (op 3555) 
v_lshrrev_b32 v65, 7, v1 // (op 3555) 
s_add_i32 s1, s0, s16 // (op 3555) 

v_and_b32 v69, 15, v1 // (op 3555) 
s_ashr_i32 s0, s1, 31 // (op 3555) 
v_lshl_add_u32 v73, v65, 4, v69 // (op 3555) 


s_cmp_eq_i32 s11, 0 // (op 3555) 
s_mov_b32 s18, scc // (op 3555) 
v_and_b32 v65, 63, v1 // (op 3555) 
s_cmp_eq_i32 s18, 0 // (op 3555) 
s_cselect_b32 s19, 0, -1 // (op 3555) 

v_lshrrev_b32 v69, 4, v65 // (op 3555) 

s_and_b32 s18, s12, 31 // (op 3555) 
s_lshl_b32 s49, 1, s18 // (op 3555) 
s_add_i32 s50, s49, s19 // (op 3555) 


s_and_b32 s19, s0, s50 // (op 3555) 


s_add_i32 s0, s1, s19 // (op 3555) 


s_ashr_i32 s1, s0, s18 // (op 3555) 


s_lshl_b32 s0, s12, 24 // (op 3555) 
s_ashr_i32 s18, s0, 31 // (op 3555) 

s_xor_b32 s0, s1, s18 // (op 3555) 

s_sub_i32 s1, s0, s18 // (op 3555) 


s_mul_hi_i32 s0, s1, s13 // (op 3555) 
s_add_i32 s18, s0, s1 // (op 3555) 

s_ashr_i32 s0, s18, 31 // (op 3555) 
s_cmp_eq_i32 s13, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s49, 0, -1 // (op 3555) 

s_and_b32 s19, s14, 31 // (op 3555) 
s_lshl_b32 s50, 1, s19 // (op 3555) 
s_add_i32 s51, s50, s49 // (op 3555) 


s_and_b32 s49, s0, s51 // (op 3555) 


s_add_i32 s0, s18, s49 // (op 3555) 


s_ashr_i32 s18, s0, s19 // (op 3555) 


s_lshl_b32 s0, s14, 24 // (op 3555) 
s_ashr_i32 s19, s0, 31 // (op 3555) 

s_xor_b32 s0, s18, s19 // (op 3555) 

s_sub_i32 s18, s0, s19 // (op 3555) 


s_mul_i32 s0, s1, s44 // (op 3555) 

s_sub_u32 s1, s16, s0 // (op 3555) 

s_mul_i32 s0, s18, s44 // (op 3555) 
s_add_u32 s0, s0, s1 // (op 3555) 


s_mul_hi_i32 s1, s16, s10 // (op 3555) 
s_add_i32 s18, s1, s16 // (op 3555) 

s_ashr_i32 s1, s18, 31 // (op 3555) 
s_cmp_eq_i32 s10, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s49, 0, -1 // (op 3555) 

s_and_b32 s19, s47, 31 // (op 3555) 
s_lshl_b32 s50, 1, s19 // (op 3555) 
s_add_i32 s51, s50, s49 // (op 3555) 


s_and_b32 s49, s1, s51 // (op 3555) 


s_add_i32 s1, s18, s49 // (op 3555) 


s_ashr_i32 s18, s1, s19 // (op 3555) 


s_lshl_b32 s1, s47, 24 // (op 3555) 
s_ashr_i32 s19, s1, 31 // (op 3555) 

s_xor_b32 s1, s18, s19 // (op 3555) 

s_sub_i32 s18, s1, s19 // (op 3555) 


s_add_u32 s50, s6, 256 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3555) 

s_mul_i32 s1, s44, s50 // (op 3555) 

s_mul_i32 s19, s48, s1 // (op 3555) 

s_mul_i32 s1, s18, s19 // (op 3555) 

s_sub_u32 s19, s16, s1 // (op 3555) 

s_mov_b32 s1, s19 // (op 3555) 
s_mul_hi_i32 s49, s1, s15 // (op 3555) 
s_add_i32 s50, s49, s1 // (op 3555) 


s_ashr_i32 s1, s50, 31 // (op 3555) 
s_cmp_eq_i32 s15, 0 // (op 3555) 
s_mov_b32 s49, scc // (op 3555) 
s_cmp_eq_i32 s49, 0 // (op 3555) 
s_cselect_b32 s51, 0, -1 // (op 3555) 

s_and_b32 s49, s46, 31 // (op 3555) 
s_lshl_b32 s72, 1, s49 // (op 3555) 
s_add_i32 s73, s72, s51 // (op 3555) 


s_and_b32 s51, s1, s73 // (op 3555) 


s_add_i32 s1, s50, s51 // (op 3555) 


s_ashr_i32 s50, s1, s49 // (op 3555) 


s_lshl_b32 s1, s46, 24 // (op 3555) 
s_ashr_i32 s49, s1, 31 // (op 3555) 

s_xor_b32 s1, s50, s49 // (op 3555) 

s_sub_i32 s50, s1, s49 // (op 3555) 


s_mul_i32 s1, s45, s44 // (op 3555) 
s_add_u32 s72, s8, 32 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 3555) 

s_sub_i32 s49, s72, s1 // (op 3555) 

s_mul_i32 s51, s50, s49 // (op 3555) 


s_sub_u32 s49, s19, s51 // (op 3555) 


s_add_u32 s19, s49, s1 // (op 3555) 


s_cmp_eq_i32 s18, 0 // (op 3555) 
s_mov_b32 s1, scc // (op 3555) 

s_cmp_eq_i32 s1, 0 // (op 3555) 
s_cselect_b32 s18, s19, s0 // (op 3555) 



v_lshl_add_u32 v65, s18, 5, v73 // (op 3555) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3555) 
s_lshr_b32 s18, s0, 1 // (op 3555) 

v_lshl_add_u32 v73, s18, 3, v69 // (op 3555) 


v_mul_lo_u32 v69, v65, s34 // (op 3555) 
v_add_u32 v69, v69, v73 // (op 3555) 


s_mul_i32 s1, s34, s9 // (op 3557) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3557) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3557) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3557) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3557) most significant: sum
s_add_u32 s1, s1, s19 // (op 3557) most significant: sum


s_mov_b32 s64, s36 // (op 3557) reg expression
s_mov_b32 s65, s37 // (op 3557) reg expression
s_mov_b32 s66, s0 // (op 3557) reg expression
s_mov_b32 s67, 131072 // (op 3557) call()

s_mul_hi_i32 s0, s16, s11 // (op 3619) 
v_and_b32 v65, -64, v1 // (op 3619) 
s_add_i32 s1, s0, s16 // (op 3619) 

v_sub_u32 v73, v1, v65 // (op 3619) 

s_ashr_i32 s0, s1, 31 // (op 3619) 
v_lshlrev_b32 v65, 2, v73 // (op 3619) 

s_cmp_eq_i32 s11, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
v_lshrrev_b32 v73, 6, v1 // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s19, 0, -1 // (op 3619) 

s_and_b32 s18, s12, 31 // (op 3619) 
s_lshl_b32 s49, 1, s18 // (op 3619) 
s_add_i32 s50, s49, s19 // (op 3619) 


s_and_b32 s19, s0, s50 // (op 3619) 


s_add_i32 s0, s1, s19 // (op 3619) 


s_ashr_i32 s1, s0, s18 // (op 3619) 


s_lshl_b32 s0, s12, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s1, s18 // (op 3619) 

s_sub_i32 s1, s0, s18 // (op 3619) 


s_mul_hi_i32 s0, s1, s13 // (op 3619) 
s_add_i32 s18, s0, s1 // (op 3619) 

s_ashr_i32 s0, s18, 31 // (op 3619) 
s_cmp_eq_i32 s13, 0 // (op 3619) 
s_mov_b32 s19, scc // (op 3619) 
s_cmp_eq_i32 s19, 0 // (op 3619) 
s_cselect_b32 s49, 0, -1 // (op 3619) 

s_and_b32 s19, s14, 31 // (op 3619) 
s_lshl_b32 s50, 1, s19 // (op 3619) 
s_add_i32 s51, s50, s49 // (op 3619) 


s_and_b32 s49, s0, s51 // (op 3619) 


s_add_i32 s0, s18, s49 // (op 3619) 


s_ashr_i32 s18, s0, s19 // (op 3619) 


s_lshl_b32 s0, s14, 24 // (op 3619) 
s_ashr_i32 s19, s0, 31 // (op 3619) 

s_xor_b32 s0, s18, s19 // (op 3619) 

s_sub_i32 s18, s0, s19 // (op 3619) 


s_add_u32 s50, s6, 256 // (op 3619) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3619) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3619) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3619) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3619) 

s_mul_i32 s0, s18, s50 // (op 3619) 

s_sub_i32 s49, s1, s0 // (op 3619) 


s_mul_hi_i32 s0, s16, s10 // (op 3619) 
s_add_i32 s1, s0, s16 // (op 3619) 

s_ashr_i32 s0, s1, 31 // (op 3619) 
s_cmp_eq_i32 s10, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s19, 0, -1 // (op 3619) 

s_and_b32 s18, s47, 31 // (op 3619) 
s_lshl_b32 s72, 1, s18 // (op 3619) 
s_add_i32 s73, s72, s19 // (op 3619) 


s_and_b32 s19, s0, s73 // (op 3619) 


s_add_i32 s0, s1, s19 // (op 3619) 


s_ashr_i32 s1, s0, s18 // (op 3619) 


s_lshl_b32 s0, s47, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s1, s18 // (op 3619) 

s_sub_i32 s1, s0, s18 // (op 3619) 


s_mul_i32 s0, s44, s50 // (op 3619) 

s_mul_i32 s18, s48, s0 // (op 3619) 

s_mul_i32 s0, s1, s18 // (op 3619) 

s_sub_u32 s18, s16, s0 // (op 3619) 

s_mul_hi_i32 s0, s18, s15 // (op 3619) 
s_add_i32 s19, s0, s18 // (op 3619) 


s_ashr_i32 s0, s19, 31 // (op 3619) 
s_cmp_eq_i32 s15, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s50, 0, -1 // (op 3619) 

s_and_b32 s18, s46, 31 // (op 3619) 
s_lshl_b32 s51, 1, s18 // (op 3619) 
s_add_i32 s72, s51, s50 // (op 3619) 


s_and_b32 s50, s0, s72 // (op 3619) 


s_add_i32 s0, s19, s50 // (op 3619) 


s_ashr_i32 s19, s0, s18 // (op 3619) 


s_lshl_b32 s0, s46, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s19, s18 // (op 3619) 

s_sub_i32 s19, s0, s18 // (op 3619) 


s_cmp_eq_i32 s1, 0 // (op 3619) 
s_mov_b32 s0, scc // (op 3619) 

s_cmp_eq_i32 s0, 0 // (op 3619) 
s_cselect_b32 s1, s19, s49 // (op 3619) 



v_lshl_add_u32 v77, s1, 8, v65 // (op 3619) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3619) 
s_lshr_b32 s18, s0, 1 // (op 3619) 

v_lshl_add_u32 v65, s18, 3, v73 // (op 3619) 


v_mul_lo_u32 v73, v65, s38 // (op 3619) 
v_add_u32 v73, v73, v77 // (op 3619) 


s_mul_i32 s1, s38, s43 // (op 3621) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3621) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3621) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3621) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3621) most significant: sum
s_add_u32 s1, s1, s19 // (op 3621) most significant: sum


s_mov_b32 s68, s40 // (op 3621) reg expression
s_mov_b32 s69, s41 // (op 3621) reg expression
s_mov_b32 s70, s0 // (op 3621) reg expression
s_mov_b32 s71, 131072 // (op 3621) call()

v_and_b32 v65, -8, v1 // (op 3712) 
v_sub_u32 v77, v1, v65 // (op 3712) 

v_lshlrev_b32 v65, 4, v77 // (op 3712) 

v_lshrrev_b32 v77, 3, v1 // (op 3712) 
v_lshl_add_u32 v81, v77, 7, v65 // (op 3712) 


s_nop 0
v_readfirstlane_b32 s0, v81 // (op 3712) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v65, -4, v1 // (op 3764) 
v_sub_u32 v77, v1, v65 // (op 3764) 

v_lshlrev_b32 v65, 6, v77 // (op 3764) 

v_lshrrev_b32 v77, 2, v1 // (op 3764) 
v_lshl_add_u32 v81, v77, 9, v65 // (op 3764) 


v_mul_lo_u32 v65, v81, 6 // (op 3764) 
v_lshrrev_b32 v77, 3, v65 // (op 3764) 

v_lshrrev_b32 v65, 4, v81 // (op 3764) 

v_lshlrev_b32 v81, 2, v65 // (op 3764) 

v_add_u32 v65, v77, v81 // (op 3764) 


v_and_b32 v77, -64, v1 // (op 3822) 
v_sub_u32 v81, v1, v77 // (op 3822) 

v_lshlrev_b32 v77, 2, v81 // (op 3822) 

v_lshrrev_b32 v81, 6, v1 // (op 3822) 
v_lshl_add_u32 v85, v81, 8, v77 // (op 3822) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 1249) 
s_and_b32 s1, s18, 4294967294 // (op 1249) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 1248) 
s_mov_b32 s49, s18 // (op 1248) convert

s_cmp_lt_u32 s1, s49 // (op 1248) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248:
s_barrier  // (op 4035) 
s_add_u32 s1, s1, 1 // (op 1250) 


v_mov_b64 v[12:13], v[14:15] // (op 3388) 

s_mov_b32 m0, s0 // (op 3388) reg expression
buffer_load_dwordx4 v12, s[20:23], 0 offen offset: 0 lds // (op 3388) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3407) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3407) most significant half

s_mul_i32 s19, s29, 6 // (op 179) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 179) least significant
s_mul_hi_u32 s49, s28, 6 // (op 179) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 179) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[18:19] // (op 179) 


v_mov_b32 v16, v61 // (op 179) 
v_mov_b32 v17, v16 // (op 179) reg expression
buffer_load_dwordx3 v[18:20], v17, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[22:24], v17, s[52:55], 0 offen offset:12 // (op 179) Load value
buffer_load_dwordx3 v[26:28], v17, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx3 v[30:32], v17, s[52:55], 0 offen offset:36 // (op 179) Load value
v_add_u32 v17, v17, v12 // (op 179) 
buffer_load_dwordx3 v[34:36], v17, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[38:40], v17, s[52:55], 0 offen offset:12 // (op 179) Load value
buffer_load_dwordx3 v[42:44], v17, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx3 v[46:48], v17, s[52:55], 0 offen offset:36 // (op 179) Load value



s_lshl_b64 s[18:19], s[28:29], 7 // (op 3483) 
s_mul_i32 s51, s19, 6 // (op 3483) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 3483) least significant
s_mul_hi_u32 s49, s18, 6 // (op 3483) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 3483) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 3483) 

v_add_u32 v61, v61, s18 // (op 3483) 



s_barrier  // (op 4048) 
v_mov_b32 v21, v73 // (op 180) 



v_mov_b32 v25, v21 // (op 180) reg expression

s_waitcnt vmcnt(7)
ds_write_b96 v65, v[18:20] offset:8192 // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v65, v[22:24] offset:8208 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v65, v[26:28] offset:8224 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v65, v[30:32] offset:8240 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v65, v[34:36] offset:8448 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v65, v[38:40] offset:8464 // (op 177) Store local data 
buffer_load_dword v29, v25, s[68:71], 0 offen // (op 180) Load value


s_lshl_b32 s18, s38, 2 // (op 3611) 
v_add_u32 v73, v73, s18 // (op 3611) 

s_waitcnt vmcnt(2)
ds_write_b96 v65, v[42:44] offset:8480 // (op 177) Store local data 
s_barrier  // (op 4038) 
s_waitcnt vmcnt(1)
ds_write_b96 v65, v[46:48] offset:8496 // (op 177) Store local data 




ds_read_b128 v[16:19], v3 // (op 169) Load local data 
s_barrier  // (op 4062) 
ds_read_b128 v[20:23], v3 offset:64 // (op 169) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4052) 
s_waitcnt vmcnt(0)
ds_write_b32 v85, v29 offset:40960 // (op 178) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 offset:8192 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:12288 // (op 171) Transpose load local data 


v_mov_b32 v12, v69 // (op 173) 

buffer_load_ubyte v13, v12, s[64:67], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v69, 4, v69 // (op 3547) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4066) 


ds_read_u8 v12, v7 offset:40960 // (op 175) Load local data 



ds_read_b96_tr_b6 v[30:32], v0 offset:8224 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:29], a[0:3], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 176) 
ds_read_b96_tr_b6 v[33:35], v0 offset:12320 // (op 320) Transpose load local data 




ds_read_u8 v12, v7 offset:40992 // (op 323) Load local data 



ds_read_b96_tr_b6 v[24:26], v0 offset:8256 // (op 476) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:23], v[30:35], a[4:7], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 324) 
ds_read_b96_tr_b6 v[27:29], v0 offset:12352 // (op 476) Transpose load local data 




ds_read_u8 v12, v7 offset:41024 // (op 479) Load local data 



ds_read_b96_tr_b6 v[30:32], v0 offset:8288 // (op 632) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:23], v[24:29], a[8:11], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 480) 
ds_read_b96_tr_b6 v[33:35], v0 offset:12384 // (op 632) Transpose load local data 




ds_read_u8 v12, v7 offset:41056 // (op 635) Load local data 



ds_read_b96_tr_b6 v[24:26], v0 offset:8320 // (op 788) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:23], v[30:35], a[12:15], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 636) 
ds_read_b96_tr_b6 v[27:29], v0 offset:12416 // (op 788) Transpose load local data 




ds_read_u8 v12, v7 offset:41088 // (op 791) Load local data 



ds_read_b96_tr_b6 v[30:32], v0 offset:8352 // (op 944) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:23], v[24:29], a[16:19], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 792) 
ds_read_b96_tr_b6 v[33:35], v0 offset:12448 // (op 944) Transpose load local data 




ds_read_u8 v12, v7 offset:41120 // (op 947) Load local data 



ds_read_b96_tr_b6 v[24:26], v0 offset:8384 // (op 1100) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:23], v[30:35], a[20:23], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 948) 
ds_read_b96_tr_b6 v[27:29], v0 offset:12480 // (op 1100) Transpose load local data 




ds_read_u8 v12, v7 offset:41152 // (op 1103) Load local data 



ds_read_b96_tr_b6 v[30:32], v0 offset:8416 // (op 1256) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:23], v[24:29], a[24:27], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1104) 
ds_read_b96_tr_b6 v[33:35], v0 offset:12512 // (op 1256) Transpose load local data 




ds_read_u8 v12, v7 offset:41184 // (op 1259) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:23], v[30:35], a[28:31], v13, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1260) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 1248) 
s_mov_b32 s49, s18 // (op 1248) convert

s_cmp_lt_u32 s1, s49 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4170) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 52) Store value


v_accvgpr_read v10, a4 // (op 1572) 
v_accvgpr_read v11, a5 // (op 1572) 
v_accvgpr_read v12, a6 // (op 1572) 
v_accvgpr_read v13, a7 // (op 1572) 
v_mul_f32 v14, v8, v10 // (op 1572) 
v_mul_f32 v15, v8, v11 // (op 1572) 
v_mul_f32 v16, v8, v12 // (op 1572) 
v_mul_f32 v17, v8, v13 // (op 1572) 



s_lshl_b32 s0, s4, 7 // (op 1574) 
v_add_u32 v0, v9, s0 // (op 1574) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1574) Store value


v_accvgpr_read v10, a8 // (op 1584) 
v_accvgpr_read v11, a9 // (op 1584) 
v_accvgpr_read v12, a10 // (op 1584) 
v_accvgpr_read v13, a11 // (op 1584) 
v_mul_f32 v14, v8, v10 // (op 1584) 
v_mul_f32 v15, v8, v11 // (op 1584) 
v_mul_f32 v16, v8, v12 // (op 1584) 
v_mul_f32 v17, v8, v13 // (op 1584) 



s_lshl_b32 s0, s4, 8 // (op 1586) 
v_add_u32 v0, v9, s0 // (op 1586) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1586) Store value


v_accvgpr_read v10, a12 // (op 1596) 
v_accvgpr_read v11, a13 // (op 1596) 
v_accvgpr_read v12, a14 // (op 1596) 
v_accvgpr_read v13, a15 // (op 1596) 
v_mul_f32 v14, v8, v10 // (op 1596) 
v_mul_f32 v15, v8, v11 // (op 1596) 
v_mul_f32 v16, v8, v12 // (op 1596) 
v_mul_f32 v17, v8, v13 // (op 1596) 



s_lshl_b32 s0, s4, 7 // (op 1598) 
s_mul_i32 s1, 3, s0 // (op 1598) 

v_add_u32 v0, v9, s1 // (op 1598) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1598) Store value


v_accvgpr_read v10, a16 // (op 1608) 
v_accvgpr_read v11, a17 // (op 1608) 
v_accvgpr_read v12, a18 // (op 1608) 
v_accvgpr_read v13, a19 // (op 1608) 
v_mul_f32 v14, v8, v10 // (op 1608) 
v_mul_f32 v15, v8, v11 // (op 1608) 
v_mul_f32 v16, v8, v12 // (op 1608) 
v_mul_f32 v17, v8, v13 // (op 1608) 



s_lshl_b32 s0, s4, 9 // (op 1610) 
v_add_u32 v0, v9, s0 // (op 1610) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1610) Store value


v_accvgpr_read v10, a20 // (op 1620) 
v_accvgpr_read v11, a21 // (op 1620) 
v_accvgpr_read v12, a22 // (op 1620) 
v_accvgpr_read v13, a23 // (op 1620) 
v_mul_f32 v14, v8, v10 // (op 1620) 
v_mul_f32 v15, v8, v11 // (op 1620) 
v_mul_f32 v16, v8, v12 // (op 1620) 
v_mul_f32 v17, v8, v13 // (op 1620) 



s_lshl_b32 s0, s4, 7 // (op 1622) 
s_mul_i32 s1, 5, s0 // (op 1622) 

v_add_u32 v0, v9, s1 // (op 1622) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1622) Store value


v_accvgpr_read v10, a24 // (op 1632) 
v_accvgpr_read v11, a25 // (op 1632) 
v_accvgpr_read v12, a26 // (op 1632) 
v_accvgpr_read v13, a27 // (op 1632) 
v_mul_f32 v14, v8, v10 // (op 1632) 
v_mul_f32 v15, v8, v11 // (op 1632) 
v_mul_f32 v16, v8, v12 // (op 1632) 
v_mul_f32 v17, v8, v13 // (op 1632) 



s_lshl_b32 s0, s4, 7 // (op 1634) 
s_mul_i32 s1, 6, s0 // (op 1634) 

v_add_u32 v0, v9, s1 // (op 1634) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1634) Store value


v_accvgpr_read v10, a28 // (op 1644) 
v_accvgpr_read v11, a29 // (op 1644) 
v_accvgpr_read v12, a30 // (op 1644) 
v_accvgpr_read v13, a31 // (op 1644) 
v_mul_f32 v14, v8, v10 // (op 1644) 
v_mul_f32 v15, v8, v11 // (op 1644) 
v_mul_f32 v16, v8, v12 // (op 1644) 
v_mul_f32 v17, v8, v13 // (op 1644) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_mul_i32 s1, 7, s0 // (op 1646) 

v_add_u32 v0, v9, s1 // (op 1646) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 135) Load value

v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v8, v14 // (op 132) 
v_mul_f32 v19, v8, v15 // (op 132) 
v_mul_f32 v20, v8, v16 // (op 132) 
v_mul_f32 v21, v8, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 136) 
v_mul_f32 v15, v5, v11 // (op 136) 
v_mul_f32 v16, v5, v12 // (op 136) 
v_mul_f32 v17, v5, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 1698) 
v_add_u32 v0, v6, s0 // (op 1698) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1698) Load value

v_accvgpr_read v14, a4 // (op 1695) 
v_accvgpr_read v15, a5 // (op 1695) 
v_accvgpr_read v16, a6 // (op 1695) 
v_accvgpr_read v17, a7 // (op 1695) 
v_mul_f32 v18, v8, v14 // (op 1695) 
v_mul_f32 v19, v8, v15 // (op 1695) 
v_mul_f32 v20, v8, v16 // (op 1695) 
v_mul_f32 v21, v8, v17 // (op 1695) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1699) 
v_mul_f32 v15, v5, v11 // (op 1699) 
v_mul_f32 v16, v5, v12 // (op 1699) 
v_mul_f32 v17, v5, v13 // (op 1699) 

v_add_f32 v10, v14, v18 // (op 1696) 
v_add_f32 v11, v15, v19 // (op 1696) 
v_add_f32 v12, v16, v20 // (op 1696) 
v_add_f32 v13, v17, v21 // (op 1696) 




s_lshl_b32 s0, s4, 7 // (op 1697) 
v_add_u32 v0, v9, s0 // (op 1697) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1697) Store value




s_lshl_b32 s0, s2, 8 // (op 1710) 
v_add_u32 v0, v6, s0 // (op 1710) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1710) Load value

v_accvgpr_read v14, a8 // (op 1707) 
v_accvgpr_read v15, a9 // (op 1707) 
v_accvgpr_read v16, a10 // (op 1707) 
v_accvgpr_read v17, a11 // (op 1707) 
v_mul_f32 v18, v8, v14 // (op 1707) 
v_mul_f32 v19, v8, v15 // (op 1707) 
v_mul_f32 v20, v8, v16 // (op 1707) 
v_mul_f32 v21, v8, v17 // (op 1707) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1711) 
v_mul_f32 v15, v5, v11 // (op 1711) 
v_mul_f32 v16, v5, v12 // (op 1711) 
v_mul_f32 v17, v5, v13 // (op 1711) 

v_add_f32 v10, v14, v18 // (op 1708) 
v_add_f32 v11, v15, v19 // (op 1708) 
v_add_f32 v12, v16, v20 // (op 1708) 
v_add_f32 v13, v17, v21 // (op 1708) 




s_lshl_b32 s0, s4, 8 // (op 1709) 
v_add_u32 v0, v9, s0 // (op 1709) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1709) Store value




s_lshl_b32 s0, s2, 7 // (op 1722) 
s_mul_i32 s1, 3, s0 // (op 1722) 

v_add_u32 v0, v6, s1 // (op 1722) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1722) Load value

v_accvgpr_read v14, a12 // (op 1719) 
v_accvgpr_read v15, a13 // (op 1719) 
v_accvgpr_read v16, a14 // (op 1719) 
v_accvgpr_read v17, a15 // (op 1719) 
v_mul_f32 v18, v8, v14 // (op 1719) 
v_mul_f32 v19, v8, v15 // (op 1719) 
v_mul_f32 v20, v8, v16 // (op 1719) 
v_mul_f32 v21, v8, v17 // (op 1719) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1723) 
v_mul_f32 v15, v5, v11 // (op 1723) 
v_mul_f32 v16, v5, v12 // (op 1723) 
v_mul_f32 v17, v5, v13 // (op 1723) 

v_add_f32 v10, v14, v18 // (op 1720) 
v_add_f32 v11, v15, v19 // (op 1720) 
v_add_f32 v12, v16, v20 // (op 1720) 
v_add_f32 v13, v17, v21 // (op 1720) 




s_lshl_b32 s0, s4, 7 // (op 1721) 
s_mul_i32 s1, 3, s0 // (op 1721) 

v_add_u32 v0, v9, s1 // (op 1721) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1721) Store value




s_lshl_b32 s0, s2, 9 // (op 1734) 
v_add_u32 v0, v6, s0 // (op 1734) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1734) Load value

v_accvgpr_read v14, a16 // (op 1731) 
v_accvgpr_read v15, a17 // (op 1731) 
v_accvgpr_read v16, a18 // (op 1731) 
v_accvgpr_read v17, a19 // (op 1731) 
v_mul_f32 v18, v8, v14 // (op 1731) 
v_mul_f32 v19, v8, v15 // (op 1731) 
v_mul_f32 v20, v8, v16 // (op 1731) 
v_mul_f32 v21, v8, v17 // (op 1731) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1735) 
v_mul_f32 v15, v5, v11 // (op 1735) 
v_mul_f32 v16, v5, v12 // (op 1735) 
v_mul_f32 v17, v5, v13 // (op 1735) 

v_add_f32 v10, v14, v18 // (op 1732) 
v_add_f32 v11, v15, v19 // (op 1732) 
v_add_f32 v12, v16, v20 // (op 1732) 
v_add_f32 v13, v17, v21 // (op 1732) 




s_lshl_b32 s0, s4, 9 // (op 1733) 
v_add_u32 v0, v9, s0 // (op 1733) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1733) Store value




s_lshl_b32 s0, s2, 7 // (op 1746) 
s_mul_i32 s1, 5, s0 // (op 1746) 

v_add_u32 v0, v6, s1 // (op 1746) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1746) Load value

v_accvgpr_read v14, a20 // (op 1743) 
v_accvgpr_read v15, a21 // (op 1743) 
v_accvgpr_read v16, a22 // (op 1743) 
v_accvgpr_read v17, a23 // (op 1743) 
v_mul_f32 v18, v8, v14 // (op 1743) 
v_mul_f32 v19, v8, v15 // (op 1743) 
v_mul_f32 v20, v8, v16 // (op 1743) 
v_mul_f32 v21, v8, v17 // (op 1743) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1747) 
v_mul_f32 v15, v5, v11 // (op 1747) 
v_mul_f32 v16, v5, v12 // (op 1747) 
v_mul_f32 v17, v5, v13 // (op 1747) 

v_add_f32 v10, v14, v18 // (op 1744) 
v_add_f32 v11, v15, v19 // (op 1744) 
v_add_f32 v12, v16, v20 // (op 1744) 
v_add_f32 v13, v17, v21 // (op 1744) 




s_lshl_b32 s0, s4, 7 // (op 1745) 
s_mul_i32 s1, 5, s0 // (op 1745) 

v_add_u32 v0, v9, s1 // (op 1745) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1745) Store value




s_lshl_b32 s0, s2, 7 // (op 1758) 
s_mul_i32 s1, 6, s0 // (op 1758) 

v_add_u32 v0, v6, s1 // (op 1758) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1758) Load value

v_accvgpr_read v14, a24 // (op 1755) 
v_accvgpr_read v15, a25 // (op 1755) 
v_accvgpr_read v16, a26 // (op 1755) 
v_accvgpr_read v17, a27 // (op 1755) 
v_mul_f32 v18, v8, v14 // (op 1755) 
v_mul_f32 v19, v8, v15 // (op 1755) 
v_mul_f32 v20, v8, v16 // (op 1755) 
v_mul_f32 v21, v8, v17 // (op 1755) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1759) 
v_mul_f32 v15, v5, v11 // (op 1759) 
v_mul_f32 v16, v5, v12 // (op 1759) 
v_mul_f32 v17, v5, v13 // (op 1759) 

v_add_f32 v10, v14, v18 // (op 1756) 
v_add_f32 v11, v15, v19 // (op 1756) 
v_add_f32 v12, v16, v20 // (op 1756) 
v_add_f32 v13, v17, v21 // (op 1756) 




s_lshl_b32 s0, s4, 7 // (op 1757) 
s_mul_i32 s1, 6, s0 // (op 1757) 

v_add_u32 v0, v9, s1 // (op 1757) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1757) Store value




s_lshl_b32 s0, s2, 7 // (op 1770) 
s_mul_i32 s1, 7, s0 // (op 1770) 

v_add_u32 v0, v6, s1 // (op 1770) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1770) Load value

v_accvgpr_read v14, a28 // (op 1767) 
v_accvgpr_read v15, a29 // (op 1767) 
v_accvgpr_read v16, a30 // (op 1767) 
v_accvgpr_read v17, a31 // (op 1767) 
v_mul_f32 v18, v8, v14 // (op 1767) 
v_mul_f32 v19, v8, v15 // (op 1767) 
v_mul_f32 v20, v8, v16 // (op 1767) 
v_mul_f32 v21, v8, v17 // (op 1767) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1771) 
v_mul_f32 v15, v5, v11 // (op 1771) 
v_mul_f32 v16, v5, v12 // (op 1771) 
v_mul_f32 v17, v5, v13 // (op 1771) 

v_add_f32 v10, v14, v18 // (op 1768) 
v_add_f32 v11, v15, v19 // (op 1768) 
v_add_f32 v12, v16, v20 // (op 1768) 
v_add_f32 v13, v17, v21 // (op 1768) 




s_lshl_b32 s0, s4, 7 // (op 1769) 
s_mul_i32 s1, 7, s0 // (op 1769) 

v_add_u32 v0, v9, s1 // (op 1769) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1769) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__end-RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 75776 // lds bytes
  .amdhsa_accum_offset 112
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
    DataTypeB: F6
    DestDataType: s
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
    macrotile: [32, 256, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 75776
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 110
    .agpr_count: 32
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

