// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_,@function
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 3824) 
s_mul_hi_i32 s1, s16, s8 // (op 4012) 
v_lshrrev_b32 v0, 6, v1 // (op 3938) 
v_and_b32 v3, 1, v0 // (op 3938) 

v_and_b32 v0, 15, v1 // (op 3938) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3938) 


v_and_b32 v0, 48, v1 // (op 3938) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3938) 


v_lshrrev_b32 v0, 6, v1 // (op 4082) 
v_and_b32 v4, 1, v0 // (op 4082) 

v_and_b32 v0, 15, v1 // (op 4082) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4082) 


v_and_b32 v0, 48, v1 // (op 4082) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4082) 


v_lshrrev_b32 v0, 7, v1 // (op 3824) 
s_add_i32 s18, s0, s16 // (op 3824) 

v_and_b32 v5, 63, v1 // (op 3824) 
s_ashr_i32 s0, s18, 31 // (op 3824) 
v_lshrrev_b32 v6, 4, v5 // (op 3824) 
s_cmp_eq_i32 s8, 0 // (op 3824) 
s_mov_b32 s19, scc // (op 3824) 
v_lshlrev_b32 v7, 2, v6 // (op 3824) 

s_cmp_eq_i32 s19, 0 // (op 3824) 
s_cselect_b32 s51, 0, -1 // (op 3824) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3824) 


s_and_b32 s19, s9, 31 // (op 3824) 
v_lshrrev_b32 v0, 2, v5 // (op 3824) 

v_and_b32 v5, 3, v0 // (op 3824) 

v_and_b32 v0, 63, v1 // (op 3886) 
v_and_b32 v7, 48, v1 // (op 3886) 
v_sub_u32 v8, v0, v7 // (op 3886) 

v_lshrrev_b32 v7, 4, v0 // (op 3886) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 3886) 


v_lshrrev_b32 v7, 7, v1 // (op 3886) 
v_lshlrev_b32 v8, 4, v7 // (op 3886) 

v_lshl_add_u32 v7, v0, 8, v8 // (op 3886) 


v_mul_lo_u32 v0, v7, 6 // (op 3886) 
v_lshrrev_b32 v8, 3, v0 // (op 3886) 

v_lshrrev_b32 v0, 4, v7 // (op 3886) 

v_lshlrev_b32 v7, 2, v0 // (op 3886) 

v_add_u32 v0, v8, v7 // (op 3886) 


v_lshrrev_b32 v7, 7, v1 // (op 3996) 
v_and_b32 v8, 15, v1 // (op 3996) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 3996) 


v_and_b32 v7, 63, v1 // (op 3996) 
v_lshrrev_b32 v8, 4, v7 // (op 3996) 

v_lshl_add_u32 v7, v8, 8, v9 // (op 3996) 


v_lshrrev_b32 v8, 7, v1 // (op 4012) 
v_and_b32 v9, 63, v1 // (op 4012) 
v_and_b32 v10, 63, v1 // (op 4043) 
v_and_b32 v11, 48, v1 // (op 4043) 
v_sub_u32 v12, v10, v11 // (op 4043) 

v_lshrrev_b32 v11, 4, v10 // (op 4043) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 4043) 


v_lshrrev_b32 v11, 7, v1 // (op 4043) 
v_lshlrev_b32 v12, 4, v11 // (op 4043) 

v_lshl_add_u32 v11, v10, 8, v12 // (op 4043) 


v_mul_lo_u32 v10, v11, 6 // (op 4043) 
v_lshrrev_b32 v12, 3, v10 // (op 4043) 

v_lshrrev_b32 v10, 4, v11 // (op 4043) 

v_lshlrev_b32 v11, 2, v10 // (op 4043) 

v_add_u32 v10, v12, v11 // (op 4043) 


v_lshrrev_b32 v11, 7, v1 // (op 4111) 
v_and_b32 v12, 15, v1 // (op 4111) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 4111) 


v_and_b32 v11, 63, v1 // (op 4111) 
v_lshrrev_b32 v12, 4, v11 // (op 4111) 

v_lshl_add_u32 v11, v12, 8, v13 // (op 4111) 


s_lshl_b32 s52, 1, s19 // (op 3824) 
v_and_b32 v12, 3, v1 // (op 3824) 
s_add_i32 s53, s52, s51 // (op 3824) 


s_and_b32 s51, s0, s53 // (op 3824) 


s_add_i32 s0, s1, s16 // (op 4012) 

s_ashr_i32 s1, s0, 31 // (op 4012) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3824) 


s_add_i32 s52, s18, s51 // (op 3824) 


v_lshrrev_b32 v5, 6, v1 // (op 3824) 
s_ashr_i32 s18, s52, s19 // (op 3824) 


v_and_b32 v12, 1, v5 // (op 3824) 

s_lshl_b32 s19, s9, 24 // (op 3824) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3824) 


s_ashr_i32 s51, s19, 31 // (op 3824) 

s_xor_b32 s19, s18, s51 // (op 3824) 

s_sub_i32 s18, s19, s51 // (op 3824) 


s_mul_hi_i32 s19, s18, s10 // (op 3824) 
s_add_i32 s51, s19, s18 // (op 3824) 

s_ashr_i32 s19, s51, 31 // (op 3824) 
s_cmp_eq_i32 s10, 0 // (op 3824) 
s_mov_b32 s52, scc // (op 3824) 
s_cmp_eq_i32 s52, 0 // (op 3824) 
s_cselect_b32 s53, 0, -1 // (op 3824) 

s_and_b32 s52, s11, 31 // (op 3824) 
v_lshrrev_b32 v12, 4, v9 // (op 4012) 
s_cmp_eq_i32 s8, 0 // (op 4012) 
s_mov_b32 s54, scc // (op 4012) 
v_lshlrev_b32 v13, 2, v12 // (op 4012) 

s_cmp_eq_i32 s54, 0 // (op 4012) 
s_cselect_b32 s55, 0, -1 // (op 4012) 

s_lshl_b32 s54, 1, s52 // (op 3824) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 4012) 


v_lshrrev_b32 v8, 2, v9 // (op 4012) 

s_add_i32 s56, s54, s53 // (op 3824) 


s_and_b32 s53, s19, s56 // (op 3824) 


s_add_i32 s19, s51, s53 // (op 3824) 


s_ashr_i32 s51, s19, s52 // (op 3824) 


s_lshl_b32 s19, s11, 24 // (op 3824) 
s_ashr_i32 s52, s19, 31 // (op 3824) 

s_xor_b32 s19, s51, s52 // (op 3824) 

s_sub_i32 s51, s19, s52 // (op 3824) 


s_mul_i32 s19, s18, s13 // (op 3824) 
s_sub_u32 s52, s16, s19 // (op 3824) 

s_mul_i32 s19, s51, s13 // (op 3824) 
s_add_u32 s19, s19, s52 // (op 3824) 

s_mul_hi_i32 s52, s16, s7 // (op 3824) 
s_add_i32 s53, s52, s16 // (op 3824) 

s_ashr_i32 s52, s53, 31 // (op 3824) 
s_cmp_eq_i32 s7, 0 // (op 3824) 
s_mov_b32 s54, scc // (op 3824) 
s_cmp_eq_i32 s54, 0 // (op 3824) 
s_cselect_b32 s56, 0, -1 // (op 3824) 

s_and_b32 s54, s14, 31 // (op 3824) 
s_lshl_b32 s57, 1, s54 // (op 3824) 
s_add_i32 s58, s57, s56 // (op 3824) 


s_and_b32 s56, s52, s58 // (op 3824) 


s_add_i32 s52, s53, s56 // (op 3824) 


s_ashr_i32 s53, s52, s54 // (op 3824) 


s_lshl_b32 s52, s14, 24 // (op 3824) 
s_ashr_i32 s54, s52, 31 // (op 3824) 

s_xor_b32 s52, s53, s54 // (op 3824) 

s_sub_i32 s53, s52, s54 // (op 3824) 


s_and_b32 s52, s9, 31 // (op 4012) 
s_lshl_b32 s54, 1, s52 // (op 4012) 
v_and_b32 v9, 3, v8 // (op 4012) 

s_add_i32 s56, s54, s55 // (op 4012) 


v_and_b32 v8, 3, v1 // (op 4012) 
s_and_b32 s54, s1, s56 // (op 4012) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 4012) 


s_add_i32 s1, s0, s54 // (op 4012) 


v_lshrrev_b32 v8, 6, v1 // (op 4012) 
s_ashr_i32 s0, s1, s52 // (op 4012) 


v_and_b32 v9, 1, v8 // (op 4012) 

s_lshl_b32 s1, s9, 24 // (op 4012) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 4012) 


s_ashr_i32 s52, s1, 31 // (op 4012) 

s_xor_b32 s1, s0, s52 // (op 4012) 

s_sub_i32 s0, s1, s52 // (op 4012) 


s_mul_hi_i32 s1, s0, s10 // (op 4012) 
s_add_i32 s52, s1, s0 // (op 4012) 

s_ashr_i32 s1, s52, 31 // (op 4012) 
s_cmp_eq_i32 s10, 0 // (op 4012) 
s_mov_b32 s54, scc // (op 4012) 
s_cmp_eq_i32 s54, 0 // (op 4012) 
s_cselect_b32 s55, 0, -1 // (op 4012) 

s_and_b32 s54, s11, 31 // (op 4012) 
s_lshl_b32 s56, 1, s54 // (op 4012) 
s_add_i32 s57, s56, s55 // (op 4012) 


s_and_b32 s55, s1, s57 // (op 4012) 


s_add_i32 s1, s52, s55 // (op 4012) 


s_ashr_i32 s52, s1, s54 // (op 4012) 


s_lshl_b32 s1, s11, 24 // (op 4012) 
s_ashr_i32 s54, s1, 31 // (op 4012) 

s_xor_b32 s1, s52, s54 // (op 4012) 

s_sub_i32 s52, s1, s54 // (op 4012) 


s_mul_i32 s1, s0, s13 // (op 4012) 
s_sub_u32 s54, s16, s1 // (op 4012) 

s_mul_i32 s1, s52, s13 // (op 4012) 
s_add_u32 s1, s1, s54 // (op 4012) 

s_mul_hi_i32 s54, s16, s7 // (op 4012) 
s_add_i32 s55, s54, s16 // (op 4012) 

s_ashr_i32 s54, s55, 31 // (op 4012) 
s_cmp_eq_i32 s7, 0 // (op 4012) 
s_mov_b32 s56, scc // (op 4012) 
s_cmp_eq_i32 s56, 0 // (op 4012) 
s_cselect_b32 s57, 0, -1 // (op 4012) 

s_and_b32 s56, s14, 31 // (op 4012) 
s_lshl_b32 s58, 1, s56 // (op 4012) 
s_add_i32 s59, s58, s57 // (op 4012) 


s_and_b32 s57, s54, s59 // (op 4012) 


s_add_i32 s54, s55, s57 // (op 4012) 


s_ashr_i32 s55, s54, s56 // (op 4012) 


s_lshl_b32 s54, s14, 24 // (op 4012) 
s_ashr_i32 s56, s54, 31 // (op 4012) 

s_xor_b32 s54, s55, s56 // (op 4012) 

s_sub_i32 s55, s54, s56 // (op 4012) 


s_add_u32 s56, s4, 32 // (op 3824) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3824) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3824) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3824) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 3824) 

s_mul_i32 s54, s13, s56 // (op 3824) 
s_mul_i32 s58, s6, s54 // (op 3824) 

s_mul_i32 s54, s53, s58 // (op 3824) 

s_sub_u32 s58, s16, s54 // (op 3824) 

s_mov_b32 s54, s58 // (op 3824) 
s_mul_hi_i32 s59, s54, s12 // (op 3824) 
s_add_i32 s60, s59, s54 // (op 3824) 


s_ashr_i32 s54, s60, 31 // (op 3824) 
s_cmp_eq_i32 s12, 0 // (op 3824) 
s_mov_b32 s59, scc // (op 3824) 
s_cmp_eq_i32 s59, 0 // (op 3824) 
s_cselect_b32 s61, 0, -1 // (op 3824) 

s_add_u32 s62, s4, 32 // (op 4012) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 4012) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4012) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4012) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 4012) 

s_mul_i32 s59, s13, s62 // (op 4012) 
s_mul_i32 s64, s6, s59 // (op 4012) 

s_mul_i32 s59, s55, s64 // (op 4012) 

s_sub_u32 s64, s16, s59 // (op 4012) 

s_mov_b32 s59, s64 // (op 4012) 
s_mul_hi_i32 s65, s59, s12 // (op 4012) 
s_add_i32 s66, s65, s59 // (op 4012) 


s_ashr_i32 s59, s66, 31 // (op 4012) 
s_cmp_eq_i32 s12, 0 // (op 4012) 
s_mov_b32 s65, scc // (op 4012) 
s_cmp_eq_i32 s65, 0 // (op 4012) 
s_cselect_b32 s67, 0, -1 // (op 4012) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 3824) 
s_lshl_b32 s68, 1, s65 // (op 3824) 
s_add_i32 s69, s68, s61 // (op 3824) 


s_and_b32 s61, s54, s69 // (op 3824) 


s_add_i32 s54, s60, s61 // (op 3824) 


s_ashr_i32 s60, s54, s65 // (op 3824) 


s_lshl_b32 s54, s48, 24 // (op 3824) 
s_ashr_i32 s61, s54, 31 // (op 3824) 

s_xor_b32 s54, s60, s61 // (op 3824) 

s_sub_i32 s60, s54, s61 // (op 3824) 


s_mul_i32 s54, s15, s13 // (op 3824) 
s_and_b32 s61, s48, 31 // (op 4012) 
s_lshl_b32 s65, 1, s61 // (op 4012) 
s_add_i32 s68, s65, s67 // (op 4012) 


s_and_b32 s65, s59, s68 // (op 4012) 


s_add_i32 s59, s66, s65 // (op 4012) 


s_ashr_i32 s65, s59, s61 // (op 4012) 


s_lshl_b32 s59, s48, 24 // (op 4012) 
s_ashr_i32 s61, s59, 31 // (op 4012) 

s_xor_b32 s59, s65, s61 // (op 4012) 

s_sub_i32 s65, s59, s61 // (op 4012) 


s_mul_i32 s59, s15, s13 // (op 4012) 
s_add_u32 s66, s20, 256 // (op 3824) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 3824) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 3824) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 3824) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 8 // (op 3824) 

s_sub_i32 s61, s66, s54 // (op 3824) 

s_mul_i32 s66, s60, s61 // (op 3824) 

s_sub_u32 s61, s58, s66 // (op 3824) 


s_add_u32 s58, s61, s54 // (op 3824) 


s_cmp_eq_i32 s53, 0 // (op 3824) 
s_mov_b32 s54, scc // (op 3824) 

s_cmp_eq_i32 s54, 0 // (op 3824) 
s_cselect_b32 s53, s58, s19 // (op 3824) 


s_mul_i32 s19, s51, s56 // (op 3824) 


v_lshl_add_u32 v9, s53, 8, v6 // (op 3824) 


s_sub_i32 s51, s18, s19 // (op 3824) 


s_cmp_eq_i32 s54, 0 // (op 3824) 
s_cselect_b32 s53, s60, s51 // (op 3824) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 3824) 


v_mul_lo_u32 v5, v6, s2 // (op 3824) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 3824) 


s_add_u32 s18, s20, 256 // (op 4012) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 4012) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 4012) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 4012) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 8 // (op 4012) 

s_sub_i32 s51, s18, s59 // (op 4012) 

s_mul_i32 s18, s65, s51 // (op 4012) 

s_sub_u32 s19, s64, s18 // (op 4012) 


s_add_u32 s18, s19, s59 // (op 4012) 


s_cmp_eq_i32 s55, 0 // (op 4012) 
s_mov_b32 s19, scc // (op 4012) 

s_cmp_eq_i32 s19, 0 // (op 4012) 
s_cselect_b32 s51, s18, s1 // (op 4012) 


s_mul_i32 s1, s52, s62 // (op 4012) 


v_lshl_add_u32 v5, s51, 8, v12 // (op 4012) 


s_sub_i32 s18, s0, s1 // (op 4012) 


s_cmp_eq_i32 s19, 0 // (op 4012) 
s_cselect_b32 s0, s65, s18 // (op 4012) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 4012) 


v_mul_lo_u32 v8, v9, s22 // (op 4012) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 4012) 


s_mul_i32 s1, s2, s5 // (op 3826) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3826) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3826) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3826) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3826) most significant: sum
s_add_u32 s1, s1, s19 // (op 3826) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3826) 

s_mul_i32 s1, s22, s5 // (op 4014) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 4014) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 4014) least significant
s_mul_hi_u32 s52, s22, s4 // (op 4014) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4014) most significant: sum
s_add_u32 s1, s1, s52 // (op 4014) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 4014) 

s_mov_b32 s56, s24 // (op 3826) reg expression
s_mov_b32 s57, s25 // (op 3826) reg expression
s_mov_b32 s58, s18 // (op 3826) reg expression
s_mov_b32 s59, 131072 // (op 3826) call()

s_mov_b32 s60, s26 // (op 4014) reg expression

s_mov_b32 s61, s27 // (op 4014) reg expression
s_mov_b32 s62, s52 // (op 4014) reg expression
s_mov_b32 s63, 131072 // (op 4014) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
v_accvgpr_write a8, 0.00000 // (op 471) call()
v_accvgpr_write a9, 0.00000 // (op 471) call()
v_accvgpr_write a10, 0.00000 // (op 471) call()
v_accvgpr_write a11, 0.00000 // (op 471) call()
v_accvgpr_write a12, 0.00000 // (op 633) call()
v_accvgpr_write a13, 0.00000 // (op 633) call()
v_accvgpr_write a14, 0.00000 // (op 633) call()
v_accvgpr_write a15, 0.00000 // (op 633) call()
v_accvgpr_write a16, 0.00000 // (op 795) call()
v_accvgpr_write a17, 0.00000 // (op 795) call()
v_accvgpr_write a18, 0.00000 // (op 795) call()
v_accvgpr_write a19, 0.00000 // (op 795) call()
v_accvgpr_write a20, 0.00000 // (op 957) call()
v_accvgpr_write a21, 0.00000 // (op 957) call()
v_accvgpr_write a22, 0.00000 // (op 957) call()
v_accvgpr_write a23, 0.00000 // (op 957) call()
v_accvgpr_write a24, 0.00000 // (op 1119) call()
v_accvgpr_write a25, 0.00000 // (op 1119) call()
v_accvgpr_write a26, 0.00000 // (op 1119) call()
v_accvgpr_write a27, 0.00000 // (op 1119) call()
v_accvgpr_write a28, 0.00000 // (op 1281) call()
v_accvgpr_write a29, 0.00000 // (op 1281) call()
v_accvgpr_write a30, 0.00000 // (op 1281) call()
v_accvgpr_write a31, 0.00000 // (op 1281) call()
s_mul_hi_i32 s0, s16, s8 // (op 3591) 
v_and_b32 v12, -4, v1 // (op 3591) 
s_add_i32 s1, s0, s16 // (op 3591) 

v_sub_u32 v13, v1, v12 // (op 3591) 

s_ashr_i32 s0, s1, 31 // (op 3591) 
v_lshlrev_b32 v12, 6, v13 // (op 3591) 

s_cmp_eq_i32 s8, 0 // (op 3591) 
s_mov_b32 s18, scc // (op 3591) 
v_lshrrev_b32 v13, 2, v1 // (op 3591) 
s_cmp_eq_i32 s18, 0 // (op 3591) 
s_cselect_b32 s19, 0, -1 // (op 3591) 

v_lshlrev_b32 v14, 1, v13 // (op 3591) 

s_and_b32 s18, s9, 31 // (op 3591) 
s_lshl_b32 s24, 1, s18 // (op 3591) 
s_add_i32 s25, s24, s19 // (op 3591) 


s_and_b32 s19, s0, s25 // (op 3591) 


s_add_i32 s0, s1, s19 // (op 3591) 


s_ashr_i32 s1, s0, s18 // (op 3591) 


s_lshl_b32 s0, s9, 24 // (op 3591) 
s_ashr_i32 s18, s0, 31 // (op 3591) 

s_xor_b32 s0, s1, s18 // (op 3591) 

s_sub_i32 s1, s0, s18 // (op 3591) 


s_mul_hi_i32 s0, s1, s10 // (op 3591) 
s_add_i32 s18, s0, s1 // (op 3591) 

s_ashr_i32 s0, s18, 31 // (op 3591) 
s_cmp_eq_i32 s10, 0 // (op 3591) 
s_mov_b32 s19, scc // (op 3591) 
s_cmp_eq_i32 s19, 0 // (op 3591) 
s_cselect_b32 s24, 0, -1 // (op 3591) 

s_and_b32 s19, s11, 31 // (op 3591) 
s_lshl_b32 s25, 1, s19 // (op 3591) 
s_add_i32 s26, s25, s24 // (op 3591) 


s_and_b32 s27, s0, s26 // (op 3591) 


s_add_i32 s0, s18, s27 // (op 3591) 


s_ashr_i32 s18, s0, s19 // (op 3591) 


s_lshl_b32 s0, s11, 24 // (op 3591) 
s_ashr_i32 s19, s0, 31 // (op 3591) 

s_xor_b32 s0, s18, s19 // (op 3591) 

s_sub_i32 s18, s0, s19 // (op 3591) 


s_mul_i32 s0, s1, s13 // (op 3591) 

s_sub_u32 s1, s16, s0 // (op 3591) 

s_mul_i32 s0, s18, s13 // (op 3591) 
s_add_u32 s0, s0, s1 // (op 3591) 


s_mul_hi_i32 s1, s16, s7 // (op 3591) 
s_add_i32 s18, s1, s16 // (op 3591) 

s_ashr_i32 s1, s18, 31 // (op 3591) 
s_cmp_eq_i32 s7, 0 // (op 3591) 
s_mov_b32 s19, scc // (op 3591) 
s_cmp_eq_i32 s19, 0 // (op 3591) 
s_cselect_b32 s24, 0, -1 // (op 3591) 

s_and_b32 s19, s14, 31 // (op 3591) 
s_lshl_b32 s25, 1, s19 // (op 3591) 
s_add_i32 s26, s25, s24 // (op 3591) 


s_and_b32 s27, s1, s26 // (op 3591) 


s_add_i32 s1, s18, s27 // (op 3591) 


s_ashr_i32 s18, s1, s19 // (op 3591) 


s_lshl_b32 s1, s14, 24 // (op 3591) 
s_ashr_i32 s19, s1, 31 // (op 3591) 

s_xor_b32 s1, s18, s19 // (op 3591) 

s_sub_i32 s18, s1, s19 // (op 3591) 


s_add_u32 s24, s4, 32 // (op 3591) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 3591) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 3591) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 3591) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 3591) 

s_mul_i32 s1, s13, s24 // (op 3591) 

s_mul_i32 s19, s6, s1 // (op 3591) 

s_mul_i32 s1, s18, s19 // (op 3591) 

s_sub_u32 s19, s16, s1 // (op 3591) 

s_mov_b32 s1, s19 // (op 3591) 
s_mul_hi_i32 s24, s1, s12 // (op 3591) 
s_add_i32 s25, s24, s1 // (op 3591) 


s_ashr_i32 s1, s25, 31 // (op 3591) 
s_cmp_eq_i32 s12, 0 // (op 3591) 
s_mov_b32 s24, scc // (op 3591) 
s_cmp_eq_i32 s24, 0 // (op 3591) 
s_cselect_b32 s26, 0, -1 // (op 3591) 

s_and_b32 s24, s48, 31 // (op 3591) 
s_lshl_b32 s27, 1, s24 // (op 3591) 
s_add_i32 s49, s27, s26 // (op 3591) 


s_and_b32 s26, s1, s49 // (op 3591) 


s_add_i32 s1, s25, s26 // (op 3591) 


s_ashr_i32 s25, s1, s24 // (op 3591) 


s_lshl_b32 s1, s48, 24 // (op 3591) 
s_ashr_i32 s24, s1, 31 // (op 3591) 

s_xor_b32 s1, s25, s24 // (op 3591) 

s_sub_i32 s25, s1, s24 // (op 3591) 


s_mul_i32 s1, s15, s13 // (op 3591) 
s_add_u32 s26, s20, 256 // (op 3591) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 3591) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 3591) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 3591) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 8 // (op 3591) 

s_sub_i32 s24, s26, s1 // (op 3591) 

s_mul_i32 s26, s25, s24 // (op 3591) 


s_sub_u32 s27, s19, s26 // (op 3591) 


s_add_u32 s19, s27, s1 // (op 3591) 


s_cmp_eq_i32 s18, 0 // (op 3591) 
s_mov_b32 s1, scc // (op 3591) 

s_cmp_eq_i32 s1, 0 // (op 3591) 
s_cselect_b32 s18, s19, s0 // (op 3591) 



v_lshl_add_u32 v13, s18, 8, v12 // (op 3591) 


v_mov_b32 v17, 0 // (op 3591) convert
v_mov_b32 v16, v13 // (op 3591) convert

v_mul_lo_u32 v13, v14, s29 // (op 3591) most significant: low of low * high
v_mul_hi_u32 v15, v14, s28 // (op 3591) most significant: high of low * low
v_mul_lo_u32 v12, v14, s28 // (op 3591) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3591) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v12 // (op 3591) least significant half
v_addc_co_u32 v15, s[0:1], v17, v13, s[0:1] // (op 3591) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 3591) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 3591) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 3591) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 3591) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3591) 

v_mov_b32 v12, v14 // (op 3591) convert

s_mul_i32 s1, s33, 6 // (op 3593) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3593) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3593) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3593) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3593) most significant: low of low * high
s_mul_i32 s24, s29, s0 // (op 3593) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3593) least significant
s_mul_hi_u32 s25, s28, s0 // (op 3593) most significant: high of low * low
s_add_u32 s19, s19, s24 // (op 3593) most significant: sum
s_add_u32 s19, s19, s25 // (op 3593) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3593) 

s_mov_b32 s24, s30 // (op 3593) reg expression
s_mov_b32 s25, s31 // (op 3593) reg expression
s_mov_b32 s26, s0 // (op 3593) reg expression
s_mov_b32 s27, 131072 // (op 3593) call()

s_mul_hi_i32 s0, s16, s8 // (op 3610) 
v_and_b32 v13, -4, v1 // (op 3610) 
s_add_i32 s1, s0, s16 // (op 3610) 

v_sub_u32 v14, v1, v13 // (op 3610) 

s_ashr_i32 s0, s1, 31 // (op 3610) 
v_lshlrev_b32 v13, 6, v14 // (op 3610) 

s_cmp_eq_i32 s8, 0 // (op 3610) 
s_mov_b32 s18, scc // (op 3610) 
v_lshrrev_b32 v14, 2, v1 // (op 3610) 
s_cmp_eq_i32 s18, 0 // (op 3610) 
s_cselect_b32 s19, 0, -1 // (op 3610) 

v_lshlrev_b32 v15, 1, v14 // (op 3610) 

s_and_b32 s18, s9, 31 // (op 3610) 
s_lshl_b32 s49, 1, s18 // (op 3610) 
s_add_i32 s50, s49, s19 // (op 3610) 


s_and_b32 s19, s0, s50 // (op 3610) 


s_add_i32 s0, s1, s19 // (op 3610) 


s_ashr_i32 s1, s0, s18 // (op 3610) 


s_lshl_b32 s0, s9, 24 // (op 3610) 
s_ashr_i32 s18, s0, 31 // (op 3610) 

s_xor_b32 s0, s1, s18 // (op 3610) 

s_sub_i32 s1, s0, s18 // (op 3610) 


s_mul_hi_i32 s0, s1, s10 // (op 3610) 
s_add_i32 s18, s0, s1 // (op 3610) 

s_ashr_i32 s0, s18, 31 // (op 3610) 
s_cmp_eq_i32 s10, 0 // (op 3610) 
s_mov_b32 s19, scc // (op 3610) 
s_cmp_eq_i32 s19, 0 // (op 3610) 
s_cselect_b32 s49, 0, -1 // (op 3610) 

s_and_b32 s19, s11, 31 // (op 3610) 
s_lshl_b32 s50, 1, s19 // (op 3610) 
s_add_i32 s51, s50, s49 // (op 3610) 


s_and_b32 s49, s0, s51 // (op 3610) 


s_add_i32 s0, s18, s49 // (op 3610) 


s_ashr_i32 s18, s0, s19 // (op 3610) 


s_lshl_b32 s0, s11, 24 // (op 3610) 
s_ashr_i32 s19, s0, 31 // (op 3610) 

s_xor_b32 s0, s18, s19 // (op 3610) 

s_sub_i32 s18, s0, s19 // (op 3610) 


s_mul_i32 s0, s1, s13 // (op 3610) 

s_sub_u32 s1, s16, s0 // (op 3610) 

s_mul_i32 s0, s18, s13 // (op 3610) 
s_add_u32 s0, s0, s1 // (op 3610) 


s_mul_hi_i32 s1, s16, s7 // (op 3610) 
s_add_i32 s18, s1, s16 // (op 3610) 

s_ashr_i32 s1, s18, 31 // (op 3610) 
s_cmp_eq_i32 s7, 0 // (op 3610) 
s_mov_b32 s19, scc // (op 3610) 
s_cmp_eq_i32 s19, 0 // (op 3610) 
s_cselect_b32 s49, 0, -1 // (op 3610) 

s_and_b32 s19, s14, 31 // (op 3610) 
s_lshl_b32 s50, 1, s19 // (op 3610) 
s_add_i32 s51, s50, s49 // (op 3610) 


s_and_b32 s49, s1, s51 // (op 3610) 


s_add_i32 s1, s18, s49 // (op 3610) 


s_ashr_i32 s18, s1, s19 // (op 3610) 


s_lshl_b32 s1, s14, 24 // (op 3610) 
s_ashr_i32 s19, s1, 31 // (op 3610) 

s_xor_b32 s1, s18, s19 // (op 3610) 

s_sub_i32 s18, s1, s19 // (op 3610) 


s_add_u32 s54, s4, 32 // (op 3610) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3610) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3610) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3610) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 3610) 

s_mul_i32 s1, s13, s50 // (op 3610) 

s_mul_i32 s19, s6, s1 // (op 3610) 

s_mul_i32 s1, s18, s19 // (op 3610) 

s_sub_u32 s19, s16, s1 // (op 3610) 

s_mov_b32 s1, s19 // (op 3610) 
s_mul_hi_i32 s49, s1, s12 // (op 3610) 
s_add_i32 s50, s49, s1 // (op 3610) 


s_ashr_i32 s1, s50, 31 // (op 3610) 
s_cmp_eq_i32 s12, 0 // (op 3610) 
s_mov_b32 s49, scc // (op 3610) 
s_cmp_eq_i32 s49, 0 // (op 3610) 
s_cselect_b32 s51, 0, -1 // (op 3610) 

s_and_b32 s49, s48, 31 // (op 3610) 
s_lshl_b32 s52, 1, s49 // (op 3610) 
s_add_i32 s53, s52, s51 // (op 3610) 


s_and_b32 s51, s1, s53 // (op 3610) 


s_add_i32 s1, s50, s51 // (op 3610) 


s_ashr_i32 s50, s1, s49 // (op 3610) 


s_lshl_b32 s1, s48, 24 // (op 3610) 
s_ashr_i32 s49, s1, 31 // (op 3610) 

s_xor_b32 s1, s50, s49 // (op 3610) 

s_sub_i32 s50, s1, s49 // (op 3610) 


s_mul_i32 s1, s15, s13 // (op 3610) 
s_add_u32 s54, s20, 256 // (op 3610) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 3610) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3610) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3610) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 8 // (op 3610) 

s_sub_i32 s49, s54, s1 // (op 3610) 

s_mul_i32 s51, s50, s49 // (op 3610) 


s_sub_u32 s49, s19, s51 // (op 3610) 


s_add_u32 s19, s49, s1 // (op 3610) 


s_cmp_eq_i32 s18, 0 // (op 3610) 
s_mov_b32 s1, scc // (op 3610) 

s_cmp_eq_i32 s1, 0 // (op 3610) 
s_cselect_b32 s18, s19, s0 // (op 3610) 



v_lshl_add_u32 v14, s18, 8, v13 // (op 3610) 


v_mov_b32 v17, 0 // (op 3610) convert
v_mov_b32 v16, v14 // (op 3610) convert

v_mul_lo_u32 v19, v15, s29 // (op 3610) most significant: low of low * high
v_mul_hi_u32 v13, v15, s28 // (op 3610) most significant: high of low * low
v_mul_lo_u32 v18, v15, s28 // (op 3610) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3610) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3610) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3610) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 3610) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 3610) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 3610) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3610) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3610) 

v_mov_b32 v13, v14 // (op 3610) convert

s_mul_i32 s1, s33, 6 // (op 3612) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3612) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3612) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3612) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3612) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3612) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3612) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3612) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3612) most significant: sum
s_add_u32 s19, s19, s50 // (op 3612) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3612) 

s_mov_b32 s24, s30 // (op 3612) reg expression
s_mov_b32 s25, s31 // (op 3612) reg expression
s_mov_b32 s26, s0 // (op 3612) reg expression
s_mov_b32 s27, 131072 // (op 3612) call()

s_mul_hi_i32 s0, s16, s8 // (op 3655) 
v_lshrrev_b32 v14, 3, v1 // (op 3655) 
s_add_i32 s1, s0, s16 // (op 3655) 

v_and_b32 v15, -8, v1 // (op 3655) 
s_ashr_i32 s0, s1, 31 // (op 3655) 
v_sub_u32 v16, v1, v15 // (op 3655) 

s_cmp_eq_i32 s8, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
v_lshlrev_b32 v15, 4, v16 // (op 3655) 

s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s19, 0, -1 // (op 3655) 

v_mov_b32 v17, 0 // (op 3655) convert
v_mov_b32 v16, v15 // (op 3655) convert

s_and_b32 s18, s9, 31 // (op 3655) 
s_lshl_b32 s49, 1, s18 // (op 3655) 
s_add_i32 s50, s49, s19 // (op 3655) 


s_and_b32 s19, s0, s50 // (op 3655) 


s_add_i32 s0, s1, s19 // (op 3655) 


s_ashr_i32 s1, s0, s18 // (op 3655) 


s_lshl_b32 s0, s9, 24 // (op 3655) 
s_ashr_i32 s18, s0, 31 // (op 3655) 

s_xor_b32 s0, s1, s18 // (op 3655) 

s_sub_i32 s1, s0, s18 // (op 3655) 


s_mul_hi_i32 s0, s1, s10 // (op 3655) 
s_add_i32 s18, s0, s1 // (op 3655) 

s_ashr_i32 s0, s18, 31 // (op 3655) 
s_cmp_eq_i32 s10, 0 // (op 3655) 
s_mov_b32 s19, scc // (op 3655) 
s_cmp_eq_i32 s19, 0 // (op 3655) 
s_cselect_b32 s49, 0, -1 // (op 3655) 

s_and_b32 s19, s11, 31 // (op 3655) 
s_lshl_b32 s50, 1, s19 // (op 3655) 
s_add_i32 s51, s50, s49 // (op 3655) 


s_and_b32 s49, s0, s51 // (op 3655) 


s_add_i32 s0, s18, s49 // (op 3655) 


s_ashr_i32 s18, s0, s19 // (op 3655) 


s_lshl_b32 s0, s11, 24 // (op 3655) 
s_ashr_i32 s19, s0, 31 // (op 3655) 

s_xor_b32 s0, s18, s19 // (op 3655) 

s_sub_i32 s18, s0, s19 // (op 3655) 


s_add_u32 s54, s4, 32 // (op 3655) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3655) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3655) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3655) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 3655) 

s_mul_i32 s0, s18, s50 // (op 3655) 

s_sub_i32 s49, s1, s0 // (op 3655) 


s_mul_hi_i32 s0, s16, s7 // (op 3655) 
s_add_i32 s1, s0, s16 // (op 3655) 

s_ashr_i32 s0, s1, 31 // (op 3655) 
s_cmp_eq_i32 s7, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s19, 0, -1 // (op 3655) 

s_and_b32 s18, s14, 31 // (op 3655) 
s_lshl_b32 s52, 1, s18 // (op 3655) 
s_add_i32 s53, s52, s19 // (op 3655) 


s_and_b32 s19, s0, s53 // (op 3655) 


s_add_i32 s0, s1, s19 // (op 3655) 


s_ashr_i32 s1, s0, s18 // (op 3655) 


s_lshl_b32 s0, s14, 24 // (op 3655) 
s_ashr_i32 s18, s0, 31 // (op 3655) 

s_xor_b32 s0, s1, s18 // (op 3655) 

s_sub_i32 s1, s0, s18 // (op 3655) 


s_mul_i32 s0, s13, s50 // (op 3655) 

s_mul_i32 s18, s6, s0 // (op 3655) 

s_mul_i32 s0, s1, s18 // (op 3655) 

s_sub_u32 s18, s16, s0 // (op 3655) 

s_mul_hi_i32 s0, s18, s12 // (op 3655) 
s_add_i32 s19, s0, s18 // (op 3655) 


s_ashr_i32 s0, s19, 31 // (op 3655) 
s_cmp_eq_i32 s12, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s50, 0, -1 // (op 3655) 

s_and_b32 s18, s48, 31 // (op 3655) 
s_lshl_b32 s51, 1, s18 // (op 3655) 
s_add_i32 s52, s51, s50 // (op 3655) 


s_and_b32 s50, s0, s52 // (op 3655) 


s_add_i32 s0, s19, s50 // (op 3655) 


s_ashr_i32 s19, s0, s18 // (op 3655) 


s_lshl_b32 s0, s48, 24 // (op 3655) 
s_ashr_i32 s18, s0, 31 // (op 3655) 

s_xor_b32 s0, s19, s18 // (op 3655) 

s_sub_i32 s19, s0, s18 // (op 3655) 


s_cmp_eq_i32 s1, 0 // (op 3655) 
s_mov_b32 s0, scc // (op 3655) 

s_cmp_eq_i32 s0, 0 // (op 3655) 
s_cselect_b32 s1, s19, s49 // (op 3655) 



v_lshl_add_u32 v15, s1, 5, v14 // (op 3655) 


v_mul_lo_u32 v19, v15, s35 // (op 3655) most significant: low of low * high
v_mul_hi_u32 v14, v15, s34 // (op 3655) most significant: high of low * low
v_mul_lo_u32 v18, v15, s34 // (op 3655) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3655) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3655) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3655) most significant half



s_mul_i32 s1, s34, s5 // (op 3657) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3657) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3657) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3657) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3657) most significant: sum
s_add_u32 s1, s1, s19 // (op 3657) most significant: sum


s_mov_b32 s52, s36 // (op 3657) reg expression
s_mov_b32 s53, s37 // (op 3657) reg expression
s_mov_b32 s54, s0 // (op 3657) reg expression
s_mov_b32 s55, 131072 // (op 3657) call()

s_mul_hi_i32 s0, s16, s8 // (op 3674) 
v_lshrrev_b32 v16, 3, v1 // (op 3674) 
s_add_i32 s1, s0, s16 // (op 3674) 

v_and_b32 v17, -8, v1 // (op 3674) 
s_ashr_i32 s0, s1, 31 // (op 3674) 
v_sub_u32 v18, v1, v17 // (op 3674) 

s_cmp_eq_i32 s8, 0 // (op 3674) 
s_mov_b32 s18, scc // (op 3674) 
v_lshlrev_b32 v17, 4, v18 // (op 3674) 

s_cmp_eq_i32 s18, 0 // (op 3674) 
s_cselect_b32 s19, 0, -1 // (op 3674) 

v_mov_b32 v19, 0 // (op 3674) convert
v_mov_b32 v18, v17 // (op 3674) convert

s_and_b32 s18, s9, 31 // (op 3674) 
s_lshl_b32 s49, 1, s18 // (op 3674) 
s_add_i32 s50, s49, s19 // (op 3674) 


s_and_b32 s19, s0, s50 // (op 3674) 


s_add_i32 s0, s1, s19 // (op 3674) 


s_ashr_i32 s1, s0, s18 // (op 3674) 


s_lshl_b32 s0, s9, 24 // (op 3674) 
s_ashr_i32 s18, s0, 31 // (op 3674) 

s_xor_b32 s0, s1, s18 // (op 3674) 

s_sub_i32 s1, s0, s18 // (op 3674) 


s_mul_hi_i32 s0, s1, s10 // (op 3674) 
s_add_i32 s18, s0, s1 // (op 3674) 

s_ashr_i32 s0, s18, 31 // (op 3674) 
s_cmp_eq_i32 s10, 0 // (op 3674) 
s_mov_b32 s19, scc // (op 3674) 
s_cmp_eq_i32 s19, 0 // (op 3674) 
s_cselect_b32 s49, 0, -1 // (op 3674) 

s_and_b32 s19, s11, 31 // (op 3674) 
s_lshl_b32 s50, 1, s19 // (op 3674) 
s_add_i32 s51, s50, s49 // (op 3674) 


s_and_b32 s49, s0, s51 // (op 3674) 


s_add_i32 s0, s18, s49 // (op 3674) 


s_ashr_i32 s18, s0, s19 // (op 3674) 


s_lshl_b32 s0, s11, 24 // (op 3674) 
s_ashr_i32 s19, s0, 31 // (op 3674) 

s_xor_b32 s0, s18, s19 // (op 3674) 

s_sub_i32 s18, s0, s19 // (op 3674) 


s_add_u32 s50, s4, 32 // (op 3674) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3674) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3674) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3674) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 3674) 

s_mul_i32 s0, s18, s50 // (op 3674) 

s_sub_i32 s49, s1, s0 // (op 3674) 


s_mul_hi_i32 s0, s16, s7 // (op 3674) 
s_add_i32 s1, s0, s16 // (op 3674) 

s_ashr_i32 s0, s1, 31 // (op 3674) 
s_cmp_eq_i32 s7, 0 // (op 3674) 
s_mov_b32 s18, scc // (op 3674) 
s_cmp_eq_i32 s18, 0 // (op 3674) 
s_cselect_b32 s19, 0, -1 // (op 3674) 

s_and_b32 s18, s14, 31 // (op 3674) 
s_lshl_b32 s64, 1, s18 // (op 3674) 
s_add_i32 s65, s64, s19 // (op 3674) 


s_and_b32 s19, s0, s65 // (op 3674) 


s_add_i32 s0, s1, s19 // (op 3674) 


s_ashr_i32 s1, s0, s18 // (op 3674) 


s_lshl_b32 s0, s14, 24 // (op 3674) 
s_ashr_i32 s18, s0, 31 // (op 3674) 

s_xor_b32 s0, s1, s18 // (op 3674) 

s_sub_i32 s1, s0, s18 // (op 3674) 


s_mul_i32 s0, s13, s50 // (op 3674) 

s_mul_i32 s18, s6, s0 // (op 3674) 

s_mul_i32 s0, s1, s18 // (op 3674) 

s_sub_u32 s18, s16, s0 // (op 3674) 

s_mul_hi_i32 s0, s18, s12 // (op 3674) 
s_add_i32 s19, s0, s18 // (op 3674) 


s_ashr_i32 s0, s19, 31 // (op 3674) 
s_cmp_eq_i32 s12, 0 // (op 3674) 
s_mov_b32 s18, scc // (op 3674) 
s_cmp_eq_i32 s18, 0 // (op 3674) 
s_cselect_b32 s50, 0, -1 // (op 3674) 

s_and_b32 s18, s48, 31 // (op 3674) 
s_lshl_b32 s51, 1, s18 // (op 3674) 
s_add_i32 s64, s51, s50 // (op 3674) 


s_and_b32 s50, s0, s64 // (op 3674) 


s_add_i32 s0, s19, s50 // (op 3674) 


s_ashr_i32 s19, s0, s18 // (op 3674) 


s_lshl_b32 s0, s48, 24 // (op 3674) 
s_ashr_i32 s18, s0, 31 // (op 3674) 

s_xor_b32 s0, s19, s18 // (op 3674) 

s_sub_i32 s19, s0, s18 // (op 3674) 


s_cmp_eq_i32 s1, 0 // (op 3674) 
s_mov_b32 s0, scc // (op 3674) 

s_cmp_eq_i32 s0, 0 // (op 3674) 
s_cselect_b32 s1, s19, s49 // (op 3674) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 3674) 


v_mul_lo_u32 v21, v17, s35 // (op 3674) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 3674) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 3674) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 3674) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3674) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3674) most significant half



s_mul_i32 s1, s34, s5 // (op 3676) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3676) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3676) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3676) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3676) most significant: sum
s_add_u32 s1, s1, s19 // (op 3676) most significant: sum


s_mov_b32 s52, s36 // (op 3676) reg expression
s_mov_b32 s53, s37 // (op 3676) reg expression
s_mov_b32 s54, s0 // (op 3676) reg expression
s_mov_b32 s55, 131072 // (op 3676) call()

s_mul_hi_i32 s0, s16, s8 // (op 3719) 
v_and_b32 v18, -64, v1 // (op 3719) 
s_add_i32 s1, s0, s16 // (op 3719) 

v_sub_u32 v19, v1, v18 // (op 3719) 

s_ashr_i32 s0, s1, 31 // (op 3719) 
v_lshlrev_b32 v18, 2, v19 // (op 3719) 

s_cmp_eq_i32 s8, 0 // (op 3719) 
s_mov_b32 s18, scc // (op 3719) 
v_lshrrev_b32 v19, 6, v1 // (op 3719) 
s_cmp_eq_i32 s18, 0 // (op 3719) 
s_cselect_b32 s19, 0, -1 // (op 3719) 

s_and_b32 s18, s9, 31 // (op 3719) 
s_lshl_b32 s49, 1, s18 // (op 3719) 
s_add_i32 s50, s49, s19 // (op 3719) 


s_and_b32 s19, s0, s50 // (op 3719) 


s_add_i32 s0, s1, s19 // (op 3719) 


s_ashr_i32 s1, s0, s18 // (op 3719) 


s_lshl_b32 s0, s9, 24 // (op 3719) 
s_ashr_i32 s18, s0, 31 // (op 3719) 

s_xor_b32 s0, s1, s18 // (op 3719) 

s_sub_i32 s1, s0, s18 // (op 3719) 


s_mul_hi_i32 s0, s1, s10 // (op 3719) 
s_add_i32 s18, s0, s1 // (op 3719) 

s_ashr_i32 s0, s18, 31 // (op 3719) 
s_cmp_eq_i32 s10, 0 // (op 3719) 
s_mov_b32 s19, scc // (op 3719) 
s_cmp_eq_i32 s19, 0 // (op 3719) 
s_cselect_b32 s49, 0, -1 // (op 3719) 

s_and_b32 s19, s11, 31 // (op 3719) 
s_lshl_b32 s50, 1, s19 // (op 3719) 
s_add_i32 s51, s50, s49 // (op 3719) 


s_and_b32 s49, s0, s51 // (op 3719) 


s_add_i32 s0, s18, s49 // (op 3719) 


s_ashr_i32 s18, s0, s19 // (op 3719) 


s_lshl_b32 s0, s11, 24 // (op 3719) 
s_ashr_i32 s19, s0, 31 // (op 3719) 

s_xor_b32 s0, s18, s19 // (op 3719) 

s_sub_i32 s18, s0, s19 // (op 3719) 


s_mul_i32 s0, s1, s13 // (op 3719) 

s_sub_u32 s1, s16, s0 // (op 3719) 

s_mul_i32 s0, s18, s13 // (op 3719) 
s_add_u32 s0, s0, s1 // (op 3719) 


s_mul_hi_i32 s1, s16, s7 // (op 3719) 
s_add_i32 s18, s1, s16 // (op 3719) 

s_ashr_i32 s1, s18, 31 // (op 3719) 
s_cmp_eq_i32 s7, 0 // (op 3719) 
s_mov_b32 s19, scc // (op 3719) 
s_cmp_eq_i32 s19, 0 // (op 3719) 
s_cselect_b32 s49, 0, -1 // (op 3719) 

s_and_b32 s19, s14, 31 // (op 3719) 
s_lshl_b32 s50, 1, s19 // (op 3719) 
s_add_i32 s51, s50, s49 // (op 3719) 


s_and_b32 s49, s1, s51 // (op 3719) 


s_add_i32 s1, s18, s49 // (op 3719) 


s_ashr_i32 s18, s1, s19 // (op 3719) 


s_lshl_b32 s1, s14, 24 // (op 3719) 
s_ashr_i32 s19, s1, 31 // (op 3719) 

s_xor_b32 s1, s18, s19 // (op 3719) 

s_sub_i32 s18, s1, s19 // (op 3719) 


s_add_u32 s50, s4, 32 // (op 3719) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3719) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3719) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3719) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 3719) 

s_mul_i32 s1, s13, s50 // (op 3719) 

s_mul_i32 s19, s6, s1 // (op 3719) 

s_mul_i32 s1, s18, s19 // (op 3719) 

s_sub_u32 s19, s16, s1 // (op 3719) 

s_mov_b32 s1, s19 // (op 3719) 
s_mul_hi_i32 s49, s1, s12 // (op 3719) 
s_add_i32 s50, s49, s1 // (op 3719) 


s_ashr_i32 s1, s50, 31 // (op 3719) 
s_cmp_eq_i32 s12, 0 // (op 3719) 
s_mov_b32 s49, scc // (op 3719) 
s_cmp_eq_i32 s49, 0 // (op 3719) 
s_cselect_b32 s51, 0, -1 // (op 3719) 

s_and_b32 s49, s48, 31 // (op 3719) 
s_lshl_b32 s64, 1, s49 // (op 3719) 
s_add_i32 s65, s64, s51 // (op 3719) 


s_and_b32 s51, s1, s65 // (op 3719) 


s_add_i32 s1, s50, s51 // (op 3719) 


s_ashr_i32 s50, s1, s49 // (op 3719) 


s_lshl_b32 s1, s48, 24 // (op 3719) 
s_ashr_i32 s49, s1, 31 // (op 3719) 

s_xor_b32 s1, s50, s49 // (op 3719) 

s_sub_i32 s50, s1, s49 // (op 3719) 


s_mul_i32 s1, s15, s13 // (op 3719) 
s_add_u32 s64, s20, 256 // (op 3719) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 3719) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3719) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3719) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 3719) 

s_sub_i32 s49, s64, s1 // (op 3719) 

s_mul_i32 s51, s50, s49 // (op 3719) 


s_sub_u32 s49, s19, s51 // (op 3719) 


s_add_u32 s19, s49, s1 // (op 3719) 


s_cmp_eq_i32 s18, 0 // (op 3719) 
s_mov_b32 s1, scc // (op 3719) 

s_cmp_eq_i32 s1, 0 // (op 3719) 
s_cselect_b32 s18, s19, s0 // (op 3719) 



v_lshl_add_u32 v20, s18, 8, v18 // (op 3719) 


v_mul_lo_u32 v18, v19, s38 // (op 3719) 

v_add_u32 v19, v20, v18 // (op 3719) 


s_mul_i32 s1, s38, s43 // (op 3721) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3721) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3721) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3721) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3721) most significant: sum
s_add_u32 s1, s1, s19 // (op 3721) most significant: sum


s_mov_b32 s64, s40 // (op 3721) reg expression
s_mov_b32 s65, s41 // (op 3721) reg expression
s_mov_b32 s66, s0 // (op 3721) reg expression
s_mov_b32 s67, 131072 // (op 3721) call()

s_mul_hi_i32 s0, s16, s8 // (op 3738) 
v_and_b32 v18, -64, v1 // (op 3738) 
s_add_i32 s1, s0, s16 // (op 3738) 

v_sub_u32 v20, v1, v18 // (op 3738) 

s_ashr_i32 s0, s1, 31 // (op 3738) 
v_lshlrev_b32 v18, 2, v20 // (op 3738) 

s_cmp_eq_i32 s8, 0 // (op 3738) 
s_mov_b32 s18, scc // (op 3738) 
v_lshrrev_b32 v20, 6, v1 // (op 3738) 
s_cmp_eq_i32 s18, 0 // (op 3738) 
s_cselect_b32 s19, 0, -1 // (op 3738) 

s_and_b32 s18, s9, 31 // (op 3738) 
s_lshl_b32 s49, 1, s18 // (op 3738) 
s_add_i32 s50, s49, s19 // (op 3738) 


s_and_b32 s19, s0, s50 // (op 3738) 


s_add_i32 s0, s1, s19 // (op 3738) 


s_ashr_i32 s1, s0, s18 // (op 3738) 


s_lshl_b32 s0, s9, 24 // (op 3738) 
s_ashr_i32 s18, s0, 31 // (op 3738) 

s_xor_b32 s0, s1, s18 // (op 3738) 

s_sub_i32 s1, s0, s18 // (op 3738) 


s_mul_hi_i32 s0, s1, s10 // (op 3738) 
s_add_i32 s18, s0, s1 // (op 3738) 

s_ashr_i32 s0, s18, 31 // (op 3738) 
s_cmp_eq_i32 s10, 0 // (op 3738) 
s_mov_b32 s19, scc // (op 3738) 
s_cmp_eq_i32 s19, 0 // (op 3738) 
s_cselect_b32 s49, 0, -1 // (op 3738) 

s_and_b32 s19, s11, 31 // (op 3738) 
s_lshl_b32 s50, 1, s19 // (op 3738) 
s_add_i32 s51, s50, s49 // (op 3738) 


s_and_b32 s49, s0, s51 // (op 3738) 


s_add_i32 s0, s18, s49 // (op 3738) 


s_ashr_i32 s18, s0, s19 // (op 3738) 


s_lshl_b32 s0, s11, 24 // (op 3738) 
s_ashr_i32 s19, s0, 31 // (op 3738) 

s_xor_b32 s0, s18, s19 // (op 3738) 

s_sub_i32 s18, s0, s19 // (op 3738) 


s_mul_i32 s0, s1, s13 // (op 3738) 

s_sub_u32 s1, s16, s0 // (op 3738) 

s_mul_i32 s0, s18, s13 // (op 3738) 
s_add_u32 s0, s0, s1 // (op 3738) 


s_mul_hi_i32 s1, s16, s7 // (op 3738) 
s_add_i32 s18, s1, s16 // (op 3738) 

s_ashr_i32 s1, s18, 31 // (op 3738) 
s_cmp_eq_i32 s7, 0 // (op 3738) 
s_mov_b32 s19, scc // (op 3738) 
s_cmp_eq_i32 s19, 0 // (op 3738) 
s_cselect_b32 s49, 0, -1 // (op 3738) 

s_and_b32 s19, s14, 31 // (op 3738) 
s_lshl_b32 s50, 1, s19 // (op 3738) 
s_add_i32 s51, s50, s49 // (op 3738) 


s_and_b32 s49, s1, s51 // (op 3738) 


s_add_i32 s1, s18, s49 // (op 3738) 


s_ashr_i32 s18, s1, s19 // (op 3738) 


s_lshl_b32 s1, s14, 24 // (op 3738) 
s_ashr_i32 s19, s1, 31 // (op 3738) 

s_xor_b32 s1, s18, s19 // (op 3738) 

s_sub_i32 s18, s1, s19 // (op 3738) 


s_add_u32 s50, s4, 32 // (op 3738) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3738) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3738) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3738) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 3738) 

s_mul_i32 s1, s13, s50 // (op 3738) 

s_mul_i32 s19, s6, s1 // (op 3738) 

s_mul_i32 s1, s18, s19 // (op 3738) 

s_sub_u32 s19, s16, s1 // (op 3738) 

s_mov_b32 s1, s19 // (op 3738) 
s_mul_hi_i32 s49, s1, s12 // (op 3738) 
s_add_i32 s50, s49, s1 // (op 3738) 


s_ashr_i32 s1, s50, 31 // (op 3738) 
s_cmp_eq_i32 s12, 0 // (op 3738) 
s_mov_b32 s49, scc // (op 3738) 
s_cmp_eq_i32 s49, 0 // (op 3738) 
s_cselect_b32 s51, 0, -1 // (op 3738) 

s_and_b32 s49, s48, 31 // (op 3738) 
s_lshl_b32 s68, 1, s49 // (op 3738) 
s_add_i32 s69, s68, s51 // (op 3738) 


s_and_b32 s51, s1, s69 // (op 3738) 


s_add_i32 s1, s50, s51 // (op 3738) 


s_ashr_i32 s50, s1, s49 // (op 3738) 


s_lshl_b32 s1, s48, 24 // (op 3738) 
s_ashr_i32 s49, s1, 31 // (op 3738) 

s_xor_b32 s1, s50, s49 // (op 3738) 

s_sub_i32 s50, s1, s49 // (op 3738) 


s_mul_i32 s1, s15, s13 // (op 3738) 
s_add_u32 s68, s20, 256 // (op 3738) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 3738) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3738) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3738) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 8 // (op 3738) 

s_sub_i32 s49, s68, s1 // (op 3738) 

s_mul_i32 s51, s50, s49 // (op 3738) 


s_sub_u32 s49, s19, s51 // (op 3738) 


s_add_u32 s19, s49, s1 // (op 3738) 


s_cmp_eq_i32 s18, 0 // (op 3738) 
s_mov_b32 s1, scc // (op 3738) 

s_cmp_eq_i32 s1, 0 // (op 3738) 
s_cselect_b32 s18, s19, s0 // (op 3738) 



v_lshl_add_u32 v21, s18, 8, v18 // (op 3738) 


v_mul_lo_u32 v18, v20, s38 // (op 3738) 

v_add_u32 v20, v21, v18 // (op 3738) 


s_mul_i32 s1, s38, s43 // (op 3740) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3740) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3740) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3740) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3740) most significant: sum
s_add_u32 s1, s1, s19 // (op 3740) most significant: sum


s_mov_b32 s64, s40 // (op 3740) reg expression
s_mov_b32 s65, s41 // (op 3740) reg expression
s_mov_b32 s66, s0 // (op 3740) reg expression
s_mov_b32 s67, 131072 // (op 3740) call()

s_mul_hi_i32 s0, s16, s8 // (op 3783) 
v_lshrrev_b32 v18, 6, v1 // (op 3783) 
s_add_i32 s1, s0, s16 // (op 3783) 

v_and_b32 v21, 1, v18 // (op 3783) 

s_ashr_i32 s0, s1, 31 // (op 3783) 
v_and_b32 v18, 15, v1 // (op 3783) 
s_cmp_eq_i32 s8, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 3783) 


s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s19, 0, -1 // (op 3783) 

v_and_b32 v18, 63, v1 // (op 3783) 
s_and_b32 s18, s9, 31 // (op 3783) 
v_lshrrev_b32 v21, 4, v18 // (op 3783) 

s_lshl_b32 s49, 1, s18 // (op 3783) 
s_add_i32 s50, s49, s19 // (op 3783) 


s_and_b32 s19, s0, s50 // (op 3783) 


s_add_i32 s0, s1, s19 // (op 3783) 


s_ashr_i32 s1, s0, s18 // (op 3783) 


s_lshl_b32 s0, s9, 24 // (op 3783) 
s_ashr_i32 s18, s0, 31 // (op 3783) 

s_xor_b32 s0, s1, s18 // (op 3783) 

s_sub_i32 s1, s0, s18 // (op 3783) 


s_mul_hi_i32 s0, s1, s10 // (op 3783) 
s_add_i32 s18, s0, s1 // (op 3783) 

s_ashr_i32 s0, s18, 31 // (op 3783) 
s_cmp_eq_i32 s10, 0 // (op 3783) 
s_mov_b32 s19, scc // (op 3783) 
s_cmp_eq_i32 s19, 0 // (op 3783) 
s_cselect_b32 s49, 0, -1 // (op 3783) 

s_and_b32 s19, s11, 31 // (op 3783) 
s_lshl_b32 s50, 1, s19 // (op 3783) 
s_add_i32 s51, s50, s49 // (op 3783) 


s_and_b32 s49, s0, s51 // (op 3783) 


s_add_i32 s0, s18, s49 // (op 3783) 


s_ashr_i32 s18, s0, s19 // (op 3783) 


s_lshl_b32 s0, s11, 24 // (op 3783) 
s_ashr_i32 s19, s0, 31 // (op 3783) 

s_xor_b32 s0, s18, s19 // (op 3783) 

s_sub_i32 s18, s0, s19 // (op 3783) 


s_add_u32 s50, s4, 32 // (op 3783) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3783) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3783) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3783) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 3783) 

s_mul_i32 s0, s18, s50 // (op 3783) 

s_sub_i32 s49, s1, s0 // (op 3783) 


s_mul_hi_i32 s0, s16, s7 // (op 3783) 
s_add_i32 s1, s0, s16 // (op 3783) 

s_ashr_i32 s0, s1, 31 // (op 3783) 
s_cmp_eq_i32 s7, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s19, 0, -1 // (op 3783) 

s_and_b32 s18, s14, 31 // (op 3783) 
s_lshl_b32 s68, 1, s18 // (op 3783) 
s_add_i32 s69, s68, s19 // (op 3783) 


s_and_b32 s19, s0, s69 // (op 3783) 


s_add_i32 s0, s1, s19 // (op 3783) 


s_ashr_i32 s1, s0, s18 // (op 3783) 


s_lshl_b32 s0, s14, 24 // (op 3783) 
s_ashr_i32 s18, s0, 31 // (op 3783) 

s_xor_b32 s0, s1, s18 // (op 3783) 

s_sub_i32 s1, s0, s18 // (op 3783) 


s_mul_i32 s0, s13, s50 // (op 3783) 

s_mul_i32 s18, s6, s0 // (op 3783) 

s_mul_i32 s0, s1, s18 // (op 3783) 

s_sub_u32 s18, s16, s0 // (op 3783) 

s_mul_hi_i32 s0, s18, s12 // (op 3783) 
s_add_i32 s19, s0, s18 // (op 3783) 


s_ashr_i32 s0, s19, 31 // (op 3783) 
s_cmp_eq_i32 s12, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s50, 0, -1 // (op 3783) 

s_and_b32 s18, s48, 31 // (op 3783) 
s_lshl_b32 s51, 1, s18 // (op 3783) 
s_add_i32 s68, s51, s50 // (op 3783) 


s_and_b32 s50, s0, s68 // (op 3783) 


s_add_i32 s0, s19, s50 // (op 3783) 


s_ashr_i32 s19, s0, s18 // (op 3783) 


s_lshl_b32 s0, s48, 24 // (op 3783) 
s_ashr_i32 s18, s0, 31 // (op 3783) 

s_xor_b32 s0, s19, s18 // (op 3783) 

s_sub_i32 s19, s0, s18 // (op 3783) 


s_cmp_eq_i32 s1, 0 // (op 3783) 
s_mov_b32 s0, scc // (op 3783) 

s_cmp_eq_i32 s0, 0 // (op 3783) 
s_cselect_b32 s1, s19, s49 // (op 3783) 



v_lshl_add_u32 v18, s1, 5, v22 // (op 3783) 


v_mul_lo_u32 v22, v18, s44 // (op 3783) 

v_add_u32 v18, v21, v22 // (op 3783) 


s_mul_i32 s1, s44, s5 // (op 3785) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3785) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3785) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3785) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3785) most significant: sum
s_add_u32 s1, s1, s19 // (op 3785) most significant: sum


s_mov_b32 s68, s46 // (op 3785) reg expression
s_mov_b32 s69, s47 // (op 3785) reg expression
s_mov_b32 s70, s0 // (op 3785) reg expression
s_mov_b32 s71, 131072 // (op 3785) call()

s_mul_hi_i32 s0, s16, s8 // (op 3802) 
v_lshrrev_b32 v21, 6, v1 // (op 3802) 
s_add_i32 s1, s0, s16 // (op 3802) 

v_and_b32 v22, 1, v21 // (op 3802) 

s_ashr_i32 s0, s1, 31 // (op 3802) 
v_and_b32 v21, 15, v1 // (op 3802) 
s_cmp_eq_i32 s8, 0 // (op 3802) 
s_mov_b32 s18, scc // (op 3802) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 3802) 


s_cmp_eq_i32 s18, 0 // (op 3802) 
s_cselect_b32 s19, 0, -1 // (op 3802) 

v_and_b32 v21, 63, v1 // (op 3802) 
s_and_b32 s18, s9, 31 // (op 3802) 
v_lshrrev_b32 v22, 4, v21 // (op 3802) 

s_lshl_b32 s49, 1, s18 // (op 3802) 
s_add_i32 s50, s49, s19 // (op 3802) 


s_and_b32 s19, s0, s50 // (op 3802) 


s_add_i32 s0, s1, s19 // (op 3802) 


s_ashr_i32 s1, s0, s18 // (op 3802) 


s_lshl_b32 s0, s9, 24 // (op 3802) 
s_ashr_i32 s18, s0, 31 // (op 3802) 

s_xor_b32 s0, s1, s18 // (op 3802) 

s_sub_i32 s1, s0, s18 // (op 3802) 


s_mul_hi_i32 s0, s1, s10 // (op 3802) 
s_add_i32 s18, s0, s1 // (op 3802) 

s_ashr_i32 s0, s18, 31 // (op 3802) 
s_cmp_eq_i32 s10, 0 // (op 3802) 
s_mov_b32 s19, scc // (op 3802) 
s_cmp_eq_i32 s19, 0 // (op 3802) 
s_cselect_b32 s49, 0, -1 // (op 3802) 

s_and_b32 s19, s11, 31 // (op 3802) 
s_lshl_b32 s50, 1, s19 // (op 3802) 
s_add_i32 s51, s50, s49 // (op 3802) 


s_and_b32 s49, s0, s51 // (op 3802) 


s_add_i32 s0, s18, s49 // (op 3802) 


s_ashr_i32 s18, s0, s19 // (op 3802) 


s_lshl_b32 s0, s11, 24 // (op 3802) 
s_ashr_i32 s19, s0, 31 // (op 3802) 

s_xor_b32 s0, s18, s19 // (op 3802) 

s_sub_i32 s18, s0, s19 // (op 3802) 


s_add_u32 s50, s4, 32 // (op 3802) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3802) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3802) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3802) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3802) 

s_mul_i32 s0, s18, s50 // (op 3802) 

s_sub_i32 s49, s1, s0 // (op 3802) 


s_mul_hi_i32 s0, s16, s7 // (op 3802) 
s_add_i32 s1, s0, s16 // (op 3802) 

s_ashr_i32 s0, s1, 31 // (op 3802) 
s_cmp_eq_i32 s7, 0 // (op 3802) 
s_mov_b32 s18, scc // (op 3802) 
s_cmp_eq_i32 s18, 0 // (op 3802) 
s_cselect_b32 s19, 0, -1 // (op 3802) 

s_and_b32 s18, s14, 31 // (op 3802) 
s_lshl_b32 s72, 1, s18 // (op 3802) 
s_add_i32 s73, s72, s19 // (op 3802) 


s_and_b32 s19, s0, s73 // (op 3802) 


s_add_i32 s0, s1, s19 // (op 3802) 


s_ashr_i32 s1, s0, s18 // (op 3802) 


s_lshl_b32 s0, s14, 24 // (op 3802) 
s_ashr_i32 s18, s0, 31 // (op 3802) 

s_xor_b32 s0, s1, s18 // (op 3802) 

s_sub_i32 s1, s0, s18 // (op 3802) 


s_mul_i32 s0, s13, s50 // (op 3802) 

s_mul_i32 s18, s6, s0 // (op 3802) 

s_mul_i32 s0, s1, s18 // (op 3802) 

s_sub_u32 s18, s16, s0 // (op 3802) 

s_mul_hi_i32 s0, s18, s12 // (op 3802) 
s_add_i32 s19, s0, s18 // (op 3802) 


s_ashr_i32 s0, s19, 31 // (op 3802) 
s_cmp_eq_i32 s12, 0 // (op 3802) 
s_mov_b32 s18, scc // (op 3802) 
s_cmp_eq_i32 s18, 0 // (op 3802) 
s_cselect_b32 s50, 0, -1 // (op 3802) 

s_and_b32 s18, s48, 31 // (op 3802) 
s_lshl_b32 s51, 1, s18 // (op 3802) 
s_add_i32 s72, s51, s50 // (op 3802) 


s_and_b32 s50, s0, s72 // (op 3802) 


s_add_i32 s0, s19, s50 // (op 3802) 


s_ashr_i32 s19, s0, s18 // (op 3802) 


s_lshl_b32 s0, s48, 24 // (op 3802) 
s_ashr_i32 s18, s0, 31 // (op 3802) 

s_xor_b32 s0, s19, s18 // (op 3802) 

s_sub_i32 s19, s0, s18 // (op 3802) 


s_cmp_eq_i32 s1, 0 // (op 3802) 
s_mov_b32 s0, scc // (op 3802) 

s_cmp_eq_i32 s0, 0 // (op 3802) 
s_cselect_b32 s1, s19, s49 // (op 3802) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 3802) 


v_mul_lo_u32 v23, v21, s44 // (op 3802) 

v_add_u32 v21, v22, v23 // (op 3802) 


s_mul_i32 s1, s44, s5 // (op 3804) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3804) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3804) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3804) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3804) most significant: sum
s_add_u32 s1, s1, s19 // (op 3804) most significant: sum


s_mov_b32 s68, s46 // (op 3804) reg expression
s_mov_b32 s69, s47 // (op 3804) reg expression
s_mov_b32 s70, s0 // (op 3804) reg expression
s_mov_b32 s71, 131072 // (op 3804) call()




buffer_load_ubyte v22, v18, s[68:71], 0 offen offset: 0 // (op 3252) Load value
v_and_b32 v23, -8, v1 // (op 3926) 
v_sub_u32 v24, v1, v23 // (op 3926) 

v_lshlrev_b32 v23, 4, v24 // (op 3926) 

v_lshrrev_b32 v24, 3, v1 // (op 3926) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 3926) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 3926) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3540) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 3540) Load value direct to lds


s_mul_i32 s1, s29, 6 // (op 3259) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3259) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3259) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3259) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3259) 


v_mov_b32 v23, v12 // (op 3259) reg expression
buffer_load_dwordx3 v[26:28], v23, s[24:27], 0 offen // (op 3259) Load value
buffer_load_dwordx3 v[30:32], v23, s[24:27], 0 offen offset:12 // (op 3259) Load value
buffer_load_dwordx3 v[34:36], v23, s[24:27], 0 offen offset:24 // (op 3259) Load value
buffer_load_dwordx3 v[38:40], v23, s[24:27], 0 offen offset:36 // (op 3259) Load value
v_add_u32 v23, v23, v24 // (op 3259) 
buffer_load_dwordx3 v[42:44], v23, s[24:27], 0 offen // (op 3259) Load value
buffer_load_dwordx3 v[46:48], v23, s[24:27], 0 offen offset:12 // (op 3259) Load value
buffer_load_dwordx3 v[50:52], v23, s[24:27], 0 offen offset:24 // (op 3259) Load value
buffer_load_dwordx3 v[54:56], v23, s[24:27], 0 offen offset:36 // (op 3259) Load value




v_mov_b32 v29, v19 // (op 3262) reg expression

buffer_load_dword v33, v29, s[64:67], 0 offen // (op 3262) Load value




buffer_load_ubyte v29, v18, s[68:71], 0 offen offset:4 // (op 3266) Load value

v_and_b32 v18, -8, v1 // (op 4070) 
v_sub_u32 v37, v1, v18 // (op 4070) 

v_lshlrev_b32 v18, 4, v37 // (op 4070) 

v_lshrrev_b32 v37, 3, v1 // (op 4070) 
v_lshl_add_u32 v41, v37, 7, v18 // (op 4070) 


s_nop 0
v_readfirstlane_b32 s0, v41 // (op 4070) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3545) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3545) reg expression
s_mov_b32 s18, 128 // (op 3545) call()
buffer_load_dwordx4 v14, s[52:55], s18 offen offset: 0 lds // (op 3545) Load value direct to lds





s_mul_i32 s1, s29, 6 // (op 3273) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3273) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3273) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3273) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 3273) 


s_lshl_b64 s[0:1], s[28:29], 7 // (op 3273) 
s_mul_i32 s19, s1, 6 // (op 3273) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 3273) least significant
s_mul_hi_u32 s49, s0, 6 // (op 3273) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3273) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 3273) 

v_add_u32 v18, v12, s0 // (op 3273) 

buffer_load_dwordx3 v[58:60], v18, s[24:27], 0 offen // (op 3273) Load value
buffer_load_dwordx3 v[62:64], v18, s[24:27], 0 offen offset:12 // (op 3273) Load value
buffer_load_dwordx3 v[66:68], v18, s[24:27], 0 offen offset:24 // (op 3273) Load value
buffer_load_dwordx3 v[70:72], v18, s[24:27], 0 offen offset:36 // (op 3273) Load value
v_add_u32 v18, v18, v14 // (op 3273) 
buffer_load_dwordx3 v[74:76], v18, s[24:27], 0 offen // (op 3273) Load value
buffer_load_dwordx3 v[78:80], v18, s[24:27], 0 offen offset:12 // (op 3273) Load value
buffer_load_dwordx3 v[82:84], v18, s[24:27], 0 offen offset:24 // (op 3273) Load value
buffer_load_dwordx3 v[86:88], v18, s[24:27], 0 offen offset:36 // (op 3273) Load value





s_lshl_b32 s0, s38, 2 // (op 3276) 
v_add_u32 v12, v19, s0 // (op 3276) 


buffer_load_dword v18, v12, s[64:67], 0 offen // (op 3276) Load value


v_and_b32 v12, -4, v1 // (op 3869) 
v_sub_u32 v19, v1, v12 // (op 3869) 

v_lshlrev_b32 v12, 6, v19 // (op 3869) 

v_lshrrev_b32 v19, 2, v1 // (op 3869) 
v_lshl_add_u32 v37, v19, 9, v12 // (op 3869) 


v_mul_lo_u32 v12, v37, 6 // (op 3869) 
v_lshrrev_b32 v19, 3, v12 // (op 3869) 

v_lshrrev_b32 v12, 4, v37 // (op 3869) 

v_lshlrev_b32 v37, 2, v12 // (op 3869) 

v_add_u32 v12, v19, v37 // (op 3869) 





s_waitcnt vmcnt(19)
ds_write_b96 v12, v[26:28] offset:8192 // (op 3279) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[30:32] offset:8208 // (op 3279) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[34:36] offset:8224 // (op 3279) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[38:40] offset:8240 // (op 3279) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[42:44] offset:8448 // (op 3279) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[46:48] offset:8464 // (op 3279) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[50:52] offset:8480 // (op 3279) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[54:56] offset:8496 // (op 3279) Store local data 

v_and_b32 v12, -64, v1 // (op 3979) 
v_sub_u32 v19, v1, v12 // (op 3979) 

v_lshlrev_b32 v12, 2, v19 // (op 3979) 

v_lshrrev_b32 v19, 6, v1 // (op 3979) 
v_lshl_add_u32 v37, v19, 8, v12 // (op 3979) 





s_waitcnt vmcnt(11)
ds_write_b32 v37, v33 offset:40960 // (op 3280) Store local data 

v_and_b32 v12, -4, v1 // (op 3840) 
v_sub_u32 v19, v1, v12 // (op 3840) 

v_lshlrev_b32 v12, 6, v19 // (op 3840) 

v_lshrrev_b32 v19, 2, v1 // (op 3840) 
v_lshl_add_u32 v37, v19, 9, v12 // (op 3840) 


v_mul_lo_u32 v12, v37, 6 // (op 3840) 
v_lshrrev_b32 v19, 3, v12 // (op 3840) 

v_lshrrev_b32 v12, 4, v37 // (op 3840) 

v_lshlrev_b32 v37, 2, v12 // (op 3840) 

v_add_u32 v12, v19, v37 // (op 3840) 


v_and_b32 v19, -8, v1 // (op 3898) 
v_sub_u32 v37, v1, v19 // (op 3898) 

v_lshlrev_b32 v19, 4, v37 // (op 3898) 

v_lshrrev_b32 v37, 3, v1 // (op 3898) 
v_lshl_add_u32 v41, v37, 7, v19 // (op 3898) 


s_nop 0
v_readfirstlane_b32 s0, v41 // (op 3898) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -64, v1 // (op 3950) 
v_sub_u32 v37, v1, v19 // (op 3950) 

v_lshlrev_b32 v19, 2, v37 // (op 3950) 

v_lshrrev_b32 v37, 6, v1 // (op 3950) 
v_lshl_add_u32 v41, v37, 8, v19 // (op 3950) 


v_and_b32 v19, -4, v1 // (op 4026) 
v_sub_u32 v37, v1, v19 // (op 4026) 

v_lshlrev_b32 v19, 6, v37 // (op 4026) 

v_lshrrev_b32 v37, 2, v1 // (op 4026) 
v_lshl_add_u32 v45, v37, 9, v19 // (op 4026) 


v_mul_lo_u32 v19, v45, 6 // (op 4026) 
v_lshrrev_b32 v37, 3, v19 // (op 4026) 

v_lshrrev_b32 v19, 4, v45 // (op 4026) 

v_lshlrev_b32 v45, 2, v19 // (op 4026) 

v_add_u32 v19, v37, v45 // (op 4026) 


v_and_b32 v37, -8, v1 // (op 4055) 
v_sub_u32 v45, v1, v37 // (op 4055) 

v_lshlrev_b32 v37, 4, v45 // (op 4055) 

v_lshrrev_b32 v45, 3, v1 // (op 4055) 
v_lshl_add_u32 v49, v45, 7, v37 // (op 4055) 


s_nop 0
v_readfirstlane_b32 s1, v49 // (op 4055) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 4094) 
v_sub_u32 v45, v1, v37 // (op 4094) 

v_lshlrev_b32 v37, 2, v45 // (op 4094) 

v_lshrrev_b32 v45, 6, v1 // (op 4094) 
v_lshl_add_u32 v49, v45, 8, v37 // (op 4094) 


s_mov_b32 s18, 0 // (op 1283) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1282) 
s_mov_b32 s19, s50 // (op 1282) convert

s_and_b32 s49, s19, 4294967294 // (op 1282) 

s_cmp_lt_u32 s18, s49 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1282 // (op 1282) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1282 if false)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1282:
s_add_u32 s18, s18, 2 // (op 1284) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4170) 



ds_read_b96_tr_b6 v[90:92], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v0 offset:12288 // (op 65) Transpose load local data 
s_barrier  // (op 4183) 



ds_read_b128 v[96:99], v3 // (op 70) Load local data 
ds_read_b128 v[100:103], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4196) 


ds_read_u8 v37, v7 offset:40960 // (op 75) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8224 // (op 340) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:103], a[0:3], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 82) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12320 // (op 340) Transpose load local data 




ds_read_u8 v37, v7 offset:40992 // (op 344) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8256 // (op 502) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:109], v[96:103], a[4:7], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 346) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12352 // (op 502) Transpose load local data 




ds_read_u8 v37, v7 offset:41024 // (op 506) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8288 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[96:103], a[8:11], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 508) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12384 // (op 664) Transpose load local data 




ds_read_u8 v37, v7 offset:41056 // (op 668) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8320 // (op 826) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:109], v[96:103], a[12:15], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 670) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12416 // (op 826) Transpose load local data 




ds_read_u8 v37, v7 offset:41088 // (op 830) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8352 // (op 988) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:103], a[16:19], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 832) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12448 // (op 988) Transpose load local data 




ds_read_u8 v37, v7 offset:41120 // (op 992) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8384 // (op 1150) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[96:103], a[20:23], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 994) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12480 // (op 1150) Transpose load local data 




ds_read_u8 v37, v7 offset:41152 // (op 1154) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8416 // (op 1312) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[96:103], a[24:27], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1156) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12512 // (op 1312) Transpose load local data 




ds_read_u8 v37, v7 offset:41184 // (op 1316) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[96:103], a[28:31], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1318) 





v_mov_b32 v37, v21 // (op 80) 

buffer_load_ubyte v22, v37, s[68:71], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 4186) 


v_mov_b64 v[14:15], v[16:17] // (op 3523) 

s_mov_b32 m0, s0 // (op 3523) reg expression
s_mov_b32 s19, 256 // (op 3523) call()
buffer_load_dwordx4 v14, s[52:55], s19 offen offset: 0 lds // (op 3523) Load value direct to lds


s_mul_i32 s51, s29, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 4) least significant
s_mul_hi_u32 s19, s28, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 4) 


v_mov_b32 v23, v13 // (op 4) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 4) 
s_mul_i32 s73, s51, 6 // (op 4) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 4) least significant
s_mul_hi_u32 s19, s50, 6 // (op 4) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 4) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 4) 

s_lshl_b32 s19, s50, 1 // (op 4) 

v_add_u32 v45, v23, s19 // (op 4) 

buffer_load_dwordx3 v[26:28], v45, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[30:32], v45, s[24:27], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[34:36], v45, s[24:27], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[38:40], v45, s[24:27], 0 offen offset:36 // (op 4) Load value
v_add_u32 v45, v45, v24 // (op 4) 
buffer_load_dwordx3 v[42:44], v45, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[46:48], v45, s[24:27], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[50:52], v45, s[24:27], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[54:56], v45, s[24:27], 0 offen offset:36 // (op 4) Load value




v_mov_b32 v45, v20 // (op 17) 
s_lshl_b32 s19, s38, 3 // (op 17) 
v_add_u32 v53, v45, s19 // (op 17) 


buffer_load_dword v33, v53, s[64:67], 0 offen // (op 17) Load value

s_barrier  // (op 4213) 



s_waitcnt vmcnt(19)
ds_write_b96 v19, v[58:60] offset:41984 // (op 225) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v19, v[62:64] offset:42000 // (op 225) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v19, v[66:68] offset:42016 // (op 225) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v19, v[70:72] offset:42032 // (op 225) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v19, v[74:76] offset:42240 // (op 225) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v19, v[78:80] offset:42256 // (op 225) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v19, v[82:84] offset:42272 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v19, v[86:88] offset:42288 // (op 225) Store local data 
s_barrier  // (op 4225) 



v_add_u32 v53, 74752, v49 // (op 236) 
s_waitcnt vmcnt(11)
ds_write_b32 v53, v18 // (op 236) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4210) 



ds_read_b96_tr_b6 v[90:92], v10 offset:41984 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v10 offset:46080 // (op 227) Transpose load local data 
s_barrier  // (op 4216) 



ds_read_b128 v[96:99], v4 offset:4096 // (op 229) Load local data 
ds_read_b128 v[100:103], v4 offset:4160 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4222) 


v_add_u32 v53, 74752, v11 // (op 231) 
ds_read_u8 v57, v53 // (op 231) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42016 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:103], a[0:3], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 234) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46112 // (op 353) Transpose load local data 




v_add_u32 v53, 74784, v11 // (op 357) 
ds_read_u8 v57, v53 // (op 357) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42048 // (op 515) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:109], v[96:103], a[4:7], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 347) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46144 // (op 515) Transpose load local data 




v_add_u32 v53, 74816, v11 // (op 519) 
ds_read_u8 v57, v53 // (op 519) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42080 // (op 677) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[96:103], a[8:11], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 509) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46176 // (op 677) Transpose load local data 




v_add_u32 v53, 74848, v11 // (op 681) 
ds_read_u8 v57, v53 // (op 681) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42112 // (op 839) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:109], v[96:103], a[12:15], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 671) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46208 // (op 839) Transpose load local data 




v_add_u32 v53, 74880, v11 // (op 843) 
ds_read_u8 v57, v53 // (op 843) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42144 // (op 1001) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:103], a[16:19], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 833) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46240 // (op 1001) Transpose load local data 




v_add_u32 v53, 74912, v11 // (op 1005) 
ds_read_u8 v57, v53 // (op 1005) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42176 // (op 1163) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[96:103], a[20:23], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 995) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46272 // (op 1163) Transpose load local data 




v_add_u32 v53, 74944, v11 // (op 1167) 
ds_read_u8 v57, v53 // (op 1167) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42208 // (op 1325) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[96:103], a[24:27], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1157) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46304 // (op 1325) Transpose load local data 




s_barrier  // (op 4322) 
v_add_u32 v53, 74976, v11 // (op 1329) 
ds_read_u8 v57, v53 // (op 1329) Load local data 

s_barrier  // (op 4326) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[96:103], a[28:31], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1319) 






buffer_load_ubyte v29, v37, s[68:71], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 4219) 
v_add_u32 v21, 8, v21 // (op 3794) 


s_add_u32 s19, 4096, s1 // (op 3535) 

s_mov_b32 m0, s19 // (op 3535) reg expression
s_mov_b32 s49, 384 // (op 3535) call()
buffer_load_dwordx4 v14, s[52:55], s49 offen offset: 0 lds // (op 3535) Load value direct to lds




v_add_co_u32 v16, vcc, 256, v16 // (op 3666) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3666) most significant half
s_mul_i32 s51, s29, 6 // (op 224) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 224) least significant
s_mul_hi_u32 s19, s28, 6 // (op 224) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 224) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[50:51] // (op 224) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 224) 
s_mul_i32 s73, s51, 6 // (op 224) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 224) least significant
s_mul_hi_u32 s19, s50, 6 // (op 224) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 224) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 224) 

s_mul_i32 s19, 3, s50 // (op 224) 

v_add_u32 v37, v23, s19 // (op 224) 

buffer_load_dwordx3 v[58:60], v37, s[24:27], 0 offen // (op 224) Load value
buffer_load_dwordx3 v[62:64], v37, s[24:27], 0 offen offset:12 // (op 224) Load value
buffer_load_dwordx3 v[66:68], v37, s[24:27], 0 offen offset:24 // (op 224) Load value
buffer_load_dwordx3 v[70:72], v37, s[24:27], 0 offen offset:36 // (op 224) Load value
v_add_u32 v37, v37, v14 // (op 224) 
buffer_load_dwordx3 v[74:76], v37, s[24:27], 0 offen // (op 224) Load value
buffer_load_dwordx3 v[78:80], v37, s[24:27], 0 offen offset:12 // (op 224) Load value
buffer_load_dwordx3 v[82:84], v37, s[24:27], 0 offen offset:24 // (op 224) Load value
buffer_load_dwordx3 v[86:88], v37, s[24:27], 0 offen offset:36 // (op 224) Load value





s_lshl_b32 s19, s38, 2 // (op 238) 
s_mul_i32 s49, 3, s19 // (op 238) 

v_add_u32 v37, v45, s49 // (op 238) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 3602) 
s_mul_i32 s73, s51, 6 // (op 3602) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 3602) least significant
s_mul_hi_u32 s19, s50, 6 // (op 3602) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 3602) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 3602) 

s_lshl_b32 s19, s50, 1 // (op 3602) 

v_add_u32 v13, v13, s19 // (op 3602) 

buffer_load_dword v18, v37, s[64:67], 0 offen // (op 238) Load value


s_barrier  // (op 4173) 
s_lshl_b32 s19, s38, 3 // (op 3730) 



v_add_u32 v20, v20, s19 // (op 3730) 

s_waitcnt vmcnt(19)
ds_write_b96 v12, v[26:28] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[30:32] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[34:36] offset:8224 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[38:40] offset:8240 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[42:44] offset:8448 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[46:48] offset:8464 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[50:52] offset:8480 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[54:56] offset:8496 // (op 6) Store local data 
s_barrier  // (op 4199) 



s_waitcnt vmcnt(11)
ds_write_b32 v41, v33 offset:40960 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1282) 
s_mov_b32 s19, s50 // (op 1282) convert

s_and_b32 s49, s19, 4294967294 // (op 1282) 

s_cmp_lt_u32 s18, s49 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1282 // (op 1282) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1282 if true)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1282:














s_barrier  // (op 4306) 
s_mul_hi_i32 s0, s16, s8 // (op 3558) 
v_and_b32 v61, -4, v1 // (op 3558) 
s_add_i32 s1, s0, s16 // (op 3558) 

v_sub_u32 v65, v1, v61 // (op 3558) 

s_ashr_i32 s0, s1, 31 // (op 3558) 
v_lshlrev_b32 v61, 6, v65 // (op 3558) 

s_cmp_eq_i32 s8, 0 // (op 3558) 
s_mov_b32 s18, scc // (op 3558) 
v_lshrrev_b32 v65, 2, v1 // (op 3558) 
s_cmp_eq_i32 s18, 0 // (op 3558) 
s_cselect_b32 s19, 0, -1 // (op 3558) 

v_lshlrev_b32 v69, 1, v65 // (op 3558) 

s_and_b32 s18, s9, 31 // (op 3558) 
s_lshl_b32 s49, 1, s18 // (op 3558) 
s_add_i32 s50, s49, s19 // (op 3558) 


s_and_b32 s19, s0, s50 // (op 3558) 


s_add_i32 s0, s1, s19 // (op 3558) 


s_ashr_i32 s1, s0, s18 // (op 3558) 


s_lshl_b32 s0, s9, 24 // (op 3558) 
s_ashr_i32 s18, s0, 31 // (op 3558) 

s_xor_b32 s0, s1, s18 // (op 3558) 

s_sub_i32 s1, s0, s18 // (op 3558) 


s_mul_hi_i32 s0, s1, s10 // (op 3558) 
s_add_i32 s18, s0, s1 // (op 3558) 

s_ashr_i32 s0, s18, 31 // (op 3558) 
s_cmp_eq_i32 s10, 0 // (op 3558) 
s_mov_b32 s19, scc // (op 3558) 
s_cmp_eq_i32 s19, 0 // (op 3558) 
s_cselect_b32 s49, 0, -1 // (op 3558) 

s_and_b32 s19, s11, 31 // (op 3558) 
s_lshl_b32 s50, 1, s19 // (op 3558) 
s_add_i32 s51, s50, s49 // (op 3558) 


s_and_b32 s49, s0, s51 // (op 3558) 


s_add_i32 s0, s18, s49 // (op 3558) 


s_ashr_i32 s18, s0, s19 // (op 3558) 


s_lshl_b32 s0, s11, 24 // (op 3558) 
s_ashr_i32 s19, s0, 31 // (op 3558) 

s_xor_b32 s0, s18, s19 // (op 3558) 

s_sub_i32 s18, s0, s19 // (op 3558) 


s_mul_i32 s0, s1, s13 // (op 3558) 

s_sub_u32 s1, s16, s0 // (op 3558) 

s_mul_i32 s0, s18, s13 // (op 3558) 
s_add_u32 s0, s0, s1 // (op 3558) 


s_mul_hi_i32 s1, s16, s7 // (op 3558) 
s_add_i32 s18, s1, s16 // (op 3558) 

s_ashr_i32 s1, s18, 31 // (op 3558) 
s_cmp_eq_i32 s7, 0 // (op 3558) 
s_mov_b32 s19, scc // (op 3558) 
s_cmp_eq_i32 s19, 0 // (op 3558) 
s_cselect_b32 s49, 0, -1 // (op 3558) 

s_and_b32 s19, s14, 31 // (op 3558) 
s_lshl_b32 s50, 1, s19 // (op 3558) 
s_add_i32 s51, s50, s49 // (op 3558) 


s_and_b32 s49, s1, s51 // (op 3558) 


s_add_i32 s1, s18, s49 // (op 3558) 


s_ashr_i32 s18, s1, s19 // (op 3558) 


s_lshl_b32 s1, s14, 24 // (op 3558) 
s_ashr_i32 s19, s1, 31 // (op 3558) 

s_xor_b32 s1, s18, s19 // (op 3558) 

s_sub_i32 s18, s1, s19 // (op 3558) 


s_add_u32 s50, s4, 32 // (op 3558) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3558) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3558) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3558) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3558) 

s_mul_i32 s1, s13, s50 // (op 3558) 

s_mul_i32 s19, s6, s1 // (op 3558) 

s_mul_i32 s1, s18, s19 // (op 3558) 

s_sub_u32 s19, s16, s1 // (op 3558) 

s_mov_b32 s1, s19 // (op 3558) 
s_mul_hi_i32 s49, s1, s12 // (op 3558) 
s_add_i32 s50, s49, s1 // (op 3558) 


s_ashr_i32 s1, s50, 31 // (op 3558) 
s_cmp_eq_i32 s12, 0 // (op 3558) 
s_mov_b32 s49, scc // (op 3558) 
s_cmp_eq_i32 s49, 0 // (op 3558) 
s_cselect_b32 s51, 0, -1 // (op 3558) 

s_and_b32 s49, s48, 31 // (op 3558) 
s_lshl_b32 s72, 1, s49 // (op 3558) 
s_add_i32 s73, s72, s51 // (op 3558) 


s_and_b32 s51, s1, s73 // (op 3558) 


s_add_i32 s1, s50, s51 // (op 3558) 


s_ashr_i32 s50, s1, s49 // (op 3558) 


s_lshl_b32 s1, s48, 24 // (op 3558) 
s_ashr_i32 s49, s1, 31 // (op 3558) 

s_xor_b32 s1, s50, s49 // (op 3558) 

s_sub_i32 s50, s1, s49 // (op 3558) 


s_mul_i32 s1, s15, s13 // (op 3558) 
s_add_u32 s72, s20, 256 // (op 3558) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3558) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3558) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3558) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3558) 

s_sub_i32 s49, s72, s1 // (op 3558) 

s_mul_i32 s51, s50, s49 // (op 3558) 


s_sub_u32 s49, s19, s51 // (op 3558) 


s_add_u32 s19, s49, s1 // (op 3558) 


s_cmp_eq_i32 s18, 0 // (op 3558) 
s_mov_b32 s1, scc // (op 3558) 

s_cmp_eq_i32 s1, 0 // (op 3558) 
s_cselect_b32 s18, s19, s0 // (op 3558) 



v_lshl_add_u32 v65, s18, 8, v61 // (op 3558) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3558) 
v_mov_b32 v13, 0 // (op 3558) convert
v_mov_b32 v12, v65 // (op 3558) convert

s_lshr_b32 s18, s0, 1 // (op 3558) 

v_lshl_add_u32 v61, s18, 8, v69 // (op 3558) 


v_mul_lo_u32 v15, v61, s29 // (op 3558) most significant: low of low * high
v_mul_hi_u32 v65, v61, s28 // (op 3558) most significant: high of low * low
v_mul_lo_u32 v14, v61, s28 // (op 3558) least significant: low of low * low
v_add_u32 v15, v15, v65 // (op 3558) most significant: sum


v_add_co_u32 v16, s[0:1], v12, v14 // (op 3558) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 3558) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 3558) most significant: low of high * low
v_mul_hi_u32 v61, v16, 6 // (op 3558) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 3558) least significant: low of low * low
v_add_u32 v13, v13, v61 // (op 3558) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3558) 

v_mov_b32 v61, v14 // (op 3558) convert

s_mul_i32 s1, s33, 6 // (op 3560) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3560) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3560) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3560) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3560) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3560) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3560) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3560) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3560) most significant: sum
s_add_u32 s19, s19, s50 // (op 3560) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3560) 

s_mov_b32 s24, s30 // (op 3560) reg expression
s_mov_b32 s25, s31 // (op 3560) reg expression
s_mov_b32 s26, s0 // (op 3560) reg expression
s_mov_b32 s27, 131072 // (op 3560) call()

s_mul_hi_i32 s0, s16, s8 // (op 3632) 
v_lshrrev_b32 v65, 3, v1 // (op 3632) 
s_add_i32 s1, s0, s16 // (op 3632) 

v_and_b32 v69, -8, v1 // (op 3632) 
s_ashr_i32 s0, s1, 31 // (op 3632) 
v_sub_u32 v73, v1, v69 // (op 3632) 

s_cmp_eq_i32 s8, 0 // (op 3632) 
s_mov_b32 s18, scc // (op 3632) 
v_lshlrev_b32 v69, 4, v73 // (op 3632) 

s_cmp_eq_i32 s18, 0 // (op 3632) 
s_cselect_b32 s19, 0, -1 // (op 3632) 

s_and_b32 s18, s9, 31 // (op 3632) 
s_lshl_b32 s49, 1, s18 // (op 3632) 
s_add_i32 s50, s49, s19 // (op 3632) 


s_and_b32 s19, s0, s50 // (op 3632) 


s_add_i32 s0, s1, s19 // (op 3632) 


s_ashr_i32 s1, s0, s18 // (op 3632) 


s_lshl_b32 s0, s9, 24 // (op 3632) 
s_ashr_i32 s18, s0, 31 // (op 3632) 

s_xor_b32 s0, s1, s18 // (op 3632) 

s_sub_i32 s1, s0, s18 // (op 3632) 


s_mul_hi_i32 s0, s1, s10 // (op 3632) 
s_add_i32 s18, s0, s1 // (op 3632) 

s_ashr_i32 s0, s18, 31 // (op 3632) 
s_cmp_eq_i32 s10, 0 // (op 3632) 
s_mov_b32 s19, scc // (op 3632) 
s_cmp_eq_i32 s19, 0 // (op 3632) 
s_cselect_b32 s49, 0, -1 // (op 3632) 

s_and_b32 s19, s11, 31 // (op 3632) 
s_lshl_b32 s50, 1, s19 // (op 3632) 
s_add_i32 s51, s50, s49 // (op 3632) 


s_and_b32 s49, s0, s51 // (op 3632) 


s_add_i32 s0, s18, s49 // (op 3632) 


s_ashr_i32 s18, s0, s19 // (op 3632) 


s_lshl_b32 s0, s11, 24 // (op 3632) 
s_ashr_i32 s19, s0, 31 // (op 3632) 

s_xor_b32 s0, s18, s19 // (op 3632) 

s_sub_i32 s18, s0, s19 // (op 3632) 


s_add_u32 s50, s4, 32 // (op 3632) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3632) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3632) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3632) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3632) 

s_mul_i32 s0, s18, s50 // (op 3632) 

s_sub_i32 s49, s1, s0 // (op 3632) 


s_mul_hi_i32 s0, s16, s7 // (op 3632) 
s_add_i32 s1, s0, s16 // (op 3632) 

s_ashr_i32 s0, s1, 31 // (op 3632) 
s_cmp_eq_i32 s7, 0 // (op 3632) 
s_mov_b32 s18, scc // (op 3632) 
s_cmp_eq_i32 s18, 0 // (op 3632) 
s_cselect_b32 s19, 0, -1 // (op 3632) 

s_and_b32 s18, s14, 31 // (op 3632) 
s_lshl_b32 s72, 1, s18 // (op 3632) 
s_add_i32 s73, s72, s19 // (op 3632) 


s_and_b32 s19, s0, s73 // (op 3632) 


s_add_i32 s0, s1, s19 // (op 3632) 


s_ashr_i32 s1, s0, s18 // (op 3632) 


s_lshl_b32 s0, s14, 24 // (op 3632) 
s_ashr_i32 s18, s0, 31 // (op 3632) 

s_xor_b32 s0, s1, s18 // (op 3632) 

s_sub_i32 s1, s0, s18 // (op 3632) 


s_mul_i32 s0, s13, s50 // (op 3632) 

s_mul_i32 s18, s6, s0 // (op 3632) 

s_mul_i32 s0, s1, s18 // (op 3632) 

s_sub_u32 s18, s16, s0 // (op 3632) 

s_mul_hi_i32 s0, s18, s12 // (op 3632) 
s_add_i32 s19, s0, s18 // (op 3632) 


s_ashr_i32 s0, s19, 31 // (op 3632) 
s_cmp_eq_i32 s12, 0 // (op 3632) 
s_mov_b32 s18, scc // (op 3632) 
s_cmp_eq_i32 s18, 0 // (op 3632) 
s_cselect_b32 s50, 0, -1 // (op 3632) 

s_and_b32 s18, s48, 31 // (op 3632) 
s_lshl_b32 s51, 1, s18 // (op 3632) 
s_add_i32 s72, s51, s50 // (op 3632) 


s_and_b32 s50, s0, s72 // (op 3632) 


s_add_i32 s0, s19, s50 // (op 3632) 


s_ashr_i32 s19, s0, s18 // (op 3632) 


s_lshl_b32 s0, s48, 24 // (op 3632) 
s_ashr_i32 s18, s0, 31 // (op 3632) 

s_xor_b32 s0, s19, s18 // (op 3632) 

s_sub_i32 s19, s0, s18 // (op 3632) 


s_cmp_eq_i32 s1, 0 // (op 3632) 
s_mov_b32 s0, scc // (op 3632) 

s_cmp_eq_i32 s0, 0 // (op 3632) 
s_cselect_b32 s1, s19, s49 // (op 3632) 



v_lshl_add_u32 v73, s1, 5, v65 // (op 3632) 


v_mul_lo_u32 v13, v73, s35 // (op 3632) most significant: low of low * high
v_mul_hi_u32 v65, v73, s34 // (op 3632) most significant: high of low * low
v_mul_lo_u32 v12, v73, s34 // (op 3632) least significant: low of low * low
v_add_u32 v13, v13, v65 // (op 3632) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3632) 
s_lshr_b32 s18, s0, 1 // (op 3632) 

v_lshl_add_u32 v65, s18, 8, v69 // (op 3632) 


v_mov_b32 v15, 0 // (op 3632) convert
v_mov_b32 v14, v65 // (op 3632) convert

v_add_co_u32 v16, s[0:1], v14, v12 // (op 3632) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 3632) most significant half



s_mul_i32 s1, s34, s5 // (op 3634) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3634) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3634) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3634) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3634) most significant: sum
s_add_u32 s1, s1, s19 // (op 3634) most significant: sum


s_mov_b32 s52, s36 // (op 3634) reg expression
s_mov_b32 s53, s37 // (op 3634) reg expression
s_mov_b32 s54, s0 // (op 3634) reg expression
s_mov_b32 s55, 131072 // (op 3634) call()

s_mul_hi_i32 s0, s16, s8 // (op 3696) 
v_and_b32 v65, -64, v1 // (op 3696) 
s_add_i32 s1, s0, s16 // (op 3696) 

v_sub_u32 v69, v1, v65 // (op 3696) 

s_ashr_i32 s0, s1, 31 // (op 3696) 
v_lshlrev_b32 v65, 2, v69 // (op 3696) 

s_cmp_eq_i32 s8, 0 // (op 3696) 
s_mov_b32 s18, scc // (op 3696) 
v_lshrrev_b32 v69, 6, v1 // (op 3696) 
s_cmp_eq_i32 s18, 0 // (op 3696) 
s_cselect_b32 s19, 0, -1 // (op 3696) 

s_and_b32 s18, s9, 31 // (op 3696) 
s_lshl_b32 s49, 1, s18 // (op 3696) 
s_add_i32 s50, s49, s19 // (op 3696) 


s_and_b32 s19, s0, s50 // (op 3696) 


s_add_i32 s0, s1, s19 // (op 3696) 


s_ashr_i32 s1, s0, s18 // (op 3696) 


s_lshl_b32 s0, s9, 24 // (op 3696) 
s_ashr_i32 s18, s0, 31 // (op 3696) 

s_xor_b32 s0, s1, s18 // (op 3696) 

s_sub_i32 s1, s0, s18 // (op 3696) 


s_mul_hi_i32 s0, s1, s10 // (op 3696) 
s_add_i32 s18, s0, s1 // (op 3696) 

s_ashr_i32 s0, s18, 31 // (op 3696) 
s_cmp_eq_i32 s10, 0 // (op 3696) 
s_mov_b32 s19, scc // (op 3696) 
s_cmp_eq_i32 s19, 0 // (op 3696) 
s_cselect_b32 s49, 0, -1 // (op 3696) 

s_and_b32 s19, s11, 31 // (op 3696) 
s_lshl_b32 s50, 1, s19 // (op 3696) 
s_add_i32 s51, s50, s49 // (op 3696) 


s_and_b32 s49, s0, s51 // (op 3696) 


s_add_i32 s0, s18, s49 // (op 3696) 


s_ashr_i32 s18, s0, s19 // (op 3696) 


s_lshl_b32 s0, s11, 24 // (op 3696) 
s_ashr_i32 s19, s0, 31 // (op 3696) 

s_xor_b32 s0, s18, s19 // (op 3696) 

s_sub_i32 s18, s0, s19 // (op 3696) 


s_mul_i32 s0, s1, s13 // (op 3696) 

s_sub_u32 s1, s16, s0 // (op 3696) 

s_mul_i32 s0, s18, s13 // (op 3696) 
s_add_u32 s0, s0, s1 // (op 3696) 


s_mul_hi_i32 s1, s16, s7 // (op 3696) 
s_add_i32 s18, s1, s16 // (op 3696) 

s_ashr_i32 s1, s18, 31 // (op 3696) 
s_cmp_eq_i32 s7, 0 // (op 3696) 
s_mov_b32 s19, scc // (op 3696) 
s_cmp_eq_i32 s19, 0 // (op 3696) 
s_cselect_b32 s49, 0, -1 // (op 3696) 

s_and_b32 s19, s14, 31 // (op 3696) 
s_lshl_b32 s50, 1, s19 // (op 3696) 
s_add_i32 s51, s50, s49 // (op 3696) 


s_and_b32 s49, s1, s51 // (op 3696) 


s_add_i32 s1, s18, s49 // (op 3696) 


s_ashr_i32 s18, s1, s19 // (op 3696) 


s_lshl_b32 s1, s14, 24 // (op 3696) 
s_ashr_i32 s19, s1, 31 // (op 3696) 

s_xor_b32 s1, s18, s19 // (op 3696) 

s_sub_i32 s18, s1, s19 // (op 3696) 


s_add_u32 s50, s4, 32 // (op 3696) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3696) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3696) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3696) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3696) 

s_mul_i32 s1, s13, s50 // (op 3696) 

s_mul_i32 s19, s6, s1 // (op 3696) 

s_mul_i32 s1, s18, s19 // (op 3696) 

s_sub_u32 s19, s16, s1 // (op 3696) 

s_mov_b32 s1, s19 // (op 3696) 
s_mul_hi_i32 s49, s1, s12 // (op 3696) 
s_add_i32 s50, s49, s1 // (op 3696) 


s_ashr_i32 s1, s50, 31 // (op 3696) 
s_cmp_eq_i32 s12, 0 // (op 3696) 
s_mov_b32 s49, scc // (op 3696) 
s_cmp_eq_i32 s49, 0 // (op 3696) 
s_cselect_b32 s51, 0, -1 // (op 3696) 

s_and_b32 s49, s48, 31 // (op 3696) 
s_lshl_b32 s72, 1, s49 // (op 3696) 
s_add_i32 s73, s72, s51 // (op 3696) 


s_and_b32 s51, s1, s73 // (op 3696) 


s_add_i32 s1, s50, s51 // (op 3696) 


s_ashr_i32 s50, s1, s49 // (op 3696) 


s_lshl_b32 s1, s48, 24 // (op 3696) 
s_ashr_i32 s49, s1, 31 // (op 3696) 

s_xor_b32 s1, s50, s49 // (op 3696) 

s_sub_i32 s50, s1, s49 // (op 3696) 


s_mul_i32 s1, s15, s13 // (op 3696) 
s_add_u32 s72, s20, 256 // (op 3696) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3696) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3696) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3696) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3696) 

s_sub_i32 s49, s72, s1 // (op 3696) 

s_mul_i32 s51, s50, s49 // (op 3696) 


s_sub_u32 s49, s19, s51 // (op 3696) 


s_add_u32 s19, s49, s1 // (op 3696) 


s_cmp_eq_i32 s18, 0 // (op 3696) 
s_mov_b32 s1, scc // (op 3696) 

s_cmp_eq_i32 s1, 0 // (op 3696) 
s_cselect_b32 s18, s19, s0 // (op 3696) 



v_lshl_add_u32 v73, s18, 8, v65 // (op 3696) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3696) 
s_lshr_b32 s18, s0, 1 // (op 3696) 

v_lshl_add_u32 v65, s18, 3, v69 // (op 3696) 


v_mul_lo_u32 v69, v65, s38 // (op 3696) 

v_add_u32 v65, v73, v69 // (op 3696) 


s_mul_i32 s1, s38, s43 // (op 3698) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3698) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3698) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3698) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3698) most significant: sum
s_add_u32 s1, s1, s19 // (op 3698) most significant: sum


s_mov_b32 s64, s40 // (op 3698) reg expression
s_mov_b32 s65, s41 // (op 3698) reg expression
s_mov_b32 s66, s0 // (op 3698) reg expression
s_mov_b32 s67, 131072 // (op 3698) call()

s_mul_hi_i32 s0, s16, s8 // (op 3760) 
v_lshrrev_b32 v69, 6, v1 // (op 3760) 
s_add_i32 s1, s0, s16 // (op 3760) 

v_and_b32 v73, 1, v69 // (op 3760) 

s_ashr_i32 s0, s1, 31 // (op 3760) 
v_and_b32 v69, 15, v1 // (op 3760) 
s_cmp_eq_i32 s8, 0 // (op 3760) 
s_mov_b32 s18, scc // (op 3760) 
v_lshl_add_u32 v77, v73, 4, v69 // (op 3760) 


s_cmp_eq_i32 s18, 0 // (op 3760) 
s_cselect_b32 s19, 0, -1 // (op 3760) 

v_and_b32 v69, 63, v1 // (op 3760) 
s_and_b32 s18, s9, 31 // (op 3760) 
v_lshrrev_b32 v73, 4, v69 // (op 3760) 

s_lshl_b32 s49, 1, s18 // (op 3760) 
s_add_i32 s50, s49, s19 // (op 3760) 


s_and_b32 s19, s0, s50 // (op 3760) 


s_add_i32 s0, s1, s19 // (op 3760) 


s_ashr_i32 s1, s0, s18 // (op 3760) 


s_lshl_b32 s0, s9, 24 // (op 3760) 
s_ashr_i32 s18, s0, 31 // (op 3760) 

s_xor_b32 s0, s1, s18 // (op 3760) 

s_sub_i32 s1, s0, s18 // (op 3760) 


s_mul_hi_i32 s0, s1, s10 // (op 3760) 
s_add_i32 s18, s0, s1 // (op 3760) 

s_ashr_i32 s0, s18, 31 // (op 3760) 
s_cmp_eq_i32 s10, 0 // (op 3760) 
s_mov_b32 s19, scc // (op 3760) 
s_cmp_eq_i32 s19, 0 // (op 3760) 
s_cselect_b32 s49, 0, -1 // (op 3760) 

s_and_b32 s19, s11, 31 // (op 3760) 
s_lshl_b32 s50, 1, s19 // (op 3760) 
s_add_i32 s51, s50, s49 // (op 3760) 


s_and_b32 s49, s0, s51 // (op 3760) 


s_add_i32 s0, s18, s49 // (op 3760) 


s_ashr_i32 s18, s0, s19 // (op 3760) 


s_lshl_b32 s0, s11, 24 // (op 3760) 
s_ashr_i32 s19, s0, 31 // (op 3760) 

s_xor_b32 s0, s18, s19 // (op 3760) 

s_sub_i32 s18, s0, s19 // (op 3760) 


s_add_u32 s50, s4, 32 // (op 3760) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3760) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3760) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3760) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3760) 

s_mul_i32 s0, s18, s50 // (op 3760) 

s_sub_i32 s49, s1, s0 // (op 3760) 


s_mul_hi_i32 s0, s16, s7 // (op 3760) 
s_add_i32 s1, s0, s16 // (op 3760) 

s_ashr_i32 s0, s1, 31 // (op 3760) 
s_cmp_eq_i32 s7, 0 // (op 3760) 
s_mov_b32 s18, scc // (op 3760) 
s_cmp_eq_i32 s18, 0 // (op 3760) 
s_cselect_b32 s19, 0, -1 // (op 3760) 

s_and_b32 s18, s14, 31 // (op 3760) 
s_lshl_b32 s72, 1, s18 // (op 3760) 
s_add_i32 s73, s72, s19 // (op 3760) 


s_and_b32 s19, s0, s73 // (op 3760) 


s_add_i32 s0, s1, s19 // (op 3760) 


s_ashr_i32 s1, s0, s18 // (op 3760) 


s_lshl_b32 s0, s14, 24 // (op 3760) 
s_ashr_i32 s18, s0, 31 // (op 3760) 

s_xor_b32 s0, s1, s18 // (op 3760) 

s_sub_i32 s1, s0, s18 // (op 3760) 


s_mul_i32 s0, s13, s50 // (op 3760) 

s_mul_i32 s18, s6, s0 // (op 3760) 

s_mul_i32 s0, s1, s18 // (op 3760) 

s_sub_u32 s18, s16, s0 // (op 3760) 

s_mul_hi_i32 s0, s18, s12 // (op 3760) 
s_add_i32 s19, s0, s18 // (op 3760) 


s_ashr_i32 s0, s19, 31 // (op 3760) 
s_cmp_eq_i32 s12, 0 // (op 3760) 
s_mov_b32 s18, scc // (op 3760) 
s_cmp_eq_i32 s18, 0 // (op 3760) 
s_cselect_b32 s50, 0, -1 // (op 3760) 

s_and_b32 s18, s48, 31 // (op 3760) 
s_lshl_b32 s51, 1, s18 // (op 3760) 
s_add_i32 s72, s51, s50 // (op 3760) 


s_and_b32 s50, s0, s72 // (op 3760) 


s_add_i32 s0, s19, s50 // (op 3760) 


s_ashr_i32 s19, s0, s18 // (op 3760) 


s_lshl_b32 s0, s48, 24 // (op 3760) 
s_ashr_i32 s18, s0, 31 // (op 3760) 

s_xor_b32 s0, s19, s18 // (op 3760) 

s_sub_i32 s19, s0, s18 // (op 3760) 


s_cmp_eq_i32 s1, 0 // (op 3760) 
s_mov_b32 s0, scc // (op 3760) 

s_cmp_eq_i32 s0, 0 // (op 3760) 
s_cselect_b32 s1, s19, s49 // (op 3760) 



v_lshl_add_u32 v69, s1, 5, v77 // (op 3760) 


v_mul_lo_u32 v77, v69, s44 // (op 3760) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 3760) 
s_lshr_b32 s18, s0, 1 // (op 3760) 

v_lshl_add_u32 v69, s18, 3, v73 // (op 3760) 


v_add_u32 v73, v69, v77 // (op 3760) 


s_mul_i32 s1, s44, s5 // (op 3762) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3762) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3762) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3762) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3762) most significant: sum
s_add_u32 s1, s1, s19 // (op 3762) most significant: sum


s_mov_b32 s68, s46 // (op 3762) reg expression
s_mov_b32 s69, s47 // (op 3762) reg expression
s_mov_b32 s70, s0 // (op 3762) reg expression
s_mov_b32 s71, 131072 // (op 3762) call()

v_and_b32 v69, -4, v1 // (op 3853) 
v_sub_u32 v77, v1, v69 // (op 3853) 

v_lshlrev_b32 v69, 6, v77 // (op 3853) 

v_lshrrev_b32 v77, 2, v1 // (op 3853) 
v_lshl_add_u32 v81, v77, 9, v69 // (op 3853) 


v_mul_lo_u32 v69, v81, 6 // (op 3853) 
v_lshrrev_b32 v77, 3, v69 // (op 3853) 

v_lshrrev_b32 v69, 4, v81 // (op 3853) 

v_lshlrev_b32 v81, 2, v69 // (op 3853) 

v_add_u32 v69, v77, v81 // (op 3853) 


v_and_b32 v77, -8, v1 // (op 3911) 
v_sub_u32 v81, v1, v77 // (op 3911) 

v_lshlrev_b32 v77, 4, v81 // (op 3911) 

v_lshrrev_b32 v81, 3, v1 // (op 3911) 
v_lshl_add_u32 v85, v81, 7, v77 // (op 3911) 


s_nop 0
v_readfirstlane_b32 s0, v85 // (op 3911) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v77, -64, v1 // (op 3963) 
v_sub_u32 v81, v1, v77 // (op 3963) 

v_lshlrev_b32 v77, 2, v81 // (op 3963) 

v_lshrrev_b32 v81, 6, v1 // (op 3963) 
v_lshl_add_u32 v85, v81, 8, v77 // (op 3963) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 1288) 
s_and_b32 s1, s18, 4294967294 // (op 1288) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 1287) 
s_mov_b32 s49, s18 // (op 1287) convert

s_cmp_lt_u32 s1, s49 // (op 1287) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 // (op 1287) Condition: Top (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1287:

s_add_u32 s1, s1, 1 // (op 1289) 
s_mul_i32 s19, s29, 6 // (op 166) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 166) least significant
s_mul_hi_u32 s49, s28, 6 // (op 166) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 166) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[18:19] // (op 166) 


v_mov_b32 v14, v61 // (op 166) 
v_mov_b32 v15, v14 // (op 166) reg expression
buffer_load_dwordx3 v[18:20], v15, s[24:27], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[22:24], v15, s[24:27], 0 offen offset:12 // (op 166) Load value
buffer_load_dwordx3 v[26:28], v15, s[24:27], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx3 v[30:32], v15, s[24:27], 0 offen offset:36 // (op 166) Load value
v_add_u32 v15, v15, v12 // (op 166) 
buffer_load_dwordx3 v[34:36], v15, s[24:27], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[38:40], v15, s[24:27], 0 offen offset:12 // (op 166) Load value
buffer_load_dwordx3 v[42:44], v15, s[24:27], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx3 v[46:48], v15, s[24:27], 0 offen offset:36 // (op 166) Load value



s_lshl_b64 s[18:19], s[28:29], 7 // (op 3548) 
s_mul_i32 s51, s19, 6 // (op 3548) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 3548) least significant
s_mul_hi_u32 s49, s18, 6 // (op 3548) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 3548) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 3548) 

v_add_u32 v61, v61, s18 // (op 3548) 

s_barrier  // (op 4176) 
s_barrier  // (op 4189) 





v_mov_b64 v[12:13], v[16:17] // (op 3528) 

s_mov_b32 m0, s0 // (op 3528) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 3528) Load value direct to lds

v_add_co_u32 v16, vcc, 128, v16 // (op 3624) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3624) most significant half
s_waitcnt vmcnt(8)
ds_write_b96 v69, v[18:20] offset:8192 // (op 167) Store local data 


v_mov_b32 v21, v65 // (op 180) 
v_mov_b32 v25, v21 // (op 180) reg expression

s_waitcnt vmcnt(7)
ds_write_b96 v69, v[22:24] offset:8208 // (op 167) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v69, v[26:28] offset:8224 // (op 167) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v69, v[30:32] offset:8240 // (op 167) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v69, v[34:36] offset:8448 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v69, v[38:40] offset:8464 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v69, v[42:44] offset:8480 // (op 167) Store local data 
buffer_load_dword v29, v25, s[64:67], 0 offen // (op 180) Load value


s_lshl_b32 s18, s38, 2 // (op 3688) 
v_add_u32 v65, v65, s18 // (op 3688) 

s_waitcnt vmcnt(2)
ds_write_b96 v69, v[46:48] offset:8496 // (op 167) Store local data 

s_barrier  // (op 4202) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4179) 






ds_read_b96_tr_b6 v[18:20], v0 offset:8192 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:12288 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v85, v29 offset:40960 // (op 178) Store local data 

s_barrier  // (op 4192) 



ds_read_b128 v[24:27], v3 // (op 171) Load local data 
ds_read_b128 v[28:31], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4206) 


ds_read_u8 v12, v7 offset:40960 // (op 173) Load local data 


v_mov_b32 v13, v73 // (op 175) 

buffer_load_ubyte v14, v13, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v73, 4, v73 // (op 3752) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:31], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 176) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8224 // (op 321) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12320 // (op 321) Transpose load local data 


ds_read_u8 v15, v7 offset:40992 // (op 325) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[24:31], a[4:7], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 327) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8256 // (op 483) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12352 // (op 483) Transpose load local data 


ds_read_u8 v15, v7 offset:41024 // (op 487) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[24:31], a[8:11], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 489) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8288 // (op 645) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12384 // (op 645) Transpose load local data 


ds_read_u8 v15, v7 offset:41056 // (op 649) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[24:31], a[12:15], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 651) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8320 // (op 807) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12416 // (op 807) Transpose load local data 


ds_read_u8 v15, v7 offset:41088 // (op 811) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[18:23], v[24:31], a[16:19], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 813) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8352 // (op 969) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12448 // (op 969) Transpose load local data 


ds_read_u8 v15, v7 offset:41120 // (op 973) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[24:31], a[20:23], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 975) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8384 // (op 1131) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12480 // (op 1131) Transpose load local data 


ds_read_u8 v15, v7 offset:41152 // (op 1135) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[18:23], v[24:31], a[24:27], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1137) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8416 // (op 1293) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12512 // (op 1293) Transpose load local data 


ds_read_u8 v15, v7 offset:41184 // (op 1297) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[24:31], a[28:31], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1299) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 1287) 
s_mov_b32 s49, s18 // (op 1287) convert

s_cmp_lt_u32 s1, s49 // (op 1287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 // (op 1287) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 if true)
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287:






















s_barrier  // (op 4310) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 1617) 
v_accvgpr_read v11, a5 // (op 1617) 
v_accvgpr_read v12, a6 // (op 1617) 
v_accvgpr_read v13, a7 // (op 1617) 
v_mul_f32 v14, v8, v10 // (op 1617) 
v_mul_f32 v15, v8, v11 // (op 1617) 
v_mul_f32 v16, v8, v12 // (op 1617) 
v_mul_f32 v17, v8, v13 // (op 1617) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:128 // (op 1619) Store value

v_accvgpr_read v10, a8 // (op 1635) 
v_accvgpr_read v11, a9 // (op 1635) 
v_accvgpr_read v12, a10 // (op 1635) 
v_accvgpr_read v13, a11 // (op 1635) 
v_mul_f32 v14, v8, v10 // (op 1635) 
v_mul_f32 v15, v8, v11 // (op 1635) 
v_mul_f32 v16, v8, v12 // (op 1635) 
v_mul_f32 v17, v8, v13 // (op 1635) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:256 // (op 1637) Store value

v_accvgpr_read v10, a12 // (op 1653) 
v_accvgpr_read v11, a13 // (op 1653) 
v_accvgpr_read v12, a14 // (op 1653) 
v_accvgpr_read v13, a15 // (op 1653) 
v_mul_f32 v14, v8, v10 // (op 1653) 
v_mul_f32 v15, v8, v11 // (op 1653) 
v_mul_f32 v16, v8, v12 // (op 1653) 
v_mul_f32 v17, v8, v13 // (op 1653) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:384 // (op 1655) Store value

v_accvgpr_read v10, a16 // (op 1671) 
v_accvgpr_read v11, a17 // (op 1671) 
v_accvgpr_read v12, a18 // (op 1671) 
v_accvgpr_read v13, a19 // (op 1671) 
v_mul_f32 v14, v8, v10 // (op 1671) 
v_mul_f32 v15, v8, v11 // (op 1671) 
v_mul_f32 v16, v8, v12 // (op 1671) 
v_mul_f32 v17, v8, v13 // (op 1671) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:512 // (op 1673) Store value

v_accvgpr_read v10, a20 // (op 1689) 
v_accvgpr_read v11, a21 // (op 1689) 
v_accvgpr_read v12, a22 // (op 1689) 
v_accvgpr_read v13, a23 // (op 1689) 
v_mul_f32 v14, v8, v10 // (op 1689) 
v_mul_f32 v15, v8, v11 // (op 1689) 
v_mul_f32 v16, v8, v12 // (op 1689) 
v_mul_f32 v17, v8, v13 // (op 1689) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:640 // (op 1691) Store value

v_accvgpr_read v10, a24 // (op 1707) 
v_accvgpr_read v11, a25 // (op 1707) 
v_accvgpr_read v12, a26 // (op 1707) 
v_accvgpr_read v13, a27 // (op 1707) 
v_mul_f32 v14, v8, v10 // (op 1707) 
v_mul_f32 v15, v8, v11 // (op 1707) 
v_mul_f32 v16, v8, v12 // (op 1707) 
v_mul_f32 v17, v8, v13 // (op 1707) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:768 // (op 1709) Store value

v_accvgpr_read v10, a28 // (op 1725) 
v_accvgpr_read v11, a29 // (op 1725) 
v_accvgpr_read v12, a30 // (op 1725) 
v_accvgpr_read v13, a31 // (op 1725) 
v_mul_f32 v14, v8, v10 // (op 1725) 
v_mul_f32 v15, v8, v11 // (op 1725) 
v_mul_f32 v16, v8, v12 // (op 1725) 
v_mul_f32 v17, v8, v13 // (op 1725) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:896 // (op 1727) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:896 // (op 1893) Load value
v_accvgpr_read v14, a28 // (op 1890) 
v_accvgpr_read v15, a29 // (op 1890) 
v_accvgpr_read v16, a30 // (op 1890) 
v_accvgpr_read v17, a31 // (op 1890) 
v_mul_f32 v18, v8, v14 // (op 1890) 
v_mul_f32 v19, v8, v15 // (op 1890) 
v_mul_f32 v20, v8, v16 // (op 1890) 
v_mul_f32 v21, v8, v17 // (op 1890) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1894) 
v_mul_f32 v15, v5, v11 // (op 1894) 
v_mul_f32 v16, v5, v12 // (op 1894) 
v_mul_f32 v17, v5, v13 // (op 1894) 

v_add_f32 v10, v14, v18 // (op 1891) 
v_add_f32 v11, v15, v19 // (op 1891) 
v_add_f32 v12, v16, v20 // (op 1891) 
v_add_f32 v13, v17, v21 // (op 1891) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:896 // (op 1892) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:768 // (op 1875) Load value
v_accvgpr_read v14, a24 // (op 1872) 
v_accvgpr_read v15, a25 // (op 1872) 
v_accvgpr_read v16, a26 // (op 1872) 
v_accvgpr_read v17, a27 // (op 1872) 
v_mul_f32 v18, v8, v14 // (op 1872) 
v_mul_f32 v19, v8, v15 // (op 1872) 
v_mul_f32 v20, v8, v16 // (op 1872) 
v_mul_f32 v21, v8, v17 // (op 1872) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1876) 
v_mul_f32 v15, v5, v11 // (op 1876) 
v_mul_f32 v16, v5, v12 // (op 1876) 
v_mul_f32 v17, v5, v13 // (op 1876) 

v_add_f32 v10, v14, v18 // (op 1873) 
v_add_f32 v11, v15, v19 // (op 1873) 
v_add_f32 v12, v16, v20 // (op 1873) 
v_add_f32 v13, v17, v21 // (op 1873) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:768 // (op 1874) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:640 // (op 1857) Load value
v_accvgpr_read v14, a20 // (op 1854) 
v_accvgpr_read v15, a21 // (op 1854) 
v_accvgpr_read v16, a22 // (op 1854) 
v_accvgpr_read v17, a23 // (op 1854) 
v_mul_f32 v18, v8, v14 // (op 1854) 
v_mul_f32 v19, v8, v15 // (op 1854) 
v_mul_f32 v20, v8, v16 // (op 1854) 
v_mul_f32 v21, v8, v17 // (op 1854) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1858) 
v_mul_f32 v15, v5, v11 // (op 1858) 
v_mul_f32 v16, v5, v12 // (op 1858) 
v_mul_f32 v17, v5, v13 // (op 1858) 

v_add_f32 v10, v14, v18 // (op 1855) 
v_add_f32 v11, v15, v19 // (op 1855) 
v_add_f32 v12, v16, v20 // (op 1855) 
v_add_f32 v13, v17, v21 // (op 1855) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:640 // (op 1856) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:512 // (op 1839) Load value
v_accvgpr_read v14, a16 // (op 1836) 
v_accvgpr_read v15, a17 // (op 1836) 
v_accvgpr_read v16, a18 // (op 1836) 
v_accvgpr_read v17, a19 // (op 1836) 
v_mul_f32 v18, v8, v14 // (op 1836) 
v_mul_f32 v19, v8, v15 // (op 1836) 
v_mul_f32 v20, v8, v16 // (op 1836) 
v_mul_f32 v21, v8, v17 // (op 1836) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1840) 
v_mul_f32 v15, v5, v11 // (op 1840) 
v_mul_f32 v16, v5, v12 // (op 1840) 
v_mul_f32 v17, v5, v13 // (op 1840) 

v_add_f32 v10, v14, v18 // (op 1837) 
v_add_f32 v11, v15, v19 // (op 1837) 
v_add_f32 v12, v16, v20 // (op 1837) 
v_add_f32 v13, v17, v21 // (op 1837) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:512 // (op 1838) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:384 // (op 1821) Load value
v_accvgpr_read v14, a12 // (op 1818) 
v_accvgpr_read v15, a13 // (op 1818) 
v_accvgpr_read v16, a14 // (op 1818) 
v_accvgpr_read v17, a15 // (op 1818) 
v_mul_f32 v18, v8, v14 // (op 1818) 
v_mul_f32 v19, v8, v15 // (op 1818) 
v_mul_f32 v20, v8, v16 // (op 1818) 
v_mul_f32 v21, v8, v17 // (op 1818) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1822) 
v_mul_f32 v15, v5, v11 // (op 1822) 
v_mul_f32 v16, v5, v12 // (op 1822) 
v_mul_f32 v17, v5, v13 // (op 1822) 

v_add_f32 v10, v14, v18 // (op 1819) 
v_add_f32 v11, v15, v19 // (op 1819) 
v_add_f32 v12, v16, v20 // (op 1819) 
v_add_f32 v13, v17, v21 // (op 1819) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:384 // (op 1820) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:256 // (op 1803) Load value
v_accvgpr_read v14, a8 // (op 1800) 
v_accvgpr_read v15, a9 // (op 1800) 
v_accvgpr_read v16, a10 // (op 1800) 
v_accvgpr_read v17, a11 // (op 1800) 
v_mul_f32 v18, v8, v14 // (op 1800) 
v_mul_f32 v19, v8, v15 // (op 1800) 
v_mul_f32 v20, v8, v16 // (op 1800) 
v_mul_f32 v21, v8, v17 // (op 1800) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1804) 
v_mul_f32 v15, v5, v11 // (op 1804) 
v_mul_f32 v16, v5, v12 // (op 1804) 
v_mul_f32 v17, v5, v13 // (op 1804) 

v_add_f32 v10, v14, v18 // (op 1801) 
v_add_f32 v11, v15, v19 // (op 1801) 
v_add_f32 v12, v16, v20 // (op 1801) 
v_add_f32 v13, v17, v21 // (op 1801) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:256 // (op 1802) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 1785) Load value
v_accvgpr_read v14, a4 // (op 1782) 
v_accvgpr_read v15, a5 // (op 1782) 
v_accvgpr_read v16, a6 // (op 1782) 
v_accvgpr_read v17, a7 // (op 1782) 
v_mul_f32 v18, v8, v14 // (op 1782) 
v_mul_f32 v19, v8, v15 // (op 1782) 
v_mul_f32 v20, v8, v16 // (op 1782) 
v_mul_f32 v21, v8, v17 // (op 1782) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1786) 
v_mul_f32 v15, v5, v11 // (op 1786) 
v_mul_f32 v16, v5, v12 // (op 1786) 
v_mul_f32 v17, v5, v13 // (op 1786) 

v_add_f32 v10, v14, v18 // (op 1783) 
v_add_f32 v11, v15, v19 // (op 1783) 
v_add_f32 v12, v16, v20 // (op 1783) 
v_add_f32 v13, v17, v21 // (op 1783) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:128 // (op 1784) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.LRR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_, .LRR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end-RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F8
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
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int64, semantic: SizeFree0 },
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
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_.kd
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
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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

