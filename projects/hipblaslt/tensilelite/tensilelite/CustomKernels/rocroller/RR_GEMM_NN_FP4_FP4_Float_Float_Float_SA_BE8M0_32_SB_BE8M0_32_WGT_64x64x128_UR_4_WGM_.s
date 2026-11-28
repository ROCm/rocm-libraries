// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_,@function
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 3494) 
s_mul_hi_i32 s1, s16, s10 // (op 3751) 
v_and_b32 v0, 63, v1 // (op 3556) 
v_and_b32 v3, 48, v1 // (op 3556) 
v_sub_u32 v4, v0, v3 // (op 3556) 

v_lshrrev_b32 v3, 4, v0 // (op 3556) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 3556) 


v_lshrrev_b32 v3, 7, v1 // (op 3556) 
v_lshlrev_b32 v4, 4, v3 // (op 3556) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 3556) 


v_lshlrev_b32 v0, 2, v3 // (op 3556) 

v_lshrrev_b32 v3, 3, v0 // (op 3556) 

v_lshrrev_b32 v0, 6, v1 // (op 3614) 
v_and_b32 v4, 1, v0 // (op 3614) 

v_and_b32 v0, 15, v1 // (op 3614) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 3614) 


v_and_b32 v0, 63, v1 // (op 3614) 
v_lshrrev_b32 v4, 4, v0 // (op 3614) 

v_lshlrev_b32 v0, 5, v4 // (op 3614) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3614) 


v_lshlrev_b32 v0, 2, v4 // (op 3614) 

v_lshrrev_b32 v4, 3, v0 // (op 3614) 

v_lshrrev_b32 v0, 7, v1 // (op 3672) 
v_and_b32 v5, 15, v1 // (op 3672) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 3672) 


v_and_b32 v0, 63, v1 // (op 3672) 
v_lshrrev_b32 v5, 4, v0 // (op 3672) 

v_lshl_add_u32 v0, v5, 6, v6 // (op 3672) 


v_lshrrev_b32 v5, 6, v1 // (op 3730) 
v_and_b32 v6, 1, v5 // (op 3730) 

v_and_b32 v5, 15, v1 // (op 3730) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 3730) 


v_and_b32 v5, 63, v1 // (op 3730) 
v_lshrrev_b32 v6, 4, v5 // (op 3730) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 3730) 


v_and_b32 v6, 63, v1 // (op 3782) 
v_and_b32 v7, 48, v1 // (op 3782) 
v_sub_u32 v8, v6, v7 // (op 3782) 

v_lshrrev_b32 v7, 4, v6 // (op 3782) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 3782) 


v_lshrrev_b32 v7, 7, v1 // (op 3782) 
v_lshlrev_b32 v8, 4, v7 // (op 3782) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 3782) 


v_lshlrev_b32 v6, 2, v7 // (op 3782) 

v_lshrrev_b32 v7, 3, v6 // (op 3782) 

v_lshrrev_b32 v6, 6, v1 // (op 3811) 
v_and_b32 v8, 1, v6 // (op 3811) 

v_and_b32 v6, 15, v1 // (op 3811) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 3811) 


v_and_b32 v6, 63, v1 // (op 3811) 
v_lshrrev_b32 v8, 4, v6 // (op 3811) 

v_lshlrev_b32 v6, 5, v8 // (op 3811) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 3811) 


v_lshlrev_b32 v6, 2, v8 // (op 3811) 

v_lshrrev_b32 v8, 3, v6 // (op 3811) 

v_lshrrev_b32 v6, 7, v1 // (op 3840) 
v_and_b32 v9, 15, v1 // (op 3840) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 3840) 


v_and_b32 v6, 63, v1 // (op 3840) 
v_lshrrev_b32 v9, 4, v6 // (op 3840) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 3840) 


v_lshrrev_b32 v9, 6, v1 // (op 3869) 
v_and_b32 v10, 1, v9 // (op 3869) 

v_and_b32 v9, 15, v1 // (op 3869) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 3869) 


v_and_b32 v9, 63, v1 // (op 3869) 
v_lshrrev_b32 v10, 4, v9 // (op 3869) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 3869) 


v_and_b32 v10, 63, v1 // (op 3898) 
v_and_b32 v11, 48, v1 // (op 3898) 
v_sub_u32 v12, v10, v11 // (op 3898) 

v_lshrrev_b32 v11, 4, v10 // (op 3898) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 3898) 


v_lshrrev_b32 v11, 7, v1 // (op 3898) 
v_lshlrev_b32 v12, 4, v11 // (op 3898) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 3898) 


v_lshlrev_b32 v10, 2, v11 // (op 3898) 

v_lshrrev_b32 v11, 3, v10 // (op 3898) 

v_lshrrev_b32 v10, 6, v1 // (op 3927) 
v_and_b32 v12, 1, v10 // (op 3927) 

v_and_b32 v10, 15, v1 // (op 3927) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3927) 


v_and_b32 v10, 63, v1 // (op 3927) 
v_lshrrev_b32 v12, 4, v10 // (op 3927) 

v_lshlrev_b32 v10, 5, v12 // (op 3927) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 3927) 


v_lshlrev_b32 v10, 2, v12 // (op 3927) 

v_lshrrev_b32 v12, 3, v10 // (op 3927) 

v_lshrrev_b32 v10, 7, v1 // (op 3956) 
v_and_b32 v13, 15, v1 // (op 3956) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 3956) 


v_and_b32 v10, 63, v1 // (op 3956) 
v_lshrrev_b32 v13, 4, v10 // (op 3956) 

v_lshl_add_u32 v10, v13, 6, v14 // (op 3956) 


v_lshrrev_b32 v13, 6, v1 // (op 3985) 
v_and_b32 v14, 1, v13 // (op 3985) 

v_and_b32 v13, 15, v1 // (op 3985) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3985) 


v_and_b32 v13, 63, v1 // (op 3985) 
v_lshrrev_b32 v14, 4, v13 // (op 3985) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 3985) 


v_and_b32 v14, 63, v1 // (op 4014) 
v_and_b32 v15, 48, v1 // (op 4014) 
v_sub_u32 v16, v14, v15 // (op 4014) 

v_lshrrev_b32 v15, 4, v14 // (op 4014) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 4014) 


v_lshrrev_b32 v15, 7, v1 // (op 4014) 
v_lshlrev_b32 v16, 4, v15 // (op 4014) 

v_lshl_add_u32 v15, v14, 6, v16 // (op 4014) 


v_lshlrev_b32 v14, 2, v15 // (op 4014) 

v_lshrrev_b32 v15, 3, v14 // (op 4014) 

v_lshrrev_b32 v14, 6, v1 // (op 4043) 
v_and_b32 v16, 1, v14 // (op 4043) 

v_and_b32 v14, 15, v1 // (op 4043) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 4043) 


v_and_b32 v14, 63, v1 // (op 4043) 
v_lshrrev_b32 v16, 4, v14 // (op 4043) 

v_lshlrev_b32 v14, 5, v16 // (op 4043) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 4043) 


v_lshlrev_b32 v14, 2, v16 // (op 4043) 

v_lshrrev_b32 v16, 3, v14 // (op 4043) 

v_lshrrev_b32 v14, 7, v1 // (op 4072) 
v_and_b32 v17, 15, v1 // (op 4072) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 4072) 


v_and_b32 v14, 63, v1 // (op 4072) 
v_lshrrev_b32 v17, 4, v14 // (op 4072) 

v_lshl_add_u32 v14, v17, 6, v18 // (op 4072) 


v_lshrrev_b32 v17, 6, v1 // (op 4101) 
v_and_b32 v18, 1, v17 // (op 4101) 

v_and_b32 v17, 15, v1 // (op 4101) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 4101) 


v_and_b32 v17, 63, v1 // (op 4101) 
v_lshrrev_b32 v18, 4, v17 // (op 4101) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 4101) 


v_lshrrev_b32 v18, 7, v1 // (op 3494) 
s_add_i32 s18, s0, s16 // (op 3494) 

v_and_b32 v19, 63, v1 // (op 3494) 
s_ashr_i32 s0, s18, 31 // (op 3494) 
v_lshrrev_b32 v20, 4, v19 // (op 3494) 
s_cmp_eq_i32 s10, 0 // (op 3494) 
s_mov_b32 s19, scc // (op 3494) 
v_lshlrev_b32 v21, 2, v20 // (op 3494) 

s_cmp_eq_i32 s19, 0 // (op 3494) 
s_cselect_b32 s51, 0, -1 // (op 3494) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 3494) 


s_and_b32 s19, s11, 31 // (op 3494) 
v_lshrrev_b32 v18, 2, v19 // (op 3494) 

v_and_b32 v19, 3, v18 // (op 3494) 

v_lshrrev_b32 v18, 7, v1 // (op 3751) 
v_and_b32 v21, 63, v1 // (op 3751) 
s_lshl_b32 s52, 1, s19 // (op 3494) 
v_and_b32 v22, 3, v1 // (op 3494) 
s_add_i32 s53, s52, s51 // (op 3494) 


s_and_b32 s51, s0, s53 // (op 3494) 


s_add_i32 s0, s1, s16 // (op 3751) 

s_ashr_i32 s1, s0, 31 // (op 3751) 
v_lshl_add_u32 v23, v19, 2, v22 // (op 3494) 


s_add_i32 s52, s18, s51 // (op 3494) 


v_lshrrev_b32 v19, 6, v1 // (op 3494) 
s_ashr_i32 s18, s52, s19 // (op 3494) 


v_and_b32 v22, 1, v19 // (op 3494) 

s_lshl_b32 s19, s11, 24 // (op 3494) 
v_lshl_add_u32 v19, v22, 4, v23 // (op 3494) 


s_ashr_i32 s51, s19, 31 // (op 3494) 

s_xor_b32 s19, s18, s51 // (op 3494) 

s_sub_i32 s18, s19, s51 // (op 3494) 


s_mul_hi_i32 s19, s18, s12 // (op 3494) 
s_add_i32 s51, s19, s18 // (op 3494) 

s_ashr_i32 s19, s51, 31 // (op 3494) 
s_cmp_eq_i32 s12, 0 // (op 3494) 
s_mov_b32 s52, scc // (op 3494) 
s_cmp_eq_i32 s52, 0 // (op 3494) 
s_cselect_b32 s53, 0, -1 // (op 3494) 

s_and_b32 s52, s13, 31 // (op 3494) 
v_lshrrev_b32 v22, 4, v21 // (op 3751) 
s_cmp_eq_i32 s10, 0 // (op 3751) 
s_mov_b32 s54, scc // (op 3751) 
v_lshlrev_b32 v23, 2, v22 // (op 3751) 

s_cmp_eq_i32 s54, 0 // (op 3751) 
s_cselect_b32 s55, 0, -1 // (op 3751) 

s_lshl_b32 s54, 1, s52 // (op 3494) 
v_lshl_add_u32 v22, v18, 4, v23 // (op 3751) 


v_lshrrev_b32 v18, 2, v21 // (op 3751) 

s_add_i32 s56, s54, s53 // (op 3494) 


s_and_b32 s53, s19, s56 // (op 3494) 


s_add_i32 s19, s51, s53 // (op 3494) 


s_ashr_i32 s51, s19, s52 // (op 3494) 


s_lshl_b32 s19, s13, 24 // (op 3494) 
s_ashr_i32 s52, s19, 31 // (op 3494) 

s_xor_b32 s19, s51, s52 // (op 3494) 

s_sub_i32 s51, s19, s52 // (op 3494) 


s_mul_i32 s19, s18, s15 // (op 3494) 
s_sub_u32 s52, s16, s19 // (op 3494) 

s_mul_i32 s19, s51, s15 // (op 3494) 
s_add_u32 s19, s19, s52 // (op 3494) 

s_mul_hi_i32 s52, s16, s9 // (op 3494) 
s_add_i32 s53, s52, s16 // (op 3494) 

s_ashr_i32 s52, s53, 31 // (op 3494) 
s_cmp_eq_i32 s9, 0 // (op 3494) 
s_mov_b32 s54, scc // (op 3494) 
s_cmp_eq_i32 s54, 0 // (op 3494) 
s_cselect_b32 s56, 0, -1 // (op 3494) 

s_and_b32 s54, s11, 31 // (op 3751) 
v_and_b32 v21, 3, v18 // (op 3751) 

s_lshl_b32 s57, 1, s54 // (op 3751) 
v_and_b32 v18, 3, v1 // (op 3751) 
s_add_i32 s58, s57, s55 // (op 3751) 


s_and_b32 s55, s1, s58 // (op 3751) 


v_lshl_add_u32 v23, v21, 2, v18 // (op 3751) 


s_add_i32 s1, s0, s55 // (op 3751) 


v_lshrrev_b32 v18, 6, v1 // (op 3751) 
s_ashr_i32 s0, s1, s54 // (op 3751) 


v_and_b32 v21, 1, v18 // (op 3751) 

s_lshl_b32 s1, s11, 24 // (op 3751) 
v_lshl_add_u32 v18, v21, 4, v23 // (op 3751) 


s_ashr_i32 s54, s1, 31 // (op 3751) 

s_xor_b32 s1, s0, s54 // (op 3751) 

s_sub_i32 s0, s1, s54 // (op 3751) 


s_mul_hi_i32 s1, s0, s12 // (op 3751) 
s_add_i32 s54, s1, s0 // (op 3751) 

s_ashr_i32 s1, s54, 31 // (op 3751) 
s_cmp_eq_i32 s12, 0 // (op 3751) 
s_mov_b32 s55, scc // (op 3751) 
s_cmp_eq_i32 s55, 0 // (op 3751) 
s_cselect_b32 s57, 0, -1 // (op 3751) 

s_and_b32 s55, s13, 31 // (op 3751) 
s_lshl_b32 s58, 1, s55 // (op 3751) 
s_add_i32 s59, s58, s57 // (op 3751) 


s_and_b32 s57, s1, s59 // (op 3751) 


s_add_i32 s1, s54, s57 // (op 3751) 


s_ashr_i32 s54, s1, s55 // (op 3751) 


s_lshl_b32 s1, s13, 24 // (op 3751) 
s_ashr_i32 s55, s1, 31 // (op 3751) 

s_xor_b32 s1, s54, s55 // (op 3751) 

s_sub_i32 s54, s1, s55 // (op 3751) 


s_mul_i32 s1, s0, s15 // (op 3751) 
s_sub_u32 s55, s16, s1 // (op 3751) 

s_mul_i32 s1, s54, s15 // (op 3751) 
s_add_u32 s1, s1, s55 // (op 3751) 

s_mul_hi_i32 s55, s16, s9 // (op 3751) 
s_add_i32 s57, s55, s16 // (op 3751) 

s_ashr_i32 s55, s57, 31 // (op 3751) 
s_cmp_eq_i32 s9, 0 // (op 3751) 
s_mov_b32 s58, scc // (op 3751) 
s_cmp_eq_i32 s58, 0 // (op 3751) 
s_cselect_b32 s59, 0, -1 // (op 3751) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 3494) 
s_lshl_b32 s60, 1, s58 // (op 3494) 
s_add_i32 s61, s60, s56 // (op 3494) 


s_and_b32 s56, s52, s61 // (op 3494) 


s_add_i32 s52, s53, s56 // (op 3494) 


s_ashr_i32 s53, s52, s58 // (op 3494) 


s_lshl_b32 s52, s46, 24 // (op 3494) 
s_ashr_i32 s56, s52, 31 // (op 3494) 

s_xor_b32 s52, s53, s56 // (op 3494) 

s_sub_i32 s53, s52, s56 // (op 3494) 


s_and_b32 s52, s46, 31 // (op 3751) 
s_lshl_b32 s56, 1, s52 // (op 3751) 
s_add_i32 s58, s56, s59 // (op 3751) 


s_and_b32 s56, s55, s58 // (op 3751) 


s_add_i32 s55, s57, s56 // (op 3751) 


s_ashr_i32 s56, s55, s52 // (op 3751) 


s_lshl_b32 s52, s46, 24 // (op 3751) 
s_ashr_i32 s55, s52, 31 // (op 3751) 

s_xor_b32 s52, s56, s55 // (op 3751) 

s_sub_i32 s56, s52, s55 // (op 3751) 


s_add_u32 s58, s6, 64 // (op 3494) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 3494) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3494) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3494) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 3494) 

s_mul_i32 s52, s15, s58 // (op 3494) 
s_mul_i32 s55, s8, s52 // (op 3494) 

s_mul_i32 s52, s53, s55 // (op 3494) 

s_sub_u32 s55, s16, s52 // (op 3494) 

s_mov_b32 s52, s55 // (op 3494) 
s_mul_hi_i32 s57, s52, s14 // (op 3494) 
s_add_i32 s60, s57, s52 // (op 3494) 


s_ashr_i32 s52, s60, 31 // (op 3494) 
s_cmp_eq_i32 s14, 0 // (op 3494) 
s_mov_b32 s57, scc // (op 3494) 
s_cmp_eq_i32 s57, 0 // (op 3494) 
s_cselect_b32 s61, 0, -1 // (op 3494) 

s_and_b32 s57, s48, 31 // (op 3494) 
s_lshl_b32 s62, 1, s57 // (op 3494) 
s_add_i32 s63, s62, s61 // (op 3494) 


s_and_b32 s61, s52, s63 // (op 3494) 


s_add_i32 s52, s60, s61 // (op 3494) 


s_ashr_i32 s60, s52, s57 // (op 3494) 


s_lshl_b32 s52, s48, 24 // (op 3494) 
s_ashr_i32 s57, s52, 31 // (op 3494) 

s_xor_b32 s52, s60, s57 // (op 3494) 

s_sub_i32 s60, s52, s57 // (op 3494) 


s_mul_i32 s52, s47, s15 // (op 3494) 
s_add_u32 s62, s20, 64 // (op 3494) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 3494) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3494) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3494) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 3494) 

s_sub_i32 s57, s62, s52 // (op 3494) 

s_mul_i32 s61, s60, s57 // (op 3494) 

s_sub_u32 s57, s55, s61 // (op 3494) 


s_add_u32 s55, s57, s52 // (op 3494) 


s_cmp_eq_i32 s53, 0 // (op 3494) 
s_mov_b32 s52, scc // (op 3494) 

s_cmp_eq_i32 s52, 0 // (op 3494) 
s_cselect_b32 s53, s55, s19 // (op 3494) 


s_mul_i32 s19, s51, s58 // (op 3494) 


v_lshl_add_u32 v21, s53, 6, v20 // (op 3494) 


s_sub_i32 s51, s18, s19 // (op 3494) 


s_cmp_eq_i32 s52, 0 // (op 3494) 
s_cselect_b32 s53, s60, s51 // (op 3494) 



v_lshl_add_u32 v20, s53, 6, v19 // (op 3494) 


v_mul_lo_u32 v19, v20, s2 // (op 3494) 

v_add_lshl_u32 v20, v21, v19, 2 // (op 3494) 


s_add_u32 s18, s6, 64 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 3751) 

s_mul_i32 s55, s15, s18 // (op 3751) 
s_mul_i32 s51, s8, s55 // (op 3751) 

s_mul_i32 s55, s56, s51 // (op 3751) 

s_sub_u32 s51, s16, s55 // (op 3751) 

s_mov_b32 s55, s51 // (op 3751) 
s_mul_hi_i32 s52, s55, s14 // (op 3751) 
s_add_i32 s53, s52, s55 // (op 3751) 


s_ashr_i32 s52, s53, 31 // (op 3751) 
s_cmp_eq_i32 s14, 0 // (op 3751) 
s_mov_b32 s55, scc // (op 3751) 
s_cmp_eq_i32 s55, 0 // (op 3751) 
s_cselect_b32 s57, 0, -1 // (op 3751) 

s_and_b32 s55, s48, 31 // (op 3751) 
s_lshl_b32 s58, 1, s55 // (op 3751) 
s_add_i32 s59, s58, s57 // (op 3751) 


s_and_b32 s57, s52, s59 // (op 3751) 


s_add_i32 s52, s53, s57 // (op 3751) 


s_ashr_i32 s53, s52, s55 // (op 3751) 


s_lshl_b32 s52, s48, 24 // (op 3751) 
s_ashr_i32 s55, s52, 31 // (op 3751) 

s_xor_b32 s52, s53, s55 // (op 3751) 

s_sub_i32 s53, s52, s55 // (op 3751) 


s_mul_i32 s52, s47, s15 // (op 3751) 
s_add_u32 s58, s20, 64 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 3751) 

s_sub_i32 s55, s58, s52 // (op 3751) 

s_mul_i32 s57, s53, s55 // (op 3751) 

s_sub_u32 s55, s51, s57 // (op 3751) 


s_add_u32 s51, s55, s52 // (op 3751) 


s_cmp_eq_i32 s56, 0 // (op 3751) 
s_mov_b32 s52, scc // (op 3751) 

s_cmp_eq_i32 s52, 0 // (op 3751) 
s_cselect_b32 s55, s51, s1 // (op 3751) 


s_mul_i32 s1, s54, s18 // (op 3751) 


v_lshl_add_u32 v19, s55, 6, v22 // (op 3751) 


s_sub_i32 s51, s0, s1 // (op 3751) 


s_cmp_eq_i32 s52, 0 // (op 3751) 
s_cselect_b32 s0, s53, s51 // (op 3751) 



v_lshl_add_u32 v21, s0, 6, v18 // (op 3751) 


v_mul_lo_u32 v18, v21, s4 // (op 3751) 

v_add_lshl_u32 v21, v19, v18, 2 // (op 3751) 


s_mul_i32 s1, s2, s7 // (op 3496) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 3496) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 3496) least significant
s_mul_hi_u32 s19, s2, s6 // (op 3496) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3496) most significant: sum
s_add_u32 s1, s1, s19 // (op 3496) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3496) 

s_mul_i32 s1, s4, s7 // (op 3753) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 3753) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 3753) least significant
s_mul_hi_u32 s52, s4, s6 // (op 3753) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3753) most significant: sum
s_add_u32 s1, s1, s52 // (op 3753) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 3753) 

s_mov_b32 s56, s22 // (op 3496) reg expression
s_mov_b32 s57, s23 // (op 3496) reg expression
s_mov_b32 s58, s18 // (op 3496) reg expression
s_mov_b32 s59, 131072 // (op 3496) call()

s_mov_b32 s60, s24 // (op 3753) reg expression

s_mov_b32 s61, s25 // (op 3753) reg expression
s_mov_b32 s62, s52 // (op 3753) reg expression
s_mov_b32 s63, 131072 // (op 3753) call()


v_mov_b32 v18, s49 // (op 2) Move value
v_mov_b32 v19, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 482) call()
v_accvgpr_write a5, 0.00000 // (op 482) call()
v_accvgpr_write a6, 0.00000 // (op 482) call()
v_accvgpr_write a7, 0.00000 // (op 482) call()
v_accvgpr_write a8, 0.00000 // (op 848) call()
v_accvgpr_write a9, 0.00000 // (op 848) call()
v_accvgpr_write a10, 0.00000 // (op 848) call()
v_accvgpr_write a11, 0.00000 // (op 848) call()
v_accvgpr_write a12, 0.00000 // (op 1095) call()
v_accvgpr_write a13, 0.00000 // (op 1095) call()
v_accvgpr_write a14, 0.00000 // (op 1095) call()
v_accvgpr_write a15, 0.00000 // (op 1095) call()
s_mul_hi_i32 s0, s16, s10 // (op 3256) 
v_and_b32 v22, -2, v1 // (op 3256) 
s_add_i32 s1, s0, s16 // (op 3256) 

v_sub_u32 v23, v1, v22 // (op 3256) 

s_ashr_i32 s0, s1, 31 // (op 3256) 
v_lshlrev_b32 v22, 5, v23 // (op 3256) 

s_cmp_eq_i32 s10, 0 // (op 3256) 
s_mov_b32 s18, scc // (op 3256) 
v_lshrrev_b32 v23, 1, v1 // (op 3256) 
s_cmp_eq_i32 s18, 0 // (op 3256) 
s_cselect_b32 s19, 0, -1 // (op 3256) 

s_and_b32 s18, s11, 31 // (op 3256) 
s_lshl_b32 s22, 1, s18 // (op 3256) 
s_add_i32 s23, s22, s19 // (op 3256) 


s_and_b32 s19, s0, s23 // (op 3256) 


s_add_i32 s0, s1, s19 // (op 3256) 


s_ashr_i32 s1, s0, s18 // (op 3256) 


s_lshl_b32 s0, s11, 24 // (op 3256) 
s_ashr_i32 s18, s0, 31 // (op 3256) 

s_xor_b32 s0, s1, s18 // (op 3256) 

s_sub_i32 s1, s0, s18 // (op 3256) 


s_mul_hi_i32 s0, s1, s12 // (op 3256) 
s_add_i32 s18, s0, s1 // (op 3256) 

s_ashr_i32 s0, s18, 31 // (op 3256) 
s_cmp_eq_i32 s12, 0 // (op 3256) 
s_mov_b32 s19, scc // (op 3256) 
s_cmp_eq_i32 s19, 0 // (op 3256) 
s_cselect_b32 s22, 0, -1 // (op 3256) 

s_and_b32 s19, s13, 31 // (op 3256) 
s_lshl_b32 s23, 1, s19 // (op 3256) 
s_add_i32 s24, s23, s22 // (op 3256) 


s_and_b32 s25, s0, s24 // (op 3256) 


s_add_i32 s0, s18, s25 // (op 3256) 


s_ashr_i32 s18, s0, s19 // (op 3256) 


s_lshl_b32 s0, s13, 24 // (op 3256) 
s_ashr_i32 s19, s0, 31 // (op 3256) 

s_xor_b32 s0, s18, s19 // (op 3256) 

s_sub_i32 s18, s0, s19 // (op 3256) 


s_mul_i32 s0, s1, s15 // (op 3256) 

s_sub_u32 s1, s16, s0 // (op 3256) 

s_mul_i32 s0, s18, s15 // (op 3256) 
s_add_u32 s0, s0, s1 // (op 3256) 


s_mul_hi_i32 s1, s16, s9 // (op 3256) 
s_add_i32 s18, s1, s16 // (op 3256) 

s_ashr_i32 s1, s18, 31 // (op 3256) 
s_cmp_eq_i32 s9, 0 // (op 3256) 
s_mov_b32 s19, scc // (op 3256) 
s_cmp_eq_i32 s19, 0 // (op 3256) 
s_cselect_b32 s22, 0, -1 // (op 3256) 

s_and_b32 s19, s46, 31 // (op 3256) 
s_lshl_b32 s23, 1, s19 // (op 3256) 
s_add_i32 s24, s23, s22 // (op 3256) 


s_and_b32 s25, s1, s24 // (op 3256) 


s_add_i32 s1, s18, s25 // (op 3256) 


s_ashr_i32 s18, s1, s19 // (op 3256) 


s_lshl_b32 s1, s46, 24 // (op 3256) 
s_ashr_i32 s19, s1, 31 // (op 3256) 

s_xor_b32 s1, s18, s19 // (op 3256) 

s_sub_i32 s18, s1, s19 // (op 3256) 


s_add_u32 s22, s6, 64 // (op 3256) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 3256) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3256) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3256) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 3256) 

s_mul_i32 s1, s15, s22 // (op 3256) 

s_mul_i32 s19, s8, s1 // (op 3256) 

s_mul_i32 s1, s18, s19 // (op 3256) 

s_sub_u32 s19, s16, s1 // (op 3256) 

s_mov_b32 s1, s19 // (op 3256) 
s_mul_hi_i32 s22, s1, s14 // (op 3256) 
s_add_i32 s23, s22, s1 // (op 3256) 


s_ashr_i32 s1, s23, 31 // (op 3256) 
s_cmp_eq_i32 s14, 0 // (op 3256) 
s_mov_b32 s22, scc // (op 3256) 
s_cmp_eq_i32 s22, 0 // (op 3256) 
s_cselect_b32 s24, 0, -1 // (op 3256) 

s_and_b32 s22, s48, 31 // (op 3256) 
s_lshl_b32 s25, 1, s22 // (op 3256) 
s_add_i32 s49, s25, s24 // (op 3256) 


s_and_b32 s24, s1, s49 // (op 3256) 


s_add_i32 s1, s23, s24 // (op 3256) 


s_ashr_i32 s23, s1, s22 // (op 3256) 


s_lshl_b32 s1, s48, 24 // (op 3256) 
s_ashr_i32 s22, s1, 31 // (op 3256) 

s_xor_b32 s1, s23, s22 // (op 3256) 

s_sub_i32 s23, s1, s22 // (op 3256) 


s_mul_i32 s1, s47, s15 // (op 3256) 
s_add_u32 s24, s20, 64 // (op 3256) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3256) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 3256) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 3256) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 3256) 

s_sub_i32 s22, s24, s1 // (op 3256) 

s_mul_i32 s24, s23, s22 // (op 3256) 


s_sub_u32 s25, s19, s24 // (op 3256) 


s_add_u32 s19, s25, s1 // (op 3256) 


s_cmp_eq_i32 s18, 0 // (op 3256) 
s_mov_b32 s1, scc // (op 3256) 

s_cmp_eq_i32 s1, 0 // (op 3256) 
s_cselect_b32 s18, s19, s0 // (op 3256) 



v_lshl_add_u32 v24, s18, 6, v22 // (op 3256) 


v_mov_b32 v27, 0 // (op 3256) convert
v_mov_b32 v26, v24 // (op 3256) convert

v_mul_lo_u32 v25, v23, s27 // (op 3256) most significant: low of low * high
v_mul_hi_u32 v22, v23, s26 // (op 3256) most significant: high of low * low
v_mul_lo_u32 v24, v23, s26 // (op 3256) least significant: low of low * low
v_add_u32 v25, v25, v22 // (op 3256) most significant: sum


v_add_co_u32 v22, s[0:1], v26, v24 // (op 3256) least significant half
v_addc_co_u32 v23, s[0:1], v27, v25, s[0:1] // (op 3256) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 3256) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 3256) 

v_mov_b32 v22, v24 // (op 3256) convert

s_mul_i32 s1, s26, s31 // (op 3258) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3258) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3258) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3258) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3258) most significant: sum
s_add_u32 s1, s1, s19 // (op 3258) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3258) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3258) 

s_mov_b32 s52, s28 // (op 3258) reg expression
s_mov_b32 s53, s29 // (op 3258) reg expression
s_mov_b32 s54, s0 // (op 3258) reg expression
s_mov_b32 s55, 131072 // (op 3258) call()

s_mul_hi_i32 s0, s16, s10 // (op 3275) 
v_and_b32 v23, -2, v1 // (op 3275) 
s_add_i32 s1, s0, s16 // (op 3275) 

v_sub_u32 v24, v1, v23 // (op 3275) 

s_ashr_i32 s0, s1, 31 // (op 3275) 
v_lshlrev_b32 v23, 5, v24 // (op 3275) 

s_cmp_eq_i32 s10, 0 // (op 3275) 
s_mov_b32 s18, scc // (op 3275) 
v_lshrrev_b32 v24, 1, v1 // (op 3275) 
s_cmp_eq_i32 s18, 0 // (op 3275) 
s_cselect_b32 s19, 0, -1 // (op 3275) 

s_and_b32 s18, s11, 31 // (op 3275) 
s_lshl_b32 s22, 1, s18 // (op 3275) 
s_add_i32 s23, s22, s19 // (op 3275) 


s_and_b32 s19, s0, s23 // (op 3275) 


s_add_i32 s0, s1, s19 // (op 3275) 


s_ashr_i32 s1, s0, s18 // (op 3275) 


s_lshl_b32 s0, s11, 24 // (op 3275) 
s_ashr_i32 s18, s0, 31 // (op 3275) 

s_xor_b32 s0, s1, s18 // (op 3275) 

s_sub_i32 s1, s0, s18 // (op 3275) 


s_mul_hi_i32 s0, s1, s12 // (op 3275) 
s_add_i32 s18, s0, s1 // (op 3275) 

s_ashr_i32 s0, s18, 31 // (op 3275) 
s_cmp_eq_i32 s12, 0 // (op 3275) 
s_mov_b32 s19, scc // (op 3275) 
s_cmp_eq_i32 s19, 0 // (op 3275) 
s_cselect_b32 s22, 0, -1 // (op 3275) 

s_and_b32 s19, s13, 31 // (op 3275) 
s_lshl_b32 s23, 1, s19 // (op 3275) 
s_add_i32 s24, s23, s22 // (op 3275) 


s_and_b32 s25, s0, s24 // (op 3275) 


s_add_i32 s0, s18, s25 // (op 3275) 


s_ashr_i32 s18, s0, s19 // (op 3275) 


s_lshl_b32 s0, s13, 24 // (op 3275) 
s_ashr_i32 s19, s0, 31 // (op 3275) 

s_xor_b32 s0, s18, s19 // (op 3275) 

s_sub_i32 s18, s0, s19 // (op 3275) 


s_mul_i32 s0, s1, s15 // (op 3275) 

s_sub_u32 s1, s16, s0 // (op 3275) 

s_mul_i32 s0, s18, s15 // (op 3275) 
s_add_u32 s0, s0, s1 // (op 3275) 


s_mul_hi_i32 s1, s16, s9 // (op 3275) 
s_add_i32 s18, s1, s16 // (op 3275) 

s_ashr_i32 s1, s18, 31 // (op 3275) 
s_cmp_eq_i32 s9, 0 // (op 3275) 
s_mov_b32 s19, scc // (op 3275) 
s_cmp_eq_i32 s19, 0 // (op 3275) 
s_cselect_b32 s22, 0, -1 // (op 3275) 

s_and_b32 s19, s46, 31 // (op 3275) 
s_lshl_b32 s23, 1, s19 // (op 3275) 
s_add_i32 s24, s23, s22 // (op 3275) 


s_and_b32 s25, s1, s24 // (op 3275) 


s_add_i32 s1, s18, s25 // (op 3275) 


s_ashr_i32 s18, s1, s19 // (op 3275) 


s_lshl_b32 s1, s46, 24 // (op 3275) 
s_ashr_i32 s19, s1, 31 // (op 3275) 

s_xor_b32 s1, s18, s19 // (op 3275) 

s_sub_i32 s18, s1, s19 // (op 3275) 


s_add_u32 s50, s6, 64 // (op 3275) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3275) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3275) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3275) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3275) 

s_mul_i32 s1, s15, s24 // (op 3275) 

s_mul_i32 s19, s8, s1 // (op 3275) 

s_mul_i32 s1, s18, s19 // (op 3275) 

s_sub_u32 s19, s16, s1 // (op 3275) 

s_mov_b32 s1, s19 // (op 3275) 
s_mul_hi_i32 s22, s1, s14 // (op 3275) 
s_add_i32 s23, s22, s1 // (op 3275) 


s_ashr_i32 s1, s23, 31 // (op 3275) 
s_cmp_eq_i32 s14, 0 // (op 3275) 
s_mov_b32 s22, scc // (op 3275) 
s_cmp_eq_i32 s22, 0 // (op 3275) 
s_cselect_b32 s24, 0, -1 // (op 3275) 

s_and_b32 s22, s48, 31 // (op 3275) 
s_lshl_b32 s25, 1, s22 // (op 3275) 
s_add_i32 s49, s25, s24 // (op 3275) 


s_and_b32 s24, s1, s49 // (op 3275) 


s_add_i32 s1, s23, s24 // (op 3275) 


s_ashr_i32 s23, s1, s22 // (op 3275) 


s_lshl_b32 s1, s48, 24 // (op 3275) 
s_ashr_i32 s22, s1, 31 // (op 3275) 

s_xor_b32 s1, s23, s22 // (op 3275) 

s_sub_i32 s23, s1, s22 // (op 3275) 


s_mul_i32 s1, s47, s15 // (op 3275) 
s_add_u32 s24, s20, 64 // (op 3275) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3275) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3275) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3275) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 3275) 

s_sub_i32 s22, s24, s1 // (op 3275) 

s_mul_i32 s24, s23, s22 // (op 3275) 


s_sub_u32 s25, s19, s24 // (op 3275) 


s_add_u32 s19, s25, s1 // (op 3275) 


s_cmp_eq_i32 s18, 0 // (op 3275) 
s_mov_b32 s1, scc // (op 3275) 

s_cmp_eq_i32 s1, 0 // (op 3275) 
s_cselect_b32 s18, s19, s0 // (op 3275) 



v_lshl_add_u32 v25, s18, 6, v23 // (op 3275) 


v_mov_b32 v27, 0 // (op 3275) convert
v_mov_b32 v26, v25 // (op 3275) convert

v_mul_lo_u32 v29, v24, s27 // (op 3275) most significant: low of low * high
v_mul_hi_u32 v23, v24, s26 // (op 3275) most significant: high of low * low
v_mul_lo_u32 v28, v24, s26 // (op 3275) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 3275) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 3275) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 3275) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 3275) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 3275) 

v_mov_b32 v23, v26 // (op 3275) convert

s_mul_i32 s1, s26, s31 // (op 3277) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3277) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3277) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3277) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3277) most significant: sum
s_add_u32 s1, s1, s19 // (op 3277) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3277) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3277) 

s_mov_b32 s52, s28 // (op 3277) reg expression
s_mov_b32 s53, s29 // (op 3277) reg expression
s_mov_b32 s54, s0 // (op 3277) reg expression
s_mov_b32 s55, 131072 // (op 3277) call()

s_mul_hi_i32 s0, s16, s10 // (op 3320) 
v_lshrrev_b32 v24, 2, v1 // (op 3320) 
s_add_i32 s1, s0, s16 // (op 3320) 

v_and_b32 v25, -4, v1 // (op 3320) 
s_ashr_i32 s0, s1, 31 // (op 3320) 
v_sub_u32 v26, v1, v25 // (op 3320) 

s_cmp_eq_i32 s10, 0 // (op 3320) 
s_mov_b32 s18, scc // (op 3320) 
v_lshlrev_b32 v25, 5, v26 // (op 3320) 

s_cmp_eq_i32 s18, 0 // (op 3320) 
s_cselect_b32 s19, 0, -1 // (op 3320) 

v_mov_b32 v27, 0 // (op 3320) convert
v_mov_b32 v26, v25 // (op 3320) convert

s_and_b32 s18, s11, 31 // (op 3320) 
s_lshl_b32 s22, 1, s18 // (op 3320) 
s_add_i32 s23, s22, s19 // (op 3320) 


s_and_b32 s19, s0, s23 // (op 3320) 


s_add_i32 s0, s1, s19 // (op 3320) 


s_ashr_i32 s1, s0, s18 // (op 3320) 


s_lshl_b32 s0, s11, 24 // (op 3320) 
s_ashr_i32 s18, s0, 31 // (op 3320) 

s_xor_b32 s0, s1, s18 // (op 3320) 

s_sub_i32 s1, s0, s18 // (op 3320) 


s_mul_hi_i32 s0, s1, s12 // (op 3320) 
s_add_i32 s18, s0, s1 // (op 3320) 

s_ashr_i32 s0, s18, 31 // (op 3320) 
s_cmp_eq_i32 s12, 0 // (op 3320) 
s_mov_b32 s19, scc // (op 3320) 
s_cmp_eq_i32 s19, 0 // (op 3320) 
s_cselect_b32 s22, 0, -1 // (op 3320) 

s_and_b32 s19, s13, 31 // (op 3320) 
s_lshl_b32 s23, 1, s19 // (op 3320) 
s_add_i32 s24, s23, s22 // (op 3320) 


s_and_b32 s25, s0, s24 // (op 3320) 


s_add_i32 s0, s18, s25 // (op 3320) 


s_ashr_i32 s18, s0, s19 // (op 3320) 


s_lshl_b32 s0, s13, 24 // (op 3320) 
s_ashr_i32 s19, s0, 31 // (op 3320) 

s_xor_b32 s0, s18, s19 // (op 3320) 

s_sub_i32 s18, s0, s19 // (op 3320) 


s_add_u32 s50, s6, 64 // (op 3320) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3320) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3320) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3320) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3320) 

s_mul_i32 s0, s18, s24 // (op 3320) 

s_sub_i32 s18, s1, s0 // (op 3320) 


s_mul_hi_i32 s19, s16, s9 // (op 3320) 
s_add_i32 s0, s19, s16 // (op 3320) 

s_ashr_i32 s1, s0, 31 // (op 3320) 
s_cmp_eq_i32 s9, 0 // (op 3320) 
s_mov_b32 s19, scc // (op 3320) 
s_cmp_eq_i32 s19, 0 // (op 3320) 
s_cselect_b32 s22, 0, -1 // (op 3320) 

s_and_b32 s19, s46, 31 // (op 3320) 
s_lshl_b32 s23, 1, s19 // (op 3320) 
s_add_i32 s49, s23, s22 // (op 3320) 


s_and_b32 s22, s1, s49 // (op 3320) 


s_add_i32 s1, s0, s22 // (op 3320) 


s_ashr_i32 s0, s1, s19 // (op 3320) 


s_lshl_b32 s1, s46, 24 // (op 3320) 
s_ashr_i32 s19, s1, 31 // (op 3320) 

s_xor_b32 s1, s0, s19 // (op 3320) 

s_sub_i32 s0, s1, s19 // (op 3320) 


s_mul_i32 s1, s15, s24 // (op 3320) 

s_mul_i32 s19, s8, s1 // (op 3320) 

s_mul_i32 s1, s0, s19 // (op 3320) 

s_sub_u32 s19, s16, s1 // (op 3320) 

s_mul_hi_i32 s1, s19, s14 // (op 3320) 
s_add_i32 s22, s1, s19 // (op 3320) 


s_ashr_i32 s1, s22, 31 // (op 3320) 
s_cmp_eq_i32 s14, 0 // (op 3320) 
s_mov_b32 s19, scc // (op 3320) 
s_cmp_eq_i32 s19, 0 // (op 3320) 
s_cselect_b32 s23, 0, -1 // (op 3320) 

s_and_b32 s19, s48, 31 // (op 3320) 
s_lshl_b32 s24, 1, s19 // (op 3320) 
s_add_i32 s25, s24, s23 // (op 3320) 


s_and_b32 s23, s1, s25 // (op 3320) 


s_add_i32 s1, s22, s23 // (op 3320) 


s_ashr_i32 s22, s1, s19 // (op 3320) 


s_lshl_b32 s1, s48, 24 // (op 3320) 
s_ashr_i32 s19, s1, 31 // (op 3320) 

s_xor_b32 s1, s22, s19 // (op 3320) 

s_sub_i32 s22, s1, s19 // (op 3320) 


s_cmp_eq_i32 s0, 0 // (op 3320) 
s_mov_b32 s1, scc // (op 3320) 

s_cmp_eq_i32 s1, 0 // (op 3320) 
s_cselect_b32 s0, s22, s18 // (op 3320) 



v_lshl_add_u32 v25, s0, 6, v24 // (op 3320) 


v_mul_lo_u32 v29, v25, s33 // (op 3320) most significant: low of low * high
v_mul_hi_u32 v24, v25, s32 // (op 3320) most significant: high of low * low
v_mul_lo_u32 v28, v25, s32 // (op 3320) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 3320) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 3320) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 3320) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 3320) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 3320) 

v_mov_b32 v24, v26 // (op 3320) convert

s_mul_i32 s1, s32, s7 // (op 3322) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3322) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3322) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3322) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3322) most significant: sum
s_add_u32 s1, s1, s19 // (op 3322) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3322) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3322) 

s_mov_b32 s64, s34 // (op 3322) reg expression
s_mov_b32 s65, s35 // (op 3322) reg expression
s_mov_b32 s66, s0 // (op 3322) reg expression
s_mov_b32 s67, 131072 // (op 3322) call()

s_mul_hi_i32 s0, s16, s10 // (op 3339) 
v_lshrrev_b32 v25, 2, v1 // (op 3339) 
s_add_i32 s1, s0, s16 // (op 3339) 

v_and_b32 v26, -4, v1 // (op 3339) 
s_ashr_i32 s0, s1, 31 // (op 3339) 
v_sub_u32 v27, v1, v26 // (op 3339) 

s_cmp_eq_i32 s10, 0 // (op 3339) 
s_mov_b32 s18, scc // (op 3339) 
v_lshlrev_b32 v26, 5, v27 // (op 3339) 

s_cmp_eq_i32 s18, 0 // (op 3339) 
s_cselect_b32 s19, 0, -1 // (op 3339) 

v_mov_b32 v29, 0 // (op 3339) convert
v_mov_b32 v28, v26 // (op 3339) convert

s_and_b32 s18, s11, 31 // (op 3339) 
s_lshl_b32 s22, 1, s18 // (op 3339) 
s_add_i32 s23, s22, s19 // (op 3339) 


s_and_b32 s19, s0, s23 // (op 3339) 


s_add_i32 s0, s1, s19 // (op 3339) 


s_ashr_i32 s1, s0, s18 // (op 3339) 


s_lshl_b32 s0, s11, 24 // (op 3339) 
s_ashr_i32 s18, s0, 31 // (op 3339) 

s_xor_b32 s0, s1, s18 // (op 3339) 

s_sub_i32 s1, s0, s18 // (op 3339) 


s_mul_hi_i32 s0, s1, s12 // (op 3339) 
s_add_i32 s18, s0, s1 // (op 3339) 

s_ashr_i32 s0, s18, 31 // (op 3339) 
s_cmp_eq_i32 s12, 0 // (op 3339) 
s_mov_b32 s19, scc // (op 3339) 
s_cmp_eq_i32 s19, 0 // (op 3339) 
s_cselect_b32 s22, 0, -1 // (op 3339) 

s_and_b32 s19, s13, 31 // (op 3339) 
s_lshl_b32 s23, 1, s19 // (op 3339) 
s_add_i32 s24, s23, s22 // (op 3339) 


s_and_b32 s25, s0, s24 // (op 3339) 


s_add_i32 s0, s18, s25 // (op 3339) 


s_ashr_i32 s18, s0, s19 // (op 3339) 


s_lshl_b32 s0, s13, 24 // (op 3339) 
s_ashr_i32 s19, s0, 31 // (op 3339) 

s_xor_b32 s0, s18, s19 // (op 3339) 

s_sub_i32 s18, s0, s19 // (op 3339) 


s_add_u32 s50, s6, 64 // (op 3339) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3339) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3339) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3339) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3339) 

s_mul_i32 s0, s18, s24 // (op 3339) 

s_sub_i32 s18, s1, s0 // (op 3339) 


s_mul_hi_i32 s19, s16, s9 // (op 3339) 
s_add_i32 s0, s19, s16 // (op 3339) 

s_ashr_i32 s1, s0, 31 // (op 3339) 
s_cmp_eq_i32 s9, 0 // (op 3339) 
s_mov_b32 s19, scc // (op 3339) 
s_cmp_eq_i32 s19, 0 // (op 3339) 
s_cselect_b32 s22, 0, -1 // (op 3339) 

s_and_b32 s19, s46, 31 // (op 3339) 
s_lshl_b32 s23, 1, s19 // (op 3339) 
s_add_i32 s49, s23, s22 // (op 3339) 


s_and_b32 s22, s1, s49 // (op 3339) 


s_add_i32 s1, s0, s22 // (op 3339) 


s_ashr_i32 s0, s1, s19 // (op 3339) 


s_lshl_b32 s1, s46, 24 // (op 3339) 
s_ashr_i32 s19, s1, 31 // (op 3339) 

s_xor_b32 s1, s0, s19 // (op 3339) 

s_sub_i32 s0, s1, s19 // (op 3339) 


s_mul_i32 s1, s15, s24 // (op 3339) 

s_mul_i32 s19, s8, s1 // (op 3339) 

s_mul_i32 s1, s0, s19 // (op 3339) 

s_sub_u32 s19, s16, s1 // (op 3339) 

s_mul_hi_i32 s1, s19, s14 // (op 3339) 
s_add_i32 s22, s1, s19 // (op 3339) 


s_ashr_i32 s1, s22, 31 // (op 3339) 
s_cmp_eq_i32 s14, 0 // (op 3339) 
s_mov_b32 s19, scc // (op 3339) 
s_cmp_eq_i32 s19, 0 // (op 3339) 
s_cselect_b32 s23, 0, -1 // (op 3339) 

s_and_b32 s19, s48, 31 // (op 3339) 
s_lshl_b32 s24, 1, s19 // (op 3339) 
s_add_i32 s25, s24, s23 // (op 3339) 


s_and_b32 s23, s1, s25 // (op 3339) 


s_add_i32 s1, s22, s23 // (op 3339) 


s_ashr_i32 s22, s1, s19 // (op 3339) 


s_lshl_b32 s1, s48, 24 // (op 3339) 
s_ashr_i32 s19, s1, 31 // (op 3339) 

s_xor_b32 s1, s22, s19 // (op 3339) 

s_sub_i32 s22, s1, s19 // (op 3339) 


s_cmp_eq_i32 s0, 0 // (op 3339) 
s_mov_b32 s1, scc // (op 3339) 

s_cmp_eq_i32 s1, 0 // (op 3339) 
s_cselect_b32 s0, s22, s18 // (op 3339) 



v_lshl_add_u32 v26, s0, 6, v25 // (op 3339) 


v_mul_lo_u32 v31, v26, s33 // (op 3339) most significant: low of low * high
v_mul_hi_u32 v25, v26, s32 // (op 3339) most significant: high of low * low
v_mul_lo_u32 v30, v26, s32 // (op 3339) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 3339) most significant: sum


v_add_co_u32 v26, s[0:1], v28, v30 // (op 3339) least significant half
v_addc_co_u32 v27, s[0:1], v29, v31, s[0:1] // (op 3339) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 3339) 


v_lshrrev_b64 v[28:29], 3, v[26:27] // (op 3339) 

v_mov_b32 v25, v28 // (op 3339) convert

s_mul_i32 s1, s32, s7 // (op 3341) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3341) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3341) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3341) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3341) most significant: sum
s_add_u32 s1, s1, s19 // (op 3341) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3341) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3341) 

s_mov_b32 s64, s34 // (op 3341) reg expression
s_mov_b32 s65, s35 // (op 3341) reg expression
s_mov_b32 s66, s0 // (op 3341) reg expression
s_mov_b32 s67, 131072 // (op 3341) call()

s_mul_hi_i32 s0, s16, s10 // (op 3384) 
v_and_b32 v26, -64, v1 // (op 3384) 
s_add_i32 s1, s0, s16 // (op 3384) 

v_sub_u32 v27, v1, v26 // (op 3384) 

s_ashr_i32 s0, s1, 31 // (op 3384) 
v_lshrrev_b32 v26, 6, v1 // (op 3384) 
s_cmp_eq_i32 s10, 0 // (op 3384) 
s_mov_b32 s18, scc // (op 3384) 
s_cmp_eq_i32 s18, 0 // (op 3384) 
s_cselect_b32 s19, 0, -1 // (op 3384) 

s_and_b32 s18, s11, 31 // (op 3384) 
s_lshl_b32 s22, 1, s18 // (op 3384) 
s_add_i32 s23, s22, s19 // (op 3384) 


s_and_b32 s19, s0, s23 // (op 3384) 


s_add_i32 s0, s1, s19 // (op 3384) 


s_ashr_i32 s1, s0, s18 // (op 3384) 


s_lshl_b32 s0, s11, 24 // (op 3384) 
s_ashr_i32 s18, s0, 31 // (op 3384) 

s_xor_b32 s0, s1, s18 // (op 3384) 

s_sub_i32 s1, s0, s18 // (op 3384) 


s_mul_hi_i32 s0, s1, s12 // (op 3384) 
s_add_i32 s18, s0, s1 // (op 3384) 

s_ashr_i32 s0, s18, 31 // (op 3384) 
s_cmp_eq_i32 s12, 0 // (op 3384) 
s_mov_b32 s19, scc // (op 3384) 
s_cmp_eq_i32 s19, 0 // (op 3384) 
s_cselect_b32 s22, 0, -1 // (op 3384) 

s_and_b32 s19, s13, 31 // (op 3384) 
s_lshl_b32 s23, 1, s19 // (op 3384) 
s_add_i32 s24, s23, s22 // (op 3384) 


s_and_b32 s25, s0, s24 // (op 3384) 


s_add_i32 s0, s18, s25 // (op 3384) 


s_ashr_i32 s18, s0, s19 // (op 3384) 


s_lshl_b32 s0, s13, 24 // (op 3384) 
s_ashr_i32 s19, s0, 31 // (op 3384) 

s_xor_b32 s0, s18, s19 // (op 3384) 

s_sub_i32 s18, s0, s19 // (op 3384) 


s_mul_i32 s0, s1, s15 // (op 3384) 

s_sub_u32 s1, s16, s0 // (op 3384) 

s_mul_i32 s0, s18, s15 // (op 3384) 
s_add_u32 s0, s0, s1 // (op 3384) 


s_mul_hi_i32 s1, s16, s9 // (op 3384) 
s_add_i32 s18, s1, s16 // (op 3384) 

s_ashr_i32 s1, s18, 31 // (op 3384) 
s_cmp_eq_i32 s9, 0 // (op 3384) 
s_mov_b32 s19, scc // (op 3384) 
s_cmp_eq_i32 s19, 0 // (op 3384) 
s_cselect_b32 s22, 0, -1 // (op 3384) 

s_and_b32 s19, s46, 31 // (op 3384) 
s_lshl_b32 s23, 1, s19 // (op 3384) 
s_add_i32 s24, s23, s22 // (op 3384) 


s_and_b32 s25, s1, s24 // (op 3384) 


s_add_i32 s1, s18, s25 // (op 3384) 


s_ashr_i32 s18, s1, s19 // (op 3384) 


s_lshl_b32 s1, s46, 24 // (op 3384) 
s_ashr_i32 s19, s1, 31 // (op 3384) 

s_xor_b32 s1, s18, s19 // (op 3384) 

s_sub_i32 s18, s1, s19 // (op 3384) 


s_add_u32 s50, s6, 64 // (op 3384) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3384) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3384) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3384) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3384) 

s_mul_i32 s1, s15, s24 // (op 3384) 

s_mul_i32 s19, s8, s1 // (op 3384) 

s_mul_i32 s1, s18, s19 // (op 3384) 

s_sub_u32 s19, s16, s1 // (op 3384) 

s_mov_b32 s1, s19 // (op 3384) 
s_mul_hi_i32 s22, s1, s14 // (op 3384) 
s_add_i32 s23, s22, s1 // (op 3384) 


s_ashr_i32 s1, s23, 31 // (op 3384) 
s_cmp_eq_i32 s14, 0 // (op 3384) 
s_mov_b32 s22, scc // (op 3384) 
s_cmp_eq_i32 s22, 0 // (op 3384) 
s_cselect_b32 s24, 0, -1 // (op 3384) 

s_and_b32 s22, s48, 31 // (op 3384) 
s_lshl_b32 s25, 1, s22 // (op 3384) 
s_add_i32 s49, s25, s24 // (op 3384) 


s_and_b32 s24, s1, s49 // (op 3384) 


s_add_i32 s1, s23, s24 // (op 3384) 


s_ashr_i32 s23, s1, s22 // (op 3384) 


s_lshl_b32 s1, s48, 24 // (op 3384) 
s_ashr_i32 s22, s1, 31 // (op 3384) 

s_xor_b32 s1, s23, s22 // (op 3384) 

s_sub_i32 s23, s1, s22 // (op 3384) 


s_mul_i32 s1, s47, s15 // (op 3384) 
s_add_u32 s24, s20, 64 // (op 3384) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3384) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3384) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3384) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 3384) 

s_sub_i32 s22, s24, s1 // (op 3384) 

s_mul_i32 s24, s23, s22 // (op 3384) 


s_sub_u32 s25, s19, s24 // (op 3384) 


s_add_u32 s19, s25, s1 // (op 3384) 


s_cmp_eq_i32 s18, 0 // (op 3384) 
s_mov_b32 s1, scc // (op 3384) 

s_cmp_eq_i32 s1, 0 // (op 3384) 
s_cselect_b32 s18, s19, s0 // (op 3384) 



v_lshl_add_u32 v28, s18, 6, v27 // (op 3384) 


v_mul_lo_u32 v27, v26, s36 // (op 3384) 

v_add_u32 v26, v28, v27 // (op 3384) 


s_mul_i32 s1, s36, s41 // (op 3386) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3386) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3386) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3386) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3386) most significant: sum
s_add_u32 s1, s1, s19 // (op 3386) most significant: sum


s_mov_b32 s68, s38 // (op 3386) reg expression
s_mov_b32 s69, s39 // (op 3386) reg expression
s_mov_b32 s70, s0 // (op 3386) reg expression
s_mov_b32 s71, 131072 // (op 3386) call()

s_mul_hi_i32 s0, s16, s10 // (op 3403) 
v_and_b32 v27, -64, v1 // (op 3403) 
s_add_i32 s1, s0, s16 // (op 3403) 

v_sub_u32 v28, v1, v27 // (op 3403) 

s_ashr_i32 s0, s1, 31 // (op 3403) 
v_lshrrev_b32 v27, 6, v1 // (op 3403) 
s_cmp_eq_i32 s10, 0 // (op 3403) 
s_mov_b32 s18, scc // (op 3403) 
s_cmp_eq_i32 s18, 0 // (op 3403) 
s_cselect_b32 s19, 0, -1 // (op 3403) 

s_and_b32 s18, s11, 31 // (op 3403) 
s_lshl_b32 s22, 1, s18 // (op 3403) 
s_add_i32 s23, s22, s19 // (op 3403) 


s_and_b32 s19, s0, s23 // (op 3403) 


s_add_i32 s0, s1, s19 // (op 3403) 


s_ashr_i32 s1, s0, s18 // (op 3403) 


s_lshl_b32 s0, s11, 24 // (op 3403) 
s_ashr_i32 s18, s0, 31 // (op 3403) 

s_xor_b32 s0, s1, s18 // (op 3403) 

s_sub_i32 s1, s0, s18 // (op 3403) 


s_mul_hi_i32 s0, s1, s12 // (op 3403) 
s_add_i32 s18, s0, s1 // (op 3403) 

s_ashr_i32 s0, s18, 31 // (op 3403) 
s_cmp_eq_i32 s12, 0 // (op 3403) 
s_mov_b32 s19, scc // (op 3403) 
s_cmp_eq_i32 s19, 0 // (op 3403) 
s_cselect_b32 s22, 0, -1 // (op 3403) 

s_and_b32 s19, s13, 31 // (op 3403) 
s_lshl_b32 s23, 1, s19 // (op 3403) 
s_add_i32 s24, s23, s22 // (op 3403) 


s_and_b32 s25, s0, s24 // (op 3403) 


s_add_i32 s0, s18, s25 // (op 3403) 


s_ashr_i32 s18, s0, s19 // (op 3403) 


s_lshl_b32 s0, s13, 24 // (op 3403) 
s_ashr_i32 s19, s0, 31 // (op 3403) 

s_xor_b32 s0, s18, s19 // (op 3403) 

s_sub_i32 s18, s0, s19 // (op 3403) 


s_mul_i32 s0, s1, s15 // (op 3403) 

s_sub_u32 s1, s16, s0 // (op 3403) 

s_mul_i32 s0, s18, s15 // (op 3403) 
s_add_u32 s0, s0, s1 // (op 3403) 


s_mul_hi_i32 s1, s16, s9 // (op 3403) 
s_add_i32 s18, s1, s16 // (op 3403) 

s_ashr_i32 s1, s18, 31 // (op 3403) 
s_cmp_eq_i32 s9, 0 // (op 3403) 
s_mov_b32 s19, scc // (op 3403) 
s_cmp_eq_i32 s19, 0 // (op 3403) 
s_cselect_b32 s22, 0, -1 // (op 3403) 

s_and_b32 s19, s46, 31 // (op 3403) 
s_lshl_b32 s23, 1, s19 // (op 3403) 
s_add_i32 s24, s23, s22 // (op 3403) 


s_and_b32 s25, s1, s24 // (op 3403) 


s_add_i32 s1, s18, s25 // (op 3403) 


s_ashr_i32 s18, s1, s19 // (op 3403) 


s_lshl_b32 s1, s46, 24 // (op 3403) 
s_ashr_i32 s19, s1, 31 // (op 3403) 

s_xor_b32 s1, s18, s19 // (op 3403) 

s_sub_i32 s18, s1, s19 // (op 3403) 


s_add_u32 s50, s6, 64 // (op 3403) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3403) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3403) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3403) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3403) 

s_mul_i32 s1, s15, s24 // (op 3403) 

s_mul_i32 s19, s8, s1 // (op 3403) 

s_mul_i32 s1, s18, s19 // (op 3403) 

s_sub_u32 s19, s16, s1 // (op 3403) 

s_mov_b32 s1, s19 // (op 3403) 
s_mul_hi_i32 s22, s1, s14 // (op 3403) 
s_add_i32 s23, s22, s1 // (op 3403) 


s_ashr_i32 s1, s23, 31 // (op 3403) 
s_cmp_eq_i32 s14, 0 // (op 3403) 
s_mov_b32 s22, scc // (op 3403) 
s_cmp_eq_i32 s22, 0 // (op 3403) 
s_cselect_b32 s24, 0, -1 // (op 3403) 

s_and_b32 s22, s48, 31 // (op 3403) 
s_lshl_b32 s25, 1, s22 // (op 3403) 
s_add_i32 s49, s25, s24 // (op 3403) 


s_and_b32 s24, s1, s49 // (op 3403) 


s_add_i32 s1, s23, s24 // (op 3403) 


s_ashr_i32 s23, s1, s22 // (op 3403) 


s_lshl_b32 s1, s48, 24 // (op 3403) 
s_ashr_i32 s22, s1, 31 // (op 3403) 

s_xor_b32 s1, s23, s22 // (op 3403) 

s_sub_i32 s23, s1, s22 // (op 3403) 


s_mul_i32 s1, s47, s15 // (op 3403) 
s_add_u32 s24, s20, 64 // (op 3403) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3403) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3403) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3403) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 3403) 

s_sub_i32 s22, s24, s1 // (op 3403) 

s_mul_i32 s24, s23, s22 // (op 3403) 


s_sub_u32 s25, s19, s24 // (op 3403) 


s_add_u32 s19, s25, s1 // (op 3403) 


s_cmp_eq_i32 s18, 0 // (op 3403) 
s_mov_b32 s1, scc // (op 3403) 

s_cmp_eq_i32 s1, 0 // (op 3403) 
s_cselect_b32 s18, s19, s0 // (op 3403) 



v_lshl_add_u32 v29, s18, 6, v28 // (op 3403) 


v_mul_lo_u32 v28, v27, s36 // (op 3403) 

v_add_u32 v27, v29, v28 // (op 3403) 


s_mul_i32 s1, s36, s41 // (op 3405) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3405) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3405) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3405) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3405) most significant: sum
s_add_u32 s1, s1, s19 // (op 3405) most significant: sum


s_mov_b32 s68, s38 // (op 3405) reg expression
s_mov_b32 s69, s39 // (op 3405) reg expression
s_mov_b32 s70, s0 // (op 3405) reg expression
s_mov_b32 s71, 131072 // (op 3405) call()

s_mul_hi_i32 s0, s16, s10 // (op 3448) 
v_lshrrev_b32 v28, 2, v1 // (op 3448) 
s_add_i32 s1, s0, s16 // (op 3448) 

v_and_b32 v29, -4, v1 // (op 3448) 
s_ashr_i32 s0, s1, 31 // (op 3448) 
v_sub_u32 v30, v1, v29 // (op 3448) 

s_cmp_eq_i32 s10, 0 // (op 3448) 
s_mov_b32 s18, scc // (op 3448) 
s_cmp_eq_i32 s18, 0 // (op 3448) 
s_cselect_b32 s19, 0, -1 // (op 3448) 

s_and_b32 s18, s11, 31 // (op 3448) 
s_lshl_b32 s22, 1, s18 // (op 3448) 
s_add_i32 s23, s22, s19 // (op 3448) 


s_and_b32 s19, s0, s23 // (op 3448) 


s_add_i32 s0, s1, s19 // (op 3448) 


s_ashr_i32 s1, s0, s18 // (op 3448) 


s_lshl_b32 s0, s11, 24 // (op 3448) 
s_ashr_i32 s18, s0, 31 // (op 3448) 

s_xor_b32 s0, s1, s18 // (op 3448) 

s_sub_i32 s1, s0, s18 // (op 3448) 


s_mul_hi_i32 s0, s1, s12 // (op 3448) 
s_add_i32 s18, s0, s1 // (op 3448) 

s_ashr_i32 s0, s18, 31 // (op 3448) 
s_cmp_eq_i32 s12, 0 // (op 3448) 
s_mov_b32 s19, scc // (op 3448) 
s_cmp_eq_i32 s19, 0 // (op 3448) 
s_cselect_b32 s22, 0, -1 // (op 3448) 

s_and_b32 s19, s13, 31 // (op 3448) 
s_lshl_b32 s23, 1, s19 // (op 3448) 
s_add_i32 s24, s23, s22 // (op 3448) 


s_and_b32 s25, s0, s24 // (op 3448) 


s_add_i32 s0, s18, s25 // (op 3448) 


s_ashr_i32 s18, s0, s19 // (op 3448) 


s_lshl_b32 s0, s13, 24 // (op 3448) 
s_ashr_i32 s19, s0, 31 // (op 3448) 

s_xor_b32 s0, s18, s19 // (op 3448) 

s_sub_i32 s18, s0, s19 // (op 3448) 


s_add_u32 s50, s6, 64 // (op 3448) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3448) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3448) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3448) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3448) 

s_mul_i32 s0, s18, s24 // (op 3448) 

s_sub_i32 s18, s1, s0 // (op 3448) 


s_mul_hi_i32 s19, s16, s9 // (op 3448) 
s_add_i32 s0, s19, s16 // (op 3448) 

s_ashr_i32 s1, s0, 31 // (op 3448) 
s_cmp_eq_i32 s9, 0 // (op 3448) 
s_mov_b32 s19, scc // (op 3448) 
s_cmp_eq_i32 s19, 0 // (op 3448) 
s_cselect_b32 s22, 0, -1 // (op 3448) 

s_and_b32 s19, s46, 31 // (op 3448) 
s_lshl_b32 s23, 1, s19 // (op 3448) 
s_add_i32 s49, s23, s22 // (op 3448) 


s_and_b32 s22, s1, s49 // (op 3448) 


s_add_i32 s1, s0, s22 // (op 3448) 


s_ashr_i32 s0, s1, s19 // (op 3448) 


s_lshl_b32 s1, s46, 24 // (op 3448) 
s_ashr_i32 s19, s1, 31 // (op 3448) 

s_xor_b32 s1, s0, s19 // (op 3448) 

s_sub_i32 s0, s1, s19 // (op 3448) 


s_mul_i32 s1, s15, s24 // (op 3448) 

s_mul_i32 s19, s8, s1 // (op 3448) 

s_mul_i32 s1, s0, s19 // (op 3448) 

s_sub_u32 s19, s16, s1 // (op 3448) 

s_mul_hi_i32 s1, s19, s14 // (op 3448) 
s_add_i32 s22, s1, s19 // (op 3448) 


s_ashr_i32 s1, s22, 31 // (op 3448) 
s_cmp_eq_i32 s14, 0 // (op 3448) 
s_mov_b32 s19, scc // (op 3448) 
s_cmp_eq_i32 s19, 0 // (op 3448) 
s_cselect_b32 s23, 0, -1 // (op 3448) 

s_and_b32 s19, s48, 31 // (op 3448) 
s_lshl_b32 s24, 1, s19 // (op 3448) 
s_add_i32 s25, s24, s23 // (op 3448) 


s_and_b32 s23, s1, s25 // (op 3448) 


s_add_i32 s1, s22, s23 // (op 3448) 


s_ashr_i32 s22, s1, s19 // (op 3448) 


s_lshl_b32 s1, s48, 24 // (op 3448) 
s_ashr_i32 s19, s1, 31 // (op 3448) 

s_xor_b32 s1, s22, s19 // (op 3448) 

s_sub_i32 s22, s1, s19 // (op 3448) 


s_cmp_eq_i32 s0, 0 // (op 3448) 
s_mov_b32 s1, scc // (op 3448) 

s_cmp_eq_i32 s1, 0 // (op 3448) 
s_cselect_b32 s0, s22, s18 // (op 3448) 



v_lshl_add_u32 v29, s0, 6, v28 // (op 3448) 


v_mul_lo_u32 v28, v29, s42 // (op 3448) 

v_add_u32 v29, v30, v28 // (op 3448) 


s_mul_i32 s1, s42, s7 // (op 3450) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3450) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3450) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3450) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3450) most significant: sum
s_add_u32 s1, s1, s19 // (op 3450) most significant: sum


s_mov_b32 s72, s44 // (op 3450) reg expression
s_mov_b32 s73, s45 // (op 3450) reg expression
s_mov_b32 s74, s0 // (op 3450) reg expression
s_mov_b32 s75, 131072 // (op 3450) call()

s_mul_hi_i32 s0, s16, s10 // (op 3467) 
v_lshrrev_b32 v28, 2, v1 // (op 3467) 
s_add_i32 s1, s0, s16 // (op 3467) 

v_and_b32 v30, -4, v1 // (op 3467) 
s_ashr_i32 s0, s1, 31 // (op 3467) 
v_sub_u32 v31, v1, v30 // (op 3467) 

s_cmp_eq_i32 s10, 0 // (op 3467) 
s_mov_b32 s18, scc // (op 3467) 
s_cmp_eq_i32 s18, 0 // (op 3467) 
s_cselect_b32 s19, 0, -1 // (op 3467) 

s_and_b32 s18, s11, 31 // (op 3467) 
s_lshl_b32 s22, 1, s18 // (op 3467) 
s_add_i32 s23, s22, s19 // (op 3467) 


s_and_b32 s19, s0, s23 // (op 3467) 


s_add_i32 s0, s1, s19 // (op 3467) 


s_ashr_i32 s1, s0, s18 // (op 3467) 


s_lshl_b32 s0, s11, 24 // (op 3467) 
s_ashr_i32 s18, s0, 31 // (op 3467) 

s_xor_b32 s0, s1, s18 // (op 3467) 

s_sub_i32 s1, s0, s18 // (op 3467) 


s_mul_hi_i32 s0, s1, s12 // (op 3467) 
s_add_i32 s18, s0, s1 // (op 3467) 

s_ashr_i32 s0, s18, 31 // (op 3467) 
s_cmp_eq_i32 s12, 0 // (op 3467) 
s_mov_b32 s19, scc // (op 3467) 
s_cmp_eq_i32 s19, 0 // (op 3467) 
s_cselect_b32 s22, 0, -1 // (op 3467) 

s_and_b32 s19, s13, 31 // (op 3467) 
s_lshl_b32 s23, 1, s19 // (op 3467) 
s_add_i32 s24, s23, s22 // (op 3467) 


s_and_b32 s25, s0, s24 // (op 3467) 


s_add_i32 s0, s18, s25 // (op 3467) 


s_ashr_i32 s18, s0, s19 // (op 3467) 


s_lshl_b32 s0, s13, 24 // (op 3467) 
s_ashr_i32 s19, s0, 31 // (op 3467) 

s_xor_b32 s0, s18, s19 // (op 3467) 

s_sub_i32 s18, s0, s19 // (op 3467) 


s_add_u32 s50, s6, 64 // (op 3467) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3467) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3467) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3467) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3467) 

s_mul_i32 s0, s18, s24 // (op 3467) 

s_sub_i32 s18, s1, s0 // (op 3467) 


s_mul_hi_i32 s19, s16, s9 // (op 3467) 
s_add_i32 s0, s19, s16 // (op 3467) 

s_ashr_i32 s1, s0, 31 // (op 3467) 
s_cmp_eq_i32 s9, 0 // (op 3467) 
s_mov_b32 s19, scc // (op 3467) 
s_cmp_eq_i32 s19, 0 // (op 3467) 
s_cselect_b32 s22, 0, -1 // (op 3467) 

s_and_b32 s19, s46, 31 // (op 3467) 
s_lshl_b32 s23, 1, s19 // (op 3467) 
s_add_i32 s49, s23, s22 // (op 3467) 


s_and_b32 s22, s1, s49 // (op 3467) 


s_add_i32 s1, s0, s22 // (op 3467) 


s_ashr_i32 s0, s1, s19 // (op 3467) 


s_lshl_b32 s1, s46, 24 // (op 3467) 
s_ashr_i32 s19, s1, 31 // (op 3467) 

s_xor_b32 s1, s0, s19 // (op 3467) 

s_sub_i32 s0, s1, s19 // (op 3467) 


s_mul_i32 s1, s15, s24 // (op 3467) 

s_mul_i32 s19, s8, s1 // (op 3467) 

s_mul_i32 s1, s0, s19 // (op 3467) 

s_sub_u32 s19, s16, s1 // (op 3467) 

s_mul_hi_i32 s1, s19, s14 // (op 3467) 
s_add_i32 s22, s1, s19 // (op 3467) 


s_ashr_i32 s1, s22, 31 // (op 3467) 
s_cmp_eq_i32 s14, 0 // (op 3467) 
s_mov_b32 s19, scc // (op 3467) 
s_cmp_eq_i32 s19, 0 // (op 3467) 
s_cselect_b32 s23, 0, -1 // (op 3467) 

s_and_b32 s19, s48, 31 // (op 3467) 
s_lshl_b32 s24, 1, s19 // (op 3467) 
s_add_i32 s25, s24, s23 // (op 3467) 


s_and_b32 s23, s1, s25 // (op 3467) 


s_add_i32 s1, s22, s23 // (op 3467) 


s_ashr_i32 s22, s1, s19 // (op 3467) 


s_lshl_b32 s1, s48, 24 // (op 3467) 
s_ashr_i32 s19, s1, 31 // (op 3467) 

s_xor_b32 s1, s22, s19 // (op 3467) 

s_sub_i32 s22, s1, s19 // (op 3467) 


s_cmp_eq_i32 s0, 0 // (op 3467) 
s_mov_b32 s1, scc // (op 3467) 

s_cmp_eq_i32 s1, 0 // (op 3467) 
s_cselect_b32 s0, s22, s18 // (op 3467) 



v_lshl_add_u32 v30, s0, 6, v28 // (op 3467) 


v_mul_lo_u32 v28, v30, s42 // (op 3467) 

v_add_u32 v30, v31, v28 // (op 3467) 


s_mul_i32 s1, s42, s7 // (op 3469) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3469) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3469) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3469) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3469) most significant: sum
s_add_u32 s1, s1, s19 // (op 3469) most significant: sum


s_mov_b32 s72, s44 // (op 3469) reg expression
s_mov_b32 s73, s45 // (op 3469) reg expression
s_mov_b32 s74, s0 // (op 3469) reg expression
s_mov_b32 s75, 131072 // (op 3469) call()




buffer_load_dwordx4 v[32:35], v24, s[64:67], 0 offen // (op 2968) Load value



buffer_load_ubyte v28, v29, s[72:75], 0 offen offset: 0 // (op 2971) Load value


v_mov_b32 v31, v22 // (op 2974) reg expression

buffer_load_dwordx4 v[36:39], v31, s[52:55], 0 offen // (op 2974) Load value



v_mov_b32 v31, v26 // (op 2977) reg expression

buffer_load_ubyte v40, v31, s[68:71], 0 offen offset: 0 // (op 2977) Load value




buffer_load_dwordx4 v[42:45], v24, s[64:67], 0 offen offset:64 // (op 2980) Load value



buffer_load_ubyte v31, v29, s[72:75], 0 offen offset:4 // (op 2983) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 2986) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 2986) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 2986) 

v_add_u32 v41, v22, s0 // (op 2986) 


buffer_load_dwordx4 v[46:49], v41, s[52:55], 0 offen // (op 2986) Load value



s_lshl_b32 s0, s36, 2 // (op 2989) 
v_add_u32 v41, v26, s0 // (op 2989) 


buffer_load_ubyte v50, v41, s[68:71], 0 offen offset: 0 // (op 2989) Load value




buffer_load_dwordx4 v[52:55], v24, s[64:67], 0 offen offset:128 // (op 2992) Load value



buffer_load_ubyte v41, v29, s[72:75], 0 offen offset:8 // (op 2995) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 2998) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 2998) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 2998) 

s_lshl_b32 s18, s0, 1 // (op 2998) 

v_add_u32 v51, v22, s18 // (op 2998) 


buffer_load_dwordx4 v[56:59], v51, s[52:55], 0 offen // (op 2998) Load value



s_lshl_b32 s0, s36, 3 // (op 3001) 
v_add_u32 v51, v26, s0 // (op 3001) 


buffer_load_ubyte v60, v51, s[68:71], 0 offen offset: 0 // (op 3001) Load value




buffer_load_dwordx4 v[62:65], v24, s[64:67], 0 offen offset:192 // (op 3004) Load value




buffer_load_ubyte v24, v29, s[72:75], 0 offen offset:12 // (op 3007) Load value



s_lshl_b64 s[0:1], s[26:27], 7 // (op 3010) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 3010) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 3010) 

s_mul_i32 s18, 3, s0 // (op 3010) 

v_add_u32 v29, v22, s18 // (op 3010) 


buffer_load_dwordx4 v[66:69], v29, s[52:55], 0 offen // (op 3010) Load value




s_lshl_b32 s0, s36, 2 // (op 3013) 
s_mul_i32 s1, 3, s0 // (op 3013) 

v_add_u32 v22, v26, s1 // (op 3013) 


buffer_load_ubyte v29, v22, s[68:71], 0 offen offset: 0 // (op 3013) Load value


v_and_b32 v22, -4, v1 // (op 3597) 
v_sub_u32 v26, v1, v22 // (op 3597) 

v_lshlrev_b32 v22, 5, v26 // (op 3597) 

v_lshrrev_b32 v26, 2, v1 // (op 3597) 
v_lshl_add_u32 v51, v26, 7, v22 // (op 3597) 


v_lshlrev_b32 v22, 2, v51 // (op 3597) 

v_lshrrev_b32 v26, 3, v22 // (op 3597) 




s_waitcnt vmcnt(15)
ds_write_b128 v26, v[32:35] // (op 3015) Store local data 

v_and_b32 v22, -4, v1 // (op 3713) 
v_sub_u32 v26, v1, v22 // (op 3713) 

v_and_b32 v22, 4294967292, v1 // (op 3713) 
v_add_u32 v51, v22, v26 // (op 3713) 





s_waitcnt vmcnt(14)
ds_write_b8 v51, v28 offset:4096 // (op 3016) Store local data 

v_and_b32 v22, -2, v1 // (op 3539) 
v_sub_u32 v26, v1, v22 // (op 3539) 

v_lshlrev_b32 v22, 5, v26 // (op 3539) 

v_lshrrev_b32 v26, 1, v1 // (op 3539) 
v_lshl_add_u32 v51, v26, 6, v22 // (op 3539) 


v_lshlrev_b32 v22, 2, v51 // (op 3539) 

v_lshrrev_b32 v26, 3, v22 // (op 3539) 




s_waitcnt vmcnt(13)
ds_write_b128 v26, v[36:39] offset:4352 // (op 3017) Store local data 

v_and_b32 v22, -64, v1 // (op 3655) 
v_sub_u32 v26, v1, v22 // (op 3655) 

v_and_b32 v22, 4294967232, v1 // (op 3655) 
v_add_u32 v51, v22, v26 // (op 3655) 





s_waitcnt vmcnt(12)
ds_write_b8 v51, v40 offset:8448 // (op 3018) Store local data 

v_and_b32 v22, -2, v1 // (op 3510) 
v_sub_u32 v26, v1, v22 // (op 3510) 

v_lshlrev_b32 v22, 5, v26 // (op 3510) 

v_lshrrev_b32 v26, 1, v1 // (op 3510) 
v_lshl_add_u32 v51, v26, 6, v22 // (op 3510) 


v_lshlrev_b32 v22, 2, v51 // (op 3510) 

v_lshrrev_b32 v26, 3, v22 // (op 3510) 

v_and_b32 v22, -4, v1 // (op 3568) 
v_sub_u32 v51, v1, v22 // (op 3568) 

v_lshlrev_b32 v22, 5, v51 // (op 3568) 

v_lshrrev_b32 v51, 2, v1 // (op 3568) 
v_lshl_add_u32 v61, v51, 7, v22 // (op 3568) 


v_lshlrev_b32 v22, 2, v61 // (op 3568) 

v_lshrrev_b32 v51, 3, v22 // (op 3568) 

v_and_b32 v22, -64, v1 // (op 3626) 
v_sub_u32 v61, v1, v22 // (op 3626) 

v_and_b32 v22, 4294967232, v1 // (op 3626) 
v_add_u32 v70, v22, v61 // (op 3626) 


v_and_b32 v22, -4, v1 // (op 3684) 
v_sub_u32 v61, v1, v22 // (op 3684) 

v_and_b32 v22, 4294967292, v1 // (op 3684) 
v_add_u32 v71, v22, v61 // (op 3684) 


v_and_b32 v22, -2, v1 // (op 3765) 
v_sub_u32 v61, v1, v22 // (op 3765) 

v_lshlrev_b32 v22, 5, v61 // (op 3765) 

v_lshrrev_b32 v61, 1, v1 // (op 3765) 
v_lshl_add_u32 v72, v61, 6, v22 // (op 3765) 


v_lshlrev_b32 v22, 2, v72 // (op 3765) 

v_lshrrev_b32 v61, 3, v22 // (op 3765) 

v_and_b32 v22, -4, v1 // (op 3794) 
v_sub_u32 v72, v1, v22 // (op 3794) 

v_lshlrev_b32 v22, 5, v72 // (op 3794) 

v_lshrrev_b32 v72, 2, v1 // (op 3794) 
v_lshl_add_u32 v73, v72, 7, v22 // (op 3794) 


v_lshlrev_b32 v22, 2, v73 // (op 3794) 

v_lshrrev_b32 v72, 3, v22 // (op 3794) 

v_and_b32 v22, -64, v1 // (op 3823) 
v_sub_u32 v73, v1, v22 // (op 3823) 

v_and_b32 v22, 4294967232, v1 // (op 3823) 
v_add_u32 v74, v22, v73 // (op 3823) 


v_and_b32 v22, -4, v1 // (op 3852) 
v_sub_u32 v73, v1, v22 // (op 3852) 

v_and_b32 v22, 4294967292, v1 // (op 3852) 
v_add_u32 v75, v22, v73 // (op 3852) 


v_and_b32 v22, -2, v1 // (op 3881) 
v_sub_u32 v73, v1, v22 // (op 3881) 

v_lshlrev_b32 v22, 5, v73 // (op 3881) 

v_lshrrev_b32 v73, 1, v1 // (op 3881) 
v_lshl_add_u32 v76, v73, 6, v22 // (op 3881) 


v_lshlrev_b32 v22, 2, v76 // (op 3881) 

v_lshrrev_b32 v73, 3, v22 // (op 3881) 

v_and_b32 v22, -4, v1 // (op 3910) 
v_sub_u32 v76, v1, v22 // (op 3910) 

v_lshlrev_b32 v22, 5, v76 // (op 3910) 

v_lshrrev_b32 v76, 2, v1 // (op 3910) 
v_lshl_add_u32 v77, v76, 7, v22 // (op 3910) 


v_lshlrev_b32 v22, 2, v77 // (op 3910) 

v_lshrrev_b32 v76, 3, v22 // (op 3910) 

v_and_b32 v22, -64, v1 // (op 3939) 
v_sub_u32 v77, v1, v22 // (op 3939) 

v_and_b32 v22, 4294967232, v1 // (op 3939) 
v_add_u32 v78, v22, v77 // (op 3939) 


v_and_b32 v22, -4, v1 // (op 3968) 
v_sub_u32 v77, v1, v22 // (op 3968) 

v_and_b32 v22, 4294967292, v1 // (op 3968) 
v_add_u32 v79, v22, v77 // (op 3968) 


v_and_b32 v22, -2, v1 // (op 3997) 
v_sub_u32 v77, v1, v22 // (op 3997) 

v_lshlrev_b32 v22, 5, v77 // (op 3997) 

v_lshrrev_b32 v77, 1, v1 // (op 3997) 
v_lshl_add_u32 v80, v77, 6, v22 // (op 3997) 


v_lshlrev_b32 v22, 2, v80 // (op 3997) 

v_lshrrev_b32 v77, 3, v22 // (op 3997) 

v_and_b32 v22, -4, v1 // (op 4026) 
v_sub_u32 v80, v1, v22 // (op 4026) 

v_lshlrev_b32 v22, 5, v80 // (op 4026) 

v_lshrrev_b32 v80, 2, v1 // (op 4026) 
v_lshl_add_u32 v81, v80, 7, v22 // (op 4026) 


v_lshlrev_b32 v22, 2, v81 // (op 4026) 

v_lshrrev_b32 v80, 3, v22 // (op 4026) 

v_and_b32 v22, -64, v1 // (op 4055) 
v_sub_u32 v81, v1, v22 // (op 4055) 

v_and_b32 v22, 4294967232, v1 // (op 4055) 
v_add_u32 v82, v22, v81 // (op 4055) 


v_and_b32 v22, -4, v1 // (op 4084) 
v_sub_u32 v81, v1, v22 // (op 4084) 

v_and_b32 v22, 4294967292, v1 // (op 4084) 
v_add_u32 v83, v22, v81 // (op 4084) 


s_mov_b32 s0, 0 // (op 881) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 880) 
s_mov_b32 s1, s18 // (op 880) convert

s_and_b32 s18, s1, 4294967292 // (op 880) 

s_cmp_lt_u32 s0, s18 // (op 880) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoop_880 // (op 880) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoop_880 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoop_880:


s_add_u32 s0, s0, 4 // (op 882) 
v_mov_b32 v22, v25 // (op 10) 

buffer_load_dwordx4 v[32:35], v22, s[64:67], 0 offen offset:256 // (op 10) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4184) 





v_mov_b32 v81, v30 // (op 25) 

buffer_load_ubyte v28, v81, s[72:75], 0 offen offset:16 // (op 25) Load value
ds_read_b64_tr_b4 v[84:85], v3 offset:4352 // (op 70) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v3 offset:4864 // (op 70) Transpose load local data 
v_mov_b32 v88, v23 // (op 4) 
s_barrier  // (op 4197) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 4) 



s_lshl_b64 s[50:51], s[18:19], 2 // (op 4) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 2 // (op 4) 

v_add_u32 v89, v88, s1 // (op 4) 


ds_read_b128 v[90:93], v4 // (op 75) Load local data 
buffer_load_dwordx4 v[36:39], v89, s[52:55], 0 offen // (op 4) Load value



v_mov_b32 v89, v27 // (op 17) 
s_lshl_b32 s1, s36, 4 // (op 17) 
v_add_u32 v94, v89, s1 // (op 17) 


s_waitcnt lgkmcnt(3)
s_barrier  // (op 4211) 
buffer_load_ubyte v40, v94, s[68:71], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 4248) 
ds_read_u8 v94, v0 offset:8448 // (op 80) Load local data 



s_barrier  // (op 4225) 
s_waitcnt vmcnt(15)
ds_write_b128 v72, v[42:45] offset:8704 // (op 253) Store local data 



ds_read_u8 v95, v5 offset:4096 // (op 85) Load local data 



s_barrier  // (op 4260) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 



ds_read_b128 v[96:99], v4 offset:2048 // (op 517) Load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v75, v31 offset:12800 // (op 255) Store local data 



ds_read_u8 v100, v5 offset:4224 // (op 521) Load local data 



s_barrier  // (op 4242) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 





ds_read_b64_tr_b4 v[84:85], v3 offset:4368 // (op 914) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:4880 // (op 914) Transpose load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v61, v[46:49] offset:13056 // (op 243) Store local data 


ds_read_u8 v94, v0 offset:8480 // (op 917) Load local data 
s_barrier  // (op 4254) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[90:93], a[8:11], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1011) 





s_waitcnt vmcnt(12)
ds_write_b8 v74, v50 offset:17152 // (op 254) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[96:99], a[12:15], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 919) 







s_barrier  // (op 4239) 
buffer_load_dwordx4 v[42:45], v22, s[64:67], 0 offen offset:320 // (op 256) Load value






ds_read_b64_tr_b4 v[84:85], v7 offset:13056 // (op 245) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:13568 // (op 245) Transpose load local data 
buffer_load_ubyte v31, v81, s[72:75], 0 offen offset:20 // (op 258) Load value


s_barrier  // (op 4245) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 242) 



s_lshl_b64 s[50:51], s[18:19], 2 // (op 242) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 242) 

s_mul_i32 s1, 5, s18 // (op 242) 

v_add_u32 v90, v88, s1 // (op 242) 


ds_read_b128 v[92:95], v8 offset:8704 // (op 247) Load local data 
buffer_load_dwordx4 v[46:49], v90, s[52:55], 0 offen // (op 242) Load value



s_lshl_b32 s1, s36, 2 // (op 257) 
s_mul_i32 s18, 5, s1 // (op 257) 

v_add_u32 v90, v89, s18 // (op 257) 


s_waitcnt lgkmcnt(3)
s_barrier  // (op 4251) 
buffer_load_ubyte v50, v90, s[68:71], 0 offen offset: 0 // (op 257) Load value



s_barrier  // (op 4272) 
ds_read_u8 v90, v6 offset:17152 // (op 249) Load local data 



s_barrier  // (op 4257) 
s_waitcnt vmcnt(15)
ds_write_b128 v76, v[52:55] offset:17408 // (op 293) Store local data 



ds_read_u8 v91, v9 offset:12800 // (op 251) Load local data 



s_barrier  // (op 4284) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[92:95], a[0:3], v90, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 



ds_read_b128 v[96:99], v8 offset:10752 // (op 529) Load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v79, v41 offset:21504 // (op 295) Store local data 



s_barrier  // (op 4385) 
s_barrier  // (op 4266) 
ds_read_u8 v100, v9 offset:12928 // (op 533) Load local data 



s_waitcnt vmcnt(13)
ds_write_b128 v73, v[56:59] offset:21760 // (op 283) Store local data 



s_barrier  // (op 4389) 
s_barrier  // (op 4278) 
ds_read_b64_tr_b4 v[102:103], v7 offset:13072 // (op 924) Transpose load local data 



ds_read_b64_tr_b4 v[104:105], v7 offset:13584 // (op 924) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v90, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 




s_barrier  // (op 4429) 
ds_read_u8 v90, v6 offset:17184 // (op 927) Load local data 
s_waitcnt vmcnt(12)
ds_write_b8 v78, v60 offset:25856 // (op 294) Store local data 
s_barrier  // (op 4433) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[92:95], a[8:11], v90, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1012) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[96:99], a[12:15], v90, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 920) 







s_barrier  // (op 4263) 
buffer_load_dwordx4 v[52:55], v22, s[64:67], 0 offen offset:384 // (op 296) Load value






ds_read_b64_tr_b4 v[84:85], v11 offset:21760 // (op 285) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v11 offset:22272 // (op 285) Transpose load local data 
buffer_load_ubyte v41, v81, s[72:75], 0 offen offset:24 // (op 298) Load value


s_barrier  // (op 4269) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 282) 



s_lshl_b64 s[50:51], s[18:19], 2 // (op 282) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 282) 

s_mul_i32 s1, 6, s18 // (op 282) 

v_add_u32 v90, v88, s1 // (op 282) 


ds_read_b128 v[92:95], v12 offset:17408 // (op 287) Load local data 
buffer_load_dwordx4 v[56:59], v90, s[52:55], 0 offen // (op 282) Load value



s_lshl_b32 s1, s36, 2 // (op 297) 
s_mul_i32 s18, 6, s1 // (op 297) 

v_add_u32 v90, v89, s18 // (op 297) 


s_waitcnt lgkmcnt(3)
s_barrier  // (op 4275) 
buffer_load_ubyte v60, v90, s[68:71], 0 offen offset: 0 // (op 297) Load value



s_barrier  // (op 4296) 
ds_read_u8 v90, v10 offset:25856 // (op 289) Load local data 



s_barrier  // (op 4281) 
s_waitcnt vmcnt(15)
ds_write_b128 v80, v[62:65] offset:8704 // (op 333) Store local data 



ds_read_u8 v91, v13 offset:21504 // (op 291) Load local data 



s_barrier  // (op 4308) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[92:95], a[0:3], v90, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 



ds_read_b128 v[96:99], v12 offset:19456 // (op 537) Load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v83, v24 offset:12800 // (op 335) Store local data 



s_barrier  // (op 4393) 
s_barrier  // (op 4290) 
ds_read_u8 v100, v13 offset:21632 // (op 541) Load local data 



s_waitcnt vmcnt(13)
ds_write_b128 v77, v[66:69] offset:13056 // (op 323) Store local data 



s_barrier  // (op 4397) 
s_barrier  // (op 4302) 
ds_read_b64_tr_b4 v[102:103], v11 offset:21776 // (op 930) Transpose load local data 



ds_read_b64_tr_b4 v[104:105], v11 offset:22288 // (op 930) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v90, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 




s_barrier  // (op 4437) 
ds_read_u8 v90, v10 offset:25888 // (op 933) Load local data 
s_waitcnt vmcnt(12)
ds_write_b8 v82, v29 offset:17152 // (op 334) Store local data 
s_barrier  // (op 4441) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[92:95], a[8:11], v90, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1013) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[96:99], a[12:15], v90, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 921) 







s_barrier  // (op 4287) 
buffer_load_dwordx4 v[62:65], v22, s[64:67], 0 offen offset:448 // (op 336) Load value




ds_read_b64_tr_b4 v[84:85], v15 offset:13056 // (op 325) Transpose load local data 



ds_read_b64_tr_b4 v[86:87], v15 offset:13568 // (op 325) Transpose load local data 
v_add_u32 v25, 256, v25 // (op 3331) 
s_barrier  // (op 4293) 
buffer_load_ubyte v24, v81, s[72:75], 0 offen offset:28 // (op 338) Load value




ds_read_b128 v[90:93], v16 offset:8704 // (op 327) Load local data 


v_add_u32 v30, 16, v30 // (op 3459) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 322) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 322) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 322) 

s_mul_i32 s1, 7, s18 // (op 322) 

v_add_u32 v22, v88, s1 // (op 322) 


s_waitcnt lgkmcnt(3)
s_barrier  // (op 4299) 
buffer_load_dwordx4 v[66:69], v22, s[52:55], 0 offen // (op 322) Load value




ds_read_u8 v22, v14 offset:17152 // (op 329) Load local data 


s_barrier  // (op 4305) 
s_lshl_b32 s1, s36, 2 // (op 337) 



ds_read_u8 v81, v17 offset:12800 // (op 331) Load local data 
s_mul_i32 s18, 7, s1 // (op 337) 




v_add_u32 v88, v89, s18 // (op 337) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 3267) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 3267) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 3267) 

s_lshl_b32 s1, s18, 2 // (op 3267) 

v_add_u32 v23, v23, s1 // (op 3267) 

ds_read_b128 v[94:97], v16 offset:10752 // (op 545) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 
buffer_load_ubyte v29, v88, s[68:71], 0 offen offset: 0 // (op 337) Load value





s_barrier  // (op 4401) 
s_barrier  // (op 4200) 
ds_read_u8 v88, v17 offset:12928 // (op 549) Load local data 



s_lshl_b32 s1, s36, 4 // (op 3395) 



s_barrier  // (op 4405) 
v_add_u32 v27, v27, s1 // (op 3395) 

ds_read_b64_tr_b4 v[98:99], v15 offset:13072 // (op 936) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v15 offset:13584 // (op 936) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[94:97], a[4:7], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 




s_barrier  // (op 4445) 
ds_read_u8 v22, v14 offset:17184 // (op 939) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v51, v[32:35] // (op 13) Store local data 
s_barrier  // (op 4449) 
s_barrier  // (op 4228) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:101], v[90:93], a[8:11], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1014) 





s_waitcnt vmcnt(14)
ds_write_b8 v71, v28 offset:4096 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:101], v[94:97], a[12:15], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 922) 




s_barrier  // (op 4187) 



s_waitcnt vmcnt(13)
ds_write_b128 v26, v[36:39] offset:4352 // (op 6) Store local data 
s_barrier  // (op 4214) 



s_waitcnt vmcnt(12)
ds_write_b8 v70, v40 offset:8448 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 880) 
s_mov_b32 s1, s18 // (op 880) convert

s_and_b32 s18, s1, 4294967292 // (op 880) 

s_cmp_lt_u32 s0, s18 // (op 880) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoop_880 // (op 880) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoop_880 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoop_880:

























s_mul_hi_i32 s0, s16, s10 // (op 3227) 
v_and_b32 v30, -2, v1 // (op 3227) 
s_add_i32 s1, s0, s16 // (op 3227) 

v_sub_u32 v51, v1, v30 // (op 3227) 

s_ashr_i32 s0, s1, 31 // (op 3227) 
v_lshlrev_b32 v30, 5, v51 // (op 3227) 

s_cmp_eq_i32 s10, 0 // (op 3227) 
s_mov_b32 s18, scc // (op 3227) 
v_lshrrev_b32 v51, 1, v1 // (op 3227) 
s_cmp_eq_i32 s18, 0 // (op 3227) 
s_cselect_b32 s19, 0, -1 // (op 3227) 

s_and_b32 s18, s11, 31 // (op 3227) 
s_lshl_b32 s22, 1, s18 // (op 3227) 
s_add_i32 s23, s22, s19 // (op 3227) 


s_and_b32 s19, s0, s23 // (op 3227) 


s_add_i32 s0, s1, s19 // (op 3227) 


s_ashr_i32 s1, s0, s18 // (op 3227) 


s_lshl_b32 s0, s11, 24 // (op 3227) 
s_ashr_i32 s18, s0, 31 // (op 3227) 

s_xor_b32 s0, s1, s18 // (op 3227) 

s_sub_i32 s1, s0, s18 // (op 3227) 


s_mul_hi_i32 s0, s1, s12 // (op 3227) 
s_add_i32 s18, s0, s1 // (op 3227) 

s_ashr_i32 s0, s18, 31 // (op 3227) 
s_cmp_eq_i32 s12, 0 // (op 3227) 
s_mov_b32 s19, scc // (op 3227) 
s_cmp_eq_i32 s19, 0 // (op 3227) 
s_cselect_b32 s22, 0, -1 // (op 3227) 

s_and_b32 s19, s13, 31 // (op 3227) 
s_lshl_b32 s23, 1, s19 // (op 3227) 
s_add_i32 s24, s23, s22 // (op 3227) 


s_and_b32 s25, s0, s24 // (op 3227) 


s_add_i32 s0, s18, s25 // (op 3227) 


s_ashr_i32 s18, s0, s19 // (op 3227) 


s_lshl_b32 s0, s13, 24 // (op 3227) 
s_ashr_i32 s19, s0, 31 // (op 3227) 

s_xor_b32 s0, s18, s19 // (op 3227) 

s_sub_i32 s18, s0, s19 // (op 3227) 


s_mul_i32 s0, s1, s15 // (op 3227) 

s_sub_u32 s1, s16, s0 // (op 3227) 

s_mul_i32 s0, s18, s15 // (op 3227) 
s_add_u32 s0, s0, s1 // (op 3227) 


s_mul_hi_i32 s1, s16, s9 // (op 3227) 
s_add_i32 s18, s1, s16 // (op 3227) 

s_ashr_i32 s1, s18, 31 // (op 3227) 
s_cmp_eq_i32 s9, 0 // (op 3227) 
s_mov_b32 s19, scc // (op 3227) 
s_cmp_eq_i32 s19, 0 // (op 3227) 
s_cselect_b32 s22, 0, -1 // (op 3227) 

s_and_b32 s19, s46, 31 // (op 3227) 
s_lshl_b32 s23, 1, s19 // (op 3227) 
s_add_i32 s24, s23, s22 // (op 3227) 


s_and_b32 s25, s1, s24 // (op 3227) 


s_add_i32 s1, s18, s25 // (op 3227) 


s_ashr_i32 s18, s1, s19 // (op 3227) 


s_lshl_b32 s1, s46, 24 // (op 3227) 
s_ashr_i32 s19, s1, 31 // (op 3227) 

s_xor_b32 s1, s18, s19 // (op 3227) 

s_sub_i32 s18, s1, s19 // (op 3227) 


s_add_u32 s50, s6, 64 // (op 3227) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3227) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3227) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3227) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3227) 

s_mul_i32 s1, s15, s24 // (op 3227) 

s_mul_i32 s19, s8, s1 // (op 3227) 

s_mul_i32 s1, s18, s19 // (op 3227) 

s_sub_u32 s19, s16, s1 // (op 3227) 

s_mov_b32 s1, s19 // (op 3227) 
s_mul_hi_i32 s22, s1, s14 // (op 3227) 
s_add_i32 s23, s22, s1 // (op 3227) 


s_ashr_i32 s1, s23, 31 // (op 3227) 
s_cmp_eq_i32 s14, 0 // (op 3227) 
s_mov_b32 s22, scc // (op 3227) 
s_cmp_eq_i32 s22, 0 // (op 3227) 
s_cselect_b32 s24, 0, -1 // (op 3227) 

s_and_b32 s22, s48, 31 // (op 3227) 
s_lshl_b32 s25, 1, s22 // (op 3227) 
s_add_i32 s49, s25, s24 // (op 3227) 


s_and_b32 s24, s1, s49 // (op 3227) 


s_add_i32 s1, s23, s24 // (op 3227) 


s_ashr_i32 s23, s1, s22 // (op 3227) 


s_lshl_b32 s1, s48, 24 // (op 3227) 
s_ashr_i32 s22, s1, 31 // (op 3227) 

s_xor_b32 s1, s23, s22 // (op 3227) 

s_sub_i32 s23, s1, s22 // (op 3227) 


s_mul_i32 s1, s47, s15 // (op 3227) 
s_add_u32 s24, s20, 64 // (op 3227) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3227) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3227) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3227) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 3227) 

s_sub_i32 s22, s24, s1 // (op 3227) 

s_mul_i32 s24, s23, s22 // (op 3227) 


s_sub_u32 s25, s19, s24 // (op 3227) 


s_add_u32 s19, s25, s1 // (op 3227) 


s_cmp_eq_i32 s18, 0 // (op 3227) 
s_mov_b32 s1, scc // (op 3227) 

s_cmp_eq_i32 s1, 0 // (op 3227) 
s_cselect_b32 s18, s19, s0 // (op 3227) 



v_lshl_add_u32 v61, s18, 6, v30 // (op 3227) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3227) 
v_mov_b32 v23, 0 // (op 3227) convert
v_mov_b32 v22, v61 // (op 3227) convert

s_lshr_b32 s18, s0, 2 // (op 3227) 

v_lshl_add_u32 v30, s18, 9, v51 // (op 3227) 


v_mul_lo_u32 v33, v30, s27 // (op 3227) most significant: low of low * high
v_mul_hi_u32 v51, v30, s26 // (op 3227) most significant: high of low * low
v_mul_lo_u32 v32, v30, s26 // (op 3227) least significant: low of low * low
v_add_u32 v33, v33, v51 // (op 3227) most significant: sum


v_add_co_u32 v34, s[0:1], v22, v32 // (op 3227) least significant half
v_addc_co_u32 v35, s[0:1], v23, v33, s[0:1] // (op 3227) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 3227) 


v_lshrrev_b64 v[22:23], 3, v[34:35] // (op 3227) 

v_mov_b32 v30, v22 // (op 3227) convert

s_mul_i32 s1, s26, s31 // (op 3229) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3229) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3229) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3229) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3229) most significant: sum
s_add_u32 s1, s1, s19 // (op 3229) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3229) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3229) 

s_mov_b32 s52, s28 // (op 3229) reg expression
s_mov_b32 s53, s29 // (op 3229) reg expression
s_mov_b32 s54, s0 // (op 3229) reg expression
s_mov_b32 s55, 131072 // (op 3229) call()

s_mul_hi_i32 s0, s16, s10 // (op 3297) 
v_lshrrev_b32 v51, 2, v1 // (op 3297) 
s_add_i32 s1, s0, s16 // (op 3297) 

v_and_b32 v61, -4, v1 // (op 3297) 
s_ashr_i32 s0, s1, 31 // (op 3297) 
v_sub_u32 v22, v1, v61 // (op 3297) 

s_cmp_eq_i32 s10, 0 // (op 3297) 
s_mov_b32 s18, scc // (op 3297) 
v_lshlrev_b32 v23, 5, v22 // (op 3297) 

s_cmp_eq_i32 s18, 0 // (op 3297) 
s_cselect_b32 s19, 0, -1 // (op 3297) 

s_and_b32 s18, s11, 31 // (op 3297) 
s_lshl_b32 s22, 1, s18 // (op 3297) 
s_add_i32 s23, s22, s19 // (op 3297) 


s_and_b32 s19, s0, s23 // (op 3297) 


s_add_i32 s0, s1, s19 // (op 3297) 


s_ashr_i32 s1, s0, s18 // (op 3297) 


s_lshl_b32 s0, s11, 24 // (op 3297) 
s_ashr_i32 s18, s0, 31 // (op 3297) 

s_xor_b32 s0, s1, s18 // (op 3297) 

s_sub_i32 s1, s0, s18 // (op 3297) 


s_mul_hi_i32 s0, s1, s12 // (op 3297) 
s_add_i32 s18, s0, s1 // (op 3297) 

s_ashr_i32 s0, s18, 31 // (op 3297) 
s_cmp_eq_i32 s12, 0 // (op 3297) 
s_mov_b32 s19, scc // (op 3297) 
s_cmp_eq_i32 s19, 0 // (op 3297) 
s_cselect_b32 s22, 0, -1 // (op 3297) 

s_and_b32 s19, s13, 31 // (op 3297) 
s_lshl_b32 s23, 1, s19 // (op 3297) 
s_add_i32 s24, s23, s22 // (op 3297) 


s_and_b32 s25, s0, s24 // (op 3297) 


s_add_i32 s0, s18, s25 // (op 3297) 


s_ashr_i32 s18, s0, s19 // (op 3297) 


s_lshl_b32 s0, s13, 24 // (op 3297) 
s_ashr_i32 s19, s0, 31 // (op 3297) 

s_xor_b32 s0, s18, s19 // (op 3297) 

s_sub_i32 s18, s0, s19 // (op 3297) 


s_add_u32 s50, s6, 64 // (op 3297) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3297) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3297) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3297) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3297) 

s_mul_i32 s0, s18, s24 // (op 3297) 

s_sub_i32 s18, s1, s0 // (op 3297) 


s_mul_hi_i32 s19, s16, s9 // (op 3297) 
s_add_i32 s0, s19, s16 // (op 3297) 

s_ashr_i32 s1, s0, 31 // (op 3297) 
s_cmp_eq_i32 s9, 0 // (op 3297) 
s_mov_b32 s19, scc // (op 3297) 
s_cmp_eq_i32 s19, 0 // (op 3297) 
s_cselect_b32 s22, 0, -1 // (op 3297) 

s_and_b32 s19, s46, 31 // (op 3297) 
s_lshl_b32 s23, 1, s19 // (op 3297) 
s_add_i32 s49, s23, s22 // (op 3297) 


s_and_b32 s22, s1, s49 // (op 3297) 


s_add_i32 s1, s0, s22 // (op 3297) 


s_ashr_i32 s0, s1, s19 // (op 3297) 


s_lshl_b32 s1, s46, 24 // (op 3297) 
s_ashr_i32 s19, s1, 31 // (op 3297) 

s_xor_b32 s1, s0, s19 // (op 3297) 

s_sub_i32 s0, s1, s19 // (op 3297) 


s_mul_i32 s1, s15, s24 // (op 3297) 

s_mul_i32 s19, s8, s1 // (op 3297) 

s_mul_i32 s1, s0, s19 // (op 3297) 

s_sub_u32 s19, s16, s1 // (op 3297) 

s_mul_hi_i32 s1, s19, s14 // (op 3297) 
s_add_i32 s22, s1, s19 // (op 3297) 


s_ashr_i32 s1, s22, 31 // (op 3297) 
s_cmp_eq_i32 s14, 0 // (op 3297) 
s_mov_b32 s19, scc // (op 3297) 
s_cmp_eq_i32 s19, 0 // (op 3297) 
s_cselect_b32 s23, 0, -1 // (op 3297) 

s_and_b32 s19, s48, 31 // (op 3297) 
s_lshl_b32 s24, 1, s19 // (op 3297) 
s_add_i32 s25, s24, s23 // (op 3297) 


s_and_b32 s23, s1, s25 // (op 3297) 


s_add_i32 s1, s22, s23 // (op 3297) 


s_ashr_i32 s22, s1, s19 // (op 3297) 


s_lshl_b32 s1, s48, 24 // (op 3297) 
s_ashr_i32 s19, s1, 31 // (op 3297) 

s_xor_b32 s1, s22, s19 // (op 3297) 

s_sub_i32 s22, s1, s19 // (op 3297) 


s_cmp_eq_i32 s0, 0 // (op 3297) 
s_mov_b32 s1, scc // (op 3297) 

s_cmp_eq_i32 s1, 0 // (op 3297) 
s_cselect_b32 s0, s22, s18 // (op 3297) 



v_lshl_add_u32 v22, s0, 6, v51 // (op 3297) 


v_mul_lo_u32 v33, v22, s33 // (op 3297) most significant: low of low * high
v_mul_hi_u32 v51, v22, s32 // (op 3297) most significant: high of low * low
v_mul_lo_u32 v32, v22, s32 // (op 3297) least significant: low of low * low
v_add_u32 v33, v33, v51 // (op 3297) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3297) 
s_lshr_b32 s18, s0, 2 // (op 3297) 

v_lshl_add_u32 v22, s18, 9, v23 // (op 3297) 


v_mov_b32 v35, 0 // (op 3297) convert
v_mov_b32 v34, v22 // (op 3297) convert

v_add_co_u32 v22, s[0:1], v34, v32 // (op 3297) least significant half
v_addc_co_u32 v23, s[0:1], v35, v33, s[0:1] // (op 3297) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 3297) 


v_lshrrev_b64 v[32:33], 3, v[22:23] // (op 3297) 

v_mov_b32 v51, v32 // (op 3297) convert

s_mul_i32 s1, s32, s7 // (op 3299) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3299) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3299) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3299) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3299) most significant: sum
s_add_u32 s1, s1, s19 // (op 3299) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3299) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3299) 

s_mov_b32 s64, s34 // (op 3299) reg expression
s_mov_b32 s65, s35 // (op 3299) reg expression
s_mov_b32 s66, s0 // (op 3299) reg expression
s_mov_b32 s67, 131072 // (op 3299) call()

s_mul_hi_i32 s0, s16, s10 // (op 3361) 
v_and_b32 v61, -64, v1 // (op 3361) 
s_add_i32 s1, s0, s16 // (op 3361) 

v_sub_u32 v22, v1, v61 // (op 3361) 

s_ashr_i32 s0, s1, 31 // (op 3361) 
v_lshrrev_b32 v23, 6, v1 // (op 3361) 
s_cmp_eq_i32 s10, 0 // (op 3361) 
s_mov_b32 s18, scc // (op 3361) 
s_cmp_eq_i32 s18, 0 // (op 3361) 
s_cselect_b32 s19, 0, -1 // (op 3361) 

s_and_b32 s18, s11, 31 // (op 3361) 
s_lshl_b32 s22, 1, s18 // (op 3361) 
s_add_i32 s23, s22, s19 // (op 3361) 


s_and_b32 s19, s0, s23 // (op 3361) 


s_add_i32 s0, s1, s19 // (op 3361) 


s_ashr_i32 s1, s0, s18 // (op 3361) 


s_lshl_b32 s0, s11, 24 // (op 3361) 
s_ashr_i32 s18, s0, 31 // (op 3361) 

s_xor_b32 s0, s1, s18 // (op 3361) 

s_sub_i32 s1, s0, s18 // (op 3361) 


s_mul_hi_i32 s0, s1, s12 // (op 3361) 
s_add_i32 s18, s0, s1 // (op 3361) 

s_ashr_i32 s0, s18, 31 // (op 3361) 
s_cmp_eq_i32 s12, 0 // (op 3361) 
s_mov_b32 s19, scc // (op 3361) 
s_cmp_eq_i32 s19, 0 // (op 3361) 
s_cselect_b32 s22, 0, -1 // (op 3361) 

s_and_b32 s19, s13, 31 // (op 3361) 
s_lshl_b32 s23, 1, s19 // (op 3361) 
s_add_i32 s24, s23, s22 // (op 3361) 


s_and_b32 s25, s0, s24 // (op 3361) 


s_add_i32 s0, s18, s25 // (op 3361) 


s_ashr_i32 s18, s0, s19 // (op 3361) 


s_lshl_b32 s0, s13, 24 // (op 3361) 
s_ashr_i32 s19, s0, 31 // (op 3361) 

s_xor_b32 s0, s18, s19 // (op 3361) 

s_sub_i32 s18, s0, s19 // (op 3361) 


s_mul_i32 s0, s1, s15 // (op 3361) 

s_sub_u32 s1, s16, s0 // (op 3361) 

s_mul_i32 s0, s18, s15 // (op 3361) 
s_add_u32 s0, s0, s1 // (op 3361) 


s_mul_hi_i32 s1, s16, s9 // (op 3361) 
s_add_i32 s18, s1, s16 // (op 3361) 

s_ashr_i32 s1, s18, 31 // (op 3361) 
s_cmp_eq_i32 s9, 0 // (op 3361) 
s_mov_b32 s19, scc // (op 3361) 
s_cmp_eq_i32 s19, 0 // (op 3361) 
s_cselect_b32 s22, 0, -1 // (op 3361) 

s_and_b32 s19, s46, 31 // (op 3361) 
s_lshl_b32 s23, 1, s19 // (op 3361) 
s_add_i32 s24, s23, s22 // (op 3361) 


s_and_b32 s25, s1, s24 // (op 3361) 


s_add_i32 s1, s18, s25 // (op 3361) 


s_ashr_i32 s18, s1, s19 // (op 3361) 


s_lshl_b32 s1, s46, 24 // (op 3361) 
s_ashr_i32 s19, s1, 31 // (op 3361) 

s_xor_b32 s1, s18, s19 // (op 3361) 

s_sub_i32 s18, s1, s19 // (op 3361) 


s_add_u32 s50, s6, 64 // (op 3361) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3361) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3361) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3361) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3361) 

s_mul_i32 s1, s15, s24 // (op 3361) 

s_mul_i32 s19, s8, s1 // (op 3361) 

s_mul_i32 s1, s18, s19 // (op 3361) 

s_sub_u32 s19, s16, s1 // (op 3361) 

s_mov_b32 s1, s19 // (op 3361) 
s_mul_hi_i32 s22, s1, s14 // (op 3361) 
s_add_i32 s23, s22, s1 // (op 3361) 


s_ashr_i32 s1, s23, 31 // (op 3361) 
s_cmp_eq_i32 s14, 0 // (op 3361) 
s_mov_b32 s22, scc // (op 3361) 
s_cmp_eq_i32 s22, 0 // (op 3361) 
s_cselect_b32 s24, 0, -1 // (op 3361) 

s_and_b32 s22, s48, 31 // (op 3361) 
s_lshl_b32 s25, 1, s22 // (op 3361) 
s_add_i32 s49, s25, s24 // (op 3361) 


s_and_b32 s24, s1, s49 // (op 3361) 


s_add_i32 s1, s23, s24 // (op 3361) 


s_ashr_i32 s23, s1, s22 // (op 3361) 


s_lshl_b32 s1, s48, 24 // (op 3361) 
s_ashr_i32 s22, s1, 31 // (op 3361) 

s_xor_b32 s1, s23, s22 // (op 3361) 

s_sub_i32 s23, s1, s22 // (op 3361) 


s_mul_i32 s1, s47, s15 // (op 3361) 
s_add_u32 s24, s20, 64 // (op 3361) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3361) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3361) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3361) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 3361) 

s_sub_i32 s22, s24, s1 // (op 3361) 

s_mul_i32 s24, s23, s22 // (op 3361) 


s_sub_u32 s25, s19, s24 // (op 3361) 


s_add_u32 s19, s25, s1 // (op 3361) 


s_cmp_eq_i32 s18, 0 // (op 3361) 
s_mov_b32 s1, scc // (op 3361) 

s_cmp_eq_i32 s1, 0 // (op 3361) 
s_cselect_b32 s18, s19, s0 // (op 3361) 



v_lshl_add_u32 v61, s18, 6, v22 // (op 3361) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3361) 
s_lshr_b32 s18, s0, 2 // (op 3361) 

v_lshl_add_u32 v22, s18, 4, v23 // (op 3361) 


v_mul_lo_u32 v23, v22, s36 // (op 3361) 

v_add_u32 v22, v61, v23 // (op 3361) 


s_mul_i32 s1, s36, s41 // (op 3363) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3363) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3363) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3363) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3363) most significant: sum
s_add_u32 s1, s1, s19 // (op 3363) most significant: sum


s_mov_b32 s68, s38 // (op 3363) reg expression
s_mov_b32 s69, s39 // (op 3363) reg expression
s_mov_b32 s70, s0 // (op 3363) reg expression
s_mov_b32 s71, 131072 // (op 3363) call()

s_mul_hi_i32 s0, s16, s10 // (op 3425) 
v_lshrrev_b32 v23, 2, v1 // (op 3425) 
s_add_i32 s1, s0, s16 // (op 3425) 

v_and_b32 v61, -4, v1 // (op 3425) 
s_ashr_i32 s0, s1, 31 // (op 3425) 
v_sub_u32 v25, v1, v61 // (op 3425) 

s_cmp_eq_i32 s10, 0 // (op 3425) 
s_mov_b32 s18, scc // (op 3425) 
s_cmp_eq_i32 s18, 0 // (op 3425) 
s_cselect_b32 s19, 0, -1 // (op 3425) 

s_and_b32 s18, s11, 31 // (op 3425) 
s_lshl_b32 s22, 1, s18 // (op 3425) 
s_add_i32 s23, s22, s19 // (op 3425) 


s_and_b32 s19, s0, s23 // (op 3425) 


s_add_i32 s0, s1, s19 // (op 3425) 


s_ashr_i32 s1, s0, s18 // (op 3425) 


s_lshl_b32 s0, s11, 24 // (op 3425) 
s_ashr_i32 s18, s0, 31 // (op 3425) 

s_xor_b32 s0, s1, s18 // (op 3425) 

s_sub_i32 s1, s0, s18 // (op 3425) 


s_mul_hi_i32 s0, s1, s12 // (op 3425) 
s_add_i32 s18, s0, s1 // (op 3425) 

s_ashr_i32 s0, s18, 31 // (op 3425) 
s_cmp_eq_i32 s12, 0 // (op 3425) 
s_mov_b32 s19, scc // (op 3425) 
s_cmp_eq_i32 s19, 0 // (op 3425) 
s_cselect_b32 s22, 0, -1 // (op 3425) 

s_and_b32 s19, s13, 31 // (op 3425) 
s_lshl_b32 s23, 1, s19 // (op 3425) 
s_add_i32 s24, s23, s22 // (op 3425) 


s_and_b32 s25, s0, s24 // (op 3425) 


s_add_i32 s0, s18, s25 // (op 3425) 


s_ashr_i32 s18, s0, s19 // (op 3425) 


s_lshl_b32 s0, s13, 24 // (op 3425) 
s_ashr_i32 s19, s0, 31 // (op 3425) 

s_xor_b32 s0, s18, s19 // (op 3425) 

s_sub_i32 s18, s0, s19 // (op 3425) 


s_add_u32 s50, s6, 64 // (op 3425) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3425) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3425) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3425) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 3425) 

s_mul_i32 s0, s18, s24 // (op 3425) 

s_sub_i32 s18, s1, s0 // (op 3425) 


s_mul_hi_i32 s19, s16, s9 // (op 3425) 
s_add_i32 s0, s19, s16 // (op 3425) 

s_ashr_i32 s1, s0, 31 // (op 3425) 
s_cmp_eq_i32 s9, 0 // (op 3425) 
s_mov_b32 s19, scc // (op 3425) 
s_cmp_eq_i32 s19, 0 // (op 3425) 
s_cselect_b32 s22, 0, -1 // (op 3425) 

s_and_b32 s19, s46, 31 // (op 3425) 
s_lshl_b32 s23, 1, s19 // (op 3425) 
s_add_i32 s49, s23, s22 // (op 3425) 


s_and_b32 s22, s1, s49 // (op 3425) 


s_add_i32 s1, s0, s22 // (op 3425) 


s_ashr_i32 s0, s1, s19 // (op 3425) 


s_lshl_b32 s1, s46, 24 // (op 3425) 
s_ashr_i32 s19, s1, 31 // (op 3425) 

s_xor_b32 s1, s0, s19 // (op 3425) 

s_sub_i32 s0, s1, s19 // (op 3425) 


s_mul_i32 s1, s15, s24 // (op 3425) 

s_mul_i32 s19, s8, s1 // (op 3425) 

s_mul_i32 s1, s0, s19 // (op 3425) 

s_sub_u32 s19, s16, s1 // (op 3425) 

s_mul_hi_i32 s1, s19, s14 // (op 3425) 
s_add_i32 s22, s1, s19 // (op 3425) 


s_ashr_i32 s1, s22, 31 // (op 3425) 
s_cmp_eq_i32 s14, 0 // (op 3425) 
s_mov_b32 s19, scc // (op 3425) 
s_cmp_eq_i32 s19, 0 // (op 3425) 
s_cselect_b32 s23, 0, -1 // (op 3425) 

s_and_b32 s19, s48, 31 // (op 3425) 
s_lshl_b32 s24, 1, s19 // (op 3425) 
s_add_i32 s25, s24, s23 // (op 3425) 


s_and_b32 s23, s1, s25 // (op 3425) 


s_add_i32 s1, s22, s23 // (op 3425) 


s_ashr_i32 s22, s1, s19 // (op 3425) 


s_lshl_b32 s1, s48, 24 // (op 3425) 
s_ashr_i32 s19, s1, 31 // (op 3425) 

s_xor_b32 s1, s22, s19 // (op 3425) 

s_sub_i32 s22, s1, s19 // (op 3425) 


s_cmp_eq_i32 s0, 0 // (op 3425) 
s_mov_b32 s1, scc // (op 3425) 

s_cmp_eq_i32 s1, 0 // (op 3425) 
s_cselect_b32 s0, s22, s18 // (op 3425) 



v_lshl_add_u32 v61, s0, 6, v23 // (op 3425) 


v_mul_lo_u32 v23, v61, s42 // (op 3425) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 3425) 
s_lshr_b32 s18, s0, 2 // (op 3425) 

v_lshl_add_u32 v61, s18, 4, v25 // (op 3425) 


v_add_u32 v25, v61, v23 // (op 3425) 


s_mul_i32 s1, s42, s7 // (op 3427) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3427) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3427) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3427) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3427) most significant: sum
s_add_u32 s1, s1, s19 // (op 3427) most significant: sum


s_mov_b32 s72, s44 // (op 3427) reg expression
s_mov_b32 s73, s45 // (op 3427) reg expression
s_mov_b32 s74, s0 // (op 3427) reg expression
s_mov_b32 s75, 131072 // (op 3427) call()

v_and_b32 v23, -2, v1 // (op 3523) 
v_sub_u32 v61, v1, v23 // (op 3523) 

v_lshlrev_b32 v23, 5, v61 // (op 3523) 

v_lshrrev_b32 v61, 1, v1 // (op 3523) 
v_lshl_add_u32 v26, v61, 6, v23 // (op 3523) 


v_lshlrev_b32 v23, 2, v26 // (op 3523) 

v_lshrrev_b32 v61, 3, v23 // (op 3523) 

v_and_b32 v23, -4, v1 // (op 3581) 
v_sub_u32 v26, v1, v23 // (op 3581) 

v_lshlrev_b32 v23, 5, v26 // (op 3581) 

v_lshrrev_b32 v26, 2, v1 // (op 3581) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 3581) 


v_lshlrev_b32 v23, 2, v27 // (op 3581) 

v_lshrrev_b32 v26, 3, v23 // (op 3581) 

v_and_b32 v23, -64, v1 // (op 3639) 
v_sub_u32 v27, v1, v23 // (op 3639) 

v_and_b32 v23, 4294967232, v1 // (op 3639) 
v_add_u32 v28, v23, v27 // (op 3639) 


v_and_b32 v23, -4, v1 // (op 3697) 
v_sub_u32 v27, v1, v23 // (op 3697) 

v_and_b32 v23, 4294967292, v1 // (op 3697) 
v_add_u32 v32, v23, v27 // (op 3697) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 886) 
s_and_b32 s18, s0, 4294967292 // (op 886) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 885) 
s_mov_b32 s19, s0 // (op 885) convert

s_cmp_lt_u32 s18, s19 // (op 885) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_885 // (op 885) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_885 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoopTail_885:


s_add_u32 s18, s18, 1 // (op 887) 
v_mov_b32 v27, v30 // (op 176) 
v_mov_b32 v29, v27 // (op 176) reg expression

buffer_load_dwordx4 v[52:55], v29, s[52:55], 0 offen // (op 176) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 3217) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 3217) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 3217) 

v_add_u32 v30, v30, s0 // (op 3217) 



s_barrier  // (op 4190) 
v_mov_b32 v27, v51 // (op 190) 




buffer_load_dwordx4 v[56:59], v27, s[64:67], 0 offen // (op 190) Load value

v_add_u32 v51, 64, v51 // (op 3289) 
s_waitcnt vmcnt(1)
ds_write_b128 v61, v[52:55] offset:4352 // (op 177) Store local data 



v_mov_b32 v27, v22 // (op 191) 
s_barrier  // (op 4203) 
v_mov_b32 v29, v27 // (op 191) reg expression




buffer_load_ubyte v31, v29, s[68:71], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s19, s36, 2 // (op 3353) 
v_add_u32 v22, v22, s19 // (op 3353) 

s_waitcnt vmcnt(1)
ds_write_b128 v26, v[56:59] // (op 187) Store local data 



v_mov_b32 v27, v25 // (op 192) 

s_barrier  // (op 4217) 
buffer_load_ubyte v29, v27, s[72:75], 0 offen offset: 0 // (op 192) Load value




v_add_u32 v25, 4, v25 // (op 3417) 
s_waitcnt vmcnt(1)
ds_write_b8 v28, v31 offset:8448 // (op 188) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4193) 



s_barrier  // (op 4231) 
ds_read_b64_tr_b4 v[52:53], v3 offset:4352 // (op 179) Transpose load local data 



ds_read_b64_tr_b4 v[54:55], v3 offset:4864 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v32, v29 offset:4096 // (op 189) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 4207) 



ds_read_b128 v[56:59], v4 // (op 181) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4221) 


ds_read_u8 v27, v0 offset:8448 // (op 183) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4235) 



ds_read_u8 v29, v5 offset:4096 // (op 185) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[56:59], a[0:3], v27, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 
ds_read_b128 v[34:37], v4 offset:2048 // (op 496) Load local data 



ds_read_u8 v31, v5 offset:4224 // (op 500) Load local data 



ds_read_b64_tr_b4 v[38:39], v3 offset:4368 // (op 860) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[34:37], a[4:7], v27, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 
ds_read_b64_tr_b4 v[40:41], v3 offset:4880 // (op 860) Transpose load local data 




ds_read_u8 v27, v0 offset:8480 // (op 863) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[56:59], a[8:11], v27, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 865) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[34:37], a[12:15], v27, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 896) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 885) 
s_mov_b32 s19, s0 // (op 885) convert

s_cmp_lt_u32 s18, s19 // (op 885) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoopTail_885 // (op 885) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopTop_KLoopTail_885 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_885:























s_barrier  // (op 4413) 





























v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalFalse_0_beta_133
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 



v_mov_b32 v0, v21 // (op 56) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v14, a4 // (op 1617) 
v_accvgpr_read v15, a5 // (op 1617) 
v_accvgpr_read v16, a6 // (op 1617) 
v_accvgpr_read v17, a7 // (op 1617) 
v_mul_f32 v10, v19, v14 // (op 1617) 
v_mul_f32 v11, v19, v15 // (op 1617) 
v_mul_f32 v12, v19, v16 // (op 1617) 
v_mul_f32 v13, v19, v17 // (op 1617) 



s_lshl_b32 s0, s4, 7 // (op 1619) 
v_add_u32 v0, v21, s0 // (op 1619) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1619) Store value


v_accvgpr_read v14, a8 // (op 1641) 
v_accvgpr_read v15, a9 // (op 1641) 
v_accvgpr_read v16, a10 // (op 1641) 
v_accvgpr_read v17, a11 // (op 1641) 
v_mul_f32 v10, v19, v14 // (op 1641) 
v_mul_f32 v11, v19, v15 // (op 1641) 
v_mul_f32 v12, v19, v16 // (op 1641) 
v_mul_f32 v13, v19, v17 // (op 1641) 



v_add_u32 v0, 128, v21 // (op 1644) 

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1644) Store value


v_accvgpr_read v14, a12 // (op 1650) 
v_accvgpr_read v15, a13 // (op 1650) 
v_accvgpr_read v16, a14 // (op 1650) 
v_accvgpr_read v17, a15 // (op 1650) 
v_mul_f32 v10, v19, v14 // (op 1650) 
v_mul_f32 v11, v19, v15 // (op 1650) 
v_mul_f32 v12, v19, v16 // (op 1650) 
v_mul_f32 v13, v19, v17 // (op 1650) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_add_u32 s0, s0, 128 // (op 1646) 
v_add_u32 v0, v21, s0 // (op 1646) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalBottom_0_beta_133
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v20 // (op 1730) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1730) Load value

v_accvgpr_read v10, a8 // (op 1722) 
v_accvgpr_read v11, a9 // (op 1722) 
v_accvgpr_read v12, a10 // (op 1722) 
v_accvgpr_read v13, a11 // (op 1722) 
v_mul_f32 v6, v19, v10 // (op 1722) 
v_mul_f32 v7, v19, v11 // (op 1722) 
v_mul_f32 v8, v19, v12 // (op 1722) 
v_mul_f32 v9, v19, v13 // (op 1722) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1729) 
v_mul_f32 v11, v18, v15 // (op 1729) 
v_mul_f32 v12, v18, v16 // (op 1729) 
v_mul_f32 v13, v18, v17 // (op 1729) 

v_add_f32 v14, v10, v6 // (op 1723) 
v_add_f32 v15, v11, v7 // (op 1723) 
v_add_f32 v16, v12, v8 // (op 1723) 
v_add_f32 v17, v13, v9 // (op 1723) 




v_add_u32 v0, 128, v21 // (op 1725) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1725) Store value




s_lshl_b32 s0, s2, 7 // (op 1734) 
s_add_u32 s0, s0, 128 // (op 1734) 
v_add_u32 v0, v20, s0 // (op 1734) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1734) Load value

v_accvgpr_read v10, a12 // (op 1736) 
v_accvgpr_read v11, a13 // (op 1736) 
v_accvgpr_read v12, a14 // (op 1736) 
v_accvgpr_read v13, a15 // (op 1736) 
v_mul_f32 v6, v19, v10 // (op 1736) 
v_mul_f32 v7, v19, v11 // (op 1736) 
v_mul_f32 v8, v19, v12 // (op 1736) 
v_mul_f32 v9, v19, v13 // (op 1736) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1732) 
v_mul_f32 v11, v18, v15 // (op 1732) 
v_mul_f32 v12, v18, v16 // (op 1732) 
v_mul_f32 v13, v18, v17 // (op 1732) 

v_add_f32 v14, v10, v6 // (op 1733) 
v_add_f32 v15, v11, v7 // (op 1733) 
v_add_f32 v16, v12, v8 // (op 1733) 
v_add_f32 v17, v13, v9 // (op 1733) 




s_lshl_b32 s0, s4, 7 // (op 1727) 
s_add_u32 s0, s0, 128 // (op 1727) 
v_add_u32 v0, v21, s0 // (op 1727) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1727) Store value




v_mov_b32 v0, v20 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 145) Load value

v_accvgpr_read v10, a0 // (op 142) 
v_accvgpr_read v11, a1 // (op 142) 
v_accvgpr_read v12, a2 // (op 142) 
v_accvgpr_read v13, a3 // (op 142) 
v_mul_f32 v6, v19, v10 // (op 142) 
v_mul_f32 v7, v19, v11 // (op 142) 
v_mul_f32 v8, v19, v12 // (op 142) 
v_mul_f32 v9, v19, v13 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 146) 
v_mul_f32 v11, v18, v15 // (op 146) 
v_mul_f32 v12, v18, v16 // (op 146) 
v_mul_f32 v13, v18, v17 // (op 146) 

v_add_f32 v14, v10, v6 // (op 143) 
v_add_f32 v15, v11, v7 // (op 143) 
v_add_f32 v16, v12, v8 // (op 143) 
v_add_f32 v17, v13, v9 // (op 143) 




v_mov_b32 v0, v21 // (op 144) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 1692) 
v_add_u32 v0, v20, s0 // (op 1692) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1692) Load value

v_accvgpr_read v10, a4 // (op 1689) 
v_accvgpr_read v11, a5 // (op 1689) 
v_accvgpr_read v12, a6 // (op 1689) 
v_accvgpr_read v13, a7 // (op 1689) 
v_mul_f32 v6, v19, v10 // (op 1689) 
v_mul_f32 v7, v19, v11 // (op 1689) 
v_mul_f32 v8, v19, v12 // (op 1689) 
v_mul_f32 v9, v19, v13 // (op 1689) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1693) 
v_mul_f32 v11, v18, v15 // (op 1693) 
v_mul_f32 v12, v18, v16 // (op 1693) 
v_mul_f32 v13, v18, v17 // (op 1693) 

v_add_f32 v14, v10, v6 // (op 1690) 
v_add_f32 v15, v11, v7 // (op 1690) 
v_add_f32 v16, v12, v8 // (op 1690) 
v_add_f32 v17, v13, v9 // (op 1690) 




s_lshl_b32 s0, s4, 7 // (op 1691) 
v_add_u32 v0, v21, s0 // (op 1691) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1691) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_
.LRR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__end:
.size RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_, .LRR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM__end-RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 26112 // lds bytes
  .amdhsa_accum_offset 108
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
    DataTypeB: F4
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_
    .symbol: RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 26112
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 106
    .agpr_count: 16
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

