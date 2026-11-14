// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_,@function
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 13729) 
s_mul_hi_i32 s1, s16, s11 // (op 13984) 
v_lshrrev_b32 v0, 7, v1 // (op 13790) 
v_and_b32 v3, 15, v1 // (op 13790) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 13790) 


v_and_b32 v0, 63, v1 // (op 13790) 
v_lshrrev_b32 v3, 4, v0 // (op 13790) 

v_lshlrev_b32 v0, 5, v3 // (op 13790) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 13790) 


v_lshlrev_b32 v0, 2, v3 // (op 13790) 

v_lshrrev_b32 v3, 3, v0 // (op 13790) 

v_and_b32 v0, 63, v1 // (op 13847) 
v_and_b32 v4, 48, v1 // (op 13847) 
v_sub_u32 v5, v0, v4 // (op 13847) 

v_lshrrev_b32 v4, 4, v0 // (op 13847) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 13847) 


v_lshrrev_b32 v4, 6, v1 // (op 13847) 
v_and_b32 v5, 1, v4 // (op 13847) 

v_lshlrev_b32 v4, 4, v5 // (op 13847) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 13847) 


v_lshlrev_b32 v0, 2, v5 // (op 13847) 

v_lshrrev_b32 v4, 3, v0 // (op 13847) 

v_lshrrev_b32 v0, 7, v1 // (op 13905) 
v_and_b32 v5, 15, v1 // (op 13905) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 13905) 


v_and_b32 v0, 63, v1 // (op 13905) 
v_lshrrev_b32 v5, 4, v0 // (op 13905) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 13905) 


v_lshrrev_b32 v5, 6, v1 // (op 13963) 
v_and_b32 v6, 1, v5 // (op 13963) 

v_and_b32 v5, 15, v1 // (op 13963) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 13963) 


v_and_b32 v5, 63, v1 // (op 13963) 
v_lshrrev_b32 v6, 4, v5 // (op 13963) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 13963) 


v_lshrrev_b32 v6, 7, v1 // (op 14030) 
v_and_b32 v7, 15, v1 // (op 14030) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 14030) 


v_and_b32 v6, 63, v1 // (op 14030) 
v_lshrrev_b32 v7, 4, v6 // (op 14030) 

v_lshlrev_b32 v6, 5, v7 // (op 14030) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 14030) 


v_lshlrev_b32 v6, 2, v7 // (op 14030) 

v_lshrrev_b32 v7, 3, v6 // (op 14030) 

v_and_b32 v6, 63, v1 // (op 14074) 
v_and_b32 v8, 48, v1 // (op 14074) 
v_sub_u32 v9, v6, v8 // (op 14074) 

v_lshrrev_b32 v8, 4, v6 // (op 14074) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 14074) 


v_lshrrev_b32 v8, 6, v1 // (op 14074) 
v_and_b32 v9, 1, v8 // (op 14074) 

v_lshlrev_b32 v8, 4, v9 // (op 14074) 

v_lshl_add_u32 v9, v6, 7, v8 // (op 14074) 


v_lshlrev_b32 v6, 2, v9 // (op 14074) 

v_lshrrev_b32 v8, 3, v6 // (op 14074) 

v_lshrrev_b32 v6, 7, v1 // (op 14103) 
v_and_b32 v9, 15, v1 // (op 14103) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 14103) 


v_and_b32 v6, 63, v1 // (op 14103) 
v_lshrrev_b32 v9, 4, v6 // (op 14103) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 14103) 


v_lshrrev_b32 v9, 6, v1 // (op 14132) 
v_and_b32 v10, 1, v9 // (op 14132) 

v_and_b32 v9, 15, v1 // (op 14132) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 14132) 


v_and_b32 v9, 63, v1 // (op 14132) 
v_lshrrev_b32 v10, 4, v9 // (op 14132) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 14132) 


v_lshrrev_b32 v10, 7, v1 // (op 14176) 
v_and_b32 v11, 15, v1 // (op 14176) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 14176) 


v_and_b32 v10, 63, v1 // (op 14176) 
v_lshrrev_b32 v11, 4, v10 // (op 14176) 

v_lshlrev_b32 v10, 5, v11 // (op 14176) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 14176) 


v_lshlrev_b32 v10, 2, v11 // (op 14176) 

v_lshrrev_b32 v11, 3, v10 // (op 14176) 

v_and_b32 v10, 63, v1 // (op 14220) 
v_and_b32 v12, 48, v1 // (op 14220) 
v_sub_u32 v13, v10, v12 // (op 14220) 

v_lshrrev_b32 v12, 4, v10 // (op 14220) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 14220) 


v_lshrrev_b32 v12, 6, v1 // (op 14220) 
v_and_b32 v13, 1, v12 // (op 14220) 

v_lshlrev_b32 v12, 4, v13 // (op 14220) 

v_lshl_add_u32 v13, v10, 7, v12 // (op 14220) 


v_lshlrev_b32 v10, 2, v13 // (op 14220) 

v_lshrrev_b32 v12, 3, v10 // (op 14220) 

v_lshrrev_b32 v10, 7, v1 // (op 14249) 
v_and_b32 v13, 15, v1 // (op 14249) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 14249) 


v_and_b32 v10, 63, v1 // (op 14249) 
v_lshrrev_b32 v13, 4, v10 // (op 14249) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 14249) 


v_lshrrev_b32 v13, 6, v1 // (op 14278) 
v_and_b32 v14, 1, v13 // (op 14278) 

v_and_b32 v13, 15, v1 // (op 14278) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 14278) 


v_and_b32 v13, 63, v1 // (op 14278) 
v_lshrrev_b32 v14, 4, v13 // (op 14278) 

v_lshl_add_u32 v13, v14, 7, v15 // (op 14278) 


v_lshrrev_b32 v14, 7, v1 // (op 14322) 
v_and_b32 v15, 15, v1 // (op 14322) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 14322) 


v_and_b32 v14, 63, v1 // (op 14322) 
v_lshrrev_b32 v15, 4, v14 // (op 14322) 

v_lshlrev_b32 v14, 5, v15 // (op 14322) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 14322) 


v_lshlrev_b32 v14, 2, v15 // (op 14322) 

v_lshrrev_b32 v15, 3, v14 // (op 14322) 

v_and_b32 v14, 63, v1 // (op 14366) 
v_and_b32 v16, 48, v1 // (op 14366) 
v_sub_u32 v17, v14, v16 // (op 14366) 

v_lshrrev_b32 v16, 4, v14 // (op 14366) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 14366) 


v_lshrrev_b32 v16, 6, v1 // (op 14366) 
v_and_b32 v17, 1, v16 // (op 14366) 

v_lshlrev_b32 v16, 4, v17 // (op 14366) 

v_lshl_add_u32 v17, v14, 7, v16 // (op 14366) 


v_lshlrev_b32 v14, 2, v17 // (op 14366) 

v_lshrrev_b32 v16, 3, v14 // (op 14366) 

v_lshrrev_b32 v14, 7, v1 // (op 14395) 
v_and_b32 v17, 15, v1 // (op 14395) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 14395) 


v_and_b32 v14, 63, v1 // (op 14395) 
v_lshrrev_b32 v17, 4, v14 // (op 14395) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 14395) 


v_lshrrev_b32 v17, 6, v1 // (op 14424) 
v_and_b32 v18, 1, v17 // (op 14424) 

v_and_b32 v17, 15, v1 // (op 14424) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 14424) 


v_and_b32 v17, 63, v1 // (op 14424) 
v_lshrrev_b32 v18, 4, v17 // (op 14424) 

v_lshl_add_u32 v17, v18, 7, v19 // (op 14424) 


v_lshrrev_b32 v18, 7, v1 // (op 13729) 
s_add_i32 s18, s0, s16 // (op 13729) 

v_and_b32 v19, 63, v1 // (op 13729) 
s_ashr_i32 s0, s18, 31 // (op 13729) 
v_lshrrev_b32 v20, 4, v19 // (op 13729) 
s_cmp_eq_i32 s11, 0 // (op 13729) 
s_mov_b32 s19, scc // (op 13729) 
v_lshlrev_b32 v21, 2, v20 // (op 13729) 

s_cmp_eq_i32 s19, 0 // (op 13729) 
s_cselect_b32 s51, 0, -1 // (op 13729) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 13729) 


s_and_b32 s19, s12, 31 // (op 13729) 
v_lshrrev_b32 v18, 2, v19 // (op 13729) 

v_and_b32 v19, 3, v18 // (op 13729) 

v_lshrrev_b32 v18, 7, v1 // (op 13984) 
v_and_b32 v21, 63, v1 // (op 13984) 
s_lshl_b32 s52, 1, s19 // (op 13729) 
v_and_b32 v22, 3, v1 // (op 13729) 
s_add_i32 s53, s52, s51 // (op 13729) 


s_and_b32 s51, s0, s53 // (op 13729) 


s_add_i32 s0, s1, s16 // (op 13984) 

s_ashr_i32 s1, s0, 31 // (op 13984) 
v_lshl_add_u32 v23, v19, 2, v22 // (op 13729) 


s_add_i32 s52, s18, s51 // (op 13729) 


v_lshrrev_b32 v19, 6, v1 // (op 13729) 
s_ashr_i32 s18, s52, s19 // (op 13729) 


v_and_b32 v22, 1, v19 // (op 13729) 

s_lshl_b32 s19, s12, 24 // (op 13729) 
v_lshl_add_u32 v19, v22, 4, v23 // (op 13729) 


s_ashr_i32 s51, s19, 31 // (op 13729) 

s_xor_b32 s19, s18, s51 // (op 13729) 

s_sub_i32 s18, s19, s51 // (op 13729) 


s_mul_hi_i32 s19, s18, s13 // (op 13729) 
s_add_i32 s51, s19, s18 // (op 13729) 

s_ashr_i32 s19, s51, 31 // (op 13729) 
s_cmp_eq_i32 s13, 0 // (op 13729) 
s_mov_b32 s52, scc // (op 13729) 
s_cmp_eq_i32 s52, 0 // (op 13729) 
s_cselect_b32 s53, 0, -1 // (op 13729) 

s_and_b32 s52, s14, 31 // (op 13729) 
v_lshrrev_b32 v22, 4, v21 // (op 13984) 
s_cmp_eq_i32 s11, 0 // (op 13984) 
s_mov_b32 s54, scc // (op 13984) 
v_lshlrev_b32 v23, 2, v22 // (op 13984) 

s_cmp_eq_i32 s54, 0 // (op 13984) 
s_cselect_b32 s55, 0, -1 // (op 13984) 

s_lshl_b32 s54, 1, s52 // (op 13729) 
v_lshl_add_u32 v22, v18, 4, v23 // (op 13984) 


v_lshrrev_b32 v18, 2, v21 // (op 13984) 

s_add_i32 s56, s54, s53 // (op 13729) 


s_and_b32 s53, s19, s56 // (op 13729) 


s_add_i32 s19, s51, s53 // (op 13729) 


s_ashr_i32 s51, s19, s52 // (op 13729) 


s_lshl_b32 s19, s14, 24 // (op 13729) 
s_ashr_i32 s52, s19, 31 // (op 13729) 

s_xor_b32 s19, s51, s52 // (op 13729) 

s_sub_i32 s51, s19, s52 // (op 13729) 


s_and_b32 s19, s12, 31 // (op 13984) 
s_lshl_b32 s52, 1, s19 // (op 13984) 
v_and_b32 v21, 3, v18 // (op 13984) 

s_add_i32 s53, s52, s55 // (op 13984) 


v_and_b32 v18, 3, v1 // (op 13984) 
s_and_b32 s52, s1, s53 // (op 13984) 


v_lshl_add_u32 v23, v21, 2, v18 // (op 13984) 


s_add_i32 s1, s0, s52 // (op 13984) 


v_lshrrev_b32 v18, 6, v1 // (op 13984) 
s_ashr_i32 s0, s1, s19 // (op 13984) 


v_and_b32 v21, 1, v18 // (op 13984) 

s_lshl_b32 s1, s12, 24 // (op 13984) 
v_lshl_add_u32 v18, v21, 4, v23 // (op 13984) 


s_ashr_i32 s19, s1, 31 // (op 13984) 

s_xor_b32 s1, s0, s19 // (op 13984) 

s_sub_i32 s0, s1, s19 // (op 13984) 


s_mul_hi_i32 s1, s0, s13 // (op 13984) 
s_add_i32 s19, s1, s0 // (op 13984) 

s_ashr_i32 s1, s19, 31 // (op 13984) 
s_cmp_eq_i32 s13, 0 // (op 13984) 
s_mov_b32 s52, scc // (op 13984) 
s_cmp_eq_i32 s52, 0 // (op 13984) 
s_cselect_b32 s53, 0, -1 // (op 13984) 

s_and_b32 s52, s14, 31 // (op 13984) 
s_lshl_b32 s54, 1, s52 // (op 13984) 
s_add_i32 s55, s54, s53 // (op 13984) 


s_and_b32 s53, s1, s55 // (op 13984) 


s_add_i32 s1, s19, s53 // (op 13984) 


s_ashr_i32 s19, s1, s52 // (op 13984) 


s_lshl_b32 s1, s14, 24 // (op 13984) 
s_ashr_i32 s52, s1, 31 // (op 13984) 

s_xor_b32 s1, s19, s52 // (op 13984) 

s_sub_i32 s19, s1, s52 // (op 13984) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 13729) 
s_sub_u32 s52, s16, s1 // (op 13729) 

s_mul_i32 s1, s51, s44 // (op 13729) 
s_add_u32 s1, s1, s52 // (op 13729) 

s_mul_hi_i32 s52, s16, s10 // (op 13729) 
s_add_i32 s53, s52, s16 // (op 13729) 

s_ashr_i32 s52, s53, 31 // (op 13729) 
s_cmp_eq_i32 s10, 0 // (op 13729) 
s_mov_b32 s54, scc // (op 13729) 
s_cmp_eq_i32 s54, 0 // (op 13729) 
s_cselect_b32 s55, 0, -1 // (op 13729) 

s_and_b32 s54, s47, 31 // (op 13729) 
s_lshl_b32 s56, 1, s54 // (op 13729) 
s_add_i32 s57, s56, s55 // (op 13729) 


s_and_b32 s55, s52, s57 // (op 13729) 


s_add_i32 s52, s53, s55 // (op 13729) 


s_ashr_i32 s53, s52, s54 // (op 13729) 


s_lshl_b32 s52, s47, 24 // (op 13729) 
s_ashr_i32 s54, s52, 31 // (op 13729) 

s_xor_b32 s52, s53, s54 // (op 13729) 

s_sub_i32 s53, s52, s54 // (op 13729) 


s_mul_i32 s52, s0, s44 // (op 13984) 
s_sub_u32 s54, s16, s52 // (op 13984) 

s_mul_i32 s52, s19, s44 // (op 13984) 
s_add_u32 s52, s52, s54 // (op 13984) 

s_mul_hi_i32 s54, s16, s10 // (op 13984) 
s_add_i32 s55, s54, s16 // (op 13984) 

s_ashr_i32 s54, s55, 31 // (op 13984) 
s_cmp_eq_i32 s10, 0 // (op 13984) 
s_mov_b32 s56, scc // (op 13984) 
s_cmp_eq_i32 s56, 0 // (op 13984) 
s_cselect_b32 s57, 0, -1 // (op 13984) 

s_and_b32 s56, s47, 31 // (op 13984) 
s_lshl_b32 s58, 1, s56 // (op 13984) 
s_add_i32 s59, s58, s57 // (op 13984) 


s_and_b32 s57, s54, s59 // (op 13984) 


s_add_i32 s54, s55, s57 // (op 13984) 


s_ashr_i32 s55, s54, s56 // (op 13984) 


s_lshl_b32 s54, s47, 24 // (op 13984) 
s_ashr_i32 s56, s54, 31 // (op 13984) 

s_xor_b32 s54, s55, s56 // (op 13984) 

s_sub_i32 s55, s54, s56 // (op 13984) 


s_add_u32 s56, s6, 128 // (op 13729) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13729) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13729) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13729) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13729) 

s_mul_i32 s54, s44, s56 // (op 13729) 
s_mul_i32 s58, s48, s54 // (op 13729) 

s_mul_i32 s54, s53, s58 // (op 13729) 

s_sub_u32 s58, s16, s54 // (op 13729) 

s_mov_b32 s54, s58 // (op 13729) 
s_mul_hi_i32 s59, s54, s15 // (op 13729) 
s_add_i32 s60, s59, s54 // (op 13729) 


s_ashr_i32 s54, s60, 31 // (op 13729) 
s_cmp_eq_i32 s15, 0 // (op 13729) 
s_mov_b32 s59, scc // (op 13729) 
s_cmp_eq_i32 s59, 0 // (op 13729) 
s_cselect_b32 s61, 0, -1 // (op 13729) 

s_and_b32 s59, s46, 31 // (op 13729) 
s_lshl_b32 s62, 1, s59 // (op 13729) 
s_add_i32 s63, s62, s61 // (op 13729) 


s_and_b32 s61, s54, s63 // (op 13729) 


s_add_i32 s54, s60, s61 // (op 13729) 


s_ashr_i32 s60, s54, s59 // (op 13729) 


s_lshl_b32 s54, s46, 24 // (op 13729) 
s_ashr_i32 s59, s54, 31 // (op 13729) 

s_xor_b32 s54, s60, s59 // (op 13729) 

s_sub_i32 s60, s54, s59 // (op 13729) 


s_mul_i32 s54, s45, s44 // (op 13729) 
s_add_u32 s62, s8, 128 // (op 13729) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 13729) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 13729) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 13729) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 13729) 

s_sub_i32 s59, s62, s54 // (op 13729) 

s_mul_i32 s61, s60, s59 // (op 13729) 

s_sub_u32 s59, s58, s61 // (op 13729) 


s_add_u32 s58, s59, s54 // (op 13729) 


s_cmp_eq_i32 s53, 0 // (op 13729) 
s_mov_b32 s54, scc // (op 13729) 

s_cmp_eq_i32 s54, 0 // (op 13729) 
s_cselect_b32 s53, s58, s1 // (op 13729) 


s_mul_i32 s1, s51, s56 // (op 13729) 


v_lshl_add_u32 v21, s53, 7, v20 // (op 13729) 


s_sub_i32 s51, s18, s1 // (op 13729) 


s_cmp_eq_i32 s54, 0 // (op 13729) 
s_cselect_b32 s1, s60, s51 // (op 13729) 



v_lshl_add_u32 v20, s1, 7, v19 // (op 13729) 


v_mul_lo_u32 v19, v20, s2 // (op 13729) 

v_add_lshl_u32 v20, v21, v19, 1 // (op 13729) 


s_add_u32 s56, s6, 128 // (op 13984) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13984) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13984) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13984) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13984) 

s_mul_i32 s1, s44, s56 // (op 13984) 
s_mul_i32 s18, s48, s1 // (op 13984) 

s_mul_i32 s1, s55, s18 // (op 13984) 

s_sub_u32 s18, s16, s1 // (op 13984) 

s_mov_b32 s1, s18 // (op 13984) 
s_mul_hi_i32 s51, s1, s15 // (op 13984) 
s_add_i32 s53, s51, s1 // (op 13984) 


s_ashr_i32 s1, s53, 31 // (op 13984) 
s_cmp_eq_i32 s15, 0 // (op 13984) 
s_mov_b32 s51, scc // (op 13984) 
s_cmp_eq_i32 s51, 0 // (op 13984) 
s_cselect_b32 s54, 0, -1 // (op 13984) 

s_and_b32 s51, s46, 31 // (op 13984) 
s_lshl_b32 s58, 1, s51 // (op 13984) 
s_add_i32 s59, s58, s54 // (op 13984) 


s_and_b32 s54, s1, s59 // (op 13984) 


s_add_i32 s1, s53, s54 // (op 13984) 


s_ashr_i32 s53, s1, s51 // (op 13984) 


s_lshl_b32 s1, s46, 24 // (op 13984) 
s_ashr_i32 s51, s1, 31 // (op 13984) 

s_xor_b32 s1, s53, s51 // (op 13984) 

s_sub_i32 s53, s1, s51 // (op 13984) 


s_mul_i32 s1, s45, s44 // (op 13984) 
s_add_u32 s58, s8, 128 // (op 13984) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 13984) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 13984) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 13984) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 13984) 

s_sub_i32 s51, s58, s1 // (op 13984) 

s_mul_i32 s54, s53, s51 // (op 13984) 

s_sub_u32 s51, s18, s54 // (op 13984) 


s_add_u32 s18, s51, s1 // (op 13984) 


s_cmp_eq_i32 s55, 0 // (op 13984) 
s_mov_b32 s1, scc // (op 13984) 

s_cmp_eq_i32 s1, 0 // (op 13984) 
s_cselect_b32 s51, s18, s52 // (op 13984) 


s_mul_i32 s18, s19, s56 // (op 13984) 


v_lshl_add_u32 v19, s51, 7, v22 // (op 13984) 


s_sub_i32 s19, s0, s18 // (op 13984) 


s_cmp_eq_i32 s1, 0 // (op 13984) 
s_cselect_b32 s0, s53, s19 // (op 13984) 



v_lshl_add_u32 v21, s0, 7, v18 // (op 13984) 


v_mul_lo_u32 v18, v21, s4 // (op 13984) 

v_add_lshl_u32 v21, v19, v18, 1 // (op 13984) 


s_mul_i32 s1, s2, s7 // (op 13731) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13731) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13731) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13731) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13731) most significant: sum
s_add_u32 s1, s1, s19 // (op 13731) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13731) 

s_mul_i32 s1, s4, s7 // (op 13986) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 13986) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13986) least significant
s_mul_hi_u32 s52, s4, s6 // (op 13986) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 13986) most significant: sum
s_add_u32 s1, s1, s52 // (op 13986) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 13986) 

s_mov_b32 s56, s20 // (op 13731) reg expression
s_mov_b32 s57, s21 // (op 13731) reg expression
s_mov_b32 s58, s18 // (op 13731) reg expression
s_mov_b32 s59, 131072 // (op 13731) call()

s_mov_b32 s60, s22 // (op 13986) reg expression

s_mov_b32 s61, s23 // (op 13986) reg expression
s_mov_b32 s62, s52 // (op 13986) reg expression
s_mov_b32 s63, 131072 // (op 13986) call()


v_mov_b32 v18, s49 // (op 2) Move value
v_mov_b32 v19, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 486) call()
v_accvgpr_write a5, 0.00000 // (op 486) call()
v_accvgpr_write a6, 0.00000 // (op 486) call()
v_accvgpr_write a7, 0.00000 // (op 486) call()
v_accvgpr_write a8, 0.00000 // (op 784) call()
v_accvgpr_write a9, 0.00000 // (op 784) call()
v_accvgpr_write a10, 0.00000 // (op 784) call()
v_accvgpr_write a11, 0.00000 // (op 784) call()
v_accvgpr_write a12, 0.00000 // (op 1082) call()
v_accvgpr_write a13, 0.00000 // (op 1082) call()
v_accvgpr_write a14, 0.00000 // (op 1082) call()
v_accvgpr_write a15, 0.00000 // (op 1082) call()
v_accvgpr_write a16, 0.00000 // (op 1516) call()
v_accvgpr_write a17, 0.00000 // (op 1516) call()
v_accvgpr_write a18, 0.00000 // (op 1516) call()
v_accvgpr_write a19, 0.00000 // (op 1516) call()
v_accvgpr_write a20, 0.00000 // (op 2009) call()
v_accvgpr_write a21, 0.00000 // (op 2009) call()
v_accvgpr_write a22, 0.00000 // (op 2009) call()
v_accvgpr_write a23, 0.00000 // (op 2009) call()
v_accvgpr_write a24, 0.00000 // (op 2010) call()
v_accvgpr_write a25, 0.00000 // (op 2010) call()
v_accvgpr_write a26, 0.00000 // (op 2010) call()
v_accvgpr_write a27, 0.00000 // (op 2010) call()
v_accvgpr_write a28, 0.00000 // (op 2011) call()
v_accvgpr_write a29, 0.00000 // (op 2011) call()
v_accvgpr_write a30, 0.00000 // (op 2011) call()
v_accvgpr_write a31, 0.00000 // (op 2011) call()
v_accvgpr_write a32, 0.00000 // (op 2806) call()
v_accvgpr_write a33, 0.00000 // (op 2806) call()
v_accvgpr_write a34, 0.00000 // (op 2806) call()
v_accvgpr_write a35, 0.00000 // (op 2806) call()
v_accvgpr_write a36, 0.00000 // (op 3299) call()
v_accvgpr_write a37, 0.00000 // (op 3299) call()
v_accvgpr_write a38, 0.00000 // (op 3299) call()
v_accvgpr_write a39, 0.00000 // (op 3299) call()
v_accvgpr_write a40, 0.00000 // (op 3300) call()
v_accvgpr_write a41, 0.00000 // (op 3300) call()
v_accvgpr_write a42, 0.00000 // (op 3300) call()
v_accvgpr_write a43, 0.00000 // (op 3300) call()
v_accvgpr_write a44, 0.00000 // (op 3301) call()
v_accvgpr_write a45, 0.00000 // (op 3301) call()
v_accvgpr_write a46, 0.00000 // (op 3301) call()
v_accvgpr_write a47, 0.00000 // (op 3301) call()
v_accvgpr_write a48, 0.00000 // (op 4096) call()
v_accvgpr_write a49, 0.00000 // (op 4096) call()
v_accvgpr_write a50, 0.00000 // (op 4096) call()
v_accvgpr_write a51, 0.00000 // (op 4096) call()
v_accvgpr_write a52, 0.00000 // (op 4589) call()
v_accvgpr_write a53, 0.00000 // (op 4589) call()
v_accvgpr_write a54, 0.00000 // (op 4589) call()
v_accvgpr_write a55, 0.00000 // (op 4589) call()
v_accvgpr_write a56, 0.00000 // (op 4590) call()
v_accvgpr_write a57, 0.00000 // (op 4590) call()
v_accvgpr_write a58, 0.00000 // (op 4590) call()
v_accvgpr_write a59, 0.00000 // (op 4590) call()
v_accvgpr_write a60, 0.00000 // (op 4591) call()
v_accvgpr_write a61, 0.00000 // (op 4591) call()
v_accvgpr_write a62, 0.00000 // (op 4591) call()
v_accvgpr_write a63, 0.00000 // (op 4591) call()
s_mul_hi_i32 s0, s16, s11 // (op 13491) 
v_lshrrev_b32 v22, 2, v1 // (op 13491) 
s_add_i32 s1, s0, s16 // (op 13491) 

v_and_b32 v23, -4, v1 // (op 13491) 
s_ashr_i32 s0, s1, 31 // (op 13491) 
v_sub_u32 v24, v1, v23 // (op 13491) 

s_cmp_eq_i32 s11, 0 // (op 13491) 
s_mov_b32 s18, scc // (op 13491) 
v_lshlrev_b32 v23, 5, v24 // (op 13491) 

s_cmp_eq_i32 s18, 0 // (op 13491) 
s_cselect_b32 s19, 0, -1 // (op 13491) 

v_mov_b32 v25, 0 // (op 13491) convert
v_mov_b32 v24, v23 // (op 13491) convert

s_and_b32 s18, s12, 31 // (op 13491) 
s_lshl_b32 s20, 1, s18 // (op 13491) 
s_add_i32 s21, s20, s19 // (op 13491) 


s_and_b32 s19, s0, s21 // (op 13491) 


s_add_i32 s0, s1, s19 // (op 13491) 


s_ashr_i32 s1, s0, s18 // (op 13491) 


s_lshl_b32 s0, s12, 24 // (op 13491) 
s_ashr_i32 s18, s0, 31 // (op 13491) 

s_xor_b32 s0, s1, s18 // (op 13491) 

s_sub_i32 s1, s0, s18 // (op 13491) 


s_mul_hi_i32 s0, s1, s13 // (op 13491) 
s_add_i32 s18, s0, s1 // (op 13491) 

s_ashr_i32 s0, s18, 31 // (op 13491) 
s_cmp_eq_i32 s13, 0 // (op 13491) 
s_mov_b32 s19, scc // (op 13491) 
s_cmp_eq_i32 s19, 0 // (op 13491) 
s_cselect_b32 s20, 0, -1 // (op 13491) 

s_and_b32 s19, s14, 31 // (op 13491) 
s_lshl_b32 s21, 1, s19 // (op 13491) 
s_add_i32 s22, s21, s20 // (op 13491) 


s_and_b32 s23, s0, s22 // (op 13491) 


s_add_i32 s0, s18, s23 // (op 13491) 


s_ashr_i32 s18, s0, s19 // (op 13491) 


s_lshl_b32 s0, s14, 24 // (op 13491) 
s_ashr_i32 s19, s0, 31 // (op 13491) 

s_xor_b32 s0, s18, s19 // (op 13491) 

s_sub_i32 s18, s0, s19 // (op 13491) 


s_mul_i32 s0, s1, s44 // (op 13491) 

s_sub_u32 s1, s16, s0 // (op 13491) 

s_mul_i32 s0, s18, s44 // (op 13491) 
s_add_u32 s0, s0, s1 // (op 13491) 


s_mul_hi_i32 s1, s16, s10 // (op 13491) 
s_add_i32 s18, s1, s16 // (op 13491) 

s_ashr_i32 s1, s18, 31 // (op 13491) 
s_cmp_eq_i32 s10, 0 // (op 13491) 
s_mov_b32 s19, scc // (op 13491) 
s_cmp_eq_i32 s19, 0 // (op 13491) 
s_cselect_b32 s20, 0, -1 // (op 13491) 

s_and_b32 s19, s47, 31 // (op 13491) 
s_lshl_b32 s21, 1, s19 // (op 13491) 
s_add_i32 s22, s21, s20 // (op 13491) 


s_and_b32 s23, s1, s22 // (op 13491) 


s_add_i32 s1, s18, s23 // (op 13491) 


s_ashr_i32 s18, s1, s19 // (op 13491) 


s_lshl_b32 s1, s47, 24 // (op 13491) 
s_ashr_i32 s19, s1, 31 // (op 13491) 

s_xor_b32 s1, s18, s19 // (op 13491) 

s_sub_i32 s18, s1, s19 // (op 13491) 


s_add_u32 s22, s6, 128 // (op 13491) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13491) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 13491) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 13491) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 13491) 

s_mul_i32 s1, s44, s22 // (op 13491) 

s_mul_i32 s19, s48, s1 // (op 13491) 

s_mul_i32 s1, s18, s19 // (op 13491) 

s_sub_u32 s19, s16, s1 // (op 13491) 

s_mov_b32 s1, s19 // (op 13491) 
s_mul_hi_i32 s20, s1, s15 // (op 13491) 
s_add_i32 s21, s20, s1 // (op 13491) 


s_ashr_i32 s1, s21, 31 // (op 13491) 
s_cmp_eq_i32 s15, 0 // (op 13491) 
s_mov_b32 s20, scc // (op 13491) 
s_cmp_eq_i32 s20, 0 // (op 13491) 
s_cselect_b32 s22, 0, -1 // (op 13491) 

s_and_b32 s20, s46, 31 // (op 13491) 
s_lshl_b32 s23, 1, s20 // (op 13491) 
s_add_i32 s49, s23, s22 // (op 13491) 


s_and_b32 s22, s1, s49 // (op 13491) 


s_add_i32 s1, s21, s22 // (op 13491) 


s_ashr_i32 s21, s1, s20 // (op 13491) 


s_lshl_b32 s1, s46, 24 // (op 13491) 
s_ashr_i32 s20, s1, 31 // (op 13491) 

s_xor_b32 s1, s21, s20 // (op 13491) 

s_sub_i32 s21, s1, s20 // (op 13491) 


s_mul_i32 s1, s45, s44 // (op 13491) 
s_add_u32 s22, s8, 128 // (op 13491) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 13491) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 13491) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 13491) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 7 // (op 13491) 

s_sub_i32 s20, s22, s1 // (op 13491) 

s_mul_i32 s22, s21, s20 // (op 13491) 


s_sub_u32 s23, s19, s22 // (op 13491) 


s_add_u32 s19, s23, s1 // (op 13491) 


s_cmp_eq_i32 s18, 0 // (op 13491) 
s_mov_b32 s1, scc // (op 13491) 

s_cmp_eq_i32 s1, 0 // (op 13491) 
s_cselect_b32 s18, s19, s0 // (op 13491) 



v_lshl_add_u32 v23, s18, 7, v22 // (op 13491) 


v_mul_lo_u32 v27, v23, s25 // (op 13491) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13491) most significant: high of low * low
v_mul_lo_u32 v26, v23, s24 // (op 13491) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 13491) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 13491) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 13491) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 13491) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 13491) 

s_mul_i32 s1, s24, s9 // (op 13493) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13493) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13493) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13493) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13493) most significant: sum
s_add_u32 s1, s1, s19 // (op 13493) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13493) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13493) 

s_mov_b32 s20, s26 // (op 13493) reg expression
s_mov_b32 s21, s27 // (op 13493) reg expression
s_mov_b32 s22, s0 // (op 13493) reg expression
s_mov_b32 s23, 131072 // (op 13493) call()

s_mul_hi_i32 s0, s16, s11 // (op 13510) 
v_lshrrev_b32 v22, 2, v1 // (op 13510) 
s_add_i32 s1, s0, s16 // (op 13510) 

v_and_b32 v23, -4, v1 // (op 13510) 
s_ashr_i32 s0, s1, 31 // (op 13510) 
v_sub_u32 v26, v1, v23 // (op 13510) 

s_cmp_eq_i32 s11, 0 // (op 13510) 
s_mov_b32 s18, scc // (op 13510) 
v_lshlrev_b32 v23, 5, v26 // (op 13510) 

s_cmp_eq_i32 s18, 0 // (op 13510) 
s_cselect_b32 s19, 0, -1 // (op 13510) 

v_mov_b32 v27, 0 // (op 13510) convert
v_mov_b32 v26, v23 // (op 13510) convert

s_and_b32 s18, s12, 31 // (op 13510) 
s_lshl_b32 s49, 1, s18 // (op 13510) 
s_add_i32 s50, s49, s19 // (op 13510) 


s_and_b32 s19, s0, s50 // (op 13510) 


s_add_i32 s0, s1, s19 // (op 13510) 


s_ashr_i32 s1, s0, s18 // (op 13510) 


s_lshl_b32 s0, s12, 24 // (op 13510) 
s_ashr_i32 s18, s0, 31 // (op 13510) 

s_xor_b32 s0, s1, s18 // (op 13510) 

s_sub_i32 s1, s0, s18 // (op 13510) 


s_mul_hi_i32 s0, s1, s13 // (op 13510) 
s_add_i32 s18, s0, s1 // (op 13510) 

s_ashr_i32 s0, s18, 31 // (op 13510) 
s_cmp_eq_i32 s13, 0 // (op 13510) 
s_mov_b32 s19, scc // (op 13510) 
s_cmp_eq_i32 s19, 0 // (op 13510) 
s_cselect_b32 s49, 0, -1 // (op 13510) 

s_and_b32 s19, s14, 31 // (op 13510) 
s_lshl_b32 s50, 1, s19 // (op 13510) 
s_add_i32 s51, s50, s49 // (op 13510) 


s_and_b32 s49, s0, s51 // (op 13510) 


s_add_i32 s0, s18, s49 // (op 13510) 


s_ashr_i32 s18, s0, s19 // (op 13510) 


s_lshl_b32 s0, s14, 24 // (op 13510) 
s_ashr_i32 s19, s0, 31 // (op 13510) 

s_xor_b32 s0, s18, s19 // (op 13510) 

s_sub_i32 s18, s0, s19 // (op 13510) 


s_mul_i32 s0, s1, s44 // (op 13510) 

s_sub_u32 s1, s16, s0 // (op 13510) 

s_mul_i32 s0, s18, s44 // (op 13510) 
s_add_u32 s0, s0, s1 // (op 13510) 


s_mul_hi_i32 s1, s16, s10 // (op 13510) 
s_add_i32 s18, s1, s16 // (op 13510) 

s_ashr_i32 s1, s18, 31 // (op 13510) 
s_cmp_eq_i32 s10, 0 // (op 13510) 
s_mov_b32 s19, scc // (op 13510) 
s_cmp_eq_i32 s19, 0 // (op 13510) 
s_cselect_b32 s49, 0, -1 // (op 13510) 

s_and_b32 s19, s47, 31 // (op 13510) 
s_lshl_b32 s50, 1, s19 // (op 13510) 
s_add_i32 s51, s50, s49 // (op 13510) 


s_and_b32 s49, s1, s51 // (op 13510) 


s_add_i32 s1, s18, s49 // (op 13510) 


s_ashr_i32 s18, s1, s19 // (op 13510) 


s_lshl_b32 s1, s47, 24 // (op 13510) 
s_ashr_i32 s19, s1, 31 // (op 13510) 

s_xor_b32 s1, s18, s19 // (op 13510) 

s_sub_i32 s18, s1, s19 // (op 13510) 


s_add_u32 s54, s6, 128 // (op 13510) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 13510) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13510) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13510) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 13510) 

s_mul_i32 s1, s44, s50 // (op 13510) 

s_mul_i32 s19, s48, s1 // (op 13510) 

s_mul_i32 s1, s18, s19 // (op 13510) 

s_sub_u32 s19, s16, s1 // (op 13510) 

s_mov_b32 s1, s19 // (op 13510) 
s_mul_hi_i32 s49, s1, s15 // (op 13510) 
s_add_i32 s50, s49, s1 // (op 13510) 


s_ashr_i32 s1, s50, 31 // (op 13510) 
s_cmp_eq_i32 s15, 0 // (op 13510) 
s_mov_b32 s49, scc // (op 13510) 
s_cmp_eq_i32 s49, 0 // (op 13510) 
s_cselect_b32 s51, 0, -1 // (op 13510) 

s_and_b32 s49, s46, 31 // (op 13510) 
s_lshl_b32 s52, 1, s49 // (op 13510) 
s_add_i32 s53, s52, s51 // (op 13510) 


s_and_b32 s51, s1, s53 // (op 13510) 


s_add_i32 s1, s50, s51 // (op 13510) 


s_ashr_i32 s50, s1, s49 // (op 13510) 


s_lshl_b32 s1, s46, 24 // (op 13510) 
s_ashr_i32 s49, s1, 31 // (op 13510) 

s_xor_b32 s1, s50, s49 // (op 13510) 

s_sub_i32 s50, s1, s49 // (op 13510) 


s_mul_i32 s1, s45, s44 // (op 13510) 
s_add_u32 s54, s8, 128 // (op 13510) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 13510) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13510) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13510) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 13510) 

s_sub_i32 s49, s54, s1 // (op 13510) 

s_mul_i32 s51, s50, s49 // (op 13510) 


s_sub_u32 s49, s19, s51 // (op 13510) 


s_add_u32 s19, s49, s1 // (op 13510) 


s_cmp_eq_i32 s18, 0 // (op 13510) 
s_mov_b32 s1, scc // (op 13510) 

s_cmp_eq_i32 s1, 0 // (op 13510) 
s_cselect_b32 s18, s19, s0 // (op 13510) 



v_lshl_add_u32 v23, s18, 7, v22 // (op 13510) 


v_mul_lo_u32 v29, v23, s25 // (op 13510) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13510) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 13510) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 13510) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 13510) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 13510) most significant half



v_lshlrev_b64 v[22:23], 2, v[28:29] // (op 13510) 

v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 13510) 

s_mul_i32 s1, s24, s9 // (op 13512) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13512) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13512) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13512) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13512) most significant: sum
s_add_u32 s1, s1, s19 // (op 13512) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13512) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13512) 

s_mov_b32 s20, s26 // (op 13512) reg expression
s_mov_b32 s21, s27 // (op 13512) reg expression
s_mov_b32 s22, s0 // (op 13512) reg expression
s_mov_b32 s23, 131072 // (op 13512) call()

s_mul_hi_i32 s0, s16, s11 // (op 13555) 
v_and_b32 v22, -4, v1 // (op 13555) 
s_add_i32 s1, s0, s16 // (op 13555) 

v_sub_u32 v23, v1, v22 // (op 13555) 

s_ashr_i32 s0, s1, 31 // (op 13555) 
v_lshlrev_b32 v22, 5, v23 // (op 13555) 

s_cmp_eq_i32 s11, 0 // (op 13555) 
s_mov_b32 s18, scc // (op 13555) 
v_lshrrev_b32 v23, 2, v1 // (op 13555) 
s_cmp_eq_i32 s18, 0 // (op 13555) 
s_cselect_b32 s19, 0, -1 // (op 13555) 

s_and_b32 s18, s12, 31 // (op 13555) 
s_lshl_b32 s49, 1, s18 // (op 13555) 
s_add_i32 s50, s49, s19 // (op 13555) 


s_and_b32 s19, s0, s50 // (op 13555) 


s_add_i32 s0, s1, s19 // (op 13555) 


s_ashr_i32 s1, s0, s18 // (op 13555) 


s_lshl_b32 s0, s12, 24 // (op 13555) 
s_ashr_i32 s18, s0, 31 // (op 13555) 

s_xor_b32 s0, s1, s18 // (op 13555) 

s_sub_i32 s1, s0, s18 // (op 13555) 


s_mul_hi_i32 s0, s1, s13 // (op 13555) 
s_add_i32 s18, s0, s1 // (op 13555) 

s_ashr_i32 s0, s18, 31 // (op 13555) 
s_cmp_eq_i32 s13, 0 // (op 13555) 
s_mov_b32 s19, scc // (op 13555) 
s_cmp_eq_i32 s19, 0 // (op 13555) 
s_cselect_b32 s49, 0, -1 // (op 13555) 

s_and_b32 s19, s14, 31 // (op 13555) 
s_lshl_b32 s50, 1, s19 // (op 13555) 
s_add_i32 s51, s50, s49 // (op 13555) 


s_and_b32 s49, s0, s51 // (op 13555) 


s_add_i32 s0, s18, s49 // (op 13555) 


s_ashr_i32 s18, s0, s19 // (op 13555) 


s_lshl_b32 s0, s14, 24 // (op 13555) 
s_ashr_i32 s19, s0, 31 // (op 13555) 

s_xor_b32 s0, s18, s19 // (op 13555) 

s_sub_i32 s18, s0, s19 // (op 13555) 


s_add_u32 s54, s6, 128 // (op 13555) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 13555) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13555) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13555) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 13555) 

s_mul_i32 s0, s18, s50 // (op 13555) 

s_sub_i32 s49, s1, s0 // (op 13555) 


s_mul_hi_i32 s0, s16, s10 // (op 13555) 
s_add_i32 s1, s0, s16 // (op 13555) 

s_ashr_i32 s0, s1, 31 // (op 13555) 
s_cmp_eq_i32 s10, 0 // (op 13555) 
s_mov_b32 s18, scc // (op 13555) 
s_cmp_eq_i32 s18, 0 // (op 13555) 
s_cselect_b32 s19, 0, -1 // (op 13555) 

s_and_b32 s18, s47, 31 // (op 13555) 
s_lshl_b32 s52, 1, s18 // (op 13555) 
s_add_i32 s53, s52, s19 // (op 13555) 


s_and_b32 s19, s0, s53 // (op 13555) 


s_add_i32 s0, s1, s19 // (op 13555) 


s_ashr_i32 s1, s0, s18 // (op 13555) 


s_lshl_b32 s0, s47, 24 // (op 13555) 
s_ashr_i32 s18, s0, 31 // (op 13555) 

s_xor_b32 s0, s1, s18 // (op 13555) 

s_sub_i32 s1, s0, s18 // (op 13555) 


s_mul_i32 s0, s44, s50 // (op 13555) 

s_mul_i32 s18, s48, s0 // (op 13555) 

s_mul_i32 s0, s1, s18 // (op 13555) 

s_sub_u32 s18, s16, s0 // (op 13555) 

s_mul_hi_i32 s0, s18, s15 // (op 13555) 
s_add_i32 s19, s0, s18 // (op 13555) 


s_ashr_i32 s0, s19, 31 // (op 13555) 
s_cmp_eq_i32 s15, 0 // (op 13555) 
s_mov_b32 s18, scc // (op 13555) 
s_cmp_eq_i32 s18, 0 // (op 13555) 
s_cselect_b32 s50, 0, -1 // (op 13555) 

s_and_b32 s18, s46, 31 // (op 13555) 
s_lshl_b32 s51, 1, s18 // (op 13555) 
s_add_i32 s52, s51, s50 // (op 13555) 


s_and_b32 s50, s0, s52 // (op 13555) 


s_add_i32 s0, s19, s50 // (op 13555) 


s_ashr_i32 s19, s0, s18 // (op 13555) 


s_lshl_b32 s0, s46, 24 // (op 13555) 
s_ashr_i32 s18, s0, 31 // (op 13555) 

s_xor_b32 s0, s19, s18 // (op 13555) 

s_sub_i32 s19, s0, s18 // (op 13555) 


s_cmp_eq_i32 s1, 0 // (op 13555) 
s_mov_b32 s0, scc // (op 13555) 

s_cmp_eq_i32 s0, 0 // (op 13555) 
s_cselect_b32 s1, s19, s49 // (op 13555) 



v_lshl_add_u32 v28, s1, 7, v22 // (op 13555) 


v_mov_b32 v31, 0 // (op 13555) convert
v_mov_b32 v30, v28 // (op 13555) convert

v_mul_lo_u32 v29, v23, s29 // (op 13555) most significant: low of low * high
v_mul_hi_u32 v22, v23, s28 // (op 13555) most significant: high of low * low
v_mul_lo_u32 v28, v23, s28 // (op 13555) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 13555) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v30 // (op 13555) least significant half
v_addc_co_u32 v29, s[0:1], v29, v31, s[0:1] // (op 13555) most significant half



v_lshlrev_b64 v[22:23], 2, v[28:29] // (op 13555) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 13555) 

s_mul_i32 s1, s28, s33 // (op 13557) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13557) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13557) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13557) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13557) most significant: sum
s_add_u32 s1, s1, s19 // (op 13557) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13557) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13557) 

s_mov_b32 s52, s30 // (op 13557) reg expression
s_mov_b32 s53, s31 // (op 13557) reg expression
s_mov_b32 s54, s0 // (op 13557) reg expression
s_mov_b32 s55, 131072 // (op 13557) call()

s_mul_hi_i32 s0, s16, s11 // (op 13574) 
v_and_b32 v22, -4, v1 // (op 13574) 
s_add_i32 s1, s0, s16 // (op 13574) 

v_sub_u32 v23, v1, v22 // (op 13574) 

s_ashr_i32 s0, s1, 31 // (op 13574) 
v_lshlrev_b32 v22, 5, v23 // (op 13574) 

s_cmp_eq_i32 s11, 0 // (op 13574) 
s_mov_b32 s18, scc // (op 13574) 
v_lshrrev_b32 v23, 2, v1 // (op 13574) 
s_cmp_eq_i32 s18, 0 // (op 13574) 
s_cselect_b32 s19, 0, -1 // (op 13574) 

s_and_b32 s18, s12, 31 // (op 13574) 
s_lshl_b32 s49, 1, s18 // (op 13574) 
s_add_i32 s50, s49, s19 // (op 13574) 


s_and_b32 s19, s0, s50 // (op 13574) 


s_add_i32 s0, s1, s19 // (op 13574) 


s_ashr_i32 s1, s0, s18 // (op 13574) 


s_lshl_b32 s0, s12, 24 // (op 13574) 
s_ashr_i32 s18, s0, 31 // (op 13574) 

s_xor_b32 s0, s1, s18 // (op 13574) 

s_sub_i32 s1, s0, s18 // (op 13574) 


s_mul_hi_i32 s0, s1, s13 // (op 13574) 
s_add_i32 s18, s0, s1 // (op 13574) 

s_ashr_i32 s0, s18, 31 // (op 13574) 
s_cmp_eq_i32 s13, 0 // (op 13574) 
s_mov_b32 s19, scc // (op 13574) 
s_cmp_eq_i32 s19, 0 // (op 13574) 
s_cselect_b32 s49, 0, -1 // (op 13574) 

s_and_b32 s19, s14, 31 // (op 13574) 
s_lshl_b32 s50, 1, s19 // (op 13574) 
s_add_i32 s51, s50, s49 // (op 13574) 


s_and_b32 s49, s0, s51 // (op 13574) 


s_add_i32 s0, s18, s49 // (op 13574) 


s_ashr_i32 s18, s0, s19 // (op 13574) 


s_lshl_b32 s0, s14, 24 // (op 13574) 
s_ashr_i32 s19, s0, 31 // (op 13574) 

s_xor_b32 s0, s18, s19 // (op 13574) 

s_sub_i32 s18, s0, s19 // (op 13574) 


s_add_u32 s50, s6, 128 // (op 13574) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13574) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 13574) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 13574) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 13574) 

s_mul_i32 s0, s18, s50 // (op 13574) 

s_sub_i32 s49, s1, s0 // (op 13574) 


s_mul_hi_i32 s0, s16, s10 // (op 13574) 
s_add_i32 s1, s0, s16 // (op 13574) 

s_ashr_i32 s0, s1, 31 // (op 13574) 
s_cmp_eq_i32 s10, 0 // (op 13574) 
s_mov_b32 s18, scc // (op 13574) 
s_cmp_eq_i32 s18, 0 // (op 13574) 
s_cselect_b32 s19, 0, -1 // (op 13574) 

s_and_b32 s18, s47, 31 // (op 13574) 
s_lshl_b32 s64, 1, s18 // (op 13574) 
s_add_i32 s65, s64, s19 // (op 13574) 


s_and_b32 s19, s0, s65 // (op 13574) 


s_add_i32 s0, s1, s19 // (op 13574) 


s_ashr_i32 s1, s0, s18 // (op 13574) 


s_lshl_b32 s0, s47, 24 // (op 13574) 
s_ashr_i32 s18, s0, 31 // (op 13574) 

s_xor_b32 s0, s1, s18 // (op 13574) 

s_sub_i32 s1, s0, s18 // (op 13574) 


s_mul_i32 s0, s44, s50 // (op 13574) 

s_mul_i32 s18, s48, s0 // (op 13574) 

s_mul_i32 s0, s1, s18 // (op 13574) 

s_sub_u32 s18, s16, s0 // (op 13574) 

s_mul_hi_i32 s0, s18, s15 // (op 13574) 
s_add_i32 s19, s0, s18 // (op 13574) 


s_ashr_i32 s0, s19, 31 // (op 13574) 
s_cmp_eq_i32 s15, 0 // (op 13574) 
s_mov_b32 s18, scc // (op 13574) 
s_cmp_eq_i32 s18, 0 // (op 13574) 
s_cselect_b32 s50, 0, -1 // (op 13574) 

s_and_b32 s18, s46, 31 // (op 13574) 
s_lshl_b32 s51, 1, s18 // (op 13574) 
s_add_i32 s64, s51, s50 // (op 13574) 


s_and_b32 s50, s0, s64 // (op 13574) 


s_add_i32 s0, s19, s50 // (op 13574) 


s_ashr_i32 s19, s0, s18 // (op 13574) 


s_lshl_b32 s0, s46, 24 // (op 13574) 
s_ashr_i32 s18, s0, 31 // (op 13574) 

s_xor_b32 s0, s19, s18 // (op 13574) 

s_sub_i32 s19, s0, s18 // (op 13574) 


s_cmp_eq_i32 s1, 0 // (op 13574) 
s_mov_b32 s0, scc // (op 13574) 

s_cmp_eq_i32 s0, 0 // (op 13574) 
s_cselect_b32 s1, s19, s49 // (op 13574) 



v_lshl_add_u32 v30, s1, 7, v22 // (op 13574) 


v_mov_b32 v33, 0 // (op 13574) convert
v_mov_b32 v32, v30 // (op 13574) convert

v_mul_lo_u32 v31, v23, s29 // (op 13574) most significant: low of low * high
v_mul_hi_u32 v22, v23, s28 // (op 13574) most significant: high of low * low
v_mul_lo_u32 v30, v23, s28 // (op 13574) least significant: low of low * low
v_add_u32 v31, v31, v22 // (op 13574) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v32 // (op 13574) least significant half
v_addc_co_u32 v31, s[0:1], v31, v33, s[0:1] // (op 13574) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 13574) 

v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 13574) 

s_mul_i32 s1, s28, s33 // (op 13576) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13576) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13576) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13576) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13576) most significant: sum
s_add_u32 s1, s1, s19 // (op 13576) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13576) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13576) 

s_mov_b32 s52, s30 // (op 13576) reg expression
s_mov_b32 s53, s31 // (op 13576) reg expression
s_mov_b32 s54, s0 // (op 13576) reg expression
s_mov_b32 s55, 131072 // (op 13576) call()

s_mul_hi_i32 s0, s16, s11 // (op 13619) 
v_lshrrev_b32 v22, 2, v1 // (op 13619) 
s_add_i32 s1, s0, s16 // (op 13619) 

v_lshlrev_b32 v23, 1, v22 // (op 13619) 

s_ashr_i32 s0, s1, 31 // (op 13619) 
v_and_b32 v22, -4, v1 // (op 13619) 
s_cmp_eq_i32 s11, 0 // (op 13619) 
s_mov_b32 s18, scc // (op 13619) 
v_sub_u32 v32, v1, v22 // (op 13619) 

s_cmp_eq_i32 s18, 0 // (op 13619) 
s_cselect_b32 s19, 0, -1 // (op 13619) 

s_and_b32 s18, s12, 31 // (op 13619) 
s_lshl_b32 s49, 1, s18 // (op 13619) 
s_add_i32 s50, s49, s19 // (op 13619) 


s_and_b32 s19, s0, s50 // (op 13619) 


s_add_i32 s0, s1, s19 // (op 13619) 


s_ashr_i32 s1, s0, s18 // (op 13619) 


s_lshl_b32 s0, s12, 24 // (op 13619) 
s_ashr_i32 s18, s0, 31 // (op 13619) 

s_xor_b32 s0, s1, s18 // (op 13619) 

s_sub_i32 s1, s0, s18 // (op 13619) 


s_mul_hi_i32 s0, s1, s13 // (op 13619) 
s_add_i32 s18, s0, s1 // (op 13619) 

s_ashr_i32 s0, s18, 31 // (op 13619) 
s_cmp_eq_i32 s13, 0 // (op 13619) 
s_mov_b32 s19, scc // (op 13619) 
s_cmp_eq_i32 s19, 0 // (op 13619) 
s_cselect_b32 s49, 0, -1 // (op 13619) 

s_and_b32 s19, s14, 31 // (op 13619) 
s_lshl_b32 s50, 1, s19 // (op 13619) 
s_add_i32 s51, s50, s49 // (op 13619) 


s_and_b32 s49, s0, s51 // (op 13619) 


s_add_i32 s0, s18, s49 // (op 13619) 


s_ashr_i32 s18, s0, s19 // (op 13619) 


s_lshl_b32 s0, s14, 24 // (op 13619) 
s_ashr_i32 s19, s0, 31 // (op 13619) 

s_xor_b32 s0, s18, s19 // (op 13619) 

s_sub_i32 s18, s0, s19 // (op 13619) 


s_mul_i32 s0, s1, s44 // (op 13619) 

s_sub_u32 s1, s16, s0 // (op 13619) 

s_mul_i32 s0, s18, s44 // (op 13619) 
s_add_u32 s0, s0, s1 // (op 13619) 


s_mul_hi_i32 s1, s16, s10 // (op 13619) 
s_add_i32 s18, s1, s16 // (op 13619) 

s_ashr_i32 s1, s18, 31 // (op 13619) 
s_cmp_eq_i32 s10, 0 // (op 13619) 
s_mov_b32 s19, scc // (op 13619) 
s_cmp_eq_i32 s19, 0 // (op 13619) 
s_cselect_b32 s49, 0, -1 // (op 13619) 

s_and_b32 s19, s47, 31 // (op 13619) 
s_lshl_b32 s50, 1, s19 // (op 13619) 
s_add_i32 s51, s50, s49 // (op 13619) 


s_and_b32 s49, s1, s51 // (op 13619) 


s_add_i32 s1, s18, s49 // (op 13619) 


s_ashr_i32 s18, s1, s19 // (op 13619) 


s_lshl_b32 s1, s47, 24 // (op 13619) 
s_ashr_i32 s19, s1, 31 // (op 13619) 

s_xor_b32 s1, s18, s19 // (op 13619) 

s_sub_i32 s18, s1, s19 // (op 13619) 


s_add_u32 s50, s6, 128 // (op 13619) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13619) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 13619) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 13619) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 13619) 

s_mul_i32 s1, s44, s50 // (op 13619) 

s_mul_i32 s19, s48, s1 // (op 13619) 

s_mul_i32 s1, s18, s19 // (op 13619) 

s_sub_u32 s19, s16, s1 // (op 13619) 

s_mov_b32 s1, s19 // (op 13619) 
s_mul_hi_i32 s49, s1, s15 // (op 13619) 
s_add_i32 s50, s49, s1 // (op 13619) 


s_ashr_i32 s1, s50, 31 // (op 13619) 
s_cmp_eq_i32 s15, 0 // (op 13619) 
s_mov_b32 s49, scc // (op 13619) 
s_cmp_eq_i32 s49, 0 // (op 13619) 
s_cselect_b32 s51, 0, -1 // (op 13619) 

s_and_b32 s49, s46, 31 // (op 13619) 
s_lshl_b32 s64, 1, s49 // (op 13619) 
s_add_i32 s65, s64, s51 // (op 13619) 


s_and_b32 s51, s1, s65 // (op 13619) 


s_add_i32 s1, s50, s51 // (op 13619) 


s_ashr_i32 s50, s1, s49 // (op 13619) 


s_lshl_b32 s1, s46, 24 // (op 13619) 
s_ashr_i32 s49, s1, 31 // (op 13619) 

s_xor_b32 s1, s50, s49 // (op 13619) 

s_sub_i32 s50, s1, s49 // (op 13619) 


s_mul_i32 s1, s45, s44 // (op 13619) 
s_add_u32 s64, s8, 128 // (op 13619) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 13619) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 13619) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 13619) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 13619) 

s_sub_i32 s49, s64, s1 // (op 13619) 

s_mul_i32 s51, s50, s49 // (op 13619) 


s_sub_u32 s49, s19, s51 // (op 13619) 


s_add_u32 s19, s49, s1 // (op 13619) 


s_cmp_eq_i32 s18, 0 // (op 13619) 
s_mov_b32 s1, scc // (op 13619) 

s_cmp_eq_i32 s1, 0 // (op 13619) 
s_cselect_b32 s18, s19, s0 // (op 13619) 



v_lshl_add_u32 v22, s18, 7, v23 // (op 13619) 


v_mul_lo_u32 v23, v22, s34 // (op 13619) 
v_add_u32 v23, v23, v32 // (op 13619) 


s_mul_i32 s1, s34, s9 // (op 13621) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13621) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13621) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13621) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13621) most significant: sum
s_add_u32 s1, s1, s19 // (op 13621) most significant: sum


s_mov_b32 s64, s36 // (op 13621) reg expression
s_mov_b32 s65, s37 // (op 13621) reg expression
s_mov_b32 s66, s0 // (op 13621) reg expression
s_mov_b32 s67, 131072 // (op 13621) call()

s_mul_hi_i32 s0, s16, s11 // (op 13638) 
v_lshrrev_b32 v22, 2, v1 // (op 13638) 
s_add_i32 s1, s0, s16 // (op 13638) 

v_lshlrev_b32 v32, 1, v22 // (op 13638) 

s_ashr_i32 s0, s1, 31 // (op 13638) 
v_and_b32 v22, -4, v1 // (op 13638) 
s_cmp_eq_i32 s11, 0 // (op 13638) 
s_mov_b32 s18, scc // (op 13638) 
v_sub_u32 v33, v1, v22 // (op 13638) 

s_cmp_eq_i32 s18, 0 // (op 13638) 
s_cselect_b32 s19, 0, -1 // (op 13638) 

s_and_b32 s18, s12, 31 // (op 13638) 
s_lshl_b32 s49, 1, s18 // (op 13638) 
s_add_i32 s50, s49, s19 // (op 13638) 


s_and_b32 s19, s0, s50 // (op 13638) 


s_add_i32 s0, s1, s19 // (op 13638) 


s_ashr_i32 s1, s0, s18 // (op 13638) 


s_lshl_b32 s0, s12, 24 // (op 13638) 
s_ashr_i32 s18, s0, 31 // (op 13638) 

s_xor_b32 s0, s1, s18 // (op 13638) 

s_sub_i32 s1, s0, s18 // (op 13638) 


s_mul_hi_i32 s0, s1, s13 // (op 13638) 
s_add_i32 s18, s0, s1 // (op 13638) 

s_ashr_i32 s0, s18, 31 // (op 13638) 
s_cmp_eq_i32 s13, 0 // (op 13638) 
s_mov_b32 s19, scc // (op 13638) 
s_cmp_eq_i32 s19, 0 // (op 13638) 
s_cselect_b32 s49, 0, -1 // (op 13638) 

s_and_b32 s19, s14, 31 // (op 13638) 
s_lshl_b32 s50, 1, s19 // (op 13638) 
s_add_i32 s51, s50, s49 // (op 13638) 


s_and_b32 s49, s0, s51 // (op 13638) 


s_add_i32 s0, s18, s49 // (op 13638) 


s_ashr_i32 s18, s0, s19 // (op 13638) 


s_lshl_b32 s0, s14, 24 // (op 13638) 
s_ashr_i32 s19, s0, 31 // (op 13638) 

s_xor_b32 s0, s18, s19 // (op 13638) 

s_sub_i32 s18, s0, s19 // (op 13638) 


s_mul_i32 s0, s1, s44 // (op 13638) 

s_sub_u32 s1, s16, s0 // (op 13638) 

s_mul_i32 s0, s18, s44 // (op 13638) 
s_add_u32 s0, s0, s1 // (op 13638) 


s_mul_hi_i32 s1, s16, s10 // (op 13638) 
s_add_i32 s18, s1, s16 // (op 13638) 

s_ashr_i32 s1, s18, 31 // (op 13638) 
s_cmp_eq_i32 s10, 0 // (op 13638) 
s_mov_b32 s19, scc // (op 13638) 
s_cmp_eq_i32 s19, 0 // (op 13638) 
s_cselect_b32 s49, 0, -1 // (op 13638) 

s_and_b32 s19, s47, 31 // (op 13638) 
s_lshl_b32 s50, 1, s19 // (op 13638) 
s_add_i32 s51, s50, s49 // (op 13638) 


s_and_b32 s49, s1, s51 // (op 13638) 


s_add_i32 s1, s18, s49 // (op 13638) 


s_ashr_i32 s18, s1, s19 // (op 13638) 


s_lshl_b32 s1, s47, 24 // (op 13638) 
s_ashr_i32 s19, s1, 31 // (op 13638) 

s_xor_b32 s1, s18, s19 // (op 13638) 

s_sub_i32 s18, s1, s19 // (op 13638) 


s_add_u32 s50, s6, 128 // (op 13638) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13638) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 13638) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 13638) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 13638) 

s_mul_i32 s1, s44, s50 // (op 13638) 

s_mul_i32 s19, s48, s1 // (op 13638) 

s_mul_i32 s1, s18, s19 // (op 13638) 

s_sub_u32 s19, s16, s1 // (op 13638) 

s_mov_b32 s1, s19 // (op 13638) 
s_mul_hi_i32 s49, s1, s15 // (op 13638) 
s_add_i32 s50, s49, s1 // (op 13638) 


s_ashr_i32 s1, s50, 31 // (op 13638) 
s_cmp_eq_i32 s15, 0 // (op 13638) 
s_mov_b32 s49, scc // (op 13638) 
s_cmp_eq_i32 s49, 0 // (op 13638) 
s_cselect_b32 s51, 0, -1 // (op 13638) 

s_and_b32 s49, s46, 31 // (op 13638) 
s_lshl_b32 s68, 1, s49 // (op 13638) 
s_add_i32 s69, s68, s51 // (op 13638) 


s_and_b32 s51, s1, s69 // (op 13638) 


s_add_i32 s1, s50, s51 // (op 13638) 


s_ashr_i32 s50, s1, s49 // (op 13638) 


s_lshl_b32 s1, s46, 24 // (op 13638) 
s_ashr_i32 s49, s1, 31 // (op 13638) 

s_xor_b32 s1, s50, s49 // (op 13638) 

s_sub_i32 s50, s1, s49 // (op 13638) 


s_mul_i32 s1, s45, s44 // (op 13638) 
s_add_u32 s68, s8, 128 // (op 13638) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 13638) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 13638) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 13638) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 13638) 

s_sub_i32 s49, s68, s1 // (op 13638) 

s_mul_i32 s51, s50, s49 // (op 13638) 


s_sub_u32 s49, s19, s51 // (op 13638) 


s_add_u32 s19, s49, s1 // (op 13638) 


s_cmp_eq_i32 s18, 0 // (op 13638) 
s_mov_b32 s1, scc // (op 13638) 

s_cmp_eq_i32 s1, 0 // (op 13638) 
s_cselect_b32 s18, s19, s0 // (op 13638) 



v_lshl_add_u32 v22, s18, 7, v32 // (op 13638) 


v_mul_lo_u32 v32, v22, s34 // (op 13638) 
v_add_u32 v32, v32, v33 // (op 13638) 


s_mul_i32 s1, s34, s9 // (op 13640) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13640) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13640) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13640) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13640) most significant: sum
s_add_u32 s1, s1, s19 // (op 13640) most significant: sum


s_mov_b32 s64, s36 // (op 13640) reg expression
s_mov_b32 s65, s37 // (op 13640) reg expression
s_mov_b32 s66, s0 // (op 13640) reg expression
s_mov_b32 s67, 131072 // (op 13640) call()

s_mul_hi_i32 s0, s16, s11 // (op 13683) 
v_and_b32 v22, -128, v1 // (op 13683) 
s_add_i32 s1, s0, s16 // (op 13683) 

v_sub_u32 v33, v1, v22 // (op 13683) 

s_ashr_i32 s0, s1, 31 // (op 13683) 
v_lshrrev_b32 v22, 7, v1 // (op 13683) 
s_cmp_eq_i32 s11, 0 // (op 13683) 
s_mov_b32 s18, scc // (op 13683) 
v_lshlrev_b32 v34, 1, v22 // (op 13683) 

s_cmp_eq_i32 s18, 0 // (op 13683) 
s_cselect_b32 s19, 0, -1 // (op 13683) 

s_and_b32 s18, s12, 31 // (op 13683) 
s_lshl_b32 s49, 1, s18 // (op 13683) 
s_add_i32 s50, s49, s19 // (op 13683) 


s_and_b32 s19, s0, s50 // (op 13683) 


s_add_i32 s0, s1, s19 // (op 13683) 


s_ashr_i32 s1, s0, s18 // (op 13683) 


s_lshl_b32 s0, s12, 24 // (op 13683) 
s_ashr_i32 s18, s0, 31 // (op 13683) 

s_xor_b32 s0, s1, s18 // (op 13683) 

s_sub_i32 s1, s0, s18 // (op 13683) 


s_mul_hi_i32 s0, s1, s13 // (op 13683) 
s_add_i32 s18, s0, s1 // (op 13683) 

s_ashr_i32 s0, s18, 31 // (op 13683) 
s_cmp_eq_i32 s13, 0 // (op 13683) 
s_mov_b32 s19, scc // (op 13683) 
s_cmp_eq_i32 s19, 0 // (op 13683) 
s_cselect_b32 s49, 0, -1 // (op 13683) 

s_and_b32 s19, s14, 31 // (op 13683) 
s_lshl_b32 s50, 1, s19 // (op 13683) 
s_add_i32 s51, s50, s49 // (op 13683) 


s_and_b32 s49, s0, s51 // (op 13683) 


s_add_i32 s0, s18, s49 // (op 13683) 


s_ashr_i32 s18, s0, s19 // (op 13683) 


s_lshl_b32 s0, s14, 24 // (op 13683) 
s_ashr_i32 s19, s0, 31 // (op 13683) 

s_xor_b32 s0, s18, s19 // (op 13683) 

s_sub_i32 s18, s0, s19 // (op 13683) 


s_add_u32 s50, s6, 128 // (op 13683) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13683) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 13683) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 13683) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 13683) 

s_mul_i32 s0, s18, s50 // (op 13683) 

s_sub_i32 s49, s1, s0 // (op 13683) 


s_mul_hi_i32 s0, s16, s10 // (op 13683) 
s_add_i32 s1, s0, s16 // (op 13683) 

s_ashr_i32 s0, s1, 31 // (op 13683) 
s_cmp_eq_i32 s10, 0 // (op 13683) 
s_mov_b32 s18, scc // (op 13683) 
s_cmp_eq_i32 s18, 0 // (op 13683) 
s_cselect_b32 s19, 0, -1 // (op 13683) 

s_and_b32 s18, s47, 31 // (op 13683) 
s_lshl_b32 s68, 1, s18 // (op 13683) 
s_add_i32 s69, s68, s19 // (op 13683) 


s_and_b32 s19, s0, s69 // (op 13683) 


s_add_i32 s0, s1, s19 // (op 13683) 


s_ashr_i32 s1, s0, s18 // (op 13683) 


s_lshl_b32 s0, s47, 24 // (op 13683) 
s_ashr_i32 s18, s0, 31 // (op 13683) 

s_xor_b32 s0, s1, s18 // (op 13683) 

s_sub_i32 s1, s0, s18 // (op 13683) 


s_mul_i32 s0, s44, s50 // (op 13683) 

s_mul_i32 s18, s48, s0 // (op 13683) 

s_mul_i32 s0, s1, s18 // (op 13683) 

s_sub_u32 s18, s16, s0 // (op 13683) 

s_mul_hi_i32 s0, s18, s15 // (op 13683) 
s_add_i32 s19, s0, s18 // (op 13683) 


s_ashr_i32 s0, s19, 31 // (op 13683) 
s_cmp_eq_i32 s15, 0 // (op 13683) 
s_mov_b32 s18, scc // (op 13683) 
s_cmp_eq_i32 s18, 0 // (op 13683) 
s_cselect_b32 s50, 0, -1 // (op 13683) 

s_and_b32 s18, s46, 31 // (op 13683) 
s_lshl_b32 s51, 1, s18 // (op 13683) 
s_add_i32 s68, s51, s50 // (op 13683) 


s_and_b32 s50, s0, s68 // (op 13683) 


s_add_i32 s0, s19, s50 // (op 13683) 


s_ashr_i32 s19, s0, s18 // (op 13683) 


s_lshl_b32 s0, s46, 24 // (op 13683) 
s_ashr_i32 s18, s0, 31 // (op 13683) 

s_xor_b32 s0, s19, s18 // (op 13683) 

s_sub_i32 s19, s0, s18 // (op 13683) 


s_cmp_eq_i32 s1, 0 // (op 13683) 
s_mov_b32 s0, scc // (op 13683) 

s_cmp_eq_i32 s0, 0 // (op 13683) 
s_cselect_b32 s1, s19, s49 // (op 13683) 



v_lshl_add_u32 v22, s1, 7, v33 // (op 13683) 


v_mul_lo_u32 v33, v34, s38 // (op 13683) 
v_add_u32 v33, v33, v22 // (op 13683) 


s_mul_i32 s1, s38, s43 // (op 13685) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13685) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13685) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13685) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13685) most significant: sum
s_add_u32 s1, s1, s19 // (op 13685) most significant: sum


s_mov_b32 s68, s40 // (op 13685) reg expression
s_mov_b32 s69, s41 // (op 13685) reg expression
s_mov_b32 s70, s0 // (op 13685) reg expression
s_mov_b32 s71, 131072 // (op 13685) call()

s_mul_hi_i32 s0, s16, s11 // (op 13702) 
v_and_b32 v22, -128, v1 // (op 13702) 
s_add_i32 s1, s0, s16 // (op 13702) 

v_sub_u32 v34, v1, v22 // (op 13702) 

s_ashr_i32 s0, s1, 31 // (op 13702) 
v_lshrrev_b32 v22, 7, v1 // (op 13702) 
s_cmp_eq_i32 s11, 0 // (op 13702) 
s_mov_b32 s18, scc // (op 13702) 
v_lshlrev_b32 v35, 1, v22 // (op 13702) 

s_cmp_eq_i32 s18, 0 // (op 13702) 
s_cselect_b32 s19, 0, -1 // (op 13702) 

s_and_b32 s18, s12, 31 // (op 13702) 
s_lshl_b32 s49, 1, s18 // (op 13702) 
s_add_i32 s50, s49, s19 // (op 13702) 


s_and_b32 s19, s0, s50 // (op 13702) 


s_add_i32 s0, s1, s19 // (op 13702) 


s_ashr_i32 s1, s0, s18 // (op 13702) 


s_lshl_b32 s0, s12, 24 // (op 13702) 
s_ashr_i32 s18, s0, 31 // (op 13702) 

s_xor_b32 s0, s1, s18 // (op 13702) 

s_sub_i32 s1, s0, s18 // (op 13702) 


s_mul_hi_i32 s0, s1, s13 // (op 13702) 
s_add_i32 s18, s0, s1 // (op 13702) 

s_ashr_i32 s0, s18, 31 // (op 13702) 
s_cmp_eq_i32 s13, 0 // (op 13702) 
s_mov_b32 s19, scc // (op 13702) 
s_cmp_eq_i32 s19, 0 // (op 13702) 
s_cselect_b32 s49, 0, -1 // (op 13702) 

s_and_b32 s19, s14, 31 // (op 13702) 
s_lshl_b32 s50, 1, s19 // (op 13702) 
s_add_i32 s51, s50, s49 // (op 13702) 


s_and_b32 s49, s0, s51 // (op 13702) 


s_add_i32 s0, s18, s49 // (op 13702) 


s_ashr_i32 s18, s0, s19 // (op 13702) 


s_lshl_b32 s0, s14, 24 // (op 13702) 
s_ashr_i32 s19, s0, 31 // (op 13702) 

s_xor_b32 s0, s18, s19 // (op 13702) 

s_sub_i32 s18, s0, s19 // (op 13702) 


s_add_u32 s50, s6, 128 // (op 13702) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13702) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13702) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13702) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13702) 

s_mul_i32 s0, s18, s50 // (op 13702) 

s_sub_i32 s49, s1, s0 // (op 13702) 


s_mul_hi_i32 s0, s16, s10 // (op 13702) 
s_add_i32 s1, s0, s16 // (op 13702) 

s_ashr_i32 s0, s1, 31 // (op 13702) 
s_cmp_eq_i32 s10, 0 // (op 13702) 
s_mov_b32 s18, scc // (op 13702) 
s_cmp_eq_i32 s18, 0 // (op 13702) 
s_cselect_b32 s19, 0, -1 // (op 13702) 

s_and_b32 s18, s47, 31 // (op 13702) 
s_lshl_b32 s72, 1, s18 // (op 13702) 
s_add_i32 s73, s72, s19 // (op 13702) 


s_and_b32 s19, s0, s73 // (op 13702) 


s_add_i32 s0, s1, s19 // (op 13702) 


s_ashr_i32 s1, s0, s18 // (op 13702) 


s_lshl_b32 s0, s47, 24 // (op 13702) 
s_ashr_i32 s18, s0, 31 // (op 13702) 

s_xor_b32 s0, s1, s18 // (op 13702) 

s_sub_i32 s1, s0, s18 // (op 13702) 


s_mul_i32 s0, s44, s50 // (op 13702) 

s_mul_i32 s18, s48, s0 // (op 13702) 

s_mul_i32 s0, s1, s18 // (op 13702) 

s_sub_u32 s18, s16, s0 // (op 13702) 

s_mul_hi_i32 s0, s18, s15 // (op 13702) 
s_add_i32 s19, s0, s18 // (op 13702) 


s_ashr_i32 s0, s19, 31 // (op 13702) 
s_cmp_eq_i32 s15, 0 // (op 13702) 
s_mov_b32 s18, scc // (op 13702) 
s_cmp_eq_i32 s18, 0 // (op 13702) 
s_cselect_b32 s50, 0, -1 // (op 13702) 

s_and_b32 s18, s46, 31 // (op 13702) 
s_lshl_b32 s51, 1, s18 // (op 13702) 
s_add_i32 s72, s51, s50 // (op 13702) 


s_and_b32 s50, s0, s72 // (op 13702) 


s_add_i32 s0, s19, s50 // (op 13702) 


s_ashr_i32 s19, s0, s18 // (op 13702) 


s_lshl_b32 s0, s46, 24 // (op 13702) 
s_ashr_i32 s18, s0, 31 // (op 13702) 

s_xor_b32 s0, s19, s18 // (op 13702) 

s_sub_i32 s19, s0, s18 // (op 13702) 


s_cmp_eq_i32 s1, 0 // (op 13702) 
s_mov_b32 s0, scc // (op 13702) 

s_cmp_eq_i32 s0, 0 // (op 13702) 
s_cselect_b32 s1, s19, s49 // (op 13702) 



v_lshl_add_u32 v22, s1, 7, v34 // (op 13702) 


v_mul_lo_u32 v34, v35, s38 // (op 13702) 
v_add_u32 v34, v34, v22 // (op 13702) 


s_mul_i32 s1, s38, s43 // (op 13704) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13704) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13704) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13704) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13704) most significant: sum
s_add_u32 s1, s1, s19 // (op 13704) most significant: sum


s_mov_b32 s68, s40 // (op 13704) reg expression
s_mov_b32 s69, s41 // (op 13704) reg expression
s_mov_b32 s70, s0 // (op 13704) reg expression
s_mov_b32 s71, 131072 // (op 13704) call()

v_and_b32 v22, -4, v1 // (op 13830) 
v_sub_u32 v35, v1, v22 // (op 13830) 

v_lshlrev_b32 v22, 5, v35 // (op 13830) 

v_lshrrev_b32 v35, 2, v1 // (op 13830) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13830) 


v_lshlrev_b32 v22, 2, v36 // (op 13830) 

v_lshrrev_b32 v35, 3, v22 // (op 13830) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13830) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13417) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13417) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13417) 


v_mov_b64 v[38:39], v[28:29] // (op 13417) reg expression
s_mov_b32 m0, s0 // (op 13417) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 13417) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v36 // (op 13417) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 13417) most significant half

s_add_u32 m0, m0, 4096 // (op 13417) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 13417) Load value direct to lds




v_mov_b64 v[36:37], s[38:39] // (op 12967) convert

v_mov_b32 v22, v33 // (op 12967) reg expression
buffer_load_ubyte v38, v22, s[68:71], 0 offen offset: 0 // (op 12967) Load value
v_add_u32 v22, v22, v36 // (op 12967) 
buffer_load_ubyte v39, v22, s[68:71], 0 offen offset: 0 // (op 12967) Load value


v_and_b32 v22, -4, v1 // (op 13773) 
v_sub_u32 v35, v1, v22 // (op 13773) 

v_lshlrev_b32 v22, 5, v35 // (op 13773) 

v_lshrrev_b32 v35, 2, v1 // (op 13773) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13773) 


v_lshlrev_b32 v22, 2, v36 // (op 13773) 

v_lshrrev_b32 v35, 3, v22 // (op 13773) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13773) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13422) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13422) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13422) 


v_mov_b64 v[40:41], v[24:25] // (op 13422) 
s_add_u32 s1, 8192, s0 // (op 13422) 
s_mov_b32 m0, s1 // (op 13422) reg expression
buffer_load_dwordx4 v40, s[20:23], 0 offen offset: 0 lds // (op 13422) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v36 // (op 13422) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 13422) most significant half

s_add_u32 m0, m0, 4096 // (op 13422) 
buffer_load_dwordx4 v40, s[20:23], 0 offen offset: 0 lds // (op 13422) Load value direct to lds





v_mov_b64 v[36:37], s[34:35] // (op 12973) convert

v_mov_b32 v22, v23 // (op 12973) 
buffer_load_ubyte v40, v22, s[64:67], 0 offen offset: 0 // (op 12973) Load value
v_add_u32 v22, v22, v36 // (op 12973) 
buffer_load_ubyte v41, v22, s[64:67], 0 offen offset: 0 // (op 12973) Load value


v_and_b32 v22, -4, v1 // (op 14057) 
v_sub_u32 v35, v1, v22 // (op 14057) 

v_lshlrev_b32 v22, 5, v35 // (op 14057) 

v_lshrrev_b32 v35, 2, v1 // (op 14057) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14057) 


v_lshlrev_b32 v22, 2, v36 // (op 14057) 

v_lshrrev_b32 v35, 3, v22 // (op 14057) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14057) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13427) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13427) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13427) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13427) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13427) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13427) 

v_mov_b32 v22, s18 // (op 13427) 
v_mov_b32 v35, s19 // (op 13427) 
v_add_co_u32 v42, s[50:51], v28, v22 // (op 13427) least significant half
v_addc_co_u32 v43, s[50:51], v29, v35, s[50:51] // (op 13427) most significant half




s_add_u32 s1, 16384, s0 // (op 13427) 
s_mov_b32 m0, s1 // (op 13427) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 13427) Load value direct to lds
v_add_co_u32 v42, s[18:19], v42, v36 // (op 13427) least significant half
v_addc_co_u32 v43, s[18:19], 0, v43, s[18:19] // (op 13427) most significant half

s_add_u32 m0, m0, 4096 // (op 13427) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 13427) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 12979) convert

s_lshl_b32 s0, s38, 2 // (op 12979) 
v_add_u32 v22, v33, s0 // (op 12979) 

buffer_load_ubyte v42, v22, s[68:71], 0 offen offset: 0 // (op 12979) Load value
v_add_u32 v22, v22, v36 // (op 12979) 
buffer_load_ubyte v43, v22, s[68:71], 0 offen offset: 0 // (op 12979) Load value


v_and_b32 v22, -4, v1 // (op 14013) 
v_sub_u32 v35, v1, v22 // (op 14013) 

v_lshlrev_b32 v22, 5, v35 // (op 14013) 

v_lshrrev_b32 v35, 2, v1 // (op 14013) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14013) 


v_lshlrev_b32 v22, 2, v36 // (op 14013) 

v_lshrrev_b32 v35, 3, v22 // (op 14013) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14013) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13429) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13429) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13429) 


v_mov_b64 v[44:45], v[24:25] // (op 13429) 
s_add_u32 s1, 24576, s0 // (op 13429) 
s_mov_b32 m0, s1 // (op 13429) reg expression
buffer_load_dwordx4 v44, s[20:23], 64 offen offset: 0 lds // (op 13429) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v36 // (op 13429) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 13429) most significant half

s_add_u32 m0, m0, 4096 // (op 13429) 
buffer_load_dwordx4 v44, s[20:23], 64 offen offset: 0 lds // (op 13429) Load value direct to lds





v_mov_b64 v[36:37], s[34:35] // (op 12985) convert

v_mov_b32 v22, v23 // (op 12985) 
buffer_load_ubyte v44, v22, s[64:67], 0 offen offset:4 // (op 12985) Load value
v_add_u32 v22, v22, v36 // (op 12985) 
buffer_load_ubyte v45, v22, s[64:67], 0 offen offset:4 // (op 12985) Load value


v_and_b32 v22, -4, v1 // (op 14203) 
v_sub_u32 v35, v1, v22 // (op 14203) 

v_lshlrev_b32 v22, 5, v35 // (op 14203) 

v_lshrrev_b32 v35, 2, v1 // (op 14203) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14203) 


v_lshlrev_b32 v22, 2, v36 // (op 14203) 

v_lshrrev_b32 v35, 3, v22 // (op 14203) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14203) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13431) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13431) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13431) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13431) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13431) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13431) 

s_lshl_b64 s[50:51], s[18:19], 1 // (op 13431) 

v_mov_b32 v22, s50 // (op 13431) 
v_mov_b32 v35, s51 // (op 13431) 
v_add_co_u32 v46, s[18:19], v28, v22 // (op 13431) least significant half
v_addc_co_u32 v47, s[18:19], v29, v35, s[18:19] // (op 13431) most significant half




s_add_u32 s1, 32768, s0 // (op 13431) 
s_mov_b32 m0, s1 // (op 13431) reg expression
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13431) Load value direct to lds
v_add_co_u32 v46, s[18:19], v46, v36 // (op 13431) least significant half
v_addc_co_u32 v47, s[18:19], 0, v47, s[18:19] // (op 13431) most significant half

s_add_u32 m0, m0, 4096 // (op 13431) 
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13431) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 12991) convert

s_lshl_b32 s0, s38, 3 // (op 12991) 
v_add_u32 v22, v33, s0 // (op 12991) 

buffer_load_ubyte v46, v22, s[68:71], 0 offen offset: 0 // (op 12991) Load value
v_add_u32 v22, v22, v36 // (op 12991) 
buffer_load_ubyte v47, v22, s[68:71], 0 offen offset: 0 // (op 12991) Load value


v_and_b32 v22, -4, v1 // (op 14159) 
v_sub_u32 v35, v1, v22 // (op 14159) 

v_lshlrev_b32 v22, 5, v35 // (op 14159) 

v_lshrrev_b32 v35, 2, v1 // (op 14159) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14159) 


v_lshlrev_b32 v22, 2, v36 // (op 14159) 

v_lshrrev_b32 v35, 3, v22 // (op 14159) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14159) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13433) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13433) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13433) 


v_mov_b64 v[48:49], v[24:25] // (op 13433) 
s_add_u32 s1, 40960, s0 // (op 13433) 
s_mov_b32 m0, s1 // (op 13433) reg expression
s_mov_b32 s18, 128 // (op 13433) call()
buffer_load_dwordx4 v48, s[20:23], s18 offen offset: 0 lds // (op 13433) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v36 // (op 13433) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 13433) most significant half

s_add_u32 m0, m0, 4096 // (op 13433) 
buffer_load_dwordx4 v48, s[20:23], s18 offen offset: 0 lds // (op 13433) Load value direct to lds






v_mov_b64 v[36:37], s[34:35] // (op 12997) convert

v_mov_b32 v22, v23 // (op 12997) 
buffer_load_ubyte v48, v22, s[64:67], 0 offen offset:8 // (op 12997) Load value
v_add_u32 v22, v22, v36 // (op 12997) 
buffer_load_ubyte v49, v22, s[64:67], 0 offen offset:8 // (op 12997) Load value


v_and_b32 v22, -4, v1 // (op 14349) 
v_sub_u32 v35, v1, v22 // (op 14349) 

v_lshlrev_b32 v22, 5, v35 // (op 14349) 

v_lshrrev_b32 v35, 2, v1 // (op 14349) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14349) 


v_lshlrev_b32 v22, 2, v36 // (op 14349) 

v_lshrrev_b32 v35, 3, v22 // (op 14349) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14349) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13435) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13435) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13435) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13435) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13435) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13435) 

s_mul_i32 s51, 3, s19 // (op 13435) most significant: low of low * high
s_mul_i32 s50, 3, s18 // (op 13435) least significant
s_mul_hi_u32 s1, 3, s18 // (op 13435) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 13435) most significant: sum


v_mov_b32 v22, s50 // (op 13435) 
v_mov_b32 v35, s51 // (op 13435) 
v_add_co_u32 v50, s[18:19], v28, v22 // (op 13435) least significant half
v_addc_co_u32 v51, s[18:19], v29, v35, s[18:19] // (op 13435) most significant half




s_add_u32 s1, 49152, s0 // (op 13435) 
s_mov_b32 m0, s1 // (op 13435) reg expression
buffer_load_dwordx4 v50, s[52:55], 0 offen offset: 0 lds // (op 13435) Load value direct to lds
v_add_co_u32 v50, s[18:19], v50, v36 // (op 13435) least significant half
v_addc_co_u32 v51, s[18:19], 0, v51, s[18:19] // (op 13435) most significant half

s_add_u32 m0, m0, 4096 // (op 13435) 
buffer_load_dwordx4 v50, s[52:55], 0 offen offset: 0 lds // (op 13435) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 13003) convert

s_lshl_b32 s0, s38, 2 // (op 13003) 
s_mul_i32 s1, 3, s0 // (op 13003) 

v_add_u32 v22, v33, s1 // (op 13003) 

buffer_load_ubyte v36, v22, s[68:71], 0 offen offset: 0 // (op 13003) Load value
v_add_u32 v22, v22, v28 // (op 13003) 
buffer_load_ubyte v37, v22, s[68:71], 0 offen offset: 0 // (op 13003) Load value



v_and_b32 v22, -4, v1 // (op 14305) 
v_sub_u32 v33, v1, v22 // (op 14305) 

v_lshlrev_b32 v22, 5, v33 // (op 14305) 

v_lshrrev_b32 v33, 2, v1 // (op 14305) 
v_lshl_add_u32 v35, v33, 7, v22 // (op 14305) 


v_lshlrev_b32 v22, 2, v35 // (op 14305) 

v_lshrrev_b32 v33, 3, v22 // (op 14305) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 14305) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13437) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13437) 

v_lshrrev_b64 v[28:29], 3, s[50:51] // (op 13437) 


v_mov_b64 v[50:51], v[24:25] // (op 13437) 
s_add_u32 s1, 57344, s0 // (op 13437) 
s_mov_b32 m0, s1 // (op 13437) reg expression
s_mov_b32 s18, 192 // (op 13437) call()
buffer_load_dwordx4 v50, s[20:23], s18 offen offset: 0 lds // (op 13437) Load value direct to lds
v_add_co_u32 v50, s[50:51], v50, v28 // (op 13437) least significant half
v_addc_co_u32 v51, s[50:51], 0, v51, s[50:51] // (op 13437) most significant half

s_add_u32 m0, m0, 4096 // (op 13437) 
buffer_load_dwordx4 v50, s[20:23], s18 offen offset: 0 lds // (op 13437) Load value direct to lds







v_mov_b64 v[24:25], s[34:35] // (op 13009) convert

v_mov_b32 v22, v23 // (op 13009) 
buffer_load_ubyte v28, v22, s[64:67], 0 offen offset:12 // (op 13009) Load value
v_add_u32 v22, v22, v24 // (op 13009) 
buffer_load_ubyte v29, v22, s[64:67], 0 offen offset:12 // (op 13009) Load value



v_and_b32 v33, -128, v1 // (op 13946) 
v_sub_u32 v35, v1, v33 // (op 13946) 

v_lshrrev_b32 v33, 7, v1 // (op 13946) 
v_lshl_add_u32 v22, v33, 8, v35 // (op 13946) 





v_add_u32 v33, 65536, v22 // (op 13012) 
s_waitcnt vmcnt(29)
ds_write_b8 v33, v38 // (op 13012) Store local data 

v_add_u32 v33, 65664, v22 // (op 13012) 
s_waitcnt vmcnt(28)
ds_write_b8 v33, v39 // (op 13012) Store local data 


v_and_b32 v33, -4, v1 // (op 13888) 
v_sub_u32 v35, v1, v33 // (op 13888) 

v_lshrrev_b32 v33, 2, v1 // (op 13888) 
v_lshl_add_u32 v22, v33, 3, v35 // (op 13888) 





v_add_u32 v33, 66048, v22 // (op 13014) 
s_waitcnt vmcnt(25)
ds_write_b8 v33, v40 // (op 13014) Store local data 

v_add_u32 v33, 66052, v22 // (op 13014) 
s_waitcnt vmcnt(24)
ds_write_b8 v33, v41 // (op 13014) Store local data 


s_barrier  // (op 14645) 



ds_read_b64_tr_b4 v[22:23], v4 // (op 13057) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v4 offset:1024 // (op 13057) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14680) 


v_add_u32 v33, 65536, v5 // (op 13097) 
ds_read_u8 v35, v33 // (op 13097) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:16 // (op 13062) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:1040 // (op 13062) Transpose load local data 


v_add_u32 v33, 65568, v5 // (op 13102) 
ds_read_u8 v54, v33 // (op 13102) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:32 // (op 13067) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v4 offset:1056 // (op 13067) Transpose load local data 


v_add_u32 v33, 65600, v5 // (op 13107) 
ds_read_u8 v55, v33 // (op 13107) Load local data 




ds_read_b64_tr_b4 v[60:61], v4 offset:48 // (op 13072) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v4 offset:1072 // (op 13072) Transpose load local data 


v_add_u32 v33, 65632, v5 // (op 13112) 
ds_read_u8 v64, v33 // (op 13112) Load local data 

s_barrier  // (op 14629) 



ds_read_b128 v[66:69], v3 offset:8192 // (op 13037) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14662) 



v_add_u32 v33, 66048, v0 // (op 13077) 
ds_read_u8 v65, v33 // (op 13077) Load local data 




ds_read_b128 v[70:73], v3 offset:10240 // (op 13042) Load local data 



v_add_u32 v33, 66176, v0 // (op 13082) 
ds_read_u8 v74, v33 // (op 13082) Load local data 




ds_read_b128 v[76:79], v3 offset:12288 // (op 13047) Load local data 



v_add_u32 v33, 66304, v0 // (op 13087) 
ds_read_u8 v75, v33 // (op 13087) Load local data 




ds_read_b128 v[80:83], v3 offset:14336 // (op 13052) Load local data 



v_add_u32 v33, 66432, v0 // (op 13092) 
ds_read_u8 v84, v33 // (op 13092) Load local data 

v_and_b32 v33, -4, v1 // (op 13745) 
v_sub_u32 v85, v1, v33 // (op 13745) 

v_lshlrev_b32 v33, 5, v85 // (op 13745) 

v_lshrrev_b32 v85, 2, v1 // (op 13745) 
v_lshl_add_u32 v86, v85, 7, v33 // (op 13745) 


v_lshlrev_b32 v33, 2, v86 // (op 13745) 

v_lshrrev_b32 v85, 3, v33 // (op 13745) 

s_nop 0
v_readfirstlane_b32 s0, v85 // (op 13745) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 13802) 
v_sub_u32 v85, v1, v33 // (op 13802) 

v_lshlrev_b32 v33, 5, v85 // (op 13802) 

v_lshrrev_b32 v85, 2, v1 // (op 13802) 
v_lshl_add_u32 v86, v85, 7, v33 // (op 13802) 


v_lshlrev_b32 v33, 2, v86 // (op 13802) 

v_lshrrev_b32 v85, 3, v33 // (op 13802) 

s_nop 0
v_readfirstlane_b32 s1, v85 // (op 13802) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 13859) 
v_sub_u32 v85, v1, v33 // (op 13859) 

v_lshrrev_b32 v33, 2, v1 // (op 13859) 
v_lshl_add_u32 v86, v33, 3, v85 // (op 13859) 


v_and_b32 v33, -128, v1 // (op 13917) 
v_sub_u32 v85, v1, v33 // (op 13917) 

v_lshrrev_b32 v33, 7, v1 // (op 13917) 
v_lshl_add_u32 v87, v33, 8, v85 // (op 13917) 


v_and_b32 v33, -4, v1 // (op 13998) 
v_sub_u32 v85, v1, v33 // (op 13998) 

v_lshlrev_b32 v33, 5, v85 // (op 13998) 

v_lshrrev_b32 v85, 2, v1 // (op 13998) 
v_lshl_add_u32 v88, v85, 7, v33 // (op 13998) 


v_lshlrev_b32 v33, 2, v88 // (op 13998) 

v_lshrrev_b32 v85, 3, v33 // (op 13998) 

s_nop 0
v_readfirstlane_b32 s18, v85 // (op 13998) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14042) 
v_sub_u32 v85, v1, v33 // (op 14042) 

v_lshlrev_b32 v33, 5, v85 // (op 14042) 

v_lshrrev_b32 v85, 2, v1 // (op 14042) 
v_lshl_add_u32 v88, v85, 7, v33 // (op 14042) 


v_lshlrev_b32 v33, 2, v88 // (op 14042) 

v_lshrrev_b32 v85, 3, v33 // (op 14042) 

s_nop 0
v_readfirstlane_b32 s19, v85 // (op 14042) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14086) 
v_sub_u32 v85, v1, v33 // (op 14086) 

v_lshrrev_b32 v33, 2, v1 // (op 14086) 
v_lshl_add_u32 v88, v33, 3, v85 // (op 14086) 


v_and_b32 v33, -128, v1 // (op 14115) 
v_sub_u32 v85, v1, v33 // (op 14115) 

v_lshrrev_b32 v33, 7, v1 // (op 14115) 
v_lshl_add_u32 v89, v33, 8, v85 // (op 14115) 


v_and_b32 v33, -4, v1 // (op 14144) 
v_sub_u32 v85, v1, v33 // (op 14144) 

v_lshlrev_b32 v33, 5, v85 // (op 14144) 

v_lshrrev_b32 v85, 2, v1 // (op 14144) 
v_lshl_add_u32 v90, v85, 7, v33 // (op 14144) 


v_lshlrev_b32 v33, 2, v90 // (op 14144) 

v_lshrrev_b32 v85, 3, v33 // (op 14144) 

s_nop 0
v_readfirstlane_b32 s49, v85 // (op 14144) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14188) 
v_sub_u32 v85, v1, v33 // (op 14188) 

v_lshlrev_b32 v33, 5, v85 // (op 14188) 

v_lshrrev_b32 v85, 2, v1 // (op 14188) 
v_lshl_add_u32 v90, v85, 7, v33 // (op 14188) 


v_lshlrev_b32 v33, 2, v90 // (op 14188) 

v_lshrrev_b32 v85, 3, v33 // (op 14188) 

s_nop 0
v_readfirstlane_b32 s50, v85 // (op 14188) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14232) 
v_sub_u32 v85, v1, v33 // (op 14232) 

v_lshrrev_b32 v33, 2, v1 // (op 14232) 
v_lshl_add_u32 v90, v33, 3, v85 // (op 14232) 


v_and_b32 v33, -128, v1 // (op 14261) 
v_sub_u32 v85, v1, v33 // (op 14261) 

v_lshrrev_b32 v33, 7, v1 // (op 14261) 
v_lshl_add_u32 v91, v33, 8, v85 // (op 14261) 


v_and_b32 v33, -4, v1 // (op 14290) 
v_sub_u32 v85, v1, v33 // (op 14290) 

v_lshlrev_b32 v33, 5, v85 // (op 14290) 

v_lshrrev_b32 v85, 2, v1 // (op 14290) 
v_lshl_add_u32 v92, v85, 7, v33 // (op 14290) 


v_lshlrev_b32 v33, 2, v92 // (op 14290) 

v_lshrrev_b32 v85, 3, v33 // (op 14290) 

s_nop 0
v_readfirstlane_b32 s51, v85 // (op 14290) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14334) 
v_sub_u32 v85, v1, v33 // (op 14334) 

v_lshlrev_b32 v33, 5, v85 // (op 14334) 

v_lshrrev_b32 v85, 2, v1 // (op 14334) 
v_lshl_add_u32 v92, v85, 7, v33 // (op 14334) 


v_lshlrev_b32 v33, 2, v92 // (op 14334) 

v_lshrrev_b32 v85, 3, v33 // (op 14334) 

s_nop 0
v_readfirstlane_b32 s72, v85 // (op 14334) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14378) 
v_sub_u32 v85, v1, v33 // (op 14378) 

v_lshrrev_b32 v33, 2, v1 // (op 14378) 
v_lshl_add_u32 v92, v33, 3, v85 // (op 14378) 


v_and_b32 v33, -128, v1 // (op 14407) 
v_sub_u32 v85, v1, v33 // (op 14407) 

v_lshrrev_b32 v33, 7, v1 // (op 14407) 
v_lshl_add_u32 v93, v33, 8, v85 // (op 14407) 


s_mov_b32 s73, 0 // (op 2901) call()
s_ashr_i64 s[74:75], s[32:33], 7 // (op 2900) 
s_mov_b32 s76, s74 // (op 2900) convert

s_and_b32 s74, s76, 4294967292 // (op 2900) 

s_cmp_lt_u32 s73, s74 // (op 2900) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2900 // (op 2900) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2900 if false)
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2900:
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[22:25], a[0:3], v65, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
s_barrier  // (op 14648) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[50:53], a[4:7], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 
s_add_u32 s73, s73, 4 // (op 2902) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[56:59], a[8:11], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 824) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[60:63], a[12:15], v65, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1122) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13369) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13369) 


v_mov_b64 v[96:97], v[30:31] // (op 13369) 
s_lshl_b64 s[74:75], s[28:29], 7 // (op 13369) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[22:25], a[16:19], v74, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1925) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[50:53], a[20:23], v74, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1833) 
s_lshr_b64 s[74:75], s[76:77], 3 // (op 13369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[56:59], a[24:27], v74, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1741) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[60:63], a[28:31], v74, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1649) 
v_mov_b32 v33, s76 // (op 13369) 
v_mov_b32 v85, s77 // (op 13369) 
v_add_co_u32 v98, s[74:75], v96, v33 // (op 13369) least significant half
v_addc_co_u32 v99, s[74:75], v97, v85, s[74:75] // (op 13369) most significant half




s_mov_b32 m0, s1 // (op 13369) reg expression
buffer_load_dwordx4 v98, s[52:55], 0 offen offset: 0 lds // (op 13369) Load value direct to lds
v_add_co_u32 v98, s[74:75], v98, v94 // (op 13369) least significant half
v_addc_co_u32 v99, s[74:75], 0, v99, s[74:75] // (op 13369) most significant half

s_add_u32 m0, m0, 4096 // (op 13369) 
buffer_load_dwordx4 v98, s[52:55], 0 offen offset: 0 lds // (op 13369) Load value direct to lds


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[22:25], a[32:35], v75, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3215) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[50:53], a[36:39], v75, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3123) 
v_mov_b64 v[94:95], s[38:39] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[56:59], a[40:43], v75, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3031) 
v_mov_b32 v33, v34 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[60:63], a[44:47], v75, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2939) 
s_lshl_b32 s74, s38, 4 // (op 25) 
v_add_u32 v85, v33, s74 // (op 25) 

buffer_load_ubyte v38, v85, s[68:71], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v85, v85, v94 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[22:25], a[48:51], v84, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4505) 
buffer_load_ubyte v39, v85, s[68:71], 0 offen offset: 0 // (op 25) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[50:53], a[52:55], v84, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4413) 
s_barrier  // (op 14632) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[56:59], a[56:59], v84, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4321) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 13364) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[60:63], a[60:63], v84, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4229) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13364) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13364) 


v_mov_b64 v[98:99], v[26:27] // (op 13364) 
v_mov_b64 v[100:101], v[98:99] // (op 13364) 
s_add_u32 s74, 8192, s0 // (op 13364) 
s_mov_b32 m0, s74 // (op 13364) reg expression
s_mov_b32 s75, 256 // (op 13364) call()
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13364) Load value direct to lds
v_add_co_u32 v100, s[76:77], v100, v94 // (op 13364) least significant half
v_addc_co_u32 v101, s[76:77], 0, v101, s[76:77] // (op 13364) most significant half

s_add_u32 m0, m0, 4096 // (op 13364) 
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13364) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 17) convert

v_mov_b32 v85, v32 // (op 17) 
v_mov_b32 v100, v85 // (op 17) 
buffer_load_ubyte v40, v100, s[64:67], 0 offen offset:16 // (op 17) Load value
v_add_u32 v100, v100, v94 // (op 17) 
buffer_load_ubyte v41, v100, s[64:67], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 14720) 



v_add_u32 v94, 66560, v89 // (op 259) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v42 // (op 259) Store local data 

v_add_u32 v94, 66688, v89 // (op 259) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v43 // (op 259) Store local data 

s_barrier  // (op 14714) 



v_add_u32 v94, 67072, v88 // (op 258) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v44 // (op 258) Store local data 

v_add_u32 v94, 67076, v88 // (op 258) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v45 // (op 258) Store local data 

s_barrier  // (op 14703) 



ds_read_b64_tr_b4 v[100:101], v8 offset:16384 // (op 251) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v8 offset:17408 // (op 251) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14717) 


v_add_u32 v94, 66560, v9 // (op 255) 
ds_read_u8 v95, v94 // (op 255) Load local data 




ds_read_b64_tr_b4 v[104:105], v8 offset:16400 // (op 533) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v8 offset:17424 // (op 533) Transpose load local data 


v_add_u32 v94, 66592, v9 // (op 537) 
ds_read_u8 v74, v94 // (op 537) Load local data 




ds_read_b64_tr_b4 v[70:71], v8 offset:16416 // (op 831) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v8 offset:17440 // (op 831) Transpose load local data 


v_add_u32 v94, 66624, v9 // (op 835) 
ds_read_u8 v75, v94 // (op 835) Load local data 




ds_read_b64_tr_b4 v[76:77], v8 offset:16432 // (op 1129) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v8 offset:17456 // (op 1129) Transpose load local data 


v_add_u32 v94, 66656, v9 // (op 1133) 
ds_read_u8 v108, v94 // (op 1133) Load local data 

s_barrier  // (op 14697) 
s_barrier  // (op 14829) 



ds_read_b128 v[66:69], v7 offset:24576 // (op 249) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14710) 



v_add_u32 v94, 67072, v6 // (op 253) 
ds_read_u8 v65, v94 // (op 253) Load local data 




ds_read_b128 v[110:113], v7 offset:26624 // (op 1654) Load local data 



v_add_u32 v94, 67200, v6 // (op 1657) 
ds_read_u8 v54, v94 // (op 1657) Load local data 




ds_read_b128 v[114:117], v7 offset:28672 // (op 2944) Load local data 



v_add_u32 v94, 67328, v6 // (op 2947) 
ds_read_u8 v55, v94 // (op 2947) Load local data 




ds_read_b128 v[118:121], v7 offset:30720 // (op 4234) Load local data 



v_add_u32 v94, 67456, v6 // (op 4237) 
ds_read_u8 v64, v94 // (op 4237) Load local data 

s_barrier  // (op 14887) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[100:103], a[0:3], v65, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 
s_barrier  // (op 14707) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[104:107], a[4:7], v65, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[70:73], a[8:11], v65, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 825) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13392) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[76:79], a[12:15], v65, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1123) 
v_lshrrev_b64 v[122:123], 3, s[76:77] // (op 13392) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13392) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13392) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13392) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:113], v[100:103], a[16:19], v54, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1926) 
s_mul_i32 s77, 5, s75 // (op 13392) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[110:113], v[104:107], a[20:23], v54, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1834) 
s_mul_i32 s76, 5, s74 // (op 13392) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:113], v[70:73], a[24:27], v54, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1742) 
s_mul_hi_u32 s78, 5, s74 // (op 13392) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[110:113], v[76:79], a[28:31], v54, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1650) 
s_add_u32 s77, s77, s78 // (op 13392) most significant: sum



v_mov_b32 v94, s76 // (op 13392) 
v_mov_b32 v109, s77 // (op 13392) 
v_add_co_u32 v110, s[74:75], v96, v94 // (op 13392) least significant half
v_addc_co_u32 v111, s[74:75], v97, v109, s[74:75] // (op 13392) most significant half




s_add_u32 s74, 16384, s19 // (op 13392) 
s_nop 0
s_mov_b32 m0, s74 // (op 13392) reg expression
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 13392) Load value direct to lds
v_add_co_u32 v110, s[76:77], v110, v122 // (op 13392) least significant half
v_addc_co_u32 v111, s[76:77], 0, v111, s[76:77] // (op 13392) most significant half

s_add_u32 m0, m0, 4096 // (op 13392) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 13392) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[114:117], v[100:103], a[32:35], v55, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3216) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[114:117], v[104:107], a[36:39], v55, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3124) 
v_mov_b64 v[112:113], s[38:39] // (op 262) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[114:117], v[70:73], a[40:43], v55, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3032) 
s_lshl_b32 s74, s38, 2 // (op 262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:117], v[76:79], a[44:47], v55, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2940) 
s_mul_i32 s75, 5, s74 // (op 262) 


v_add_u32 v94, v33, s75 // (op 262) 

buffer_load_ubyte v42, v94, s[68:71], 0 offen offset: 0 // (op 262) Load value
v_add_u32 v94, v94, v112 // (op 262) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[118:121], v[100:103], a[48:51], v64, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4506) 
buffer_load_ubyte v43, v94, s[68:71], 0 offen offset: 0 // (op 262) Load value




s_barrier  // (op 14700) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:121], v[104:107], a[52:55], v64, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4414) 


s_lshl_b64 s[74:75], s[24:25], 6 // (op 13387) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[118:121], v[70:73], a[56:59], v64, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4322) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13387) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13387) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[118:121], v[76:79], a[60:63], v64, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4230) 
v_mov_b64 v[100:101], v[98:99] // (op 13387) 


s_add_u32 s74, 24576, s18 // (op 13387) 
s_mov_b32 m0, s74 // (op 13387) reg expression
s_mov_b32 s75, 320 // (op 13387) call()
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13387) Load value direct to lds
v_add_co_u32 v100, s[76:77], v100, v94 // (op 13387) least significant half
v_addc_co_u32 v101, s[76:77], 0, v101, s[76:77] // (op 13387) most significant half

s_add_u32 m0, m0, 4096 // (op 13387) 
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13387) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 261) convert

v_mov_b32 v100, v85 // (op 261) 
buffer_load_ubyte v44, v100, s[64:67], 0 offen offset:20 // (op 261) Load value
v_add_u32 v100, v100, v94 // (op 261) 
buffer_load_ubyte v45, v100, s[64:67], 0 offen offset:20 // (op 261) Load value


s_barrier  // (op 14746) 



v_add_u32 v94, 66560, v91 // (op 299) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v46 // (op 299) Store local data 

v_add_u32 v94, 66688, v91 // (op 299) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v47 // (op 299) Store local data 

s_barrier  // (op 14740) 



v_add_u32 v94, 67072, v90 // (op 298) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v48 // (op 298) Store local data 

v_add_u32 v94, 67076, v90 // (op 298) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v49 // (op 298) Store local data 

s_barrier  // (op 14729) 



ds_read_b64_tr_b4 v[22:23], v12 offset:32768 // (op 291) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v12 offset:33792 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14743) 


v_add_u32 v94, 66560, v13 // (op 295) 
ds_read_u8 v35, v94 // (op 295) Load local data 




ds_read_b64_tr_b4 v[50:51], v12 offset:32784 // (op 541) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v12 offset:33808 // (op 541) Transpose load local data 


v_add_u32 v94, 66592, v13 // (op 545) 
ds_read_u8 v54, v94 // (op 545) Load local data 




ds_read_b64_tr_b4 v[56:57], v12 offset:32800 // (op 839) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v12 offset:33824 // (op 839) Transpose load local data 


v_add_u32 v94, 66624, v13 // (op 843) 
ds_read_u8 v55, v94 // (op 843) Load local data 




ds_read_b64_tr_b4 v[60:61], v12 offset:32816 // (op 1137) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v12 offset:33840 // (op 1137) Transpose load local data 


v_add_u32 v94, 66656, v13 // (op 1141) 
ds_read_u8 v84, v94 // (op 1141) Load local data 

s_barrier  // (op 14723) 
s_barrier  // (op 14835) 



ds_read_b128 v[66:69], v11 offset:40960 // (op 289) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14736) 



v_add_u32 v94, 67072, v10 // (op 293) 
ds_read_u8 v65, v94 // (op 293) Load local data 




ds_read_b128 v[80:83], v11 offset:43008 // (op 1660) Load local data 



v_add_u32 v94, 67200, v10 // (op 1663) 
ds_read_u8 v74, v94 // (op 1663) Load local data 




ds_read_b128 v[100:103], v11 offset:45056 // (op 2950) Load local data 



v_add_u32 v94, 67328, v10 // (op 2953) 
ds_read_u8 v75, v94 // (op 2953) Load local data 




ds_read_b128 v[104:107], v11 offset:47104 // (op 4240) Load local data 



v_add_u32 v94, 67456, v10 // (op 4243) 
ds_read_u8 v95, v94 // (op 4243) Load local data 

s_barrier  // (op 14893) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[22:25], a[0:3], v65, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 
s_barrier  // (op 14733) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[50:53], a[4:7], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13402) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[56:59], a[8:11], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 826) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13402) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[60:63], a[12:15], v65, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1124) 
v_lshrrev_b64 v[108:109], 3, s[76:77] // (op 13402) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13402) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13402) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13402) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[22:25], a[16:19], v74, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1927) 
s_mul_i32 s77, 6, s75 // (op 13402) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[50:53], a[20:23], v74, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1835) 
s_mul_i32 s76, 6, s74 // (op 13402) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[56:59], a[24:27], v74, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1743) 
s_mul_hi_u32 s78, 6, s74 // (op 13402) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[60:63], a[28:31], v74, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1651) 
s_add_u32 s77, s77, s78 // (op 13402) most significant: sum


v_mov_b32 v94, s76 // (op 13402) 
v_mov_b32 v110, s77 // (op 13402) 
v_add_co_u32 v112, s[74:75], v96, v94 // (op 13402) least significant half
v_addc_co_u32 v113, s[74:75], v97, v110, s[74:75] // (op 13402) most significant half




s_add_u32 s74, 32768, s50 // (op 13402) 
s_nop 0
s_mov_b32 m0, s74 // (op 13402) reg expression
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 13402) Load value direct to lds
v_add_co_u32 v112, s[76:77], v112, v108 // (op 13402) least significant half
v_addc_co_u32 v113, s[76:77], 0, v113, s[76:77] // (op 13402) most significant half

s_add_u32 m0, m0, 4096 // (op 13402) 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 13402) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:103], v[22:25], a[32:35], v75, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3217) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:103], v[50:53], a[36:39], v75, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3125) 
v_mov_b64 v[108:109], s[38:39] // (op 302) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[56:59], a[40:43], v75, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3033) 
s_lshl_b32 s74, s38, 2 // (op 302) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[60:63], a[44:47], v75, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2941) 
s_mul_i32 s75, 6, s74 // (op 302) 


v_add_u32 v94, v33, s75 // (op 302) 

buffer_load_ubyte v46, v94, s[68:71], 0 offen offset: 0 // (op 302) Load value
v_add_u32 v94, v94, v108 // (op 302) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[22:25], a[48:51], v95, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4507) 
buffer_load_ubyte v47, v94, s[68:71], 0 offen offset: 0 // (op 302) Load value


s_barrier  // (op 14726) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[50:53], a[52:55], v95, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4415) 

s_lshl_b64 s[74:75], s[24:25], 6 // (op 13397) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:107], v[56:59], a[56:59], v95, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4323) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13397) 

v_lshrrev_b64 v[100:101], 3, s[76:77] // (op 13397) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:107], v[60:63], a[60:63], v95, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4231) 
v_mov_b64 v[102:103], v[98:99] // (op 13397) 


s_add_u32 s74, 40960, s49 // (op 13397) 
s_mov_b32 m0, s74 // (op 13397) reg expression
s_mov_b32 s75, 384 // (op 13397) call()
buffer_load_dwordx4 v102, s[20:23], s75 offen offset: 0 lds // (op 13397) Load value direct to lds
v_add_co_u32 v102, s[76:77], v102, v100 // (op 13397) least significant half
v_addc_co_u32 v103, s[76:77], 0, v103, s[76:77] // (op 13397) most significant half

s_add_u32 m0, m0, 4096 // (op 13397) 
buffer_load_dwordx4 v102, s[20:23], s75 offen offset: 0 lds // (op 13397) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 301) convert

v_mov_b32 v100, v85 // (op 301) 
buffer_load_ubyte v48, v100, s[64:67], 0 offen offset:24 // (op 301) Load value
v_add_u32 v100, v100, v94 // (op 301) 
buffer_load_ubyte v49, v100, s[64:67], 0 offen offset:24 // (op 301) Load value


s_barrier  // (op 14772) 



v_add_u32 v94, 66560, v93 // (op 339) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v36 // (op 339) Store local data 

v_add_u32 v94, 66688, v93 // (op 339) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v37 // (op 339) Store local data 

s_barrier  // (op 14766) 



v_add_u32 v94, 67072, v92 // (op 338) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v28 // (op 338) Store local data 

v_add_u32 v94, 67076, v92 // (op 338) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v29 // (op 338) Store local data 

s_barrier  // (op 14755) 



ds_read_b64_tr_b4 v[70:71], v16 offset:49152 // (op 331) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v16 offset:50176 // (op 331) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14769) 


v_add_u32 v94, 66560, v17 // (op 335) 
ds_read_u8 v64, v94 // (op 335) Load local data 




ds_read_b64_tr_b4 v[76:77], v16 offset:49168 // (op 549) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v16 offset:50192 // (op 549) Transpose load local data 


v_add_u32 v94, 66592, v17 // (op 553) 
ds_read_u8 v74, v94 // (op 553) Load local data 




ds_read_b64_tr_b4 v[80:81], v16 offset:49184 // (op 847) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v16 offset:50208 // (op 847) Transpose load local data 


v_add_u32 v94, 66624, v17 // (op 851) 
ds_read_u8 v75, v94 // (op 851) Load local data 




ds_read_b64_tr_b4 v[100:101], v16 offset:49200 // (op 1145) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v16 offset:50224 // (op 1145) Transpose load local data 


v_add_u32 v94, 66656, v17 // (op 1149) 
ds_read_u8 v95, v94 // (op 1149) Load local data 

s_barrier  // (op 14749) 
s_barrier  // (op 14841) 



ds_read_b128 v[66:69], v15 offset:57344 // (op 329) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14762) 



v_add_u32 v94, 67072, v14 // (op 333) 
ds_read_u8 v65, v94 // (op 333) Load local data 




ds_read_b128 v[104:107], v15 offset:59392 // (op 1666) Load local data 



v_add_u32 v94, 67200, v14 // (op 1669) 
ds_read_u8 v54, v94 // (op 1669) Load local data 




ds_read_b128 v[108:111], v15 offset:61440 // (op 2956) Load local data 



v_add_u32 v94, 67328, v14 // (op 2959) 
ds_read_u8 v55, v94 // (op 2959) Load local data 




ds_read_b128 v[112:115], v15 offset:63488 // (op 4246) Load local data 



v_add_u32 v94, 67456, v14 // (op 4249) 
ds_read_u8 v84, v94 // (op 4249) Load local data 

s_barrier  // (op 14899) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:73], a[0:3], v65, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
s_barrier  // (op 14759) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[76:79], a[4:7], v65, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13412) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[80:83], a[8:11], v65, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 827) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13412) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[100:103], a[12:15], v65, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1125) 
v_lshrrev_b64 v[116:117], 3, s[76:77] // (op 13412) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13412) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13412) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13412) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[70:73], a[16:19], v54, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1928) 
s_mul_i32 s77, 7, s75 // (op 13412) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[76:79], a[20:23], v54, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1836) 
s_mul_i32 s76, 7, s74 // (op 13412) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[80:83], a[24:27], v54, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1744) 
s_mul_hi_u32 s78, 7, s74 // (op 13412) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:107], v[100:103], a[28:31], v54, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1652) 
s_add_u32 s77, s77, s78 // (op 13412) most significant: sum



v_mov_b32 v94, s76 // (op 13412) 
v_mov_b32 v104, s77 // (op 13412) 
v_add_co_u32 v106, s[74:75], v96, v94 // (op 13412) least significant half
v_addc_co_u32 v107, s[74:75], v97, v104, s[74:75] // (op 13412) most significant half




s_add_u32 s74, 49152, s72 // (op 13412) 
s_nop 0
s_mov_b32 m0, s74 // (op 13412) reg expression
buffer_load_dwordx4 v106, s[52:55], 0 offen offset: 0 lds // (op 13412) Load value direct to lds
v_add_co_u32 v106, s[76:77], v106, v116 // (op 13412) least significant half
v_addc_co_u32 v107, s[76:77], 0, v107, s[76:77] // (op 13412) most significant half

s_add_u32 m0, m0, 4096 // (op 13412) 
buffer_load_dwordx4 v106, s[52:55], 0 offen offset: 0 lds // (op 13412) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[70:73], a[32:35], v55, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3218) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[76:79], a[36:39], v55, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3126) 
v_mov_b64 v[96:97], s[38:39] // (op 342) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[80:83], a[40:43], v55, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3034) 
s_lshl_b32 s74, s38, 2 // (op 342) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[100:103], a[44:47], v55, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2942) 
s_mul_i32 s75, 7, s74 // (op 342) 


v_add_u32 v94, v33, s75 // (op 342) 

buffer_load_ubyte v36, v94, s[68:71], 0 offen offset: 0 // (op 342) Load value
v_add_u32 v94, v94, v96 // (op 342) 
s_lshl_b64 s[74:75], s[28:29], 7 // (op 13566) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13566) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13566) 

s_lshl_b64 s[76:77], s[74:75], 2 // (op 13566) 

v_mov_b32 v104, s76 // (op 13566) 
v_mov_b32 v105, s77 // (op 13566) 
v_add_co_u32 v30, s[74:75], v30, v104 // (op 13566) least significant half
v_addc_co_u32 v31, s[74:75], v31, v105, s[74:75] // (op 13566) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[70:73], a[48:51], v84, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4508) 
buffer_load_ubyte v37, v94, s[68:71], 0 offen offset: 0 // (op 342) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[76:79], a[52:55], v84, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4416) 
s_barrier  // (op 14752) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[80:83], a[56:59], v84, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4324) 
s_lshl_b32 s74, s38, 4 // (op 13694) 
v_add_u32 v34, v34, s74 // (op 13694) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[100:103], a[60:63], v84, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4232) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 13407) 



s_lshl_b64 s[76:77], s[74:75], 2 // (op 13407) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13407) 


v_mov_b64 v[96:97], v[98:99] // (op 13407) 
s_add_u32 s74, 57344, s51 // (op 13407) 
s_mov_b32 m0, s74 // (op 13407) reg expression
s_mov_b32 s75, 448 // (op 13407) call()
buffer_load_dwordx4 v96, s[20:23], s75 offen offset: 0 lds // (op 13407) Load value direct to lds
v_add_co_u32 v96, s[76:77], v96, v94 // (op 13407) least significant half
v_addc_co_u32 v97, s[76:77], 0, v97, s[76:77] // (op 13407) most significant half

s_add_u32 m0, m0, 4096 // (op 13407) 
buffer_load_dwordx4 v96, s[20:23], s75 offen offset: 0 lds // (op 13407) Load value direct to lds






v_add_co_u32 v26, vcc, 256, v26 // (op 13502) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 13502) most significant half
v_mov_b64 v[94:95], s[34:35] // (op 341) convert

v_mov_b32 v33, v85 // (op 341) 
buffer_load_ubyte v28, v33, s[64:67], 0 offen offset:28 // (op 341) Load value
v_add_u32 v33, v33, v94 // (op 341) 
buffer_load_ubyte v29, v33, s[64:67], 0 offen offset:28 // (op 341) Load value



s_barrier  // (op 14683) 
v_add_u32 v32, 16, v32 // (op 13630) 



v_add_u32 v33, 65536, v87 // (op 30) 
s_waitcnt vmcnt(29)
ds_write_b8 v33, v38 // (op 30) Store local data 

v_add_u32 v33, 65664, v87 // (op 30) 
s_waitcnt vmcnt(28)
ds_write_b8 v33, v39 // (op 30) Store local data 

s_barrier  // (op 14665) 



v_add_u32 v33, 66048, v86 // (op 21) 
s_waitcnt vmcnt(25)
ds_write_b8 v33, v40 // (op 21) Store local data 

v_add_u32 v33, 66052, v86 // (op 21) 
s_waitcnt vmcnt(24)
ds_write_b8 v33, v41 // (op 21) Store local data 

s_barrier  // (op 14654) 



ds_read_b64_tr_b4 v[22:23], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v4 offset:1024 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14690) 


v_add_u32 v33, 65536, v5 // (op 87) 
ds_read_u8 v35, v33 // (op 87) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:16 // (op 521) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:1040 // (op 521) Transpose load local data 


v_add_u32 v33, 65568, v5 // (op 525) 
ds_read_u8 v54, v33 // (op 525) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:32 // (op 819) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v4 offset:1056 // (op 819) Transpose load local data 


v_add_u32 v33, 65600, v5 // (op 823) 
ds_read_u8 v55, v33 // (op 823) Load local data 




ds_read_b64_tr_b4 v[60:61], v4 offset:48 // (op 1117) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v4 offset:1072 // (op 1117) Transpose load local data 


v_add_u32 v33, 65632, v5 // (op 1121) 
ds_read_u8 v64, v33 // (op 1121) Load local data 

s_barrier  // (op 14638) 



ds_read_b128 v[66:69], v3 offset:8192 // (op 72) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14672) 



v_add_u32 v33, 66048, v0 // (op 82) 
ds_read_u8 v65, v33 // (op 82) Load local data 




ds_read_b128 v[70:73], v3 offset:10240 // (op 1644) Load local data 



v_add_u32 v33, 66176, v0 // (op 1647) 
ds_read_u8 v74, v33 // (op 1647) Load local data 




ds_read_b128 v[76:79], v3 offset:12288 // (op 2934) Load local data 



v_add_u32 v33, 66304, v0 // (op 2937) 
ds_read_u8 v75, v33 // (op 2937) Load local data 




ds_read_b128 v[80:83], v3 offset:14336 // (op 4224) Load local data 



v_add_u32 v33, 66432, v0 // (op 4227) 
ds_read_u8 v84, v33 // (op 4227) Load local data 

s_ashr_i64 s[74:75], s[32:33], 7 // (op 2900) 
s_mov_b32 s76, s74 // (op 2900) convert

s_and_b32 s74, s76, 4294967292 // (op 2900) 

s_cmp_lt_u32 s73, s74 // (op 2900) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2900 // (op 2900) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2900 if true)
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2900:


























s_barrier  // (op 14853) 
s_mul_hi_i32 s0, s16, s11 // (op 13450) 
v_lshrrev_b32 v22, 2, v1 // (op 13450) 
s_add_i32 s1, s0, s16 // (op 13450) 

v_and_b32 v23, -4, v1 // (op 13450) 
s_ashr_i32 s0, s1, 31 // (op 13450) 
v_sub_u32 v24, v1, v23 // (op 13450) 

s_cmp_eq_i32 s11, 0 // (op 13450) 
s_mov_b32 s18, scc // (op 13450) 
v_lshlrev_b32 v23, 5, v24 // (op 13450) 

s_cmp_eq_i32 s18, 0 // (op 13450) 
s_cselect_b32 s19, 0, -1 // (op 13450) 

s_and_b32 s18, s12, 31 // (op 13450) 
s_lshl_b32 s49, 1, s18 // (op 13450) 
s_add_i32 s50, s49, s19 // (op 13450) 


s_and_b32 s19, s0, s50 // (op 13450) 


s_add_i32 s0, s1, s19 // (op 13450) 


s_ashr_i32 s1, s0, s18 // (op 13450) 


s_lshl_b32 s0, s12, 24 // (op 13450) 
s_ashr_i32 s18, s0, 31 // (op 13450) 

s_xor_b32 s0, s1, s18 // (op 13450) 

s_sub_i32 s1, s0, s18 // (op 13450) 


s_mul_hi_i32 s0, s1, s13 // (op 13450) 
s_add_i32 s18, s0, s1 // (op 13450) 

s_ashr_i32 s0, s18, 31 // (op 13450) 
s_cmp_eq_i32 s13, 0 // (op 13450) 
s_mov_b32 s19, scc // (op 13450) 
s_cmp_eq_i32 s19, 0 // (op 13450) 
s_cselect_b32 s49, 0, -1 // (op 13450) 

s_and_b32 s19, s14, 31 // (op 13450) 
s_lshl_b32 s50, 1, s19 // (op 13450) 
s_add_i32 s51, s50, s49 // (op 13450) 


s_and_b32 s49, s0, s51 // (op 13450) 


s_add_i32 s0, s18, s49 // (op 13450) 


s_ashr_i32 s18, s0, s19 // (op 13450) 


s_lshl_b32 s0, s14, 24 // (op 13450) 
s_ashr_i32 s19, s0, 31 // (op 13450) 

s_xor_b32 s0, s18, s19 // (op 13450) 

s_sub_i32 s18, s0, s19 // (op 13450) 


s_mul_i32 s0, s1, s44 // (op 13450) 

s_sub_u32 s1, s16, s0 // (op 13450) 

s_mul_i32 s0, s18, s44 // (op 13450) 
s_add_u32 s0, s0, s1 // (op 13450) 


s_mul_hi_i32 s1, s16, s10 // (op 13450) 
s_add_i32 s18, s1, s16 // (op 13450) 

s_ashr_i32 s1, s18, 31 // (op 13450) 
s_cmp_eq_i32 s10, 0 // (op 13450) 
s_mov_b32 s19, scc // (op 13450) 
s_cmp_eq_i32 s19, 0 // (op 13450) 
s_cselect_b32 s49, 0, -1 // (op 13450) 

s_and_b32 s19, s47, 31 // (op 13450) 
s_lshl_b32 s50, 1, s19 // (op 13450) 
s_add_i32 s51, s50, s49 // (op 13450) 


s_and_b32 s49, s1, s51 // (op 13450) 


s_add_i32 s1, s18, s49 // (op 13450) 


s_ashr_i32 s18, s1, s19 // (op 13450) 


s_lshl_b32 s1, s47, 24 // (op 13450) 
s_ashr_i32 s19, s1, 31 // (op 13450) 

s_xor_b32 s1, s18, s19 // (op 13450) 

s_sub_i32 s18, s1, s19 // (op 13450) 


s_add_u32 s50, s6, 128 // (op 13450) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13450) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13450) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13450) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13450) 

s_mul_i32 s1, s44, s50 // (op 13450) 

s_mul_i32 s19, s48, s1 // (op 13450) 

s_mul_i32 s1, s18, s19 // (op 13450) 

s_sub_u32 s19, s16, s1 // (op 13450) 

s_mov_b32 s1, s19 // (op 13450) 
s_mul_hi_i32 s49, s1, s15 // (op 13450) 
s_add_i32 s50, s49, s1 // (op 13450) 


s_ashr_i32 s1, s50, 31 // (op 13450) 
s_cmp_eq_i32 s15, 0 // (op 13450) 
s_mov_b32 s49, scc // (op 13450) 
s_cmp_eq_i32 s49, 0 // (op 13450) 
s_cselect_b32 s51, 0, -1 // (op 13450) 

s_and_b32 s49, s46, 31 // (op 13450) 
s_lshl_b32 s72, 1, s49 // (op 13450) 
s_add_i32 s73, s72, s51 // (op 13450) 


s_and_b32 s51, s1, s73 // (op 13450) 


s_add_i32 s1, s50, s51 // (op 13450) 


s_ashr_i32 s50, s1, s49 // (op 13450) 


s_lshl_b32 s1, s46, 24 // (op 13450) 
s_ashr_i32 s49, s1, 31 // (op 13450) 

s_xor_b32 s1, s50, s49 // (op 13450) 

s_sub_i32 s50, s1, s49 // (op 13450) 


s_mul_i32 s1, s45, s44 // (op 13450) 
s_add_u32 s72, s8, 128 // (op 13450) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 13450) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 13450) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 13450) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 13450) 

s_sub_i32 s49, s72, s1 // (op 13450) 

s_mul_i32 s51, s50, s49 // (op 13450) 


s_sub_u32 s49, s19, s51 // (op 13450) 


s_add_u32 s19, s49, s1 // (op 13450) 


s_cmp_eq_i32 s18, 0 // (op 13450) 
s_mov_b32 s1, scc // (op 13450) 

s_cmp_eq_i32 s1, 0 // (op 13450) 
s_cselect_b32 s18, s19, s0 // (op 13450) 



v_lshl_add_u32 v24, s18, 7, v22 // (op 13450) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13450) 
s_lshr_b32 s18, s0, 2 // (op 13450) 

v_lshl_add_u32 v22, s18, 9, v23 // (op 13450) 


v_mov_b32 v27, 0 // (op 13450) convert
v_mov_b32 v26, v22 // (op 13450) convert

v_mul_lo_u32 v23, v24, s25 // (op 13450) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 13450) most significant: high of low * low
v_mul_lo_u32 v22, v24, s24 // (op 13450) least significant: low of low * low
v_add_u32 v23, v23, v25 // (op 13450) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v26 // (op 13450) least significant half
v_addc_co_u32 v23, s[0:1], v23, v27, s[0:1] // (op 13450) most significant half



v_lshlrev_b64 v[24:25], 2, v[22:23] // (op 13450) 

v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 13450) 

s_mul_i32 s1, s24, s9 // (op 13452) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13452) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13452) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13452) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13452) most significant: sum
s_add_u32 s1, s1, s19 // (op 13452) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13452) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13452) 

s_mov_b32 s20, s26 // (op 13452) reg expression
s_mov_b32 s21, s27 // (op 13452) reg expression
s_mov_b32 s22, s0 // (op 13452) reg expression
s_mov_b32 s23, 131072 // (op 13452) call()

s_mul_hi_i32 s0, s16, s11 // (op 13532) 
v_and_b32 v24, -4, v1 // (op 13532) 
s_add_i32 s1, s0, s16 // (op 13532) 

v_sub_u32 v25, v1, v24 // (op 13532) 

s_ashr_i32 s0, s1, 31 // (op 13532) 
v_lshlrev_b32 v24, 5, v25 // (op 13532) 

s_cmp_eq_i32 s11, 0 // (op 13532) 
s_mov_b32 s18, scc // (op 13532) 
v_lshrrev_b32 v25, 2, v1 // (op 13532) 
s_cmp_eq_i32 s18, 0 // (op 13532) 
s_cselect_b32 s19, 0, -1 // (op 13532) 

s_and_b32 s18, s12, 31 // (op 13532) 
s_lshl_b32 s49, 1, s18 // (op 13532) 
s_add_i32 s50, s49, s19 // (op 13532) 


s_and_b32 s19, s0, s50 // (op 13532) 


s_add_i32 s0, s1, s19 // (op 13532) 


s_ashr_i32 s1, s0, s18 // (op 13532) 


s_lshl_b32 s0, s12, 24 // (op 13532) 
s_ashr_i32 s18, s0, 31 // (op 13532) 

s_xor_b32 s0, s1, s18 // (op 13532) 

s_sub_i32 s1, s0, s18 // (op 13532) 


s_mul_hi_i32 s0, s1, s13 // (op 13532) 
s_add_i32 s18, s0, s1 // (op 13532) 

s_ashr_i32 s0, s18, 31 // (op 13532) 
s_cmp_eq_i32 s13, 0 // (op 13532) 
s_mov_b32 s19, scc // (op 13532) 
s_cmp_eq_i32 s19, 0 // (op 13532) 
s_cselect_b32 s49, 0, -1 // (op 13532) 

s_and_b32 s19, s14, 31 // (op 13532) 
s_lshl_b32 s50, 1, s19 // (op 13532) 
s_add_i32 s51, s50, s49 // (op 13532) 


s_and_b32 s49, s0, s51 // (op 13532) 


s_add_i32 s0, s18, s49 // (op 13532) 


s_ashr_i32 s18, s0, s19 // (op 13532) 


s_lshl_b32 s0, s14, 24 // (op 13532) 
s_ashr_i32 s19, s0, 31 // (op 13532) 

s_xor_b32 s0, s18, s19 // (op 13532) 

s_sub_i32 s18, s0, s19 // (op 13532) 


s_add_u32 s50, s6, 128 // (op 13532) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13532) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13532) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13532) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13532) 

s_mul_i32 s0, s18, s50 // (op 13532) 

s_sub_i32 s49, s1, s0 // (op 13532) 


s_mul_hi_i32 s0, s16, s10 // (op 13532) 
s_add_i32 s1, s0, s16 // (op 13532) 

s_ashr_i32 s0, s1, 31 // (op 13532) 
s_cmp_eq_i32 s10, 0 // (op 13532) 
s_mov_b32 s18, scc // (op 13532) 
s_cmp_eq_i32 s18, 0 // (op 13532) 
s_cselect_b32 s19, 0, -1 // (op 13532) 

s_and_b32 s18, s47, 31 // (op 13532) 
s_lshl_b32 s72, 1, s18 // (op 13532) 
s_add_i32 s73, s72, s19 // (op 13532) 


s_and_b32 s19, s0, s73 // (op 13532) 


s_add_i32 s0, s1, s19 // (op 13532) 


s_ashr_i32 s1, s0, s18 // (op 13532) 


s_lshl_b32 s0, s47, 24 // (op 13532) 
s_ashr_i32 s18, s0, 31 // (op 13532) 

s_xor_b32 s0, s1, s18 // (op 13532) 

s_sub_i32 s1, s0, s18 // (op 13532) 


s_mul_i32 s0, s44, s50 // (op 13532) 

s_mul_i32 s18, s48, s0 // (op 13532) 

s_mul_i32 s0, s1, s18 // (op 13532) 

s_sub_u32 s18, s16, s0 // (op 13532) 

s_mul_hi_i32 s0, s18, s15 // (op 13532) 
s_add_i32 s19, s0, s18 // (op 13532) 


s_ashr_i32 s0, s19, 31 // (op 13532) 
s_cmp_eq_i32 s15, 0 // (op 13532) 
s_mov_b32 s18, scc // (op 13532) 
s_cmp_eq_i32 s18, 0 // (op 13532) 
s_cselect_b32 s50, 0, -1 // (op 13532) 

s_and_b32 s18, s46, 31 // (op 13532) 
s_lshl_b32 s51, 1, s18 // (op 13532) 
s_add_i32 s72, s51, s50 // (op 13532) 


s_and_b32 s50, s0, s72 // (op 13532) 


s_add_i32 s0, s19, s50 // (op 13532) 


s_ashr_i32 s19, s0, s18 // (op 13532) 


s_lshl_b32 s0, s46, 24 // (op 13532) 
s_ashr_i32 s18, s0, 31 // (op 13532) 

s_xor_b32 s0, s19, s18 // (op 13532) 

s_sub_i32 s19, s0, s18 // (op 13532) 


s_cmp_eq_i32 s1, 0 // (op 13532) 
s_mov_b32 s0, scc // (op 13532) 

s_cmp_eq_i32 s0, 0 // (op 13532) 
s_cselect_b32 s1, s19, s49 // (op 13532) 



v_lshl_add_u32 v26, s1, 7, v24 // (op 13532) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13532) 
v_mov_b32 v31, 0 // (op 13532) convert
v_mov_b32 v30, v26 // (op 13532) convert

s_lshr_b32 s18, s0, 2 // (op 13532) 

v_lshl_add_u32 v24, s18, 9, v25 // (op 13532) 


v_mul_lo_u32 v27, v24, s29 // (op 13532) most significant: low of low * high
v_mul_hi_u32 v25, v24, s28 // (op 13532) most significant: high of low * low
v_mul_lo_u32 v26, v24, s28 // (op 13532) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 13532) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v30 // (op 13532) least significant half
v_addc_co_u32 v27, s[0:1], v27, v31, s[0:1] // (op 13532) most significant half



v_lshlrev_b64 v[24:25], 2, v[26:27] // (op 13532) 

v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 13532) 

s_mul_i32 s1, s28, s33 // (op 13534) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13534) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13534) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13534) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13534) most significant: sum
s_add_u32 s1, s1, s19 // (op 13534) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13534) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13534) 

s_mov_b32 s52, s30 // (op 13534) reg expression
s_mov_b32 s53, s31 // (op 13534) reg expression
s_mov_b32 s54, s0 // (op 13534) reg expression
s_mov_b32 s55, 131072 // (op 13534) call()

s_mul_hi_i32 s0, s16, s11 // (op 13596) 
v_lshrrev_b32 v24, 2, v1 // (op 13596) 
s_add_i32 s1, s0, s16 // (op 13596) 

v_lshlrev_b32 v25, 1, v24 // (op 13596) 

s_ashr_i32 s0, s1, 31 // (op 13596) 
v_and_b32 v24, -4, v1 // (op 13596) 
s_cmp_eq_i32 s11, 0 // (op 13596) 
s_mov_b32 s18, scc // (op 13596) 
v_sub_u32 v30, v1, v24 // (op 13596) 

s_cmp_eq_i32 s18, 0 // (op 13596) 
s_cselect_b32 s19, 0, -1 // (op 13596) 

s_and_b32 s18, s12, 31 // (op 13596) 
s_lshl_b32 s49, 1, s18 // (op 13596) 
s_add_i32 s50, s49, s19 // (op 13596) 


s_and_b32 s19, s0, s50 // (op 13596) 


s_add_i32 s0, s1, s19 // (op 13596) 


s_ashr_i32 s1, s0, s18 // (op 13596) 


s_lshl_b32 s0, s12, 24 // (op 13596) 
s_ashr_i32 s18, s0, 31 // (op 13596) 

s_xor_b32 s0, s1, s18 // (op 13596) 

s_sub_i32 s1, s0, s18 // (op 13596) 


s_mul_hi_i32 s0, s1, s13 // (op 13596) 
s_add_i32 s18, s0, s1 // (op 13596) 

s_ashr_i32 s0, s18, 31 // (op 13596) 
s_cmp_eq_i32 s13, 0 // (op 13596) 
s_mov_b32 s19, scc // (op 13596) 
s_cmp_eq_i32 s19, 0 // (op 13596) 
s_cselect_b32 s49, 0, -1 // (op 13596) 

s_and_b32 s19, s14, 31 // (op 13596) 
s_lshl_b32 s50, 1, s19 // (op 13596) 
s_add_i32 s51, s50, s49 // (op 13596) 


s_and_b32 s49, s0, s51 // (op 13596) 


s_add_i32 s0, s18, s49 // (op 13596) 


s_ashr_i32 s18, s0, s19 // (op 13596) 


s_lshl_b32 s0, s14, 24 // (op 13596) 
s_ashr_i32 s19, s0, 31 // (op 13596) 

s_xor_b32 s0, s18, s19 // (op 13596) 

s_sub_i32 s18, s0, s19 // (op 13596) 


s_mul_i32 s0, s1, s44 // (op 13596) 

s_sub_u32 s1, s16, s0 // (op 13596) 

s_mul_i32 s0, s18, s44 // (op 13596) 
s_add_u32 s0, s0, s1 // (op 13596) 


s_mul_hi_i32 s1, s16, s10 // (op 13596) 
s_add_i32 s18, s1, s16 // (op 13596) 

s_ashr_i32 s1, s18, 31 // (op 13596) 
s_cmp_eq_i32 s10, 0 // (op 13596) 
s_mov_b32 s19, scc // (op 13596) 
s_cmp_eq_i32 s19, 0 // (op 13596) 
s_cselect_b32 s49, 0, -1 // (op 13596) 

s_and_b32 s19, s47, 31 // (op 13596) 
s_lshl_b32 s50, 1, s19 // (op 13596) 
s_add_i32 s51, s50, s49 // (op 13596) 


s_and_b32 s49, s1, s51 // (op 13596) 


s_add_i32 s1, s18, s49 // (op 13596) 


s_ashr_i32 s18, s1, s19 // (op 13596) 


s_lshl_b32 s1, s47, 24 // (op 13596) 
s_ashr_i32 s19, s1, 31 // (op 13596) 

s_xor_b32 s1, s18, s19 // (op 13596) 

s_sub_i32 s18, s1, s19 // (op 13596) 


s_add_u32 s50, s6, 128 // (op 13596) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13596) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13596) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13596) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13596) 

s_mul_i32 s1, s44, s50 // (op 13596) 

s_mul_i32 s19, s48, s1 // (op 13596) 

s_mul_i32 s1, s18, s19 // (op 13596) 

s_sub_u32 s19, s16, s1 // (op 13596) 

s_mov_b32 s1, s19 // (op 13596) 
s_mul_hi_i32 s49, s1, s15 // (op 13596) 
s_add_i32 s50, s49, s1 // (op 13596) 


s_ashr_i32 s1, s50, 31 // (op 13596) 
s_cmp_eq_i32 s15, 0 // (op 13596) 
s_mov_b32 s49, scc // (op 13596) 
s_cmp_eq_i32 s49, 0 // (op 13596) 
s_cselect_b32 s51, 0, -1 // (op 13596) 

s_and_b32 s49, s46, 31 // (op 13596) 
s_lshl_b32 s72, 1, s49 // (op 13596) 
s_add_i32 s73, s72, s51 // (op 13596) 


s_and_b32 s51, s1, s73 // (op 13596) 


s_add_i32 s1, s50, s51 // (op 13596) 


s_ashr_i32 s50, s1, s49 // (op 13596) 


s_lshl_b32 s1, s46, 24 // (op 13596) 
s_ashr_i32 s49, s1, 31 // (op 13596) 

s_xor_b32 s1, s50, s49 // (op 13596) 

s_sub_i32 s50, s1, s49 // (op 13596) 


s_mul_i32 s1, s45, s44 // (op 13596) 
s_add_u32 s72, s8, 128 // (op 13596) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 13596) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 13596) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 13596) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 13596) 

s_sub_i32 s49, s72, s1 // (op 13596) 

s_mul_i32 s51, s50, s49 // (op 13596) 


s_sub_u32 s49, s19, s51 // (op 13596) 


s_add_u32 s19, s49, s1 // (op 13596) 


s_cmp_eq_i32 s18, 0 // (op 13596) 
s_mov_b32 s1, scc // (op 13596) 

s_cmp_eq_i32 s1, 0 // (op 13596) 
s_cselect_b32 s18, s19, s0 // (op 13596) 



v_lshl_add_u32 v24, s18, 7, v25 // (op 13596) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13596) 
s_lshr_b32 s18, s0, 2 // (op 13596) 

v_lshl_add_u32 v25, s18, 4, v30 // (op 13596) 


v_mul_lo_u32 v30, v24, s34 // (op 13596) 
v_add_u32 v30, v30, v25 // (op 13596) 


s_mul_i32 s1, s34, s9 // (op 13598) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13598) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13598) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13598) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13598) most significant: sum
s_add_u32 s1, s1, s19 // (op 13598) most significant: sum


s_mov_b32 s64, s36 // (op 13598) reg expression
s_mov_b32 s65, s37 // (op 13598) reg expression
s_mov_b32 s66, s0 // (op 13598) reg expression
s_mov_b32 s67, 131072 // (op 13598) call()

s_mul_hi_i32 s0, s16, s11 // (op 13660) 
v_and_b32 v24, -128, v1 // (op 13660) 
s_add_i32 s1, s0, s16 // (op 13660) 

v_sub_u32 v25, v1, v24 // (op 13660) 

s_ashr_i32 s0, s1, 31 // (op 13660) 
v_lshrrev_b32 v24, 7, v1 // (op 13660) 
s_cmp_eq_i32 s11, 0 // (op 13660) 
s_mov_b32 s18, scc // (op 13660) 
v_lshlrev_b32 v31, 1, v24 // (op 13660) 

s_cmp_eq_i32 s18, 0 // (op 13660) 
s_cselect_b32 s19, 0, -1 // (op 13660) 

s_and_b32 s18, s12, 31 // (op 13660) 
s_lshl_b32 s49, 1, s18 // (op 13660) 
s_add_i32 s50, s49, s19 // (op 13660) 


s_and_b32 s19, s0, s50 // (op 13660) 


s_add_i32 s0, s1, s19 // (op 13660) 


s_ashr_i32 s1, s0, s18 // (op 13660) 


s_lshl_b32 s0, s12, 24 // (op 13660) 
s_ashr_i32 s18, s0, 31 // (op 13660) 

s_xor_b32 s0, s1, s18 // (op 13660) 

s_sub_i32 s1, s0, s18 // (op 13660) 


s_mul_hi_i32 s0, s1, s13 // (op 13660) 
s_add_i32 s18, s0, s1 // (op 13660) 

s_ashr_i32 s0, s18, 31 // (op 13660) 
s_cmp_eq_i32 s13, 0 // (op 13660) 
s_mov_b32 s19, scc // (op 13660) 
s_cmp_eq_i32 s19, 0 // (op 13660) 
s_cselect_b32 s49, 0, -1 // (op 13660) 

s_and_b32 s19, s14, 31 // (op 13660) 
s_lshl_b32 s50, 1, s19 // (op 13660) 
s_add_i32 s51, s50, s49 // (op 13660) 


s_and_b32 s49, s0, s51 // (op 13660) 


s_add_i32 s0, s18, s49 // (op 13660) 


s_ashr_i32 s18, s0, s19 // (op 13660) 


s_lshl_b32 s0, s14, 24 // (op 13660) 
s_ashr_i32 s19, s0, 31 // (op 13660) 

s_xor_b32 s0, s18, s19 // (op 13660) 

s_sub_i32 s18, s0, s19 // (op 13660) 


s_add_u32 s50, s6, 128 // (op 13660) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13660) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13660) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13660) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13660) 

s_mul_i32 s0, s18, s50 // (op 13660) 

s_sub_i32 s49, s1, s0 // (op 13660) 


s_mul_hi_i32 s0, s16, s10 // (op 13660) 
s_add_i32 s1, s0, s16 // (op 13660) 

s_ashr_i32 s0, s1, 31 // (op 13660) 
s_cmp_eq_i32 s10, 0 // (op 13660) 
s_mov_b32 s18, scc // (op 13660) 
s_cmp_eq_i32 s18, 0 // (op 13660) 
s_cselect_b32 s19, 0, -1 // (op 13660) 

s_and_b32 s18, s47, 31 // (op 13660) 
s_lshl_b32 s72, 1, s18 // (op 13660) 
s_add_i32 s73, s72, s19 // (op 13660) 


s_and_b32 s19, s0, s73 // (op 13660) 


s_add_i32 s0, s1, s19 // (op 13660) 


s_ashr_i32 s1, s0, s18 // (op 13660) 


s_lshl_b32 s0, s47, 24 // (op 13660) 
s_ashr_i32 s18, s0, 31 // (op 13660) 

s_xor_b32 s0, s1, s18 // (op 13660) 

s_sub_i32 s1, s0, s18 // (op 13660) 


s_mul_i32 s0, s44, s50 // (op 13660) 

s_mul_i32 s18, s48, s0 // (op 13660) 

s_mul_i32 s0, s1, s18 // (op 13660) 

s_sub_u32 s18, s16, s0 // (op 13660) 

s_mul_hi_i32 s0, s18, s15 // (op 13660) 
s_add_i32 s19, s0, s18 // (op 13660) 


s_ashr_i32 s0, s19, 31 // (op 13660) 
s_cmp_eq_i32 s15, 0 // (op 13660) 
s_mov_b32 s18, scc // (op 13660) 
s_cmp_eq_i32 s18, 0 // (op 13660) 
s_cselect_b32 s50, 0, -1 // (op 13660) 

s_and_b32 s18, s46, 31 // (op 13660) 
s_lshl_b32 s51, 1, s18 // (op 13660) 
s_add_i32 s72, s51, s50 // (op 13660) 


s_and_b32 s50, s0, s72 // (op 13660) 


s_add_i32 s0, s19, s50 // (op 13660) 


s_ashr_i32 s19, s0, s18 // (op 13660) 


s_lshl_b32 s0, s46, 24 // (op 13660) 
s_ashr_i32 s18, s0, 31 // (op 13660) 

s_xor_b32 s0, s19, s18 // (op 13660) 

s_sub_i32 s19, s0, s18 // (op 13660) 


s_cmp_eq_i32 s1, 0 // (op 13660) 
s_mov_b32 s0, scc // (op 13660) 

s_cmp_eq_i32 s0, 0 // (op 13660) 
s_cselect_b32 s1, s19, s49 // (op 13660) 



v_lshl_add_u32 v24, s1, 7, v25 // (op 13660) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13660) 
s_lshr_b32 s18, s0, 2 // (op 13660) 

v_lshl_add_u32 v25, s18, 4, v31 // (op 13660) 


v_mul_lo_u32 v31, v25, s38 // (op 13660) 
v_add_u32 v31, v31, v24 // (op 13660) 


s_mul_i32 s1, s38, s43 // (op 13662) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13662) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13662) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13662) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13662) most significant: sum
s_add_u32 s1, s1, s19 // (op 13662) most significant: sum


s_mov_b32 s68, s40 // (op 13662) reg expression
s_mov_b32 s69, s41 // (op 13662) reg expression
s_mov_b32 s70, s0 // (op 13662) reg expression
s_mov_b32 s71, 131072 // (op 13662) call()

v_and_b32 v24, -4, v1 // (op 13758) 
v_sub_u32 v25, v1, v24 // (op 13758) 

v_lshlrev_b32 v24, 5, v25 // (op 13758) 

v_lshrrev_b32 v25, 2, v1 // (op 13758) 
v_lshl_add_u32 v32, v25, 7, v24 // (op 13758) 


v_lshlrev_b32 v24, 2, v32 // (op 13758) 

v_lshrrev_b32 v25, 3, v24 // (op 13758) 

s_nop 0
v_readfirstlane_b32 s0, v25 // (op 13758) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 13815) 
v_sub_u32 v25, v1, v24 // (op 13815) 

v_lshlrev_b32 v24, 5, v25 // (op 13815) 

v_lshrrev_b32 v25, 2, v1 // (op 13815) 
v_lshl_add_u32 v32, v25, 7, v24 // (op 13815) 


v_lshlrev_b32 v24, 2, v32 // (op 13815) 

v_lshrrev_b32 v25, 3, v24 // (op 13815) 

s_nop 0
v_readfirstlane_b32 s1, v25 // (op 13815) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 13872) 
v_sub_u32 v25, v1, v24 // (op 13872) 

v_lshrrev_b32 v24, 2, v1 // (op 13872) 
v_lshl_add_u32 v32, v24, 3, v25 // (op 13872) 


v_and_b32 v24, -128, v1 // (op 13930) 
v_sub_u32 v25, v1, v24 // (op 13930) 

v_lshrrev_b32 v24, 7, v1 // (op 13930) 
v_lshl_add_u32 v33, v24, 8, v25 // (op 13930) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 4196) 
s_and_b32 s49, s18, 4294967292 // (op 4196) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 4195) 
s_mov_b32 s50, s18 // (op 4195) convert

s_cmp_lt_u32 s49, s50 // (op 4195) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4195 // (op 4195) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4195 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4195:
s_barrier  // (op 14635) 
s_add_u32 s49, s49, 1 // (op 4197) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 13374) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13374) 

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 13374) 


v_mov_b64 v[28:29], v[22:23] // (op 13374) 
v_mov_b64 v[34:35], v[28:29] // (op 13374) 
s_add_u32 s18, 8192, s0 // (op 13374) 
s_mov_b32 m0, s18 // (op 13374) reg expression
buffer_load_dwordx4 v34, s[20:23], 0 offen offset: 0 lds // (op 13374) Load value direct to lds
v_add_co_u32 v34, s[50:51], v34, v24 // (op 13374) least significant half
v_addc_co_u32 v35, s[50:51], 0, v35, s[50:51] // (op 13374) most significant half

s_add_u32 m0, m0, 4096 // (op 13374) 
buffer_load_dwordx4 v34, s[20:23], 0 offen offset: 0 lds // (op 13374) Load value direct to lds




v_add_co_u32 v22, s[18:19], 64, v22 // (op 13440) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 13440) most significant half

s_barrier  // (op 14651) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 13380) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13380) 

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 13380) 


v_mov_b64 v[28:29], v[26:27] // (op 13380) 
v_mov_b64 v[34:35], v[28:29] // (op 13380) reg expression
s_mov_b32 m0, s1 // (op 13380) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13380) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v24 // (op 13380) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 13380) most significant half

s_add_u32 m0, m0, 4096 // (op 13380) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13380) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 13524) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13524) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13524) 

v_mov_b32 v24, s18 // (op 13524) 
v_mov_b32 v25, s19 // (op 13524) 
v_add_co_u32 v26, s[50:51], v26, v24 // (op 13524) least significant half
v_addc_co_u32 v27, s[50:51], v27, v25, s[50:51] // (op 13524) most significant half





v_mov_b64 v[24:25], s[34:35] // (op 195) convert

v_mov_b32 v28, v30 // (op 195) 
v_mov_b32 v29, v28 // (op 195) 
buffer_load_ubyte v34, v29, s[64:67], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v29, v29, v24 // (op 195) 
buffer_load_ubyte v35, v29, s[64:67], 0 offen offset: 0 // (op 195) Load value



v_add_u32 v30, 4, v30 // (op 13588) 

s_barrier  // (op 14668) 
v_mov_b64 v[24:25], s[38:39] // (op 196) convert




v_mov_b32 v28, v31 // (op 196) 
v_mov_b32 v29, v28 // (op 196) reg expression
v_add_u32 v36, 66048, v32 // (op 192) 
s_waitcnt vmcnt(1)
ds_write_b8 v36, v34 // (op 192) Store local data 

v_add_u32 v36, 66052, v32 // (op 192) 
buffer_load_ubyte v38, v29, s[68:71], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v29, v29, v24 // (op 196) 
buffer_load_ubyte v39, v29, s[68:71], 0 offen offset: 0 // (op 196) Load value



s_lshl_b32 s18, s38, 2 // (op 13652) 
v_add_u32 v31, v31, s18 // (op 13652) 

s_waitcnt vmcnt(2)
ds_write_b8 v36, v35 // (op 192) Store local data 


s_barrier  // (op 14641) 



s_barrier  // (op 14686) 
ds_read_b128 v[34:37], v3 offset:8192 // (op 183) Load local data 



s_barrier  // (op 14658) 
v_add_u32 v24, 65536, v33 // (op 193) 



ds_read_b64_tr_b4 v[40:41], v4 // (op 185) Transpose load local data 
ds_read_b64_tr_b4 v[42:43], v4 offset:1024 // (op 185) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v38 // (op 193) Store local data 

v_add_u32 v24, 65664, v33 // (op 193) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14676) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v39 // (op 193) Store local data 





v_add_u32 v24, 66048, v0 // (op 187) 
ds_read_u8 v25, v24 // (op 187) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 14693) 


v_add_u32 v24, 65536, v5 // (op 189) 
ds_read_u8 v28, v24 // (op 189) Load local data 




ds_read_b64_tr_b4 v[44:45], v4 offset:16 // (op 500) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v4 offset:1040 // (op 500) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[40:43], a[0:3], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 


v_add_u32 v24, 65568, v5 // (op 504) 
ds_read_u8 v29, v24 // (op 504) Load local data 




ds_read_b64_tr_b4 v[48:49], v4 offset:32 // (op 798) Transpose load local data 
ds_read_b64_tr_b4 v[50:51], v4 offset:1056 // (op 798) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[44:47], a[4:7], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 


v_add_u32 v24, 65600, v5 // (op 802) 
ds_read_u8 v38, v24 // (op 802) Load local data 




ds_read_b64_tr_b4 v[52:53], v4 offset:48 // (op 1096) Transpose load local data 
ds_read_b64_tr_b4 v[54:55], v4 offset:1072 // (op 1096) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[48:51], a[8:11], v25, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 803) 


v_add_u32 v24, 65632, v5 // (op 1100) 
ds_read_u8 v39, v24 // (op 1100) Load local data 




ds_read_b128 v[56:59], v3 offset:10240 // (op 1528) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[52:55], a[12:15], v25, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 





v_add_u32 v24, 66176, v0 // (op 1531) 
ds_read_u8 v25, v24 // (op 1531) Load local data 




ds_read_b128 v[34:37], v3 offset:12288 // (op 2818) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[40:43], a[16:19], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1533) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1564) 
v_add_u32 v24, 66304, v0 // (op 2821) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v25, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1595) 
ds_read_u8 v60, v24 // (op 2821) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[52:55], a[28:31], v25, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1626) 
ds_read_b128 v[62:65], v3 offset:14336 // (op 4108) Load local data 





v_add_u32 v61, 66432, v0 // (op 4111) 
ds_read_u8 v24, v61 // (op 4111) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[40:43], a[32:35], v60, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2823) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v60, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v60, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2885) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[52:55], a[44:47], v60, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2916) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[40:43], a[48:51], v24, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4113) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[62:65], v[44:47], a[52:55], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:65], v[48:51], a[56:59], v24, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4175) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[62:65], v[52:55], a[60:63], v24, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4206) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 4195) 
s_mov_b32 s50, s18 // (op 4195) convert

s_cmp_lt_u32 s49, s50 // (op 4195) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4195 // (op 4195) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4195 if true)
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4195:























s_barrier  // (op 14871) 





























v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 

v_lshrrev_b32 v14, 16, v10 // (op 56) 
v_lshrrev_b32 v15, 16, v11 // (op 56) 
v_lshrrev_b32 v16, 16, v12 // (op 56) 
v_lshrrev_b32 v17, 16, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v21 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 5877) 
v_accvgpr_read v15, a5 // (op 5877) 
v_accvgpr_read v16, a6 // (op 5877) 
v_accvgpr_read v17, a7 // (op 5877) 
v_mul_f32 v10, v19, v14 // (op 5877) 
v_mul_f32 v11, v19, v15 // (op 5877) 
v_mul_f32 v12, v19, v16 // (op 5877) 
v_mul_f32 v13, v19, v17 // (op 5877) 

v_lshrrev_b32 v14, 16, v10 // (op 5879) 
v_lshrrev_b32 v15, 16, v11 // (op 5879) 
v_lshrrev_b32 v16, 16, v12 // (op 5879) 
v_lshrrev_b32 v17, 16, v13 // (op 5879) 


v_pack_b32_f16 v12, v14, v15 // (op 5880) 
v_pack_b32_f16 v13, v16, v17 // (op 5880) 

s_lshl_b32 s0, s4, 6 // (op 5880) 
v_add_u32 v0, v21, s0 // (op 5880) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5880) Store value



v_accvgpr_read v14, a8 // (op 5891) 
v_accvgpr_read v15, a9 // (op 5891) 
v_accvgpr_read v16, a10 // (op 5891) 
v_accvgpr_read v17, a11 // (op 5891) 
v_mul_f32 v10, v19, v14 // (op 5891) 
v_mul_f32 v11, v19, v15 // (op 5891) 
v_mul_f32 v12, v19, v16 // (op 5891) 
v_mul_f32 v13, v19, v17 // (op 5891) 

v_lshrrev_b32 v14, 16, v10 // (op 5893) 
v_lshrrev_b32 v15, 16, v11 // (op 5893) 
v_lshrrev_b32 v16, 16, v12 // (op 5893) 
v_lshrrev_b32 v17, 16, v13 // (op 5893) 


v_pack_b32_f16 v12, v14, v15 // (op 5894) 
v_pack_b32_f16 v13, v16, v17 // (op 5894) 

s_lshl_b32 s0, s4, 7 // (op 5894) 
v_add_u32 v0, v21, s0 // (op 5894) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5894) Store value



v_accvgpr_read v14, a12 // (op 5905) 
v_accvgpr_read v15, a13 // (op 5905) 
v_accvgpr_read v16, a14 // (op 5905) 
v_accvgpr_read v17, a15 // (op 5905) 
v_mul_f32 v10, v19, v14 // (op 5905) 
v_mul_f32 v11, v19, v15 // (op 5905) 
v_mul_f32 v12, v19, v16 // (op 5905) 
v_mul_f32 v13, v19, v17 // (op 5905) 

v_lshrrev_b32 v14, 16, v10 // (op 5907) 
v_lshrrev_b32 v15, 16, v11 // (op 5907) 
v_lshrrev_b32 v16, 16, v12 // (op 5907) 
v_lshrrev_b32 v17, 16, v13 // (op 5907) 


v_pack_b32_f16 v12, v14, v15 // (op 5908) 
v_pack_b32_f16 v13, v16, v17 // (op 5908) 

s_lshl_b32 s0, s4, 6 // (op 5908) 
s_mul_i32 s1, 3, s0 // (op 5908) 

v_add_u32 v0, v21, s1 // (op 5908) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5908) Store value



v_accvgpr_read v14, a16 // (op 5941) 
v_accvgpr_read v15, a17 // (op 5941) 
v_accvgpr_read v16, a18 // (op 5941) 
v_accvgpr_read v17, a19 // (op 5941) 
v_mul_f32 v10, v19, v14 // (op 5941) 
v_mul_f32 v11, v19, v15 // (op 5941) 
v_mul_f32 v12, v19, v16 // (op 5941) 
v_mul_f32 v13, v19, v17 // (op 5941) 

v_lshrrev_b32 v14, 16, v10 // (op 5943) 
v_lshrrev_b32 v15, 16, v11 // (op 5943) 
v_lshrrev_b32 v16, 16, v12 // (op 5943) 
v_lshrrev_b32 v17, 16, v13 // (op 5943) 


v_pack_b32_f16 v12, v14, v15 // (op 5945) 
v_pack_b32_f16 v13, v16, v17 // (op 5945) 

v_add_u32 v0, 64, v21 // (op 5945) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5945) Store value



v_accvgpr_read v14, a20 // (op 5955) 
v_accvgpr_read v15, a21 // (op 5955) 
v_accvgpr_read v16, a22 // (op 5955) 
v_accvgpr_read v17, a23 // (op 5955) 
v_mul_f32 v10, v19, v14 // (op 5955) 
v_mul_f32 v11, v19, v15 // (op 5955) 
v_mul_f32 v12, v19, v16 // (op 5955) 
v_mul_f32 v13, v19, v17 // (op 5955) 

v_lshrrev_b32 v14, 16, v10 // (op 5957) 
v_lshrrev_b32 v15, 16, v11 // (op 5957) 
v_lshrrev_b32 v16, 16, v12 // (op 5957) 
v_lshrrev_b32 v17, 16, v13 // (op 5957) 


v_pack_b32_f16 v12, v14, v15 // (op 5947) 
v_pack_b32_f16 v13, v16, v17 // (op 5947) 

s_lshl_b32 s0, s4, 6 // (op 5947) 
s_add_u32 s0, s0, 64 // (op 5947) 
v_add_u32 v0, v21, s0 // (op 5947) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5947) Store value



v_accvgpr_read v14, a24 // (op 5961) 
v_accvgpr_read v15, a25 // (op 5961) 
v_accvgpr_read v16, a26 // (op 5961) 
v_accvgpr_read v17, a27 // (op 5961) 
v_mul_f32 v10, v19, v14 // (op 5961) 
v_mul_f32 v11, v19, v15 // (op 5961) 
v_mul_f32 v12, v19, v16 // (op 5961) 
v_mul_f32 v13, v19, v17 // (op 5961) 

v_lshrrev_b32 v14, 16, v10 // (op 5963) 
v_lshrrev_b32 v15, 16, v11 // (op 5963) 
v_lshrrev_b32 v16, 16, v12 // (op 5963) 
v_lshrrev_b32 v17, 16, v13 // (op 5963) 


v_pack_b32_f16 v12, v14, v15 // (op 5949) 
v_pack_b32_f16 v13, v16, v17 // (op 5949) 

s_lshl_b32 s0, s4, 7 // (op 5949) 
s_add_u32 s0, s0, 64 // (op 5949) 
v_add_u32 v0, v21, s0 // (op 5949) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5949) Store value



v_accvgpr_read v14, a28 // (op 5967) 
v_accvgpr_read v15, a29 // (op 5967) 
v_accvgpr_read v16, a30 // (op 5967) 
v_accvgpr_read v17, a31 // (op 5967) 
v_mul_f32 v10, v19, v14 // (op 5967) 
v_mul_f32 v11, v19, v15 // (op 5967) 
v_mul_f32 v12, v19, v16 // (op 5967) 
v_mul_f32 v13, v19, v17 // (op 5967) 

v_lshrrev_b32 v14, 16, v10 // (op 5969) 
v_lshrrev_b32 v15, 16, v11 // (op 5969) 
v_lshrrev_b32 v16, 16, v12 // (op 5969) 
v_lshrrev_b32 v17, 16, v13 // (op 5969) 


v_pack_b32_f16 v12, v14, v15 // (op 5951) 
v_pack_b32_f16 v13, v16, v17 // (op 5951) 

s_lshl_b32 s0, s4, 6 // (op 5951) 
s_mul_i32 s1, 3, s0 // (op 5951) 
s_add_u32 s1, s1, 64 // (op 5951) 

v_add_u32 v0, v21, s1 // (op 5951) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 5951) Store value



v_accvgpr_read v14, a32 // (op 6017) 
v_accvgpr_read v15, a33 // (op 6017) 
v_accvgpr_read v16, a34 // (op 6017) 
v_accvgpr_read v17, a35 // (op 6017) 
v_mul_f32 v10, v19, v14 // (op 6017) 
v_mul_f32 v11, v19, v15 // (op 6017) 
v_mul_f32 v12, v19, v16 // (op 6017) 
v_mul_f32 v13, v19, v17 // (op 6017) 

v_lshrrev_b32 v14, 16, v10 // (op 6019) 
v_lshrrev_b32 v15, 16, v11 // (op 6019) 
v_lshrrev_b32 v16, 16, v12 // (op 6019) 
v_lshrrev_b32 v17, 16, v13 // (op 6019) 


v_pack_b32_f16 v12, v14, v15 // (op 6021) 
v_pack_b32_f16 v13, v16, v17 // (op 6021) 

v_add_u32 v0, 128, v21 // (op 6021) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6021) Store value



v_accvgpr_read v14, a36 // (op 6031) 
v_accvgpr_read v15, a37 // (op 6031) 
v_accvgpr_read v16, a38 // (op 6031) 
v_accvgpr_read v17, a39 // (op 6031) 
v_mul_f32 v10, v19, v14 // (op 6031) 
v_mul_f32 v11, v19, v15 // (op 6031) 
v_mul_f32 v12, v19, v16 // (op 6031) 
v_mul_f32 v13, v19, v17 // (op 6031) 

v_lshrrev_b32 v14, 16, v10 // (op 6033) 
v_lshrrev_b32 v15, 16, v11 // (op 6033) 
v_lshrrev_b32 v16, 16, v12 // (op 6033) 
v_lshrrev_b32 v17, 16, v13 // (op 6033) 


v_pack_b32_f16 v12, v14, v15 // (op 6023) 
v_pack_b32_f16 v13, v16, v17 // (op 6023) 

s_lshl_b32 s0, s4, 6 // (op 6023) 
s_add_u32 s0, s0, 128 // (op 6023) 
v_add_u32 v0, v21, s0 // (op 6023) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6023) Store value



v_accvgpr_read v14, a40 // (op 6037) 
v_accvgpr_read v15, a41 // (op 6037) 
v_accvgpr_read v16, a42 // (op 6037) 
v_accvgpr_read v17, a43 // (op 6037) 
v_mul_f32 v10, v19, v14 // (op 6037) 
v_mul_f32 v11, v19, v15 // (op 6037) 
v_mul_f32 v12, v19, v16 // (op 6037) 
v_mul_f32 v13, v19, v17 // (op 6037) 

v_lshrrev_b32 v14, 16, v10 // (op 6039) 
v_lshrrev_b32 v15, 16, v11 // (op 6039) 
v_lshrrev_b32 v16, 16, v12 // (op 6039) 
v_lshrrev_b32 v17, 16, v13 // (op 6039) 


v_pack_b32_f16 v12, v14, v15 // (op 6025) 
v_pack_b32_f16 v13, v16, v17 // (op 6025) 

s_lshl_b32 s0, s4, 7 // (op 6025) 
s_add_u32 s0, s0, 128 // (op 6025) 
v_add_u32 v0, v21, s0 // (op 6025) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6025) Store value



v_accvgpr_read v14, a44 // (op 6043) 
v_accvgpr_read v15, a45 // (op 6043) 
v_accvgpr_read v16, a46 // (op 6043) 
v_accvgpr_read v17, a47 // (op 6043) 
v_mul_f32 v10, v19, v14 // (op 6043) 
v_mul_f32 v11, v19, v15 // (op 6043) 
v_mul_f32 v12, v19, v16 // (op 6043) 
v_mul_f32 v13, v19, v17 // (op 6043) 

v_lshrrev_b32 v14, 16, v10 // (op 6045) 
v_lshrrev_b32 v15, 16, v11 // (op 6045) 
v_lshrrev_b32 v16, 16, v12 // (op 6045) 
v_lshrrev_b32 v17, 16, v13 // (op 6045) 


v_pack_b32_f16 v12, v14, v15 // (op 6027) 
v_pack_b32_f16 v13, v16, v17 // (op 6027) 

s_lshl_b32 s0, s4, 6 // (op 6027) 
s_mul_i32 s1, 3, s0 // (op 6027) 
s_add_u32 s1, s1, 128 // (op 6027) 

v_add_u32 v0, v21, s1 // (op 6027) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6027) Store value



v_accvgpr_read v14, a48 // (op 6093) 
v_accvgpr_read v15, a49 // (op 6093) 
v_accvgpr_read v16, a50 // (op 6093) 
v_accvgpr_read v17, a51 // (op 6093) 
v_mul_f32 v10, v19, v14 // (op 6093) 
v_mul_f32 v11, v19, v15 // (op 6093) 
v_mul_f32 v12, v19, v16 // (op 6093) 
v_mul_f32 v13, v19, v17 // (op 6093) 

v_lshrrev_b32 v14, 16, v10 // (op 6095) 
v_lshrrev_b32 v15, 16, v11 // (op 6095) 
v_lshrrev_b32 v16, 16, v12 // (op 6095) 
v_lshrrev_b32 v17, 16, v13 // (op 6095) 


v_pack_b32_f16 v12, v14, v15 // (op 6097) 
v_pack_b32_f16 v13, v16, v17 // (op 6097) 

v_add_u32 v0, 192, v21 // (op 6097) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6097) Store value



v_accvgpr_read v14, a52 // (op 6107) 
v_accvgpr_read v15, a53 // (op 6107) 
v_accvgpr_read v16, a54 // (op 6107) 
v_accvgpr_read v17, a55 // (op 6107) 
v_mul_f32 v10, v19, v14 // (op 6107) 
v_mul_f32 v11, v19, v15 // (op 6107) 
v_mul_f32 v12, v19, v16 // (op 6107) 
v_mul_f32 v13, v19, v17 // (op 6107) 

v_lshrrev_b32 v14, 16, v10 // (op 6109) 
v_lshrrev_b32 v15, 16, v11 // (op 6109) 
v_lshrrev_b32 v16, 16, v12 // (op 6109) 
v_lshrrev_b32 v17, 16, v13 // (op 6109) 


v_pack_b32_f16 v12, v14, v15 // (op 6099) 
v_pack_b32_f16 v13, v16, v17 // (op 6099) 

s_lshl_b32 s0, s4, 6 // (op 6099) 
s_add_u32 s0, s0, 192 // (op 6099) 
v_add_u32 v0, v21, s0 // (op 6099) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6099) Store value



v_accvgpr_read v14, a56 // (op 6113) 
v_accvgpr_read v15, a57 // (op 6113) 
v_accvgpr_read v16, a58 // (op 6113) 
v_accvgpr_read v17, a59 // (op 6113) 
v_mul_f32 v10, v19, v14 // (op 6113) 
v_mul_f32 v11, v19, v15 // (op 6113) 
v_mul_f32 v12, v19, v16 // (op 6113) 
v_mul_f32 v13, v19, v17 // (op 6113) 

v_lshrrev_b32 v14, 16, v10 // (op 6115) 
v_lshrrev_b32 v15, 16, v11 // (op 6115) 
v_lshrrev_b32 v16, 16, v12 // (op 6115) 
v_lshrrev_b32 v17, 16, v13 // (op 6115) 


v_pack_b32_f16 v12, v14, v15 // (op 6101) 
v_pack_b32_f16 v13, v16, v17 // (op 6101) 

s_lshl_b32 s0, s4, 7 // (op 6101) 
s_add_u32 s0, s0, 192 // (op 6101) 
v_add_u32 v0, v21, s0 // (op 6101) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6101) Store value



v_accvgpr_read v14, a60 // (op 6119) 
v_accvgpr_read v15, a61 // (op 6119) 
v_accvgpr_read v16, a62 // (op 6119) 
v_accvgpr_read v17, a63 // (op 6119) 
v_mul_f32 v10, v19, v14 // (op 6119) 
v_mul_f32 v11, v19, v15 // (op 6119) 
v_mul_f32 v12, v19, v16 // (op 6119) 
v_mul_f32 v13, v19, v17 // (op 6119) 

v_lshrrev_b32 v14, 16, v10 // (op 6121) 
v_lshrrev_b32 v15, 16, v11 // (op 6121) 
v_lshrrev_b32 v16, 16, v12 // (op 6121) 
v_lshrrev_b32 v17, 16, v13 // (op 6121) 


v_pack_b32_f16 v12, v14, v15 // (op 6103) 
v_pack_b32_f16 v13, v16, v17 // (op 6103) 

s_lshl_b32 s0, s4, 6 // (op 6103) 
s_mul_i32 s1, 3, s0 // (op 6103) 
s_add_u32 s1, s1, 192 // (op 6103) 

v_add_u32 v0, v21, s1 // (op 6103) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 6103) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_135
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_135:


v_add_u32 v0, 192, v20 // (op 6491) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6491) Load value

v_accvgpr_read v12, a48 // (op 6478) 
v_accvgpr_read v13, a49 // (op 6478) 
v_accvgpr_read v14, a50 // (op 6478) 
v_accvgpr_read v15, a51 // (op 6478) 
v_mul_f32 v8, v19, v12 // (op 6478) 
v_mul_f32 v9, v19, v13 // (op 6478) 
v_mul_f32 v10, v19, v14 // (op 6478) 
v_mul_f32 v11, v19, v15 // (op 6478) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6490) 
v_lshrrev_b32 v7, 16, v16 // (op 6490) 
v_lshlrev_b32 v6, 16, v6 // (op 6490) 
v_lshlrev_b32 v7, 16, v7 // (op 6490) 
v_mul_f32 v12, v18, v6 // (op 6490) 
v_mul_f32 v13, v18, v7 // (op 6490) 
v_and_b32 v6, 65535, v17 // (op 6490) 
v_lshrrev_b32 v7, 16, v17 // (op 6490) 
v_lshlrev_b32 v6, 16, v6 // (op 6490) 
v_lshlrev_b32 v7, 16, v7 // (op 6490) 
v_mul_f32 v14, v18, v6 // (op 6490) 
v_mul_f32 v15, v18, v7 // (op 6490) 


v_add_f32 v4, v12, v8 // (op 6479) 
v_add_f32 v5, v13, v9 // (op 6479) 
v_add_f32 v6, v14, v10 // (op 6479) 
v_add_f32 v7, v15, v11 // (op 6479) 


v_lshrrev_b32 v8, 16, v4 // (op 6480) 
v_lshrrev_b32 v9, 16, v5 // (op 6480) 
v_lshrrev_b32 v10, 16, v6 // (op 6480) 
v_lshrrev_b32 v11, 16, v7 // (op 6480) 


v_pack_b32_f16 v6, v8, v9 // (op 6482) 
v_pack_b32_f16 v7, v10, v11 // (op 6482) 

v_add_u32 v0, 192, v21 // (op 6482) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6482) Store value





s_lshl_b32 s0, s2, 6 // (op 6496) 
s_add_u32 s0, s0, 192 // (op 6496) 
v_add_u32 v0, v20, s0 // (op 6496) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6496) Load value

v_accvgpr_read v12, a52 // (op 6508) 
v_accvgpr_read v13, a53 // (op 6508) 
v_accvgpr_read v14, a54 // (op 6508) 
v_accvgpr_read v15, a55 // (op 6508) 
v_mul_f32 v8, v19, v12 // (op 6508) 
v_mul_f32 v9, v19, v13 // (op 6508) 
v_mul_f32 v10, v19, v14 // (op 6508) 
v_mul_f32 v11, v19, v15 // (op 6508) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6493) 
v_lshrrev_b32 v7, 16, v16 // (op 6493) 
v_lshlrev_b32 v6, 16, v6 // (op 6493) 
v_lshlrev_b32 v7, 16, v7 // (op 6493) 
v_mul_f32 v12, v18, v6 // (op 6493) 
v_mul_f32 v13, v18, v7 // (op 6493) 
v_and_b32 v6, 65535, v17 // (op 6493) 
v_lshrrev_b32 v7, 16, v17 // (op 6493) 
v_lshlrev_b32 v6, 16, v6 // (op 6493) 
v_lshlrev_b32 v7, 16, v7 // (op 6493) 
v_mul_f32 v14, v18, v6 // (op 6493) 
v_mul_f32 v15, v18, v7 // (op 6493) 


v_add_f32 v4, v12, v8 // (op 6494) 
v_add_f32 v5, v13, v9 // (op 6494) 
v_add_f32 v6, v14, v10 // (op 6494) 
v_add_f32 v7, v15, v11 // (op 6494) 


v_lshrrev_b32 v8, 16, v4 // (op 6495) 
v_lshrrev_b32 v9, 16, v5 // (op 6495) 
v_lshrrev_b32 v10, 16, v6 // (op 6495) 
v_lshrrev_b32 v11, 16, v7 // (op 6495) 


v_pack_b32_f16 v6, v8, v9 // (op 6484) 
v_pack_b32_f16 v7, v10, v11 // (op 6484) 

s_lshl_b32 s0, s4, 6 // (op 6484) 
s_add_u32 s0, s0, 192 // (op 6484) 
v_add_u32 v0, v21, s0 // (op 6484) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6484) Store value





s_lshl_b32 s0, s2, 7 // (op 6501) 
s_add_u32 s0, s0, 192 // (op 6501) 
v_add_u32 v0, v20, s0 // (op 6501) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6501) Load value

v_accvgpr_read v12, a56 // (op 6510) 
v_accvgpr_read v13, a57 // (op 6510) 
v_accvgpr_read v14, a58 // (op 6510) 
v_accvgpr_read v15, a59 // (op 6510) 
v_mul_f32 v8, v19, v12 // (op 6510) 
v_mul_f32 v9, v19, v13 // (op 6510) 
v_mul_f32 v10, v19, v14 // (op 6510) 
v_mul_f32 v11, v19, v15 // (op 6510) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6498) 
v_lshrrev_b32 v7, 16, v16 // (op 6498) 
v_lshlrev_b32 v6, 16, v6 // (op 6498) 
v_lshlrev_b32 v7, 16, v7 // (op 6498) 
v_mul_f32 v12, v18, v6 // (op 6498) 
v_mul_f32 v13, v18, v7 // (op 6498) 
v_and_b32 v6, 65535, v17 // (op 6498) 
v_lshrrev_b32 v7, 16, v17 // (op 6498) 
v_lshlrev_b32 v6, 16, v6 // (op 6498) 
v_lshlrev_b32 v7, 16, v7 // (op 6498) 
v_mul_f32 v14, v18, v6 // (op 6498) 
v_mul_f32 v15, v18, v7 // (op 6498) 


v_add_f32 v4, v12, v8 // (op 6499) 
v_add_f32 v5, v13, v9 // (op 6499) 
v_add_f32 v6, v14, v10 // (op 6499) 
v_add_f32 v7, v15, v11 // (op 6499) 


v_lshrrev_b32 v8, 16, v4 // (op 6500) 
v_lshrrev_b32 v9, 16, v5 // (op 6500) 
v_lshrrev_b32 v10, 16, v6 // (op 6500) 
v_lshrrev_b32 v11, 16, v7 // (op 6500) 


v_pack_b32_f16 v6, v8, v9 // (op 6486) 
v_pack_b32_f16 v7, v10, v11 // (op 6486) 

s_lshl_b32 s0, s4, 7 // (op 6486) 
s_add_u32 s0, s0, 192 // (op 6486) 
v_add_u32 v0, v21, s0 // (op 6486) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6486) Store value





s_lshl_b32 s0, s2, 6 // (op 6506) 
s_mul_i32 s1, 3, s0 // (op 6506) 
s_add_u32 s1, s1, 192 // (op 6506) 

v_add_u32 v0, v20, s1 // (op 6506) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6506) Load value

v_accvgpr_read v12, a60 // (op 6512) 
v_accvgpr_read v13, a61 // (op 6512) 
v_accvgpr_read v14, a62 // (op 6512) 
v_accvgpr_read v15, a63 // (op 6512) 
v_mul_f32 v8, v19, v12 // (op 6512) 
v_mul_f32 v9, v19, v13 // (op 6512) 
v_mul_f32 v10, v19, v14 // (op 6512) 
v_mul_f32 v11, v19, v15 // (op 6512) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6503) 
v_lshrrev_b32 v7, 16, v16 // (op 6503) 
v_lshlrev_b32 v6, 16, v6 // (op 6503) 
v_lshlrev_b32 v7, 16, v7 // (op 6503) 
v_mul_f32 v12, v18, v6 // (op 6503) 
v_mul_f32 v13, v18, v7 // (op 6503) 
v_and_b32 v6, 65535, v17 // (op 6503) 
v_lshrrev_b32 v7, 16, v17 // (op 6503) 
v_lshlrev_b32 v6, 16, v6 // (op 6503) 
v_lshlrev_b32 v7, 16, v7 // (op 6503) 
v_mul_f32 v14, v18, v6 // (op 6503) 
v_mul_f32 v15, v18, v7 // (op 6503) 


v_add_f32 v4, v12, v8 // (op 6504) 
v_add_f32 v5, v13, v9 // (op 6504) 
v_add_f32 v6, v14, v10 // (op 6504) 
v_add_f32 v7, v15, v11 // (op 6504) 


v_lshrrev_b32 v8, 16, v4 // (op 6505) 
v_lshrrev_b32 v9, 16, v5 // (op 6505) 
v_lshrrev_b32 v10, 16, v6 // (op 6505) 
v_lshrrev_b32 v11, 16, v7 // (op 6505) 


v_pack_b32_f16 v6, v8, v9 // (op 6488) 
v_pack_b32_f16 v7, v10, v11 // (op 6488) 

s_lshl_b32 s0, s4, 6 // (op 6488) 
s_mul_i32 s1, 3, s0 // (op 6488) 
s_add_u32 s1, s1, 192 // (op 6488) 

v_add_u32 v0, v21, s1 // (op 6488) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6488) Store value





v_add_u32 v0, 128, v20 // (op 6404) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6404) Load value

v_accvgpr_read v12, a32 // (op 6391) 
v_accvgpr_read v13, a33 // (op 6391) 
v_accvgpr_read v14, a34 // (op 6391) 
v_accvgpr_read v15, a35 // (op 6391) 
v_mul_f32 v8, v19, v12 // (op 6391) 
v_mul_f32 v9, v19, v13 // (op 6391) 
v_mul_f32 v10, v19, v14 // (op 6391) 
v_mul_f32 v11, v19, v15 // (op 6391) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6403) 
v_lshrrev_b32 v7, 16, v16 // (op 6403) 
v_lshlrev_b32 v6, 16, v6 // (op 6403) 
v_lshlrev_b32 v7, 16, v7 // (op 6403) 
v_mul_f32 v12, v18, v6 // (op 6403) 
v_mul_f32 v13, v18, v7 // (op 6403) 
v_and_b32 v6, 65535, v17 // (op 6403) 
v_lshrrev_b32 v7, 16, v17 // (op 6403) 
v_lshlrev_b32 v6, 16, v6 // (op 6403) 
v_lshlrev_b32 v7, 16, v7 // (op 6403) 
v_mul_f32 v14, v18, v6 // (op 6403) 
v_mul_f32 v15, v18, v7 // (op 6403) 


v_add_f32 v4, v12, v8 // (op 6392) 
v_add_f32 v5, v13, v9 // (op 6392) 
v_add_f32 v6, v14, v10 // (op 6392) 
v_add_f32 v7, v15, v11 // (op 6392) 


v_lshrrev_b32 v8, 16, v4 // (op 6393) 
v_lshrrev_b32 v9, 16, v5 // (op 6393) 
v_lshrrev_b32 v10, 16, v6 // (op 6393) 
v_lshrrev_b32 v11, 16, v7 // (op 6393) 


v_pack_b32_f16 v6, v8, v9 // (op 6395) 
v_pack_b32_f16 v7, v10, v11 // (op 6395) 

v_add_u32 v0, 128, v21 // (op 6395) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6395) Store value





s_lshl_b32 s0, s2, 6 // (op 6409) 
s_add_u32 s0, s0, 128 // (op 6409) 
v_add_u32 v0, v20, s0 // (op 6409) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6409) Load value

v_accvgpr_read v12, a36 // (op 6421) 
v_accvgpr_read v13, a37 // (op 6421) 
v_accvgpr_read v14, a38 // (op 6421) 
v_accvgpr_read v15, a39 // (op 6421) 
v_mul_f32 v8, v19, v12 // (op 6421) 
v_mul_f32 v9, v19, v13 // (op 6421) 
v_mul_f32 v10, v19, v14 // (op 6421) 
v_mul_f32 v11, v19, v15 // (op 6421) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6406) 
v_lshrrev_b32 v7, 16, v16 // (op 6406) 
v_lshlrev_b32 v6, 16, v6 // (op 6406) 
v_lshlrev_b32 v7, 16, v7 // (op 6406) 
v_mul_f32 v12, v18, v6 // (op 6406) 
v_mul_f32 v13, v18, v7 // (op 6406) 
v_and_b32 v6, 65535, v17 // (op 6406) 
v_lshrrev_b32 v7, 16, v17 // (op 6406) 
v_lshlrev_b32 v6, 16, v6 // (op 6406) 
v_lshlrev_b32 v7, 16, v7 // (op 6406) 
v_mul_f32 v14, v18, v6 // (op 6406) 
v_mul_f32 v15, v18, v7 // (op 6406) 


v_add_f32 v4, v12, v8 // (op 6407) 
v_add_f32 v5, v13, v9 // (op 6407) 
v_add_f32 v6, v14, v10 // (op 6407) 
v_add_f32 v7, v15, v11 // (op 6407) 


v_lshrrev_b32 v8, 16, v4 // (op 6408) 
v_lshrrev_b32 v9, 16, v5 // (op 6408) 
v_lshrrev_b32 v10, 16, v6 // (op 6408) 
v_lshrrev_b32 v11, 16, v7 // (op 6408) 


v_pack_b32_f16 v6, v8, v9 // (op 6397) 
v_pack_b32_f16 v7, v10, v11 // (op 6397) 

s_lshl_b32 s0, s4, 6 // (op 6397) 
s_add_u32 s0, s0, 128 // (op 6397) 
v_add_u32 v0, v21, s0 // (op 6397) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6397) Store value





s_lshl_b32 s0, s2, 7 // (op 6414) 
s_add_u32 s0, s0, 128 // (op 6414) 
v_add_u32 v0, v20, s0 // (op 6414) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6414) Load value

v_accvgpr_read v12, a40 // (op 6423) 
v_accvgpr_read v13, a41 // (op 6423) 
v_accvgpr_read v14, a42 // (op 6423) 
v_accvgpr_read v15, a43 // (op 6423) 
v_mul_f32 v8, v19, v12 // (op 6423) 
v_mul_f32 v9, v19, v13 // (op 6423) 
v_mul_f32 v10, v19, v14 // (op 6423) 
v_mul_f32 v11, v19, v15 // (op 6423) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6411) 
v_lshrrev_b32 v7, 16, v16 // (op 6411) 
v_lshlrev_b32 v6, 16, v6 // (op 6411) 
v_lshlrev_b32 v7, 16, v7 // (op 6411) 
v_mul_f32 v12, v18, v6 // (op 6411) 
v_mul_f32 v13, v18, v7 // (op 6411) 
v_and_b32 v6, 65535, v17 // (op 6411) 
v_lshrrev_b32 v7, 16, v17 // (op 6411) 
v_lshlrev_b32 v6, 16, v6 // (op 6411) 
v_lshlrev_b32 v7, 16, v7 // (op 6411) 
v_mul_f32 v14, v18, v6 // (op 6411) 
v_mul_f32 v15, v18, v7 // (op 6411) 


v_add_f32 v4, v12, v8 // (op 6412) 
v_add_f32 v5, v13, v9 // (op 6412) 
v_add_f32 v6, v14, v10 // (op 6412) 
v_add_f32 v7, v15, v11 // (op 6412) 


v_lshrrev_b32 v8, 16, v4 // (op 6413) 
v_lshrrev_b32 v9, 16, v5 // (op 6413) 
v_lshrrev_b32 v10, 16, v6 // (op 6413) 
v_lshrrev_b32 v11, 16, v7 // (op 6413) 


v_pack_b32_f16 v6, v8, v9 // (op 6399) 
v_pack_b32_f16 v7, v10, v11 // (op 6399) 

s_lshl_b32 s0, s4, 7 // (op 6399) 
s_add_u32 s0, s0, 128 // (op 6399) 
v_add_u32 v0, v21, s0 // (op 6399) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6399) Store value





s_lshl_b32 s0, s2, 6 // (op 6419) 
s_mul_i32 s1, 3, s0 // (op 6419) 
s_add_u32 s1, s1, 128 // (op 6419) 

v_add_u32 v0, v20, s1 // (op 6419) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6419) Load value

v_accvgpr_read v12, a44 // (op 6425) 
v_accvgpr_read v13, a45 // (op 6425) 
v_accvgpr_read v14, a46 // (op 6425) 
v_accvgpr_read v15, a47 // (op 6425) 
v_mul_f32 v8, v19, v12 // (op 6425) 
v_mul_f32 v9, v19, v13 // (op 6425) 
v_mul_f32 v10, v19, v14 // (op 6425) 
v_mul_f32 v11, v19, v15 // (op 6425) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6416) 
v_lshrrev_b32 v7, 16, v16 // (op 6416) 
v_lshlrev_b32 v6, 16, v6 // (op 6416) 
v_lshlrev_b32 v7, 16, v7 // (op 6416) 
v_mul_f32 v12, v18, v6 // (op 6416) 
v_mul_f32 v13, v18, v7 // (op 6416) 
v_and_b32 v6, 65535, v17 // (op 6416) 
v_lshrrev_b32 v7, 16, v17 // (op 6416) 
v_lshlrev_b32 v6, 16, v6 // (op 6416) 
v_lshlrev_b32 v7, 16, v7 // (op 6416) 
v_mul_f32 v14, v18, v6 // (op 6416) 
v_mul_f32 v15, v18, v7 // (op 6416) 


v_add_f32 v4, v12, v8 // (op 6417) 
v_add_f32 v5, v13, v9 // (op 6417) 
v_add_f32 v6, v14, v10 // (op 6417) 
v_add_f32 v7, v15, v11 // (op 6417) 


v_lshrrev_b32 v8, 16, v4 // (op 6418) 
v_lshrrev_b32 v9, 16, v5 // (op 6418) 
v_lshrrev_b32 v10, 16, v6 // (op 6418) 
v_lshrrev_b32 v11, 16, v7 // (op 6418) 


v_pack_b32_f16 v6, v8, v9 // (op 6401) 
v_pack_b32_f16 v7, v10, v11 // (op 6401) 

s_lshl_b32 s0, s4, 6 // (op 6401) 
s_mul_i32 s1, 3, s0 // (op 6401) 
s_add_u32 s1, s1, 128 // (op 6401) 

v_add_u32 v0, v21, s1 // (op 6401) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6401) Store value





v_add_u32 v0, 64, v20 // (op 6317) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6317) Load value

v_accvgpr_read v12, a16 // (op 6304) 
v_accvgpr_read v13, a17 // (op 6304) 
v_accvgpr_read v14, a18 // (op 6304) 
v_accvgpr_read v15, a19 // (op 6304) 
v_mul_f32 v8, v19, v12 // (op 6304) 
v_mul_f32 v9, v19, v13 // (op 6304) 
v_mul_f32 v10, v19, v14 // (op 6304) 
v_mul_f32 v11, v19, v15 // (op 6304) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6316) 
v_lshrrev_b32 v7, 16, v16 // (op 6316) 
v_lshlrev_b32 v6, 16, v6 // (op 6316) 
v_lshlrev_b32 v7, 16, v7 // (op 6316) 
v_mul_f32 v12, v18, v6 // (op 6316) 
v_mul_f32 v13, v18, v7 // (op 6316) 
v_and_b32 v6, 65535, v17 // (op 6316) 
v_lshrrev_b32 v7, 16, v17 // (op 6316) 
v_lshlrev_b32 v6, 16, v6 // (op 6316) 
v_lshlrev_b32 v7, 16, v7 // (op 6316) 
v_mul_f32 v14, v18, v6 // (op 6316) 
v_mul_f32 v15, v18, v7 // (op 6316) 


v_add_f32 v4, v12, v8 // (op 6305) 
v_add_f32 v5, v13, v9 // (op 6305) 
v_add_f32 v6, v14, v10 // (op 6305) 
v_add_f32 v7, v15, v11 // (op 6305) 


v_lshrrev_b32 v8, 16, v4 // (op 6306) 
v_lshrrev_b32 v9, 16, v5 // (op 6306) 
v_lshrrev_b32 v10, 16, v6 // (op 6306) 
v_lshrrev_b32 v11, 16, v7 // (op 6306) 


v_pack_b32_f16 v6, v8, v9 // (op 6308) 
v_pack_b32_f16 v7, v10, v11 // (op 6308) 

v_add_u32 v0, 64, v21 // (op 6308) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6308) Store value





s_lshl_b32 s0, s2, 6 // (op 6322) 
s_add_u32 s0, s0, 64 // (op 6322) 
v_add_u32 v0, v20, s0 // (op 6322) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6322) Load value

v_accvgpr_read v12, a20 // (op 6334) 
v_accvgpr_read v13, a21 // (op 6334) 
v_accvgpr_read v14, a22 // (op 6334) 
v_accvgpr_read v15, a23 // (op 6334) 
v_mul_f32 v8, v19, v12 // (op 6334) 
v_mul_f32 v9, v19, v13 // (op 6334) 
v_mul_f32 v10, v19, v14 // (op 6334) 
v_mul_f32 v11, v19, v15 // (op 6334) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6319) 
v_lshrrev_b32 v7, 16, v16 // (op 6319) 
v_lshlrev_b32 v6, 16, v6 // (op 6319) 
v_lshlrev_b32 v7, 16, v7 // (op 6319) 
v_mul_f32 v12, v18, v6 // (op 6319) 
v_mul_f32 v13, v18, v7 // (op 6319) 
v_and_b32 v6, 65535, v17 // (op 6319) 
v_lshrrev_b32 v7, 16, v17 // (op 6319) 
v_lshlrev_b32 v6, 16, v6 // (op 6319) 
v_lshlrev_b32 v7, 16, v7 // (op 6319) 
v_mul_f32 v14, v18, v6 // (op 6319) 
v_mul_f32 v15, v18, v7 // (op 6319) 


v_add_f32 v4, v12, v8 // (op 6320) 
v_add_f32 v5, v13, v9 // (op 6320) 
v_add_f32 v6, v14, v10 // (op 6320) 
v_add_f32 v7, v15, v11 // (op 6320) 


v_lshrrev_b32 v8, 16, v4 // (op 6321) 
v_lshrrev_b32 v9, 16, v5 // (op 6321) 
v_lshrrev_b32 v10, 16, v6 // (op 6321) 
v_lshrrev_b32 v11, 16, v7 // (op 6321) 


v_pack_b32_f16 v6, v8, v9 // (op 6310) 
v_pack_b32_f16 v7, v10, v11 // (op 6310) 

s_lshl_b32 s0, s4, 6 // (op 6310) 
s_add_u32 s0, s0, 64 // (op 6310) 
v_add_u32 v0, v21, s0 // (op 6310) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6310) Store value





s_lshl_b32 s0, s2, 7 // (op 6327) 
s_add_u32 s0, s0, 64 // (op 6327) 
v_add_u32 v0, v20, s0 // (op 6327) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6327) Load value

v_accvgpr_read v12, a24 // (op 6336) 
v_accvgpr_read v13, a25 // (op 6336) 
v_accvgpr_read v14, a26 // (op 6336) 
v_accvgpr_read v15, a27 // (op 6336) 
v_mul_f32 v8, v19, v12 // (op 6336) 
v_mul_f32 v9, v19, v13 // (op 6336) 
v_mul_f32 v10, v19, v14 // (op 6336) 
v_mul_f32 v11, v19, v15 // (op 6336) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6324) 
v_lshrrev_b32 v7, 16, v16 // (op 6324) 
v_lshlrev_b32 v6, 16, v6 // (op 6324) 
v_lshlrev_b32 v7, 16, v7 // (op 6324) 
v_mul_f32 v12, v18, v6 // (op 6324) 
v_mul_f32 v13, v18, v7 // (op 6324) 
v_and_b32 v6, 65535, v17 // (op 6324) 
v_lshrrev_b32 v7, 16, v17 // (op 6324) 
v_lshlrev_b32 v6, 16, v6 // (op 6324) 
v_lshlrev_b32 v7, 16, v7 // (op 6324) 
v_mul_f32 v14, v18, v6 // (op 6324) 
v_mul_f32 v15, v18, v7 // (op 6324) 


v_add_f32 v4, v12, v8 // (op 6325) 
v_add_f32 v5, v13, v9 // (op 6325) 
v_add_f32 v6, v14, v10 // (op 6325) 
v_add_f32 v7, v15, v11 // (op 6325) 


v_lshrrev_b32 v8, 16, v4 // (op 6326) 
v_lshrrev_b32 v9, 16, v5 // (op 6326) 
v_lshrrev_b32 v10, 16, v6 // (op 6326) 
v_lshrrev_b32 v11, 16, v7 // (op 6326) 


v_pack_b32_f16 v6, v8, v9 // (op 6312) 
v_pack_b32_f16 v7, v10, v11 // (op 6312) 

s_lshl_b32 s0, s4, 7 // (op 6312) 
s_add_u32 s0, s0, 64 // (op 6312) 
v_add_u32 v0, v21, s0 // (op 6312) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6312) Store value





s_lshl_b32 s0, s2, 6 // (op 6332) 
s_mul_i32 s1, 3, s0 // (op 6332) 
s_add_u32 s1, s1, 64 // (op 6332) 

v_add_u32 v0, v20, s1 // (op 6332) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6332) Load value

v_accvgpr_read v12, a28 // (op 6338) 
v_accvgpr_read v13, a29 // (op 6338) 
v_accvgpr_read v14, a30 // (op 6338) 
v_accvgpr_read v15, a31 // (op 6338) 
v_mul_f32 v8, v19, v12 // (op 6338) 
v_mul_f32 v9, v19, v13 // (op 6338) 
v_mul_f32 v10, v19, v14 // (op 6338) 
v_mul_f32 v11, v19, v15 // (op 6338) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6329) 
v_lshrrev_b32 v7, 16, v16 // (op 6329) 
v_lshlrev_b32 v6, 16, v6 // (op 6329) 
v_lshlrev_b32 v7, 16, v7 // (op 6329) 
v_mul_f32 v12, v18, v6 // (op 6329) 
v_mul_f32 v13, v18, v7 // (op 6329) 
v_and_b32 v6, 65535, v17 // (op 6329) 
v_lshrrev_b32 v7, 16, v17 // (op 6329) 
v_lshlrev_b32 v6, 16, v6 // (op 6329) 
v_lshlrev_b32 v7, 16, v7 // (op 6329) 
v_mul_f32 v14, v18, v6 // (op 6329) 
v_mul_f32 v15, v18, v7 // (op 6329) 


v_add_f32 v4, v12, v8 // (op 6330) 
v_add_f32 v5, v13, v9 // (op 6330) 
v_add_f32 v6, v14, v10 // (op 6330) 
v_add_f32 v7, v15, v11 // (op 6330) 


v_lshrrev_b32 v8, 16, v4 // (op 6331) 
v_lshrrev_b32 v9, 16, v5 // (op 6331) 
v_lshrrev_b32 v10, 16, v6 // (op 6331) 
v_lshrrev_b32 v11, 16, v7 // (op 6331) 


v_pack_b32_f16 v6, v8, v9 // (op 6314) 
v_pack_b32_f16 v7, v10, v11 // (op 6314) 

s_lshl_b32 s0, s4, 6 // (op 6314) 
s_mul_i32 s1, 3, s0 // (op 6314) 
s_add_u32 s1, s1, 64 // (op 6314) 

v_add_u32 v0, v21, s1 // (op 6314) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6314) Store value





v_mov_b32 v0, v20 // (op 148) reg expression

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v8, v19, v12 // (op 144) 
v_mul_f32 v9, v19, v13 // (op 144) 
v_mul_f32 v10, v19, v14 // (op 144) 
v_mul_f32 v11, v19, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 149) 
v_lshrrev_b32 v7, 16, v16 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v12, v18, v6 // (op 149) 
v_mul_f32 v13, v18, v7 // (op 149) 
v_and_b32 v6, 65535, v17 // (op 149) 
v_lshrrev_b32 v7, 16, v17 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 


v_add_f32 v4, v12, v8 // (op 145) 
v_add_f32 v5, v13, v9 // (op 145) 
v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 


v_lshrrev_b32 v8, 16, v4 // (op 146) 
v_lshrrev_b32 v9, 16, v5 // (op 146) 
v_lshrrev_b32 v10, 16, v6 // (op 146) 
v_lshrrev_b32 v11, 16, v7 // (op 146) 


v_pack_b32_f16 v6, v8, v9 // (op 147) 
v_pack_b32_f16 v7, v10, v11 // (op 147) 

v_mov_b32 v0, v21 // (op 147) reg expression

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 6225) 
v_add_u32 v0, v20, s0 // (op 6225) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6225) Load value

v_accvgpr_read v12, a4 // (op 6221) 
v_accvgpr_read v13, a5 // (op 6221) 
v_accvgpr_read v14, a6 // (op 6221) 
v_accvgpr_read v15, a7 // (op 6221) 
v_mul_f32 v8, v19, v12 // (op 6221) 
v_mul_f32 v9, v19, v13 // (op 6221) 
v_mul_f32 v10, v19, v14 // (op 6221) 
v_mul_f32 v11, v19, v15 // (op 6221) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6226) 
v_lshrrev_b32 v7, 16, v16 // (op 6226) 
v_lshlrev_b32 v6, 16, v6 // (op 6226) 
v_lshlrev_b32 v7, 16, v7 // (op 6226) 
v_mul_f32 v12, v18, v6 // (op 6226) 
v_mul_f32 v13, v18, v7 // (op 6226) 
v_and_b32 v6, 65535, v17 // (op 6226) 
v_lshrrev_b32 v7, 16, v17 // (op 6226) 
v_lshlrev_b32 v6, 16, v6 // (op 6226) 
v_lshlrev_b32 v7, 16, v7 // (op 6226) 
v_mul_f32 v14, v18, v6 // (op 6226) 
v_mul_f32 v15, v18, v7 // (op 6226) 


v_add_f32 v4, v12, v8 // (op 6222) 
v_add_f32 v5, v13, v9 // (op 6222) 
v_add_f32 v6, v14, v10 // (op 6222) 
v_add_f32 v7, v15, v11 // (op 6222) 


v_lshrrev_b32 v8, 16, v4 // (op 6223) 
v_lshrrev_b32 v9, 16, v5 // (op 6223) 
v_lshrrev_b32 v10, 16, v6 // (op 6223) 
v_lshrrev_b32 v11, 16, v7 // (op 6223) 


v_pack_b32_f16 v6, v8, v9 // (op 6224) 
v_pack_b32_f16 v7, v10, v11 // (op 6224) 

s_lshl_b32 s0, s4, 6 // (op 6224) 
v_add_u32 v0, v21, s0 // (op 6224) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6224) Store value





s_lshl_b32 s0, s2, 7 // (op 6239) 
v_add_u32 v0, v20, s0 // (op 6239) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6239) Load value

v_accvgpr_read v12, a8 // (op 6235) 
v_accvgpr_read v13, a9 // (op 6235) 
v_accvgpr_read v14, a10 // (op 6235) 
v_accvgpr_read v15, a11 // (op 6235) 
v_mul_f32 v8, v19, v12 // (op 6235) 
v_mul_f32 v9, v19, v13 // (op 6235) 
v_mul_f32 v10, v19, v14 // (op 6235) 
v_mul_f32 v11, v19, v15 // (op 6235) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6240) 
v_lshrrev_b32 v7, 16, v16 // (op 6240) 
v_lshlrev_b32 v6, 16, v6 // (op 6240) 
v_lshlrev_b32 v7, 16, v7 // (op 6240) 
v_mul_f32 v12, v18, v6 // (op 6240) 
v_mul_f32 v13, v18, v7 // (op 6240) 
v_and_b32 v6, 65535, v17 // (op 6240) 
v_lshrrev_b32 v7, 16, v17 // (op 6240) 
v_lshlrev_b32 v6, 16, v6 // (op 6240) 
v_lshlrev_b32 v7, 16, v7 // (op 6240) 
v_mul_f32 v14, v18, v6 // (op 6240) 
v_mul_f32 v15, v18, v7 // (op 6240) 


v_add_f32 v4, v12, v8 // (op 6236) 
v_add_f32 v5, v13, v9 // (op 6236) 
v_add_f32 v6, v14, v10 // (op 6236) 
v_add_f32 v7, v15, v11 // (op 6236) 


v_lshrrev_b32 v8, 16, v4 // (op 6237) 
v_lshrrev_b32 v9, 16, v5 // (op 6237) 
v_lshrrev_b32 v10, 16, v6 // (op 6237) 
v_lshrrev_b32 v11, 16, v7 // (op 6237) 


v_pack_b32_f16 v6, v8, v9 // (op 6238) 
v_pack_b32_f16 v7, v10, v11 // (op 6238) 

s_lshl_b32 s0, s4, 7 // (op 6238) 
v_add_u32 v0, v21, s0 // (op 6238) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6238) Store value





s_lshl_b32 s0, s2, 6 // (op 6253) 
s_mul_i32 s1, 3, s0 // (op 6253) 

v_add_u32 v0, v20, s1 // (op 6253) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 6253) Load value

v_accvgpr_read v12, a12 // (op 6249) 
v_accvgpr_read v13, a13 // (op 6249) 
v_accvgpr_read v14, a14 // (op 6249) 
v_accvgpr_read v15, a15 // (op 6249) 
v_mul_f32 v8, v19, v12 // (op 6249) 
v_mul_f32 v9, v19, v13 // (op 6249) 
v_mul_f32 v10, v19, v14 // (op 6249) 
v_mul_f32 v11, v19, v15 // (op 6249) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 6254) 
v_lshrrev_b32 v7, 16, v16 // (op 6254) 
v_lshlrev_b32 v6, 16, v6 // (op 6254) 
v_lshlrev_b32 v7, 16, v7 // (op 6254) 
v_mul_f32 v12, v18, v6 // (op 6254) 
v_mul_f32 v13, v18, v7 // (op 6254) 
v_and_b32 v6, 65535, v17 // (op 6254) 
v_lshrrev_b32 v7, 16, v17 // (op 6254) 
v_lshlrev_b32 v6, 16, v6 // (op 6254) 
v_lshlrev_b32 v7, 16, v7 // (op 6254) 
v_mul_f32 v14, v18, v6 // (op 6254) 
v_mul_f32 v15, v18, v7 // (op 6254) 


v_add_f32 v4, v12, v8 // (op 6250) 
v_add_f32 v5, v13, v9 // (op 6250) 
v_add_f32 v6, v14, v10 // (op 6250) 
v_add_f32 v7, v15, v11 // (op 6250) 


v_lshrrev_b32 v8, 16, v4 // (op 6251) 
v_lshrrev_b32 v9, 16, v5 // (op 6251) 
v_lshrrev_b32 v10, 16, v6 // (op 6251) 
v_lshrrev_b32 v11, 16, v7 // (op 6251) 


v_pack_b32_f16 v6, v8, v9 // (op 6252) 
v_pack_b32_f16 v7, v10, v11 // (op 6252) 

s_lshl_b32 s0, s4, 6 // (op 6252) 
s_mul_i32 s1, 3, s0 // (op 6252) 

v_add_u32 v0, v21, s1 // (op 6252) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 6252) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.LRR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end:
.size RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_, .LRR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end-RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 188
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
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
    DataTypeB: F4
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
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
    .symbol: RR_GEMM_TT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 79
    .vgpr_count: 124
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

