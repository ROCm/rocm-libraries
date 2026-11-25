// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_,@function
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 13649) 
s_mul_hi_i32 s1, s16, s11 // (op 13904) 
v_lshrrev_b32 v0, 7, v1 // (op 13710) 
v_and_b32 v3, 15, v1 // (op 13710) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 13710) 


v_and_b32 v0, 63, v1 // (op 13710) 
v_lshrrev_b32 v3, 4, v0 // (op 13710) 

v_lshlrev_b32 v0, 5, v3 // (op 13710) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 13710) 


v_lshlrev_b32 v0, 2, v3 // (op 13710) 

v_lshrrev_b32 v3, 3, v0 // (op 13710) 

v_and_b32 v0, 63, v1 // (op 13767) 
v_and_b32 v4, 48, v1 // (op 13767) 
v_sub_u32 v5, v0, v4 // (op 13767) 

v_lshrrev_b32 v4, 4, v0 // (op 13767) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 13767) 


v_lshrrev_b32 v4, 6, v1 // (op 13767) 
v_and_b32 v5, 1, v4 // (op 13767) 

v_lshlrev_b32 v4, 4, v5 // (op 13767) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 13767) 


v_lshlrev_b32 v0, 2, v5 // (op 13767) 

v_lshrrev_b32 v4, 3, v0 // (op 13767) 

v_lshrrev_b32 v0, 7, v1 // (op 13825) 
v_and_b32 v5, 15, v1 // (op 13825) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 13825) 


v_and_b32 v0, 63, v1 // (op 13825) 
v_lshrrev_b32 v5, 4, v0 // (op 13825) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 13825) 


v_lshrrev_b32 v5, 6, v1 // (op 13883) 
v_and_b32 v6, 1, v5 // (op 13883) 

v_and_b32 v5, 15, v1 // (op 13883) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 13883) 


v_and_b32 v5, 63, v1 // (op 13883) 
v_lshrrev_b32 v6, 4, v5 // (op 13883) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 13883) 


v_lshrrev_b32 v6, 7, v1 // (op 13950) 
v_and_b32 v7, 15, v1 // (op 13950) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 13950) 


v_and_b32 v6, 63, v1 // (op 13950) 
v_lshrrev_b32 v7, 4, v6 // (op 13950) 

v_lshlrev_b32 v6, 5, v7 // (op 13950) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 13950) 


v_lshlrev_b32 v6, 2, v7 // (op 13950) 

v_lshrrev_b32 v7, 3, v6 // (op 13950) 

v_and_b32 v6, 63, v1 // (op 13994) 
v_and_b32 v8, 48, v1 // (op 13994) 
v_sub_u32 v9, v6, v8 // (op 13994) 

v_lshrrev_b32 v8, 4, v6 // (op 13994) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 13994) 


v_lshrrev_b32 v8, 6, v1 // (op 13994) 
v_and_b32 v9, 1, v8 // (op 13994) 

v_lshlrev_b32 v8, 4, v9 // (op 13994) 

v_lshl_add_u32 v9, v6, 7, v8 // (op 13994) 


v_lshlrev_b32 v6, 2, v9 // (op 13994) 

v_lshrrev_b32 v8, 3, v6 // (op 13994) 

v_lshrrev_b32 v6, 7, v1 // (op 14023) 
v_and_b32 v9, 15, v1 // (op 14023) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 14023) 


v_and_b32 v6, 63, v1 // (op 14023) 
v_lshrrev_b32 v9, 4, v6 // (op 14023) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 14023) 


v_lshrrev_b32 v9, 6, v1 // (op 14052) 
v_and_b32 v10, 1, v9 // (op 14052) 

v_and_b32 v9, 15, v1 // (op 14052) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 14052) 


v_and_b32 v9, 63, v1 // (op 14052) 
v_lshrrev_b32 v10, 4, v9 // (op 14052) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 14052) 


v_lshrrev_b32 v10, 7, v1 // (op 14096) 
v_and_b32 v11, 15, v1 // (op 14096) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 14096) 


v_and_b32 v10, 63, v1 // (op 14096) 
v_lshrrev_b32 v11, 4, v10 // (op 14096) 

v_lshlrev_b32 v10, 5, v11 // (op 14096) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 14096) 


v_lshlrev_b32 v10, 2, v11 // (op 14096) 

v_lshrrev_b32 v11, 3, v10 // (op 14096) 

v_and_b32 v10, 63, v1 // (op 14140) 
v_and_b32 v12, 48, v1 // (op 14140) 
v_sub_u32 v13, v10, v12 // (op 14140) 

v_lshrrev_b32 v12, 4, v10 // (op 14140) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 14140) 


v_lshrrev_b32 v12, 6, v1 // (op 14140) 
v_and_b32 v13, 1, v12 // (op 14140) 

v_lshlrev_b32 v12, 4, v13 // (op 14140) 

v_lshl_add_u32 v13, v10, 7, v12 // (op 14140) 


v_lshlrev_b32 v10, 2, v13 // (op 14140) 

v_lshrrev_b32 v12, 3, v10 // (op 14140) 

v_lshrrev_b32 v10, 7, v1 // (op 14169) 
v_and_b32 v13, 15, v1 // (op 14169) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 14169) 


v_and_b32 v10, 63, v1 // (op 14169) 
v_lshrrev_b32 v13, 4, v10 // (op 14169) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 14169) 


v_lshrrev_b32 v13, 6, v1 // (op 14198) 
v_and_b32 v14, 1, v13 // (op 14198) 

v_and_b32 v13, 15, v1 // (op 14198) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 14198) 


v_and_b32 v13, 63, v1 // (op 14198) 
v_lshrrev_b32 v14, 4, v13 // (op 14198) 

v_lshl_add_u32 v13, v14, 7, v15 // (op 14198) 


v_lshrrev_b32 v14, 7, v1 // (op 14242) 
v_and_b32 v15, 15, v1 // (op 14242) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 14242) 


v_and_b32 v14, 63, v1 // (op 14242) 
v_lshrrev_b32 v15, 4, v14 // (op 14242) 

v_lshlrev_b32 v14, 5, v15 // (op 14242) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 14242) 


v_lshlrev_b32 v14, 2, v15 // (op 14242) 

v_lshrrev_b32 v15, 3, v14 // (op 14242) 

v_and_b32 v14, 63, v1 // (op 14286) 
v_and_b32 v16, 48, v1 // (op 14286) 
v_sub_u32 v17, v14, v16 // (op 14286) 

v_lshrrev_b32 v16, 4, v14 // (op 14286) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 14286) 


v_lshrrev_b32 v16, 6, v1 // (op 14286) 
v_and_b32 v17, 1, v16 // (op 14286) 

v_lshlrev_b32 v16, 4, v17 // (op 14286) 

v_lshl_add_u32 v17, v14, 7, v16 // (op 14286) 


v_lshlrev_b32 v14, 2, v17 // (op 14286) 

v_lshrrev_b32 v16, 3, v14 // (op 14286) 

v_lshrrev_b32 v14, 7, v1 // (op 14315) 
v_and_b32 v17, 15, v1 // (op 14315) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 14315) 


v_and_b32 v14, 63, v1 // (op 14315) 
v_lshrrev_b32 v17, 4, v14 // (op 14315) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 14315) 


v_lshrrev_b32 v17, 6, v1 // (op 14344) 
v_and_b32 v18, 1, v17 // (op 14344) 

v_and_b32 v17, 15, v1 // (op 14344) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 14344) 


v_and_b32 v17, 63, v1 // (op 14344) 
v_lshrrev_b32 v18, 4, v17 // (op 14344) 

v_lshl_add_u32 v17, v18, 7, v19 // (op 14344) 


v_lshrrev_b32 v18, 7, v1 // (op 13649) 
s_add_i32 s18, s0, s16 // (op 13649) 

v_and_b32 v19, 63, v1 // (op 13649) 
s_ashr_i32 s0, s18, 31 // (op 13649) 
v_lshrrev_b32 v20, 4, v19 // (op 13649) 
s_cmp_eq_i32 s11, 0 // (op 13649) 
s_mov_b32 s19, scc // (op 13649) 
v_lshlrev_b32 v21, 2, v20 // (op 13649) 

s_cmp_eq_i32 s19, 0 // (op 13649) 
s_cselect_b32 s51, 0, -1 // (op 13649) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 13649) 


s_and_b32 s19, s12, 31 // (op 13649) 
v_lshrrev_b32 v18, 2, v19 // (op 13649) 

v_and_b32 v19, 3, v18 // (op 13649) 

v_lshrrev_b32 v18, 7, v1 // (op 13904) 
v_and_b32 v21, 63, v1 // (op 13904) 
s_lshl_b32 s52, 1, s19 // (op 13649) 
v_and_b32 v22, 3, v1 // (op 13649) 
s_add_i32 s53, s52, s51 // (op 13649) 


s_and_b32 s51, s0, s53 // (op 13649) 


s_add_i32 s0, s1, s16 // (op 13904) 

s_ashr_i32 s1, s0, 31 // (op 13904) 
v_lshl_add_u32 v23, v19, 2, v22 // (op 13649) 


s_add_i32 s52, s18, s51 // (op 13649) 


v_lshrrev_b32 v19, 6, v1 // (op 13649) 
s_ashr_i32 s18, s52, s19 // (op 13649) 


v_and_b32 v22, 1, v19 // (op 13649) 

s_lshl_b32 s19, s12, 24 // (op 13649) 
v_lshl_add_u32 v19, v22, 4, v23 // (op 13649) 


s_ashr_i32 s51, s19, 31 // (op 13649) 

s_xor_b32 s19, s18, s51 // (op 13649) 

s_sub_i32 s18, s19, s51 // (op 13649) 


s_mul_hi_i32 s19, s18, s13 // (op 13649) 
s_add_i32 s51, s19, s18 // (op 13649) 

s_ashr_i32 s19, s51, 31 // (op 13649) 
s_cmp_eq_i32 s13, 0 // (op 13649) 
s_mov_b32 s52, scc // (op 13649) 
s_cmp_eq_i32 s52, 0 // (op 13649) 
s_cselect_b32 s53, 0, -1 // (op 13649) 

s_and_b32 s52, s14, 31 // (op 13649) 
v_lshrrev_b32 v22, 4, v21 // (op 13904) 
s_cmp_eq_i32 s11, 0 // (op 13904) 
s_mov_b32 s54, scc // (op 13904) 
v_lshlrev_b32 v23, 2, v22 // (op 13904) 

s_cmp_eq_i32 s54, 0 // (op 13904) 
s_cselect_b32 s55, 0, -1 // (op 13904) 

s_lshl_b32 s54, 1, s52 // (op 13649) 
v_lshl_add_u32 v22, v18, 4, v23 // (op 13904) 


v_lshrrev_b32 v18, 2, v21 // (op 13904) 

s_add_i32 s56, s54, s53 // (op 13649) 


s_and_b32 s53, s19, s56 // (op 13649) 


s_add_i32 s19, s51, s53 // (op 13649) 


s_ashr_i32 s51, s19, s52 // (op 13649) 


s_lshl_b32 s19, s14, 24 // (op 13649) 
s_ashr_i32 s52, s19, 31 // (op 13649) 

s_xor_b32 s19, s51, s52 // (op 13649) 

s_sub_i32 s51, s19, s52 // (op 13649) 


s_and_b32 s19, s12, 31 // (op 13904) 
s_lshl_b32 s52, 1, s19 // (op 13904) 
v_and_b32 v21, 3, v18 // (op 13904) 

s_add_i32 s53, s52, s55 // (op 13904) 


v_and_b32 v18, 3, v1 // (op 13904) 
s_and_b32 s52, s1, s53 // (op 13904) 


v_lshl_add_u32 v23, v21, 2, v18 // (op 13904) 


s_add_i32 s1, s0, s52 // (op 13904) 


v_lshrrev_b32 v18, 6, v1 // (op 13904) 
s_ashr_i32 s0, s1, s19 // (op 13904) 


v_and_b32 v21, 1, v18 // (op 13904) 

s_lshl_b32 s1, s12, 24 // (op 13904) 
v_lshl_add_u32 v18, v21, 4, v23 // (op 13904) 


s_ashr_i32 s19, s1, 31 // (op 13904) 

s_xor_b32 s1, s0, s19 // (op 13904) 

s_sub_i32 s0, s1, s19 // (op 13904) 


s_mul_hi_i32 s1, s0, s13 // (op 13904) 
s_add_i32 s19, s1, s0 // (op 13904) 

s_ashr_i32 s1, s19, 31 // (op 13904) 
s_cmp_eq_i32 s13, 0 // (op 13904) 
s_mov_b32 s52, scc // (op 13904) 
s_cmp_eq_i32 s52, 0 // (op 13904) 
s_cselect_b32 s53, 0, -1 // (op 13904) 

s_and_b32 s52, s14, 31 // (op 13904) 
s_lshl_b32 s54, 1, s52 // (op 13904) 
s_add_i32 s55, s54, s53 // (op 13904) 


s_and_b32 s53, s1, s55 // (op 13904) 


s_add_i32 s1, s19, s53 // (op 13904) 


s_ashr_i32 s19, s1, s52 // (op 13904) 


s_lshl_b32 s1, s14, 24 // (op 13904) 
s_ashr_i32 s52, s1, 31 // (op 13904) 

s_xor_b32 s1, s19, s52 // (op 13904) 

s_sub_i32 s19, s1, s52 // (op 13904) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 13649) 
s_sub_u32 s52, s16, s1 // (op 13649) 

s_mul_i32 s1, s51, s44 // (op 13649) 
s_add_u32 s1, s1, s52 // (op 13649) 

s_mul_hi_i32 s52, s16, s10 // (op 13649) 
s_add_i32 s53, s52, s16 // (op 13649) 

s_ashr_i32 s52, s53, 31 // (op 13649) 
s_cmp_eq_i32 s10, 0 // (op 13649) 
s_mov_b32 s54, scc // (op 13649) 
s_cmp_eq_i32 s54, 0 // (op 13649) 
s_cselect_b32 s55, 0, -1 // (op 13649) 

s_and_b32 s54, s47, 31 // (op 13649) 
s_lshl_b32 s56, 1, s54 // (op 13649) 
s_add_i32 s57, s56, s55 // (op 13649) 


s_and_b32 s55, s52, s57 // (op 13649) 


s_add_i32 s52, s53, s55 // (op 13649) 


s_ashr_i32 s53, s52, s54 // (op 13649) 


s_lshl_b32 s52, s47, 24 // (op 13649) 
s_ashr_i32 s54, s52, 31 // (op 13649) 

s_xor_b32 s52, s53, s54 // (op 13649) 

s_sub_i32 s53, s52, s54 // (op 13649) 


s_mul_i32 s52, s0, s44 // (op 13904) 
s_sub_u32 s54, s16, s52 // (op 13904) 

s_mul_i32 s52, s19, s44 // (op 13904) 
s_add_u32 s52, s52, s54 // (op 13904) 

s_mul_hi_i32 s54, s16, s10 // (op 13904) 
s_add_i32 s55, s54, s16 // (op 13904) 

s_ashr_i32 s54, s55, 31 // (op 13904) 
s_cmp_eq_i32 s10, 0 // (op 13904) 
s_mov_b32 s56, scc // (op 13904) 
s_cmp_eq_i32 s56, 0 // (op 13904) 
s_cselect_b32 s57, 0, -1 // (op 13904) 

s_and_b32 s56, s47, 31 // (op 13904) 
s_lshl_b32 s58, 1, s56 // (op 13904) 
s_add_i32 s59, s58, s57 // (op 13904) 


s_and_b32 s57, s54, s59 // (op 13904) 


s_add_i32 s54, s55, s57 // (op 13904) 


s_ashr_i32 s55, s54, s56 // (op 13904) 


s_lshl_b32 s54, s47, 24 // (op 13904) 
s_ashr_i32 s56, s54, 31 // (op 13904) 

s_xor_b32 s54, s55, s56 // (op 13904) 

s_sub_i32 s55, s54, s56 // (op 13904) 


s_add_u32 s56, s6, 128 // (op 13649) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13649) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13649) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13649) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13649) 

s_mul_i32 s54, s44, s56 // (op 13649) 
s_mul_i32 s58, s48, s54 // (op 13649) 

s_mul_i32 s54, s53, s58 // (op 13649) 

s_sub_u32 s58, s16, s54 // (op 13649) 

s_mov_b32 s54, s58 // (op 13649) 
s_mul_hi_i32 s59, s54, s15 // (op 13649) 
s_add_i32 s60, s59, s54 // (op 13649) 


s_ashr_i32 s54, s60, 31 // (op 13649) 
s_cmp_eq_i32 s15, 0 // (op 13649) 
s_mov_b32 s59, scc // (op 13649) 
s_cmp_eq_i32 s59, 0 // (op 13649) 
s_cselect_b32 s61, 0, -1 // (op 13649) 

s_and_b32 s59, s46, 31 // (op 13649) 
s_lshl_b32 s62, 1, s59 // (op 13649) 
s_add_i32 s63, s62, s61 // (op 13649) 


s_and_b32 s61, s54, s63 // (op 13649) 


s_add_i32 s54, s60, s61 // (op 13649) 


s_ashr_i32 s60, s54, s59 // (op 13649) 


s_lshl_b32 s54, s46, 24 // (op 13649) 
s_ashr_i32 s59, s54, 31 // (op 13649) 

s_xor_b32 s54, s60, s59 // (op 13649) 

s_sub_i32 s60, s54, s59 // (op 13649) 


s_mul_i32 s54, s45, s44 // (op 13649) 
s_add_u32 s62, s8, 128 // (op 13649) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 13649) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 13649) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 13649) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 13649) 

s_sub_i32 s59, s62, s54 // (op 13649) 

s_mul_i32 s61, s60, s59 // (op 13649) 

s_sub_u32 s59, s58, s61 // (op 13649) 


s_add_u32 s58, s59, s54 // (op 13649) 


s_cmp_eq_i32 s53, 0 // (op 13649) 
s_mov_b32 s54, scc // (op 13649) 

s_cmp_eq_i32 s54, 0 // (op 13649) 
s_cselect_b32 s53, s58, s1 // (op 13649) 


s_mul_i32 s1, s51, s56 // (op 13649) 


v_lshl_add_u32 v21, s53, 7, v20 // (op 13649) 


s_sub_i32 s51, s18, s1 // (op 13649) 


s_cmp_eq_i32 s54, 0 // (op 13649) 
s_cselect_b32 s1, s60, s51 // (op 13649) 



v_lshl_add_u32 v20, s1, 7, v19 // (op 13649) 


v_mul_lo_u32 v19, v20, s2 // (op 13649) 

v_add_lshl_u32 v20, v21, v19, 2 // (op 13649) 


s_add_u32 s56, s6, 128 // (op 13904) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13904) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13904) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13904) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13904) 

s_mul_i32 s1, s44, s56 // (op 13904) 
s_mul_i32 s18, s48, s1 // (op 13904) 

s_mul_i32 s1, s55, s18 // (op 13904) 

s_sub_u32 s18, s16, s1 // (op 13904) 

s_mov_b32 s1, s18 // (op 13904) 
s_mul_hi_i32 s51, s1, s15 // (op 13904) 
s_add_i32 s53, s51, s1 // (op 13904) 


s_ashr_i32 s1, s53, 31 // (op 13904) 
s_cmp_eq_i32 s15, 0 // (op 13904) 
s_mov_b32 s51, scc // (op 13904) 
s_cmp_eq_i32 s51, 0 // (op 13904) 
s_cselect_b32 s54, 0, -1 // (op 13904) 

s_and_b32 s51, s46, 31 // (op 13904) 
s_lshl_b32 s58, 1, s51 // (op 13904) 
s_add_i32 s59, s58, s54 // (op 13904) 


s_and_b32 s54, s1, s59 // (op 13904) 


s_add_i32 s1, s53, s54 // (op 13904) 


s_ashr_i32 s53, s1, s51 // (op 13904) 


s_lshl_b32 s1, s46, 24 // (op 13904) 
s_ashr_i32 s51, s1, 31 // (op 13904) 

s_xor_b32 s1, s53, s51 // (op 13904) 

s_sub_i32 s53, s1, s51 // (op 13904) 


s_mul_i32 s1, s45, s44 // (op 13904) 
s_add_u32 s58, s8, 128 // (op 13904) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 13904) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 13904) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 13904) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 13904) 

s_sub_i32 s51, s58, s1 // (op 13904) 

s_mul_i32 s54, s53, s51 // (op 13904) 

s_sub_u32 s51, s18, s54 // (op 13904) 


s_add_u32 s18, s51, s1 // (op 13904) 


s_cmp_eq_i32 s55, 0 // (op 13904) 
s_mov_b32 s1, scc // (op 13904) 

s_cmp_eq_i32 s1, 0 // (op 13904) 
s_cselect_b32 s51, s18, s52 // (op 13904) 


s_mul_i32 s18, s19, s56 // (op 13904) 


v_lshl_add_u32 v19, s51, 7, v22 // (op 13904) 


s_sub_i32 s19, s0, s18 // (op 13904) 


s_cmp_eq_i32 s1, 0 // (op 13904) 
s_cselect_b32 s0, s53, s19 // (op 13904) 



v_lshl_add_u32 v21, s0, 7, v18 // (op 13904) 


v_mul_lo_u32 v18, v21, s4 // (op 13904) 

v_add_lshl_u32 v21, v19, v18, 2 // (op 13904) 


s_mul_i32 s1, s2, s7 // (op 13651) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13651) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13651) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13651) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13651) most significant: sum
s_add_u32 s1, s1, s19 // (op 13651) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13651) 

s_mul_i32 s1, s4, s7 // (op 13906) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 13906) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13906) least significant
s_mul_hi_u32 s52, s4, s6 // (op 13906) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 13906) most significant: sum
s_add_u32 s1, s1, s52 // (op 13906) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 13906) 

s_mov_b32 s56, s20 // (op 13651) reg expression
s_mov_b32 s57, s21 // (op 13651) reg expression
s_mov_b32 s58, s18 // (op 13651) reg expression
s_mov_b32 s59, 131072 // (op 13651) call()

s_mov_b32 s60, s22 // (op 13906) reg expression

s_mov_b32 s61, s23 // (op 13906) reg expression
s_mov_b32 s62, s52 // (op 13906) reg expression
s_mov_b32 s63, 131072 // (op 13906) call()


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
v_accvgpr_write a8, 0.00000 // (op 780) call()
v_accvgpr_write a9, 0.00000 // (op 780) call()
v_accvgpr_write a10, 0.00000 // (op 780) call()
v_accvgpr_write a11, 0.00000 // (op 780) call()
v_accvgpr_write a12, 0.00000 // (op 1078) call()
v_accvgpr_write a13, 0.00000 // (op 1078) call()
v_accvgpr_write a14, 0.00000 // (op 1078) call()
v_accvgpr_write a15, 0.00000 // (op 1078) call()
v_accvgpr_write a16, 0.00000 // (op 1512) call()
v_accvgpr_write a17, 0.00000 // (op 1512) call()
v_accvgpr_write a18, 0.00000 // (op 1512) call()
v_accvgpr_write a19, 0.00000 // (op 1512) call()
v_accvgpr_write a20, 0.00000 // (op 2005) call()
v_accvgpr_write a21, 0.00000 // (op 2005) call()
v_accvgpr_write a22, 0.00000 // (op 2005) call()
v_accvgpr_write a23, 0.00000 // (op 2005) call()
v_accvgpr_write a24, 0.00000 // (op 2006) call()
v_accvgpr_write a25, 0.00000 // (op 2006) call()
v_accvgpr_write a26, 0.00000 // (op 2006) call()
v_accvgpr_write a27, 0.00000 // (op 2006) call()
v_accvgpr_write a28, 0.00000 // (op 2007) call()
v_accvgpr_write a29, 0.00000 // (op 2007) call()
v_accvgpr_write a30, 0.00000 // (op 2007) call()
v_accvgpr_write a31, 0.00000 // (op 2007) call()
v_accvgpr_write a32, 0.00000 // (op 2802) call()
v_accvgpr_write a33, 0.00000 // (op 2802) call()
v_accvgpr_write a34, 0.00000 // (op 2802) call()
v_accvgpr_write a35, 0.00000 // (op 2802) call()
v_accvgpr_write a36, 0.00000 // (op 3295) call()
v_accvgpr_write a37, 0.00000 // (op 3295) call()
v_accvgpr_write a38, 0.00000 // (op 3295) call()
v_accvgpr_write a39, 0.00000 // (op 3295) call()
v_accvgpr_write a40, 0.00000 // (op 3296) call()
v_accvgpr_write a41, 0.00000 // (op 3296) call()
v_accvgpr_write a42, 0.00000 // (op 3296) call()
v_accvgpr_write a43, 0.00000 // (op 3296) call()
v_accvgpr_write a44, 0.00000 // (op 3297) call()
v_accvgpr_write a45, 0.00000 // (op 3297) call()
v_accvgpr_write a46, 0.00000 // (op 3297) call()
v_accvgpr_write a47, 0.00000 // (op 3297) call()
v_accvgpr_write a48, 0.00000 // (op 4092) call()
v_accvgpr_write a49, 0.00000 // (op 4092) call()
v_accvgpr_write a50, 0.00000 // (op 4092) call()
v_accvgpr_write a51, 0.00000 // (op 4092) call()
v_accvgpr_write a52, 0.00000 // (op 4585) call()
v_accvgpr_write a53, 0.00000 // (op 4585) call()
v_accvgpr_write a54, 0.00000 // (op 4585) call()
v_accvgpr_write a55, 0.00000 // (op 4585) call()
v_accvgpr_write a56, 0.00000 // (op 4586) call()
v_accvgpr_write a57, 0.00000 // (op 4586) call()
v_accvgpr_write a58, 0.00000 // (op 4586) call()
v_accvgpr_write a59, 0.00000 // (op 4586) call()
v_accvgpr_write a60, 0.00000 // (op 4587) call()
v_accvgpr_write a61, 0.00000 // (op 4587) call()
v_accvgpr_write a62, 0.00000 // (op 4587) call()
v_accvgpr_write a63, 0.00000 // (op 4587) call()
s_mul_hi_i32 s0, s16, s11 // (op 13411) 
v_lshrrev_b32 v22, 2, v1 // (op 13411) 
s_add_i32 s1, s0, s16 // (op 13411) 

v_and_b32 v23, -4, v1 // (op 13411) 
s_ashr_i32 s0, s1, 31 // (op 13411) 
v_sub_u32 v24, v1, v23 // (op 13411) 

s_cmp_eq_i32 s11, 0 // (op 13411) 
s_mov_b32 s18, scc // (op 13411) 
v_lshlrev_b32 v23, 5, v24 // (op 13411) 

s_cmp_eq_i32 s18, 0 // (op 13411) 
s_cselect_b32 s19, 0, -1 // (op 13411) 

v_mov_b32 v25, 0 // (op 13411) convert
v_mov_b32 v24, v23 // (op 13411) convert

s_and_b32 s18, s12, 31 // (op 13411) 
s_lshl_b32 s20, 1, s18 // (op 13411) 
s_add_i32 s21, s20, s19 // (op 13411) 


s_and_b32 s19, s0, s21 // (op 13411) 


s_add_i32 s0, s1, s19 // (op 13411) 


s_ashr_i32 s1, s0, s18 // (op 13411) 


s_lshl_b32 s0, s12, 24 // (op 13411) 
s_ashr_i32 s18, s0, 31 // (op 13411) 

s_xor_b32 s0, s1, s18 // (op 13411) 

s_sub_i32 s1, s0, s18 // (op 13411) 


s_mul_hi_i32 s0, s1, s13 // (op 13411) 
s_add_i32 s18, s0, s1 // (op 13411) 

s_ashr_i32 s0, s18, 31 // (op 13411) 
s_cmp_eq_i32 s13, 0 // (op 13411) 
s_mov_b32 s19, scc // (op 13411) 
s_cmp_eq_i32 s19, 0 // (op 13411) 
s_cselect_b32 s20, 0, -1 // (op 13411) 

s_and_b32 s19, s14, 31 // (op 13411) 
s_lshl_b32 s21, 1, s19 // (op 13411) 
s_add_i32 s22, s21, s20 // (op 13411) 


s_and_b32 s23, s0, s22 // (op 13411) 


s_add_i32 s0, s18, s23 // (op 13411) 


s_ashr_i32 s18, s0, s19 // (op 13411) 


s_lshl_b32 s0, s14, 24 // (op 13411) 
s_ashr_i32 s19, s0, 31 // (op 13411) 

s_xor_b32 s0, s18, s19 // (op 13411) 

s_sub_i32 s18, s0, s19 // (op 13411) 


s_mul_i32 s0, s1, s44 // (op 13411) 

s_sub_u32 s1, s16, s0 // (op 13411) 

s_mul_i32 s0, s18, s44 // (op 13411) 
s_add_u32 s0, s0, s1 // (op 13411) 


s_mul_hi_i32 s1, s16, s10 // (op 13411) 
s_add_i32 s18, s1, s16 // (op 13411) 

s_ashr_i32 s1, s18, 31 // (op 13411) 
s_cmp_eq_i32 s10, 0 // (op 13411) 
s_mov_b32 s19, scc // (op 13411) 
s_cmp_eq_i32 s19, 0 // (op 13411) 
s_cselect_b32 s20, 0, -1 // (op 13411) 

s_and_b32 s19, s47, 31 // (op 13411) 
s_lshl_b32 s21, 1, s19 // (op 13411) 
s_add_i32 s22, s21, s20 // (op 13411) 


s_and_b32 s23, s1, s22 // (op 13411) 


s_add_i32 s1, s18, s23 // (op 13411) 


s_ashr_i32 s18, s1, s19 // (op 13411) 


s_lshl_b32 s1, s47, 24 // (op 13411) 
s_ashr_i32 s19, s1, 31 // (op 13411) 

s_xor_b32 s1, s18, s19 // (op 13411) 

s_sub_i32 s18, s1, s19 // (op 13411) 


s_add_u32 s22, s6, 128 // (op 13411) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13411) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 13411) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 13411) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 13411) 

s_mul_i32 s1, s44, s22 // (op 13411) 

s_mul_i32 s19, s48, s1 // (op 13411) 

s_mul_i32 s1, s18, s19 // (op 13411) 

s_sub_u32 s19, s16, s1 // (op 13411) 

s_mov_b32 s1, s19 // (op 13411) 
s_mul_hi_i32 s20, s1, s15 // (op 13411) 
s_add_i32 s21, s20, s1 // (op 13411) 


s_ashr_i32 s1, s21, 31 // (op 13411) 
s_cmp_eq_i32 s15, 0 // (op 13411) 
s_mov_b32 s20, scc // (op 13411) 
s_cmp_eq_i32 s20, 0 // (op 13411) 
s_cselect_b32 s22, 0, -1 // (op 13411) 

s_and_b32 s20, s46, 31 // (op 13411) 
s_lshl_b32 s23, 1, s20 // (op 13411) 
s_add_i32 s49, s23, s22 // (op 13411) 


s_and_b32 s22, s1, s49 // (op 13411) 


s_add_i32 s1, s21, s22 // (op 13411) 


s_ashr_i32 s21, s1, s20 // (op 13411) 


s_lshl_b32 s1, s46, 24 // (op 13411) 
s_ashr_i32 s20, s1, 31 // (op 13411) 

s_xor_b32 s1, s21, s20 // (op 13411) 

s_sub_i32 s21, s1, s20 // (op 13411) 


s_mul_i32 s1, s45, s44 // (op 13411) 
s_add_u32 s22, s8, 128 // (op 13411) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 13411) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 13411) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 13411) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 7 // (op 13411) 

s_sub_i32 s20, s22, s1 // (op 13411) 

s_mul_i32 s22, s21, s20 // (op 13411) 


s_sub_u32 s23, s19, s22 // (op 13411) 


s_add_u32 s19, s23, s1 // (op 13411) 


s_cmp_eq_i32 s18, 0 // (op 13411) 
s_mov_b32 s1, scc // (op 13411) 

s_cmp_eq_i32 s1, 0 // (op 13411) 
s_cselect_b32 s18, s19, s0 // (op 13411) 



v_lshl_add_u32 v23, s18, 7, v22 // (op 13411) 


v_mul_lo_u32 v27, v23, s25 // (op 13411) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13411) most significant: high of low * low
v_mul_lo_u32 v26, v23, s24 // (op 13411) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 13411) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 13411) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 13411) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 13411) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 13411) 

s_mul_i32 s1, s24, s9 // (op 13413) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13413) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13413) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13413) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13413) most significant: sum
s_add_u32 s1, s1, s19 // (op 13413) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13413) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13413) 

s_mov_b32 s20, s26 // (op 13413) reg expression
s_mov_b32 s21, s27 // (op 13413) reg expression
s_mov_b32 s22, s0 // (op 13413) reg expression
s_mov_b32 s23, 131072 // (op 13413) call()

s_mul_hi_i32 s0, s16, s11 // (op 13430) 
v_lshrrev_b32 v22, 2, v1 // (op 13430) 
s_add_i32 s1, s0, s16 // (op 13430) 

v_and_b32 v23, -4, v1 // (op 13430) 
s_ashr_i32 s0, s1, 31 // (op 13430) 
v_sub_u32 v26, v1, v23 // (op 13430) 

s_cmp_eq_i32 s11, 0 // (op 13430) 
s_mov_b32 s18, scc // (op 13430) 
v_lshlrev_b32 v23, 5, v26 // (op 13430) 

s_cmp_eq_i32 s18, 0 // (op 13430) 
s_cselect_b32 s19, 0, -1 // (op 13430) 

v_mov_b32 v27, 0 // (op 13430) convert
v_mov_b32 v26, v23 // (op 13430) convert

s_and_b32 s18, s12, 31 // (op 13430) 
s_lshl_b32 s49, 1, s18 // (op 13430) 
s_add_i32 s50, s49, s19 // (op 13430) 


s_and_b32 s19, s0, s50 // (op 13430) 


s_add_i32 s0, s1, s19 // (op 13430) 


s_ashr_i32 s1, s0, s18 // (op 13430) 


s_lshl_b32 s0, s12, 24 // (op 13430) 
s_ashr_i32 s18, s0, 31 // (op 13430) 

s_xor_b32 s0, s1, s18 // (op 13430) 

s_sub_i32 s1, s0, s18 // (op 13430) 


s_mul_hi_i32 s0, s1, s13 // (op 13430) 
s_add_i32 s18, s0, s1 // (op 13430) 

s_ashr_i32 s0, s18, 31 // (op 13430) 
s_cmp_eq_i32 s13, 0 // (op 13430) 
s_mov_b32 s19, scc // (op 13430) 
s_cmp_eq_i32 s19, 0 // (op 13430) 
s_cselect_b32 s49, 0, -1 // (op 13430) 

s_and_b32 s19, s14, 31 // (op 13430) 
s_lshl_b32 s50, 1, s19 // (op 13430) 
s_add_i32 s51, s50, s49 // (op 13430) 


s_and_b32 s49, s0, s51 // (op 13430) 


s_add_i32 s0, s18, s49 // (op 13430) 


s_ashr_i32 s18, s0, s19 // (op 13430) 


s_lshl_b32 s0, s14, 24 // (op 13430) 
s_ashr_i32 s19, s0, 31 // (op 13430) 

s_xor_b32 s0, s18, s19 // (op 13430) 

s_sub_i32 s18, s0, s19 // (op 13430) 


s_mul_i32 s0, s1, s44 // (op 13430) 

s_sub_u32 s1, s16, s0 // (op 13430) 

s_mul_i32 s0, s18, s44 // (op 13430) 
s_add_u32 s0, s0, s1 // (op 13430) 


s_mul_hi_i32 s1, s16, s10 // (op 13430) 
s_add_i32 s18, s1, s16 // (op 13430) 

s_ashr_i32 s1, s18, 31 // (op 13430) 
s_cmp_eq_i32 s10, 0 // (op 13430) 
s_mov_b32 s19, scc // (op 13430) 
s_cmp_eq_i32 s19, 0 // (op 13430) 
s_cselect_b32 s49, 0, -1 // (op 13430) 

s_and_b32 s19, s47, 31 // (op 13430) 
s_lshl_b32 s50, 1, s19 // (op 13430) 
s_add_i32 s51, s50, s49 // (op 13430) 


s_and_b32 s49, s1, s51 // (op 13430) 


s_add_i32 s1, s18, s49 // (op 13430) 


s_ashr_i32 s18, s1, s19 // (op 13430) 


s_lshl_b32 s1, s47, 24 // (op 13430) 
s_ashr_i32 s19, s1, 31 // (op 13430) 

s_xor_b32 s1, s18, s19 // (op 13430) 

s_sub_i32 s18, s1, s19 // (op 13430) 


s_add_u32 s54, s6, 128 // (op 13430) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 13430) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13430) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13430) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 13430) 

s_mul_i32 s1, s44, s50 // (op 13430) 

s_mul_i32 s19, s48, s1 // (op 13430) 

s_mul_i32 s1, s18, s19 // (op 13430) 

s_sub_u32 s19, s16, s1 // (op 13430) 

s_mov_b32 s1, s19 // (op 13430) 
s_mul_hi_i32 s49, s1, s15 // (op 13430) 
s_add_i32 s50, s49, s1 // (op 13430) 


s_ashr_i32 s1, s50, 31 // (op 13430) 
s_cmp_eq_i32 s15, 0 // (op 13430) 
s_mov_b32 s49, scc // (op 13430) 
s_cmp_eq_i32 s49, 0 // (op 13430) 
s_cselect_b32 s51, 0, -1 // (op 13430) 

s_and_b32 s49, s46, 31 // (op 13430) 
s_lshl_b32 s52, 1, s49 // (op 13430) 
s_add_i32 s53, s52, s51 // (op 13430) 


s_and_b32 s51, s1, s53 // (op 13430) 


s_add_i32 s1, s50, s51 // (op 13430) 


s_ashr_i32 s50, s1, s49 // (op 13430) 


s_lshl_b32 s1, s46, 24 // (op 13430) 
s_ashr_i32 s49, s1, 31 // (op 13430) 

s_xor_b32 s1, s50, s49 // (op 13430) 

s_sub_i32 s50, s1, s49 // (op 13430) 


s_mul_i32 s1, s45, s44 // (op 13430) 
s_add_u32 s54, s8, 128 // (op 13430) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 13430) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13430) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13430) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 13430) 

s_sub_i32 s49, s54, s1 // (op 13430) 

s_mul_i32 s51, s50, s49 // (op 13430) 


s_sub_u32 s49, s19, s51 // (op 13430) 


s_add_u32 s19, s49, s1 // (op 13430) 


s_cmp_eq_i32 s18, 0 // (op 13430) 
s_mov_b32 s1, scc // (op 13430) 

s_cmp_eq_i32 s1, 0 // (op 13430) 
s_cselect_b32 s18, s19, s0 // (op 13430) 



v_lshl_add_u32 v23, s18, 7, v22 // (op 13430) 


v_mul_lo_u32 v29, v23, s25 // (op 13430) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13430) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 13430) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 13430) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 13430) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 13430) most significant half



v_lshlrev_b64 v[22:23], 2, v[28:29] // (op 13430) 

v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 13430) 

s_mul_i32 s1, s24, s9 // (op 13432) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13432) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13432) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13432) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13432) most significant: sum
s_add_u32 s1, s1, s19 // (op 13432) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13432) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13432) 

s_mov_b32 s20, s26 // (op 13432) reg expression
s_mov_b32 s21, s27 // (op 13432) reg expression
s_mov_b32 s22, s0 // (op 13432) reg expression
s_mov_b32 s23, 131072 // (op 13432) call()

s_mul_hi_i32 s0, s16, s11 // (op 13475) 
v_and_b32 v22, -4, v1 // (op 13475) 
s_add_i32 s1, s0, s16 // (op 13475) 

v_sub_u32 v23, v1, v22 // (op 13475) 

s_ashr_i32 s0, s1, 31 // (op 13475) 
v_lshlrev_b32 v22, 5, v23 // (op 13475) 

s_cmp_eq_i32 s11, 0 // (op 13475) 
s_mov_b32 s18, scc // (op 13475) 
v_lshrrev_b32 v23, 2, v1 // (op 13475) 
s_cmp_eq_i32 s18, 0 // (op 13475) 
s_cselect_b32 s19, 0, -1 // (op 13475) 

s_and_b32 s18, s12, 31 // (op 13475) 
s_lshl_b32 s49, 1, s18 // (op 13475) 
s_add_i32 s50, s49, s19 // (op 13475) 


s_and_b32 s19, s0, s50 // (op 13475) 


s_add_i32 s0, s1, s19 // (op 13475) 


s_ashr_i32 s1, s0, s18 // (op 13475) 


s_lshl_b32 s0, s12, 24 // (op 13475) 
s_ashr_i32 s18, s0, 31 // (op 13475) 

s_xor_b32 s0, s1, s18 // (op 13475) 

s_sub_i32 s1, s0, s18 // (op 13475) 


s_mul_hi_i32 s0, s1, s13 // (op 13475) 
s_add_i32 s18, s0, s1 // (op 13475) 

s_ashr_i32 s0, s18, 31 // (op 13475) 
s_cmp_eq_i32 s13, 0 // (op 13475) 
s_mov_b32 s19, scc // (op 13475) 
s_cmp_eq_i32 s19, 0 // (op 13475) 
s_cselect_b32 s49, 0, -1 // (op 13475) 

s_and_b32 s19, s14, 31 // (op 13475) 
s_lshl_b32 s50, 1, s19 // (op 13475) 
s_add_i32 s51, s50, s49 // (op 13475) 


s_and_b32 s49, s0, s51 // (op 13475) 


s_add_i32 s0, s18, s49 // (op 13475) 


s_ashr_i32 s18, s0, s19 // (op 13475) 


s_lshl_b32 s0, s14, 24 // (op 13475) 
s_ashr_i32 s19, s0, 31 // (op 13475) 

s_xor_b32 s0, s18, s19 // (op 13475) 

s_sub_i32 s18, s0, s19 // (op 13475) 


s_add_u32 s54, s6, 128 // (op 13475) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 13475) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 13475) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 13475) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 13475) 

s_mul_i32 s0, s18, s50 // (op 13475) 

s_sub_i32 s49, s1, s0 // (op 13475) 


s_mul_hi_i32 s0, s16, s10 // (op 13475) 
s_add_i32 s1, s0, s16 // (op 13475) 

s_ashr_i32 s0, s1, 31 // (op 13475) 
s_cmp_eq_i32 s10, 0 // (op 13475) 
s_mov_b32 s18, scc // (op 13475) 
s_cmp_eq_i32 s18, 0 // (op 13475) 
s_cselect_b32 s19, 0, -1 // (op 13475) 

s_and_b32 s18, s47, 31 // (op 13475) 
s_lshl_b32 s52, 1, s18 // (op 13475) 
s_add_i32 s53, s52, s19 // (op 13475) 


s_and_b32 s19, s0, s53 // (op 13475) 


s_add_i32 s0, s1, s19 // (op 13475) 


s_ashr_i32 s1, s0, s18 // (op 13475) 


s_lshl_b32 s0, s47, 24 // (op 13475) 
s_ashr_i32 s18, s0, 31 // (op 13475) 

s_xor_b32 s0, s1, s18 // (op 13475) 

s_sub_i32 s1, s0, s18 // (op 13475) 


s_mul_i32 s0, s44, s50 // (op 13475) 

s_mul_i32 s18, s48, s0 // (op 13475) 

s_mul_i32 s0, s1, s18 // (op 13475) 

s_sub_u32 s18, s16, s0 // (op 13475) 

s_mul_hi_i32 s0, s18, s15 // (op 13475) 
s_add_i32 s19, s0, s18 // (op 13475) 


s_ashr_i32 s0, s19, 31 // (op 13475) 
s_cmp_eq_i32 s15, 0 // (op 13475) 
s_mov_b32 s18, scc // (op 13475) 
s_cmp_eq_i32 s18, 0 // (op 13475) 
s_cselect_b32 s50, 0, -1 // (op 13475) 

s_and_b32 s18, s46, 31 // (op 13475) 
s_lshl_b32 s51, 1, s18 // (op 13475) 
s_add_i32 s52, s51, s50 // (op 13475) 


s_and_b32 s50, s0, s52 // (op 13475) 


s_add_i32 s0, s19, s50 // (op 13475) 


s_ashr_i32 s19, s0, s18 // (op 13475) 


s_lshl_b32 s0, s46, 24 // (op 13475) 
s_ashr_i32 s18, s0, 31 // (op 13475) 

s_xor_b32 s0, s19, s18 // (op 13475) 

s_sub_i32 s19, s0, s18 // (op 13475) 


s_cmp_eq_i32 s1, 0 // (op 13475) 
s_mov_b32 s0, scc // (op 13475) 

s_cmp_eq_i32 s0, 0 // (op 13475) 
s_cselect_b32 s1, s19, s49 // (op 13475) 



v_lshl_add_u32 v28, s1, 7, v22 // (op 13475) 


v_mov_b32 v31, 0 // (op 13475) convert
v_mov_b32 v30, v28 // (op 13475) convert

v_mul_lo_u32 v29, v23, s29 // (op 13475) most significant: low of low * high
v_mul_hi_u32 v22, v23, s28 // (op 13475) most significant: high of low * low
v_mul_lo_u32 v28, v23, s28 // (op 13475) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 13475) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v30 // (op 13475) least significant half
v_addc_co_u32 v29, s[0:1], v29, v31, s[0:1] // (op 13475) most significant half



v_lshlrev_b64 v[22:23], 2, v[28:29] // (op 13475) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 13475) 

s_mul_i32 s1, s28, s33 // (op 13477) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13477) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13477) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13477) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13477) most significant: sum
s_add_u32 s1, s1, s19 // (op 13477) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13477) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13477) 

s_mov_b32 s52, s30 // (op 13477) reg expression
s_mov_b32 s53, s31 // (op 13477) reg expression
s_mov_b32 s54, s0 // (op 13477) reg expression
s_mov_b32 s55, 131072 // (op 13477) call()

s_mul_hi_i32 s0, s16, s11 // (op 13494) 
v_and_b32 v22, -4, v1 // (op 13494) 
s_add_i32 s1, s0, s16 // (op 13494) 

v_sub_u32 v23, v1, v22 // (op 13494) 

s_ashr_i32 s0, s1, 31 // (op 13494) 
v_lshlrev_b32 v22, 5, v23 // (op 13494) 

s_cmp_eq_i32 s11, 0 // (op 13494) 
s_mov_b32 s18, scc // (op 13494) 
v_lshrrev_b32 v23, 2, v1 // (op 13494) 
s_cmp_eq_i32 s18, 0 // (op 13494) 
s_cselect_b32 s19, 0, -1 // (op 13494) 

s_and_b32 s18, s12, 31 // (op 13494) 
s_lshl_b32 s49, 1, s18 // (op 13494) 
s_add_i32 s50, s49, s19 // (op 13494) 


s_and_b32 s19, s0, s50 // (op 13494) 


s_add_i32 s0, s1, s19 // (op 13494) 


s_ashr_i32 s1, s0, s18 // (op 13494) 


s_lshl_b32 s0, s12, 24 // (op 13494) 
s_ashr_i32 s18, s0, 31 // (op 13494) 

s_xor_b32 s0, s1, s18 // (op 13494) 

s_sub_i32 s1, s0, s18 // (op 13494) 


s_mul_hi_i32 s0, s1, s13 // (op 13494) 
s_add_i32 s18, s0, s1 // (op 13494) 

s_ashr_i32 s0, s18, 31 // (op 13494) 
s_cmp_eq_i32 s13, 0 // (op 13494) 
s_mov_b32 s19, scc // (op 13494) 
s_cmp_eq_i32 s19, 0 // (op 13494) 
s_cselect_b32 s49, 0, -1 // (op 13494) 

s_and_b32 s19, s14, 31 // (op 13494) 
s_lshl_b32 s50, 1, s19 // (op 13494) 
s_add_i32 s51, s50, s49 // (op 13494) 


s_and_b32 s49, s0, s51 // (op 13494) 


s_add_i32 s0, s18, s49 // (op 13494) 


s_ashr_i32 s18, s0, s19 // (op 13494) 


s_lshl_b32 s0, s14, 24 // (op 13494) 
s_ashr_i32 s19, s0, 31 // (op 13494) 

s_xor_b32 s0, s18, s19 // (op 13494) 

s_sub_i32 s18, s0, s19 // (op 13494) 


s_add_u32 s50, s6, 128 // (op 13494) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13494) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 13494) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 13494) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 13494) 

s_mul_i32 s0, s18, s50 // (op 13494) 

s_sub_i32 s49, s1, s0 // (op 13494) 


s_mul_hi_i32 s0, s16, s10 // (op 13494) 
s_add_i32 s1, s0, s16 // (op 13494) 

s_ashr_i32 s0, s1, 31 // (op 13494) 
s_cmp_eq_i32 s10, 0 // (op 13494) 
s_mov_b32 s18, scc // (op 13494) 
s_cmp_eq_i32 s18, 0 // (op 13494) 
s_cselect_b32 s19, 0, -1 // (op 13494) 

s_and_b32 s18, s47, 31 // (op 13494) 
s_lshl_b32 s64, 1, s18 // (op 13494) 
s_add_i32 s65, s64, s19 // (op 13494) 


s_and_b32 s19, s0, s65 // (op 13494) 


s_add_i32 s0, s1, s19 // (op 13494) 


s_ashr_i32 s1, s0, s18 // (op 13494) 


s_lshl_b32 s0, s47, 24 // (op 13494) 
s_ashr_i32 s18, s0, 31 // (op 13494) 

s_xor_b32 s0, s1, s18 // (op 13494) 

s_sub_i32 s1, s0, s18 // (op 13494) 


s_mul_i32 s0, s44, s50 // (op 13494) 

s_mul_i32 s18, s48, s0 // (op 13494) 

s_mul_i32 s0, s1, s18 // (op 13494) 

s_sub_u32 s18, s16, s0 // (op 13494) 

s_mul_hi_i32 s0, s18, s15 // (op 13494) 
s_add_i32 s19, s0, s18 // (op 13494) 


s_ashr_i32 s0, s19, 31 // (op 13494) 
s_cmp_eq_i32 s15, 0 // (op 13494) 
s_mov_b32 s18, scc // (op 13494) 
s_cmp_eq_i32 s18, 0 // (op 13494) 
s_cselect_b32 s50, 0, -1 // (op 13494) 

s_and_b32 s18, s46, 31 // (op 13494) 
s_lshl_b32 s51, 1, s18 // (op 13494) 
s_add_i32 s64, s51, s50 // (op 13494) 


s_and_b32 s50, s0, s64 // (op 13494) 


s_add_i32 s0, s19, s50 // (op 13494) 


s_ashr_i32 s19, s0, s18 // (op 13494) 


s_lshl_b32 s0, s46, 24 // (op 13494) 
s_ashr_i32 s18, s0, 31 // (op 13494) 

s_xor_b32 s0, s19, s18 // (op 13494) 

s_sub_i32 s19, s0, s18 // (op 13494) 


s_cmp_eq_i32 s1, 0 // (op 13494) 
s_mov_b32 s0, scc // (op 13494) 

s_cmp_eq_i32 s0, 0 // (op 13494) 
s_cselect_b32 s1, s19, s49 // (op 13494) 



v_lshl_add_u32 v30, s1, 7, v22 // (op 13494) 


v_mov_b32 v33, 0 // (op 13494) convert
v_mov_b32 v32, v30 // (op 13494) convert

v_mul_lo_u32 v31, v23, s29 // (op 13494) most significant: low of low * high
v_mul_hi_u32 v22, v23, s28 // (op 13494) most significant: high of low * low
v_mul_lo_u32 v30, v23, s28 // (op 13494) least significant: low of low * low
v_add_u32 v31, v31, v22 // (op 13494) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v32 // (op 13494) least significant half
v_addc_co_u32 v31, s[0:1], v31, v33, s[0:1] // (op 13494) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 13494) 

v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 13494) 

s_mul_i32 s1, s28, s33 // (op 13496) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13496) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13496) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13496) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13496) most significant: sum
s_add_u32 s1, s1, s19 // (op 13496) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13496) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13496) 

s_mov_b32 s52, s30 // (op 13496) reg expression
s_mov_b32 s53, s31 // (op 13496) reg expression
s_mov_b32 s54, s0 // (op 13496) reg expression
s_mov_b32 s55, 131072 // (op 13496) call()

s_mul_hi_i32 s0, s16, s11 // (op 13539) 
v_lshrrev_b32 v22, 2, v1 // (op 13539) 
s_add_i32 s1, s0, s16 // (op 13539) 

v_lshlrev_b32 v23, 1, v22 // (op 13539) 

s_ashr_i32 s0, s1, 31 // (op 13539) 
v_and_b32 v22, -4, v1 // (op 13539) 
s_cmp_eq_i32 s11, 0 // (op 13539) 
s_mov_b32 s18, scc // (op 13539) 
v_sub_u32 v32, v1, v22 // (op 13539) 

s_cmp_eq_i32 s18, 0 // (op 13539) 
s_cselect_b32 s19, 0, -1 // (op 13539) 

s_and_b32 s18, s12, 31 // (op 13539) 
s_lshl_b32 s49, 1, s18 // (op 13539) 
s_add_i32 s50, s49, s19 // (op 13539) 


s_and_b32 s19, s0, s50 // (op 13539) 


s_add_i32 s0, s1, s19 // (op 13539) 


s_ashr_i32 s1, s0, s18 // (op 13539) 


s_lshl_b32 s0, s12, 24 // (op 13539) 
s_ashr_i32 s18, s0, 31 // (op 13539) 

s_xor_b32 s0, s1, s18 // (op 13539) 

s_sub_i32 s1, s0, s18 // (op 13539) 


s_mul_hi_i32 s0, s1, s13 // (op 13539) 
s_add_i32 s18, s0, s1 // (op 13539) 

s_ashr_i32 s0, s18, 31 // (op 13539) 
s_cmp_eq_i32 s13, 0 // (op 13539) 
s_mov_b32 s19, scc // (op 13539) 
s_cmp_eq_i32 s19, 0 // (op 13539) 
s_cselect_b32 s49, 0, -1 // (op 13539) 

s_and_b32 s19, s14, 31 // (op 13539) 
s_lshl_b32 s50, 1, s19 // (op 13539) 
s_add_i32 s51, s50, s49 // (op 13539) 


s_and_b32 s49, s0, s51 // (op 13539) 


s_add_i32 s0, s18, s49 // (op 13539) 


s_ashr_i32 s18, s0, s19 // (op 13539) 


s_lshl_b32 s0, s14, 24 // (op 13539) 
s_ashr_i32 s19, s0, 31 // (op 13539) 

s_xor_b32 s0, s18, s19 // (op 13539) 

s_sub_i32 s18, s0, s19 // (op 13539) 


s_mul_i32 s0, s1, s44 // (op 13539) 

s_sub_u32 s1, s16, s0 // (op 13539) 

s_mul_i32 s0, s18, s44 // (op 13539) 
s_add_u32 s0, s0, s1 // (op 13539) 


s_mul_hi_i32 s1, s16, s10 // (op 13539) 
s_add_i32 s18, s1, s16 // (op 13539) 

s_ashr_i32 s1, s18, 31 // (op 13539) 
s_cmp_eq_i32 s10, 0 // (op 13539) 
s_mov_b32 s19, scc // (op 13539) 
s_cmp_eq_i32 s19, 0 // (op 13539) 
s_cselect_b32 s49, 0, -1 // (op 13539) 

s_and_b32 s19, s47, 31 // (op 13539) 
s_lshl_b32 s50, 1, s19 // (op 13539) 
s_add_i32 s51, s50, s49 // (op 13539) 


s_and_b32 s49, s1, s51 // (op 13539) 


s_add_i32 s1, s18, s49 // (op 13539) 


s_ashr_i32 s18, s1, s19 // (op 13539) 


s_lshl_b32 s1, s47, 24 // (op 13539) 
s_ashr_i32 s19, s1, 31 // (op 13539) 

s_xor_b32 s1, s18, s19 // (op 13539) 

s_sub_i32 s18, s1, s19 // (op 13539) 


s_add_u32 s50, s6, 128 // (op 13539) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13539) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 13539) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 13539) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 13539) 

s_mul_i32 s1, s44, s50 // (op 13539) 

s_mul_i32 s19, s48, s1 // (op 13539) 

s_mul_i32 s1, s18, s19 // (op 13539) 

s_sub_u32 s19, s16, s1 // (op 13539) 

s_mov_b32 s1, s19 // (op 13539) 
s_mul_hi_i32 s49, s1, s15 // (op 13539) 
s_add_i32 s50, s49, s1 // (op 13539) 


s_ashr_i32 s1, s50, 31 // (op 13539) 
s_cmp_eq_i32 s15, 0 // (op 13539) 
s_mov_b32 s49, scc // (op 13539) 
s_cmp_eq_i32 s49, 0 // (op 13539) 
s_cselect_b32 s51, 0, -1 // (op 13539) 

s_and_b32 s49, s46, 31 // (op 13539) 
s_lshl_b32 s64, 1, s49 // (op 13539) 
s_add_i32 s65, s64, s51 // (op 13539) 


s_and_b32 s51, s1, s65 // (op 13539) 


s_add_i32 s1, s50, s51 // (op 13539) 


s_ashr_i32 s50, s1, s49 // (op 13539) 


s_lshl_b32 s1, s46, 24 // (op 13539) 
s_ashr_i32 s49, s1, 31 // (op 13539) 

s_xor_b32 s1, s50, s49 // (op 13539) 

s_sub_i32 s50, s1, s49 // (op 13539) 


s_mul_i32 s1, s45, s44 // (op 13539) 
s_add_u32 s64, s8, 128 // (op 13539) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 13539) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 13539) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 13539) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 13539) 

s_sub_i32 s49, s64, s1 // (op 13539) 

s_mul_i32 s51, s50, s49 // (op 13539) 


s_sub_u32 s49, s19, s51 // (op 13539) 


s_add_u32 s19, s49, s1 // (op 13539) 


s_cmp_eq_i32 s18, 0 // (op 13539) 
s_mov_b32 s1, scc // (op 13539) 

s_cmp_eq_i32 s1, 0 // (op 13539) 
s_cselect_b32 s18, s19, s0 // (op 13539) 



v_lshl_add_u32 v22, s18, 7, v23 // (op 13539) 


v_mul_lo_u32 v23, v22, s34 // (op 13539) 
v_add_u32 v23, v23, v32 // (op 13539) 


s_mul_i32 s1, s34, s9 // (op 13541) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13541) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13541) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13541) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13541) most significant: sum
s_add_u32 s1, s1, s19 // (op 13541) most significant: sum


s_mov_b32 s64, s36 // (op 13541) reg expression
s_mov_b32 s65, s37 // (op 13541) reg expression
s_mov_b32 s66, s0 // (op 13541) reg expression
s_mov_b32 s67, 131072 // (op 13541) call()

s_mul_hi_i32 s0, s16, s11 // (op 13558) 
v_lshrrev_b32 v22, 2, v1 // (op 13558) 
s_add_i32 s1, s0, s16 // (op 13558) 

v_lshlrev_b32 v32, 1, v22 // (op 13558) 

s_ashr_i32 s0, s1, 31 // (op 13558) 
v_and_b32 v22, -4, v1 // (op 13558) 
s_cmp_eq_i32 s11, 0 // (op 13558) 
s_mov_b32 s18, scc // (op 13558) 
v_sub_u32 v33, v1, v22 // (op 13558) 

s_cmp_eq_i32 s18, 0 // (op 13558) 
s_cselect_b32 s19, 0, -1 // (op 13558) 

s_and_b32 s18, s12, 31 // (op 13558) 
s_lshl_b32 s49, 1, s18 // (op 13558) 
s_add_i32 s50, s49, s19 // (op 13558) 


s_and_b32 s19, s0, s50 // (op 13558) 


s_add_i32 s0, s1, s19 // (op 13558) 


s_ashr_i32 s1, s0, s18 // (op 13558) 


s_lshl_b32 s0, s12, 24 // (op 13558) 
s_ashr_i32 s18, s0, 31 // (op 13558) 

s_xor_b32 s0, s1, s18 // (op 13558) 

s_sub_i32 s1, s0, s18 // (op 13558) 


s_mul_hi_i32 s0, s1, s13 // (op 13558) 
s_add_i32 s18, s0, s1 // (op 13558) 

s_ashr_i32 s0, s18, 31 // (op 13558) 
s_cmp_eq_i32 s13, 0 // (op 13558) 
s_mov_b32 s19, scc // (op 13558) 
s_cmp_eq_i32 s19, 0 // (op 13558) 
s_cselect_b32 s49, 0, -1 // (op 13558) 

s_and_b32 s19, s14, 31 // (op 13558) 
s_lshl_b32 s50, 1, s19 // (op 13558) 
s_add_i32 s51, s50, s49 // (op 13558) 


s_and_b32 s49, s0, s51 // (op 13558) 


s_add_i32 s0, s18, s49 // (op 13558) 


s_ashr_i32 s18, s0, s19 // (op 13558) 


s_lshl_b32 s0, s14, 24 // (op 13558) 
s_ashr_i32 s19, s0, 31 // (op 13558) 

s_xor_b32 s0, s18, s19 // (op 13558) 

s_sub_i32 s18, s0, s19 // (op 13558) 


s_mul_i32 s0, s1, s44 // (op 13558) 

s_sub_u32 s1, s16, s0 // (op 13558) 

s_mul_i32 s0, s18, s44 // (op 13558) 
s_add_u32 s0, s0, s1 // (op 13558) 


s_mul_hi_i32 s1, s16, s10 // (op 13558) 
s_add_i32 s18, s1, s16 // (op 13558) 

s_ashr_i32 s1, s18, 31 // (op 13558) 
s_cmp_eq_i32 s10, 0 // (op 13558) 
s_mov_b32 s19, scc // (op 13558) 
s_cmp_eq_i32 s19, 0 // (op 13558) 
s_cselect_b32 s49, 0, -1 // (op 13558) 

s_and_b32 s19, s47, 31 // (op 13558) 
s_lshl_b32 s50, 1, s19 // (op 13558) 
s_add_i32 s51, s50, s49 // (op 13558) 


s_and_b32 s49, s1, s51 // (op 13558) 


s_add_i32 s1, s18, s49 // (op 13558) 


s_ashr_i32 s18, s1, s19 // (op 13558) 


s_lshl_b32 s1, s47, 24 // (op 13558) 
s_ashr_i32 s19, s1, 31 // (op 13558) 

s_xor_b32 s1, s18, s19 // (op 13558) 

s_sub_i32 s18, s1, s19 // (op 13558) 


s_add_u32 s50, s6, 128 // (op 13558) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13558) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 13558) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 13558) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 13558) 

s_mul_i32 s1, s44, s50 // (op 13558) 

s_mul_i32 s19, s48, s1 // (op 13558) 

s_mul_i32 s1, s18, s19 // (op 13558) 

s_sub_u32 s19, s16, s1 // (op 13558) 

s_mov_b32 s1, s19 // (op 13558) 
s_mul_hi_i32 s49, s1, s15 // (op 13558) 
s_add_i32 s50, s49, s1 // (op 13558) 


s_ashr_i32 s1, s50, 31 // (op 13558) 
s_cmp_eq_i32 s15, 0 // (op 13558) 
s_mov_b32 s49, scc // (op 13558) 
s_cmp_eq_i32 s49, 0 // (op 13558) 
s_cselect_b32 s51, 0, -1 // (op 13558) 

s_and_b32 s49, s46, 31 // (op 13558) 
s_lshl_b32 s68, 1, s49 // (op 13558) 
s_add_i32 s69, s68, s51 // (op 13558) 


s_and_b32 s51, s1, s69 // (op 13558) 


s_add_i32 s1, s50, s51 // (op 13558) 


s_ashr_i32 s50, s1, s49 // (op 13558) 


s_lshl_b32 s1, s46, 24 // (op 13558) 
s_ashr_i32 s49, s1, 31 // (op 13558) 

s_xor_b32 s1, s50, s49 // (op 13558) 

s_sub_i32 s50, s1, s49 // (op 13558) 


s_mul_i32 s1, s45, s44 // (op 13558) 
s_add_u32 s68, s8, 128 // (op 13558) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 13558) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 13558) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 13558) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 13558) 

s_sub_i32 s49, s68, s1 // (op 13558) 

s_mul_i32 s51, s50, s49 // (op 13558) 


s_sub_u32 s49, s19, s51 // (op 13558) 


s_add_u32 s19, s49, s1 // (op 13558) 


s_cmp_eq_i32 s18, 0 // (op 13558) 
s_mov_b32 s1, scc // (op 13558) 

s_cmp_eq_i32 s1, 0 // (op 13558) 
s_cselect_b32 s18, s19, s0 // (op 13558) 



v_lshl_add_u32 v22, s18, 7, v32 // (op 13558) 


v_mul_lo_u32 v32, v22, s34 // (op 13558) 
v_add_u32 v32, v32, v33 // (op 13558) 


s_mul_i32 s1, s34, s9 // (op 13560) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13560) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13560) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13560) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13560) most significant: sum
s_add_u32 s1, s1, s19 // (op 13560) most significant: sum


s_mov_b32 s64, s36 // (op 13560) reg expression
s_mov_b32 s65, s37 // (op 13560) reg expression
s_mov_b32 s66, s0 // (op 13560) reg expression
s_mov_b32 s67, 131072 // (op 13560) call()

s_mul_hi_i32 s0, s16, s11 // (op 13603) 
v_and_b32 v22, -128, v1 // (op 13603) 
s_add_i32 s1, s0, s16 // (op 13603) 

v_sub_u32 v33, v1, v22 // (op 13603) 

s_ashr_i32 s0, s1, 31 // (op 13603) 
v_lshrrev_b32 v22, 7, v1 // (op 13603) 
s_cmp_eq_i32 s11, 0 // (op 13603) 
s_mov_b32 s18, scc // (op 13603) 
v_lshlrev_b32 v34, 1, v22 // (op 13603) 

s_cmp_eq_i32 s18, 0 // (op 13603) 
s_cselect_b32 s19, 0, -1 // (op 13603) 

s_and_b32 s18, s12, 31 // (op 13603) 
s_lshl_b32 s49, 1, s18 // (op 13603) 
s_add_i32 s50, s49, s19 // (op 13603) 


s_and_b32 s19, s0, s50 // (op 13603) 


s_add_i32 s0, s1, s19 // (op 13603) 


s_ashr_i32 s1, s0, s18 // (op 13603) 


s_lshl_b32 s0, s12, 24 // (op 13603) 
s_ashr_i32 s18, s0, 31 // (op 13603) 

s_xor_b32 s0, s1, s18 // (op 13603) 

s_sub_i32 s1, s0, s18 // (op 13603) 


s_mul_hi_i32 s0, s1, s13 // (op 13603) 
s_add_i32 s18, s0, s1 // (op 13603) 

s_ashr_i32 s0, s18, 31 // (op 13603) 
s_cmp_eq_i32 s13, 0 // (op 13603) 
s_mov_b32 s19, scc // (op 13603) 
s_cmp_eq_i32 s19, 0 // (op 13603) 
s_cselect_b32 s49, 0, -1 // (op 13603) 

s_and_b32 s19, s14, 31 // (op 13603) 
s_lshl_b32 s50, 1, s19 // (op 13603) 
s_add_i32 s51, s50, s49 // (op 13603) 


s_and_b32 s49, s0, s51 // (op 13603) 


s_add_i32 s0, s18, s49 // (op 13603) 


s_ashr_i32 s18, s0, s19 // (op 13603) 


s_lshl_b32 s0, s14, 24 // (op 13603) 
s_ashr_i32 s19, s0, 31 // (op 13603) 

s_xor_b32 s0, s18, s19 // (op 13603) 

s_sub_i32 s18, s0, s19 // (op 13603) 


s_add_u32 s50, s6, 128 // (op 13603) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13603) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 13603) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 13603) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 13603) 

s_mul_i32 s0, s18, s50 // (op 13603) 

s_sub_i32 s49, s1, s0 // (op 13603) 


s_mul_hi_i32 s0, s16, s10 // (op 13603) 
s_add_i32 s1, s0, s16 // (op 13603) 

s_ashr_i32 s0, s1, 31 // (op 13603) 
s_cmp_eq_i32 s10, 0 // (op 13603) 
s_mov_b32 s18, scc // (op 13603) 
s_cmp_eq_i32 s18, 0 // (op 13603) 
s_cselect_b32 s19, 0, -1 // (op 13603) 

s_and_b32 s18, s47, 31 // (op 13603) 
s_lshl_b32 s68, 1, s18 // (op 13603) 
s_add_i32 s69, s68, s19 // (op 13603) 


s_and_b32 s19, s0, s69 // (op 13603) 


s_add_i32 s0, s1, s19 // (op 13603) 


s_ashr_i32 s1, s0, s18 // (op 13603) 


s_lshl_b32 s0, s47, 24 // (op 13603) 
s_ashr_i32 s18, s0, 31 // (op 13603) 

s_xor_b32 s0, s1, s18 // (op 13603) 

s_sub_i32 s1, s0, s18 // (op 13603) 


s_mul_i32 s0, s44, s50 // (op 13603) 

s_mul_i32 s18, s48, s0 // (op 13603) 

s_mul_i32 s0, s1, s18 // (op 13603) 

s_sub_u32 s18, s16, s0 // (op 13603) 

s_mul_hi_i32 s0, s18, s15 // (op 13603) 
s_add_i32 s19, s0, s18 // (op 13603) 


s_ashr_i32 s0, s19, 31 // (op 13603) 
s_cmp_eq_i32 s15, 0 // (op 13603) 
s_mov_b32 s18, scc // (op 13603) 
s_cmp_eq_i32 s18, 0 // (op 13603) 
s_cselect_b32 s50, 0, -1 // (op 13603) 

s_and_b32 s18, s46, 31 // (op 13603) 
s_lshl_b32 s51, 1, s18 // (op 13603) 
s_add_i32 s68, s51, s50 // (op 13603) 


s_and_b32 s50, s0, s68 // (op 13603) 


s_add_i32 s0, s19, s50 // (op 13603) 


s_ashr_i32 s19, s0, s18 // (op 13603) 


s_lshl_b32 s0, s46, 24 // (op 13603) 
s_ashr_i32 s18, s0, 31 // (op 13603) 

s_xor_b32 s0, s19, s18 // (op 13603) 

s_sub_i32 s19, s0, s18 // (op 13603) 


s_cmp_eq_i32 s1, 0 // (op 13603) 
s_mov_b32 s0, scc // (op 13603) 

s_cmp_eq_i32 s0, 0 // (op 13603) 
s_cselect_b32 s1, s19, s49 // (op 13603) 



v_lshl_add_u32 v22, s1, 7, v33 // (op 13603) 


v_mul_lo_u32 v33, v34, s38 // (op 13603) 
v_add_u32 v33, v33, v22 // (op 13603) 


s_mul_i32 s1, s38, s43 // (op 13605) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13605) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13605) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13605) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13605) most significant: sum
s_add_u32 s1, s1, s19 // (op 13605) most significant: sum


s_mov_b32 s68, s40 // (op 13605) reg expression
s_mov_b32 s69, s41 // (op 13605) reg expression
s_mov_b32 s70, s0 // (op 13605) reg expression
s_mov_b32 s71, 131072 // (op 13605) call()

s_mul_hi_i32 s0, s16, s11 // (op 13622) 
v_and_b32 v22, -128, v1 // (op 13622) 
s_add_i32 s1, s0, s16 // (op 13622) 

v_sub_u32 v34, v1, v22 // (op 13622) 

s_ashr_i32 s0, s1, 31 // (op 13622) 
v_lshrrev_b32 v22, 7, v1 // (op 13622) 
s_cmp_eq_i32 s11, 0 // (op 13622) 
s_mov_b32 s18, scc // (op 13622) 
v_lshlrev_b32 v35, 1, v22 // (op 13622) 

s_cmp_eq_i32 s18, 0 // (op 13622) 
s_cselect_b32 s19, 0, -1 // (op 13622) 

s_and_b32 s18, s12, 31 // (op 13622) 
s_lshl_b32 s49, 1, s18 // (op 13622) 
s_add_i32 s50, s49, s19 // (op 13622) 


s_and_b32 s19, s0, s50 // (op 13622) 


s_add_i32 s0, s1, s19 // (op 13622) 


s_ashr_i32 s1, s0, s18 // (op 13622) 


s_lshl_b32 s0, s12, 24 // (op 13622) 
s_ashr_i32 s18, s0, 31 // (op 13622) 

s_xor_b32 s0, s1, s18 // (op 13622) 

s_sub_i32 s1, s0, s18 // (op 13622) 


s_mul_hi_i32 s0, s1, s13 // (op 13622) 
s_add_i32 s18, s0, s1 // (op 13622) 

s_ashr_i32 s0, s18, 31 // (op 13622) 
s_cmp_eq_i32 s13, 0 // (op 13622) 
s_mov_b32 s19, scc // (op 13622) 
s_cmp_eq_i32 s19, 0 // (op 13622) 
s_cselect_b32 s49, 0, -1 // (op 13622) 

s_and_b32 s19, s14, 31 // (op 13622) 
s_lshl_b32 s50, 1, s19 // (op 13622) 
s_add_i32 s51, s50, s49 // (op 13622) 


s_and_b32 s49, s0, s51 // (op 13622) 


s_add_i32 s0, s18, s49 // (op 13622) 


s_ashr_i32 s18, s0, s19 // (op 13622) 


s_lshl_b32 s0, s14, 24 // (op 13622) 
s_ashr_i32 s19, s0, 31 // (op 13622) 

s_xor_b32 s0, s18, s19 // (op 13622) 

s_sub_i32 s18, s0, s19 // (op 13622) 


s_add_u32 s50, s6, 128 // (op 13622) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13622) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13622) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13622) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13622) 

s_mul_i32 s0, s18, s50 // (op 13622) 

s_sub_i32 s49, s1, s0 // (op 13622) 


s_mul_hi_i32 s0, s16, s10 // (op 13622) 
s_add_i32 s1, s0, s16 // (op 13622) 

s_ashr_i32 s0, s1, 31 // (op 13622) 
s_cmp_eq_i32 s10, 0 // (op 13622) 
s_mov_b32 s18, scc // (op 13622) 
s_cmp_eq_i32 s18, 0 // (op 13622) 
s_cselect_b32 s19, 0, -1 // (op 13622) 

s_and_b32 s18, s47, 31 // (op 13622) 
s_lshl_b32 s72, 1, s18 // (op 13622) 
s_add_i32 s73, s72, s19 // (op 13622) 


s_and_b32 s19, s0, s73 // (op 13622) 


s_add_i32 s0, s1, s19 // (op 13622) 


s_ashr_i32 s1, s0, s18 // (op 13622) 


s_lshl_b32 s0, s47, 24 // (op 13622) 
s_ashr_i32 s18, s0, 31 // (op 13622) 

s_xor_b32 s0, s1, s18 // (op 13622) 

s_sub_i32 s1, s0, s18 // (op 13622) 


s_mul_i32 s0, s44, s50 // (op 13622) 

s_mul_i32 s18, s48, s0 // (op 13622) 

s_mul_i32 s0, s1, s18 // (op 13622) 

s_sub_u32 s18, s16, s0 // (op 13622) 

s_mul_hi_i32 s0, s18, s15 // (op 13622) 
s_add_i32 s19, s0, s18 // (op 13622) 


s_ashr_i32 s0, s19, 31 // (op 13622) 
s_cmp_eq_i32 s15, 0 // (op 13622) 
s_mov_b32 s18, scc // (op 13622) 
s_cmp_eq_i32 s18, 0 // (op 13622) 
s_cselect_b32 s50, 0, -1 // (op 13622) 

s_and_b32 s18, s46, 31 // (op 13622) 
s_lshl_b32 s51, 1, s18 // (op 13622) 
s_add_i32 s72, s51, s50 // (op 13622) 


s_and_b32 s50, s0, s72 // (op 13622) 


s_add_i32 s0, s19, s50 // (op 13622) 


s_ashr_i32 s19, s0, s18 // (op 13622) 


s_lshl_b32 s0, s46, 24 // (op 13622) 
s_ashr_i32 s18, s0, 31 // (op 13622) 

s_xor_b32 s0, s19, s18 // (op 13622) 

s_sub_i32 s19, s0, s18 // (op 13622) 


s_cmp_eq_i32 s1, 0 // (op 13622) 
s_mov_b32 s0, scc // (op 13622) 

s_cmp_eq_i32 s0, 0 // (op 13622) 
s_cselect_b32 s1, s19, s49 // (op 13622) 



v_lshl_add_u32 v22, s1, 7, v34 // (op 13622) 


v_mul_lo_u32 v34, v35, s38 // (op 13622) 
v_add_u32 v34, v34, v22 // (op 13622) 


s_mul_i32 s1, s38, s43 // (op 13624) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13624) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13624) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13624) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13624) most significant: sum
s_add_u32 s1, s1, s19 // (op 13624) most significant: sum


s_mov_b32 s68, s40 // (op 13624) reg expression
s_mov_b32 s69, s41 // (op 13624) reg expression
s_mov_b32 s70, s0 // (op 13624) reg expression
s_mov_b32 s71, 131072 // (op 13624) call()

v_and_b32 v22, -4, v1 // (op 13750) 
v_sub_u32 v35, v1, v22 // (op 13750) 

v_lshlrev_b32 v22, 5, v35 // (op 13750) 

v_lshrrev_b32 v35, 2, v1 // (op 13750) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13750) 


v_lshlrev_b32 v22, 2, v36 // (op 13750) 

v_lshrrev_b32 v35, 3, v22 // (op 13750) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13750) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13337) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13337) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13337) 


v_mov_b64 v[38:39], v[28:29] // (op 13337) reg expression
s_mov_b32 m0, s0 // (op 13337) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 13337) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v36 // (op 13337) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 13337) most significant half

s_add_u32 m0, m0, 4096 // (op 13337) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 13337) Load value direct to lds




v_mov_b64 v[36:37], s[38:39] // (op 12887) convert

v_mov_b32 v22, v33 // (op 12887) reg expression
buffer_load_ubyte v38, v22, s[68:71], 0 offen offset: 0 // (op 12887) Load value
v_add_u32 v22, v22, v36 // (op 12887) 
buffer_load_ubyte v39, v22, s[68:71], 0 offen offset: 0 // (op 12887) Load value


v_and_b32 v22, -4, v1 // (op 13693) 
v_sub_u32 v35, v1, v22 // (op 13693) 

v_lshlrev_b32 v22, 5, v35 // (op 13693) 

v_lshrrev_b32 v35, 2, v1 // (op 13693) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13693) 


v_lshlrev_b32 v22, 2, v36 // (op 13693) 

v_lshrrev_b32 v35, 3, v22 // (op 13693) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13693) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13342) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13342) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13342) 


v_mov_b64 v[40:41], v[24:25] // (op 13342) 
s_add_u32 s1, 8192, s0 // (op 13342) 
s_mov_b32 m0, s1 // (op 13342) reg expression
buffer_load_dwordx4 v40, s[20:23], 0 offen offset: 0 lds // (op 13342) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v36 // (op 13342) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 13342) most significant half

s_add_u32 m0, m0, 4096 // (op 13342) 
buffer_load_dwordx4 v40, s[20:23], 0 offen offset: 0 lds // (op 13342) Load value direct to lds





v_mov_b64 v[36:37], s[34:35] // (op 12893) convert

v_mov_b32 v22, v23 // (op 12893) 
buffer_load_ubyte v40, v22, s[64:67], 0 offen offset: 0 // (op 12893) Load value
v_add_u32 v22, v22, v36 // (op 12893) 
buffer_load_ubyte v41, v22, s[64:67], 0 offen offset: 0 // (op 12893) Load value


v_and_b32 v22, -4, v1 // (op 13977) 
v_sub_u32 v35, v1, v22 // (op 13977) 

v_lshlrev_b32 v22, 5, v35 // (op 13977) 

v_lshrrev_b32 v35, 2, v1 // (op 13977) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13977) 


v_lshlrev_b32 v22, 2, v36 // (op 13977) 

v_lshrrev_b32 v35, 3, v22 // (op 13977) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13977) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13347) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13347) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13347) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13347) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13347) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13347) 

v_mov_b32 v22, s18 // (op 13347) 
v_mov_b32 v35, s19 // (op 13347) 
v_add_co_u32 v42, s[50:51], v28, v22 // (op 13347) least significant half
v_addc_co_u32 v43, s[50:51], v29, v35, s[50:51] // (op 13347) most significant half




s_add_u32 s1, 16384, s0 // (op 13347) 
s_mov_b32 m0, s1 // (op 13347) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 13347) Load value direct to lds
v_add_co_u32 v42, s[18:19], v42, v36 // (op 13347) least significant half
v_addc_co_u32 v43, s[18:19], 0, v43, s[18:19] // (op 13347) most significant half

s_add_u32 m0, m0, 4096 // (op 13347) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 13347) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 12899) convert

s_lshl_b32 s0, s38, 2 // (op 12899) 
v_add_u32 v22, v33, s0 // (op 12899) 

buffer_load_ubyte v42, v22, s[68:71], 0 offen offset: 0 // (op 12899) Load value
v_add_u32 v22, v22, v36 // (op 12899) 
buffer_load_ubyte v43, v22, s[68:71], 0 offen offset: 0 // (op 12899) Load value


v_and_b32 v22, -4, v1 // (op 13933) 
v_sub_u32 v35, v1, v22 // (op 13933) 

v_lshlrev_b32 v22, 5, v35 // (op 13933) 

v_lshrrev_b32 v35, 2, v1 // (op 13933) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 13933) 


v_lshlrev_b32 v22, 2, v36 // (op 13933) 

v_lshrrev_b32 v35, 3, v22 // (op 13933) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13933) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13349) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13349) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13349) 


v_mov_b64 v[44:45], v[24:25] // (op 13349) 
s_add_u32 s1, 24576, s0 // (op 13349) 
s_mov_b32 m0, s1 // (op 13349) reg expression
buffer_load_dwordx4 v44, s[20:23], 64 offen offset: 0 lds // (op 13349) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v36 // (op 13349) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 13349) most significant half

s_add_u32 m0, m0, 4096 // (op 13349) 
buffer_load_dwordx4 v44, s[20:23], 64 offen offset: 0 lds // (op 13349) Load value direct to lds





v_mov_b64 v[36:37], s[34:35] // (op 12905) convert

v_mov_b32 v22, v23 // (op 12905) 
buffer_load_ubyte v44, v22, s[64:67], 0 offen offset:4 // (op 12905) Load value
v_add_u32 v22, v22, v36 // (op 12905) 
buffer_load_ubyte v45, v22, s[64:67], 0 offen offset:4 // (op 12905) Load value


v_and_b32 v22, -4, v1 // (op 14123) 
v_sub_u32 v35, v1, v22 // (op 14123) 

v_lshlrev_b32 v22, 5, v35 // (op 14123) 

v_lshrrev_b32 v35, 2, v1 // (op 14123) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14123) 


v_lshlrev_b32 v22, 2, v36 // (op 14123) 

v_lshrrev_b32 v35, 3, v22 // (op 14123) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14123) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13351) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13351) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13351) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13351) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13351) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13351) 

s_lshl_b64 s[50:51], s[18:19], 1 // (op 13351) 

v_mov_b32 v22, s50 // (op 13351) 
v_mov_b32 v35, s51 // (op 13351) 
v_add_co_u32 v46, s[18:19], v28, v22 // (op 13351) least significant half
v_addc_co_u32 v47, s[18:19], v29, v35, s[18:19] // (op 13351) most significant half




s_add_u32 s1, 32768, s0 // (op 13351) 
s_mov_b32 m0, s1 // (op 13351) reg expression
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13351) Load value direct to lds
v_add_co_u32 v46, s[18:19], v46, v36 // (op 13351) least significant half
v_addc_co_u32 v47, s[18:19], 0, v47, s[18:19] // (op 13351) most significant half

s_add_u32 m0, m0, 4096 // (op 13351) 
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13351) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 12911) convert

s_lshl_b32 s0, s38, 3 // (op 12911) 
v_add_u32 v22, v33, s0 // (op 12911) 

buffer_load_ubyte v46, v22, s[68:71], 0 offen offset: 0 // (op 12911) Load value
v_add_u32 v22, v22, v36 // (op 12911) 
buffer_load_ubyte v47, v22, s[68:71], 0 offen offset: 0 // (op 12911) Load value


v_and_b32 v22, -4, v1 // (op 14079) 
v_sub_u32 v35, v1, v22 // (op 14079) 

v_lshlrev_b32 v22, 5, v35 // (op 14079) 

v_lshrrev_b32 v35, 2, v1 // (op 14079) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14079) 


v_lshlrev_b32 v22, 2, v36 // (op 14079) 

v_lshrrev_b32 v35, 3, v22 // (op 14079) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14079) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13353) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13353) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13353) 


v_mov_b64 v[48:49], v[24:25] // (op 13353) 
s_add_u32 s1, 40960, s0 // (op 13353) 
s_mov_b32 m0, s1 // (op 13353) reg expression
s_mov_b32 s18, 128 // (op 13353) call()
buffer_load_dwordx4 v48, s[20:23], s18 offen offset: 0 lds // (op 13353) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v36 // (op 13353) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 13353) most significant half

s_add_u32 m0, m0, 4096 // (op 13353) 
buffer_load_dwordx4 v48, s[20:23], s18 offen offset: 0 lds // (op 13353) Load value direct to lds






v_mov_b64 v[36:37], s[34:35] // (op 12917) convert

v_mov_b32 v22, v23 // (op 12917) 
buffer_load_ubyte v48, v22, s[64:67], 0 offen offset:8 // (op 12917) Load value
v_add_u32 v22, v22, v36 // (op 12917) 
buffer_load_ubyte v49, v22, s[64:67], 0 offen offset:8 // (op 12917) Load value


v_and_b32 v22, -4, v1 // (op 14269) 
v_sub_u32 v35, v1, v22 // (op 14269) 

v_lshlrev_b32 v22, 5, v35 // (op 14269) 

v_lshrrev_b32 v35, 2, v1 // (op 14269) 
v_lshl_add_u32 v36, v35, 7, v22 // (op 14269) 


v_lshlrev_b32 v22, 2, v36 // (op 14269) 

v_lshrrev_b32 v35, 3, v22 // (op 14269) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 14269) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 13355) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13355) 

v_lshrrev_b64 v[36:37], 3, s[50:51] // (op 13355) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 13355) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13355) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13355) 

s_mul_i32 s51, 3, s19 // (op 13355) most significant: low of low * high
s_mul_i32 s50, 3, s18 // (op 13355) least significant
s_mul_hi_u32 s1, 3, s18 // (op 13355) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 13355) most significant: sum


v_mov_b32 v22, s50 // (op 13355) 
v_mov_b32 v35, s51 // (op 13355) 
v_add_co_u32 v50, s[18:19], v28, v22 // (op 13355) least significant half
v_addc_co_u32 v51, s[18:19], v29, v35, s[18:19] // (op 13355) most significant half




s_add_u32 s1, 49152, s0 // (op 13355) 
s_mov_b32 m0, s1 // (op 13355) reg expression
buffer_load_dwordx4 v50, s[52:55], 0 offen offset: 0 lds // (op 13355) Load value direct to lds
v_add_co_u32 v50, s[18:19], v50, v36 // (op 13355) least significant half
v_addc_co_u32 v51, s[18:19], 0, v51, s[18:19] // (op 13355) most significant half

s_add_u32 m0, m0, 4096 // (op 13355) 
buffer_load_dwordx4 v50, s[52:55], 0 offen offset: 0 lds // (op 13355) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 12923) convert

s_lshl_b32 s0, s38, 2 // (op 12923) 
s_mul_i32 s1, 3, s0 // (op 12923) 

v_add_u32 v22, v33, s1 // (op 12923) 

buffer_load_ubyte v36, v22, s[68:71], 0 offen offset: 0 // (op 12923) Load value
v_add_u32 v22, v22, v28 // (op 12923) 
buffer_load_ubyte v37, v22, s[68:71], 0 offen offset: 0 // (op 12923) Load value



v_and_b32 v22, -4, v1 // (op 14225) 
v_sub_u32 v33, v1, v22 // (op 14225) 

v_lshlrev_b32 v22, 5, v33 // (op 14225) 

v_lshrrev_b32 v33, 2, v1 // (op 14225) 
v_lshl_add_u32 v35, v33, 7, v22 // (op 14225) 


v_lshlrev_b32 v22, 2, v35 // (op 14225) 

v_lshrrev_b32 v33, 3, v22 // (op 14225) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 14225) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 13357) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13357) 

v_lshrrev_b64 v[28:29], 3, s[50:51] // (op 13357) 


v_mov_b64 v[50:51], v[24:25] // (op 13357) 
s_add_u32 s1, 57344, s0 // (op 13357) 
s_mov_b32 m0, s1 // (op 13357) reg expression
s_mov_b32 s18, 192 // (op 13357) call()
buffer_load_dwordx4 v50, s[20:23], s18 offen offset: 0 lds // (op 13357) Load value direct to lds
v_add_co_u32 v50, s[50:51], v50, v28 // (op 13357) least significant half
v_addc_co_u32 v51, s[50:51], 0, v51, s[50:51] // (op 13357) most significant half

s_add_u32 m0, m0, 4096 // (op 13357) 
buffer_load_dwordx4 v50, s[20:23], s18 offen offset: 0 lds // (op 13357) Load value direct to lds







v_mov_b64 v[24:25], s[34:35] // (op 12929) convert

v_mov_b32 v22, v23 // (op 12929) 
buffer_load_ubyte v28, v22, s[64:67], 0 offen offset:12 // (op 12929) Load value
v_add_u32 v22, v22, v24 // (op 12929) 
buffer_load_ubyte v29, v22, s[64:67], 0 offen offset:12 // (op 12929) Load value



v_and_b32 v33, -128, v1 // (op 13866) 
v_sub_u32 v35, v1, v33 // (op 13866) 

v_lshrrev_b32 v33, 7, v1 // (op 13866) 
v_lshl_add_u32 v22, v33, 8, v35 // (op 13866) 





v_add_u32 v33, 65536, v22 // (op 12932) 
s_waitcnt vmcnt(29)
ds_write_b8 v33, v38 // (op 12932) Store local data 

v_add_u32 v33, 65664, v22 // (op 12932) 
s_waitcnt vmcnt(28)
ds_write_b8 v33, v39 // (op 12932) Store local data 


v_and_b32 v33, -4, v1 // (op 13808) 
v_sub_u32 v35, v1, v33 // (op 13808) 

v_lshrrev_b32 v33, 2, v1 // (op 13808) 
v_lshl_add_u32 v22, v33, 3, v35 // (op 13808) 





v_add_u32 v33, 66048, v22 // (op 12934) 
s_waitcnt vmcnt(25)
ds_write_b8 v33, v40 // (op 12934) Store local data 

v_add_u32 v33, 66052, v22 // (op 12934) 
s_waitcnt vmcnt(24)
ds_write_b8 v33, v41 // (op 12934) Store local data 


s_barrier  // (op 14565) 



ds_read_b64_tr_b4 v[22:23], v4 // (op 12977) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v4 offset:1024 // (op 12977) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14600) 


v_add_u32 v33, 65536, v5 // (op 13017) 
ds_read_u8 v35, v33 // (op 13017) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:16 // (op 12982) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:1040 // (op 12982) Transpose load local data 


v_add_u32 v33, 65568, v5 // (op 13022) 
ds_read_u8 v54, v33 // (op 13022) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:32 // (op 12987) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v4 offset:1056 // (op 12987) Transpose load local data 


v_add_u32 v33, 65600, v5 // (op 13027) 
ds_read_u8 v55, v33 // (op 13027) Load local data 




ds_read_b64_tr_b4 v[60:61], v4 offset:48 // (op 12992) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v4 offset:1072 // (op 12992) Transpose load local data 


v_add_u32 v33, 65632, v5 // (op 13032) 
ds_read_u8 v64, v33 // (op 13032) Load local data 

s_barrier  // (op 14549) 



ds_read_b128 v[66:69], v3 offset:8192 // (op 12957) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14582) 



v_add_u32 v33, 66048, v0 // (op 12997) 
ds_read_u8 v65, v33 // (op 12997) Load local data 




ds_read_b128 v[70:73], v3 offset:10240 // (op 12962) Load local data 



v_add_u32 v33, 66176, v0 // (op 13002) 
ds_read_u8 v74, v33 // (op 13002) Load local data 




ds_read_b128 v[76:79], v3 offset:12288 // (op 12967) Load local data 



v_add_u32 v33, 66304, v0 // (op 13007) 
ds_read_u8 v75, v33 // (op 13007) Load local data 




ds_read_b128 v[80:83], v3 offset:14336 // (op 12972) Load local data 



v_add_u32 v33, 66432, v0 // (op 13012) 
ds_read_u8 v84, v33 // (op 13012) Load local data 

v_and_b32 v33, -4, v1 // (op 13665) 
v_sub_u32 v85, v1, v33 // (op 13665) 

v_lshlrev_b32 v33, 5, v85 // (op 13665) 

v_lshrrev_b32 v85, 2, v1 // (op 13665) 
v_lshl_add_u32 v86, v85, 7, v33 // (op 13665) 


v_lshlrev_b32 v33, 2, v86 // (op 13665) 

v_lshrrev_b32 v85, 3, v33 // (op 13665) 

s_nop 0
v_readfirstlane_b32 s0, v85 // (op 13665) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 13722) 
v_sub_u32 v85, v1, v33 // (op 13722) 

v_lshlrev_b32 v33, 5, v85 // (op 13722) 

v_lshrrev_b32 v85, 2, v1 // (op 13722) 
v_lshl_add_u32 v86, v85, 7, v33 // (op 13722) 


v_lshlrev_b32 v33, 2, v86 // (op 13722) 

v_lshrrev_b32 v85, 3, v33 // (op 13722) 

s_nop 0
v_readfirstlane_b32 s1, v85 // (op 13722) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 13779) 
v_sub_u32 v85, v1, v33 // (op 13779) 

v_lshrrev_b32 v33, 2, v1 // (op 13779) 
v_lshl_add_u32 v86, v33, 3, v85 // (op 13779) 


v_and_b32 v33, -128, v1 // (op 13837) 
v_sub_u32 v85, v1, v33 // (op 13837) 

v_lshrrev_b32 v33, 7, v1 // (op 13837) 
v_lshl_add_u32 v87, v33, 8, v85 // (op 13837) 


v_and_b32 v33, -4, v1 // (op 13918) 
v_sub_u32 v85, v1, v33 // (op 13918) 

v_lshlrev_b32 v33, 5, v85 // (op 13918) 

v_lshrrev_b32 v85, 2, v1 // (op 13918) 
v_lshl_add_u32 v88, v85, 7, v33 // (op 13918) 


v_lshlrev_b32 v33, 2, v88 // (op 13918) 

v_lshrrev_b32 v85, 3, v33 // (op 13918) 

s_nop 0
v_readfirstlane_b32 s18, v85 // (op 13918) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 13962) 
v_sub_u32 v85, v1, v33 // (op 13962) 

v_lshlrev_b32 v33, 5, v85 // (op 13962) 

v_lshrrev_b32 v85, 2, v1 // (op 13962) 
v_lshl_add_u32 v88, v85, 7, v33 // (op 13962) 


v_lshlrev_b32 v33, 2, v88 // (op 13962) 

v_lshrrev_b32 v85, 3, v33 // (op 13962) 

s_nop 0
v_readfirstlane_b32 s19, v85 // (op 13962) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14006) 
v_sub_u32 v85, v1, v33 // (op 14006) 

v_lshrrev_b32 v33, 2, v1 // (op 14006) 
v_lshl_add_u32 v88, v33, 3, v85 // (op 14006) 


v_and_b32 v33, -128, v1 // (op 14035) 
v_sub_u32 v85, v1, v33 // (op 14035) 

v_lshrrev_b32 v33, 7, v1 // (op 14035) 
v_lshl_add_u32 v89, v33, 8, v85 // (op 14035) 


v_and_b32 v33, -4, v1 // (op 14064) 
v_sub_u32 v85, v1, v33 // (op 14064) 

v_lshlrev_b32 v33, 5, v85 // (op 14064) 

v_lshrrev_b32 v85, 2, v1 // (op 14064) 
v_lshl_add_u32 v90, v85, 7, v33 // (op 14064) 


v_lshlrev_b32 v33, 2, v90 // (op 14064) 

v_lshrrev_b32 v85, 3, v33 // (op 14064) 

s_nop 0
v_readfirstlane_b32 s49, v85 // (op 14064) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14108) 
v_sub_u32 v85, v1, v33 // (op 14108) 

v_lshlrev_b32 v33, 5, v85 // (op 14108) 

v_lshrrev_b32 v85, 2, v1 // (op 14108) 
v_lshl_add_u32 v90, v85, 7, v33 // (op 14108) 


v_lshlrev_b32 v33, 2, v90 // (op 14108) 

v_lshrrev_b32 v85, 3, v33 // (op 14108) 

s_nop 0
v_readfirstlane_b32 s50, v85 // (op 14108) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14152) 
v_sub_u32 v85, v1, v33 // (op 14152) 

v_lshrrev_b32 v33, 2, v1 // (op 14152) 
v_lshl_add_u32 v90, v33, 3, v85 // (op 14152) 


v_and_b32 v33, -128, v1 // (op 14181) 
v_sub_u32 v85, v1, v33 // (op 14181) 

v_lshrrev_b32 v33, 7, v1 // (op 14181) 
v_lshl_add_u32 v91, v33, 8, v85 // (op 14181) 


v_and_b32 v33, -4, v1 // (op 14210) 
v_sub_u32 v85, v1, v33 // (op 14210) 

v_lshlrev_b32 v33, 5, v85 // (op 14210) 

v_lshrrev_b32 v85, 2, v1 // (op 14210) 
v_lshl_add_u32 v92, v85, 7, v33 // (op 14210) 


v_lshlrev_b32 v33, 2, v92 // (op 14210) 

v_lshrrev_b32 v85, 3, v33 // (op 14210) 

s_nop 0
v_readfirstlane_b32 s51, v85 // (op 14210) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14254) 
v_sub_u32 v85, v1, v33 // (op 14254) 

v_lshlrev_b32 v33, 5, v85 // (op 14254) 

v_lshrrev_b32 v85, 2, v1 // (op 14254) 
v_lshl_add_u32 v92, v85, 7, v33 // (op 14254) 


v_lshlrev_b32 v33, 2, v92 // (op 14254) 

v_lshrrev_b32 v85, 3, v33 // (op 14254) 

s_nop 0
v_readfirstlane_b32 s72, v85 // (op 14254) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 14298) 
v_sub_u32 v85, v1, v33 // (op 14298) 

v_lshrrev_b32 v33, 2, v1 // (op 14298) 
v_lshl_add_u32 v92, v33, 3, v85 // (op 14298) 


v_and_b32 v33, -128, v1 // (op 14327) 
v_sub_u32 v85, v1, v33 // (op 14327) 

v_lshrrev_b32 v33, 7, v1 // (op 14327) 
v_lshl_add_u32 v93, v33, 8, v85 // (op 14327) 


s_mov_b32 s73, 0 // (op 2804) call()
s_ashr_i64 s[74:75], s[32:33], 7 // (op 2803) 
s_mov_b32 s76, s74 // (op 2803) convert

s_and_b32 s74, s76, 4294967292 // (op 2803) 

s_cmp_lt_u32 s73, s74 // (op 2803) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2803 // (op 2803) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2803 if false)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2803:
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[22:25], a[0:3], v65, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 
s_barrier  // (op 14568) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[50:53], a[4:7], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 
s_add_u32 s73, s73, 4 // (op 2805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[56:59], a[8:11], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 820) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13289) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[60:63], a[12:15], v65, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1118) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13289) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13289) 


v_mov_b64 v[96:97], v[30:31] // (op 13289) 
s_lshl_b64 s[74:75], s[28:29], 7 // (op 13289) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[22:25], a[16:19], v74, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1921) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13289) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[50:53], a[20:23], v74, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1829) 
s_lshr_b64 s[74:75], s[76:77], 3 // (op 13289) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[56:59], a[24:27], v74, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1737) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13289) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[60:63], a[28:31], v74, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1645) 
v_mov_b32 v33, s76 // (op 13289) 
v_mov_b32 v85, s77 // (op 13289) 
v_add_co_u32 v98, s[74:75], v96, v33 // (op 13289) least significant half
v_addc_co_u32 v99, s[74:75], v97, v85, s[74:75] // (op 13289) most significant half




s_mov_b32 m0, s1 // (op 13289) reg expression
buffer_load_dwordx4 v98, s[52:55], 0 offen offset: 0 lds // (op 13289) Load value direct to lds
v_add_co_u32 v98, s[74:75], v98, v94 // (op 13289) least significant half
v_addc_co_u32 v99, s[74:75], 0, v99, s[74:75] // (op 13289) most significant half

s_add_u32 m0, m0, 4096 // (op 13289) 
buffer_load_dwordx4 v98, s[52:55], 0 offen offset: 0 lds // (op 13289) Load value direct to lds


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[22:25], a[32:35], v75, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3211) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[50:53], a[36:39], v75, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3119) 
v_mov_b64 v[94:95], s[38:39] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[56:59], a[40:43], v75, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3027) 
v_mov_b32 v33, v34 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[60:63], a[44:47], v75, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2935) 
s_lshl_b32 s74, s38, 4 // (op 25) 
v_add_u32 v85, v33, s74 // (op 25) 

buffer_load_ubyte v38, v85, s[68:71], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v85, v85, v94 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[22:25], a[48:51], v84, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4501) 
buffer_load_ubyte v39, v85, s[68:71], 0 offen offset: 0 // (op 25) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[50:53], a[52:55], v84, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4409) 
s_barrier  // (op 14552) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[56:59], a[56:59], v84, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4317) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 13284) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[60:63], a[60:63], v84, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4225) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13284) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13284) 


v_mov_b64 v[98:99], v[26:27] // (op 13284) 
v_mov_b64 v[100:101], v[98:99] // (op 13284) 
s_add_u32 s74, 8192, s0 // (op 13284) 
s_mov_b32 m0, s74 // (op 13284) reg expression
s_mov_b32 s75, 256 // (op 13284) call()
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13284) Load value direct to lds
v_add_co_u32 v100, s[76:77], v100, v94 // (op 13284) least significant half
v_addc_co_u32 v101, s[76:77], 0, v101, s[76:77] // (op 13284) most significant half

s_add_u32 m0, m0, 4096 // (op 13284) 
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13284) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 17) convert

v_mov_b32 v85, v32 // (op 17) 
v_mov_b32 v100, v85 // (op 17) 
buffer_load_ubyte v40, v100, s[64:67], 0 offen offset:16 // (op 17) Load value
v_add_u32 v100, v100, v94 // (op 17) 
buffer_load_ubyte v41, v100, s[64:67], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 14640) 



v_add_u32 v94, 66560, v89 // (op 255) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v42 // (op 255) Store local data 

v_add_u32 v94, 66688, v89 // (op 255) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v43 // (op 255) Store local data 

s_barrier  // (op 14634) 



v_add_u32 v94, 67072, v88 // (op 254) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v44 // (op 254) Store local data 

v_add_u32 v94, 67076, v88 // (op 254) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v45 // (op 254) Store local data 

s_barrier  // (op 14623) 



ds_read_b64_tr_b4 v[100:101], v8 offset:16384 // (op 247) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v8 offset:17408 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14637) 


v_add_u32 v94, 66560, v9 // (op 251) 
ds_read_u8 v95, v94 // (op 251) Load local data 




ds_read_b64_tr_b4 v[104:105], v8 offset:16400 // (op 529) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v8 offset:17424 // (op 529) Transpose load local data 


v_add_u32 v94, 66592, v9 // (op 533) 
ds_read_u8 v74, v94 // (op 533) Load local data 




ds_read_b64_tr_b4 v[70:71], v8 offset:16416 // (op 827) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v8 offset:17440 // (op 827) Transpose load local data 


v_add_u32 v94, 66624, v9 // (op 831) 
ds_read_u8 v75, v94 // (op 831) Load local data 




ds_read_b64_tr_b4 v[76:77], v8 offset:16432 // (op 1125) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v8 offset:17456 // (op 1125) Transpose load local data 


v_add_u32 v94, 66656, v9 // (op 1129) 
ds_read_u8 v108, v94 // (op 1129) Load local data 

s_barrier  // (op 14617) 
s_barrier  // (op 14745) 



ds_read_b128 v[66:69], v7 offset:24576 // (op 245) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14630) 



v_add_u32 v94, 67072, v6 // (op 249) 
ds_read_u8 v65, v94 // (op 249) Load local data 




ds_read_b128 v[110:113], v7 offset:26624 // (op 1650) Load local data 



v_add_u32 v94, 67200, v6 // (op 1653) 
ds_read_u8 v54, v94 // (op 1653) Load local data 




ds_read_b128 v[114:117], v7 offset:28672 // (op 2940) Load local data 



v_add_u32 v94, 67328, v6 // (op 2943) 
ds_read_u8 v55, v94 // (op 2943) Load local data 




ds_read_b128 v[118:121], v7 offset:30720 // (op 4230) Load local data 



v_add_u32 v94, 67456, v6 // (op 4233) 
ds_read_u8 v64, v94 // (op 4233) Load local data 

s_barrier  // (op 14803) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[100:103], a[0:3], v65, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 
s_barrier  // (op 14627) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[104:107], a[4:7], v65, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13312) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[70:73], a[8:11], v65, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 821) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13312) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[76:79], a[12:15], v65, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1119) 
v_lshrrev_b64 v[122:123], 3, s[76:77] // (op 13312) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13312) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13312) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13312) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:113], v[100:103], a[16:19], v54, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1922) 
s_mul_i32 s77, 5, s75 // (op 13312) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[110:113], v[104:107], a[20:23], v54, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1830) 
s_mul_i32 s76, 5, s74 // (op 13312) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:113], v[70:73], a[24:27], v54, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1738) 
s_mul_hi_u32 s78, 5, s74 // (op 13312) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[110:113], v[76:79], a[28:31], v54, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1646) 
s_add_u32 s77, s77, s78 // (op 13312) most significant: sum



v_mov_b32 v94, s76 // (op 13312) 
v_mov_b32 v109, s77 // (op 13312) 
v_add_co_u32 v110, s[74:75], v96, v94 // (op 13312) least significant half
v_addc_co_u32 v111, s[74:75], v97, v109, s[74:75] // (op 13312) most significant half




s_add_u32 s74, 16384, s19 // (op 13312) 
s_nop 0
s_mov_b32 m0, s74 // (op 13312) reg expression
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 13312) Load value direct to lds
v_add_co_u32 v110, s[76:77], v110, v122 // (op 13312) least significant half
v_addc_co_u32 v111, s[76:77], 0, v111, s[76:77] // (op 13312) most significant half

s_add_u32 m0, m0, 4096 // (op 13312) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 13312) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[114:117], v[100:103], a[32:35], v55, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3212) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[114:117], v[104:107], a[36:39], v55, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3120) 
v_mov_b64 v[112:113], s[38:39] // (op 258) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[114:117], v[70:73], a[40:43], v55, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3028) 
s_lshl_b32 s74, s38, 2 // (op 258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:117], v[76:79], a[44:47], v55, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2936) 
s_mul_i32 s75, 5, s74 // (op 258) 


v_add_u32 v94, v33, s75 // (op 258) 

buffer_load_ubyte v42, v94, s[68:71], 0 offen offset: 0 // (op 258) Load value
v_add_u32 v94, v94, v112 // (op 258) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[118:121], v[100:103], a[48:51], v64, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4502) 
buffer_load_ubyte v43, v94, s[68:71], 0 offen offset: 0 // (op 258) Load value




s_barrier  // (op 14620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:121], v[104:107], a[52:55], v64, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4410) 


s_lshl_b64 s[74:75], s[24:25], 6 // (op 13307) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[118:121], v[70:73], a[56:59], v64, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4318) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13307) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13307) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[118:121], v[76:79], a[60:63], v64, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4226) 
v_mov_b64 v[100:101], v[98:99] // (op 13307) 


s_add_u32 s74, 24576, s18 // (op 13307) 
s_mov_b32 m0, s74 // (op 13307) reg expression
s_mov_b32 s75, 320 // (op 13307) call()
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13307) Load value direct to lds
v_add_co_u32 v100, s[76:77], v100, v94 // (op 13307) least significant half
v_addc_co_u32 v101, s[76:77], 0, v101, s[76:77] // (op 13307) most significant half

s_add_u32 m0, m0, 4096 // (op 13307) 
buffer_load_dwordx4 v100, s[20:23], s75 offen offset: 0 lds // (op 13307) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 257) convert

v_mov_b32 v100, v85 // (op 257) 
buffer_load_ubyte v44, v100, s[64:67], 0 offen offset:20 // (op 257) Load value
v_add_u32 v100, v100, v94 // (op 257) 
buffer_load_ubyte v45, v100, s[64:67], 0 offen offset:20 // (op 257) Load value


s_barrier  // (op 14666) 



v_add_u32 v94, 66560, v91 // (op 295) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v46 // (op 295) Store local data 

v_add_u32 v94, 66688, v91 // (op 295) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v47 // (op 295) Store local data 

s_barrier  // (op 14660) 



v_add_u32 v94, 67072, v90 // (op 294) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v48 // (op 294) Store local data 

v_add_u32 v94, 67076, v90 // (op 294) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v49 // (op 294) Store local data 

s_barrier  // (op 14649) 



ds_read_b64_tr_b4 v[22:23], v12 offset:32768 // (op 287) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v12 offset:33792 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14663) 


v_add_u32 v94, 66560, v13 // (op 291) 
ds_read_u8 v35, v94 // (op 291) Load local data 




ds_read_b64_tr_b4 v[50:51], v12 offset:32784 // (op 537) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v12 offset:33808 // (op 537) Transpose load local data 


v_add_u32 v94, 66592, v13 // (op 541) 
ds_read_u8 v54, v94 // (op 541) Load local data 




ds_read_b64_tr_b4 v[56:57], v12 offset:32800 // (op 835) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v12 offset:33824 // (op 835) Transpose load local data 


v_add_u32 v94, 66624, v13 // (op 839) 
ds_read_u8 v55, v94 // (op 839) Load local data 




ds_read_b64_tr_b4 v[60:61], v12 offset:32816 // (op 1133) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v12 offset:33840 // (op 1133) Transpose load local data 


v_add_u32 v94, 66656, v13 // (op 1137) 
ds_read_u8 v84, v94 // (op 1137) Load local data 

s_barrier  // (op 14643) 
s_barrier  // (op 14751) 



ds_read_b128 v[66:69], v11 offset:40960 // (op 285) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14656) 



v_add_u32 v94, 67072, v10 // (op 289) 
ds_read_u8 v65, v94 // (op 289) Load local data 




ds_read_b128 v[80:83], v11 offset:43008 // (op 1656) Load local data 



v_add_u32 v94, 67200, v10 // (op 1659) 
ds_read_u8 v74, v94 // (op 1659) Load local data 




ds_read_b128 v[100:103], v11 offset:45056 // (op 2946) Load local data 



v_add_u32 v94, 67328, v10 // (op 2949) 
ds_read_u8 v75, v94 // (op 2949) Load local data 




ds_read_b128 v[104:107], v11 offset:47104 // (op 4236) Load local data 



v_add_u32 v94, 67456, v10 // (op 4239) 
ds_read_u8 v95, v94 // (op 4239) Load local data 

s_barrier  // (op 14809) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[22:25], a[0:3], v65, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 
s_barrier  // (op 14653) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[50:53], a[4:7], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13322) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[56:59], a[8:11], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 822) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13322) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[60:63], a[12:15], v65, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1120) 
v_lshrrev_b64 v[108:109], 3, s[76:77] // (op 13322) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13322) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13322) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13322) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[22:25], a[16:19], v74, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1923) 
s_mul_i32 s77, 6, s75 // (op 13322) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[50:53], a[20:23], v74, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1831) 
s_mul_i32 s76, 6, s74 // (op 13322) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[56:59], a[24:27], v74, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1739) 
s_mul_hi_u32 s78, 6, s74 // (op 13322) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[60:63], a[28:31], v74, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1647) 
s_add_u32 s77, s77, s78 // (op 13322) most significant: sum


v_mov_b32 v94, s76 // (op 13322) 
v_mov_b32 v110, s77 // (op 13322) 
v_add_co_u32 v112, s[74:75], v96, v94 // (op 13322) least significant half
v_addc_co_u32 v113, s[74:75], v97, v110, s[74:75] // (op 13322) most significant half




s_add_u32 s74, 32768, s50 // (op 13322) 
s_nop 0
s_mov_b32 m0, s74 // (op 13322) reg expression
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 13322) Load value direct to lds
v_add_co_u32 v112, s[76:77], v112, v108 // (op 13322) least significant half
v_addc_co_u32 v113, s[76:77], 0, v113, s[76:77] // (op 13322) most significant half

s_add_u32 m0, m0, 4096 // (op 13322) 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 13322) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:103], v[22:25], a[32:35], v75, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3213) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:103], v[50:53], a[36:39], v75, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3121) 
v_mov_b64 v[108:109], s[38:39] // (op 298) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[56:59], a[40:43], v75, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3029) 
s_lshl_b32 s74, s38, 2 // (op 298) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[60:63], a[44:47], v75, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2937) 
s_mul_i32 s75, 6, s74 // (op 298) 


v_add_u32 v94, v33, s75 // (op 298) 

buffer_load_ubyte v46, v94, s[68:71], 0 offen offset: 0 // (op 298) Load value
v_add_u32 v94, v94, v108 // (op 298) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[22:25], a[48:51], v95, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4503) 
buffer_load_ubyte v47, v94, s[68:71], 0 offen offset: 0 // (op 298) Load value


s_barrier  // (op 14646) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[50:53], a[52:55], v95, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4411) 

s_lshl_b64 s[74:75], s[24:25], 6 // (op 13317) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:107], v[56:59], a[56:59], v95, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4319) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13317) 

v_lshrrev_b64 v[100:101], 3, s[76:77] // (op 13317) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:107], v[60:63], a[60:63], v95, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4227) 
v_mov_b64 v[102:103], v[98:99] // (op 13317) 


s_add_u32 s74, 40960, s49 // (op 13317) 
s_mov_b32 m0, s74 // (op 13317) reg expression
s_mov_b32 s75, 384 // (op 13317) call()
buffer_load_dwordx4 v102, s[20:23], s75 offen offset: 0 lds // (op 13317) Load value direct to lds
v_add_co_u32 v102, s[76:77], v102, v100 // (op 13317) least significant half
v_addc_co_u32 v103, s[76:77], 0, v103, s[76:77] // (op 13317) most significant half

s_add_u32 m0, m0, 4096 // (op 13317) 
buffer_load_dwordx4 v102, s[20:23], s75 offen offset: 0 lds // (op 13317) Load value direct to lds





v_mov_b64 v[94:95], s[34:35] // (op 297) convert

v_mov_b32 v100, v85 // (op 297) 
buffer_load_ubyte v48, v100, s[64:67], 0 offen offset:24 // (op 297) Load value
v_add_u32 v100, v100, v94 // (op 297) 
buffer_load_ubyte v49, v100, s[64:67], 0 offen offset:24 // (op 297) Load value


s_barrier  // (op 14692) 



v_add_u32 v94, 66560, v93 // (op 335) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v36 // (op 335) Store local data 

v_add_u32 v94, 66688, v93 // (op 335) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v37 // (op 335) Store local data 

s_barrier  // (op 14686) 



v_add_u32 v94, 67072, v92 // (op 334) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v28 // (op 334) Store local data 

v_add_u32 v94, 67076, v92 // (op 334) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v29 // (op 334) Store local data 

s_barrier  // (op 14675) 



ds_read_b64_tr_b4 v[70:71], v16 offset:49152 // (op 327) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v16 offset:50176 // (op 327) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14689) 


v_add_u32 v94, 66560, v17 // (op 331) 
ds_read_u8 v64, v94 // (op 331) Load local data 




ds_read_b64_tr_b4 v[76:77], v16 offset:49168 // (op 545) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v16 offset:50192 // (op 545) Transpose load local data 


v_add_u32 v94, 66592, v17 // (op 549) 
ds_read_u8 v74, v94 // (op 549) Load local data 




ds_read_b64_tr_b4 v[80:81], v16 offset:49184 // (op 843) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v16 offset:50208 // (op 843) Transpose load local data 


v_add_u32 v94, 66624, v17 // (op 847) 
ds_read_u8 v75, v94 // (op 847) Load local data 




ds_read_b64_tr_b4 v[100:101], v16 offset:49200 // (op 1141) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v16 offset:50224 // (op 1141) Transpose load local data 


v_add_u32 v94, 66656, v17 // (op 1145) 
ds_read_u8 v95, v94 // (op 1145) Load local data 

s_barrier  // (op 14669) 
s_barrier  // (op 14757) 



ds_read_b128 v[66:69], v15 offset:57344 // (op 325) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14682) 



v_add_u32 v94, 67072, v14 // (op 329) 
ds_read_u8 v65, v94 // (op 329) Load local data 




ds_read_b128 v[104:107], v15 offset:59392 // (op 1662) Load local data 



v_add_u32 v94, 67200, v14 // (op 1665) 
ds_read_u8 v54, v94 // (op 1665) Load local data 




ds_read_b128 v[108:111], v15 offset:61440 // (op 2952) Load local data 



v_add_u32 v94, 67328, v14 // (op 2955) 
ds_read_u8 v55, v94 // (op 2955) Load local data 




ds_read_b128 v[112:115], v15 offset:63488 // (op 4242) Load local data 



v_add_u32 v94, 67456, v14 // (op 4245) 
ds_read_u8 v84, v94 // (op 4245) Load local data 

s_barrier  // (op 14815) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:73], a[0:3], v65, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 
s_barrier  // (op 14679) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[76:79], a[4:7], v65, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 
s_lshl_b64 s[74:75], s[28:29], 6 // (op 13332) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[80:83], a[8:11], v65, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 823) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13332) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[100:103], a[12:15], v65, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1121) 
v_lshrrev_b64 v[116:117], 3, s[76:77] // (op 13332) 


s_lshl_b64 s[74:75], s[28:29], 7 // (op 13332) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13332) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13332) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[70:73], a[16:19], v54, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1924) 
s_mul_i32 s77, 7, s75 // (op 13332) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[76:79], a[20:23], v54, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1832) 
s_mul_i32 s76, 7, s74 // (op 13332) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[80:83], a[24:27], v54, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1740) 
s_mul_hi_u32 s78, 7, s74 // (op 13332) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:107], v[100:103], a[28:31], v54, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1648) 
s_add_u32 s77, s77, s78 // (op 13332) most significant: sum



v_mov_b32 v94, s76 // (op 13332) 
v_mov_b32 v104, s77 // (op 13332) 
v_add_co_u32 v106, s[74:75], v96, v94 // (op 13332) least significant half
v_addc_co_u32 v107, s[74:75], v97, v104, s[74:75] // (op 13332) most significant half




s_add_u32 s74, 49152, s72 // (op 13332) 
s_nop 0
s_mov_b32 m0, s74 // (op 13332) reg expression
buffer_load_dwordx4 v106, s[52:55], 0 offen offset: 0 lds // (op 13332) Load value direct to lds
v_add_co_u32 v106, s[76:77], v106, v116 // (op 13332) least significant half
v_addc_co_u32 v107, s[76:77], 0, v107, s[76:77] // (op 13332) most significant half

s_add_u32 m0, m0, 4096 // (op 13332) 
buffer_load_dwordx4 v106, s[52:55], 0 offen offset: 0 lds // (op 13332) Load value direct to lds



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[70:73], a[32:35], v55, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3214) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[76:79], a[36:39], v55, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3122) 
v_mov_b64 v[96:97], s[38:39] // (op 338) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[80:83], a[40:43], v55, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3030) 
s_lshl_b32 s74, s38, 2 // (op 338) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[100:103], a[44:47], v55, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2938) 
s_mul_i32 s75, 7, s74 // (op 338) 


v_add_u32 v94, v33, s75 // (op 338) 

buffer_load_ubyte v36, v94, s[68:71], 0 offen offset: 0 // (op 338) Load value
v_add_u32 v94, v94, v96 // (op 338) 
s_lshl_b64 s[74:75], s[28:29], 7 // (op 13486) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 13486) 

s_lshr_b64 s[74:75], s[76:77], 3 // (op 13486) 

s_lshl_b64 s[76:77], s[74:75], 2 // (op 13486) 

v_mov_b32 v104, s76 // (op 13486) 
v_mov_b32 v105, s77 // (op 13486) 
v_add_co_u32 v30, s[74:75], v30, v104 // (op 13486) least significant half
v_addc_co_u32 v31, s[74:75], v31, v105, s[74:75] // (op 13486) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[70:73], a[48:51], v84, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4504) 
buffer_load_ubyte v37, v94, s[68:71], 0 offen offset: 0 // (op 338) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[76:79], a[52:55], v84, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4412) 
s_barrier  // (op 14672) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[80:83], a[56:59], v84, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4320) 
s_lshl_b32 s74, s38, 4 // (op 13614) 
v_add_u32 v34, v34, s74 // (op 13614) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[100:103], a[60:63], v84, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4228) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 13327) 



s_lshl_b64 s[76:77], s[74:75], 2 // (op 13327) 

v_lshrrev_b64 v[94:95], 3, s[76:77] // (op 13327) 


v_mov_b64 v[96:97], v[98:99] // (op 13327) 
s_add_u32 s74, 57344, s51 // (op 13327) 
s_mov_b32 m0, s74 // (op 13327) reg expression
s_mov_b32 s75, 448 // (op 13327) call()
buffer_load_dwordx4 v96, s[20:23], s75 offen offset: 0 lds // (op 13327) Load value direct to lds
v_add_co_u32 v96, s[76:77], v96, v94 // (op 13327) least significant half
v_addc_co_u32 v97, s[76:77], 0, v97, s[76:77] // (op 13327) most significant half

s_add_u32 m0, m0, 4096 // (op 13327) 
buffer_load_dwordx4 v96, s[20:23], s75 offen offset: 0 lds // (op 13327) Load value direct to lds






v_add_co_u32 v26, vcc, 256, v26 // (op 13422) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 13422) most significant half
v_mov_b64 v[94:95], s[34:35] // (op 337) convert

v_mov_b32 v33, v85 // (op 337) 
buffer_load_ubyte v28, v33, s[64:67], 0 offen offset:28 // (op 337) Load value
v_add_u32 v33, v33, v94 // (op 337) 
buffer_load_ubyte v29, v33, s[64:67], 0 offen offset:28 // (op 337) Load value



s_barrier  // (op 14603) 
v_add_u32 v32, 16, v32 // (op 13550) 



v_add_u32 v33, 65536, v87 // (op 30) 
s_waitcnt vmcnt(29)
ds_write_b8 v33, v38 // (op 30) Store local data 

v_add_u32 v33, 65664, v87 // (op 30) 
s_waitcnt vmcnt(28)
ds_write_b8 v33, v39 // (op 30) Store local data 

s_barrier  // (op 14585) 



v_add_u32 v33, 66048, v86 // (op 21) 
s_waitcnt vmcnt(25)
ds_write_b8 v33, v40 // (op 21) Store local data 

v_add_u32 v33, 66052, v86 // (op 21) 
s_waitcnt vmcnt(24)
ds_write_b8 v33, v41 // (op 21) Store local data 

s_barrier  // (op 14574) 



ds_read_b64_tr_b4 v[22:23], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v4 offset:1024 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14610) 


v_add_u32 v33, 65536, v5 // (op 85) 
ds_read_u8 v35, v33 // (op 85) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:16 // (op 517) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:1040 // (op 517) Transpose load local data 


v_add_u32 v33, 65568, v5 // (op 521) 
ds_read_u8 v54, v33 // (op 521) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:32 // (op 815) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v4 offset:1056 // (op 815) Transpose load local data 


v_add_u32 v33, 65600, v5 // (op 819) 
ds_read_u8 v55, v33 // (op 819) Load local data 




ds_read_b64_tr_b4 v[60:61], v4 offset:48 // (op 1113) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v4 offset:1072 // (op 1113) Transpose load local data 


v_add_u32 v33, 65632, v5 // (op 1117) 
ds_read_u8 v64, v33 // (op 1117) Load local data 

s_barrier  // (op 14558) 



ds_read_b128 v[66:69], v3 offset:8192 // (op 70) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 14592) 



v_add_u32 v33, 66048, v0 // (op 80) 
ds_read_u8 v65, v33 // (op 80) Load local data 




ds_read_b128 v[70:73], v3 offset:10240 // (op 1640) Load local data 



v_add_u32 v33, 66176, v0 // (op 1643) 
ds_read_u8 v74, v33 // (op 1643) Load local data 




ds_read_b128 v[76:79], v3 offset:12288 // (op 2930) Load local data 



v_add_u32 v33, 66304, v0 // (op 2933) 
ds_read_u8 v75, v33 // (op 2933) Load local data 




ds_read_b128 v[80:83], v3 offset:14336 // (op 4220) Load local data 



v_add_u32 v33, 66432, v0 // (op 4223) 
ds_read_u8 v84, v33 // (op 4223) Load local data 

s_ashr_i64 s[74:75], s[32:33], 7 // (op 2803) 
s_mov_b32 s76, s74 // (op 2803) convert

s_and_b32 s74, s76, 4294967292 // (op 2803) 

s_cmp_lt_u32 s73, s74 // (op 2803) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2803 // (op 2803) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_2803 if true)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_2803:


























s_barrier  // (op 14769) 
s_mul_hi_i32 s0, s16, s11 // (op 13370) 
v_lshrrev_b32 v22, 2, v1 // (op 13370) 
s_add_i32 s1, s0, s16 // (op 13370) 

v_and_b32 v23, -4, v1 // (op 13370) 
s_ashr_i32 s0, s1, 31 // (op 13370) 
v_sub_u32 v24, v1, v23 // (op 13370) 

s_cmp_eq_i32 s11, 0 // (op 13370) 
s_mov_b32 s18, scc // (op 13370) 
v_lshlrev_b32 v23, 5, v24 // (op 13370) 

s_cmp_eq_i32 s18, 0 // (op 13370) 
s_cselect_b32 s19, 0, -1 // (op 13370) 

s_and_b32 s18, s12, 31 // (op 13370) 
s_lshl_b32 s49, 1, s18 // (op 13370) 
s_add_i32 s50, s49, s19 // (op 13370) 


s_and_b32 s19, s0, s50 // (op 13370) 


s_add_i32 s0, s1, s19 // (op 13370) 


s_ashr_i32 s1, s0, s18 // (op 13370) 


s_lshl_b32 s0, s12, 24 // (op 13370) 
s_ashr_i32 s18, s0, 31 // (op 13370) 

s_xor_b32 s0, s1, s18 // (op 13370) 

s_sub_i32 s1, s0, s18 // (op 13370) 


s_mul_hi_i32 s0, s1, s13 // (op 13370) 
s_add_i32 s18, s0, s1 // (op 13370) 

s_ashr_i32 s0, s18, 31 // (op 13370) 
s_cmp_eq_i32 s13, 0 // (op 13370) 
s_mov_b32 s19, scc // (op 13370) 
s_cmp_eq_i32 s19, 0 // (op 13370) 
s_cselect_b32 s49, 0, -1 // (op 13370) 

s_and_b32 s19, s14, 31 // (op 13370) 
s_lshl_b32 s50, 1, s19 // (op 13370) 
s_add_i32 s51, s50, s49 // (op 13370) 


s_and_b32 s49, s0, s51 // (op 13370) 


s_add_i32 s0, s18, s49 // (op 13370) 


s_ashr_i32 s18, s0, s19 // (op 13370) 


s_lshl_b32 s0, s14, 24 // (op 13370) 
s_ashr_i32 s19, s0, 31 // (op 13370) 

s_xor_b32 s0, s18, s19 // (op 13370) 

s_sub_i32 s18, s0, s19 // (op 13370) 


s_mul_i32 s0, s1, s44 // (op 13370) 

s_sub_u32 s1, s16, s0 // (op 13370) 

s_mul_i32 s0, s18, s44 // (op 13370) 
s_add_u32 s0, s0, s1 // (op 13370) 


s_mul_hi_i32 s1, s16, s10 // (op 13370) 
s_add_i32 s18, s1, s16 // (op 13370) 

s_ashr_i32 s1, s18, 31 // (op 13370) 
s_cmp_eq_i32 s10, 0 // (op 13370) 
s_mov_b32 s19, scc // (op 13370) 
s_cmp_eq_i32 s19, 0 // (op 13370) 
s_cselect_b32 s49, 0, -1 // (op 13370) 

s_and_b32 s19, s47, 31 // (op 13370) 
s_lshl_b32 s50, 1, s19 // (op 13370) 
s_add_i32 s51, s50, s49 // (op 13370) 


s_and_b32 s49, s1, s51 // (op 13370) 


s_add_i32 s1, s18, s49 // (op 13370) 


s_ashr_i32 s18, s1, s19 // (op 13370) 


s_lshl_b32 s1, s47, 24 // (op 13370) 
s_ashr_i32 s19, s1, 31 // (op 13370) 

s_xor_b32 s1, s18, s19 // (op 13370) 

s_sub_i32 s18, s1, s19 // (op 13370) 


s_add_u32 s50, s6, 128 // (op 13370) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13370) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13370) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13370) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13370) 

s_mul_i32 s1, s44, s50 // (op 13370) 

s_mul_i32 s19, s48, s1 // (op 13370) 

s_mul_i32 s1, s18, s19 // (op 13370) 

s_sub_u32 s19, s16, s1 // (op 13370) 

s_mov_b32 s1, s19 // (op 13370) 
s_mul_hi_i32 s49, s1, s15 // (op 13370) 
s_add_i32 s50, s49, s1 // (op 13370) 


s_ashr_i32 s1, s50, 31 // (op 13370) 
s_cmp_eq_i32 s15, 0 // (op 13370) 
s_mov_b32 s49, scc // (op 13370) 
s_cmp_eq_i32 s49, 0 // (op 13370) 
s_cselect_b32 s51, 0, -1 // (op 13370) 

s_and_b32 s49, s46, 31 // (op 13370) 
s_lshl_b32 s72, 1, s49 // (op 13370) 
s_add_i32 s73, s72, s51 // (op 13370) 


s_and_b32 s51, s1, s73 // (op 13370) 


s_add_i32 s1, s50, s51 // (op 13370) 


s_ashr_i32 s50, s1, s49 // (op 13370) 


s_lshl_b32 s1, s46, 24 // (op 13370) 
s_ashr_i32 s49, s1, 31 // (op 13370) 

s_xor_b32 s1, s50, s49 // (op 13370) 

s_sub_i32 s50, s1, s49 // (op 13370) 


s_mul_i32 s1, s45, s44 // (op 13370) 
s_add_u32 s72, s8, 128 // (op 13370) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 13370) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 13370) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 13370) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 13370) 

s_sub_i32 s49, s72, s1 // (op 13370) 

s_mul_i32 s51, s50, s49 // (op 13370) 


s_sub_u32 s49, s19, s51 // (op 13370) 


s_add_u32 s19, s49, s1 // (op 13370) 


s_cmp_eq_i32 s18, 0 // (op 13370) 
s_mov_b32 s1, scc // (op 13370) 

s_cmp_eq_i32 s1, 0 // (op 13370) 
s_cselect_b32 s18, s19, s0 // (op 13370) 



v_lshl_add_u32 v24, s18, 7, v22 // (op 13370) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13370) 
s_lshr_b32 s18, s0, 2 // (op 13370) 

v_lshl_add_u32 v22, s18, 9, v23 // (op 13370) 


v_mov_b32 v27, 0 // (op 13370) convert
v_mov_b32 v26, v22 // (op 13370) convert

v_mul_lo_u32 v23, v24, s25 // (op 13370) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 13370) most significant: high of low * low
v_mul_lo_u32 v22, v24, s24 // (op 13370) least significant: low of low * low
v_add_u32 v23, v23, v25 // (op 13370) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v26 // (op 13370) least significant half
v_addc_co_u32 v23, s[0:1], v23, v27, s[0:1] // (op 13370) most significant half



v_lshlrev_b64 v[24:25], 2, v[22:23] // (op 13370) 

v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 13370) 

s_mul_i32 s1, s24, s9 // (op 13372) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 13372) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13372) least significant
s_mul_hi_u32 s19, s24, s8 // (op 13372) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13372) most significant: sum
s_add_u32 s1, s1, s19 // (op 13372) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13372) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13372) 

s_mov_b32 s20, s26 // (op 13372) reg expression
s_mov_b32 s21, s27 // (op 13372) reg expression
s_mov_b32 s22, s0 // (op 13372) reg expression
s_mov_b32 s23, 131072 // (op 13372) call()

s_mul_hi_i32 s0, s16, s11 // (op 13452) 
v_and_b32 v24, -4, v1 // (op 13452) 
s_add_i32 s1, s0, s16 // (op 13452) 

v_sub_u32 v25, v1, v24 // (op 13452) 

s_ashr_i32 s0, s1, 31 // (op 13452) 
v_lshlrev_b32 v24, 5, v25 // (op 13452) 

s_cmp_eq_i32 s11, 0 // (op 13452) 
s_mov_b32 s18, scc // (op 13452) 
v_lshrrev_b32 v25, 2, v1 // (op 13452) 
s_cmp_eq_i32 s18, 0 // (op 13452) 
s_cselect_b32 s19, 0, -1 // (op 13452) 

s_and_b32 s18, s12, 31 // (op 13452) 
s_lshl_b32 s49, 1, s18 // (op 13452) 
s_add_i32 s50, s49, s19 // (op 13452) 


s_and_b32 s19, s0, s50 // (op 13452) 


s_add_i32 s0, s1, s19 // (op 13452) 


s_ashr_i32 s1, s0, s18 // (op 13452) 


s_lshl_b32 s0, s12, 24 // (op 13452) 
s_ashr_i32 s18, s0, 31 // (op 13452) 

s_xor_b32 s0, s1, s18 // (op 13452) 

s_sub_i32 s1, s0, s18 // (op 13452) 


s_mul_hi_i32 s0, s1, s13 // (op 13452) 
s_add_i32 s18, s0, s1 // (op 13452) 

s_ashr_i32 s0, s18, 31 // (op 13452) 
s_cmp_eq_i32 s13, 0 // (op 13452) 
s_mov_b32 s19, scc // (op 13452) 
s_cmp_eq_i32 s19, 0 // (op 13452) 
s_cselect_b32 s49, 0, -1 // (op 13452) 

s_and_b32 s19, s14, 31 // (op 13452) 
s_lshl_b32 s50, 1, s19 // (op 13452) 
s_add_i32 s51, s50, s49 // (op 13452) 


s_and_b32 s49, s0, s51 // (op 13452) 


s_add_i32 s0, s18, s49 // (op 13452) 


s_ashr_i32 s18, s0, s19 // (op 13452) 


s_lshl_b32 s0, s14, 24 // (op 13452) 
s_ashr_i32 s19, s0, 31 // (op 13452) 

s_xor_b32 s0, s18, s19 // (op 13452) 

s_sub_i32 s18, s0, s19 // (op 13452) 


s_add_u32 s50, s6, 128 // (op 13452) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13452) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13452) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13452) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13452) 

s_mul_i32 s0, s18, s50 // (op 13452) 

s_sub_i32 s49, s1, s0 // (op 13452) 


s_mul_hi_i32 s0, s16, s10 // (op 13452) 
s_add_i32 s1, s0, s16 // (op 13452) 

s_ashr_i32 s0, s1, 31 // (op 13452) 
s_cmp_eq_i32 s10, 0 // (op 13452) 
s_mov_b32 s18, scc // (op 13452) 
s_cmp_eq_i32 s18, 0 // (op 13452) 
s_cselect_b32 s19, 0, -1 // (op 13452) 

s_and_b32 s18, s47, 31 // (op 13452) 
s_lshl_b32 s72, 1, s18 // (op 13452) 
s_add_i32 s73, s72, s19 // (op 13452) 


s_and_b32 s19, s0, s73 // (op 13452) 


s_add_i32 s0, s1, s19 // (op 13452) 


s_ashr_i32 s1, s0, s18 // (op 13452) 


s_lshl_b32 s0, s47, 24 // (op 13452) 
s_ashr_i32 s18, s0, 31 // (op 13452) 

s_xor_b32 s0, s1, s18 // (op 13452) 

s_sub_i32 s1, s0, s18 // (op 13452) 


s_mul_i32 s0, s44, s50 // (op 13452) 

s_mul_i32 s18, s48, s0 // (op 13452) 

s_mul_i32 s0, s1, s18 // (op 13452) 

s_sub_u32 s18, s16, s0 // (op 13452) 

s_mul_hi_i32 s0, s18, s15 // (op 13452) 
s_add_i32 s19, s0, s18 // (op 13452) 


s_ashr_i32 s0, s19, 31 // (op 13452) 
s_cmp_eq_i32 s15, 0 // (op 13452) 
s_mov_b32 s18, scc // (op 13452) 
s_cmp_eq_i32 s18, 0 // (op 13452) 
s_cselect_b32 s50, 0, -1 // (op 13452) 

s_and_b32 s18, s46, 31 // (op 13452) 
s_lshl_b32 s51, 1, s18 // (op 13452) 
s_add_i32 s72, s51, s50 // (op 13452) 


s_and_b32 s50, s0, s72 // (op 13452) 


s_add_i32 s0, s19, s50 // (op 13452) 


s_ashr_i32 s19, s0, s18 // (op 13452) 


s_lshl_b32 s0, s46, 24 // (op 13452) 
s_ashr_i32 s18, s0, 31 // (op 13452) 

s_xor_b32 s0, s19, s18 // (op 13452) 

s_sub_i32 s19, s0, s18 // (op 13452) 


s_cmp_eq_i32 s1, 0 // (op 13452) 
s_mov_b32 s0, scc // (op 13452) 

s_cmp_eq_i32 s0, 0 // (op 13452) 
s_cselect_b32 s1, s19, s49 // (op 13452) 



v_lshl_add_u32 v26, s1, 7, v24 // (op 13452) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13452) 
v_mov_b32 v31, 0 // (op 13452) convert
v_mov_b32 v30, v26 // (op 13452) convert

s_lshr_b32 s18, s0, 2 // (op 13452) 

v_lshl_add_u32 v24, s18, 9, v25 // (op 13452) 


v_mul_lo_u32 v27, v24, s29 // (op 13452) most significant: low of low * high
v_mul_hi_u32 v25, v24, s28 // (op 13452) most significant: high of low * low
v_mul_lo_u32 v26, v24, s28 // (op 13452) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 13452) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v30 // (op 13452) least significant half
v_addc_co_u32 v27, s[0:1], v27, v31, s[0:1] // (op 13452) most significant half



v_lshlrev_b64 v[24:25], 2, v[26:27] // (op 13452) 

v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 13452) 

s_mul_i32 s1, s28, s33 // (op 13454) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 13454) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13454) least significant
s_mul_hi_u32 s19, s28, s32 // (op 13454) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13454) most significant: sum
s_add_u32 s1, s1, s19 // (op 13454) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13454) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13454) 

s_mov_b32 s52, s30 // (op 13454) reg expression
s_mov_b32 s53, s31 // (op 13454) reg expression
s_mov_b32 s54, s0 // (op 13454) reg expression
s_mov_b32 s55, 131072 // (op 13454) call()

s_mul_hi_i32 s0, s16, s11 // (op 13516) 
v_lshrrev_b32 v24, 2, v1 // (op 13516) 
s_add_i32 s1, s0, s16 // (op 13516) 

v_lshlrev_b32 v25, 1, v24 // (op 13516) 

s_ashr_i32 s0, s1, 31 // (op 13516) 
v_and_b32 v24, -4, v1 // (op 13516) 
s_cmp_eq_i32 s11, 0 // (op 13516) 
s_mov_b32 s18, scc // (op 13516) 
v_sub_u32 v30, v1, v24 // (op 13516) 

s_cmp_eq_i32 s18, 0 // (op 13516) 
s_cselect_b32 s19, 0, -1 // (op 13516) 

s_and_b32 s18, s12, 31 // (op 13516) 
s_lshl_b32 s49, 1, s18 // (op 13516) 
s_add_i32 s50, s49, s19 // (op 13516) 


s_and_b32 s19, s0, s50 // (op 13516) 


s_add_i32 s0, s1, s19 // (op 13516) 


s_ashr_i32 s1, s0, s18 // (op 13516) 


s_lshl_b32 s0, s12, 24 // (op 13516) 
s_ashr_i32 s18, s0, 31 // (op 13516) 

s_xor_b32 s0, s1, s18 // (op 13516) 

s_sub_i32 s1, s0, s18 // (op 13516) 


s_mul_hi_i32 s0, s1, s13 // (op 13516) 
s_add_i32 s18, s0, s1 // (op 13516) 

s_ashr_i32 s0, s18, 31 // (op 13516) 
s_cmp_eq_i32 s13, 0 // (op 13516) 
s_mov_b32 s19, scc // (op 13516) 
s_cmp_eq_i32 s19, 0 // (op 13516) 
s_cselect_b32 s49, 0, -1 // (op 13516) 

s_and_b32 s19, s14, 31 // (op 13516) 
s_lshl_b32 s50, 1, s19 // (op 13516) 
s_add_i32 s51, s50, s49 // (op 13516) 


s_and_b32 s49, s0, s51 // (op 13516) 


s_add_i32 s0, s18, s49 // (op 13516) 


s_ashr_i32 s18, s0, s19 // (op 13516) 


s_lshl_b32 s0, s14, 24 // (op 13516) 
s_ashr_i32 s19, s0, 31 // (op 13516) 

s_xor_b32 s0, s18, s19 // (op 13516) 

s_sub_i32 s18, s0, s19 // (op 13516) 


s_mul_i32 s0, s1, s44 // (op 13516) 

s_sub_u32 s1, s16, s0 // (op 13516) 

s_mul_i32 s0, s18, s44 // (op 13516) 
s_add_u32 s0, s0, s1 // (op 13516) 


s_mul_hi_i32 s1, s16, s10 // (op 13516) 
s_add_i32 s18, s1, s16 // (op 13516) 

s_ashr_i32 s1, s18, 31 // (op 13516) 
s_cmp_eq_i32 s10, 0 // (op 13516) 
s_mov_b32 s19, scc // (op 13516) 
s_cmp_eq_i32 s19, 0 // (op 13516) 
s_cselect_b32 s49, 0, -1 // (op 13516) 

s_and_b32 s19, s47, 31 // (op 13516) 
s_lshl_b32 s50, 1, s19 // (op 13516) 
s_add_i32 s51, s50, s49 // (op 13516) 


s_and_b32 s49, s1, s51 // (op 13516) 


s_add_i32 s1, s18, s49 // (op 13516) 


s_ashr_i32 s18, s1, s19 // (op 13516) 


s_lshl_b32 s1, s47, 24 // (op 13516) 
s_ashr_i32 s19, s1, 31 // (op 13516) 

s_xor_b32 s1, s18, s19 // (op 13516) 

s_sub_i32 s18, s1, s19 // (op 13516) 


s_add_u32 s50, s6, 128 // (op 13516) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13516) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13516) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13516) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13516) 

s_mul_i32 s1, s44, s50 // (op 13516) 

s_mul_i32 s19, s48, s1 // (op 13516) 

s_mul_i32 s1, s18, s19 // (op 13516) 

s_sub_u32 s19, s16, s1 // (op 13516) 

s_mov_b32 s1, s19 // (op 13516) 
s_mul_hi_i32 s49, s1, s15 // (op 13516) 
s_add_i32 s50, s49, s1 // (op 13516) 


s_ashr_i32 s1, s50, 31 // (op 13516) 
s_cmp_eq_i32 s15, 0 // (op 13516) 
s_mov_b32 s49, scc // (op 13516) 
s_cmp_eq_i32 s49, 0 // (op 13516) 
s_cselect_b32 s51, 0, -1 // (op 13516) 

s_and_b32 s49, s46, 31 // (op 13516) 
s_lshl_b32 s72, 1, s49 // (op 13516) 
s_add_i32 s73, s72, s51 // (op 13516) 


s_and_b32 s51, s1, s73 // (op 13516) 


s_add_i32 s1, s50, s51 // (op 13516) 


s_ashr_i32 s50, s1, s49 // (op 13516) 


s_lshl_b32 s1, s46, 24 // (op 13516) 
s_ashr_i32 s49, s1, 31 // (op 13516) 

s_xor_b32 s1, s50, s49 // (op 13516) 

s_sub_i32 s50, s1, s49 // (op 13516) 


s_mul_i32 s1, s45, s44 // (op 13516) 
s_add_u32 s72, s8, 128 // (op 13516) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 13516) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 13516) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 13516) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 13516) 

s_sub_i32 s49, s72, s1 // (op 13516) 

s_mul_i32 s51, s50, s49 // (op 13516) 


s_sub_u32 s49, s19, s51 // (op 13516) 


s_add_u32 s19, s49, s1 // (op 13516) 


s_cmp_eq_i32 s18, 0 // (op 13516) 
s_mov_b32 s1, scc // (op 13516) 

s_cmp_eq_i32 s1, 0 // (op 13516) 
s_cselect_b32 s18, s19, s0 // (op 13516) 



v_lshl_add_u32 v24, s18, 7, v25 // (op 13516) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13516) 
s_lshr_b32 s18, s0, 2 // (op 13516) 

v_lshl_add_u32 v25, s18, 4, v30 // (op 13516) 


v_mul_lo_u32 v30, v24, s34 // (op 13516) 
v_add_u32 v30, v30, v25 // (op 13516) 


s_mul_i32 s1, s34, s9 // (op 13518) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 13518) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13518) least significant
s_mul_hi_u32 s19, s34, s8 // (op 13518) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13518) most significant: sum
s_add_u32 s1, s1, s19 // (op 13518) most significant: sum


s_mov_b32 s64, s36 // (op 13518) reg expression
s_mov_b32 s65, s37 // (op 13518) reg expression
s_mov_b32 s66, s0 // (op 13518) reg expression
s_mov_b32 s67, 131072 // (op 13518) call()

s_mul_hi_i32 s0, s16, s11 // (op 13580) 
v_and_b32 v24, -128, v1 // (op 13580) 
s_add_i32 s1, s0, s16 // (op 13580) 

v_sub_u32 v25, v1, v24 // (op 13580) 

s_ashr_i32 s0, s1, 31 // (op 13580) 
v_lshrrev_b32 v24, 7, v1 // (op 13580) 
s_cmp_eq_i32 s11, 0 // (op 13580) 
s_mov_b32 s18, scc // (op 13580) 
v_lshlrev_b32 v31, 1, v24 // (op 13580) 

s_cmp_eq_i32 s18, 0 // (op 13580) 
s_cselect_b32 s19, 0, -1 // (op 13580) 

s_and_b32 s18, s12, 31 // (op 13580) 
s_lshl_b32 s49, 1, s18 // (op 13580) 
s_add_i32 s50, s49, s19 // (op 13580) 


s_and_b32 s19, s0, s50 // (op 13580) 


s_add_i32 s0, s1, s19 // (op 13580) 


s_ashr_i32 s1, s0, s18 // (op 13580) 


s_lshl_b32 s0, s12, 24 // (op 13580) 
s_ashr_i32 s18, s0, 31 // (op 13580) 

s_xor_b32 s0, s1, s18 // (op 13580) 

s_sub_i32 s1, s0, s18 // (op 13580) 


s_mul_hi_i32 s0, s1, s13 // (op 13580) 
s_add_i32 s18, s0, s1 // (op 13580) 

s_ashr_i32 s0, s18, 31 // (op 13580) 
s_cmp_eq_i32 s13, 0 // (op 13580) 
s_mov_b32 s19, scc // (op 13580) 
s_cmp_eq_i32 s19, 0 // (op 13580) 
s_cselect_b32 s49, 0, -1 // (op 13580) 

s_and_b32 s19, s14, 31 // (op 13580) 
s_lshl_b32 s50, 1, s19 // (op 13580) 
s_add_i32 s51, s50, s49 // (op 13580) 


s_and_b32 s49, s0, s51 // (op 13580) 


s_add_i32 s0, s18, s49 // (op 13580) 


s_ashr_i32 s18, s0, s19 // (op 13580) 


s_lshl_b32 s0, s14, 24 // (op 13580) 
s_ashr_i32 s19, s0, 31 // (op 13580) 

s_xor_b32 s0, s18, s19 // (op 13580) 

s_sub_i32 s18, s0, s19 // (op 13580) 


s_add_u32 s50, s6, 128 // (op 13580) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 13580) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 13580) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 13580) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 13580) 

s_mul_i32 s0, s18, s50 // (op 13580) 

s_sub_i32 s49, s1, s0 // (op 13580) 


s_mul_hi_i32 s0, s16, s10 // (op 13580) 
s_add_i32 s1, s0, s16 // (op 13580) 

s_ashr_i32 s0, s1, 31 // (op 13580) 
s_cmp_eq_i32 s10, 0 // (op 13580) 
s_mov_b32 s18, scc // (op 13580) 
s_cmp_eq_i32 s18, 0 // (op 13580) 
s_cselect_b32 s19, 0, -1 // (op 13580) 

s_and_b32 s18, s47, 31 // (op 13580) 
s_lshl_b32 s72, 1, s18 // (op 13580) 
s_add_i32 s73, s72, s19 // (op 13580) 


s_and_b32 s19, s0, s73 // (op 13580) 


s_add_i32 s0, s1, s19 // (op 13580) 


s_ashr_i32 s1, s0, s18 // (op 13580) 


s_lshl_b32 s0, s47, 24 // (op 13580) 
s_ashr_i32 s18, s0, 31 // (op 13580) 

s_xor_b32 s0, s1, s18 // (op 13580) 

s_sub_i32 s1, s0, s18 // (op 13580) 


s_mul_i32 s0, s44, s50 // (op 13580) 

s_mul_i32 s18, s48, s0 // (op 13580) 

s_mul_i32 s0, s1, s18 // (op 13580) 

s_sub_u32 s18, s16, s0 // (op 13580) 

s_mul_hi_i32 s0, s18, s15 // (op 13580) 
s_add_i32 s19, s0, s18 // (op 13580) 


s_ashr_i32 s0, s19, 31 // (op 13580) 
s_cmp_eq_i32 s15, 0 // (op 13580) 
s_mov_b32 s18, scc // (op 13580) 
s_cmp_eq_i32 s18, 0 // (op 13580) 
s_cselect_b32 s50, 0, -1 // (op 13580) 

s_and_b32 s18, s46, 31 // (op 13580) 
s_lshl_b32 s51, 1, s18 // (op 13580) 
s_add_i32 s72, s51, s50 // (op 13580) 


s_and_b32 s50, s0, s72 // (op 13580) 


s_add_i32 s0, s19, s50 // (op 13580) 


s_ashr_i32 s19, s0, s18 // (op 13580) 


s_lshl_b32 s0, s46, 24 // (op 13580) 
s_ashr_i32 s18, s0, 31 // (op 13580) 

s_xor_b32 s0, s19, s18 // (op 13580) 

s_sub_i32 s19, s0, s18 // (op 13580) 


s_cmp_eq_i32 s1, 0 // (op 13580) 
s_mov_b32 s0, scc // (op 13580) 

s_cmp_eq_i32 s0, 0 // (op 13580) 
s_cselect_b32 s1, s19, s49 // (op 13580) 



v_lshl_add_u32 v24, s1, 7, v25 // (op 13580) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 13580) 
s_lshr_b32 s18, s0, 2 // (op 13580) 

v_lshl_add_u32 v25, s18, 4, v31 // (op 13580) 


v_mul_lo_u32 v31, v25, s38 // (op 13580) 
v_add_u32 v31, v31, v24 // (op 13580) 


s_mul_i32 s1, s38, s43 // (op 13582) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 13582) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 13582) least significant
s_mul_hi_u32 s19, s38, s42 // (op 13582) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13582) most significant: sum
s_add_u32 s1, s1, s19 // (op 13582) most significant: sum


s_mov_b32 s68, s40 // (op 13582) reg expression
s_mov_b32 s69, s41 // (op 13582) reg expression
s_mov_b32 s70, s0 // (op 13582) reg expression
s_mov_b32 s71, 131072 // (op 13582) call()

v_and_b32 v24, -4, v1 // (op 13678) 
v_sub_u32 v25, v1, v24 // (op 13678) 

v_lshlrev_b32 v24, 5, v25 // (op 13678) 

v_lshrrev_b32 v25, 2, v1 // (op 13678) 
v_lshl_add_u32 v32, v25, 7, v24 // (op 13678) 


v_lshlrev_b32 v24, 2, v32 // (op 13678) 

v_lshrrev_b32 v25, 3, v24 // (op 13678) 

s_nop 0
v_readfirstlane_b32 s0, v25 // (op 13678) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 13735) 
v_sub_u32 v25, v1, v24 // (op 13735) 

v_lshlrev_b32 v24, 5, v25 // (op 13735) 

v_lshrrev_b32 v25, 2, v1 // (op 13735) 
v_lshl_add_u32 v32, v25, 7, v24 // (op 13735) 


v_lshlrev_b32 v24, 2, v32 // (op 13735) 

v_lshrrev_b32 v25, 3, v24 // (op 13735) 

s_nop 0
v_readfirstlane_b32 s1, v25 // (op 13735) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 13792) 
v_sub_u32 v25, v1, v24 // (op 13792) 

v_lshrrev_b32 v24, 2, v1 // (op 13792) 
v_lshl_add_u32 v32, v24, 3, v25 // (op 13792) 


v_and_b32 v24, -128, v1 // (op 13850) 
v_sub_u32 v25, v1, v24 // (op 13850) 

v_lshrrev_b32 v24, 7, v1 // (op 13850) 
v_lshl_add_u32 v33, v24, 8, v25 // (op 13850) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 4192) 
s_and_b32 s49, s18, 4294967292 // (op 4192) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 4191) 
s_mov_b32 s50, s18 // (op 4191) convert

s_cmp_lt_u32 s49, s50 // (op 4191) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4191 // (op 4191) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4191 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4191:
s_barrier  // (op 14555) 
s_add_u32 s49, s49, 1 // (op 4193) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 13294) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13294) 

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 13294) 


v_mov_b64 v[28:29], v[22:23] // (op 13294) 
v_mov_b64 v[34:35], v[28:29] // (op 13294) 
s_add_u32 s18, 8192, s0 // (op 13294) 
s_mov_b32 m0, s18 // (op 13294) reg expression
buffer_load_dwordx4 v34, s[20:23], 0 offen offset: 0 lds // (op 13294) Load value direct to lds
v_add_co_u32 v34, s[50:51], v34, v24 // (op 13294) least significant half
v_addc_co_u32 v35, s[50:51], 0, v35, s[50:51] // (op 13294) most significant half

s_add_u32 m0, m0, 4096 // (op 13294) 
buffer_load_dwordx4 v34, s[20:23], 0 offen offset: 0 lds // (op 13294) Load value direct to lds




v_add_co_u32 v22, s[18:19], 64, v22 // (op 13360) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 13360) most significant half

s_barrier  // (op 14571) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 13300) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13300) 

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 13300) 


v_mov_b64 v[28:29], v[26:27] // (op 13300) 
v_mov_b64 v[34:35], v[28:29] // (op 13300) reg expression
s_mov_b32 m0, s1 // (op 13300) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13300) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v24 // (op 13300) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 13300) most significant half

s_add_u32 m0, m0, 4096 // (op 13300) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13300) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 13444) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 13444) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 13444) 

v_mov_b32 v24, s18 // (op 13444) 
v_mov_b32 v25, s19 // (op 13444) 
v_add_co_u32 v26, s[50:51], v26, v24 // (op 13444) least significant half
v_addc_co_u32 v27, s[50:51], v27, v25, s[50:51] // (op 13444) most significant half





v_mov_b64 v[24:25], s[34:35] // (op 191) convert

v_mov_b32 v28, v30 // (op 191) 
v_mov_b32 v29, v28 // (op 191) 
buffer_load_ubyte v34, v29, s[64:67], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v29, v29, v24 // (op 191) 
buffer_load_ubyte v35, v29, s[64:67], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v30, 4, v30 // (op 13508) 

s_barrier  // (op 14588) 
v_mov_b64 v[24:25], s[38:39] // (op 192) convert




v_mov_b32 v28, v31 // (op 192) 
v_mov_b32 v29, v28 // (op 192) reg expression
v_add_u32 v36, 66048, v32 // (op 188) 
s_waitcnt vmcnt(1)
ds_write_b8 v36, v34 // (op 188) Store local data 

v_add_u32 v36, 66052, v32 // (op 188) 
buffer_load_ubyte v38, v29, s[68:71], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v29, v29, v24 // (op 192) 
buffer_load_ubyte v39, v29, s[68:71], 0 offen offset: 0 // (op 192) Load value



s_lshl_b32 s18, s38, 2 // (op 13572) 
v_add_u32 v31, v31, s18 // (op 13572) 

s_waitcnt vmcnt(2)
ds_write_b8 v36, v35 // (op 188) Store local data 


s_barrier  // (op 14561) 



s_barrier  // (op 14606) 
ds_read_b128 v[34:37], v3 offset:8192 // (op 179) Load local data 



s_barrier  // (op 14578) 
v_add_u32 v24, 65536, v33 // (op 189) 



ds_read_b64_tr_b4 v[40:41], v4 // (op 181) Transpose load local data 
ds_read_b64_tr_b4 v[42:43], v4 offset:1024 // (op 181) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v38 // (op 189) Store local data 

v_add_u32 v24, 65664, v33 // (op 189) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 14596) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v39 // (op 189) Store local data 





v_add_u32 v24, 66048, v0 // (op 183) 
ds_read_u8 v25, v24 // (op 183) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 14613) 


v_add_u32 v24, 65536, v5 // (op 185) 
ds_read_u8 v28, v24 // (op 185) Load local data 




ds_read_b64_tr_b4 v[44:45], v4 offset:16 // (op 496) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v4 offset:1040 // (op 496) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[40:43], a[0:3], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 


v_add_u32 v24, 65568, v5 // (op 500) 
ds_read_u8 v29, v24 // (op 500) Load local data 




ds_read_b64_tr_b4 v[48:49], v4 offset:32 // (op 794) Transpose load local data 
ds_read_b64_tr_b4 v[50:51], v4 offset:1056 // (op 794) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[44:47], a[4:7], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 


v_add_u32 v24, 65600, v5 // (op 798) 
ds_read_u8 v38, v24 // (op 798) Load local data 




ds_read_b64_tr_b4 v[52:53], v4 offset:48 // (op 1092) Transpose load local data 
ds_read_b64_tr_b4 v[54:55], v4 offset:1072 // (op 1092) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[48:51], a[8:11], v25, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 799) 


v_add_u32 v24, 65632, v5 // (op 1096) 
ds_read_u8 v39, v24 // (op 1096) Load local data 




ds_read_b128 v[56:59], v3 offset:10240 // (op 1524) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[52:55], a[12:15], v25, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 





v_add_u32 v24, 66176, v0 // (op 1527) 
ds_read_u8 v25, v24 // (op 1527) Load local data 




ds_read_b128 v[34:37], v3 offset:12288 // (op 2814) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[40:43], a[16:19], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1529) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1560) 
v_add_u32 v24, 66304, v0 // (op 2817) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v25, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1591) 
ds_read_u8 v60, v24 // (op 2817) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[52:55], a[28:31], v25, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1622) 
ds_read_b128 v[62:65], v3 offset:14336 // (op 4104) Load local data 





v_add_u32 v61, 66432, v0 // (op 4107) 
ds_read_u8 v24, v61 // (op 4107) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[40:43], a[32:35], v60, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v60, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2850) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v60, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2881) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[52:55], a[44:47], v60, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2912) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[40:43], a[48:51], v24, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4109) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[62:65], v[44:47], a[52:55], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4140) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:65], v[48:51], a[56:59], v24, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4171) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[62:65], v[52:55], a[60:63], v24, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4202) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 4191) 
s_mov_b32 s50, s18 // (op 4191) convert

s_cmp_lt_u32 s49, s50 // (op 4191) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4191 // (op 4191) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_4191 if true)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_4191:























s_barrier  // (op 14787) 





























v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_133
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


v_accvgpr_read v14, a4 // (op 5873) 
v_accvgpr_read v15, a5 // (op 5873) 
v_accvgpr_read v16, a6 // (op 5873) 
v_accvgpr_read v17, a7 // (op 5873) 
v_mul_f32 v10, v19, v14 // (op 5873) 
v_mul_f32 v11, v19, v15 // (op 5873) 
v_mul_f32 v12, v19, v16 // (op 5873) 
v_mul_f32 v13, v19, v17 // (op 5873) 



s_lshl_b32 s0, s4, 7 // (op 5875) 
v_add_u32 v0, v21, s0 // (op 5875) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5875) Store value


v_accvgpr_read v14, a8 // (op 5885) 
v_accvgpr_read v15, a9 // (op 5885) 
v_accvgpr_read v16, a10 // (op 5885) 
v_accvgpr_read v17, a11 // (op 5885) 
v_mul_f32 v10, v19, v14 // (op 5885) 
v_mul_f32 v11, v19, v15 // (op 5885) 
v_mul_f32 v12, v19, v16 // (op 5885) 
v_mul_f32 v13, v19, v17 // (op 5885) 



s_lshl_b32 s0, s4, 8 // (op 5887) 
v_add_u32 v0, v21, s0 // (op 5887) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5887) Store value


v_accvgpr_read v14, a12 // (op 5897) 
v_accvgpr_read v15, a13 // (op 5897) 
v_accvgpr_read v16, a14 // (op 5897) 
v_accvgpr_read v17, a15 // (op 5897) 
v_mul_f32 v10, v19, v14 // (op 5897) 
v_mul_f32 v11, v19, v15 // (op 5897) 
v_mul_f32 v12, v19, v16 // (op 5897) 
v_mul_f32 v13, v19, v17 // (op 5897) 



s_lshl_b32 s0, s4, 7 // (op 5899) 
s_mul_i32 s1, 3, s0 // (op 5899) 

v_add_u32 v0, v21, s1 // (op 5899) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5899) Store value


v_accvgpr_read v14, a16 // (op 5931) 
v_accvgpr_read v15, a17 // (op 5931) 
v_accvgpr_read v16, a18 // (op 5931) 
v_accvgpr_read v17, a19 // (op 5931) 
v_mul_f32 v10, v19, v14 // (op 5931) 
v_mul_f32 v11, v19, v15 // (op 5931) 
v_mul_f32 v12, v19, v16 // (op 5931) 
v_mul_f32 v13, v19, v17 // (op 5931) 



v_add_u32 v0, 128, v21 // (op 5934) 

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5934) Store value


v_accvgpr_read v14, a20 // (op 5944) 
v_accvgpr_read v15, a21 // (op 5944) 
v_accvgpr_read v16, a22 // (op 5944) 
v_accvgpr_read v17, a23 // (op 5944) 
v_mul_f32 v10, v19, v14 // (op 5944) 
v_mul_f32 v11, v19, v15 // (op 5944) 
v_mul_f32 v12, v19, v16 // (op 5944) 
v_mul_f32 v13, v19, v17 // (op 5944) 



s_lshl_b32 s0, s4, 7 // (op 5936) 
s_add_u32 s0, s0, 128 // (op 5936) 
v_add_u32 v0, v21, s0 // (op 5936) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5936) Store value


v_accvgpr_read v14, a24 // (op 5949) 
v_accvgpr_read v15, a25 // (op 5949) 
v_accvgpr_read v16, a26 // (op 5949) 
v_accvgpr_read v17, a27 // (op 5949) 
v_mul_f32 v10, v19, v14 // (op 5949) 
v_mul_f32 v11, v19, v15 // (op 5949) 
v_mul_f32 v12, v19, v16 // (op 5949) 
v_mul_f32 v13, v19, v17 // (op 5949) 



s_lshl_b32 s0, s4, 8 // (op 5938) 
s_add_u32 s0, s0, 128 // (op 5938) 
v_add_u32 v0, v21, s0 // (op 5938) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5938) Store value


v_accvgpr_read v14, a28 // (op 5954) 
v_accvgpr_read v15, a29 // (op 5954) 
v_accvgpr_read v16, a30 // (op 5954) 
v_accvgpr_read v17, a31 // (op 5954) 
v_mul_f32 v10, v19, v14 // (op 5954) 
v_mul_f32 v11, v19, v15 // (op 5954) 
v_mul_f32 v12, v19, v16 // (op 5954) 
v_mul_f32 v13, v19, v17 // (op 5954) 



s_lshl_b32 s0, s4, 7 // (op 5940) 
s_mul_i32 s1, 3, s0 // (op 5940) 
s_add_u32 s1, s1, 128 // (op 5940) 

v_add_u32 v0, v21, s1 // (op 5940) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 5940) Store value


v_accvgpr_read v14, a32 // (op 5999) 
v_accvgpr_read v15, a33 // (op 5999) 
v_accvgpr_read v16, a34 // (op 5999) 
v_accvgpr_read v17, a35 // (op 5999) 
v_mul_f32 v10, v19, v14 // (op 5999) 
v_mul_f32 v11, v19, v15 // (op 5999) 
v_mul_f32 v12, v19, v16 // (op 5999) 
v_mul_f32 v13, v19, v17 // (op 5999) 



v_add_u32 v0, 256, v21 // (op 6002) 

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6002) Store value


v_accvgpr_read v14, a36 // (op 6012) 
v_accvgpr_read v15, a37 // (op 6012) 
v_accvgpr_read v16, a38 // (op 6012) 
v_accvgpr_read v17, a39 // (op 6012) 
v_mul_f32 v10, v19, v14 // (op 6012) 
v_mul_f32 v11, v19, v15 // (op 6012) 
v_mul_f32 v12, v19, v16 // (op 6012) 
v_mul_f32 v13, v19, v17 // (op 6012) 



s_lshl_b32 s0, s4, 7 // (op 6004) 
s_add_u32 s0, s0, 256 // (op 6004) 
v_add_u32 v0, v21, s0 // (op 6004) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6004) Store value


v_accvgpr_read v14, a40 // (op 6017) 
v_accvgpr_read v15, a41 // (op 6017) 
v_accvgpr_read v16, a42 // (op 6017) 
v_accvgpr_read v17, a43 // (op 6017) 
v_mul_f32 v10, v19, v14 // (op 6017) 
v_mul_f32 v11, v19, v15 // (op 6017) 
v_mul_f32 v12, v19, v16 // (op 6017) 
v_mul_f32 v13, v19, v17 // (op 6017) 



s_lshl_b32 s0, s4, 8 // (op 6006) 
s_add_u32 s0, s0, 256 // (op 6006) 
v_add_u32 v0, v21, s0 // (op 6006) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6006) Store value


v_accvgpr_read v14, a44 // (op 6022) 
v_accvgpr_read v15, a45 // (op 6022) 
v_accvgpr_read v16, a46 // (op 6022) 
v_accvgpr_read v17, a47 // (op 6022) 
v_mul_f32 v10, v19, v14 // (op 6022) 
v_mul_f32 v11, v19, v15 // (op 6022) 
v_mul_f32 v12, v19, v16 // (op 6022) 
v_mul_f32 v13, v19, v17 // (op 6022) 



s_lshl_b32 s0, s4, 7 // (op 6008) 
s_mul_i32 s1, 3, s0 // (op 6008) 
s_add_u32 s1, s1, 256 // (op 6008) 

v_add_u32 v0, v21, s1 // (op 6008) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6008) Store value


v_accvgpr_read v14, a48 // (op 6067) 
v_accvgpr_read v15, a49 // (op 6067) 
v_accvgpr_read v16, a50 // (op 6067) 
v_accvgpr_read v17, a51 // (op 6067) 
v_mul_f32 v10, v19, v14 // (op 6067) 
v_mul_f32 v11, v19, v15 // (op 6067) 
v_mul_f32 v12, v19, v16 // (op 6067) 
v_mul_f32 v13, v19, v17 // (op 6067) 



v_add_u32 v0, 384, v21 // (op 6070) 

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6070) Store value


v_accvgpr_read v14, a52 // (op 6080) 
v_accvgpr_read v15, a53 // (op 6080) 
v_accvgpr_read v16, a54 // (op 6080) 
v_accvgpr_read v17, a55 // (op 6080) 
v_mul_f32 v10, v19, v14 // (op 6080) 
v_mul_f32 v11, v19, v15 // (op 6080) 
v_mul_f32 v12, v19, v16 // (op 6080) 
v_mul_f32 v13, v19, v17 // (op 6080) 



s_lshl_b32 s0, s4, 7 // (op 6072) 
s_add_u32 s0, s0, 384 // (op 6072) 
v_add_u32 v0, v21, s0 // (op 6072) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6072) Store value


v_accvgpr_read v14, a56 // (op 6085) 
v_accvgpr_read v15, a57 // (op 6085) 
v_accvgpr_read v16, a58 // (op 6085) 
v_accvgpr_read v17, a59 // (op 6085) 
v_mul_f32 v10, v19, v14 // (op 6085) 
v_mul_f32 v11, v19, v15 // (op 6085) 
v_mul_f32 v12, v19, v16 // (op 6085) 
v_mul_f32 v13, v19, v17 // (op 6085) 



s_lshl_b32 s0, s4, 8 // (op 6074) 
s_add_u32 s0, s0, 384 // (op 6074) 
v_add_u32 v0, v21, s0 // (op 6074) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6074) Store value


v_accvgpr_read v14, a60 // (op 6090) 
v_accvgpr_read v15, a61 // (op 6090) 
v_accvgpr_read v16, a62 // (op 6090) 
v_accvgpr_read v17, a63 // (op 6090) 
v_mul_f32 v10, v19, v14 // (op 6090) 
v_mul_f32 v11, v19, v15 // (op 6090) 
v_mul_f32 v12, v19, v16 // (op 6090) 
v_mul_f32 v13, v19, v17 // (op 6090) 



s_lshl_b32 s0, s4, 7 // (op 6076) 
s_mul_i32 s1, 3, s0 // (op 6076) 
s_add_u32 s1, s1, 384 // (op 6076) 

v_add_u32 v0, v21, s1 // (op 6076) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 6076) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 384, v20 // (op 6434) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6434) Load value

v_accvgpr_read v10, a48 // (op 6422) 
v_accvgpr_read v11, a49 // (op 6422) 
v_accvgpr_read v12, a50 // (op 6422) 
v_accvgpr_read v13, a51 // (op 6422) 
v_mul_f32 v6, v19, v10 // (op 6422) 
v_mul_f32 v7, v19, v11 // (op 6422) 
v_mul_f32 v8, v19, v12 // (op 6422) 
v_mul_f32 v9, v19, v13 // (op 6422) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6433) 
v_mul_f32 v11, v18, v15 // (op 6433) 
v_mul_f32 v12, v18, v16 // (op 6433) 
v_mul_f32 v13, v18, v17 // (op 6433) 

v_add_f32 v14, v10, v6 // (op 6423) 
v_add_f32 v15, v11, v7 // (op 6423) 
v_add_f32 v16, v12, v8 // (op 6423) 
v_add_f32 v17, v13, v9 // (op 6423) 




v_add_u32 v0, 384, v21 // (op 6425) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6425) Store value




s_lshl_b32 s0, s2, 7 // (op 6438) 
s_add_u32 s0, s0, 384 // (op 6438) 
v_add_u32 v0, v20, s0 // (op 6438) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6438) Load value

v_accvgpr_read v10, a52 // (op 6448) 
v_accvgpr_read v11, a53 // (op 6448) 
v_accvgpr_read v12, a54 // (op 6448) 
v_accvgpr_read v13, a55 // (op 6448) 
v_mul_f32 v6, v19, v10 // (op 6448) 
v_mul_f32 v7, v19, v11 // (op 6448) 
v_mul_f32 v8, v19, v12 // (op 6448) 
v_mul_f32 v9, v19, v13 // (op 6448) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6436) 
v_mul_f32 v11, v18, v15 // (op 6436) 
v_mul_f32 v12, v18, v16 // (op 6436) 
v_mul_f32 v13, v18, v17 // (op 6436) 

v_add_f32 v14, v10, v6 // (op 6437) 
v_add_f32 v15, v11, v7 // (op 6437) 
v_add_f32 v16, v12, v8 // (op 6437) 
v_add_f32 v17, v13, v9 // (op 6437) 




s_lshl_b32 s0, s4, 7 // (op 6427) 
s_add_u32 s0, s0, 384 // (op 6427) 
v_add_u32 v0, v21, s0 // (op 6427) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6427) Store value




s_lshl_b32 s0, s2, 8 // (op 6442) 
s_add_u32 s0, s0, 384 // (op 6442) 
v_add_u32 v0, v20, s0 // (op 6442) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6442) Load value

v_accvgpr_read v10, a56 // (op 6450) 
v_accvgpr_read v11, a57 // (op 6450) 
v_accvgpr_read v12, a58 // (op 6450) 
v_accvgpr_read v13, a59 // (op 6450) 
v_mul_f32 v6, v19, v10 // (op 6450) 
v_mul_f32 v7, v19, v11 // (op 6450) 
v_mul_f32 v8, v19, v12 // (op 6450) 
v_mul_f32 v9, v19, v13 // (op 6450) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6440) 
v_mul_f32 v11, v18, v15 // (op 6440) 
v_mul_f32 v12, v18, v16 // (op 6440) 
v_mul_f32 v13, v18, v17 // (op 6440) 

v_add_f32 v14, v10, v6 // (op 6441) 
v_add_f32 v15, v11, v7 // (op 6441) 
v_add_f32 v16, v12, v8 // (op 6441) 
v_add_f32 v17, v13, v9 // (op 6441) 




s_lshl_b32 s0, s4, 8 // (op 6429) 
s_add_u32 s0, s0, 384 // (op 6429) 
v_add_u32 v0, v21, s0 // (op 6429) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6429) Store value




s_lshl_b32 s0, s2, 7 // (op 6446) 
s_mul_i32 s1, 3, s0 // (op 6446) 
s_add_u32 s1, s1, 384 // (op 6446) 

v_add_u32 v0, v20, s1 // (op 6446) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6446) Load value

v_accvgpr_read v10, a60 // (op 6452) 
v_accvgpr_read v11, a61 // (op 6452) 
v_accvgpr_read v12, a62 // (op 6452) 
v_accvgpr_read v13, a63 // (op 6452) 
v_mul_f32 v6, v19, v10 // (op 6452) 
v_mul_f32 v7, v19, v11 // (op 6452) 
v_mul_f32 v8, v19, v12 // (op 6452) 
v_mul_f32 v9, v19, v13 // (op 6452) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6444) 
v_mul_f32 v11, v18, v15 // (op 6444) 
v_mul_f32 v12, v18, v16 // (op 6444) 
v_mul_f32 v13, v18, v17 // (op 6444) 

v_add_f32 v14, v10, v6 // (op 6445) 
v_add_f32 v15, v11, v7 // (op 6445) 
v_add_f32 v16, v12, v8 // (op 6445) 
v_add_f32 v17, v13, v9 // (op 6445) 




s_lshl_b32 s0, s4, 7 // (op 6431) 
s_mul_i32 s1, 3, s0 // (op 6431) 
s_add_u32 s1, s1, 384 // (op 6431) 

v_add_u32 v0, v21, s1 // (op 6431) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6431) Store value




v_add_u32 v0, 128, v20 // (op 6276) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6276) Load value

v_accvgpr_read v10, a16 // (op 6264) 
v_accvgpr_read v11, a17 // (op 6264) 
v_accvgpr_read v12, a18 // (op 6264) 
v_accvgpr_read v13, a19 // (op 6264) 
v_mul_f32 v6, v19, v10 // (op 6264) 
v_mul_f32 v7, v19, v11 // (op 6264) 
v_mul_f32 v8, v19, v12 // (op 6264) 
v_mul_f32 v9, v19, v13 // (op 6264) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6275) 
v_mul_f32 v11, v18, v15 // (op 6275) 
v_mul_f32 v12, v18, v16 // (op 6275) 
v_mul_f32 v13, v18, v17 // (op 6275) 

v_add_f32 v14, v10, v6 // (op 6265) 
v_add_f32 v15, v11, v7 // (op 6265) 
v_add_f32 v16, v12, v8 // (op 6265) 
v_add_f32 v17, v13, v9 // (op 6265) 




v_add_u32 v0, 128, v21 // (op 6267) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6267) Store value




s_lshl_b32 s0, s2, 7 // (op 6280) 
s_add_u32 s0, s0, 128 // (op 6280) 
v_add_u32 v0, v20, s0 // (op 6280) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6280) Load value

v_accvgpr_read v10, a20 // (op 6290) 
v_accvgpr_read v11, a21 // (op 6290) 
v_accvgpr_read v12, a22 // (op 6290) 
v_accvgpr_read v13, a23 // (op 6290) 
v_mul_f32 v6, v19, v10 // (op 6290) 
v_mul_f32 v7, v19, v11 // (op 6290) 
v_mul_f32 v8, v19, v12 // (op 6290) 
v_mul_f32 v9, v19, v13 // (op 6290) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6278) 
v_mul_f32 v11, v18, v15 // (op 6278) 
v_mul_f32 v12, v18, v16 // (op 6278) 
v_mul_f32 v13, v18, v17 // (op 6278) 

v_add_f32 v14, v10, v6 // (op 6279) 
v_add_f32 v15, v11, v7 // (op 6279) 
v_add_f32 v16, v12, v8 // (op 6279) 
v_add_f32 v17, v13, v9 // (op 6279) 




s_lshl_b32 s0, s4, 7 // (op 6269) 
s_add_u32 s0, s0, 128 // (op 6269) 
v_add_u32 v0, v21, s0 // (op 6269) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6269) Store value




s_lshl_b32 s0, s2, 8 // (op 6284) 
s_add_u32 s0, s0, 128 // (op 6284) 
v_add_u32 v0, v20, s0 // (op 6284) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6284) Load value

v_accvgpr_read v10, a24 // (op 6292) 
v_accvgpr_read v11, a25 // (op 6292) 
v_accvgpr_read v12, a26 // (op 6292) 
v_accvgpr_read v13, a27 // (op 6292) 
v_mul_f32 v6, v19, v10 // (op 6292) 
v_mul_f32 v7, v19, v11 // (op 6292) 
v_mul_f32 v8, v19, v12 // (op 6292) 
v_mul_f32 v9, v19, v13 // (op 6292) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6282) 
v_mul_f32 v11, v18, v15 // (op 6282) 
v_mul_f32 v12, v18, v16 // (op 6282) 
v_mul_f32 v13, v18, v17 // (op 6282) 

v_add_f32 v14, v10, v6 // (op 6283) 
v_add_f32 v15, v11, v7 // (op 6283) 
v_add_f32 v16, v12, v8 // (op 6283) 
v_add_f32 v17, v13, v9 // (op 6283) 




s_lshl_b32 s0, s4, 8 // (op 6271) 
s_add_u32 s0, s0, 128 // (op 6271) 
v_add_u32 v0, v21, s0 // (op 6271) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6271) Store value




s_lshl_b32 s0, s2, 7 // (op 6288) 
s_mul_i32 s1, 3, s0 // (op 6288) 
s_add_u32 s1, s1, 128 // (op 6288) 

v_add_u32 v0, v20, s1 // (op 6288) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6288) Load value

v_accvgpr_read v10, a28 // (op 6294) 
v_accvgpr_read v11, a29 // (op 6294) 
v_accvgpr_read v12, a30 // (op 6294) 
v_accvgpr_read v13, a31 // (op 6294) 
v_mul_f32 v6, v19, v10 // (op 6294) 
v_mul_f32 v7, v19, v11 // (op 6294) 
v_mul_f32 v8, v19, v12 // (op 6294) 
v_mul_f32 v9, v19, v13 // (op 6294) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6286) 
v_mul_f32 v11, v18, v15 // (op 6286) 
v_mul_f32 v12, v18, v16 // (op 6286) 
v_mul_f32 v13, v18, v17 // (op 6286) 

v_add_f32 v14, v10, v6 // (op 6287) 
v_add_f32 v15, v11, v7 // (op 6287) 
v_add_f32 v16, v12, v8 // (op 6287) 
v_add_f32 v17, v13, v9 // (op 6287) 




s_lshl_b32 s0, s4, 7 // (op 6273) 
s_mul_i32 s1, 3, s0 // (op 6273) 
s_add_u32 s1, s1, 128 // (op 6273) 

v_add_u32 v0, v21, s1 // (op 6273) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6273) Store value




v_add_u32 v0, 256, v20 // (op 6355) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6355) Load value

v_accvgpr_read v10, a32 // (op 6343) 
v_accvgpr_read v11, a33 // (op 6343) 
v_accvgpr_read v12, a34 // (op 6343) 
v_accvgpr_read v13, a35 // (op 6343) 
v_mul_f32 v6, v19, v10 // (op 6343) 
v_mul_f32 v7, v19, v11 // (op 6343) 
v_mul_f32 v8, v19, v12 // (op 6343) 
v_mul_f32 v9, v19, v13 // (op 6343) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6354) 
v_mul_f32 v11, v18, v15 // (op 6354) 
v_mul_f32 v12, v18, v16 // (op 6354) 
v_mul_f32 v13, v18, v17 // (op 6354) 

v_add_f32 v14, v10, v6 // (op 6344) 
v_add_f32 v15, v11, v7 // (op 6344) 
v_add_f32 v16, v12, v8 // (op 6344) 
v_add_f32 v17, v13, v9 // (op 6344) 




v_add_u32 v0, 256, v21 // (op 6346) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6346) Store value




s_lshl_b32 s0, s2, 7 // (op 6359) 
s_add_u32 s0, s0, 256 // (op 6359) 
v_add_u32 v0, v20, s0 // (op 6359) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6359) Load value

v_accvgpr_read v10, a36 // (op 6369) 
v_accvgpr_read v11, a37 // (op 6369) 
v_accvgpr_read v12, a38 // (op 6369) 
v_accvgpr_read v13, a39 // (op 6369) 
v_mul_f32 v6, v19, v10 // (op 6369) 
v_mul_f32 v7, v19, v11 // (op 6369) 
v_mul_f32 v8, v19, v12 // (op 6369) 
v_mul_f32 v9, v19, v13 // (op 6369) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6357) 
v_mul_f32 v11, v18, v15 // (op 6357) 
v_mul_f32 v12, v18, v16 // (op 6357) 
v_mul_f32 v13, v18, v17 // (op 6357) 

v_add_f32 v14, v10, v6 // (op 6358) 
v_add_f32 v15, v11, v7 // (op 6358) 
v_add_f32 v16, v12, v8 // (op 6358) 
v_add_f32 v17, v13, v9 // (op 6358) 




s_lshl_b32 s0, s4, 7 // (op 6348) 
s_add_u32 s0, s0, 256 // (op 6348) 
v_add_u32 v0, v21, s0 // (op 6348) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6348) Store value




s_lshl_b32 s0, s2, 8 // (op 6363) 
s_add_u32 s0, s0, 256 // (op 6363) 
v_add_u32 v0, v20, s0 // (op 6363) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6363) Load value

v_accvgpr_read v10, a40 // (op 6371) 
v_accvgpr_read v11, a41 // (op 6371) 
v_accvgpr_read v12, a42 // (op 6371) 
v_accvgpr_read v13, a43 // (op 6371) 
v_mul_f32 v6, v19, v10 // (op 6371) 
v_mul_f32 v7, v19, v11 // (op 6371) 
v_mul_f32 v8, v19, v12 // (op 6371) 
v_mul_f32 v9, v19, v13 // (op 6371) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6361) 
v_mul_f32 v11, v18, v15 // (op 6361) 
v_mul_f32 v12, v18, v16 // (op 6361) 
v_mul_f32 v13, v18, v17 // (op 6361) 

v_add_f32 v14, v10, v6 // (op 6362) 
v_add_f32 v15, v11, v7 // (op 6362) 
v_add_f32 v16, v12, v8 // (op 6362) 
v_add_f32 v17, v13, v9 // (op 6362) 




s_lshl_b32 s0, s4, 8 // (op 6350) 
s_add_u32 s0, s0, 256 // (op 6350) 
v_add_u32 v0, v21, s0 // (op 6350) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6350) Store value




s_lshl_b32 s0, s2, 7 // (op 6367) 
s_mul_i32 s1, 3, s0 // (op 6367) 
s_add_u32 s1, s1, 256 // (op 6367) 

v_add_u32 v0, v20, s1 // (op 6367) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6367) Load value

v_accvgpr_read v10, a44 // (op 6373) 
v_accvgpr_read v11, a45 // (op 6373) 
v_accvgpr_read v12, a46 // (op 6373) 
v_accvgpr_read v13, a47 // (op 6373) 
v_mul_f32 v6, v19, v10 // (op 6373) 
v_mul_f32 v7, v19, v11 // (op 6373) 
v_mul_f32 v8, v19, v12 // (op 6373) 
v_mul_f32 v9, v19, v13 // (op 6373) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6365) 
v_mul_f32 v11, v18, v15 // (op 6365) 
v_mul_f32 v12, v18, v16 // (op 6365) 
v_mul_f32 v13, v18, v17 // (op 6365) 

v_add_f32 v14, v10, v6 // (op 6366) 
v_add_f32 v15, v11, v7 // (op 6366) 
v_add_f32 v16, v12, v8 // (op 6366) 
v_add_f32 v17, v13, v9 // (op 6366) 




s_lshl_b32 s0, s4, 7 // (op 6352) 
s_mul_i32 s1, 3, s0 // (op 6352) 
s_add_u32 s1, s1, 256 // (op 6352) 

v_add_u32 v0, v21, s1 // (op 6352) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6352) Store value




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




s_lshl_b32 s0, s2, 7 // (op 6190) 
v_add_u32 v0, v20, s0 // (op 6190) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6190) Load value

v_accvgpr_read v10, a4 // (op 6187) 
v_accvgpr_read v11, a5 // (op 6187) 
v_accvgpr_read v12, a6 // (op 6187) 
v_accvgpr_read v13, a7 // (op 6187) 
v_mul_f32 v6, v19, v10 // (op 6187) 
v_mul_f32 v7, v19, v11 // (op 6187) 
v_mul_f32 v8, v19, v12 // (op 6187) 
v_mul_f32 v9, v19, v13 // (op 6187) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6191) 
v_mul_f32 v11, v18, v15 // (op 6191) 
v_mul_f32 v12, v18, v16 // (op 6191) 
v_mul_f32 v13, v18, v17 // (op 6191) 

v_add_f32 v14, v10, v6 // (op 6188) 
v_add_f32 v15, v11, v7 // (op 6188) 
v_add_f32 v16, v12, v8 // (op 6188) 
v_add_f32 v17, v13, v9 // (op 6188) 




s_lshl_b32 s0, s4, 7 // (op 6189) 
v_add_u32 v0, v21, s0 // (op 6189) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6189) Store value




s_lshl_b32 s0, s2, 8 // (op 6202) 
v_add_u32 v0, v20, s0 // (op 6202) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6202) Load value

v_accvgpr_read v10, a8 // (op 6199) 
v_accvgpr_read v11, a9 // (op 6199) 
v_accvgpr_read v12, a10 // (op 6199) 
v_accvgpr_read v13, a11 // (op 6199) 
v_mul_f32 v6, v19, v10 // (op 6199) 
v_mul_f32 v7, v19, v11 // (op 6199) 
v_mul_f32 v8, v19, v12 // (op 6199) 
v_mul_f32 v9, v19, v13 // (op 6199) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6203) 
v_mul_f32 v11, v18, v15 // (op 6203) 
v_mul_f32 v12, v18, v16 // (op 6203) 
v_mul_f32 v13, v18, v17 // (op 6203) 

v_add_f32 v14, v10, v6 // (op 6200) 
v_add_f32 v15, v11, v7 // (op 6200) 
v_add_f32 v16, v12, v8 // (op 6200) 
v_add_f32 v17, v13, v9 // (op 6200) 




s_lshl_b32 s0, s4, 8 // (op 6201) 
v_add_u32 v0, v21, s0 // (op 6201) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6201) Store value




s_lshl_b32 s0, s2, 7 // (op 6214) 
s_mul_i32 s1, 3, s0 // (op 6214) 

v_add_u32 v0, v20, s1 // (op 6214) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 6214) Load value

v_accvgpr_read v10, a12 // (op 6211) 
v_accvgpr_read v11, a13 // (op 6211) 
v_accvgpr_read v12, a14 // (op 6211) 
v_accvgpr_read v13, a15 // (op 6211) 
v_mul_f32 v6, v19, v10 // (op 6211) 
v_mul_f32 v7, v19, v11 // (op 6211) 
v_mul_f32 v8, v19, v12 // (op 6211) 
v_mul_f32 v9, v19, v13 // (op 6211) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 6215) 
v_mul_f32 v11, v18, v15 // (op 6215) 
v_mul_f32 v12, v18, v16 // (op 6215) 
v_mul_f32 v13, v18, v17 // (op 6215) 

v_add_f32 v14, v10, v6 // (op 6212) 
v_add_f32 v15, v11, v7 // (op 6212) 
v_add_f32 v16, v12, v8 // (op 6212) 
v_add_f32 v17, v13, v9 // (op 6212) 




s_lshl_b32 s0, s4, 7 // (op 6213) 
s_mul_i32 s1, 3, s0 // (op 6213) 

v_add_u32 v0, v21, s1 // (op 6213) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 6213) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.LRR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end:
.size RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_, .LRR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end-RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
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
  - .name: RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
    .symbol: RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_.kd
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

