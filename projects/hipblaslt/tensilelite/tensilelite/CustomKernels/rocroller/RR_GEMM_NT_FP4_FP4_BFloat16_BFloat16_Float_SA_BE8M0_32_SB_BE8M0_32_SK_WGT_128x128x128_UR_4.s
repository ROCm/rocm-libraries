// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.p2align 8
.type RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4,@function
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_exec_begin:


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
s_load_dwordx2 s[50:51], s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_and_b32 v0, 63, v1 // (op 28194) 
v_and_b32 v3, 48, v1 // (op 28194) 
v_sub_u32 v4, v0, v3 // (op 28194) 

v_lshrrev_b32 v3, 4, v0 // (op 28194) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 28194) 


v_lshrrev_b32 v3, 7, v1 // (op 28194) 
v_lshlrev_b32 v4, 4, v3 // (op 28194) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 28194) 


v_lshlrev_b32 v0, 2, v3 // (op 28194) 

v_lshrrev_b32 v3, 3, v0 // (op 28194) 

v_and_b32 v0, 63, v1 // (op 28269) 
v_and_b32 v4, 48, v1 // (op 28269) 
v_sub_u32 v5, v0, v4 // (op 28269) 

v_lshrrev_b32 v4, 4, v0 // (op 28269) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 28269) 


v_lshrrev_b32 v4, 6, v1 // (op 28269) 
v_and_b32 v5, 1, v4 // (op 28269) 

v_lshlrev_b32 v4, 4, v5 // (op 28269) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 28269) 


v_lshlrev_b32 v0, 2, v5 // (op 28269) 

v_lshrrev_b32 v4, 3, v0 // (op 28269) 

v_lshrrev_b32 v0, 7, v1 // (op 28345) 
v_and_b32 v5, 15, v1 // (op 28345) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 28345) 


v_and_b32 v0, 63, v1 // (op 28345) 
v_lshrrev_b32 v5, 4, v0 // (op 28345) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 28345) 


v_lshrrev_b32 v5, 6, v1 // (op 28421) 
v_and_b32 v6, 1, v5 // (op 28421) 

v_and_b32 v5, 15, v1 // (op 28421) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 28421) 


v_and_b32 v5, 63, v1 // (op 28421) 
v_lshrrev_b32 v6, 4, v5 // (op 28421) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 28421) 


v_and_b32 v6, 63, v1 // (op 28548) 
v_and_b32 v7, 48, v1 // (op 28548) 
v_sub_u32 v8, v6, v7 // (op 28548) 

v_lshrrev_b32 v7, 4, v6 // (op 28548) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 28548) 


v_lshrrev_b32 v7, 7, v1 // (op 28548) 
v_lshlrev_b32 v8, 4, v7 // (op 28548) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 28548) 


v_lshlrev_b32 v6, 2, v7 // (op 28548) 

v_lshrrev_b32 v7, 3, v6 // (op 28548) 

v_and_b32 v6, 63, v1 // (op 28623) 
v_and_b32 v8, 48, v1 // (op 28623) 
v_sub_u32 v9, v6, v8 // (op 28623) 

v_lshrrev_b32 v8, 4, v6 // (op 28623) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 28623) 


v_lshrrev_b32 v8, 6, v1 // (op 28623) 
v_and_b32 v9, 1, v8 // (op 28623) 

v_lshlrev_b32 v8, 4, v9 // (op 28623) 

v_lshl_add_u32 v9, v6, 7, v8 // (op 28623) 


v_lshlrev_b32 v6, 2, v9 // (op 28623) 

v_lshrrev_b32 v8, 3, v6 // (op 28623) 

v_lshrrev_b32 v6, 7, v1 // (op 28699) 
v_and_b32 v9, 15, v1 // (op 28699) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 28699) 


v_and_b32 v6, 63, v1 // (op 28699) 
v_lshrrev_b32 v9, 4, v6 // (op 28699) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 28699) 


v_lshrrev_b32 v9, 6, v1 // (op 28775) 
v_and_b32 v10, 1, v9 // (op 28775) 

v_and_b32 v9, 15, v1 // (op 28775) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 28775) 


v_and_b32 v9, 63, v1 // (op 28775) 
v_lshrrev_b32 v10, 4, v9 // (op 28775) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 28775) 


v_and_b32 v10, 63, v1 // (op 28889) 
v_and_b32 v11, 48, v1 // (op 28889) 
v_sub_u32 v12, v10, v11 // (op 28889) 

v_lshrrev_b32 v11, 4, v10 // (op 28889) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 28889) 


v_lshrrev_b32 v11, 7, v1 // (op 28889) 
v_lshlrev_b32 v12, 4, v11 // (op 28889) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 28889) 


v_lshlrev_b32 v10, 2, v11 // (op 28889) 

v_lshrrev_b32 v11, 3, v10 // (op 28889) 

v_and_b32 v10, 63, v1 // (op 28933) 
v_and_b32 v12, 48, v1 // (op 28933) 
v_sub_u32 v13, v10, v12 // (op 28933) 

v_lshrrev_b32 v12, 4, v10 // (op 28933) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 28933) 


v_lshrrev_b32 v12, 6, v1 // (op 28933) 
v_and_b32 v13, 1, v12 // (op 28933) 

v_lshlrev_b32 v12, 4, v13 // (op 28933) 

v_lshl_add_u32 v13, v10, 7, v12 // (op 28933) 


v_lshlrev_b32 v10, 2, v13 // (op 28933) 

v_lshrrev_b32 v12, 3, v10 // (op 28933) 

v_lshrrev_b32 v10, 7, v1 // (op 28962) 
v_and_b32 v13, 15, v1 // (op 28962) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 28962) 


v_and_b32 v10, 63, v1 // (op 28962) 
v_lshrrev_b32 v13, 4, v10 // (op 28962) 

v_lshl_add_u32 v10, v13, 7, v14 // (op 28962) 


v_lshrrev_b32 v13, 6, v1 // (op 28991) 
v_and_b32 v14, 1, v13 // (op 28991) 

v_and_b32 v13, 15, v1 // (op 28991) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 28991) 


v_and_b32 v13, 63, v1 // (op 28991) 
v_lshrrev_b32 v14, 4, v13 // (op 28991) 

v_lshl_add_u32 v13, v14, 7, v15 // (op 28991) 


v_and_b32 v14, 63, v1 // (op 29035) 
v_and_b32 v15, 48, v1 // (op 29035) 
v_sub_u32 v16, v14, v15 // (op 29035) 

v_lshrrev_b32 v15, 4, v14 // (op 29035) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 29035) 


v_lshrrev_b32 v15, 7, v1 // (op 29035) 
v_lshlrev_b32 v16, 4, v15 // (op 29035) 

v_lshl_add_u32 v15, v14, 7, v16 // (op 29035) 


v_lshlrev_b32 v14, 2, v15 // (op 29035) 

v_lshrrev_b32 v15, 3, v14 // (op 29035) 

v_and_b32 v14, 63, v1 // (op 29079) 
v_and_b32 v16, 48, v1 // (op 29079) 
v_sub_u32 v17, v14, v16 // (op 29079) 

v_lshrrev_b32 v16, 4, v14 // (op 29079) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 29079) 


v_lshrrev_b32 v16, 6, v1 // (op 29079) 
v_and_b32 v17, 1, v16 // (op 29079) 

v_lshlrev_b32 v16, 4, v17 // (op 29079) 

v_lshl_add_u32 v17, v14, 7, v16 // (op 29079) 


v_lshlrev_b32 v14, 2, v17 // (op 29079) 

v_lshrrev_b32 v16, 3, v14 // (op 29079) 

v_lshrrev_b32 v14, 7, v1 // (op 29108) 
v_and_b32 v17, 15, v1 // (op 29108) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 29108) 


v_and_b32 v14, 63, v1 // (op 29108) 
v_lshrrev_b32 v17, 4, v14 // (op 29108) 

v_lshl_add_u32 v14, v17, 7, v18 // (op 29108) 


v_lshrrev_b32 v17, 6, v1 // (op 29137) 
v_and_b32 v18, 1, v17 // (op 29137) 

v_and_b32 v17, 15, v1 // (op 29137) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 29137) 


v_and_b32 v17, 63, v1 // (op 29137) 
v_lshrrev_b32 v18, 4, v17 // (op 29137) 

v_lshl_add_u32 v17, v18, 7, v19 // (op 29137) 


v_and_b32 v18, 63, v1 // (op 29181) 
v_and_b32 v19, 48, v1 // (op 29181) 
v_sub_u32 v20, v18, v19 // (op 29181) 

v_lshrrev_b32 v19, 4, v18 // (op 29181) 

v_lshl_add_u32 v18, v19, 5, v20 // (op 29181) 


v_lshrrev_b32 v19, 7, v1 // (op 29181) 
v_lshlrev_b32 v20, 4, v19 // (op 29181) 

v_lshl_add_u32 v19, v18, 7, v20 // (op 29181) 


v_lshlrev_b32 v18, 2, v19 // (op 29181) 

v_lshrrev_b32 v19, 3, v18 // (op 29181) 

v_and_b32 v18, 63, v1 // (op 29225) 
v_and_b32 v20, 48, v1 // (op 29225) 
v_sub_u32 v21, v18, v20 // (op 29225) 

v_lshrrev_b32 v20, 4, v18 // (op 29225) 

v_lshl_add_u32 v18, v20, 5, v21 // (op 29225) 


v_lshrrev_b32 v20, 6, v1 // (op 29225) 
v_and_b32 v21, 1, v20 // (op 29225) 

v_lshlrev_b32 v20, 4, v21 // (op 29225) 

v_lshl_add_u32 v21, v18, 7, v20 // (op 29225) 


v_lshlrev_b32 v18, 2, v21 // (op 29225) 

v_lshrrev_b32 v20, 3, v18 // (op 29225) 

v_lshrrev_b32 v18, 7, v1 // (op 29254) 
v_and_b32 v21, 15, v1 // (op 29254) 
v_lshl_add_u32 v22, v18, 4, v21 // (op 29254) 


v_and_b32 v18, 63, v1 // (op 29254) 
v_lshrrev_b32 v21, 4, v18 // (op 29254) 

v_lshl_add_u32 v18, v21, 7, v22 // (op 29254) 


v_lshrrev_b32 v21, 6, v1 // (op 29283) 
v_and_b32 v22, 1, v21 // (op 29283) 

v_and_b32 v21, 15, v1 // (op 29283) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 29283) 


v_and_b32 v21, 63, v1 // (op 29283) 
v_lshrrev_b32 v22, 4, v21 // (op 29283) 

v_lshl_add_u32 v21, v22, 7, v23 // (op 29283) 


v_and_b32 v22, 63, v1 // (op 29327) 
v_and_b32 v23, 48, v1 // (op 29327) 
v_sub_u32 v24, v22, v23 // (op 29327) 

v_lshrrev_b32 v23, 4, v22 // (op 29327) 

v_lshl_add_u32 v22, v23, 5, v24 // (op 29327) 


v_lshrrev_b32 v23, 7, v1 // (op 29327) 
v_lshlrev_b32 v24, 4, v23 // (op 29327) 

v_lshl_add_u32 v23, v22, 7, v24 // (op 29327) 


v_lshlrev_b32 v22, 2, v23 // (op 29327) 

v_lshrrev_b32 v23, 3, v22 // (op 29327) 

v_and_b32 v22, 63, v1 // (op 29371) 
v_and_b32 v24, 48, v1 // (op 29371) 
v_sub_u32 v25, v22, v24 // (op 29371) 

v_lshrrev_b32 v24, 4, v22 // (op 29371) 

v_lshl_add_u32 v22, v24, 5, v25 // (op 29371) 


v_lshrrev_b32 v24, 6, v1 // (op 29371) 
v_and_b32 v25, 1, v24 // (op 29371) 

v_lshlrev_b32 v24, 4, v25 // (op 29371) 

v_lshl_add_u32 v25, v22, 7, v24 // (op 29371) 


v_lshlrev_b32 v22, 2, v25 // (op 29371) 

v_lshrrev_b32 v24, 3, v22 // (op 29371) 

v_lshrrev_b32 v22, 7, v1 // (op 29400) 
v_and_b32 v25, 15, v1 // (op 29400) 
v_lshl_add_u32 v26, v22, 4, v25 // (op 29400) 


v_and_b32 v22, 63, v1 // (op 29400) 
v_lshrrev_b32 v25, 4, v22 // (op 29400) 

v_lshl_add_u32 v22, v25, 7, v26 // (op 29400) 


v_lshrrev_b32 v25, 6, v1 // (op 29429) 
v_and_b32 v26, 1, v25 // (op 29429) 

v_and_b32 v25, 15, v1 // (op 29429) 
v_lshl_add_u32 v27, v26, 4, v25 // (op 29429) 


v_and_b32 v25, 63, v1 // (op 29429) 
v_lshrrev_b32 v26, 4, v25 // (op 29429) 

v_lshl_add_u32 v25, v26, 7, v27 // (op 29429) 


v_and_b32 v26, 63, v1 // (op 29473) 
v_and_b32 v27, 48, v1 // (op 29473) 
v_sub_u32 v28, v26, v27 // (op 29473) 

v_lshrrev_b32 v27, 4, v26 // (op 29473) 

v_lshl_add_u32 v26, v27, 5, v28 // (op 29473) 


v_lshrrev_b32 v27, 7, v1 // (op 29473) 
v_lshlrev_b32 v28, 4, v27 // (op 29473) 

v_lshl_add_u32 v27, v26, 7, v28 // (op 29473) 


v_lshlrev_b32 v26, 2, v27 // (op 29473) 

v_lshrrev_b32 v27, 3, v26 // (op 29473) 

v_and_b32 v26, 63, v1 // (op 29517) 
v_and_b32 v28, 48, v1 // (op 29517) 
v_sub_u32 v29, v26, v28 // (op 29517) 

v_lshrrev_b32 v28, 4, v26 // (op 29517) 

v_lshl_add_u32 v26, v28, 5, v29 // (op 29517) 


v_lshrrev_b32 v28, 6, v1 // (op 29517) 
v_and_b32 v29, 1, v28 // (op 29517) 

v_lshlrev_b32 v28, 4, v29 // (op 29517) 

v_lshl_add_u32 v29, v26, 7, v28 // (op 29517) 


v_lshlrev_b32 v26, 2, v29 // (op 29517) 

v_lshrrev_b32 v28, 3, v26 // (op 29517) 

v_lshrrev_b32 v26, 7, v1 // (op 29546) 
v_and_b32 v29, 15, v1 // (op 29546) 
v_lshl_add_u32 v30, v26, 4, v29 // (op 29546) 


v_and_b32 v26, 63, v1 // (op 29546) 
v_lshrrev_b32 v29, 4, v26 // (op 29546) 

v_lshl_add_u32 v26, v29, 7, v30 // (op 29546) 


v_lshrrev_b32 v29, 6, v1 // (op 29575) 
v_and_b32 v30, 1, v29 // (op 29575) 

v_and_b32 v29, 15, v1 // (op 29575) 
v_lshl_add_u32 v31, v30, 4, v29 // (op 29575) 


v_and_b32 v29, 63, v1 // (op 29575) 
v_lshrrev_b32 v30, 4, v29 // (op 29575) 

v_lshl_add_u32 v29, v30, 7, v31 // (op 29575) 


v_and_b32 v30, 63, v1 // (op 29619) 
v_and_b32 v31, 48, v1 // (op 29619) 
v_sub_u32 v32, v30, v31 // (op 29619) 

v_lshrrev_b32 v31, 4, v30 // (op 29619) 

v_lshl_add_u32 v30, v31, 5, v32 // (op 29619) 


v_lshrrev_b32 v31, 7, v1 // (op 29619) 
v_lshlrev_b32 v32, 4, v31 // (op 29619) 

v_lshl_add_u32 v31, v30, 7, v32 // (op 29619) 


v_lshlrev_b32 v30, 2, v31 // (op 29619) 

v_lshrrev_b32 v31, 3, v30 // (op 29619) 

v_and_b32 v30, 63, v1 // (op 29663) 
v_and_b32 v32, 48, v1 // (op 29663) 
v_sub_u32 v33, v30, v32 // (op 29663) 

v_lshrrev_b32 v32, 4, v30 // (op 29663) 

v_lshl_add_u32 v30, v32, 5, v33 // (op 29663) 


v_lshrrev_b32 v32, 6, v1 // (op 29663) 
v_and_b32 v33, 1, v32 // (op 29663) 

v_lshlrev_b32 v32, 4, v33 // (op 29663) 

v_lshl_add_u32 v33, v30, 7, v32 // (op 29663) 


v_lshlrev_b32 v30, 2, v33 // (op 29663) 

v_lshrrev_b32 v32, 3, v30 // (op 29663) 

v_lshrrev_b32 v30, 7, v1 // (op 29692) 
v_and_b32 v33, 15, v1 // (op 29692) 
v_lshl_add_u32 v34, v30, 4, v33 // (op 29692) 


v_and_b32 v30, 63, v1 // (op 29692) 
v_lshrrev_b32 v33, 4, v30 // (op 29692) 

v_lshl_add_u32 v30, v33, 7, v34 // (op 29692) 


v_lshrrev_b32 v33, 6, v1 // (op 29721) 
v_and_b32 v34, 1, v33 // (op 29721) 

v_and_b32 v33, 15, v1 // (op 29721) 
v_lshl_add_u32 v35, v34, 4, v33 // (op 29721) 


v_and_b32 v33, 63, v1 // (op 29721) 
v_lshrrev_b32 v34, 4, v33 // (op 29721) 

v_lshl_add_u32 v33, v34, 7, v35 // (op 29721) 


v_mov_b32 v34, s6 // (op 2) Move value
v_mov_b32 v35, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 395) call()
s_mov_b32 s1, 0 // (op 396) call()
s_mov_b32 s6, 0 // (op 397) call()
s_mov_b32 s7, 0 // (op 398) call()
s_mov_b32 s18, 0 // (op 399) call()
s_mov_b32 s19, 0 // (op 400) call()
s_mov_b32 s0, 0 // (op 295) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v36, 7, v1 // (op 28105) 
s_mul_i32 s52, s16, s8 // (op 28105) 
s_add_u32 s52, s52, s0 // (op 28105) 
v_and_b32 v37, 63, v1 // (op 28105) 
s_mul_hi_u32 s53, s52, s13 // (op 28105) 
v_lshrrev_b32 v38, 4, v37 // (op 28105) 
s_sub_u32 s54, s52, s53 // (op 28105) 
v_lshlrev_b32 v39, 2, v38 // (op 28105) 

v_lshl_add_u32 v38, v36, 4, v39 // (op 28105) 


s_lshr_b32 s55, s54, 1 // (op 28105) 

s_add_u32 s54, s55, s53 // (op 28105) 


v_lshrrev_b32 v36, 2, v37 // (op 28105) 

s_lshr_b32 s53, s54, s11 // (op 28105) 

v_and_b32 v37, 3, v36 // (op 28105) 

s_and_b32 s54, s11, 2147483648 // (op 28105) 
v_and_b32 v36, 3, v1 // (op 28105) 
s_cmp_eq_i32 s54, 0 // (op 28105) 
s_mov_b32 s55, scc // (op 28105) 

v_lshl_add_u32 v39, v37, 2, v36 // (op 28105) 


s_cmp_eq_i32 s55, 0 // (op 28105) 
s_cselect_b32 s54, s52, s53 // (op 28105) 



v_lshrrev_b32 v36, 6, v1 // (op 28105) 
s_mul_hi_u32 s52, s54, s12 // (op 28105) 
v_and_b32 v37, 1, v36 // (op 28105) 

s_sub_u32 s53, s54, s52 // (op 28105) 
v_lshl_add_u32 v36, v37, 4, v39 // (op 28105) 


s_lshr_b32 s55, s53, 1 // (op 28105) 

s_add_u32 s53, s55, s52 // (op 28105) 


s_lshr_b32 s52, s53, s10 // (op 28105) 

s_and_b32 s53, s10, 2147483648 // (op 28105) 
s_cmp_eq_i32 s53, 0 // (op 28105) 
s_mov_b32 s55, scc // (op 28105) 

s_cmp_eq_i32 s55, 0 // (op 28105) 
s_cselect_b32 s53, s54, s52 // (op 28105) 


s_mul_i32 s52, s53, s14 // (op 28105) 
v_lshl_add_u32 v37, s53, 7, v38 // (op 28105) 


s_sub_u32 s53, s54, s52 // (op 28105) 


v_lshl_add_u32 v38, s53, 7, v36 // (op 28105) 


v_mul_lo_u32 v36, v38, s4 // (op 28105) 

v_add_lshl_u32 v38, v37, v36, 1 // (op 28105) 


s_mul_i32 s53, s4, s23 // (op 28107) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 28107) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 28107) least significant
s_mul_hi_u32 s55, s4, s22 // (op 28107) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 28107) most significant: sum
s_add_u32 s53, s53, s55 // (op 28107) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 28107) 

s_mov_b32 s56, s20 // (op 28107) reg expression
s_mov_b32 s57, s21 // (op 28107) reg expression
s_mov_b32 s58, s54 // (op 28107) reg expression
s_mov_b32 s59, 131072 // (op 28107) call()

v_and_b32 v36, 63, v1 // (op 28211) 
v_and_b32 v37, 48, v1 // (op 28211) 
v_sub_u32 v39, v36, v37 // (op 28211) 

v_lshrrev_b32 v37, 4, v36 // (op 28211) 

v_lshl_add_u32 v36, v37, 5, v39 // (op 28211) 


v_lshrrev_b32 v37, 7, v1 // (op 28211) 
v_lshlrev_b32 v39, 4, v37 // (op 28211) 

v_lshl_add_u32 v37, v36, 7, v39 // (op 28211) 


v_lshlrev_b32 v36, 2, v37 // (op 28211) 

v_lshrrev_b32 v37, 3, v36 // (op 28211) 

v_and_b32 v36, 63, v1 // (op 28286) 
v_and_b32 v39, 48, v1 // (op 28286) 
v_sub_u32 v40, v36, v39 // (op 28286) 

v_lshrrev_b32 v39, 4, v36 // (op 28286) 

v_lshl_add_u32 v36, v39, 5, v40 // (op 28286) 


v_lshrrev_b32 v39, 6, v1 // (op 28286) 
v_and_b32 v40, 1, v39 // (op 28286) 

v_lshlrev_b32 v39, 4, v40 // (op 28286) 

v_lshl_add_u32 v40, v36, 7, v39 // (op 28286) 


v_lshlrev_b32 v36, 2, v40 // (op 28286) 

v_lshrrev_b32 v39, 3, v36 // (op 28286) 

v_lshrrev_b32 v36, 7, v1 // (op 28362) 
v_and_b32 v40, 15, v1 // (op 28362) 
v_lshl_add_u32 v41, v36, 4, v40 // (op 28362) 


v_and_b32 v36, 63, v1 // (op 28362) 
v_lshrrev_b32 v40, 4, v36 // (op 28362) 

v_lshl_add_u32 v36, v40, 7, v41 // (op 28362) 


v_lshrrev_b32 v40, 6, v1 // (op 28438) 
v_and_b32 v41, 1, v40 // (op 28438) 

v_and_b32 v40, 15, v1 // (op 28438) 
v_lshl_add_u32 v42, v41, 4, v40 // (op 28438) 


v_and_b32 v40, 63, v1 // (op 28438) 
v_lshrrev_b32 v41, 4, v40 // (op 28438) 

v_lshl_add_u32 v40, v41, 7, v42 // (op 28438) 


v_lshrrev_b32 v41, 7, v1 // (op 28461) 
s_mul_i32 s52, s16, s8 // (op 28461) 
s_add_u32 s52, s52, s0 // (op 28461) 
v_and_b32 v42, 63, v1 // (op 28461) 
s_mul_hi_u32 s53, s52, s13 // (op 28461) 
v_lshrrev_b32 v43, 4, v42 // (op 28461) 
s_sub_u32 s54, s52, s53 // (op 28461) 
v_lshlrev_b32 v44, 2, v43 // (op 28461) 

s_lshr_b32 s55, s54, 1 // (op 28461) 

v_lshl_add_u32 v43, v41, 4, v44 // (op 28461) 


s_add_u32 s54, s55, s53 // (op 28461) 


v_lshrrev_b32 v41, 2, v42 // (op 28461) 

s_lshr_b32 s53, s54, s11 // (op 28461) 

v_and_b32 v42, 3, v41 // (op 28461) 

s_and_b32 s54, s11, 2147483648 // (op 28461) 
v_and_b32 v41, 3, v1 // (op 28461) 
s_cmp_eq_i32 s54, 0 // (op 28461) 
s_mov_b32 s55, scc // (op 28461) 

v_lshl_add_u32 v44, v42, 2, v41 // (op 28461) 


s_cmp_eq_i32 s55, 0 // (op 28461) 
s_cselect_b32 s54, s52, s53 // (op 28461) 



v_lshrrev_b32 v41, 6, v1 // (op 28461) 
s_mul_hi_u32 s55, s54, s12 // (op 28461) 
v_and_b32 v42, 1, v41 // (op 28461) 

s_sub_u32 s52, s54, s55 // (op 28461) 
v_lshl_add_u32 v41, v42, 4, v44 // (op 28461) 


s_lshr_b32 s53, s52, 1 // (op 28461) 

s_add_u32 s52, s53, s55 // (op 28461) 


s_lshr_b32 s53, s52, s10 // (op 28461) 

s_and_b32 s52, s10, 2147483648 // (op 28461) 
s_cmp_eq_i32 s52, 0 // (op 28461) 
s_mov_b32 s55, scc // (op 28461) 

s_cmp_eq_i32 s55, 0 // (op 28461) 
s_cselect_b32 s52, s54, s53 // (op 28461) 


s_mul_i32 s53, s52, s14 // (op 28461) 
v_lshl_add_u32 v42, s52, 7, v43 // (op 28461) 


s_sub_u32 s52, s54, s53 // (op 28461) 


v_lshl_add_u32 v43, s52, 7, v41 // (op 28461) 


v_mul_lo_u32 v41, v43, s24 // (op 28461) 

v_add_lshl_u32 v43, v42, v41, 1 // (op 28461) 


s_mul_i32 s53, s24, s23 // (op 28463) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 28463) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 28463) least significant
s_mul_hi_u32 s55, s24, s22 // (op 28463) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 28463) most significant: sum
s_add_u32 s53, s53, s55 // (op 28463) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 28463) 

s_mov_b32 s60, s26 // (op 28463) reg expression
s_mov_b32 s61, s27 // (op 28463) reg expression
s_mov_b32 s62, s54 // (op 28463) reg expression
s_mov_b32 s63, 131072 // (op 28463) call()

v_lshrrev_b32 v41, 5, v1 // (op 28815) 
v_lshlrev_b32 v42, 2, v41 // (op 28815) 

v_lshl_add_u32 v41, s16, 7, v42 // (op 28815) 

v_and_b32 v42, -32, v1 // (op 28815) 
v_sub_u32 v44, v1, v42 // (op 28815) 

v_lshl_add_u32 v42, v41, 7, v44 // (op 28815) 


v_lshlrev_b32 v41, 2, v42 // (op 28815) 

s_lshl_b32 s52, s15, 16 // (op 28817) 
s_mov_b32 s64, s28 // (op 28817) reg expression
s_mov_b32 s65, s29 // (op 28817) reg expression
s_mov_b32 s66, s52 // (op 28817) reg expression
s_mov_b32 s67, 131072 // (op 28817) call()

s_mul_i32 s6, s8, s16 // (op 299) 
s_add_u32 s6, s6, s0 // (op 299) 
s_mul_hi_u32 s52, s6, s13 // (op 300) 
s_sub_u32 s53, s6, s52 // (op 300) 
s_lshr_b32 s54, s53, 1 // (op 300) 

s_add_u32 s53, s54, s52 // (op 300) 


s_lshr_b32 s52, s53, s11 // (op 300) 

s_and_b32 s53, s11, 2147483648 // (op 300) 
s_cmp_eq_i32 s53, 0 // (op 300) 
s_mov_b32 s54, scc // (op 300) 

s_cmp_eq_i32 s54, 0 // (op 300) 
s_cselect_b32 s53, s6, s52 // (op 300) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 300) 
s_mul_i32 s52, s53, s54 // (op 300) 


s_sub_u32 s18, s6, s52 // (op 300) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 301) 
s_cmp_gt_u32 s52, s18 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_sub_u32 s55, s52, s18 // (op 301) 

s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s52, 0, s55 // (op 301) 


s_cmp_gt_u32 s8, s0 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_sub_u32 s54, s8, s0 // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s55, 0, s54 // (op 301) 


s_cmp_lt_u32 s52, s55 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s54, s55, s52 // (op 301) 



s_cmp_gt_u32 s9, s6 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_sub_u32 s52, s9, s6 // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s53, 0, s52 // (op 301) 


s_cmp_lt_u32 s54, s53 // (op 301) 
s_mov_b32 s52, scc // (op 301) 
s_cmp_eq_i32 s52, 0 // (op 301) 
s_cselect_b32 s7, s53, s54 // (op 301) 



s_add_u32 s52, s18, s7 // (op 302) 
s_sub_u32 s19, s52, 1 // (op 302) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 7260) call()
v_accvgpr_write a5, 0.00000 // (op 7260) call()
v_accvgpr_write a6, 0.00000 // (op 7260) call()
v_accvgpr_write a7, 0.00000 // (op 7260) call()
v_accvgpr_write a8, 0.00000 // (op 7558) call()
v_accvgpr_write a9, 0.00000 // (op 7558) call()
v_accvgpr_write a10, 0.00000 // (op 7558) call()
v_accvgpr_write a11, 0.00000 // (op 7558) call()
v_accvgpr_write a12, 0.00000 // (op 7856) call()
v_accvgpr_write a13, 0.00000 // (op 7856) call()
v_accvgpr_write a14, 0.00000 // (op 7856) call()
v_accvgpr_write a15, 0.00000 // (op 7856) call()
v_accvgpr_write a16, 0.00000 // (op 8290) call()
v_accvgpr_write a17, 0.00000 // (op 8290) call()
v_accvgpr_write a18, 0.00000 // (op 8290) call()
v_accvgpr_write a19, 0.00000 // (op 8290) call()
v_accvgpr_write a20, 0.00000 // (op 8783) call()
v_accvgpr_write a21, 0.00000 // (op 8783) call()
v_accvgpr_write a22, 0.00000 // (op 8783) call()
v_accvgpr_write a23, 0.00000 // (op 8783) call()
v_accvgpr_write a24, 0.00000 // (op 8784) call()
v_accvgpr_write a25, 0.00000 // (op 8784) call()
v_accvgpr_write a26, 0.00000 // (op 8784) call()
v_accvgpr_write a27, 0.00000 // (op 8784) call()
v_accvgpr_write a28, 0.00000 // (op 8785) call()
v_accvgpr_write a29, 0.00000 // (op 8785) call()
v_accvgpr_write a30, 0.00000 // (op 8785) call()
v_accvgpr_write a31, 0.00000 // (op 8785) call()
v_accvgpr_write a32, 0.00000 // (op 9580) call()
v_accvgpr_write a33, 0.00000 // (op 9580) call()
v_accvgpr_write a34, 0.00000 // (op 9580) call()
v_accvgpr_write a35, 0.00000 // (op 9580) call()
v_accvgpr_write a36, 0.00000 // (op 10073) call()
v_accvgpr_write a37, 0.00000 // (op 10073) call()
v_accvgpr_write a38, 0.00000 // (op 10073) call()
v_accvgpr_write a39, 0.00000 // (op 10073) call()
v_accvgpr_write a40, 0.00000 // (op 10074) call()
v_accvgpr_write a41, 0.00000 // (op 10074) call()
v_accvgpr_write a42, 0.00000 // (op 10074) call()
v_accvgpr_write a43, 0.00000 // (op 10074) call()
v_accvgpr_write a44, 0.00000 // (op 10075) call()
v_accvgpr_write a45, 0.00000 // (op 10075) call()
v_accvgpr_write a46, 0.00000 // (op 10075) call()
v_accvgpr_write a47, 0.00000 // (op 10075) call()
v_accvgpr_write a48, 0.00000 // (op 10870) call()
v_accvgpr_write a49, 0.00000 // (op 10870) call()
v_accvgpr_write a50, 0.00000 // (op 10870) call()
v_accvgpr_write a51, 0.00000 // (op 10870) call()
v_accvgpr_write a52, 0.00000 // (op 11363) call()
v_accvgpr_write a53, 0.00000 // (op 11363) call()
v_accvgpr_write a54, 0.00000 // (op 11363) call()
v_accvgpr_write a55, 0.00000 // (op 11363) call()
v_accvgpr_write a56, 0.00000 // (op 11364) call()
v_accvgpr_write a57, 0.00000 // (op 11364) call()
v_accvgpr_write a58, 0.00000 // (op 11364) call()
v_accvgpr_write a59, 0.00000 // (op 11364) call()
v_accvgpr_write a60, 0.00000 // (op 11365) call()
v_accvgpr_write a61, 0.00000 // (op 11365) call()
v_accvgpr_write a62, 0.00000 // (op 11365) call()
v_accvgpr_write a63, 0.00000 // (op 11365) call()
v_and_b32 v42, -4, v1 // (op 27672) 
s_mul_i32 s52, s16, s8 // (op 27672) 
s_add_u32 s52, s52, s0 // (op 27672) 
v_sub_u32 v44, v1, v42 // (op 27672) 

s_mul_hi_u32 s53, s52, s13 // (op 27672) 
v_lshlrev_b32 v42, 5, v44 // (op 27672) 

s_sub_u32 s54, s52, s53 // (op 27672) 
v_lshrrev_b32 v44, 2, v1 // (op 27672) 
s_lshr_b32 s55, s54, 1 // (op 27672) 

s_add_u32 s54, s55, s53 // (op 27672) 


s_lshr_b32 s53, s54, s11 // (op 27672) 

s_and_b32 s54, s11, 2147483648 // (op 27672) 
s_cmp_eq_i32 s54, 0 // (op 27672) 
s_mov_b32 s55, scc // (op 27672) 

s_cmp_eq_i32 s55, 0 // (op 27672) 
s_cselect_b32 s54, s52, s53 // (op 27672) 


s_mul_hi_u32 s53, s54, s12 // (op 27672) 
s_sub_u32 s55, s54, s53 // (op 27672) 
s_lshr_b32 s68, s55, 1 // (op 27672) 

s_add_u32 s55, s68, s53 // (op 27672) 


s_lshr_b32 s53, s55, s10 // (op 27672) 

s_and_b32 s55, s10, 2147483648 // (op 27672) 
s_cmp_eq_i32 s55, 0 // (op 27672) 
s_mov_b32 s68, scc // (op 27672) 

s_cmp_eq_i32 s68, 0 // (op 27672) 
s_cselect_b32 s55, s54, s53 // (op 27672) 


v_lshl_add_u32 v45, s55, 7, v42 // (op 27672) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 27672) 
v_mov_b32 v47, 0 // (op 27672) convert
v_mov_b32 v46, v45 // (op 27672) convert

s_mul_i32 s53, s54, s68 // (op 27672) 


s_sub_u32 s54, s52, s53 // (op 27672) 


v_lshl_add_u32 v42, s54, 7, v44 // (op 27672) 


v_mul_lo_u32 v45, v42, s31 // (op 27672) most significant: low of low * high
v_mul_hi_u32 v48, v42, s30 // (op 27672) most significant: high of low * low
v_mul_lo_u32 v44, v42, s30 // (op 27672) least significant: low of low * low
v_add_u32 v45, v45, v48 // (op 27672) most significant: sum


v_add_co_u32 v48, s[52:53], v46, v44 // (op 27672) least significant half
v_addc_co_u32 v49, s[52:53], v47, v45, s[52:53] // (op 27672) most significant half

v_lshlrev_b64 v[48:49], 2, v[48:49] // (op 27672) 


v_lshrrev_b64 v[44:45], 3, v[48:49] // (op 27672) 

s_mul_i32 s53, s30, s3 // (op 27674) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27674) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27674) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27674) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27674) most significant: sum
s_add_u32 s53, s53, s55 // (op 27674) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27674) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27674) 

s_mov_b32 s68, s32 // (op 27674) reg expression
s_mov_b32 s69, s33 // (op 27674) reg expression
s_mov_b32 s70, s52 // (op 27674) reg expression
s_mov_b32 s71, 131072 // (op 27674) call()

v_and_b32 v42, -4, v1 // (op 27691) 
s_mul_i32 s52, s16, s8 // (op 27691) 
s_add_u32 s52, s52, s0 // (op 27691) 
v_sub_u32 v46, v1, v42 // (op 27691) 

s_mul_hi_u32 s53, s52, s13 // (op 27691) 
v_lshlrev_b32 v42, 5, v46 // (op 27691) 

s_sub_u32 s54, s52, s53 // (op 27691) 
v_lshrrev_b32 v46, 2, v1 // (op 27691) 
s_lshr_b32 s55, s54, 1 // (op 27691) 

s_add_u32 s54, s55, s53 // (op 27691) 


s_lshr_b32 s53, s54, s11 // (op 27691) 

s_and_b32 s54, s11, 2147483648 // (op 27691) 
s_cmp_eq_i32 s54, 0 // (op 27691) 
s_mov_b32 s55, scc // (op 27691) 

s_cmp_eq_i32 s55, 0 // (op 27691) 
s_cselect_b32 s54, s52, s53 // (op 27691) 


s_mul_hi_u32 s53, s54, s12 // (op 27691) 
s_sub_u32 s55, s54, s53 // (op 27691) 
s_lshr_b32 s72, s55, 1 // (op 27691) 

s_add_u32 s55, s72, s53 // (op 27691) 


s_lshr_b32 s53, s55, s10 // (op 27691) 

s_and_b32 s55, s10, 2147483648 // (op 27691) 
s_cmp_eq_i32 s55, 0 // (op 27691) 
s_mov_b32 s72, scc // (op 27691) 

s_cmp_eq_i32 s72, 0 // (op 27691) 
s_cselect_b32 s55, s54, s53 // (op 27691) 


v_lshl_add_u32 v47, s55, 7, v42 // (op 27691) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 27691) 
v_mov_b32 v49, 0 // (op 27691) convert
v_mov_b32 v48, v47 // (op 27691) convert

s_mul_i32 s53, s54, s72 // (op 27691) 


s_sub_u32 s54, s52, s53 // (op 27691) 


v_lshl_add_u32 v42, s54, 7, v46 // (op 27691) 


v_mul_lo_u32 v47, v42, s31 // (op 27691) most significant: low of low * high
v_mul_hi_u32 v50, v42, s30 // (op 27691) most significant: high of low * low
v_mul_lo_u32 v46, v42, s30 // (op 27691) least significant: low of low * low
v_add_u32 v47, v47, v50 // (op 27691) most significant: sum


v_add_co_u32 v50, s[52:53], v48, v46 // (op 27691) least significant half
v_addc_co_u32 v51, s[52:53], v49, v47, s[52:53] // (op 27691) most significant half

v_lshlrev_b64 v[50:51], 2, v[50:51] // (op 27691) 


v_lshrrev_b64 v[46:47], 3, v[50:51] // (op 27691) 

s_mul_i32 s53, s30, s3 // (op 27693) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27693) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27693) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27693) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27693) most significant: sum
s_add_u32 s53, s53, s55 // (op 27693) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27693) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27693) 

s_mov_b32 s68, s32 // (op 27693) reg expression
s_mov_b32 s69, s33 // (op 27693) reg expression
s_mov_b32 s70, s52 // (op 27693) reg expression
s_mov_b32 s71, 131072 // (op 27693) call()

v_and_b32 v42, -4, v1 // (op 27800) 
s_mul_i32 s52, s16, s8 // (op 27800) 
s_add_u32 s52, s52, s0 // (op 27800) 
v_sub_u32 v48, v1, v42 // (op 27800) 

s_mul_hi_u32 s53, s52, s13 // (op 27800) 
v_lshlrev_b32 v42, 5, v48 // (op 27800) 

s_sub_u32 s54, s52, s53 // (op 27800) 
v_lshrrev_b32 v48, 2, v1 // (op 27800) 
s_lshr_b32 s55, s54, 1 // (op 27800) 

s_add_u32 s54, s55, s53 // (op 27800) 


s_lshr_b32 s53, s54, s11 // (op 27800) 

s_and_b32 s54, s11, 2147483648 // (op 27800) 
s_cmp_eq_i32 s54, 0 // (op 27800) 
s_mov_b32 s55, scc // (op 27800) 

s_cmp_eq_i32 s55, 0 // (op 27800) 
s_cselect_b32 s54, s52, s53 // (op 27800) 


s_mul_hi_u32 s53, s54, s12 // (op 27800) 
s_sub_u32 s55, s54, s53 // (op 27800) 
s_lshr_b32 s72, s55, 1 // (op 27800) 

s_add_u32 s55, s72, s53 // (op 27800) 


s_lshr_b32 s53, s55, s10 // (op 27800) 

s_and_b32 s55, s10, 2147483648 // (op 27800) 
s_cmp_eq_i32 s55, 0 // (op 27800) 
s_mov_b32 s72, scc // (op 27800) 

s_cmp_eq_i32 s72, 0 // (op 27800) 
s_cselect_b32 s55, s54, s53 // (op 27800) 


s_mul_i32 s53, s55, s14 // (op 27800) 

s_sub_u32 s55, s54, s53 // (op 27800) 

v_lshl_add_u32 v49, s55, 7, v42 // (op 27800) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 27800) 
v_mov_b32 v51, 0 // (op 27800) convert
v_mov_b32 v50, v49 // (op 27800) convert

s_mul_i32 s53, s54, s72 // (op 27800) 


s_sub_u32 s54, s52, s53 // (op 27800) 


v_lshl_add_u32 v42, s54, 7, v48 // (op 27800) 


v_mul_lo_u32 v49, v42, s35 // (op 27800) most significant: low of low * high
v_mul_hi_u32 v52, v42, s34 // (op 27800) most significant: high of low * low
v_mul_lo_u32 v48, v42, s34 // (op 27800) least significant: low of low * low
v_add_u32 v49, v49, v52 // (op 27800) most significant: sum

v_add_co_u32 v48, s[52:53], v48, v50 // (op 27800) least significant half
v_addc_co_u32 v49, s[52:53], v49, v51, s[52:53] // (op 27800) most significant half



v_lshlrev_b64 v[50:51], 2, v[48:49] // (op 27800) 

v_lshrrev_b64 v[48:49], 3, v[50:51] // (op 27800) 

s_mul_i32 s53, s34, s3 // (op 27802) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 27802) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 27802) least significant
s_mul_hi_u32 s55, s34, s2 // (op 27802) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27802) most significant: sum
s_add_u32 s53, s53, s55 // (op 27802) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27802) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27802) 

s_mov_b32 s72, s36 // (op 27802) reg expression
s_mov_b32 s73, s37 // (op 27802) reg expression
s_mov_b32 s74, s52 // (op 27802) reg expression
s_mov_b32 s75, 131072 // (op 27802) call()

v_and_b32 v42, -4, v1 // (op 27819) 
s_mul_i32 s52, s16, s8 // (op 27819) 
s_add_u32 s52, s52, s0 // (op 27819) 
v_sub_u32 v50, v1, v42 // (op 27819) 

s_mul_hi_u32 s53, s52, s13 // (op 27819) 
v_lshlrev_b32 v42, 5, v50 // (op 27819) 

s_sub_u32 s54, s52, s53 // (op 27819) 
v_lshrrev_b32 v50, 2, v1 // (op 27819) 
s_lshr_b32 s55, s54, 1 // (op 27819) 

s_add_u32 s54, s55, s53 // (op 27819) 


s_lshr_b32 s53, s54, s11 // (op 27819) 

s_and_b32 s54, s11, 2147483648 // (op 27819) 
s_cmp_eq_i32 s54, 0 // (op 27819) 
s_mov_b32 s55, scc // (op 27819) 

s_cmp_eq_i32 s55, 0 // (op 27819) 
s_cselect_b32 s54, s52, s53 // (op 27819) 


s_mul_hi_u32 s53, s54, s12 // (op 27819) 
s_sub_u32 s55, s54, s53 // (op 27819) 
s_lshr_b32 s76, s55, 1 // (op 27819) 

s_add_u32 s55, s76, s53 // (op 27819) 


s_lshr_b32 s53, s55, s10 // (op 27819) 

s_and_b32 s55, s10, 2147483648 // (op 27819) 
s_cmp_eq_i32 s55, 0 // (op 27819) 
s_mov_b32 s76, scc // (op 27819) 

s_cmp_eq_i32 s76, 0 // (op 27819) 
s_cselect_b32 s55, s54, s53 // (op 27819) 


s_mul_i32 s53, s55, s14 // (op 27819) 

s_sub_u32 s55, s54, s53 // (op 27819) 

v_lshl_add_u32 v51, s55, 7, v42 // (op 27819) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 27819) 
v_mov_b32 v53, 0 // (op 27819) convert
v_mov_b32 v52, v51 // (op 27819) convert

s_mul_i32 s53, s54, s76 // (op 27819) 


s_sub_u32 s54, s52, s53 // (op 27819) 


v_lshl_add_u32 v42, s54, 7, v50 // (op 27819) 


v_mul_lo_u32 v51, v42, s35 // (op 27819) most significant: low of low * high
v_mul_hi_u32 v54, v42, s34 // (op 27819) most significant: high of low * low
v_mul_lo_u32 v50, v42, s34 // (op 27819) least significant: low of low * low
v_add_u32 v51, v51, v54 // (op 27819) most significant: sum

v_add_co_u32 v50, s[52:53], v50, v52 // (op 27819) least significant half
v_addc_co_u32 v51, s[52:53], v51, v53, s[52:53] // (op 27819) most significant half



v_lshlrev_b64 v[52:53], 2, v[50:51] // (op 27819) 

v_lshrrev_b64 v[50:51], 3, v[52:53] // (op 27819) 

s_mul_i32 s53, s34, s3 // (op 27821) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 27821) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 27821) least significant
s_mul_hi_u32 s55, s34, s2 // (op 27821) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27821) most significant: sum
s_add_u32 s53, s53, s55 // (op 27821) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27821) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27821) 

s_mov_b32 s72, s36 // (op 27821) reg expression
s_mov_b32 s73, s37 // (op 27821) reg expression
s_mov_b32 s74, s52 // (op 27821) reg expression
s_mov_b32 s75, 131072 // (op 27821) call()

v_and_b32 v42, -128, v1 // (op 27928) 
s_mul_i32 s52, s16, s8 // (op 27928) 
s_add_u32 s52, s52, s0 // (op 27928) 
v_sub_u32 v52, v1, v42 // (op 27928) 

s_mul_hi_u32 s53, s52, s13 // (op 27928) 
v_lshrrev_b32 v42, 7, v1 // (op 27928) 
s_sub_u32 s54, s52, s53 // (op 27928) 
v_lshlrev_b32 v53, 1, v42 // (op 27928) 

s_lshr_b32 s55, s54, 1 // (op 27928) 

s_add_u32 s54, s55, s53 // (op 27928) 


s_lshr_b32 s53, s54, s11 // (op 27928) 

s_and_b32 s54, s11, 2147483648 // (op 27928) 
s_cmp_eq_i32 s54, 0 // (op 27928) 
s_mov_b32 s55, scc // (op 27928) 

s_cmp_eq_i32 s55, 0 // (op 27928) 
s_cselect_b32 s54, s52, s53 // (op 27928) 


s_mul_hi_u32 s53, s54, s12 // (op 27928) 
s_sub_u32 s55, s54, s53 // (op 27928) 
s_lshr_b32 s76, s55, 1 // (op 27928) 

s_add_u32 s55, s76, s53 // (op 27928) 


s_lshr_b32 s53, s55, s10 // (op 27928) 

s_and_b32 s55, s10, 2147483648 // (op 27928) 
s_cmp_eq_i32 s55, 0 // (op 27928) 
s_mov_b32 s76, scc // (op 27928) 

s_cmp_eq_i32 s76, 0 // (op 27928) 
s_cselect_b32 s55, s54, s53 // (op 27928) 


v_lshl_add_u32 v42, s55, 7, v52 // (op 27928) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 27928) 
s_mul_i32 s53, s54, s76 // (op 27928) 


s_sub_u32 s54, s52, s53 // (op 27928) 


v_lshl_add_u32 v52, s54, 2, v53 // (op 27928) 


v_mul_lo_u32 v53, v52, s38 // (op 27928) 

v_add_u32 v52, v42, v53 // (op 27928) 


s_mul_i32 s53, s38, s43 // (op 27930) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27930) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27930) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27930) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27930) most significant: sum
s_add_u32 s53, s53, s55 // (op 27930) most significant: sum


s_mov_b32 s76, s40 // (op 27930) reg expression
s_mov_b32 s77, s41 // (op 27930) reg expression
s_mov_b32 s78, s52 // (op 27930) reg expression
s_mov_b32 s79, 131072 // (op 27930) call()

v_and_b32 v42, -128, v1 // (op 27947) 
s_mul_i32 s52, s16, s8 // (op 27947) 
s_add_u32 s52, s52, s0 // (op 27947) 
v_sub_u32 v53, v1, v42 // (op 27947) 

s_mul_hi_u32 s53, s52, s13 // (op 27947) 
v_lshrrev_b32 v42, 7, v1 // (op 27947) 
s_sub_u32 s54, s52, s53 // (op 27947) 
v_lshlrev_b32 v54, 1, v42 // (op 27947) 

s_lshr_b32 s55, s54, 1 // (op 27947) 

s_add_u32 s54, s55, s53 // (op 27947) 


s_lshr_b32 s53, s54, s11 // (op 27947) 

s_and_b32 s54, s11, 2147483648 // (op 27947) 
s_cmp_eq_i32 s54, 0 // (op 27947) 
s_mov_b32 s55, scc // (op 27947) 

s_cmp_eq_i32 s55, 0 // (op 27947) 
s_cselect_b32 s54, s52, s53 // (op 27947) 


s_mul_hi_u32 s53, s54, s12 // (op 27947) 
s_sub_u32 s55, s54, s53 // (op 27947) 
s_lshr_b32 s80, s55, 1 // (op 27947) 

s_add_u32 s55, s80, s53 // (op 27947) 


s_lshr_b32 s53, s55, s10 // (op 27947) 

s_and_b32 s55, s10, 2147483648 // (op 27947) 
s_cmp_eq_i32 s55, 0 // (op 27947) 
s_mov_b32 s80, scc // (op 27947) 

s_cmp_eq_i32 s80, 0 // (op 27947) 
s_cselect_b32 s55, s54, s53 // (op 27947) 


v_lshl_add_u32 v42, s55, 7, v53 // (op 27947) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 27947) 
s_mul_i32 s53, s54, s80 // (op 27947) 


s_sub_u32 s54, s52, s53 // (op 27947) 


v_lshl_add_u32 v53, s54, 2, v54 // (op 27947) 


v_mul_lo_u32 v54, v53, s38 // (op 27947) 

v_add_u32 v53, v42, v54 // (op 27947) 


s_mul_i32 s53, s38, s43 // (op 27949) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27949) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27949) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27949) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27949) most significant: sum
s_add_u32 s53, s53, s55 // (op 27949) most significant: sum


s_mov_b32 s76, s40 // (op 27949) reg expression
s_mov_b32 s77, s41 // (op 27949) reg expression
s_mov_b32 s78, s52 // (op 27949) reg expression
s_mov_b32 s79, 131072 // (op 27949) call()

v_and_b32 v42, -128, v1 // (op 28056) 
s_mul_i32 s52, s16, s8 // (op 28056) 
s_add_u32 s52, s52, s0 // (op 28056) 
v_sub_u32 v54, v1, v42 // (op 28056) 

s_mul_hi_u32 s53, s52, s13 // (op 28056) 
v_lshrrev_b32 v42, 7, v1 // (op 28056) 
s_sub_u32 s54, s52, s53 // (op 28056) 
v_lshlrev_b32 v55, 1, v42 // (op 28056) 

s_lshr_b32 s55, s54, 1 // (op 28056) 

s_add_u32 s54, s55, s53 // (op 28056) 


s_lshr_b32 s53, s54, s11 // (op 28056) 

s_and_b32 s54, s11, 2147483648 // (op 28056) 
s_cmp_eq_i32 s54, 0 // (op 28056) 
s_mov_b32 s55, scc // (op 28056) 

s_cmp_eq_i32 s55, 0 // (op 28056) 
s_cselect_b32 s54, s52, s53 // (op 28056) 


s_mul_hi_u32 s53, s54, s12 // (op 28056) 
s_sub_u32 s55, s54, s53 // (op 28056) 
s_lshr_b32 s80, s55, 1 // (op 28056) 

s_add_u32 s55, s80, s53 // (op 28056) 


s_lshr_b32 s53, s55, s10 // (op 28056) 

s_and_b32 s55, s10, 2147483648 // (op 28056) 
s_cmp_eq_i32 s55, 0 // (op 28056) 
s_mov_b32 s80, scc // (op 28056) 

s_cmp_eq_i32 s80, 0 // (op 28056) 
s_cselect_b32 s55, s54, s53 // (op 28056) 


s_mul_i32 s53, s55, s14 // (op 28056) 

s_sub_u32 s55, s54, s53 // (op 28056) 

v_lshl_add_u32 v42, s55, 7, v54 // (op 28056) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 28056) 
s_mul_i32 s53, s54, s80 // (op 28056) 


s_sub_u32 s54, s52, s53 // (op 28056) 


v_lshl_add_u32 v54, s54, 2, v55 // (op 28056) 


v_mul_lo_u32 v55, v54, s44 // (op 28056) 
v_add_u32 v55, v55, v42 // (op 28056) 


s_mul_i32 s53, s44, s43 // (op 28058) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 28058) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 28058) least significant
s_mul_hi_u32 s55, s44, s42 // (op 28058) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 28058) most significant: sum
s_add_u32 s53, s53, s55 // (op 28058) most significant: sum


s_mov_b32 s80, s46 // (op 28058) reg expression
s_mov_b32 s81, s47 // (op 28058) reg expression
s_mov_b32 s82, s52 // (op 28058) reg expression
s_mov_b32 s83, 131072 // (op 28058) call()

v_and_b32 v42, -128, v1 // (op 28075) 
s_mul_i32 s52, s16, s8 // (op 28075) 
s_add_u32 s52, s52, s0 // (op 28075) 
v_sub_u32 v54, v1, v42 // (op 28075) 

s_mul_hi_u32 s53, s52, s13 // (op 28075) 
v_lshrrev_b32 v42, 7, v1 // (op 28075) 
s_sub_u32 s54, s52, s53 // (op 28075) 
v_lshlrev_b32 v56, 1, v42 // (op 28075) 

s_lshr_b32 s55, s54, 1 // (op 28075) 

s_add_u32 s54, s55, s53 // (op 28075) 


s_lshr_b32 s53, s54, s11 // (op 28075) 

s_and_b32 s54, s11, 2147483648 // (op 28075) 
s_cmp_eq_i32 s54, 0 // (op 28075) 
s_mov_b32 s55, scc // (op 28075) 

s_cmp_eq_i32 s55, 0 // (op 28075) 
s_cselect_b32 s54, s52, s53 // (op 28075) 


s_mul_hi_u32 s53, s54, s12 // (op 28075) 
s_sub_u32 s55, s54, s53 // (op 28075) 
s_lshr_b32 s84, s55, 1 // (op 28075) 

s_add_u32 s55, s84, s53 // (op 28075) 


s_lshr_b32 s53, s55, s10 // (op 28075) 

s_and_b32 s55, s10, 2147483648 // (op 28075) 
s_cmp_eq_i32 s55, 0 // (op 28075) 
s_mov_b32 s84, scc // (op 28075) 

s_cmp_eq_i32 s84, 0 // (op 28075) 
s_cselect_b32 s55, s54, s53 // (op 28075) 


s_mul_i32 s53, s55, s14 // (op 28075) 

s_sub_u32 s55, s54, s53 // (op 28075) 

v_lshl_add_u32 v42, s55, 7, v54 // (op 28075) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 28075) 
s_mul_i32 s53, s54, s84 // (op 28075) 


s_sub_u32 s54, s52, s53 // (op 28075) 


v_lshl_add_u32 v54, s54, 2, v56 // (op 28075) 


v_mul_lo_u32 v56, v54, s44 // (op 28075) 
v_add_u32 v56, v56, v42 // (op 28075) 


s_mul_i32 s53, s44, s43 // (op 28077) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 28077) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 28077) least significant
s_mul_hi_u32 s55, s44, s42 // (op 28077) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 28077) most significant: sum
s_add_u32 s53, s53, s55 // (op 28077) most significant: sum


s_mov_b32 s80, s46 // (op 28077) reg expression
s_mov_b32 s81, s47 // (op 28077) reg expression
s_mov_b32 s82, s52 // (op 28077) reg expression
s_mov_b32 s83, 131072 // (op 28077) call()

v_and_b32 v42, -4, v1 // (op 28252) 
v_sub_u32 v54, v1, v42 // (op 28252) 

v_lshlrev_b32 v42, 5, v54 // (op 28252) 

v_lshrrev_b32 v54, 2, v1 // (op 28252) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 28252) 


v_lshlrev_b32 v42, 2, v57 // (op 28252) 

v_lshrrev_b32 v54, 3, v42 // (op 28252) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 28252) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30151) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27514) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27514) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27514) 


v_mov_b64 v[60:61], v[48:49] // (op 27514) reg expression
s_mov_b32 m0, s52 // (op 27514) reg expression
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 27514) Load value direct to lds
v_add_co_u32 v60, s[54:55], v60, v58 // (op 27514) least significant half
v_addc_co_u32 v61, s[54:55], 0, v61, s[54:55] // (op 27514) most significant half

s_add_u32 m0, m0, 4096 // (op 27514) 
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 27514) Load value direct to lds




v_mov_b64 v[58:59], s[44:45] // (op 26989) convert

v_mov_b32 v42, v55 // (op 26989) reg expression
buffer_load_ubyte v60, v42, s[80:83], 0 offen offset: 0 // (op 26989) Load value
v_add_u32 v42, v42, v58 // (op 26989) 
buffer_load_ubyte v61, v42, s[80:83], 0 offen offset: 0 // (op 26989) Load value


v_and_b32 v42, -4, v1 // (op 28177) 
v_sub_u32 v54, v1, v42 // (op 28177) 

v_lshlrev_b32 v42, 5, v54 // (op 28177) 

v_lshrrev_b32 v54, 2, v1 // (op 28177) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 28177) 


v_lshlrev_b32 v42, 2, v57 // (op 28177) 

v_lshrrev_b32 v54, 3, v42 // (op 28177) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 28177) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30132) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27519) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27519) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27519) 


v_mov_b64 v[62:63], v[44:45] // (op 27519) reg expression
s_add_u32 s53, 8192, s52 // (op 27519) 
s_mov_b32 m0, s53 // (op 27519) reg expression
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 27519) Load value direct to lds
v_add_co_u32 v62, s[54:55], v62, v58 // (op 27519) least significant half
v_addc_co_u32 v63, s[54:55], 0, v63, s[54:55] // (op 27519) most significant half

s_add_u32 m0, m0, 4096 // (op 27519) 
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 27519) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 26995) convert

v_mov_b32 v42, v52 // (op 26995) reg expression
buffer_load_ubyte v62, v42, s[76:79], 0 offen offset: 0 // (op 26995) Load value
v_add_u32 v42, v42, v58 // (op 26995) 
buffer_load_ubyte v63, v42, s[76:79], 0 offen offset: 0 // (op 26995) Load value


v_and_b32 v42, -4, v1 // (op 29354) 
v_sub_u32 v54, v1, v42 // (op 29354) 

v_lshlrev_b32 v42, 5, v54 // (op 29354) 

v_lshrrev_b32 v54, 2, v1 // (op 29354) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29354) 


v_lshlrev_b32 v42, 2, v57 // (op 29354) 

v_lshrrev_b32 v54, 3, v42 // (op 29354) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29354) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30398) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27524) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27524) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27524) 


s_lshl_b64 s[54:55], s[34:35], 7 // (op 27524) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27524) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27524) 

v_mov_b32 v42, s54 // (op 27524) 
v_mov_b32 v54, s55 // (op 27524) 
v_add_co_u32 v64, s[84:85], v48, v42 // (op 27524) least significant half
v_addc_co_u32 v65, s[84:85], v49, v54, s[84:85] // (op 27524) most significant half




s_add_u32 s53, 16384, s52 // (op 27524) 
s_mov_b32 m0, s53 // (op 27524) reg expression
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 27524) Load value direct to lds
v_add_co_u32 v64, s[54:55], v64, v58 // (op 27524) least significant half
v_addc_co_u32 v65, s[54:55], 0, v65, s[54:55] // (op 27524) most significant half

s_add_u32 m0, m0, 4096 // (op 27524) 
buffer_load_dwordx4 v64, s[72:75], 0 offen offset: 0 lds // (op 27524) Load value direct to lds





v_mov_b64 v[58:59], s[44:45] // (op 27001) convert

s_lshl_b32 s52, s44, 2 // (op 27001) 
v_add_u32 v42, v55, s52 // (op 27001) 

buffer_load_ubyte v64, v42, s[80:83], 0 offen offset: 0 // (op 27001) Load value
v_add_u32 v42, v42, v58 // (op 27001) 
buffer_load_ubyte v65, v42, s[80:83], 0 offen offset: 0 // (op 27001) Load value


v_and_b32 v42, -4, v1 // (op 29310) 
v_sub_u32 v54, v1, v42 // (op 29310) 

v_lshlrev_b32 v42, 5, v54 // (op 29310) 

v_lshrrev_b32 v54, 2, v1 // (op 29310) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29310) 


v_lshlrev_b32 v42, 2, v57 // (op 29310) 

v_lshrrev_b32 v54, 3, v42 // (op 29310) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29310) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30388) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27526) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27526) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27526) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27526) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27526) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27526) 

v_mov_b32 v42, s54 // (op 27526) 
v_mov_b32 v54, s55 // (op 27526) 
v_add_co_u32 v66, s[84:85], v44, v42 // (op 27526) least significant half
v_addc_co_u32 v67, s[84:85], v45, v54, s[84:85] // (op 27526) most significant half




s_add_u32 s53, 24576, s52 // (op 27526) 
s_mov_b32 m0, s53 // (op 27526) reg expression
buffer_load_dwordx4 v66, s[68:71], 0 offen offset: 0 lds // (op 27526) Load value direct to lds
v_add_co_u32 v66, s[54:55], v66, v58 // (op 27526) least significant half
v_addc_co_u32 v67, s[54:55], 0, v67, s[54:55] // (op 27526) most significant half

s_add_u32 m0, m0, 4096 // (op 27526) 
buffer_load_dwordx4 v66, s[68:71], 0 offen offset: 0 lds // (op 27526) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 27007) convert

s_lshl_b32 s52, s38, 2 // (op 27007) 
v_add_u32 v42, v52, s52 // (op 27007) 

buffer_load_ubyte v66, v42, s[76:79], 0 offen offset: 0 // (op 27007) Load value
v_add_u32 v42, v42, v58 // (op 27007) 
buffer_load_ubyte v67, v42, s[76:79], 0 offen offset: 0 // (op 27007) Load value


v_and_b32 v42, -4, v1 // (op 29500) 
v_sub_u32 v54, v1, v42 // (op 29500) 

v_lshlrev_b32 v42, 5, v54 // (op 29500) 

v_lshrrev_b32 v54, 2, v1 // (op 29500) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29500) 


v_lshlrev_b32 v42, 2, v57 // (op 29500) 

v_lshrrev_b32 v54, 3, v42 // (op 29500) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29500) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30430) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27528) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27528) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27528) 


s_lshl_b64 s[54:55], s[34:35], 7 // (op 27528) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27528) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27528) 

s_lshl_b64 s[84:85], s[54:55], 1 // (op 27528) 

v_mov_b32 v42, s84 // (op 27528) 
v_mov_b32 v54, s85 // (op 27528) 
v_add_co_u32 v68, s[54:55], v48, v42 // (op 27528) least significant half
v_addc_co_u32 v69, s[54:55], v49, v54, s[54:55] // (op 27528) most significant half




s_add_u32 s53, 32768, s52 // (op 27528) 
s_mov_b32 m0, s53 // (op 27528) reg expression
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 27528) Load value direct to lds
v_add_co_u32 v68, s[54:55], v68, v58 // (op 27528) least significant half
v_addc_co_u32 v69, s[54:55], 0, v69, s[54:55] // (op 27528) most significant half

s_add_u32 m0, m0, 4096 // (op 27528) 
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 27528) Load value direct to lds





v_mov_b64 v[58:59], s[44:45] // (op 27013) convert

s_lshl_b32 s52, s44, 3 // (op 27013) 
v_add_u32 v42, v55, s52 // (op 27013) 

buffer_load_ubyte v68, v42, s[80:83], 0 offen offset: 0 // (op 27013) Load value
v_add_u32 v42, v42, v58 // (op 27013) 
buffer_load_ubyte v69, v42, s[80:83], 0 offen offset: 0 // (op 27013) Load value


v_and_b32 v42, -4, v1 // (op 29456) 
v_sub_u32 v54, v1, v42 // (op 29456) 

v_lshlrev_b32 v42, 5, v54 // (op 29456) 

v_lshrrev_b32 v54, 2, v1 // (op 29456) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29456) 


v_lshlrev_b32 v42, 2, v57 // (op 29456) 

v_lshrrev_b32 v54, 3, v42 // (op 29456) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29456) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30420) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27530) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27530) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27530) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27530) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27530) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27530) 

s_lshl_b64 s[84:85], s[54:55], 1 // (op 27530) 

v_mov_b32 v42, s84 // (op 27530) 
v_mov_b32 v54, s85 // (op 27530) 
v_add_co_u32 v70, s[54:55], v44, v42 // (op 27530) least significant half
v_addc_co_u32 v71, s[54:55], v45, v54, s[54:55] // (op 27530) most significant half




s_add_u32 s53, 40960, s52 // (op 27530) 
s_mov_b32 m0, s53 // (op 27530) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 27530) Load value direct to lds
v_add_co_u32 v70, s[54:55], v70, v58 // (op 27530) least significant half
v_addc_co_u32 v71, s[54:55], 0, v71, s[54:55] // (op 27530) most significant half

s_add_u32 m0, m0, 4096 // (op 27530) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 27530) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 27019) convert

s_lshl_b32 s52, s38, 3 // (op 27019) 
v_add_u32 v42, v52, s52 // (op 27019) 

buffer_load_ubyte v70, v42, s[76:79], 0 offen offset: 0 // (op 27019) Load value
v_add_u32 v42, v42, v58 // (op 27019) 
buffer_load_ubyte v71, v42, s[76:79], 0 offen offset: 0 // (op 27019) Load value


v_and_b32 v42, -4, v1 // (op 29646) 
v_sub_u32 v54, v1, v42 // (op 29646) 

v_lshlrev_b32 v42, 5, v54 // (op 29646) 

v_lshrrev_b32 v54, 2, v1 // (op 29646) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29646) 


v_lshlrev_b32 v42, 2, v57 // (op 29646) 

v_lshrrev_b32 v54, 3, v42 // (op 29646) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29646) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30462) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27532) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27532) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27532) 


s_lshl_b64 s[54:55], s[34:35], 7 // (op 27532) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27532) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27532) 

s_mul_i32 s85, 3, s55 // (op 27532) most significant: low of low * high
s_mul_i32 s84, 3, s54 // (op 27532) least significant
s_mul_hi_u32 s53, 3, s54 // (op 27532) most significant: high of low * low
s_add_u32 s85, s85, s53 // (op 27532) most significant: sum


v_mov_b32 v42, s84 // (op 27532) 
v_mov_b32 v54, s85 // (op 27532) 
v_add_co_u32 v72, s[54:55], v48, v42 // (op 27532) least significant half
v_addc_co_u32 v73, s[54:55], v49, v54, s[54:55] // (op 27532) most significant half




s_add_u32 s53, 49152, s52 // (op 27532) 
s_mov_b32 m0, s53 // (op 27532) reg expression
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 27532) Load value direct to lds
v_add_co_u32 v72, s[54:55], v72, v58 // (op 27532) least significant half
v_addc_co_u32 v73, s[54:55], 0, v73, s[54:55] // (op 27532) most significant half

s_add_u32 m0, m0, 4096 // (op 27532) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 27532) Load value direct to lds






v_mov_b64 v[48:49], s[44:45] // (op 27025) convert

s_lshl_b32 s52, s44, 2 // (op 27025) 
s_mul_i32 s53, 3, s52 // (op 27025) 

v_add_u32 v42, v55, s53 // (op 27025) 

buffer_load_ubyte v58, v42, s[80:83], 0 offen offset: 0 // (op 27025) Load value
v_add_u32 v42, v42, v48 // (op 27025) 
buffer_load_ubyte v59, v42, s[80:83], 0 offen offset: 0 // (op 27025) Load value



v_and_b32 v42, -4, v1 // (op 29602) 
v_sub_u32 v57, v1, v42 // (op 29602) 

v_lshlrev_b32 v42, 5, v57 // (op 29602) 

v_lshrrev_b32 v57, 2, v1 // (op 29602) 
v_lshl_add_u32 v48, v57, 7, v42 // (op 29602) 


v_lshlrev_b32 v42, 2, v48 // (op 29602) 

v_lshrrev_b32 v57, 3, v42 // (op 29602) 

s_nop 0
v_readfirstlane_b32 s52, v57 // (op 29602) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30452) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27534) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27534) 

v_lshrrev_b64 v[48:49], 3, s[84:85] // (op 27534) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27534) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27534) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27534) 

s_mul_i32 s85, 3, s55 // (op 27534) most significant: low of low * high
s_mul_i32 s84, 3, s54 // (op 27534) least significant
s_mul_hi_u32 s53, 3, s54 // (op 27534) most significant: high of low * low
s_add_u32 s85, s85, s53 // (op 27534) most significant: sum


v_mov_b32 v42, s84 // (op 27534) 
v_mov_b32 v57, s85 // (op 27534) 
v_add_co_u32 v54, s[54:55], v44, v42 // (op 27534) least significant half
v_addc_co_u32 v55, s[54:55], v45, v57, s[54:55] // (op 27534) most significant half




s_add_u32 s53, 57344, s52 // (op 27534) 
s_mov_b32 m0, s53 // (op 27534) reg expression
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27534) Load value direct to lds
v_add_co_u32 v54, s[54:55], v54, v48 // (op 27534) least significant half
v_addc_co_u32 v55, s[54:55], 0, v55, s[54:55] // (op 27534) most significant half

s_add_u32 m0, m0, 4096 // (op 27534) 
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27534) Load value direct to lds






v_mov_b64 v[44:45], s[38:39] // (op 27031) convert

s_lshl_b32 s52, s38, 2 // (op 27031) 
s_mul_i32 s53, 3, s52 // (op 27031) 

v_add_u32 v42, v52, s53 // (op 27031) 

buffer_load_ubyte v48, v42, s[76:79], 0 offen offset: 0 // (op 27031) Load value
v_add_u32 v42, v42, v44 // (op 27031) 
buffer_load_ubyte v49, v42, s[76:79], 0 offen offset: 0 // (op 27031) Load value



v_and_b32 v42, -128, v1 // (op 28404) 
v_sub_u32 v52, v1, v42 // (op 28404) 

v_lshrrev_b32 v42, 7, v1 // (op 28404) 
v_lshl_add_u32 v57, v42, 8, v52 // (op 28404) 


s_barrier  // (op 30192) 



v_add_u32 v42, 65536, v57 // (op 27034) 
s_waitcnt vmcnt(29)
ds_write_b8 v42, v60 // (op 27034) Store local data 

v_add_u32 v42, 65664, v57 // (op 27034) 
s_waitcnt vmcnt(28)
ds_write_b8 v42, v61 // (op 27034) Store local data 


v_and_b32 v42, -128, v1 // (op 28328) 
v_sub_u32 v52, v1, v42 // (op 28328) 

v_lshrrev_b32 v42, 7, v1 // (op 28328) 
v_lshl_add_u32 v57, v42, 8, v52 // (op 28328) 


s_barrier  // (op 30171) 



v_add_u32 v42, 66048, v57 // (op 27036) 
s_waitcnt vmcnt(25)
ds_write_b8 v42, v62 // (op 27036) Store local data 

v_add_u32 v42, 66176, v57 // (op 27036) 
s_waitcnt vmcnt(24)
ds_write_b8 v42, v63 // (op 27036) Store local data 


s_barrier  // (op 30148) 



ds_read_b64_tr_b4 v[72:73], v39 // (op 27079) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v39 offset:1024 // (op 27079) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30189) 


v_add_u32 v42, 65536, v40 // (op 27119) 
ds_read_u8 v52, v42 // (op 27119) Load local data 




ds_read_b64_tr_b4 v[76:77], v39 offset:16 // (op 27084) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v39 offset:1040 // (op 27084) Transpose load local data 


v_add_u32 v42, 65568, v40 // (op 27124) 
ds_read_u8 v57, v42 // (op 27124) Load local data 




ds_read_b64_tr_b4 v[80:81], v39 offset:32 // (op 27089) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v39 offset:1056 // (op 27089) Transpose load local data 


v_add_u32 v42, 65600, v40 // (op 27129) 
ds_read_u8 v44, v42 // (op 27129) Load local data 




ds_read_b64_tr_b4 v[84:85], v39 offset:48 // (op 27094) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v39 offset:1072 // (op 27094) Transpose load local data 


v_add_u32 v42, 65632, v40 // (op 27134) 
ds_read_u8 v45, v42 // (op 27134) Load local data 

s_barrier  // (op 30129) 



ds_read_b64_tr_b4 v[88:89], v37 offset:8192 // (op 27059) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v37 offset:9216 // (op 27059) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30168) 


v_add_u32 v42, 66048, v36 // (op 27099) 
ds_read_u8 v54, v42 // (op 27099) Load local data 




ds_read_b64_tr_b4 v[92:93], v37 offset:8208 // (op 27064) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v37 offset:9232 // (op 27064) Transpose load local data 


v_add_u32 v42, 66080, v36 // (op 27104) 
ds_read_u8 v55, v42 // (op 27104) Load local data 




ds_read_b64_tr_b4 v[96:97], v37 offset:8224 // (op 27069) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v37 offset:9248 // (op 27069) Transpose load local data 


v_add_u32 v42, 66112, v36 // (op 27109) 
ds_read_u8 v100, v42 // (op 27109) Load local data 




ds_read_b64_tr_b4 v[102:103], v37 offset:8240 // (op 27074) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v37 offset:9264 // (op 27074) Transpose load local data 


v_add_u32 v42, 66144, v36 // (op 27114) 
ds_read_u8 v101, v42 // (op 27114) Load local data 

v_and_b32 v42, -4, v1 // (op 28149) 
v_sub_u32 v106, v1, v42 // (op 28149) 

v_lshlrev_b32 v42, 5, v106 // (op 28149) 

v_lshrrev_b32 v106, 2, v1 // (op 28149) 
v_lshl_add_u32 v107, v106, 7, v42 // (op 28149) 


v_lshlrev_b32 v42, 2, v107 // (op 28149) 

v_lshrrev_b32 v106, 3, v42 // (op 28149) 

s_nop 0
v_readfirstlane_b32 s52, v106 // (op 28149) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 28224) 
v_sub_u32 v106, v1, v42 // (op 28224) 

v_lshlrev_b32 v42, 5, v106 // (op 28224) 

v_lshrrev_b32 v106, 2, v1 // (op 28224) 
v_lshl_add_u32 v107, v106, 7, v42 // (op 28224) 


v_lshlrev_b32 v42, 2, v107 // (op 28224) 

v_lshrrev_b32 v106, 3, v42 // (op 28224) 

s_nop 0
v_readfirstlane_b32 s53, v106 // (op 28224) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 28299) 
v_sub_u32 v106, v1, v42 // (op 28299) 

v_lshrrev_b32 v42, 7, v1 // (op 28299) 
v_lshl_add_u32 v107, v42, 8, v106 // (op 28299) 


v_and_b32 v42, -128, v1 // (op 28375) 
v_sub_u32 v106, v1, v42 // (op 28375) 

v_lshrrev_b32 v42, 7, v1 // (op 28375) 
v_lshl_add_u32 v108, v42, 8, v106 // (op 28375) 


v_and_b32 v42, -4, v1 // (op 29295) 
v_sub_u32 v106, v1, v42 // (op 29295) 

v_lshlrev_b32 v42, 5, v106 // (op 29295) 

v_lshrrev_b32 v106, 2, v1 // (op 29295) 
v_lshl_add_u32 v109, v106, 7, v42 // (op 29295) 


v_lshlrev_b32 v42, 2, v109 // (op 29295) 

v_lshrrev_b32 v106, 3, v42 // (op 29295) 

s_nop 0
v_readfirstlane_b32 s54, v106 // (op 29295) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29339) 
v_sub_u32 v106, v1, v42 // (op 29339) 

v_lshlrev_b32 v42, 5, v106 // (op 29339) 

v_lshrrev_b32 v106, 2, v1 // (op 29339) 
v_lshl_add_u32 v109, v106, 7, v42 // (op 29339) 


v_lshlrev_b32 v42, 2, v109 // (op 29339) 

v_lshrrev_b32 v106, 3, v42 // (op 29339) 

s_nop 0
v_readfirstlane_b32 s55, v106 // (op 29339) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29383) 
v_sub_u32 v106, v1, v42 // (op 29383) 

v_lshrrev_b32 v42, 7, v1 // (op 29383) 
v_lshl_add_u32 v109, v42, 8, v106 // (op 29383) 


v_and_b32 v42, -128, v1 // (op 29412) 
v_sub_u32 v106, v1, v42 // (op 29412) 

v_lshrrev_b32 v42, 7, v1 // (op 29412) 
v_lshl_add_u32 v110, v42, 8, v106 // (op 29412) 


v_and_b32 v42, -4, v1 // (op 29441) 
v_sub_u32 v106, v1, v42 // (op 29441) 

v_lshlrev_b32 v42, 5, v106 // (op 29441) 

v_lshrrev_b32 v106, 2, v1 // (op 29441) 
v_lshl_add_u32 v111, v106, 7, v42 // (op 29441) 


v_lshlrev_b32 v42, 2, v111 // (op 29441) 

v_lshrrev_b32 v106, 3, v42 // (op 29441) 

s_nop 0
v_readfirstlane_b32 s84, v106 // (op 29441) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29485) 
v_sub_u32 v106, v1, v42 // (op 29485) 

v_lshlrev_b32 v42, 5, v106 // (op 29485) 

v_lshrrev_b32 v106, 2, v1 // (op 29485) 
v_lshl_add_u32 v111, v106, 7, v42 // (op 29485) 


v_lshlrev_b32 v42, 2, v111 // (op 29485) 

v_lshrrev_b32 v106, 3, v42 // (op 29485) 

s_nop 0
v_readfirstlane_b32 s85, v106 // (op 29485) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29529) 
v_sub_u32 v106, v1, v42 // (op 29529) 

v_lshrrev_b32 v42, 7, v1 // (op 29529) 
v_lshl_add_u32 v111, v42, 8, v106 // (op 29529) 


v_and_b32 v42, -128, v1 // (op 29558) 
v_sub_u32 v106, v1, v42 // (op 29558) 

v_lshrrev_b32 v42, 7, v1 // (op 29558) 
v_lshl_add_u32 v112, v42, 8, v106 // (op 29558) 


v_and_b32 v42, -4, v1 // (op 29587) 
v_sub_u32 v106, v1, v42 // (op 29587) 

v_lshlrev_b32 v42, 5, v106 // (op 29587) 

v_lshrrev_b32 v106, 2, v1 // (op 29587) 
v_lshl_add_u32 v113, v106, 7, v42 // (op 29587) 


v_lshlrev_b32 v42, 2, v113 // (op 29587) 

v_lshrrev_b32 v106, 3, v42 // (op 29587) 

s_nop 0
v_readfirstlane_b32 s86, v106 // (op 29587) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29631) 
v_sub_u32 v106, v1, v42 // (op 29631) 

v_lshlrev_b32 v42, 5, v106 // (op 29631) 

v_lshrrev_b32 v106, 2, v1 // (op 29631) 
v_lshl_add_u32 v113, v106, 7, v42 // (op 29631) 


v_lshlrev_b32 v42, 2, v113 // (op 29631) 

v_lshrrev_b32 v106, 3, v42 // (op 29631) 

s_nop 0
v_readfirstlane_b32 s87, v106 // (op 29631) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29675) 
v_sub_u32 v106, v1, v42 // (op 29675) 

v_lshrrev_b32 v42, 7, v1 // (op 29675) 
v_lshl_add_u32 v113, v42, 8, v106 // (op 29675) 


v_and_b32 v42, -128, v1 // (op 29704) 
v_sub_u32 v106, v1, v42 // (op 29704) 

v_lshrrev_b32 v42, 7, v1 // (op 29704) 
v_lshl_add_u32 v114, v42, 8, v106 // (op 29704) 


s_mov_b32 s1, 0 // (op 9582) call()
s_and_b32 s88, s7, 4294967292 // (op 9581) 
s_cmp_lt_u32 s1, s88 // (op 9581) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9581 // (op 9581) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9581 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9581:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[72:75], a[0:3], v54, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
s_barrier  // (op 30154) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[76:79], a[4:7], v54, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7300) 
s_add_u32 s1, s1, 4 // (op 9583) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[80:83], a[8:11], v54, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7598) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27391) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[84:87], a[12:15], v54, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7896) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27391) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27391) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[72:75], a[16:19], v55, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8699) 
v_mov_b64 v[118:119], v[50:51] // (op 27391) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[76:79], a[20:23], v55, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8607) 
s_lshl_b64 s[88:89], s[34:35], 7 // (op 27391) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[80:83], a[24:27], v55, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8515) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27391) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[84:87], a[28:31], v55, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8423) 
s_lshr_b64 s[88:89], s[90:91], 3 // (op 27391) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27391) 

v_mov_b32 v42, s90 // (op 27391) 
v_mov_b32 v106, s91 // (op 27391) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[72:75], a[32:35], v100, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9989) 
v_add_co_u32 v120, s[88:89], v118, v42 // (op 27391) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[76:79], a[36:39], v100, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9897) 
v_addc_co_u32 v121, s[88:89], v119, v106, s[88:89] // (op 27391) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[80:83], a[40:43], v100, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9805) 
s_mov_b32 m0, s53 // (op 27391) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[84:87], a[44:47], v100, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9713) 
buffer_load_dwordx4 v120, s[72:75], 0 offen offset: 0 lds // (op 27391) Load value direct to lds
v_add_co_u32 v120, s[88:89], v120, v116 // (op 27391) least significant half
v_addc_co_u32 v121, s[88:89], 0, v121, s[88:89] // (op 27391) most significant half

s_add_u32 m0, m0, 4096 // (op 27391) 
buffer_load_dwordx4 v120, s[72:75], 0 offen offset: 0 lds // (op 27391) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[72:75], a[48:51], v101, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11279) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[76:79], a[52:55], v101, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11187) 
v_mov_b64 v[116:117], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[80:83], a[56:59], v101, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11095) 
v_mov_b32 v42, v56 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[84:87], a[60:63], v101, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11003) 
s_lshl_b32 s88, s44, 4 // (op 25) 
v_add_u32 v106, v42, s88 // (op 25) 

buffer_load_ubyte v60, v106, s[80:83], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v106, v106, v116 // (op 25) 
buffer_load_ubyte v61, v106, s[80:83], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 30135) 

s_lshl_b64 s[88:89], s[30:31], 6 // (op 27386) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27386) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27386) 


v_mov_b64 v[120:121], v[46:47] // (op 27386) 
s_lshl_b64 s[88:89], s[30:31], 7 // (op 27386) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27386) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27386) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27386) 

v_mov_b32 v106, s90 // (op 27386) 
v_mov_b32 v115, s91 // (op 27386) 
v_add_co_u32 v122, s[88:89], v120, v106 // (op 27386) least significant half
v_addc_co_u32 v123, s[88:89], v121, v115, s[88:89] // (op 27386) most significant half




s_add_u32 s88, 8192, s52 // (op 27386) 
s_nop 0
s_mov_b32 m0, s88 // (op 27386) reg expression
buffer_load_dwordx4 v122, s[68:71], 0 offen offset: 0 lds // (op 27386) Load value direct to lds
v_add_co_u32 v122, s[90:91], v122, v116 // (op 27386) least significant half
v_addc_co_u32 v123, s[90:91], 0, v123, s[90:91] // (op 27386) most significant half

s_add_u32 m0, m0, 4096 // (op 27386) 
buffer_load_dwordx4 v122, s[68:71], 0 offen offset: 0 lds // (op 27386) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 17) convert

v_mov_b32 v106, v53 // (op 17) 
s_lshl_b32 s88, s38, 4 // (op 17) 
v_add_u32 v115, v106, s88 // (op 17) 

buffer_load_ubyte v62, v115, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v115, v115, v116 // (op 17) 
buffer_load_ubyte v63, v115, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 30414) 



v_add_u32 v115, 66560, v110 // (op 7033) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v64 // (op 7033) Store local data 

v_add_u32 v115, 66688, v110 // (op 7033) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v65 // (op 7033) Store local data 

s_barrier  // (op 30408) 



v_add_u32 v115, 67072, v109 // (op 7032) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v66 // (op 7032) Store local data 

v_add_u32 v115, 67200, v109 // (op 7032) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v67 // (op 7032) Store local data 

s_barrier  // (op 30394) 



ds_read_b64_tr_b4 v[122:123], v24 offset:16384 // (op 7025) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v24 offset:17408 // (op 7025) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30411) 


v_add_u32 v115, 66560, v25 // (op 7029) 
ds_read_u8 v116, v115 // (op 7029) Load local data 




ds_read_b64_tr_b4 v[126:127], v24 offset:16400 // (op 7307) Transpose load local data 
ds_read_b64_tr_b4 v[128:129], v24 offset:17424 // (op 7307) Transpose load local data 


v_add_u32 v115, 66592, v25 // (op 7311) 
ds_read_u8 v55, v115 // (op 7311) Load local data 




ds_read_b64_tr_b4 v[92:93], v24 offset:16416 // (op 7605) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v24 offset:17440 // (op 7605) Transpose load local data 


v_add_u32 v115, 66624, v25 // (op 7609) 
ds_read_u8 v100, v115 // (op 7609) Load local data 




ds_read_b64_tr_b4 v[96:97], v24 offset:16432 // (op 7903) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v24 offset:17456 // (op 7903) Transpose load local data 


v_add_u32 v115, 66656, v25 // (op 7907) 
ds_read_u8 v117, v115 // (op 7907) Load local data 

s_barrier  // (op 30385) 
s_barrier  // (op 30893) 



ds_read_b64_tr_b4 v[88:89], v23 offset:24576 // (op 7023) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v23 offset:25600 // (op 7023) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30404) 


v_add_u32 v115, 67072, v22 // (op 7027) 
ds_read_u8 v54, v115 // (op 7027) Load local data 




ds_read_b64_tr_b4 v[130:131], v23 offset:24592 // (op 8428) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v23 offset:25616 // (op 8428) Transpose load local data 


v_add_u32 v115, 67104, v22 // (op 8431) 
ds_read_u8 v57, v115 // (op 8431) Load local data 




ds_read_b64_tr_b4 v[134:135], v23 offset:24608 // (op 9718) Transpose load local data 
ds_read_b64_tr_b4 v[136:137], v23 offset:25632 // (op 9718) Transpose load local data 


v_add_u32 v115, 67136, v22 // (op 9721) 
ds_read_u8 v44, v115 // (op 9721) Load local data 




ds_read_b64_tr_b4 v[138:139], v23 offset:24624 // (op 11008) Transpose load local data 
ds_read_b64_tr_b4 v[140:141], v23 offset:25648 // (op 11008) Transpose load local data 


v_add_u32 v115, 67168, v22 // (op 11011) 
ds_read_u8 v45, v115 // (op 11011) Load local data 

s_barrier  // (op 30951) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[122:125], a[0:3], v54, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7030) 
s_barrier  // (op 30401) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[126:129], a[4:7], v54, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7301) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27467) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[92:95], a[8:11], v54, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7599) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27467) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[96:99], a[12:15], v54, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7897) 
v_lshrrev_b64 v[142:143], 3, s[90:91] // (op 27467) 


s_lshl_b64 s[88:89], s[34:35], 7 // (op 27467) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[130:133], v[122:125], a[16:19], v57, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8700) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27467) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:133], v[126:129], a[20:23], v57, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8608) 
s_lshr_b64 s[88:89], s[90:91], 3 // (op 27467) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[130:133], v[92:95], a[24:27], v57, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8516) 
s_mul_i32 s91, 5, s89 // (op 27467) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[130:133], v[96:99], a[28:31], v57, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8424) 
s_mul_i32 s90, 5, s88 // (op 27467) least significant

s_mul_hi_u32 s92, 5, s88 // (op 27467) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27467) most significant: sum


v_mov_b32 v115, s90 // (op 27467) 
v_mov_b32 v130, s91 // (op 27467) 
v_add_co_u32 v132, s[88:89], v118, v115 // (op 27467) least significant half
v_addc_co_u32 v133, s[88:89], v119, v130, s[88:89] // (op 27467) most significant half




s_add_u32 s88, 16384, s55 // (op 27467) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:137], v[122:125], a[32:35], v44, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9990) 
s_mov_b32 m0, s88 // (op 27467) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:137], v[126:129], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9898) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 27467) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:137], v[92:95], a[40:43], v44, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9806) 
v_add_co_u32 v132, s[90:91], v132, v142 // (op 27467) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:137], v[96:99], a[44:47], v44, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9714) 
v_addc_co_u32 v133, s[90:91], 0, v133, s[90:91] // (op 27467) most significant half


s_add_u32 m0, m0, 4096 // (op 27467) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 27467) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[138:141], v[122:125], a[48:51], v45, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11280) 



v_mov_b64 v[122:123], s[44:45] // (op 7036) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[138:141], v[126:129], a[52:55], v45, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11188) 
s_lshl_b32 s88, s44, 2 // (op 7036) 

s_mul_i32 s89, 5, s88 // (op 7036) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[138:141], v[92:95], a[56:59], v45, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11096) 
v_add_u32 v115, v42, s89 // (op 7036) 

buffer_load_ubyte v64, v115, s[80:83], 0 offen offset: 0 // (op 7036) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:141], v[96:99], a[60:63], v45, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11004) 
v_add_u32 v115, v115, v122 // (op 7036) 


buffer_load_ubyte v65, v115, s[80:83], 0 offen offset: 0 // (op 7036) Load value


s_barrier  // (op 30391) 

s_lshl_b64 s[88:89], s[30:31], 6 // (op 27462) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27462) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27462) 


s_lshl_b64 s[88:89], s[30:31], 7 // (op 27462) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27462) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27462) 

s_mul_i32 s91, 5, s89 // (op 27462) most significant: low of low * high
s_mul_i32 s90, 5, s88 // (op 27462) least significant
s_mul_hi_u32 s92, 5, s88 // (op 27462) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27462) most significant: sum


v_mov_b32 v115, s90 // (op 27462) 
v_mov_b32 v122, s91 // (op 27462) 
v_add_co_u32 v124, s[88:89], v120, v115 // (op 27462) least significant half
v_addc_co_u32 v125, s[88:89], v121, v122, s[88:89] // (op 27462) most significant half




s_add_u32 s88, 24576, s54 // (op 27462) 
s_nop 0
s_mov_b32 m0, s88 // (op 27462) reg expression
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27462) Load value direct to lds
v_add_co_u32 v124, s[90:91], v124, v116 // (op 27462) least significant half
v_addc_co_u32 v125, s[90:91], 0, v125, s[90:91] // (op 27462) most significant half

s_add_u32 m0, m0, 4096 // (op 27462) 
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27462) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 7035) convert

s_lshl_b32 s88, s38, 2 // (op 7035) 
s_mul_i32 s89, 5, s88 // (op 7035) 

v_add_u32 v115, v106, s89 // (op 7035) 

buffer_load_ubyte v66, v115, s[76:79], 0 offen offset: 0 // (op 7035) Load value
v_add_u32 v115, v115, v116 // (op 7035) 
buffer_load_ubyte v67, v115, s[76:79], 0 offen offset: 0 // (op 7035) Load value


s_barrier  // (op 30446) 



v_add_u32 v115, 66560, v112 // (op 7073) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v68 // (op 7073) Store local data 

v_add_u32 v115, 66688, v112 // (op 7073) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v69 // (op 7073) Store local data 

s_barrier  // (op 30440) 



v_add_u32 v115, 67072, v111 // (op 7072) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v70 // (op 7072) Store local data 

v_add_u32 v115, 67200, v111 // (op 7072) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v71 // (op 7072) Store local data 

s_barrier  // (op 30426) 



ds_read_b64_tr_b4 v[72:73], v28 offset:32768 // (op 7065) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v28 offset:33792 // (op 7065) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30443) 


v_add_u32 v115, 66560, v29 // (op 7069) 
ds_read_u8 v52, v115 // (op 7069) Load local data 




ds_read_b64_tr_b4 v[76:77], v28 offset:32784 // (op 7315) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v28 offset:33808 // (op 7315) Transpose load local data 


v_add_u32 v115, 66592, v29 // (op 7319) 
ds_read_u8 v57, v115 // (op 7319) Load local data 




ds_read_b64_tr_b4 v[80:81], v28 offset:32800 // (op 7613) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v28 offset:33824 // (op 7613) Transpose load local data 


v_add_u32 v115, 66624, v29 // (op 7617) 
ds_read_u8 v44, v115 // (op 7617) Load local data 




ds_read_b64_tr_b4 v[84:85], v28 offset:32816 // (op 7911) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v28 offset:33840 // (op 7911) Transpose load local data 


v_add_u32 v115, 66656, v29 // (op 7915) 
ds_read_u8 v101, v115 // (op 7915) Load local data 

s_barrier  // (op 30417) 
s_barrier  // (op 30899) 



ds_read_b64_tr_b4 v[88:89], v27 offset:40960 // (op 7063) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v27 offset:41984 // (op 7063) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30436) 


v_add_u32 v115, 67072, v26 // (op 7067) 
ds_read_u8 v54, v115 // (op 7067) Load local data 




ds_read_b64_tr_b4 v[102:103], v27 offset:40976 // (op 8434) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v27 offset:42000 // (op 8434) Transpose load local data 


v_add_u32 v115, 67104, v26 // (op 8437) 
ds_read_u8 v55, v115 // (op 8437) Load local data 




ds_read_b64_tr_b4 v[122:123], v27 offset:40992 // (op 9724) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v27 offset:42016 // (op 9724) Transpose load local data 


v_add_u32 v115, 67136, v26 // (op 9727) 
ds_read_u8 v100, v115 // (op 9727) Load local data 




ds_read_b64_tr_b4 v[126:127], v27 offset:41008 // (op 11014) Transpose load local data 
ds_read_b64_tr_b4 v[128:129], v27 offset:42032 // (op 11014) Transpose load local data 


v_add_u32 v115, 67168, v26 // (op 11017) 
ds_read_u8 v116, v115 // (op 11017) Load local data 

s_barrier  // (op 30957) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[72:75], a[0:3], v54, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7070) 
s_barrier  // (op 30433) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[76:79], a[4:7], v54, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7302) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27477) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[80:83], a[8:11], v54, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7600) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27477) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[84:87], a[12:15], v54, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7898) 
v_lshrrev_b64 v[130:131], 3, s[90:91] // (op 27477) 


s_lshl_b64 s[88:89], s[34:35], 7 // (op 27477) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[102:105], v[72:75], a[16:19], v55, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8701) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27477) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:105], v[76:79], a[20:23], v55, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8609) 
s_lshr_b64 s[88:89], s[90:91], 3 // (op 27477) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:105], v[80:83], a[24:27], v55, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8517) 
s_mul_i32 s91, 6, s89 // (op 27477) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:105], v[84:87], a[28:31], v55, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8425) 
s_mul_i32 s90, 6, s88 // (op 27477) least significant
s_mul_hi_u32 s92, 6, s88 // (op 27477) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27477) most significant: sum


v_mov_b32 v115, s90 // (op 27477) 
v_mov_b32 v117, s91 // (op 27477) 
v_add_co_u32 v132, s[88:89], v118, v115 // (op 27477) least significant half
v_addc_co_u32 v133, s[88:89], v119, v117, s[88:89] // (op 27477) most significant half




s_add_u32 s88, 32768, s85 // (op 27477) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:125], v[72:75], a[32:35], v100, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9991) 
s_mov_b32 m0, s88 // (op 27477) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[122:125], v[76:79], a[36:39], v100, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9899) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 27477) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[122:125], v[80:83], a[40:43], v100, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9807) 
v_add_co_u32 v132, s[90:91], v132, v130 // (op 27477) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[122:125], v[84:87], a[44:47], v100, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9715) 
v_addc_co_u32 v133, s[90:91], 0, v133, s[90:91] // (op 27477) most significant half


s_add_u32 m0, m0, 4096 // (op 27477) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 27477) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[126:129], v[72:75], a[48:51], v116, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11281) 

v_mov_b64 v[122:123], s[44:45] // (op 7076) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:129], v[76:79], a[52:55], v116, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11189) 
s_lshl_b32 s88, s44, 2 // (op 7076) 
s_mul_i32 s89, 6, s88 // (op 7076) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:129], v[80:83], a[56:59], v116, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11097) 
v_add_u32 v115, v42, s89 // (op 7076) 

buffer_load_ubyte v68, v115, s[80:83], 0 offen offset: 0 // (op 7076) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:129], v[84:87], a[60:63], v116, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11005) 
v_add_u32 v115, v115, v122 // (op 7076) 


buffer_load_ubyte v69, v115, s[80:83], 0 offen offset: 0 // (op 7076) Load value


s_barrier  // (op 30423) 

s_lshl_b64 s[88:89], s[30:31], 6 // (op 27472) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27472) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27472) 


s_lshl_b64 s[88:89], s[30:31], 7 // (op 27472) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27472) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27472) 

s_mul_i32 s91, 6, s89 // (op 27472) most significant: low of low * high
s_mul_i32 s90, 6, s88 // (op 27472) least significant
s_mul_hi_u32 s92, 6, s88 // (op 27472) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27472) most significant: sum


v_mov_b32 v115, s90 // (op 27472) 
v_mov_b32 v122, s91 // (op 27472) 
v_add_co_u32 v124, s[88:89], v120, v115 // (op 27472) least significant half
v_addc_co_u32 v125, s[88:89], v121, v122, s[88:89] // (op 27472) most significant half




s_add_u32 s88, 40960, s84 // (op 27472) 
s_nop 0
s_mov_b32 m0, s88 // (op 27472) reg expression
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27472) Load value direct to lds
v_add_co_u32 v124, s[90:91], v124, v116 // (op 27472) least significant half
v_addc_co_u32 v125, s[90:91], 0, v125, s[90:91] // (op 27472) most significant half

s_add_u32 m0, m0, 4096 // (op 27472) 
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27472) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 7075) convert

s_lshl_b32 s88, s38, 2 // (op 7075) 
s_mul_i32 s89, 6, s88 // (op 7075) 

v_add_u32 v115, v106, s89 // (op 7075) 

buffer_load_ubyte v70, v115, s[76:79], 0 offen offset: 0 // (op 7075) Load value
v_add_u32 v115, v115, v116 // (op 7075) 
buffer_load_ubyte v71, v115, s[76:79], 0 offen offset: 0 // (op 7075) Load value


s_barrier  // (op 30478) 



v_add_u32 v115, 66560, v114 // (op 7113) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v58 // (op 7113) Store local data 

v_add_u32 v115, 66688, v114 // (op 7113) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v59 // (op 7113) Store local data 

s_barrier  // (op 30472) 



v_add_u32 v115, 67072, v113 // (op 7112) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v48 // (op 7112) Store local data 

v_add_u32 v115, 67200, v113 // (op 7112) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v49 // (op 7112) Store local data 

s_barrier  // (op 30458) 



ds_read_b64_tr_b4 v[92:93], v32 offset:49152 // (op 7105) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v32 offset:50176 // (op 7105) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30475) 


v_add_u32 v115, 66560, v33 // (op 7109) 
ds_read_u8 v45, v115 // (op 7109) Load local data 




ds_read_b64_tr_b4 v[96:97], v32 offset:49168 // (op 7323) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v32 offset:50192 // (op 7323) Transpose load local data 


v_add_u32 v115, 66592, v33 // (op 7327) 
ds_read_u8 v55, v115 // (op 7327) Load local data 




ds_read_b64_tr_b4 v[102:103], v32 offset:49184 // (op 7621) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v32 offset:50208 // (op 7621) Transpose load local data 


v_add_u32 v115, 66624, v33 // (op 7625) 
ds_read_u8 v100, v115 // (op 7625) Load local data 




ds_read_b64_tr_b4 v[122:123], v32 offset:49200 // (op 7919) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v32 offset:50224 // (op 7919) Transpose load local data 


v_add_u32 v115, 66656, v33 // (op 7923) 
ds_read_u8 v116, v115 // (op 7923) Load local data 

s_barrier  // (op 30449) 
s_barrier  // (op 30905) 



ds_read_b64_tr_b4 v[88:89], v31 offset:57344 // (op 7103) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v31 offset:58368 // (op 7103) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30468) 


v_add_u32 v115, 67072, v30 // (op 7107) 
ds_read_u8 v54, v115 // (op 7107) Load local data 




ds_read_b64_tr_b4 v[126:127], v31 offset:57360 // (op 8440) Transpose load local data 
ds_read_b64_tr_b4 v[128:129], v31 offset:58384 // (op 8440) Transpose load local data 


v_add_u32 v115, 67104, v30 // (op 8443) 
ds_read_u8 v57, v115 // (op 8443) Load local data 




ds_read_b64_tr_b4 v[130:131], v31 offset:57376 // (op 9730) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v31 offset:58400 // (op 9730) Transpose load local data 


v_add_u32 v115, 67136, v30 // (op 9733) 
ds_read_u8 v44, v115 // (op 9733) Load local data 




ds_read_b64_tr_b4 v[134:135], v31 offset:57392 // (op 11020) Transpose load local data 
ds_read_b64_tr_b4 v[136:137], v31 offset:58416 // (op 11020) Transpose load local data 


v_add_u32 v115, 67168, v30 // (op 11023) 
ds_read_u8 v101, v115 // (op 11023) Load local data 

s_barrier  // (op 30963) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[92:95], a[0:3], v54, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7110) 
s_barrier  // (op 30465) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[96:99], a[4:7], v54, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7303) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27487) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[102:105], a[8:11], v54, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7601) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27487) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[122:125], a[12:15], v54, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7899) 
v_lshrrev_b64 v[138:139], 3, s[90:91] // (op 27487) 


s_lshl_b64 s[88:89], s[34:35], 7 // (op 27487) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[126:129], v[92:95], a[16:19], v57, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8702) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27487) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:129], v[96:99], a[20:23], v57, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8610) 
s_lshr_b64 s[88:89], s[90:91], 3 // (op 27487) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:129], v[102:105], a[24:27], v57, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8518) 
s_mul_i32 s91, 7, s89 // (op 27487) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[126:129], v[122:125], a[28:31], v57, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8426) 
s_mul_i32 s90, 7, s88 // (op 27487) least significant

s_mul_hi_u32 s92, 7, s88 // (op 27487) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27487) most significant: sum


v_mov_b32 v115, s90 // (op 27487) 
v_mov_b32 v117, s91 // (op 27487) 
v_add_co_u32 v126, s[88:89], v118, v115 // (op 27487) least significant half
v_addc_co_u32 v127, s[88:89], v119, v117, s[88:89] // (op 27487) most significant half




s_add_u32 s88, 49152, s87 // (op 27487) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:133], v[92:95], a[32:35], v44, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9992) 
s_mov_b32 m0, s88 // (op 27487) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:133], v[96:99], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9900) 
buffer_load_dwordx4 v126, s[72:75], 0 offen offset: 0 lds // (op 27487) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:133], v[102:105], a[40:43], v44, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9808) 
v_add_co_u32 v126, s[90:91], v126, v138 // (op 27487) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:133], v[122:125], a[44:47], v44, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9716) 
v_addc_co_u32 v127, s[90:91], 0, v127, s[90:91] // (op 27487) most significant half


s_add_u32 m0, m0, 4096 // (op 27487) 
buffer_load_dwordx4 v126, s[72:75], 0 offen offset: 0 lds // (op 27487) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:137], v[92:95], a[48:51], v101, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11282) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:137], v[96:99], a[52:55], v101, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11190) 
v_mov_b64 v[118:119], s[44:45] // (op 7116) convert

s_lshl_b32 s88, s44, 2 // (op 7116) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:137], v[102:105], a[56:59], v101, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11098) 
s_mul_i32 s89, 7, s88 // (op 7116) 

v_add_u32 v115, v42, s89 // (op 7116) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:137], v[122:125], a[60:63], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11006) 
buffer_load_ubyte v58, v115, s[80:83], 0 offen offset: 0 // (op 7116) Load value



v_add_u32 v115, v115, v118 // (op 7116) 
s_lshl_b64 s[88:89], s[34:35], 7 // (op 27811) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27811) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27811) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27811) 

v_mov_b32 v116, s90 // (op 27811) 
v_mov_b32 v117, s91 // (op 27811) 
v_add_co_u32 v50, s[88:89], v50, v116 // (op 27811) least significant half
v_addc_co_u32 v51, s[88:89], v51, v117, s[88:89] // (op 27811) most significant half




buffer_load_ubyte v59, v115, s[80:83], 0 offen offset: 0 // (op 7116) Load value



s_barrier  // (op 30455) 
s_lshl_b32 s88, s44, 4 // (op 28067) 

v_add_u32 v56, v56, s88 // (op 28067) 

s_lshl_b64 s[88:89], s[30:31], 6 // (op 27482) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27482) 

v_lshrrev_b64 v[118:119], 3, s[90:91] // (op 27482) 


s_lshl_b64 s[88:89], s[30:31], 7 // (op 27482) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27482) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27482) 

s_mul_i32 s91, 7, s89 // (op 27482) most significant: low of low * high
s_mul_i32 s90, 7, s88 // (op 27482) least significant
s_mul_hi_u32 s92, 7, s88 // (op 27482) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27482) most significant: sum


v_mov_b32 v42, s90 // (op 27482) 
v_mov_b32 v115, s91 // (op 27482) 
v_add_co_u32 v116, s[88:89], v120, v42 // (op 27482) least significant half
v_addc_co_u32 v117, s[88:89], v121, v115, s[88:89] // (op 27482) most significant half




s_add_u32 s88, 57344, s86 // (op 27482) 
s_nop 0
s_mov_b32 m0, s88 // (op 27482) reg expression
buffer_load_dwordx4 v116, s[68:71], 0 offen offset: 0 lds // (op 27482) Load value direct to lds
v_add_co_u32 v116, s[90:91], v116, v118 // (op 27482) least significant half
v_addc_co_u32 v117, s[90:91], 0, v117, s[90:91] // (op 27482) most significant half

s_add_u32 m0, m0, 4096 // (op 27482) 
buffer_load_dwordx4 v116, s[68:71], 0 offen offset: 0 lds // (op 27482) Load value direct to lds





v_mov_b64 v[116:117], s[38:39] // (op 7115) convert

s_lshl_b32 s88, s38, 2 // (op 7115) 
s_mul_i32 s89, 7, s88 // (op 7115) 

v_add_u32 v42, v106, s89 // (op 7115) 

s_lshl_b64 s[88:89], s[30:31], 7 // (op 27683) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27683) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27683) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27683) 

v_mov_b32 v115, s90 // (op 27683) 
v_mov_b32 v118, s91 // (op 27683) 
v_add_co_u32 v46, s[88:89], v46, v115 // (op 27683) least significant half
v_addc_co_u32 v47, s[88:89], v47, v118, s[88:89] // (op 27683) most significant half




buffer_load_ubyte v48, v42, s[76:79], 0 offen offset: 0 // (op 7115) Load value
v_add_u32 v42, v42, v116 // (op 7115) 
buffer_load_ubyte v49, v42, s[76:79], 0 offen offset: 0 // (op 7115) Load value



s_barrier  // (op 30195) 
s_lshl_b32 s88, s38, 4 // (op 27939) 



v_add_u32 v53, v53, s88 // (op 27939) 

v_add_u32 v42, 65536, v108 // (op 30) 
s_waitcnt vmcnt(29)
ds_write_b8 v42, v60 // (op 30) Store local data 

v_add_u32 v42, 65664, v108 // (op 30) 
s_waitcnt vmcnt(28)
ds_write_b8 v42, v61 // (op 30) Store local data 

s_barrier  // (op 30174) 



v_add_u32 v42, 66048, v107 // (op 21) 
s_waitcnt vmcnt(25)
ds_write_b8 v42, v62 // (op 21) Store local data 

v_add_u32 v42, 66176, v107 // (op 21) 
s_waitcnt vmcnt(24)
ds_write_b8 v42, v63 // (op 21) Store local data 

s_barrier  // (op 30160) 



ds_read_b64_tr_b4 v[72:73], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:1024 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30202) 


v_add_u32 v42, 65536, v5 // (op 87) 
ds_read_u8 v52, v42 // (op 87) Load local data 




ds_read_b64_tr_b4 v[76:77], v4 offset:16 // (op 7295) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v4 offset:1040 // (op 7295) Transpose load local data 


v_add_u32 v42, 65568, v5 // (op 7299) 
ds_read_u8 v57, v42 // (op 7299) Load local data 




ds_read_b64_tr_b4 v[80:81], v4 offset:32 // (op 7593) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v4 offset:1056 // (op 7593) Transpose load local data 


v_add_u32 v42, 65600, v5 // (op 7597) 
ds_read_u8 v44, v42 // (op 7597) Load local data 




ds_read_b64_tr_b4 v[84:85], v4 offset:48 // (op 7891) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v4 offset:1072 // (op 7891) Transpose load local data 


v_add_u32 v42, 65632, v5 // (op 7895) 
ds_read_u8 v45, v42 // (op 7895) Load local data 

s_barrier  // (op 30141) 



ds_read_b64_tr_b4 v[88:89], v3 offset:8192 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:9216 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30181) 


v_add_u32 v42, 66048, v0 // (op 82) 
ds_read_u8 v54, v42 // (op 82) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:8208 // (op 8418) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:9232 // (op 8418) Transpose load local data 


v_add_u32 v42, 66080, v0 // (op 8421) 
ds_read_u8 v55, v42 // (op 8421) Load local data 




ds_read_b64_tr_b4 v[96:97], v3 offset:8224 // (op 9708) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v3 offset:9248 // (op 9708) Transpose load local data 


v_add_u32 v42, 66112, v0 // (op 9711) 
ds_read_u8 v100, v42 // (op 9711) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:8240 // (op 10998) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:9264 // (op 10998) Transpose load local data 


v_add_u32 v42, 66144, v0 // (op 11001) 
ds_read_u8 v101, v42 // (op 11001) Load local data 

s_and_b32 s88, s7, 4294967292 // (op 9581) 
s_cmp_lt_u32 s1, s88 // (op 9581) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9581 // (op 9581) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9581 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9581:




























s_barrier  // (op 30917) 
s_and_b32 s52, s7, 4294967292 // (op 27571) 
v_and_b32 v42, -4, v1 // (op 27571) 
s_add_u32 s53, s0, s52 // (op 27571) 

v_sub_u32 v44, v1, v42 // (op 27571) 

v_lshlrev_b32 v42, 5, v44 // (op 27571) 

s_mul_i32 s52, s16, s8 // (op 27571) 
s_add_u32 s52, s52, s53 // (op 27571) 

v_lshrrev_b32 v44, 2, v1 // (op 27571) 
s_mul_hi_u32 s53, s52, s13 // (op 27571) 
s_sub_u32 s54, s52, s53 // (op 27571) 
s_lshr_b32 s55, s54, 1 // (op 27571) 

s_add_u32 s54, s55, s53 // (op 27571) 


s_lshr_b32 s53, s54, s11 // (op 27571) 

s_and_b32 s54, s11, 2147483648 // (op 27571) 
s_cmp_eq_i32 s54, 0 // (op 27571) 
s_mov_b32 s55, scc // (op 27571) 

s_cmp_eq_i32 s55, 0 // (op 27571) 
s_cselect_b32 s54, s52, s53 // (op 27571) 


s_mul_hi_u32 s53, s54, s12 // (op 27571) 
s_sub_u32 s55, s54, s53 // (op 27571) 
s_lshr_b32 s84, s55, 1 // (op 27571) 

s_add_u32 s55, s84, s53 // (op 27571) 


s_lshr_b32 s53, s55, s10 // (op 27571) 

s_and_b32 s55, s10, 2147483648 // (op 27571) 
s_cmp_eq_i32 s55, 0 // (op 27571) 
s_mov_b32 s84, scc // (op 27571) 

s_cmp_eq_i32 s84, 0 // (op 27571) 
s_cselect_b32 s55, s54, s53 // (op 27571) 


v_lshl_add_u32 v46, s55, 7, v42 // (op 27571) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27571) 
v_mov_b32 v57, 0 // (op 27571) convert
v_mov_b32 v56, v46 // (op 27571) convert

s_mul_i32 s53, s54, s84 // (op 27571) 


s_sub_u32 s54, s52, s53 // (op 27571) 


v_lshl_add_u32 v42, s54, 7, v44 // (op 27571) 


v_mul_lo_u32 v47, v42, s31 // (op 27571) most significant: low of low * high
v_mul_hi_u32 v44, v42, s30 // (op 27571) most significant: high of low * low
v_mul_lo_u32 v46, v42, s30 // (op 27571) least significant: low of low * low
v_add_u32 v47, v47, v44 // (op 27571) most significant: sum


v_add_co_u32 v50, s[52:53], v56, v46 // (op 27571) least significant half
v_addc_co_u32 v51, s[52:53], v57, v47, s[52:53] // (op 27571) most significant half

v_lshlrev_b64 v[50:51], 2, v[50:51] // (op 27571) 


v_lshrrev_b64 v[46:47], 3, v[50:51] // (op 27571) 

s_mul_i32 s53, s30, s3 // (op 27573) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27573) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27573) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27573) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27573) most significant: sum
s_add_u32 s53, s53, s55 // (op 27573) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27573) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27573) 

s_mov_b32 s68, s32 // (op 27573) reg expression
s_mov_b32 s69, s33 // (op 27573) reg expression
s_mov_b32 s70, s52 // (op 27573) reg expression
s_mov_b32 s71, 131072 // (op 27573) call()

s_and_b32 s52, s7, 4294967292 // (op 27735) 
v_and_b32 v42, -4, v1 // (op 27735) 
s_add_u32 s53, s0, s52 // (op 27735) 

v_sub_u32 v44, v1, v42 // (op 27735) 

v_lshlrev_b32 v42, 5, v44 // (op 27735) 

s_mul_i32 s52, s16, s8 // (op 27735) 
s_add_u32 s52, s52, s53 // (op 27735) 

v_lshrrev_b32 v44, 2, v1 // (op 27735) 
s_mul_hi_u32 s53, s52, s13 // (op 27735) 
s_sub_u32 s54, s52, s53 // (op 27735) 
s_lshr_b32 s55, s54, 1 // (op 27735) 

s_add_u32 s54, s55, s53 // (op 27735) 


s_lshr_b32 s53, s54, s11 // (op 27735) 

s_and_b32 s54, s11, 2147483648 // (op 27735) 
s_cmp_eq_i32 s54, 0 // (op 27735) 
s_mov_b32 s55, scc // (op 27735) 

s_cmp_eq_i32 s55, 0 // (op 27735) 
s_cselect_b32 s54, s52, s53 // (op 27735) 


s_mul_hi_u32 s53, s54, s12 // (op 27735) 
s_sub_u32 s55, s54, s53 // (op 27735) 
s_lshr_b32 s84, s55, 1 // (op 27735) 

s_add_u32 s55, s84, s53 // (op 27735) 


s_lshr_b32 s53, s55, s10 // (op 27735) 

s_and_b32 s55, s10, 2147483648 // (op 27735) 
s_cmp_eq_i32 s55, 0 // (op 27735) 
s_mov_b32 s84, scc // (op 27735) 

s_cmp_eq_i32 s84, 0 // (op 27735) 
s_cselect_b32 s55, s54, s53 // (op 27735) 


s_mul_i32 s53, s55, s14 // (op 27735) 

s_sub_u32 s55, s54, s53 // (op 27735) 

v_lshl_add_u32 v50, s55, 7, v42 // (op 27735) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27735) 
v_mov_b32 v53, 0 // (op 27735) convert
v_mov_b32 v52, v50 // (op 27735) convert

s_mul_i32 s53, s54, s84 // (op 27735) 


s_sub_u32 s54, s52, s53 // (op 27735) 


v_lshl_add_u32 v42, s54, 7, v44 // (op 27735) 


v_mul_lo_u32 v51, v42, s35 // (op 27735) most significant: low of low * high
v_mul_hi_u32 v44, v42, s34 // (op 27735) most significant: high of low * low
v_mul_lo_u32 v50, v42, s34 // (op 27735) least significant: low of low * low
v_add_u32 v51, v51, v44 // (op 27735) most significant: sum

v_add_co_u32 v50, s[52:53], v50, v52 // (op 27735) least significant half
v_addc_co_u32 v51, s[52:53], v51, v53, s[52:53] // (op 27735) most significant half



v_lshlrev_b64 v[52:53], 2, v[50:51] // (op 27735) 

v_lshrrev_b64 v[50:51], 3, v[52:53] // (op 27735) 

s_mul_i32 s53, s34, s3 // (op 27737) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 27737) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 27737) least significant
s_mul_hi_u32 s55, s34, s2 // (op 27737) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27737) most significant: sum
s_add_u32 s53, s53, s55 // (op 27737) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27737) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27737) 

s_mov_b32 s72, s36 // (op 27737) reg expression
s_mov_b32 s73, s37 // (op 27737) reg expression
s_mov_b32 s74, s52 // (op 27737) reg expression
s_mov_b32 s75, 131072 // (op 27737) call()

s_and_b32 s52, s7, 4294967292 // (op 27863) 
v_and_b32 v42, -128, v1 // (op 27863) 
s_add_u32 s53, s0, s52 // (op 27863) 

v_sub_u32 v44, v1, v42 // (op 27863) 

v_lshrrev_b32 v42, 7, v1 // (op 27863) 
s_mul_i32 s52, s16, s8 // (op 27863) 
s_add_u32 s52, s52, s53 // (op 27863) 

v_lshlrev_b32 v52, 1, v42 // (op 27863) 

s_mul_hi_u32 s53, s52, s13 // (op 27863) 
s_sub_u32 s54, s52, s53 // (op 27863) 
s_lshr_b32 s55, s54, 1 // (op 27863) 

s_add_u32 s54, s55, s53 // (op 27863) 


s_lshr_b32 s53, s54, s11 // (op 27863) 

s_and_b32 s54, s11, 2147483648 // (op 27863) 
s_cmp_eq_i32 s54, 0 // (op 27863) 
s_mov_b32 s55, scc // (op 27863) 

s_cmp_eq_i32 s55, 0 // (op 27863) 
s_cselect_b32 s54, s52, s53 // (op 27863) 


s_mul_hi_u32 s53, s54, s12 // (op 27863) 
s_sub_u32 s55, s54, s53 // (op 27863) 
s_lshr_b32 s84, s55, 1 // (op 27863) 

s_add_u32 s55, s84, s53 // (op 27863) 


s_lshr_b32 s53, s55, s10 // (op 27863) 

s_and_b32 s55, s10, 2147483648 // (op 27863) 
s_cmp_eq_i32 s55, 0 // (op 27863) 
s_mov_b32 s84, scc // (op 27863) 

s_cmp_eq_i32 s84, 0 // (op 27863) 
s_cselect_b32 s55, s54, s53 // (op 27863) 


v_lshl_add_u32 v42, s55, 7, v44 // (op 27863) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27863) 
s_mul_i32 s53, s54, s84 // (op 27863) 


s_sub_u32 s54, s52, s53 // (op 27863) 


v_lshl_add_u32 v44, s54, 2, v52 // (op 27863) 


v_mul_lo_u32 v52, v44, s38 // (op 27863) 

v_add_u32 v44, v42, v52 // (op 27863) 


s_mul_i32 s53, s38, s43 // (op 27865) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27865) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27865) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27865) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27865) most significant: sum
s_add_u32 s53, s53, s55 // (op 27865) most significant: sum


s_mov_b32 s76, s40 // (op 27865) reg expression
s_mov_b32 s77, s41 // (op 27865) reg expression
s_mov_b32 s78, s52 // (op 27865) reg expression
s_mov_b32 s79, 131072 // (op 27865) call()

s_and_b32 s52, s7, 4294967292 // (op 27991) 
v_and_b32 v42, -128, v1 // (op 27991) 
s_add_u32 s53, s0, s52 // (op 27991) 

v_sub_u32 v52, v1, v42 // (op 27991) 

v_lshrrev_b32 v42, 7, v1 // (op 27991) 
s_mul_i32 s52, s16, s8 // (op 27991) 
s_add_u32 s52, s52, s53 // (op 27991) 

v_lshlrev_b32 v53, 1, v42 // (op 27991) 

s_mul_hi_u32 s53, s52, s13 // (op 27991) 
s_sub_u32 s54, s52, s53 // (op 27991) 
s_lshr_b32 s55, s54, 1 // (op 27991) 

s_add_u32 s54, s55, s53 // (op 27991) 


s_lshr_b32 s53, s54, s11 // (op 27991) 

s_and_b32 s54, s11, 2147483648 // (op 27991) 
s_cmp_eq_i32 s54, 0 // (op 27991) 
s_mov_b32 s55, scc // (op 27991) 

s_cmp_eq_i32 s55, 0 // (op 27991) 
s_cselect_b32 s54, s52, s53 // (op 27991) 


s_mul_hi_u32 s53, s54, s12 // (op 27991) 
s_sub_u32 s55, s54, s53 // (op 27991) 
s_lshr_b32 s84, s55, 1 // (op 27991) 

s_add_u32 s55, s84, s53 // (op 27991) 


s_lshr_b32 s53, s55, s10 // (op 27991) 

s_and_b32 s55, s10, 2147483648 // (op 27991) 
s_cmp_eq_i32 s55, 0 // (op 27991) 
s_mov_b32 s84, scc // (op 27991) 

s_cmp_eq_i32 s84, 0 // (op 27991) 
s_cselect_b32 s55, s54, s53 // (op 27991) 


s_mul_i32 s53, s55, s14 // (op 27991) 

s_sub_u32 s55, s54, s53 // (op 27991) 

v_lshl_add_u32 v42, s55, 7, v52 // (op 27991) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27991) 
s_mul_i32 s53, s54, s84 // (op 27991) 


s_sub_u32 s54, s52, s53 // (op 27991) 


v_lshl_add_u32 v52, s54, 2, v53 // (op 27991) 


v_mul_lo_u32 v53, v52, s44 // (op 27991) 
v_add_u32 v53, v53, v42 // (op 27991) 


s_mul_i32 s53, s44, s43 // (op 27993) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 27993) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 27993) least significant
s_mul_hi_u32 s55, s44, s42 // (op 27993) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27993) most significant: sum
s_add_u32 s53, s53, s55 // (op 27993) most significant: sum


s_mov_b32 s80, s46 // (op 27993) reg expression
s_mov_b32 s81, s47 // (op 27993) reg expression
s_mov_b32 s82, s52 // (op 27993) reg expression
s_mov_b32 s83, 131072 // (op 27993) call()

v_and_b32 v42, -4, v1 // (op 28162) 
v_sub_u32 v52, v1, v42 // (op 28162) 

v_lshlrev_b32 v42, 5, v52 // (op 28162) 

v_lshrrev_b32 v52, 2, v1 // (op 28162) 
v_lshl_add_u32 v56, v52, 7, v42 // (op 28162) 


v_lshlrev_b32 v42, 2, v56 // (op 28162) 

v_lshrrev_b32 v52, 3, v42 // (op 28162) 

s_nop 0
v_readfirstlane_b32 s52, v52 // (op 28162) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 28237) 
v_sub_u32 v52, v1, v42 // (op 28237) 

v_lshlrev_b32 v42, 5, v52 // (op 28237) 

v_lshrrev_b32 v52, 2, v1 // (op 28237) 
v_lshl_add_u32 v56, v52, 7, v42 // (op 28237) 


v_lshlrev_b32 v42, 2, v56 // (op 28237) 

v_lshrrev_b32 v52, 3, v42 // (op 28237) 

s_nop 0
v_readfirstlane_b32 s53, v52 // (op 28237) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 28312) 
v_sub_u32 v52, v1, v42 // (op 28312) 

v_lshrrev_b32 v42, 7, v1 // (op 28312) 
v_lshl_add_u32 v56, v42, 8, v52 // (op 28312) 


v_and_b32 v42, -128, v1 // (op 28388) 
v_sub_u32 v52, v1, v42 // (op 28388) 

v_lshrrev_b32 v42, 7, v1 // (op 28388) 
v_lshl_add_u32 v57, v42, 8, v52 // (op 28388) 


s_and_b32 s54, s7, 4294967292 // (op 10970) 
s_cmp_lt_u32 s54, s7 // (op 10969) 
















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10969 // (op 10969) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10969 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10969:
s_barrier  // (op 30138) 
s_add_u32 s54, s54, 1 // (op 10971) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 27449) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27449) 

v_lshrrev_b64 v[48:49], 3, s[86:87] // (op 27449) 


v_mov_b64 v[54:55], v[46:47] // (op 27449) 
v_mov_b64 v[58:59], v[54:55] // (op 27449) reg expression
s_add_u32 s55, 8192, s52 // (op 27449) 
s_mov_b32 m0, s55 // (op 27449) reg expression
buffer_load_dwordx4 v58, s[68:71], 0 offen offset: 0 lds // (op 27449) Load value direct to lds
v_add_co_u32 v58, s[84:85], v58, v48 // (op 27449) least significant half
v_addc_co_u32 v59, s[84:85], 0, v59, s[84:85] // (op 27449) most significant half

s_add_u32 m0, m0, 4096 // (op 27449) 
buffer_load_dwordx4 v58, s[68:71], 0 offen offset: 0 lds // (op 27449) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 27561) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27561) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 27561) 

v_mov_b32 v42, s84 // (op 27561) 
v_mov_b32 v45, s85 // (op 27561) 
v_add_co_u32 v46, s[86:87], v46, v42 // (op 27561) least significant half
v_addc_co_u32 v47, s[86:87], v47, v45, s[86:87] // (op 27561) most significant half




s_barrier  // (op 30157) 

s_lshl_b64 s[84:85], s[34:35], 6 // (op 27455) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27455) 

v_lshrrev_b64 v[48:49], 3, s[86:87] // (op 27455) 


v_mov_b64 v[54:55], v[50:51] // (op 27455) 
v_mov_b64 v[58:59], v[54:55] // (op 27455) reg expression
s_mov_b32 m0, s53 // (op 27455) reg expression
buffer_load_dwordx4 v58, s[72:75], 0 offen offset: 0 lds // (op 27455) Load value direct to lds
v_add_co_u32 v58, s[84:85], v58, v48 // (op 27455) least significant half
v_addc_co_u32 v59, s[84:85], 0, v59, s[84:85] // (op 27455) most significant half

s_add_u32 m0, m0, 4096 // (op 27455) 
buffer_load_dwordx4 v58, s[72:75], 0 offen offset: 0 lds // (op 27455) Load value direct to lds



s_lshl_b64 s[84:85], s[34:35], 7 // (op 27727) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27727) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 27727) 

v_mov_b32 v42, s84 // (op 27727) 
v_mov_b32 v45, s85 // (op 27727) 
v_add_co_u32 v50, s[86:87], v50, v42 // (op 27727) least significant half
v_addc_co_u32 v51, s[86:87], v51, v45, s[86:87] // (op 27727) most significant half





v_mov_b64 v[48:49], s[38:39] // (op 6969) convert

v_mov_b32 v42, v44 // (op 6969) 
v_mov_b32 v45, v42 // (op 6969) reg expression
buffer_load_ubyte v54, v45, s[76:79], 0 offen offset: 0 // (op 6969) Load value
v_add_u32 v45, v45, v48 // (op 6969) 
buffer_load_ubyte v55, v45, s[76:79], 0 offen offset: 0 // (op 6969) Load value



s_lshl_b32 s55, s38, 2 // (op 27855) 
v_add_u32 v44, v44, s55 // (op 27855) 


s_barrier  // (op 30177) 
v_mov_b64 v[48:49], s[44:45] // (op 6970) convert




v_mov_b32 v42, v53 // (op 6970) 
v_mov_b32 v45, v42 // (op 6970) reg expression
v_add_u32 v52, 66048, v56 // (op 6966) 
s_waitcnt vmcnt(1)
ds_write_b8 v52, v54 // (op 6966) Store local data 

v_add_u32 v52, 66176, v56 // (op 6966) 
buffer_load_ubyte v58, v45, s[80:83], 0 offen offset: 0 // (op 6970) Load value
v_add_u32 v45, v45, v48 // (op 6970) 
buffer_load_ubyte v59, v45, s[80:83], 0 offen offset: 0 // (op 6970) Load value



s_lshl_b32 s55, s44, 2 // (op 27983) 
v_add_u32 v53, v53, s55 // (op 27983) 

s_waitcnt vmcnt(2)
ds_write_b8 v52, v55 // (op 6966) Store local data 


s_barrier  // (op 30144) 



s_barrier  // (op 30198) 
ds_read_b64_tr_b4 v[60:61], v3 offset:8192 // (op 6957) Transpose load local data 



v_add_u32 v42, 65536, v57 // (op 6967) 
ds_read_b64_tr_b4 v[62:63], v3 offset:9216 // (op 6957) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v42, v58 // (op 6967) Store local data 

s_barrier  // (op 30164) 
v_add_u32 v42, 65664, v57 // (op 6967) 



ds_read_b64_tr_b4 v[64:65], v4 // (op 6959) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v4 offset:1024 // (op 6959) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v42, v59 // (op 6967) Store local data 


s_waitcnt lgkmcnt(6)
s_barrier  // (op 30185) 


v_add_u32 v42, 66048, v0 // (op 6961) 
ds_read_u8 v45, v42 // (op 6961) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 30205) 


v_add_u32 v42, 65536, v5 // (op 6963) 
ds_read_u8 v52, v42 // (op 6963) Load local data 




ds_read_b64_tr_b4 v[68:69], v4 offset:16 // (op 7274) Transpose load local data 
ds_read_b64_tr_b4 v[70:71], v4 offset:1040 // (op 7274) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:63], v[64:67], a[0:3], v45, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6964) 


v_add_u32 v42, 65568, v5 // (op 7278) 
ds_read_u8 v48, v42 // (op 7278) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:32 // (op 7572) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:1056 // (op 7572) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:63], v[68:71], a[4:7], v45, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7279) 


v_add_u32 v42, 65600, v5 // (op 7576) 
ds_read_u8 v49, v42 // (op 7576) Load local data 




ds_read_b64_tr_b4 v[76:77], v4 offset:48 // (op 7870) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v4 offset:1072 // (op 7870) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:63], v[72:75], a[8:11], v45, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7577) 


v_add_u32 v42, 65632, v5 // (op 7874) 
ds_read_u8 v54, v42 // (op 7874) Load local data 




ds_read_b64_tr_b4 v[80:81], v3 offset:8208 // (op 8302) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:9232 // (op 8302) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:63], v[76:79], a[12:15], v45, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7875) 




v_add_u32 v42, 66080, v0 // (op 8305) 
ds_read_u8 v45, v42 // (op 8305) Load local data 




ds_read_b64_tr_b4 v[58:59], v3 offset:8224 // (op 9592) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v3 offset:9248 // (op 9592) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[64:67], a[16:19], v45, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8307) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[68:71], a[20:23], v45, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8338) 
v_add_u32 v42, 66112, v0 // (op 9595) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[72:75], a[24:27], v45, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8369) 
ds_read_u8 v55, v42 // (op 9595) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[76:79], a[28:31], v45, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8400) 
ds_read_b64_tr_b4 v[84:85], v3 offset:8240 // (op 10882) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v3 offset:9264 // (op 10882) Transpose load local data 


v_add_u32 v42, 66144, v0 // (op 10885) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[64:67], a[32:35], v55, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9597) 
ds_read_u8 v45, v42 // (op 10885) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[68:71], a[36:39], v55, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9628) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[58:61], v[72:75], a[40:43], v55, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9659) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[58:61], v[76:79], a[44:47], v55, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9690) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[64:67], a[48:51], v45, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10887) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[68:71], a[52:55], v45, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10918) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[72:75], a[56:59], v45, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10949) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[76:79], a[60:63], v45, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10980) 




s_cmp_lt_u32 s54, s7 // (op 10969) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10969 // (op 10969) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10969 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10969:









s_barrier  // (op 30935) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_322



v_add_u32 v42, 49152, v41 // (op 12724) 
buffer_store_dword a48, v42, s[64:67], 0 offen sc1 // (op 12724) Store value

v_add_u32 v42, 49664, v41 // (op 12724) 
buffer_store_dword a49, v42, s[64:67], 0 offen sc1 // (op 12724) Store value

v_add_u32 v42, 50176, v41 // (op 12724) 
buffer_store_dword a50, v42, s[64:67], 0 offen sc1 // (op 12724) Store value

v_add_u32 v42, 50688, v41 // (op 12724) 
buffer_store_dword a51, v42, s[64:67], 0 offen sc1 // (op 12724) Store value




v_add_u32 v42, 49280, v41 // (op 12726) 
buffer_store_dword a52, v42, s[64:67], 0 offen sc1 // (op 12726) Store value

v_add_u32 v42, 49792, v41 // (op 12726) 
buffer_store_dword a53, v42, s[64:67], 0 offen sc1 // (op 12726) Store value

v_add_u32 v42, 50304, v41 // (op 12726) 
buffer_store_dword a54, v42, s[64:67], 0 offen sc1 // (op 12726) Store value

v_add_u32 v42, 50816, v41 // (op 12726) 
buffer_store_dword a55, v42, s[64:67], 0 offen sc1 // (op 12726) Store value




v_add_u32 v42, 49408, v41 // (op 12728) 
buffer_store_dword a56, v42, s[64:67], 0 offen sc1 // (op 12728) Store value

v_add_u32 v42, 49920, v41 // (op 12728) 
buffer_store_dword a57, v42, s[64:67], 0 offen sc1 // (op 12728) Store value

v_add_u32 v42, 50432, v41 // (op 12728) 
buffer_store_dword a58, v42, s[64:67], 0 offen sc1 // (op 12728) Store value

v_add_u32 v42, 50944, v41 // (op 12728) 
buffer_store_dword a59, v42, s[64:67], 0 offen sc1 // (op 12728) Store value




v_add_u32 v42, 49536, v41 // (op 12730) 
buffer_store_dword a60, v42, s[64:67], 0 offen sc1 // (op 12730) Store value

v_add_u32 v42, 50048, v41 // (op 12730) 
buffer_store_dword a61, v42, s[64:67], 0 offen sc1 // (op 12730) Store value

v_add_u32 v42, 50560, v41 // (op 12730) 
buffer_store_dword a62, v42, s[64:67], 0 offen sc1 // (op 12730) Store value

v_add_u32 v42, 51072, v41 // (op 12730) 
buffer_store_dword a63, v42, s[64:67], 0 offen sc1 // (op 12730) Store value




v_add_u32 v42, 32768, v41 // (op 12700) 
buffer_store_dword a32, v42, s[64:67], 0 offen sc1 // (op 12700) Store value

v_add_u32 v42, 33280, v41 // (op 12700) 
buffer_store_dword a33, v42, s[64:67], 0 offen sc1 // (op 12700) Store value

v_add_u32 v42, 33792, v41 // (op 12700) 
buffer_store_dword a34, v42, s[64:67], 0 offen sc1 // (op 12700) Store value

v_add_u32 v42, 34304, v41 // (op 12700) 
buffer_store_dword a35, v42, s[64:67], 0 offen sc1 // (op 12700) Store value




v_add_u32 v42, 32896, v41 // (op 12702) 
buffer_store_dword a36, v42, s[64:67], 0 offen sc1 // (op 12702) Store value

v_add_u32 v42, 33408, v41 // (op 12702) 
buffer_store_dword a37, v42, s[64:67], 0 offen sc1 // (op 12702) Store value

v_add_u32 v42, 33920, v41 // (op 12702) 
buffer_store_dword a38, v42, s[64:67], 0 offen sc1 // (op 12702) Store value

v_add_u32 v42, 34432, v41 // (op 12702) 
buffer_store_dword a39, v42, s[64:67], 0 offen sc1 // (op 12702) Store value




v_add_u32 v42, 33024, v41 // (op 12704) 
buffer_store_dword a40, v42, s[64:67], 0 offen sc1 // (op 12704) Store value

v_add_u32 v42, 33536, v41 // (op 12704) 
buffer_store_dword a41, v42, s[64:67], 0 offen sc1 // (op 12704) Store value

v_add_u32 v42, 34048, v41 // (op 12704) 
buffer_store_dword a42, v42, s[64:67], 0 offen sc1 // (op 12704) Store value

v_add_u32 v42, 34560, v41 // (op 12704) 
buffer_store_dword a43, v42, s[64:67], 0 offen sc1 // (op 12704) Store value




v_add_u32 v42, 33152, v41 // (op 12706) 
buffer_store_dword a44, v42, s[64:67], 0 offen sc1 // (op 12706) Store value

v_add_u32 v42, 33664, v41 // (op 12706) 
buffer_store_dword a45, v42, s[64:67], 0 offen sc1 // (op 12706) Store value

v_add_u32 v42, 34176, v41 // (op 12706) 
buffer_store_dword a46, v42, s[64:67], 0 offen sc1 // (op 12706) Store value

v_add_u32 v42, 34688, v41 // (op 12706) 
buffer_store_dword a47, v42, s[64:67], 0 offen sc1 // (op 12706) Store value




v_add_u32 v42, 16384, v41 // (op 12676) 
buffer_store_dword a16, v42, s[64:67], 0 offen sc1 // (op 12676) Store value

v_add_u32 v42, 16896, v41 // (op 12676) 
buffer_store_dword a17, v42, s[64:67], 0 offen sc1 // (op 12676) Store value

v_add_u32 v42, 17408, v41 // (op 12676) 
buffer_store_dword a18, v42, s[64:67], 0 offen sc1 // (op 12676) Store value

v_add_u32 v42, 17920, v41 // (op 12676) 
buffer_store_dword a19, v42, s[64:67], 0 offen sc1 // (op 12676) Store value




v_add_u32 v42, 16512, v41 // (op 12678) 
buffer_store_dword a20, v42, s[64:67], 0 offen sc1 // (op 12678) Store value

v_add_u32 v42, 17024, v41 // (op 12678) 
buffer_store_dword a21, v42, s[64:67], 0 offen sc1 // (op 12678) Store value

v_add_u32 v42, 17536, v41 // (op 12678) 
buffer_store_dword a22, v42, s[64:67], 0 offen sc1 // (op 12678) Store value

v_add_u32 v42, 18048, v41 // (op 12678) 
buffer_store_dword a23, v42, s[64:67], 0 offen sc1 // (op 12678) Store value




v_add_u32 v42, 16640, v41 // (op 12680) 
buffer_store_dword a24, v42, s[64:67], 0 offen sc1 // (op 12680) Store value

v_add_u32 v42, 17152, v41 // (op 12680) 
buffer_store_dword a25, v42, s[64:67], 0 offen sc1 // (op 12680) Store value

v_add_u32 v42, 17664, v41 // (op 12680) 
buffer_store_dword a26, v42, s[64:67], 0 offen sc1 // (op 12680) Store value

v_add_u32 v42, 18176, v41 // (op 12680) 
buffer_store_dword a27, v42, s[64:67], 0 offen sc1 // (op 12680) Store value




v_add_u32 v42, 16768, v41 // (op 12682) 
buffer_store_dword a28, v42, s[64:67], 0 offen sc1 // (op 12682) Store value

v_add_u32 v42, 17280, v41 // (op 12682) 
buffer_store_dword a29, v42, s[64:67], 0 offen sc1 // (op 12682) Store value

v_add_u32 v42, 17792, v41 // (op 12682) 
buffer_store_dword a30, v42, s[64:67], 0 offen sc1 // (op 12682) Store value

v_add_u32 v42, 18304, v41 // (op 12682) 
buffer_store_dword a31, v42, s[64:67], 0 offen sc1 // (op 12682) Store value




buffer_store_dword a0, v41, s[64:67], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v41, s[64:67], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a2, v41, s[64:67], 0 offen offset:1024 sc1 // (op 319) Store value
buffer_store_dword a3, v41, s[64:67], 0 offen offset:1536 sc1 // (op 319) Store value



buffer_store_dword a4, v41, s[64:67], 0 offen offset:128 sc1 // (op 12650) Store value
buffer_store_dword a5, v41, s[64:67], 0 offen offset:640 sc1 // (op 12650) Store value
buffer_store_dword a6, v41, s[64:67], 0 offen offset:1152 sc1 // (op 12650) Store value
buffer_store_dword a7, v41, s[64:67], 0 offen offset:1664 sc1 // (op 12650) Store value



buffer_store_dword a8, v41, s[64:67], 0 offen offset:256 sc1 // (op 12651) Store value
buffer_store_dword a9, v41, s[64:67], 0 offen offset:768 sc1 // (op 12651) Store value
buffer_store_dword a10, v41, s[64:67], 0 offen offset:1280 sc1 // (op 12651) Store value
buffer_store_dword a11, v41, s[64:67], 0 offen offset:1792 sc1 // (op 12651) Store value



buffer_store_dword a12, v41, s[64:67], 0 offen offset:384 sc1 // (op 12652) Store value
buffer_store_dword a13, v41, s[64:67], 0 offen offset:896 sc1 // (op 12652) Store value
buffer_store_dword a14, v41, s[64:67], 0 offen offset:1408 sc1 // (op 12652) Store value
buffer_store_dword a15, v41, s[64:67], 0 offen offset:1920 sc1 // (op 12652) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_327
s_mov_b32 s52, 1 // (op 325) call()
s_mov_b32 s55, 0 // (op 326) convert
s_mov_b32 s54, s16 // (op 326) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 326) 
s_add_u32 s84, s48, s54 // (op 326) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 326) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 326) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_322
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_322:
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_322:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 338) 
s_mov_b32 s52, scc // (op 338) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 338) 
s_mov_b32 s53, s54 // (op 338) convert

s_sub_u32 s54, s53, 1 // (op 338) 

s_cmp_lt_u32 s19, s54 // (op 338) 
s_mov_b32 s53, scc // (op 338) 

s_and_b32 s54, s52, s53 // (op 338) 
s_cmp_lg_u32 s54, 0 // (op 338) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 28842) 
v_lshrrev_b32 v36, 5, v1 // (op 28842) 
s_add_u32 s54, s53, s52 // (op 28842) 
s_lshl_b32 s54, s54, 7 // (op 28842) 

v_lshlrev_b32 v37, 2, v36 // (op 28842) 

v_add_lshl_u32 v36, s54, v37, 7 // (op 28842) 


v_and_b32 v37, -32, v1 // (op 28842) 
v_sub_u32 v39, v1, v37 // (op 28842) 

v_add_lshl_u32 v37, v36, v39, 2 // (op 28842) 


s_lshl_b32 s53, s15, 16 // (op 28844) 
s_mov_b32 s64, s28 // (op 28844) reg expression
s_mov_b32 s65, s29 // (op 28844) reg expression
s_mov_b32 s66, s53 // (op 28844) reg expression
s_mov_b32 s67, 131072 // (op 28844) call()

RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_342:
s_add_u32 s53, s16, 1 // (op 340) 
s_add_u32 s54, s53, s52 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 340) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 340) 

s_add_u32 s84, s48, s54 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 340) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 340) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 342) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_345
s_mov_b32 s53, 0 // (op 343) call()
s_add_u32 s54, s16, 1 // (op 344) 
s_add_u32 s84, s54, s52 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 344) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 344) 

s_add_u32 s54, s48, s84 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 344) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 344) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v44, v37, s[64:67], 0 offen // (op 357) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:512 // (op 357) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1024 // (op 357) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1536 // (op 357) Load value
v_accvgpr_read v48, a0 // (op 360) reg expression
v_accvgpr_read v49, a1 // (op 360) reg expression
v_accvgpr_read v50, a2 // (op 360) reg expression
v_accvgpr_read v51, a3 // (op 360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 362) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 362) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 362) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 362) 

v_accvgpr_write a0, v48 // (op 364) reg expression
v_accvgpr_write a1, v49 // (op 364) reg expression
v_accvgpr_write a2, v50 // (op 364) reg expression
v_accvgpr_write a3, v51 // (op 364) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:128 // (op 12799) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:640 // (op 12799) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1152 // (op 12799) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1664 // (op 12799) Load value
v_accvgpr_read v48, a4 // (op 12800) reg expression
v_accvgpr_read v49, a5 // (op 12800) reg expression
v_accvgpr_read v50, a6 // (op 12800) reg expression
v_accvgpr_read v51, a7 // (op 12800) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12801) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12801) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12801) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12801) 

v_accvgpr_write a4, v48 // (op 12802) reg expression
v_accvgpr_write a5, v49 // (op 12802) reg expression
v_accvgpr_write a6, v50 // (op 12802) reg expression
v_accvgpr_write a7, v51 // (op 12802) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:256 // (op 12806) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:768 // (op 12806) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1280 // (op 12806) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1792 // (op 12806) Load value
v_accvgpr_read v48, a8 // (op 12807) reg expression
v_accvgpr_read v49, a9 // (op 12807) reg expression
v_accvgpr_read v50, a10 // (op 12807) reg expression
v_accvgpr_read v51, a11 // (op 12807) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12808) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12808) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12808) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12808) 

v_accvgpr_write a8, v48 // (op 12809) reg expression
v_accvgpr_write a9, v49 // (op 12809) reg expression
v_accvgpr_write a10, v50 // (op 12809) reg expression
v_accvgpr_write a11, v51 // (op 12809) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:384 // (op 12813) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:896 // (op 12813) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1408 // (op 12813) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1920 // (op 12813) Load value
v_accvgpr_read v48, a12 // (op 12814) reg expression
v_accvgpr_read v49, a13 // (op 12814) reg expression
v_accvgpr_read v50, a14 // (op 12814) reg expression
v_accvgpr_read v51, a15 // (op 12814) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12815) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12815) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12815) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12815) 

v_accvgpr_write a12, v48 // (op 12816) reg expression
v_accvgpr_write a13, v49 // (op 12816) reg expression
v_accvgpr_write a14, v50 // (op 12816) reg expression
v_accvgpr_write a15, v51 // (op 12816) reg expression




v_mov_b32 v36, 16384 // (op 12850) 
v_add_i32 v39, v37, v36 // (op 12850) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12850) Load value

v_mov_b32 v36, 16896 // (op 12850) 
v_add_i32 v39, v37, v36 // (op 12850) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12850) Load value

v_mov_b32 v36, 17408 // (op 12850) 
v_add_i32 v39, v37, v36 // (op 12850) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12850) Load value

v_mov_b32 v36, 17920 // (op 12850) 
v_add_i32 v39, v37, v36 // (op 12850) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12850) Load value

v_accvgpr_read v48, a16 // (op 12847) reg expression
v_accvgpr_read v49, a17 // (op 12847) reg expression
v_accvgpr_read v50, a18 // (op 12847) reg expression
v_accvgpr_read v51, a19 // (op 12847) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12848) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12848) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12848) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12848) 

v_accvgpr_write a16, v48 // (op 12849) reg expression
v_accvgpr_write a17, v49 // (op 12849) reg expression
v_accvgpr_write a18, v50 // (op 12849) reg expression
v_accvgpr_write a19, v51 // (op 12849) reg expression




v_mov_b32 v36, 16512 // (op 12855) 
v_add_i32 v39, v37, v36 // (op 12855) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12855) Load value

v_mov_b32 v36, 17024 // (op 12855) 
v_add_i32 v39, v37, v36 // (op 12855) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12855) Load value

v_mov_b32 v36, 17536 // (op 12855) 
v_add_i32 v39, v37, v36 // (op 12855) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12855) Load value

v_mov_b32 v36, 18048 // (op 12855) 
v_add_i32 v39, v37, v36 // (op 12855) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12855) Load value

v_accvgpr_read v48, a20 // (op 12852) reg expression
v_accvgpr_read v49, a21 // (op 12852) reg expression
v_accvgpr_read v50, a22 // (op 12852) reg expression
v_accvgpr_read v51, a23 // (op 12852) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12853) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12853) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12853) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12853) 

v_accvgpr_write a20, v48 // (op 12854) reg expression
v_accvgpr_write a21, v49 // (op 12854) reg expression
v_accvgpr_write a22, v50 // (op 12854) reg expression
v_accvgpr_write a23, v51 // (op 12854) reg expression




v_mov_b32 v36, 16640 // (op 12860) 
v_add_i32 v39, v37, v36 // (op 12860) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12860) Load value

v_mov_b32 v36, 17152 // (op 12860) 
v_add_i32 v39, v37, v36 // (op 12860) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12860) Load value

v_mov_b32 v36, 17664 // (op 12860) 
v_add_i32 v39, v37, v36 // (op 12860) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12860) Load value

v_mov_b32 v36, 18176 // (op 12860) 
v_add_i32 v39, v37, v36 // (op 12860) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12860) Load value

v_accvgpr_read v48, a24 // (op 12857) reg expression
v_accvgpr_read v49, a25 // (op 12857) reg expression
v_accvgpr_read v50, a26 // (op 12857) reg expression
v_accvgpr_read v51, a27 // (op 12857) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12858) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12858) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12858) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12858) 

v_accvgpr_write a24, v48 // (op 12859) reg expression
v_accvgpr_write a25, v49 // (op 12859) reg expression
v_accvgpr_write a26, v50 // (op 12859) reg expression
v_accvgpr_write a27, v51 // (op 12859) reg expression




v_mov_b32 v36, 16768 // (op 12865) 
v_add_i32 v39, v37, v36 // (op 12865) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12865) Load value

v_mov_b32 v36, 17280 // (op 12865) 
v_add_i32 v39, v37, v36 // (op 12865) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12865) Load value

v_mov_b32 v36, 17792 // (op 12865) 
v_add_i32 v39, v37, v36 // (op 12865) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12865) Load value

v_mov_b32 v36, 18304 // (op 12865) 
v_add_i32 v39, v37, v36 // (op 12865) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12865) Load value

v_accvgpr_read v48, a28 // (op 12862) reg expression
v_accvgpr_read v49, a29 // (op 12862) reg expression
v_accvgpr_read v50, a30 // (op 12862) reg expression
v_accvgpr_read v51, a31 // (op 12862) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12863) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12863) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12863) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12863) 

v_accvgpr_write a28, v48 // (op 12864) reg expression
v_accvgpr_write a29, v49 // (op 12864) reg expression
v_accvgpr_write a30, v50 // (op 12864) reg expression
v_accvgpr_write a31, v51 // (op 12864) reg expression




v_mov_b32 v36, 32768 // (op 12898) 
v_add_i32 v39, v37, v36 // (op 12898) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12898) Load value

v_mov_b32 v36, 33280 // (op 12898) 
v_add_i32 v39, v37, v36 // (op 12898) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12898) Load value

v_mov_b32 v36, 33792 // (op 12898) 
v_add_i32 v39, v37, v36 // (op 12898) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12898) Load value

v_mov_b32 v36, 34304 // (op 12898) 
v_add_i32 v39, v37, v36 // (op 12898) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12898) Load value

v_accvgpr_read v48, a32 // (op 12895) reg expression
v_accvgpr_read v49, a33 // (op 12895) reg expression
v_accvgpr_read v50, a34 // (op 12895) reg expression
v_accvgpr_read v51, a35 // (op 12895) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12896) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12896) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12896) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12896) 

v_accvgpr_write a32, v48 // (op 12897) reg expression
v_accvgpr_write a33, v49 // (op 12897) reg expression
v_accvgpr_write a34, v50 // (op 12897) reg expression
v_accvgpr_write a35, v51 // (op 12897) reg expression




v_mov_b32 v36, 32896 // (op 12903) 
v_add_i32 v39, v37, v36 // (op 12903) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12903) Load value

v_mov_b32 v36, 33408 // (op 12903) 
v_add_i32 v39, v37, v36 // (op 12903) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12903) Load value

v_mov_b32 v36, 33920 // (op 12903) 
v_add_i32 v39, v37, v36 // (op 12903) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12903) Load value

v_mov_b32 v36, 34432 // (op 12903) 
v_add_i32 v39, v37, v36 // (op 12903) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12903) Load value

v_accvgpr_read v48, a36 // (op 12900) reg expression
v_accvgpr_read v49, a37 // (op 12900) reg expression
v_accvgpr_read v50, a38 // (op 12900) reg expression
v_accvgpr_read v51, a39 // (op 12900) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12901) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12901) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12901) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12901) 

v_accvgpr_write a36, v48 // (op 12902) reg expression
v_accvgpr_write a37, v49 // (op 12902) reg expression
v_accvgpr_write a38, v50 // (op 12902) reg expression
v_accvgpr_write a39, v51 // (op 12902) reg expression




v_mov_b32 v36, 33024 // (op 12908) 
v_add_i32 v39, v37, v36 // (op 12908) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12908) Load value

v_mov_b32 v36, 33536 // (op 12908) 
v_add_i32 v39, v37, v36 // (op 12908) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12908) Load value

v_mov_b32 v36, 34048 // (op 12908) 
v_add_i32 v39, v37, v36 // (op 12908) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12908) Load value

v_mov_b32 v36, 34560 // (op 12908) 
v_add_i32 v39, v37, v36 // (op 12908) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12908) Load value

v_accvgpr_read v48, a40 // (op 12905) reg expression
v_accvgpr_read v49, a41 // (op 12905) reg expression
v_accvgpr_read v50, a42 // (op 12905) reg expression
v_accvgpr_read v51, a43 // (op 12905) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12906) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12906) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12906) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12906) 

v_accvgpr_write a40, v48 // (op 12907) reg expression
v_accvgpr_write a41, v49 // (op 12907) reg expression
v_accvgpr_write a42, v50 // (op 12907) reg expression
v_accvgpr_write a43, v51 // (op 12907) reg expression




v_mov_b32 v36, 33152 // (op 12913) 
v_add_i32 v39, v37, v36 // (op 12913) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12913) Load value

v_mov_b32 v36, 33664 // (op 12913) 
v_add_i32 v39, v37, v36 // (op 12913) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12913) Load value

v_mov_b32 v36, 34176 // (op 12913) 
v_add_i32 v39, v37, v36 // (op 12913) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12913) Load value

v_mov_b32 v36, 34688 // (op 12913) 
v_add_i32 v39, v37, v36 // (op 12913) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12913) Load value

v_accvgpr_read v48, a44 // (op 12910) reg expression
v_accvgpr_read v49, a45 // (op 12910) reg expression
v_accvgpr_read v50, a46 // (op 12910) reg expression
v_accvgpr_read v51, a47 // (op 12910) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12911) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12911) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12911) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12911) 

v_accvgpr_write a44, v48 // (op 12912) reg expression
v_accvgpr_write a45, v49 // (op 12912) reg expression
v_accvgpr_write a46, v50 // (op 12912) reg expression
v_accvgpr_write a47, v51 // (op 12912) reg expression




v_mov_b32 v36, 49152 // (op 12946) 
v_add_i32 v39, v37, v36 // (op 12946) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12946) Load value

v_mov_b32 v36, 49664 // (op 12946) 
v_add_i32 v39, v37, v36 // (op 12946) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12946) Load value

v_mov_b32 v36, 50176 // (op 12946) 
v_add_i32 v39, v37, v36 // (op 12946) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12946) Load value

v_mov_b32 v36, 50688 // (op 12946) 
v_add_i32 v39, v37, v36 // (op 12946) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12946) Load value

v_accvgpr_read v48, a48 // (op 12943) reg expression
v_accvgpr_read v49, a49 // (op 12943) reg expression
v_accvgpr_read v50, a50 // (op 12943) reg expression
v_accvgpr_read v51, a51 // (op 12943) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12944) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12944) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12944) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12944) 

v_accvgpr_write a48, v48 // (op 12945) reg expression
v_accvgpr_write a49, v49 // (op 12945) reg expression
v_accvgpr_write a50, v50 // (op 12945) reg expression
v_accvgpr_write a51, v51 // (op 12945) reg expression




v_mov_b32 v36, 49280 // (op 12951) 
v_add_i32 v39, v37, v36 // (op 12951) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12951) Load value

v_mov_b32 v36, 49792 // (op 12951) 
v_add_i32 v39, v37, v36 // (op 12951) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12951) Load value

v_mov_b32 v36, 50304 // (op 12951) 
v_add_i32 v39, v37, v36 // (op 12951) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12951) Load value

v_mov_b32 v36, 50816 // (op 12951) 
v_add_i32 v39, v37, v36 // (op 12951) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12951) Load value

v_accvgpr_read v48, a52 // (op 12948) reg expression
v_accvgpr_read v49, a53 // (op 12948) reg expression
v_accvgpr_read v50, a54 // (op 12948) reg expression
v_accvgpr_read v51, a55 // (op 12948) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12949) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12949) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12949) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12949) 

v_accvgpr_write a52, v48 // (op 12950) reg expression
v_accvgpr_write a53, v49 // (op 12950) reg expression
v_accvgpr_write a54, v50 // (op 12950) reg expression
v_accvgpr_write a55, v51 // (op 12950) reg expression




v_mov_b32 v36, 49408 // (op 12956) 
v_add_i32 v39, v37, v36 // (op 12956) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12956) Load value

v_mov_b32 v36, 49920 // (op 12956) 
v_add_i32 v39, v37, v36 // (op 12956) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12956) Load value

v_mov_b32 v36, 50432 // (op 12956) 
v_add_i32 v39, v37, v36 // (op 12956) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12956) Load value

v_mov_b32 v36, 50944 // (op 12956) 
v_add_i32 v39, v37, v36 // (op 12956) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12956) Load value

v_accvgpr_read v48, a56 // (op 12953) reg expression
v_accvgpr_read v49, a57 // (op 12953) reg expression
v_accvgpr_read v50, a58 // (op 12953) reg expression
v_accvgpr_read v51, a59 // (op 12953) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12954) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12954) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12954) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12954) 

v_accvgpr_write a56, v48 // (op 12955) reg expression
v_accvgpr_write a57, v49 // (op 12955) reg expression
v_accvgpr_write a58, v50 // (op 12955) reg expression
v_accvgpr_write a59, v51 // (op 12955) reg expression




v_mov_b32 v36, 49536 // (op 12961) 
v_add_i32 v39, v37, v36 // (op 12961) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12961) Load value

v_mov_b32 v36, 50048 // (op 12961) 
v_add_i32 v39, v37, v36 // (op 12961) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12961) Load value

v_mov_b32 v36, 50560 // (op 12961) 
v_add_i32 v39, v37, v36 // (op 12961) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12961) Load value

v_mov_b32 v36, 51072 // (op 12961) 
v_add_i32 v39, v37, v36 // (op 12961) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12961) Load value

v_accvgpr_read v48, a60 // (op 12958) reg expression
v_accvgpr_read v49, a61 // (op 12958) reg expression
v_accvgpr_read v50, a62 // (op 12958) reg expression
v_accvgpr_read v51, a63 // (op 12958) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12959) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12959) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12959) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12959) 

v_accvgpr_write a60, v48 // (op 12960) reg expression
v_accvgpr_write a61, v49 // (op 12960) reg expression
v_accvgpr_write a62, v50 // (op 12960) reg expression
v_accvgpr_write a63, v51 // (op 12960) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_338
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_338:
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v34 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135
v_accvgpr_read v44, a0 // (op 50) 
v_accvgpr_read v45, a1 // (op 50) 
v_accvgpr_read v46, a2 // (op 50) 
v_accvgpr_read v47, a3 // (op 50) 
v_mul_f32 v48, v35, v44 // (op 50) 
v_mul_f32 v49, v35, v45 // (op 50) 
v_mul_f32 v50, v35, v46 // (op 50) 
v_mul_f32 v51, v35, v47 // (op 50) 

v_lshrrev_b32 v44, 16, v48 // (op 56) 
v_lshrrev_b32 v45, 16, v49 // (op 56) 
v_lshrrev_b32 v46, 16, v50 // (op 56) 
v_lshrrev_b32 v47, 16, v51 // (op 56) 


v_pack_b32_f16 v36, v44, v45 // (op 58) 
v_pack_b32_f16 v37, v46, v47 // (op 58) 

v_mov_b32 v39, v43 // (op 58) reg expression

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v44, a4 // (op 13043) 
v_accvgpr_read v45, a5 // (op 13043) 
v_accvgpr_read v46, a6 // (op 13043) 
v_accvgpr_read v47, a7 // (op 13043) 
v_mul_f32 v48, v35, v44 // (op 13043) 
v_mul_f32 v49, v35, v45 // (op 13043) 
v_mul_f32 v50, v35, v46 // (op 13043) 
v_mul_f32 v51, v35, v47 // (op 13043) 

v_lshrrev_b32 v44, 16, v48 // (op 13045) 
v_lshrrev_b32 v45, 16, v49 // (op 13045) 
v_lshrrev_b32 v46, 16, v50 // (op 13045) 
v_lshrrev_b32 v47, 16, v51 // (op 13045) 


v_pack_b32_f16 v36, v44, v45 // (op 13046) 
v_pack_b32_f16 v37, v46, v47 // (op 13046) 

s_lshl_b32 s52, s24, 6 // (op 13046) 
v_add_u32 v39, v43, s52 // (op 13046) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13046) Store value



v_accvgpr_read v44, a8 // (op 13057) 
v_accvgpr_read v45, a9 // (op 13057) 
v_accvgpr_read v46, a10 // (op 13057) 
v_accvgpr_read v47, a11 // (op 13057) 
v_mul_f32 v48, v35, v44 // (op 13057) 
v_mul_f32 v49, v35, v45 // (op 13057) 
v_mul_f32 v50, v35, v46 // (op 13057) 
v_mul_f32 v51, v35, v47 // (op 13057) 

v_lshrrev_b32 v44, 16, v48 // (op 13059) 
v_lshrrev_b32 v45, 16, v49 // (op 13059) 
v_lshrrev_b32 v46, 16, v50 // (op 13059) 
v_lshrrev_b32 v47, 16, v51 // (op 13059) 


v_pack_b32_f16 v36, v44, v45 // (op 13060) 
v_pack_b32_f16 v37, v46, v47 // (op 13060) 

s_lshl_b32 s52, s24, 7 // (op 13060) 
v_add_u32 v39, v43, s52 // (op 13060) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13060) Store value



v_accvgpr_read v44, a12 // (op 13071) 
v_accvgpr_read v45, a13 // (op 13071) 
v_accvgpr_read v46, a14 // (op 13071) 
v_accvgpr_read v47, a15 // (op 13071) 
v_mul_f32 v48, v35, v44 // (op 13071) 
v_mul_f32 v49, v35, v45 // (op 13071) 
v_mul_f32 v50, v35, v46 // (op 13071) 
v_mul_f32 v51, v35, v47 // (op 13071) 

v_lshrrev_b32 v44, 16, v48 // (op 13073) 
v_lshrrev_b32 v45, 16, v49 // (op 13073) 
v_lshrrev_b32 v46, 16, v50 // (op 13073) 
v_lshrrev_b32 v47, 16, v51 // (op 13073) 


v_pack_b32_f16 v36, v44, v45 // (op 13074) 
v_pack_b32_f16 v37, v46, v47 // (op 13074) 

s_lshl_b32 s52, s24, 6 // (op 13074) 
s_mul_i32 s53, 3, s52 // (op 13074) 

v_add_u32 v39, v43, s53 // (op 13074) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13074) Store value



v_accvgpr_read v44, a16 // (op 13107) 
v_accvgpr_read v45, a17 // (op 13107) 
v_accvgpr_read v46, a18 // (op 13107) 
v_accvgpr_read v47, a19 // (op 13107) 
v_mul_f32 v48, v35, v44 // (op 13107) 
v_mul_f32 v49, v35, v45 // (op 13107) 
v_mul_f32 v50, v35, v46 // (op 13107) 
v_mul_f32 v51, v35, v47 // (op 13107) 

v_lshrrev_b32 v44, 16, v48 // (op 13109) 
v_lshrrev_b32 v45, 16, v49 // (op 13109) 
v_lshrrev_b32 v46, 16, v50 // (op 13109) 
v_lshrrev_b32 v47, 16, v51 // (op 13109) 


v_pack_b32_f16 v36, v44, v45 // (op 13111) 
v_pack_b32_f16 v37, v46, v47 // (op 13111) 

v_add_u32 v39, 64, v43 // (op 13111) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13111) Store value



v_accvgpr_read v44, a20 // (op 13121) 
v_accvgpr_read v45, a21 // (op 13121) 
v_accvgpr_read v46, a22 // (op 13121) 
v_accvgpr_read v47, a23 // (op 13121) 
v_mul_f32 v48, v35, v44 // (op 13121) 
v_mul_f32 v49, v35, v45 // (op 13121) 
v_mul_f32 v50, v35, v46 // (op 13121) 
v_mul_f32 v51, v35, v47 // (op 13121) 

v_lshrrev_b32 v44, 16, v48 // (op 13123) 
v_lshrrev_b32 v45, 16, v49 // (op 13123) 
v_lshrrev_b32 v46, 16, v50 // (op 13123) 
v_lshrrev_b32 v47, 16, v51 // (op 13123) 


v_pack_b32_f16 v36, v44, v45 // (op 13113) 
v_pack_b32_f16 v37, v46, v47 // (op 13113) 

s_lshl_b32 s52, s24, 6 // (op 13113) 
s_add_u32 s52, s52, 64 // (op 13113) 
v_add_u32 v39, v43, s52 // (op 13113) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13113) Store value



v_accvgpr_read v44, a24 // (op 13127) 
v_accvgpr_read v45, a25 // (op 13127) 
v_accvgpr_read v46, a26 // (op 13127) 
v_accvgpr_read v47, a27 // (op 13127) 
v_mul_f32 v48, v35, v44 // (op 13127) 
v_mul_f32 v49, v35, v45 // (op 13127) 
v_mul_f32 v50, v35, v46 // (op 13127) 
v_mul_f32 v51, v35, v47 // (op 13127) 

v_lshrrev_b32 v44, 16, v48 // (op 13129) 
v_lshrrev_b32 v45, 16, v49 // (op 13129) 
v_lshrrev_b32 v46, 16, v50 // (op 13129) 
v_lshrrev_b32 v47, 16, v51 // (op 13129) 


v_pack_b32_f16 v36, v44, v45 // (op 13115) 
v_pack_b32_f16 v37, v46, v47 // (op 13115) 

s_lshl_b32 s52, s24, 7 // (op 13115) 
s_add_u32 s52, s52, 64 // (op 13115) 
v_add_u32 v39, v43, s52 // (op 13115) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13115) Store value



v_accvgpr_read v44, a28 // (op 13133) 
v_accvgpr_read v45, a29 // (op 13133) 
v_accvgpr_read v46, a30 // (op 13133) 
v_accvgpr_read v47, a31 // (op 13133) 
v_mul_f32 v48, v35, v44 // (op 13133) 
v_mul_f32 v49, v35, v45 // (op 13133) 
v_mul_f32 v50, v35, v46 // (op 13133) 
v_mul_f32 v51, v35, v47 // (op 13133) 

v_lshrrev_b32 v44, 16, v48 // (op 13135) 
v_lshrrev_b32 v45, 16, v49 // (op 13135) 
v_lshrrev_b32 v46, 16, v50 // (op 13135) 
v_lshrrev_b32 v47, 16, v51 // (op 13135) 


v_pack_b32_f16 v36, v44, v45 // (op 13117) 
v_pack_b32_f16 v37, v46, v47 // (op 13117) 

s_lshl_b32 s52, s24, 6 // (op 13117) 
s_mul_i32 s53, 3, s52 // (op 13117) 
s_add_u32 s53, s53, 64 // (op 13117) 

v_add_u32 v39, v43, s53 // (op 13117) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13117) Store value



v_accvgpr_read v44, a32 // (op 13183) 
v_accvgpr_read v45, a33 // (op 13183) 
v_accvgpr_read v46, a34 // (op 13183) 
v_accvgpr_read v47, a35 // (op 13183) 
v_mul_f32 v48, v35, v44 // (op 13183) 
v_mul_f32 v49, v35, v45 // (op 13183) 
v_mul_f32 v50, v35, v46 // (op 13183) 
v_mul_f32 v51, v35, v47 // (op 13183) 

v_lshrrev_b32 v44, 16, v48 // (op 13185) 
v_lshrrev_b32 v45, 16, v49 // (op 13185) 
v_lshrrev_b32 v46, 16, v50 // (op 13185) 
v_lshrrev_b32 v47, 16, v51 // (op 13185) 


v_pack_b32_f16 v36, v44, v45 // (op 13187) 
v_pack_b32_f16 v37, v46, v47 // (op 13187) 

v_add_u32 v39, 128, v43 // (op 13187) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13187) Store value



v_accvgpr_read v44, a36 // (op 13197) 
v_accvgpr_read v45, a37 // (op 13197) 
v_accvgpr_read v46, a38 // (op 13197) 
v_accvgpr_read v47, a39 // (op 13197) 
v_mul_f32 v48, v35, v44 // (op 13197) 
v_mul_f32 v49, v35, v45 // (op 13197) 
v_mul_f32 v50, v35, v46 // (op 13197) 
v_mul_f32 v51, v35, v47 // (op 13197) 

v_lshrrev_b32 v44, 16, v48 // (op 13199) 
v_lshrrev_b32 v45, 16, v49 // (op 13199) 
v_lshrrev_b32 v46, 16, v50 // (op 13199) 
v_lshrrev_b32 v47, 16, v51 // (op 13199) 


v_pack_b32_f16 v36, v44, v45 // (op 13189) 
v_pack_b32_f16 v37, v46, v47 // (op 13189) 

s_lshl_b32 s52, s24, 6 // (op 13189) 
s_add_u32 s52, s52, 128 // (op 13189) 
v_add_u32 v39, v43, s52 // (op 13189) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13189) Store value



v_accvgpr_read v44, a40 // (op 13203) 
v_accvgpr_read v45, a41 // (op 13203) 
v_accvgpr_read v46, a42 // (op 13203) 
v_accvgpr_read v47, a43 // (op 13203) 
v_mul_f32 v48, v35, v44 // (op 13203) 
v_mul_f32 v49, v35, v45 // (op 13203) 
v_mul_f32 v50, v35, v46 // (op 13203) 
v_mul_f32 v51, v35, v47 // (op 13203) 

v_lshrrev_b32 v44, 16, v48 // (op 13205) 
v_lshrrev_b32 v45, 16, v49 // (op 13205) 
v_lshrrev_b32 v46, 16, v50 // (op 13205) 
v_lshrrev_b32 v47, 16, v51 // (op 13205) 


v_pack_b32_f16 v36, v44, v45 // (op 13191) 
v_pack_b32_f16 v37, v46, v47 // (op 13191) 

s_lshl_b32 s52, s24, 7 // (op 13191) 
s_add_u32 s52, s52, 128 // (op 13191) 
v_add_u32 v39, v43, s52 // (op 13191) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13191) Store value



v_accvgpr_read v44, a44 // (op 13209) 
v_accvgpr_read v45, a45 // (op 13209) 
v_accvgpr_read v46, a46 // (op 13209) 
v_accvgpr_read v47, a47 // (op 13209) 
v_mul_f32 v48, v35, v44 // (op 13209) 
v_mul_f32 v49, v35, v45 // (op 13209) 
v_mul_f32 v50, v35, v46 // (op 13209) 
v_mul_f32 v51, v35, v47 // (op 13209) 

v_lshrrev_b32 v44, 16, v48 // (op 13211) 
v_lshrrev_b32 v45, 16, v49 // (op 13211) 
v_lshrrev_b32 v46, 16, v50 // (op 13211) 
v_lshrrev_b32 v47, 16, v51 // (op 13211) 


v_pack_b32_f16 v36, v44, v45 // (op 13193) 
v_pack_b32_f16 v37, v46, v47 // (op 13193) 

s_lshl_b32 s52, s24, 6 // (op 13193) 
s_mul_i32 s53, 3, s52 // (op 13193) 
s_add_u32 s53, s53, 128 // (op 13193) 

v_add_u32 v39, v43, s53 // (op 13193) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13193) Store value



v_accvgpr_read v44, a48 // (op 13259) 
v_accvgpr_read v45, a49 // (op 13259) 
v_accvgpr_read v46, a50 // (op 13259) 
v_accvgpr_read v47, a51 // (op 13259) 
v_mul_f32 v48, v35, v44 // (op 13259) 
v_mul_f32 v49, v35, v45 // (op 13259) 
v_mul_f32 v50, v35, v46 // (op 13259) 
v_mul_f32 v51, v35, v47 // (op 13259) 

v_lshrrev_b32 v44, 16, v48 // (op 13261) 
v_lshrrev_b32 v45, 16, v49 // (op 13261) 
v_lshrrev_b32 v46, 16, v50 // (op 13261) 
v_lshrrev_b32 v47, 16, v51 // (op 13261) 


v_pack_b32_f16 v36, v44, v45 // (op 13263) 
v_pack_b32_f16 v37, v46, v47 // (op 13263) 

v_add_u32 v39, 192, v43 // (op 13263) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13263) Store value



v_accvgpr_read v44, a52 // (op 13273) 
v_accvgpr_read v45, a53 // (op 13273) 
v_accvgpr_read v46, a54 // (op 13273) 
v_accvgpr_read v47, a55 // (op 13273) 
v_mul_f32 v48, v35, v44 // (op 13273) 
v_mul_f32 v49, v35, v45 // (op 13273) 
v_mul_f32 v50, v35, v46 // (op 13273) 
v_mul_f32 v51, v35, v47 // (op 13273) 

v_lshrrev_b32 v44, 16, v48 // (op 13275) 
v_lshrrev_b32 v45, 16, v49 // (op 13275) 
v_lshrrev_b32 v46, 16, v50 // (op 13275) 
v_lshrrev_b32 v47, 16, v51 // (op 13275) 


v_pack_b32_f16 v36, v44, v45 // (op 13265) 
v_pack_b32_f16 v37, v46, v47 // (op 13265) 

s_lshl_b32 s52, s24, 6 // (op 13265) 
s_add_u32 s52, s52, 192 // (op 13265) 
v_add_u32 v39, v43, s52 // (op 13265) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13265) Store value



v_accvgpr_read v44, a56 // (op 13279) 
v_accvgpr_read v45, a57 // (op 13279) 
v_accvgpr_read v46, a58 // (op 13279) 
v_accvgpr_read v47, a59 // (op 13279) 
v_mul_f32 v48, v35, v44 // (op 13279) 
v_mul_f32 v49, v35, v45 // (op 13279) 
v_mul_f32 v50, v35, v46 // (op 13279) 
v_mul_f32 v51, v35, v47 // (op 13279) 

v_lshrrev_b32 v44, 16, v48 // (op 13281) 
v_lshrrev_b32 v45, 16, v49 // (op 13281) 
v_lshrrev_b32 v46, 16, v50 // (op 13281) 
v_lshrrev_b32 v47, 16, v51 // (op 13281) 


v_pack_b32_f16 v36, v44, v45 // (op 13267) 
v_pack_b32_f16 v37, v46, v47 // (op 13267) 

s_lshl_b32 s52, s24, 7 // (op 13267) 
s_add_u32 s52, s52, 192 // (op 13267) 
v_add_u32 v39, v43, s52 // (op 13267) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13267) Store value



v_accvgpr_read v44, a60 // (op 13285) 
v_accvgpr_read v45, a61 // (op 13285) 
v_accvgpr_read v46, a62 // (op 13285) 
v_accvgpr_read v47, a63 // (op 13285) 
v_mul_f32 v48, v35, v44 // (op 13285) 
v_mul_f32 v49, v35, v45 // (op 13285) 
v_mul_f32 v50, v35, v46 // (op 13285) 
v_mul_f32 v51, v35, v47 // (op 13285) 

v_lshrrev_b32 v44, 16, v48 // (op 13287) 
v_lshrrev_b32 v45, 16, v49 // (op 13287) 
v_lshrrev_b32 v46, 16, v50 // (op 13287) 
v_lshrrev_b32 v47, 16, v51 // (op 13287) 


v_pack_b32_f16 v36, v44, v45 // (op 13269) 
v_pack_b32_f16 v37, v46, v47 // (op 13269) 

s_lshl_b32 s52, s24, 6 // (op 13269) 
s_mul_i32 s53, 3, s52 // (op 13269) 
s_add_u32 s53, s53, 192 // (op 13269) 

v_add_u32 v39, v43, s53 // (op 13269) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13269) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135:


v_add_u32 v36, 192, v38 // (op 13657) 

buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13657) Load value

v_accvgpr_read v44, a48 // (op 13644) 
v_accvgpr_read v45, a49 // (op 13644) 
v_accvgpr_read v46, a50 // (op 13644) 
v_accvgpr_read v47, a51 // (op 13644) 
v_mul_f32 v48, v35, v44 // (op 13644) 
v_mul_f32 v49, v35, v45 // (op 13644) 
v_mul_f32 v50, v35, v46 // (op 13644) 
v_mul_f32 v51, v35, v47 // (op 13644) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13656) 
v_lshrrev_b32 v37, 16, v40 // (op 13656) 
v_lshlrev_b32 v36, 16, v36 // (op 13656) 
v_lshlrev_b32 v37, 16, v37 // (op 13656) 
v_mul_f32 v44, v34, v36 // (op 13656) 
v_mul_f32 v45, v34, v37 // (op 13656) 
v_and_b32 v36, 65535, v41 // (op 13656) 
v_lshrrev_b32 v37, 16, v41 // (op 13656) 
v_lshlrev_b32 v36, 16, v36 // (op 13656) 
v_lshlrev_b32 v37, 16, v37 // (op 13656) 
v_mul_f32 v46, v34, v36 // (op 13656) 
v_mul_f32 v47, v34, v37 // (op 13656) 


v_add_f32 v52, v44, v48 // (op 13645) 
v_add_f32 v53, v45, v49 // (op 13645) 
v_add_f32 v54, v46, v50 // (op 13645) 
v_add_f32 v55, v47, v51 // (op 13645) 


v_lshrrev_b32 v44, 16, v52 // (op 13646) 
v_lshrrev_b32 v45, 16, v53 // (op 13646) 
v_lshrrev_b32 v46, 16, v54 // (op 13646) 
v_lshrrev_b32 v47, 16, v55 // (op 13646) 


v_pack_b32_f16 v36, v44, v45 // (op 13648) 
v_pack_b32_f16 v37, v46, v47 // (op 13648) 

v_add_u32 v39, 192, v43 // (op 13648) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13648) Store value





s_lshl_b32 s52, s4, 6 // (op 13662) 
s_add_u32 s52, s52, 192 // (op 13662) 
v_add_u32 v36, v38, s52 // (op 13662) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13662) Load value

v_accvgpr_read v44, a52 // (op 13674) 
v_accvgpr_read v45, a53 // (op 13674) 
v_accvgpr_read v46, a54 // (op 13674) 
v_accvgpr_read v47, a55 // (op 13674) 
v_mul_f32 v48, v35, v44 // (op 13674) 
v_mul_f32 v49, v35, v45 // (op 13674) 
v_mul_f32 v50, v35, v46 // (op 13674) 
v_mul_f32 v51, v35, v47 // (op 13674) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13659) 
v_lshrrev_b32 v37, 16, v40 // (op 13659) 
v_lshlrev_b32 v36, 16, v36 // (op 13659) 
v_lshlrev_b32 v37, 16, v37 // (op 13659) 
v_mul_f32 v44, v34, v36 // (op 13659) 
v_mul_f32 v45, v34, v37 // (op 13659) 
v_and_b32 v36, 65535, v41 // (op 13659) 
v_lshrrev_b32 v37, 16, v41 // (op 13659) 
v_lshlrev_b32 v36, 16, v36 // (op 13659) 
v_lshlrev_b32 v37, 16, v37 // (op 13659) 
v_mul_f32 v46, v34, v36 // (op 13659) 
v_mul_f32 v47, v34, v37 // (op 13659) 


v_add_f32 v52, v44, v48 // (op 13660) 
v_add_f32 v53, v45, v49 // (op 13660) 
v_add_f32 v54, v46, v50 // (op 13660) 
v_add_f32 v55, v47, v51 // (op 13660) 


v_lshrrev_b32 v44, 16, v52 // (op 13661) 
v_lshrrev_b32 v45, 16, v53 // (op 13661) 
v_lshrrev_b32 v46, 16, v54 // (op 13661) 
v_lshrrev_b32 v47, 16, v55 // (op 13661) 


v_pack_b32_f16 v36, v44, v45 // (op 13650) 
v_pack_b32_f16 v37, v46, v47 // (op 13650) 

s_lshl_b32 s52, s24, 6 // (op 13650) 
s_add_u32 s52, s52, 192 // (op 13650) 
v_add_u32 v39, v43, s52 // (op 13650) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13650) Store value





s_lshl_b32 s52, s4, 7 // (op 13667) 
s_add_u32 s52, s52, 192 // (op 13667) 
v_add_u32 v36, v38, s52 // (op 13667) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13667) Load value

v_accvgpr_read v44, a56 // (op 13676) 
v_accvgpr_read v45, a57 // (op 13676) 
v_accvgpr_read v46, a58 // (op 13676) 
v_accvgpr_read v47, a59 // (op 13676) 
v_mul_f32 v48, v35, v44 // (op 13676) 
v_mul_f32 v49, v35, v45 // (op 13676) 
v_mul_f32 v50, v35, v46 // (op 13676) 
v_mul_f32 v51, v35, v47 // (op 13676) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13664) 
v_lshrrev_b32 v37, 16, v40 // (op 13664) 
v_lshlrev_b32 v36, 16, v36 // (op 13664) 
v_lshlrev_b32 v37, 16, v37 // (op 13664) 
v_mul_f32 v44, v34, v36 // (op 13664) 
v_mul_f32 v45, v34, v37 // (op 13664) 
v_and_b32 v36, 65535, v41 // (op 13664) 
v_lshrrev_b32 v37, 16, v41 // (op 13664) 
v_lshlrev_b32 v36, 16, v36 // (op 13664) 
v_lshlrev_b32 v37, 16, v37 // (op 13664) 
v_mul_f32 v46, v34, v36 // (op 13664) 
v_mul_f32 v47, v34, v37 // (op 13664) 


v_add_f32 v52, v44, v48 // (op 13665) 
v_add_f32 v53, v45, v49 // (op 13665) 
v_add_f32 v54, v46, v50 // (op 13665) 
v_add_f32 v55, v47, v51 // (op 13665) 


v_lshrrev_b32 v44, 16, v52 // (op 13666) 
v_lshrrev_b32 v45, 16, v53 // (op 13666) 
v_lshrrev_b32 v46, 16, v54 // (op 13666) 
v_lshrrev_b32 v47, 16, v55 // (op 13666) 


v_pack_b32_f16 v36, v44, v45 // (op 13652) 
v_pack_b32_f16 v37, v46, v47 // (op 13652) 

s_lshl_b32 s52, s24, 7 // (op 13652) 
s_add_u32 s52, s52, 192 // (op 13652) 
v_add_u32 v39, v43, s52 // (op 13652) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13652) Store value





s_lshl_b32 s52, s4, 6 // (op 13672) 
s_mul_i32 s53, 3, s52 // (op 13672) 
s_add_u32 s53, s53, 192 // (op 13672) 

v_add_u32 v36, v38, s53 // (op 13672) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13672) Load value

v_accvgpr_read v44, a60 // (op 13678) 
v_accvgpr_read v45, a61 // (op 13678) 
v_accvgpr_read v46, a62 // (op 13678) 
v_accvgpr_read v47, a63 // (op 13678) 
v_mul_f32 v48, v35, v44 // (op 13678) 
v_mul_f32 v49, v35, v45 // (op 13678) 
v_mul_f32 v50, v35, v46 // (op 13678) 
v_mul_f32 v51, v35, v47 // (op 13678) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13669) 
v_lshrrev_b32 v37, 16, v40 // (op 13669) 
v_lshlrev_b32 v36, 16, v36 // (op 13669) 
v_lshlrev_b32 v37, 16, v37 // (op 13669) 
v_mul_f32 v44, v34, v36 // (op 13669) 
v_mul_f32 v45, v34, v37 // (op 13669) 
v_and_b32 v36, 65535, v41 // (op 13669) 
v_lshrrev_b32 v37, 16, v41 // (op 13669) 
v_lshlrev_b32 v36, 16, v36 // (op 13669) 
v_lshlrev_b32 v37, 16, v37 // (op 13669) 
v_mul_f32 v46, v34, v36 // (op 13669) 
v_mul_f32 v47, v34, v37 // (op 13669) 


v_add_f32 v52, v44, v48 // (op 13670) 
v_add_f32 v53, v45, v49 // (op 13670) 
v_add_f32 v54, v46, v50 // (op 13670) 
v_add_f32 v55, v47, v51 // (op 13670) 


v_lshrrev_b32 v44, 16, v52 // (op 13671) 
v_lshrrev_b32 v45, 16, v53 // (op 13671) 
v_lshrrev_b32 v46, 16, v54 // (op 13671) 
v_lshrrev_b32 v47, 16, v55 // (op 13671) 


v_pack_b32_f16 v36, v44, v45 // (op 13654) 
v_pack_b32_f16 v37, v46, v47 // (op 13654) 

s_lshl_b32 s52, s24, 6 // (op 13654) 
s_mul_i32 s53, 3, s52 // (op 13654) 
s_add_u32 s53, s53, 192 // (op 13654) 

v_add_u32 v39, v43, s53 // (op 13654) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13654) Store value





v_add_u32 v36, 128, v38 // (op 13570) 

buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13570) Load value

v_accvgpr_read v44, a32 // (op 13557) 
v_accvgpr_read v45, a33 // (op 13557) 
v_accvgpr_read v46, a34 // (op 13557) 
v_accvgpr_read v47, a35 // (op 13557) 
v_mul_f32 v48, v35, v44 // (op 13557) 
v_mul_f32 v49, v35, v45 // (op 13557) 
v_mul_f32 v50, v35, v46 // (op 13557) 
v_mul_f32 v51, v35, v47 // (op 13557) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13569) 
v_lshrrev_b32 v37, 16, v40 // (op 13569) 
v_lshlrev_b32 v36, 16, v36 // (op 13569) 
v_lshlrev_b32 v37, 16, v37 // (op 13569) 
v_mul_f32 v44, v34, v36 // (op 13569) 
v_mul_f32 v45, v34, v37 // (op 13569) 
v_and_b32 v36, 65535, v41 // (op 13569) 
v_lshrrev_b32 v37, 16, v41 // (op 13569) 
v_lshlrev_b32 v36, 16, v36 // (op 13569) 
v_lshlrev_b32 v37, 16, v37 // (op 13569) 
v_mul_f32 v46, v34, v36 // (op 13569) 
v_mul_f32 v47, v34, v37 // (op 13569) 


v_add_f32 v52, v44, v48 // (op 13558) 
v_add_f32 v53, v45, v49 // (op 13558) 
v_add_f32 v54, v46, v50 // (op 13558) 
v_add_f32 v55, v47, v51 // (op 13558) 


v_lshrrev_b32 v44, 16, v52 // (op 13559) 
v_lshrrev_b32 v45, 16, v53 // (op 13559) 
v_lshrrev_b32 v46, 16, v54 // (op 13559) 
v_lshrrev_b32 v47, 16, v55 // (op 13559) 


v_pack_b32_f16 v36, v44, v45 // (op 13561) 
v_pack_b32_f16 v37, v46, v47 // (op 13561) 

v_add_u32 v39, 128, v43 // (op 13561) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13561) Store value





s_lshl_b32 s52, s4, 6 // (op 13575) 
s_add_u32 s52, s52, 128 // (op 13575) 
v_add_u32 v36, v38, s52 // (op 13575) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13575) Load value

v_accvgpr_read v44, a36 // (op 13587) 
v_accvgpr_read v45, a37 // (op 13587) 
v_accvgpr_read v46, a38 // (op 13587) 
v_accvgpr_read v47, a39 // (op 13587) 
v_mul_f32 v48, v35, v44 // (op 13587) 
v_mul_f32 v49, v35, v45 // (op 13587) 
v_mul_f32 v50, v35, v46 // (op 13587) 
v_mul_f32 v51, v35, v47 // (op 13587) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13572) 
v_lshrrev_b32 v37, 16, v40 // (op 13572) 
v_lshlrev_b32 v36, 16, v36 // (op 13572) 
v_lshlrev_b32 v37, 16, v37 // (op 13572) 
v_mul_f32 v44, v34, v36 // (op 13572) 
v_mul_f32 v45, v34, v37 // (op 13572) 
v_and_b32 v36, 65535, v41 // (op 13572) 
v_lshrrev_b32 v37, 16, v41 // (op 13572) 
v_lshlrev_b32 v36, 16, v36 // (op 13572) 
v_lshlrev_b32 v37, 16, v37 // (op 13572) 
v_mul_f32 v46, v34, v36 // (op 13572) 
v_mul_f32 v47, v34, v37 // (op 13572) 


v_add_f32 v52, v44, v48 // (op 13573) 
v_add_f32 v53, v45, v49 // (op 13573) 
v_add_f32 v54, v46, v50 // (op 13573) 
v_add_f32 v55, v47, v51 // (op 13573) 


v_lshrrev_b32 v44, 16, v52 // (op 13574) 
v_lshrrev_b32 v45, 16, v53 // (op 13574) 
v_lshrrev_b32 v46, 16, v54 // (op 13574) 
v_lshrrev_b32 v47, 16, v55 // (op 13574) 


v_pack_b32_f16 v36, v44, v45 // (op 13563) 
v_pack_b32_f16 v37, v46, v47 // (op 13563) 

s_lshl_b32 s52, s24, 6 // (op 13563) 
s_add_u32 s52, s52, 128 // (op 13563) 
v_add_u32 v39, v43, s52 // (op 13563) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13563) Store value





s_lshl_b32 s52, s4, 7 // (op 13580) 
s_add_u32 s52, s52, 128 // (op 13580) 
v_add_u32 v36, v38, s52 // (op 13580) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13580) Load value

v_accvgpr_read v44, a40 // (op 13589) 
v_accvgpr_read v45, a41 // (op 13589) 
v_accvgpr_read v46, a42 // (op 13589) 
v_accvgpr_read v47, a43 // (op 13589) 
v_mul_f32 v48, v35, v44 // (op 13589) 
v_mul_f32 v49, v35, v45 // (op 13589) 
v_mul_f32 v50, v35, v46 // (op 13589) 
v_mul_f32 v51, v35, v47 // (op 13589) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13577) 
v_lshrrev_b32 v37, 16, v40 // (op 13577) 
v_lshlrev_b32 v36, 16, v36 // (op 13577) 
v_lshlrev_b32 v37, 16, v37 // (op 13577) 
v_mul_f32 v44, v34, v36 // (op 13577) 
v_mul_f32 v45, v34, v37 // (op 13577) 
v_and_b32 v36, 65535, v41 // (op 13577) 
v_lshrrev_b32 v37, 16, v41 // (op 13577) 
v_lshlrev_b32 v36, 16, v36 // (op 13577) 
v_lshlrev_b32 v37, 16, v37 // (op 13577) 
v_mul_f32 v46, v34, v36 // (op 13577) 
v_mul_f32 v47, v34, v37 // (op 13577) 


v_add_f32 v52, v44, v48 // (op 13578) 
v_add_f32 v53, v45, v49 // (op 13578) 
v_add_f32 v54, v46, v50 // (op 13578) 
v_add_f32 v55, v47, v51 // (op 13578) 


v_lshrrev_b32 v44, 16, v52 // (op 13579) 
v_lshrrev_b32 v45, 16, v53 // (op 13579) 
v_lshrrev_b32 v46, 16, v54 // (op 13579) 
v_lshrrev_b32 v47, 16, v55 // (op 13579) 


v_pack_b32_f16 v36, v44, v45 // (op 13565) 
v_pack_b32_f16 v37, v46, v47 // (op 13565) 

s_lshl_b32 s52, s24, 7 // (op 13565) 
s_add_u32 s52, s52, 128 // (op 13565) 
v_add_u32 v39, v43, s52 // (op 13565) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13565) Store value





s_lshl_b32 s52, s4, 6 // (op 13585) 
s_mul_i32 s53, 3, s52 // (op 13585) 
s_add_u32 s53, s53, 128 // (op 13585) 

v_add_u32 v36, v38, s53 // (op 13585) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13585) Load value

v_accvgpr_read v44, a44 // (op 13591) 
v_accvgpr_read v45, a45 // (op 13591) 
v_accvgpr_read v46, a46 // (op 13591) 
v_accvgpr_read v47, a47 // (op 13591) 
v_mul_f32 v48, v35, v44 // (op 13591) 
v_mul_f32 v49, v35, v45 // (op 13591) 
v_mul_f32 v50, v35, v46 // (op 13591) 
v_mul_f32 v51, v35, v47 // (op 13591) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13582) 
v_lshrrev_b32 v37, 16, v40 // (op 13582) 
v_lshlrev_b32 v36, 16, v36 // (op 13582) 
v_lshlrev_b32 v37, 16, v37 // (op 13582) 
v_mul_f32 v44, v34, v36 // (op 13582) 
v_mul_f32 v45, v34, v37 // (op 13582) 
v_and_b32 v36, 65535, v41 // (op 13582) 
v_lshrrev_b32 v37, 16, v41 // (op 13582) 
v_lshlrev_b32 v36, 16, v36 // (op 13582) 
v_lshlrev_b32 v37, 16, v37 // (op 13582) 
v_mul_f32 v46, v34, v36 // (op 13582) 
v_mul_f32 v47, v34, v37 // (op 13582) 


v_add_f32 v52, v44, v48 // (op 13583) 
v_add_f32 v53, v45, v49 // (op 13583) 
v_add_f32 v54, v46, v50 // (op 13583) 
v_add_f32 v55, v47, v51 // (op 13583) 


v_lshrrev_b32 v44, 16, v52 // (op 13584) 
v_lshrrev_b32 v45, 16, v53 // (op 13584) 
v_lshrrev_b32 v46, 16, v54 // (op 13584) 
v_lshrrev_b32 v47, 16, v55 // (op 13584) 


v_pack_b32_f16 v36, v44, v45 // (op 13567) 
v_pack_b32_f16 v37, v46, v47 // (op 13567) 

s_lshl_b32 s52, s24, 6 // (op 13567) 
s_mul_i32 s53, 3, s52 // (op 13567) 
s_add_u32 s53, s53, 128 // (op 13567) 

v_add_u32 v39, v43, s53 // (op 13567) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13567) Store value





v_add_u32 v36, 64, v38 // (op 13483) 

buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13483) Load value

v_accvgpr_read v44, a16 // (op 13470) 
v_accvgpr_read v45, a17 // (op 13470) 
v_accvgpr_read v46, a18 // (op 13470) 
v_accvgpr_read v47, a19 // (op 13470) 
v_mul_f32 v48, v35, v44 // (op 13470) 
v_mul_f32 v49, v35, v45 // (op 13470) 
v_mul_f32 v50, v35, v46 // (op 13470) 
v_mul_f32 v51, v35, v47 // (op 13470) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13482) 
v_lshrrev_b32 v37, 16, v40 // (op 13482) 
v_lshlrev_b32 v36, 16, v36 // (op 13482) 
v_lshlrev_b32 v37, 16, v37 // (op 13482) 
v_mul_f32 v44, v34, v36 // (op 13482) 
v_mul_f32 v45, v34, v37 // (op 13482) 
v_and_b32 v36, 65535, v41 // (op 13482) 
v_lshrrev_b32 v37, 16, v41 // (op 13482) 
v_lshlrev_b32 v36, 16, v36 // (op 13482) 
v_lshlrev_b32 v37, 16, v37 // (op 13482) 
v_mul_f32 v46, v34, v36 // (op 13482) 
v_mul_f32 v47, v34, v37 // (op 13482) 


v_add_f32 v52, v44, v48 // (op 13471) 
v_add_f32 v53, v45, v49 // (op 13471) 
v_add_f32 v54, v46, v50 // (op 13471) 
v_add_f32 v55, v47, v51 // (op 13471) 


v_lshrrev_b32 v44, 16, v52 // (op 13472) 
v_lshrrev_b32 v45, 16, v53 // (op 13472) 
v_lshrrev_b32 v46, 16, v54 // (op 13472) 
v_lshrrev_b32 v47, 16, v55 // (op 13472) 


v_pack_b32_f16 v36, v44, v45 // (op 13474) 
v_pack_b32_f16 v37, v46, v47 // (op 13474) 

v_add_u32 v39, 64, v43 // (op 13474) 

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13474) Store value





s_lshl_b32 s52, s4, 6 // (op 13488) 
s_add_u32 s52, s52, 64 // (op 13488) 
v_add_u32 v36, v38, s52 // (op 13488) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13488) Load value

v_accvgpr_read v44, a20 // (op 13500) 
v_accvgpr_read v45, a21 // (op 13500) 
v_accvgpr_read v46, a22 // (op 13500) 
v_accvgpr_read v47, a23 // (op 13500) 
v_mul_f32 v48, v35, v44 // (op 13500) 
v_mul_f32 v49, v35, v45 // (op 13500) 
v_mul_f32 v50, v35, v46 // (op 13500) 
v_mul_f32 v51, v35, v47 // (op 13500) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13485) 
v_lshrrev_b32 v37, 16, v40 // (op 13485) 
v_lshlrev_b32 v36, 16, v36 // (op 13485) 
v_lshlrev_b32 v37, 16, v37 // (op 13485) 
v_mul_f32 v44, v34, v36 // (op 13485) 
v_mul_f32 v45, v34, v37 // (op 13485) 
v_and_b32 v36, 65535, v41 // (op 13485) 
v_lshrrev_b32 v37, 16, v41 // (op 13485) 
v_lshlrev_b32 v36, 16, v36 // (op 13485) 
v_lshlrev_b32 v37, 16, v37 // (op 13485) 
v_mul_f32 v46, v34, v36 // (op 13485) 
v_mul_f32 v47, v34, v37 // (op 13485) 


v_add_f32 v52, v44, v48 // (op 13486) 
v_add_f32 v53, v45, v49 // (op 13486) 
v_add_f32 v54, v46, v50 // (op 13486) 
v_add_f32 v55, v47, v51 // (op 13486) 


v_lshrrev_b32 v44, 16, v52 // (op 13487) 
v_lshrrev_b32 v45, 16, v53 // (op 13487) 
v_lshrrev_b32 v46, 16, v54 // (op 13487) 
v_lshrrev_b32 v47, 16, v55 // (op 13487) 


v_pack_b32_f16 v36, v44, v45 // (op 13476) 
v_pack_b32_f16 v37, v46, v47 // (op 13476) 

s_lshl_b32 s52, s24, 6 // (op 13476) 
s_add_u32 s52, s52, 64 // (op 13476) 
v_add_u32 v39, v43, s52 // (op 13476) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13476) Store value





s_lshl_b32 s52, s4, 7 // (op 13493) 
s_add_u32 s52, s52, 64 // (op 13493) 
v_add_u32 v36, v38, s52 // (op 13493) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13493) Load value

v_accvgpr_read v44, a24 // (op 13502) 
v_accvgpr_read v45, a25 // (op 13502) 
v_accvgpr_read v46, a26 // (op 13502) 
v_accvgpr_read v47, a27 // (op 13502) 
v_mul_f32 v48, v35, v44 // (op 13502) 
v_mul_f32 v49, v35, v45 // (op 13502) 
v_mul_f32 v50, v35, v46 // (op 13502) 
v_mul_f32 v51, v35, v47 // (op 13502) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13490) 
v_lshrrev_b32 v37, 16, v40 // (op 13490) 
v_lshlrev_b32 v36, 16, v36 // (op 13490) 
v_lshlrev_b32 v37, 16, v37 // (op 13490) 
v_mul_f32 v44, v34, v36 // (op 13490) 
v_mul_f32 v45, v34, v37 // (op 13490) 
v_and_b32 v36, 65535, v41 // (op 13490) 
v_lshrrev_b32 v37, 16, v41 // (op 13490) 
v_lshlrev_b32 v36, 16, v36 // (op 13490) 
v_lshlrev_b32 v37, 16, v37 // (op 13490) 
v_mul_f32 v46, v34, v36 // (op 13490) 
v_mul_f32 v47, v34, v37 // (op 13490) 


v_add_f32 v52, v44, v48 // (op 13491) 
v_add_f32 v53, v45, v49 // (op 13491) 
v_add_f32 v54, v46, v50 // (op 13491) 
v_add_f32 v55, v47, v51 // (op 13491) 


v_lshrrev_b32 v44, 16, v52 // (op 13492) 
v_lshrrev_b32 v45, 16, v53 // (op 13492) 
v_lshrrev_b32 v46, 16, v54 // (op 13492) 
v_lshrrev_b32 v47, 16, v55 // (op 13492) 


v_pack_b32_f16 v36, v44, v45 // (op 13478) 
v_pack_b32_f16 v37, v46, v47 // (op 13478) 

s_lshl_b32 s52, s24, 7 // (op 13478) 
s_add_u32 s52, s52, 64 // (op 13478) 
v_add_u32 v39, v43, s52 // (op 13478) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13478) Store value





s_lshl_b32 s52, s4, 6 // (op 13498) 
s_mul_i32 s53, 3, s52 // (op 13498) 
s_add_u32 s53, s53, 64 // (op 13498) 

v_add_u32 v36, v38, s53 // (op 13498) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13498) Load value

v_accvgpr_read v44, a28 // (op 13504) 
v_accvgpr_read v45, a29 // (op 13504) 
v_accvgpr_read v46, a30 // (op 13504) 
v_accvgpr_read v47, a31 // (op 13504) 
v_mul_f32 v48, v35, v44 // (op 13504) 
v_mul_f32 v49, v35, v45 // (op 13504) 
v_mul_f32 v50, v35, v46 // (op 13504) 
v_mul_f32 v51, v35, v47 // (op 13504) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13495) 
v_lshrrev_b32 v37, 16, v40 // (op 13495) 
v_lshlrev_b32 v36, 16, v36 // (op 13495) 
v_lshlrev_b32 v37, 16, v37 // (op 13495) 
v_mul_f32 v44, v34, v36 // (op 13495) 
v_mul_f32 v45, v34, v37 // (op 13495) 
v_and_b32 v36, 65535, v41 // (op 13495) 
v_lshrrev_b32 v37, 16, v41 // (op 13495) 
v_lshlrev_b32 v36, 16, v36 // (op 13495) 
v_lshlrev_b32 v37, 16, v37 // (op 13495) 
v_mul_f32 v46, v34, v36 // (op 13495) 
v_mul_f32 v47, v34, v37 // (op 13495) 


v_add_f32 v52, v44, v48 // (op 13496) 
v_add_f32 v53, v45, v49 // (op 13496) 
v_add_f32 v54, v46, v50 // (op 13496) 
v_add_f32 v55, v47, v51 // (op 13496) 


v_lshrrev_b32 v44, 16, v52 // (op 13497) 
v_lshrrev_b32 v45, 16, v53 // (op 13497) 
v_lshrrev_b32 v46, 16, v54 // (op 13497) 
v_lshrrev_b32 v47, 16, v55 // (op 13497) 


v_pack_b32_f16 v36, v44, v45 // (op 13480) 
v_pack_b32_f16 v37, v46, v47 // (op 13480) 

s_lshl_b32 s52, s24, 6 // (op 13480) 
s_mul_i32 s53, 3, s52 // (op 13480) 
s_add_u32 s53, s53, 64 // (op 13480) 

v_add_u32 v39, v43, s53 // (op 13480) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13480) Store value





v_mov_b32 v36, v38 // (op 148) reg expression

buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v44, a0 // (op 144) 
v_accvgpr_read v45, a1 // (op 144) 
v_accvgpr_read v46, a2 // (op 144) 
v_accvgpr_read v47, a3 // (op 144) 
v_mul_f32 v48, v35, v44 // (op 144) 
v_mul_f32 v49, v35, v45 // (op 144) 
v_mul_f32 v50, v35, v46 // (op 144) 
v_mul_f32 v51, v35, v47 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 149) 
v_lshrrev_b32 v37, 16, v40 // (op 149) 
v_lshlrev_b32 v36, 16, v36 // (op 149) 
v_lshlrev_b32 v37, 16, v37 // (op 149) 
v_mul_f32 v44, v34, v36 // (op 149) 
v_mul_f32 v45, v34, v37 // (op 149) 
v_and_b32 v36, 65535, v41 // (op 149) 
v_lshrrev_b32 v37, 16, v41 // (op 149) 
v_lshlrev_b32 v36, 16, v36 // (op 149) 
v_lshlrev_b32 v37, 16, v37 // (op 149) 
v_mul_f32 v46, v34, v36 // (op 149) 
v_mul_f32 v47, v34, v37 // (op 149) 


v_add_f32 v52, v44, v48 // (op 145) 
v_add_f32 v53, v45, v49 // (op 145) 
v_add_f32 v54, v46, v50 // (op 145) 
v_add_f32 v55, v47, v51 // (op 145) 


v_lshrrev_b32 v44, 16, v52 // (op 146) 
v_lshrrev_b32 v45, 16, v53 // (op 146) 
v_lshrrev_b32 v46, 16, v54 // (op 146) 
v_lshrrev_b32 v47, 16, v55 // (op 146) 


v_pack_b32_f16 v36, v44, v45 // (op 147) 
v_pack_b32_f16 v37, v46, v47 // (op 147) 

v_mov_b32 v39, v43 // (op 147) reg expression

buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s52, s4, 6 // (op 13391) 
v_add_u32 v36, v38, s52 // (op 13391) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13391) Load value

v_accvgpr_read v44, a4 // (op 13387) 
v_accvgpr_read v45, a5 // (op 13387) 
v_accvgpr_read v46, a6 // (op 13387) 
v_accvgpr_read v47, a7 // (op 13387) 
v_mul_f32 v48, v35, v44 // (op 13387) 
v_mul_f32 v49, v35, v45 // (op 13387) 
v_mul_f32 v50, v35, v46 // (op 13387) 
v_mul_f32 v51, v35, v47 // (op 13387) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13392) 
v_lshrrev_b32 v37, 16, v40 // (op 13392) 
v_lshlrev_b32 v36, 16, v36 // (op 13392) 
v_lshlrev_b32 v37, 16, v37 // (op 13392) 
v_mul_f32 v44, v34, v36 // (op 13392) 
v_mul_f32 v45, v34, v37 // (op 13392) 
v_and_b32 v36, 65535, v41 // (op 13392) 
v_lshrrev_b32 v37, 16, v41 // (op 13392) 
v_lshlrev_b32 v36, 16, v36 // (op 13392) 
v_lshlrev_b32 v37, 16, v37 // (op 13392) 
v_mul_f32 v46, v34, v36 // (op 13392) 
v_mul_f32 v47, v34, v37 // (op 13392) 


v_add_f32 v52, v44, v48 // (op 13388) 
v_add_f32 v53, v45, v49 // (op 13388) 
v_add_f32 v54, v46, v50 // (op 13388) 
v_add_f32 v55, v47, v51 // (op 13388) 


v_lshrrev_b32 v44, 16, v52 // (op 13389) 
v_lshrrev_b32 v45, 16, v53 // (op 13389) 
v_lshrrev_b32 v46, 16, v54 // (op 13389) 
v_lshrrev_b32 v47, 16, v55 // (op 13389) 


v_pack_b32_f16 v36, v44, v45 // (op 13390) 
v_pack_b32_f16 v37, v46, v47 // (op 13390) 

s_lshl_b32 s52, s24, 6 // (op 13390) 
v_add_u32 v39, v43, s52 // (op 13390) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13390) Store value





s_lshl_b32 s52, s4, 7 // (op 13405) 
v_add_u32 v36, v38, s52 // (op 13405) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13405) Load value

v_accvgpr_read v44, a8 // (op 13401) 
v_accvgpr_read v45, a9 // (op 13401) 
v_accvgpr_read v46, a10 // (op 13401) 
v_accvgpr_read v47, a11 // (op 13401) 
v_mul_f32 v48, v35, v44 // (op 13401) 
v_mul_f32 v49, v35, v45 // (op 13401) 
v_mul_f32 v50, v35, v46 // (op 13401) 
v_mul_f32 v51, v35, v47 // (op 13401) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13406) 
v_lshrrev_b32 v37, 16, v40 // (op 13406) 
v_lshlrev_b32 v36, 16, v36 // (op 13406) 
v_lshlrev_b32 v37, 16, v37 // (op 13406) 
v_mul_f32 v44, v34, v36 // (op 13406) 
v_mul_f32 v45, v34, v37 // (op 13406) 
v_and_b32 v36, 65535, v41 // (op 13406) 
v_lshrrev_b32 v37, 16, v41 // (op 13406) 
v_lshlrev_b32 v36, 16, v36 // (op 13406) 
v_lshlrev_b32 v37, 16, v37 // (op 13406) 
v_mul_f32 v46, v34, v36 // (op 13406) 
v_mul_f32 v47, v34, v37 // (op 13406) 


v_add_f32 v52, v44, v48 // (op 13402) 
v_add_f32 v53, v45, v49 // (op 13402) 
v_add_f32 v54, v46, v50 // (op 13402) 
v_add_f32 v55, v47, v51 // (op 13402) 


v_lshrrev_b32 v44, 16, v52 // (op 13403) 
v_lshrrev_b32 v45, 16, v53 // (op 13403) 
v_lshrrev_b32 v46, 16, v54 // (op 13403) 
v_lshrrev_b32 v47, 16, v55 // (op 13403) 


v_pack_b32_f16 v36, v44, v45 // (op 13404) 
v_pack_b32_f16 v37, v46, v47 // (op 13404) 

s_lshl_b32 s52, s24, 7 // (op 13404) 
v_add_u32 v39, v43, s52 // (op 13404) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13404) Store value





s_lshl_b32 s52, s4, 6 // (op 13419) 
s_mul_i32 s53, 3, s52 // (op 13419) 

v_add_u32 v36, v38, s53 // (op 13419) 


buffer_load_dwordx2 v[40:41], v36, s[56:59], 0 offen // (op 13419) Load value

v_accvgpr_read v44, a12 // (op 13415) 
v_accvgpr_read v45, a13 // (op 13415) 
v_accvgpr_read v46, a14 // (op 13415) 
v_accvgpr_read v47, a15 // (op 13415) 
v_mul_f32 v48, v35, v44 // (op 13415) 
v_mul_f32 v49, v35, v45 // (op 13415) 
v_mul_f32 v50, v35, v46 // (op 13415) 
v_mul_f32 v51, v35, v47 // (op 13415) 

s_waitcnt vmcnt(0)
v_and_b32 v36, 65535, v40 // (op 13420) 
v_lshrrev_b32 v37, 16, v40 // (op 13420) 
v_lshlrev_b32 v36, 16, v36 // (op 13420) 
v_lshlrev_b32 v37, 16, v37 // (op 13420) 
v_mul_f32 v44, v34, v36 // (op 13420) 
v_mul_f32 v45, v34, v37 // (op 13420) 
v_and_b32 v36, 65535, v41 // (op 13420) 
v_lshrrev_b32 v37, 16, v41 // (op 13420) 
v_lshlrev_b32 v36, 16, v36 // (op 13420) 
v_lshlrev_b32 v37, 16, v37 // (op 13420) 
v_mul_f32 v46, v34, v36 // (op 13420) 
v_mul_f32 v47, v34, v37 // (op 13420) 


v_add_f32 v52, v44, v48 // (op 13416) 
v_add_f32 v53, v45, v49 // (op 13416) 
v_add_f32 v54, v46, v50 // (op 13416) 
v_add_f32 v55, v47, v51 // (op 13416) 


v_lshrrev_b32 v44, 16, v52 // (op 13417) 
v_lshrrev_b32 v45, 16, v53 // (op 13417) 
v_lshrrev_b32 v46, 16, v54 // (op 13417) 
v_lshrrev_b32 v47, 16, v55 // (op 13417) 


v_pack_b32_f16 v36, v44, v45 // (op 13418) 
v_pack_b32_f16 v37, v46, v47 // (op 13418) 

s_lshl_b32 s52, s24, 6 // (op 13418) 
s_mul_i32 s53, 3, s52 // (op 13418) 

v_add_u32 v39, v43, s53 // (op 13418) 


buffer_store_dwordx2 v[36:37], v39, s[60:63], 0 offen // (op 13418) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_381:


s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_294:

























s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s0, 0 // (op 450) call()
s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 28132) 
s_mul_i32 s15, s16, s50 // (op 28132) 
s_add_u32 s15, s15, s0 // (op 28132) 
v_and_b32 v3, 63, v1 // (op 28132) 
s_add_u32 s6, s15, s9 // (op 28132) 

v_lshrrev_b32 v4, 4, v3 // (op 28132) 
s_mul_hi_u32 s15, s6, s13 // (op 28132) 
v_lshlrev_b32 v5, 2, v4 // (op 28132) 

s_sub_u32 s7, s6, s15 // (op 28132) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 28132) 


s_lshr_b32 s8, s7, 1 // (op 28132) 

v_lshrrev_b32 v0, 2, v3 // (op 28132) 

s_add_u32 s7, s8, s15 // (op 28132) 


v_and_b32 v3, 3, v0 // (op 28132) 

s_lshr_b32 s8, s7, s11 // (op 28132) 

v_and_b32 v0, 3, v1 // (op 28132) 
s_and_b32 s7, s11, 2147483648 // (op 28132) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 28132) 


s_cmp_eq_i32 s7, 0 // (op 28132) 
s_mov_b32 s15, scc // (op 28132) 

v_lshrrev_b32 v0, 6, v1 // (op 28132) 
s_cmp_eq_i32 s15, 0 // (op 28132) 
s_cselect_b32 s7, s6, s8 // (op 28132) 



v_and_b32 v3, 1, v0 // (op 28132) 

s_mul_hi_u32 s6, s7, s12 // (op 28132) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 28132) 


s_sub_u32 s8, s7, s6 // (op 28132) 
s_lshr_b32 s15, s8, 1 // (op 28132) 

s_add_u32 s8, s15, s6 // (op 28132) 


s_lshr_b32 s6, s8, s10 // (op 28132) 

s_and_b32 s8, s10, 2147483648 // (op 28132) 
s_cmp_eq_i32 s8, 0 // (op 28132) 
s_mov_b32 s15, scc // (op 28132) 

s_cmp_eq_i32 s15, 0 // (op 28132) 
s_cselect_b32 s8, s7, s6 // (op 28132) 


s_mul_i32 s6, s8, s14 // (op 28132) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 28132) 


s_sub_u32 s8, s7, s6 // (op 28132) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 28132) 


v_mul_lo_u32 v0, v4, s4 // (op 28132) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 28132) 


s_mul_i32 s19, s4, s23 // (op 28134) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 28134) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 28134) least significant
s_mul_hi_u32 s6, s4, s22 // (op 28134) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28134) most significant: sum
s_add_u32 s19, s19, s6 // (op 28134) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 28134) 

s_mov_b32 s56, s20 // (op 28134) reg expression
s_mov_b32 s57, s21 // (op 28134) reg expression
s_mov_b32 s58, s28 // (op 28134) reg expression
s_mov_b32 s59, 131072 // (op 28134) call()

v_lshrrev_b32 v0, 7, v1 // (op 28486) 
s_mul_i32 s15, s16, s50 // (op 28486) 
s_add_u32 s15, s15, s0 // (op 28486) 
v_and_b32 v3, 63, v1 // (op 28486) 
s_add_u32 s6, s15, s9 // (op 28486) 

v_lshrrev_b32 v5, 4, v3 // (op 28486) 
s_mul_hi_u32 s15, s6, s13 // (op 28486) 
v_lshlrev_b32 v22, 2, v5 // (op 28486) 

s_sub_u32 s7, s6, s15 // (op 28486) 
v_lshl_add_u32 v5, v0, 4, v22 // (op 28486) 


s_lshr_b32 s8, s7, 1 // (op 28486) 

v_lshrrev_b32 v0, 2, v3 // (op 28486) 

s_add_u32 s7, s8, s15 // (op 28486) 


v_and_b32 v3, 3, v0 // (op 28486) 

s_lshr_b32 s8, s7, s11 // (op 28486) 

v_and_b32 v0, 3, v1 // (op 28486) 
s_and_b32 s7, s11, 2147483648 // (op 28486) 
v_lshl_add_u32 v22, v3, 2, v0 // (op 28486) 


s_cmp_eq_i32 s7, 0 // (op 28486) 
s_mov_b32 s15, scc // (op 28486) 

v_lshrrev_b32 v0, 6, v1 // (op 28486) 
s_cmp_eq_i32 s15, 0 // (op 28486) 
s_cselect_b32 s7, s6, s8 // (op 28486) 



v_and_b32 v3, 1, v0 // (op 28486) 

s_mul_hi_u32 s6, s7, s12 // (op 28486) 
v_lshl_add_u32 v0, v3, 4, v22 // (op 28486) 


s_sub_u32 s8, s7, s6 // (op 28486) 
s_lshr_b32 s15, s8, 1 // (op 28486) 

s_add_u32 s8, s15, s6 // (op 28486) 


s_lshr_b32 s6, s8, s10 // (op 28486) 

s_and_b32 s8, s10, 2147483648 // (op 28486) 
s_cmp_eq_i32 s8, 0 // (op 28486) 
s_mov_b32 s15, scc // (op 28486) 

s_cmp_eq_i32 s15, 0 // (op 28486) 
s_cselect_b32 s8, s7, s6 // (op 28486) 


s_mul_i32 s6, s8, s14 // (op 28486) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 28486) 


s_sub_u32 s8, s7, s6 // (op 28486) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 28486) 


v_mul_lo_u32 v0, v5, s24 // (op 28486) 

v_add_lshl_u32 v5, v3, v0, 1 // (op 28486) 


s_mul_i32 s19, s24, s23 // (op 28488) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 28488) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 28488) least significant
s_mul_hi_u32 s6, s24, s22 // (op 28488) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28488) most significant: sum
s_add_u32 s19, s19, s6 // (op 28488) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 28488) 

s_mov_b32 s60, s26 // (op 28488) reg expression
s_mov_b32 s61, s27 // (op 28488) reg expression
s_mov_b32 s62, s28 // (op 28488) reg expression
s_mov_b32 s63, 131072 // (op 28488) call()

v_and_b32 v0, 63, v1 // (op 28565) 
v_and_b32 v3, 48, v1 // (op 28565) 
v_sub_u32 v22, v0, v3 // (op 28565) 

v_lshrrev_b32 v3, 4, v0 // (op 28565) 

v_lshl_add_u32 v0, v3, 5, v22 // (op 28565) 


v_lshrrev_b32 v3, 7, v1 // (op 28565) 
v_lshlrev_b32 v22, 4, v3 // (op 28565) 

v_lshl_add_u32 v3, v0, 7, v22 // (op 28565) 


v_lshlrev_b32 v0, 2, v3 // (op 28565) 

v_lshrrev_b32 v3, 3, v0 // (op 28565) 

v_and_b32 v0, 63, v1 // (op 28640) 
v_and_b32 v22, 48, v1 // (op 28640) 
v_sub_u32 v23, v0, v22 // (op 28640) 

v_lshrrev_b32 v22, 4, v0 // (op 28640) 

v_lshl_add_u32 v0, v22, 5, v23 // (op 28640) 


v_lshrrev_b32 v22, 6, v1 // (op 28640) 
v_and_b32 v23, 1, v22 // (op 28640) 

v_lshlrev_b32 v22, 4, v23 // (op 28640) 

v_lshl_add_u32 v23, v0, 7, v22 // (op 28640) 


v_lshlrev_b32 v0, 2, v23 // (op 28640) 

v_lshrrev_b32 v22, 3, v0 // (op 28640) 

v_lshrrev_b32 v0, 7, v1 // (op 28716) 
v_and_b32 v23, 15, v1 // (op 28716) 
v_lshl_add_u32 v24, v0, 4, v23 // (op 28716) 


v_and_b32 v0, 63, v1 // (op 28716) 
v_lshrrev_b32 v23, 4, v0 // (op 28716) 

v_lshl_add_u32 v0, v23, 7, v24 // (op 28716) 


v_lshrrev_b32 v23, 6, v1 // (op 28792) 
v_and_b32 v24, 1, v23 // (op 28792) 

v_and_b32 v23, 15, v1 // (op 28792) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 28792) 


v_and_b32 v23, 63, v1 // (op 28792) 
v_lshrrev_b32 v24, 4, v23 // (op 28792) 

v_lshl_add_u32 v23, v24, 7, v25 // (op 28792) 


v_accvgpr_write a0, 0.00000 // (op 184) call()
v_accvgpr_write a1, 0.00000 // (op 184) call()
v_accvgpr_write a2, 0.00000 // (op 184) call()
v_accvgpr_write a3, 0.00000 // (op 184) call()
v_accvgpr_write a4, 0.00000 // (op 768) call()
v_accvgpr_write a5, 0.00000 // (op 768) call()
v_accvgpr_write a6, 0.00000 // (op 768) call()
v_accvgpr_write a7, 0.00000 // (op 768) call()
v_accvgpr_write a8, 0.00000 // (op 1066) call()
v_accvgpr_write a9, 0.00000 // (op 1066) call()
v_accvgpr_write a10, 0.00000 // (op 1066) call()
v_accvgpr_write a11, 0.00000 // (op 1066) call()
v_accvgpr_write a12, 0.00000 // (op 1364) call()
v_accvgpr_write a13, 0.00000 // (op 1364) call()
v_accvgpr_write a14, 0.00000 // (op 1364) call()
v_accvgpr_write a15, 0.00000 // (op 1364) call()
v_accvgpr_write a16, 0.00000 // (op 1798) call()
v_accvgpr_write a17, 0.00000 // (op 1798) call()
v_accvgpr_write a18, 0.00000 // (op 1798) call()
v_accvgpr_write a19, 0.00000 // (op 1798) call()
v_accvgpr_write a20, 0.00000 // (op 2291) call()
v_accvgpr_write a21, 0.00000 // (op 2291) call()
v_accvgpr_write a22, 0.00000 // (op 2291) call()
v_accvgpr_write a23, 0.00000 // (op 2291) call()
v_accvgpr_write a24, 0.00000 // (op 2292) call()
v_accvgpr_write a25, 0.00000 // (op 2292) call()
v_accvgpr_write a26, 0.00000 // (op 2292) call()
v_accvgpr_write a27, 0.00000 // (op 2292) call()
v_accvgpr_write a28, 0.00000 // (op 2293) call()
v_accvgpr_write a29, 0.00000 // (op 2293) call()
v_accvgpr_write a30, 0.00000 // (op 2293) call()
v_accvgpr_write a31, 0.00000 // (op 2293) call()
v_accvgpr_write a32, 0.00000 // (op 3088) call()
v_accvgpr_write a33, 0.00000 // (op 3088) call()
v_accvgpr_write a34, 0.00000 // (op 3088) call()
v_accvgpr_write a35, 0.00000 // (op 3088) call()
v_accvgpr_write a36, 0.00000 // (op 3581) call()
v_accvgpr_write a37, 0.00000 // (op 3581) call()
v_accvgpr_write a38, 0.00000 // (op 3581) call()
v_accvgpr_write a39, 0.00000 // (op 3581) call()
v_accvgpr_write a40, 0.00000 // (op 3582) call()
v_accvgpr_write a41, 0.00000 // (op 3582) call()
v_accvgpr_write a42, 0.00000 // (op 3582) call()
v_accvgpr_write a43, 0.00000 // (op 3582) call()
v_accvgpr_write a44, 0.00000 // (op 3583) call()
v_accvgpr_write a45, 0.00000 // (op 3583) call()
v_accvgpr_write a46, 0.00000 // (op 3583) call()
v_accvgpr_write a47, 0.00000 // (op 3583) call()
v_accvgpr_write a48, 0.00000 // (op 4378) call()
v_accvgpr_write a49, 0.00000 // (op 4378) call()
v_accvgpr_write a50, 0.00000 // (op 4378) call()
v_accvgpr_write a51, 0.00000 // (op 4378) call()
v_accvgpr_write a52, 0.00000 // (op 4871) call()
v_accvgpr_write a53, 0.00000 // (op 4871) call()
v_accvgpr_write a54, 0.00000 // (op 4871) call()
v_accvgpr_write a55, 0.00000 // (op 4871) call()
v_accvgpr_write a56, 0.00000 // (op 4872) call()
v_accvgpr_write a57, 0.00000 // (op 4872) call()
v_accvgpr_write a58, 0.00000 // (op 4872) call()
v_accvgpr_write a59, 0.00000 // (op 4872) call()
v_accvgpr_write a60, 0.00000 // (op 4873) call()
v_accvgpr_write a61, 0.00000 // (op 4873) call()
v_accvgpr_write a62, 0.00000 // (op 4873) call()
v_accvgpr_write a63, 0.00000 // (op 4873) call()
v_and_b32 v24, -4, v1 // (op 27612) 
s_mul_i32 s15, s16, s50 // (op 27612) 
s_add_u32 s15, s15, s0 // (op 27612) 
v_sub_u32 v25, v1, v24 // (op 27612) 

s_add_u32 s6, s15, s9 // (op 27612) 

v_lshlrev_b32 v24, 5, v25 // (op 27612) 

s_mul_hi_u32 s15, s6, s13 // (op 27612) 
v_lshrrev_b32 v25, 2, v1 // (op 27612) 
s_sub_u32 s7, s6, s15 // (op 27612) 
s_lshr_b32 s8, s7, 1 // (op 27612) 

s_add_u32 s7, s8, s15 // (op 27612) 


s_lshr_b32 s8, s7, s11 // (op 27612) 

s_and_b32 s7, s11, 2147483648 // (op 27612) 
s_cmp_eq_i32 s7, 0 // (op 27612) 
s_mov_b32 s15, scc // (op 27612) 

s_cmp_eq_i32 s15, 0 // (op 27612) 
s_cselect_b32 s7, s6, s8 // (op 27612) 


s_mul_hi_u32 s8, s7, s12 // (op 27612) 
s_sub_u32 s15, s7, s8 // (op 27612) 
s_lshr_b32 s18, s15, 1 // (op 27612) 

s_add_u32 s15, s18, s8 // (op 27612) 


s_lshr_b32 s8, s15, s10 // (op 27612) 

s_and_b32 s15, s10, 2147483648 // (op 27612) 
s_cmp_eq_i32 s15, 0 // (op 27612) 
s_mov_b32 s18, scc // (op 27612) 

s_cmp_eq_i32 s18, 0 // (op 27612) 
s_cselect_b32 s15, s7, s8 // (op 27612) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27612) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27612) 
v_mov_b32 v33, 0 // (op 27612) convert
v_mov_b32 v32, v26 // (op 27612) convert

s_mul_i32 s8, s7, s18 // (op 27612) 


s_sub_u32 s7, s6, s8 // (op 27612) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27612) 


v_mul_lo_u32 v31, v24, s31 // (op 27612) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27612) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 27612) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 27612) most significant: sum


v_add_co_u32 v24, s[18:19], v32, v30 // (op 27612) least significant half
v_addc_co_u32 v25, s[18:19], v33, v31, s[18:19] // (op 27612) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27612) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 27612) 

s_mul_i32 s19, s30, s3 // (op 27614) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27614) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27614) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27614) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27614) most significant: sum
s_add_u32 s19, s19, s6 // (op 27614) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27614) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27614) 

s_mov_b32 s68, s32 // (op 27614) reg expression
s_mov_b32 s69, s33 // (op 27614) reg expression
s_mov_b32 s70, s18 // (op 27614) reg expression
s_mov_b32 s71, 131072 // (op 27614) call()

v_and_b32 v24, -4, v1 // (op 27631) 
s_mul_i32 s15, s16, s50 // (op 27631) 
s_add_u32 s15, s15, s0 // (op 27631) 
v_sub_u32 v25, v1, v24 // (op 27631) 

s_add_u32 s6, s15, s9 // (op 27631) 

v_lshlrev_b32 v24, 5, v25 // (op 27631) 

s_mul_hi_u32 s15, s6, s13 // (op 27631) 
v_lshrrev_b32 v25, 2, v1 // (op 27631) 
s_sub_u32 s7, s6, s15 // (op 27631) 
s_lshr_b32 s8, s7, 1 // (op 27631) 

s_add_u32 s7, s8, s15 // (op 27631) 


s_lshr_b32 s8, s7, s11 // (op 27631) 

s_and_b32 s7, s11, 2147483648 // (op 27631) 
s_cmp_eq_i32 s7, 0 // (op 27631) 
s_mov_b32 s15, scc // (op 27631) 

s_cmp_eq_i32 s15, 0 // (op 27631) 
s_cselect_b32 s7, s6, s8 // (op 27631) 


s_mul_hi_u32 s8, s7, s12 // (op 27631) 
s_sub_u32 s15, s7, s8 // (op 27631) 
s_lshr_b32 s18, s15, 1 // (op 27631) 

s_add_u32 s15, s18, s8 // (op 27631) 


s_lshr_b32 s8, s15, s10 // (op 27631) 

s_and_b32 s15, s10, 2147483648 // (op 27631) 
s_cmp_eq_i32 s15, 0 // (op 27631) 
s_mov_b32 s18, scc // (op 27631) 

s_cmp_eq_i32 s18, 0 // (op 27631) 
s_cselect_b32 s15, s7, s8 // (op 27631) 


v_lshl_add_u32 v28, s15, 7, v24 // (op 27631) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27631) 
v_mov_b32 v33, 0 // (op 27631) convert
v_mov_b32 v32, v28 // (op 27631) convert

s_mul_i32 s8, s7, s18 // (op 27631) 


s_sub_u32 s7, s6, s8 // (op 27631) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27631) 


v_mul_lo_u32 v29, v24, s31 // (op 27631) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27631) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 27631) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 27631) most significant: sum


v_add_co_u32 v24, s[18:19], v32, v28 // (op 27631) least significant half
v_addc_co_u32 v25, s[18:19], v33, v29, s[18:19] // (op 27631) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27631) 


v_lshrrev_b64 v[28:29], 3, v[24:25] // (op 27631) 

s_mul_i32 s19, s30, s3 // (op 27633) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27633) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27633) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27633) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27633) most significant: sum
s_add_u32 s19, s19, s6 // (op 27633) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27633) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27633) 

s_mov_b32 s68, s32 // (op 27633) reg expression
s_mov_b32 s69, s33 // (op 27633) reg expression
s_mov_b32 s70, s18 // (op 27633) reg expression
s_mov_b32 s71, 131072 // (op 27633) call()

v_and_b32 v24, -4, v1 // (op 27758) 
s_mul_i32 s15, s16, s50 // (op 27758) 
s_add_u32 s15, s15, s0 // (op 27758) 
v_sub_u32 v25, v1, v24 // (op 27758) 

s_add_u32 s6, s15, s9 // (op 27758) 

v_lshlrev_b32 v24, 5, v25 // (op 27758) 

s_mul_hi_u32 s15, s6, s13 // (op 27758) 
v_lshrrev_b32 v25, 2, v1 // (op 27758) 
s_sub_u32 s7, s6, s15 // (op 27758) 
s_lshr_b32 s8, s7, 1 // (op 27758) 

s_add_u32 s7, s8, s15 // (op 27758) 


s_lshr_b32 s8, s7, s11 // (op 27758) 

s_and_b32 s7, s11, 2147483648 // (op 27758) 
s_cmp_eq_i32 s7, 0 // (op 27758) 
s_mov_b32 s15, scc // (op 27758) 

s_cmp_eq_i32 s15, 0 // (op 27758) 
s_cselect_b32 s7, s6, s8 // (op 27758) 


s_mul_hi_u32 s8, s7, s12 // (op 27758) 
s_sub_u32 s15, s7, s8 // (op 27758) 
s_lshr_b32 s18, s15, 1 // (op 27758) 

s_add_u32 s15, s18, s8 // (op 27758) 


s_lshr_b32 s8, s15, s10 // (op 27758) 

s_and_b32 s15, s10, 2147483648 // (op 27758) 
s_cmp_eq_i32 s15, 0 // (op 27758) 
s_mov_b32 s18, scc // (op 27758) 

s_cmp_eq_i32 s18, 0 // (op 27758) 
s_cselect_b32 s15, s7, s8 // (op 27758) 


s_mul_i32 s8, s15, s14 // (op 27758) 

s_sub_u32 s15, s7, s8 // (op 27758) 

v_lshl_add_u32 v30, s15, 7, v24 // (op 27758) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27758) 
v_mov_b32 v33, 0 // (op 27758) convert
v_mov_b32 v32, v30 // (op 27758) convert

s_mul_i32 s8, s7, s18 // (op 27758) 


s_sub_u32 s7, s6, s8 // (op 27758) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27758) 


v_mul_lo_u32 v31, v24, s35 // (op 27758) most significant: low of low * high
v_mul_hi_u32 v25, v24, s34 // (op 27758) most significant: high of low * low
v_mul_lo_u32 v30, v24, s34 // (op 27758) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 27758) most significant: sum

v_add_co_u32 v30, s[18:19], v30, v32 // (op 27758) least significant half
v_addc_co_u32 v31, s[18:19], v31, v33, s[18:19] // (op 27758) most significant half



v_lshlrev_b64 v[24:25], 2, v[30:31] // (op 27758) 

v_lshrrev_b64 v[30:31], 3, v[24:25] // (op 27758) 

s_mul_i32 s19, s34, s3 // (op 27760) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 27760) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 27760) least significant
s_mul_hi_u32 s6, s34, s2 // (op 27760) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27760) most significant: sum
s_add_u32 s19, s19, s6 // (op 27760) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27760) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27760) 

s_mov_b32 s72, s36 // (op 27760) reg expression
s_mov_b32 s73, s37 // (op 27760) reg expression
s_mov_b32 s74, s18 // (op 27760) reg expression
s_mov_b32 s75, 131072 // (op 27760) call()

v_and_b32 v24, -4, v1 // (op 27777) 
s_mul_i32 s15, s16, s50 // (op 27777) 
s_add_u32 s15, s15, s0 // (op 27777) 
v_sub_u32 v25, v1, v24 // (op 27777) 

s_add_u32 s6, s15, s9 // (op 27777) 

v_lshlrev_b32 v24, 5, v25 // (op 27777) 

s_mul_hi_u32 s15, s6, s13 // (op 27777) 
v_lshrrev_b32 v25, 2, v1 // (op 27777) 
s_sub_u32 s7, s6, s15 // (op 27777) 
s_lshr_b32 s8, s7, 1 // (op 27777) 

s_add_u32 s7, s8, s15 // (op 27777) 


s_lshr_b32 s8, s7, s11 // (op 27777) 

s_and_b32 s7, s11, 2147483648 // (op 27777) 
s_cmp_eq_i32 s7, 0 // (op 27777) 
s_mov_b32 s15, scc // (op 27777) 

s_cmp_eq_i32 s15, 0 // (op 27777) 
s_cselect_b32 s7, s6, s8 // (op 27777) 


s_mul_hi_u32 s8, s7, s12 // (op 27777) 
s_sub_u32 s15, s7, s8 // (op 27777) 
s_lshr_b32 s18, s15, 1 // (op 27777) 

s_add_u32 s15, s18, s8 // (op 27777) 


s_lshr_b32 s8, s15, s10 // (op 27777) 

s_and_b32 s15, s10, 2147483648 // (op 27777) 
s_cmp_eq_i32 s15, 0 // (op 27777) 
s_mov_b32 s18, scc // (op 27777) 

s_cmp_eq_i32 s18, 0 // (op 27777) 
s_cselect_b32 s15, s7, s8 // (op 27777) 


s_mul_i32 s8, s15, s14 // (op 27777) 

s_sub_u32 s15, s7, s8 // (op 27777) 

v_lshl_add_u32 v32, s15, 7, v24 // (op 27777) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27777) 
v_mov_b32 v37, 0 // (op 27777) convert
v_mov_b32 v36, v32 // (op 27777) convert

s_mul_i32 s8, s7, s18 // (op 27777) 


s_sub_u32 s7, s6, s8 // (op 27777) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27777) 


v_mul_lo_u32 v33, v24, s35 // (op 27777) most significant: low of low * high
v_mul_hi_u32 v25, v24, s34 // (op 27777) most significant: high of low * low
v_mul_lo_u32 v32, v24, s34 // (op 27777) least significant: low of low * low
v_add_u32 v33, v33, v25 // (op 27777) most significant: sum

v_add_co_u32 v32, s[18:19], v32, v36 // (op 27777) least significant half
v_addc_co_u32 v33, s[18:19], v33, v37, s[18:19] // (op 27777) most significant half



v_lshlrev_b64 v[24:25], 2, v[32:33] // (op 27777) 

v_lshrrev_b64 v[32:33], 3, v[24:25] // (op 27777) 

s_mul_i32 s19, s34, s3 // (op 27779) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 27779) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 27779) least significant
s_mul_hi_u32 s6, s34, s2 // (op 27779) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27779) most significant: sum
s_add_u32 s19, s19, s6 // (op 27779) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27779) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27779) 

s_mov_b32 s72, s36 // (op 27779) reg expression
s_mov_b32 s73, s37 // (op 27779) reg expression
s_mov_b32 s74, s18 // (op 27779) reg expression
s_mov_b32 s75, 131072 // (op 27779) call()

v_and_b32 v24, -128, v1 // (op 27886) 
s_mul_i32 s15, s16, s50 // (op 27886) 
s_add_u32 s15, s15, s0 // (op 27886) 
v_sub_u32 v25, v1, v24 // (op 27886) 

s_add_u32 s6, s15, s9 // (op 27886) 

v_lshrrev_b32 v24, 7, v1 // (op 27886) 
s_mul_hi_u32 s15, s6, s13 // (op 27886) 
v_lshlrev_b32 v36, 1, v24 // (op 27886) 

s_sub_u32 s7, s6, s15 // (op 27886) 
s_lshr_b32 s8, s7, 1 // (op 27886) 

s_add_u32 s7, s8, s15 // (op 27886) 


s_lshr_b32 s8, s7, s11 // (op 27886) 

s_and_b32 s7, s11, 2147483648 // (op 27886) 
s_cmp_eq_i32 s7, 0 // (op 27886) 
s_mov_b32 s15, scc // (op 27886) 

s_cmp_eq_i32 s15, 0 // (op 27886) 
s_cselect_b32 s7, s6, s8 // (op 27886) 


s_mul_hi_u32 s8, s7, s12 // (op 27886) 
s_sub_u32 s15, s7, s8 // (op 27886) 
s_lshr_b32 s18, s15, 1 // (op 27886) 

s_add_u32 s15, s18, s8 // (op 27886) 


s_lshr_b32 s8, s15, s10 // (op 27886) 

s_and_b32 s15, s10, 2147483648 // (op 27886) 
s_cmp_eq_i32 s15, 0 // (op 27886) 
s_mov_b32 s18, scc // (op 27886) 

s_cmp_eq_i32 s18, 0 // (op 27886) 
s_cselect_b32 s15, s7, s8 // (op 27886) 


v_lshl_add_u32 v24, s15, 7, v25 // (op 27886) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27886) 
s_mul_i32 s8, s7, s18 // (op 27886) 


s_sub_u32 s7, s6, s8 // (op 27886) 


v_lshl_add_u32 v25, s7, 2, v36 // (op 27886) 


v_mul_lo_u32 v36, v25, s38 // (op 27886) 

v_add_u32 v25, v24, v36 // (op 27886) 


s_mul_i32 s19, s38, s43 // (op 27888) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27888) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27888) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27888) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27888) most significant: sum
s_add_u32 s19, s19, s6 // (op 27888) most significant: sum


s_mov_b32 s76, s40 // (op 27888) reg expression
s_mov_b32 s77, s41 // (op 27888) reg expression
s_mov_b32 s78, s18 // (op 27888) reg expression
s_mov_b32 s79, 131072 // (op 27888) call()

v_and_b32 v24, -128, v1 // (op 27905) 
s_mul_i32 s15, s16, s50 // (op 27905) 
s_add_u32 s15, s15, s0 // (op 27905) 
v_sub_u32 v36, v1, v24 // (op 27905) 

s_add_u32 s6, s15, s9 // (op 27905) 

v_lshrrev_b32 v24, 7, v1 // (op 27905) 
s_mul_hi_u32 s15, s6, s13 // (op 27905) 
v_lshlrev_b32 v37, 1, v24 // (op 27905) 

s_sub_u32 s7, s6, s15 // (op 27905) 
s_lshr_b32 s8, s7, 1 // (op 27905) 

s_add_u32 s7, s8, s15 // (op 27905) 


s_lshr_b32 s8, s7, s11 // (op 27905) 

s_and_b32 s7, s11, 2147483648 // (op 27905) 
s_cmp_eq_i32 s7, 0 // (op 27905) 
s_mov_b32 s15, scc // (op 27905) 

s_cmp_eq_i32 s15, 0 // (op 27905) 
s_cselect_b32 s7, s6, s8 // (op 27905) 


s_mul_hi_u32 s8, s7, s12 // (op 27905) 
s_sub_u32 s15, s7, s8 // (op 27905) 
s_lshr_b32 s18, s15, 1 // (op 27905) 

s_add_u32 s15, s18, s8 // (op 27905) 


s_lshr_b32 s8, s15, s10 // (op 27905) 

s_and_b32 s15, s10, 2147483648 // (op 27905) 
s_cmp_eq_i32 s15, 0 // (op 27905) 
s_mov_b32 s18, scc // (op 27905) 

s_cmp_eq_i32 s18, 0 // (op 27905) 
s_cselect_b32 s15, s7, s8 // (op 27905) 


v_lshl_add_u32 v24, s15, 7, v36 // (op 27905) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27905) 
s_mul_i32 s8, s7, s18 // (op 27905) 


s_sub_u32 s7, s6, s8 // (op 27905) 


v_lshl_add_u32 v36, s7, 2, v37 // (op 27905) 


v_mul_lo_u32 v37, v36, s38 // (op 27905) 

v_add_u32 v36, v24, v37 // (op 27905) 


s_mul_i32 s19, s38, s43 // (op 27907) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27907) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27907) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27907) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27907) most significant: sum
s_add_u32 s19, s19, s6 // (op 27907) most significant: sum


s_mov_b32 s76, s40 // (op 27907) reg expression
s_mov_b32 s77, s41 // (op 27907) reg expression
s_mov_b32 s78, s18 // (op 27907) reg expression
s_mov_b32 s79, 131072 // (op 27907) call()

v_and_b32 v24, -128, v1 // (op 28014) 
s_mul_i32 s15, s16, s50 // (op 28014) 
s_add_u32 s15, s15, s0 // (op 28014) 
v_sub_u32 v37, v1, v24 // (op 28014) 

s_add_u32 s6, s15, s9 // (op 28014) 

v_lshrrev_b32 v24, 7, v1 // (op 28014) 
s_mul_hi_u32 s15, s6, s13 // (op 28014) 
v_lshlrev_b32 v38, 1, v24 // (op 28014) 

s_sub_u32 s7, s6, s15 // (op 28014) 
s_lshr_b32 s8, s7, 1 // (op 28014) 

s_add_u32 s7, s8, s15 // (op 28014) 


s_lshr_b32 s8, s7, s11 // (op 28014) 

s_and_b32 s7, s11, 2147483648 // (op 28014) 
s_cmp_eq_i32 s7, 0 // (op 28014) 
s_mov_b32 s15, scc // (op 28014) 

s_cmp_eq_i32 s15, 0 // (op 28014) 
s_cselect_b32 s7, s6, s8 // (op 28014) 


s_mul_hi_u32 s8, s7, s12 // (op 28014) 
s_sub_u32 s15, s7, s8 // (op 28014) 
s_lshr_b32 s18, s15, 1 // (op 28014) 

s_add_u32 s15, s18, s8 // (op 28014) 


s_lshr_b32 s8, s15, s10 // (op 28014) 

s_and_b32 s15, s10, 2147483648 // (op 28014) 
s_cmp_eq_i32 s15, 0 // (op 28014) 
s_mov_b32 s18, scc // (op 28014) 

s_cmp_eq_i32 s18, 0 // (op 28014) 
s_cselect_b32 s15, s7, s8 // (op 28014) 


s_mul_i32 s8, s15, s14 // (op 28014) 

s_sub_u32 s15, s7, s8 // (op 28014) 

v_lshl_add_u32 v24, s15, 7, v37 // (op 28014) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 28014) 
s_mul_i32 s8, s7, s18 // (op 28014) 


s_sub_u32 s7, s6, s8 // (op 28014) 


v_lshl_add_u32 v37, s7, 2, v38 // (op 28014) 


v_mul_lo_u32 v38, v37, s44 // (op 28014) 
v_add_u32 v38, v38, v24 // (op 28014) 


s_mul_i32 s19, s44, s43 // (op 28016) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 28016) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 28016) least significant
s_mul_hi_u32 s6, s44, s42 // (op 28016) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28016) most significant: sum
s_add_u32 s19, s19, s6 // (op 28016) most significant: sum


s_mov_b32 s80, s46 // (op 28016) reg expression
s_mov_b32 s81, s47 // (op 28016) reg expression
s_mov_b32 s82, s18 // (op 28016) reg expression
s_mov_b32 s83, 131072 // (op 28016) call()

v_and_b32 v24, -128, v1 // (op 28033) 
s_mul_i32 s15, s16, s50 // (op 28033) 
s_add_u32 s15, s15, s0 // (op 28033) 
v_sub_u32 v37, v1, v24 // (op 28033) 

s_add_u32 s6, s15, s9 // (op 28033) 

v_lshrrev_b32 v24, 7, v1 // (op 28033) 
s_mul_hi_u32 s15, s6, s13 // (op 28033) 
v_lshlrev_b32 v39, 1, v24 // (op 28033) 

s_sub_u32 s7, s6, s15 // (op 28033) 
s_lshr_b32 s8, s7, 1 // (op 28033) 

s_add_u32 s7, s8, s15 // (op 28033) 


s_lshr_b32 s8, s7, s11 // (op 28033) 

s_and_b32 s7, s11, 2147483648 // (op 28033) 
s_cmp_eq_i32 s7, 0 // (op 28033) 
s_mov_b32 s15, scc // (op 28033) 

s_cmp_eq_i32 s15, 0 // (op 28033) 
s_cselect_b32 s7, s6, s8 // (op 28033) 


s_mul_hi_u32 s8, s7, s12 // (op 28033) 
s_sub_u32 s15, s7, s8 // (op 28033) 
s_lshr_b32 s18, s15, 1 // (op 28033) 

s_add_u32 s15, s18, s8 // (op 28033) 


s_lshr_b32 s8, s15, s10 // (op 28033) 

s_and_b32 s15, s10, 2147483648 // (op 28033) 
s_cmp_eq_i32 s15, 0 // (op 28033) 
s_mov_b32 s18, scc // (op 28033) 

s_cmp_eq_i32 s18, 0 // (op 28033) 
s_cselect_b32 s15, s7, s8 // (op 28033) 


s_mul_i32 s8, s15, s14 // (op 28033) 

s_sub_u32 s15, s7, s8 // (op 28033) 

v_lshl_add_u32 v24, s15, 7, v37 // (op 28033) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 28033) 
s_mul_i32 s8, s7, s18 // (op 28033) 


s_sub_u32 s7, s6, s8 // (op 28033) 


v_lshl_add_u32 v37, s7, 2, v39 // (op 28033) 


v_mul_lo_u32 v39, v37, s44 // (op 28033) 
v_add_u32 v39, v39, v24 // (op 28033) 


s_mul_i32 s19, s44, s43 // (op 28035) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 28035) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 28035) least significant
s_mul_hi_u32 s6, s44, s42 // (op 28035) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28035) most significant: sum
s_add_u32 s19, s19, s6 // (op 28035) most significant: sum


s_mov_b32 s80, s46 // (op 28035) reg expression
s_mov_b32 s81, s47 // (op 28035) reg expression
s_mov_b32 s82, s18 // (op 28035) reg expression
s_mov_b32 s83, 131072 // (op 28035) call()

v_and_b32 v24, -4, v1 // (op 28606) 
v_sub_u32 v37, v1, v24 // (op 28606) 

v_lshlrev_b32 v24, 5, v37 // (op 28606) 

v_lshrrev_b32 v37, 2, v1 // (op 28606) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28606) 


v_lshlrev_b32 v24, 2, v40 // (op 28606) 

v_lshrrev_b32 v37, 3, v24 // (op 28606) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28606) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30231) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27492) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27492) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27492) 


v_mov_b64 v[42:43], v[30:31] // (op 27492) reg expression
s_mov_b32 m0, s15 // (op 27492) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 27492) Load value direct to lds
v_add_co_u32 v42, s[18:19], v42, v40 // (op 27492) least significant half
v_addc_co_u32 v43, s[18:19], 0, v43, s[18:19] // (op 27492) most significant half

s_add_u32 m0, m0, 4096 // (op 27492) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 27492) Load value direct to lds




v_mov_b64 v[40:41], s[44:45] // (op 26568) convert

v_mov_b32 v24, v38 // (op 26568) reg expression
buffer_load_ubyte v42, v24, s[80:83], 0 offen offset: 0 // (op 26568) Load value
v_add_u32 v24, v24, v40 // (op 26568) 
buffer_load_ubyte v43, v24, s[80:83], 0 offen offset: 0 // (op 26568) Load value


v_and_b32 v24, -4, v1 // (op 28531) 
v_sub_u32 v37, v1, v24 // (op 28531) 

v_lshlrev_b32 v24, 5, v37 // (op 28531) 

v_lshrrev_b32 v37, 2, v1 // (op 28531) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28531) 


v_lshlrev_b32 v24, 2, v40 // (op 28531) 

v_lshrrev_b32 v37, 3, v24 // (op 28531) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28531) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30212) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27497) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27497) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27497) 


v_mov_b64 v[44:45], v[26:27] // (op 27497) reg expression
s_add_u32 s6, 8192, s15 // (op 27497) 
s_mov_b32 m0, s6 // (op 27497) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 27497) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v40 // (op 27497) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 27497) most significant half

s_add_u32 m0, m0, 4096 // (op 27497) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 27497) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26574) convert

v_mov_b32 v24, v25 // (op 26574) reg expression
buffer_load_ubyte v44, v24, s[76:79], 0 offen offset: 0 // (op 26574) Load value
v_add_u32 v24, v24, v40 // (op 26574) 
buffer_load_ubyte v45, v24, s[76:79], 0 offen offset: 0 // (op 26574) Load value


v_and_b32 v24, -4, v1 // (op 28916) 
v_sub_u32 v37, v1, v24 // (op 28916) 

v_lshlrev_b32 v24, 5, v37 // (op 28916) 

v_lshrrev_b32 v37, 2, v1 // (op 28916) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28916) 


v_lshlrev_b32 v24, 2, v40 // (op 28916) 

v_lshrrev_b32 v37, 3, v24 // (op 28916) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28916) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30302) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27502) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27502) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27502) 


s_lshl_b64 s[18:19], s[34:35], 7 // (op 27502) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27502) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27502) 

v_mov_b32 v24, s18 // (op 27502) 
v_mov_b32 v37, s19 // (op 27502) 
v_add_co_u32 v46, s[28:29], v30, v24 // (op 27502) least significant half
v_addc_co_u32 v47, s[28:29], v31, v37, s[28:29] // (op 27502) most significant half




s_add_u32 s6, 16384, s15 // (op 27502) 
s_mov_b32 m0, s6 // (op 27502) reg expression
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 27502) Load value direct to lds
v_add_co_u32 v46, s[18:19], v46, v40 // (op 27502) least significant half
v_addc_co_u32 v47, s[18:19], 0, v47, s[18:19] // (op 27502) most significant half

s_add_u32 m0, m0, 4096 // (op 27502) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 27502) Load value direct to lds





v_mov_b64 v[40:41], s[44:45] // (op 26580) convert

s_lshl_b32 s15, s44, 2 // (op 26580) 
v_add_u32 v24, v38, s15 // (op 26580) 

buffer_load_ubyte v46, v24, s[80:83], 0 offen offset: 0 // (op 26580) Load value
v_add_u32 v24, v24, v40 // (op 26580) 
buffer_load_ubyte v47, v24, s[80:83], 0 offen offset: 0 // (op 26580) Load value


v_and_b32 v24, -4, v1 // (op 28872) 
v_sub_u32 v37, v1, v24 // (op 28872) 

v_lshlrev_b32 v24, 5, v37 // (op 28872) 

v_lshrrev_b32 v37, 2, v1 // (op 28872) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28872) 


v_lshlrev_b32 v24, 2, v40 // (op 28872) 

v_lshrrev_b32 v37, 3, v24 // (op 28872) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28872) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30292) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27504) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27504) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27504) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27504) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27504) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27504) 

v_mov_b32 v24, s18 // (op 27504) 
v_mov_b32 v37, s19 // (op 27504) 
v_add_co_u32 v48, s[28:29], v26, v24 // (op 27504) least significant half
v_addc_co_u32 v49, s[28:29], v27, v37, s[28:29] // (op 27504) most significant half




s_add_u32 s6, 24576, s15 // (op 27504) 
s_mov_b32 m0, s6 // (op 27504) reg expression
buffer_load_dwordx4 v48, s[68:71], 0 offen offset: 0 lds // (op 27504) Load value direct to lds
v_add_co_u32 v48, s[18:19], v48, v40 // (op 27504) least significant half
v_addc_co_u32 v49, s[18:19], 0, v49, s[18:19] // (op 27504) most significant half

s_add_u32 m0, m0, 4096 // (op 27504) 
buffer_load_dwordx4 v48, s[68:71], 0 offen offset: 0 lds // (op 27504) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26586) convert

s_lshl_b32 s15, s38, 2 // (op 26586) 
v_add_u32 v24, v25, s15 // (op 26586) 

buffer_load_ubyte v48, v24, s[76:79], 0 offen offset: 0 // (op 26586) Load value
v_add_u32 v24, v24, v40 // (op 26586) 
buffer_load_ubyte v49, v24, s[76:79], 0 offen offset: 0 // (op 26586) Load value


v_and_b32 v24, -4, v1 // (op 29062) 
v_sub_u32 v37, v1, v24 // (op 29062) 

v_lshlrev_b32 v24, 5, v37 // (op 29062) 

v_lshrrev_b32 v37, 2, v1 // (op 29062) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 29062) 


v_lshlrev_b32 v24, 2, v40 // (op 29062) 

v_lshrrev_b32 v37, 3, v24 // (op 29062) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 29062) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30334) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27506) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27506) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27506) 


s_lshl_b64 s[18:19], s[34:35], 7 // (op 27506) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27506) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27506) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 27506) 

v_mov_b32 v24, s28 // (op 27506) 
v_mov_b32 v37, s29 // (op 27506) 
v_add_co_u32 v50, s[18:19], v30, v24 // (op 27506) least significant half
v_addc_co_u32 v51, s[18:19], v31, v37, s[18:19] // (op 27506) most significant half




s_add_u32 s6, 32768, s15 // (op 27506) 
s_mov_b32 m0, s6 // (op 27506) reg expression
buffer_load_dwordx4 v50, s[72:75], 0 offen offset: 0 lds // (op 27506) Load value direct to lds
v_add_co_u32 v50, s[18:19], v50, v40 // (op 27506) least significant half
v_addc_co_u32 v51, s[18:19], 0, v51, s[18:19] // (op 27506) most significant half

s_add_u32 m0, m0, 4096 // (op 27506) 
buffer_load_dwordx4 v50, s[72:75], 0 offen offset: 0 lds // (op 27506) Load value direct to lds





v_mov_b64 v[40:41], s[44:45] // (op 26592) convert

s_lshl_b32 s15, s44, 3 // (op 26592) 
v_add_u32 v24, v38, s15 // (op 26592) 

buffer_load_ubyte v50, v24, s[80:83], 0 offen offset: 0 // (op 26592) Load value
v_add_u32 v24, v24, v40 // (op 26592) 
buffer_load_ubyte v51, v24, s[80:83], 0 offen offset: 0 // (op 26592) Load value


v_and_b32 v24, -4, v1 // (op 29018) 
v_sub_u32 v37, v1, v24 // (op 29018) 

v_lshlrev_b32 v24, 5, v37 // (op 29018) 

v_lshrrev_b32 v37, 2, v1 // (op 29018) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 29018) 


v_lshlrev_b32 v24, 2, v40 // (op 29018) 

v_lshrrev_b32 v37, 3, v24 // (op 29018) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 29018) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30324) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27508) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27508) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27508) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27508) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27508) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27508) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 27508) 

v_mov_b32 v24, s28 // (op 27508) 
v_mov_b32 v37, s29 // (op 27508) 
v_add_co_u32 v52, s[18:19], v26, v24 // (op 27508) least significant half
v_addc_co_u32 v53, s[18:19], v27, v37, s[18:19] // (op 27508) most significant half




s_add_u32 s6, 40960, s15 // (op 27508) 
s_mov_b32 m0, s6 // (op 27508) reg expression
buffer_load_dwordx4 v52, s[68:71], 0 offen offset: 0 lds // (op 27508) Load value direct to lds
v_add_co_u32 v52, s[18:19], v52, v40 // (op 27508) least significant half
v_addc_co_u32 v53, s[18:19], 0, v53, s[18:19] // (op 27508) most significant half

s_add_u32 m0, m0, 4096 // (op 27508) 
buffer_load_dwordx4 v52, s[68:71], 0 offen offset: 0 lds // (op 27508) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26598) convert

s_lshl_b32 s15, s38, 3 // (op 26598) 
v_add_u32 v24, v25, s15 // (op 26598) 

buffer_load_ubyte v52, v24, s[76:79], 0 offen offset: 0 // (op 26598) Load value
v_add_u32 v24, v24, v40 // (op 26598) 
buffer_load_ubyte v53, v24, s[76:79], 0 offen offset: 0 // (op 26598) Load value


v_and_b32 v24, -4, v1 // (op 29208) 
v_sub_u32 v37, v1, v24 // (op 29208) 

v_lshlrev_b32 v24, 5, v37 // (op 29208) 

v_lshrrev_b32 v37, 2, v1 // (op 29208) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 29208) 


v_lshlrev_b32 v24, 2, v40 // (op 29208) 

v_lshrrev_b32 v37, 3, v24 // (op 29208) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 29208) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30366) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27510) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27510) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27510) 


s_lshl_b64 s[18:19], s[34:35], 7 // (op 27510) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27510) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27510) 

s_mul_i32 s29, 3, s19 // (op 27510) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 27510) least significant
s_mul_hi_u32 s6, 3, s18 // (op 27510) most significant: high of low * low
s_add_u32 s29, s29, s6 // (op 27510) most significant: sum


v_mov_b32 v24, s28 // (op 27510) 
v_mov_b32 v37, s29 // (op 27510) 
v_add_co_u32 v54, s[18:19], v30, v24 // (op 27510) least significant half
v_addc_co_u32 v55, s[18:19], v31, v37, s[18:19] // (op 27510) most significant half




s_add_u32 s6, 49152, s15 // (op 27510) 
s_mov_b32 m0, s6 // (op 27510) reg expression
buffer_load_dwordx4 v54, s[72:75], 0 offen offset: 0 lds // (op 27510) Load value direct to lds
v_add_co_u32 v54, s[18:19], v54, v40 // (op 27510) least significant half
v_addc_co_u32 v55, s[18:19], 0, v55, s[18:19] // (op 27510) most significant half

s_add_u32 m0, m0, 4096 // (op 27510) 
buffer_load_dwordx4 v54, s[72:75], 0 offen offset: 0 lds // (op 27510) Load value direct to lds






v_mov_b64 v[30:31], s[44:45] // (op 26604) convert

s_lshl_b32 s15, s44, 2 // (op 26604) 
s_mul_i32 s6, 3, s15 // (op 26604) 

v_add_u32 v24, v38, s6 // (op 26604) 

buffer_load_ubyte v40, v24, s[80:83], 0 offen offset: 0 // (op 26604) Load value
v_add_u32 v24, v24, v30 // (op 26604) 
buffer_load_ubyte v41, v24, s[80:83], 0 offen offset: 0 // (op 26604) Load value



v_and_b32 v24, -4, v1 // (op 29164) 
v_sub_u32 v30, v1, v24 // (op 29164) 

v_lshlrev_b32 v24, 5, v30 // (op 29164) 

v_lshrrev_b32 v30, 2, v1 // (op 29164) 
v_lshl_add_u32 v31, v30, 7, v24 // (op 29164) 


v_lshlrev_b32 v24, 2, v31 // (op 29164) 

v_lshrrev_b32 v30, 3, v24 // (op 29164) 

s_nop 0
v_readfirstlane_b32 s15, v30 // (op 29164) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30356) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27512) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27512) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27512) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27512) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27512) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27512) 

s_mul_i32 s29, 3, s19 // (op 27512) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 27512) least significant
s_mul_hi_u32 s6, 3, s18 // (op 27512) most significant: high of low * low
s_add_u32 s29, s29, s6 // (op 27512) most significant: sum


v_mov_b32 v24, s28 // (op 27512) 
v_mov_b32 v37, s29 // (op 27512) 
v_add_co_u32 v54, s[18:19], v26, v24 // (op 27512) least significant half
v_addc_co_u32 v55, s[18:19], v27, v37, s[18:19] // (op 27512) most significant half




s_add_u32 s6, 57344, s15 // (op 27512) 
s_mov_b32 m0, s6 // (op 27512) reg expression
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27512) Load value direct to lds
v_add_co_u32 v54, s[18:19], v54, v30 // (op 27512) least significant half
v_addc_co_u32 v55, s[18:19], 0, v55, s[18:19] // (op 27512) most significant half

s_add_u32 m0, m0, 4096 // (op 27512) 
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27512) Load value direct to lds






v_mov_b64 v[26:27], s[38:39] // (op 26610) convert

s_lshl_b32 s15, s38, 2 // (op 26610) 
s_mul_i32 s6, 3, s15 // (op 26610) 

v_add_u32 v24, v25, s6 // (op 26610) 

buffer_load_ubyte v30, v24, s[76:79], 0 offen offset: 0 // (op 26610) Load value
v_add_u32 v24, v24, v26 // (op 26610) 
buffer_load_ubyte v31, v24, s[76:79], 0 offen offset: 0 // (op 26610) Load value



v_and_b32 v24, -128, v1 // (op 28758) 
v_sub_u32 v25, v1, v24 // (op 28758) 

v_lshrrev_b32 v24, 7, v1 // (op 28758) 
v_lshl_add_u32 v26, v24, 8, v25 // (op 28758) 


s_barrier  // (op 30272) 



v_add_u32 v27, 65536, v26 // (op 26613) 
s_waitcnt vmcnt(29)
ds_write_b8 v27, v42 // (op 26613) Store local data 

v_add_u32 v27, 65664, v26 // (op 26613) 
s_waitcnt vmcnt(28)
ds_write_b8 v27, v43 // (op 26613) Store local data 


v_and_b32 v24, -128, v1 // (op 28682) 
v_sub_u32 v25, v1, v24 // (op 28682) 

v_lshrrev_b32 v24, 7, v1 // (op 28682) 
v_lshl_add_u32 v26, v24, 8, v25 // (op 28682) 


s_barrier  // (op 30251) 



v_add_u32 v27, 66048, v26 // (op 26615) 
s_waitcnt vmcnt(25)
ds_write_b8 v27, v44 // (op 26615) Store local data 

v_add_u32 v27, 66176, v26 // (op 26615) 
s_waitcnt vmcnt(24)
ds_write_b8 v27, v45 // (op 26615) Store local data 


s_barrier  // (op 30228) 



ds_read_b64_tr_b4 v[24:25], v22 // (op 26658) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v22 offset:1024 // (op 26658) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30269) 


v_add_u32 v37, 65536, v23 // (op 26698) 
ds_read_u8 v38, v37 // (op 26698) Load local data 




ds_read_b64_tr_b4 v[54:55], v22 offset:16 // (op 26663) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v22 offset:1040 // (op 26663) Transpose load local data 


v_add_u32 v37, 65568, v23 // (op 26703) 
ds_read_u8 v58, v37 // (op 26703) Load local data 




ds_read_b64_tr_b4 v[60:61], v22 offset:32 // (op 26668) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v22 offset:1056 // (op 26668) Transpose load local data 


v_add_u32 v37, 65600, v23 // (op 26708) 
ds_read_u8 v59, v37 // (op 26708) Load local data 




ds_read_b64_tr_b4 v[64:65], v22 offset:48 // (op 26673) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v22 offset:1072 // (op 26673) Transpose load local data 


v_add_u32 v37, 65632, v23 // (op 26713) 
ds_read_u8 v68, v37 // (op 26713) Load local data 

s_barrier  // (op 30209) 



ds_read_b64_tr_b4 v[70:71], v3 offset:8192 // (op 26638) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:9216 // (op 26638) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30248) 


v_add_u32 v37, 66048, v0 // (op 26678) 
ds_read_u8 v69, v37 // (op 26678) Load local data 




ds_read_b64_tr_b4 v[74:75], v3 offset:8208 // (op 26643) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:9232 // (op 26643) Transpose load local data 


v_add_u32 v37, 66080, v0 // (op 26683) 
ds_read_u8 v78, v37 // (op 26683) Load local data 




ds_read_b64_tr_b4 v[80:81], v3 offset:8224 // (op 26648) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:9248 // (op 26648) Transpose load local data 


v_add_u32 v37, 66112, v0 // (op 26688) 
ds_read_u8 v79, v37 // (op 26688) Load local data 




ds_read_b64_tr_b4 v[84:85], v3 offset:8240 // (op 26653) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:9264 // (op 26653) Transpose load local data 


v_add_u32 v37, 66144, v0 // (op 26693) 
ds_read_u8 v88, v37 // (op 26693) Load local data 

v_and_b32 v37, -4, v1 // (op 28503) 
v_sub_u32 v89, v1, v37 // (op 28503) 

v_lshlrev_b32 v37, 5, v89 // (op 28503) 

v_lshrrev_b32 v89, 2, v1 // (op 28503) 
v_lshl_add_u32 v90, v89, 7, v37 // (op 28503) 


v_lshlrev_b32 v37, 2, v90 // (op 28503) 

v_lshrrev_b32 v89, 3, v37 // (op 28503) 

s_nop 0
v_readfirstlane_b32 s15, v89 // (op 28503) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 28578) 
v_sub_u32 v89, v1, v37 // (op 28578) 

v_lshlrev_b32 v37, 5, v89 // (op 28578) 

v_lshrrev_b32 v89, 2, v1 // (op 28578) 
v_lshl_add_u32 v90, v89, 7, v37 // (op 28578) 


v_lshlrev_b32 v37, 2, v90 // (op 28578) 

v_lshrrev_b32 v89, 3, v37 // (op 28578) 

s_nop 0
v_readfirstlane_b32 s6, v89 // (op 28578) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 28653) 
v_sub_u32 v89, v1, v37 // (op 28653) 

v_lshrrev_b32 v37, 7, v1 // (op 28653) 
v_lshl_add_u32 v90, v37, 8, v89 // (op 28653) 


v_and_b32 v37, -128, v1 // (op 28729) 
v_sub_u32 v89, v1, v37 // (op 28729) 

v_lshrrev_b32 v37, 7, v1 // (op 28729) 
v_lshl_add_u32 v91, v37, 8, v89 // (op 28729) 


v_and_b32 v37, -4, v1 // (op 28857) 
v_sub_u32 v89, v1, v37 // (op 28857) 

v_lshlrev_b32 v37, 5, v89 // (op 28857) 

v_lshrrev_b32 v89, 2, v1 // (op 28857) 
v_lshl_add_u32 v92, v89, 7, v37 // (op 28857) 


v_lshlrev_b32 v37, 2, v92 // (op 28857) 

v_lshrrev_b32 v89, 3, v37 // (op 28857) 

s_nop 0
v_readfirstlane_b32 s7, v89 // (op 28857) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 28901) 
v_sub_u32 v89, v1, v37 // (op 28901) 

v_lshlrev_b32 v37, 5, v89 // (op 28901) 

v_lshrrev_b32 v89, 2, v1 // (op 28901) 
v_lshl_add_u32 v92, v89, 7, v37 // (op 28901) 


v_lshlrev_b32 v37, 2, v92 // (op 28901) 

v_lshrrev_b32 v89, 3, v37 // (op 28901) 

s_nop 0
v_readfirstlane_b32 s8, v89 // (op 28901) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 28945) 
v_sub_u32 v89, v1, v37 // (op 28945) 

v_lshrrev_b32 v37, 7, v1 // (op 28945) 
v_lshl_add_u32 v92, v37, 8, v89 // (op 28945) 


v_and_b32 v37, -128, v1 // (op 28974) 
v_sub_u32 v89, v1, v37 // (op 28974) 

v_lshrrev_b32 v37, 7, v1 // (op 28974) 
v_lshl_add_u32 v93, v37, 8, v89 // (op 28974) 


v_and_b32 v37, -4, v1 // (op 29003) 
v_sub_u32 v89, v1, v37 // (op 29003) 

v_lshlrev_b32 v37, 5, v89 // (op 29003) 

v_lshrrev_b32 v89, 2, v1 // (op 29003) 
v_lshl_add_u32 v94, v89, 7, v37 // (op 29003) 


v_lshlrev_b32 v37, 2, v94 // (op 29003) 

v_lshrrev_b32 v89, 3, v37 // (op 29003) 

s_nop 0
v_readfirstlane_b32 s18, v89 // (op 29003) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 29047) 
v_sub_u32 v89, v1, v37 // (op 29047) 

v_lshlrev_b32 v37, 5, v89 // (op 29047) 

v_lshrrev_b32 v89, 2, v1 // (op 29047) 
v_lshl_add_u32 v94, v89, 7, v37 // (op 29047) 


v_lshlrev_b32 v37, 2, v94 // (op 29047) 

v_lshrrev_b32 v89, 3, v37 // (op 29047) 

s_nop 0
v_readfirstlane_b32 s19, v89 // (op 29047) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 29091) 
v_sub_u32 v89, v1, v37 // (op 29091) 

v_lshrrev_b32 v37, 7, v1 // (op 29091) 
v_lshl_add_u32 v94, v37, 8, v89 // (op 29091) 


v_and_b32 v37, -128, v1 // (op 29120) 
v_sub_u32 v89, v1, v37 // (op 29120) 

v_lshrrev_b32 v37, 7, v1 // (op 29120) 
v_lshl_add_u32 v95, v37, 8, v89 // (op 29120) 


v_and_b32 v37, -4, v1 // (op 29149) 
v_sub_u32 v89, v1, v37 // (op 29149) 

v_lshlrev_b32 v37, 5, v89 // (op 29149) 

v_lshrrev_b32 v89, 2, v1 // (op 29149) 
v_lshl_add_u32 v96, v89, 7, v37 // (op 29149) 


v_lshlrev_b32 v37, 2, v96 // (op 29149) 

v_lshrrev_b32 v89, 3, v37 // (op 29149) 

s_nop 0
v_readfirstlane_b32 s28, v89 // (op 29149) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 29193) 
v_sub_u32 v89, v1, v37 // (op 29193) 

v_lshlrev_b32 v37, 5, v89 // (op 29193) 

v_lshrrev_b32 v89, 2, v1 // (op 29193) 
v_lshl_add_u32 v96, v89, 7, v37 // (op 29193) 


v_lshlrev_b32 v37, 2, v96 // (op 29193) 

v_lshrrev_b32 v89, 3, v37 // (op 29193) 

s_nop 0
v_readfirstlane_b32 s29, v89 // (op 29193) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 29237) 
v_sub_u32 v89, v1, v37 // (op 29237) 

v_lshrrev_b32 v37, 7, v1 // (op 29237) 
v_lshl_add_u32 v96, v37, 8, v89 // (op 29237) 


v_and_b32 v37, -128, v1 // (op 29266) 
v_sub_u32 v89, v1, v37 // (op 29266) 

v_lshrrev_b32 v37, 7, v1 // (op 29266) 
v_lshl_add_u32 v97, v37, 8, v89 // (op 29266) 


s_mov_b32 s1, 0 // (op 3183) call()
s_ashr_i64 s[48:49], s[2:3], 7 // (op 3182) 
s_mov_b32 s52, s48 // (op 3182) convert

s_and_b32 s48, s52, 4294967292 // (op 3182) 

s_cmp_lt_u32 s1, s48 // (op 3182) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3182 // (op 3182) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3182 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3182:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[24:27], a[0:3], v69, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 198) 
s_barrier  // (op 30234) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[54:57], a[4:7], v69, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 808) 
s_add_u32 s1, s1, 4 // (op 3184) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[60:63], a[8:11], v69, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1106) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27401) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[64:67], a[12:15], v69, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1404) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27401) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27401) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[24:27], a[16:19], v78, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2207) 
v_mov_b64 v[100:101], v[32:33] // (op 27401) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[54:57], a[20:23], v78, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2115) 
s_lshl_b64 s[48:49], s[34:35], 7 // (op 27401) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[60:63], a[24:27], v78, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2023) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27401) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[64:67], a[28:31], v78, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1931) 
s_lshr_b64 s[48:49], s[52:53], 3 // (op 27401) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27401) 

v_mov_b32 v37, s52 // (op 27401) 
v_mov_b32 v89, s53 // (op 27401) 
v_add_co_u32 v102, s[48:49], v100, v37 // (op 27401) least significant half
v_addc_co_u32 v103, s[48:49], v101, v89, s[48:49] // (op 27401) most significant half




s_mov_b32 m0, s6 // (op 27401) reg expression
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 27401) Load value direct to lds
v_add_co_u32 v102, s[48:49], v102, v98 // (op 27401) least significant half
v_addc_co_u32 v103, s[48:49], 0, v103, s[48:49] // (op 27401) most significant half

s_add_u32 m0, m0, 4096 // (op 27401) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 27401) Load value direct to lds


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[24:27], a[32:35], v79, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3497) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[54:57], a[36:39], v79, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3405) 
v_mov_b64 v[98:99], s[44:45] // (op 204) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[60:63], a[40:43], v79, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3313) 
v_mov_b32 v37, v39 // (op 204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[64:67], a[44:47], v79, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3221) 
s_lshl_b32 s48, s44, 4 // (op 204) 
v_add_u32 v89, v37, s48 // (op 204) 

buffer_load_ubyte v42, v89, s[80:83], 0 offen offset: 0 // (op 204) Load value
v_add_u32 v89, v89, v98 // (op 204) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[24:27], a[48:51], v88, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4787) 
buffer_load_ubyte v43, v89, s[80:83], 0 offen offset: 0 // (op 204) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[54:57], a[52:55], v88, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4695) 
s_barrier  // (op 30215) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[60:63], a[56:59], v88, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4603) 
s_lshl_b64 s[48:49], s[30:31], 6 // (op 27396) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[64:67], a[60:63], v88, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4511) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27396) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27396) 


v_mov_b64 v[102:103], v[28:29] // (op 27396) 
s_lshl_b64 s[48:49], s[30:31], 7 // (op 27396) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27396) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27396) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27396) 

v_mov_b32 v89, s52 // (op 27396) 
v_mov_b32 v104, s53 // (op 27396) 
v_add_co_u32 v106, s[48:49], v102, v89 // (op 27396) least significant half
v_addc_co_u32 v107, s[48:49], v103, v104, s[48:49] // (op 27396) most significant half




s_add_u32 s48, 8192, s15 // (op 27396) 
s_nop 0
s_mov_b32 m0, s48 // (op 27396) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27396) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v98 // (op 27396) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27396) most significant half

s_add_u32 m0, m0, 4096 // (op 27396) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27396) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 203) convert

v_mov_b32 v89, v36 // (op 203) 
s_lshl_b32 s48, s38, 4 // (op 203) 
v_add_u32 v104, v89, s48 // (op 203) 

buffer_load_ubyte v44, v104, s[76:79], 0 offen offset: 0 // (op 203) Load value
v_add_u32 v104, v104, v98 // (op 203) 
buffer_load_ubyte v45, v104, s[76:79], 0 offen offset: 0 // (op 203) Load value


s_barrier  // (op 30318) 



v_add_u32 v98, 66560, v93 // (op 541) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v46 // (op 541) Store local data 

v_add_u32 v98, 66688, v93 // (op 541) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v47 // (op 541) Store local data 

s_barrier  // (op 30312) 



v_add_u32 v98, 67072, v92 // (op 540) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v48 // (op 540) Store local data 

v_add_u32 v98, 67200, v92 // (op 540) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v49 // (op 540) Store local data 

s_barrier  // (op 30298) 



ds_read_b64_tr_b4 v[104:105], v12 offset:16384 // (op 533) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v12 offset:17408 // (op 533) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30315) 


v_add_u32 v98, 66560, v13 // (op 537) 
ds_read_u8 v99, v98 // (op 537) Load local data 




ds_read_b64_tr_b4 v[108:109], v12 offset:16400 // (op 815) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v12 offset:17424 // (op 815) Transpose load local data 


v_add_u32 v98, 66592, v13 // (op 819) 
ds_read_u8 v78, v98 // (op 819) Load local data 




ds_read_b64_tr_b4 v[74:75], v12 offset:16416 // (op 1113) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v12 offset:17440 // (op 1113) Transpose load local data 


v_add_u32 v98, 66624, v13 // (op 1117) 
ds_read_u8 v79, v98 // (op 1117) Load local data 




ds_read_b64_tr_b4 v[80:81], v12 offset:16432 // (op 1411) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v12 offset:17456 // (op 1411) Transpose load local data 


v_add_u32 v98, 66656, v13 // (op 1415) 
ds_read_u8 v112, v98 // (op 1415) Load local data 

s_barrier  // (op 30289) 
s_barrier  // (op 30577) 



ds_read_b64_tr_b4 v[70:71], v11 offset:24576 // (op 531) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v11 offset:25600 // (op 531) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30308) 


v_add_u32 v98, 67072, v10 // (op 535) 
ds_read_u8 v69, v98 // (op 535) Load local data 




ds_read_b64_tr_b4 v[114:115], v11 offset:24592 // (op 1936) Transpose load local data 
ds_read_b64_tr_b4 v[116:117], v11 offset:25616 // (op 1936) Transpose load local data 


v_add_u32 v98, 67104, v10 // (op 1939) 
ds_read_u8 v58, v98 // (op 1939) Load local data 




ds_read_b64_tr_b4 v[118:119], v11 offset:24608 // (op 3226) Transpose load local data 
ds_read_b64_tr_b4 v[120:121], v11 offset:25632 // (op 3226) Transpose load local data 


v_add_u32 v98, 67136, v10 // (op 3229) 
ds_read_u8 v59, v98 // (op 3229) Load local data 




ds_read_b64_tr_b4 v[122:123], v11 offset:24624 // (op 4516) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v11 offset:25648 // (op 4516) Transpose load local data 


v_add_u32 v98, 67168, v10 // (op 4519) 
ds_read_u8 v68, v98 // (op 4519) Load local data 

s_barrier  // (op 30635) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[104:107], a[0:3], v69, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 538) 
s_barrier  // (op 30305) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[108:111], a[4:7], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 809) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27424) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[74:77], a[8:11], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1107) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27424) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[80:83], a[12:15], v69, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1405) 
v_lshrrev_b64 v[126:127], 3, s[52:53] // (op 27424) 


s_lshl_b64 s[48:49], s[34:35], 7 // (op 27424) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:117], v[104:107], a[16:19], v58, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2208) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27424) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:117], v[108:111], a[20:23], v58, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2116) 
s_lshr_b64 s[48:49], s[52:53], 3 // (op 27424) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[74:77], a[24:27], v58, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2024) 
s_mul_i32 s53, 5, s49 // (op 27424) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:117], v[80:83], a[28:31], v58, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1932) 
s_mul_i32 s52, 5, s48 // (op 27424) least significant

s_mul_hi_u32 s54, 5, s48 // (op 27424) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27424) most significant: sum


v_mov_b32 v98, s52 // (op 27424) 
v_mov_b32 v113, s53 // (op 27424) 
v_add_co_u32 v114, s[48:49], v100, v98 // (op 27424) least significant half
v_addc_co_u32 v115, s[48:49], v101, v113, s[48:49] // (op 27424) most significant half




s_add_u32 s48, 16384, s8 // (op 27424) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[118:121], v[104:107], a[32:35], v59, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3498) 
s_mov_b32 m0, s48 // (op 27424) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:121], v[108:111], a[36:39], v59, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3406) 
buffer_load_dwordx4 v114, s[72:75], 0 offen offset: 0 lds // (op 27424) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[118:121], v[74:77], a[40:43], v59, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3314) 
v_add_co_u32 v114, s[52:53], v114, v126 // (op 27424) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[118:121], v[80:83], a[44:47], v59, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3222) 
v_addc_co_u32 v115, s[52:53], 0, v115, s[52:53] // (op 27424) most significant half


s_add_u32 m0, m0, 4096 // (op 27424) 
buffer_load_dwordx4 v114, s[72:75], 0 offen offset: 0 lds // (op 27424) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[122:125], v[104:107], a[48:51], v68, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4788) 



v_mov_b64 v[98:99], s[44:45] // (op 544) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[122:125], v[108:111], a[52:55], v68, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4696) 
s_lshl_b32 s48, s44, 2 // (op 544) 

s_mul_i32 s49, 5, s48 // (op 544) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[122:125], v[74:77], a[56:59], v68, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4604) 
v_add_u32 v104, v37, s49 // (op 544) 

buffer_load_ubyte v46, v104, s[80:83], 0 offen offset: 0 // (op 544) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[122:125], v[80:83], a[60:63], v68, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4512) 
v_add_u32 v104, v104, v98 // (op 544) 


buffer_load_ubyte v47, v104, s[80:83], 0 offen offset: 0 // (op 544) Load value


s_barrier  // (op 30295) 

s_lshl_b64 s[48:49], s[30:31], 6 // (op 27419) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27419) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27419) 


s_lshl_b64 s[48:49], s[30:31], 7 // (op 27419) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27419) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27419) 

s_mul_i32 s53, 5, s49 // (op 27419) most significant: low of low * high
s_mul_i32 s52, 5, s48 // (op 27419) least significant
s_mul_hi_u32 s54, 5, s48 // (op 27419) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27419) most significant: sum


v_mov_b32 v104, s52 // (op 27419) 
v_mov_b32 v105, s53 // (op 27419) 
v_add_co_u32 v106, s[48:49], v102, v104 // (op 27419) least significant half
v_addc_co_u32 v107, s[48:49], v103, v105, s[48:49] // (op 27419) most significant half




s_add_u32 s48, 24576, s7 // (op 27419) 
s_nop 0
s_mov_b32 m0, s48 // (op 27419) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27419) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v98 // (op 27419) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27419) most significant half

s_add_u32 m0, m0, 4096 // (op 27419) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27419) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 543) convert

s_lshl_b32 s48, s38, 2 // (op 543) 
s_mul_i32 s49, 5, s48 // (op 543) 

v_add_u32 v104, v89, s49 // (op 543) 

buffer_load_ubyte v48, v104, s[76:79], 0 offen offset: 0 // (op 543) Load value
v_add_u32 v104, v104, v98 // (op 543) 
buffer_load_ubyte v49, v104, s[76:79], 0 offen offset: 0 // (op 543) Load value


s_barrier  // (op 30350) 



v_add_u32 v98, 66560, v95 // (op 581) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v50 // (op 581) Store local data 

v_add_u32 v98, 66688, v95 // (op 581) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v51 // (op 581) Store local data 

s_barrier  // (op 30344) 



v_add_u32 v98, 67072, v94 // (op 580) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v52 // (op 580) Store local data 

v_add_u32 v98, 67200, v94 // (op 580) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v53 // (op 580) Store local data 

s_barrier  // (op 30330) 



ds_read_b64_tr_b4 v[24:25], v16 offset:32768 // (op 573) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v16 offset:33792 // (op 573) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30347) 


v_add_u32 v98, 66560, v17 // (op 577) 
ds_read_u8 v38, v98 // (op 577) Load local data 




ds_read_b64_tr_b4 v[54:55], v16 offset:32784 // (op 823) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v16 offset:33808 // (op 823) Transpose load local data 


v_add_u32 v98, 66592, v17 // (op 827) 
ds_read_u8 v58, v98 // (op 827) Load local data 




ds_read_b64_tr_b4 v[60:61], v16 offset:32800 // (op 1121) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v16 offset:33824 // (op 1121) Transpose load local data 


v_add_u32 v98, 66624, v17 // (op 1125) 
ds_read_u8 v59, v98 // (op 1125) Load local data 




ds_read_b64_tr_b4 v[64:65], v16 offset:32816 // (op 1419) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v16 offset:33840 // (op 1419) Transpose load local data 


v_add_u32 v98, 66656, v17 // (op 1423) 
ds_read_u8 v88, v98 // (op 1423) Load local data 

s_barrier  // (op 30321) 
s_barrier  // (op 30583) 



ds_read_b64_tr_b4 v[70:71], v15 offset:40960 // (op 571) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v15 offset:41984 // (op 571) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30340) 


v_add_u32 v98, 67072, v14 // (op 575) 
ds_read_u8 v69, v98 // (op 575) Load local data 




ds_read_b64_tr_b4 v[84:85], v15 offset:40976 // (op 1942) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v15 offset:42000 // (op 1942) Transpose load local data 


v_add_u32 v98, 67104, v14 // (op 1945) 
ds_read_u8 v78, v98 // (op 1945) Load local data 




ds_read_b64_tr_b4 v[104:105], v15 offset:40992 // (op 3232) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v15 offset:42016 // (op 3232) Transpose load local data 


v_add_u32 v98, 67136, v14 // (op 3235) 
ds_read_u8 v79, v98 // (op 3235) Load local data 




ds_read_b64_tr_b4 v[108:109], v15 offset:41008 // (op 4522) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v15 offset:42032 // (op 4522) Transpose load local data 


v_add_u32 v98, 67168, v14 // (op 4525) 
ds_read_u8 v99, v98 // (op 4525) Load local data 

s_barrier  // (op 30641) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[24:27], a[0:3], v69, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 578) 
s_barrier  // (op 30337) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[54:57], a[4:7], v69, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 810) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27434) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[60:63], a[8:11], v69, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1108) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27434) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[64:67], a[12:15], v69, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1406) 
v_lshrrev_b64 v[112:113], 3, s[52:53] // (op 27434) 


s_lshl_b64 s[48:49], s[34:35], 7 // (op 27434) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[24:27], a[16:19], v78, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2209) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27434) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[54:57], a[20:23], v78, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2117) 
s_lshr_b64 s[48:49], s[52:53], 3 // (op 27434) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[60:63], a[24:27], v78, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2025) 
s_mul_i32 s53, 6, s49 // (op 27434) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[64:67], a[28:31], v78, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1933) 
s_mul_i32 s52, 6, s48 // (op 27434) least significant
s_mul_hi_u32 s54, 6, s48 // (op 27434) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27434) most significant: sum


v_mov_b32 v98, s52 // (op 27434) 
v_mov_b32 v114, s53 // (op 27434) 
v_add_co_u32 v116, s[48:49], v100, v98 // (op 27434) least significant half
v_addc_co_u32 v117, s[48:49], v101, v114, s[48:49] // (op 27434) most significant half




s_add_u32 s48, 32768, s19 // (op 27434) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[24:27], a[32:35], v79, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3499) 
s_mov_b32 m0, s48 // (op 27434) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[54:57], a[36:39], v79, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3407) 
buffer_load_dwordx4 v116, s[72:75], 0 offen offset: 0 lds // (op 27434) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[60:63], a[40:43], v79, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3315) 
v_add_co_u32 v116, s[52:53], v116, v112 // (op 27434) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[64:67], a[44:47], v79, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3223) 
v_addc_co_u32 v117, s[52:53], 0, v117, s[52:53] // (op 27434) most significant half


s_add_u32 m0, m0, 4096 // (op 27434) 
buffer_load_dwordx4 v116, s[72:75], 0 offen offset: 0 lds // (op 27434) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[24:27], a[48:51], v99, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4789) 

v_mov_b64 v[104:105], s[44:45] // (op 584) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:111], v[54:57], a[52:55], v99, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4697) 
s_lshl_b32 s48, s44, 2 // (op 584) 
s_mul_i32 s49, 6, s48 // (op 584) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[60:63], a[56:59], v99, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4605) 
v_add_u32 v98, v37, s49 // (op 584) 

buffer_load_ubyte v50, v98, s[80:83], 0 offen offset: 0 // (op 584) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[64:67], a[60:63], v99, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4513) 
v_add_u32 v98, v98, v104 // (op 584) 


buffer_load_ubyte v51, v98, s[80:83], 0 offen offset: 0 // (op 584) Load value


s_barrier  // (op 30327) 

s_lshl_b64 s[48:49], s[30:31], 6 // (op 27429) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27429) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27429) 


s_lshl_b64 s[48:49], s[30:31], 7 // (op 27429) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27429) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27429) 

s_mul_i32 s53, 6, s49 // (op 27429) most significant: low of low * high
s_mul_i32 s52, 6, s48 // (op 27429) least significant
s_mul_hi_u32 s54, 6, s48 // (op 27429) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27429) most significant: sum


v_mov_b32 v104, s52 // (op 27429) 
v_mov_b32 v105, s53 // (op 27429) 
v_add_co_u32 v106, s[48:49], v102, v104 // (op 27429) least significant half
v_addc_co_u32 v107, s[48:49], v103, v105, s[48:49] // (op 27429) most significant half




s_add_u32 s48, 40960, s18 // (op 27429) 
s_nop 0
s_mov_b32 m0, s48 // (op 27429) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27429) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v98 // (op 27429) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27429) most significant half

s_add_u32 m0, m0, 4096 // (op 27429) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27429) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 583) convert

s_lshl_b32 s48, s38, 2 // (op 583) 
s_mul_i32 s49, 6, s48 // (op 583) 

v_add_u32 v104, v89, s49 // (op 583) 

buffer_load_ubyte v52, v104, s[76:79], 0 offen offset: 0 // (op 583) Load value
v_add_u32 v104, v104, v98 // (op 583) 
buffer_load_ubyte v53, v104, s[76:79], 0 offen offset: 0 // (op 583) Load value


s_barrier  // (op 30382) 



v_add_u32 v98, 66560, v97 // (op 621) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v40 // (op 621) Store local data 

v_add_u32 v98, 66688, v97 // (op 621) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v41 // (op 621) Store local data 

s_barrier  // (op 30376) 



v_add_u32 v98, 67072, v96 // (op 620) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v30 // (op 620) Store local data 

v_add_u32 v98, 67200, v96 // (op 620) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v31 // (op 620) Store local data 

s_barrier  // (op 30362) 



ds_read_b64_tr_b4 v[74:75], v20 offset:49152 // (op 613) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v20 offset:50176 // (op 613) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30379) 


v_add_u32 v98, 66560, v21 // (op 617) 
ds_read_u8 v68, v98 // (op 617) Load local data 




ds_read_b64_tr_b4 v[80:81], v20 offset:49168 // (op 831) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v20 offset:50192 // (op 831) Transpose load local data 


v_add_u32 v98, 66592, v21 // (op 835) 
ds_read_u8 v78, v98 // (op 835) Load local data 




ds_read_b64_tr_b4 v[84:85], v20 offset:49184 // (op 1129) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v20 offset:50208 // (op 1129) Transpose load local data 


v_add_u32 v98, 66624, v21 // (op 1133) 
ds_read_u8 v79, v98 // (op 1133) Load local data 




ds_read_b64_tr_b4 v[104:105], v20 offset:49200 // (op 1427) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v20 offset:50224 // (op 1427) Transpose load local data 


v_add_u32 v98, 66656, v21 // (op 1431) 
ds_read_u8 v99, v98 // (op 1431) Load local data 

s_barrier  // (op 30353) 
s_barrier  // (op 30589) 



ds_read_b64_tr_b4 v[70:71], v19 offset:57344 // (op 611) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v19 offset:58368 // (op 611) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30372) 


v_add_u32 v98, 67072, v18 // (op 615) 
ds_read_u8 v69, v98 // (op 615) Load local data 




ds_read_b64_tr_b4 v[108:109], v19 offset:57360 // (op 1948) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v19 offset:58384 // (op 1948) Transpose load local data 


v_add_u32 v98, 67104, v18 // (op 1951) 
ds_read_u8 v58, v98 // (op 1951) Load local data 




ds_read_b64_tr_b4 v[112:113], v19 offset:57376 // (op 3238) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v19 offset:58400 // (op 3238) Transpose load local data 


v_add_u32 v98, 67136, v18 // (op 3241) 
ds_read_u8 v59, v98 // (op 3241) Load local data 




ds_read_b64_tr_b4 v[116:117], v19 offset:57392 // (op 4528) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v19 offset:58416 // (op 4528) Transpose load local data 


v_add_u32 v98, 67168, v18 // (op 4531) 
ds_read_u8 v88, v98 // (op 4531) Load local data 

s_barrier  // (op 30647) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[74:77], a[0:3], v69, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 618) 
s_barrier  // (op 30369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[80:83], a[4:7], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 811) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27444) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[84:87], a[8:11], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1109) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27444) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[104:107], a[12:15], v69, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1407) 
v_lshrrev_b64 v[120:121], 3, s[52:53] // (op 27444) 


s_lshl_b64 s[48:49], s[34:35], 7 // (op 27444) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[74:77], a[16:19], v58, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2210) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27444) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[80:83], a[20:23], v58, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2118) 
s_lshr_b64 s[48:49], s[52:53], 3 // (op 27444) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[84:87], a[24:27], v58, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2026) 
s_mul_i32 s53, 7, s49 // (op 27444) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[104:107], a[28:31], v58, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1934) 
s_mul_i32 s52, 7, s48 // (op 27444) least significant

s_mul_hi_u32 s54, 7, s48 // (op 27444) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27444) most significant: sum


v_mov_b32 v98, s52 // (op 27444) 
v_mov_b32 v108, s53 // (op 27444) 
v_add_co_u32 v110, s[48:49], v100, v98 // (op 27444) least significant half
v_addc_co_u32 v111, s[48:49], v101, v108, s[48:49] // (op 27444) most significant half




s_add_u32 s48, 49152, s29 // (op 27444) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[74:77], a[32:35], v59, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3500) 
s_mov_b32 m0, s48 // (op 27444) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[80:83], a[36:39], v59, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3408) 
buffer_load_dwordx4 v110, s[72:75], 0 offen offset: 0 lds // (op 27444) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[84:87], a[40:43], v59, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3316) 
v_add_co_u32 v110, s[52:53], v110, v120 // (op 27444) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[104:107], a[44:47], v59, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3224) 
v_addc_co_u32 v111, s[52:53], 0, v111, s[52:53] // (op 27444) most significant half


s_add_u32 m0, m0, 4096 // (op 27444) 
buffer_load_dwordx4 v110, s[72:75], 0 offen offset: 0 lds // (op 27444) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[74:77], a[48:51], v88, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4790) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[80:83], a[52:55], v88, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4698) 
v_mov_b64 v[100:101], s[44:45] // (op 624) convert

s_lshl_b32 s48, s44, 2 // (op 624) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[84:87], a[56:59], v88, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4606) 
s_mul_i32 s49, 7, s48 // (op 624) 

v_add_u32 v98, v37, s49 // (op 624) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[104:107], a[60:63], v88, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4514) 
buffer_load_ubyte v40, v98, s[80:83], 0 offen offset: 0 // (op 624) Load value



v_add_u32 v98, v98, v100 // (op 624) 
s_lshl_b64 s[48:49], s[34:35], 7 // (op 27769) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27769) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27769) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27769) 

v_mov_b32 v99, s52 // (op 27769) 
v_mov_b32 v104, s53 // (op 27769) 
v_add_co_u32 v32, s[48:49], v32, v99 // (op 27769) least significant half
v_addc_co_u32 v33, s[48:49], v33, v104, s[48:49] // (op 27769) most significant half




buffer_load_ubyte v41, v98, s[80:83], 0 offen offset: 0 // (op 624) Load value



s_barrier  // (op 30359) 
s_lshl_b32 s48, s44, 4 // (op 28025) 

v_add_u32 v39, v39, s48 // (op 28025) 

s_lshl_b64 s[48:49], s[30:31], 6 // (op 27439) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27439) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27439) 


s_lshl_b64 s[48:49], s[30:31], 7 // (op 27439) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27439) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27439) 

s_mul_i32 s53, 7, s49 // (op 27439) most significant: low of low * high
s_mul_i32 s52, 7, s48 // (op 27439) least significant
s_mul_hi_u32 s54, 7, s48 // (op 27439) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27439) most significant: sum


v_mov_b32 v37, s52 // (op 27439) 
v_mov_b32 v100, s53 // (op 27439) 
v_add_co_u32 v104, s[48:49], v102, v37 // (op 27439) least significant half
v_addc_co_u32 v105, s[48:49], v103, v100, s[48:49] // (op 27439) most significant half




s_add_u32 s48, 57344, s28 // (op 27439) 
s_nop 0
s_mov_b32 m0, s48 // (op 27439) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 27439) Load value direct to lds
v_add_co_u32 v104, s[52:53], v104, v98 // (op 27439) least significant half
v_addc_co_u32 v105, s[52:53], 0, v105, s[52:53] // (op 27439) most significant half

s_add_u32 m0, m0, 4096 // (op 27439) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 27439) Load value direct to lds





v_mov_b64 v[98:99], s[38:39] // (op 623) convert

s_lshl_b32 s48, s38, 2 // (op 623) 
s_mul_i32 s49, 7, s48 // (op 623) 

v_add_u32 v37, v89, s49 // (op 623) 

s_lshl_b64 s[48:49], s[30:31], 7 // (op 27623) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27623) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27623) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27623) 

v_mov_b32 v100, s52 // (op 27623) 
v_mov_b32 v101, s53 // (op 27623) 
v_add_co_u32 v28, s[48:49], v28, v100 // (op 27623) least significant half
v_addc_co_u32 v29, s[48:49], v29, v101, s[48:49] // (op 27623) most significant half




buffer_load_ubyte v30, v37, s[76:79], 0 offen offset: 0 // (op 623) Load value
v_add_u32 v37, v37, v98 // (op 623) 
buffer_load_ubyte v31, v37, s[76:79], 0 offen offset: 0 // (op 623) Load value



s_barrier  // (op 30275) 
s_lshl_b32 s48, s38, 4 // (op 27897) 



v_add_u32 v36, v36, s48 // (op 27897) 

v_add_u32 v37, 65536, v91 // (op 201) 
s_waitcnt vmcnt(29)
ds_write_b8 v37, v42 // (op 201) Store local data 

v_add_u32 v37, 65664, v91 // (op 201) 
s_waitcnt vmcnt(28)
ds_write_b8 v37, v43 // (op 201) Store local data 

s_barrier  // (op 30254) 



v_add_u32 v37, 66048, v90 // (op 200) 
s_waitcnt vmcnt(25)
ds_write_b8 v37, v44 // (op 200) Store local data 

v_add_u32 v37, 66176, v90 // (op 200) 
s_waitcnt vmcnt(24)
ds_write_b8 v37, v45 // (op 200) Store local data 

s_barrier  // (op 30240) 



ds_read_b64_tr_b4 v[24:25], v8 // (op 193) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v8 offset:1024 // (op 193) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 30282) 


v_add_u32 v37, 65536, v9 // (op 197) 
ds_read_u8 v38, v37 // (op 197) Load local data 




ds_read_b64_tr_b4 v[54:55], v8 offset:16 // (op 803) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v8 offset:1040 // (op 803) Transpose load local data 


v_add_u32 v37, 65568, v9 // (op 807) 
ds_read_u8 v58, v37 // (op 807) Load local data 




ds_read_b64_tr_b4 v[60:61], v8 offset:32 // (op 1101) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v8 offset:1056 // (op 1101) Transpose load local data 


v_add_u32 v37, 65600, v9 // (op 1105) 
ds_read_u8 v59, v37 // (op 1105) Load local data 




ds_read_b64_tr_b4 v[64:65], v8 offset:48 // (op 1399) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v8 offset:1072 // (op 1399) Transpose load local data 


v_add_u32 v37, 65632, v9 // (op 1403) 
ds_read_u8 v68, v37 // (op 1403) Load local data 

s_barrier  // (op 30221) 



ds_read_b64_tr_b4 v[70:71], v7 offset:8192 // (op 191) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v7 offset:9216 // (op 191) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 30261) 


v_add_u32 v37, 66048, v6 // (op 195) 
ds_read_u8 v69, v37 // (op 195) Load local data 




ds_read_b64_tr_b4 v[74:75], v7 offset:8208 // (op 1926) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:9232 // (op 1926) Transpose load local data 


v_add_u32 v37, 66080, v6 // (op 1929) 
ds_read_u8 v78, v37 // (op 1929) Load local data 




ds_read_b64_tr_b4 v[80:81], v7 offset:8224 // (op 3216) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v7 offset:9248 // (op 3216) Transpose load local data 


v_add_u32 v37, 66112, v6 // (op 3219) 
ds_read_u8 v79, v37 // (op 3219) Load local data 




ds_read_b64_tr_b4 v[84:85], v7 offset:8240 // (op 4506) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:9264 // (op 4506) Transpose load local data 


v_add_u32 v37, 66144, v6 // (op 4509) 
ds_read_u8 v88, v37 // (op 4509) Load local data 

s_ashr_i64 s[48:49], s[2:3], 7 // (op 3182) 
s_mov_b32 s52, s48 // (op 3182) convert

s_and_b32 s48, s52, 4294967292 // (op 3182) 

s_cmp_lt_u32 s1, s48 // (op 3182) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3182 // (op 3182) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3182 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3182:




























s_barrier  // (op 30601) 
v_and_b32 v24, -4, v1 // (op 27547) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27547) 
v_sub_u32 v25, v1, v24 // (op 27547) 

s_and_b32 s15, s18, 4294967292 // (op 27547) 
v_lshlrev_b32 v24, 5, v25 // (op 27547) 

s_add_u32 s6, s0, s15 // (op 27547) 

v_lshrrev_b32 v25, 2, v1 // (op 27547) 
s_mul_i32 s15, s16, s50 // (op 27547) 
s_add_u32 s15, s15, s6 // (op 27547) 

s_add_u32 s6, s15, s9 // (op 27547) 

s_mul_hi_u32 s15, s6, s13 // (op 27547) 
s_sub_u32 s7, s6, s15 // (op 27547) 
s_lshr_b32 s8, s7, 1 // (op 27547) 

s_add_u32 s7, s8, s15 // (op 27547) 


s_lshr_b32 s8, s7, s11 // (op 27547) 

s_and_b32 s7, s11, 2147483648 // (op 27547) 
s_cmp_eq_i32 s7, 0 // (op 27547) 
s_mov_b32 s15, scc // (op 27547) 

s_cmp_eq_i32 s15, 0 // (op 27547) 
s_cselect_b32 s7, s6, s8 // (op 27547) 


s_mul_hi_u32 s8, s7, s12 // (op 27547) 
s_sub_u32 s15, s7, s8 // (op 27547) 
s_lshr_b32 s28, s15, 1 // (op 27547) 

s_add_u32 s15, s28, s8 // (op 27547) 


s_lshr_b32 s8, s15, s10 // (op 27547) 

s_and_b32 s15, s10, 2147483648 // (op 27547) 
s_cmp_eq_i32 s15, 0 // (op 27547) 
s_mov_b32 s28, scc // (op 27547) 

s_cmp_eq_i32 s28, 0 // (op 27547) 
s_cselect_b32 s15, s7, s8 // (op 27547) 


s_mul_i32 s8, s7, s18 // (op 27547) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27547) 


s_sub_u32 s7, s6, s8 // (op 27547) 


v_mov_b32 v29, 0 // (op 27547) convert
v_mov_b32 v28, v26 // (op 27547) convert

v_lshl_add_u32 v24, s7, 7, v25 // (op 27547) 


v_mul_lo_u32 v27, v24, s31 // (op 27547) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27547) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 27547) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 27547) most significant: sum


v_add_co_u32 v24, s[18:19], v28, v26 // (op 27547) least significant half
v_addc_co_u32 v25, s[18:19], v29, v27, s[18:19] // (op 27547) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27547) 


v_lshrrev_b64 v[32:33], 3, v[24:25] // (op 27547) 

s_mul_i32 s19, s30, s3 // (op 27549) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27549) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27549) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27549) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27549) most significant: sum
s_add_u32 s19, s19, s6 // (op 27549) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27549) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27549) 

s_mov_b32 s68, s32 // (op 27549) reg expression
s_mov_b32 s69, s33 // (op 27549) reg expression
s_mov_b32 s70, s18 // (op 27549) reg expression
s_mov_b32 s71, 131072 // (op 27549) call()

v_and_b32 v24, -4, v1 // (op 27713) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27713) 
v_sub_u32 v25, v1, v24 // (op 27713) 

s_and_b32 s15, s18, 4294967292 // (op 27713) 
v_lshlrev_b32 v24, 5, v25 // (op 27713) 

s_add_u32 s6, s0, s15 // (op 27713) 

v_lshrrev_b32 v25, 2, v1 // (op 27713) 
s_mul_i32 s15, s16, s50 // (op 27713) 
s_add_u32 s15, s15, s6 // (op 27713) 

s_add_u32 s6, s15, s9 // (op 27713) 

s_mul_hi_u32 s15, s6, s13 // (op 27713) 
s_sub_u32 s7, s6, s15 // (op 27713) 
s_lshr_b32 s8, s7, 1 // (op 27713) 

s_add_u32 s7, s8, s15 // (op 27713) 


s_lshr_b32 s8, s7, s11 // (op 27713) 

s_and_b32 s7, s11, 2147483648 // (op 27713) 
s_cmp_eq_i32 s7, 0 // (op 27713) 
s_mov_b32 s15, scc // (op 27713) 

s_cmp_eq_i32 s15, 0 // (op 27713) 
s_cselect_b32 s7, s6, s8 // (op 27713) 


s_mul_hi_u32 s8, s7, s12 // (op 27713) 
s_sub_u32 s15, s7, s8 // (op 27713) 
s_lshr_b32 s28, s15, 1 // (op 27713) 

s_add_u32 s15, s28, s8 // (op 27713) 


s_lshr_b32 s8, s15, s10 // (op 27713) 

s_and_b32 s15, s10, 2147483648 // (op 27713) 
s_cmp_eq_i32 s15, 0 // (op 27713) 
s_mov_b32 s28, scc // (op 27713) 

s_cmp_eq_i32 s28, 0 // (op 27713) 
s_cselect_b32 s15, s7, s8 // (op 27713) 


s_mul_i32 s8, s15, s14 // (op 27713) 

s_sub_u32 s15, s7, s8 // (op 27713) 

s_mul_i32 s8, s7, s18 // (op 27713) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27713) 


s_sub_u32 s7, s6, s8 // (op 27713) 


v_mov_b32 v29, 0 // (op 27713) convert
v_mov_b32 v28, v26 // (op 27713) convert

v_lshl_add_u32 v24, s7, 7, v25 // (op 27713) 


v_mul_lo_u32 v27, v24, s35 // (op 27713) most significant: low of low * high
v_mul_hi_u32 v25, v24, s34 // (op 27713) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 27713) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 27713) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v28 // (op 27713) least significant half
v_addc_co_u32 v27, s[18:19], v27, v29, s[18:19] // (op 27713) most significant half



v_lshlrev_b64 v[24:25], 2, v[26:27] // (op 27713) 

v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 27713) 

s_mul_i32 s19, s34, s3 // (op 27715) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 27715) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 27715) least significant
s_mul_hi_u32 s6, s34, s2 // (op 27715) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27715) most significant: sum
s_add_u32 s19, s19, s6 // (op 27715) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27715) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27715) 

s_mov_b32 s72, s36 // (op 27715) reg expression
s_mov_b32 s73, s37 // (op 27715) reg expression
s_mov_b32 s74, s18 // (op 27715) reg expression
s_mov_b32 s75, 131072 // (op 27715) call()

v_and_b32 v24, -128, v1 // (op 27841) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27841) 
v_sub_u32 v25, v1, v24 // (op 27841) 

s_and_b32 s15, s18, 4294967292 // (op 27841) 
v_lshrrev_b32 v24, 7, v1 // (op 27841) 
s_add_u32 s6, s0, s15 // (op 27841) 

v_lshlrev_b32 v28, 1, v24 // (op 27841) 

s_mul_i32 s15, s16, s50 // (op 27841) 
s_add_u32 s15, s15, s6 // (op 27841) 

s_add_u32 s6, s15, s9 // (op 27841) 

s_mul_hi_u32 s15, s6, s13 // (op 27841) 
s_sub_u32 s7, s6, s15 // (op 27841) 
s_lshr_b32 s8, s7, 1 // (op 27841) 

s_add_u32 s7, s8, s15 // (op 27841) 


s_lshr_b32 s8, s7, s11 // (op 27841) 

s_and_b32 s7, s11, 2147483648 // (op 27841) 
s_cmp_eq_i32 s7, 0 // (op 27841) 
s_mov_b32 s15, scc // (op 27841) 

s_cmp_eq_i32 s15, 0 // (op 27841) 
s_cselect_b32 s7, s6, s8 // (op 27841) 


s_mul_hi_u32 s8, s7, s12 // (op 27841) 
s_sub_u32 s15, s7, s8 // (op 27841) 
s_lshr_b32 s28, s15, 1 // (op 27841) 

s_add_u32 s15, s28, s8 // (op 27841) 


s_lshr_b32 s8, s15, s10 // (op 27841) 

s_and_b32 s15, s10, 2147483648 // (op 27841) 
s_cmp_eq_i32 s15, 0 // (op 27841) 
s_mov_b32 s28, scc // (op 27841) 

s_cmp_eq_i32 s28, 0 // (op 27841) 
s_cselect_b32 s15, s7, s8 // (op 27841) 


s_mul_i32 s8, s7, s18 // (op 27841) 


v_lshl_add_u32 v24, s15, 7, v25 // (op 27841) 


s_sub_u32 s7, s6, s8 // (op 27841) 


v_lshl_add_u32 v25, s7, 2, v28 // (op 27841) 


v_mul_lo_u32 v28, v25, s38 // (op 27841) 

v_add_u32 v25, v24, v28 // (op 27841) 


s_mul_i32 s19, s38, s43 // (op 27843) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27843) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27843) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27843) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27843) most significant: sum
s_add_u32 s19, s19, s6 // (op 27843) most significant: sum


s_mov_b32 s76, s40 // (op 27843) reg expression
s_mov_b32 s77, s41 // (op 27843) reg expression
s_mov_b32 s78, s18 // (op 27843) reg expression
s_mov_b32 s79, 131072 // (op 27843) call()

v_and_b32 v24, -128, v1 // (op 27969) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27969) 
v_sub_u32 v28, v1, v24 // (op 27969) 

s_and_b32 s15, s18, 4294967292 // (op 27969) 
v_lshrrev_b32 v24, 7, v1 // (op 27969) 
s_add_u32 s6, s0, s15 // (op 27969) 

v_lshlrev_b32 v29, 1, v24 // (op 27969) 

s_mul_i32 s15, s16, s50 // (op 27969) 
s_add_u32 s15, s15, s6 // (op 27969) 

s_add_u32 s6, s15, s9 // (op 27969) 

s_mul_hi_u32 s15, s6, s13 // (op 27969) 
s_sub_u32 s7, s6, s15 // (op 27969) 
s_lshr_b32 s8, s7, 1 // (op 27969) 

s_add_u32 s7, s8, s15 // (op 27969) 


s_lshr_b32 s8, s7, s11 // (op 27969) 

s_and_b32 s7, s11, 2147483648 // (op 27969) 
s_cmp_eq_i32 s7, 0 // (op 27969) 
s_mov_b32 s15, scc // (op 27969) 

s_cmp_eq_i32 s15, 0 // (op 27969) 
s_cselect_b32 s7, s6, s8 // (op 27969) 


s_mul_hi_u32 s8, s7, s12 // (op 27969) 
s_sub_u32 s15, s7, s8 // (op 27969) 
s_lshr_b32 s28, s15, 1 // (op 27969) 

s_add_u32 s15, s28, s8 // (op 27969) 


s_lshr_b32 s8, s15, s10 // (op 27969) 

s_and_b32 s15, s10, 2147483648 // (op 27969) 
s_cmp_eq_i32 s15, 0 // (op 27969) 
s_mov_b32 s28, scc // (op 27969) 

s_cmp_eq_i32 s28, 0 // (op 27969) 
s_cselect_b32 s15, s7, s8 // (op 27969) 


s_mul_i32 s8, s15, s14 // (op 27969) 

s_sub_u32 s15, s7, s8 // (op 27969) 

s_mul_i32 s8, s7, s18 // (op 27969) 


v_lshl_add_u32 v24, s15, 7, v28 // (op 27969) 


s_sub_u32 s7, s6, s8 // (op 27969) 


v_lshl_add_u32 v28, s7, 2, v29 // (op 27969) 


v_mul_lo_u32 v29, v28, s44 // (op 27969) 
v_add_u32 v29, v29, v24 // (op 27969) 


s_mul_i32 s19, s44, s43 // (op 27971) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 27971) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 27971) least significant
s_mul_hi_u32 s6, s44, s42 // (op 27971) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27971) most significant: sum
s_add_u32 s19, s19, s6 // (op 27971) most significant: sum


s_mov_b32 s80, s46 // (op 27971) reg expression
s_mov_b32 s81, s47 // (op 27971) reg expression
s_mov_b32 s82, s18 // (op 27971) reg expression
s_mov_b32 s83, 131072 // (op 27971) call()

v_and_b32 v24, -4, v1 // (op 28516) 
v_sub_u32 v28, v1, v24 // (op 28516) 

v_lshlrev_b32 v24, 5, v28 // (op 28516) 

v_lshrrev_b32 v28, 2, v1 // (op 28516) 
v_lshl_add_u32 v36, v28, 7, v24 // (op 28516) 


v_lshlrev_b32 v24, 2, v36 // (op 28516) 

v_lshrrev_b32 v28, 3, v24 // (op 28516) 

s_nop 0
v_readfirstlane_b32 s15, v28 // (op 28516) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 28591) 
v_sub_u32 v28, v1, v24 // (op 28591) 

v_lshlrev_b32 v24, 5, v28 // (op 28591) 

v_lshrrev_b32 v28, 2, v1 // (op 28591) 
v_lshl_add_u32 v36, v28, 7, v24 // (op 28591) 


v_lshlrev_b32 v24, 2, v36 // (op 28591) 

v_lshrrev_b32 v28, 3, v24 // (op 28591) 

s_nop 0
v_readfirstlane_b32 s6, v28 // (op 28591) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -128, v1 // (op 28666) 
v_sub_u32 v28, v1, v24 // (op 28666) 

v_lshrrev_b32 v24, 7, v1 // (op 28666) 
v_lshl_add_u32 v36, v24, 8, v28 // (op 28666) 


v_and_b32 v24, -128, v1 // (op 28742) 
v_sub_u32 v28, v1, v24 // (op 28742) 

v_lshrrev_b32 v24, 7, v1 // (op 28742) 
v_lshl_add_u32 v37, v24, 8, v28 // (op 28742) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4478) 
s_and_b32 s7, s18, 4294967292 // (op 4478) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4477) 
s_mov_b32 s8, s18 // (op 4477) convert

s_cmp_lt_u32 s7, s8 // (op 4477) 

















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4477 // (op 4477) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4477 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4477:
s_barrier  // (op 30218) 
s_add_u32 s7, s7, 1 // (op 4479) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27406) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27406) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27406) 


v_mov_b64 v[38:39], v[32:33] // (op 27406) 
v_mov_b64 v[40:41], v[38:39] // (op 27406) reg expression
s_add_u32 s8, 8192, s15 // (op 27406) 
s_mov_b32 m0, s8 // (op 27406) reg expression
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 27406) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v30 // (op 27406) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 27406) most significant half

s_add_u32 m0, m0, 4096 // (op 27406) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 27406) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 27537) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27537) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27537) 

v_mov_b32 v24, s18 // (op 27537) 
v_mov_b32 v28, s19 // (op 27537) 
v_add_co_u32 v32, s[28:29], v32, v24 // (op 27537) least significant half
v_addc_co_u32 v33, s[28:29], v33, v28, s[28:29] // (op 27537) most significant half




s_barrier  // (op 30237) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27412) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27412) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27412) 


v_mov_b64 v[38:39], v[26:27] // (op 27412) 
v_mov_b64 v[40:41], v[38:39] // (op 27412) reg expression
s_mov_b32 m0, s6 // (op 27412) reg expression
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 27412) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v30 // (op 27412) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 27412) most significant half

s_add_u32 m0, m0, 4096 // (op 27412) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 27412) Load value direct to lds



s_lshl_b64 s[18:19], s[34:35], 7 // (op 27705) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27705) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27705) 

v_mov_b32 v24, s18 // (op 27705) 
v_mov_b32 v28, s19 // (op 27705) 
v_add_co_u32 v26, s[28:29], v26, v24 // (op 27705) least significant half
v_addc_co_u32 v27, s[28:29], v27, v28, s[28:29] // (op 27705) most significant half





v_mov_b64 v[30:31], s[38:39] // (op 477) convert

v_mov_b32 v24, v25 // (op 477) 
v_mov_b32 v28, v24 // (op 477) reg expression
buffer_load_ubyte v38, v28, s[76:79], 0 offen offset: 0 // (op 477) Load value
v_add_u32 v28, v28, v30 // (op 477) 
buffer_load_ubyte v39, v28, s[76:79], 0 offen offset: 0 // (op 477) Load value



s_lshl_b32 s8, s38, 2 // (op 27833) 
v_add_u32 v25, v25, s8 // (op 27833) 


s_barrier  // (op 30257) 
v_mov_b64 v[30:31], s[44:45] // (op 478) convert




v_mov_b32 v24, v29 // (op 478) 
v_mov_b32 v28, v24 // (op 478) reg expression
v_add_u32 v40, 66048, v36 // (op 474) 
s_waitcnt vmcnt(1)
ds_write_b8 v40, v38 // (op 474) Store local data 

v_add_u32 v40, 66176, v36 // (op 474) 
buffer_load_ubyte v42, v28, s[80:83], 0 offen offset: 0 // (op 478) Load value
v_add_u32 v28, v28, v30 // (op 478) 
buffer_load_ubyte v43, v28, s[80:83], 0 offen offset: 0 // (op 478) Load value



s_lshl_b32 s8, s44, 2 // (op 27961) 
v_add_u32 v29, v29, s8 // (op 27961) 

s_waitcnt vmcnt(2)
ds_write_b8 v40, v39 // (op 474) Store local data 


s_barrier  // (op 30224) 



s_barrier  // (op 30278) 
ds_read_b64_tr_b4 v[38:39], v7 offset:8192 // (op 465) Transpose load local data 



v_add_u32 v24, 65536, v37 // (op 475) 
ds_read_b64_tr_b4 v[40:41], v7 offset:9216 // (op 465) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v42 // (op 475) Store local data 

s_barrier  // (op 30244) 
v_add_u32 v24, 65664, v37 // (op 475) 



ds_read_b64_tr_b4 v[44:45], v8 // (op 467) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v8 offset:1024 // (op 467) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v43 // (op 475) Store local data 


s_waitcnt lgkmcnt(6)
s_barrier  // (op 30265) 


v_add_u32 v24, 66048, v6 // (op 469) 
ds_read_u8 v28, v24 // (op 469) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 30285) 


v_add_u32 v24, 65536, v9 // (op 471) 
ds_read_u8 v30, v24 // (op 471) Load local data 




ds_read_b64_tr_b4 v[48:49], v8 offset:16 // (op 782) Transpose load local data 
ds_read_b64_tr_b4 v[50:51], v8 offset:1040 // (op 782) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[44:47], a[0:3], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 472) 


v_add_u32 v24, 65568, v9 // (op 786) 
ds_read_u8 v31, v24 // (op 786) Load local data 




ds_read_b64_tr_b4 v[52:53], v8 offset:32 // (op 1080) Transpose load local data 
ds_read_b64_tr_b4 v[54:55], v8 offset:1056 // (op 1080) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:41], v[48:51], a[4:7], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 787) 


v_add_u32 v24, 65600, v9 // (op 1084) 
ds_read_u8 v42, v24 // (op 1084) Load local data 




ds_read_b64_tr_b4 v[56:57], v8 offset:48 // (op 1378) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v8 offset:1072 // (op 1378) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[52:55], a[8:11], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1085) 


v_add_u32 v24, 65632, v9 // (op 1382) 
ds_read_u8 v43, v24 // (op 1382) Load local data 




ds_read_b64_tr_b4 v[60:61], v7 offset:8208 // (op 1810) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v7 offset:9232 // (op 1810) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[56:59], a[12:15], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1383) 




v_add_u32 v24, 66080, v6 // (op 1813) 
ds_read_u8 v28, v24 // (op 1813) Load local data 




ds_read_b64_tr_b4 v[38:39], v7 offset:8224 // (op 3100) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v7 offset:9248 // (op 3100) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[44:47], a[16:19], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1815) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[48:51], a[20:23], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1846) 
v_add_u32 v24, 66112, v6 // (op 3103) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[52:55], a[24:27], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1877) 
ds_read_u8 v64, v24 // (op 3103) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[56:59], a[28:31], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1908) 
ds_read_b64_tr_b4 v[66:67], v7 offset:8240 // (op 4390) Transpose load local data 


ds_read_b64_tr_b4 v[68:69], v7 offset:9264 // (op 4390) Transpose load local data 


v_add_u32 v24, 66144, v6 // (op 4393) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:41], v[44:47], a[32:35], v64, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3105) 
ds_read_u8 v28, v24 // (op 4393) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[38:41], v[48:51], a[36:39], v64, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3136) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[38:41], v[52:55], a[40:43], v64, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3167) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[38:41], v[56:59], a[44:47], v64, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3198) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:69], v[44:47], a[48:51], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4395) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:69], v[48:51], a[52:55], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4426) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:69], v[52:55], a[56:59], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4457) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:69], v[56:59], a[60:63], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4488) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 4477) 
s_mov_b32 s8, s18 // (op 4477) convert

s_cmp_lt_u32 s7, s8 // (op 4477) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4477 // (op 4477) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4477 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4477:









s_barrier  // (op 30619) 




v_cmp_eq_f32 vcc, 0.00000, v34 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_238
v_accvgpr_read v22, a0 // (op 246) 
v_accvgpr_read v23, a1 // (op 246) 
v_accvgpr_read v24, a2 // (op 246) 
v_accvgpr_read v25, a3 // (op 246) 
v_mul_f32 v26, v35, v22 // (op 246) 
v_mul_f32 v27, v35, v23 // (op 246) 
v_mul_f32 v28, v35, v24 // (op 246) 
v_mul_f32 v29, v35, v25 // (op 246) 

v_lshrrev_b32 v22, 16, v26 // (op 248) 
v_lshrrev_b32 v23, 16, v27 // (op 248) 
v_lshrrev_b32 v24, 16, v28 // (op 248) 
v_lshrrev_b32 v25, 16, v29 // (op 248) 


v_pack_b32_f16 v26, v22, v23 // (op 249) 
v_pack_b32_f16 v27, v24, v25 // (op 249) 

v_mov_b32 v0, v5 // (op 249) reg expression

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 249) Store value



v_accvgpr_read v22, a4 // (op 6159) 
v_accvgpr_read v23, a5 // (op 6159) 
v_accvgpr_read v24, a6 // (op 6159) 
v_accvgpr_read v25, a7 // (op 6159) 
v_mul_f32 v26, v35, v22 // (op 6159) 
v_mul_f32 v27, v35, v23 // (op 6159) 
v_mul_f32 v28, v35, v24 // (op 6159) 
v_mul_f32 v29, v35, v25 // (op 6159) 

v_lshrrev_b32 v22, 16, v26 // (op 6161) 
v_lshrrev_b32 v23, 16, v27 // (op 6161) 
v_lshrrev_b32 v24, 16, v28 // (op 6161) 
v_lshrrev_b32 v25, 16, v29 // (op 6161) 


v_pack_b32_f16 v26, v22, v23 // (op 6162) 
v_pack_b32_f16 v27, v24, v25 // (op 6162) 

s_lshl_b32 s15, s24, 6 // (op 6162) 
v_add_u32 v0, v5, s15 // (op 6162) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6162) Store value



v_accvgpr_read v22, a8 // (op 6173) 
v_accvgpr_read v23, a9 // (op 6173) 
v_accvgpr_read v24, a10 // (op 6173) 
v_accvgpr_read v25, a11 // (op 6173) 
v_mul_f32 v26, v35, v22 // (op 6173) 
v_mul_f32 v27, v35, v23 // (op 6173) 
v_mul_f32 v28, v35, v24 // (op 6173) 
v_mul_f32 v29, v35, v25 // (op 6173) 

v_lshrrev_b32 v22, 16, v26 // (op 6175) 
v_lshrrev_b32 v23, 16, v27 // (op 6175) 
v_lshrrev_b32 v24, 16, v28 // (op 6175) 
v_lshrrev_b32 v25, 16, v29 // (op 6175) 


v_pack_b32_f16 v26, v22, v23 // (op 6176) 
v_pack_b32_f16 v27, v24, v25 // (op 6176) 

s_lshl_b32 s15, s24, 7 // (op 6176) 
v_add_u32 v0, v5, s15 // (op 6176) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6176) Store value



v_accvgpr_read v22, a12 // (op 6187) 
v_accvgpr_read v23, a13 // (op 6187) 
v_accvgpr_read v24, a14 // (op 6187) 
v_accvgpr_read v25, a15 // (op 6187) 
v_mul_f32 v26, v35, v22 // (op 6187) 
v_mul_f32 v27, v35, v23 // (op 6187) 
v_mul_f32 v28, v35, v24 // (op 6187) 
v_mul_f32 v29, v35, v25 // (op 6187) 

v_lshrrev_b32 v22, 16, v26 // (op 6189) 
v_lshrrev_b32 v23, 16, v27 // (op 6189) 
v_lshrrev_b32 v24, 16, v28 // (op 6189) 
v_lshrrev_b32 v25, 16, v29 // (op 6189) 


v_pack_b32_f16 v26, v22, v23 // (op 6190) 
v_pack_b32_f16 v27, v24, v25 // (op 6190) 

s_lshl_b32 s15, s24, 6 // (op 6190) 
s_mul_i32 s6, 3, s15 // (op 6190) 

v_add_u32 v0, v5, s6 // (op 6190) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6190) Store value



v_accvgpr_read v22, a16 // (op 6223) 
v_accvgpr_read v23, a17 // (op 6223) 
v_accvgpr_read v24, a18 // (op 6223) 
v_accvgpr_read v25, a19 // (op 6223) 
v_mul_f32 v26, v35, v22 // (op 6223) 
v_mul_f32 v27, v35, v23 // (op 6223) 
v_mul_f32 v28, v35, v24 // (op 6223) 
v_mul_f32 v29, v35, v25 // (op 6223) 

v_lshrrev_b32 v22, 16, v26 // (op 6225) 
v_lshrrev_b32 v23, 16, v27 // (op 6225) 
v_lshrrev_b32 v24, 16, v28 // (op 6225) 
v_lshrrev_b32 v25, 16, v29 // (op 6225) 


v_pack_b32_f16 v26, v22, v23 // (op 6227) 
v_pack_b32_f16 v27, v24, v25 // (op 6227) 

v_add_u32 v0, 64, v5 // (op 6227) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6227) Store value



v_accvgpr_read v22, a20 // (op 6237) 
v_accvgpr_read v23, a21 // (op 6237) 
v_accvgpr_read v24, a22 // (op 6237) 
v_accvgpr_read v25, a23 // (op 6237) 
v_mul_f32 v26, v35, v22 // (op 6237) 
v_mul_f32 v27, v35, v23 // (op 6237) 
v_mul_f32 v28, v35, v24 // (op 6237) 
v_mul_f32 v29, v35, v25 // (op 6237) 

v_lshrrev_b32 v22, 16, v26 // (op 6239) 
v_lshrrev_b32 v23, 16, v27 // (op 6239) 
v_lshrrev_b32 v24, 16, v28 // (op 6239) 
v_lshrrev_b32 v25, 16, v29 // (op 6239) 


v_pack_b32_f16 v26, v22, v23 // (op 6229) 
v_pack_b32_f16 v27, v24, v25 // (op 6229) 

s_lshl_b32 s15, s24, 6 // (op 6229) 
s_add_u32 s15, s15, 64 // (op 6229) 
v_add_u32 v0, v5, s15 // (op 6229) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6229) Store value



v_accvgpr_read v22, a24 // (op 6243) 
v_accvgpr_read v23, a25 // (op 6243) 
v_accvgpr_read v24, a26 // (op 6243) 
v_accvgpr_read v25, a27 // (op 6243) 
v_mul_f32 v26, v35, v22 // (op 6243) 
v_mul_f32 v27, v35, v23 // (op 6243) 
v_mul_f32 v28, v35, v24 // (op 6243) 
v_mul_f32 v29, v35, v25 // (op 6243) 

v_lshrrev_b32 v22, 16, v26 // (op 6245) 
v_lshrrev_b32 v23, 16, v27 // (op 6245) 
v_lshrrev_b32 v24, 16, v28 // (op 6245) 
v_lshrrev_b32 v25, 16, v29 // (op 6245) 


v_pack_b32_f16 v26, v22, v23 // (op 6231) 
v_pack_b32_f16 v27, v24, v25 // (op 6231) 

s_lshl_b32 s15, s24, 7 // (op 6231) 
s_add_u32 s15, s15, 64 // (op 6231) 
v_add_u32 v0, v5, s15 // (op 6231) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6231) Store value



v_accvgpr_read v22, a28 // (op 6249) 
v_accvgpr_read v23, a29 // (op 6249) 
v_accvgpr_read v24, a30 // (op 6249) 
v_accvgpr_read v25, a31 // (op 6249) 
v_mul_f32 v26, v35, v22 // (op 6249) 
v_mul_f32 v27, v35, v23 // (op 6249) 
v_mul_f32 v28, v35, v24 // (op 6249) 
v_mul_f32 v29, v35, v25 // (op 6249) 

v_lshrrev_b32 v22, 16, v26 // (op 6251) 
v_lshrrev_b32 v23, 16, v27 // (op 6251) 
v_lshrrev_b32 v24, 16, v28 // (op 6251) 
v_lshrrev_b32 v25, 16, v29 // (op 6251) 


v_pack_b32_f16 v26, v22, v23 // (op 6233) 
v_pack_b32_f16 v27, v24, v25 // (op 6233) 

s_lshl_b32 s15, s24, 6 // (op 6233) 
s_mul_i32 s6, 3, s15 // (op 6233) 
s_add_u32 s6, s6, 64 // (op 6233) 

v_add_u32 v0, v5, s6 // (op 6233) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6233) Store value



v_accvgpr_read v22, a32 // (op 6299) 
v_accvgpr_read v23, a33 // (op 6299) 
v_accvgpr_read v24, a34 // (op 6299) 
v_accvgpr_read v25, a35 // (op 6299) 
v_mul_f32 v26, v35, v22 // (op 6299) 
v_mul_f32 v27, v35, v23 // (op 6299) 
v_mul_f32 v28, v35, v24 // (op 6299) 
v_mul_f32 v29, v35, v25 // (op 6299) 

v_lshrrev_b32 v22, 16, v26 // (op 6301) 
v_lshrrev_b32 v23, 16, v27 // (op 6301) 
v_lshrrev_b32 v24, 16, v28 // (op 6301) 
v_lshrrev_b32 v25, 16, v29 // (op 6301) 


v_pack_b32_f16 v26, v22, v23 // (op 6303) 
v_pack_b32_f16 v27, v24, v25 // (op 6303) 

v_add_u32 v0, 128, v5 // (op 6303) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6303) Store value



v_accvgpr_read v22, a36 // (op 6313) 
v_accvgpr_read v23, a37 // (op 6313) 
v_accvgpr_read v24, a38 // (op 6313) 
v_accvgpr_read v25, a39 // (op 6313) 
v_mul_f32 v26, v35, v22 // (op 6313) 
v_mul_f32 v27, v35, v23 // (op 6313) 
v_mul_f32 v28, v35, v24 // (op 6313) 
v_mul_f32 v29, v35, v25 // (op 6313) 

v_lshrrev_b32 v22, 16, v26 // (op 6315) 
v_lshrrev_b32 v23, 16, v27 // (op 6315) 
v_lshrrev_b32 v24, 16, v28 // (op 6315) 
v_lshrrev_b32 v25, 16, v29 // (op 6315) 


v_pack_b32_f16 v26, v22, v23 // (op 6305) 
v_pack_b32_f16 v27, v24, v25 // (op 6305) 

s_lshl_b32 s15, s24, 6 // (op 6305) 
s_add_u32 s15, s15, 128 // (op 6305) 
v_add_u32 v0, v5, s15 // (op 6305) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6305) Store value



v_accvgpr_read v22, a40 // (op 6319) 
v_accvgpr_read v23, a41 // (op 6319) 
v_accvgpr_read v24, a42 // (op 6319) 
v_accvgpr_read v25, a43 // (op 6319) 
v_mul_f32 v26, v35, v22 // (op 6319) 
v_mul_f32 v27, v35, v23 // (op 6319) 
v_mul_f32 v28, v35, v24 // (op 6319) 
v_mul_f32 v29, v35, v25 // (op 6319) 

v_lshrrev_b32 v22, 16, v26 // (op 6321) 
v_lshrrev_b32 v23, 16, v27 // (op 6321) 
v_lshrrev_b32 v24, 16, v28 // (op 6321) 
v_lshrrev_b32 v25, 16, v29 // (op 6321) 


v_pack_b32_f16 v26, v22, v23 // (op 6307) 
v_pack_b32_f16 v27, v24, v25 // (op 6307) 

s_lshl_b32 s15, s24, 7 // (op 6307) 
s_add_u32 s15, s15, 128 // (op 6307) 
v_add_u32 v0, v5, s15 // (op 6307) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6307) Store value



v_accvgpr_read v22, a44 // (op 6325) 
v_accvgpr_read v23, a45 // (op 6325) 
v_accvgpr_read v24, a46 // (op 6325) 
v_accvgpr_read v25, a47 // (op 6325) 
v_mul_f32 v26, v35, v22 // (op 6325) 
v_mul_f32 v27, v35, v23 // (op 6325) 
v_mul_f32 v28, v35, v24 // (op 6325) 
v_mul_f32 v29, v35, v25 // (op 6325) 

v_lshrrev_b32 v22, 16, v26 // (op 6327) 
v_lshrrev_b32 v23, 16, v27 // (op 6327) 
v_lshrrev_b32 v24, 16, v28 // (op 6327) 
v_lshrrev_b32 v25, 16, v29 // (op 6327) 


v_pack_b32_f16 v26, v22, v23 // (op 6309) 
v_pack_b32_f16 v27, v24, v25 // (op 6309) 

s_lshl_b32 s15, s24, 6 // (op 6309) 
s_mul_i32 s6, 3, s15 // (op 6309) 
s_add_u32 s6, s6, 128 // (op 6309) 

v_add_u32 v0, v5, s6 // (op 6309) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6309) Store value



v_accvgpr_read v22, a48 // (op 6375) 
v_accvgpr_read v23, a49 // (op 6375) 
v_accvgpr_read v24, a50 // (op 6375) 
v_accvgpr_read v25, a51 // (op 6375) 
v_mul_f32 v26, v35, v22 // (op 6375) 
v_mul_f32 v27, v35, v23 // (op 6375) 
v_mul_f32 v28, v35, v24 // (op 6375) 
v_mul_f32 v29, v35, v25 // (op 6375) 

v_lshrrev_b32 v22, 16, v26 // (op 6377) 
v_lshrrev_b32 v23, 16, v27 // (op 6377) 
v_lshrrev_b32 v24, 16, v28 // (op 6377) 
v_lshrrev_b32 v25, 16, v29 // (op 6377) 


v_pack_b32_f16 v26, v22, v23 // (op 6379) 
v_pack_b32_f16 v27, v24, v25 // (op 6379) 

v_add_u32 v0, 192, v5 // (op 6379) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6379) Store value



v_accvgpr_read v22, a52 // (op 6389) 
v_accvgpr_read v23, a53 // (op 6389) 
v_accvgpr_read v24, a54 // (op 6389) 
v_accvgpr_read v25, a55 // (op 6389) 
v_mul_f32 v26, v35, v22 // (op 6389) 
v_mul_f32 v27, v35, v23 // (op 6389) 
v_mul_f32 v28, v35, v24 // (op 6389) 
v_mul_f32 v29, v35, v25 // (op 6389) 

v_lshrrev_b32 v22, 16, v26 // (op 6391) 
v_lshrrev_b32 v23, 16, v27 // (op 6391) 
v_lshrrev_b32 v24, 16, v28 // (op 6391) 
v_lshrrev_b32 v25, 16, v29 // (op 6391) 


v_pack_b32_f16 v26, v22, v23 // (op 6381) 
v_pack_b32_f16 v27, v24, v25 // (op 6381) 

s_lshl_b32 s15, s24, 6 // (op 6381) 
s_add_u32 s15, s15, 192 // (op 6381) 
v_add_u32 v0, v5, s15 // (op 6381) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6381) Store value



v_accvgpr_read v22, a56 // (op 6395) 
v_accvgpr_read v23, a57 // (op 6395) 
v_accvgpr_read v24, a58 // (op 6395) 
v_accvgpr_read v25, a59 // (op 6395) 
v_mul_f32 v26, v35, v22 // (op 6395) 
v_mul_f32 v27, v35, v23 // (op 6395) 
v_mul_f32 v28, v35, v24 // (op 6395) 
v_mul_f32 v29, v35, v25 // (op 6395) 

v_lshrrev_b32 v22, 16, v26 // (op 6397) 
v_lshrrev_b32 v23, 16, v27 // (op 6397) 
v_lshrrev_b32 v24, 16, v28 // (op 6397) 
v_lshrrev_b32 v25, 16, v29 // (op 6397) 


v_pack_b32_f16 v26, v22, v23 // (op 6383) 
v_pack_b32_f16 v27, v24, v25 // (op 6383) 

s_lshl_b32 s15, s24, 7 // (op 6383) 
s_add_u32 s15, s15, 192 // (op 6383) 
v_add_u32 v0, v5, s15 // (op 6383) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6383) Store value



v_accvgpr_read v22, a60 // (op 6401) 
v_accvgpr_read v23, a61 // (op 6401) 
v_accvgpr_read v24, a62 // (op 6401) 
v_accvgpr_read v25, a63 // (op 6401) 
v_mul_f32 v26, v35, v22 // (op 6401) 
v_mul_f32 v27, v35, v23 // (op 6401) 
v_mul_f32 v28, v35, v24 // (op 6401) 
v_mul_f32 v29, v35, v25 // (op 6401) 

v_lshrrev_b32 v22, 16, v26 // (op 6403) 
v_lshrrev_b32 v23, 16, v27 // (op 6403) 
v_lshrrev_b32 v24, 16, v28 // (op 6403) 
v_lshrrev_b32 v25, 16, v29 // (op 6403) 


v_pack_b32_f16 v26, v22, v23 // (op 6385) 
v_pack_b32_f16 v27, v24, v25 // (op 6385) 

s_lshl_b32 s15, s24, 6 // (op 6385) 
s_mul_i32 s6, 3, s15 // (op 6385) 
s_add_u32 s6, s6, 192 // (op 6385) 

v_add_u32 v0, v5, s6 // (op 6385) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6385) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_238
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_238:


v_add_u32 v0, 192, v4 // (op 6773) 

buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6773) Load value

v_accvgpr_read v24, a48 // (op 6760) 
v_accvgpr_read v25, a49 // (op 6760) 
v_accvgpr_read v26, a50 // (op 6760) 
v_accvgpr_read v27, a51 // (op 6760) 
v_mul_f32 v28, v35, v24 // (op 6760) 
v_mul_f32 v29, v35, v25 // (op 6760) 
v_mul_f32 v30, v35, v26 // (op 6760) 
v_mul_f32 v31, v35, v27 // (op 6760) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6772) 
v_lshrrev_b32 v33, 16, v22 // (op 6772) 
v_lshlrev_b32 v32, 16, v32 // (op 6772) 
v_lshlrev_b32 v33, 16, v33 // (op 6772) 
v_mul_f32 v24, v34, v32 // (op 6772) 
v_mul_f32 v25, v34, v33 // (op 6772) 
v_and_b32 v32, 65535, v23 // (op 6772) 
v_lshrrev_b32 v33, 16, v23 // (op 6772) 
v_lshlrev_b32 v32, 16, v32 // (op 6772) 
v_lshlrev_b32 v33, 16, v33 // (op 6772) 
v_mul_f32 v26, v34, v32 // (op 6772) 
v_mul_f32 v27, v34, v33 // (op 6772) 


v_add_f32 v36, v24, v28 // (op 6761) 
v_add_f32 v37, v25, v29 // (op 6761) 
v_add_f32 v38, v26, v30 // (op 6761) 
v_add_f32 v39, v27, v31 // (op 6761) 


v_lshrrev_b32 v22, 16, v36 // (op 6762) 
v_lshrrev_b32 v23, 16, v37 // (op 6762) 
v_lshrrev_b32 v24, 16, v38 // (op 6762) 
v_lshrrev_b32 v25, 16, v39 // (op 6762) 


v_pack_b32_f16 v26, v22, v23 // (op 6764) 
v_pack_b32_f16 v27, v24, v25 // (op 6764) 

v_add_u32 v0, 192, v5 // (op 6764) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6764) Store value





s_lshl_b32 s15, s4, 6 // (op 6778) 
s_add_u32 s15, s15, 192 // (op 6778) 
v_add_u32 v0, v4, s15 // (op 6778) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6778) Load value

v_accvgpr_read v24, a52 // (op 6790) 
v_accvgpr_read v25, a53 // (op 6790) 
v_accvgpr_read v26, a54 // (op 6790) 
v_accvgpr_read v27, a55 // (op 6790) 
v_mul_f32 v28, v35, v24 // (op 6790) 
v_mul_f32 v29, v35, v25 // (op 6790) 
v_mul_f32 v30, v35, v26 // (op 6790) 
v_mul_f32 v31, v35, v27 // (op 6790) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6775) 
v_lshrrev_b32 v33, 16, v22 // (op 6775) 
v_lshlrev_b32 v32, 16, v32 // (op 6775) 
v_lshlrev_b32 v33, 16, v33 // (op 6775) 
v_mul_f32 v24, v34, v32 // (op 6775) 
v_mul_f32 v25, v34, v33 // (op 6775) 
v_and_b32 v32, 65535, v23 // (op 6775) 
v_lshrrev_b32 v33, 16, v23 // (op 6775) 
v_lshlrev_b32 v32, 16, v32 // (op 6775) 
v_lshlrev_b32 v33, 16, v33 // (op 6775) 
v_mul_f32 v26, v34, v32 // (op 6775) 
v_mul_f32 v27, v34, v33 // (op 6775) 


v_add_f32 v36, v24, v28 // (op 6776) 
v_add_f32 v37, v25, v29 // (op 6776) 
v_add_f32 v38, v26, v30 // (op 6776) 
v_add_f32 v39, v27, v31 // (op 6776) 


v_lshrrev_b32 v22, 16, v36 // (op 6777) 
v_lshrrev_b32 v23, 16, v37 // (op 6777) 
v_lshrrev_b32 v24, 16, v38 // (op 6777) 
v_lshrrev_b32 v25, 16, v39 // (op 6777) 


v_pack_b32_f16 v26, v22, v23 // (op 6766) 
v_pack_b32_f16 v27, v24, v25 // (op 6766) 

s_lshl_b32 s15, s24, 6 // (op 6766) 
s_add_u32 s15, s15, 192 // (op 6766) 
v_add_u32 v0, v5, s15 // (op 6766) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6766) Store value





s_lshl_b32 s15, s4, 7 // (op 6783) 
s_add_u32 s15, s15, 192 // (op 6783) 
v_add_u32 v0, v4, s15 // (op 6783) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6783) Load value

v_accvgpr_read v24, a56 // (op 6792) 
v_accvgpr_read v25, a57 // (op 6792) 
v_accvgpr_read v26, a58 // (op 6792) 
v_accvgpr_read v27, a59 // (op 6792) 
v_mul_f32 v28, v35, v24 // (op 6792) 
v_mul_f32 v29, v35, v25 // (op 6792) 
v_mul_f32 v30, v35, v26 // (op 6792) 
v_mul_f32 v31, v35, v27 // (op 6792) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6780) 
v_lshrrev_b32 v33, 16, v22 // (op 6780) 
v_lshlrev_b32 v32, 16, v32 // (op 6780) 
v_lshlrev_b32 v33, 16, v33 // (op 6780) 
v_mul_f32 v24, v34, v32 // (op 6780) 
v_mul_f32 v25, v34, v33 // (op 6780) 
v_and_b32 v32, 65535, v23 // (op 6780) 
v_lshrrev_b32 v33, 16, v23 // (op 6780) 
v_lshlrev_b32 v32, 16, v32 // (op 6780) 
v_lshlrev_b32 v33, 16, v33 // (op 6780) 
v_mul_f32 v26, v34, v32 // (op 6780) 
v_mul_f32 v27, v34, v33 // (op 6780) 


v_add_f32 v36, v24, v28 // (op 6781) 
v_add_f32 v37, v25, v29 // (op 6781) 
v_add_f32 v38, v26, v30 // (op 6781) 
v_add_f32 v39, v27, v31 // (op 6781) 


v_lshrrev_b32 v22, 16, v36 // (op 6782) 
v_lshrrev_b32 v23, 16, v37 // (op 6782) 
v_lshrrev_b32 v24, 16, v38 // (op 6782) 
v_lshrrev_b32 v25, 16, v39 // (op 6782) 


v_pack_b32_f16 v26, v22, v23 // (op 6768) 
v_pack_b32_f16 v27, v24, v25 // (op 6768) 

s_lshl_b32 s15, s24, 7 // (op 6768) 
s_add_u32 s15, s15, 192 // (op 6768) 
v_add_u32 v0, v5, s15 // (op 6768) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6768) Store value





s_lshl_b32 s15, s4, 6 // (op 6788) 
s_mul_i32 s6, 3, s15 // (op 6788) 
s_add_u32 s6, s6, 192 // (op 6788) 

v_add_u32 v0, v4, s6 // (op 6788) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6788) Load value

v_accvgpr_read v24, a60 // (op 6794) 
v_accvgpr_read v25, a61 // (op 6794) 
v_accvgpr_read v26, a62 // (op 6794) 
v_accvgpr_read v27, a63 // (op 6794) 
v_mul_f32 v28, v35, v24 // (op 6794) 
v_mul_f32 v29, v35, v25 // (op 6794) 
v_mul_f32 v30, v35, v26 // (op 6794) 
v_mul_f32 v31, v35, v27 // (op 6794) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6785) 
v_lshrrev_b32 v33, 16, v22 // (op 6785) 
v_lshlrev_b32 v32, 16, v32 // (op 6785) 
v_lshlrev_b32 v33, 16, v33 // (op 6785) 
v_mul_f32 v24, v34, v32 // (op 6785) 
v_mul_f32 v25, v34, v33 // (op 6785) 
v_and_b32 v32, 65535, v23 // (op 6785) 
v_lshrrev_b32 v33, 16, v23 // (op 6785) 
v_lshlrev_b32 v32, 16, v32 // (op 6785) 
v_lshlrev_b32 v33, 16, v33 // (op 6785) 
v_mul_f32 v26, v34, v32 // (op 6785) 
v_mul_f32 v27, v34, v33 // (op 6785) 


v_add_f32 v36, v24, v28 // (op 6786) 
v_add_f32 v37, v25, v29 // (op 6786) 
v_add_f32 v38, v26, v30 // (op 6786) 
v_add_f32 v39, v27, v31 // (op 6786) 


v_lshrrev_b32 v22, 16, v36 // (op 6787) 
v_lshrrev_b32 v23, 16, v37 // (op 6787) 
v_lshrrev_b32 v24, 16, v38 // (op 6787) 
v_lshrrev_b32 v25, 16, v39 // (op 6787) 


v_pack_b32_f16 v26, v22, v23 // (op 6770) 
v_pack_b32_f16 v27, v24, v25 // (op 6770) 

s_lshl_b32 s15, s24, 6 // (op 6770) 
s_mul_i32 s6, 3, s15 // (op 6770) 
s_add_u32 s6, s6, 192 // (op 6770) 

v_add_u32 v0, v5, s6 // (op 6770) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6770) Store value





v_add_u32 v0, 128, v4 // (op 6686) 

buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6686) Load value

v_accvgpr_read v24, a32 // (op 6673) 
v_accvgpr_read v25, a33 // (op 6673) 
v_accvgpr_read v26, a34 // (op 6673) 
v_accvgpr_read v27, a35 // (op 6673) 
v_mul_f32 v28, v35, v24 // (op 6673) 
v_mul_f32 v29, v35, v25 // (op 6673) 
v_mul_f32 v30, v35, v26 // (op 6673) 
v_mul_f32 v31, v35, v27 // (op 6673) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6685) 
v_lshrrev_b32 v33, 16, v22 // (op 6685) 
v_lshlrev_b32 v32, 16, v32 // (op 6685) 
v_lshlrev_b32 v33, 16, v33 // (op 6685) 
v_mul_f32 v24, v34, v32 // (op 6685) 
v_mul_f32 v25, v34, v33 // (op 6685) 
v_and_b32 v32, 65535, v23 // (op 6685) 
v_lshrrev_b32 v33, 16, v23 // (op 6685) 
v_lshlrev_b32 v32, 16, v32 // (op 6685) 
v_lshlrev_b32 v33, 16, v33 // (op 6685) 
v_mul_f32 v26, v34, v32 // (op 6685) 
v_mul_f32 v27, v34, v33 // (op 6685) 


v_add_f32 v36, v24, v28 // (op 6674) 
v_add_f32 v37, v25, v29 // (op 6674) 
v_add_f32 v38, v26, v30 // (op 6674) 
v_add_f32 v39, v27, v31 // (op 6674) 


v_lshrrev_b32 v22, 16, v36 // (op 6675) 
v_lshrrev_b32 v23, 16, v37 // (op 6675) 
v_lshrrev_b32 v24, 16, v38 // (op 6675) 
v_lshrrev_b32 v25, 16, v39 // (op 6675) 


v_pack_b32_f16 v26, v22, v23 // (op 6677) 
v_pack_b32_f16 v27, v24, v25 // (op 6677) 

v_add_u32 v0, 128, v5 // (op 6677) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6677) Store value





s_lshl_b32 s15, s4, 6 // (op 6691) 
s_add_u32 s15, s15, 128 // (op 6691) 
v_add_u32 v0, v4, s15 // (op 6691) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6691) Load value

v_accvgpr_read v24, a36 // (op 6703) 
v_accvgpr_read v25, a37 // (op 6703) 
v_accvgpr_read v26, a38 // (op 6703) 
v_accvgpr_read v27, a39 // (op 6703) 
v_mul_f32 v28, v35, v24 // (op 6703) 
v_mul_f32 v29, v35, v25 // (op 6703) 
v_mul_f32 v30, v35, v26 // (op 6703) 
v_mul_f32 v31, v35, v27 // (op 6703) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6688) 
v_lshrrev_b32 v33, 16, v22 // (op 6688) 
v_lshlrev_b32 v32, 16, v32 // (op 6688) 
v_lshlrev_b32 v33, 16, v33 // (op 6688) 
v_mul_f32 v24, v34, v32 // (op 6688) 
v_mul_f32 v25, v34, v33 // (op 6688) 
v_and_b32 v32, 65535, v23 // (op 6688) 
v_lshrrev_b32 v33, 16, v23 // (op 6688) 
v_lshlrev_b32 v32, 16, v32 // (op 6688) 
v_lshlrev_b32 v33, 16, v33 // (op 6688) 
v_mul_f32 v26, v34, v32 // (op 6688) 
v_mul_f32 v27, v34, v33 // (op 6688) 


v_add_f32 v36, v24, v28 // (op 6689) 
v_add_f32 v37, v25, v29 // (op 6689) 
v_add_f32 v38, v26, v30 // (op 6689) 
v_add_f32 v39, v27, v31 // (op 6689) 


v_lshrrev_b32 v22, 16, v36 // (op 6690) 
v_lshrrev_b32 v23, 16, v37 // (op 6690) 
v_lshrrev_b32 v24, 16, v38 // (op 6690) 
v_lshrrev_b32 v25, 16, v39 // (op 6690) 


v_pack_b32_f16 v26, v22, v23 // (op 6679) 
v_pack_b32_f16 v27, v24, v25 // (op 6679) 

s_lshl_b32 s15, s24, 6 // (op 6679) 
s_add_u32 s15, s15, 128 // (op 6679) 
v_add_u32 v0, v5, s15 // (op 6679) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6679) Store value





s_lshl_b32 s15, s4, 7 // (op 6696) 
s_add_u32 s15, s15, 128 // (op 6696) 
v_add_u32 v0, v4, s15 // (op 6696) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6696) Load value

v_accvgpr_read v24, a40 // (op 6705) 
v_accvgpr_read v25, a41 // (op 6705) 
v_accvgpr_read v26, a42 // (op 6705) 
v_accvgpr_read v27, a43 // (op 6705) 
v_mul_f32 v28, v35, v24 // (op 6705) 
v_mul_f32 v29, v35, v25 // (op 6705) 
v_mul_f32 v30, v35, v26 // (op 6705) 
v_mul_f32 v31, v35, v27 // (op 6705) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6693) 
v_lshrrev_b32 v33, 16, v22 // (op 6693) 
v_lshlrev_b32 v32, 16, v32 // (op 6693) 
v_lshlrev_b32 v33, 16, v33 // (op 6693) 
v_mul_f32 v24, v34, v32 // (op 6693) 
v_mul_f32 v25, v34, v33 // (op 6693) 
v_and_b32 v32, 65535, v23 // (op 6693) 
v_lshrrev_b32 v33, 16, v23 // (op 6693) 
v_lshlrev_b32 v32, 16, v32 // (op 6693) 
v_lshlrev_b32 v33, 16, v33 // (op 6693) 
v_mul_f32 v26, v34, v32 // (op 6693) 
v_mul_f32 v27, v34, v33 // (op 6693) 


v_add_f32 v36, v24, v28 // (op 6694) 
v_add_f32 v37, v25, v29 // (op 6694) 
v_add_f32 v38, v26, v30 // (op 6694) 
v_add_f32 v39, v27, v31 // (op 6694) 


v_lshrrev_b32 v22, 16, v36 // (op 6695) 
v_lshrrev_b32 v23, 16, v37 // (op 6695) 
v_lshrrev_b32 v24, 16, v38 // (op 6695) 
v_lshrrev_b32 v25, 16, v39 // (op 6695) 


v_pack_b32_f16 v26, v22, v23 // (op 6681) 
v_pack_b32_f16 v27, v24, v25 // (op 6681) 

s_lshl_b32 s15, s24, 7 // (op 6681) 
s_add_u32 s15, s15, 128 // (op 6681) 
v_add_u32 v0, v5, s15 // (op 6681) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6681) Store value





s_lshl_b32 s15, s4, 6 // (op 6701) 
s_mul_i32 s6, 3, s15 // (op 6701) 
s_add_u32 s6, s6, 128 // (op 6701) 

v_add_u32 v0, v4, s6 // (op 6701) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6701) Load value

v_accvgpr_read v24, a44 // (op 6707) 
v_accvgpr_read v25, a45 // (op 6707) 
v_accvgpr_read v26, a46 // (op 6707) 
v_accvgpr_read v27, a47 // (op 6707) 
v_mul_f32 v28, v35, v24 // (op 6707) 
v_mul_f32 v29, v35, v25 // (op 6707) 
v_mul_f32 v30, v35, v26 // (op 6707) 
v_mul_f32 v31, v35, v27 // (op 6707) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6698) 
v_lshrrev_b32 v33, 16, v22 // (op 6698) 
v_lshlrev_b32 v32, 16, v32 // (op 6698) 
v_lshlrev_b32 v33, 16, v33 // (op 6698) 
v_mul_f32 v24, v34, v32 // (op 6698) 
v_mul_f32 v25, v34, v33 // (op 6698) 
v_and_b32 v32, 65535, v23 // (op 6698) 
v_lshrrev_b32 v33, 16, v23 // (op 6698) 
v_lshlrev_b32 v32, 16, v32 // (op 6698) 
v_lshlrev_b32 v33, 16, v33 // (op 6698) 
v_mul_f32 v26, v34, v32 // (op 6698) 
v_mul_f32 v27, v34, v33 // (op 6698) 


v_add_f32 v36, v24, v28 // (op 6699) 
v_add_f32 v37, v25, v29 // (op 6699) 
v_add_f32 v38, v26, v30 // (op 6699) 
v_add_f32 v39, v27, v31 // (op 6699) 


v_lshrrev_b32 v22, 16, v36 // (op 6700) 
v_lshrrev_b32 v23, 16, v37 // (op 6700) 
v_lshrrev_b32 v24, 16, v38 // (op 6700) 
v_lshrrev_b32 v25, 16, v39 // (op 6700) 


v_pack_b32_f16 v26, v22, v23 // (op 6683) 
v_pack_b32_f16 v27, v24, v25 // (op 6683) 

s_lshl_b32 s15, s24, 6 // (op 6683) 
s_mul_i32 s6, 3, s15 // (op 6683) 
s_add_u32 s6, s6, 128 // (op 6683) 

v_add_u32 v0, v5, s6 // (op 6683) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6683) Store value





v_add_u32 v0, 64, v4 // (op 6599) 

buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6599) Load value

v_accvgpr_read v24, a16 // (op 6586) 
v_accvgpr_read v25, a17 // (op 6586) 
v_accvgpr_read v26, a18 // (op 6586) 
v_accvgpr_read v27, a19 // (op 6586) 
v_mul_f32 v28, v35, v24 // (op 6586) 
v_mul_f32 v29, v35, v25 // (op 6586) 
v_mul_f32 v30, v35, v26 // (op 6586) 
v_mul_f32 v31, v35, v27 // (op 6586) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6598) 
v_lshrrev_b32 v33, 16, v22 // (op 6598) 
v_lshlrev_b32 v32, 16, v32 // (op 6598) 
v_lshlrev_b32 v33, 16, v33 // (op 6598) 
v_mul_f32 v24, v34, v32 // (op 6598) 
v_mul_f32 v25, v34, v33 // (op 6598) 
v_and_b32 v32, 65535, v23 // (op 6598) 
v_lshrrev_b32 v33, 16, v23 // (op 6598) 
v_lshlrev_b32 v32, 16, v32 // (op 6598) 
v_lshlrev_b32 v33, 16, v33 // (op 6598) 
v_mul_f32 v26, v34, v32 // (op 6598) 
v_mul_f32 v27, v34, v33 // (op 6598) 


v_add_f32 v36, v24, v28 // (op 6587) 
v_add_f32 v37, v25, v29 // (op 6587) 
v_add_f32 v38, v26, v30 // (op 6587) 
v_add_f32 v39, v27, v31 // (op 6587) 


v_lshrrev_b32 v22, 16, v36 // (op 6588) 
v_lshrrev_b32 v23, 16, v37 // (op 6588) 
v_lshrrev_b32 v24, 16, v38 // (op 6588) 
v_lshrrev_b32 v25, 16, v39 // (op 6588) 


v_pack_b32_f16 v26, v22, v23 // (op 6590) 
v_pack_b32_f16 v27, v24, v25 // (op 6590) 

v_add_u32 v0, 64, v5 // (op 6590) 

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6590) Store value





s_lshl_b32 s15, s4, 6 // (op 6604) 
s_add_u32 s15, s15, 64 // (op 6604) 
v_add_u32 v0, v4, s15 // (op 6604) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6604) Load value

v_accvgpr_read v24, a20 // (op 6616) 
v_accvgpr_read v25, a21 // (op 6616) 
v_accvgpr_read v26, a22 // (op 6616) 
v_accvgpr_read v27, a23 // (op 6616) 
v_mul_f32 v28, v35, v24 // (op 6616) 
v_mul_f32 v29, v35, v25 // (op 6616) 
v_mul_f32 v30, v35, v26 // (op 6616) 
v_mul_f32 v31, v35, v27 // (op 6616) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6601) 
v_lshrrev_b32 v33, 16, v22 // (op 6601) 
v_lshlrev_b32 v32, 16, v32 // (op 6601) 
v_lshlrev_b32 v33, 16, v33 // (op 6601) 
v_mul_f32 v24, v34, v32 // (op 6601) 
v_mul_f32 v25, v34, v33 // (op 6601) 
v_and_b32 v32, 65535, v23 // (op 6601) 
v_lshrrev_b32 v33, 16, v23 // (op 6601) 
v_lshlrev_b32 v32, 16, v32 // (op 6601) 
v_lshlrev_b32 v33, 16, v33 // (op 6601) 
v_mul_f32 v26, v34, v32 // (op 6601) 
v_mul_f32 v27, v34, v33 // (op 6601) 


v_add_f32 v36, v24, v28 // (op 6602) 
v_add_f32 v37, v25, v29 // (op 6602) 
v_add_f32 v38, v26, v30 // (op 6602) 
v_add_f32 v39, v27, v31 // (op 6602) 


v_lshrrev_b32 v22, 16, v36 // (op 6603) 
v_lshrrev_b32 v23, 16, v37 // (op 6603) 
v_lshrrev_b32 v24, 16, v38 // (op 6603) 
v_lshrrev_b32 v25, 16, v39 // (op 6603) 


v_pack_b32_f16 v26, v22, v23 // (op 6592) 
v_pack_b32_f16 v27, v24, v25 // (op 6592) 

s_lshl_b32 s15, s24, 6 // (op 6592) 
s_add_u32 s15, s15, 64 // (op 6592) 
v_add_u32 v0, v5, s15 // (op 6592) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6592) Store value





s_lshl_b32 s15, s4, 7 // (op 6609) 
s_add_u32 s15, s15, 64 // (op 6609) 
v_add_u32 v0, v4, s15 // (op 6609) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6609) Load value

v_accvgpr_read v24, a24 // (op 6618) 
v_accvgpr_read v25, a25 // (op 6618) 
v_accvgpr_read v26, a26 // (op 6618) 
v_accvgpr_read v27, a27 // (op 6618) 
v_mul_f32 v28, v35, v24 // (op 6618) 
v_mul_f32 v29, v35, v25 // (op 6618) 
v_mul_f32 v30, v35, v26 // (op 6618) 
v_mul_f32 v31, v35, v27 // (op 6618) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6606) 
v_lshrrev_b32 v33, 16, v22 // (op 6606) 
v_lshlrev_b32 v32, 16, v32 // (op 6606) 
v_lshlrev_b32 v33, 16, v33 // (op 6606) 
v_mul_f32 v24, v34, v32 // (op 6606) 
v_mul_f32 v25, v34, v33 // (op 6606) 
v_and_b32 v32, 65535, v23 // (op 6606) 
v_lshrrev_b32 v33, 16, v23 // (op 6606) 
v_lshlrev_b32 v32, 16, v32 // (op 6606) 
v_lshlrev_b32 v33, 16, v33 // (op 6606) 
v_mul_f32 v26, v34, v32 // (op 6606) 
v_mul_f32 v27, v34, v33 // (op 6606) 


v_add_f32 v36, v24, v28 // (op 6607) 
v_add_f32 v37, v25, v29 // (op 6607) 
v_add_f32 v38, v26, v30 // (op 6607) 
v_add_f32 v39, v27, v31 // (op 6607) 


v_lshrrev_b32 v22, 16, v36 // (op 6608) 
v_lshrrev_b32 v23, 16, v37 // (op 6608) 
v_lshrrev_b32 v24, 16, v38 // (op 6608) 
v_lshrrev_b32 v25, 16, v39 // (op 6608) 


v_pack_b32_f16 v26, v22, v23 // (op 6594) 
v_pack_b32_f16 v27, v24, v25 // (op 6594) 

s_lshl_b32 s15, s24, 7 // (op 6594) 
s_add_u32 s15, s15, 64 // (op 6594) 
v_add_u32 v0, v5, s15 // (op 6594) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6594) Store value





s_lshl_b32 s15, s4, 6 // (op 6614) 
s_mul_i32 s6, 3, s15 // (op 6614) 
s_add_u32 s6, s6, 64 // (op 6614) 

v_add_u32 v0, v4, s6 // (op 6614) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6614) Load value

v_accvgpr_read v24, a28 // (op 6620) 
v_accvgpr_read v25, a29 // (op 6620) 
v_accvgpr_read v26, a30 // (op 6620) 
v_accvgpr_read v27, a31 // (op 6620) 
v_mul_f32 v28, v35, v24 // (op 6620) 
v_mul_f32 v29, v35, v25 // (op 6620) 
v_mul_f32 v30, v35, v26 // (op 6620) 
v_mul_f32 v31, v35, v27 // (op 6620) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6611) 
v_lshrrev_b32 v33, 16, v22 // (op 6611) 
v_lshlrev_b32 v32, 16, v32 // (op 6611) 
v_lshlrev_b32 v33, 16, v33 // (op 6611) 
v_mul_f32 v24, v34, v32 // (op 6611) 
v_mul_f32 v25, v34, v33 // (op 6611) 
v_and_b32 v32, 65535, v23 // (op 6611) 
v_lshrrev_b32 v33, 16, v23 // (op 6611) 
v_lshlrev_b32 v32, 16, v32 // (op 6611) 
v_lshlrev_b32 v33, 16, v33 // (op 6611) 
v_mul_f32 v26, v34, v32 // (op 6611) 
v_mul_f32 v27, v34, v33 // (op 6611) 


v_add_f32 v36, v24, v28 // (op 6612) 
v_add_f32 v37, v25, v29 // (op 6612) 
v_add_f32 v38, v26, v30 // (op 6612) 
v_add_f32 v39, v27, v31 // (op 6612) 


v_lshrrev_b32 v22, 16, v36 // (op 6613) 
v_lshrrev_b32 v23, 16, v37 // (op 6613) 
v_lshrrev_b32 v24, 16, v38 // (op 6613) 
v_lshrrev_b32 v25, 16, v39 // (op 6613) 


v_pack_b32_f16 v26, v22, v23 // (op 6596) 
v_pack_b32_f16 v27, v24, v25 // (op 6596) 

s_lshl_b32 s15, s24, 6 // (op 6596) 
s_mul_i32 s6, 3, s15 // (op 6596) 
s_add_u32 s6, s6, 64 // (op 6596) 

v_add_u32 v0, v5, s6 // (op 6596) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6596) Store value





v_mov_b32 v0, v4 // (op 263) reg expression

buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 263) Load value

v_accvgpr_read v24, a0 // (op 259) 
v_accvgpr_read v25, a1 // (op 259) 
v_accvgpr_read v26, a2 // (op 259) 
v_accvgpr_read v27, a3 // (op 259) 
v_mul_f32 v28, v35, v24 // (op 259) 
v_mul_f32 v29, v35, v25 // (op 259) 
v_mul_f32 v30, v35, v26 // (op 259) 
v_mul_f32 v31, v35, v27 // (op 259) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 264) 
v_lshrrev_b32 v33, 16, v22 // (op 264) 
v_lshlrev_b32 v32, 16, v32 // (op 264) 
v_lshlrev_b32 v33, 16, v33 // (op 264) 
v_mul_f32 v24, v34, v32 // (op 264) 
v_mul_f32 v25, v34, v33 // (op 264) 
v_and_b32 v32, 65535, v23 // (op 264) 
v_lshrrev_b32 v33, 16, v23 // (op 264) 
v_lshlrev_b32 v32, 16, v32 // (op 264) 
v_lshlrev_b32 v33, 16, v33 // (op 264) 
v_mul_f32 v26, v34, v32 // (op 264) 
v_mul_f32 v27, v34, v33 // (op 264) 


v_add_f32 v36, v24, v28 // (op 260) 
v_add_f32 v37, v25, v29 // (op 260) 
v_add_f32 v38, v26, v30 // (op 260) 
v_add_f32 v39, v27, v31 // (op 260) 


v_lshrrev_b32 v22, 16, v36 // (op 261) 
v_lshrrev_b32 v23, 16, v37 // (op 261) 
v_lshrrev_b32 v24, 16, v38 // (op 261) 
v_lshrrev_b32 v25, 16, v39 // (op 261) 


v_pack_b32_f16 v26, v22, v23 // (op 262) 
v_pack_b32_f16 v27, v24, v25 // (op 262) 

v_mov_b32 v0, v5 // (op 262) reg expression

buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 262) Store value





s_lshl_b32 s15, s4, 6 // (op 6507) 
v_add_u32 v0, v4, s15 // (op 6507) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6507) Load value

v_accvgpr_read v24, a4 // (op 6503) 
v_accvgpr_read v25, a5 // (op 6503) 
v_accvgpr_read v26, a6 // (op 6503) 
v_accvgpr_read v27, a7 // (op 6503) 
v_mul_f32 v28, v35, v24 // (op 6503) 
v_mul_f32 v29, v35, v25 // (op 6503) 
v_mul_f32 v30, v35, v26 // (op 6503) 
v_mul_f32 v31, v35, v27 // (op 6503) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6508) 
v_lshrrev_b32 v33, 16, v22 // (op 6508) 
v_lshlrev_b32 v32, 16, v32 // (op 6508) 
v_lshlrev_b32 v33, 16, v33 // (op 6508) 
v_mul_f32 v24, v34, v32 // (op 6508) 
v_mul_f32 v25, v34, v33 // (op 6508) 
v_and_b32 v32, 65535, v23 // (op 6508) 
v_lshrrev_b32 v33, 16, v23 // (op 6508) 
v_lshlrev_b32 v32, 16, v32 // (op 6508) 
v_lshlrev_b32 v33, 16, v33 // (op 6508) 
v_mul_f32 v26, v34, v32 // (op 6508) 
v_mul_f32 v27, v34, v33 // (op 6508) 


v_add_f32 v36, v24, v28 // (op 6504) 
v_add_f32 v37, v25, v29 // (op 6504) 
v_add_f32 v38, v26, v30 // (op 6504) 
v_add_f32 v39, v27, v31 // (op 6504) 


v_lshrrev_b32 v22, 16, v36 // (op 6505) 
v_lshrrev_b32 v23, 16, v37 // (op 6505) 
v_lshrrev_b32 v24, 16, v38 // (op 6505) 
v_lshrrev_b32 v25, 16, v39 // (op 6505) 


v_pack_b32_f16 v26, v22, v23 // (op 6506) 
v_pack_b32_f16 v27, v24, v25 // (op 6506) 

s_lshl_b32 s15, s24, 6 // (op 6506) 
v_add_u32 v0, v5, s15 // (op 6506) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6506) Store value





s_lshl_b32 s15, s4, 7 // (op 6521) 
v_add_u32 v0, v4, s15 // (op 6521) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6521) Load value

v_accvgpr_read v24, a8 // (op 6517) 
v_accvgpr_read v25, a9 // (op 6517) 
v_accvgpr_read v26, a10 // (op 6517) 
v_accvgpr_read v27, a11 // (op 6517) 
v_mul_f32 v28, v35, v24 // (op 6517) 
v_mul_f32 v29, v35, v25 // (op 6517) 
v_mul_f32 v30, v35, v26 // (op 6517) 
v_mul_f32 v31, v35, v27 // (op 6517) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6522) 
v_lshrrev_b32 v33, 16, v22 // (op 6522) 
v_lshlrev_b32 v32, 16, v32 // (op 6522) 
v_lshlrev_b32 v33, 16, v33 // (op 6522) 
v_mul_f32 v24, v34, v32 // (op 6522) 
v_mul_f32 v25, v34, v33 // (op 6522) 
v_and_b32 v32, 65535, v23 // (op 6522) 
v_lshrrev_b32 v33, 16, v23 // (op 6522) 
v_lshlrev_b32 v32, 16, v32 // (op 6522) 
v_lshlrev_b32 v33, 16, v33 // (op 6522) 
v_mul_f32 v26, v34, v32 // (op 6522) 
v_mul_f32 v27, v34, v33 // (op 6522) 


v_add_f32 v36, v24, v28 // (op 6518) 
v_add_f32 v37, v25, v29 // (op 6518) 
v_add_f32 v38, v26, v30 // (op 6518) 
v_add_f32 v39, v27, v31 // (op 6518) 


v_lshrrev_b32 v22, 16, v36 // (op 6519) 
v_lshrrev_b32 v23, 16, v37 // (op 6519) 
v_lshrrev_b32 v24, 16, v38 // (op 6519) 
v_lshrrev_b32 v25, 16, v39 // (op 6519) 


v_pack_b32_f16 v26, v22, v23 // (op 6520) 
v_pack_b32_f16 v27, v24, v25 // (op 6520) 

s_lshl_b32 s15, s24, 7 // (op 6520) 
v_add_u32 v0, v5, s15 // (op 6520) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6520) Store value





s_lshl_b32 s15, s4, 6 // (op 6535) 
s_mul_i32 s6, 3, s15 // (op 6535) 

v_add_u32 v0, v4, s6 // (op 6535) 


buffer_load_dwordx2 v[22:23], v0, s[56:59], 0 offen // (op 6535) Load value

v_accvgpr_read v24, a12 // (op 6531) 
v_accvgpr_read v25, a13 // (op 6531) 
v_accvgpr_read v26, a14 // (op 6531) 
v_accvgpr_read v27, a15 // (op 6531) 
v_mul_f32 v28, v35, v24 // (op 6531) 
v_mul_f32 v29, v35, v25 // (op 6531) 
v_mul_f32 v30, v35, v26 // (op 6531) 
v_mul_f32 v31, v35, v27 // (op 6531) 

s_waitcnt vmcnt(0)
v_and_b32 v32, 65535, v22 // (op 6536) 
v_lshrrev_b32 v33, 16, v22 // (op 6536) 
v_lshlrev_b32 v32, 16, v32 // (op 6536) 
v_lshlrev_b32 v33, 16, v33 // (op 6536) 
v_mul_f32 v24, v34, v32 // (op 6536) 
v_mul_f32 v25, v34, v33 // (op 6536) 
v_and_b32 v32, 65535, v23 // (op 6536) 
v_lshrrev_b32 v33, 16, v23 // (op 6536) 
v_lshlrev_b32 v32, 16, v32 // (op 6536) 
v_lshlrev_b32 v33, 16, v33 // (op 6536) 
v_mul_f32 v26, v34, v32 // (op 6536) 
v_mul_f32 v27, v34, v33 // (op 6536) 


v_add_f32 v36, v24, v28 // (op 6532) 
v_add_f32 v37, v25, v29 // (op 6532) 
v_add_f32 v38, v26, v30 // (op 6532) 
v_add_f32 v39, v27, v31 // (op 6532) 


v_lshrrev_b32 v22, 16, v36 // (op 6533) 
v_lshrrev_b32 v23, 16, v37 // (op 6533) 
v_lshrrev_b32 v24, 16, v38 // (op 6533) 
v_lshrrev_b32 v25, 16, v39 // (op 6533) 


v_pack_b32_f16 v26, v22, v23 // (op 6534) 
v_pack_b32_f16 v27, v24, v25 // (op 6534) 

s_lshl_b32 s15, s24, 6 // (op 6534) 
s_mul_i32 s6, 3, s15 // (op 6534) 

v_add_u32 v0, v5, s6 // (op 6534) 


buffer_store_dwordx2 v[26:27], v0, s[60:63], 0 offen // (op 6534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_238:
s_add_u32 s0, s0, s1 // (op 451) 


s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_449:

































































s_endpgm  // End of RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end:
.size RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4, .LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end-RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
  .amdhsa_next_free_vgpr 208
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
  .amdhsa_accum_offset 144
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
    args: [ { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideC0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: uint32, semantic: RR_SKNumSKTilesPerWG },
            { type: uint32, semantic: RR_SKNumSKTiles },
            { type: int32, semantic: RR_SKMagicShiftsNumTilesN },
            { type: int32, semantic: RR_SKMagicShiftsIters },
            { type: uint32, semantic: RR_SKMagicMultipleNumTilesN },
            { type: uint32, semantic: RR_SKMagicMultipleIters },
            { type: uint32, semantic: RR_SKNumTilesN },
            { type: uint32, semantic: RR_SKNumWGs },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: address, semantic: RR_SKAddressPartials },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressFlags },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 2
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 4
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
    .symbol: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 93
    .vgpr_count: 144
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_18
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: numSKTilesPerWG
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: numSKTiles
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicShifts_31
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShifts_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: numTiles1
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: numWGs
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_19
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_14
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_0_16
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 168
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 176
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 180
        .value_kind: by_value
...

.end_amdgpu_metadata

