// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.p2align 8
.type RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4,@function
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 28071) 
v_and_b32 v3, 48, v1 // (op 28071) 
v_sub_u32 v4, v0, v3 // (op 28071) 

v_lshrrev_b32 v3, 4, v0 // (op 28071) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 28071) 


v_lshrrev_b32 v3, 7, v1 // (op 28071) 
v_lshlrev_b32 v4, 4, v3 // (op 28071) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 28071) 


v_lshlrev_b32 v0, 2, v3 // (op 28071) 

v_lshrrev_b32 v3, 3, v0 // (op 28071) 

v_lshrrev_b32 v0, 6, v1 // (op 28146) 
v_and_b32 v4, 1, v0 // (op 28146) 

v_and_b32 v0, 15, v1 // (op 28146) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 28146) 


v_and_b32 v0, 63, v1 // (op 28146) 
v_lshrrev_b32 v4, 4, v0 // (op 28146) 

v_lshlrev_b32 v0, 5, v4 // (op 28146) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 28146) 


v_lshlrev_b32 v0, 2, v4 // (op 28146) 

v_lshrrev_b32 v4, 3, v0 // (op 28146) 

v_lshrrev_b32 v0, 7, v1 // (op 28222) 
v_and_b32 v5, 15, v1 // (op 28222) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 28222) 


v_and_b32 v0, 63, v1 // (op 28222) 
v_lshrrev_b32 v5, 4, v0 // (op 28222) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 28222) 


v_lshrrev_b32 v5, 6, v1 // (op 28298) 
v_and_b32 v6, 1, v5 // (op 28298) 

v_and_b32 v5, 15, v1 // (op 28298) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 28298) 


v_and_b32 v5, 63, v1 // (op 28298) 
v_lshrrev_b32 v6, 4, v5 // (op 28298) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 28298) 


v_and_b32 v6, 63, v1 // (op 28425) 
v_and_b32 v7, 48, v1 // (op 28425) 
v_sub_u32 v8, v6, v7 // (op 28425) 

v_lshrrev_b32 v7, 4, v6 // (op 28425) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 28425) 


v_lshrrev_b32 v7, 7, v1 // (op 28425) 
v_lshlrev_b32 v8, 4, v7 // (op 28425) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 28425) 


v_lshlrev_b32 v6, 2, v7 // (op 28425) 

v_lshrrev_b32 v7, 3, v6 // (op 28425) 

v_lshrrev_b32 v6, 6, v1 // (op 28500) 
v_and_b32 v8, 1, v6 // (op 28500) 

v_and_b32 v6, 15, v1 // (op 28500) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 28500) 


v_and_b32 v6, 63, v1 // (op 28500) 
v_lshrrev_b32 v8, 4, v6 // (op 28500) 

v_lshlrev_b32 v6, 5, v8 // (op 28500) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 28500) 


v_lshlrev_b32 v6, 2, v8 // (op 28500) 

v_lshrrev_b32 v8, 3, v6 // (op 28500) 

v_lshrrev_b32 v6, 7, v1 // (op 28576) 
v_and_b32 v9, 15, v1 // (op 28576) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 28576) 


v_and_b32 v6, 63, v1 // (op 28576) 
v_lshrrev_b32 v9, 4, v6 // (op 28576) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 28576) 


v_lshrrev_b32 v9, 6, v1 // (op 28652) 
v_and_b32 v10, 1, v9 // (op 28652) 

v_and_b32 v9, 15, v1 // (op 28652) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 28652) 


v_and_b32 v9, 63, v1 // (op 28652) 
v_lshrrev_b32 v10, 4, v9 // (op 28652) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 28652) 


v_and_b32 v10, 63, v1 // (op 28766) 
v_and_b32 v11, 48, v1 // (op 28766) 
v_sub_u32 v12, v10, v11 // (op 28766) 

v_lshrrev_b32 v11, 4, v10 // (op 28766) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 28766) 


v_lshrrev_b32 v11, 7, v1 // (op 28766) 
v_lshlrev_b32 v12, 4, v11 // (op 28766) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 28766) 


v_lshlrev_b32 v10, 2, v11 // (op 28766) 

v_lshrrev_b32 v11, 3, v10 // (op 28766) 

v_lshrrev_b32 v10, 6, v1 // (op 28810) 
v_and_b32 v12, 1, v10 // (op 28810) 

v_and_b32 v10, 15, v1 // (op 28810) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 28810) 


v_and_b32 v10, 63, v1 // (op 28810) 
v_lshrrev_b32 v12, 4, v10 // (op 28810) 

v_lshlrev_b32 v10, 5, v12 // (op 28810) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 28810) 


v_lshlrev_b32 v10, 2, v12 // (op 28810) 

v_lshrrev_b32 v12, 3, v10 // (op 28810) 

v_lshrrev_b32 v10, 7, v1 // (op 28839) 
v_and_b32 v13, 15, v1 // (op 28839) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 28839) 


v_and_b32 v10, 63, v1 // (op 28839) 
v_lshrrev_b32 v13, 4, v10 // (op 28839) 

v_lshl_add_u32 v10, v13, 7, v14 // (op 28839) 


v_lshrrev_b32 v13, 6, v1 // (op 28868) 
v_and_b32 v14, 1, v13 // (op 28868) 

v_and_b32 v13, 15, v1 // (op 28868) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 28868) 


v_and_b32 v13, 63, v1 // (op 28868) 
v_lshrrev_b32 v14, 4, v13 // (op 28868) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 28868) 


v_and_b32 v14, 63, v1 // (op 28912) 
v_and_b32 v15, 48, v1 // (op 28912) 
v_sub_u32 v16, v14, v15 // (op 28912) 

v_lshrrev_b32 v15, 4, v14 // (op 28912) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 28912) 


v_lshrrev_b32 v15, 7, v1 // (op 28912) 
v_lshlrev_b32 v16, 4, v15 // (op 28912) 

v_lshl_add_u32 v15, v14, 7, v16 // (op 28912) 


v_lshlrev_b32 v14, 2, v15 // (op 28912) 

v_lshrrev_b32 v15, 3, v14 // (op 28912) 

v_lshrrev_b32 v14, 6, v1 // (op 28956) 
v_and_b32 v16, 1, v14 // (op 28956) 

v_and_b32 v14, 15, v1 // (op 28956) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 28956) 


v_and_b32 v14, 63, v1 // (op 28956) 
v_lshrrev_b32 v16, 4, v14 // (op 28956) 

v_lshlrev_b32 v14, 5, v16 // (op 28956) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 28956) 


v_lshlrev_b32 v14, 2, v16 // (op 28956) 

v_lshrrev_b32 v16, 3, v14 // (op 28956) 

v_lshrrev_b32 v14, 7, v1 // (op 28985) 
v_and_b32 v17, 15, v1 // (op 28985) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 28985) 


v_and_b32 v14, 63, v1 // (op 28985) 
v_lshrrev_b32 v17, 4, v14 // (op 28985) 

v_lshl_add_u32 v14, v17, 7, v18 // (op 28985) 


v_lshrrev_b32 v17, 6, v1 // (op 29014) 
v_and_b32 v18, 1, v17 // (op 29014) 

v_and_b32 v17, 15, v1 // (op 29014) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 29014) 


v_and_b32 v17, 63, v1 // (op 29014) 
v_lshrrev_b32 v18, 4, v17 // (op 29014) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 29014) 


v_and_b32 v18, 63, v1 // (op 29058) 
v_and_b32 v19, 48, v1 // (op 29058) 
v_sub_u32 v20, v18, v19 // (op 29058) 

v_lshrrev_b32 v19, 4, v18 // (op 29058) 

v_lshl_add_u32 v18, v19, 5, v20 // (op 29058) 


v_lshrrev_b32 v19, 7, v1 // (op 29058) 
v_lshlrev_b32 v20, 4, v19 // (op 29058) 

v_lshl_add_u32 v19, v18, 7, v20 // (op 29058) 


v_lshlrev_b32 v18, 2, v19 // (op 29058) 

v_lshrrev_b32 v19, 3, v18 // (op 29058) 

v_lshrrev_b32 v18, 6, v1 // (op 29102) 
v_and_b32 v20, 1, v18 // (op 29102) 

v_and_b32 v18, 15, v1 // (op 29102) 
v_lshl_add_u32 v21, v20, 4, v18 // (op 29102) 


v_and_b32 v18, 63, v1 // (op 29102) 
v_lshrrev_b32 v20, 4, v18 // (op 29102) 

v_lshlrev_b32 v18, 5, v20 // (op 29102) 

v_lshl_add_u32 v20, v21, 7, v18 // (op 29102) 


v_lshlrev_b32 v18, 2, v20 // (op 29102) 

v_lshrrev_b32 v20, 3, v18 // (op 29102) 

v_lshrrev_b32 v18, 7, v1 // (op 29131) 
v_and_b32 v21, 15, v1 // (op 29131) 
v_lshl_add_u32 v22, v18, 4, v21 // (op 29131) 


v_and_b32 v18, 63, v1 // (op 29131) 
v_lshrrev_b32 v21, 4, v18 // (op 29131) 

v_lshl_add_u32 v18, v21, 7, v22 // (op 29131) 


v_lshrrev_b32 v21, 6, v1 // (op 29160) 
v_and_b32 v22, 1, v21 // (op 29160) 

v_and_b32 v21, 15, v1 // (op 29160) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 29160) 


v_and_b32 v21, 63, v1 // (op 29160) 
v_lshrrev_b32 v22, 4, v21 // (op 29160) 

v_lshl_add_u32 v21, v23, 2, v22 // (op 29160) 


v_and_b32 v22, 63, v1 // (op 29204) 
v_and_b32 v23, 48, v1 // (op 29204) 
v_sub_u32 v24, v22, v23 // (op 29204) 

v_lshrrev_b32 v23, 4, v22 // (op 29204) 

v_lshl_add_u32 v22, v23, 5, v24 // (op 29204) 


v_lshrrev_b32 v23, 7, v1 // (op 29204) 
v_lshlrev_b32 v24, 4, v23 // (op 29204) 

v_lshl_add_u32 v23, v22, 7, v24 // (op 29204) 


v_lshlrev_b32 v22, 2, v23 // (op 29204) 

v_lshrrev_b32 v23, 3, v22 // (op 29204) 

v_lshrrev_b32 v22, 6, v1 // (op 29248) 
v_and_b32 v24, 1, v22 // (op 29248) 

v_and_b32 v22, 15, v1 // (op 29248) 
v_lshl_add_u32 v25, v24, 4, v22 // (op 29248) 


v_and_b32 v22, 63, v1 // (op 29248) 
v_lshrrev_b32 v24, 4, v22 // (op 29248) 

v_lshlrev_b32 v22, 5, v24 // (op 29248) 

v_lshl_add_u32 v24, v25, 7, v22 // (op 29248) 


v_lshlrev_b32 v22, 2, v24 // (op 29248) 

v_lshrrev_b32 v24, 3, v22 // (op 29248) 

v_lshrrev_b32 v22, 7, v1 // (op 29277) 
v_and_b32 v25, 15, v1 // (op 29277) 
v_lshl_add_u32 v26, v22, 4, v25 // (op 29277) 


v_and_b32 v22, 63, v1 // (op 29277) 
v_lshrrev_b32 v25, 4, v22 // (op 29277) 

v_lshl_add_u32 v22, v25, 7, v26 // (op 29277) 


v_lshrrev_b32 v25, 6, v1 // (op 29306) 
v_and_b32 v26, 1, v25 // (op 29306) 

v_and_b32 v25, 15, v1 // (op 29306) 
v_lshl_add_u32 v27, v26, 4, v25 // (op 29306) 


v_and_b32 v25, 63, v1 // (op 29306) 
v_lshrrev_b32 v26, 4, v25 // (op 29306) 

v_lshl_add_u32 v25, v27, 2, v26 // (op 29306) 


v_and_b32 v26, 63, v1 // (op 29350) 
v_and_b32 v27, 48, v1 // (op 29350) 
v_sub_u32 v28, v26, v27 // (op 29350) 

v_lshrrev_b32 v27, 4, v26 // (op 29350) 

v_lshl_add_u32 v26, v27, 5, v28 // (op 29350) 


v_lshrrev_b32 v27, 7, v1 // (op 29350) 
v_lshlrev_b32 v28, 4, v27 // (op 29350) 

v_lshl_add_u32 v27, v26, 7, v28 // (op 29350) 


v_lshlrev_b32 v26, 2, v27 // (op 29350) 

v_lshrrev_b32 v27, 3, v26 // (op 29350) 

v_lshrrev_b32 v26, 6, v1 // (op 29394) 
v_and_b32 v28, 1, v26 // (op 29394) 

v_and_b32 v26, 15, v1 // (op 29394) 
v_lshl_add_u32 v29, v28, 4, v26 // (op 29394) 


v_and_b32 v26, 63, v1 // (op 29394) 
v_lshrrev_b32 v28, 4, v26 // (op 29394) 

v_lshlrev_b32 v26, 5, v28 // (op 29394) 

v_lshl_add_u32 v28, v29, 7, v26 // (op 29394) 


v_lshlrev_b32 v26, 2, v28 // (op 29394) 

v_lshrrev_b32 v28, 3, v26 // (op 29394) 

v_lshrrev_b32 v26, 7, v1 // (op 29423) 
v_and_b32 v29, 15, v1 // (op 29423) 
v_lshl_add_u32 v30, v26, 4, v29 // (op 29423) 


v_and_b32 v26, 63, v1 // (op 29423) 
v_lshrrev_b32 v29, 4, v26 // (op 29423) 

v_lshl_add_u32 v26, v29, 7, v30 // (op 29423) 


v_lshrrev_b32 v29, 6, v1 // (op 29452) 
v_and_b32 v30, 1, v29 // (op 29452) 

v_and_b32 v29, 15, v1 // (op 29452) 
v_lshl_add_u32 v31, v30, 4, v29 // (op 29452) 


v_and_b32 v29, 63, v1 // (op 29452) 
v_lshrrev_b32 v30, 4, v29 // (op 29452) 

v_lshl_add_u32 v29, v31, 2, v30 // (op 29452) 


v_and_b32 v30, 63, v1 // (op 29496) 
v_and_b32 v31, 48, v1 // (op 29496) 
v_sub_u32 v32, v30, v31 // (op 29496) 

v_lshrrev_b32 v31, 4, v30 // (op 29496) 

v_lshl_add_u32 v30, v31, 5, v32 // (op 29496) 


v_lshrrev_b32 v31, 7, v1 // (op 29496) 
v_lshlrev_b32 v32, 4, v31 // (op 29496) 

v_lshl_add_u32 v31, v30, 7, v32 // (op 29496) 


v_lshlrev_b32 v30, 2, v31 // (op 29496) 

v_lshrrev_b32 v31, 3, v30 // (op 29496) 

v_lshrrev_b32 v30, 6, v1 // (op 29540) 
v_and_b32 v32, 1, v30 // (op 29540) 

v_and_b32 v30, 15, v1 // (op 29540) 
v_lshl_add_u32 v33, v32, 4, v30 // (op 29540) 


v_and_b32 v30, 63, v1 // (op 29540) 
v_lshrrev_b32 v32, 4, v30 // (op 29540) 

v_lshlrev_b32 v30, 5, v32 // (op 29540) 

v_lshl_add_u32 v32, v33, 7, v30 // (op 29540) 


v_lshlrev_b32 v30, 2, v32 // (op 29540) 

v_lshrrev_b32 v32, 3, v30 // (op 29540) 

v_lshrrev_b32 v30, 7, v1 // (op 29569) 
v_and_b32 v33, 15, v1 // (op 29569) 
v_lshl_add_u32 v34, v30, 4, v33 // (op 29569) 


v_and_b32 v30, 63, v1 // (op 29569) 
v_lshrrev_b32 v33, 4, v30 // (op 29569) 

v_lshl_add_u32 v30, v33, 7, v34 // (op 29569) 


v_lshrrev_b32 v33, 6, v1 // (op 29598) 
v_and_b32 v34, 1, v33 // (op 29598) 

v_and_b32 v33, 15, v1 // (op 29598) 
v_lshl_add_u32 v35, v34, 4, v33 // (op 29598) 


v_and_b32 v33, 63, v1 // (op 29598) 
v_lshrrev_b32 v34, 4, v33 // (op 29598) 

v_lshl_add_u32 v33, v35, 2, v34 // (op 29598) 


v_mov_b32 v34, s6 // (op 2) Move value
v_mov_b32 v35, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 387) call()
s_mov_b32 s1, 0 // (op 388) call()
s_mov_b32 s6, 0 // (op 389) call()
s_mov_b32 s7, 0 // (op 390) call()
s_mov_b32 s18, 0 // (op 391) call()
s_mov_b32 s19, 0 // (op 392) call()
s_mov_b32 s0, 0 // (op 287) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v36, 7, v1 // (op 27982) 
s_mul_i32 s52, s16, s8 // (op 27982) 
s_add_u32 s52, s52, s0 // (op 27982) 
v_and_b32 v37, 63, v1 // (op 27982) 
s_mul_hi_u32 s53, s52, s13 // (op 27982) 
v_lshrrev_b32 v38, 4, v37 // (op 27982) 
s_sub_u32 s54, s52, s53 // (op 27982) 
v_lshlrev_b32 v39, 2, v38 // (op 27982) 

v_lshl_add_u32 v38, v36, 4, v39 // (op 27982) 


s_lshr_b32 s55, s54, 1 // (op 27982) 

s_add_u32 s54, s55, s53 // (op 27982) 


v_lshrrev_b32 v36, 2, v37 // (op 27982) 

s_lshr_b32 s53, s54, s11 // (op 27982) 

v_and_b32 v37, 3, v36 // (op 27982) 

s_and_b32 s54, s11, 2147483648 // (op 27982) 
v_and_b32 v36, 3, v1 // (op 27982) 
s_cmp_eq_i32 s54, 0 // (op 27982) 
s_mov_b32 s55, scc // (op 27982) 

v_lshl_add_u32 v39, v37, 2, v36 // (op 27982) 


s_cmp_eq_i32 s55, 0 // (op 27982) 
s_cselect_b32 s54, s52, s53 // (op 27982) 



v_lshrrev_b32 v36, 6, v1 // (op 27982) 
s_mul_hi_u32 s52, s54, s12 // (op 27982) 
v_and_b32 v37, 1, v36 // (op 27982) 

s_sub_u32 s53, s54, s52 // (op 27982) 
v_lshl_add_u32 v36, v37, 4, v39 // (op 27982) 


s_lshr_b32 s55, s53, 1 // (op 27982) 

s_add_u32 s53, s55, s52 // (op 27982) 


s_lshr_b32 s52, s53, s10 // (op 27982) 

s_and_b32 s53, s10, 2147483648 // (op 27982) 
s_cmp_eq_i32 s53, 0 // (op 27982) 
s_mov_b32 s55, scc // (op 27982) 

s_cmp_eq_i32 s55, 0 // (op 27982) 
s_cselect_b32 s53, s54, s52 // (op 27982) 


s_mul_i32 s52, s53, s14 // (op 27982) 
v_lshl_add_u32 v37, s53, 7, v38 // (op 27982) 


s_sub_u32 s53, s54, s52 // (op 27982) 


v_lshl_add_u32 v38, s53, 7, v36 // (op 27982) 


v_mul_lo_u32 v36, v38, s4 // (op 27982) 

v_add_lshl_u32 v38, v37, v36, 2 // (op 27982) 


s_mul_i32 s53, s4, s23 // (op 27984) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 27984) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 27984) least significant
s_mul_hi_u32 s55, s4, s22 // (op 27984) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27984) most significant: sum
s_add_u32 s53, s53, s55 // (op 27984) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27984) 

s_mov_b32 s56, s20 // (op 27984) reg expression
s_mov_b32 s57, s21 // (op 27984) reg expression
s_mov_b32 s58, s54 // (op 27984) reg expression
s_mov_b32 s59, 131072 // (op 27984) call()

v_and_b32 v36, 63, v1 // (op 28088) 
v_and_b32 v37, 48, v1 // (op 28088) 
v_sub_u32 v39, v36, v37 // (op 28088) 

v_lshrrev_b32 v37, 4, v36 // (op 28088) 

v_lshl_add_u32 v36, v37, 5, v39 // (op 28088) 


v_lshrrev_b32 v37, 7, v1 // (op 28088) 
v_lshlrev_b32 v39, 4, v37 // (op 28088) 

v_lshl_add_u32 v37, v36, 7, v39 // (op 28088) 


v_lshlrev_b32 v36, 2, v37 // (op 28088) 

v_lshrrev_b32 v37, 3, v36 // (op 28088) 

v_lshrrev_b32 v36, 6, v1 // (op 28163) 
v_and_b32 v39, 1, v36 // (op 28163) 

v_and_b32 v36, 15, v1 // (op 28163) 
v_lshl_add_u32 v40, v39, 4, v36 // (op 28163) 


v_and_b32 v36, 63, v1 // (op 28163) 
v_lshrrev_b32 v39, 4, v36 // (op 28163) 

v_lshlrev_b32 v36, 5, v39 // (op 28163) 

v_lshl_add_u32 v39, v40, 7, v36 // (op 28163) 


v_lshlrev_b32 v36, 2, v39 // (op 28163) 

v_lshrrev_b32 v39, 3, v36 // (op 28163) 

v_lshrrev_b32 v36, 7, v1 // (op 28239) 
v_and_b32 v40, 15, v1 // (op 28239) 
v_lshl_add_u32 v41, v36, 4, v40 // (op 28239) 


v_and_b32 v36, 63, v1 // (op 28239) 
v_lshrrev_b32 v40, 4, v36 // (op 28239) 

v_lshl_add_u32 v36, v40, 7, v41 // (op 28239) 


v_lshrrev_b32 v40, 6, v1 // (op 28315) 
v_and_b32 v41, 1, v40 // (op 28315) 

v_and_b32 v40, 15, v1 // (op 28315) 
v_lshl_add_u32 v42, v41, 4, v40 // (op 28315) 


v_and_b32 v40, 63, v1 // (op 28315) 
v_lshrrev_b32 v41, 4, v40 // (op 28315) 

v_lshl_add_u32 v40, v42, 2, v41 // (op 28315) 


v_lshrrev_b32 v41, 7, v1 // (op 28338) 
s_mul_i32 s52, s16, s8 // (op 28338) 
s_add_u32 s52, s52, s0 // (op 28338) 
v_and_b32 v42, 63, v1 // (op 28338) 
s_mul_hi_u32 s53, s52, s13 // (op 28338) 
v_lshrrev_b32 v43, 4, v42 // (op 28338) 
s_sub_u32 s54, s52, s53 // (op 28338) 
v_lshlrev_b32 v44, 2, v43 // (op 28338) 

s_lshr_b32 s55, s54, 1 // (op 28338) 

v_lshl_add_u32 v43, v41, 4, v44 // (op 28338) 


s_add_u32 s54, s55, s53 // (op 28338) 


v_lshrrev_b32 v41, 2, v42 // (op 28338) 

s_lshr_b32 s53, s54, s11 // (op 28338) 

v_and_b32 v42, 3, v41 // (op 28338) 

s_and_b32 s54, s11, 2147483648 // (op 28338) 
v_and_b32 v41, 3, v1 // (op 28338) 
s_cmp_eq_i32 s54, 0 // (op 28338) 
s_mov_b32 s55, scc // (op 28338) 

v_lshl_add_u32 v44, v42, 2, v41 // (op 28338) 


s_cmp_eq_i32 s55, 0 // (op 28338) 
s_cselect_b32 s54, s52, s53 // (op 28338) 



v_lshrrev_b32 v41, 6, v1 // (op 28338) 
s_mul_hi_u32 s55, s54, s12 // (op 28338) 
v_and_b32 v42, 1, v41 // (op 28338) 

s_sub_u32 s52, s54, s55 // (op 28338) 
v_lshl_add_u32 v41, v42, 4, v44 // (op 28338) 


s_lshr_b32 s53, s52, 1 // (op 28338) 

s_add_u32 s52, s53, s55 // (op 28338) 


s_lshr_b32 s53, s52, s10 // (op 28338) 

s_and_b32 s52, s10, 2147483648 // (op 28338) 
s_cmp_eq_i32 s52, 0 // (op 28338) 
s_mov_b32 s55, scc // (op 28338) 

s_cmp_eq_i32 s55, 0 // (op 28338) 
s_cselect_b32 s52, s54, s53 // (op 28338) 


s_mul_i32 s53, s52, s14 // (op 28338) 
v_lshl_add_u32 v42, s52, 7, v43 // (op 28338) 


s_sub_u32 s52, s54, s53 // (op 28338) 


v_lshl_add_u32 v43, s52, 7, v41 // (op 28338) 


v_mul_lo_u32 v41, v43, s24 // (op 28338) 

v_add_lshl_u32 v43, v42, v41, 2 // (op 28338) 


s_mul_i32 s53, s24, s23 // (op 28340) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 28340) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 28340) least significant
s_mul_hi_u32 s55, s24, s22 // (op 28340) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 28340) most significant: sum
s_add_u32 s53, s53, s55 // (op 28340) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 28340) 

s_mov_b32 s60, s26 // (op 28340) reg expression
s_mov_b32 s61, s27 // (op 28340) reg expression
s_mov_b32 s62, s54 // (op 28340) reg expression
s_mov_b32 s63, 131072 // (op 28340) call()

v_lshrrev_b32 v41, 5, v1 // (op 28692) 
v_lshlrev_b32 v42, 2, v41 // (op 28692) 

v_lshl_add_u32 v41, s16, 7, v42 // (op 28692) 

v_and_b32 v42, -32, v1 // (op 28692) 
v_sub_u32 v44, v1, v42 // (op 28692) 

v_lshl_add_u32 v42, v41, 7, v44 // (op 28692) 


v_lshlrev_b32 v41, 2, v42 // (op 28692) 

s_lshl_b32 s52, s15, 16 // (op 28694) 
s_mov_b32 s64, s28 // (op 28694) reg expression
s_mov_b32 s65, s29 // (op 28694) reg expression
s_mov_b32 s66, s52 // (op 28694) reg expression
s_mov_b32 s67, 131072 // (op 28694) call()

s_mul_i32 s6, s8, s16 // (op 291) 
s_add_u32 s6, s6, s0 // (op 291) 
s_mul_hi_u32 s52, s6, s13 // (op 292) 
s_sub_u32 s53, s6, s52 // (op 292) 
s_lshr_b32 s54, s53, 1 // (op 292) 

s_add_u32 s53, s54, s52 // (op 292) 


s_lshr_b32 s52, s53, s11 // (op 292) 

s_and_b32 s53, s11, 2147483648 // (op 292) 
s_cmp_eq_i32 s53, 0 // (op 292) 
s_mov_b32 s54, scc // (op 292) 

s_cmp_eq_i32 s54, 0 // (op 292) 
s_cselect_b32 s53, s6, s52 // (op 292) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 292) 
s_mul_i32 s52, s53, s54 // (op 292) 


s_sub_u32 s18, s6, s52 // (op 292) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 293) 
s_cmp_gt_u32 s52, s18 // (op 293) 
s_mov_b32 s54, scc // (op 293) 
s_sub_u32 s55, s52, s18 // (op 293) 

s_cmp_eq_i32 s54, 0 // (op 293) 
s_cselect_b32 s52, 0, s55 // (op 293) 


s_cmp_gt_u32 s8, s0 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_sub_u32 s54, s8, s0 // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s55, 0, s54 // (op 293) 


s_cmp_lt_u32 s52, s55 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s54, s55, s52 // (op 293) 



s_cmp_gt_u32 s9, s6 // (op 293) 
s_mov_b32 s55, scc // (op 293) 
s_sub_u32 s52, s9, s6 // (op 293) 
s_cmp_eq_i32 s55, 0 // (op 293) 
s_cselect_b32 s53, 0, s52 // (op 293) 


s_cmp_lt_u32 s54, s53 // (op 293) 
s_mov_b32 s52, scc // (op 293) 
s_cmp_eq_i32 s52, 0 // (op 293) 
s_cselect_b32 s7, s53, s54 // (op 293) 



s_add_u32 s52, s18, s7 // (op 294) 
s_sub_u32 s19, s52, 1 // (op 294) 

v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 7192) call()
v_accvgpr_write a5, 0.00000 // (op 7192) call()
v_accvgpr_write a6, 0.00000 // (op 7192) call()
v_accvgpr_write a7, 0.00000 // (op 7192) call()
v_accvgpr_write a8, 0.00000 // (op 7490) call()
v_accvgpr_write a9, 0.00000 // (op 7490) call()
v_accvgpr_write a10, 0.00000 // (op 7490) call()
v_accvgpr_write a11, 0.00000 // (op 7490) call()
v_accvgpr_write a12, 0.00000 // (op 7788) call()
v_accvgpr_write a13, 0.00000 // (op 7788) call()
v_accvgpr_write a14, 0.00000 // (op 7788) call()
v_accvgpr_write a15, 0.00000 // (op 7788) call()
v_accvgpr_write a16, 0.00000 // (op 8222) call()
v_accvgpr_write a17, 0.00000 // (op 8222) call()
v_accvgpr_write a18, 0.00000 // (op 8222) call()
v_accvgpr_write a19, 0.00000 // (op 8222) call()
v_accvgpr_write a20, 0.00000 // (op 8715) call()
v_accvgpr_write a21, 0.00000 // (op 8715) call()
v_accvgpr_write a22, 0.00000 // (op 8715) call()
v_accvgpr_write a23, 0.00000 // (op 8715) call()
v_accvgpr_write a24, 0.00000 // (op 8716) call()
v_accvgpr_write a25, 0.00000 // (op 8716) call()
v_accvgpr_write a26, 0.00000 // (op 8716) call()
v_accvgpr_write a27, 0.00000 // (op 8716) call()
v_accvgpr_write a28, 0.00000 // (op 8717) call()
v_accvgpr_write a29, 0.00000 // (op 8717) call()
v_accvgpr_write a30, 0.00000 // (op 8717) call()
v_accvgpr_write a31, 0.00000 // (op 8717) call()
v_accvgpr_write a32, 0.00000 // (op 9512) call()
v_accvgpr_write a33, 0.00000 // (op 9512) call()
v_accvgpr_write a34, 0.00000 // (op 9512) call()
v_accvgpr_write a35, 0.00000 // (op 9512) call()
v_accvgpr_write a36, 0.00000 // (op 10005) call()
v_accvgpr_write a37, 0.00000 // (op 10005) call()
v_accvgpr_write a38, 0.00000 // (op 10005) call()
v_accvgpr_write a39, 0.00000 // (op 10005) call()
v_accvgpr_write a40, 0.00000 // (op 10006) call()
v_accvgpr_write a41, 0.00000 // (op 10006) call()
v_accvgpr_write a42, 0.00000 // (op 10006) call()
v_accvgpr_write a43, 0.00000 // (op 10006) call()
v_accvgpr_write a44, 0.00000 // (op 10007) call()
v_accvgpr_write a45, 0.00000 // (op 10007) call()
v_accvgpr_write a46, 0.00000 // (op 10007) call()
v_accvgpr_write a47, 0.00000 // (op 10007) call()
v_accvgpr_write a48, 0.00000 // (op 10802) call()
v_accvgpr_write a49, 0.00000 // (op 10802) call()
v_accvgpr_write a50, 0.00000 // (op 10802) call()
v_accvgpr_write a51, 0.00000 // (op 10802) call()
v_accvgpr_write a52, 0.00000 // (op 11295) call()
v_accvgpr_write a53, 0.00000 // (op 11295) call()
v_accvgpr_write a54, 0.00000 // (op 11295) call()
v_accvgpr_write a55, 0.00000 // (op 11295) call()
v_accvgpr_write a56, 0.00000 // (op 11296) call()
v_accvgpr_write a57, 0.00000 // (op 11296) call()
v_accvgpr_write a58, 0.00000 // (op 11296) call()
v_accvgpr_write a59, 0.00000 // (op 11296) call()
v_accvgpr_write a60, 0.00000 // (op 11297) call()
v_accvgpr_write a61, 0.00000 // (op 11297) call()
v_accvgpr_write a62, 0.00000 // (op 11297) call()
v_accvgpr_write a63, 0.00000 // (op 11297) call()
v_and_b32 v42, -4, v1 // (op 27549) 
s_mul_i32 s52, s16, s8 // (op 27549) 
s_add_u32 s52, s52, s0 // (op 27549) 
v_sub_u32 v44, v1, v42 // (op 27549) 

s_mul_hi_u32 s53, s52, s13 // (op 27549) 
v_lshlrev_b32 v42, 5, v44 // (op 27549) 

s_sub_u32 s54, s52, s53 // (op 27549) 
v_lshrrev_b32 v44, 2, v1 // (op 27549) 
s_lshr_b32 s55, s54, 1 // (op 27549) 

s_add_u32 s54, s55, s53 // (op 27549) 


s_lshr_b32 s53, s54, s11 // (op 27549) 

s_and_b32 s54, s11, 2147483648 // (op 27549) 
s_cmp_eq_i32 s54, 0 // (op 27549) 
s_mov_b32 s55, scc // (op 27549) 

s_cmp_eq_i32 s55, 0 // (op 27549) 
s_cselect_b32 s54, s52, s53 // (op 27549) 


s_mul_hi_u32 s53, s54, s12 // (op 27549) 
s_sub_u32 s55, s54, s53 // (op 27549) 
s_lshr_b32 s68, s55, 1 // (op 27549) 

s_add_u32 s55, s68, s53 // (op 27549) 


s_lshr_b32 s53, s55, s10 // (op 27549) 

s_and_b32 s55, s10, 2147483648 // (op 27549) 
s_cmp_eq_i32 s55, 0 // (op 27549) 
s_mov_b32 s68, scc // (op 27549) 

s_cmp_eq_i32 s68, 0 // (op 27549) 
s_cselect_b32 s55, s54, s53 // (op 27549) 


v_lshl_add_u32 v45, s55, 7, v42 // (op 27549) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 27549) 
v_mov_b32 v47, 0 // (op 27549) convert
v_mov_b32 v46, v45 // (op 27549) convert

s_mul_i32 s53, s54, s68 // (op 27549) 


s_sub_u32 s54, s52, s53 // (op 27549) 


v_lshl_add_u32 v42, s54, 7, v44 // (op 27549) 


v_mul_lo_u32 v45, v42, s31 // (op 27549) most significant: low of low * high
v_mul_hi_u32 v48, v42, s30 // (op 27549) most significant: high of low * low
v_mul_lo_u32 v44, v42, s30 // (op 27549) least significant: low of low * low
v_add_u32 v45, v45, v48 // (op 27549) most significant: sum


v_add_co_u32 v48, s[52:53], v46, v44 // (op 27549) least significant half
v_addc_co_u32 v49, s[52:53], v47, v45, s[52:53] // (op 27549) most significant half

v_lshlrev_b64 v[48:49], 2, v[48:49] // (op 27549) 


v_lshrrev_b64 v[44:45], 3, v[48:49] // (op 27549) 

s_mul_i32 s53, s30, s3 // (op 27551) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27551) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27551) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27551) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27551) most significant: sum
s_add_u32 s53, s53, s55 // (op 27551) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27551) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27551) 

s_mov_b32 s68, s32 // (op 27551) reg expression
s_mov_b32 s69, s33 // (op 27551) reg expression
s_mov_b32 s70, s52 // (op 27551) reg expression
s_mov_b32 s71, 131072 // (op 27551) call()

v_and_b32 v42, -4, v1 // (op 27568) 
s_mul_i32 s52, s16, s8 // (op 27568) 
s_add_u32 s52, s52, s0 // (op 27568) 
v_sub_u32 v46, v1, v42 // (op 27568) 

s_mul_hi_u32 s53, s52, s13 // (op 27568) 
v_lshlrev_b32 v42, 5, v46 // (op 27568) 

s_sub_u32 s54, s52, s53 // (op 27568) 
v_lshrrev_b32 v46, 2, v1 // (op 27568) 
s_lshr_b32 s55, s54, 1 // (op 27568) 

s_add_u32 s54, s55, s53 // (op 27568) 


s_lshr_b32 s53, s54, s11 // (op 27568) 

s_and_b32 s54, s11, 2147483648 // (op 27568) 
s_cmp_eq_i32 s54, 0 // (op 27568) 
s_mov_b32 s55, scc // (op 27568) 

s_cmp_eq_i32 s55, 0 // (op 27568) 
s_cselect_b32 s54, s52, s53 // (op 27568) 


s_mul_hi_u32 s53, s54, s12 // (op 27568) 
s_sub_u32 s55, s54, s53 // (op 27568) 
s_lshr_b32 s72, s55, 1 // (op 27568) 

s_add_u32 s55, s72, s53 // (op 27568) 


s_lshr_b32 s53, s55, s10 // (op 27568) 

s_and_b32 s55, s10, 2147483648 // (op 27568) 
s_cmp_eq_i32 s55, 0 // (op 27568) 
s_mov_b32 s72, scc // (op 27568) 

s_cmp_eq_i32 s72, 0 // (op 27568) 
s_cselect_b32 s55, s54, s53 // (op 27568) 


v_lshl_add_u32 v47, s55, 7, v42 // (op 27568) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 27568) 
v_mov_b32 v49, 0 // (op 27568) convert
v_mov_b32 v48, v47 // (op 27568) convert

s_mul_i32 s53, s54, s72 // (op 27568) 


s_sub_u32 s54, s52, s53 // (op 27568) 


v_lshl_add_u32 v42, s54, 7, v46 // (op 27568) 


v_mul_lo_u32 v47, v42, s31 // (op 27568) most significant: low of low * high
v_mul_hi_u32 v50, v42, s30 // (op 27568) most significant: high of low * low
v_mul_lo_u32 v46, v42, s30 // (op 27568) least significant: low of low * low
v_add_u32 v47, v47, v50 // (op 27568) most significant: sum


v_add_co_u32 v50, s[52:53], v48, v46 // (op 27568) least significant half
v_addc_co_u32 v51, s[52:53], v49, v47, s[52:53] // (op 27568) most significant half

v_lshlrev_b64 v[50:51], 2, v[50:51] // (op 27568) 


v_lshrrev_b64 v[46:47], 3, v[50:51] // (op 27568) 

s_mul_i32 s53, s30, s3 // (op 27570) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27570) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27570) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27570) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27570) most significant: sum
s_add_u32 s53, s53, s55 // (op 27570) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27570) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27570) 

s_mov_b32 s68, s32 // (op 27570) reg expression
s_mov_b32 s69, s33 // (op 27570) reg expression
s_mov_b32 s70, s52 // (op 27570) reg expression
s_mov_b32 s71, 131072 // (op 27570) call()

v_lshrrev_b32 v42, 2, v1 // (op 27677) 
s_mul_i32 s52, s16, s8 // (op 27677) 
s_add_u32 s52, s52, s0 // (op 27677) 
v_and_b32 v48, -4, v1 // (op 27677) 
s_mul_hi_u32 s53, s52, s13 // (op 27677) 
v_sub_u32 v49, v1, v48 // (op 27677) 

s_sub_u32 s54, s52, s53 // (op 27677) 
v_lshlrev_b32 v48, 5, v49 // (op 27677) 

s_lshr_b32 s55, s54, 1 // (op 27677) 

s_add_u32 s54, s55, s53 // (op 27677) 


s_lshr_b32 s53, s54, s11 // (op 27677) 

s_and_b32 s54, s11, 2147483648 // (op 27677) 
s_cmp_eq_i32 s54, 0 // (op 27677) 
s_mov_b32 s55, scc // (op 27677) 

s_cmp_eq_i32 s55, 0 // (op 27677) 
s_cselect_b32 s54, s52, s53 // (op 27677) 


s_mul_hi_u32 s53, s54, s12 // (op 27677) 
s_sub_u32 s55, s54, s53 // (op 27677) 
s_lshr_b32 s72, s55, 1 // (op 27677) 

s_add_u32 s55, s72, s53 // (op 27677) 


s_lshr_b32 s53, s55, s10 // (op 27677) 

s_and_b32 s55, s10, 2147483648 // (op 27677) 
s_cmp_eq_i32 s55, 0 // (op 27677) 
s_mov_b32 s72, scc // (op 27677) 

s_cmp_eq_i32 s72, 0 // (op 27677) 
s_cselect_b32 s55, s54, s53 // (op 27677) 


s_mul_i32 s53, s55, s14 // (op 27677) 

s_sub_u32 s55, s54, s53 // (op 27677) 

v_lshl_add_u32 v49, s55, 7, v42 // (op 27677) 


v_mul_lo_u32 v51, v49, s35 // (op 27677) most significant: low of low * high
v_mul_hi_u32 v42, v49, s34 // (op 27677) most significant: high of low * low
v_mul_lo_u32 v50, v49, s34 // (op 27677) least significant: low of low * low
v_add_u32 v51, v51, v42 // (op 27677) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 27677) 
s_mul_i32 s53, s54, s72 // (op 27677) 


s_sub_u32 s54, s52, s53 // (op 27677) 


v_lshl_add_u32 v42, s54, 7, v48 // (op 27677) 


v_mov_b32 v49, 0 // (op 27677) convert
v_mov_b32 v48, v42 // (op 27677) convert

v_add_co_u32 v52, s[52:53], v48, v50 // (op 27677) least significant half
v_addc_co_u32 v53, s[52:53], v49, v51, s[52:53] // (op 27677) most significant half

v_lshlrev_b64 v[52:53], 2, v[52:53] // (op 27677) 


v_lshrrev_b64 v[48:49], 3, v[52:53] // (op 27677) 

s_mul_i32 s53, s34, s23 // (op 27679) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 27679) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 27679) least significant
s_mul_hi_u32 s55, s34, s22 // (op 27679) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27679) most significant: sum
s_add_u32 s53, s53, s55 // (op 27679) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27679) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27679) 

s_mov_b32 s72, s36 // (op 27679) reg expression
s_mov_b32 s73, s37 // (op 27679) reg expression
s_mov_b32 s74, s52 // (op 27679) reg expression
s_mov_b32 s75, 131072 // (op 27679) call()

v_lshrrev_b32 v42, 2, v1 // (op 27696) 
s_mul_i32 s52, s16, s8 // (op 27696) 
s_add_u32 s52, s52, s0 // (op 27696) 
v_and_b32 v50, -4, v1 // (op 27696) 
s_mul_hi_u32 s53, s52, s13 // (op 27696) 
v_sub_u32 v51, v1, v50 // (op 27696) 

s_sub_u32 s54, s52, s53 // (op 27696) 
v_lshlrev_b32 v50, 5, v51 // (op 27696) 

s_lshr_b32 s55, s54, 1 // (op 27696) 

s_add_u32 s54, s55, s53 // (op 27696) 


s_lshr_b32 s53, s54, s11 // (op 27696) 

s_and_b32 s54, s11, 2147483648 // (op 27696) 
s_cmp_eq_i32 s54, 0 // (op 27696) 
s_mov_b32 s55, scc // (op 27696) 

s_cmp_eq_i32 s55, 0 // (op 27696) 
s_cselect_b32 s54, s52, s53 // (op 27696) 


s_mul_hi_u32 s53, s54, s12 // (op 27696) 
s_sub_u32 s55, s54, s53 // (op 27696) 
s_lshr_b32 s76, s55, 1 // (op 27696) 

s_add_u32 s55, s76, s53 // (op 27696) 


s_lshr_b32 s53, s55, s10 // (op 27696) 

s_and_b32 s55, s10, 2147483648 // (op 27696) 
s_cmp_eq_i32 s55, 0 // (op 27696) 
s_mov_b32 s76, scc // (op 27696) 

s_cmp_eq_i32 s76, 0 // (op 27696) 
s_cselect_b32 s55, s54, s53 // (op 27696) 


s_mul_i32 s53, s55, s14 // (op 27696) 

s_sub_u32 s55, s54, s53 // (op 27696) 

v_lshl_add_u32 v51, s55, 7, v42 // (op 27696) 


v_mul_lo_u32 v53, v51, s35 // (op 27696) most significant: low of low * high
v_mul_hi_u32 v42, v51, s34 // (op 27696) most significant: high of low * low
v_mul_lo_u32 v52, v51, s34 // (op 27696) least significant: low of low * low
v_add_u32 v53, v53, v42 // (op 27696) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 27696) 
s_mul_i32 s53, s54, s76 // (op 27696) 


s_sub_u32 s54, s52, s53 // (op 27696) 


v_lshl_add_u32 v42, s54, 7, v50 // (op 27696) 


v_mov_b32 v51, 0 // (op 27696) convert
v_mov_b32 v50, v42 // (op 27696) convert

v_add_co_u32 v54, s[52:53], v50, v52 // (op 27696) least significant half
v_addc_co_u32 v55, s[52:53], v51, v53, s[52:53] // (op 27696) most significant half

v_lshlrev_b64 v[54:55], 2, v[54:55] // (op 27696) 


v_lshrrev_b64 v[50:51], 3, v[54:55] // (op 27696) 

s_mul_i32 s53, s34, s23 // (op 27698) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 27698) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 27698) least significant
s_mul_hi_u32 s55, s34, s22 // (op 27698) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27698) most significant: sum
s_add_u32 s53, s53, s55 // (op 27698) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27698) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27698) 

s_mov_b32 s72, s36 // (op 27698) reg expression
s_mov_b32 s73, s37 // (op 27698) reg expression
s_mov_b32 s74, s52 // (op 27698) reg expression
s_mov_b32 s75, 131072 // (op 27698) call()

v_and_b32 v42, -128, v1 // (op 27805) 
s_mul_i32 s52, s16, s8 // (op 27805) 
s_add_u32 s52, s52, s0 // (op 27805) 
v_sub_u32 v52, v1, v42 // (op 27805) 

s_mul_hi_u32 s53, s52, s13 // (op 27805) 
v_lshrrev_b32 v42, 7, v1 // (op 27805) 
s_sub_u32 s54, s52, s53 // (op 27805) 
v_lshlrev_b32 v53, 1, v42 // (op 27805) 

s_lshr_b32 s55, s54, 1 // (op 27805) 

s_add_u32 s54, s55, s53 // (op 27805) 


s_lshr_b32 s53, s54, s11 // (op 27805) 

s_and_b32 s54, s11, 2147483648 // (op 27805) 
s_cmp_eq_i32 s54, 0 // (op 27805) 
s_mov_b32 s55, scc // (op 27805) 

s_cmp_eq_i32 s55, 0 // (op 27805) 
s_cselect_b32 s54, s52, s53 // (op 27805) 


s_mul_hi_u32 s53, s54, s12 // (op 27805) 
s_sub_u32 s55, s54, s53 // (op 27805) 
s_lshr_b32 s76, s55, 1 // (op 27805) 

s_add_u32 s55, s76, s53 // (op 27805) 


s_lshr_b32 s53, s55, s10 // (op 27805) 

s_and_b32 s55, s10, 2147483648 // (op 27805) 
s_cmp_eq_i32 s55, 0 // (op 27805) 
s_mov_b32 s76, scc // (op 27805) 

s_cmp_eq_i32 s76, 0 // (op 27805) 
s_cselect_b32 s55, s54, s53 // (op 27805) 


v_lshl_add_u32 v42, s55, 7, v52 // (op 27805) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 27805) 
s_mul_i32 s53, s54, s76 // (op 27805) 


s_sub_u32 s54, s52, s53 // (op 27805) 


v_lshl_add_u32 v52, s54, 2, v53 // (op 27805) 


v_mul_lo_u32 v53, v52, s38 // (op 27805) 

v_add_u32 v52, v42, v53 // (op 27805) 


s_mul_i32 s53, s38, s43 // (op 27807) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27807) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27807) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27807) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27807) most significant: sum
s_add_u32 s53, s53, s55 // (op 27807) most significant: sum


s_mov_b32 s76, s40 // (op 27807) reg expression
s_mov_b32 s77, s41 // (op 27807) reg expression
s_mov_b32 s78, s52 // (op 27807) reg expression
s_mov_b32 s79, 131072 // (op 27807) call()

v_and_b32 v42, -128, v1 // (op 27824) 
s_mul_i32 s52, s16, s8 // (op 27824) 
s_add_u32 s52, s52, s0 // (op 27824) 
v_sub_u32 v53, v1, v42 // (op 27824) 

s_mul_hi_u32 s53, s52, s13 // (op 27824) 
v_lshrrev_b32 v42, 7, v1 // (op 27824) 
s_sub_u32 s54, s52, s53 // (op 27824) 
v_lshlrev_b32 v54, 1, v42 // (op 27824) 

s_lshr_b32 s55, s54, 1 // (op 27824) 

s_add_u32 s54, s55, s53 // (op 27824) 


s_lshr_b32 s53, s54, s11 // (op 27824) 

s_and_b32 s54, s11, 2147483648 // (op 27824) 
s_cmp_eq_i32 s54, 0 // (op 27824) 
s_mov_b32 s55, scc // (op 27824) 

s_cmp_eq_i32 s55, 0 // (op 27824) 
s_cselect_b32 s54, s52, s53 // (op 27824) 


s_mul_hi_u32 s53, s54, s12 // (op 27824) 
s_sub_u32 s55, s54, s53 // (op 27824) 
s_lshr_b32 s80, s55, 1 // (op 27824) 

s_add_u32 s55, s80, s53 // (op 27824) 


s_lshr_b32 s53, s55, s10 // (op 27824) 

s_and_b32 s55, s10, 2147483648 // (op 27824) 
s_cmp_eq_i32 s55, 0 // (op 27824) 
s_mov_b32 s80, scc // (op 27824) 

s_cmp_eq_i32 s80, 0 // (op 27824) 
s_cselect_b32 s55, s54, s53 // (op 27824) 


v_lshl_add_u32 v42, s55, 7, v53 // (op 27824) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 27824) 
s_mul_i32 s53, s54, s80 // (op 27824) 


s_sub_u32 s54, s52, s53 // (op 27824) 


v_lshl_add_u32 v53, s54, 2, v54 // (op 27824) 


v_mul_lo_u32 v54, v53, s38 // (op 27824) 

v_add_u32 v53, v42, v54 // (op 27824) 


s_mul_i32 s53, s38, s43 // (op 27826) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27826) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27826) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27826) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27826) most significant: sum
s_add_u32 s53, s53, s55 // (op 27826) most significant: sum


s_mov_b32 s76, s40 // (op 27826) reg expression
s_mov_b32 s77, s41 // (op 27826) reg expression
s_mov_b32 s78, s52 // (op 27826) reg expression
s_mov_b32 s79, 131072 // (op 27826) call()

v_lshrrev_b32 v42, 2, v1 // (op 27933) 
s_mul_i32 s52, s16, s8 // (op 27933) 
s_add_u32 s52, s52, s0 // (op 27933) 
v_lshlrev_b32 v54, 1, v42 // (op 27933) 

s_mul_hi_u32 s53, s52, s13 // (op 27933) 
v_and_b32 v42, -4, v1 // (op 27933) 
s_sub_u32 s54, s52, s53 // (op 27933) 
v_sub_u32 v55, v1, v42 // (op 27933) 

s_lshr_b32 s55, s54, 1 // (op 27933) 

s_add_u32 s54, s55, s53 // (op 27933) 


s_lshr_b32 s53, s54, s11 // (op 27933) 

s_and_b32 s54, s11, 2147483648 // (op 27933) 
s_cmp_eq_i32 s54, 0 // (op 27933) 
s_mov_b32 s55, scc // (op 27933) 

s_cmp_eq_i32 s55, 0 // (op 27933) 
s_cselect_b32 s54, s52, s53 // (op 27933) 


s_mul_hi_u32 s53, s54, s12 // (op 27933) 
s_sub_u32 s55, s54, s53 // (op 27933) 
s_lshr_b32 s80, s55, 1 // (op 27933) 

s_add_u32 s55, s80, s53 // (op 27933) 


s_lshr_b32 s53, s55, s10 // (op 27933) 

s_and_b32 s55, s10, 2147483648 // (op 27933) 
s_cmp_eq_i32 s55, 0 // (op 27933) 
s_mov_b32 s80, scc // (op 27933) 

s_cmp_eq_i32 s80, 0 // (op 27933) 
s_cselect_b32 s55, s54, s53 // (op 27933) 


s_mul_i32 s53, s55, s14 // (op 27933) 

s_sub_u32 s55, s54, s53 // (op 27933) 

v_lshl_add_u32 v42, s55, 7, v54 // (op 27933) 


v_mul_lo_u32 v54, v42, s44 // (op 27933) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 27933) 
s_mul_i32 s53, s54, s80 // (op 27933) 


s_sub_u32 s54, s52, s53 // (op 27933) 


v_lshl_add_u32 v42, s54, 2, v55 // (op 27933) 


v_add_u32 v55, v42, v54 // (op 27933) 


s_mul_i32 s53, s44, s23 // (op 27935) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 27935) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 27935) least significant
s_mul_hi_u32 s55, s44, s22 // (op 27935) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27935) most significant: sum
s_add_u32 s53, s53, s55 // (op 27935) most significant: sum


s_mov_b32 s80, s46 // (op 27935) reg expression
s_mov_b32 s81, s47 // (op 27935) reg expression
s_mov_b32 s82, s52 // (op 27935) reg expression
s_mov_b32 s83, 131072 // (op 27935) call()

v_lshrrev_b32 v42, 2, v1 // (op 27952) 
s_mul_i32 s52, s16, s8 // (op 27952) 
s_add_u32 s52, s52, s0 // (op 27952) 
v_lshlrev_b32 v54, 1, v42 // (op 27952) 

s_mul_hi_u32 s53, s52, s13 // (op 27952) 
v_and_b32 v42, -4, v1 // (op 27952) 
s_sub_u32 s54, s52, s53 // (op 27952) 
v_sub_u32 v56, v1, v42 // (op 27952) 

s_lshr_b32 s55, s54, 1 // (op 27952) 

s_add_u32 s54, s55, s53 // (op 27952) 


s_lshr_b32 s53, s54, s11 // (op 27952) 

s_and_b32 s54, s11, 2147483648 // (op 27952) 
s_cmp_eq_i32 s54, 0 // (op 27952) 
s_mov_b32 s55, scc // (op 27952) 

s_cmp_eq_i32 s55, 0 // (op 27952) 
s_cselect_b32 s54, s52, s53 // (op 27952) 


s_mul_hi_u32 s53, s54, s12 // (op 27952) 
s_sub_u32 s55, s54, s53 // (op 27952) 
s_lshr_b32 s84, s55, 1 // (op 27952) 

s_add_u32 s55, s84, s53 // (op 27952) 


s_lshr_b32 s53, s55, s10 // (op 27952) 

s_and_b32 s55, s10, 2147483648 // (op 27952) 
s_cmp_eq_i32 s55, 0 // (op 27952) 
s_mov_b32 s84, scc // (op 27952) 

s_cmp_eq_i32 s84, 0 // (op 27952) 
s_cselect_b32 s55, s54, s53 // (op 27952) 


s_mul_i32 s53, s55, s14 // (op 27952) 

s_sub_u32 s55, s54, s53 // (op 27952) 

v_lshl_add_u32 v42, s55, 7, v54 // (op 27952) 


v_mul_lo_u32 v54, v42, s44 // (op 27952) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 27952) 
s_mul_i32 s53, s54, s84 // (op 27952) 


s_sub_u32 s54, s52, s53 // (op 27952) 


v_lshl_add_u32 v42, s54, 2, v56 // (op 27952) 


v_add_u32 v56, v42, v54 // (op 27952) 


s_mul_i32 s53, s44, s23 // (op 27954) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 27954) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 27954) least significant
s_mul_hi_u32 s55, s44, s22 // (op 27954) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27954) most significant: sum
s_add_u32 s53, s53, s55 // (op 27954) most significant: sum


s_mov_b32 s80, s46 // (op 27954) reg expression
s_mov_b32 s81, s47 // (op 27954) reg expression
s_mov_b32 s82, s52 // (op 27954) reg expression
s_mov_b32 s83, 131072 // (op 27954) call()

v_and_b32 v42, -4, v1 // (op 28129) 
v_sub_u32 v54, v1, v42 // (op 28129) 

v_lshlrev_b32 v42, 5, v54 // (op 28129) 

v_lshrrev_b32 v54, 2, v1 // (op 28129) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 28129) 


v_lshlrev_b32 v42, 2, v57 // (op 28129) 

v_lshrrev_b32 v54, 3, v42 // (op 28129) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 28129) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30028) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27391) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27391) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27391) 


v_mov_b64 v[60:61], v[48:49] // (op 27391) 
s_mov_b32 m0, s52 // (op 27391) reg expression
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 27391) Load value direct to lds
v_add_co_u32 v60, s[54:55], v60, v58 // (op 27391) least significant half
v_addc_co_u32 v61, s[54:55], 0, v61, s[54:55] // (op 27391) most significant half

s_add_u32 m0, m0, 4096 // (op 27391) 
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 27391) Load value direct to lds




v_mov_b64 v[58:59], s[44:45] // (op 26866) convert

v_mov_b32 v42, v55 // (op 26866) 
buffer_load_ubyte v60, v42, s[80:83], 0 offen offset: 0 // (op 26866) Load value
v_add_u32 v42, v42, v58 // (op 26866) 
buffer_load_ubyte v61, v42, s[80:83], 0 offen offset: 0 // (op 26866) Load value


v_and_b32 v42, -4, v1 // (op 28054) 
v_sub_u32 v54, v1, v42 // (op 28054) 

v_lshlrev_b32 v42, 5, v54 // (op 28054) 

v_lshrrev_b32 v54, 2, v1 // (op 28054) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 28054) 


v_lshlrev_b32 v42, 2, v57 // (op 28054) 

v_lshrrev_b32 v54, 3, v42 // (op 28054) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 28054) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30009) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27396) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27396) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27396) 


v_mov_b64 v[62:63], v[44:45] // (op 27396) reg expression
s_add_u32 s53, 8192, s52 // (op 27396) 
s_mov_b32 m0, s53 // (op 27396) reg expression
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 27396) Load value direct to lds
v_add_co_u32 v62, s[54:55], v62, v58 // (op 27396) least significant half
v_addc_co_u32 v63, s[54:55], 0, v63, s[54:55] // (op 27396) most significant half

s_add_u32 m0, m0, 4096 // (op 27396) 
buffer_load_dwordx4 v62, s[68:71], 0 offen offset: 0 lds // (op 27396) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 26872) convert

v_mov_b32 v42, v52 // (op 26872) reg expression
buffer_load_ubyte v62, v42, s[76:79], 0 offen offset: 0 // (op 26872) Load value
v_add_u32 v42, v42, v58 // (op 26872) 
buffer_load_ubyte v63, v42, s[76:79], 0 offen offset: 0 // (op 26872) Load value


v_and_b32 v42, -4, v1 // (op 29231) 
v_sub_u32 v54, v1, v42 // (op 29231) 

v_lshlrev_b32 v42, 5, v54 // (op 29231) 

v_lshrrev_b32 v54, 2, v1 // (op 29231) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29231) 


v_lshlrev_b32 v42, 2, v57 // (op 29231) 

v_lshrrev_b32 v54, 3, v42 // (op 29231) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29231) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30275) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27401) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27401) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27401) 


v_mov_b64 v[64:65], v[48:49] // (op 27401) 
s_add_u32 s53, 16384, s52 // (op 27401) 
s_mov_b32 m0, s53 // (op 27401) reg expression
buffer_load_dwordx4 v64, s[72:75], 64 offen offset: 0 lds // (op 27401) Load value direct to lds
v_add_co_u32 v64, s[54:55], v64, v58 // (op 27401) least significant half
v_addc_co_u32 v65, s[54:55], 0, v65, s[54:55] // (op 27401) most significant half

s_add_u32 m0, m0, 4096 // (op 27401) 
buffer_load_dwordx4 v64, s[72:75], 64 offen offset: 0 lds // (op 27401) Load value direct to lds





v_mov_b64 v[58:59], s[44:45] // (op 26878) convert

v_mov_b32 v42, v55 // (op 26878) 
buffer_load_ubyte v64, v42, s[80:83], 0 offen offset:4 // (op 26878) Load value
v_add_u32 v42, v42, v58 // (op 26878) 
buffer_load_ubyte v65, v42, s[80:83], 0 offen offset:4 // (op 26878) Load value


v_and_b32 v42, -4, v1 // (op 29187) 
v_sub_u32 v54, v1, v42 // (op 29187) 

v_lshlrev_b32 v42, 5, v54 // (op 29187) 

v_lshrrev_b32 v54, 2, v1 // (op 29187) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29187) 


v_lshlrev_b32 v42, 2, v57 // (op 29187) 

v_lshrrev_b32 v54, 3, v42 // (op 29187) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29187) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30265) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27403) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27403) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27403) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27403) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27403) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27403) 

v_mov_b32 v42, s54 // (op 27403) 
v_mov_b32 v54, s55 // (op 27403) 
v_add_co_u32 v66, s[84:85], v44, v42 // (op 27403) least significant half
v_addc_co_u32 v67, s[84:85], v45, v54, s[84:85] // (op 27403) most significant half




s_add_u32 s53, 24576, s52 // (op 27403) 
s_mov_b32 m0, s53 // (op 27403) reg expression
buffer_load_dwordx4 v66, s[68:71], 0 offen offset: 0 lds // (op 27403) Load value direct to lds
v_add_co_u32 v66, s[54:55], v66, v58 // (op 27403) least significant half
v_addc_co_u32 v67, s[54:55], 0, v67, s[54:55] // (op 27403) most significant half

s_add_u32 m0, m0, 4096 // (op 27403) 
buffer_load_dwordx4 v66, s[68:71], 0 offen offset: 0 lds // (op 27403) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 26884) convert

s_lshl_b32 s52, s38, 2 // (op 26884) 
v_add_u32 v42, v52, s52 // (op 26884) 

buffer_load_ubyte v66, v42, s[76:79], 0 offen offset: 0 // (op 26884) Load value
v_add_u32 v42, v42, v58 // (op 26884) 
buffer_load_ubyte v67, v42, s[76:79], 0 offen offset: 0 // (op 26884) Load value


v_and_b32 v42, -4, v1 // (op 29377) 
v_sub_u32 v54, v1, v42 // (op 29377) 

v_lshlrev_b32 v42, 5, v54 // (op 29377) 

v_lshrrev_b32 v54, 2, v1 // (op 29377) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29377) 


v_lshlrev_b32 v42, 2, v57 // (op 29377) 

v_lshrrev_b32 v54, 3, v42 // (op 29377) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29377) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30307) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27405) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27405) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27405) 


v_mov_b64 v[68:69], v[48:49] // (op 27405) 
s_add_u32 s53, 32768, s52 // (op 27405) 
s_mov_b32 m0, s53 // (op 27405) reg expression
s_mov_b32 s54, 128 // (op 27405) call()
buffer_load_dwordx4 v68, s[72:75], s54 offen offset: 0 lds // (op 27405) Load value direct to lds
v_add_co_u32 v68, s[84:85], v68, v58 // (op 27405) least significant half
v_addc_co_u32 v69, s[84:85], 0, v69, s[84:85] // (op 27405) most significant half

s_add_u32 m0, m0, 4096 // (op 27405) 
buffer_load_dwordx4 v68, s[72:75], s54 offen offset: 0 lds // (op 27405) Load value direct to lds






v_mov_b64 v[58:59], s[44:45] // (op 26890) convert

v_mov_b32 v42, v55 // (op 26890) 
buffer_load_ubyte v68, v42, s[80:83], 0 offen offset:8 // (op 26890) Load value
v_add_u32 v42, v42, v58 // (op 26890) 
buffer_load_ubyte v69, v42, s[80:83], 0 offen offset:8 // (op 26890) Load value


v_and_b32 v42, -4, v1 // (op 29333) 
v_sub_u32 v54, v1, v42 // (op 29333) 

v_lshlrev_b32 v42, 5, v54 // (op 29333) 

v_lshrrev_b32 v54, 2, v1 // (op 29333) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29333) 


v_lshlrev_b32 v42, 2, v57 // (op 29333) 

v_lshrrev_b32 v54, 3, v42 // (op 29333) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29333) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30297) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27407) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27407) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27407) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27407) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27407) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27407) 

s_lshl_b64 s[84:85], s[54:55], 1 // (op 27407) 

v_mov_b32 v42, s84 // (op 27407) 
v_mov_b32 v54, s85 // (op 27407) 
v_add_co_u32 v70, s[54:55], v44, v42 // (op 27407) least significant half
v_addc_co_u32 v71, s[54:55], v45, v54, s[54:55] // (op 27407) most significant half




s_add_u32 s53, 40960, s52 // (op 27407) 
s_mov_b32 m0, s53 // (op 27407) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 27407) Load value direct to lds
v_add_co_u32 v70, s[54:55], v70, v58 // (op 27407) least significant half
v_addc_co_u32 v71, s[54:55], 0, v71, s[54:55] // (op 27407) most significant half

s_add_u32 m0, m0, 4096 // (op 27407) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 27407) Load value direct to lds





v_mov_b64 v[58:59], s[38:39] // (op 26896) convert

s_lshl_b32 s52, s38, 3 // (op 26896) 
v_add_u32 v42, v52, s52 // (op 26896) 

buffer_load_ubyte v70, v42, s[76:79], 0 offen offset: 0 // (op 26896) Load value
v_add_u32 v42, v42, v58 // (op 26896) 
buffer_load_ubyte v71, v42, s[76:79], 0 offen offset: 0 // (op 26896) Load value


v_and_b32 v42, -4, v1 // (op 29523) 
v_sub_u32 v54, v1, v42 // (op 29523) 

v_lshlrev_b32 v42, 5, v54 // (op 29523) 

v_lshrrev_b32 v54, 2, v1 // (op 29523) 
v_lshl_add_u32 v57, v54, 7, v42 // (op 29523) 


v_lshlrev_b32 v42, 2, v57 // (op 29523) 

v_lshrrev_b32 v54, 3, v42 // (op 29523) 

s_nop 0
v_readfirstlane_b32 s52, v54 // (op 29523) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30339) 

s_lshl_b64 s[54:55], s[34:35], 6 // (op 27409) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27409) 

v_lshrrev_b64 v[58:59], 3, s[84:85] // (op 27409) 


v_mov_b64 v[72:73], v[48:49] // (op 27409) 
s_add_u32 s53, 49152, s52 // (op 27409) 
s_mov_b32 m0, s53 // (op 27409) reg expression
s_mov_b32 s54, 192 // (op 27409) call()
buffer_load_dwordx4 v72, s[72:75], s54 offen offset: 0 lds // (op 27409) Load value direct to lds
v_add_co_u32 v72, s[84:85], v72, v58 // (op 27409) least significant half
v_addc_co_u32 v73, s[84:85], 0, v73, s[84:85] // (op 27409) most significant half

s_add_u32 m0, m0, 4096 // (op 27409) 
buffer_load_dwordx4 v72, s[72:75], s54 offen offset: 0 lds // (op 27409) Load value direct to lds







v_mov_b64 v[48:49], s[44:45] // (op 26902) convert

v_mov_b32 v42, v55 // (op 26902) 
buffer_load_ubyte v58, v42, s[80:83], 0 offen offset:12 // (op 26902) Load value
v_add_u32 v42, v42, v48 // (op 26902) 
buffer_load_ubyte v59, v42, s[80:83], 0 offen offset:12 // (op 26902) Load value



v_and_b32 v42, -4, v1 // (op 29479) 
v_sub_u32 v57, v1, v42 // (op 29479) 

v_lshlrev_b32 v42, 5, v57 // (op 29479) 

v_lshrrev_b32 v57, 2, v1 // (op 29479) 
v_lshl_add_u32 v48, v57, 7, v42 // (op 29479) 


v_lshlrev_b32 v42, 2, v48 // (op 29479) 

v_lshrrev_b32 v57, 3, v42 // (op 29479) 

s_nop 0
v_readfirstlane_b32 s52, v57 // (op 29479) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30329) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 27411) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27411) 

v_lshrrev_b64 v[48:49], 3, s[84:85] // (op 27411) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 27411) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 27411) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 27411) 

s_mul_i32 s85, 3, s55 // (op 27411) most significant: low of low * high
s_mul_i32 s84, 3, s54 // (op 27411) least significant
s_mul_hi_u32 s53, 3, s54 // (op 27411) most significant: high of low * low
s_add_u32 s85, s85, s53 // (op 27411) most significant: sum


v_mov_b32 v42, s84 // (op 27411) 
v_mov_b32 v57, s85 // (op 27411) 
v_add_co_u32 v54, s[54:55], v44, v42 // (op 27411) least significant half
v_addc_co_u32 v55, s[54:55], v45, v57, s[54:55] // (op 27411) most significant half




s_add_u32 s53, 57344, s52 // (op 27411) 
s_mov_b32 m0, s53 // (op 27411) reg expression
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27411) Load value direct to lds
v_add_co_u32 v54, s[54:55], v54, v48 // (op 27411) least significant half
v_addc_co_u32 v55, s[54:55], 0, v55, s[54:55] // (op 27411) most significant half

s_add_u32 m0, m0, 4096 // (op 27411) 
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27411) Load value direct to lds






v_mov_b64 v[44:45], s[38:39] // (op 26908) convert

s_lshl_b32 s52, s38, 2 // (op 26908) 
s_mul_i32 s53, 3, s52 // (op 26908) 

v_add_u32 v42, v52, s53 // (op 26908) 

buffer_load_ubyte v48, v42, s[76:79], 0 offen offset: 0 // (op 26908) Load value
v_add_u32 v42, v42, v44 // (op 26908) 
buffer_load_ubyte v49, v42, s[76:79], 0 offen offset: 0 // (op 26908) Load value



v_and_b32 v42, -4, v1 // (op 28281) 
v_sub_u32 v52, v1, v42 // (op 28281) 

v_lshrrev_b32 v42, 2, v1 // (op 28281) 
v_lshl_add_u32 v57, v42, 3, v52 // (op 28281) 


s_barrier  // (op 30069) 



v_add_u32 v42, 65536, v57 // (op 26911) 
s_waitcnt vmcnt(29)
ds_write_b8 v42, v60 // (op 26911) Store local data 

v_add_u32 v42, 65540, v57 // (op 26911) 
s_waitcnt vmcnt(28)
ds_write_b8 v42, v61 // (op 26911) Store local data 


v_and_b32 v42, -128, v1 // (op 28205) 
v_sub_u32 v52, v1, v42 // (op 28205) 

v_lshrrev_b32 v42, 7, v1 // (op 28205) 
v_lshl_add_u32 v57, v42, 8, v52 // (op 28205) 


s_barrier  // (op 30048) 



v_add_u32 v42, 66048, v57 // (op 26913) 
s_waitcnt vmcnt(25)
ds_write_b8 v42, v62 // (op 26913) Store local data 

v_add_u32 v42, 66176, v57 // (op 26913) 
s_waitcnt vmcnt(24)
ds_write_b8 v42, v63 // (op 26913) Store local data 


s_barrier  // (op 30025) 



ds_read_b128 v[72:75], v39 // (op 26956) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30066) 



v_add_u32 v42, 65536, v40 // (op 26996) 
ds_read_u8 v52, v42 // (op 26996) Load local data 




ds_read_b128 v[76:79], v39 offset:2048 // (op 26961) Load local data 



v_add_u32 v42, 65664, v40 // (op 27001) 
ds_read_u8 v57, v42 // (op 27001) Load local data 




ds_read_b128 v[80:83], v39 offset:4096 // (op 26966) Load local data 



v_add_u32 v42, 65792, v40 // (op 27006) 
ds_read_u8 v44, v42 // (op 27006) Load local data 




ds_read_b128 v[84:87], v39 offset:6144 // (op 26971) Load local data 



v_add_u32 v42, 65920, v40 // (op 27011) 
ds_read_u8 v45, v42 // (op 27011) Load local data 

s_barrier  // (op 30006) 



ds_read_b64_tr_b4 v[88:89], v37 offset:8192 // (op 26936) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v37 offset:9216 // (op 26936) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30045) 


v_add_u32 v42, 66048, v36 // (op 26976) 
ds_read_u8 v54, v42 // (op 26976) Load local data 




ds_read_b64_tr_b4 v[92:93], v37 offset:8208 // (op 26941) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v37 offset:9232 // (op 26941) Transpose load local data 


v_add_u32 v42, 66080, v36 // (op 26981) 
ds_read_u8 v55, v42 // (op 26981) Load local data 




ds_read_b64_tr_b4 v[96:97], v37 offset:8224 // (op 26946) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v37 offset:9248 // (op 26946) Transpose load local data 


v_add_u32 v42, 66112, v36 // (op 26986) 
ds_read_u8 v100, v42 // (op 26986) Load local data 




ds_read_b64_tr_b4 v[102:103], v37 offset:8240 // (op 26951) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v37 offset:9264 // (op 26951) Transpose load local data 


v_add_u32 v42, 66144, v36 // (op 26991) 
ds_read_u8 v101, v42 // (op 26991) Load local data 

v_and_b32 v42, -4, v1 // (op 28026) 
v_sub_u32 v106, v1, v42 // (op 28026) 

v_lshlrev_b32 v42, 5, v106 // (op 28026) 

v_lshrrev_b32 v106, 2, v1 // (op 28026) 
v_lshl_add_u32 v107, v106, 7, v42 // (op 28026) 


v_lshlrev_b32 v42, 2, v107 // (op 28026) 

v_lshrrev_b32 v106, 3, v42 // (op 28026) 

s_nop 0
v_readfirstlane_b32 s52, v106 // (op 28026) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 28101) 
v_sub_u32 v106, v1, v42 // (op 28101) 

v_lshlrev_b32 v42, 5, v106 // (op 28101) 

v_lshrrev_b32 v106, 2, v1 // (op 28101) 
v_lshl_add_u32 v107, v106, 7, v42 // (op 28101) 


v_lshlrev_b32 v42, 2, v107 // (op 28101) 

v_lshrrev_b32 v106, 3, v42 // (op 28101) 

s_nop 0
v_readfirstlane_b32 s53, v106 // (op 28101) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 28176) 
v_sub_u32 v106, v1, v42 // (op 28176) 

v_lshrrev_b32 v42, 7, v1 // (op 28176) 
v_lshl_add_u32 v107, v42, 8, v106 // (op 28176) 


v_and_b32 v42, -4, v1 // (op 28252) 
v_sub_u32 v106, v1, v42 // (op 28252) 

v_lshrrev_b32 v42, 2, v1 // (op 28252) 
v_lshl_add_u32 v108, v42, 3, v106 // (op 28252) 


v_and_b32 v42, -4, v1 // (op 29172) 
v_sub_u32 v106, v1, v42 // (op 29172) 

v_lshlrev_b32 v42, 5, v106 // (op 29172) 

v_lshrrev_b32 v106, 2, v1 // (op 29172) 
v_lshl_add_u32 v109, v106, 7, v42 // (op 29172) 


v_lshlrev_b32 v42, 2, v109 // (op 29172) 

v_lshrrev_b32 v106, 3, v42 // (op 29172) 

s_nop 0
v_readfirstlane_b32 s54, v106 // (op 29172) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29216) 
v_sub_u32 v106, v1, v42 // (op 29216) 

v_lshlrev_b32 v42, 5, v106 // (op 29216) 

v_lshrrev_b32 v106, 2, v1 // (op 29216) 
v_lshl_add_u32 v109, v106, 7, v42 // (op 29216) 


v_lshlrev_b32 v42, 2, v109 // (op 29216) 

v_lshrrev_b32 v106, 3, v42 // (op 29216) 

s_nop 0
v_readfirstlane_b32 s55, v106 // (op 29216) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29260) 
v_sub_u32 v106, v1, v42 // (op 29260) 

v_lshrrev_b32 v42, 7, v1 // (op 29260) 
v_lshl_add_u32 v109, v42, 8, v106 // (op 29260) 


v_and_b32 v42, -4, v1 // (op 29289) 
v_sub_u32 v106, v1, v42 // (op 29289) 

v_lshrrev_b32 v42, 2, v1 // (op 29289) 
v_lshl_add_u32 v110, v42, 3, v106 // (op 29289) 


v_and_b32 v42, -4, v1 // (op 29318) 
v_sub_u32 v106, v1, v42 // (op 29318) 

v_lshlrev_b32 v42, 5, v106 // (op 29318) 

v_lshrrev_b32 v106, 2, v1 // (op 29318) 
v_lshl_add_u32 v111, v106, 7, v42 // (op 29318) 


v_lshlrev_b32 v42, 2, v111 // (op 29318) 

v_lshrrev_b32 v106, 3, v42 // (op 29318) 

s_nop 0
v_readfirstlane_b32 s84, v106 // (op 29318) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29362) 
v_sub_u32 v106, v1, v42 // (op 29362) 

v_lshlrev_b32 v42, 5, v106 // (op 29362) 

v_lshrrev_b32 v106, 2, v1 // (op 29362) 
v_lshl_add_u32 v111, v106, 7, v42 // (op 29362) 


v_lshlrev_b32 v42, 2, v111 // (op 29362) 

v_lshrrev_b32 v106, 3, v42 // (op 29362) 

s_nop 0
v_readfirstlane_b32 s85, v106 // (op 29362) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29406) 
v_sub_u32 v106, v1, v42 // (op 29406) 

v_lshrrev_b32 v42, 7, v1 // (op 29406) 
v_lshl_add_u32 v111, v42, 8, v106 // (op 29406) 


v_and_b32 v42, -4, v1 // (op 29435) 
v_sub_u32 v106, v1, v42 // (op 29435) 

v_lshrrev_b32 v42, 2, v1 // (op 29435) 
v_lshl_add_u32 v112, v42, 3, v106 // (op 29435) 


v_and_b32 v42, -4, v1 // (op 29464) 
v_sub_u32 v106, v1, v42 // (op 29464) 

v_lshlrev_b32 v42, 5, v106 // (op 29464) 

v_lshrrev_b32 v106, 2, v1 // (op 29464) 
v_lshl_add_u32 v113, v106, 7, v42 // (op 29464) 


v_lshlrev_b32 v42, 2, v113 // (op 29464) 

v_lshrrev_b32 v106, 3, v42 // (op 29464) 

s_nop 0
v_readfirstlane_b32 s86, v106 // (op 29464) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 29508) 
v_sub_u32 v106, v1, v42 // (op 29508) 

v_lshlrev_b32 v42, 5, v106 // (op 29508) 

v_lshrrev_b32 v106, 2, v1 // (op 29508) 
v_lshl_add_u32 v113, v106, 7, v42 // (op 29508) 


v_lshlrev_b32 v42, 2, v113 // (op 29508) 

v_lshrrev_b32 v106, 3, v42 // (op 29508) 

s_nop 0
v_readfirstlane_b32 s87, v106 // (op 29508) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 29552) 
v_sub_u32 v106, v1, v42 // (op 29552) 

v_lshrrev_b32 v42, 7, v1 // (op 29552) 
v_lshl_add_u32 v113, v42, 8, v106 // (op 29552) 


v_and_b32 v42, -4, v1 // (op 29581) 
v_sub_u32 v106, v1, v42 // (op 29581) 

v_lshrrev_b32 v42, 2, v1 // (op 29581) 
v_lshl_add_u32 v114, v42, 3, v106 // (op 29581) 


s_mov_b32 s1, 0 // (op 9607) call()
s_and_b32 s88, s7, 4294967292 // (op 9606) 
s_cmp_lt_u32 s1, s88 // (op 9606) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9606 // (op 9606) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9606 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9606:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[72:75], a[0:3], v54, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 
s_barrier  // (op 30031) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[76:79], a[4:7], v54, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7232) 
s_add_u32 s1, s1, 4 // (op 9608) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[80:83], a[8:11], v54, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7530) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[84:87], a[12:15], v54, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7828) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27268) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27268) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[72:75], a[16:19], v55, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8631) 
v_mov_b64 v[118:119], v[50:51] // (op 27268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[76:79], a[20:23], v55, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8539) 
v_mov_b64 v[120:121], v[118:119] // (op 27268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[80:83], a[24:27], v55, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8447) 
s_mov_b32 m0, s53 // (op 27268) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[84:87], a[28:31], v55, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8355) 
s_mov_b32 s88, 256 // (op 27268) call()
buffer_load_dwordx4 v120, s[72:75], s88 offen offset: 0 lds // (op 27268) Load value direct to lds
v_add_co_u32 v120, s[90:91], v120, v116 // (op 27268) least significant half
v_addc_co_u32 v121, s[90:91], 0, v121, s[90:91] // (op 27268) most significant half

s_add_u32 m0, m0, 4096 // (op 27268) 
buffer_load_dwordx4 v120, s[72:75], s88 offen offset: 0 lds // (op 27268) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[72:75], a[32:35], v100, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9921) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[76:79], a[36:39], v100, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9829) 
v_mov_b64 v[116:117], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[80:83], a[40:43], v100, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9737) 
v_mov_b32 v42, v56 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[84:87], a[44:47], v100, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9645) 
v_mov_b32 v106, v42 // (op 25) 
buffer_load_ubyte v60, v106, s[80:83], 0 offen offset:16 // (op 25) Load value
v_add_u32 v106, v106, v116 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[72:75], a[48:51], v101, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11211) 
buffer_load_ubyte v61, v106, s[80:83], 0 offen offset:16 // (op 25) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[76:79], a[52:55], v101, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11119) 
s_barrier  // (op 30012) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[80:83], a[56:59], v101, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11027) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 27263) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[84:87], a[60:63], v101, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10935) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27263) 

v_lshrrev_b64 v[116:117], 3, s[90:91] // (op 27263) 


v_mov_b64 v[120:121], v[46:47] // (op 27263) 
s_lshl_b64 s[88:89], s[30:31], 7 // (op 27263) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27263) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27263) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27263) 

v_mov_b32 v106, s90 // (op 27263) 
v_mov_b32 v115, s91 // (op 27263) 
v_add_co_u32 v122, s[88:89], v120, v106 // (op 27263) least significant half
v_addc_co_u32 v123, s[88:89], v121, v115, s[88:89] // (op 27263) most significant half




s_add_u32 s88, 8192, s52 // (op 27263) 
s_nop 0
s_mov_b32 m0, s88 // (op 27263) reg expression
buffer_load_dwordx4 v122, s[68:71], 0 offen offset: 0 lds // (op 27263) Load value direct to lds
v_add_co_u32 v122, s[90:91], v122, v116 // (op 27263) least significant half
v_addc_co_u32 v123, s[90:91], 0, v123, s[90:91] // (op 27263) most significant half

s_add_u32 m0, m0, 4096 // (op 27263) 
buffer_load_dwordx4 v122, s[68:71], 0 offen offset: 0 lds // (op 27263) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 17) convert

v_mov_b32 v106, v53 // (op 17) 
s_lshl_b32 s88, s38, 4 // (op 17) 
v_add_u32 v115, v106, s88 // (op 17) 

buffer_load_ubyte v62, v115, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v115, v115, v116 // (op 17) 
buffer_load_ubyte v63, v115, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 30291) 



v_add_u32 v115, 66560, v110 // (op 6965) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v64 // (op 6965) Store local data 

v_add_u32 v115, 66564, v110 // (op 6965) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v65 // (op 6965) Store local data 

s_barrier  // (op 30285) 



v_add_u32 v115, 67072, v109 // (op 6964) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v66 // (op 6964) Store local data 

v_add_u32 v115, 67200, v109 // (op 6964) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v67 // (op 6964) Store local data 

s_barrier  // (op 30271) 



ds_read_b128 v[122:125], v24 offset:16384 // (op 6957) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30288) 



v_add_u32 v115, 66560, v25 // (op 6961) 
ds_read_u8 v116, v115 // (op 6961) Load local data 




ds_read_b128 v[126:129], v24 offset:18432 // (op 7239) Load local data 



v_add_u32 v115, 66688, v25 // (op 7243) 
ds_read_u8 v55, v115 // (op 7243) Load local data 




ds_read_b128 v[92:95], v24 offset:20480 // (op 7537) Load local data 



v_add_u32 v115, 66816, v25 // (op 7541) 
ds_read_u8 v100, v115 // (op 7541) Load local data 




ds_read_b128 v[96:99], v24 offset:22528 // (op 7835) Load local data 



v_add_u32 v115, 66944, v25 // (op 7839) 
ds_read_u8 v117, v115 // (op 7839) Load local data 

s_barrier  // (op 30262) 
s_barrier  // (op 30702) 



ds_read_b64_tr_b4 v[88:89], v23 offset:24576 // (op 6955) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v23 offset:25600 // (op 6955) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30281) 


v_add_u32 v115, 67072, v22 // (op 6959) 
ds_read_u8 v54, v115 // (op 6959) Load local data 




ds_read_b64_tr_b4 v[130:131], v23 offset:24592 // (op 8360) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v23 offset:25616 // (op 8360) Transpose load local data 


v_add_u32 v115, 67104, v22 // (op 8363) 
ds_read_u8 v57, v115 // (op 8363) Load local data 




ds_read_b64_tr_b4 v[134:135], v23 offset:24608 // (op 9650) Transpose load local data 
ds_read_b64_tr_b4 v[136:137], v23 offset:25632 // (op 9650) Transpose load local data 


v_add_u32 v115, 67136, v22 // (op 9653) 
ds_read_u8 v44, v115 // (op 9653) Load local data 




ds_read_b64_tr_b4 v[138:139], v23 offset:24624 // (op 10940) Transpose load local data 
ds_read_b64_tr_b4 v[140:141], v23 offset:25648 // (op 10940) Transpose load local data 


v_add_u32 v115, 67168, v22 // (op 10943) 
ds_read_u8 v45, v115 // (op 10943) Load local data 

s_barrier  // (op 30760) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[122:125], a[0:3], v54, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6962) 
s_barrier  // (op 30278) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[126:129], a[4:7], v54, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7233) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[92:95], a[8:11], v54, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7531) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27344) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[96:99], a[12:15], v54, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7829) 
v_lshrrev_b64 v[142:143], 3, s[90:91] // (op 27344) 


v_mov_b64 v[144:145], v[118:119] // (op 27344) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[130:133], v[122:125], a[16:19], v57, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8632) 
s_add_u32 s88, 16384, s55 // (op 27344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:133], v[126:129], a[20:23], v57, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8540) 
s_mov_b32 m0, s88 // (op 27344) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[130:133], v[92:95], a[24:27], v57, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8448) 
s_mov_b32 s89, 320 // (op 27344) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[130:133], v[96:99], a[28:31], v57, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8356) 
buffer_load_dwordx4 v144, s[72:75], s89 offen offset: 0 lds // (op 27344) Load value direct to lds

v_add_co_u32 v144, s[90:91], v144, v142 // (op 27344) least significant half
v_addc_co_u32 v145, s[90:91], 0, v145, s[90:91] // (op 27344) most significant half

s_add_u32 m0, m0, 4096 // (op 27344) 
buffer_load_dwordx4 v144, s[72:75], s89 offen offset: 0 lds // (op 27344) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:137], v[122:125], a[32:35], v44, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9922) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:137], v[126:129], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9830) 
v_mov_b64 v[130:131], s[44:45] // (op 6968) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:137], v[92:95], a[40:43], v44, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9738) 
v_mov_b32 v115, v42 // (op 6968) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:137], v[96:99], a[44:47], v44, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9646) 
buffer_load_ubyte v64, v115, s[80:83], 0 offen offset:20 // (op 6968) Load value

v_add_u32 v115, v115, v130 // (op 6968) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[138:141], v[122:125], a[48:51], v45, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11212) 
buffer_load_ubyte v65, v115, s[80:83], 0 offen offset:20 // (op 6968) Load value




s_barrier  // (op 30268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[138:141], v[126:129], a[52:55], v45, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11120) 


s_lshl_b64 s[88:89], s[30:31], 6 // (op 27339) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[138:141], v[92:95], a[56:59], v45, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11028) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27339) 

v_lshrrev_b64 v[122:123], 3, s[90:91] // (op 27339) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:141], v[96:99], a[60:63], v45, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10936) 
s_lshl_b64 s[88:89], s[30:31], 7 // (op 27339) 


s_lshl_b64 s[90:91], s[88:89], 2 // (op 27339) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27339) 

s_mul_i32 s91, 5, s89 // (op 27339) most significant: low of low * high
s_mul_i32 s90, 5, s88 // (op 27339) least significant
s_mul_hi_u32 s92, 5, s88 // (op 27339) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27339) most significant: sum


v_mov_b32 v115, s90 // (op 27339) 
v_mov_b32 v116, s91 // (op 27339) 
v_add_co_u32 v124, s[88:89], v120, v115 // (op 27339) least significant half
v_addc_co_u32 v125, s[88:89], v121, v116, s[88:89] // (op 27339) most significant half




s_add_u32 s88, 24576, s54 // (op 27339) 
s_nop 0
s_mov_b32 m0, s88 // (op 27339) reg expression
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27339) Load value direct to lds
v_add_co_u32 v124, s[90:91], v124, v122 // (op 27339) least significant half
v_addc_co_u32 v125, s[90:91], 0, v125, s[90:91] // (op 27339) most significant half

s_add_u32 m0, m0, 4096 // (op 27339) 
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27339) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 6967) convert

s_lshl_b32 s88, s38, 2 // (op 6967) 
s_mul_i32 s89, 5, s88 // (op 6967) 

v_add_u32 v115, v106, s89 // (op 6967) 

buffer_load_ubyte v66, v115, s[76:79], 0 offen offset: 0 // (op 6967) Load value
v_add_u32 v115, v115, v116 // (op 6967) 
buffer_load_ubyte v67, v115, s[76:79], 0 offen offset: 0 // (op 6967) Load value


s_barrier  // (op 30323) 



v_add_u32 v115, 66560, v112 // (op 7005) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v68 // (op 7005) Store local data 

v_add_u32 v115, 66564, v112 // (op 7005) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v69 // (op 7005) Store local data 

s_barrier  // (op 30317) 



v_add_u32 v115, 67072, v111 // (op 7004) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v70 // (op 7004) Store local data 

v_add_u32 v115, 67200, v111 // (op 7004) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v71 // (op 7004) Store local data 

s_barrier  // (op 30303) 



ds_read_b128 v[72:75], v28 offset:32768 // (op 6997) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30320) 



v_add_u32 v115, 66560, v29 // (op 7001) 
ds_read_u8 v52, v115 // (op 7001) Load local data 




ds_read_b128 v[76:79], v28 offset:34816 // (op 7247) Load local data 



v_add_u32 v115, 66688, v29 // (op 7251) 
ds_read_u8 v57, v115 // (op 7251) Load local data 




ds_read_b128 v[80:83], v28 offset:36864 // (op 7545) Load local data 



v_add_u32 v115, 66816, v29 // (op 7549) 
ds_read_u8 v44, v115 // (op 7549) Load local data 




ds_read_b128 v[84:87], v28 offset:38912 // (op 7843) Load local data 



v_add_u32 v115, 66944, v29 // (op 7847) 
ds_read_u8 v101, v115 // (op 7847) Load local data 

s_barrier  // (op 30294) 
s_barrier  // (op 30708) 



ds_read_b64_tr_b4 v[88:89], v27 offset:40960 // (op 6995) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v27 offset:41984 // (op 6995) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30313) 


v_add_u32 v115, 67072, v26 // (op 6999) 
ds_read_u8 v54, v115 // (op 6999) Load local data 




ds_read_b64_tr_b4 v[102:103], v27 offset:40976 // (op 8366) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v27 offset:42000 // (op 8366) Transpose load local data 


v_add_u32 v115, 67104, v26 // (op 8369) 
ds_read_u8 v55, v115 // (op 8369) Load local data 




ds_read_b64_tr_b4 v[122:123], v27 offset:40992 // (op 9656) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v27 offset:42016 // (op 9656) Transpose load local data 


v_add_u32 v115, 67136, v26 // (op 9659) 
ds_read_u8 v100, v115 // (op 9659) Load local data 




ds_read_b64_tr_b4 v[126:127], v27 offset:41008 // (op 10946) Transpose load local data 
ds_read_b64_tr_b4 v[128:129], v27 offset:42032 // (op 10946) Transpose load local data 


v_add_u32 v115, 67168, v26 // (op 10949) 
ds_read_u8 v116, v115 // (op 10949) Load local data 

s_barrier  // (op 30766) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[72:75], a[0:3], v54, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7002) 
s_barrier  // (op 30310) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[76:79], a[4:7], v54, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7234) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27354) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[80:83], a[8:11], v54, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7532) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27354) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[84:87], a[12:15], v54, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7830) 
v_lshrrev_b64 v[130:131], 3, s[90:91] // (op 27354) 


v_mov_b64 v[132:133], v[118:119] // (op 27354) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[102:105], v[72:75], a[16:19], v55, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8633) 
s_add_u32 s88, 32768, s85 // (op 27354) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:105], v[76:79], a[20:23], v55, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8541) 
s_mov_b32 m0, s88 // (op 27354) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:105], v[80:83], a[24:27], v55, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8449) 
s_mov_b32 s89, 384 // (op 27354) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:105], v[84:87], a[28:31], v55, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8357) 
buffer_load_dwordx4 v132, s[72:75], s89 offen offset: 0 lds // (op 27354) Load value direct to lds
v_add_co_u32 v132, s[90:91], v132, v130 // (op 27354) least significant half
v_addc_co_u32 v133, s[90:91], 0, v133, s[90:91] // (op 27354) most significant half

s_add_u32 m0, m0, 4096 // (op 27354) 
buffer_load_dwordx4 v132, s[72:75], s89 offen offset: 0 lds // (op 27354) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:125], v[72:75], a[32:35], v100, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9923) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[122:125], v[76:79], a[36:39], v100, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9831) 
v_mov_b64 v[130:131], s[44:45] // (op 7008) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[122:125], v[80:83], a[40:43], v100, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9739) 
v_mov_b32 v115, v42 // (op 7008) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[122:125], v[84:87], a[44:47], v100, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9647) 
buffer_load_ubyte v68, v115, s[80:83], 0 offen offset:24 // (op 7008) Load value

v_add_u32 v115, v115, v130 // (op 7008) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[126:129], v[72:75], a[48:51], v116, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11213) 
buffer_load_ubyte v69, v115, s[80:83], 0 offen offset:24 // (op 7008) Load value


s_barrier  // (op 30300) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:129], v[76:79], a[52:55], v116, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11121) 

s_lshl_b64 s[88:89], s[30:31], 6 // (op 27349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:129], v[80:83], a[56:59], v116, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11029) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27349) 

v_lshrrev_b64 v[122:123], 3, s[90:91] // (op 27349) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:129], v[84:87], a[60:63], v116, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10937) 
s_lshl_b64 s[88:89], s[30:31], 7 // (op 27349) 


s_lshl_b64 s[90:91], s[88:89], 2 // (op 27349) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27349) 

s_mul_i32 s91, 6, s89 // (op 27349) most significant: low of low * high
s_mul_i32 s90, 6, s88 // (op 27349) least significant
s_mul_hi_u32 s92, 6, s88 // (op 27349) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27349) most significant: sum


v_mov_b32 v115, s90 // (op 27349) 
v_mov_b32 v116, s91 // (op 27349) 
v_add_co_u32 v124, s[88:89], v120, v115 // (op 27349) least significant half
v_addc_co_u32 v125, s[88:89], v121, v116, s[88:89] // (op 27349) most significant half




s_add_u32 s88, 40960, s84 // (op 27349) 
s_nop 0
s_mov_b32 m0, s88 // (op 27349) reg expression
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27349) Load value direct to lds
v_add_co_u32 v124, s[90:91], v124, v122 // (op 27349) least significant half
v_addc_co_u32 v125, s[90:91], 0, v125, s[90:91] // (op 27349) most significant half

s_add_u32 m0, m0, 4096 // (op 27349) 
buffer_load_dwordx4 v124, s[68:71], 0 offen offset: 0 lds // (op 27349) Load value direct to lds




v_mov_b64 v[116:117], s[38:39] // (op 7007) convert

s_lshl_b32 s88, s38, 2 // (op 7007) 
s_mul_i32 s89, 6, s88 // (op 7007) 

v_add_u32 v115, v106, s89 // (op 7007) 

buffer_load_ubyte v70, v115, s[76:79], 0 offen offset: 0 // (op 7007) Load value
v_add_u32 v115, v115, v116 // (op 7007) 
buffer_load_ubyte v71, v115, s[76:79], 0 offen offset: 0 // (op 7007) Load value


s_barrier  // (op 30355) 



v_add_u32 v115, 66560, v114 // (op 7045) 
s_waitcnt vmcnt(29)
ds_write_b8 v115, v58 // (op 7045) Store local data 

v_add_u32 v115, 66564, v114 // (op 7045) 
s_waitcnt vmcnt(28)
ds_write_b8 v115, v59 // (op 7045) Store local data 

s_barrier  // (op 30349) 



v_add_u32 v115, 67072, v113 // (op 7044) 
s_waitcnt vmcnt(25)
ds_write_b8 v115, v48 // (op 7044) Store local data 

v_add_u32 v115, 67200, v113 // (op 7044) 
s_waitcnt vmcnt(24)
ds_write_b8 v115, v49 // (op 7044) Store local data 

s_barrier  // (op 30335) 



ds_read_b128 v[92:95], v32 offset:49152 // (op 7037) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30352) 



v_add_u32 v115, 66560, v33 // (op 7041) 
ds_read_u8 v45, v115 // (op 7041) Load local data 




ds_read_b128 v[96:99], v32 offset:51200 // (op 7255) Load local data 



v_add_u32 v115, 66688, v33 // (op 7259) 
ds_read_u8 v55, v115 // (op 7259) Load local data 




ds_read_b128 v[102:105], v32 offset:53248 // (op 7553) Load local data 



v_add_u32 v115, 66816, v33 // (op 7557) 
ds_read_u8 v100, v115 // (op 7557) Load local data 




ds_read_b128 v[122:125], v32 offset:55296 // (op 7851) Load local data 



v_add_u32 v115, 66944, v33 // (op 7855) 
ds_read_u8 v116, v115 // (op 7855) Load local data 

s_barrier  // (op 30326) 
s_barrier  // (op 30714) 



ds_read_b64_tr_b4 v[88:89], v31 offset:57344 // (op 7035) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v31 offset:58368 // (op 7035) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30345) 


v_add_u32 v115, 67072, v30 // (op 7039) 
ds_read_u8 v54, v115 // (op 7039) Load local data 




ds_read_b64_tr_b4 v[126:127], v31 offset:57360 // (op 8372) Transpose load local data 
ds_read_b64_tr_b4 v[128:129], v31 offset:58384 // (op 8372) Transpose load local data 


v_add_u32 v115, 67104, v30 // (op 8375) 
ds_read_u8 v57, v115 // (op 8375) Load local data 




ds_read_b64_tr_b4 v[130:131], v31 offset:57376 // (op 9662) Transpose load local data 
ds_read_b64_tr_b4 v[132:133], v31 offset:58400 // (op 9662) Transpose load local data 


v_add_u32 v115, 67136, v30 // (op 9665) 
ds_read_u8 v44, v115 // (op 9665) Load local data 




ds_read_b64_tr_b4 v[134:135], v31 offset:57392 // (op 10952) Transpose load local data 
ds_read_b64_tr_b4 v[136:137], v31 offset:58416 // (op 10952) Transpose load local data 


v_add_u32 v115, 67168, v30 // (op 10955) 
ds_read_u8 v101, v115 // (op 10955) Load local data 

s_barrier  // (op 30772) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[92:95], a[0:3], v54, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7042) 
s_barrier  // (op 30342) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[96:99], a[4:7], v54, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7235) 
s_lshl_b64 s[88:89], s[34:35], 6 // (op 27364) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[102:105], a[8:11], v54, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7533) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27364) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[122:125], a[12:15], v54, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7831) 
v_lshrrev_b64 v[138:139], 3, s[90:91] // (op 27364) 


v_mov_b64 v[140:141], v[118:119] // (op 27364) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[126:129], v[92:95], a[16:19], v57, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8634) 
s_add_u32 s88, 49152, s87 // (op 27364) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:129], v[96:99], a[20:23], v57, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8542) 
s_mov_b32 m0, s88 // (op 27364) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:129], v[102:105], a[24:27], v57, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8450) 
s_mov_b32 s89, 448 // (op 27364) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[126:129], v[122:125], a[28:31], v57, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8358) 
buffer_load_dwordx4 v140, s[72:75], s89 offen offset: 0 lds // (op 27364) Load value direct to lds

v_add_co_u32 v140, s[90:91], v140, v138 // (op 27364) least significant half
v_addc_co_u32 v141, s[90:91], 0, v141, s[90:91] // (op 27364) most significant half

s_add_u32 m0, m0, 4096 // (op 27364) 
buffer_load_dwordx4 v140, s[72:75], s89 offen offset: 0 lds // (op 27364) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:133], v[92:95], a[32:35], v44, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9924) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:133], v[96:99], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9832) 
v_add_co_u32 v50, vcc, 256, v50 // (op 27688) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:133], v[102:105], a[40:43], v44, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9740) 
v_addc_co_u32 v51, vcc, 0, v51, vcc // (op 27688) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:133], v[122:125], a[44:47], v44, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9648) 
v_mov_b64 v[118:119], s[44:45] // (op 7048) convert


v_mov_b32 v115, v42 // (op 7048) 
buffer_load_ubyte v58, v115, s[80:83], 0 offen offset:28 // (op 7048) Load value
v_add_u32 v115, v115, v118 // (op 7048) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:137], v[92:95], a[48:51], v101, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11214) 
buffer_load_ubyte v59, v115, s[80:83], 0 offen offset:28 // (op 7048) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:137], v[96:99], a[52:55], v101, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11122) 
s_barrier  // (op 30332) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:137], v[102:105], a[56:59], v101, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11030) 
v_add_u32 v56, 16, v56 // (op 27944) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 27359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:137], v[122:125], a[60:63], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10938) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27359) 




v_lshrrev_b64 v[118:119], 3, s[90:91] // (op 27359) 


s_lshl_b64 s[88:89], s[30:31], 7 // (op 27359) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27359) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27359) 

s_mul_i32 s91, 7, s89 // (op 27359) most significant: low of low * high
s_mul_i32 s90, 7, s88 // (op 27359) least significant
s_mul_hi_u32 s92, 7, s88 // (op 27359) most significant: high of low * low
s_add_u32 s91, s91, s92 // (op 27359) most significant: sum


v_mov_b32 v42, s90 // (op 27359) 
v_mov_b32 v115, s91 // (op 27359) 
v_add_co_u32 v116, s[88:89], v120, v42 // (op 27359) least significant half
v_addc_co_u32 v117, s[88:89], v121, v115, s[88:89] // (op 27359) most significant half




s_add_u32 s88, 57344, s86 // (op 27359) 
s_nop 0
s_mov_b32 m0, s88 // (op 27359) reg expression
buffer_load_dwordx4 v116, s[68:71], 0 offen offset: 0 lds // (op 27359) Load value direct to lds
v_add_co_u32 v116, s[90:91], v116, v118 // (op 27359) least significant half
v_addc_co_u32 v117, s[90:91], 0, v117, s[90:91] // (op 27359) most significant half

s_add_u32 m0, m0, 4096 // (op 27359) 
buffer_load_dwordx4 v116, s[68:71], 0 offen offset: 0 lds // (op 27359) Load value direct to lds





v_mov_b64 v[116:117], s[38:39] // (op 7047) convert

s_lshl_b32 s88, s38, 2 // (op 7047) 
s_mul_i32 s89, 7, s88 // (op 7047) 

v_add_u32 v42, v106, s89 // (op 7047) 

s_lshl_b64 s[88:89], s[30:31], 7 // (op 27560) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 27560) 

s_lshr_b64 s[88:89], s[90:91], 3 // (op 27560) 

s_lshl_b64 s[90:91], s[88:89], 2 // (op 27560) 

v_mov_b32 v115, s90 // (op 27560) 
v_mov_b32 v118, s91 // (op 27560) 
v_add_co_u32 v46, s[88:89], v46, v115 // (op 27560) least significant half
v_addc_co_u32 v47, s[88:89], v47, v118, s[88:89] // (op 27560) most significant half




buffer_load_ubyte v48, v42, s[76:79], 0 offen offset: 0 // (op 7047) Load value
v_add_u32 v42, v42, v116 // (op 7047) 
buffer_load_ubyte v49, v42, s[76:79], 0 offen offset: 0 // (op 7047) Load value



s_barrier  // (op 30072) 
s_lshl_b32 s88, s38, 4 // (op 27816) 



v_add_u32 v53, v53, s88 // (op 27816) 

v_add_u32 v42, 65536, v108 // (op 30) 
s_waitcnt vmcnt(29)
ds_write_b8 v42, v60 // (op 30) Store local data 

v_add_u32 v42, 65540, v108 // (op 30) 
s_waitcnt vmcnt(28)
ds_write_b8 v42, v61 // (op 30) Store local data 

s_barrier  // (op 30051) 



v_add_u32 v42, 66048, v107 // (op 21) 
s_waitcnt vmcnt(25)
ds_write_b8 v42, v62 // (op 21) Store local data 

v_add_u32 v42, 66176, v107 // (op 21) 
s_waitcnt vmcnt(24)
ds_write_b8 v42, v63 // (op 21) Store local data 

s_barrier  // (op 30037) 



ds_read_b128 v[72:75], v4 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30079) 



v_add_u32 v42, 65536, v5 // (op 85) 
ds_read_u8 v52, v42 // (op 85) Load local data 




ds_read_b128 v[76:79], v4 offset:2048 // (op 7227) Load local data 



v_add_u32 v42, 65664, v5 // (op 7231) 
ds_read_u8 v57, v42 // (op 7231) Load local data 




ds_read_b128 v[80:83], v4 offset:4096 // (op 7525) Load local data 



v_add_u32 v42, 65792, v5 // (op 7529) 
ds_read_u8 v44, v42 // (op 7529) Load local data 




ds_read_b128 v[84:87], v4 offset:6144 // (op 7823) Load local data 



v_add_u32 v42, 65920, v5 // (op 7827) 
ds_read_u8 v45, v42 // (op 7827) Load local data 

s_barrier  // (op 30018) 



ds_read_b64_tr_b4 v[88:89], v3 offset:8192 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:9216 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30058) 


v_add_u32 v42, 66048, v0 // (op 80) 
ds_read_u8 v54, v42 // (op 80) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:8208 // (op 8350) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:9232 // (op 8350) Transpose load local data 


v_add_u32 v42, 66080, v0 // (op 8353) 
ds_read_u8 v55, v42 // (op 8353) Load local data 




ds_read_b64_tr_b4 v[96:97], v3 offset:8224 // (op 9640) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v3 offset:9248 // (op 9640) Transpose load local data 


v_add_u32 v42, 66112, v0 // (op 9643) 
ds_read_u8 v100, v42 // (op 9643) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:8240 // (op 10930) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:9264 // (op 10930) Transpose load local data 


v_add_u32 v42, 66144, v0 // (op 10933) 
ds_read_u8 v101, v42 // (op 10933) Load local data 

s_and_b32 s88, s7, 4294967292 // (op 9606) 
s_cmp_lt_u32 s1, s88 // (op 9606) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9606 // (op 9606) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_9606 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_9606:



























s_barrier  // (op 30726) 
s_and_b32 s52, s7, 4294967292 // (op 27448) 
v_and_b32 v42, -4, v1 // (op 27448) 
s_add_u32 s53, s0, s52 // (op 27448) 

v_sub_u32 v46, v1, v42 // (op 27448) 

v_lshlrev_b32 v42, 5, v46 // (op 27448) 

s_mul_i32 s52, s16, s8 // (op 27448) 
s_add_u32 s52, s52, s53 // (op 27448) 

v_lshrrev_b32 v46, 2, v1 // (op 27448) 
s_mul_hi_u32 s53, s52, s13 // (op 27448) 
s_sub_u32 s54, s52, s53 // (op 27448) 
s_lshr_b32 s55, s54, 1 // (op 27448) 

s_add_u32 s54, s55, s53 // (op 27448) 


s_lshr_b32 s53, s54, s11 // (op 27448) 

s_and_b32 s54, s11, 2147483648 // (op 27448) 
s_cmp_eq_i32 s54, 0 // (op 27448) 
s_mov_b32 s55, scc // (op 27448) 

s_cmp_eq_i32 s55, 0 // (op 27448) 
s_cselect_b32 s54, s52, s53 // (op 27448) 


s_mul_hi_u32 s53, s54, s12 // (op 27448) 
s_sub_u32 s55, s54, s53 // (op 27448) 
s_lshr_b32 s84, s55, 1 // (op 27448) 

s_add_u32 s55, s84, s53 // (op 27448) 


s_lshr_b32 s53, s55, s10 // (op 27448) 

s_and_b32 s55, s10, 2147483648 // (op 27448) 
s_cmp_eq_i32 s55, 0 // (op 27448) 
s_mov_b32 s84, scc // (op 27448) 

s_cmp_eq_i32 s84, 0 // (op 27448) 
s_cselect_b32 s55, s54, s53 // (op 27448) 


v_lshl_add_u32 v47, s55, 7, v42 // (op 27448) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27448) 
v_mov_b32 v57, 0 // (op 27448) convert
v_mov_b32 v56, v47 // (op 27448) convert

s_mul_i32 s53, s54, s84 // (op 27448) 


s_sub_u32 s54, s52, s53 // (op 27448) 


v_lshl_add_u32 v42, s54, 7, v46 // (op 27448) 


v_mul_lo_u32 v47, v42, s31 // (op 27448) most significant: low of low * high
v_mul_hi_u32 v50, v42, s30 // (op 27448) most significant: high of low * low
v_mul_lo_u32 v46, v42, s30 // (op 27448) least significant: low of low * low
v_add_u32 v47, v47, v50 // (op 27448) most significant: sum


v_add_co_u32 v50, s[52:53], v56, v46 // (op 27448) least significant half
v_addc_co_u32 v51, s[52:53], v57, v47, s[52:53] // (op 27448) most significant half

v_lshlrev_b64 v[50:51], 2, v[50:51] // (op 27448) 


v_lshrrev_b64 v[46:47], 3, v[50:51] // (op 27448) 

s_mul_i32 s53, s30, s3 // (op 27450) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 27450) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 27450) least significant
s_mul_hi_u32 s55, s30, s2 // (op 27450) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27450) most significant: sum
s_add_u32 s53, s53, s55 // (op 27450) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27450) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27450) 

s_mov_b32 s68, s32 // (op 27450) reg expression
s_mov_b32 s69, s33 // (op 27450) reg expression
s_mov_b32 s70, s52 // (op 27450) reg expression
s_mov_b32 s71, 131072 // (op 27450) call()

s_and_b32 s52, s7, 4294967292 // (op 27612) 
v_lshrrev_b32 v42, 2, v1 // (op 27612) 
s_add_u32 s53, s0, s52 // (op 27612) 

v_and_b32 v50, -4, v1 // (op 27612) 
v_sub_u32 v51, v1, v50 // (op 27612) 

s_mul_i32 s52, s16, s8 // (op 27612) 
s_add_u32 s52, s52, s53 // (op 27612) 

v_lshlrev_b32 v50, 5, v51 // (op 27612) 

s_mul_hi_u32 s53, s52, s13 // (op 27612) 
s_sub_u32 s54, s52, s53 // (op 27612) 
s_lshr_b32 s55, s54, 1 // (op 27612) 

s_add_u32 s54, s55, s53 // (op 27612) 


s_lshr_b32 s53, s54, s11 // (op 27612) 

s_and_b32 s54, s11, 2147483648 // (op 27612) 
s_cmp_eq_i32 s54, 0 // (op 27612) 
s_mov_b32 s55, scc // (op 27612) 

s_cmp_eq_i32 s55, 0 // (op 27612) 
s_cselect_b32 s54, s52, s53 // (op 27612) 


s_mul_hi_u32 s53, s54, s12 // (op 27612) 
s_sub_u32 s55, s54, s53 // (op 27612) 
s_lshr_b32 s84, s55, 1 // (op 27612) 

s_add_u32 s55, s84, s53 // (op 27612) 


s_lshr_b32 s53, s55, s10 // (op 27612) 

s_and_b32 s55, s10, 2147483648 // (op 27612) 
s_cmp_eq_i32 s55, 0 // (op 27612) 
s_mov_b32 s84, scc // (op 27612) 

s_cmp_eq_i32 s84, 0 // (op 27612) 
s_cselect_b32 s55, s54, s53 // (op 27612) 


s_mul_i32 s53, s55, s14 // (op 27612) 

s_sub_u32 s55, s54, s53 // (op 27612) 

v_lshl_add_u32 v51, s55, 7, v42 // (op 27612) 


v_mul_lo_u32 v53, v51, s35 // (op 27612) most significant: low of low * high
v_mul_hi_u32 v42, v51, s34 // (op 27612) most significant: high of low * low
v_mul_lo_u32 v52, v51, s34 // (op 27612) least significant: low of low * low
v_add_u32 v53, v53, v42 // (op 27612) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27612) 
s_mul_i32 s53, s54, s84 // (op 27612) 


s_sub_u32 s54, s52, s53 // (op 27612) 


v_lshl_add_u32 v42, s54, 7, v50 // (op 27612) 


v_mov_b32 v51, 0 // (op 27612) convert
v_mov_b32 v50, v42 // (op 27612) convert

v_add_co_u32 v56, s[52:53], v50, v52 // (op 27612) least significant half
v_addc_co_u32 v57, s[52:53], v51, v53, s[52:53] // (op 27612) most significant half

v_lshlrev_b64 v[56:57], 2, v[56:57] // (op 27612) 


v_lshrrev_b64 v[50:51], 3, v[56:57] // (op 27612) 

s_mul_i32 s53, s34, s23 // (op 27614) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 27614) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 27614) least significant
s_mul_hi_u32 s55, s34, s22 // (op 27614) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27614) most significant: sum
s_add_u32 s53, s53, s55 // (op 27614) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 27614) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 27614) 

s_mov_b32 s72, s36 // (op 27614) reg expression
s_mov_b32 s73, s37 // (op 27614) reg expression
s_mov_b32 s74, s52 // (op 27614) reg expression
s_mov_b32 s75, 131072 // (op 27614) call()

s_and_b32 s52, s7, 4294967292 // (op 27740) 
v_and_b32 v42, -128, v1 // (op 27740) 
s_add_u32 s53, s0, s52 // (op 27740) 

v_sub_u32 v52, v1, v42 // (op 27740) 

v_lshrrev_b32 v42, 7, v1 // (op 27740) 
s_mul_i32 s52, s16, s8 // (op 27740) 
s_add_u32 s52, s52, s53 // (op 27740) 

v_lshlrev_b32 v53, 1, v42 // (op 27740) 

s_mul_hi_u32 s53, s52, s13 // (op 27740) 
s_sub_u32 s54, s52, s53 // (op 27740) 
s_lshr_b32 s55, s54, 1 // (op 27740) 

s_add_u32 s54, s55, s53 // (op 27740) 


s_lshr_b32 s53, s54, s11 // (op 27740) 

s_and_b32 s54, s11, 2147483648 // (op 27740) 
s_cmp_eq_i32 s54, 0 // (op 27740) 
s_mov_b32 s55, scc // (op 27740) 

s_cmp_eq_i32 s55, 0 // (op 27740) 
s_cselect_b32 s54, s52, s53 // (op 27740) 


s_mul_hi_u32 s53, s54, s12 // (op 27740) 
s_sub_u32 s55, s54, s53 // (op 27740) 
s_lshr_b32 s84, s55, 1 // (op 27740) 

s_add_u32 s55, s84, s53 // (op 27740) 


s_lshr_b32 s53, s55, s10 // (op 27740) 

s_and_b32 s55, s10, 2147483648 // (op 27740) 
s_cmp_eq_i32 s55, 0 // (op 27740) 
s_mov_b32 s84, scc // (op 27740) 

s_cmp_eq_i32 s84, 0 // (op 27740) 
s_cselect_b32 s55, s54, s53 // (op 27740) 


v_lshl_add_u32 v42, s55, 7, v52 // (op 27740) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 27740) 
s_mul_i32 s53, s54, s84 // (op 27740) 


s_sub_u32 s54, s52, s53 // (op 27740) 


v_lshl_add_u32 v52, s54, 2, v53 // (op 27740) 


v_mul_lo_u32 v53, v52, s38 // (op 27740) 

v_add_u32 v52, v42, v53 // (op 27740) 


s_mul_i32 s53, s38, s43 // (op 27742) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 27742) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 27742) least significant
s_mul_hi_u32 s55, s38, s42 // (op 27742) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27742) most significant: sum
s_add_u32 s53, s53, s55 // (op 27742) most significant: sum


s_mov_b32 s76, s40 // (op 27742) reg expression
s_mov_b32 s77, s41 // (op 27742) reg expression
s_mov_b32 s78, s52 // (op 27742) reg expression
s_mov_b32 s79, 131072 // (op 27742) call()

s_and_b32 s52, s7, 4294967292 // (op 27868) 
v_lshrrev_b32 v42, 2, v1 // (op 27868) 
s_add_u32 s53, s0, s52 // (op 27868) 

v_lshlrev_b32 v53, 1, v42 // (op 27868) 

v_and_b32 v42, -4, v1 // (op 27868) 
s_mul_i32 s52, s16, s8 // (op 27868) 
s_add_u32 s52, s52, s53 // (op 27868) 

v_sub_u32 v56, v1, v42 // (op 27868) 

s_mul_hi_u32 s53, s52, s13 // (op 27868) 
s_sub_u32 s54, s52, s53 // (op 27868) 
s_lshr_b32 s55, s54, 1 // (op 27868) 

s_add_u32 s54, s55, s53 // (op 27868) 


s_lshr_b32 s53, s54, s11 // (op 27868) 

s_and_b32 s54, s11, 2147483648 // (op 27868) 
s_cmp_eq_i32 s54, 0 // (op 27868) 
s_mov_b32 s55, scc // (op 27868) 

s_cmp_eq_i32 s55, 0 // (op 27868) 
s_cselect_b32 s54, s52, s53 // (op 27868) 


s_mul_hi_u32 s53, s54, s12 // (op 27868) 
s_sub_u32 s55, s54, s53 // (op 27868) 
s_lshr_b32 s84, s55, 1 // (op 27868) 

s_add_u32 s55, s84, s53 // (op 27868) 


s_lshr_b32 s53, s55, s10 // (op 27868) 

s_and_b32 s55, s10, 2147483648 // (op 27868) 
s_cmp_eq_i32 s55, 0 // (op 27868) 
s_mov_b32 s84, scc // (op 27868) 

s_cmp_eq_i32 s84, 0 // (op 27868) 
s_cselect_b32 s55, s54, s53 // (op 27868) 


s_mul_i32 s53, s55, s14 // (op 27868) 

s_sub_u32 s55, s54, s53 // (op 27868) 

v_lshl_add_u32 v42, s55, 7, v53 // (op 27868) 


v_mul_lo_u32 v53, v42, s44 // (op 27868) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 27868) 
s_mul_i32 s53, s54, s84 // (op 27868) 


s_sub_u32 s54, s52, s53 // (op 27868) 


v_lshl_add_u32 v42, s54, 2, v56 // (op 27868) 


v_add_u32 v56, v42, v53 // (op 27868) 


s_mul_i32 s53, s44, s23 // (op 27870) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 27870) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 27870) least significant
s_mul_hi_u32 s55, s44, s22 // (op 27870) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27870) most significant: sum
s_add_u32 s53, s53, s55 // (op 27870) most significant: sum


s_mov_b32 s80, s46 // (op 27870) reg expression
s_mov_b32 s81, s47 // (op 27870) reg expression
s_mov_b32 s82, s52 // (op 27870) reg expression
s_mov_b32 s83, 131072 // (op 27870) call()

v_and_b32 v42, -4, v1 // (op 28039) 
v_sub_u32 v53, v1, v42 // (op 28039) 

v_lshlrev_b32 v42, 5, v53 // (op 28039) 

v_lshrrev_b32 v53, 2, v1 // (op 28039) 
v_lshl_add_u32 v57, v53, 7, v42 // (op 28039) 


v_lshlrev_b32 v42, 2, v57 // (op 28039) 

v_lshrrev_b32 v53, 3, v42 // (op 28039) 

s_nop 0
v_readfirstlane_b32 s52, v53 // (op 28039) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -4, v1 // (op 28114) 
v_sub_u32 v53, v1, v42 // (op 28114) 

v_lshlrev_b32 v42, 5, v53 // (op 28114) 

v_lshrrev_b32 v53, 2, v1 // (op 28114) 
v_lshl_add_u32 v57, v53, 7, v42 // (op 28114) 


v_lshlrev_b32 v42, 2, v57 // (op 28114) 

v_lshrrev_b32 v53, 3, v42 // (op 28114) 

s_nop 0
v_readfirstlane_b32 s53, v53 // (op 28114) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v42, -128, v1 // (op 28189) 
v_sub_u32 v53, v1, v42 // (op 28189) 

v_lshrrev_b32 v42, 7, v1 // (op 28189) 
v_lshl_add_u32 v57, v42, 8, v53 // (op 28189) 


v_and_b32 v42, -4, v1 // (op 28265) 
v_sub_u32 v53, v1, v42 // (op 28265) 

v_lshrrev_b32 v42, 2, v1 // (op 28265) 
v_lshl_add_u32 v60, v42, 3, v53 // (op 28265) 


s_and_b32 s54, s7, 4294967292 // (op 10902) 
s_cmp_lt_u32 s54, s7 // (op 10901) 

















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10901 // (op 10901) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10901 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10901:
s_barrier  // (op 30015) 
s_add_u32 s54, s54, 1 // (op 10903) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 27326) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27326) 

v_lshrrev_b64 v[44:45], 3, s[86:87] // (op 27326) 


v_mov_b64 v[48:49], v[46:47] // (op 27326) 
v_mov_b64 v[54:55], v[48:49] // (op 27326) reg expression
s_add_u32 s55, 8192, s52 // (op 27326) 
s_mov_b32 m0, s55 // (op 27326) reg expression
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27326) Load value direct to lds
v_add_co_u32 v54, s[84:85], v54, v44 // (op 27326) least significant half
v_addc_co_u32 v55, s[84:85], 0, v55, s[84:85] // (op 27326) most significant half

s_add_u32 m0, m0, 4096 // (op 27326) 
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27326) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 27438) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27438) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 27438) 

v_mov_b32 v42, s84 // (op 27438) 
v_mov_b32 v44, s85 // (op 27438) 
v_add_co_u32 v46, s[86:87], v46, v42 // (op 27438) least significant half
v_addc_co_u32 v47, s[86:87], v47, v44, s[86:87] // (op 27438) most significant half




s_barrier  // (op 30034) 

s_lshl_b64 s[84:85], s[34:35], 6 // (op 27332) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 27332) 

v_lshrrev_b64 v[44:45], 3, s[86:87] // (op 27332) 


v_mov_b64 v[48:49], v[50:51] // (op 27332) 
v_mov_b64 v[54:55], v[48:49] // (op 27332) 
s_mov_b32 m0, s53 // (op 27332) reg expression
buffer_load_dwordx4 v54, s[72:75], 0 offen offset: 0 lds // (op 27332) Load value direct to lds
v_add_co_u32 v54, s[84:85], v54, v44 // (op 27332) least significant half
v_addc_co_u32 v55, s[84:85], 0, v55, s[84:85] // (op 27332) most significant half

s_add_u32 m0, m0, 4096 // (op 27332) 
buffer_load_dwordx4 v54, s[72:75], 0 offen offset: 0 lds // (op 27332) Load value direct to lds



v_add_co_u32 v50, s[84:85], 64, v50 // (op 27604) least significant half
v_addc_co_u32 v51, s[84:85], 0, v51, s[84:85] // (op 27604) most significant half


v_mov_b64 v[44:45], s[38:39] // (op 6901) convert

v_mov_b32 v42, v52 // (op 6901) 
v_mov_b32 v48, v42 // (op 6901) reg expression
buffer_load_ubyte v54, v48, s[76:79], 0 offen offset: 0 // (op 6901) Load value
v_add_u32 v48, v48, v44 // (op 6901) 
buffer_load_ubyte v55, v48, s[76:79], 0 offen offset: 0 // (op 6901) Load value



s_lshl_b32 s55, s38, 2 // (op 27732) 
v_add_u32 v52, v52, s55 // (op 27732) 


s_barrier  // (op 30054) 
v_mov_b64 v[44:45], s[44:45] // (op 6902) convert




v_mov_b32 v42, v56 // (op 6902) 
v_mov_b32 v53, v42 // (op 6902) 
v_add_u32 v48, 66048, v57 // (op 6898) 
s_waitcnt vmcnt(1)
ds_write_b8 v48, v54 // (op 6898) Store local data 

v_add_u32 v48, 66176, v57 // (op 6898) 
buffer_load_ubyte v58, v53, s[80:83], 0 offen offset: 0 // (op 6902) Load value
v_add_u32 v53, v53, v44 // (op 6902) 
buffer_load_ubyte v59, v53, s[80:83], 0 offen offset: 0 // (op 6902) Load value



v_add_u32 v56, 4, v56 // (op 27860) 
s_waitcnt vmcnt(2)
ds_write_b8 v48, v55 // (op 6898) Store local data 


s_barrier  // (op 30021) 



s_barrier  // (op 30075) 
ds_read_b64_tr_b4 v[62:63], v3 offset:8192 // (op 6889) Transpose load local data 



v_add_u32 v42, 65536, v60 // (op 6899) 
ds_read_b64_tr_b4 v[64:65], v3 offset:9216 // (op 6889) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v42, v58 // (op 6899) Store local data 

s_barrier  // (op 30041) 
v_add_u32 v42, 65540, v60 // (op 6899) 



ds_read_b128 v[66:69], v4 // (op 6891) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v42, v59 // (op 6899) Store local data 


s_waitcnt lgkmcnt(5)
s_barrier  // (op 30062) 


v_add_u32 v42, 66048, v0 // (op 6893) 
ds_read_u8 v61, v42 // (op 6893) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 30082) 



v_add_u32 v42, 65536, v5 // (op 6895) 
ds_read_u8 v44, v42 // (op 6895) Load local data 




ds_read_b128 v[70:73], v4 offset:2048 // (op 7206) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[66:69], a[0:3], v61, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6896) 



v_add_u32 v42, 65664, v5 // (op 7210) 
ds_read_u8 v45, v42 // (op 7210) Load local data 




ds_read_b128 v[74:77], v4 offset:4096 // (op 7504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[70:73], a[4:7], v61, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7211) 



v_add_u32 v42, 65792, v5 // (op 7508) 
ds_read_u8 v48, v42 // (op 7508) Load local data 




ds_read_b128 v[78:81], v4 offset:6144 // (op 7802) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[74:77], a[8:11], v61, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7509) 



v_add_u32 v42, 65920, v5 // (op 7806) 
ds_read_u8 v49, v42 // (op 7806) Load local data 




ds_read_b64_tr_b4 v[82:83], v3 offset:8208 // (op 8234) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v3 offset:9232 // (op 8234) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[78:81], a[12:15], v61, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7807) 




v_add_u32 v42, 66080, v0 // (op 8237) 
ds_read_u8 v53, v42 // (op 8237) Load local data 




ds_read_b64_tr_b4 v[62:63], v3 offset:8224 // (op 9524) Transpose load local data 
ds_read_b64_tr_b4 v[64:65], v3 offset:9248 // (op 9524) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:85], v[66:69], a[16:19], v53, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8239) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[82:85], v[70:73], a[20:23], v53, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8270) 
v_add_u32 v42, 66112, v0 // (op 9527) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[74:77], a[24:27], v53, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8301) 
ds_read_u8 v61, v42 // (op 9527) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[78:81], a[28:31], v53, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8332) 
ds_read_b64_tr_b4 v[86:87], v3 offset:8240 // (op 10814) Transpose load local data 


ds_read_b64_tr_b4 v[88:89], v3 offset:9264 // (op 10814) Transpose load local data 


v_add_u32 v42, 66144, v0 // (op 10817) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[66:69], a[32:35], v61, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9529) 
ds_read_u8 v53, v42 // (op 10817) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[70:73], a[36:39], v61, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9560) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[74:77], a[40:43], v61, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9591) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:65], v[78:81], a[44:47], v61, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9622) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[66:69], a[48:51], v53, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10819) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[70:73], a[52:55], v53, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10850) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[74:77], a[56:59], v53, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10881) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[78:81], a[60:63], v53, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10912) 




s_cmp_lt_u32 s54, s7 // (op 10901) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10901 // (op 10901) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_10901 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_10901:









s_barrier  // (op 30744) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 288) 
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_314



v_add_u32 v42, 49152, v41 // (op 12656) 
buffer_store_dword a48, v42, s[64:67], 0 offen sc1 // (op 12656) Store value

v_add_u32 v42, 49664, v41 // (op 12656) 
buffer_store_dword a49, v42, s[64:67], 0 offen sc1 // (op 12656) Store value

v_add_u32 v42, 50176, v41 // (op 12656) 
buffer_store_dword a50, v42, s[64:67], 0 offen sc1 // (op 12656) Store value

v_add_u32 v42, 50688, v41 // (op 12656) 
buffer_store_dword a51, v42, s[64:67], 0 offen sc1 // (op 12656) Store value




v_add_u32 v42, 49280, v41 // (op 12658) 
buffer_store_dword a52, v42, s[64:67], 0 offen sc1 // (op 12658) Store value

v_add_u32 v42, 49792, v41 // (op 12658) 
buffer_store_dword a53, v42, s[64:67], 0 offen sc1 // (op 12658) Store value

v_add_u32 v42, 50304, v41 // (op 12658) 
buffer_store_dword a54, v42, s[64:67], 0 offen sc1 // (op 12658) Store value

v_add_u32 v42, 50816, v41 // (op 12658) 
buffer_store_dword a55, v42, s[64:67], 0 offen sc1 // (op 12658) Store value




v_add_u32 v42, 49408, v41 // (op 12660) 
buffer_store_dword a56, v42, s[64:67], 0 offen sc1 // (op 12660) Store value

v_add_u32 v42, 49920, v41 // (op 12660) 
buffer_store_dword a57, v42, s[64:67], 0 offen sc1 // (op 12660) Store value

v_add_u32 v42, 50432, v41 // (op 12660) 
buffer_store_dword a58, v42, s[64:67], 0 offen sc1 // (op 12660) Store value

v_add_u32 v42, 50944, v41 // (op 12660) 
buffer_store_dword a59, v42, s[64:67], 0 offen sc1 // (op 12660) Store value




v_add_u32 v42, 49536, v41 // (op 12662) 
buffer_store_dword a60, v42, s[64:67], 0 offen sc1 // (op 12662) Store value

v_add_u32 v42, 50048, v41 // (op 12662) 
buffer_store_dword a61, v42, s[64:67], 0 offen sc1 // (op 12662) Store value

v_add_u32 v42, 50560, v41 // (op 12662) 
buffer_store_dword a62, v42, s[64:67], 0 offen sc1 // (op 12662) Store value

v_add_u32 v42, 51072, v41 // (op 12662) 
buffer_store_dword a63, v42, s[64:67], 0 offen sc1 // (op 12662) Store value




v_add_u32 v42, 32768, v41 // (op 12632) 
buffer_store_dword a32, v42, s[64:67], 0 offen sc1 // (op 12632) Store value

v_add_u32 v42, 33280, v41 // (op 12632) 
buffer_store_dword a33, v42, s[64:67], 0 offen sc1 // (op 12632) Store value

v_add_u32 v42, 33792, v41 // (op 12632) 
buffer_store_dword a34, v42, s[64:67], 0 offen sc1 // (op 12632) Store value

v_add_u32 v42, 34304, v41 // (op 12632) 
buffer_store_dword a35, v42, s[64:67], 0 offen sc1 // (op 12632) Store value




v_add_u32 v42, 32896, v41 // (op 12634) 
buffer_store_dword a36, v42, s[64:67], 0 offen sc1 // (op 12634) Store value

v_add_u32 v42, 33408, v41 // (op 12634) 
buffer_store_dword a37, v42, s[64:67], 0 offen sc1 // (op 12634) Store value

v_add_u32 v42, 33920, v41 // (op 12634) 
buffer_store_dword a38, v42, s[64:67], 0 offen sc1 // (op 12634) Store value

v_add_u32 v42, 34432, v41 // (op 12634) 
buffer_store_dword a39, v42, s[64:67], 0 offen sc1 // (op 12634) Store value




v_add_u32 v42, 33024, v41 // (op 12636) 
buffer_store_dword a40, v42, s[64:67], 0 offen sc1 // (op 12636) Store value

v_add_u32 v42, 33536, v41 // (op 12636) 
buffer_store_dword a41, v42, s[64:67], 0 offen sc1 // (op 12636) Store value

v_add_u32 v42, 34048, v41 // (op 12636) 
buffer_store_dword a42, v42, s[64:67], 0 offen sc1 // (op 12636) Store value

v_add_u32 v42, 34560, v41 // (op 12636) 
buffer_store_dword a43, v42, s[64:67], 0 offen sc1 // (op 12636) Store value




v_add_u32 v42, 33152, v41 // (op 12638) 
buffer_store_dword a44, v42, s[64:67], 0 offen sc1 // (op 12638) Store value

v_add_u32 v42, 33664, v41 // (op 12638) 
buffer_store_dword a45, v42, s[64:67], 0 offen sc1 // (op 12638) Store value

v_add_u32 v42, 34176, v41 // (op 12638) 
buffer_store_dword a46, v42, s[64:67], 0 offen sc1 // (op 12638) Store value

v_add_u32 v42, 34688, v41 // (op 12638) 
buffer_store_dword a47, v42, s[64:67], 0 offen sc1 // (op 12638) Store value




v_add_u32 v42, 16384, v41 // (op 12608) 
buffer_store_dword a16, v42, s[64:67], 0 offen sc1 // (op 12608) Store value

v_add_u32 v42, 16896, v41 // (op 12608) 
buffer_store_dword a17, v42, s[64:67], 0 offen sc1 // (op 12608) Store value

v_add_u32 v42, 17408, v41 // (op 12608) 
buffer_store_dword a18, v42, s[64:67], 0 offen sc1 // (op 12608) Store value

v_add_u32 v42, 17920, v41 // (op 12608) 
buffer_store_dword a19, v42, s[64:67], 0 offen sc1 // (op 12608) Store value




v_add_u32 v42, 16512, v41 // (op 12610) 
buffer_store_dword a20, v42, s[64:67], 0 offen sc1 // (op 12610) Store value

v_add_u32 v42, 17024, v41 // (op 12610) 
buffer_store_dword a21, v42, s[64:67], 0 offen sc1 // (op 12610) Store value

v_add_u32 v42, 17536, v41 // (op 12610) 
buffer_store_dword a22, v42, s[64:67], 0 offen sc1 // (op 12610) Store value

v_add_u32 v42, 18048, v41 // (op 12610) 
buffer_store_dword a23, v42, s[64:67], 0 offen sc1 // (op 12610) Store value




v_add_u32 v42, 16640, v41 // (op 12612) 
buffer_store_dword a24, v42, s[64:67], 0 offen sc1 // (op 12612) Store value

v_add_u32 v42, 17152, v41 // (op 12612) 
buffer_store_dword a25, v42, s[64:67], 0 offen sc1 // (op 12612) Store value

v_add_u32 v42, 17664, v41 // (op 12612) 
buffer_store_dword a26, v42, s[64:67], 0 offen sc1 // (op 12612) Store value

v_add_u32 v42, 18176, v41 // (op 12612) 
buffer_store_dword a27, v42, s[64:67], 0 offen sc1 // (op 12612) Store value




v_add_u32 v42, 16768, v41 // (op 12614) 
buffer_store_dword a28, v42, s[64:67], 0 offen sc1 // (op 12614) Store value

v_add_u32 v42, 17280, v41 // (op 12614) 
buffer_store_dword a29, v42, s[64:67], 0 offen sc1 // (op 12614) Store value

v_add_u32 v42, 17792, v41 // (op 12614) 
buffer_store_dword a30, v42, s[64:67], 0 offen sc1 // (op 12614) Store value

v_add_u32 v42, 18304, v41 // (op 12614) 
buffer_store_dword a31, v42, s[64:67], 0 offen sc1 // (op 12614) Store value




buffer_store_dword a0, v41, s[64:67], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v41, s[64:67], 0 offen offset:512 sc1 // (op 311) Store value
buffer_store_dword a2, v41, s[64:67], 0 offen offset:1024 sc1 // (op 311) Store value
buffer_store_dword a3, v41, s[64:67], 0 offen offset:1536 sc1 // (op 311) Store value



buffer_store_dword a4, v41, s[64:67], 0 offen offset:128 sc1 // (op 12582) Store value
buffer_store_dword a5, v41, s[64:67], 0 offen offset:640 sc1 // (op 12582) Store value
buffer_store_dword a6, v41, s[64:67], 0 offen offset:1152 sc1 // (op 12582) Store value
buffer_store_dword a7, v41, s[64:67], 0 offen offset:1664 sc1 // (op 12582) Store value



buffer_store_dword a8, v41, s[64:67], 0 offen offset:256 sc1 // (op 12583) Store value
buffer_store_dword a9, v41, s[64:67], 0 offen offset:768 sc1 // (op 12583) Store value
buffer_store_dword a10, v41, s[64:67], 0 offen offset:1280 sc1 // (op 12583) Store value
buffer_store_dword a11, v41, s[64:67], 0 offen offset:1792 sc1 // (op 12583) Store value



buffer_store_dword a12, v41, s[64:67], 0 offen offset:384 sc1 // (op 12584) Store value
buffer_store_dword a13, v41, s[64:67], 0 offen offset:896 sc1 // (op 12584) Store value
buffer_store_dword a14, v41, s[64:67], 0 offen offset:1408 sc1 // (op 12584) Store value
buffer_store_dword a15, v41, s[64:67], 0 offen offset:1920 sc1 // (op 12584) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_319
s_mov_b32 s52, 1 // (op 317) call()
s_mov_b32 s55, 0 // (op 318) convert
s_mov_b32 s54, s16 // (op 318) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 318) 
s_add_u32 s84, s48, s54 // (op 318) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 318) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 318) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_314
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_314:
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_314:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 330) 
s_mov_b32 s52, scc // (op 330) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 330) 
s_mov_b32 s53, s54 // (op 330) convert

s_sub_u32 s54, s53, 1 // (op 330) 

s_cmp_lt_u32 s19, s54 // (op 330) 
s_mov_b32 s53, scc // (op 330) 

s_and_b32 s54, s52, s53 // (op 330) 
s_cmp_lg_u32 s54, 0 // (op 330) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 28719) 
v_lshrrev_b32 v36, 5, v1 // (op 28719) 
s_add_u32 s54, s53, s52 // (op 28719) 
s_lshl_b32 s54, s54, 7 // (op 28719) 

v_lshlrev_b32 v37, 2, v36 // (op 28719) 

v_add_lshl_u32 v36, s54, v37, 7 // (op 28719) 


v_and_b32 v37, -32, v1 // (op 28719) 
v_sub_u32 v39, v1, v37 // (op 28719) 

v_add_lshl_u32 v37, v36, v39, 2 // (op 28719) 


s_lshl_b32 s53, s15, 16 // (op 28721) 
s_mov_b32 s64, s28 // (op 28721) reg expression
s_mov_b32 s65, s29 // (op 28721) reg expression
s_mov_b32 s66, s53 // (op 28721) reg expression
s_mov_b32 s67, 131072 // (op 28721) call()

RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_334:
s_add_u32 s53, s16, 1 // (op 332) 
s_add_u32 s54, s53, s52 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 332) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 332) 

s_add_u32 s84, s48, s54 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 332) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 332) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 334) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_337
s_mov_b32 s53, 0 // (op 335) call()
s_add_u32 s54, s16, 1 // (op 336) 
s_add_u32 s84, s54, s52 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 336) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 336) 

s_add_u32 s54, s48, s84 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 336) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 336) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_337:




buffer_load_dword v44, v37, s[64:67], 0 offen // (op 349) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:512 // (op 349) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1024 // (op 349) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1536 // (op 349) Load value
v_accvgpr_read v48, a0 // (op 352) reg expression
v_accvgpr_read v49, a1 // (op 352) reg expression
v_accvgpr_read v50, a2 // (op 352) reg expression
v_accvgpr_read v51, a3 // (op 352) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 354) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 354) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 354) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 354) 

v_accvgpr_write a0, v48 // (op 356) reg expression
v_accvgpr_write a1, v49 // (op 356) reg expression
v_accvgpr_write a2, v50 // (op 356) reg expression
v_accvgpr_write a3, v51 // (op 356) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:128 // (op 12731) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:640 // (op 12731) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1152 // (op 12731) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1664 // (op 12731) Load value
v_accvgpr_read v48, a4 // (op 12732) reg expression
v_accvgpr_read v49, a5 // (op 12732) reg expression
v_accvgpr_read v50, a6 // (op 12732) reg expression
v_accvgpr_read v51, a7 // (op 12732) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12733) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12733) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12733) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12733) 

v_accvgpr_write a4, v48 // (op 12734) reg expression
v_accvgpr_write a5, v49 // (op 12734) reg expression
v_accvgpr_write a6, v50 // (op 12734) reg expression
v_accvgpr_write a7, v51 // (op 12734) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:256 // (op 12738) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:768 // (op 12738) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1280 // (op 12738) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1792 // (op 12738) Load value
v_accvgpr_read v48, a8 // (op 12739) reg expression
v_accvgpr_read v49, a9 // (op 12739) reg expression
v_accvgpr_read v50, a10 // (op 12739) reg expression
v_accvgpr_read v51, a11 // (op 12739) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12740) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12740) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12740) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12740) 

v_accvgpr_write a8, v48 // (op 12741) reg expression
v_accvgpr_write a9, v49 // (op 12741) reg expression
v_accvgpr_write a10, v50 // (op 12741) reg expression
v_accvgpr_write a11, v51 // (op 12741) reg expression




buffer_load_dword v44, v37, s[64:67], 0 offen offset:384 // (op 12745) Load value
buffer_load_dword v45, v37, s[64:67], 0 offen offset:896 // (op 12745) Load value
buffer_load_dword v46, v37, s[64:67], 0 offen offset:1408 // (op 12745) Load value
buffer_load_dword v47, v37, s[64:67], 0 offen offset:1920 // (op 12745) Load value
v_accvgpr_read v48, a12 // (op 12746) reg expression
v_accvgpr_read v49, a13 // (op 12746) reg expression
v_accvgpr_read v50, a14 // (op 12746) reg expression
v_accvgpr_read v51, a15 // (op 12746) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12747) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12747) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12747) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12747) 

v_accvgpr_write a12, v48 // (op 12748) reg expression
v_accvgpr_write a13, v49 // (op 12748) reg expression
v_accvgpr_write a14, v50 // (op 12748) reg expression
v_accvgpr_write a15, v51 // (op 12748) reg expression




v_mov_b32 v36, 16384 // (op 12782) 
v_add_i32 v39, v37, v36 // (op 12782) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12782) Load value

v_mov_b32 v36, 16896 // (op 12782) 
v_add_i32 v39, v37, v36 // (op 12782) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12782) Load value

v_mov_b32 v36, 17408 // (op 12782) 
v_add_i32 v39, v37, v36 // (op 12782) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12782) Load value

v_mov_b32 v36, 17920 // (op 12782) 
v_add_i32 v39, v37, v36 // (op 12782) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12782) Load value

v_accvgpr_read v48, a16 // (op 12779) reg expression
v_accvgpr_read v49, a17 // (op 12779) reg expression
v_accvgpr_read v50, a18 // (op 12779) reg expression
v_accvgpr_read v51, a19 // (op 12779) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12780) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12780) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12780) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12780) 

v_accvgpr_write a16, v48 // (op 12781) reg expression
v_accvgpr_write a17, v49 // (op 12781) reg expression
v_accvgpr_write a18, v50 // (op 12781) reg expression
v_accvgpr_write a19, v51 // (op 12781) reg expression




v_mov_b32 v36, 16512 // (op 12787) 
v_add_i32 v39, v37, v36 // (op 12787) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12787) Load value

v_mov_b32 v36, 17024 // (op 12787) 
v_add_i32 v39, v37, v36 // (op 12787) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12787) Load value

v_mov_b32 v36, 17536 // (op 12787) 
v_add_i32 v39, v37, v36 // (op 12787) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12787) Load value

v_mov_b32 v36, 18048 // (op 12787) 
v_add_i32 v39, v37, v36 // (op 12787) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12787) Load value

v_accvgpr_read v48, a20 // (op 12784) reg expression
v_accvgpr_read v49, a21 // (op 12784) reg expression
v_accvgpr_read v50, a22 // (op 12784) reg expression
v_accvgpr_read v51, a23 // (op 12784) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12785) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12785) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12785) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12785) 

v_accvgpr_write a20, v48 // (op 12786) reg expression
v_accvgpr_write a21, v49 // (op 12786) reg expression
v_accvgpr_write a22, v50 // (op 12786) reg expression
v_accvgpr_write a23, v51 // (op 12786) reg expression




v_mov_b32 v36, 16640 // (op 12792) 
v_add_i32 v39, v37, v36 // (op 12792) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12792) Load value

v_mov_b32 v36, 17152 // (op 12792) 
v_add_i32 v39, v37, v36 // (op 12792) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12792) Load value

v_mov_b32 v36, 17664 // (op 12792) 
v_add_i32 v39, v37, v36 // (op 12792) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12792) Load value

v_mov_b32 v36, 18176 // (op 12792) 
v_add_i32 v39, v37, v36 // (op 12792) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12792) Load value

v_accvgpr_read v48, a24 // (op 12789) reg expression
v_accvgpr_read v49, a25 // (op 12789) reg expression
v_accvgpr_read v50, a26 // (op 12789) reg expression
v_accvgpr_read v51, a27 // (op 12789) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12790) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12790) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12790) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12790) 

v_accvgpr_write a24, v48 // (op 12791) reg expression
v_accvgpr_write a25, v49 // (op 12791) reg expression
v_accvgpr_write a26, v50 // (op 12791) reg expression
v_accvgpr_write a27, v51 // (op 12791) reg expression




v_mov_b32 v36, 16768 // (op 12797) 
v_add_i32 v39, v37, v36 // (op 12797) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12797) Load value

v_mov_b32 v36, 17280 // (op 12797) 
v_add_i32 v39, v37, v36 // (op 12797) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12797) Load value

v_mov_b32 v36, 17792 // (op 12797) 
v_add_i32 v39, v37, v36 // (op 12797) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12797) Load value

v_mov_b32 v36, 18304 // (op 12797) 
v_add_i32 v39, v37, v36 // (op 12797) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12797) Load value

v_accvgpr_read v48, a28 // (op 12794) reg expression
v_accvgpr_read v49, a29 // (op 12794) reg expression
v_accvgpr_read v50, a30 // (op 12794) reg expression
v_accvgpr_read v51, a31 // (op 12794) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12795) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12795) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12795) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12795) 

v_accvgpr_write a28, v48 // (op 12796) reg expression
v_accvgpr_write a29, v49 // (op 12796) reg expression
v_accvgpr_write a30, v50 // (op 12796) reg expression
v_accvgpr_write a31, v51 // (op 12796) reg expression




v_mov_b32 v36, 32768 // (op 12830) 
v_add_i32 v39, v37, v36 // (op 12830) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12830) Load value

v_mov_b32 v36, 33280 // (op 12830) 
v_add_i32 v39, v37, v36 // (op 12830) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12830) Load value

v_mov_b32 v36, 33792 // (op 12830) 
v_add_i32 v39, v37, v36 // (op 12830) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12830) Load value

v_mov_b32 v36, 34304 // (op 12830) 
v_add_i32 v39, v37, v36 // (op 12830) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12830) Load value

v_accvgpr_read v48, a32 // (op 12827) reg expression
v_accvgpr_read v49, a33 // (op 12827) reg expression
v_accvgpr_read v50, a34 // (op 12827) reg expression
v_accvgpr_read v51, a35 // (op 12827) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12828) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12828) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12828) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12828) 

v_accvgpr_write a32, v48 // (op 12829) reg expression
v_accvgpr_write a33, v49 // (op 12829) reg expression
v_accvgpr_write a34, v50 // (op 12829) reg expression
v_accvgpr_write a35, v51 // (op 12829) reg expression




v_mov_b32 v36, 32896 // (op 12835) 
v_add_i32 v39, v37, v36 // (op 12835) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12835) Load value

v_mov_b32 v36, 33408 // (op 12835) 
v_add_i32 v39, v37, v36 // (op 12835) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12835) Load value

v_mov_b32 v36, 33920 // (op 12835) 
v_add_i32 v39, v37, v36 // (op 12835) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12835) Load value

v_mov_b32 v36, 34432 // (op 12835) 
v_add_i32 v39, v37, v36 // (op 12835) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12835) Load value

v_accvgpr_read v48, a36 // (op 12832) reg expression
v_accvgpr_read v49, a37 // (op 12832) reg expression
v_accvgpr_read v50, a38 // (op 12832) reg expression
v_accvgpr_read v51, a39 // (op 12832) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12833) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12833) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12833) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12833) 

v_accvgpr_write a36, v48 // (op 12834) reg expression
v_accvgpr_write a37, v49 // (op 12834) reg expression
v_accvgpr_write a38, v50 // (op 12834) reg expression
v_accvgpr_write a39, v51 // (op 12834) reg expression




v_mov_b32 v36, 33024 // (op 12840) 
v_add_i32 v39, v37, v36 // (op 12840) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12840) Load value

v_mov_b32 v36, 33536 // (op 12840) 
v_add_i32 v39, v37, v36 // (op 12840) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12840) Load value

v_mov_b32 v36, 34048 // (op 12840) 
v_add_i32 v39, v37, v36 // (op 12840) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12840) Load value

v_mov_b32 v36, 34560 // (op 12840) 
v_add_i32 v39, v37, v36 // (op 12840) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12840) Load value

v_accvgpr_read v48, a40 // (op 12837) reg expression
v_accvgpr_read v49, a41 // (op 12837) reg expression
v_accvgpr_read v50, a42 // (op 12837) reg expression
v_accvgpr_read v51, a43 // (op 12837) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12838) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12838) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12838) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12838) 

v_accvgpr_write a40, v48 // (op 12839) reg expression
v_accvgpr_write a41, v49 // (op 12839) reg expression
v_accvgpr_write a42, v50 // (op 12839) reg expression
v_accvgpr_write a43, v51 // (op 12839) reg expression




v_mov_b32 v36, 33152 // (op 12845) 
v_add_i32 v39, v37, v36 // (op 12845) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12845) Load value

v_mov_b32 v36, 33664 // (op 12845) 
v_add_i32 v39, v37, v36 // (op 12845) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12845) Load value

v_mov_b32 v36, 34176 // (op 12845) 
v_add_i32 v39, v37, v36 // (op 12845) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12845) Load value

v_mov_b32 v36, 34688 // (op 12845) 
v_add_i32 v39, v37, v36 // (op 12845) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12845) Load value

v_accvgpr_read v48, a44 // (op 12842) reg expression
v_accvgpr_read v49, a45 // (op 12842) reg expression
v_accvgpr_read v50, a46 // (op 12842) reg expression
v_accvgpr_read v51, a47 // (op 12842) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12843) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12843) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12843) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12843) 

v_accvgpr_write a44, v48 // (op 12844) reg expression
v_accvgpr_write a45, v49 // (op 12844) reg expression
v_accvgpr_write a46, v50 // (op 12844) reg expression
v_accvgpr_write a47, v51 // (op 12844) reg expression




v_mov_b32 v36, 49152 // (op 12878) 
v_add_i32 v39, v37, v36 // (op 12878) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12878) Load value

v_mov_b32 v36, 49664 // (op 12878) 
v_add_i32 v39, v37, v36 // (op 12878) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12878) Load value

v_mov_b32 v36, 50176 // (op 12878) 
v_add_i32 v39, v37, v36 // (op 12878) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12878) Load value

v_mov_b32 v36, 50688 // (op 12878) 
v_add_i32 v39, v37, v36 // (op 12878) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12878) Load value

v_accvgpr_read v48, a48 // (op 12875) reg expression
v_accvgpr_read v49, a49 // (op 12875) reg expression
v_accvgpr_read v50, a50 // (op 12875) reg expression
v_accvgpr_read v51, a51 // (op 12875) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12876) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12876) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12876) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12876) 

v_accvgpr_write a48, v48 // (op 12877) reg expression
v_accvgpr_write a49, v49 // (op 12877) reg expression
v_accvgpr_write a50, v50 // (op 12877) reg expression
v_accvgpr_write a51, v51 // (op 12877) reg expression




v_mov_b32 v36, 49280 // (op 12883) 
v_add_i32 v39, v37, v36 // (op 12883) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12883) Load value

v_mov_b32 v36, 49792 // (op 12883) 
v_add_i32 v39, v37, v36 // (op 12883) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12883) Load value

v_mov_b32 v36, 50304 // (op 12883) 
v_add_i32 v39, v37, v36 // (op 12883) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12883) Load value

v_mov_b32 v36, 50816 // (op 12883) 
v_add_i32 v39, v37, v36 // (op 12883) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12883) Load value

v_accvgpr_read v48, a52 // (op 12880) reg expression
v_accvgpr_read v49, a53 // (op 12880) reg expression
v_accvgpr_read v50, a54 // (op 12880) reg expression
v_accvgpr_read v51, a55 // (op 12880) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12881) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12881) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12881) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12881) 

v_accvgpr_write a52, v48 // (op 12882) reg expression
v_accvgpr_write a53, v49 // (op 12882) reg expression
v_accvgpr_write a54, v50 // (op 12882) reg expression
v_accvgpr_write a55, v51 // (op 12882) reg expression




v_mov_b32 v36, 49408 // (op 12888) 
v_add_i32 v39, v37, v36 // (op 12888) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12888) Load value

v_mov_b32 v36, 49920 // (op 12888) 
v_add_i32 v39, v37, v36 // (op 12888) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12888) Load value

v_mov_b32 v36, 50432 // (op 12888) 
v_add_i32 v39, v37, v36 // (op 12888) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12888) Load value

v_mov_b32 v36, 50944 // (op 12888) 
v_add_i32 v39, v37, v36 // (op 12888) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12888) Load value

v_accvgpr_read v48, a56 // (op 12885) reg expression
v_accvgpr_read v49, a57 // (op 12885) reg expression
v_accvgpr_read v50, a58 // (op 12885) reg expression
v_accvgpr_read v51, a59 // (op 12885) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12886) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12886) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12886) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12886) 

v_accvgpr_write a56, v48 // (op 12887) reg expression
v_accvgpr_write a57, v49 // (op 12887) reg expression
v_accvgpr_write a58, v50 // (op 12887) reg expression
v_accvgpr_write a59, v51 // (op 12887) reg expression




v_mov_b32 v36, 49536 // (op 12893) 
v_add_i32 v39, v37, v36 // (op 12893) 

buffer_load_dword v44, v39, s[64:67], 0 offen // (op 12893) Load value

v_mov_b32 v36, 50048 // (op 12893) 
v_add_i32 v39, v37, v36 // (op 12893) 

buffer_load_dword v45, v39, s[64:67], 0 offen // (op 12893) Load value

v_mov_b32 v36, 50560 // (op 12893) 
v_add_i32 v39, v37, v36 // (op 12893) 

buffer_load_dword v46, v39, s[64:67], 0 offen // (op 12893) Load value

v_mov_b32 v36, 51072 // (op 12893) 
v_add_i32 v39, v37, v36 // (op 12893) 

buffer_load_dword v47, v39, s[64:67], 0 offen // (op 12893) Load value

v_accvgpr_read v48, a60 // (op 12890) reg expression
v_accvgpr_read v49, a61 // (op 12890) reg expression
v_accvgpr_read v50, a62 // (op 12890) reg expression
v_accvgpr_read v51, a63 // (op 12890) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v48, v48, v44 // (op 12891) 
s_waitcnt vmcnt(2)
v_add_f32 v49, v49, v45 // (op 12891) 
s_waitcnt vmcnt(1)
v_add_f32 v50, v50, v46 // (op 12891) 
s_waitcnt vmcnt(0)
v_add_f32 v51, v51, v47 // (op 12891) 

v_accvgpr_write a60, v48 // (op 12892) reg expression
v_accvgpr_write a61, v49 // (op 12892) reg expression
v_accvgpr_write a62, v50 // (op 12892) reg expression
v_accvgpr_write a63, v51 // (op 12892) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_330
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_330:
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v34 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_133
v_accvgpr_read v44, a0 // (op 50) 
v_accvgpr_read v45, a1 // (op 50) 
v_accvgpr_read v46, a2 // (op 50) 
v_accvgpr_read v47, a3 // (op 50) 
v_mul_f32 v48, v35, v44 // (op 50) 
v_mul_f32 v49, v35, v45 // (op 50) 
v_mul_f32 v50, v35, v46 // (op 50) 
v_mul_f32 v51, v35, v47 // (op 50) 



v_mov_b32 v36, v43 // (op 56) reg expression

buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v44, a4 // (op 12975) 
v_accvgpr_read v45, a5 // (op 12975) 
v_accvgpr_read v46, a6 // (op 12975) 
v_accvgpr_read v47, a7 // (op 12975) 
v_mul_f32 v48, v35, v44 // (op 12975) 
v_mul_f32 v49, v35, v45 // (op 12975) 
v_mul_f32 v50, v35, v46 // (op 12975) 
v_mul_f32 v51, v35, v47 // (op 12975) 



s_lshl_b32 s52, s24, 7 // (op 12977) 
v_add_u32 v36, v43, s52 // (op 12977) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 12977) Store value


v_accvgpr_read v44, a8 // (op 12987) 
v_accvgpr_read v45, a9 // (op 12987) 
v_accvgpr_read v46, a10 // (op 12987) 
v_accvgpr_read v47, a11 // (op 12987) 
v_mul_f32 v48, v35, v44 // (op 12987) 
v_mul_f32 v49, v35, v45 // (op 12987) 
v_mul_f32 v50, v35, v46 // (op 12987) 
v_mul_f32 v51, v35, v47 // (op 12987) 



s_lshl_b32 s52, s24, 8 // (op 12989) 
v_add_u32 v36, v43, s52 // (op 12989) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 12989) Store value


v_accvgpr_read v44, a12 // (op 12999) 
v_accvgpr_read v45, a13 // (op 12999) 
v_accvgpr_read v46, a14 // (op 12999) 
v_accvgpr_read v47, a15 // (op 12999) 
v_mul_f32 v48, v35, v44 // (op 12999) 
v_mul_f32 v49, v35, v45 // (op 12999) 
v_mul_f32 v50, v35, v46 // (op 12999) 
v_mul_f32 v51, v35, v47 // (op 12999) 



s_lshl_b32 s52, s24, 7 // (op 13001) 
s_mul_i32 s53, 3, s52 // (op 13001) 

v_add_u32 v36, v43, s53 // (op 13001) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13001) Store value


v_accvgpr_read v44, a16 // (op 13033) 
v_accvgpr_read v45, a17 // (op 13033) 
v_accvgpr_read v46, a18 // (op 13033) 
v_accvgpr_read v47, a19 // (op 13033) 
v_mul_f32 v48, v35, v44 // (op 13033) 
v_mul_f32 v49, v35, v45 // (op 13033) 
v_mul_f32 v50, v35, v46 // (op 13033) 
v_mul_f32 v51, v35, v47 // (op 13033) 



v_add_u32 v36, 128, v43 // (op 13036) 

buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13036) Store value


v_accvgpr_read v44, a20 // (op 13046) 
v_accvgpr_read v45, a21 // (op 13046) 
v_accvgpr_read v46, a22 // (op 13046) 
v_accvgpr_read v47, a23 // (op 13046) 
v_mul_f32 v48, v35, v44 // (op 13046) 
v_mul_f32 v49, v35, v45 // (op 13046) 
v_mul_f32 v50, v35, v46 // (op 13046) 
v_mul_f32 v51, v35, v47 // (op 13046) 



s_lshl_b32 s52, s24, 7 // (op 13038) 
s_add_u32 s52, s52, 128 // (op 13038) 
v_add_u32 v36, v43, s52 // (op 13038) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13038) Store value


v_accvgpr_read v44, a24 // (op 13051) 
v_accvgpr_read v45, a25 // (op 13051) 
v_accvgpr_read v46, a26 // (op 13051) 
v_accvgpr_read v47, a27 // (op 13051) 
v_mul_f32 v48, v35, v44 // (op 13051) 
v_mul_f32 v49, v35, v45 // (op 13051) 
v_mul_f32 v50, v35, v46 // (op 13051) 
v_mul_f32 v51, v35, v47 // (op 13051) 



s_lshl_b32 s52, s24, 8 // (op 13040) 
s_add_u32 s52, s52, 128 // (op 13040) 
v_add_u32 v36, v43, s52 // (op 13040) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13040) Store value


v_accvgpr_read v44, a28 // (op 13056) 
v_accvgpr_read v45, a29 // (op 13056) 
v_accvgpr_read v46, a30 // (op 13056) 
v_accvgpr_read v47, a31 // (op 13056) 
v_mul_f32 v48, v35, v44 // (op 13056) 
v_mul_f32 v49, v35, v45 // (op 13056) 
v_mul_f32 v50, v35, v46 // (op 13056) 
v_mul_f32 v51, v35, v47 // (op 13056) 



s_lshl_b32 s52, s24, 7 // (op 13042) 
s_mul_i32 s53, 3, s52 // (op 13042) 
s_add_u32 s53, s53, 128 // (op 13042) 

v_add_u32 v36, v43, s53 // (op 13042) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13042) Store value


v_accvgpr_read v44, a32 // (op 13101) 
v_accvgpr_read v45, a33 // (op 13101) 
v_accvgpr_read v46, a34 // (op 13101) 
v_accvgpr_read v47, a35 // (op 13101) 
v_mul_f32 v48, v35, v44 // (op 13101) 
v_mul_f32 v49, v35, v45 // (op 13101) 
v_mul_f32 v50, v35, v46 // (op 13101) 
v_mul_f32 v51, v35, v47 // (op 13101) 



v_add_u32 v36, 256, v43 // (op 13104) 

buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13104) Store value


v_accvgpr_read v44, a36 // (op 13114) 
v_accvgpr_read v45, a37 // (op 13114) 
v_accvgpr_read v46, a38 // (op 13114) 
v_accvgpr_read v47, a39 // (op 13114) 
v_mul_f32 v48, v35, v44 // (op 13114) 
v_mul_f32 v49, v35, v45 // (op 13114) 
v_mul_f32 v50, v35, v46 // (op 13114) 
v_mul_f32 v51, v35, v47 // (op 13114) 



s_lshl_b32 s52, s24, 7 // (op 13106) 
s_add_u32 s52, s52, 256 // (op 13106) 
v_add_u32 v36, v43, s52 // (op 13106) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13106) Store value


v_accvgpr_read v44, a40 // (op 13119) 
v_accvgpr_read v45, a41 // (op 13119) 
v_accvgpr_read v46, a42 // (op 13119) 
v_accvgpr_read v47, a43 // (op 13119) 
v_mul_f32 v48, v35, v44 // (op 13119) 
v_mul_f32 v49, v35, v45 // (op 13119) 
v_mul_f32 v50, v35, v46 // (op 13119) 
v_mul_f32 v51, v35, v47 // (op 13119) 



s_lshl_b32 s52, s24, 8 // (op 13108) 
s_add_u32 s52, s52, 256 // (op 13108) 
v_add_u32 v36, v43, s52 // (op 13108) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13108) Store value


v_accvgpr_read v44, a44 // (op 13124) 
v_accvgpr_read v45, a45 // (op 13124) 
v_accvgpr_read v46, a46 // (op 13124) 
v_accvgpr_read v47, a47 // (op 13124) 
v_mul_f32 v48, v35, v44 // (op 13124) 
v_mul_f32 v49, v35, v45 // (op 13124) 
v_mul_f32 v50, v35, v46 // (op 13124) 
v_mul_f32 v51, v35, v47 // (op 13124) 



s_lshl_b32 s52, s24, 7 // (op 13110) 
s_mul_i32 s53, 3, s52 // (op 13110) 
s_add_u32 s53, s53, 256 // (op 13110) 

v_add_u32 v36, v43, s53 // (op 13110) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13110) Store value


v_accvgpr_read v44, a48 // (op 13169) 
v_accvgpr_read v45, a49 // (op 13169) 
v_accvgpr_read v46, a50 // (op 13169) 
v_accvgpr_read v47, a51 // (op 13169) 
v_mul_f32 v48, v35, v44 // (op 13169) 
v_mul_f32 v49, v35, v45 // (op 13169) 
v_mul_f32 v50, v35, v46 // (op 13169) 
v_mul_f32 v51, v35, v47 // (op 13169) 



v_add_u32 v36, 384, v43 // (op 13172) 

buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13172) Store value


v_accvgpr_read v44, a52 // (op 13182) 
v_accvgpr_read v45, a53 // (op 13182) 
v_accvgpr_read v46, a54 // (op 13182) 
v_accvgpr_read v47, a55 // (op 13182) 
v_mul_f32 v48, v35, v44 // (op 13182) 
v_mul_f32 v49, v35, v45 // (op 13182) 
v_mul_f32 v50, v35, v46 // (op 13182) 
v_mul_f32 v51, v35, v47 // (op 13182) 



s_lshl_b32 s52, s24, 7 // (op 13174) 
s_add_u32 s52, s52, 384 // (op 13174) 
v_add_u32 v36, v43, s52 // (op 13174) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13174) Store value


v_accvgpr_read v44, a56 // (op 13187) 
v_accvgpr_read v45, a57 // (op 13187) 
v_accvgpr_read v46, a58 // (op 13187) 
v_accvgpr_read v47, a59 // (op 13187) 
v_mul_f32 v48, v35, v44 // (op 13187) 
v_mul_f32 v49, v35, v45 // (op 13187) 
v_mul_f32 v50, v35, v46 // (op 13187) 
v_mul_f32 v51, v35, v47 // (op 13187) 



s_lshl_b32 s52, s24, 8 // (op 13176) 
s_add_u32 s52, s52, 384 // (op 13176) 
v_add_u32 v36, v43, s52 // (op 13176) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13176) Store value


v_accvgpr_read v44, a60 // (op 13192) 
v_accvgpr_read v45, a61 // (op 13192) 
v_accvgpr_read v46, a62 // (op 13192) 
v_accvgpr_read v47, a63 // (op 13192) 
v_mul_f32 v48, v35, v44 // (op 13192) 
v_mul_f32 v49, v35, v45 // (op 13192) 
v_mul_f32 v50, v35, v46 // (op 13192) 
v_mul_f32 v51, v35, v47 // (op 13192) 



s_lshl_b32 s52, s24, 7 // (op 13178) 
s_mul_i32 s53, 3, s52 // (op 13178) 
s_add_u32 s53, s53, 384 // (op 13178) 

v_add_u32 v36, v43, s53 // (op 13178) 


buffer_store_dwordx4 v[48:51], v36, s[60:63], 0 offen // (op 13178) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_133
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_133:


v_add_u32 v36, 384, v38 // (op 13536) 

buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13536) Load value

v_accvgpr_read v48, a48 // (op 13524) 
v_accvgpr_read v49, a49 // (op 13524) 
v_accvgpr_read v50, a50 // (op 13524) 
v_accvgpr_read v51, a51 // (op 13524) 
v_mul_f32 v52, v35, v48 // (op 13524) 
v_mul_f32 v53, v35, v49 // (op 13524) 
v_mul_f32 v54, v35, v50 // (op 13524) 
v_mul_f32 v55, v35, v51 // (op 13524) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13535) 
v_mul_f32 v49, v34, v45 // (op 13535) 
v_mul_f32 v50, v34, v46 // (op 13535) 
v_mul_f32 v51, v34, v47 // (op 13535) 

v_add_f32 v44, v48, v52 // (op 13525) 
v_add_f32 v45, v49, v53 // (op 13525) 
v_add_f32 v46, v50, v54 // (op 13525) 
v_add_f32 v47, v51, v55 // (op 13525) 




v_add_u32 v36, 384, v43 // (op 13527) 

buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13527) Store value




s_lshl_b32 s52, s4, 7 // (op 13540) 
s_add_u32 s52, s52, 384 // (op 13540) 
v_add_u32 v36, v38, s52 // (op 13540) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13540) Load value

v_accvgpr_read v48, a52 // (op 13550) 
v_accvgpr_read v49, a53 // (op 13550) 
v_accvgpr_read v50, a54 // (op 13550) 
v_accvgpr_read v51, a55 // (op 13550) 
v_mul_f32 v52, v35, v48 // (op 13550) 
v_mul_f32 v53, v35, v49 // (op 13550) 
v_mul_f32 v54, v35, v50 // (op 13550) 
v_mul_f32 v55, v35, v51 // (op 13550) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13538) 
v_mul_f32 v49, v34, v45 // (op 13538) 
v_mul_f32 v50, v34, v46 // (op 13538) 
v_mul_f32 v51, v34, v47 // (op 13538) 

v_add_f32 v44, v48, v52 // (op 13539) 
v_add_f32 v45, v49, v53 // (op 13539) 
v_add_f32 v46, v50, v54 // (op 13539) 
v_add_f32 v47, v51, v55 // (op 13539) 




s_lshl_b32 s52, s24, 7 // (op 13529) 
s_add_u32 s52, s52, 384 // (op 13529) 
v_add_u32 v36, v43, s52 // (op 13529) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13529) Store value




s_lshl_b32 s52, s4, 8 // (op 13544) 
s_add_u32 s52, s52, 384 // (op 13544) 
v_add_u32 v36, v38, s52 // (op 13544) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13544) Load value

v_accvgpr_read v48, a56 // (op 13552) 
v_accvgpr_read v49, a57 // (op 13552) 
v_accvgpr_read v50, a58 // (op 13552) 
v_accvgpr_read v51, a59 // (op 13552) 
v_mul_f32 v52, v35, v48 // (op 13552) 
v_mul_f32 v53, v35, v49 // (op 13552) 
v_mul_f32 v54, v35, v50 // (op 13552) 
v_mul_f32 v55, v35, v51 // (op 13552) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13542) 
v_mul_f32 v49, v34, v45 // (op 13542) 
v_mul_f32 v50, v34, v46 // (op 13542) 
v_mul_f32 v51, v34, v47 // (op 13542) 

v_add_f32 v44, v48, v52 // (op 13543) 
v_add_f32 v45, v49, v53 // (op 13543) 
v_add_f32 v46, v50, v54 // (op 13543) 
v_add_f32 v47, v51, v55 // (op 13543) 




s_lshl_b32 s52, s24, 8 // (op 13531) 
s_add_u32 s52, s52, 384 // (op 13531) 
v_add_u32 v36, v43, s52 // (op 13531) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13531) Store value




s_lshl_b32 s52, s4, 7 // (op 13548) 
s_mul_i32 s53, 3, s52 // (op 13548) 
s_add_u32 s53, s53, 384 // (op 13548) 

v_add_u32 v36, v38, s53 // (op 13548) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13548) Load value

v_accvgpr_read v48, a60 // (op 13554) 
v_accvgpr_read v49, a61 // (op 13554) 
v_accvgpr_read v50, a62 // (op 13554) 
v_accvgpr_read v51, a63 // (op 13554) 
v_mul_f32 v52, v35, v48 // (op 13554) 
v_mul_f32 v53, v35, v49 // (op 13554) 
v_mul_f32 v54, v35, v50 // (op 13554) 
v_mul_f32 v55, v35, v51 // (op 13554) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13546) 
v_mul_f32 v49, v34, v45 // (op 13546) 
v_mul_f32 v50, v34, v46 // (op 13546) 
v_mul_f32 v51, v34, v47 // (op 13546) 

v_add_f32 v44, v48, v52 // (op 13547) 
v_add_f32 v45, v49, v53 // (op 13547) 
v_add_f32 v46, v50, v54 // (op 13547) 
v_add_f32 v47, v51, v55 // (op 13547) 




s_lshl_b32 s52, s24, 7 // (op 13533) 
s_mul_i32 s53, 3, s52 // (op 13533) 
s_add_u32 s53, s53, 384 // (op 13533) 

v_add_u32 v36, v43, s53 // (op 13533) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13533) Store value




v_add_u32 v36, 256, v38 // (op 13457) 

s_nop 0
buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13457) Load value

v_accvgpr_read v48, a32 // (op 13445) 
v_accvgpr_read v49, a33 // (op 13445) 
v_accvgpr_read v50, a34 // (op 13445) 
v_accvgpr_read v51, a35 // (op 13445) 
v_mul_f32 v52, v35, v48 // (op 13445) 
v_mul_f32 v53, v35, v49 // (op 13445) 
v_mul_f32 v54, v35, v50 // (op 13445) 
v_mul_f32 v55, v35, v51 // (op 13445) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13456) 
v_mul_f32 v49, v34, v45 // (op 13456) 
v_mul_f32 v50, v34, v46 // (op 13456) 
v_mul_f32 v51, v34, v47 // (op 13456) 

v_add_f32 v44, v48, v52 // (op 13446) 
v_add_f32 v45, v49, v53 // (op 13446) 
v_add_f32 v46, v50, v54 // (op 13446) 
v_add_f32 v47, v51, v55 // (op 13446) 




v_add_u32 v36, 256, v43 // (op 13448) 

buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13448) Store value




s_lshl_b32 s52, s4, 7 // (op 13461) 
s_add_u32 s52, s52, 256 // (op 13461) 
v_add_u32 v36, v38, s52 // (op 13461) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13461) Load value

v_accvgpr_read v48, a36 // (op 13471) 
v_accvgpr_read v49, a37 // (op 13471) 
v_accvgpr_read v50, a38 // (op 13471) 
v_accvgpr_read v51, a39 // (op 13471) 
v_mul_f32 v52, v35, v48 // (op 13471) 
v_mul_f32 v53, v35, v49 // (op 13471) 
v_mul_f32 v54, v35, v50 // (op 13471) 
v_mul_f32 v55, v35, v51 // (op 13471) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13459) 
v_mul_f32 v49, v34, v45 // (op 13459) 
v_mul_f32 v50, v34, v46 // (op 13459) 
v_mul_f32 v51, v34, v47 // (op 13459) 

v_add_f32 v44, v48, v52 // (op 13460) 
v_add_f32 v45, v49, v53 // (op 13460) 
v_add_f32 v46, v50, v54 // (op 13460) 
v_add_f32 v47, v51, v55 // (op 13460) 




s_lshl_b32 s52, s24, 7 // (op 13450) 
s_add_u32 s52, s52, 256 // (op 13450) 
v_add_u32 v36, v43, s52 // (op 13450) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13450) Store value




s_lshl_b32 s52, s4, 8 // (op 13465) 
s_add_u32 s52, s52, 256 // (op 13465) 
v_add_u32 v36, v38, s52 // (op 13465) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13465) Load value

v_accvgpr_read v48, a40 // (op 13473) 
v_accvgpr_read v49, a41 // (op 13473) 
v_accvgpr_read v50, a42 // (op 13473) 
v_accvgpr_read v51, a43 // (op 13473) 
v_mul_f32 v52, v35, v48 // (op 13473) 
v_mul_f32 v53, v35, v49 // (op 13473) 
v_mul_f32 v54, v35, v50 // (op 13473) 
v_mul_f32 v55, v35, v51 // (op 13473) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13463) 
v_mul_f32 v49, v34, v45 // (op 13463) 
v_mul_f32 v50, v34, v46 // (op 13463) 
v_mul_f32 v51, v34, v47 // (op 13463) 

v_add_f32 v44, v48, v52 // (op 13464) 
v_add_f32 v45, v49, v53 // (op 13464) 
v_add_f32 v46, v50, v54 // (op 13464) 
v_add_f32 v47, v51, v55 // (op 13464) 




s_lshl_b32 s52, s24, 8 // (op 13452) 
s_add_u32 s52, s52, 256 // (op 13452) 
v_add_u32 v36, v43, s52 // (op 13452) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13452) Store value




s_lshl_b32 s52, s4, 7 // (op 13469) 
s_mul_i32 s53, 3, s52 // (op 13469) 
s_add_u32 s53, s53, 256 // (op 13469) 

v_add_u32 v36, v38, s53 // (op 13469) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13469) Load value

v_accvgpr_read v48, a44 // (op 13475) 
v_accvgpr_read v49, a45 // (op 13475) 
v_accvgpr_read v50, a46 // (op 13475) 
v_accvgpr_read v51, a47 // (op 13475) 
v_mul_f32 v52, v35, v48 // (op 13475) 
v_mul_f32 v53, v35, v49 // (op 13475) 
v_mul_f32 v54, v35, v50 // (op 13475) 
v_mul_f32 v55, v35, v51 // (op 13475) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13467) 
v_mul_f32 v49, v34, v45 // (op 13467) 
v_mul_f32 v50, v34, v46 // (op 13467) 
v_mul_f32 v51, v34, v47 // (op 13467) 

v_add_f32 v44, v48, v52 // (op 13468) 
v_add_f32 v45, v49, v53 // (op 13468) 
v_add_f32 v46, v50, v54 // (op 13468) 
v_add_f32 v47, v51, v55 // (op 13468) 




s_lshl_b32 s52, s24, 7 // (op 13454) 
s_mul_i32 s53, 3, s52 // (op 13454) 
s_add_u32 s53, s53, 256 // (op 13454) 

v_add_u32 v36, v43, s53 // (op 13454) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13454) Store value




v_add_u32 v36, 128, v38 // (op 13378) 

s_nop 0
buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13378) Load value

v_accvgpr_read v48, a16 // (op 13366) 
v_accvgpr_read v49, a17 // (op 13366) 
v_accvgpr_read v50, a18 // (op 13366) 
v_accvgpr_read v51, a19 // (op 13366) 
v_mul_f32 v52, v35, v48 // (op 13366) 
v_mul_f32 v53, v35, v49 // (op 13366) 
v_mul_f32 v54, v35, v50 // (op 13366) 
v_mul_f32 v55, v35, v51 // (op 13366) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13377) 
v_mul_f32 v49, v34, v45 // (op 13377) 
v_mul_f32 v50, v34, v46 // (op 13377) 
v_mul_f32 v51, v34, v47 // (op 13377) 

v_add_f32 v44, v48, v52 // (op 13367) 
v_add_f32 v45, v49, v53 // (op 13367) 
v_add_f32 v46, v50, v54 // (op 13367) 
v_add_f32 v47, v51, v55 // (op 13367) 




v_add_u32 v36, 128, v43 // (op 13369) 

buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13369) Store value




s_lshl_b32 s52, s4, 7 // (op 13382) 
s_add_u32 s52, s52, 128 // (op 13382) 
v_add_u32 v36, v38, s52 // (op 13382) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13382) Load value

v_accvgpr_read v48, a20 // (op 13392) 
v_accvgpr_read v49, a21 // (op 13392) 
v_accvgpr_read v50, a22 // (op 13392) 
v_accvgpr_read v51, a23 // (op 13392) 
v_mul_f32 v52, v35, v48 // (op 13392) 
v_mul_f32 v53, v35, v49 // (op 13392) 
v_mul_f32 v54, v35, v50 // (op 13392) 
v_mul_f32 v55, v35, v51 // (op 13392) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13380) 
v_mul_f32 v49, v34, v45 // (op 13380) 
v_mul_f32 v50, v34, v46 // (op 13380) 
v_mul_f32 v51, v34, v47 // (op 13380) 

v_add_f32 v44, v48, v52 // (op 13381) 
v_add_f32 v45, v49, v53 // (op 13381) 
v_add_f32 v46, v50, v54 // (op 13381) 
v_add_f32 v47, v51, v55 // (op 13381) 




s_lshl_b32 s52, s24, 7 // (op 13371) 
s_add_u32 s52, s52, 128 // (op 13371) 
v_add_u32 v36, v43, s52 // (op 13371) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13371) Store value




s_lshl_b32 s52, s4, 8 // (op 13386) 
s_add_u32 s52, s52, 128 // (op 13386) 
v_add_u32 v36, v38, s52 // (op 13386) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13386) Load value

v_accvgpr_read v48, a24 // (op 13394) 
v_accvgpr_read v49, a25 // (op 13394) 
v_accvgpr_read v50, a26 // (op 13394) 
v_accvgpr_read v51, a27 // (op 13394) 
v_mul_f32 v52, v35, v48 // (op 13394) 
v_mul_f32 v53, v35, v49 // (op 13394) 
v_mul_f32 v54, v35, v50 // (op 13394) 
v_mul_f32 v55, v35, v51 // (op 13394) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13384) 
v_mul_f32 v49, v34, v45 // (op 13384) 
v_mul_f32 v50, v34, v46 // (op 13384) 
v_mul_f32 v51, v34, v47 // (op 13384) 

v_add_f32 v44, v48, v52 // (op 13385) 
v_add_f32 v45, v49, v53 // (op 13385) 
v_add_f32 v46, v50, v54 // (op 13385) 
v_add_f32 v47, v51, v55 // (op 13385) 




s_lshl_b32 s52, s24, 8 // (op 13373) 
s_add_u32 s52, s52, 128 // (op 13373) 
v_add_u32 v36, v43, s52 // (op 13373) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13373) Store value




s_lshl_b32 s52, s4, 7 // (op 13390) 
s_mul_i32 s53, 3, s52 // (op 13390) 
s_add_u32 s53, s53, 128 // (op 13390) 

v_add_u32 v36, v38, s53 // (op 13390) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13390) Load value

v_accvgpr_read v48, a28 // (op 13396) 
v_accvgpr_read v49, a29 // (op 13396) 
v_accvgpr_read v50, a30 // (op 13396) 
v_accvgpr_read v51, a31 // (op 13396) 
v_mul_f32 v52, v35, v48 // (op 13396) 
v_mul_f32 v53, v35, v49 // (op 13396) 
v_mul_f32 v54, v35, v50 // (op 13396) 
v_mul_f32 v55, v35, v51 // (op 13396) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13388) 
v_mul_f32 v49, v34, v45 // (op 13388) 
v_mul_f32 v50, v34, v46 // (op 13388) 
v_mul_f32 v51, v34, v47 // (op 13388) 

v_add_f32 v44, v48, v52 // (op 13389) 
v_add_f32 v45, v49, v53 // (op 13389) 
v_add_f32 v46, v50, v54 // (op 13389) 
v_add_f32 v47, v51, v55 // (op 13389) 




s_lshl_b32 s52, s24, 7 // (op 13375) 
s_mul_i32 s53, 3, s52 // (op 13375) 
s_add_u32 s53, s53, 128 // (op 13375) 

v_add_u32 v36, v43, s53 // (op 13375) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13375) Store value




v_mov_b32 v36, v38 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 145) Load value

v_accvgpr_read v48, a0 // (op 142) 
v_accvgpr_read v49, a1 // (op 142) 
v_accvgpr_read v50, a2 // (op 142) 
v_accvgpr_read v51, a3 // (op 142) 
v_mul_f32 v52, v35, v48 // (op 142) 
v_mul_f32 v53, v35, v49 // (op 142) 
v_mul_f32 v54, v35, v50 // (op 142) 
v_mul_f32 v55, v35, v51 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 146) 
v_mul_f32 v49, v34, v45 // (op 146) 
v_mul_f32 v50, v34, v46 // (op 146) 
v_mul_f32 v51, v34, v47 // (op 146) 

v_add_f32 v44, v48, v52 // (op 143) 
v_add_f32 v45, v49, v53 // (op 143) 
v_add_f32 v46, v50, v54 // (op 143) 
v_add_f32 v47, v51, v55 // (op 143) 




v_mov_b32 v36, v43 // (op 144) reg expression

buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s52, s4, 7 // (op 13292) 
v_add_u32 v36, v38, s52 // (op 13292) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13292) Load value

v_accvgpr_read v48, a4 // (op 13289) 
v_accvgpr_read v49, a5 // (op 13289) 
v_accvgpr_read v50, a6 // (op 13289) 
v_accvgpr_read v51, a7 // (op 13289) 
v_mul_f32 v52, v35, v48 // (op 13289) 
v_mul_f32 v53, v35, v49 // (op 13289) 
v_mul_f32 v54, v35, v50 // (op 13289) 
v_mul_f32 v55, v35, v51 // (op 13289) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13293) 
v_mul_f32 v49, v34, v45 // (op 13293) 
v_mul_f32 v50, v34, v46 // (op 13293) 
v_mul_f32 v51, v34, v47 // (op 13293) 

v_add_f32 v44, v48, v52 // (op 13290) 
v_add_f32 v45, v49, v53 // (op 13290) 
v_add_f32 v46, v50, v54 // (op 13290) 
v_add_f32 v47, v51, v55 // (op 13290) 




s_lshl_b32 s52, s24, 7 // (op 13291) 
v_add_u32 v36, v43, s52 // (op 13291) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13291) Store value




s_lshl_b32 s52, s4, 8 // (op 13304) 
v_add_u32 v36, v38, s52 // (op 13304) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13304) Load value

v_accvgpr_read v48, a8 // (op 13301) 
v_accvgpr_read v49, a9 // (op 13301) 
v_accvgpr_read v50, a10 // (op 13301) 
v_accvgpr_read v51, a11 // (op 13301) 
v_mul_f32 v52, v35, v48 // (op 13301) 
v_mul_f32 v53, v35, v49 // (op 13301) 
v_mul_f32 v54, v35, v50 // (op 13301) 
v_mul_f32 v55, v35, v51 // (op 13301) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13305) 
v_mul_f32 v49, v34, v45 // (op 13305) 
v_mul_f32 v50, v34, v46 // (op 13305) 
v_mul_f32 v51, v34, v47 // (op 13305) 

v_add_f32 v44, v48, v52 // (op 13302) 
v_add_f32 v45, v49, v53 // (op 13302) 
v_add_f32 v46, v50, v54 // (op 13302) 
v_add_f32 v47, v51, v55 // (op 13302) 




s_lshl_b32 s52, s24, 8 // (op 13303) 
v_add_u32 v36, v43, s52 // (op 13303) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13303) Store value




s_lshl_b32 s52, s4, 7 // (op 13316) 
s_mul_i32 s53, 3, s52 // (op 13316) 

v_add_u32 v36, v38, s53 // (op 13316) 


buffer_load_dwordx4 v[44:47], v36, s[56:59], 0 offen // (op 13316) Load value

v_accvgpr_read v48, a12 // (op 13313) 
v_accvgpr_read v49, a13 // (op 13313) 
v_accvgpr_read v50, a14 // (op 13313) 
v_accvgpr_read v51, a15 // (op 13313) 
v_mul_f32 v52, v35, v48 // (op 13313) 
v_mul_f32 v53, v35, v49 // (op 13313) 
v_mul_f32 v54, v35, v50 // (op 13313) 
v_mul_f32 v55, v35, v51 // (op 13313) 

s_waitcnt vmcnt(0)
v_mul_f32 v48, v34, v44 // (op 13317) 
v_mul_f32 v49, v34, v45 // (op 13317) 
v_mul_f32 v50, v34, v46 // (op 13317) 
v_mul_f32 v51, v34, v47 // (op 13317) 

v_add_f32 v44, v48, v52 // (op 13314) 
v_add_f32 v45, v49, v53 // (op 13314) 
v_add_f32 v46, v50, v54 // (op 13314) 
v_add_f32 v47, v51, v55 // (op 13314) 




s_lshl_b32 s52, s24, 7 // (op 13315) 
s_mul_i32 s53, 3, s52 // (op 13315) 

v_add_u32 v36, v43, s53 // (op 13315) 


buffer_store_dwordx4 v[44:47], v36, s[60:63], 0 offen // (op 13315) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_373:


s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_286:

























s_mov_b32 s0, 0 // (op 427) call()
s_mov_b32 s1, 0 // (op 428) call()
s_mov_b32 s0, 0 // (op 442) call()
s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 7, v1 // (op 28009) 
s_mul_i32 s15, s16, s50 // (op 28009) 
s_add_u32 s15, s15, s0 // (op 28009) 
v_and_b32 v3, 63, v1 // (op 28009) 
s_add_u32 s6, s15, s9 // (op 28009) 

v_lshrrev_b32 v4, 4, v3 // (op 28009) 
s_mul_hi_u32 s15, s6, s13 // (op 28009) 
v_lshlrev_b32 v5, 2, v4 // (op 28009) 

s_sub_u32 s7, s6, s15 // (op 28009) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 28009) 


s_lshr_b32 s8, s7, 1 // (op 28009) 

v_lshrrev_b32 v0, 2, v3 // (op 28009) 

s_add_u32 s7, s8, s15 // (op 28009) 


v_and_b32 v3, 3, v0 // (op 28009) 

s_lshr_b32 s8, s7, s11 // (op 28009) 

v_and_b32 v0, 3, v1 // (op 28009) 
s_and_b32 s7, s11, 2147483648 // (op 28009) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 28009) 


s_cmp_eq_i32 s7, 0 // (op 28009) 
s_mov_b32 s15, scc // (op 28009) 

v_lshrrev_b32 v0, 6, v1 // (op 28009) 
s_cmp_eq_i32 s15, 0 // (op 28009) 
s_cselect_b32 s7, s6, s8 // (op 28009) 



v_and_b32 v3, 1, v0 // (op 28009) 

s_mul_hi_u32 s6, s7, s12 // (op 28009) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 28009) 


s_sub_u32 s8, s7, s6 // (op 28009) 
s_lshr_b32 s15, s8, 1 // (op 28009) 

s_add_u32 s8, s15, s6 // (op 28009) 


s_lshr_b32 s6, s8, s10 // (op 28009) 

s_and_b32 s8, s10, 2147483648 // (op 28009) 
s_cmp_eq_i32 s8, 0 // (op 28009) 
s_mov_b32 s15, scc // (op 28009) 

s_cmp_eq_i32 s15, 0 // (op 28009) 
s_cselect_b32 s8, s7, s6 // (op 28009) 


s_mul_i32 s6, s8, s14 // (op 28009) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 28009) 


s_sub_u32 s8, s7, s6 // (op 28009) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 28009) 


v_mul_lo_u32 v0, v4, s4 // (op 28009) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 28009) 


s_mul_i32 s19, s4, s23 // (op 28011) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 28011) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 28011) least significant
s_mul_hi_u32 s6, s4, s22 // (op 28011) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28011) most significant: sum
s_add_u32 s19, s19, s6 // (op 28011) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 28011) 

s_mov_b32 s56, s20 // (op 28011) reg expression
s_mov_b32 s57, s21 // (op 28011) reg expression
s_mov_b32 s58, s28 // (op 28011) reg expression
s_mov_b32 s59, 131072 // (op 28011) call()

v_lshrrev_b32 v0, 7, v1 // (op 28363) 
s_mul_i32 s15, s16, s50 // (op 28363) 
s_add_u32 s15, s15, s0 // (op 28363) 
v_and_b32 v3, 63, v1 // (op 28363) 
s_add_u32 s6, s15, s9 // (op 28363) 

v_lshrrev_b32 v5, 4, v3 // (op 28363) 
s_mul_hi_u32 s15, s6, s13 // (op 28363) 
v_lshlrev_b32 v22, 2, v5 // (op 28363) 

s_sub_u32 s7, s6, s15 // (op 28363) 
v_lshl_add_u32 v5, v0, 4, v22 // (op 28363) 


s_lshr_b32 s8, s7, 1 // (op 28363) 

v_lshrrev_b32 v0, 2, v3 // (op 28363) 

s_add_u32 s7, s8, s15 // (op 28363) 


v_and_b32 v3, 3, v0 // (op 28363) 

s_lshr_b32 s8, s7, s11 // (op 28363) 

v_and_b32 v0, 3, v1 // (op 28363) 
s_and_b32 s7, s11, 2147483648 // (op 28363) 
v_lshl_add_u32 v22, v3, 2, v0 // (op 28363) 


s_cmp_eq_i32 s7, 0 // (op 28363) 
s_mov_b32 s15, scc // (op 28363) 

v_lshrrev_b32 v0, 6, v1 // (op 28363) 
s_cmp_eq_i32 s15, 0 // (op 28363) 
s_cselect_b32 s7, s6, s8 // (op 28363) 



v_and_b32 v3, 1, v0 // (op 28363) 

s_mul_hi_u32 s6, s7, s12 // (op 28363) 
v_lshl_add_u32 v0, v3, 4, v22 // (op 28363) 


s_sub_u32 s8, s7, s6 // (op 28363) 
s_lshr_b32 s15, s8, 1 // (op 28363) 

s_add_u32 s8, s15, s6 // (op 28363) 


s_lshr_b32 s6, s8, s10 // (op 28363) 

s_and_b32 s8, s10, 2147483648 // (op 28363) 
s_cmp_eq_i32 s8, 0 // (op 28363) 
s_mov_b32 s15, scc // (op 28363) 

s_cmp_eq_i32 s15, 0 // (op 28363) 
s_cselect_b32 s8, s7, s6 // (op 28363) 


s_mul_i32 s6, s8, s14 // (op 28363) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 28363) 


s_sub_u32 s8, s7, s6 // (op 28363) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 28363) 


v_mul_lo_u32 v0, v5, s24 // (op 28363) 

v_add_lshl_u32 v5, v3, v0, 2 // (op 28363) 


s_mul_i32 s19, s24, s23 // (op 28365) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 28365) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 28365) least significant
s_mul_hi_u32 s6, s24, s22 // (op 28365) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 28365) most significant: sum
s_add_u32 s19, s19, s6 // (op 28365) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 28365) 

s_mov_b32 s60, s26 // (op 28365) reg expression
s_mov_b32 s61, s27 // (op 28365) reg expression
s_mov_b32 s62, s28 // (op 28365) reg expression
s_mov_b32 s63, 131072 // (op 28365) call()

v_and_b32 v0, 63, v1 // (op 28442) 
v_and_b32 v3, 48, v1 // (op 28442) 
v_sub_u32 v22, v0, v3 // (op 28442) 

v_lshrrev_b32 v3, 4, v0 // (op 28442) 

v_lshl_add_u32 v0, v3, 5, v22 // (op 28442) 


v_lshrrev_b32 v3, 7, v1 // (op 28442) 
v_lshlrev_b32 v22, 4, v3 // (op 28442) 

v_lshl_add_u32 v3, v0, 7, v22 // (op 28442) 


v_lshlrev_b32 v0, 2, v3 // (op 28442) 

v_lshrrev_b32 v3, 3, v0 // (op 28442) 

v_lshrrev_b32 v0, 6, v1 // (op 28517) 
v_and_b32 v22, 1, v0 // (op 28517) 

v_and_b32 v0, 15, v1 // (op 28517) 
v_lshl_add_u32 v23, v22, 4, v0 // (op 28517) 


v_and_b32 v0, 63, v1 // (op 28517) 
v_lshrrev_b32 v22, 4, v0 // (op 28517) 

v_lshlrev_b32 v0, 5, v22 // (op 28517) 

v_lshl_add_u32 v22, v23, 7, v0 // (op 28517) 


v_lshlrev_b32 v0, 2, v22 // (op 28517) 

v_lshrrev_b32 v22, 3, v0 // (op 28517) 

v_lshrrev_b32 v0, 7, v1 // (op 28593) 
v_and_b32 v23, 15, v1 // (op 28593) 
v_lshl_add_u32 v24, v0, 4, v23 // (op 28593) 


v_and_b32 v0, 63, v1 // (op 28593) 
v_lshrrev_b32 v23, 4, v0 // (op 28593) 

v_lshl_add_u32 v0, v23, 7, v24 // (op 28593) 


v_lshrrev_b32 v23, 6, v1 // (op 28669) 
v_and_b32 v24, 1, v23 // (op 28669) 

v_and_b32 v23, 15, v1 // (op 28669) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 28669) 


v_and_b32 v23, 63, v1 // (op 28669) 
v_lshrrev_b32 v24, 4, v23 // (op 28669) 

v_lshl_add_u32 v23, v25, 2, v24 // (op 28669) 


v_accvgpr_write a0, 0.00000 // (op 180) call()
v_accvgpr_write a1, 0.00000 // (op 180) call()
v_accvgpr_write a2, 0.00000 // (op 180) call()
v_accvgpr_write a3, 0.00000 // (op 180) call()
v_accvgpr_write a4, 0.00000 // (op 760) call()
v_accvgpr_write a5, 0.00000 // (op 760) call()
v_accvgpr_write a6, 0.00000 // (op 760) call()
v_accvgpr_write a7, 0.00000 // (op 760) call()
v_accvgpr_write a8, 0.00000 // (op 1058) call()
v_accvgpr_write a9, 0.00000 // (op 1058) call()
v_accvgpr_write a10, 0.00000 // (op 1058) call()
v_accvgpr_write a11, 0.00000 // (op 1058) call()
v_accvgpr_write a12, 0.00000 // (op 1356) call()
v_accvgpr_write a13, 0.00000 // (op 1356) call()
v_accvgpr_write a14, 0.00000 // (op 1356) call()
v_accvgpr_write a15, 0.00000 // (op 1356) call()
v_accvgpr_write a16, 0.00000 // (op 1790) call()
v_accvgpr_write a17, 0.00000 // (op 1790) call()
v_accvgpr_write a18, 0.00000 // (op 1790) call()
v_accvgpr_write a19, 0.00000 // (op 1790) call()
v_accvgpr_write a20, 0.00000 // (op 2283) call()
v_accvgpr_write a21, 0.00000 // (op 2283) call()
v_accvgpr_write a22, 0.00000 // (op 2283) call()
v_accvgpr_write a23, 0.00000 // (op 2283) call()
v_accvgpr_write a24, 0.00000 // (op 2284) call()
v_accvgpr_write a25, 0.00000 // (op 2284) call()
v_accvgpr_write a26, 0.00000 // (op 2284) call()
v_accvgpr_write a27, 0.00000 // (op 2284) call()
v_accvgpr_write a28, 0.00000 // (op 2285) call()
v_accvgpr_write a29, 0.00000 // (op 2285) call()
v_accvgpr_write a30, 0.00000 // (op 2285) call()
v_accvgpr_write a31, 0.00000 // (op 2285) call()
v_accvgpr_write a32, 0.00000 // (op 3080) call()
v_accvgpr_write a33, 0.00000 // (op 3080) call()
v_accvgpr_write a34, 0.00000 // (op 3080) call()
v_accvgpr_write a35, 0.00000 // (op 3080) call()
v_accvgpr_write a36, 0.00000 // (op 3573) call()
v_accvgpr_write a37, 0.00000 // (op 3573) call()
v_accvgpr_write a38, 0.00000 // (op 3573) call()
v_accvgpr_write a39, 0.00000 // (op 3573) call()
v_accvgpr_write a40, 0.00000 // (op 3574) call()
v_accvgpr_write a41, 0.00000 // (op 3574) call()
v_accvgpr_write a42, 0.00000 // (op 3574) call()
v_accvgpr_write a43, 0.00000 // (op 3574) call()
v_accvgpr_write a44, 0.00000 // (op 3575) call()
v_accvgpr_write a45, 0.00000 // (op 3575) call()
v_accvgpr_write a46, 0.00000 // (op 3575) call()
v_accvgpr_write a47, 0.00000 // (op 3575) call()
v_accvgpr_write a48, 0.00000 // (op 4370) call()
v_accvgpr_write a49, 0.00000 // (op 4370) call()
v_accvgpr_write a50, 0.00000 // (op 4370) call()
v_accvgpr_write a51, 0.00000 // (op 4370) call()
v_accvgpr_write a52, 0.00000 // (op 4863) call()
v_accvgpr_write a53, 0.00000 // (op 4863) call()
v_accvgpr_write a54, 0.00000 // (op 4863) call()
v_accvgpr_write a55, 0.00000 // (op 4863) call()
v_accvgpr_write a56, 0.00000 // (op 4864) call()
v_accvgpr_write a57, 0.00000 // (op 4864) call()
v_accvgpr_write a58, 0.00000 // (op 4864) call()
v_accvgpr_write a59, 0.00000 // (op 4864) call()
v_accvgpr_write a60, 0.00000 // (op 4865) call()
v_accvgpr_write a61, 0.00000 // (op 4865) call()
v_accvgpr_write a62, 0.00000 // (op 4865) call()
v_accvgpr_write a63, 0.00000 // (op 4865) call()
v_and_b32 v24, -4, v1 // (op 27489) 
s_mul_i32 s15, s16, s50 // (op 27489) 
s_add_u32 s15, s15, s0 // (op 27489) 
v_sub_u32 v25, v1, v24 // (op 27489) 

s_add_u32 s6, s15, s9 // (op 27489) 

v_lshlrev_b32 v24, 5, v25 // (op 27489) 

s_mul_hi_u32 s15, s6, s13 // (op 27489) 
v_lshrrev_b32 v25, 2, v1 // (op 27489) 
s_sub_u32 s7, s6, s15 // (op 27489) 
s_lshr_b32 s8, s7, 1 // (op 27489) 

s_add_u32 s7, s8, s15 // (op 27489) 


s_lshr_b32 s8, s7, s11 // (op 27489) 

s_and_b32 s7, s11, 2147483648 // (op 27489) 
s_cmp_eq_i32 s7, 0 // (op 27489) 
s_mov_b32 s15, scc // (op 27489) 

s_cmp_eq_i32 s15, 0 // (op 27489) 
s_cselect_b32 s7, s6, s8 // (op 27489) 


s_mul_hi_u32 s8, s7, s12 // (op 27489) 
s_sub_u32 s15, s7, s8 // (op 27489) 
s_lshr_b32 s18, s15, 1 // (op 27489) 

s_add_u32 s15, s18, s8 // (op 27489) 


s_lshr_b32 s8, s15, s10 // (op 27489) 

s_and_b32 s15, s10, 2147483648 // (op 27489) 
s_cmp_eq_i32 s15, 0 // (op 27489) 
s_mov_b32 s18, scc // (op 27489) 

s_cmp_eq_i32 s18, 0 // (op 27489) 
s_cselect_b32 s15, s7, s8 // (op 27489) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27489) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27489) 
v_mov_b32 v33, 0 // (op 27489) convert
v_mov_b32 v32, v26 // (op 27489) convert

s_mul_i32 s8, s7, s18 // (op 27489) 


s_sub_u32 s7, s6, s8 // (op 27489) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27489) 


v_mul_lo_u32 v31, v24, s31 // (op 27489) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27489) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 27489) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 27489) most significant: sum


v_add_co_u32 v24, s[18:19], v32, v30 // (op 27489) least significant half
v_addc_co_u32 v25, s[18:19], v33, v31, s[18:19] // (op 27489) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27489) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 27489) 

s_mul_i32 s19, s30, s3 // (op 27491) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27491) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27491) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27491) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27491) most significant: sum
s_add_u32 s19, s19, s6 // (op 27491) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27491) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27491) 

s_mov_b32 s68, s32 // (op 27491) reg expression
s_mov_b32 s69, s33 // (op 27491) reg expression
s_mov_b32 s70, s18 // (op 27491) reg expression
s_mov_b32 s71, 131072 // (op 27491) call()

v_and_b32 v24, -4, v1 // (op 27508) 
s_mul_i32 s15, s16, s50 // (op 27508) 
s_add_u32 s15, s15, s0 // (op 27508) 
v_sub_u32 v25, v1, v24 // (op 27508) 

s_add_u32 s6, s15, s9 // (op 27508) 

v_lshlrev_b32 v24, 5, v25 // (op 27508) 

s_mul_hi_u32 s15, s6, s13 // (op 27508) 
v_lshrrev_b32 v25, 2, v1 // (op 27508) 
s_sub_u32 s7, s6, s15 // (op 27508) 
s_lshr_b32 s8, s7, 1 // (op 27508) 

s_add_u32 s7, s8, s15 // (op 27508) 


s_lshr_b32 s8, s7, s11 // (op 27508) 

s_and_b32 s7, s11, 2147483648 // (op 27508) 
s_cmp_eq_i32 s7, 0 // (op 27508) 
s_mov_b32 s15, scc // (op 27508) 

s_cmp_eq_i32 s15, 0 // (op 27508) 
s_cselect_b32 s7, s6, s8 // (op 27508) 


s_mul_hi_u32 s8, s7, s12 // (op 27508) 
s_sub_u32 s15, s7, s8 // (op 27508) 
s_lshr_b32 s18, s15, 1 // (op 27508) 

s_add_u32 s15, s18, s8 // (op 27508) 


s_lshr_b32 s8, s15, s10 // (op 27508) 

s_and_b32 s15, s10, 2147483648 // (op 27508) 
s_cmp_eq_i32 s15, 0 // (op 27508) 
s_mov_b32 s18, scc // (op 27508) 

s_cmp_eq_i32 s18, 0 // (op 27508) 
s_cselect_b32 s15, s7, s8 // (op 27508) 


v_lshl_add_u32 v28, s15, 7, v24 // (op 27508) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27508) 
v_mov_b32 v33, 0 // (op 27508) convert
v_mov_b32 v32, v28 // (op 27508) convert

s_mul_i32 s8, s7, s18 // (op 27508) 


s_sub_u32 s7, s6, s8 // (op 27508) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27508) 


v_mul_lo_u32 v29, v24, s31 // (op 27508) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27508) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 27508) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 27508) most significant: sum


v_add_co_u32 v24, s[18:19], v32, v28 // (op 27508) least significant half
v_addc_co_u32 v25, s[18:19], v33, v29, s[18:19] // (op 27508) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27508) 


v_lshrrev_b64 v[28:29], 3, v[24:25] // (op 27508) 

s_mul_i32 s19, s30, s3 // (op 27510) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27510) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27510) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27510) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27510) most significant: sum
s_add_u32 s19, s19, s6 // (op 27510) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27510) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27510) 

s_mov_b32 s68, s32 // (op 27510) reg expression
s_mov_b32 s69, s33 // (op 27510) reg expression
s_mov_b32 s70, s18 // (op 27510) reg expression
s_mov_b32 s71, 131072 // (op 27510) call()

v_lshrrev_b32 v24, 2, v1 // (op 27635) 
s_mul_i32 s15, s16, s50 // (op 27635) 
s_add_u32 s15, s15, s0 // (op 27635) 
v_and_b32 v25, -4, v1 // (op 27635) 
s_add_u32 s6, s15, s9 // (op 27635) 

v_sub_u32 v30, v1, v25 // (op 27635) 

s_mul_hi_u32 s15, s6, s13 // (op 27635) 
v_lshlrev_b32 v25, 5, v30 // (op 27635) 

s_sub_u32 s7, s6, s15 // (op 27635) 
s_lshr_b32 s8, s7, 1 // (op 27635) 

s_add_u32 s7, s8, s15 // (op 27635) 


s_lshr_b32 s8, s7, s11 // (op 27635) 

s_and_b32 s7, s11, 2147483648 // (op 27635) 
s_cmp_eq_i32 s7, 0 // (op 27635) 
s_mov_b32 s15, scc // (op 27635) 

s_cmp_eq_i32 s15, 0 // (op 27635) 
s_cselect_b32 s7, s6, s8 // (op 27635) 


s_mul_hi_u32 s8, s7, s12 // (op 27635) 
s_sub_u32 s15, s7, s8 // (op 27635) 
s_lshr_b32 s18, s15, 1 // (op 27635) 

s_add_u32 s15, s18, s8 // (op 27635) 


s_lshr_b32 s8, s15, s10 // (op 27635) 

s_and_b32 s15, s10, 2147483648 // (op 27635) 
s_cmp_eq_i32 s15, 0 // (op 27635) 
s_mov_b32 s18, scc // (op 27635) 

s_cmp_eq_i32 s18, 0 // (op 27635) 
s_cselect_b32 s15, s7, s8 // (op 27635) 


s_mul_i32 s8, s15, s14 // (op 27635) 

s_sub_u32 s15, s7, s8 // (op 27635) 

v_lshl_add_u32 v30, s15, 7, v24 // (op 27635) 


v_mul_lo_u32 v33, v30, s35 // (op 27635) most significant: low of low * high
v_mul_hi_u32 v24, v30, s34 // (op 27635) most significant: high of low * low
v_mul_lo_u32 v32, v30, s34 // (op 27635) least significant: low of low * low
v_add_u32 v33, v33, v24 // (op 27635) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27635) 
s_mul_i32 s8, s7, s18 // (op 27635) 


s_sub_u32 s7, s6, s8 // (op 27635) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27635) 


v_mov_b32 v31, 0 // (op 27635) convert
v_mov_b32 v30, v24 // (op 27635) convert

v_add_co_u32 v24, s[18:19], v30, v32 // (op 27635) least significant half
v_addc_co_u32 v25, s[18:19], v31, v33, s[18:19] // (op 27635) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27635) 


v_lshrrev_b64 v[30:31], 3, v[24:25] // (op 27635) 

s_mul_i32 s19, s34, s23 // (op 27637) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 27637) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 27637) least significant
s_mul_hi_u32 s6, s34, s22 // (op 27637) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27637) most significant: sum
s_add_u32 s19, s19, s6 // (op 27637) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27637) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27637) 

s_mov_b32 s72, s36 // (op 27637) reg expression
s_mov_b32 s73, s37 // (op 27637) reg expression
s_mov_b32 s74, s18 // (op 27637) reg expression
s_mov_b32 s75, 131072 // (op 27637) call()

v_lshrrev_b32 v24, 2, v1 // (op 27654) 
s_mul_i32 s15, s16, s50 // (op 27654) 
s_add_u32 s15, s15, s0 // (op 27654) 
v_and_b32 v25, -4, v1 // (op 27654) 
s_add_u32 s6, s15, s9 // (op 27654) 

v_sub_u32 v32, v1, v25 // (op 27654) 

s_mul_hi_u32 s15, s6, s13 // (op 27654) 
v_lshlrev_b32 v25, 5, v32 // (op 27654) 

s_sub_u32 s7, s6, s15 // (op 27654) 
s_lshr_b32 s8, s7, 1 // (op 27654) 

s_add_u32 s7, s8, s15 // (op 27654) 


s_lshr_b32 s8, s7, s11 // (op 27654) 

s_and_b32 s7, s11, 2147483648 // (op 27654) 
s_cmp_eq_i32 s7, 0 // (op 27654) 
s_mov_b32 s15, scc // (op 27654) 

s_cmp_eq_i32 s15, 0 // (op 27654) 
s_cselect_b32 s7, s6, s8 // (op 27654) 


s_mul_hi_u32 s8, s7, s12 // (op 27654) 
s_sub_u32 s15, s7, s8 // (op 27654) 
s_lshr_b32 s18, s15, 1 // (op 27654) 

s_add_u32 s15, s18, s8 // (op 27654) 


s_lshr_b32 s8, s15, s10 // (op 27654) 

s_and_b32 s15, s10, 2147483648 // (op 27654) 
s_cmp_eq_i32 s15, 0 // (op 27654) 
s_mov_b32 s18, scc // (op 27654) 

s_cmp_eq_i32 s18, 0 // (op 27654) 
s_cselect_b32 s15, s7, s8 // (op 27654) 


s_mul_i32 s8, s15, s14 // (op 27654) 

s_sub_u32 s15, s7, s8 // (op 27654) 

v_lshl_add_u32 v32, s15, 7, v24 // (op 27654) 


v_mul_lo_u32 v37, v32, s35 // (op 27654) most significant: low of low * high
v_mul_hi_u32 v24, v32, s34 // (op 27654) most significant: high of low * low
v_mul_lo_u32 v36, v32, s34 // (op 27654) least significant: low of low * low
v_add_u32 v37, v37, v24 // (op 27654) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27654) 
s_mul_i32 s8, s7, s18 // (op 27654) 


s_sub_u32 s7, s6, s8 // (op 27654) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 27654) 


v_mov_b32 v33, 0 // (op 27654) convert
v_mov_b32 v32, v24 // (op 27654) convert

v_add_co_u32 v24, s[18:19], v32, v36 // (op 27654) least significant half
v_addc_co_u32 v25, s[18:19], v33, v37, s[18:19] // (op 27654) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27654) 


v_lshrrev_b64 v[32:33], 3, v[24:25] // (op 27654) 

s_mul_i32 s19, s34, s23 // (op 27656) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 27656) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 27656) least significant
s_mul_hi_u32 s6, s34, s22 // (op 27656) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27656) most significant: sum
s_add_u32 s19, s19, s6 // (op 27656) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27656) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27656) 

s_mov_b32 s72, s36 // (op 27656) reg expression
s_mov_b32 s73, s37 // (op 27656) reg expression
s_mov_b32 s74, s18 // (op 27656) reg expression
s_mov_b32 s75, 131072 // (op 27656) call()

v_and_b32 v24, -128, v1 // (op 27763) 
s_mul_i32 s15, s16, s50 // (op 27763) 
s_add_u32 s15, s15, s0 // (op 27763) 
v_sub_u32 v25, v1, v24 // (op 27763) 

s_add_u32 s6, s15, s9 // (op 27763) 

v_lshrrev_b32 v24, 7, v1 // (op 27763) 
s_mul_hi_u32 s15, s6, s13 // (op 27763) 
v_lshlrev_b32 v36, 1, v24 // (op 27763) 

s_sub_u32 s7, s6, s15 // (op 27763) 
s_lshr_b32 s8, s7, 1 // (op 27763) 

s_add_u32 s7, s8, s15 // (op 27763) 


s_lshr_b32 s8, s7, s11 // (op 27763) 

s_and_b32 s7, s11, 2147483648 // (op 27763) 
s_cmp_eq_i32 s7, 0 // (op 27763) 
s_mov_b32 s15, scc // (op 27763) 

s_cmp_eq_i32 s15, 0 // (op 27763) 
s_cselect_b32 s7, s6, s8 // (op 27763) 


s_mul_hi_u32 s8, s7, s12 // (op 27763) 
s_sub_u32 s15, s7, s8 // (op 27763) 
s_lshr_b32 s18, s15, 1 // (op 27763) 

s_add_u32 s15, s18, s8 // (op 27763) 


s_lshr_b32 s8, s15, s10 // (op 27763) 

s_and_b32 s15, s10, 2147483648 // (op 27763) 
s_cmp_eq_i32 s15, 0 // (op 27763) 
s_mov_b32 s18, scc // (op 27763) 

s_cmp_eq_i32 s18, 0 // (op 27763) 
s_cselect_b32 s15, s7, s8 // (op 27763) 


v_lshl_add_u32 v24, s15, 7, v25 // (op 27763) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27763) 
s_mul_i32 s8, s7, s18 // (op 27763) 


s_sub_u32 s7, s6, s8 // (op 27763) 


v_lshl_add_u32 v25, s7, 2, v36 // (op 27763) 


v_mul_lo_u32 v36, v25, s38 // (op 27763) 

v_add_u32 v25, v24, v36 // (op 27763) 


s_mul_i32 s19, s38, s43 // (op 27765) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27765) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27765) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27765) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27765) most significant: sum
s_add_u32 s19, s19, s6 // (op 27765) most significant: sum


s_mov_b32 s76, s40 // (op 27765) reg expression
s_mov_b32 s77, s41 // (op 27765) reg expression
s_mov_b32 s78, s18 // (op 27765) reg expression
s_mov_b32 s79, 131072 // (op 27765) call()

v_and_b32 v24, -128, v1 // (op 27782) 
s_mul_i32 s15, s16, s50 // (op 27782) 
s_add_u32 s15, s15, s0 // (op 27782) 
v_sub_u32 v36, v1, v24 // (op 27782) 

s_add_u32 s6, s15, s9 // (op 27782) 

v_lshrrev_b32 v24, 7, v1 // (op 27782) 
s_mul_hi_u32 s15, s6, s13 // (op 27782) 
v_lshlrev_b32 v37, 1, v24 // (op 27782) 

s_sub_u32 s7, s6, s15 // (op 27782) 
s_lshr_b32 s8, s7, 1 // (op 27782) 

s_add_u32 s7, s8, s15 // (op 27782) 


s_lshr_b32 s8, s7, s11 // (op 27782) 

s_and_b32 s7, s11, 2147483648 // (op 27782) 
s_cmp_eq_i32 s7, 0 // (op 27782) 
s_mov_b32 s15, scc // (op 27782) 

s_cmp_eq_i32 s15, 0 // (op 27782) 
s_cselect_b32 s7, s6, s8 // (op 27782) 


s_mul_hi_u32 s8, s7, s12 // (op 27782) 
s_sub_u32 s15, s7, s8 // (op 27782) 
s_lshr_b32 s18, s15, 1 // (op 27782) 

s_add_u32 s15, s18, s8 // (op 27782) 


s_lshr_b32 s8, s15, s10 // (op 27782) 

s_and_b32 s15, s10, 2147483648 // (op 27782) 
s_cmp_eq_i32 s15, 0 // (op 27782) 
s_mov_b32 s18, scc // (op 27782) 

s_cmp_eq_i32 s18, 0 // (op 27782) 
s_cselect_b32 s15, s7, s8 // (op 27782) 


v_lshl_add_u32 v24, s15, 7, v36 // (op 27782) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 27782) 
s_mul_i32 s8, s7, s18 // (op 27782) 


s_sub_u32 s7, s6, s8 // (op 27782) 


v_lshl_add_u32 v36, s7, 2, v37 // (op 27782) 


v_mul_lo_u32 v37, v36, s38 // (op 27782) 

v_add_u32 v36, v24, v37 // (op 27782) 


s_mul_i32 s19, s38, s43 // (op 27784) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27784) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27784) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27784) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27784) most significant: sum
s_add_u32 s19, s19, s6 // (op 27784) most significant: sum


s_mov_b32 s76, s40 // (op 27784) reg expression
s_mov_b32 s77, s41 // (op 27784) reg expression
s_mov_b32 s78, s18 // (op 27784) reg expression
s_mov_b32 s79, 131072 // (op 27784) call()

v_lshrrev_b32 v24, 2, v1 // (op 27891) 
s_mul_i32 s15, s16, s50 // (op 27891) 
s_add_u32 s15, s15, s0 // (op 27891) 
v_lshlrev_b32 v37, 1, v24 // (op 27891) 

s_add_u32 s6, s15, s9 // (op 27891) 

v_and_b32 v24, -4, v1 // (op 27891) 
s_mul_hi_u32 s15, s6, s13 // (op 27891) 
v_sub_u32 v38, v1, v24 // (op 27891) 

s_sub_u32 s7, s6, s15 // (op 27891) 
s_lshr_b32 s8, s7, 1 // (op 27891) 

s_add_u32 s7, s8, s15 // (op 27891) 


s_lshr_b32 s8, s7, s11 // (op 27891) 

s_and_b32 s7, s11, 2147483648 // (op 27891) 
s_cmp_eq_i32 s7, 0 // (op 27891) 
s_mov_b32 s15, scc // (op 27891) 

s_cmp_eq_i32 s15, 0 // (op 27891) 
s_cselect_b32 s7, s6, s8 // (op 27891) 


s_mul_hi_u32 s8, s7, s12 // (op 27891) 
s_sub_u32 s15, s7, s8 // (op 27891) 
s_lshr_b32 s18, s15, 1 // (op 27891) 

s_add_u32 s15, s18, s8 // (op 27891) 


s_lshr_b32 s8, s15, s10 // (op 27891) 

s_and_b32 s15, s10, 2147483648 // (op 27891) 
s_cmp_eq_i32 s15, 0 // (op 27891) 
s_mov_b32 s18, scc // (op 27891) 

s_cmp_eq_i32 s18, 0 // (op 27891) 
s_cselect_b32 s15, s7, s8 // (op 27891) 


s_mul_i32 s8, s15, s14 // (op 27891) 

s_sub_u32 s15, s7, s8 // (op 27891) 

v_lshl_add_u32 v24, s15, 7, v37 // (op 27891) 


v_mul_lo_u32 v37, v24, s44 // (op 27891) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 27891) 
s_mul_i32 s8, s7, s18 // (op 27891) 


s_sub_u32 s7, s6, s8 // (op 27891) 


v_lshl_add_u32 v24, s7, 2, v38 // (op 27891) 


v_add_u32 v38, v24, v37 // (op 27891) 


s_mul_i32 s19, s44, s23 // (op 27893) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 27893) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 27893) least significant
s_mul_hi_u32 s6, s44, s22 // (op 27893) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27893) most significant: sum
s_add_u32 s19, s19, s6 // (op 27893) most significant: sum


s_mov_b32 s80, s46 // (op 27893) reg expression
s_mov_b32 s81, s47 // (op 27893) reg expression
s_mov_b32 s82, s18 // (op 27893) reg expression
s_mov_b32 s83, 131072 // (op 27893) call()

v_lshrrev_b32 v24, 2, v1 // (op 27910) 
s_mul_i32 s15, s16, s50 // (op 27910) 
s_add_u32 s15, s15, s0 // (op 27910) 
v_lshlrev_b32 v37, 1, v24 // (op 27910) 

s_add_u32 s6, s15, s9 // (op 27910) 

v_and_b32 v24, -4, v1 // (op 27910) 
s_mul_hi_u32 s15, s6, s13 // (op 27910) 
v_sub_u32 v39, v1, v24 // (op 27910) 

s_sub_u32 s7, s6, s15 // (op 27910) 
s_lshr_b32 s8, s7, 1 // (op 27910) 

s_add_u32 s7, s8, s15 // (op 27910) 


s_lshr_b32 s8, s7, s11 // (op 27910) 

s_and_b32 s7, s11, 2147483648 // (op 27910) 
s_cmp_eq_i32 s7, 0 // (op 27910) 
s_mov_b32 s15, scc // (op 27910) 

s_cmp_eq_i32 s15, 0 // (op 27910) 
s_cselect_b32 s7, s6, s8 // (op 27910) 


s_mul_hi_u32 s8, s7, s12 // (op 27910) 
s_sub_u32 s15, s7, s8 // (op 27910) 
s_lshr_b32 s18, s15, 1 // (op 27910) 

s_add_u32 s15, s18, s8 // (op 27910) 


s_lshr_b32 s8, s15, s10 // (op 27910) 

s_and_b32 s15, s10, 2147483648 // (op 27910) 
s_cmp_eq_i32 s15, 0 // (op 27910) 
s_mov_b32 s18, scc // (op 27910) 

s_cmp_eq_i32 s18, 0 // (op 27910) 
s_cselect_b32 s15, s7, s8 // (op 27910) 


s_mul_i32 s8, s15, s14 // (op 27910) 

s_sub_u32 s15, s7, s8 // (op 27910) 

v_lshl_add_u32 v24, s15, 7, v37 // (op 27910) 


v_mul_lo_u32 v37, v24, s44 // (op 27910) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 27910) 
s_mul_i32 s8, s7, s18 // (op 27910) 


s_sub_u32 s7, s6, s8 // (op 27910) 


v_lshl_add_u32 v24, s7, 2, v39 // (op 27910) 


v_add_u32 v39, v24, v37 // (op 27910) 


s_mul_i32 s19, s44, s23 // (op 27912) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 27912) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 27912) least significant
s_mul_hi_u32 s6, s44, s22 // (op 27912) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27912) most significant: sum
s_add_u32 s19, s19, s6 // (op 27912) most significant: sum


s_mov_b32 s80, s46 // (op 27912) reg expression
s_mov_b32 s81, s47 // (op 27912) reg expression
s_mov_b32 s82, s18 // (op 27912) reg expression
s_mov_b32 s83, 131072 // (op 27912) call()

v_and_b32 v24, -4, v1 // (op 28483) 
v_sub_u32 v37, v1, v24 // (op 28483) 

v_lshlrev_b32 v24, 5, v37 // (op 28483) 

v_lshrrev_b32 v37, 2, v1 // (op 28483) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28483) 


v_lshlrev_b32 v24, 2, v40 // (op 28483) 

v_lshrrev_b32 v37, 3, v24 // (op 28483) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28483) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30108) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27369) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27369) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27369) 


v_mov_b64 v[42:43], v[30:31] // (op 27369) 
s_mov_b32 m0, s15 // (op 27369) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 27369) Load value direct to lds
v_add_co_u32 v42, s[18:19], v42, v40 // (op 27369) least significant half
v_addc_co_u32 v43, s[18:19], 0, v43, s[18:19] // (op 27369) most significant half

s_add_u32 m0, m0, 4096 // (op 27369) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 27369) Load value direct to lds




v_mov_b64 v[40:41], s[44:45] // (op 26445) convert

v_mov_b32 v24, v38 // (op 26445) 
buffer_load_ubyte v42, v24, s[80:83], 0 offen offset: 0 // (op 26445) Load value
v_add_u32 v24, v24, v40 // (op 26445) 
buffer_load_ubyte v43, v24, s[80:83], 0 offen offset: 0 // (op 26445) Load value


v_and_b32 v24, -4, v1 // (op 28408) 
v_sub_u32 v37, v1, v24 // (op 28408) 

v_lshlrev_b32 v24, 5, v37 // (op 28408) 

v_lshrrev_b32 v37, 2, v1 // (op 28408) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28408) 


v_lshlrev_b32 v24, 2, v40 // (op 28408) 

v_lshrrev_b32 v37, 3, v24 // (op 28408) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28408) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30089) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27374) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27374) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27374) 


v_mov_b64 v[44:45], v[26:27] // (op 27374) reg expression
s_add_u32 s6, 8192, s15 // (op 27374) 
s_mov_b32 m0, s6 // (op 27374) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 27374) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v40 // (op 27374) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 27374) most significant half

s_add_u32 m0, m0, 4096 // (op 27374) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 27374) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26451) convert

v_mov_b32 v24, v25 // (op 26451) reg expression
buffer_load_ubyte v44, v24, s[76:79], 0 offen offset: 0 // (op 26451) Load value
v_add_u32 v24, v24, v40 // (op 26451) 
buffer_load_ubyte v45, v24, s[76:79], 0 offen offset: 0 // (op 26451) Load value


v_and_b32 v24, -4, v1 // (op 28793) 
v_sub_u32 v37, v1, v24 // (op 28793) 

v_lshlrev_b32 v24, 5, v37 // (op 28793) 

v_lshrrev_b32 v37, 2, v1 // (op 28793) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28793) 


v_lshlrev_b32 v24, 2, v40 // (op 28793) 

v_lshrrev_b32 v37, 3, v24 // (op 28793) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28793) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30179) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27379) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27379) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27379) 


v_mov_b64 v[46:47], v[30:31] // (op 27379) 
s_add_u32 s6, 16384, s15 // (op 27379) 
s_mov_b32 m0, s6 // (op 27379) reg expression
buffer_load_dwordx4 v46, s[72:75], 64 offen offset: 0 lds // (op 27379) Load value direct to lds
v_add_co_u32 v46, s[18:19], v46, v40 // (op 27379) least significant half
v_addc_co_u32 v47, s[18:19], 0, v47, s[18:19] // (op 27379) most significant half

s_add_u32 m0, m0, 4096 // (op 27379) 
buffer_load_dwordx4 v46, s[72:75], 64 offen offset: 0 lds // (op 27379) Load value direct to lds





v_mov_b64 v[40:41], s[44:45] // (op 26457) convert

v_mov_b32 v24, v38 // (op 26457) 
buffer_load_ubyte v46, v24, s[80:83], 0 offen offset:4 // (op 26457) Load value
v_add_u32 v24, v24, v40 // (op 26457) 
buffer_load_ubyte v47, v24, s[80:83], 0 offen offset:4 // (op 26457) Load value


v_and_b32 v24, -4, v1 // (op 28749) 
v_sub_u32 v37, v1, v24 // (op 28749) 

v_lshlrev_b32 v24, 5, v37 // (op 28749) 

v_lshrrev_b32 v37, 2, v1 // (op 28749) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28749) 


v_lshlrev_b32 v24, 2, v40 // (op 28749) 

v_lshrrev_b32 v37, 3, v24 // (op 28749) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28749) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30169) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27381) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27381) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27381) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27381) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27381) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27381) 

v_mov_b32 v24, s18 // (op 27381) 
v_mov_b32 v37, s19 // (op 27381) 
v_add_co_u32 v48, s[28:29], v26, v24 // (op 27381) least significant half
v_addc_co_u32 v49, s[28:29], v27, v37, s[28:29] // (op 27381) most significant half




s_add_u32 s6, 24576, s15 // (op 27381) 
s_mov_b32 m0, s6 // (op 27381) reg expression
buffer_load_dwordx4 v48, s[68:71], 0 offen offset: 0 lds // (op 27381) Load value direct to lds
v_add_co_u32 v48, s[18:19], v48, v40 // (op 27381) least significant half
v_addc_co_u32 v49, s[18:19], 0, v49, s[18:19] // (op 27381) most significant half

s_add_u32 m0, m0, 4096 // (op 27381) 
buffer_load_dwordx4 v48, s[68:71], 0 offen offset: 0 lds // (op 27381) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26463) convert

s_lshl_b32 s15, s38, 2 // (op 26463) 
v_add_u32 v24, v25, s15 // (op 26463) 

buffer_load_ubyte v48, v24, s[76:79], 0 offen offset: 0 // (op 26463) Load value
v_add_u32 v24, v24, v40 // (op 26463) 
buffer_load_ubyte v49, v24, s[76:79], 0 offen offset: 0 // (op 26463) Load value


v_and_b32 v24, -4, v1 // (op 28939) 
v_sub_u32 v37, v1, v24 // (op 28939) 

v_lshlrev_b32 v24, 5, v37 // (op 28939) 

v_lshrrev_b32 v37, 2, v1 // (op 28939) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28939) 


v_lshlrev_b32 v24, 2, v40 // (op 28939) 

v_lshrrev_b32 v37, 3, v24 // (op 28939) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28939) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30211) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27383) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27383) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27383) 


v_mov_b64 v[50:51], v[30:31] // (op 27383) 
s_add_u32 s6, 32768, s15 // (op 27383) 
s_mov_b32 m0, s6 // (op 27383) reg expression
s_mov_b32 s7, 128 // (op 27383) call()
buffer_load_dwordx4 v50, s[72:75], s7 offen offset: 0 lds // (op 27383) Load value direct to lds
v_add_co_u32 v50, s[18:19], v50, v40 // (op 27383) least significant half
v_addc_co_u32 v51, s[18:19], 0, v51, s[18:19] // (op 27383) most significant half

s_add_u32 m0, m0, 4096 // (op 27383) 
buffer_load_dwordx4 v50, s[72:75], s7 offen offset: 0 lds // (op 27383) Load value direct to lds






v_mov_b64 v[40:41], s[44:45] // (op 26469) convert

v_mov_b32 v24, v38 // (op 26469) 
buffer_load_ubyte v50, v24, s[80:83], 0 offen offset:8 // (op 26469) Load value
v_add_u32 v24, v24, v40 // (op 26469) 
buffer_load_ubyte v51, v24, s[80:83], 0 offen offset:8 // (op 26469) Load value


v_and_b32 v24, -4, v1 // (op 28895) 
v_sub_u32 v37, v1, v24 // (op 28895) 

v_lshlrev_b32 v24, 5, v37 // (op 28895) 

v_lshrrev_b32 v37, 2, v1 // (op 28895) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 28895) 


v_lshlrev_b32 v24, 2, v40 // (op 28895) 

v_lshrrev_b32 v37, 3, v24 // (op 28895) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 28895) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30201) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27385) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27385) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27385) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27385) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27385) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27385) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 27385) 

v_mov_b32 v24, s28 // (op 27385) 
v_mov_b32 v37, s29 // (op 27385) 
v_add_co_u32 v52, s[18:19], v26, v24 // (op 27385) least significant half
v_addc_co_u32 v53, s[18:19], v27, v37, s[18:19] // (op 27385) most significant half




s_add_u32 s6, 40960, s15 // (op 27385) 
s_mov_b32 m0, s6 // (op 27385) reg expression
buffer_load_dwordx4 v52, s[68:71], 0 offen offset: 0 lds // (op 27385) Load value direct to lds
v_add_co_u32 v52, s[18:19], v52, v40 // (op 27385) least significant half
v_addc_co_u32 v53, s[18:19], 0, v53, s[18:19] // (op 27385) most significant half

s_add_u32 m0, m0, 4096 // (op 27385) 
buffer_load_dwordx4 v52, s[68:71], 0 offen offset: 0 lds // (op 27385) Load value direct to lds





v_mov_b64 v[40:41], s[38:39] // (op 26475) convert

s_lshl_b32 s15, s38, 3 // (op 26475) 
v_add_u32 v24, v25, s15 // (op 26475) 

buffer_load_ubyte v52, v24, s[76:79], 0 offen offset: 0 // (op 26475) Load value
v_add_u32 v24, v24, v40 // (op 26475) 
buffer_load_ubyte v53, v24, s[76:79], 0 offen offset: 0 // (op 26475) Load value


v_and_b32 v24, -4, v1 // (op 29085) 
v_sub_u32 v37, v1, v24 // (op 29085) 

v_lshlrev_b32 v24, 5, v37 // (op 29085) 

v_lshrrev_b32 v37, 2, v1 // (op 29085) 
v_lshl_add_u32 v40, v37, 7, v24 // (op 29085) 


v_lshlrev_b32 v24, 2, v40 // (op 29085) 

v_lshrrev_b32 v37, 3, v24 // (op 29085) 

s_nop 0
v_readfirstlane_b32 s15, v37 // (op 29085) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30243) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27387) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27387) 

v_lshrrev_b64 v[40:41], 3, s[28:29] // (op 27387) 


v_mov_b64 v[54:55], v[30:31] // (op 27387) 
s_add_u32 s6, 49152, s15 // (op 27387) 
s_mov_b32 m0, s6 // (op 27387) reg expression
s_mov_b32 s7, 192 // (op 27387) call()
buffer_load_dwordx4 v54, s[72:75], s7 offen offset: 0 lds // (op 27387) Load value direct to lds
v_add_co_u32 v54, s[18:19], v54, v40 // (op 27387) least significant half
v_addc_co_u32 v55, s[18:19], 0, v55, s[18:19] // (op 27387) most significant half

s_add_u32 m0, m0, 4096 // (op 27387) 
buffer_load_dwordx4 v54, s[72:75], s7 offen offset: 0 lds // (op 27387) Load value direct to lds







v_mov_b64 v[30:31], s[44:45] // (op 26481) convert

v_mov_b32 v24, v38 // (op 26481) 
buffer_load_ubyte v40, v24, s[80:83], 0 offen offset:12 // (op 26481) Load value
v_add_u32 v24, v24, v30 // (op 26481) 
buffer_load_ubyte v41, v24, s[80:83], 0 offen offset:12 // (op 26481) Load value



v_and_b32 v24, -4, v1 // (op 29041) 
v_sub_u32 v30, v1, v24 // (op 29041) 

v_lshlrev_b32 v24, 5, v30 // (op 29041) 

v_lshrrev_b32 v30, 2, v1 // (op 29041) 
v_lshl_add_u32 v31, v30, 7, v24 // (op 29041) 


v_lshlrev_b32 v24, 2, v31 // (op 29041) 

v_lshrrev_b32 v30, 3, v24 // (op 29041) 

s_nop 0
v_readfirstlane_b32 s15, v30 // (op 29041) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 30233) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27389) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27389) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27389) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 27389) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27389) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27389) 

s_mul_i32 s29, 3, s19 // (op 27389) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 27389) least significant
s_mul_hi_u32 s6, 3, s18 // (op 27389) most significant: high of low * low
s_add_u32 s29, s29, s6 // (op 27389) most significant: sum


v_mov_b32 v24, s28 // (op 27389) 
v_mov_b32 v37, s29 // (op 27389) 
v_add_co_u32 v54, s[18:19], v26, v24 // (op 27389) least significant half
v_addc_co_u32 v55, s[18:19], v27, v37, s[18:19] // (op 27389) most significant half




s_add_u32 s6, 57344, s15 // (op 27389) 
s_mov_b32 m0, s6 // (op 27389) reg expression
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27389) Load value direct to lds
v_add_co_u32 v54, s[18:19], v54, v30 // (op 27389) least significant half
v_addc_co_u32 v55, s[18:19], 0, v55, s[18:19] // (op 27389) most significant half

s_add_u32 m0, m0, 4096 // (op 27389) 
buffer_load_dwordx4 v54, s[68:71], 0 offen offset: 0 lds // (op 27389) Load value direct to lds






v_mov_b64 v[26:27], s[38:39] // (op 26487) convert

s_lshl_b32 s15, s38, 2 // (op 26487) 
s_mul_i32 s6, 3, s15 // (op 26487) 

v_add_u32 v24, v25, s6 // (op 26487) 

buffer_load_ubyte v30, v24, s[76:79], 0 offen offset: 0 // (op 26487) Load value
v_add_u32 v24, v24, v26 // (op 26487) 
buffer_load_ubyte v31, v24, s[76:79], 0 offen offset: 0 // (op 26487) Load value



v_and_b32 v24, -4, v1 // (op 28635) 
v_sub_u32 v25, v1, v24 // (op 28635) 

v_lshrrev_b32 v24, 2, v1 // (op 28635) 
v_lshl_add_u32 v26, v24, 3, v25 // (op 28635) 


s_barrier  // (op 30149) 



v_add_u32 v27, 65536, v26 // (op 26490) 
s_waitcnt vmcnt(29)
ds_write_b8 v27, v42 // (op 26490) Store local data 

v_add_u32 v27, 65540, v26 // (op 26490) 
s_waitcnt vmcnt(28)
ds_write_b8 v27, v43 // (op 26490) Store local data 


v_and_b32 v24, -128, v1 // (op 28559) 
v_sub_u32 v25, v1, v24 // (op 28559) 

v_lshrrev_b32 v24, 7, v1 // (op 28559) 
v_lshl_add_u32 v26, v24, 8, v25 // (op 28559) 


s_barrier  // (op 30128) 



v_add_u32 v27, 66048, v26 // (op 26492) 
s_waitcnt vmcnt(25)
ds_write_b8 v27, v44 // (op 26492) Store local data 

v_add_u32 v27, 66176, v26 // (op 26492) 
s_waitcnt vmcnt(24)
ds_write_b8 v27, v45 // (op 26492) Store local data 


s_barrier  // (op 30105) 



ds_read_b128 v[24:27], v22 // (op 26535) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30146) 



v_add_u32 v37, 65536, v23 // (op 26575) 
ds_read_u8 v38, v37 // (op 26575) Load local data 




ds_read_b128 v[54:57], v22 offset:2048 // (op 26540) Load local data 



v_add_u32 v37, 65664, v23 // (op 26580) 
ds_read_u8 v58, v37 // (op 26580) Load local data 




ds_read_b128 v[60:63], v22 offset:4096 // (op 26545) Load local data 



v_add_u32 v37, 65792, v23 // (op 26585) 
ds_read_u8 v59, v37 // (op 26585) Load local data 




ds_read_b128 v[64:67], v22 offset:6144 // (op 26550) Load local data 



v_add_u32 v37, 65920, v23 // (op 26590) 
ds_read_u8 v68, v37 // (op 26590) Load local data 

s_barrier  // (op 30086) 



ds_read_b64_tr_b4 v[70:71], v3 offset:8192 // (op 26515) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:9216 // (op 26515) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30125) 


v_add_u32 v37, 66048, v0 // (op 26555) 
ds_read_u8 v69, v37 // (op 26555) Load local data 




ds_read_b64_tr_b4 v[74:75], v3 offset:8208 // (op 26520) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:9232 // (op 26520) Transpose load local data 


v_add_u32 v37, 66080, v0 // (op 26560) 
ds_read_u8 v78, v37 // (op 26560) Load local data 




ds_read_b64_tr_b4 v[80:81], v3 offset:8224 // (op 26525) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:9248 // (op 26525) Transpose load local data 


v_add_u32 v37, 66112, v0 // (op 26565) 
ds_read_u8 v79, v37 // (op 26565) Load local data 




ds_read_b64_tr_b4 v[84:85], v3 offset:8240 // (op 26530) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:9264 // (op 26530) Transpose load local data 


v_add_u32 v37, 66144, v0 // (op 26570) 
ds_read_u8 v88, v37 // (op 26570) Load local data 

v_and_b32 v37, -4, v1 // (op 28380) 
v_sub_u32 v89, v1, v37 // (op 28380) 

v_lshlrev_b32 v37, 5, v89 // (op 28380) 

v_lshrrev_b32 v89, 2, v1 // (op 28380) 
v_lshl_add_u32 v90, v89, 7, v37 // (op 28380) 


v_lshlrev_b32 v37, 2, v90 // (op 28380) 

v_lshrrev_b32 v89, 3, v37 // (op 28380) 

s_nop 0
v_readfirstlane_b32 s15, v89 // (op 28380) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 28455) 
v_sub_u32 v89, v1, v37 // (op 28455) 

v_lshlrev_b32 v37, 5, v89 // (op 28455) 

v_lshrrev_b32 v89, 2, v1 // (op 28455) 
v_lshl_add_u32 v90, v89, 7, v37 // (op 28455) 


v_lshlrev_b32 v37, 2, v90 // (op 28455) 

v_lshrrev_b32 v89, 3, v37 // (op 28455) 

s_nop 0
v_readfirstlane_b32 s6, v89 // (op 28455) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 28530) 
v_sub_u32 v89, v1, v37 // (op 28530) 

v_lshrrev_b32 v37, 7, v1 // (op 28530) 
v_lshl_add_u32 v90, v37, 8, v89 // (op 28530) 


v_and_b32 v37, -4, v1 // (op 28606) 
v_sub_u32 v89, v1, v37 // (op 28606) 

v_lshrrev_b32 v37, 2, v1 // (op 28606) 
v_lshl_add_u32 v91, v37, 3, v89 // (op 28606) 


v_and_b32 v37, -4, v1 // (op 28734) 
v_sub_u32 v89, v1, v37 // (op 28734) 

v_lshlrev_b32 v37, 5, v89 // (op 28734) 

v_lshrrev_b32 v89, 2, v1 // (op 28734) 
v_lshl_add_u32 v92, v89, 7, v37 // (op 28734) 


v_lshlrev_b32 v37, 2, v92 // (op 28734) 

v_lshrrev_b32 v89, 3, v37 // (op 28734) 

s_nop 0
v_readfirstlane_b32 s7, v89 // (op 28734) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 28778) 
v_sub_u32 v89, v1, v37 // (op 28778) 

v_lshlrev_b32 v37, 5, v89 // (op 28778) 

v_lshrrev_b32 v89, 2, v1 // (op 28778) 
v_lshl_add_u32 v92, v89, 7, v37 // (op 28778) 


v_lshlrev_b32 v37, 2, v92 // (op 28778) 

v_lshrrev_b32 v89, 3, v37 // (op 28778) 

s_nop 0
v_readfirstlane_b32 s8, v89 // (op 28778) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 28822) 
v_sub_u32 v89, v1, v37 // (op 28822) 

v_lshrrev_b32 v37, 7, v1 // (op 28822) 
v_lshl_add_u32 v92, v37, 8, v89 // (op 28822) 


v_and_b32 v37, -4, v1 // (op 28851) 
v_sub_u32 v89, v1, v37 // (op 28851) 

v_lshrrev_b32 v37, 2, v1 // (op 28851) 
v_lshl_add_u32 v93, v37, 3, v89 // (op 28851) 


v_and_b32 v37, -4, v1 // (op 28880) 
v_sub_u32 v89, v1, v37 // (op 28880) 

v_lshlrev_b32 v37, 5, v89 // (op 28880) 

v_lshrrev_b32 v89, 2, v1 // (op 28880) 
v_lshl_add_u32 v94, v89, 7, v37 // (op 28880) 


v_lshlrev_b32 v37, 2, v94 // (op 28880) 

v_lshrrev_b32 v89, 3, v37 // (op 28880) 

s_nop 0
v_readfirstlane_b32 s18, v89 // (op 28880) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 28924) 
v_sub_u32 v89, v1, v37 // (op 28924) 

v_lshlrev_b32 v37, 5, v89 // (op 28924) 

v_lshrrev_b32 v89, 2, v1 // (op 28924) 
v_lshl_add_u32 v94, v89, 7, v37 // (op 28924) 


v_lshlrev_b32 v37, 2, v94 // (op 28924) 

v_lshrrev_b32 v89, 3, v37 // (op 28924) 

s_nop 0
v_readfirstlane_b32 s19, v89 // (op 28924) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 28968) 
v_sub_u32 v89, v1, v37 // (op 28968) 

v_lshrrev_b32 v37, 7, v1 // (op 28968) 
v_lshl_add_u32 v94, v37, 8, v89 // (op 28968) 


v_and_b32 v37, -4, v1 // (op 28997) 
v_sub_u32 v89, v1, v37 // (op 28997) 

v_lshrrev_b32 v37, 2, v1 // (op 28997) 
v_lshl_add_u32 v95, v37, 3, v89 // (op 28997) 


v_and_b32 v37, -4, v1 // (op 29026) 
v_sub_u32 v89, v1, v37 // (op 29026) 

v_lshlrev_b32 v37, 5, v89 // (op 29026) 

v_lshrrev_b32 v89, 2, v1 // (op 29026) 
v_lshl_add_u32 v96, v89, 7, v37 // (op 29026) 


v_lshlrev_b32 v37, 2, v96 // (op 29026) 

v_lshrrev_b32 v89, 3, v37 // (op 29026) 

s_nop 0
v_readfirstlane_b32 s28, v89 // (op 29026) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 29070) 
v_sub_u32 v89, v1, v37 // (op 29070) 

v_lshlrev_b32 v37, 5, v89 // (op 29070) 

v_lshrrev_b32 v89, 2, v1 // (op 29070) 
v_lshl_add_u32 v96, v89, 7, v37 // (op 29070) 


v_lshlrev_b32 v37, 2, v96 // (op 29070) 

v_lshrrev_b32 v89, 3, v37 // (op 29070) 

s_nop 0
v_readfirstlane_b32 s29, v89 // (op 29070) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -128, v1 // (op 29114) 
v_sub_u32 v89, v1, v37 // (op 29114) 

v_lshrrev_b32 v37, 7, v1 // (op 29114) 
v_lshl_add_u32 v96, v37, 8, v89 // (op 29114) 


v_and_b32 v37, -4, v1 // (op 29143) 
v_sub_u32 v89, v1, v37 // (op 29143) 

v_lshrrev_b32 v37, 2, v1 // (op 29143) 
v_lshl_add_u32 v97, v37, 3, v89 // (op 29143) 


s_mov_b32 s1, 0 // (op 3175) call()
s_ashr_i64 s[48:49], s[2:3], 7 // (op 3174) 
s_mov_b32 s52, s48 // (op 3174) convert

s_and_b32 s48, s52, 4294967292 // (op 3174) 

s_cmp_lt_u32 s1, s48 // (op 3174) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3174 // (op 3174) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3174 if false)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3174:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[24:27], a[0:3], v69, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 194) 
s_barrier  // (op 30111) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[54:57], a[4:7], v69, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 800) 
s_add_u32 s1, s1, 4 // (op 3176) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[60:63], a[8:11], v69, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1098) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27278) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[64:67], a[12:15], v69, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1396) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27278) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27278) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[24:27], a[16:19], v78, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2199) 
v_mov_b64 v[100:101], v[32:33] // (op 27278) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[54:57], a[20:23], v78, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2107) 
v_mov_b64 v[102:103], v[100:101] // (op 27278) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[60:63], a[24:27], v78, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2015) 
s_mov_b32 m0, s6 // (op 27278) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[64:67], a[28:31], v78, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1923) 
s_mov_b32 s48, 256 // (op 27278) call()
buffer_load_dwordx4 v102, s[72:75], s48 offen offset: 0 lds // (op 27278) Load value direct to lds
v_add_co_u32 v102, s[52:53], v102, v98 // (op 27278) least significant half
v_addc_co_u32 v103, s[52:53], 0, v103, s[52:53] // (op 27278) most significant half

s_add_u32 m0, m0, 4096 // (op 27278) 
buffer_load_dwordx4 v102, s[72:75], s48 offen offset: 0 lds // (op 27278) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[24:27], a[32:35], v79, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3489) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[54:57], a[36:39], v79, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3397) 
v_mov_b64 v[98:99], s[44:45] // (op 200) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[60:63], a[40:43], v79, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3305) 
v_mov_b32 v37, v39 // (op 200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[64:67], a[44:47], v79, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3213) 
v_mov_b32 v89, v37 // (op 200) 
buffer_load_ubyte v42, v89, s[80:83], 0 offen offset:16 // (op 200) Load value
v_add_u32 v89, v89, v98 // (op 200) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[24:27], a[48:51], v88, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4779) 
buffer_load_ubyte v43, v89, s[80:83], 0 offen offset:16 // (op 200) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[54:57], a[52:55], v88, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4687) 
s_barrier  // (op 30092) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[60:63], a[56:59], v88, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4595) 
s_lshl_b64 s[48:49], s[30:31], 6 // (op 27273) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[64:67], a[60:63], v88, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4503) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27273) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27273) 


v_mov_b64 v[102:103], v[28:29] // (op 27273) 
s_lshl_b64 s[48:49], s[30:31], 7 // (op 27273) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27273) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27273) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27273) 

v_mov_b32 v89, s52 // (op 27273) 
v_mov_b32 v104, s53 // (op 27273) 
v_add_co_u32 v106, s[48:49], v102, v89 // (op 27273) least significant half
v_addc_co_u32 v107, s[48:49], v103, v104, s[48:49] // (op 27273) most significant half




s_add_u32 s48, 8192, s15 // (op 27273) 
s_nop 0
s_mov_b32 m0, s48 // (op 27273) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27273) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v98 // (op 27273) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27273) most significant half

s_add_u32 m0, m0, 4096 // (op 27273) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27273) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 199) convert

v_mov_b32 v89, v36 // (op 199) 
s_lshl_b32 s48, s38, 4 // (op 199) 
v_add_u32 v104, v89, s48 // (op 199) 

buffer_load_ubyte v44, v104, s[76:79], 0 offen offset: 0 // (op 199) Load value
v_add_u32 v104, v104, v98 // (op 199) 
buffer_load_ubyte v45, v104, s[76:79], 0 offen offset: 0 // (op 199) Load value


s_barrier  // (op 30195) 



v_add_u32 v98, 66560, v93 // (op 533) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v46 // (op 533) Store local data 

v_add_u32 v98, 66564, v93 // (op 533) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v47 // (op 533) Store local data 

s_barrier  // (op 30189) 



v_add_u32 v98, 67072, v92 // (op 532) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v48 // (op 532) Store local data 

v_add_u32 v98, 67200, v92 // (op 532) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v49 // (op 532) Store local data 

s_barrier  // (op 30175) 



ds_read_b128 v[104:107], v12 offset:16384 // (op 525) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30192) 



v_add_u32 v98, 66560, v13 // (op 529) 
ds_read_u8 v99, v98 // (op 529) Load local data 




ds_read_b128 v[108:111], v12 offset:18432 // (op 807) Load local data 



v_add_u32 v98, 66688, v13 // (op 811) 
ds_read_u8 v78, v98 // (op 811) Load local data 




ds_read_b128 v[74:77], v12 offset:20480 // (op 1105) Load local data 



v_add_u32 v98, 66816, v13 // (op 1109) 
ds_read_u8 v79, v98 // (op 1109) Load local data 




ds_read_b128 v[80:83], v12 offset:22528 // (op 1403) Load local data 



v_add_u32 v98, 66944, v13 // (op 1407) 
ds_read_u8 v112, v98 // (op 1407) Load local data 

s_barrier  // (op 30166) 
s_barrier  // (op 30446) 



ds_read_b64_tr_b4 v[70:71], v11 offset:24576 // (op 523) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v11 offset:25600 // (op 523) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30185) 


v_add_u32 v98, 67072, v10 // (op 527) 
ds_read_u8 v69, v98 // (op 527) Load local data 




ds_read_b64_tr_b4 v[114:115], v11 offset:24592 // (op 1928) Transpose load local data 
ds_read_b64_tr_b4 v[116:117], v11 offset:25616 // (op 1928) Transpose load local data 


v_add_u32 v98, 67104, v10 // (op 1931) 
ds_read_u8 v58, v98 // (op 1931) Load local data 




ds_read_b64_tr_b4 v[118:119], v11 offset:24608 // (op 3218) Transpose load local data 
ds_read_b64_tr_b4 v[120:121], v11 offset:25632 // (op 3218) Transpose load local data 


v_add_u32 v98, 67136, v10 // (op 3221) 
ds_read_u8 v59, v98 // (op 3221) Load local data 




ds_read_b64_tr_b4 v[122:123], v11 offset:24624 // (op 4508) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v11 offset:25648 // (op 4508) Transpose load local data 


v_add_u32 v98, 67168, v10 // (op 4511) 
ds_read_u8 v68, v98 // (op 4511) Load local data 

s_barrier  // (op 30504) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[104:107], a[0:3], v69, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 530) 
s_barrier  // (op 30182) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[108:111], a[4:7], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 801) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27301) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[74:77], a[8:11], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1099) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27301) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[80:83], a[12:15], v69, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1397) 
v_lshrrev_b64 v[126:127], 3, s[52:53] // (op 27301) 


v_mov_b64 v[128:129], v[100:101] // (op 27301) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:117], v[104:107], a[16:19], v58, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2200) 
s_add_u32 s48, 16384, s8 // (op 27301) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:117], v[108:111], a[20:23], v58, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2108) 
s_mov_b32 m0, s48 // (op 27301) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[74:77], a[24:27], v58, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2016) 
s_mov_b32 s49, 320 // (op 27301) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:117], v[80:83], a[28:31], v58, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1924) 
buffer_load_dwordx4 v128, s[72:75], s49 offen offset: 0 lds // (op 27301) Load value direct to lds

v_add_co_u32 v128, s[52:53], v128, v126 // (op 27301) least significant half
v_addc_co_u32 v129, s[52:53], 0, v129, s[52:53] // (op 27301) most significant half

s_add_u32 m0, m0, 4096 // (op 27301) 
buffer_load_dwordx4 v128, s[72:75], s49 offen offset: 0 lds // (op 27301) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[118:121], v[104:107], a[32:35], v59, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3490) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:121], v[108:111], a[36:39], v59, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3398) 
v_mov_b64 v[116:117], s[44:45] // (op 536) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[118:121], v[74:77], a[40:43], v59, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3306) 
v_mov_b32 v98, v37 // (op 536) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[118:121], v[80:83], a[44:47], v59, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3214) 
buffer_load_ubyte v46, v98, s[80:83], 0 offen offset:20 // (op 536) Load value

v_add_u32 v98, v98, v116 // (op 536) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[122:125], v[104:107], a[48:51], v68, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4780) 
buffer_load_ubyte v47, v98, s[80:83], 0 offen offset:20 // (op 536) Load value




s_barrier  // (op 30172) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[122:125], v[108:111], a[52:55], v68, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4688) 


s_lshl_b64 s[48:49], s[30:31], 6 // (op 27296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[122:125], v[74:77], a[56:59], v68, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4596) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27296) 

v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27296) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[122:125], v[80:83], a[60:63], v68, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4504) 
s_lshl_b64 s[48:49], s[30:31], 7 // (op 27296) 


s_lshl_b64 s[52:53], s[48:49], 2 // (op 27296) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27296) 

s_mul_i32 s53, 5, s49 // (op 27296) most significant: low of low * high
s_mul_i32 s52, 5, s48 // (op 27296) least significant
s_mul_hi_u32 s54, 5, s48 // (op 27296) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27296) most significant: sum


v_mov_b32 v104, s52 // (op 27296) 
v_mov_b32 v105, s53 // (op 27296) 
v_add_co_u32 v106, s[48:49], v102, v104 // (op 27296) least significant half
v_addc_co_u32 v107, s[48:49], v103, v105, s[48:49] // (op 27296) most significant half




s_add_u32 s48, 24576, s7 // (op 27296) 
s_nop 0
s_mov_b32 m0, s48 // (op 27296) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27296) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v98 // (op 27296) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27296) most significant half

s_add_u32 m0, m0, 4096 // (op 27296) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27296) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 535) convert

s_lshl_b32 s48, s38, 2 // (op 535) 
s_mul_i32 s49, 5, s48 // (op 535) 

v_add_u32 v104, v89, s49 // (op 535) 

buffer_load_ubyte v48, v104, s[76:79], 0 offen offset: 0 // (op 535) Load value
v_add_u32 v104, v104, v98 // (op 535) 
buffer_load_ubyte v49, v104, s[76:79], 0 offen offset: 0 // (op 535) Load value


s_barrier  // (op 30227) 



v_add_u32 v98, 66560, v95 // (op 573) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v50 // (op 573) Store local data 

v_add_u32 v98, 66564, v95 // (op 573) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v51 // (op 573) Store local data 

s_barrier  // (op 30221) 



v_add_u32 v98, 67072, v94 // (op 572) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v52 // (op 572) Store local data 

v_add_u32 v98, 67200, v94 // (op 572) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v53 // (op 572) Store local data 

s_barrier  // (op 30207) 



ds_read_b128 v[24:27], v16 offset:32768 // (op 565) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30224) 



v_add_u32 v98, 66560, v17 // (op 569) 
ds_read_u8 v38, v98 // (op 569) Load local data 




ds_read_b128 v[54:57], v16 offset:34816 // (op 815) Load local data 



v_add_u32 v98, 66688, v17 // (op 819) 
ds_read_u8 v58, v98 // (op 819) Load local data 




ds_read_b128 v[60:63], v16 offset:36864 // (op 1113) Load local data 



v_add_u32 v98, 66816, v17 // (op 1117) 
ds_read_u8 v59, v98 // (op 1117) Load local data 




ds_read_b128 v[64:67], v16 offset:38912 // (op 1411) Load local data 



v_add_u32 v98, 66944, v17 // (op 1415) 
ds_read_u8 v88, v98 // (op 1415) Load local data 

s_barrier  // (op 30198) 
s_barrier  // (op 30452) 



ds_read_b64_tr_b4 v[70:71], v15 offset:40960 // (op 563) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v15 offset:41984 // (op 563) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30217) 


v_add_u32 v98, 67072, v14 // (op 567) 
ds_read_u8 v69, v98 // (op 567) Load local data 




ds_read_b64_tr_b4 v[84:85], v15 offset:40976 // (op 1934) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v15 offset:42000 // (op 1934) Transpose load local data 


v_add_u32 v98, 67104, v14 // (op 1937) 
ds_read_u8 v78, v98 // (op 1937) Load local data 




ds_read_b64_tr_b4 v[104:105], v15 offset:40992 // (op 3224) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v15 offset:42016 // (op 3224) Transpose load local data 


v_add_u32 v98, 67136, v14 // (op 3227) 
ds_read_u8 v79, v98 // (op 3227) Load local data 




ds_read_b64_tr_b4 v[108:109], v15 offset:41008 // (op 4514) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v15 offset:42032 // (op 4514) Transpose load local data 


v_add_u32 v98, 67168, v14 // (op 4517) 
ds_read_u8 v99, v98 // (op 4517) Load local data 

s_barrier  // (op 30510) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[24:27], a[0:3], v69, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 570) 
s_barrier  // (op 30214) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[54:57], a[4:7], v69, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 802) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[60:63], a[8:11], v69, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1100) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27311) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[64:67], a[12:15], v69, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1398) 
v_lshrrev_b64 v[112:113], 3, s[52:53] // (op 27311) 


v_mov_b64 v[114:115], v[100:101] // (op 27311) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[24:27], a[16:19], v78, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2201) 
s_add_u32 s48, 32768, s19 // (op 27311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[54:57], a[20:23], v78, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2109) 
s_mov_b32 m0, s48 // (op 27311) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[60:63], a[24:27], v78, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2017) 
s_mov_b32 s49, 384 // (op 27311) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[64:67], a[28:31], v78, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1925) 
buffer_load_dwordx4 v114, s[72:75], s49 offen offset: 0 lds // (op 27311) Load value direct to lds
v_add_co_u32 v114, s[52:53], v114, v112 // (op 27311) least significant half
v_addc_co_u32 v115, s[52:53], 0, v115, s[52:53] // (op 27311) most significant half

s_add_u32 m0, m0, 4096 // (op 27311) 
buffer_load_dwordx4 v114, s[72:75], s49 offen offset: 0 lds // (op 27311) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[24:27], a[32:35], v79, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3491) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[54:57], a[36:39], v79, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3399) 
v_mov_b64 v[112:113], s[44:45] // (op 576) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[60:63], a[40:43], v79, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3307) 
v_mov_b32 v98, v37 // (op 576) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[64:67], a[44:47], v79, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3215) 
buffer_load_ubyte v50, v98, s[80:83], 0 offen offset:24 // (op 576) Load value

v_add_u32 v98, v98, v112 // (op 576) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[24:27], a[48:51], v99, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4781) 
buffer_load_ubyte v51, v98, s[80:83], 0 offen offset:24 // (op 576) Load value


s_barrier  // (op 30204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:111], v[54:57], a[52:55], v99, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4689) 

s_lshl_b64 s[48:49], s[30:31], 6 // (op 27306) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[60:63], a[56:59], v99, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4597) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27306) 

v_lshrrev_b64 v[104:105], 3, s[52:53] // (op 27306) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[64:67], a[60:63], v99, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4505) 
s_lshl_b64 s[48:49], s[30:31], 7 // (op 27306) 


s_lshl_b64 s[52:53], s[48:49], 2 // (op 27306) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27306) 

s_mul_i32 s53, 6, s49 // (op 27306) most significant: low of low * high
s_mul_i32 s52, 6, s48 // (op 27306) least significant
s_mul_hi_u32 s54, 6, s48 // (op 27306) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27306) most significant: sum


v_mov_b32 v98, s52 // (op 27306) 
v_mov_b32 v99, s53 // (op 27306) 
v_add_co_u32 v106, s[48:49], v102, v98 // (op 27306) least significant half
v_addc_co_u32 v107, s[48:49], v103, v99, s[48:49] // (op 27306) most significant half




s_add_u32 s48, 40960, s18 // (op 27306) 
s_nop 0
s_mov_b32 m0, s48 // (op 27306) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27306) Load value direct to lds
v_add_co_u32 v106, s[52:53], v106, v104 // (op 27306) least significant half
v_addc_co_u32 v107, s[52:53], 0, v107, s[52:53] // (op 27306) most significant half

s_add_u32 m0, m0, 4096 // (op 27306) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 27306) Load value direct to lds




v_mov_b64 v[98:99], s[38:39] // (op 575) convert

s_lshl_b32 s48, s38, 2 // (op 575) 
s_mul_i32 s49, 6, s48 // (op 575) 

v_add_u32 v104, v89, s49 // (op 575) 

buffer_load_ubyte v52, v104, s[76:79], 0 offen offset: 0 // (op 575) Load value
v_add_u32 v104, v104, v98 // (op 575) 
buffer_load_ubyte v53, v104, s[76:79], 0 offen offset: 0 // (op 575) Load value


s_barrier  // (op 30259) 



v_add_u32 v98, 66560, v97 // (op 613) 
s_waitcnt vmcnt(29)
ds_write_b8 v98, v40 // (op 613) Store local data 

v_add_u32 v98, 66564, v97 // (op 613) 
s_waitcnt vmcnt(28)
ds_write_b8 v98, v41 // (op 613) Store local data 

s_barrier  // (op 30253) 



v_add_u32 v98, 67072, v96 // (op 612) 
s_waitcnt vmcnt(25)
ds_write_b8 v98, v30 // (op 612) Store local data 

v_add_u32 v98, 67200, v96 // (op 612) 
s_waitcnt vmcnt(24)
ds_write_b8 v98, v31 // (op 612) Store local data 

s_barrier  // (op 30239) 



ds_read_b128 v[74:77], v20 offset:49152 // (op 605) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30256) 



v_add_u32 v98, 66560, v21 // (op 609) 
ds_read_u8 v68, v98 // (op 609) Load local data 




ds_read_b128 v[80:83], v20 offset:51200 // (op 823) Load local data 



v_add_u32 v98, 66688, v21 // (op 827) 
ds_read_u8 v78, v98 // (op 827) Load local data 




ds_read_b128 v[84:87], v20 offset:53248 // (op 1121) Load local data 



v_add_u32 v98, 66816, v21 // (op 1125) 
ds_read_u8 v79, v98 // (op 1125) Load local data 




ds_read_b128 v[104:107], v20 offset:55296 // (op 1419) Load local data 



v_add_u32 v98, 66944, v21 // (op 1423) 
ds_read_u8 v99, v98 // (op 1423) Load local data 

s_barrier  // (op 30230) 
s_barrier  // (op 30458) 



ds_read_b64_tr_b4 v[70:71], v19 offset:57344 // (op 603) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v19 offset:58368 // (op 603) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30249) 


v_add_u32 v98, 67072, v18 // (op 607) 
ds_read_u8 v69, v98 // (op 607) Load local data 




ds_read_b64_tr_b4 v[108:109], v19 offset:57360 // (op 1940) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v19 offset:58384 // (op 1940) Transpose load local data 


v_add_u32 v98, 67104, v18 // (op 1943) 
ds_read_u8 v58, v98 // (op 1943) Load local data 




ds_read_b64_tr_b4 v[112:113], v19 offset:57376 // (op 3230) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v19 offset:58400 // (op 3230) Transpose load local data 


v_add_u32 v98, 67136, v18 // (op 3233) 
ds_read_u8 v59, v98 // (op 3233) Load local data 




ds_read_b64_tr_b4 v[116:117], v19 offset:57392 // (op 4520) Transpose load local data 
ds_read_b64_tr_b4 v[118:119], v19 offset:58416 // (op 4520) Transpose load local data 


v_add_u32 v98, 67168, v18 // (op 4523) 
ds_read_u8 v88, v98 // (op 4523) Load local data 

s_barrier  // (op 30516) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[74:77], a[0:3], v69, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 610) 
s_barrier  // (op 30246) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[80:83], a[4:7], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 803) 
s_lshl_b64 s[48:49], s[34:35], 6 // (op 27321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[84:87], a[8:11], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27321) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[104:107], a[12:15], v69, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1399) 
v_lshrrev_b64 v[120:121], 3, s[52:53] // (op 27321) 


v_mov_b64 v[122:123], v[100:101] // (op 27321) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[74:77], a[16:19], v58, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2202) 
s_add_u32 s48, 49152, s29 // (op 27321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[80:83], a[20:23], v58, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2110) 
s_mov_b32 m0, s48 // (op 27321) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[84:87], a[24:27], v58, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2018) 
s_mov_b32 s49, 448 // (op 27321) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[104:107], a[28:31], v58, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1926) 
buffer_load_dwordx4 v122, s[72:75], s49 offen offset: 0 lds // (op 27321) Load value direct to lds

v_add_co_u32 v122, s[52:53], v122, v120 // (op 27321) least significant half
v_addc_co_u32 v123, s[52:53], 0, v123, s[52:53] // (op 27321) most significant half

s_add_u32 m0, m0, 4096 // (op 27321) 
buffer_load_dwordx4 v122, s[72:75], s49 offen offset: 0 lds // (op 27321) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[74:77], a[32:35], v59, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3492) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[80:83], a[36:39], v59, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3400) 
v_add_co_u32 v32, vcc, 256, v32 // (op 27646) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[84:87], a[40:43], v59, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3308) 
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 27646) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[104:107], a[44:47], v59, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3216) 
v_mov_b64 v[100:101], s[44:45] // (op 616) convert


v_mov_b32 v98, v37 // (op 616) 
buffer_load_ubyte v40, v98, s[80:83], 0 offen offset:28 // (op 616) Load value
v_add_u32 v98, v98, v100 // (op 616) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[74:77], a[48:51], v88, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4782) 
buffer_load_ubyte v41, v98, s[80:83], 0 offen offset:28 // (op 616) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[80:83], a[52:55], v88, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4690) 
s_barrier  // (op 30236) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[84:87], a[56:59], v88, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4598) 
v_add_u32 v39, 16, v39 // (op 27902) 
s_lshl_b64 s[48:49], s[30:31], 6 // (op 27316) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[104:107], a[60:63], v88, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4506) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27316) 




v_lshrrev_b64 v[98:99], 3, s[52:53] // (op 27316) 


s_lshl_b64 s[48:49], s[30:31], 7 // (op 27316) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27316) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27316) 

s_mul_i32 s53, 7, s49 // (op 27316) most significant: low of low * high
s_mul_i32 s52, 7, s48 // (op 27316) least significant
s_mul_hi_u32 s54, 7, s48 // (op 27316) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 27316) most significant: sum


v_mov_b32 v37, s52 // (op 27316) 
v_mov_b32 v100, s53 // (op 27316) 
v_add_co_u32 v104, s[48:49], v102, v37 // (op 27316) least significant half
v_addc_co_u32 v105, s[48:49], v103, v100, s[48:49] // (op 27316) most significant half




s_add_u32 s48, 57344, s28 // (op 27316) 
s_nop 0
s_mov_b32 m0, s48 // (op 27316) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 27316) Load value direct to lds
v_add_co_u32 v104, s[52:53], v104, v98 // (op 27316) least significant half
v_addc_co_u32 v105, s[52:53], 0, v105, s[52:53] // (op 27316) most significant half

s_add_u32 m0, m0, 4096 // (op 27316) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 27316) Load value direct to lds





v_mov_b64 v[98:99], s[38:39] // (op 615) convert

s_lshl_b32 s48, s38, 2 // (op 615) 
s_mul_i32 s49, 7, s48 // (op 615) 

v_add_u32 v37, v89, s49 // (op 615) 

s_lshl_b64 s[48:49], s[30:31], 7 // (op 27500) 
s_lshl_b64 s[52:53], s[48:49], 2 // (op 27500) 

s_lshr_b64 s[48:49], s[52:53], 3 // (op 27500) 

s_lshl_b64 s[52:53], s[48:49], 2 // (op 27500) 

v_mov_b32 v100, s52 // (op 27500) 
v_mov_b32 v101, s53 // (op 27500) 
v_add_co_u32 v28, s[48:49], v28, v100 // (op 27500) least significant half
v_addc_co_u32 v29, s[48:49], v29, v101, s[48:49] // (op 27500) most significant half




buffer_load_ubyte v30, v37, s[76:79], 0 offen offset: 0 // (op 615) Load value
v_add_u32 v37, v37, v98 // (op 615) 
buffer_load_ubyte v31, v37, s[76:79], 0 offen offset: 0 // (op 615) Load value



s_barrier  // (op 30152) 
s_lshl_b32 s48, s38, 4 // (op 27774) 



v_add_u32 v36, v36, s48 // (op 27774) 

v_add_u32 v37, 65536, v91 // (op 197) 
s_waitcnt vmcnt(29)
ds_write_b8 v37, v42 // (op 197) Store local data 

v_add_u32 v37, 65540, v91 // (op 197) 
s_waitcnt vmcnt(28)
ds_write_b8 v37, v43 // (op 197) Store local data 

s_barrier  // (op 30131) 



v_add_u32 v37, 66048, v90 // (op 196) 
s_waitcnt vmcnt(25)
ds_write_b8 v37, v44 // (op 196) Store local data 

v_add_u32 v37, 66176, v90 // (op 196) 
s_waitcnt vmcnt(24)
ds_write_b8 v37, v45 // (op 196) Store local data 

s_barrier  // (op 30117) 



ds_read_b128 v[24:27], v8 // (op 189) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 30159) 



v_add_u32 v37, 65536, v9 // (op 193) 
ds_read_u8 v38, v37 // (op 193) Load local data 




ds_read_b128 v[54:57], v8 offset:2048 // (op 795) Load local data 



v_add_u32 v37, 65664, v9 // (op 799) 
ds_read_u8 v58, v37 // (op 799) Load local data 




ds_read_b128 v[60:63], v8 offset:4096 // (op 1093) Load local data 



v_add_u32 v37, 65792, v9 // (op 1097) 
ds_read_u8 v59, v37 // (op 1097) Load local data 




ds_read_b128 v[64:67], v8 offset:6144 // (op 1391) Load local data 



v_add_u32 v37, 65920, v9 // (op 1395) 
ds_read_u8 v68, v37 // (op 1395) Load local data 

s_barrier  // (op 30098) 



ds_read_b64_tr_b4 v[70:71], v7 offset:8192 // (op 187) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v7 offset:9216 // (op 187) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 30138) 


v_add_u32 v37, 66048, v6 // (op 191) 
ds_read_u8 v69, v37 // (op 191) Load local data 




ds_read_b64_tr_b4 v[74:75], v7 offset:8208 // (op 1918) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:9232 // (op 1918) Transpose load local data 


v_add_u32 v37, 66080, v6 // (op 1921) 
ds_read_u8 v78, v37 // (op 1921) Load local data 




ds_read_b64_tr_b4 v[80:81], v7 offset:8224 // (op 3208) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v7 offset:9248 // (op 3208) Transpose load local data 


v_add_u32 v37, 66112, v6 // (op 3211) 
ds_read_u8 v79, v37 // (op 3211) Load local data 




ds_read_b64_tr_b4 v[84:85], v7 offset:8240 // (op 4498) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:9264 // (op 4498) Transpose load local data 


v_add_u32 v37, 66144, v6 // (op 4501) 
ds_read_u8 v88, v37 // (op 4501) Load local data 

s_ashr_i64 s[48:49], s[2:3], 7 // (op 3174) 
s_mov_b32 s52, s48 // (op 3174) convert

s_and_b32 s48, s52, 4294967292 // (op 3174) 

s_cmp_lt_u32 s1, s48 // (op 3174) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3174 // (op 3174) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3174 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3174:



























s_barrier  // (op 30470) 
v_and_b32 v24, -4, v1 // (op 27424) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27424) 
v_sub_u32 v25, v1, v24 // (op 27424) 

s_and_b32 s15, s18, 4294967292 // (op 27424) 
v_lshlrev_b32 v24, 5, v25 // (op 27424) 

s_add_u32 s6, s0, s15 // (op 27424) 

v_lshrrev_b32 v25, 2, v1 // (op 27424) 
s_mul_i32 s15, s16, s50 // (op 27424) 
s_add_u32 s15, s15, s6 // (op 27424) 

s_add_u32 s6, s15, s9 // (op 27424) 

s_mul_hi_u32 s15, s6, s13 // (op 27424) 
s_sub_u32 s7, s6, s15 // (op 27424) 
s_lshr_b32 s8, s7, 1 // (op 27424) 

s_add_u32 s7, s8, s15 // (op 27424) 


s_lshr_b32 s8, s7, s11 // (op 27424) 

s_and_b32 s7, s11, 2147483648 // (op 27424) 
s_cmp_eq_i32 s7, 0 // (op 27424) 
s_mov_b32 s15, scc // (op 27424) 

s_cmp_eq_i32 s15, 0 // (op 27424) 
s_cselect_b32 s7, s6, s8 // (op 27424) 


s_mul_hi_u32 s8, s7, s12 // (op 27424) 
s_sub_u32 s15, s7, s8 // (op 27424) 
s_lshr_b32 s28, s15, 1 // (op 27424) 

s_add_u32 s15, s28, s8 // (op 27424) 


s_lshr_b32 s8, s15, s10 // (op 27424) 

s_and_b32 s15, s10, 2147483648 // (op 27424) 
s_cmp_eq_i32 s15, 0 // (op 27424) 
s_mov_b32 s28, scc // (op 27424) 

s_cmp_eq_i32 s28, 0 // (op 27424) 
s_cselect_b32 s15, s7, s8 // (op 27424) 


s_mul_i32 s8, s7, s18 // (op 27424) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27424) 


s_sub_u32 s7, s6, s8 // (op 27424) 


v_mov_b32 v29, 0 // (op 27424) convert
v_mov_b32 v28, v26 // (op 27424) convert

v_lshl_add_u32 v24, s7, 7, v25 // (op 27424) 


v_mul_lo_u32 v27, v24, s31 // (op 27424) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 27424) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 27424) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 27424) most significant: sum


v_add_co_u32 v24, s[18:19], v28, v26 // (op 27424) least significant half
v_addc_co_u32 v25, s[18:19], v29, v27, s[18:19] // (op 27424) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27424) 


v_lshrrev_b64 v[32:33], 3, v[24:25] // (op 27424) 

s_mul_i32 s19, s30, s3 // (op 27426) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 27426) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 27426) least significant
s_mul_hi_u32 s6, s30, s2 // (op 27426) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27426) most significant: sum
s_add_u32 s19, s19, s6 // (op 27426) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27426) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27426) 

s_mov_b32 s68, s32 // (op 27426) reg expression
s_mov_b32 s69, s33 // (op 27426) reg expression
s_mov_b32 s70, s18 // (op 27426) reg expression
s_mov_b32 s71, 131072 // (op 27426) call()

v_lshrrev_b32 v24, 2, v1 // (op 27590) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27590) 
v_and_b32 v25, -4, v1 // (op 27590) 
s_and_b32 s15, s18, 4294967292 // (op 27590) 
v_sub_u32 v26, v1, v25 // (op 27590) 

s_add_u32 s6, s0, s15 // (op 27590) 

v_lshlrev_b32 v25, 5, v26 // (op 27590) 

s_mul_i32 s15, s16, s50 // (op 27590) 
s_add_u32 s15, s15, s6 // (op 27590) 

s_add_u32 s6, s15, s9 // (op 27590) 

s_mul_hi_u32 s15, s6, s13 // (op 27590) 
s_sub_u32 s7, s6, s15 // (op 27590) 
s_lshr_b32 s8, s7, 1 // (op 27590) 

s_add_u32 s7, s8, s15 // (op 27590) 


s_lshr_b32 s8, s7, s11 // (op 27590) 

s_and_b32 s7, s11, 2147483648 // (op 27590) 
s_cmp_eq_i32 s7, 0 // (op 27590) 
s_mov_b32 s15, scc // (op 27590) 

s_cmp_eq_i32 s15, 0 // (op 27590) 
s_cselect_b32 s7, s6, s8 // (op 27590) 


s_mul_hi_u32 s8, s7, s12 // (op 27590) 
s_sub_u32 s15, s7, s8 // (op 27590) 
s_lshr_b32 s28, s15, 1 // (op 27590) 

s_add_u32 s15, s28, s8 // (op 27590) 


s_lshr_b32 s8, s15, s10 // (op 27590) 

s_and_b32 s15, s10, 2147483648 // (op 27590) 
s_cmp_eq_i32 s15, 0 // (op 27590) 
s_mov_b32 s28, scc // (op 27590) 

s_cmp_eq_i32 s28, 0 // (op 27590) 
s_cselect_b32 s15, s7, s8 // (op 27590) 


s_mul_i32 s8, s15, s14 // (op 27590) 

s_sub_u32 s15, s7, s8 // (op 27590) 

s_mul_i32 s8, s7, s18 // (op 27590) 


v_lshl_add_u32 v26, s15, 7, v24 // (op 27590) 


s_sub_u32 s7, s6, s8 // (op 27590) 


v_mul_lo_u32 v29, v26, s35 // (op 27590) most significant: low of low * high
v_mul_hi_u32 v24, v26, s34 // (op 27590) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 27590) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 27590) most significant: sum


v_lshl_add_u32 v24, s7, 7, v25 // (op 27590) 


v_mov_b32 v27, 0 // (op 27590) convert
v_mov_b32 v26, v24 // (op 27590) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 27590) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 27590) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 27590) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 27590) 

s_mul_i32 s19, s34, s23 // (op 27592) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 27592) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 27592) least significant
s_mul_hi_u32 s6, s34, s22 // (op 27592) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27592) most significant: sum
s_add_u32 s19, s19, s6 // (op 27592) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 27592) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 27592) 

s_mov_b32 s72, s36 // (op 27592) reg expression
s_mov_b32 s73, s37 // (op 27592) reg expression
s_mov_b32 s74, s18 // (op 27592) reg expression
s_mov_b32 s75, 131072 // (op 27592) call()

v_and_b32 v24, -128, v1 // (op 27718) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27718) 
v_sub_u32 v25, v1, v24 // (op 27718) 

s_and_b32 s15, s18, 4294967292 // (op 27718) 
v_lshrrev_b32 v24, 7, v1 // (op 27718) 
s_add_u32 s6, s0, s15 // (op 27718) 

v_lshlrev_b32 v28, 1, v24 // (op 27718) 

s_mul_i32 s15, s16, s50 // (op 27718) 
s_add_u32 s15, s15, s6 // (op 27718) 

s_add_u32 s6, s15, s9 // (op 27718) 

s_mul_hi_u32 s15, s6, s13 // (op 27718) 
s_sub_u32 s7, s6, s15 // (op 27718) 
s_lshr_b32 s8, s7, 1 // (op 27718) 

s_add_u32 s7, s8, s15 // (op 27718) 


s_lshr_b32 s8, s7, s11 // (op 27718) 

s_and_b32 s7, s11, 2147483648 // (op 27718) 
s_cmp_eq_i32 s7, 0 // (op 27718) 
s_mov_b32 s15, scc // (op 27718) 

s_cmp_eq_i32 s15, 0 // (op 27718) 
s_cselect_b32 s7, s6, s8 // (op 27718) 


s_mul_hi_u32 s8, s7, s12 // (op 27718) 
s_sub_u32 s15, s7, s8 // (op 27718) 
s_lshr_b32 s28, s15, 1 // (op 27718) 

s_add_u32 s15, s28, s8 // (op 27718) 


s_lshr_b32 s8, s15, s10 // (op 27718) 

s_and_b32 s15, s10, 2147483648 // (op 27718) 
s_cmp_eq_i32 s15, 0 // (op 27718) 
s_mov_b32 s28, scc // (op 27718) 

s_cmp_eq_i32 s28, 0 // (op 27718) 
s_cselect_b32 s15, s7, s8 // (op 27718) 


s_mul_i32 s8, s7, s18 // (op 27718) 


v_lshl_add_u32 v24, s15, 7, v25 // (op 27718) 


s_sub_u32 s7, s6, s8 // (op 27718) 


v_lshl_add_u32 v25, s7, 2, v28 // (op 27718) 


v_mul_lo_u32 v28, v25, s38 // (op 27718) 

v_add_u32 v25, v24, v28 // (op 27718) 


s_mul_i32 s19, s38, s43 // (op 27720) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 27720) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 27720) least significant
s_mul_hi_u32 s6, s38, s42 // (op 27720) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27720) most significant: sum
s_add_u32 s19, s19, s6 // (op 27720) most significant: sum


s_mov_b32 s76, s40 // (op 27720) reg expression
s_mov_b32 s77, s41 // (op 27720) reg expression
s_mov_b32 s78, s18 // (op 27720) reg expression
s_mov_b32 s79, 131072 // (op 27720) call()

v_lshrrev_b32 v24, 2, v1 // (op 27846) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 27846) 
v_lshlrev_b32 v28, 1, v24 // (op 27846) 

s_and_b32 s15, s18, 4294967292 // (op 27846) 
v_and_b32 v24, -4, v1 // (op 27846) 
s_add_u32 s6, s0, s15 // (op 27846) 

v_sub_u32 v29, v1, v24 // (op 27846) 

s_mul_i32 s15, s16, s50 // (op 27846) 
s_add_u32 s15, s15, s6 // (op 27846) 

s_add_u32 s6, s15, s9 // (op 27846) 

s_mul_hi_u32 s15, s6, s13 // (op 27846) 
s_sub_u32 s7, s6, s15 // (op 27846) 
s_lshr_b32 s8, s7, 1 // (op 27846) 

s_add_u32 s7, s8, s15 // (op 27846) 


s_lshr_b32 s8, s7, s11 // (op 27846) 

s_and_b32 s7, s11, 2147483648 // (op 27846) 
s_cmp_eq_i32 s7, 0 // (op 27846) 
s_mov_b32 s15, scc // (op 27846) 

s_cmp_eq_i32 s15, 0 // (op 27846) 
s_cselect_b32 s7, s6, s8 // (op 27846) 


s_mul_hi_u32 s8, s7, s12 // (op 27846) 
s_sub_u32 s15, s7, s8 // (op 27846) 
s_lshr_b32 s28, s15, 1 // (op 27846) 

s_add_u32 s15, s28, s8 // (op 27846) 


s_lshr_b32 s8, s15, s10 // (op 27846) 

s_and_b32 s15, s10, 2147483648 // (op 27846) 
s_cmp_eq_i32 s15, 0 // (op 27846) 
s_mov_b32 s28, scc // (op 27846) 

s_cmp_eq_i32 s28, 0 // (op 27846) 
s_cselect_b32 s15, s7, s8 // (op 27846) 


s_mul_i32 s8, s15, s14 // (op 27846) 

s_sub_u32 s15, s7, s8 // (op 27846) 

v_lshl_add_u32 v24, s15, 7, v28 // (op 27846) 


s_mul_i32 s8, s7, s18 // (op 27846) 


v_mul_lo_u32 v28, v24, s44 // (op 27846) 

s_sub_u32 s7, s6, s8 // (op 27846) 


v_lshl_add_u32 v24, s7, 2, v29 // (op 27846) 


v_add_u32 v29, v24, v28 // (op 27846) 


s_mul_i32 s19, s44, s23 // (op 27848) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 27848) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 27848) least significant
s_mul_hi_u32 s6, s44, s22 // (op 27848) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 27848) most significant: sum
s_add_u32 s19, s19, s6 // (op 27848) most significant: sum


s_mov_b32 s80, s46 // (op 27848) reg expression
s_mov_b32 s81, s47 // (op 27848) reg expression
s_mov_b32 s82, s18 // (op 27848) reg expression
s_mov_b32 s83, 131072 // (op 27848) call()

v_and_b32 v24, -4, v1 // (op 28393) 
v_sub_u32 v28, v1, v24 // (op 28393) 

v_lshlrev_b32 v24, 5, v28 // (op 28393) 

v_lshrrev_b32 v28, 2, v1 // (op 28393) 
v_lshl_add_u32 v36, v28, 7, v24 // (op 28393) 


v_lshlrev_b32 v24, 2, v36 // (op 28393) 

v_lshrrev_b32 v28, 3, v24 // (op 28393) 

s_nop 0
v_readfirstlane_b32 s15, v28 // (op 28393) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 28468) 
v_sub_u32 v28, v1, v24 // (op 28468) 

v_lshlrev_b32 v24, 5, v28 // (op 28468) 

v_lshrrev_b32 v28, 2, v1 // (op 28468) 
v_lshl_add_u32 v36, v28, 7, v24 // (op 28468) 


v_lshlrev_b32 v24, 2, v36 // (op 28468) 

v_lshrrev_b32 v28, 3, v24 // (op 28468) 

s_nop 0
v_readfirstlane_b32 s6, v28 // (op 28468) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -128, v1 // (op 28543) 
v_sub_u32 v28, v1, v24 // (op 28543) 

v_lshrrev_b32 v24, 7, v1 // (op 28543) 
v_lshl_add_u32 v36, v24, 8, v28 // (op 28543) 


v_and_b32 v24, -4, v1 // (op 28619) 
v_sub_u32 v28, v1, v24 // (op 28619) 

v_lshrrev_b32 v24, 2, v1 // (op 28619) 
v_lshl_add_u32 v37, v24, 3, v28 // (op 28619) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4470) 
s_and_b32 s7, s18, 4294967292 // (op 4470) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4469) 
s_mov_b32 s8, s18 // (op 4469) convert

s_cmp_lt_u32 s7, s8 // (op 4469) 


















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4469 // (op 4469) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4469 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4469:
s_barrier  // (op 30095) 
s_add_u32 s7, s7, 1 // (op 4471) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 27283) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27283) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27283) 


v_mov_b64 v[38:39], v[32:33] // (op 27283) 
v_mov_b64 v[40:41], v[38:39] // (op 27283) reg expression
s_add_u32 s8, 8192, s15 // (op 27283) 
s_mov_b32 m0, s8 // (op 27283) reg expression
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 27283) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v30 // (op 27283) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 27283) most significant half

s_add_u32 m0, m0, 4096 // (op 27283) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 27283) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 27414) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27414) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 27414) 

v_mov_b32 v24, s18 // (op 27414) 
v_mov_b32 v28, s19 // (op 27414) 
v_add_co_u32 v32, s[28:29], v32, v24 // (op 27414) least significant half
v_addc_co_u32 v33, s[28:29], v33, v28, s[28:29] // (op 27414) most significant half




s_barrier  // (op 30114) 

s_lshl_b64 s[18:19], s[34:35], 6 // (op 27289) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 27289) 

v_lshrrev_b64 v[30:31], 3, s[28:29] // (op 27289) 


v_mov_b64 v[38:39], v[26:27] // (op 27289) 
v_mov_b64 v[40:41], v[38:39] // (op 27289) 
s_mov_b32 m0, s6 // (op 27289) reg expression
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 27289) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v30 // (op 27289) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 27289) most significant half

s_add_u32 m0, m0, 4096 // (op 27289) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 27289) Load value direct to lds



v_add_co_u32 v26, s[18:19], 64, v26 // (op 27582) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 27582) most significant half


v_mov_b64 v[30:31], s[38:39] // (op 469) convert

v_mov_b32 v24, v25 // (op 469) 
v_mov_b32 v28, v24 // (op 469) reg expression
buffer_load_ubyte v38, v28, s[76:79], 0 offen offset: 0 // (op 469) Load value
v_add_u32 v28, v28, v30 // (op 469) 
buffer_load_ubyte v39, v28, s[76:79], 0 offen offset: 0 // (op 469) Load value



s_lshl_b32 s8, s38, 2 // (op 27710) 
v_add_u32 v25, v25, s8 // (op 27710) 


s_barrier  // (op 30134) 
v_mov_b64 v[30:31], s[44:45] // (op 470) convert




v_mov_b32 v24, v29 // (op 470) 
v_mov_b32 v28, v24 // (op 470) 
v_add_u32 v40, 66048, v36 // (op 466) 
s_waitcnt vmcnt(1)
ds_write_b8 v40, v38 // (op 466) Store local data 

v_add_u32 v40, 66176, v36 // (op 466) 
buffer_load_ubyte v42, v28, s[80:83], 0 offen offset: 0 // (op 470) Load value
v_add_u32 v28, v28, v30 // (op 470) 
buffer_load_ubyte v43, v28, s[80:83], 0 offen offset: 0 // (op 470) Load value



v_add_u32 v29, 4, v29 // (op 27838) 
s_waitcnt vmcnt(2)
ds_write_b8 v40, v39 // (op 466) Store local data 


s_barrier  // (op 30101) 



s_barrier  // (op 30155) 
ds_read_b64_tr_b4 v[38:39], v7 offset:8192 // (op 457) Transpose load local data 



v_add_u32 v24, 65536, v37 // (op 467) 
ds_read_b64_tr_b4 v[40:41], v7 offset:9216 // (op 457) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v42 // (op 467) Store local data 

s_barrier  // (op 30121) 
v_add_u32 v24, 65540, v37 // (op 467) 



ds_read_b128 v[44:47], v8 // (op 459) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v43 // (op 467) Store local data 


s_waitcnt lgkmcnt(5)
s_barrier  // (op 30142) 


v_add_u32 v24, 66048, v6 // (op 461) 
ds_read_u8 v28, v24 // (op 461) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 30162) 



v_add_u32 v24, 65536, v9 // (op 463) 
ds_read_u8 v30, v24 // (op 463) Load local data 




ds_read_b128 v[48:51], v8 offset:2048 // (op 774) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[44:47], a[0:3], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 464) 



v_add_u32 v24, 65664, v9 // (op 778) 
ds_read_u8 v31, v24 // (op 778) Load local data 




ds_read_b128 v[52:55], v8 offset:4096 // (op 1072) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:41], v[48:51], a[4:7], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 779) 



v_add_u32 v24, 65792, v9 // (op 1076) 
ds_read_u8 v42, v24 // (op 1076) Load local data 




ds_read_b128 v[56:59], v8 offset:6144 // (op 1370) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[52:55], a[8:11], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1077) 



v_add_u32 v24, 65920, v9 // (op 1374) 
ds_read_u8 v43, v24 // (op 1374) Load local data 




ds_read_b64_tr_b4 v[60:61], v7 offset:8208 // (op 1802) Transpose load local data 
ds_read_b64_tr_b4 v[62:63], v7 offset:9232 // (op 1802) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[56:59], a[12:15], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1375) 




v_add_u32 v24, 66080, v6 // (op 1805) 
ds_read_u8 v28, v24 // (op 1805) Load local data 




ds_read_b64_tr_b4 v[38:39], v7 offset:8224 // (op 3092) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v7 offset:9248 // (op 3092) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[44:47], a[16:19], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1807) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[48:51], a[20:23], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1838) 
v_add_u32 v24, 66112, v6 // (op 3095) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[52:55], a[24:27], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1869) 
ds_read_u8 v64, v24 // (op 3095) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[56:59], a[28:31], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1900) 
ds_read_b64_tr_b4 v[66:67], v7 offset:8240 // (op 4382) Transpose load local data 


ds_read_b64_tr_b4 v[68:69], v7 offset:9264 // (op 4382) Transpose load local data 


v_add_u32 v24, 66144, v6 // (op 4385) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:41], v[44:47], a[32:35], v64, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3097) 
ds_read_u8 v28, v24 // (op 4385) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[38:41], v[48:51], a[36:39], v64, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3128) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[38:41], v[52:55], a[40:43], v64, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3159) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[38:41], v[56:59], a[44:47], v64, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3190) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:69], v[44:47], a[48:51], v28, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4387) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:69], v[48:51], a[52:55], v28, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4418) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:69], v[52:55], a[56:59], v28, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4449) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:69], v[56:59], a[60:63], v28, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4480) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 4469) 
s_mov_b32 s8, s18 // (op 4469) convert

s_cmp_lt_u32 s7, s8 // (op 4469) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4469 // (op 4469) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4469 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4469:









s_barrier  // (op 30488) 




v_cmp_eq_f32 vcc, 0.00000, v34 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_234
v_accvgpr_read v22, a0 // (op 242) 
v_accvgpr_read v23, a1 // (op 242) 
v_accvgpr_read v24, a2 // (op 242) 
v_accvgpr_read v25, a3 // (op 242) 
v_mul_f32 v26, v35, v22 // (op 242) 
v_mul_f32 v27, v35, v23 // (op 242) 
v_mul_f32 v28, v35, v24 // (op 242) 
v_mul_f32 v29, v35, v25 // (op 242) 



v_mov_b32 v0, v5 // (op 244) reg expression

buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 244) Store value


v_accvgpr_read v22, a4 // (op 6151) 
v_accvgpr_read v23, a5 // (op 6151) 
v_accvgpr_read v24, a6 // (op 6151) 
v_accvgpr_read v25, a7 // (op 6151) 
v_mul_f32 v26, v35, v22 // (op 6151) 
v_mul_f32 v27, v35, v23 // (op 6151) 
v_mul_f32 v28, v35, v24 // (op 6151) 
v_mul_f32 v29, v35, v25 // (op 6151) 



s_lshl_b32 s15, s24, 7 // (op 6153) 
v_add_u32 v0, v5, s15 // (op 6153) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6153) Store value


v_accvgpr_read v22, a8 // (op 6163) 
v_accvgpr_read v23, a9 // (op 6163) 
v_accvgpr_read v24, a10 // (op 6163) 
v_accvgpr_read v25, a11 // (op 6163) 
v_mul_f32 v26, v35, v22 // (op 6163) 
v_mul_f32 v27, v35, v23 // (op 6163) 
v_mul_f32 v28, v35, v24 // (op 6163) 
v_mul_f32 v29, v35, v25 // (op 6163) 



s_lshl_b32 s15, s24, 8 // (op 6165) 
v_add_u32 v0, v5, s15 // (op 6165) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6165) Store value


v_accvgpr_read v22, a12 // (op 6175) 
v_accvgpr_read v23, a13 // (op 6175) 
v_accvgpr_read v24, a14 // (op 6175) 
v_accvgpr_read v25, a15 // (op 6175) 
v_mul_f32 v26, v35, v22 // (op 6175) 
v_mul_f32 v27, v35, v23 // (op 6175) 
v_mul_f32 v28, v35, v24 // (op 6175) 
v_mul_f32 v29, v35, v25 // (op 6175) 



s_lshl_b32 s15, s24, 7 // (op 6177) 
s_mul_i32 s6, 3, s15 // (op 6177) 

v_add_u32 v0, v5, s6 // (op 6177) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6177) Store value


v_accvgpr_read v22, a16 // (op 6209) 
v_accvgpr_read v23, a17 // (op 6209) 
v_accvgpr_read v24, a18 // (op 6209) 
v_accvgpr_read v25, a19 // (op 6209) 
v_mul_f32 v26, v35, v22 // (op 6209) 
v_mul_f32 v27, v35, v23 // (op 6209) 
v_mul_f32 v28, v35, v24 // (op 6209) 
v_mul_f32 v29, v35, v25 // (op 6209) 



v_add_u32 v0, 128, v5 // (op 6212) 

buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6212) Store value


v_accvgpr_read v22, a20 // (op 6222) 
v_accvgpr_read v23, a21 // (op 6222) 
v_accvgpr_read v24, a22 // (op 6222) 
v_accvgpr_read v25, a23 // (op 6222) 
v_mul_f32 v26, v35, v22 // (op 6222) 
v_mul_f32 v27, v35, v23 // (op 6222) 
v_mul_f32 v28, v35, v24 // (op 6222) 
v_mul_f32 v29, v35, v25 // (op 6222) 



s_lshl_b32 s15, s24, 7 // (op 6214) 
s_add_u32 s15, s15, 128 // (op 6214) 
v_add_u32 v0, v5, s15 // (op 6214) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6214) Store value


v_accvgpr_read v22, a24 // (op 6227) 
v_accvgpr_read v23, a25 // (op 6227) 
v_accvgpr_read v24, a26 // (op 6227) 
v_accvgpr_read v25, a27 // (op 6227) 
v_mul_f32 v26, v35, v22 // (op 6227) 
v_mul_f32 v27, v35, v23 // (op 6227) 
v_mul_f32 v28, v35, v24 // (op 6227) 
v_mul_f32 v29, v35, v25 // (op 6227) 



s_lshl_b32 s15, s24, 8 // (op 6216) 
s_add_u32 s15, s15, 128 // (op 6216) 
v_add_u32 v0, v5, s15 // (op 6216) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6216) Store value


v_accvgpr_read v22, a28 // (op 6232) 
v_accvgpr_read v23, a29 // (op 6232) 
v_accvgpr_read v24, a30 // (op 6232) 
v_accvgpr_read v25, a31 // (op 6232) 
v_mul_f32 v26, v35, v22 // (op 6232) 
v_mul_f32 v27, v35, v23 // (op 6232) 
v_mul_f32 v28, v35, v24 // (op 6232) 
v_mul_f32 v29, v35, v25 // (op 6232) 



s_lshl_b32 s15, s24, 7 // (op 6218) 
s_mul_i32 s6, 3, s15 // (op 6218) 
s_add_u32 s6, s6, 128 // (op 6218) 

v_add_u32 v0, v5, s6 // (op 6218) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6218) Store value


v_accvgpr_read v22, a32 // (op 6277) 
v_accvgpr_read v23, a33 // (op 6277) 
v_accvgpr_read v24, a34 // (op 6277) 
v_accvgpr_read v25, a35 // (op 6277) 
v_mul_f32 v26, v35, v22 // (op 6277) 
v_mul_f32 v27, v35, v23 // (op 6277) 
v_mul_f32 v28, v35, v24 // (op 6277) 
v_mul_f32 v29, v35, v25 // (op 6277) 



v_add_u32 v0, 256, v5 // (op 6280) 

buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6280) Store value


v_accvgpr_read v22, a36 // (op 6290) 
v_accvgpr_read v23, a37 // (op 6290) 
v_accvgpr_read v24, a38 // (op 6290) 
v_accvgpr_read v25, a39 // (op 6290) 
v_mul_f32 v26, v35, v22 // (op 6290) 
v_mul_f32 v27, v35, v23 // (op 6290) 
v_mul_f32 v28, v35, v24 // (op 6290) 
v_mul_f32 v29, v35, v25 // (op 6290) 



s_lshl_b32 s15, s24, 7 // (op 6282) 
s_add_u32 s15, s15, 256 // (op 6282) 
v_add_u32 v0, v5, s15 // (op 6282) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6282) Store value


v_accvgpr_read v22, a40 // (op 6295) 
v_accvgpr_read v23, a41 // (op 6295) 
v_accvgpr_read v24, a42 // (op 6295) 
v_accvgpr_read v25, a43 // (op 6295) 
v_mul_f32 v26, v35, v22 // (op 6295) 
v_mul_f32 v27, v35, v23 // (op 6295) 
v_mul_f32 v28, v35, v24 // (op 6295) 
v_mul_f32 v29, v35, v25 // (op 6295) 



s_lshl_b32 s15, s24, 8 // (op 6284) 
s_add_u32 s15, s15, 256 // (op 6284) 
v_add_u32 v0, v5, s15 // (op 6284) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6284) Store value


v_accvgpr_read v22, a44 // (op 6300) 
v_accvgpr_read v23, a45 // (op 6300) 
v_accvgpr_read v24, a46 // (op 6300) 
v_accvgpr_read v25, a47 // (op 6300) 
v_mul_f32 v26, v35, v22 // (op 6300) 
v_mul_f32 v27, v35, v23 // (op 6300) 
v_mul_f32 v28, v35, v24 // (op 6300) 
v_mul_f32 v29, v35, v25 // (op 6300) 



s_lshl_b32 s15, s24, 7 // (op 6286) 
s_mul_i32 s6, 3, s15 // (op 6286) 
s_add_u32 s6, s6, 256 // (op 6286) 

v_add_u32 v0, v5, s6 // (op 6286) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6286) Store value


v_accvgpr_read v22, a48 // (op 6345) 
v_accvgpr_read v23, a49 // (op 6345) 
v_accvgpr_read v24, a50 // (op 6345) 
v_accvgpr_read v25, a51 // (op 6345) 
v_mul_f32 v26, v35, v22 // (op 6345) 
v_mul_f32 v27, v35, v23 // (op 6345) 
v_mul_f32 v28, v35, v24 // (op 6345) 
v_mul_f32 v29, v35, v25 // (op 6345) 



v_add_u32 v0, 384, v5 // (op 6348) 

buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6348) Store value


v_accvgpr_read v22, a52 // (op 6358) 
v_accvgpr_read v23, a53 // (op 6358) 
v_accvgpr_read v24, a54 // (op 6358) 
v_accvgpr_read v25, a55 // (op 6358) 
v_mul_f32 v26, v35, v22 // (op 6358) 
v_mul_f32 v27, v35, v23 // (op 6358) 
v_mul_f32 v28, v35, v24 // (op 6358) 
v_mul_f32 v29, v35, v25 // (op 6358) 



s_lshl_b32 s15, s24, 7 // (op 6350) 
s_add_u32 s15, s15, 384 // (op 6350) 
v_add_u32 v0, v5, s15 // (op 6350) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6350) Store value


v_accvgpr_read v22, a56 // (op 6363) 
v_accvgpr_read v23, a57 // (op 6363) 
v_accvgpr_read v24, a58 // (op 6363) 
v_accvgpr_read v25, a59 // (op 6363) 
v_mul_f32 v26, v35, v22 // (op 6363) 
v_mul_f32 v27, v35, v23 // (op 6363) 
v_mul_f32 v28, v35, v24 // (op 6363) 
v_mul_f32 v29, v35, v25 // (op 6363) 



s_lshl_b32 s15, s24, 8 // (op 6352) 
s_add_u32 s15, s15, 384 // (op 6352) 
v_add_u32 v0, v5, s15 // (op 6352) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6352) Store value


v_accvgpr_read v22, a60 // (op 6368) 
v_accvgpr_read v23, a61 // (op 6368) 
v_accvgpr_read v24, a62 // (op 6368) 
v_accvgpr_read v25, a63 // (op 6368) 
v_mul_f32 v26, v35, v22 // (op 6368) 
v_mul_f32 v27, v35, v23 // (op 6368) 
v_mul_f32 v28, v35, v24 // (op 6368) 
v_mul_f32 v29, v35, v25 // (op 6368) 



s_lshl_b32 s15, s24, 7 // (op 6354) 
s_mul_i32 s6, 3, s15 // (op 6354) 
s_add_u32 s6, s6, 384 // (op 6354) 

v_add_u32 v0, v5, s6 // (op 6354) 


buffer_store_dwordx4 v[26:29], v0, s[60:63], 0 offen // (op 6354) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_234
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_234:


v_add_u32 v0, 384, v4 // (op 6712) 

buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6712) Load value

v_accvgpr_read v26, a48 // (op 6700) 
v_accvgpr_read v27, a49 // (op 6700) 
v_accvgpr_read v28, a50 // (op 6700) 
v_accvgpr_read v29, a51 // (op 6700) 
v_mul_f32 v30, v35, v26 // (op 6700) 
v_mul_f32 v31, v35, v27 // (op 6700) 
v_mul_f32 v32, v35, v28 // (op 6700) 
v_mul_f32 v33, v35, v29 // (op 6700) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6711) 
v_mul_f32 v27, v34, v23 // (op 6711) 
v_mul_f32 v28, v34, v24 // (op 6711) 
v_mul_f32 v29, v34, v25 // (op 6711) 

v_add_f32 v22, v26, v30 // (op 6701) 
v_add_f32 v23, v27, v31 // (op 6701) 
v_add_f32 v24, v28, v32 // (op 6701) 
v_add_f32 v25, v29, v33 // (op 6701) 




v_add_u32 v0, 384, v5 // (op 6703) 

buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6703) Store value




s_lshl_b32 s15, s4, 7 // (op 6716) 
s_add_u32 s15, s15, 384 // (op 6716) 
v_add_u32 v0, v4, s15 // (op 6716) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6716) Load value

v_accvgpr_read v26, a52 // (op 6726) 
v_accvgpr_read v27, a53 // (op 6726) 
v_accvgpr_read v28, a54 // (op 6726) 
v_accvgpr_read v29, a55 // (op 6726) 
v_mul_f32 v30, v35, v26 // (op 6726) 
v_mul_f32 v31, v35, v27 // (op 6726) 
v_mul_f32 v32, v35, v28 // (op 6726) 
v_mul_f32 v33, v35, v29 // (op 6726) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6714) 
v_mul_f32 v27, v34, v23 // (op 6714) 
v_mul_f32 v28, v34, v24 // (op 6714) 
v_mul_f32 v29, v34, v25 // (op 6714) 

v_add_f32 v22, v26, v30 // (op 6715) 
v_add_f32 v23, v27, v31 // (op 6715) 
v_add_f32 v24, v28, v32 // (op 6715) 
v_add_f32 v25, v29, v33 // (op 6715) 




s_lshl_b32 s15, s24, 7 // (op 6705) 
s_add_u32 s15, s15, 384 // (op 6705) 
v_add_u32 v0, v5, s15 // (op 6705) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6705) Store value




s_lshl_b32 s15, s4, 8 // (op 6720) 
s_add_u32 s15, s15, 384 // (op 6720) 
v_add_u32 v0, v4, s15 // (op 6720) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6720) Load value

v_accvgpr_read v26, a56 // (op 6728) 
v_accvgpr_read v27, a57 // (op 6728) 
v_accvgpr_read v28, a58 // (op 6728) 
v_accvgpr_read v29, a59 // (op 6728) 
v_mul_f32 v30, v35, v26 // (op 6728) 
v_mul_f32 v31, v35, v27 // (op 6728) 
v_mul_f32 v32, v35, v28 // (op 6728) 
v_mul_f32 v33, v35, v29 // (op 6728) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6718) 
v_mul_f32 v27, v34, v23 // (op 6718) 
v_mul_f32 v28, v34, v24 // (op 6718) 
v_mul_f32 v29, v34, v25 // (op 6718) 

v_add_f32 v22, v26, v30 // (op 6719) 
v_add_f32 v23, v27, v31 // (op 6719) 
v_add_f32 v24, v28, v32 // (op 6719) 
v_add_f32 v25, v29, v33 // (op 6719) 




s_lshl_b32 s15, s24, 8 // (op 6707) 
s_add_u32 s15, s15, 384 // (op 6707) 
v_add_u32 v0, v5, s15 // (op 6707) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6707) Store value




s_lshl_b32 s15, s4, 7 // (op 6724) 
s_mul_i32 s6, 3, s15 // (op 6724) 
s_add_u32 s6, s6, 384 // (op 6724) 

v_add_u32 v0, v4, s6 // (op 6724) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6724) Load value

v_accvgpr_read v26, a60 // (op 6730) 
v_accvgpr_read v27, a61 // (op 6730) 
v_accvgpr_read v28, a62 // (op 6730) 
v_accvgpr_read v29, a63 // (op 6730) 
v_mul_f32 v30, v35, v26 // (op 6730) 
v_mul_f32 v31, v35, v27 // (op 6730) 
v_mul_f32 v32, v35, v28 // (op 6730) 
v_mul_f32 v33, v35, v29 // (op 6730) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6722) 
v_mul_f32 v27, v34, v23 // (op 6722) 
v_mul_f32 v28, v34, v24 // (op 6722) 
v_mul_f32 v29, v34, v25 // (op 6722) 

v_add_f32 v22, v26, v30 // (op 6723) 
v_add_f32 v23, v27, v31 // (op 6723) 
v_add_f32 v24, v28, v32 // (op 6723) 
v_add_f32 v25, v29, v33 // (op 6723) 




s_lshl_b32 s15, s24, 7 // (op 6709) 
s_mul_i32 s6, 3, s15 // (op 6709) 
s_add_u32 s6, s6, 384 // (op 6709) 

v_add_u32 v0, v5, s6 // (op 6709) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6709) Store value




v_add_u32 v0, 256, v4 // (op 6633) 

s_nop 0
buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6633) Load value

v_accvgpr_read v26, a32 // (op 6621) 
v_accvgpr_read v27, a33 // (op 6621) 
v_accvgpr_read v28, a34 // (op 6621) 
v_accvgpr_read v29, a35 // (op 6621) 
v_mul_f32 v30, v35, v26 // (op 6621) 
v_mul_f32 v31, v35, v27 // (op 6621) 
v_mul_f32 v32, v35, v28 // (op 6621) 
v_mul_f32 v33, v35, v29 // (op 6621) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6632) 
v_mul_f32 v27, v34, v23 // (op 6632) 
v_mul_f32 v28, v34, v24 // (op 6632) 
v_mul_f32 v29, v34, v25 // (op 6632) 

v_add_f32 v22, v26, v30 // (op 6622) 
v_add_f32 v23, v27, v31 // (op 6622) 
v_add_f32 v24, v28, v32 // (op 6622) 
v_add_f32 v25, v29, v33 // (op 6622) 




v_add_u32 v0, 256, v5 // (op 6624) 

buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6624) Store value




s_lshl_b32 s15, s4, 7 // (op 6637) 
s_add_u32 s15, s15, 256 // (op 6637) 
v_add_u32 v0, v4, s15 // (op 6637) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6637) Load value

v_accvgpr_read v26, a36 // (op 6647) 
v_accvgpr_read v27, a37 // (op 6647) 
v_accvgpr_read v28, a38 // (op 6647) 
v_accvgpr_read v29, a39 // (op 6647) 
v_mul_f32 v30, v35, v26 // (op 6647) 
v_mul_f32 v31, v35, v27 // (op 6647) 
v_mul_f32 v32, v35, v28 // (op 6647) 
v_mul_f32 v33, v35, v29 // (op 6647) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6635) 
v_mul_f32 v27, v34, v23 // (op 6635) 
v_mul_f32 v28, v34, v24 // (op 6635) 
v_mul_f32 v29, v34, v25 // (op 6635) 

v_add_f32 v22, v26, v30 // (op 6636) 
v_add_f32 v23, v27, v31 // (op 6636) 
v_add_f32 v24, v28, v32 // (op 6636) 
v_add_f32 v25, v29, v33 // (op 6636) 




s_lshl_b32 s15, s24, 7 // (op 6626) 
s_add_u32 s15, s15, 256 // (op 6626) 
v_add_u32 v0, v5, s15 // (op 6626) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6626) Store value




s_lshl_b32 s15, s4, 8 // (op 6641) 
s_add_u32 s15, s15, 256 // (op 6641) 
v_add_u32 v0, v4, s15 // (op 6641) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6641) Load value

v_accvgpr_read v26, a40 // (op 6649) 
v_accvgpr_read v27, a41 // (op 6649) 
v_accvgpr_read v28, a42 // (op 6649) 
v_accvgpr_read v29, a43 // (op 6649) 
v_mul_f32 v30, v35, v26 // (op 6649) 
v_mul_f32 v31, v35, v27 // (op 6649) 
v_mul_f32 v32, v35, v28 // (op 6649) 
v_mul_f32 v33, v35, v29 // (op 6649) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6639) 
v_mul_f32 v27, v34, v23 // (op 6639) 
v_mul_f32 v28, v34, v24 // (op 6639) 
v_mul_f32 v29, v34, v25 // (op 6639) 

v_add_f32 v22, v26, v30 // (op 6640) 
v_add_f32 v23, v27, v31 // (op 6640) 
v_add_f32 v24, v28, v32 // (op 6640) 
v_add_f32 v25, v29, v33 // (op 6640) 




s_lshl_b32 s15, s24, 8 // (op 6628) 
s_add_u32 s15, s15, 256 // (op 6628) 
v_add_u32 v0, v5, s15 // (op 6628) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6628) Store value




s_lshl_b32 s15, s4, 7 // (op 6645) 
s_mul_i32 s6, 3, s15 // (op 6645) 
s_add_u32 s6, s6, 256 // (op 6645) 

v_add_u32 v0, v4, s6 // (op 6645) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6645) Load value

v_accvgpr_read v26, a44 // (op 6651) 
v_accvgpr_read v27, a45 // (op 6651) 
v_accvgpr_read v28, a46 // (op 6651) 
v_accvgpr_read v29, a47 // (op 6651) 
v_mul_f32 v30, v35, v26 // (op 6651) 
v_mul_f32 v31, v35, v27 // (op 6651) 
v_mul_f32 v32, v35, v28 // (op 6651) 
v_mul_f32 v33, v35, v29 // (op 6651) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6643) 
v_mul_f32 v27, v34, v23 // (op 6643) 
v_mul_f32 v28, v34, v24 // (op 6643) 
v_mul_f32 v29, v34, v25 // (op 6643) 

v_add_f32 v22, v26, v30 // (op 6644) 
v_add_f32 v23, v27, v31 // (op 6644) 
v_add_f32 v24, v28, v32 // (op 6644) 
v_add_f32 v25, v29, v33 // (op 6644) 




s_lshl_b32 s15, s24, 7 // (op 6630) 
s_mul_i32 s6, 3, s15 // (op 6630) 
s_add_u32 s6, s6, 256 // (op 6630) 

v_add_u32 v0, v5, s6 // (op 6630) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6630) Store value




v_add_u32 v0, 128, v4 // (op 6554) 

s_nop 0
buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6554) Load value

v_accvgpr_read v26, a16 // (op 6542) 
v_accvgpr_read v27, a17 // (op 6542) 
v_accvgpr_read v28, a18 // (op 6542) 
v_accvgpr_read v29, a19 // (op 6542) 
v_mul_f32 v30, v35, v26 // (op 6542) 
v_mul_f32 v31, v35, v27 // (op 6542) 
v_mul_f32 v32, v35, v28 // (op 6542) 
v_mul_f32 v33, v35, v29 // (op 6542) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6553) 
v_mul_f32 v27, v34, v23 // (op 6553) 
v_mul_f32 v28, v34, v24 // (op 6553) 
v_mul_f32 v29, v34, v25 // (op 6553) 

v_add_f32 v22, v26, v30 // (op 6543) 
v_add_f32 v23, v27, v31 // (op 6543) 
v_add_f32 v24, v28, v32 // (op 6543) 
v_add_f32 v25, v29, v33 // (op 6543) 




v_add_u32 v0, 128, v5 // (op 6545) 

buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6545) Store value




s_lshl_b32 s15, s4, 7 // (op 6558) 
s_add_u32 s15, s15, 128 // (op 6558) 
v_add_u32 v0, v4, s15 // (op 6558) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6558) Load value

v_accvgpr_read v26, a20 // (op 6568) 
v_accvgpr_read v27, a21 // (op 6568) 
v_accvgpr_read v28, a22 // (op 6568) 
v_accvgpr_read v29, a23 // (op 6568) 
v_mul_f32 v30, v35, v26 // (op 6568) 
v_mul_f32 v31, v35, v27 // (op 6568) 
v_mul_f32 v32, v35, v28 // (op 6568) 
v_mul_f32 v33, v35, v29 // (op 6568) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6556) 
v_mul_f32 v27, v34, v23 // (op 6556) 
v_mul_f32 v28, v34, v24 // (op 6556) 
v_mul_f32 v29, v34, v25 // (op 6556) 

v_add_f32 v22, v26, v30 // (op 6557) 
v_add_f32 v23, v27, v31 // (op 6557) 
v_add_f32 v24, v28, v32 // (op 6557) 
v_add_f32 v25, v29, v33 // (op 6557) 




s_lshl_b32 s15, s24, 7 // (op 6547) 
s_add_u32 s15, s15, 128 // (op 6547) 
v_add_u32 v0, v5, s15 // (op 6547) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6547) Store value




s_lshl_b32 s15, s4, 8 // (op 6562) 
s_add_u32 s15, s15, 128 // (op 6562) 
v_add_u32 v0, v4, s15 // (op 6562) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6562) Load value

v_accvgpr_read v26, a24 // (op 6570) 
v_accvgpr_read v27, a25 // (op 6570) 
v_accvgpr_read v28, a26 // (op 6570) 
v_accvgpr_read v29, a27 // (op 6570) 
v_mul_f32 v30, v35, v26 // (op 6570) 
v_mul_f32 v31, v35, v27 // (op 6570) 
v_mul_f32 v32, v35, v28 // (op 6570) 
v_mul_f32 v33, v35, v29 // (op 6570) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6560) 
v_mul_f32 v27, v34, v23 // (op 6560) 
v_mul_f32 v28, v34, v24 // (op 6560) 
v_mul_f32 v29, v34, v25 // (op 6560) 

v_add_f32 v22, v26, v30 // (op 6561) 
v_add_f32 v23, v27, v31 // (op 6561) 
v_add_f32 v24, v28, v32 // (op 6561) 
v_add_f32 v25, v29, v33 // (op 6561) 




s_lshl_b32 s15, s24, 8 // (op 6549) 
s_add_u32 s15, s15, 128 // (op 6549) 
v_add_u32 v0, v5, s15 // (op 6549) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6549) Store value




s_lshl_b32 s15, s4, 7 // (op 6566) 
s_mul_i32 s6, 3, s15 // (op 6566) 
s_add_u32 s6, s6, 128 // (op 6566) 

v_add_u32 v0, v4, s6 // (op 6566) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6566) Load value

v_accvgpr_read v26, a28 // (op 6572) 
v_accvgpr_read v27, a29 // (op 6572) 
v_accvgpr_read v28, a30 // (op 6572) 
v_accvgpr_read v29, a31 // (op 6572) 
v_mul_f32 v30, v35, v26 // (op 6572) 
v_mul_f32 v31, v35, v27 // (op 6572) 
v_mul_f32 v32, v35, v28 // (op 6572) 
v_mul_f32 v33, v35, v29 // (op 6572) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6564) 
v_mul_f32 v27, v34, v23 // (op 6564) 
v_mul_f32 v28, v34, v24 // (op 6564) 
v_mul_f32 v29, v34, v25 // (op 6564) 

v_add_f32 v22, v26, v30 // (op 6565) 
v_add_f32 v23, v27, v31 // (op 6565) 
v_add_f32 v24, v28, v32 // (op 6565) 
v_add_f32 v25, v29, v33 // (op 6565) 




s_lshl_b32 s15, s24, 7 // (op 6551) 
s_mul_i32 s6, 3, s15 // (op 6551) 
s_add_u32 s6, s6, 128 // (op 6551) 

v_add_u32 v0, v5, s6 // (op 6551) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6551) Store value




v_mov_b32 v0, v4 // (op 257) reg expression

s_nop 0
buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 257) Load value

v_accvgpr_read v26, a0 // (op 254) 
v_accvgpr_read v27, a1 // (op 254) 
v_accvgpr_read v28, a2 // (op 254) 
v_accvgpr_read v29, a3 // (op 254) 
v_mul_f32 v30, v35, v26 // (op 254) 
v_mul_f32 v31, v35, v27 // (op 254) 
v_mul_f32 v32, v35, v28 // (op 254) 
v_mul_f32 v33, v35, v29 // (op 254) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 258) 
v_mul_f32 v27, v34, v23 // (op 258) 
v_mul_f32 v28, v34, v24 // (op 258) 
v_mul_f32 v29, v34, v25 // (op 258) 

v_add_f32 v22, v26, v30 // (op 255) 
v_add_f32 v23, v27, v31 // (op 255) 
v_add_f32 v24, v28, v32 // (op 255) 
v_add_f32 v25, v29, v33 // (op 255) 




v_mov_b32 v0, v5 // (op 256) reg expression

buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 256) Store value




s_lshl_b32 s15, s4, 7 // (op 6468) 
v_add_u32 v0, v4, s15 // (op 6468) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6468) Load value

v_accvgpr_read v26, a4 // (op 6465) 
v_accvgpr_read v27, a5 // (op 6465) 
v_accvgpr_read v28, a6 // (op 6465) 
v_accvgpr_read v29, a7 // (op 6465) 
v_mul_f32 v30, v35, v26 // (op 6465) 
v_mul_f32 v31, v35, v27 // (op 6465) 
v_mul_f32 v32, v35, v28 // (op 6465) 
v_mul_f32 v33, v35, v29 // (op 6465) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6469) 
v_mul_f32 v27, v34, v23 // (op 6469) 
v_mul_f32 v28, v34, v24 // (op 6469) 
v_mul_f32 v29, v34, v25 // (op 6469) 

v_add_f32 v22, v26, v30 // (op 6466) 
v_add_f32 v23, v27, v31 // (op 6466) 
v_add_f32 v24, v28, v32 // (op 6466) 
v_add_f32 v25, v29, v33 // (op 6466) 




s_lshl_b32 s15, s24, 7 // (op 6467) 
v_add_u32 v0, v5, s15 // (op 6467) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6467) Store value




s_lshl_b32 s15, s4, 8 // (op 6480) 
v_add_u32 v0, v4, s15 // (op 6480) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6480) Load value

v_accvgpr_read v26, a8 // (op 6477) 
v_accvgpr_read v27, a9 // (op 6477) 
v_accvgpr_read v28, a10 // (op 6477) 
v_accvgpr_read v29, a11 // (op 6477) 
v_mul_f32 v30, v35, v26 // (op 6477) 
v_mul_f32 v31, v35, v27 // (op 6477) 
v_mul_f32 v32, v35, v28 // (op 6477) 
v_mul_f32 v33, v35, v29 // (op 6477) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6481) 
v_mul_f32 v27, v34, v23 // (op 6481) 
v_mul_f32 v28, v34, v24 // (op 6481) 
v_mul_f32 v29, v34, v25 // (op 6481) 

v_add_f32 v22, v26, v30 // (op 6478) 
v_add_f32 v23, v27, v31 // (op 6478) 
v_add_f32 v24, v28, v32 // (op 6478) 
v_add_f32 v25, v29, v33 // (op 6478) 




s_lshl_b32 s15, s24, 8 // (op 6479) 
v_add_u32 v0, v5, s15 // (op 6479) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6479) Store value




s_lshl_b32 s15, s4, 7 // (op 6492) 
s_mul_i32 s6, 3, s15 // (op 6492) 

v_add_u32 v0, v4, s6 // (op 6492) 


buffer_load_dwordx4 v[22:25], v0, s[56:59], 0 offen // (op 6492) Load value

v_accvgpr_read v26, a12 // (op 6489) 
v_accvgpr_read v27, a13 // (op 6489) 
v_accvgpr_read v28, a14 // (op 6489) 
v_accvgpr_read v29, a15 // (op 6489) 
v_mul_f32 v30, v35, v26 // (op 6489) 
v_mul_f32 v31, v35, v27 // (op 6489) 
v_mul_f32 v32, v35, v28 // (op 6489) 
v_mul_f32 v33, v35, v29 // (op 6489) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v34, v22 // (op 6493) 
v_mul_f32 v27, v34, v23 // (op 6493) 
v_mul_f32 v28, v34, v24 // (op 6493) 
v_mul_f32 v29, v34, v25 // (op 6493) 

v_add_f32 v22, v26, v30 // (op 6490) 
v_add_f32 v23, v27, v31 // (op 6490) 
v_add_f32 v24, v28, v32 // (op 6490) 
v_add_f32 v25, v29, v33 // (op 6490) 




s_lshl_b32 s15, s24, 7 // (op 6491) 
s_mul_i32 s6, 3, s15 // (op 6491) 

v_add_u32 v0, v5, s6 // (op 6491) 


buffer_store_dwordx4 v[22:25], v0, s[60:63], 0 offen // (op 6491) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_234:
s_add_u32 s0, s0, s1 // (op 443) 


s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_441:

































































s_endpgm  // End of RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.LRR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end:
.size RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4, .LRR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end-RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
  .amdhsa_accum_offset 148
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
  - .name: RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
    .symbol: RR_GEMM_NN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 93
    .vgpr_count: 146
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_19
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
      - .name: Tensor_2_size_1_12
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
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_14
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_15
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_17
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

