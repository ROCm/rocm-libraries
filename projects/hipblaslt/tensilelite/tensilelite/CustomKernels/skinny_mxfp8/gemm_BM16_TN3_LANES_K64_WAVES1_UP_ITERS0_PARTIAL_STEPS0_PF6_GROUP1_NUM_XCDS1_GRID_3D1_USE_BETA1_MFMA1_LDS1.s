.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1:
s_load_dword s59, s[0:1], 104
s_load_dword s48, s[0:1], 96
s_load_dword s49, s[0:1], 100
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s58, s[0:1], 72
s_load_dword s57, s[0:1], 68
s_load_dword s56, s[0:1], 64
s_load_dword s55, s[0:1], 60
s_load_dword s54, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[66:67], s[0:1], 32
s_load_dwordx2 s[64:65], s[0:1], 24
s_load_dwordx2 s[62:63], s[0:1], 16
s_load_dwordx2 s[60:61], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s72, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
s_mov_b32 s53, 0
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s73, s53, 98304
v_and_b32 v13, 7, v5
v_add_u32 v15, 0, v6
v_lshrrev_b32 v14, 1, v15
v_xor_b32 v7, v14, v13
v_lshlrev_b32 v7, 4, v7
v_add_u32 v15, 8, v6
v_lshrrev_b32 v14, 1, v15
v_xor_b32 v8, v14, v13
v_lshlrev_b32 v8, 4, v8
v_and_b32 v15, 15, v5
v_lshrrev_b32 v16, 3, v15
v_lshlrev_b32 v16, 10, v16
v_and_b32 v13, 7, v15
v_lshlrev_b32 v13, 7, v13
v_add_u32 v16, v16, v13
v_lshrrev_b32 v14, 1, v15
v_lshrrev_b32 v13, 4, v5
v_xor_b32 v13, v14, v13
v_lshlrev_b32 v13, 4, v13
v_add_u32 v16, v16, v13
s_add_u32 s74, s73, 0
v_add_u32 v9, s74, v16
v_xor_b32 v10, 64, v9
s_add_u32 s74, s73, 65536
v_add_u32 v11, s74, v16
v_xor_b32 v12, 64, v11
s_waitcnt lgkmcnt(0)
s_add_u32 s74, s36, 3
s_lshr_b32 s74, s74, 2
s_lshl_b32 s74, s74, 2
s_add_u32 s68, s57, -1
s_mul_i32 s68, s68, s55
s_add_u32 s68, s68, s74
s_add_u32 s69, s58, -1
s_mul_i32 s69, s69, s56
s_add_u32 s69, s69, s74
s_add_u32 s50, s36, 255
s_lshr_b32 s50, s50, 8
s_lshl_b32 s50, s50, 8
s_add_u32 s70, s57, 31
s_lshr_b32 s70, s70, 5
s_mul_i32 s70, s70, s50
s_add_u32 s71, s58, 31
s_lshr_b32 s71, s71, 5
s_mul_i32 s71, s71, s50
s_add_u32 s26, s58, -1
s_mul_i32 s26, s26, s54
s_lshl_b32 s74, s57, 1
s_add_u32 s26, s26, s74
s_add_u32 s34, s58, -1
s_mul_i32 s34, s34, s72
s_lshl_b32 s74, s57, 1
s_add_u32 s34, s34, s74
s_mul_i32 s30, s57, s59
s_mul_i32 s30, s30, s58
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s57, s58
s_lshl_b32 s42, s57, 2
s_cmp_le_u32 s57, s58
s_cselect_b64 s[8:9], s[60:61], s[62:63]
s_cselect_b32 s10, s68, s69
s_cselect_b64 s[12:13], s[62:63], s[60:61]
s_cselect_b32 s14, s69, s68
s_cselect_b64 s[16:17], s[64:65], s[66:67]
s_cselect_b32 s18, s70, s71
s_cselect_b64 s[20:21], s[66:67], s[64:65]
s_cselect_b32 s22, s71, s70
s_cselect_b32 s51, s55, s56
s_cselect_b32 s52, s56, s55
s_cselect_b32 s39, 2, s54
s_cselect_b32 s40, s54, 2
s_cselect_b32 s43, 2, s72
s_cselect_b32 s44, s72, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s57, s58
s_max_u32 s46, s57, s58
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
s_lshl_b32 s54, s3, 4
s_mul_i32 s55, s2, 48
s_add_u32 s60, s54, 0
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s45
v_mul_lo_u32 v32, v13, s51
v_add_u32 v32, v32, v7
v_mov_b32 v37, -2147483648
v_cndmask_b32 v32, v37, v32, s[62:63]
s_add_u32 s60, s54, 8
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s45
v_mul_lo_u32 v33, v13, s51
v_add_u32 v33, v33, v8
v_mov_b32 v37, -2147483648
v_cndmask_b32 v33, v37, v33, s[62:63]
s_add_u32 s60, s55, 0
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v26, v13, s52
v_add_u32 v26, v26, v7
v_mov_b32 v37, -2147483648
v_cndmask_b32 v26, v37, v26, s[62:63]
s_add_u32 s60, s55, 8
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v27, v13, s52
v_add_u32 v27, v27, v8
v_mov_b32 v37, -2147483648
v_cndmask_b32 v27, v37, v27, s[62:63]
s_add_u32 s60, s55, 16
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v28, v13, s52
v_add_u32 v28, v28, v7
v_mov_b32 v37, -2147483648
v_cndmask_b32 v28, v37, v28, s[62:63]
s_add_u32 s60, s55, 24
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v29, v13, s52
v_add_u32 v29, v29, v8
v_mov_b32 v37, -2147483648
v_cndmask_b32 v29, v37, v29, s[62:63]
s_add_u32 s60, s55, 32
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v30, v13, s52
v_add_u32 v30, v30, v7
v_mov_b32 v37, -2147483648
v_cndmask_b32 v30, v37, v30, s[62:63]
s_add_u32 s60, s55, 40
v_add_u32 v13, s60, v6
v_cmp_lt_u32 s[62:63], v13, s46
v_mul_lo_u32 v31, v13, s52
v_add_u32 v31, v31, v8
v_mov_b32 v37, -2147483648
v_cndmask_b32 v31, v37, v31, s[62:63]
s_lshr_b32 s60, s54, 5
s_mul_i32 s60, s60, s50
v_add_u32 v5, s60, v4
s_and_b32 s59, 16, s54
s_lshr_b32 s59, s59, 1
s_add_u32 s60, s55, 0
s_lshr_b32 s61, s60, 5
s_mul_i32 s61, s61, s50
v_add_u32 v34, s61, v4
s_and_b32 s56, 16, s60
s_lshr_b32 s56, s56, 1
s_add_u32 s60, s55, 16
s_lshr_b32 s61, s60, 5
s_mul_i32 s61, s61, s50
v_add_u32 v35, s61, v4
s_and_b32 s57, 16, s60
s_lshr_b32 s57, s57, 1
s_add_u32 s60, s55, 32
s_lshr_b32 s61, s60, 5
s_mul_i32 s61, s61, s50
v_add_u32 v36, s61, v4
s_and_b32 s58, 16, s60
s_lshr_b32 s58, s58, 1
v_mov_b32 v14, 0
v_mov_b32 v15, 0
v_mov_b32 v16, 0
v_mov_b32 v17, 0
v_mov_b32 v18, 0
v_mov_b32 v19, 0
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v24, 0
v_mov_b32 v25, 0
s_cmp_eq_u32 s53, 0
s_cbranch_scc0 label_c_prefetch_skip_0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s60, s5, 2147483647
s_cmp_eq_u32 s60, 0
s_cbranch_scc1 label_c_prefetch_skip_0
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_1
s_add_u32 s61, s54, 0
v_add_u32 v56, s61, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s43
s_add_u32 s61, s55, 0
v_add_u32 v50, s61, v3
v_mul_lo_u32 v53, v50, s44
s_add_u32 s61, s55, 16
v_add_u32 v51, s61, v3
v_mul_lo_u32 v54, v51, s44
s_add_u32 s61, s55, 32
v_add_u32 v52, s61, v3
v_mul_lo_u32 v55, v52, s44
s_mov_b32 s64, s44
s_lshl_b32 s65, s44, 1
s_mul_i32 s66, s44, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[74:75], v50, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v38, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v50, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v39, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v50, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v40, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v50, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v41, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[74:75], v51, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v42, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v51, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v43, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v51, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v44, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v51, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v45, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[74:75], v52, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v46, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v52, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v47, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v52, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v48, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v52, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v49, v13, s[32:35], 0 offen offset:0
s_branch label_epilogue_end_2
label_orientation_b_1:
s_add_u32 s61, s3, 1
s_lshl_b32 s61, s61, 4
s_add_u32 s62, s2, 1
s_mul_i32 s62, s62, 48
s_cmp_le_u32 s61, s45
s_cselect_b32 s63, 1, 0
s_cmp_le_u32 s62, s46
s_cselect_b32 s64, 1, 0
s_and_b32 s60, s63, s64
s_add_u32 s61, s54, 0
v_add_u32 v56, s61, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s43
s_add_u32 s61, s55, 0
v_add_u32 v50, s61, v3
v_mul_lo_u32 v53, v50, s44
s_add_u32 s61, s55, 16
v_add_u32 v51, s61, v3
v_mul_lo_u32 v54, v51, s44
s_add_u32 s61, s55, 32
v_add_u32 v52, s61, v3
v_mul_lo_u32 v55, v52, s44
s_mov_b32 s64, s44
s_lshl_b32 s65, s44, 1
s_mul_i32 s66, s44, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_c_vector_3
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[74:75], v50, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v38, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v50, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v39, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v50, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v40, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v50, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v41, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[74:75], v51, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v42, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v51, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v43, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v51, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v44, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v51, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v45, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[74:75], v52, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v46, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v52, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v47, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v52, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v48, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v52, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_load_ushort v49, v13, s[32:35], 0 offen offset:0
s_branch label_c_loaded_4
label_c_vector_3:
v_add_u32 v13, v37, v53
buffer_load_dwordx2 v[38:39], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v54
buffer_load_dwordx2 v[40:41], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v37, v55
buffer_load_dwordx2 v[42:43], v13, s[32:35], 0 offen offset:0
label_c_loaded_4:
label_epilogue_end_2:
label_c_prefetch_skip_0:
s_mul_i32 s62, s4, s37
s_sub_u32 s63, s36, s62
s_min_i32 s63, s63, s37
s_lshr_b32 s63, s63, 8
s_add_u32 s64, s63, 0
s_sub_u32 s64, s64, s53
s_mov_b32 s60, s64
s_lshl_b32 s61, s53, 8
s_add_u32 s61, s61, s62
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1920
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 18304
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 m0, s73, 32768
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 36864
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 40960
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 45056
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 33792
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 37888
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 41984
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 46080
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 34688
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 38784
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 42880
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 46976
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 35712
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 39808
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 43904
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 48000
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v58, v34, s[20:23], s61 offen offset:0
buffer_load_dword v59, v35, s[20:23], s61 offen offset:0
buffer_load_dword v60, v36, s[20:23], s61 offen offset:0
buffer_load_dword v61, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 m0, s73, 49152
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 53248
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 57344
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 61440
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 50176
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 54272
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 58368
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 62464
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 51072
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 55168
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 59264
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 63360
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 52096
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 56192
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 60288
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 64384
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v62, v34, s[20:23], s61 offen offset:0
buffer_load_dword v63, v35, s[20:23], s61 offen offset:0
buffer_load_dword v64, v36, s[20:23], s61 offen offset:0
buffer_load_dword v65, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 m0, s73, 65536
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 69632
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 73728
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 77824
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 66560
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 70656
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 74752
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 78848
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 67456
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 71552
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 75648
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 79744
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 68480
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 72576
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 76672
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 80768
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v66, v34, s[20:23], s61 offen offset:0
buffer_load_dword v67, v35, s[20:23], s61 offen offset:0
buffer_load_dword v68, v36, s[20:23], s61 offen offset:0
buffer_load_dword v69, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 m0, s73, 81920
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 86016
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 90112
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 94208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 82944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 87040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 91136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 95232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 83840
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 87936
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 92032
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 96128
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 84864
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 88960
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 93056
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 97152
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
buffer_load_dword v70, v34, s[20:23], s61 offen offset:0
buffer_load_dword v71, v35, s[20:23], s61 offen offset:0
buffer_load_dword v72, v36, s[20:23], s61 offen offset:0
buffer_load_dword v73, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_mul_hi_i32 s62, s60, 715827883
s_lshr_b32 s63, s62, 31
s_add_u32 s62, s62, s63
s_mul_i32 s62, s62, 6
s_add_u32 s62, s62, -6
s_max_i32 s62, s62, 0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_cmp_gt_i32 s62, 0
s_mov_b32 s63, 0
s_cbranch_scc0 label_kloop_end_6
.p2align 4
label_kloop_5:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 32768
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 36864
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 40960
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 45056
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 33792
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 37888
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 41984
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 46080
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 38784
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 46976
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 35712
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 39808
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 43904
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 48000
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
buffer_load_dword v58, v34, s[20:23], s61 offen offset:0
buffer_load_dword v59, v35, s[20:23], s61 offen offset:0
buffer_load_dword v60, v36, s[20:23], s61 offen offset:0
buffer_load_dword v61, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 49152
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 53248
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 57344
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 61440
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 50176
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 54272
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 58368
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 62464
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 51072
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 55168
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 59264
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 63360
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 52096
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 56192
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 60288
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 64384
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
buffer_load_dword v62, v34, s[20:23], s61 offen offset:0
buffer_load_dword v63, v35, s[20:23], s61 offen offset:0
buffer_load_dword v64, v36, s[20:23], s61 offen offset:0
buffer_load_dword v65, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 65536
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 69632
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 73728
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 77824
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 66560
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 70656
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 74752
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 78848
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 67456
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 71552
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 75648
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 79744
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 68480
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 72576
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 76672
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 80768
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
buffer_load_dword v66, v34, s[20:23], s61 offen offset:0
buffer_load_dword v67, v35, s[20:23], s61 offen offset:0
buffer_load_dword v68, v36, s[20:23], s61 offen offset:0
buffer_load_dword v69, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 81920
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 86016
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 90112
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 94208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 82944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 87040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 91136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 95232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 83840
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 87936
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 92032
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 96128
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 84864
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 88960
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 93056
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 97152
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
buffer_load_dword v70, v34, s[20:23], s61 offen offset:0
buffer_load_dword v71, v35, s[20:23], s61 offen offset:0
buffer_load_dword v72, v36, s[20:23], s61 offen offset:0
buffer_load_dword v73, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_add_u32 s63, s63, 6
s_cmp_lt_i32 s63, s62
s_cbranch_scc1 label_kloop_5
label_kloop_end_6:
s_sub_u32 s63, s60, s62
s_cmp_eq_i32 s63, 1
s_cbranch_scc0 label_arm_end_7
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_7:
s_cmp_eq_i32 s63, 2
s_cbranch_scc0 label_arm_end_8
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_8:
s_cmp_eq_i32 s63, 3
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(60)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s63, 4
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(60)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(40)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s63, 5
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(60)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(40)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s63, 6
s_cbranch_scc0 label_arm_end_12
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(60)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(40)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
s_waitcnt vmcnt(0)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s63, 7
s_cbranch_scc0 label_arm_end_13
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(60)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_cmp_eq_i32 s63, 8
s_cbranch_scc0 label_arm_end_14
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(63)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(60)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
s_waitcnt vmcnt(40)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(0)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_14:
s_cmp_eq_i32 s63, 9
s_cbranch_scc0 label_arm_end_15
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 32768
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 36864
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 40960
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 45056
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 33792
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 37888
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 41984
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 46080
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 38784
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 46976
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 35712
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 39808
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 43904
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 48000
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
buffer_load_dword v58, v34, s[20:23], s61 offen offset:0
buffer_load_dword v59, v35, s[20:23], s61 offen offset:0
buffer_load_dword v60, v36, s[20:23], s61 offen offset:0
buffer_load_dword v61, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt vmcnt(63)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(63)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
s_waitcnt vmcnt(60)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(0)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_15:
s_cmp_eq_i32 s63, 10
s_cbranch_scc0 label_arm_end_16
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 32768
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 36864
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 40960
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 45056
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 33792
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 37888
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 41984
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 46080
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 38784
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 46976
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 35712
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 39808
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 43904
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 48000
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
buffer_load_dword v58, v34, s[20:23], s61 offen offset:0
buffer_load_dword v59, v35, s[20:23], s61 offen offset:0
buffer_load_dword v60, v36, s[20:23], s61 offen offset:0
buffer_load_dword v61, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 49152
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 53248
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 57344
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 61440
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 50176
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 54272
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 58368
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 62464
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 51072
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 55168
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 59264
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 63360
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 52096
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 56192
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 60288
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 64384
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
buffer_load_dword v62, v34, s[20:23], s61 offen offset:0
buffer_load_dword v63, v35, s[20:23], s61 offen offset:0
buffer_load_dword v64, v36, s[20:23], s61 offen offset:0
buffer_load_dword v65, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt vmcnt(63)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
s_waitcnt vmcnt(63)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_waitcnt vmcnt(60)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(40)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(20)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(0)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_16:
s_cmp_eq_i32 s63, 11
s_cbranch_scc0 label_arm_end_17
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s73
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 4096
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 8192
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 12288
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 1024
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 5120
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 9216
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 13312
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 14208
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 2944
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 7040
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 11136
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 15232
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
buffer_load_dword v50, v34, s[20:23], s61 offen offset:0
buffer_load_dword v51, v35, s[20:23], s61 offen offset:0
buffer_load_dword v52, v36, s[20:23], s61 offen offset:0
buffer_load_dword v53, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 16384
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 20480
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 24576
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 28672
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 17408
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 21504
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 25600
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 29696
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 22400
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 30592
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 19328
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 23424
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 27520
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 31616
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
buffer_load_dword v54, v34, s[20:23], s61 offen offset:0
buffer_load_dword v55, v35, s[20:23], s61 offen offset:0
buffer_load_dword v56, v36, s[20:23], s61 offen offset:0
buffer_load_dword v57, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 32768
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 36864
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 40960
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 45056
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 33792
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 37888
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 41984
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 46080
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 38784
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 46976
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 35712
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 39808
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 43904
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 48000
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
buffer_load_dword v58, v34, s[20:23], s61 offen offset:0
buffer_load_dword v59, v35, s[20:23], s61 offen offset:0
buffer_load_dword v60, v36, s[20:23], s61 offen offset:0
buffer_load_dword v61, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 49152
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 53248
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 57344
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 61440
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 50176
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 54272
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 58368
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 62464
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 51072
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 55168
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 59264
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 63360
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 52096
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 56192
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 60288
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 64384
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
buffer_load_dword v62, v34, s[20:23], s61 offen offset:0
buffer_load_dword v63, v35, s[20:23], s61 offen offset:0
buffer_load_dword v64, v36, s[20:23], s61 offen offset:0
buffer_load_dword v65, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 65536
s_nop 0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 69632
s_nop 0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 73728
s_nop 0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 77824
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:0 lds
s_add_u32 m0, s73, 66560
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 70656
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 74752
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:0 lds
s_add_u32 m0, s73, 78848
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:0 lds
s_waitcnt vmcnt(63)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s73, 67456
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v26, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 71552
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v28, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 75648
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v30, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 79744
s_nop 0
buffer_load_dwordx4 v32, s[8:11], s61 offen offset:128 lds
s_add_u32 m0, s73, 68480
s_nop 0
buffer_load_dwordx4 v27, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 72576
s_nop 0
buffer_load_dwordx4 v29, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 76672
s_nop 0
buffer_load_dwordx4 v31, s[12:15], s61 offen offset:128 lds
s_add_u32 m0, s73, 80768
s_nop 0
buffer_load_dwordx4 v33, s[8:11], s61 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v11 offset:16384
ds_read_b128 v[78:81], v12 offset:16384
ds_read_b128 v[82:85], v11 offset:20480
ds_read_b128 v[86:89], v12 offset:20480
ds_read_b128 v[90:93], v11 offset:24576
ds_read_b128 v[94:97], v12 offset:24576
ds_read_b128 v[98:101], v11 offset:28672
ds_read_b128 v[102:105], v12 offset:28672
buffer_load_dword v66, v34, s[20:23], s61 offen offset:0
buffer_load_dword v67, v35, s[20:23], s61 offen offset:0
buffer_load_dword v68, v36, s[20:23], s61 offen offset:0
buffer_load_dword v69, v5, s[16:19], s61 offen offset:0
s_add_u32 s61, s61, 256
s_waitcnt vmcnt(63)
v_lshrrev_b32 v70, s56, v70
v_lshrrev_b32 v71, s57, v71
v_lshrrev_b32 v72, s58, v72
v_lshrrev_b32 v73, s59, v73
ds_read_b128 v[106:109], v11 offset:18432
ds_read_b128 v[110:113], v12 offset:18432
ds_read_b128 v[114:117], v11 offset:22528
ds_read_b128 v[118:121], v12 offset:22528
ds_read_b128 v[122:125], v11 offset:26624
ds_read_b128 v[126:129], v12 offset:26624
ds_read_b128 v[130:133], v11 offset:30720
ds_read_b128 v[134:137], v12 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v70, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v71, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v72, v73 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:0
ds_read_b128 v[78:81], v10 offset:0
ds_read_b128 v[82:85], v9 offset:4096
ds_read_b128 v[86:89], v10 offset:4096
ds_read_b128 v[90:93], v9 offset:8192
ds_read_b128 v[94:97], v10 offset:8192
ds_read_b128 v[98:101], v9 offset:12288
ds_read_b128 v[102:105], v10 offset:12288
s_waitcnt vmcnt(63)
v_lshrrev_b32 v50, s56, v50
v_lshrrev_b32 v51, s57, v51
v_lshrrev_b32 v52, s58, v52
v_lshrrev_b32 v53, s59, v53
ds_read_b128 v[106:109], v9 offset:2048
ds_read_b128 v[110:113], v10 offset:2048
ds_read_b128 v[114:117], v9 offset:6144
ds_read_b128 v[118:121], v10 offset:6144
ds_read_b128 v[122:125], v9 offset:10240
ds_read_b128 v[126:129], v10 offset:10240
ds_read_b128 v[130:133], v9 offset:14336
ds_read_b128 v[134:137], v10 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v50, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v51, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v52, v53 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(63)
ds_read_b128 v[74:77], v9 offset:16384
ds_read_b128 v[78:81], v10 offset:16384
ds_read_b128 v[82:85], v9 offset:20480
ds_read_b128 v[86:89], v10 offset:20480
ds_read_b128 v[90:93], v9 offset:24576
ds_read_b128 v[94:97], v10 offset:24576
ds_read_b128 v[98:101], v9 offset:28672
ds_read_b128 v[102:105], v10 offset:28672
s_waitcnt vmcnt(60)
v_lshrrev_b32 v54, s56, v54
v_lshrrev_b32 v55, s57, v55
v_lshrrev_b32 v56, s58, v56
v_lshrrev_b32 v57, s59, v57
ds_read_b128 v[106:109], v9 offset:18432
ds_read_b128 v[110:113], v10 offset:18432
ds_read_b128 v[114:117], v9 offset:22528
ds_read_b128 v[118:121], v10 offset:22528
ds_read_b128 v[122:125], v9 offset:26624
ds_read_b128 v[126:129], v10 offset:26624
ds_read_b128 v[130:133], v9 offset:30720
ds_read_b128 v[134:137], v10 offset:30720
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v54, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v55, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(52)
ds_read_b128 v[74:77], v9 offset:32768
ds_read_b128 v[78:81], v10 offset:32768
ds_read_b128 v[82:85], v9 offset:36864
ds_read_b128 v[86:89], v10 offset:36864
ds_read_b128 v[90:93], v9 offset:40960
ds_read_b128 v[94:97], v10 offset:40960
ds_read_b128 v[98:101], v9 offset:45056
ds_read_b128 v[102:105], v10 offset:45056
s_waitcnt vmcnt(40)
v_lshrrev_b32 v58, s56, v58
v_lshrrev_b32 v59, s57, v59
v_lshrrev_b32 v60, s58, v60
v_lshrrev_b32 v61, s59, v61
ds_read_b128 v[106:109], v9 offset:34816
ds_read_b128 v[110:113], v10 offset:34816
ds_read_b128 v[114:117], v9 offset:38912
ds_read_b128 v[118:121], v10 offset:38912
ds_read_b128 v[122:125], v9 offset:43008
ds_read_b128 v[126:129], v10 offset:43008
ds_read_b128 v[130:133], v9 offset:47104
ds_read_b128 v[134:137], v10 offset:47104
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v58, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v59, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v60, v61 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(32)
ds_read_b128 v[74:77], v9 offset:49152
ds_read_b128 v[78:81], v10 offset:49152
ds_read_b128 v[82:85], v9 offset:53248
ds_read_b128 v[86:89], v10 offset:53248
ds_read_b128 v[90:93], v9 offset:57344
ds_read_b128 v[94:97], v10 offset:57344
ds_read_b128 v[98:101], v9 offset:61440
ds_read_b128 v[102:105], v10 offset:61440
s_waitcnt vmcnt(20)
v_lshrrev_b32 v62, s56, v62
v_lshrrev_b32 v63, s57, v63
v_lshrrev_b32 v64, s58, v64
v_lshrrev_b32 v65, s59, v65
ds_read_b128 v[106:109], v9 offset:51200
ds_read_b128 v[110:113], v10 offset:51200
ds_read_b128 v[114:117], v9 offset:55296
ds_read_b128 v[118:121], v10 offset:55296
ds_read_b128 v[122:125], v9 offset:59392
ds_read_b128 v[126:129], v10 offset:59392
ds_read_b128 v[130:133], v9 offset:63488
ds_read_b128 v[134:137], v10 offset:63488
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v62, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v63, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v64, v65 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(12)
ds_read_b128 v[74:77], v11 offset:0
ds_read_b128 v[78:81], v12 offset:0
ds_read_b128 v[82:85], v11 offset:4096
ds_read_b128 v[86:89], v12 offset:4096
ds_read_b128 v[90:93], v11 offset:8192
ds_read_b128 v[94:97], v12 offset:8192
ds_read_b128 v[98:101], v11 offset:12288
ds_read_b128 v[102:105], v12 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v66, s56, v66
v_lshrrev_b32 v67, s57, v67
v_lshrrev_b32 v68, s58, v68
v_lshrrev_b32 v69, s59, v69
ds_read_b128 v[106:109], v11 offset:2048
ds_read_b128 v[110:113], v12 offset:2048
ds_read_b128 v[114:117], v11 offset:6144
ds_read_b128 v[118:121], v12 offset:6144
ds_read_b128 v[122:125], v11 offset:10240
ds_read_b128 v[126:129], v12 offset:10240
ds_read_b128 v[130:133], v11 offset:14336
ds_read_b128 v[134:137], v12 offset:14336
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[74:81], v[98:105], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:89], v[98:105], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[98:105], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[106:113], v[130:137], v[14:17], v66, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:121], v[130:137], v[18:21], v67, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[130:137], v[22:25], v68, v69 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_17:
s_waitcnt 0
s_nop 11
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_19
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_21
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[6:7], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[6:7], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], s[6:7], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[6:7], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[6:7], v[24:25] op_sel_hi:[0,1]
s_and_b32 s60, s5, 2147483647
s_cmp_eq_u32 s60, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
v_lshlrev_b32 v37, 16, v38
v_fmac_f32 v14, s5, v37
v_lshlrev_b32 v37, 16, v39
v_fmac_f32 v15, s5, v37
v_lshlrev_b32 v37, 16, v40
v_fmac_f32 v16, s5, v37
v_lshlrev_b32 v37, 16, v41
v_fmac_f32 v17, s5, v37
v_lshlrev_b32 v37, 16, v42
v_fmac_f32 v18, s5, v37
v_lshlrev_b32 v37, 16, v43
v_fmac_f32 v19, s5, v37
v_lshlrev_b32 v37, 16, v44
v_fmac_f32 v20, s5, v37
v_lshlrev_b32 v37, 16, v45
v_fmac_f32 v21, s5, v37
v_lshlrev_b32 v37, 16, v46
v_fmac_f32 v22, s5, v37
v_lshlrev_b32 v37, 16, v47
v_fmac_f32 v23, s5, v37
v_lshlrev_b32 v37, 16, v48
v_fmac_f32 v24, s5, v37
v_lshlrev_b32 v37, 16, v49
v_fmac_f32 v25, s5, v37
label_no_beta_23:
s_add_u32 s69, s54, 0
v_add_u32 v56, s69, v1
v_cmp_lt_u32 s[60:61], v56, s45
v_mul_lo_u32 v37, v56, s39
s_add_u32 s69, s55, 0
v_add_u32 v50, s69, v3
v_mul_lo_u32 v53, v50, s40
s_add_u32 s69, s55, 16
v_add_u32 v51, s69, v3
v_mul_lo_u32 v54, v51, s40
s_add_u32 s69, s55, 32
v_add_u32 v52, s69, v3
v_mul_lo_u32 v55, v52, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_lshrrev_b32 v14, 16, v14
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[70:71], v50, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v14, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v13, v37, v53
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v50, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v15, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v13, v37, v53
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v50, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v16, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v50, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v17, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[70:71], v51, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v18, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v13, v37, v54
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v51, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v19, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v13, v37, v54
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v51, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v20, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v51, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v21, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[70:71], v52, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v22, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v13, v37, v55
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v52, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v23, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v13, v37, v55
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v52, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v24, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v52, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v25, v13, s[24:27], 0 offen offset:0
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s69, s4, s38
s_lshl_b32 s69, s69, 2
s_add_u32 s70, s54, 0
v_add_u32 v56, s70, v1
v_cmp_lt_u32 s[60:61], v56, s45
v_mul_lo_u32 v37, v56, s41
v_add_u32 v37, s69, v37
s_add_u32 s70, s55, 0
v_add_u32 v50, s70, v3
v_mul_lo_u32 v53, v50, s42
s_add_u32 s70, s55, 16
v_add_u32 v51, s70, v3
v_mul_lo_u32 v54, v51, s42
s_add_u32 s70, s55, 32
v_add_u32 v52, s70, v3
v_mul_lo_u32 v55, v52, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[70:71], v50, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v14, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v50, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v15, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v50, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v16, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v50, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v17, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[70:71], v51, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v18, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v51, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v19, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v51, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v20, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v51, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v21, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[70:71], v52, s65
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v22, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v52, s66
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v23, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v52, s67
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v24, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v52, s68
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v25, v13, s[28:31], 0 offen offset:0
label_stored_22:
s_branch label_epilogue_end_20
label_orientation_b_19:
s_add_u32 s61, s3, 1
s_lshl_b32 s61, s61, 4
s_add_u32 s62, s2, 1
s_mul_i32 s62, s62, 48
s_cmp_le_u32 s61, s45
s_cselect_b32 s63, 1, 0
s_cmp_le_u32 s62, s46
s_cselect_b32 s64, 1, 0
s_and_b32 s60, s63, s64
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_24
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[6:7], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[6:7], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], s[6:7], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[6:7], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[6:7], v[24:25] op_sel_hi:[0,1]
s_and_b32 s61, s5, 2147483647
s_cmp_eq_u32 s61, 0
s_cbranch_scc1 label_no_beta_26
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s60, 1
s_cbranch_scc0 label_c_masked_27
v_lshlrev_b32 v37, 16, v38
v_fmac_f32 v14, s5, v37
v_and_b32 v37, -65536, v38
v_fmac_f32 v15, s5, v37
v_lshlrev_b32 v37, 16, v39
v_fmac_f32 v16, s5, v37
v_and_b32 v37, -65536, v39
v_fmac_f32 v17, s5, v37
v_lshlrev_b32 v37, 16, v40
v_fmac_f32 v18, s5, v37
v_and_b32 v37, -65536, v40
v_fmac_f32 v19, s5, v37
v_lshlrev_b32 v37, 16, v41
v_fmac_f32 v20, s5, v37
v_and_b32 v37, -65536, v41
v_fmac_f32 v21, s5, v37
v_lshlrev_b32 v37, 16, v42
v_fmac_f32 v22, s5, v37
v_and_b32 v37, -65536, v42
v_fmac_f32 v23, s5, v37
v_lshlrev_b32 v37, 16, v43
v_fmac_f32 v24, s5, v37
v_and_b32 v37, -65536, v43
v_fmac_f32 v25, s5, v37
s_branch label_c_added_28
label_c_masked_27:
v_lshlrev_b32 v37, 16, v38
v_fmac_f32 v14, s5, v37
v_lshlrev_b32 v37, 16, v39
v_fmac_f32 v15, s5, v37
v_lshlrev_b32 v37, 16, v40
v_fmac_f32 v16, s5, v37
v_lshlrev_b32 v37, 16, v41
v_fmac_f32 v17, s5, v37
v_lshlrev_b32 v37, 16, v42
v_fmac_f32 v18, s5, v37
v_lshlrev_b32 v37, 16, v43
v_fmac_f32 v19, s5, v37
v_lshlrev_b32 v37, 16, v44
v_fmac_f32 v20, s5, v37
v_lshlrev_b32 v37, 16, v45
v_fmac_f32 v21, s5, v37
v_lshlrev_b32 v37, 16, v46
v_fmac_f32 v22, s5, v37
v_lshlrev_b32 v37, 16, v47
v_fmac_f32 v23, s5, v37
v_lshlrev_b32 v37, 16, v48
v_fmac_f32 v24, s5, v37
v_lshlrev_b32 v37, 16, v49
v_fmac_f32 v25, s5, v37
label_c_added_28:
label_no_beta_26:
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_vector_29
s_add_u32 s61, s54, 0
v_add_u32 v56, s61, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s39
s_add_u32 s61, s55, 0
v_add_u32 v50, s61, v3
v_mul_lo_u32 v53, v50, s40
s_add_u32 s61, s55, 16
v_add_u32 v51, s61, v3
v_mul_lo_u32 v54, v51, s40
s_add_u32 s61, s55, 32
v_add_u32 v52, s61, v3
v_mul_lo_u32 v55, v52, s40
s_mov_b32 s64, s40
s_lshl_b32 s65, s40, 1
s_mul_i32 s66, s40, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
v_lshrrev_b32 v14, 16, v14
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[74:75], v50, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v14, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v50, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v15, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v13, v37, v53
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v50, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v16, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v13, v37, v53
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v50, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v17, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[74:75], v51, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v18, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v51, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v19, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v13, v37, v54
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v51, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v20, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v13, v37, v54
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v51, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v21, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[74:75], v52, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v22, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v52, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v23, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v13, v37, v55
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v52, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v24, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v13, v37, v55
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v52, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_short v25, v13, s[24:27], 0 offen offset:0
s_branch label_tile_stored_30
label_vector_29:
s_add_u32 s61, s54, 0
v_add_u32 v56, s61, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s39
s_add_u32 s61, s55, 0
v_add_u32 v50, s61, v3
v_mul_lo_u32 v53, v50, s40
s_add_u32 s61, s55, 16
v_add_u32 v51, s61, v3
v_mul_lo_u32 v54, v51, s40
s_add_u32 s61, s55, 32
v_add_u32 v52, s61, v3
v_mul_lo_u32 v55, v52, s40
s_mov_b32 s64, s40
s_lshl_b32 s65, s40, 1
s_mul_i32 s66, s40, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
v_add_u32 v13, v37, v53
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_lshrrev_b32 v16, 16, v16
v_lshrrev_b32 v17, 16, v17
v_pack_b32_f16 v14, v14, v15
v_pack_b32_f16 v15, v16, v17
buffer_store_dwordx2 v[14:15], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v37, v54
v_lshrrev_b32 v18, 16, v18
v_lshrrev_b32 v19, 16, v19
v_lshrrev_b32 v20, 16, v20
v_lshrrev_b32 v21, 16, v21
v_pack_b32_f16 v18, v18, v19
v_pack_b32_f16 v19, v20, v21
buffer_store_dwordx2 v[18:19], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v37, v55
v_lshrrev_b32 v22, 16, v22
v_lshrrev_b32 v23, 16, v23
v_lshrrev_b32 v24, 16, v24
v_lshrrev_b32 v25, 16, v25
v_pack_b32_f16 v22, v22, v23
v_pack_b32_f16 v23, v24, v25
buffer_store_dwordx2 v[22:23], v13, s[24:27], 0 offen offset:0
label_tile_stored_30:
s_branch label_stored_25
label_workspace_24:
s_mul_i32 s61, s4, s38
s_lshl_b32 s61, s61, 2
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_vector_31
s_add_u32 s71, s54, 0
v_add_u32 v56, s71, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s41
v_add_u32 v37, s61, v37
s_add_u32 s71, s55, 0
v_add_u32 v50, s71, v3
v_mul_lo_u32 v53, v50, s42
s_add_u32 s71, s55, 16
v_add_u32 v51, s71, v3
v_mul_lo_u32 v54, v51, s42
s_add_u32 s71, s55, 32
v_add_u32 v52, s71, v3
v_mul_lo_u32 v55, v52, s42
s_mov_b32 s64, s42
s_lshl_b32 s65, s42, 1
s_mul_i32 s66, s42, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
v_add_u32 v13, v37, v53
v_cmp_lt_i32 s[74:75], v50, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v14, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v50, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v15, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v50, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v16, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v53
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v50, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v17, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_cmp_lt_i32 s[74:75], v51, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v18, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v51, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v19, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v51, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v20, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v51, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v21, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_cmp_lt_i32 s[74:75], v52, s67
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v22, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[74:75], v52, s68
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v23, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s65, v13
v_cmp_lt_i32 s[74:75], v52, s69
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v24, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
v_add_u32 v13, s66, v13
v_cmp_lt_i32 s[74:75], v52, s70
s_and_b64 s[74:75], s[74:75], s[62:63]
v_cndmask_b32 v13, -1, v13, s[74:75]
buffer_store_dword v25, v13, s[28:31], 0 offen offset:0
s_branch label_tile_stored_32
label_vector_31:
s_add_u32 s71, s54, 0
v_add_u32 v56, s71, v1
v_cmp_lt_u32 s[62:63], v56, s45
v_mul_lo_u32 v37, v56, s41
v_add_u32 v37, s61, v37
s_add_u32 s71, s55, 0
v_add_u32 v50, s71, v3
v_mul_lo_u32 v53, v50, s42
s_add_u32 s71, s55, 16
v_add_u32 v51, s71, v3
v_mul_lo_u32 v54, v51, s42
s_add_u32 s71, s55, 32
v_add_u32 v52, s71, v3
v_mul_lo_u32 v55, v52, s42
s_mov_b32 s64, s42
s_lshl_b32 s65, s42, 1
s_mul_i32 s66, s42, 3
s_sub_u32 s67, s46, 0
s_sub_u32 s68, s46, 1
s_sub_u32 s69, s46, 2
s_sub_u32 s70, s46, 3
v_add_u32 v13, v37, v53
buffer_store_dwordx4 v[14:17], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v54
buffer_store_dwordx4 v[18:21], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v37, v55
buffer_store_dwordx4 v[22:25], v13, s[28:31], 0 offen offset:0
label_tile_stored_32:
label_stored_25:
label_epilogue_end_20:
label_tile_end_18:
s_endpgm
.Lgemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 144
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr 76
  .amdhsa_group_segment_fixed_size 98304
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.amdgpu_metadata
---
custom.config:
  Source:
    Origin: skinny_mxfp8
    Repository: "TensileLite C++ skinny GEMM emitter (sweep selection)"
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: address, semantic: AddressD },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressMXScaleA },
            { type: address, semantic: AddressMXScaleB },
            { type: uint64, semantic: ConstantZero },
            { type: address, semantic: AddressC },
            { type: int32, semantic: StrideD0Bytes },
            { type: int32, semantic: StrideA0Bytes },
            { type: int32, semantic: StrideB0Bytes },
            { type: int32, semantic: SizeFree0 },
            { type: int32, semantic: SizeFree1 },
            { type: int32, semantic: SizeSum },
            { type: int32, semantic: Spk },
            { type: int32, semantic: StrideC0Bytes },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int32, semantic: NumWorkGroupsX },
            { type: int32, semantic: NumWorkGroupsY },
            { type: int32, semantic: NumWorkGroupsZ } ]
    macrotile: [48, 16, 4096]
    threads: [64, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
    kStep: 256
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 256
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  AssertStrideAMultiple: { 1: 4 }
  AssertStrideBMultiple: { 1: 4 }
  AssertTensorBytesLessThan: 4294967296
  StaggerU: 0
  MXScaleFormat: HostPreSwizzle
  WavefrontSize: 64
amdhsa.version: [1, 1]
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.kernels:
  - .name: gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN3_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 82
    .vgpr_count: 144
    .agpr_count: 0
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .args:
      - .name: D
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: A
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: B
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .name: scalesA
        .size: 8
        .offset: 24
        .value_kind: global_buffer
        .address_space: global
      - .name: scalesB
        .size: 8
        .offset: 32
        .value_kind: global_buffer
        .address_space: global
      - .name: ws
        .size: 8
        .offset: 40
        .value_kind: global_buffer
        .address_space: global
      - .name: C
        .size: 8
        .offset: 48
        .value_kind: global_buffer
        .address_space: global
      - .name: ldD
        .size: 4
        .offset: 56
        .value_kind: by_value
      - .name: ldA
        .size: 4
        .offset: 60
        .value_kind: by_value
      - .name: ldB
        .size: 4
        .offset: 64
        .value_kind: by_value
      - .name: m
        .size: 4
        .offset: 68
        .value_kind: by_value
      - .name: n
        .size: 4
        .offset: 72
        .value_kind: by_value
      - .name: k
        .size: 4
        .offset: 76
        .value_kind: by_value
      - .name: spk
        .size: 4
        .offset: 80
        .value_kind: by_value
      - .name: ldC
        .size: 4
        .offset: 84
        .value_kind: by_value
      - .name: alpha
        .size: 4
        .offset: 88
        .value_kind: by_value
      - .name: beta
        .size: 4
        .offset: 92
        .value_kind: by_value
      - .name: numWorkGroupsX
        .size: 4
        .offset: 96
        .value_kind: by_value
      - .name: numWorkGroupsY
        .size: 4
        .offset: 100
        .value_kind: by_value
      - .name: numWorkGroupsZ
        .size: 4
        .offset: 104
        .value_kind: by_value
...
.end_amdgpu_metadata
