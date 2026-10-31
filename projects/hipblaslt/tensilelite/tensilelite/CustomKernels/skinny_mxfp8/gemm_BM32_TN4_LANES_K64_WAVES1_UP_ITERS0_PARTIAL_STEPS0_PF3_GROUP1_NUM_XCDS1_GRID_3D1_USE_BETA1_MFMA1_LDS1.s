.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.globl gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1,@function
gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1:
s_load_dword s58, s[0:1], 104
s_load_dword s48, s[0:1], 96
s_load_dword s49, s[0:1], 100
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s57, s[0:1], 72
s_load_dword s56, s[0:1], 68
s_load_dword s55, s[0:1], 64
s_load_dword s54, s[0:1], 60
s_load_dword s53, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[66:67], s[0:1], 32
s_load_dwordx2 s[64:65], s[0:1], 24
s_load_dwordx2 s[62:63], s[0:1], 16
s_load_dwordx2 s[60:61], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s71, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
s_waitcnt lgkmcnt(0)
s_add_u32 s72, s36, 3
s_lshr_b32 s72, s72, 2
s_lshl_b32 s72, s72, 2
s_add_u32 s59, s56, -1
s_mul_i32 s59, s59, s54
s_add_u32 s59, s59, s72
s_add_u32 s68, s57, -1
s_mul_i32 s68, s68, s55
s_add_u32 s68, s68, s72
s_add_u32 s50, s36, 255
s_lshr_b32 s50, s50, 8
s_lshl_b32 s50, s50, 8
s_add_u32 s69, s56, 31
s_lshr_b32 s69, s69, 5
s_mul_i32 s69, s69, s50
s_add_u32 s70, s57, 31
s_lshr_b32 s70, s70, 5
s_mul_i32 s70, s70, s50
s_add_u32 s26, s57, -1
s_mul_i32 s26, s26, s53
s_lshl_b32 s72, s56, 1
s_add_u32 s26, s26, s72
s_add_u32 s34, s57, -1
s_mul_i32 s34, s34, s71
s_lshl_b32 s72, s56, 1
s_add_u32 s34, s34, s72
s_mul_i32 s30, s56, s58
s_mul_i32 s30, s30, s57
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s56, s57
s_lshl_b32 s42, s56, 2
s_cmp_le_u32 s56, s57
s_cselect_b64 s[8:9], s[60:61], s[62:63]
s_cselect_b32 s10, s59, s68
s_cselect_b64 s[12:13], s[62:63], s[60:61]
s_cselect_b32 s14, s68, s59
s_cselect_b64 s[16:17], s[64:65], s[66:67]
s_cselect_b32 s18, s69, s70
s_cselect_b64 s[20:21], s[66:67], s[64:65]
s_cselect_b32 s22, s70, s69
s_cselect_b32 s51, s54, s55
s_cselect_b32 s52, s55, s54
s_cselect_b32 s39, 2, s53
s_cselect_b32 s40, s53, 2
s_cselect_b32 s43, 2, s71
s_cselect_b32 s44, s71, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s56, s57
s_max_u32 s46, s56, s57
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
s_mov_b32 s53, 0
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s54, s53, 73728
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
s_add_u32 s55, s54, 0
v_add_u32 v9, s55, v16
v_xor_b32 v10, 64, v9
s_add_u32 s55, s54, 65536
v_add_u32 v11, s55, v16
v_xor_b32 v12, 64, v11
s_lshl_b32 s55, s3, 5
s_lshl_b32 s56, s2, 6
s_add_u32 s57, s55, 0
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s45
s_nop 1
v_mul_lo_u32 v54, v13, s51
v_add_u32 v54, v54, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v54, v92, v54, s[58:59]
s_add_u32 s57, s55, 8
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s45
s_nop 1
v_mul_lo_u32 v55, v13, s51
v_add_u32 v55, v55, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v55, v92, v55, s[58:59]
s_add_u32 s57, s55, 16
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s45
s_nop 1
v_mul_lo_u32 v56, v13, s51
v_add_u32 v56, v56, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v56, v92, v56, s[58:59]
s_add_u32 s57, s55, 24
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s45
s_nop 1
v_mul_lo_u32 v57, v13, s51
v_add_u32 v57, v57, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v57, v92, v57, s[58:59]
s_add_u32 s57, s56, 0
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v46, v13, s52
v_add_u32 v46, v46, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v46, v92, v46, s[58:59]
s_add_u32 s57, s56, 8
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v47, v13, s52
v_add_u32 v47, v47, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v47, v92, v47, s[58:59]
s_add_u32 s57, s56, 16
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v48, v13, s52
v_add_u32 v48, v48, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v48, v92, v48, s[58:59]
s_add_u32 s57, s56, 24
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v49, v13, s52
v_add_u32 v49, v49, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v49, v92, v49, s[58:59]
s_add_u32 s57, s56, 32
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v50, v13, s52
v_add_u32 v50, v50, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v50, v92, v50, s[58:59]
s_add_u32 s57, s56, 40
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v51, v13, s52
v_add_u32 v51, v51, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v51, v92, v51, s[58:59]
s_add_u32 s57, s56, 48
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v52, v13, s52
v_add_u32 v52, v52, v7
v_mov_b32 v92, -2147483648
v_cndmask_b32 v52, v92, v52, s[58:59]
s_add_u32 s57, s56, 56
v_add_u32 v13, s57, v6
v_cmp_lt_u32 s[58:59], v13, s46
s_nop 1
v_mul_lo_u32 v53, v13, s52
v_add_u32 v53, v53, v8
v_mov_b32 v92, -2147483648
v_cndmask_b32 v53, v92, v53, s[58:59]
s_lshr_b32 s57, s55, 5
s_mul_i32 s57, s57, s50
v_add_u32 v5, s57, v4
s_add_u32 s57, s56, 0
s_lshr_b32 s58, s57, 5
s_mul_i32 s58, s58, s50
v_add_u32 v58, s58, v4
s_add_u32 s57, s56, 32
s_lshr_b32 s58, s57, 5
s_mul_i32 s58, s58, s50
v_add_u32 v59, s58, v4
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
v_mov_b32 v26, 0
v_mov_b32 v27, 0
v_mov_b32 v28, 0
v_mov_b32 v29, 0
v_mov_b32 v30, 0
v_mov_b32 v31, 0
v_mov_b32 v32, 0
v_mov_b32 v33, 0
v_mov_b32 v34, 0
v_mov_b32 v35, 0
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
s_nop 7
s_cmp_eq_u32 s53, 0
s_cbranch_scc0 label_c_prefetch_skip_0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s57, s5, 2147483647
s_cmp_eq_u32 s57, 0
s_cbranch_scc1 label_c_prefetch_skip_0
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_1
s_add_u32 s69, s55, 0
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s43
s_add_u32 s69, s55, 16
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s43
s_add_u32 s69, s56, 0
v_add_u32 v94, s69, v3
v_mul_lo_u32 v98, v94, s44
s_add_u32 s69, s56, 16
v_add_u32 v95, s69, v3
v_mul_lo_u32 v99, v95, s44
s_add_u32 s69, s56, 32
v_add_u32 v96, s69, v3
v_mul_lo_u32 v100, v96, s44
s_add_u32 s69, s56, 48
v_add_u32 v97, s69, v3
v_mul_lo_u32 v101, v97, s44
s_mov_b32 s62, s44
s_lshl_b32 s63, s44, 1
s_mul_i32 s64, s44, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v60, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v61, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v62, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v63, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v68, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v69, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v70, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v71, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v76, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v77, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v78, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v79, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v84, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v85, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v86, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v87, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v64, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v65, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v66, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v67, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v72, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v73, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v74, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v75, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v80, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v81, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v82, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v83, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v88, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v89, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v90, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v91, v13, s[32:35], 0 offen offset:0
s_branch label_epilogue_end_2
label_orientation_b_1:
s_add_u32 s58, s3, 1
s_lshl_b32 s58, s58, 5
s_add_u32 s59, s2, 1
s_lshl_b32 s59, s59, 6
s_cmp_le_u32 s58, s45
s_cselect_b32 s60, 1, 0
s_cmp_le_u32 s59, s46
s_cselect_b32 s61, 1, 0
s_and_b32 s57, s60, s61
s_add_u32 s69, s55, 0
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s43
s_add_u32 s69, s55, 16
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s43
s_add_u32 s69, s56, 0
v_add_u32 v94, s69, v3
v_mul_lo_u32 v98, v94, s44
s_add_u32 s69, s56, 16
v_add_u32 v95, s69, v3
v_mul_lo_u32 v99, v95, s44
s_add_u32 s69, s56, 32
v_add_u32 v96, s69, v3
v_mul_lo_u32 v100, v96, s44
s_add_u32 s69, s56, 48
v_add_u32 v97, s69, v3
v_mul_lo_u32 v101, v97, s44
s_mov_b32 s62, s44
s_lshl_b32 s63, s44, 1
s_mul_i32 s64, s44, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
s_cmp_eq_u32 s57, 1
s_cbranch_scc1 label_c_vector_3
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v60, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v61, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v62, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v63, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v68, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v69, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v70, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v71, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v76, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v77, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v78, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v79, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v84, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v85, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v86, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v87, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v64, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v65, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v66, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v67, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v72, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v73, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v74, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v75, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v80, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v81, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v82, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v83, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v88, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v89, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v90, v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_load_ushort v91, v13, s[32:35], 0 offen offset:0
s_branch label_c_loaded_4
label_c_vector_3:
v_add_u32 v13, v92, v98
buffer_load_dwordx2 v[60:61], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v99
buffer_load_dwordx2 v[62:63], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v100
buffer_load_dwordx2 v[64:65], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v92, v101
buffer_load_dwordx2 v[66:67], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v98
buffer_load_dwordx2 v[68:69], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v99
buffer_load_dwordx2 v[70:71], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v100
buffer_load_dwordx2 v[72:73], v13, s[32:35], 0 offen offset:0
v_add_u32 v13, v93, v101
buffer_load_dwordx2 v[74:75], v13, s[32:35], 0 offen offset:0
label_c_loaded_4:
label_epilogue_end_2:
label_c_prefetch_skip_0:
s_mul_i32 s59, s4, s37
s_sub_u32 s60, s36, s59
s_min_i32 s60, s60, s37
s_lshr_b32 s60, s60, 8
s_add_u32 s61, s60, 0
s_sub_u32 s61, s61, s53
s_mov_b32 s57, s61
s_lshl_b32 s58, s53, 8
s_add_u32 s58, s58, s59
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
buffer_load_dword v92, v58, s[20:23], s58 offen offset:0
buffer_load_dword v93, v59, s[20:23], s58 offen offset:0
buffer_load_dword v94, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 40960
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 45056
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 41984
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 46080
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 42880
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 46976
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 43904
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 48000
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
buffer_load_dword v96, v58, s[20:23], s58 offen offset:0
buffer_load_dword v97, v59, s[20:23], s58 offen offset:0
buffer_load_dword v98, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_add_u32 m0, s54, 49152
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 53248
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 57344
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 61440
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 65536
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 69632
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 50176
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 54272
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 58368
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 62464
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 66560
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 70656
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 51072
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 55168
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 59264
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 63360
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 67456
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 71552
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 52096
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 56192
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 60288
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 64384
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 68480
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 72576
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
buffer_load_dword v100, v58, s[20:23], s58 offen offset:0
buffer_load_dword v101, v59, s[20:23], s58 offen offset:0
buffer_load_dword v102, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_mul_hi_i32 s59, s57, 1431655766
s_lshr_b32 s60, s59, 31
s_add_u32 s59, s59, s60
s_mul_i32 s59, s59, 3
s_add_u32 s59, s59, -3
s_max_i32 s59, s59, 0
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:0
ds_read_b128 v[108:111], v10 offset:0
ds_read_b128 v[112:115], v9 offset:4096
ds_read_b128 v[116:119], v10 offset:4096
ds_read_b128 v[120:123], v9 offset:8192
ds_read_b128 v[124:127], v10 offset:8192
ds_read_b128 v[128:131], v9 offset:12288
ds_read_b128 v[132:135], v10 offset:12288
ds_read_b128 v[136:139], v9 offset:16384
ds_read_b128 v[140:143], v10 offset:16384
ds_read_b128 v[144:147], v9 offset:20480
ds_read_b128 v[148:151], v10 offset:20480
s_cmp_gt_i32 s59, 0
s_mov_b32 s60, 0
s_cbranch_scc0 label_kloop_end_6
.p2align 4
label_kloop_5:
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v92, v58, s[20:23], s58 offen offset:0
buffer_load_dword v93, v59, s[20:23], s58 offen offset:0
buffer_load_dword v94, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 40960
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 45056
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 41984
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 46080
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 42880
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 46976
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 43904
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 48000
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:49152
ds_read_b128 v[108:111], v10 offset:49152
ds_read_b128 v[112:115], v9 offset:53248
ds_read_b128 v[116:119], v10 offset:53248
ds_read_b128 v[120:123], v9 offset:57344
ds_read_b128 v[124:127], v10 offset:57344
ds_read_b128 v[128:131], v9 offset:61440
ds_read_b128 v[132:135], v10 offset:61440
ds_read_b128 v[136:139], v11 offset:0
ds_read_b128 v[140:143], v12 offset:0
ds_read_b128 v[144:147], v11 offset:4096
ds_read_b128 v[148:151], v12 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v96, v58, s[20:23], s58 offen offset:0
buffer_load_dword v97, v59, s[20:23], s58 offen offset:0
buffer_load_dword v98, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:51200
ds_read_b128 v[156:159], v10 offset:51200
ds_read_b128 v[160:163], v9 offset:55296
ds_read_b128 v[164:167], v10 offset:55296
ds_read_b128 v[168:171], v9 offset:59392
ds_read_b128 v[172:175], v10 offset:59392
ds_read_b128 v[176:179], v9 offset:63488
ds_read_b128 v[180:183], v10 offset:63488
ds_read_b128 v[184:187], v11 offset:2048
ds_read_b128 v[188:191], v12 offset:2048
ds_read_b128 v[192:195], v11 offset:6144
ds_read_b128 v[196:199], v12 offset:6144
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 49152
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 53248
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 57344
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 61440
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 65536
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 69632
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 50176
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 54272
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 58368
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 62464
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 66560
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 70656
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 51072
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 55168
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 59264
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 63360
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 67456
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 71552
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 52096
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 56192
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 60288
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 64384
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 68480
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 72576
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:0
ds_read_b128 v[108:111], v10 offset:0
ds_read_b128 v[112:115], v9 offset:4096
ds_read_b128 v[116:119], v10 offset:4096
ds_read_b128 v[120:123], v9 offset:8192
ds_read_b128 v[124:127], v10 offset:8192
ds_read_b128 v[128:131], v9 offset:12288
ds_read_b128 v[132:135], v10 offset:12288
ds_read_b128 v[136:139], v9 offset:16384
ds_read_b128 v[140:143], v10 offset:16384
ds_read_b128 v[144:147], v9 offset:20480
ds_read_b128 v[148:151], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v100, v58, s[20:23], s58 offen offset:0
buffer_load_dword v101, v59, s[20:23], s58 offen offset:0
buffer_load_dword v102, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_add_u32 s60, s60, 3
s_cmp_lt_i32 s60, s59
s_cbranch_scc1 label_kloop_5
label_kloop_end_6:
s_sub_u32 s60, s57, s59
s_cmp_eq_i32 s60, 1
s_cbranch_scc0 label_arm_end_7
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_7:
s_cmp_eq_i32 s60, 2
s_cbranch_scc0 label_arm_end_8
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(42)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(27)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_8:
s_cmp_eq_i32 s60, 3
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(42)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(27)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(15)
ds_read_b128 v[104:107], v9 offset:49152
ds_read_b128 v[108:111], v10 offset:49152
ds_read_b128 v[112:115], v9 offset:53248
ds_read_b128 v[116:119], v10 offset:53248
ds_read_b128 v[120:123], v9 offset:57344
ds_read_b128 v[124:127], v10 offset:57344
ds_read_b128 v[128:131], v9 offset:61440
ds_read_b128 v[132:135], v10 offset:61440
ds_read_b128 v[136:139], v11 offset:0
ds_read_b128 v[140:143], v12 offset:0
ds_read_b128 v[144:147], v11 offset:4096
ds_read_b128 v[148:151], v12 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[152:155], v9 offset:51200
ds_read_b128 v[156:159], v10 offset:51200
ds_read_b128 v[160:163], v9 offset:55296
ds_read_b128 v[164:167], v10 offset:55296
ds_read_b128 v[168:171], v9 offset:59392
ds_read_b128 v[172:175], v10 offset:59392
ds_read_b128 v[176:179], v9 offset:63488
ds_read_b128 v[180:183], v10 offset:63488
ds_read_b128 v[184:187], v11 offset:2048
ds_read_b128 v[188:191], v12 offset:2048
ds_read_b128 v[192:195], v11 offset:6144
ds_read_b128 v[196:199], v12 offset:6144
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s60, 4
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v92, v58, s[20:23], s58 offen offset:0
buffer_load_dword v93, v59, s[20:23], s58 offen offset:0
buffer_load_dword v94, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(42)
ds_read_b128 v[104:107], v9 offset:49152
ds_read_b128 v[108:111], v10 offset:49152
ds_read_b128 v[112:115], v9 offset:53248
ds_read_b128 v[116:119], v10 offset:53248
ds_read_b128 v[120:123], v9 offset:57344
ds_read_b128 v[124:127], v10 offset:57344
ds_read_b128 v[128:131], v9 offset:61440
ds_read_b128 v[132:135], v10 offset:61440
ds_read_b128 v[136:139], v11 offset:0
ds_read_b128 v[140:143], v12 offset:0
ds_read_b128 v[144:147], v11 offset:4096
ds_read_b128 v[148:151], v12 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(27)
ds_read_b128 v[152:155], v9 offset:51200
ds_read_b128 v[156:159], v10 offset:51200
ds_read_b128 v[160:163], v9 offset:55296
ds_read_b128 v[164:167], v10 offset:55296
ds_read_b128 v[168:171], v9 offset:59392
ds_read_b128 v[172:175], v10 offset:59392
ds_read_b128 v[176:179], v9 offset:63488
ds_read_b128 v[180:183], v10 offset:63488
ds_read_b128 v[184:187], v11 offset:2048
ds_read_b128 v[188:191], v12 offset:2048
ds_read_b128 v[192:195], v11 offset:6144
ds_read_b128 v[196:199], v12 offset:6144
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(15)
ds_read_b128 v[104:107], v9 offset:0
ds_read_b128 v[108:111], v10 offset:0
ds_read_b128 v[112:115], v9 offset:4096
ds_read_b128 v[116:119], v10 offset:4096
ds_read_b128 v[120:123], v9 offset:8192
ds_read_b128 v[124:127], v10 offset:8192
ds_read_b128 v[128:131], v9 offset:12288
ds_read_b128 v[132:135], v10 offset:12288
ds_read_b128 v[136:139], v9 offset:16384
ds_read_b128 v[140:143], v10 offset:16384
ds_read_b128 v[144:147], v9 offset:20480
ds_read_b128 v[148:151], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s60, 5
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v92, v58, s[20:23], s58 offen offset:0
buffer_load_dword v93, v59, s[20:23], s58 offen offset:0
buffer_load_dword v94, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 40960
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 45056
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:0 lds
s_add_u32 m0, s54, 41984
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 46080
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v46, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v48, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v50, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v52, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 42880
s_nop 0
buffer_load_dwordx4 v54, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 46976
s_nop 0
buffer_load_dwordx4 v56, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v47, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v49, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v51, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v53, s[12:15], s58 offen offset:128 lds
s_add_u32 m0, s54, 43904
s_nop 0
buffer_load_dwordx4 v55, s[8:11], s58 offen offset:128 lds
s_add_u32 m0, s54, 48000
s_nop 0
buffer_load_dwordx4 v57, s[8:11], s58 offen offset:128 lds
s_waitcnt vmcnt(63)
ds_read_b128 v[104:107], v9 offset:49152
ds_read_b128 v[108:111], v10 offset:49152
ds_read_b128 v[112:115], v9 offset:53248
ds_read_b128 v[116:119], v10 offset:53248
ds_read_b128 v[120:123], v9 offset:57344
ds_read_b128 v[124:127], v10 offset:57344
ds_read_b128 v[128:131], v9 offset:61440
ds_read_b128 v[132:135], v10 offset:61440
ds_read_b128 v[136:139], v11 offset:0
ds_read_b128 v[140:143], v12 offset:0
ds_read_b128 v[144:147], v11 offset:4096
ds_read_b128 v[148:151], v12 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v96, v58, s[20:23], s58 offen offset:0
buffer_load_dword v97, v59, s[20:23], s58 offen offset:0
buffer_load_dword v98, v5, s[16:19], s58 offen offset:0
s_add_u32 s58, s58, 256
s_waitcnt vmcnt(54)
ds_read_b128 v[152:155], v9 offset:51200
ds_read_b128 v[156:159], v10 offset:51200
ds_read_b128 v[160:163], v9 offset:55296
ds_read_b128 v[164:167], v10 offset:55296
ds_read_b128 v[168:171], v9 offset:59392
ds_read_b128 v[172:175], v10 offset:59392
ds_read_b128 v[176:179], v9 offset:63488
ds_read_b128 v[180:183], v10 offset:63488
ds_read_b128 v[184:187], v11 offset:2048
ds_read_b128 v[188:191], v12 offset:2048
ds_read_b128 v[192:195], v11 offset:6144
ds_read_b128 v[196:199], v12 offset:6144
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(42)
ds_read_b128 v[104:107], v9 offset:0
ds_read_b128 v[108:111], v10 offset:0
ds_read_b128 v[112:115], v9 offset:4096
ds_read_b128 v[116:119], v10 offset:4096
ds_read_b128 v[120:123], v9 offset:8192
ds_read_b128 v[124:127], v10 offset:8192
ds_read_b128 v[128:131], v9 offset:12288
ds_read_b128 v[132:135], v10 offset:12288
ds_read_b128 v[136:139], v9 offset:16384
ds_read_b128 v[140:143], v10 offset:16384
ds_read_b128 v[144:147], v9 offset:20480
ds_read_b128 v[148:151], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v100, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v101, v102 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(27)
ds_read_b128 v[152:155], v9 offset:2048
ds_read_b128 v[156:159], v10 offset:2048
ds_read_b128 v[160:163], v9 offset:6144
ds_read_b128 v[164:167], v10 offset:6144
ds_read_b128 v[168:171], v9 offset:10240
ds_read_b128 v[172:175], v10 offset:10240
ds_read_b128 v[176:179], v9 offset:14336
ds_read_b128 v[180:183], v10 offset:14336
ds_read_b128 v[184:187], v9 offset:18432
ds_read_b128 v[188:191], v10 offset:18432
ds_read_b128 v[192:195], v9 offset:22528
ds_read_b128 v[196:199], v10 offset:22528
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(15)
ds_read_b128 v[104:107], v9 offset:24576
ds_read_b128 v[108:111], v10 offset:24576
ds_read_b128 v[112:115], v9 offset:28672
ds_read_b128 v[116:119], v10 offset:28672
ds_read_b128 v[120:123], v9 offset:32768
ds_read_b128 v[124:127], v10 offset:32768
ds_read_b128 v[128:131], v9 offset:36864
ds_read_b128 v[132:135], v10 offset:36864
ds_read_b128 v[136:139], v9 offset:40960
ds_read_b128 v[140:143], v10 offset:40960
ds_read_b128 v[144:147], v9 offset:45056
ds_read_b128 v[148:151], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v92, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v92, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v92, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v92, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v93, v94 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v93, v94 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v93, v94 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[152:155], v9 offset:26624
ds_read_b128 v[156:159], v10 offset:26624
ds_read_b128 v[160:163], v9 offset:30720
ds_read_b128 v[164:167], v10 offset:30720
ds_read_b128 v[168:171], v9 offset:34816
ds_read_b128 v[172:175], v10 offset:34816
ds_read_b128 v[176:179], v9 offset:38912
ds_read_b128 v[180:183], v10 offset:38912
ds_read_b128 v[184:187], v9 offset:43008
ds_read_b128 v[188:191], v10 offset:43008
ds_read_b128 v[192:195], v9 offset:47104
ds_read_b128 v[196:199], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[104:111], v[136:143], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[104:111], v[144:151], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[112:119], v[136:143], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[112:119], v[144:151], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[120:127], v[136:143], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[144:151], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[136:143], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[144:151], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[184:191], v[14:17], v96, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[192:199], v[18:21], v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[160:167], v[184:191], v[22:25], v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[160:167], v[192:199], v[26:29], v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[168:175], v[184:191], v[30:33], v97, v98 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[192:199], v[34:37], v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[176:183], v[184:191], v[38:41], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[176:183], v[192:199], v[42:45], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_nop 15
s_nop 15
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_13
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_15
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[6:7], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[6:7], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], s[6:7], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[6:7], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[6:7], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], s[6:7], v[26:27] op_sel_hi:[0,1]
v_pk_mul_f32 v[28:29], s[6:7], v[28:29] op_sel_hi:[0,1]
v_pk_mul_f32 v[30:31], s[6:7], v[30:31] op_sel_hi:[0,1]
v_pk_mul_f32 v[32:33], s[6:7], v[32:33] op_sel_hi:[0,1]
v_pk_mul_f32 v[34:35], s[6:7], v[34:35] op_sel_hi:[0,1]
v_pk_mul_f32 v[36:37], s[6:7], v[36:37] op_sel_hi:[0,1]
v_pk_mul_f32 v[38:39], s[6:7], v[38:39] op_sel_hi:[0,1]
v_pk_mul_f32 v[40:41], s[6:7], v[40:41] op_sel_hi:[0,1]
v_pk_mul_f32 v[42:43], s[6:7], v[42:43] op_sel_hi:[0,1]
v_pk_mul_f32 v[44:45], s[6:7], v[44:45] op_sel_hi:[0,1]
s_and_b32 s57, s5, 2147483647
s_cmp_eq_u32 s57, 0
s_cbranch_scc1 label_no_beta_17
s_waitcnt vmcnt(0)
v_lshlrev_b32 v92, 16, v60
v_fmac_f32 v14, s5, v92
v_lshlrev_b32 v92, 16, v61
v_fmac_f32 v15, s5, v92
v_lshlrev_b32 v92, 16, v62
v_fmac_f32 v16, s5, v92
v_lshlrev_b32 v92, 16, v63
v_fmac_f32 v17, s5, v92
v_lshlrev_b32 v92, 16, v68
v_fmac_f32 v22, s5, v92
v_lshlrev_b32 v92, 16, v69
v_fmac_f32 v23, s5, v92
v_lshlrev_b32 v92, 16, v70
v_fmac_f32 v24, s5, v92
v_lshlrev_b32 v92, 16, v71
v_fmac_f32 v25, s5, v92
v_lshlrev_b32 v92, 16, v76
v_fmac_f32 v30, s5, v92
v_lshlrev_b32 v92, 16, v77
v_fmac_f32 v31, s5, v92
v_lshlrev_b32 v92, 16, v78
v_fmac_f32 v32, s5, v92
v_lshlrev_b32 v92, 16, v79
v_fmac_f32 v33, s5, v92
v_lshlrev_b32 v92, 16, v84
v_fmac_f32 v38, s5, v92
v_lshlrev_b32 v92, 16, v85
v_fmac_f32 v39, s5, v92
v_lshlrev_b32 v92, 16, v86
v_fmac_f32 v40, s5, v92
v_lshlrev_b32 v92, 16, v87
v_fmac_f32 v41, s5, v92
v_lshlrev_b32 v92, 16, v64
v_fmac_f32 v18, s5, v92
v_lshlrev_b32 v92, 16, v65
v_fmac_f32 v19, s5, v92
v_lshlrev_b32 v92, 16, v66
v_fmac_f32 v20, s5, v92
v_lshlrev_b32 v92, 16, v67
v_fmac_f32 v21, s5, v92
v_lshlrev_b32 v92, 16, v72
v_fmac_f32 v26, s5, v92
v_lshlrev_b32 v92, 16, v73
v_fmac_f32 v27, s5, v92
v_lshlrev_b32 v92, 16, v74
v_fmac_f32 v28, s5, v92
v_lshlrev_b32 v92, 16, v75
v_fmac_f32 v29, s5, v92
v_lshlrev_b32 v92, 16, v80
v_fmac_f32 v34, s5, v92
v_lshlrev_b32 v92, 16, v81
v_fmac_f32 v35, s5, v92
v_lshlrev_b32 v92, 16, v82
v_fmac_f32 v36, s5, v92
v_lshlrev_b32 v92, 16, v83
v_fmac_f32 v37, s5, v92
v_lshlrev_b32 v92, 16, v88
v_fmac_f32 v42, s5, v92
v_lshlrev_b32 v92, 16, v89
v_fmac_f32 v43, s5, v92
v_lshlrev_b32 v92, 16, v90
v_fmac_f32 v44, s5, v92
v_lshlrev_b32 v92, 16, v91
v_fmac_f32 v45, s5, v92
label_no_beta_17:
s_add_u32 s57, s55, 0
v_add_u32 v102, s57, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s39
s_add_u32 s57, s55, 16
v_add_u32 v102, s57, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s39
s_add_u32 s57, s56, 0
v_add_u32 v94, s57, v3
v_mul_lo_u32 v98, v94, s40
s_add_u32 s57, s56, 16
v_add_u32 v95, s57, v3
v_mul_lo_u32 v99, v95, s40
s_add_u32 s57, s56, 32
v_add_u32 v96, s57, v3
v_mul_lo_u32 v100, v96, s40
s_add_u32 s57, s56, 48
v_add_u32 v97, s57, v3
v_mul_lo_u32 v101, v97, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_lshrrev_b32 v14, 16, v14
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v14, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v15, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v16, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v17, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v22, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v23, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v24, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v25, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v30, 16, v30
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v30, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v31, 16, v31
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v31, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v32, 16, v32
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v32, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v33, 16, v33
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v33, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v38, 16, v38
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v38, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v39, 16, v39
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v39, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v40, 16, v40
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v40, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v41, 16, v41
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v41, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v18, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v19, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v20, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v21, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v26, 16, v26
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v26, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v27, 16, v27
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v27, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v28, 16, v28
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v28, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v29, 16, v29
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v29, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v34, 16, v34
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v34, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v35, 16, v35
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v35, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v36, 16, v36
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v36, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v37, 16, v37
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v37, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v42, 16, v42
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v42, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v43, 16, v43
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v43, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v44, 16, v44
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v44, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v45, 16, v45
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v45, v13, s[24:27], 0 offen offset:0
s_branch label_stored_16
label_workspace_15:
s_mul_i32 s57, s4, s38
s_lshl_b32 s57, s57, 2
s_add_u32 s69, s55, 0
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s41
v_add_u32 v92, s57, v92
s_add_u32 s69, s55, 16
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s41
v_add_u32 v93, s57, v93
s_add_u32 s69, s56, 0
v_add_u32 v94, s69, v3
v_mul_lo_u32 v98, v94, s42
s_add_u32 s69, s56, 16
v_add_u32 v95, s69, v3
v_mul_lo_u32 v99, v95, s42
s_add_u32 s69, s56, 32
v_add_u32 v96, s69, v3
v_mul_lo_u32 v100, v96, s42
s_add_u32 s69, s56, 48
v_add_u32 v97, s69, v3
v_mul_lo_u32 v101, v97, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v14, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v15, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v16, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v17, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v22, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v23, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v24, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v25, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v30, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v31, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v32, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v33, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v38, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v39, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v40, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v41, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v18, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v19, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v20, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v21, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v26, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v27, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v28, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v29, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v34, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v35, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v36, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v37, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v42, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v43, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v44, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v45, v13, s[28:31], 0 offen offset:0
label_stored_16:
s_branch label_epilogue_end_14
label_orientation_b_13:
s_add_u32 s58, s3, 1
s_lshl_b32 s58, s58, 5
s_add_u32 s59, s2, 1
s_lshl_b32 s59, s59, 6
s_cmp_le_u32 s58, s45
s_cselect_b32 s60, 1, 0
s_cmp_le_u32 s59, s46
s_cselect_b32 s61, 1, 0
s_and_b32 s57, s60, s61
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_18
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[6:7], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[6:7], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], s[6:7], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[6:7], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[6:7], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], s[6:7], v[26:27] op_sel_hi:[0,1]
v_pk_mul_f32 v[28:29], s[6:7], v[28:29] op_sel_hi:[0,1]
v_pk_mul_f32 v[30:31], s[6:7], v[30:31] op_sel_hi:[0,1]
v_pk_mul_f32 v[32:33], s[6:7], v[32:33] op_sel_hi:[0,1]
v_pk_mul_f32 v[34:35], s[6:7], v[34:35] op_sel_hi:[0,1]
v_pk_mul_f32 v[36:37], s[6:7], v[36:37] op_sel_hi:[0,1]
v_pk_mul_f32 v[38:39], s[6:7], v[38:39] op_sel_hi:[0,1]
v_pk_mul_f32 v[40:41], s[6:7], v[40:41] op_sel_hi:[0,1]
v_pk_mul_f32 v[42:43], s[6:7], v[42:43] op_sel_hi:[0,1]
v_pk_mul_f32 v[44:45], s[6:7], v[44:45] op_sel_hi:[0,1]
s_and_b32 s58, s5, 2147483647
s_cmp_eq_u32 s58, 0
s_cbranch_scc1 label_no_beta_20
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s57, 1
s_cbranch_scc0 label_c_masked_21
v_lshlrev_b32 v92, 16, v60
v_fmac_f32 v14, s5, v92
v_and_b32 v92, -65536, v60
v_fmac_f32 v15, s5, v92
v_lshlrev_b32 v92, 16, v61
v_fmac_f32 v16, s5, v92
v_and_b32 v92, -65536, v61
v_fmac_f32 v17, s5, v92
v_lshlrev_b32 v92, 16, v62
v_fmac_f32 v22, s5, v92
v_and_b32 v92, -65536, v62
v_fmac_f32 v23, s5, v92
v_lshlrev_b32 v92, 16, v63
v_fmac_f32 v24, s5, v92
v_and_b32 v92, -65536, v63
v_fmac_f32 v25, s5, v92
v_lshlrev_b32 v92, 16, v64
v_fmac_f32 v30, s5, v92
v_and_b32 v92, -65536, v64
v_fmac_f32 v31, s5, v92
v_lshlrev_b32 v92, 16, v65
v_fmac_f32 v32, s5, v92
v_and_b32 v92, -65536, v65
v_fmac_f32 v33, s5, v92
v_lshlrev_b32 v92, 16, v66
v_fmac_f32 v38, s5, v92
v_and_b32 v92, -65536, v66
v_fmac_f32 v39, s5, v92
v_lshlrev_b32 v92, 16, v67
v_fmac_f32 v40, s5, v92
v_and_b32 v92, -65536, v67
v_fmac_f32 v41, s5, v92
v_lshlrev_b32 v92, 16, v68
v_fmac_f32 v18, s5, v92
v_and_b32 v92, -65536, v68
v_fmac_f32 v19, s5, v92
v_lshlrev_b32 v92, 16, v69
v_fmac_f32 v20, s5, v92
v_and_b32 v92, -65536, v69
v_fmac_f32 v21, s5, v92
v_lshlrev_b32 v92, 16, v70
v_fmac_f32 v26, s5, v92
v_and_b32 v92, -65536, v70
v_fmac_f32 v27, s5, v92
v_lshlrev_b32 v92, 16, v71
v_fmac_f32 v28, s5, v92
v_and_b32 v92, -65536, v71
v_fmac_f32 v29, s5, v92
v_lshlrev_b32 v92, 16, v72
v_fmac_f32 v34, s5, v92
v_and_b32 v92, -65536, v72
v_fmac_f32 v35, s5, v92
v_lshlrev_b32 v92, 16, v73
v_fmac_f32 v36, s5, v92
v_and_b32 v92, -65536, v73
v_fmac_f32 v37, s5, v92
v_lshlrev_b32 v92, 16, v74
v_fmac_f32 v42, s5, v92
v_and_b32 v92, -65536, v74
v_fmac_f32 v43, s5, v92
v_lshlrev_b32 v92, 16, v75
v_fmac_f32 v44, s5, v92
v_and_b32 v92, -65536, v75
v_fmac_f32 v45, s5, v92
s_branch label_c_added_22
label_c_masked_21:
v_lshlrev_b32 v92, 16, v60
v_fmac_f32 v14, s5, v92
v_lshlrev_b32 v92, 16, v61
v_fmac_f32 v15, s5, v92
v_lshlrev_b32 v92, 16, v62
v_fmac_f32 v16, s5, v92
v_lshlrev_b32 v92, 16, v63
v_fmac_f32 v17, s5, v92
v_lshlrev_b32 v92, 16, v68
v_fmac_f32 v22, s5, v92
v_lshlrev_b32 v92, 16, v69
v_fmac_f32 v23, s5, v92
v_lshlrev_b32 v92, 16, v70
v_fmac_f32 v24, s5, v92
v_lshlrev_b32 v92, 16, v71
v_fmac_f32 v25, s5, v92
v_lshlrev_b32 v92, 16, v76
v_fmac_f32 v30, s5, v92
v_lshlrev_b32 v92, 16, v77
v_fmac_f32 v31, s5, v92
v_lshlrev_b32 v92, 16, v78
v_fmac_f32 v32, s5, v92
v_lshlrev_b32 v92, 16, v79
v_fmac_f32 v33, s5, v92
v_lshlrev_b32 v92, 16, v84
v_fmac_f32 v38, s5, v92
v_lshlrev_b32 v92, 16, v85
v_fmac_f32 v39, s5, v92
v_lshlrev_b32 v92, 16, v86
v_fmac_f32 v40, s5, v92
v_lshlrev_b32 v92, 16, v87
v_fmac_f32 v41, s5, v92
v_lshlrev_b32 v92, 16, v64
v_fmac_f32 v18, s5, v92
v_lshlrev_b32 v92, 16, v65
v_fmac_f32 v19, s5, v92
v_lshlrev_b32 v92, 16, v66
v_fmac_f32 v20, s5, v92
v_lshlrev_b32 v92, 16, v67
v_fmac_f32 v21, s5, v92
v_lshlrev_b32 v92, 16, v72
v_fmac_f32 v26, s5, v92
v_lshlrev_b32 v92, 16, v73
v_fmac_f32 v27, s5, v92
v_lshlrev_b32 v92, 16, v74
v_fmac_f32 v28, s5, v92
v_lshlrev_b32 v92, 16, v75
v_fmac_f32 v29, s5, v92
v_lshlrev_b32 v92, 16, v80
v_fmac_f32 v34, s5, v92
v_lshlrev_b32 v92, 16, v81
v_fmac_f32 v35, s5, v92
v_lshlrev_b32 v92, 16, v82
v_fmac_f32 v36, s5, v92
v_lshlrev_b32 v92, 16, v83
v_fmac_f32 v37, s5, v92
v_lshlrev_b32 v92, 16, v88
v_fmac_f32 v42, s5, v92
v_lshlrev_b32 v92, 16, v89
v_fmac_f32 v43, s5, v92
v_lshlrev_b32 v92, 16, v90
v_fmac_f32 v44, s5, v92
v_lshlrev_b32 v92, 16, v91
v_fmac_f32 v45, s5, v92
label_c_added_22:
label_no_beta_20:
s_cmp_eq_u32 s57, 1
s_cbranch_scc1 label_vector_23
s_add_u32 s69, s55, 0
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s39
s_add_u32 s69, s55, 16
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s39
s_add_u32 s69, s56, 0
v_add_u32 v94, s69, v3
v_mul_lo_u32 v98, v94, s40
s_add_u32 s69, s56, 16
v_add_u32 v95, s69, v3
v_mul_lo_u32 v99, v95, s40
s_add_u32 s69, s56, 32
v_add_u32 v96, s69, v3
v_mul_lo_u32 v100, v96, s40
s_add_u32 s69, s56, 48
v_add_u32 v97, s69, v3
v_mul_lo_u32 v101, v97, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_lshrrev_b32 v14, 16, v14
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v14, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v15, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v16, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v17, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v22, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v23, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v24, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v25, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v30, 16, v30
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v30, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v31, 16, v31
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v31, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v32, 16, v32
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v32, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v33, 16, v33
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v33, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v38, 16, v38
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v38, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v39, 16, v39
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v39, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v40, 16, v40
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v40, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v41, 16, v41
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v41, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v18, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v19, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v20, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v21, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v26, 16, v26
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v26, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v27, 16, v27
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v27, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v28, 16, v28
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v28, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v29, 16, v29
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v29, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v34, 16, v34
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v34, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v35, 16, v35
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v35, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v36, 16, v36
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v36, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v37, 16, v37
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v37, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v42, 16, v42
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v42, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v43, 16, v43
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v43, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v44, 16, v44
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v44, v13, s[24:27], 0 offen offset:0
v_lshrrev_b32 v45, 16, v45
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_short v45, v13, s[24:27], 0 offen offset:0
s_branch label_tile_stored_24
label_vector_23:
s_add_u32 s69, s55, 0
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s39
s_add_u32 s69, s55, 16
v_add_u32 v102, s69, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s39
s_add_u32 s69, s56, 0
v_add_u32 v94, s69, v3
v_mul_lo_u32 v98, v94, s40
s_add_u32 s69, s56, 16
v_add_u32 v95, s69, v3
v_mul_lo_u32 v99, v95, s40
s_add_u32 s69, s56, 32
v_add_u32 v96, s69, v3
v_mul_lo_u32 v100, v96, s40
s_add_u32 s69, s56, 48
v_add_u32 v97, s69, v3
v_mul_lo_u32 v101, v97, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v92, v98
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_lshrrev_b32 v16, 16, v16
v_lshrrev_b32 v17, 16, v17
v_pack_b32_f16 v14, v14, v15
v_pack_b32_f16 v15, v16, v17
buffer_store_dwordx2 v[14:15], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v92, v99
v_lshrrev_b32 v22, 16, v22
v_lshrrev_b32 v23, 16, v23
v_lshrrev_b32 v24, 16, v24
v_lshrrev_b32 v25, 16, v25
v_pack_b32_f16 v22, v22, v23
v_pack_b32_f16 v23, v24, v25
buffer_store_dwordx2 v[22:23], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v92, v100
v_lshrrev_b32 v30, 16, v30
v_lshrrev_b32 v31, 16, v31
v_lshrrev_b32 v32, 16, v32
v_lshrrev_b32 v33, 16, v33
v_pack_b32_f16 v30, v30, v31
v_pack_b32_f16 v31, v32, v33
buffer_store_dwordx2 v[30:31], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v92, v101
v_lshrrev_b32 v38, 16, v38
v_lshrrev_b32 v39, 16, v39
v_lshrrev_b32 v40, 16, v40
v_lshrrev_b32 v41, 16, v41
v_pack_b32_f16 v38, v38, v39
v_pack_b32_f16 v39, v40, v41
buffer_store_dwordx2 v[38:39], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v93, v98
v_lshrrev_b32 v18, 16, v18
v_lshrrev_b32 v19, 16, v19
v_lshrrev_b32 v20, 16, v20
v_lshrrev_b32 v21, 16, v21
v_pack_b32_f16 v18, v18, v19
v_pack_b32_f16 v19, v20, v21
buffer_store_dwordx2 v[18:19], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v93, v99
v_lshrrev_b32 v26, 16, v26
v_lshrrev_b32 v27, 16, v27
v_lshrrev_b32 v28, 16, v28
v_lshrrev_b32 v29, 16, v29
v_pack_b32_f16 v26, v26, v27
v_pack_b32_f16 v27, v28, v29
buffer_store_dwordx2 v[26:27], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v93, v100
v_lshrrev_b32 v34, 16, v34
v_lshrrev_b32 v35, 16, v35
v_lshrrev_b32 v36, 16, v36
v_lshrrev_b32 v37, 16, v37
v_pack_b32_f16 v34, v34, v35
v_pack_b32_f16 v35, v36, v37
buffer_store_dwordx2 v[34:35], v13, s[24:27], 0 offen offset:0
v_add_u32 v13, v93, v101
v_lshrrev_b32 v42, 16, v42
v_lshrrev_b32 v43, 16, v43
v_lshrrev_b32 v44, 16, v44
v_lshrrev_b32 v45, 16, v45
v_pack_b32_f16 v42, v42, v43
v_pack_b32_f16 v43, v44, v45
buffer_store_dwordx2 v[42:43], v13, s[24:27], 0 offen offset:0
label_tile_stored_24:
s_branch label_stored_19
label_workspace_18:
s_mul_i32 s69, s4, s38
s_lshl_b32 s69, s69, 2
s_cmp_eq_u32 s57, 1
s_cbranch_scc1 label_vector_25
s_add_u32 s70, s55, 0
v_add_u32 v102, s70, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s41
v_add_u32 v92, s69, v92
s_add_u32 s70, s55, 16
v_add_u32 v102, s70, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s41
v_add_u32 v93, s69, v93
s_add_u32 s70, s56, 0
v_add_u32 v94, s70, v3
v_mul_lo_u32 v98, v94, s42
s_add_u32 s70, s56, 16
v_add_u32 v95, s70, v3
v_mul_lo_u32 v99, v95, s42
s_add_u32 s70, s56, 32
v_add_u32 v96, s70, v3
v_mul_lo_u32 v100, v96, s42
s_add_u32 s70, s56, 48
v_add_u32 v97, s70, v3
v_mul_lo_u32 v101, v97, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v92, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v14, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v15, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v16, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v17, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v22, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v23, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v24, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v25, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v30, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v31, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v32, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v33, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v38, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v39, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v40, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[58:59]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v41, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_cmp_lt_i32 s[70:71], v94, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v18, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v94, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v19, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v94, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v20, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v94, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v21, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_cmp_lt_i32 s[70:71], v95, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v26, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v95, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v27, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v95, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v28, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v95, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v29, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_cmp_lt_i32 s[70:71], v96, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v34, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v96, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v35, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v96, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v36, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v96, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v37, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_cmp_lt_i32 s[70:71], v97, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v42, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s62, v13
v_cmp_lt_i32 s[70:71], v97, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v43, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s63, v13
v_cmp_lt_i32 s[70:71], v97, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v44, v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
v_add_u32 v13, s64, v13
v_cmp_lt_i32 s[70:71], v97, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v13, -1, v13, s[70:71]
buffer_store_dword v45, v13, s[28:31], 0 offen offset:0
s_branch label_tile_stored_26
label_vector_25:
s_add_u32 s70, s55, 0
v_add_u32 v102, s70, v1
v_cmp_lt_u32 s[58:59], v102, s45
s_nop 1
v_mul_lo_u32 v92, v102, s41
v_add_u32 v92, s69, v92
s_add_u32 s70, s55, 16
v_add_u32 v102, s70, v1
v_cmp_lt_u32 s[60:61], v102, s45
s_nop 1
v_mul_lo_u32 v93, v102, s41
v_add_u32 v93, s69, v93
s_add_u32 s70, s56, 0
v_add_u32 v94, s70, v3
v_mul_lo_u32 v98, v94, s42
s_add_u32 s70, s56, 16
v_add_u32 v95, s70, v3
v_mul_lo_u32 v99, v95, s42
s_add_u32 s70, s56, 32
v_add_u32 v96, s70, v3
v_mul_lo_u32 v100, v96, s42
s_add_u32 s70, s56, 48
v_add_u32 v97, s70, v3
v_mul_lo_u32 v101, v97, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v13, v92, v98
buffer_store_dwordx4 v[14:17], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v99
buffer_store_dwordx4 v[22:25], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v100
buffer_store_dwordx4 v[30:33], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v92, v101
buffer_store_dwordx4 v[38:41], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v98
buffer_store_dwordx4 v[18:21], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v99
buffer_store_dwordx4 v[26:29], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v100
buffer_store_dwordx4 v[34:37], v13, s[28:31], 0 offen offset:0
v_add_u32 v13, v93, v101
buffer_store_dwordx4 v[42:45], v13, s[28:31], 0 offen offset:0
label_tile_stored_26:
label_stored_19:
label_epilogue_end_14:
label_tile_end_12:
s_endpgm
.Lgemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1, .Lgemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end - gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 200
  .amdhsa_next_free_vgpr 200
  .amdhsa_next_free_sgpr 73
  .amdhsa_group_segment_fixed_size 73728
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
    Repository: "TensileLite C++ skinny GEMM emitter (selection sweep)"
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
    macrotile: [32, 4, 4096]
    threads: [64, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  AssertSummationElementMultiple: 256
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 131072, 1: 131072 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  AssertStrideAMultiple: { 1: 4 }
  AssertStrideBMultiple: { 1: 4 }
  StaggerU: 0
  MXScaleFormat: HostPreSwizzle
  WavefrontSize: 64
amdhsa.version: [1, 1]
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.kernels:
  - .name: gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM32_TN4_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 73728
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 79
    .vgpr_count: 200
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
