.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1:
s_load_dword s46, s[0:1], 96
s_load_dword s35, s[0:1], 80
s_load_dword s32, s[0:1], 88
s_load_dword s34, s[0:1], 76
s_load_dword s56, s[0:1], 72
s_load_dword s55, s[0:1], 68
s_load_dword s54, s[0:1], 64
s_load_dword s53, s[0:1], 60
s_load_dword s52, s[0:1], 56
s_load_dwordx2 s[24:25], s[0:1], 40
s_load_dwordx2 s[64:65], s[0:1], 32
s_load_dwordx2 s[62:63], s[0:1], 24
s_load_dwordx2 s[60:61], s[0:1], 16
s_load_dwordx2 s[58:59], s[0:1], 8
s_load_dwordx2 s[20:21], s[0:1], 0
s_load_dword s3, s[0:1], 92
s_load_dword s69, s[0:1], 84
s_load_dwordx2 s[28:29], s[0:1], 48
v_lshrrev_b32 v1, 6, v0
s_nop 0
v_readfirstlane_b32 s51, v1
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s70, s51, 36864
v_and_b32 v11, 7, v5
v_add_u32 v13, 0, v6
v_lshrrev_b32 v12, 1, v13
v_xor_b32 v7, v12, v11
v_lshlrev_b32 v7, 4, v7
v_add_u32 v13, 8, v6
v_lshrrev_b32 v12, 1, v13
v_xor_b32 v8, v12, v11
v_lshlrev_b32 v8, 4, v8
v_and_b32 v13, 15, v5
v_lshrrev_b32 v14, 3, v13
v_lshlrev_b32 v14, 10, v14
v_and_b32 v11, 7, v13
v_lshlrev_b32 v11, 7, v11
v_add_u32 v14, v14, v11
v_lshrrev_b32 v12, 1, v13
v_lshrrev_b32 v11, 4, v5
v_xor_b32 v11, v12, v11
v_lshlrev_b32 v11, 4, v11
v_add_u32 v14, v14, v11
s_add_u32 s71, s70, 0
v_add_u32 v9, s71, v14
v_xor_b32 v10, 64, v9
s_add_u32 s71, s51, -1
s_max_i32 s71, s71, 0
s_lshl_b32 s71, s71, 11
s_add_u32 s71, s71, 73728
v_lshlrev_b32 v11, 4, v5
v_add_u32 v11, s71, v11
s_waitcnt lgkmcnt(0)
s_add_u32 s71, s34, 3
s_lshr_b32 s71, s71, 2
s_lshl_b32 s71, s71, 2
s_add_u32 s57, s55, -1
s_mul_i32 s57, s57, s53
s_add_u32 s57, s57, s71
s_add_u32 s66, s56, -1
s_mul_i32 s66, s66, s54
s_add_u32 s66, s66, s71
s_add_u32 s47, s34, 255
s_lshr_b32 s47, s47, 8
s_lshl_b32 s47, s47, 8
s_add_u32 s67, s55, 31
s_lshr_b32 s67, s67, 5
s_mul_i32 s67, s67, s47
s_add_u32 s68, s56, 31
s_lshr_b32 s68, s68, 5
s_mul_i32 s68, s68, s47
s_add_u32 s22, s56, -1
s_mul_i32 s22, s22, s52
s_lshl_b32 s71, s55, 1
s_add_u32 s22, s22, s71
s_add_u32 s30, s56, -1
s_mul_i32 s30, s30, s69
s_lshl_b32 s71, s55, 1
s_add_u32 s30, s30, s71
s_add_u32 s50, s34, s35
s_add_u32 s50, s50, -1
s_abs_i32 s72, s35
s_abs_i32 s73, s50
s_xor_b32 s74, s50, s35
s_ashr_i32 s74, s74, 31
v_cvt_f32_u32 v5, s72
v_rcp_iflag_f32 v5, v5
s_sub_u32 s75, 0, s72
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s76, v5
s_mul_i32 s77, s75, s76
s_mul_hi_u32 s77, s76, s77
s_add_u32 s76, s76, s77
s_mul_hi_u32 s75, s73, s76
s_mul_i32 s77, s75, s72
s_sub_u32 s77, s73, s77
s_add_u32 s78, s75, 1
s_sub_u32 s79, s77, s72
s_cmp_ge_u32 s77, s72
s_cselect_b32 s75, s78, s75
s_cselect_b32 s77, s79, s77
s_add_u32 s78, s75, 1
s_cmp_ge_u32 s77, s72
s_cselect_b32 s75, s78, s75
s_xor_b32 s75, s75, s74
s_sub_u32 s50, s75, s74
s_mul_i32 s26, s55, s50
s_mul_i32 s26, s26, s56
s_lshl_b32 s26, s26, 2
s_mul_i32 s36, s55, s56
s_lshl_b32 s40, s55, 2
s_cmp_le_u32 s55, s56
s_cselect_b64 s[4:5], s[58:59], s[60:61]
s_cselect_b32 s6, s57, s66
s_cselect_b64 s[8:9], s[60:61], s[58:59]
s_cselect_b32 s10, s66, s57
s_cselect_b64 s[12:13], s[62:63], s[64:65]
s_cselect_b32 s14, s67, s68
s_cselect_b64 s[16:17], s[64:65], s[62:63]
s_cselect_b32 s18, s68, s67
s_cselect_b32 s48, s53, s54
s_cselect_b32 s49, s54, s53
s_cselect_b32 s37, 2, s52
s_cselect_b32 s38, s52, 2
s_cselect_b32 s41, 2, s69
s_cselect_b32 s42, s69, 2
s_cselect_b32 s39, 4, s40
s_cselect_b32 s40, s40, 4
s_cselect_b32 s45, 1, 0
s_min_u32 s43, s55, s56
s_max_u32 s44, s55, s56
s_mov_b32 s7, 131072
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_add_u32 s53, s44, 31
s_lshr_b32 s53, s53, 5
s_add_u32 s55, s43, 15
s_lshr_b32 s55, s55, 4
s_mul_i32 s54, s53, s55
s_mul_i32 s52, s54, s50
s_add_u32 s56, s52, 7
s_lshr_b32 s56, s56, 3
s_add_u32 s58, s52, -1
s_and_b32 s57, 7, s58
s_add_u32 s57, s57, 1
s_lshl_b32 s58, s53, 3
s_cmp_lt_i32 s2, s52
s_cbranch_scc0 label_tile_loop_end_1
label_tile_loop_0:
s_barrier
s_lshr_b32 s62, s2, 3
s_and_b32 s63, 7, s2
s_sub_u32 s64, s63, s57
s_max_i32 s64, s64, 0
s_mul_i32 s61, s63, s56
s_add_u32 s61, s61, s62
s_sub_u32 s61, s61, s64
s_abs_i32 s63, s54
s_abs_i32 s64, s61
s_xor_b32 s65, s61, s54
s_ashr_i32 s65, s65, 31
v_cvt_f32_u32 v5, s63
v_rcp_iflag_f32 v5, v5
s_sub_u32 s66, 0, s63
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s67, v5
s_mul_i32 s68, s66, s67
s_mul_hi_u32 s68, s67, s68
s_add_u32 s67, s67, s68
s_mul_hi_u32 s66, s64, s67
s_mul_i32 s68, s66, s63
s_sub_u32 s68, s64, s68
s_add_u32 s69, s66, 1
s_sub_u32 s71, s68, s63
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_cselect_b32 s68, s71, s68
s_add_u32 s69, s66, 1
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_xor_b32 s66, s66, s65
s_sub_u32 s60, s66, s65
s_mul_i32 s62, s60, s54
s_sub_u32 s62, s61, s62
s_abs_i32 s67, s58
s_abs_i32 s68, s62
s_xor_b32 s69, s62, s58
s_ashr_i32 s69, s69, 31
v_cvt_f32_u32 v5, s67
v_rcp_iflag_f32 v5, v5
s_sub_u32 s71, 0, s67
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s72, v5
s_mul_i32 s73, s71, s72
s_mul_hi_u32 s73, s72, s73
s_add_u32 s72, s72, s73
s_mul_hi_u32 s71, s68, s72
s_mul_i32 s73, s71, s67
s_sub_u32 s73, s68, s73
s_add_u32 s74, s71, 1
s_sub_u32 s75, s73, s67
s_cmp_ge_u32 s73, s67
s_cselect_b32 s71, s74, s71
s_cselect_b32 s73, s75, s73
s_add_u32 s74, s71, 1
s_cmp_ge_u32 s73, s67
s_cselect_b32 s71, s74, s71
s_xor_b32 s71, s71, s69
s_sub_u32 s63, s71, s69
s_mul_i32 s64, s63, s58
s_sub_u32 s64, s62, s64
s_lshl_b32 s65, s63, 3
s_sub_u32 s66, s55, s65
s_min_i32 s66, s66, 8
s_abs_i32 s67, s66
s_abs_i32 s68, s64
s_xor_b32 s69, s64, s66
s_ashr_i32 s69, s69, 31
v_cvt_f32_u32 v5, s67
v_rcp_iflag_f32 v5, v5
s_sub_u32 s71, 0, s67
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s72, v5
s_mul_i32 s73, s71, s72
s_mul_hi_u32 s73, s72, s73
s_add_u32 s72, s72, s73
s_mul_hi_u32 s71, s68, s72
s_mul_i32 s73, s71, s67
s_sub_u32 s73, s68, s73
s_add_u32 s74, s71, 1
s_sub_u32 s75, s73, s67
s_cmp_ge_u32 s73, s67
s_cselect_b32 s71, s74, s71
s_cselect_b32 s73, s75, s73
s_add_u32 s74, s71, 1
s_cmp_ge_u32 s73, s67
s_cselect_b32 s71, s74, s71
s_xor_b32 s71, s71, s69
s_sub_u32 s50, s71, s69
s_mul_i32 s59, s50, s66
s_sub_u32 s59, s64, s59
s_add_u32 s59, s59, s65
s_lshl_b32 s61, s59, 4
s_lshl_b32 s62, s50, 5
s_add_u32 s64, s61, 0
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s43
v_mul_lo_u32 v24, v27, s48
v_add_u32 v24, v24, v7
v_mov_b32 v36, -2147483648
v_cndmask_b32 v24, v36, v24, s[66:67]
s_add_u32 s64, s61, 8
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s43
v_mul_lo_u32 v25, v27, s48
v_add_u32 v25, v25, v8
v_mov_b32 v36, -2147483648
v_cndmask_b32 v25, v36, v25, s[66:67]
s_add_u32 s64, s62, 0
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s44
v_mul_lo_u32 v20, v27, s49
v_add_u32 v20, v20, v7
v_mov_b32 v36, -2147483648
v_cndmask_b32 v20, v36, v20, s[66:67]
s_add_u32 s64, s62, 8
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s44
v_mul_lo_u32 v21, v27, s49
v_add_u32 v21, v21, v8
v_mov_b32 v36, -2147483648
v_cndmask_b32 v21, v36, v21, s[66:67]
s_add_u32 s64, s62, 16
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s44
v_mul_lo_u32 v22, v27, s49
v_add_u32 v22, v22, v7
v_mov_b32 v36, -2147483648
v_cndmask_b32 v22, v36, v22, s[66:67]
s_add_u32 s64, s62, 24
v_add_u32 v27, s64, v6
v_cmp_lt_u32 s[66:67], v27, s44
v_mul_lo_u32 v23, v27, s49
v_add_u32 v23, v23, v8
v_mov_b32 v36, -2147483648
v_cndmask_b32 v23, v36, v23, s[66:67]
s_lshr_b32 s64, s61, 5
s_mul_i32 s64, s64, s47
v_add_u32 v26, s64, v4
s_and_b32 s63, 16, s61
s_lshr_b32 s63, s63, 1
s_add_u32 s64, s62, 0
s_lshr_b32 s65, s64, 5
s_mul_i32 s65, s65, s47
v_add_u32 v5, s65, v4
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
v_mov_b32 v16, 0
v_mov_b32 v17, 0
v_mov_b32 v18, 0
v_mov_b32 v19, 0
s_cmp_eq_u32 s51, 0
s_cbranch_scc0 label_c_prefetch_skip_2
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_c_prefetch_skip_2
s_and_b32 s64, s3, 2147483647
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s65, s61, 0
v_add_u32 v41, s65, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s41
s_add_u32 s65, s62, 0
v_add_u32 v37, s65, v3
v_mul_lo_u32 v39, v37, s42
s_add_u32 s65, s62, 16
v_add_u32 v38, s65, v3
v_mul_lo_u32 v40, v38, s42
s_mov_b32 s71, s42
s_lshl_b32 s72, s42, 1
s_mul_i32 s73, s42, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[68:69], v37, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v28, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v37, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v29, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v37, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v30, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v37, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v31, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[68:69], v38, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v32, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v38, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v33, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v38, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v34, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v38, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v35, v27, s[28:31], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s65, s59, 1
s_lshl_b32 s65, s65, 4
s_add_u32 s66, s50, 1
s_lshl_b32 s66, s66, 5
s_cmp_le_u32 s65, s43
s_cselect_b32 s67, 1, 0
s_cmp_le_u32 s66, s44
s_cselect_b32 s68, 1, 0
s_and_b32 s64, s67, s68
s_add_u32 s65, s61, 0
v_add_u32 v41, s65, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s41
s_add_u32 s65, s62, 0
v_add_u32 v37, s65, v3
v_mul_lo_u32 v39, v37, s42
s_add_u32 s65, s62, 16
v_add_u32 v38, s65, v3
v_mul_lo_u32 v40, v38, s42
s_mov_b32 s71, s42
s_lshl_b32 s72, s42, 1
s_mul_i32 s73, s42, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[68:69], v37, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v28, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v37, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v29, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v37, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v30, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v37, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v31, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[68:69], v38, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v32, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v38, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v33, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v38, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v34, v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v38, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_load_ushort v35, v27, s[28:31], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v27, v36, v39
buffer_load_dwordx2 v[28:29], v27, s[28:31], 0 offen offset:0
v_add_u32 v27, v36, v40
buffer_load_dwordx2 v[30:31], v27, s[28:31], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_mul_i32 s66, s60, s35
s_sub_u32 s67, s34, s66
s_min_i32 s67, s67, s35
s_lshr_b32 s67, s67, 8
s_add_u32 s68, s67, 1
s_sub_u32 s68, s68, s51
s_lshr_b32 s64, s68, 1
s_lshl_b32 s65, s51, 8
s_add_u32 s65, s65, s66
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 1920
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
buffer_load_dword v36, v5, s[16:19], s65 offen offset:0
buffer_load_dword v37, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 14208
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 18304
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
buffer_load_dword v38, v5, s[16:19], s65 offen offset:0
buffer_load_dword v39, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 m0, s70, 24576
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 28672
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 32768
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 25600
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 29696
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 33792
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 26496
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 30592
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 34688
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 27520
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 31616
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 35712
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
buffer_load_dword v40, v5, s[16:19], s65 offen offset:0
buffer_load_dword v41, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_mul_hi_i32 s66, s64, 1431655766
s_lshr_b32 s67, s66, 31
s_add_u32 s66, s66, s67
s_mul_i32 s66, s66, 3
s_add_u32 s66, s66, -3
s_max_i32 s66, s66, 0
s_waitcnt vmcnt(36)
ds_read_b128 v[42:45], v9 offset:0
ds_read_b128 v[46:49], v10 offset:0
ds_read_b128 v[50:53], v9 offset:4096
ds_read_b128 v[54:57], v10 offset:4096
ds_read_b128 v[58:61], v9 offset:8192
ds_read_b128 v[62:65], v10 offset:8192
s_cmp_gt_i32 s66, 0
s_mov_b32 s67, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
buffer_load_dword v36, v5, s[16:19], s65 offen offset:0
buffer_load_dword v37, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 14208
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:24576
ds_read_b128 v[46:49], v10 offset:24576
ds_read_b128 v[50:53], v9 offset:28672
ds_read_b128 v[54:57], v10 offset:28672
ds_read_b128 v[58:61], v9 offset:32768
ds_read_b128 v[62:65], v10 offset:32768
buffer_load_dword v38, v5, s[16:19], s65 offen offset:0
buffer_load_dword v39, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 24576
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 28672
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 32768
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 25600
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 29696
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 33792
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v41, s63, v41
ds_read_b128 v[66:69], v9 offset:26624
ds_read_b128 v[70:73], v10 offset:26624
ds_read_b128 v[74:77], v9 offset:30720
ds_read_b128 v[78:81], v10 offset:30720
ds_read_b128 v[82:85], v9 offset:34816
ds_read_b128 v[86:89], v10 offset:34816
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 30592
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 34688
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 27520
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 31616
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 35712
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:0
ds_read_b128 v[46:49], v10 offset:0
ds_read_b128 v[50:53], v9 offset:4096
ds_read_b128 v[54:57], v10 offset:4096
ds_read_b128 v[58:61], v9 offset:8192
ds_read_b128 v[62:65], v10 offset:8192
buffer_load_dword v40, v5, s[16:19], s65 offen offset:0
buffer_load_dword v41, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 s67, s67, 3
s_cmp_lt_i32 s67, s66
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s67, s64, s66
s_cmp_eq_i32 s67, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(28)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s67, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(28)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
s_waitcnt vmcnt(14)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s67, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(28)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
s_waitcnt vmcnt(14)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[42:45], v9 offset:24576
ds_read_b128 v[46:49], v10 offset:24576
ds_read_b128 v[50:53], v9 offset:28672
ds_read_b128 v[54:57], v10 offset:28672
ds_read_b128 v[58:61], v9 offset:32768
ds_read_b128 v[62:65], v10 offset:32768
s_waitcnt vmcnt(0)
v_lshrrev_b32 v41, s63, v41
ds_read_b128 v[66:69], v9 offset:26624
ds_read_b128 v[70:73], v10 offset:26624
ds_read_b128 v[74:77], v9 offset:30720
ds_read_b128 v[78:81], v10 offset:30720
ds_read_b128 v[82:85], v9 offset:34816
ds_read_b128 v[86:89], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s67, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
buffer_load_dword v36, v5, s[16:19], s65 offen offset:0
buffer_load_dword v37, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(28)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[42:45], v9 offset:24576
ds_read_b128 v[46:49], v10 offset:24576
ds_read_b128 v[50:53], v9 offset:28672
ds_read_b128 v[54:57], v10 offset:28672
ds_read_b128 v[58:61], v9 offset:32768
ds_read_b128 v[62:65], v10 offset:32768
s_waitcnt vmcnt(14)
v_lshrrev_b32 v41, s63, v41
ds_read_b128 v[66:69], v9 offset:26624
ds_read_b128 v[70:73], v10 offset:26624
ds_read_b128 v[74:77], v9 offset:30720
ds_read_b128 v[78:81], v10 offset:30720
ds_read_b128 v[82:85], v9 offset:34816
ds_read_b128 v[86:89], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[42:45], v9 offset:0
ds_read_b128 v[46:49], v10 offset:0
ds_read_b128 v[50:53], v9 offset:4096
ds_read_b128 v[54:57], v10 offset:4096
ds_read_b128 v[58:61], v9 offset:8192
ds_read_b128 v[62:65], v10 offset:8192
s_waitcnt vmcnt(0)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s67, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
buffer_load_dword v36, v5, s[16:19], s65 offen offset:0
buffer_load_dword v37, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s70, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 14208
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s70, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[42:45], v9 offset:24576
ds_read_b128 v[46:49], v10 offset:24576
ds_read_b128 v[50:53], v9 offset:28672
ds_read_b128 v[54:57], v10 offset:28672
ds_read_b128 v[58:61], v9 offset:32768
ds_read_b128 v[62:65], v10 offset:32768
buffer_load_dword v38, v5, s[16:19], s65 offen offset:0
buffer_load_dword v39, v26, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(28)
v_lshrrev_b32 v41, s63, v41
ds_read_b128 v[66:69], v9 offset:26624
ds_read_b128 v[70:73], v10 offset:26624
ds_read_b128 v[74:77], v9 offset:30720
ds_read_b128 v[78:81], v10 offset:30720
ds_read_b128 v[82:85], v9 offset:34816
ds_read_b128 v[86:89], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v40, v41 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v40, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[42:45], v9 offset:0
ds_read_b128 v[46:49], v10 offset:0
ds_read_b128 v[50:53], v9 offset:4096
ds_read_b128 v[54:57], v10 offset:4096
ds_read_b128 v[58:61], v9 offset:8192
ds_read_b128 v[62:65], v10 offset:8192
s_waitcnt vmcnt(14)
v_lshrrev_b32 v37, s63, v37
ds_read_b128 v[66:69], v9 offset:2048
ds_read_b128 v[70:73], v10 offset:2048
ds_read_b128 v[74:77], v9 offset:6144
ds_read_b128 v[78:81], v10 offset:6144
ds_read_b128 v[82:85], v9 offset:10240
ds_read_b128 v[86:89], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
ds_read_b128 v[50:53], v9 offset:16384
ds_read_b128 v[54:57], v10 offset:16384
ds_read_b128 v[58:61], v9 offset:20480
ds_read_b128 v[62:65], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v39, s63, v39
ds_read_b128 v[66:69], v9 offset:14336
ds_read_b128 v[70:73], v10 offset:14336
ds_read_b128 v[74:77], v9 offset:18432
ds_read_b128 v[78:81], v10 offset:18432
ds_read_b128 v[82:85], v9 offset:22528
ds_read_b128 v[86:89], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[42:49], v[58:65], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[50:57], v[58:65], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:73], v[82:89], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:81], v[82:89], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_waitcnt 0
s_nop 11
s_cmp_eq_u32 s51, 0
s_cbranch_scc1 label_wave_zero_15
ds_write_b128 v11, v[12:15] offset:0
ds_write_b128 v11, v[16:19] offset:1024
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_tile_end_14
label_wave_zero_15:
s_barrier
ds_read_b128 v[36:39], v11 offset:0
ds_read_b128 v[40:43], v11 offset:1024
s_waitcnt lgkmcnt(0)
v_add_f32 v12, v12, v36
v_add_f32 v13, v13, v37
v_add_f32 v14, v14, v38
v_add_f32 v15, v15, v39
v_add_f32 v16, v16, v40
v_add_f32 v17, v17, v41
v_add_f32 v18, v18, v42
v_add_f32 v19, v19, v43
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_16
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_18
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[32:33], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[32:33], v[18:19] op_sel_hi:[0,1]
s_and_b32 s64, s3, 2147483647
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_no_beta_20
s_waitcnt vmcnt(0)
v_lshlrev_b32 v36, 16, v28
v_fmac_f32 v12, s3, v36
v_lshlrev_b32 v36, 16, v29
v_fmac_f32 v13, s3, v36
v_lshlrev_b32 v36, 16, v30
v_fmac_f32 v14, s3, v36
v_lshlrev_b32 v36, 16, v31
v_fmac_f32 v15, s3, v36
v_lshlrev_b32 v36, 16, v32
v_fmac_f32 v16, s3, v36
v_lshlrev_b32 v36, 16, v33
v_fmac_f32 v17, s3, v36
v_lshlrev_b32 v36, 16, v34
v_fmac_f32 v18, s3, v36
v_lshlrev_b32 v36, 16, v35
v_fmac_f32 v19, s3, v36
label_no_beta_20:
s_add_u32 s69, s61, 0
v_add_u32 v41, s69, v1
v_cmp_lt_u32 s[64:65], v41, s43
v_mul_lo_u32 v36, v41, s37
s_add_u32 s69, s62, 0
v_add_u32 v37, s69, v3
v_mul_lo_u32 v39, v37, s38
s_add_u32 s69, s62, 16
v_add_u32 v38, s69, v3
v_mul_lo_u32 v40, v38, s38
s_mov_b32 s66, s38
s_lshl_b32 s67, s38, 1
s_mul_i32 s68, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[76:77], v37, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v12, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v27, v36, v39
v_add_u32 v27, s66, v27
v_cmp_lt_i32 s[76:77], v37, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v13, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v27, v36, v39
v_add_u32 v27, s67, v27
v_cmp_lt_i32 s[76:77], v37, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v14, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v27, v36, v39
v_add_u32 v27, s68, v27
v_cmp_lt_i32 s[76:77], v37, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v15, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[76:77], v38, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v16, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v27, v36, v40
v_add_u32 v27, s66, v27
v_cmp_lt_i32 s[76:77], v38, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v17, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v27, v36, v40
v_add_u32 v27, s67, v27
v_cmp_lt_i32 s[76:77], v38, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v18, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v27, v36, v40
v_add_u32 v27, s68, v27
v_cmp_lt_i32 s[76:77], v38, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_short v19, v27, s[20:23], 0 offen offset:0
s_branch label_stored_19
label_workspace_18:
s_mul_i32 s69, s60, s36
s_lshl_b32 s69, s69, 2
s_add_u32 s75, s61, 0
v_add_u32 v41, s75, v1
v_cmp_lt_u32 s[64:65], v41, s43
v_mul_lo_u32 v36, v41, s39
v_add_u32 v36, s69, v36
s_add_u32 s75, s62, 0
v_add_u32 v37, s75, v3
v_mul_lo_u32 v39, v37, s40
s_add_u32 s75, s62, 16
v_add_u32 v38, s75, v3
v_mul_lo_u32 v40, v38, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[76:77], v37, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v12, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s66, v27
v_cmp_lt_i32 s[76:77], v37, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v13, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s67, v27
v_cmp_lt_i32 s[76:77], v37, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v14, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s68, v27
v_cmp_lt_i32 s[76:77], v37, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v15, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[76:77], v38, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v16, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s66, v27
v_cmp_lt_i32 s[76:77], v38, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v17, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s67, v27
v_cmp_lt_i32 s[76:77], v38, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v18, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s68, v27
v_cmp_lt_i32 s[76:77], v38, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v27, -1, v27, s[76:77]
buffer_store_dword v19, v27, s[24:27], 0 offen offset:0
label_stored_19:
s_branch label_epilogue_end_17
label_orientation_b_16:
s_add_u32 s65, s59, 1
s_lshl_b32 s65, s65, 4
s_add_u32 s66, s50, 1
s_lshl_b32 s66, s66, 5
s_cmp_le_u32 s65, s43
s_cselect_b32 s67, 1, 0
s_cmp_le_u32 s66, s44
s_cselect_b32 s68, 1, 0
s_and_b32 s64, s67, s68
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_21
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[32:33], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[32:33], v[18:19] op_sel_hi:[0,1]
s_and_b32 s65, s3, 2147483647
s_cmp_eq_u32 s65, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s64, 1
s_cbranch_scc0 label_c_masked_24
v_lshlrev_b32 v36, 16, v28
v_fmac_f32 v12, s3, v36
v_and_b32 v36, -65536, v28
v_fmac_f32 v13, s3, v36
v_lshlrev_b32 v36, 16, v29
v_fmac_f32 v14, s3, v36
v_and_b32 v36, -65536, v29
v_fmac_f32 v15, s3, v36
v_lshlrev_b32 v36, 16, v30
v_fmac_f32 v16, s3, v36
v_and_b32 v36, -65536, v30
v_fmac_f32 v17, s3, v36
v_lshlrev_b32 v36, 16, v31
v_fmac_f32 v18, s3, v36
v_and_b32 v36, -65536, v31
v_fmac_f32 v19, s3, v36
s_branch label_c_added_25
label_c_masked_24:
v_lshlrev_b32 v36, 16, v28
v_fmac_f32 v12, s3, v36
v_lshlrev_b32 v36, 16, v29
v_fmac_f32 v13, s3, v36
v_lshlrev_b32 v36, 16, v30
v_fmac_f32 v14, s3, v36
v_lshlrev_b32 v36, 16, v31
v_fmac_f32 v15, s3, v36
v_lshlrev_b32 v36, 16, v32
v_fmac_f32 v16, s3, v36
v_lshlrev_b32 v36, 16, v33
v_fmac_f32 v17, s3, v36
v_lshlrev_b32 v36, 16, v34
v_fmac_f32 v18, s3, v36
v_lshlrev_b32 v36, 16, v35
v_fmac_f32 v19, s3, v36
label_c_added_25:
label_no_beta_23:
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_vector_26
s_add_u32 s65, s61, 0
v_add_u32 v41, s65, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s37
s_add_u32 s65, s62, 0
v_add_u32 v37, s65, v3
v_mul_lo_u32 v39, v37, s38
s_add_u32 s65, s62, 16
v_add_u32 v38, s65, v3
v_mul_lo_u32 v40, v38, s38
s_mov_b32 s71, s38
s_lshl_b32 s72, s38, 1
s_mul_i32 s73, s38, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[68:69], v37, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v12, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v27, v36, v39
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v37, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v13, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v27, v36, v39
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v37, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v14, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v27, v36, v39
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v37, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v15, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[68:69], v38, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v16, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v27, v36, v40
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v38, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v17, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v27, v36, v40
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v38, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v18, v27, s[20:23], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v27, v36, v40
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v38, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_short v19, v27, s[20:23], 0 offen offset:0
s_branch label_tile_stored_27
label_vector_26:
s_add_u32 s65, s61, 0
v_add_u32 v41, s65, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s37
s_add_u32 s65, s62, 0
v_add_u32 v37, s65, v3
v_mul_lo_u32 v39, v37, s38
s_add_u32 s65, s62, 16
v_add_u32 v38, s65, v3
v_mul_lo_u32 v40, v38, s38
s_mov_b32 s71, s38
s_lshl_b32 s72, s38, 1
s_mul_i32 s73, s38, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
v_add_u32 v27, v36, v39
v_lshrrev_b32 v12, 16, v12
v_lshrrev_b32 v13, 16, v13
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_pack_b32_f16 v12, v12, v13
v_pack_b32_f16 v13, v14, v15
buffer_store_dwordx2 v[12:13], v27, s[20:23], 0 offen offset:0
v_add_u32 v27, v36, v40
v_lshrrev_b32 v16, 16, v16
v_lshrrev_b32 v17, 16, v17
v_lshrrev_b32 v18, 16, v18
v_lshrrev_b32 v19, 16, v19
v_pack_b32_f16 v16, v16, v17
v_pack_b32_f16 v17, v18, v19
buffer_store_dwordx2 v[16:17], v27, s[20:23], 0 offen offset:0
label_tile_stored_27:
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s65, s60, s36
s_lshl_b32 s65, s65, 2
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_vector_28
s_add_u32 s68, s61, 0
v_add_u32 v41, s68, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s39
v_add_u32 v36, s65, v36
s_add_u32 s68, s62, 0
v_add_u32 v37, s68, v3
v_mul_lo_u32 v39, v37, s40
s_add_u32 s68, s62, 16
v_add_u32 v38, s68, v3
v_mul_lo_u32 v40, v38, s40
s_mov_b32 s71, s40
s_lshl_b32 s72, s40, 1
s_mul_i32 s73, s40, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
v_add_u32 v27, v36, v39
v_cmp_lt_i32 s[68:69], v37, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v12, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v37, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v13, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v37, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v14, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v39
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v37, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v15, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_cmp_lt_i32 s[68:69], v38, s74
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v16, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s71, v27
v_cmp_lt_i32 s[68:69], v38, s75
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v17, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s72, v27
v_cmp_lt_i32 s[68:69], v38, s76
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v18, v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
v_add_u32 v27, s73, v27
v_cmp_lt_i32 s[68:69], v38, s77
s_and_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32 v27, -1, v27, s[68:69]
buffer_store_dword v19, v27, s[24:27], 0 offen offset:0
s_branch label_tile_stored_29
label_vector_28:
s_add_u32 s68, s61, 0
v_add_u32 v41, s68, v1
v_cmp_lt_u32 s[66:67], v41, s43
v_mul_lo_u32 v36, v41, s39
v_add_u32 v36, s65, v36
s_add_u32 s68, s62, 0
v_add_u32 v37, s68, v3
v_mul_lo_u32 v39, v37, s40
s_add_u32 s68, s62, 16
v_add_u32 v38, s68, v3
v_mul_lo_u32 v40, v38, s40
s_mov_b32 s71, s40
s_lshl_b32 s72, s40, 1
s_mul_i32 s73, s40, 3
s_sub_u32 s74, s44, 0
s_sub_u32 s75, s44, 1
s_sub_u32 s76, s44, 2
s_sub_u32 s77, s44, 3
v_add_u32 v27, v36, v39
buffer_store_dwordx4 v[12:15], v27, s[24:27], 0 offen offset:0
v_add_u32 v27, v36, v40
buffer_store_dwordx4 v[16:19], v27, s[24:27], 0 offen offset:0
label_tile_stored_29:
label_stored_22:
label_epilogue_end_17:
label_tile_end_14:
s_add_u32 s2, s2, s46
s_cmp_lt_i32 s2, s52
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 96
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr 80
  .amdhsa_group_segment_fixed_size 75776
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
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
    macrotile: [32, 16, 4096]
    threads: [128, 1, 1]
    grid: [TilesWideSkinnyBatch, One, One]
    kStep: 256
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 256
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 1073741825 }
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
  - .name: gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 75776
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .sgpr_count: 86
    .vgpr_count: 96
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
