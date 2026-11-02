.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1:
s_load_dword s61, s[0:1], 104
s_load_dword s48, s[0:1], 96
s_load_dword s49, s[0:1], 100
s_load_dword s53, s[0:1], 108
s_load_dword s54, s[0:1], 112
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s60, s[0:1], 72
s_load_dword s59, s[0:1], 68
s_load_dword s58, s[0:1], 64
s_load_dword s57, s[0:1], 60
s_load_dword s56, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[68:69], s[0:1], 32
s_load_dwordx2 s[66:67], s[0:1], 24
s_load_dwordx2 s[64:65], s[0:1], 16
s_load_dwordx2 s[62:63], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s74, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
s_mov_b32 s55, 0
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s75, s55, 49152
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
s_add_u32 s76, s75, 0
v_add_u32 v9, s76, v14
v_xor_b32 v10, 64, v9
s_waitcnt lgkmcnt(0)
s_add_u32 s76, s36, 3
s_lshr_b32 s76, s76, 2
s_lshl_b32 s76, s76, 2
s_add_u32 s70, s59, -1
s_mul_i32 s70, s70, s57
s_add_u32 s70, s70, s76
s_add_u32 s71, s60, -1
s_mul_i32 s71, s71, s58
s_add_u32 s71, s71, s76
s_add_u32 s50, s36, 255
s_lshr_b32 s50, s50, 8
s_lshl_b32 s50, s50, 8
s_add_u32 s72, s59, 31
s_lshr_b32 s72, s72, 5
s_mul_i32 s72, s72, s50
s_add_u32 s73, s60, 31
s_lshr_b32 s73, s73, 5
s_mul_i32 s73, s73, s50
s_add_u32 s26, s60, -1
s_mul_i32 s26, s26, s56
s_lshl_b32 s76, s59, 1
s_add_u32 s26, s26, s76
s_add_u32 s34, s60, -1
s_mul_i32 s34, s34, s74
s_lshl_b32 s76, s59, 1
s_add_u32 s34, s34, s76
s_mul_i32 s30, s59, s61
s_mul_i32 s30, s30, s60
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s59, s60
s_lshl_b32 s42, s59, 2
s_cmp_le_u32 s59, s60
s_cselect_b64 s[8:9], s[62:63], s[64:65]
s_cselect_b32 s10, s70, s71
s_cselect_b64 s[12:13], s[64:65], s[62:63]
s_cselect_b32 s14, s71, s70
s_cselect_b64 s[16:17], s[66:67], s[68:69]
s_cselect_b32 s18, s72, s73
s_cselect_b64 s[20:21], s[68:69], s[66:67]
s_cselect_b32 s22, s73, s72
s_cselect_b32 s51, s57, s58
s_cselect_b32 s52, s58, s57
s_cselect_b32 s39, 2, s56
s_cselect_b32 s40, s56, 2
s_cselect_b32 s43, 2, s74
s_cselect_b32 s44, s74, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s59, s60
s_max_u32 s46, s59, s60
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
s_mov_b32 s58, 1
s_mov_b32 s59, 0
s_cmp_lt_u32 s54, 2
s_cbranch_scc1 label_tile_decoded_0
s_mov_b32 s61, s2
s_cmp_lt_u32 s2, s53
s_cbranch_scc1 label_tile_known_1
s_mov_b32 s58, 0
s_sub_u32 s61, s2, s53
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
s_sub_u32 s70, s68, s63
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_cselect_b32 s68, s70, s68
s_add_u32 s69, s66, 1
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_xor_b32 s66, s66, s65
s_sub_u32 s62, s66, s65
s_mul_i32 s59, s62, s54
s_sub_u32 s59, s61, s59
s_add_u32 s61, s53, s62
label_tile_known_1:
s_add_u32 s48, s46, 15
s_lshr_b32 s48, s48, 4
s_abs_i32 s63, s48
s_abs_i32 s64, s61
s_xor_b32 s65, s61, s48
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
s_sub_u32 s70, s68, s63
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_cselect_b32 s68, s70, s68
s_add_u32 s69, s66, 1
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_xor_b32 s66, s66, s65
s_sub_u32 s3, s66, s65
s_mul_i32 s2, s3, s48
s_sub_u32 s2, s61, s2
s_mov_b32 s4, 0
label_tile_decoded_0:
s_lshr_b32 s61, s2, 3
s_and_b32 s61, 1, s61
s_and_b32 s62, 7, s2
s_lshl_b32 s62, s62, 1
s_add_u32 s61, s61, s62
s_and_b32 s62, -16, s2
s_add_u32 s61, s61, s62
s_and_b32 s62, -16, s48
s_cmp_lt_u32 s2, s62
s_cselect_b32 s2, s61, s2
s_cmp_eq_u32 s58, 0
s_cbranch_scc1 label_piece_window_2
s_mov_b32 s56, 0
s_mov_b32 s57, s46
s_mov_b32 s60, 2
s_branch label_window_done_3
label_piece_window_2:
s_mov_b32 s61, 16
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
s_sub_u32 s70, s68, s63
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_cselect_b32 s68, s70, s68
s_add_u32 s69, s66, 1
s_cmp_ge_u32 s68, s63
s_cselect_b32 s66, s69, s66
s_xor_b32 s66, s66, s65
s_sub_u32 s61, s66, s65
s_mul_i32 s56, s59, s61
s_lshl_b32 s62, s2, 4
s_add_u32 s56, s56, s62
s_add_u32 s57, s56, s61
s_min_i32 s57, s57, s46
s_sub_u32 s57, s57, s56
s_cmp_gt_i32 s57, 0
s_cbranch_scc1 label_piece_live_4
s_endpgm
label_piece_live_4:
s_lshr_b32 s60, s61, 3
s_max_i32 s60, s60, 1
label_window_done_3:
s_lshl_b32 s53, s3, 4
s_lshl_b32 s54, s2, 4
s_sub_u32 s63, s56, s54
s_lshr_b32 s63, s63, 3
s_cmp_lg_u32 s58, 0
s_cselect_b32 s63, 0, s63
s_lshr_b32 s64, s63, 1
s_lshl_b32 s64, s64, 12
s_and_b32 s65, 1, s63
s_lshl_b32 s65, s65, 10
s_add_u32 s64, s64, s65
s_add_u32 s62, s75, s64
s_add_u32 s64, s53, 0
v_add_u32 v24, s64, v6
v_cmp_lt_u32 s[66:67], v24, s45
v_mul_lo_u32 v18, v24, s51
v_add_u32 v18, v18, v7
v_mov_b32 v25, -2147483648
v_cndmask_b32 v18, v25, v18, s[66:67]
s_add_u32 s64, s53, 8
v_add_u32 v24, s64, v6
v_cmp_lt_u32 s[66:67], v24, s45
v_mul_lo_u32 v19, v24, s51
v_add_u32 v19, v19, v8
v_mov_b32 v25, -2147483648
v_cndmask_b32 v19, v25, v19, s[66:67]
s_lshl_b32 s64, s63, 3
s_add_u32 s64, s64, s54
s_and_b32 s65, 1, s63
s_lshl_b32 s65, s65, 6
v_xor_b32 v24, s65, v7
v_xor_b32 v25, s65, v8
s_add_u32 s66, s64, 0
v_add_u32 v26, s66, v6
v_sub_u32 v27, v26, s56
v_cmp_lt_u32 s[68:69], v27, s57
v_mul_lo_u32 v16, v26, s52
v_add_u32 v16, v16, v24
v_mov_b32 v27, -2147483648
v_cndmask_b32 v16, v27, v16, s[68:69]
s_add_u32 s66, s64, 8
v_add_u32 v26, s66, v6
v_sub_u32 v27, v26, s56
v_cmp_lt_u32 s[68:69], v27, s57
v_mul_lo_u32 v17, v26, s52
v_add_u32 v17, v17, v25
v_mov_b32 v27, -2147483648
v_cndmask_b32 v17, v27, v17, s[68:69]
s_lshr_b32 s64, s53, 5
s_mul_i32 s64, s64, s50
v_add_u32 v11, s64, v4
s_and_b32 s61, 16, s53
s_lshr_b32 s61, s61, 1
s_add_u32 s64, s54, 0
s_lshr_b32 s65, s64, 5
s_mul_i32 s65, s65, s50
v_add_u32 v5, s65, v4
s_and_b32 s59, 16, s64
s_lshr_b32 s59, s59, 1
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
s_cmp_eq_u32 s55, 0
s_cbranch_scc0 label_c_prefetch_skip_5
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_5
s_and_b32 s63, s5, 2147483647
s_cmp_eq_u32 s63, 0
s_cbranch_scc1 label_c_prefetch_skip_5
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_6
s_add_u32 s73, s53, 0
v_add_u32 v28, s73, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s43
s_add_u32 s73, s54, 0
v_add_u32 v26, s73, v3
v_mul_lo_u32 v27, v26, s44
s_mov_b32 s66, s44
s_lshl_b32 s67, s44, 1
s_mul_i32 s68, s44, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v20, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v21, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v22, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v23, v24, s[32:35], 0 offen offset:0
s_branch label_epilogue_end_7
label_orientation_b_6:
s_add_u32 s64, s3, 1
s_lshl_b32 s64, s64, 4
s_add_u32 s65, s2, 1
s_lshl_b32 s65, s65, 4
s_cmp_le_u32 s64, s45
s_cselect_b32 s66, 1, 0
s_cmp_le_u32 s65, s46
s_cselect_b32 s67, 1, 0
s_and_b32 s63, s66, s67
s_and_b32 s63, s63, s58
s_add_u32 s73, s53, 0
v_add_u32 v28, s73, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s43
s_add_u32 s73, s54, 0
v_add_u32 v26, s73, v3
v_mul_lo_u32 v27, v26, s44
s_mov_b32 s66, s44
s_lshl_b32 s67, s44, 1
s_mul_i32 s68, s44, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
s_cmp_eq_u32 s63, 1
s_cbranch_scc1 label_c_vector_8
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v20, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v21, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v22, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_load_ushort v23, v24, s[32:35], 0 offen offset:0
s_branch label_c_loaded_9
label_c_vector_8:
v_add_u32 v24, v25, v27
buffer_load_dwordx2 v[20:21], v24, s[32:35], 0 offen offset:0
label_c_loaded_9:
label_epilogue_end_7:
label_c_prefetch_skip_5:
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_units_variant_1_11
s_mul_i32 s66, s4, s37
s_sub_u32 s67, s36, s66
s_min_i32 s67, s67, s37
s_and_b32 s64, s67, 255
s_lshr_b32 s67, s67, 8
s_add_u32 s68, s67, 0
s_sub_u32 s68, s68, s55
s_mov_b32 s60, s68
s_lshl_b32 s63, s55, 8
s_add_u32 s63, s63, s66
s_lshl_b32 s65, s67, 8
s_add_u32 s65, s65, s66
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 34688
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 41984
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 42880
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 43904
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v34, v5, s[20:23], s63 offen offset:0
buffer_load_dword v35, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_mul_hi_i32 s66, s60, 715827883
s_lshr_b32 s67, s66, 31
s_add_u32 s66, s66, s67
s_mul_i32 s66, s66, 6
s_add_u32 s66, s66, -6
s_max_i32 s66, s66, 0
s_waitcnt vmcnt(56)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_cmp_gt_i32 s66, 0
s_mov_b32 s67, 0
s_cbranch_scc0 label_kloop_end_13
.p2align 4
label_kloop_12:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 41984
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 43904
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
buffer_load_dword v34, v5, s[20:23], s63 offen offset:0
buffer_load_dword v35, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 s67, s67, 6
s_cmp_lt_i32 s67, s66
s_cbranch_scc1 label_kloop_12
label_kloop_end_13:
s_sub_u32 s67, s60, s66
s_cmp_eq_i32 s67, 1
s_cbranch_scc0 label_arm_end_14
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_14:
s_cmp_eq_i32 s67, 2
s_cbranch_scc0 label_arm_end_15
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_15:
s_cmp_eq_i32 s67, 3
s_cbranch_scc0 label_arm_end_16
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(30)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_16:
s_cmp_eq_i32 s67, 4
s_cbranch_scc0 label_arm_end_17
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(30)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_17:
s_cmp_eq_i32 s67, 5
s_cbranch_scc0 label_arm_end_18
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(30)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(10)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_18:
s_cmp_eq_i32 s67, 6
s_cbranch_scc0 label_arm_end_19
s_waitcnt vmcnt(50)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(30)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(10)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(0)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_19:
s_cmp_eq_i32 s67, 7
s_cbranch_scc0 label_arm_end_20
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(50)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(40)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(30)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(20)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(10)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(0)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_20:
s_cmp_eq_i32 s67, 8
s_cbranch_scc0 label_arm_end_21
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(50)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(40)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(30)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(20)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(10)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_21:
s_cmp_eq_i32 s67, 9
s_cbranch_scc0 label_arm_end_22
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(50)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(40)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(30)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(20)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(10)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_22:
s_cmp_eq_i32 s67, 10
s_cbranch_scc0 label_arm_end_23
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(50)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(40)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(30)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(10)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(0)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_23:
s_cmp_eq_i32 s67, 11
s_cbranch_scc0 label_arm_end_24
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(54)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s62, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(50)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(30)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(20)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(10)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(0)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_24:
s_waitcnt 0
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_tail_skip_25
label_tail_begin_26:
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s65 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s62, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s65 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s62, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s65 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s62, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s65 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s65 offen offset:128 lds
buffer_load_dword v24, v5, s[20:23], s65 offen offset:0
buffer_load_dword v25, v11, s[16:19], s65 offen offset:0
s_waitcnt vmcnt(0)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(0)
s_mov_b32 s66, 0
s_cmp_ge_i32 s64, 64
s_cbranch_scc0 label_tail_edge_27
s_branch label_tail_chunk_end_29
label_tail_edge_27:
s_cmp_gt_i32 s64, 0
s_cbranch_scc0 label_tail_zero_28
s_or_b32 s66, s66, 1
s_branch label_tail_chunk_end_29
label_tail_zero_28:
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
label_tail_chunk_end_29:
s_sub_u32 s67, s64, 64
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_30
s_branch label_tail_chunk_end_32
label_tail_edge_30:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_31
s_or_b32 s66, s66, 2
s_branch label_tail_chunk_end_32
label_tail_zero_31:
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
label_tail_chunk_end_32:
s_sub_u32 s67, s64, 128
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_33
s_branch label_tail_chunk_end_35
label_tail_edge_33:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_34
s_or_b32 s66, s66, 4
s_branch label_tail_chunk_end_35
label_tail_zero_34:
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
label_tail_chunk_end_35:
s_sub_u32 s67, s64, 192
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_36
s_branch label_tail_chunk_end_38
label_tail_edge_36:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_37
s_or_b32 s66, s66, 8
s_branch label_tail_chunk_end_38
label_tail_zero_37:
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v66, 0
v_mov_b32 v67, 0
label_tail_chunk_end_38:
s_bitcmp1_b32 s66, 0
s_cbranch_scc0 label_tail_mask_skip_39
v_sub_u32 v68, s64, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v36, v36, v70
v_and_b32 v44, v44, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v37, v37, v70
v_and_b32 v45, v45, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v38, v38, v70
v_and_b32 v46, v46, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v39, v39, v70
v_and_b32 v47, v47, v70
label_tail_mask_skip_39:
s_sub_u32 s67, s64, 64
s_bitcmp1_b32 s66, 1
s_cbranch_scc0 label_tail_mask_skip_40
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v40, v40, v70
v_and_b32 v48, v48, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v41, v41, v70
v_and_b32 v49, v49, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v42, v42, v70
v_and_b32 v50, v50, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v43, v43, v70
v_and_b32 v51, v51, v70
label_tail_mask_skip_40:
s_sub_u32 s67, s64, 128
s_bitcmp1_b32 s66, 2
s_cbranch_scc0 label_tail_mask_skip_41
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v52, v52, v70
v_and_b32 v60, v60, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v53, v53, v70
v_and_b32 v61, v61, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v54, v54, v70
v_and_b32 v62, v62, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v55, v55, v70
v_and_b32 v63, v63, v70
label_tail_mask_skip_41:
s_sub_u32 s67, s64, 192
s_bitcmp1_b32 s66, 3
s_cbranch_scc0 label_tail_mask_skip_42
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v56, v56, v70
v_and_b32 v64, v64, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v57, v57, v70
v_and_b32 v65, v65, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v58, v58, v70
v_and_b32 v66, v66, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v59, v59, v70
v_and_b32 v67, v67, v70
label_tail_mask_skip_42:
v_sub_u32 v68, s64, v2
v_sub_u32 v68, v68, v2
v_cmp_gt_i32 s[66:67], v68, 0
v_mov_b32 v71, 65535
s_nop 0
v_cndmask_b32 v69, 0, v71, s[66:67]
v_add_u32 v68, -128, v68
v_cmp_gt_i32 s[66:67], v68, 0
v_mov_b32 v71, -65536
s_nop 0
v_cndmask_b32 v68, 0, v71, s[66:67]
v_or_b32 v69, v69, v68
v_and_b32 v70, 2139062143, v69
v_sub_u32 v70, 2139062143, v70
v_and_b32 v24, v24, v69
v_or_b32 v24, v24, v70
v_and_b32 v25, v25, v69
v_or_b32 v25, v25, v70
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_tail_skip_25:
s_branch label_units_variants_end_10
label_units_variant_1_11:
s_mul_i32 s66, s4, s37
s_sub_u32 s67, s36, s66
s_min_i32 s67, s67, s37
s_and_b32 s64, s67, 255
s_lshr_b32 s67, s67, 8
s_add_u32 s68, s67, 0
s_sub_u32 s68, s68, s55
s_mov_b32 s60, s68
s_lshl_b32 s63, s55, 8
s_add_u32 s63, s63, s66
s_lshl_b32 s65, s67, 8
s_add_u32 s65, s65, s66
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 34688
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 m0, s62, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s62, 42880
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
buffer_load_dword v34, v5, s[20:23], s63 offen offset:0
buffer_load_dword v35, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_mul_hi_i32 s66, s60, 715827883
s_lshr_b32 s67, s66, 31
s_add_u32 s66, s66, s67
s_mul_i32 s66, s66, 6
s_add_u32 s66, s66, -6
s_max_i32 s66, s66, 0
s_waitcnt vmcnt(45)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_cmp_gt_i32 s66, 0
s_mov_b32 s67, 0
s_cbranch_scc0 label_kloop_end_44
.p2align 4
label_kloop_43:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 42880
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
buffer_load_dword v34, v5, s[20:23], s63 offen offset:0
buffer_load_dword v35, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_add_u32 s67, s67, 6
s_cmp_lt_i32 s67, s66
s_cbranch_scc1 label_kloop_43
label_kloop_end_44:
s_sub_u32 s67, s60, s66
s_cmp_eq_i32 s67, 1
s_cbranch_scc0 label_arm_end_45
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_45:
s_cmp_eq_i32 s67, 2
s_cbranch_scc0 label_arm_end_46
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(32)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_46:
s_cmp_eq_i32 s67, 3
s_cbranch_scc0 label_arm_end_47
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(32)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_47:
s_cmp_eq_i32 s67, 4
s_cbranch_scc0 label_arm_end_48
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(32)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(16)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_48:
s_cmp_eq_i32 s67, 5
s_cbranch_scc0 label_arm_end_49
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(32)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(16)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(8)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_49:
s_cmp_eq_i32 s67, 6
s_cbranch_scc0 label_arm_end_50
s_waitcnt vmcnt(40)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(32)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(16)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(8)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(0)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_50:
s_cmp_eq_i32 s67, 7
s_cbranch_scc0 label_arm_end_51
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(40)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(32)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(24)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(16)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(8)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(0)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_51:
s_cmp_eq_i32 s67, 8
s_cbranch_scc0 label_arm_end_52
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(40)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(32)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(24)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(16)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(8)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_52:
s_cmp_eq_i32 s67, 9
s_cbranch_scc0 label_arm_end_53
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(40)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(32)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(24)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(16)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(8)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_53:
s_cmp_eq_i32 s67, 10
s_cbranch_scc0 label_arm_end_54
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(40)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
s_waitcnt vmcnt(32)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(24)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(16)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(8)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(0)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_54:
s_cmp_eq_i32 s67, 11
s_cbranch_scc0 label_arm_end_55
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
buffer_load_dword v24, v5, s[20:23], s63 offen offset:0
buffer_load_dword v25, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
buffer_load_dword v26, v5, s[20:23], s63 offen offset:0
buffer_load_dword v27, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
buffer_load_dword v28, v5, s[20:23], s63 offen offset:0
buffer_load_dword v29, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
buffer_load_dword v30, v5, s[20:23], s63 offen offset:0
buffer_load_dword v31, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:0 lds
s_add_u32 m0, s75, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:0 lds
s_add_u32 m0, s75, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:0 lds
s_waitcnt vmcnt(43)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s62, 34688
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[12:15], s63 offen offset:128 lds
s_add_u32 m0, s75, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s63 offen offset:128 lds
s_add_u32 m0, s75, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s63 offen offset:128 lds
s_waitcnt vmcnt(43)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
buffer_load_dword v32, v5, s[20:23], s63 offen offset:0
buffer_load_dword v33, v11, s[16:19], s63 offen offset:0
s_add_u32 s63, s63, 256
s_waitcnt vmcnt(40)
v_lshrrev_b32 v34, s59, v34
v_lshrrev_b32 v35, s61, v35
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(37)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_waitcnt vmcnt(32)
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(29)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
s_waitcnt vmcnt(24)
v_lshrrev_b32 v26, s59, v26
v_lshrrev_b32 v27, s61, v27
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(21)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
s_waitcnt vmcnt(16)
v_lshrrev_b32 v28, s59, v28
v_lshrrev_b32 v29, s61, v29
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(13)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
s_waitcnt vmcnt(8)
v_lshrrev_b32 v30, s59, v30
v_lshrrev_b32 v31, s61, v31
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(5)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
s_waitcnt vmcnt(0)
v_lshrrev_b32 v32, s59, v32
v_lshrrev_b32 v33, s61, v33
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_55:
s_waitcnt 0
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_tail_skip_56
label_tail_begin_57:
s_mov_b32 m0, s62
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s65 offen offset:0 lds
s_add_u32 m0, s75, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s75, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s62, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s65 offen offset:128 lds
s_add_u32 m0, s75, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s75, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s65 offen offset:128 lds
buffer_load_dword v24, v5, s[20:23], s65 offen offset:0
buffer_load_dword v25, v11, s[16:19], s65 offen offset:0
s_waitcnt vmcnt(0)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
s_waitcnt lgkmcnt(0)
s_mov_b32 s66, 0
s_cmp_ge_i32 s64, 64
s_cbranch_scc0 label_tail_edge_58
s_branch label_tail_chunk_end_60
label_tail_edge_58:
s_cmp_gt_i32 s64, 0
s_cbranch_scc0 label_tail_zero_59
s_or_b32 s66, s66, 1
s_branch label_tail_chunk_end_60
label_tail_zero_59:
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
label_tail_chunk_end_60:
s_sub_u32 s67, s64, 64
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_61
s_branch label_tail_chunk_end_63
label_tail_edge_61:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_62
s_or_b32 s66, s66, 2
s_branch label_tail_chunk_end_63
label_tail_zero_62:
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
label_tail_chunk_end_63:
s_sub_u32 s67, s64, 128
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_64
s_branch label_tail_chunk_end_66
label_tail_edge_64:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_65
s_or_b32 s66, s66, 4
s_branch label_tail_chunk_end_66
label_tail_zero_65:
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
label_tail_chunk_end_66:
s_sub_u32 s67, s64, 192
s_cmp_ge_i32 s67, 64
s_cbranch_scc0 label_tail_edge_67
s_branch label_tail_chunk_end_69
label_tail_edge_67:
s_cmp_gt_i32 s67, 0
s_cbranch_scc0 label_tail_zero_68
s_or_b32 s66, s66, 8
s_branch label_tail_chunk_end_69
label_tail_zero_68:
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v66, 0
v_mov_b32 v67, 0
label_tail_chunk_end_69:
s_bitcmp1_b32 s66, 0
s_cbranch_scc0 label_tail_mask_skip_70
v_sub_u32 v68, s64, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v36, v36, v70
v_and_b32 v44, v44, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v37, v37, v70
v_and_b32 v45, v45, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v38, v38, v70
v_and_b32 v46, v46, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v39, v39, v70
v_and_b32 v47, v47, v70
label_tail_mask_skip_70:
s_sub_u32 s67, s64, 64
s_bitcmp1_b32 s66, 1
s_cbranch_scc0 label_tail_mask_skip_71
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v40, v40, v70
v_and_b32 v48, v48, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v41, v41, v70
v_and_b32 v49, v49, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v42, v42, v70
v_and_b32 v50, v50, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v43, v43, v70
v_and_b32 v51, v51, v70
label_tail_mask_skip_71:
s_sub_u32 s67, s64, 128
s_bitcmp1_b32 s66, 2
s_cbranch_scc0 label_tail_mask_skip_72
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v52, v52, v70
v_and_b32 v60, v60, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v53, v53, v70
v_and_b32 v61, v61, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v54, v54, v70
v_and_b32 v62, v62, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v55, v55, v70
v_and_b32 v63, v63, v70
label_tail_mask_skip_72:
s_sub_u32 s67, s64, 192
s_bitcmp1_b32 s66, 3
s_cbranch_scc0 label_tail_mask_skip_73
v_sub_u32 v68, s67, v2
v_sub_u32 v69, 0, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 4, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v56, v56, v70
v_and_b32 v64, v64, v70
v_sub_u32 v69, 4, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 8, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v57, v57, v70
v_and_b32 v65, v65, v70
v_sub_u32 v69, 8, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 12, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v58, v58, v70
v_and_b32 v66, v66, v70
v_sub_u32 v69, 12, v68
v_ashrrev_i32 v69, 31, v69
v_sub_u32 v70, 16, v68
v_max_i32 v70, 0, v70
v_min_i32 v70, 3, v70
v_lshlrev_b32 v70, 3, v70
v_lshrrev_b32 v70, v70, -1
v_and_b32 v70, v70, v69
v_and_b32 v59, v59, v70
v_and_b32 v67, v67, v70
label_tail_mask_skip_73:
v_sub_u32 v68, s64, v2
v_sub_u32 v68, v68, v2
v_cmp_gt_i32 s[66:67], v68, 0
v_mov_b32 v71, 65535
s_nop 0
v_cndmask_b32 v69, 0, v71, s[66:67]
v_add_u32 v68, -128, v68
v_cmp_gt_i32 s[66:67], v68, 0
v_mov_b32 v71, -65536
s_nop 0
v_cndmask_b32 v68, 0, v71, s[66:67]
v_or_b32 v69, v69, v68
v_and_b32 v70, 2139062143, v69
v_sub_u32 v70, 2139062143, v70
v_and_b32 v24, v24, v69
v_or_b32 v24, v24, v70
v_and_b32 v25, v25, v69
v_or_b32 v25, v25, v70
v_lshrrev_b32 v24, s59, v24
v_lshrrev_b32 v25, s61, v25
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_tail_skip_56:
label_units_variants_end_10:
s_nop 11
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_75
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_77
v_pk_mul_f32 v[12:13], s[6:7], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
s_and_b32 s60, s5, 2147483647
s_cmp_eq_u32 s60, 0
s_cbranch_scc1 label_no_beta_79
s_waitcnt vmcnt(0)
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v22
v_fmac_f32 v14, s5, v25
v_lshlrev_b32 v25, 16, v23
v_fmac_f32 v15, s5, v25
label_no_beta_79:
s_add_u32 s60, s53, 0
v_add_u32 v28, s60, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s39
s_add_u32 s60, s54, 0
v_add_u32 v26, s60, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v12, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v13, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v14, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v15, v24, s[24:27], 0 offen offset:0
s_branch label_stored_78
label_workspace_77:
s_mul_i32 s60, s4, s38
s_lshl_b32 s60, s60, 2
s_add_u32 s63, s53, 0
v_add_u32 v28, s63, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s60, v25
s_add_u32 s63, s54, 0
v_add_u32 v26, s63, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s66, s42
s_lshl_b32 s67, s42, 1
s_mul_i32 s68, s42, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v12, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v13, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v14, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v15, v24, s[28:31], 0 offen offset:0
label_stored_78:
s_branch label_epilogue_end_76
label_orientation_b_75:
s_add_u32 s63, s3, 1
s_lshl_b32 s63, s63, 4
s_add_u32 s64, s2, 1
s_lshl_b32 s64, s64, 4
s_cmp_le_u32 s63, s45
s_cselect_b32 s65, 1, 0
s_cmp_le_u32 s64, s46
s_cselect_b32 s66, 1, 0
s_and_b32 s60, s65, s66
s_and_b32 s60, s60, s58
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_80
v_pk_mul_f32 v[12:13], s[6:7], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
s_and_b32 s63, s5, 2147483647
s_cmp_eq_u32 s63, 0
s_cbranch_scc1 label_no_beta_82
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s60, 1
s_cbranch_scc0 label_c_masked_83
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_and_b32 v25, -65536, v20
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v14, s5, v25
v_and_b32 v25, -65536, v21
v_fmac_f32 v15, s5, v25
s_branch label_c_added_84
label_c_masked_83:
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v22
v_fmac_f32 v14, s5, v25
v_lshlrev_b32 v25, 16, v23
v_fmac_f32 v15, s5, v25
label_c_added_84:
label_no_beta_82:
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_vector_85
s_add_u32 s63, s53, 0
v_add_u32 v28, s63, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s39
s_add_u32 s63, s54, 0
v_add_u32 v26, s63, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v12, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v13, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v14, v24, s[24:27], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_short v15, v24, s[24:27], 0 offen offset:0
s_branch label_tile_stored_86
label_vector_85:
s_add_u32 s63, s53, 0
v_add_u32 v28, s63, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s39
s_add_u32 s63, s54, 0
v_add_u32 v26, s63, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_add_u32 v24, v25, v27
v_cvt_pk_bf16_f32 v12, v12, v13
v_cvt_pk_bf16_f32 v13, v14, v15
buffer_store_dwordx2 v[12:13], v24, s[24:27], 0 offen offset:0
label_tile_stored_86:
s_branch label_stored_81
label_workspace_80:
s_mul_i32 s63, s4, s38
s_lshl_b32 s63, s63, 2
s_cmp_eq_u32 s60, 1
s_cbranch_scc1 label_vector_87
s_add_u32 s73, s53, 0
v_add_u32 v28, s73, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s63, v25
s_add_u32 s73, s54, 0
v_add_u32 v26, s73, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s66, s42
s_lshl_b32 s67, s42, 1
s_mul_i32 s68, s42, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_add_u32 v24, v25, v27
v_add_u32 v28, s69, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v12, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s66, v24
v_add_u32 v28, s70, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v13, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s67, v24
v_add_u32 v28, s71, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v14, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s68, v24
v_add_u32 v28, s72, v26
v_cmp_lt_u32 s[76:77], v28, s57
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v24, -1, v24, s[76:77]
buffer_store_dword v15, v24, s[28:31], 0 offen offset:0
s_branch label_tile_stored_88
label_vector_87:
s_add_u32 s73, s53, 0
v_add_u32 v28, s73, v1
v_cmp_lt_u32 s[64:65], v28, s45
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s63, v25
s_add_u32 s73, s54, 0
v_add_u32 v26, s73, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s66, s42
s_lshl_b32 s67, s42, 1
s_mul_i32 s68, s42, 3
s_sub_u32 s69, 0, s56
s_sub_u32 s70, 1, s56
s_sub_u32 s71, 2, s56
s_sub_u32 s72, 3, s56
v_add_u32 v24, v25, v27
buffer_store_dwordx4 v[12:15], v24, s[28:31], 0 offen offset:0
label_tile_stored_88:
label_stored_81:
label_epilogue_end_76:
label_tile_end_74:
s_endpgm
.Lgemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 120
  .amdhsa_accum_offset 72
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr 78
  .amdhsa_group_segment_fixed_size 49152
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
            { type: int32, semantic: NumWorkGroupsZ },
            { type: int32, semantic: ConstantZero },
            { type: int32, semantic: ConstantZero } ]
    macrotile: [16, 16, 4096]
    threads: [64, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
    kStep: 256
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 32
  AssertTensorBytesLessThan: 2147483648
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 2147483393 }
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
  - .name: gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 120
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 84
    .vgpr_count: 72
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
      - .name: wholeTiles
        .size: 4
        .offset: 108
        .value_kind: by_value
      - .name: pieces
        .size: 4
        .offset: 112
        .value_kind: by_value
...
.end_amdgpu_metadata
