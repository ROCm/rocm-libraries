.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1:
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
s_mul_i32 s70, s51, 24576
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
s_lshl_b32 s71, s71, 10
s_add_u32 s71, s71, 49152
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
s_add_u32 s53, s44, 15
s_lshr_b32 s53, s53, 4
s_add_u32 s55, s43, 15
s_lshr_b32 s55, s55, 4
s_mul_i32 s54, s53, s55
s_mul_i32 s52, s54, s50
s_add_u32 s56, s52, 7
s_lshr_b32 s56, s56, 3
s_add_u32 s58, s52, -1
s_and_b32 s57, 7, s58
s_add_u32 s57, s57, 1
s_cmp_lt_i32 s2, s52
s_cbranch_scc0 label_tile_loop_end_1
label_tile_loop_0:
s_barrier
s_lshr_b32 s60, s2, 3
s_and_b32 s61, 7, s2
s_sub_u32 s62, s61, s57
s_max_i32 s62, s62, 0
s_mul_i32 s59, s61, s56
s_add_u32 s59, s59, s60
s_sub_u32 s59, s59, s62
s_abs_i32 s61, s54
s_abs_i32 s62, s59
s_xor_b32 s63, s59, s54
s_ashr_i32 s63, s63, 31
v_cvt_f32_u32 v5, s61
v_rcp_iflag_f32 v5, v5
s_sub_u32 s64, 0, s61
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s65, v5
s_mul_i32 s66, s64, s65
s_mul_hi_u32 s66, s65, s66
s_add_u32 s65, s65, s66
s_mul_hi_u32 s64, s62, s65
s_mul_i32 s66, s64, s61
s_sub_u32 s66, s62, s66
s_add_u32 s67, s64, 1
s_sub_u32 s68, s66, s61
s_cmp_ge_u32 s66, s61
s_cselect_b32 s64, s67, s64
s_cselect_b32 s66, s68, s66
s_add_u32 s67, s64, 1
s_cmp_ge_u32 s66, s61
s_cselect_b32 s64, s67, s64
s_xor_b32 s64, s64, s63
s_sub_u32 s58, s64, s63
s_mul_i32 s60, s58, s54
s_sub_u32 s60, s59, s60
s_abs_i32 s61, s53
s_abs_i32 s62, s60
s_xor_b32 s63, s60, s53
s_ashr_i32 s63, s63, 31
v_cvt_f32_u32 v5, s61
v_rcp_iflag_f32 v5, v5
s_sub_u32 s64, 0, s61
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s65, v5
s_mul_i32 s66, s64, s65
s_mul_hi_u32 s66, s65, s66
s_add_u32 s65, s65, s66
s_mul_hi_u32 s64, s62, s65
s_mul_i32 s66, s64, s61
s_sub_u32 s66, s62, s66
s_add_u32 s67, s64, 1
s_sub_u32 s68, s66, s61
s_cmp_ge_u32 s66, s61
s_cselect_b32 s64, s67, s64
s_cselect_b32 s66, s68, s66
s_add_u32 s67, s64, 1
s_cmp_ge_u32 s66, s61
s_cselect_b32 s64, s67, s64
s_xor_b32 s64, s64, s63
s_sub_u32 s55, s64, s63
s_mul_i32 s50, s55, s53
s_sub_u32 s50, s60, s50
s_lshl_b32 s59, s55, 4
s_lshl_b32 s60, s50, 4
s_add_u32 s63, s59, 0
v_add_u32 v21, s63, v6
v_cmp_lt_u32 s[64:65], v21, s43
v_mul_lo_u32 v18, v21, s48
v_add_u32 v18, v18, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v18, v26, v18, s[64:65]
s_add_u32 s63, s59, 8
v_add_u32 v21, s63, v6
v_cmp_lt_u32 s[64:65], v21, s43
v_mul_lo_u32 v19, v21, s48
v_add_u32 v19, v19, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v19, v26, v19, s[64:65]
s_add_u32 s63, s60, 0
v_add_u32 v21, s63, v6
v_cmp_lt_u32 s[64:65], v21, s44
v_mul_lo_u32 v16, v21, s49
v_add_u32 v16, v16, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v16, v26, v16, s[64:65]
s_add_u32 s63, s60, 8
v_add_u32 v21, s63, v6
v_cmp_lt_u32 s[64:65], v21, s44
v_mul_lo_u32 v17, v21, s49
v_add_u32 v17, v17, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v17, v26, v17, s[64:65]
s_lshr_b32 s63, s59, 5
s_mul_i32 s63, s63, s47
v_add_u32 v20, s63, v4
s_and_b32 s62, 16, s59
s_lshr_b32 s62, s62, 1
s_add_u32 s63, s60, 0
s_lshr_b32 s64, s63, 5
s_mul_i32 s64, s64, s47
v_add_u32 v5, s64, v4
s_and_b32 s61, 16, s63
s_lshr_b32 s61, s61, 1
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
s_cmp_eq_u32 s51, 0
s_cbranch_scc0 label_c_prefetch_skip_2
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_c_prefetch_skip_2
s_and_b32 s63, s3, 2147483647
s_cmp_eq_u32 s63, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s69, s59, 0
v_add_u32 v29, s69, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s41
s_add_u32 s69, s60, 0
v_add_u32 v27, s69, v3
v_mul_lo_u32 v28, v27, s42
s_mov_b32 s66, s42
s_lshl_b32 s67, s42, 1
s_mul_i32 s68, s42, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v22, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v23, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v24, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v25, v21, s[28:31], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s64, s55, 1
s_lshl_b32 s64, s64, 4
s_add_u32 s65, s50, 1
s_lshl_b32 s65, s65, 4
s_cmp_le_u32 s64, s43
s_cselect_b32 s66, 1, 0
s_cmp_le_u32 s65, s44
s_cselect_b32 s67, 1, 0
s_and_b32 s63, s66, s67
s_add_u32 s69, s59, 0
v_add_u32 v29, s69, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s41
s_add_u32 s69, s60, 0
v_add_u32 v27, s69, v3
v_mul_lo_u32 v28, v27, s42
s_mov_b32 s66, s42
s_lshl_b32 s67, s42, 1
s_mul_i32 s68, s42, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
s_cmp_eq_u32 s63, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v22, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v23, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v24, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_load_ushort v25, v21, s[28:31], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v21, v26, v28
buffer_load_dwordx2 v[22:23], v21, s[28:31], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_mul_i32 s67, s58, s35
s_sub_u32 s68, s34, s67
s_min_i32 s68, s68, s35
s_and_b32 s65, s68, 255
s_lshr_b32 s68, s68, 8
s_add_u32 s69, s68, 1
s_sub_u32 s69, s69, s51
s_lshr_b32 s63, s69, 1
s_lshl_b32 s64, s51, 8
s_add_u32 s64, s64, s67
s_lshl_b32 s66, s68, 8
s_add_u32 s66, s66, s67
s_and_b32 s71, 1, s68
s_cmp_eq_u32 s51, s71
s_cselect_b32 s72, 1, 0
s_cmp_lg_u32 s72, 0
s_cselect_b32 s65, s65, 0
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
buffer_load_dword v26, v5, s[16:19], s64 offen offset:0
buffer_load_dword v27, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
buffer_load_dword v28, v5, s[16:19], s64 offen offset:0
buffer_load_dword v29, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_add_u32 m0, s70, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 18304
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
buffer_load_dword v30, v5, s[16:19], s64 offen offset:0
buffer_load_dword v31, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_mul_hi_i32 s67, s63, 1431655766
s_lshr_b32 s68, s67, 31
s_add_u32 s67, s67, s68
s_mul_i32 s67, s67, 3
s_add_u32 s67, s67, -3
s_max_i32 s67, s67, 0
s_waitcnt vmcnt(26)
ds_read_b128 v[32:35], v9 offset:0
ds_read_b128 v[36:39], v10 offset:0
ds_read_b128 v[40:43], v9 offset:4096
ds_read_b128 v[44:47], v10 offset:4096
s_cmp_gt_i32 s67, 0
s_mov_b32 s68, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s64 offen offset:0
buffer_load_dword v27, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:16384
ds_read_b128 v[36:39], v10 offset:16384
ds_read_b128 v[40:43], v9 offset:20480
ds_read_b128 v[44:47], v10 offset:20480
buffer_load_dword v28, v5, s[16:19], s64 offen offset:0
buffer_load_dword v29, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v30, s61, v30
v_lshrrev_b32 v31, s62, v31
ds_read_b128 v[48:51], v9 offset:18432
ds_read_b128 v[52:55], v10 offset:18432
ds_read_b128 v[56:59], v9 offset:22528
ds_read_b128 v[60:63], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:0
ds_read_b128 v[36:39], v10 offset:0
ds_read_b128 v[40:43], v9 offset:4096
ds_read_b128 v[44:47], v10 offset:4096
buffer_load_dword v30, v5, s[16:19], s64 offen offset:0
buffer_load_dword v31, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_add_u32 s68, s68, 3
s_cmp_lt_i32 s68, s67
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s68, s63, s67
s_cmp_eq_i32 s68, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(20)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s68, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(20)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
s_waitcnt vmcnt(10)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s68, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(20)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
s_waitcnt vmcnt(10)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[32:35], v9 offset:16384
ds_read_b128 v[36:39], v10 offset:16384
ds_read_b128 v[40:43], v9 offset:20480
ds_read_b128 v[44:47], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v30, s61, v30
v_lshrrev_b32 v31, s62, v31
ds_read_b128 v[48:51], v9 offset:18432
ds_read_b128 v[52:55], v10 offset:18432
ds_read_b128 v[56:59], v9 offset:22528
ds_read_b128 v[60:63], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s68, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s64 offen offset:0
buffer_load_dword v27, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_waitcnt vmcnt(20)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[32:35], v9 offset:16384
ds_read_b128 v[36:39], v10 offset:16384
ds_read_b128 v[40:43], v9 offset:20480
ds_read_b128 v[44:47], v10 offset:20480
s_waitcnt vmcnt(10)
v_lshrrev_b32 v30, s61, v30
v_lshrrev_b32 v31, s62, v31
ds_read_b128 v[48:51], v9 offset:18432
ds_read_b128 v[52:55], v10 offset:18432
ds_read_b128 v[56:59], v9 offset:22528
ds_read_b128 v[60:63], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[32:35], v9 offset:0
ds_read_b128 v[36:39], v10 offset:0
ds_read_b128 v[40:43], v9 offset:4096
ds_read_b128 v[44:47], v10 offset:4096
s_waitcnt vmcnt(0)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s68, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s64 offen offset:0
buffer_load_dword v27, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:0 lds
s_add_u32 m0, s70, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:0 lds
s_add_u32 m0, s70, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:0 lds
s_waitcnt vmcnt(24)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s70, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s64 offen offset:128 lds
s_add_u32 m0, s70, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s64 offen offset:128 lds
s_add_u32 m0, s70, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s64 offen offset:128 lds
s_waitcnt vmcnt(24)
ds_read_b128 v[32:35], v9 offset:16384
ds_read_b128 v[36:39], v10 offset:16384
ds_read_b128 v[40:43], v9 offset:20480
ds_read_b128 v[44:47], v10 offset:20480
buffer_load_dword v28, v5, s[16:19], s64 offen offset:0
buffer_load_dword v29, v20, s[12:15], s64 offen offset:0
s_add_u32 s64, s64, 512
s_waitcnt vmcnt(20)
v_lshrrev_b32 v30, s61, v30
v_lshrrev_b32 v31, s62, v31
ds_read_b128 v[48:51], v9 offset:18432
ds_read_b128 v[52:55], v10 offset:18432
ds_read_b128 v[56:59], v9 offset:22528
ds_read_b128 v[60:63], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[32:35], v9 offset:0
ds_read_b128 v[36:39], v10 offset:0
ds_read_b128 v[40:43], v9 offset:4096
ds_read_b128 v[44:47], v10 offset:4096
s_waitcnt vmcnt(10)
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[32:35], v9 offset:8192
ds_read_b128 v[36:39], v10 offset:8192
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v28, s61, v28
v_lshrrev_b32 v29, s62, v29
ds_read_b128 v[48:51], v9 offset:10240
ds_read_b128 v[52:55], v10 offset:10240
ds_read_b128 v[56:59], v9 offset:14336
ds_read_b128 v[60:63], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_waitcnt 0
s_cmp_eq_u32 s65, 0
s_cbranch_scc1 label_tail_skip_14
label_tail_begin_15:
s_mov_b32 m0, s70
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s66 offen offset:0 lds
s_add_u32 m0, s70, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s66 offen offset:0 lds
s_add_u32 m0, s70, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s66 offen offset:0 lds
s_add_u32 m0, s70, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s66 offen offset:0 lds
s_add_u32 m0, s70, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s66 offen offset:128 lds
s_add_u32 m0, s70, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s66 offen offset:128 lds
s_add_u32 m0, s70, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s66 offen offset:128 lds
s_add_u32 m0, s70, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s66 offen offset:128 lds
buffer_load_dword v26, v5, s[16:19], s66 offen offset:0
buffer_load_dword v27, v20, s[12:15], s66 offen offset:0
s_waitcnt vmcnt(0)
ds_read_b128 v[32:35], v9 offset:0
ds_read_b128 v[36:39], v10 offset:0
ds_read_b128 v[40:43], v9 offset:4096
ds_read_b128 v[44:47], v10 offset:4096
ds_read_b128 v[48:51], v9 offset:2048
ds_read_b128 v[52:55], v10 offset:2048
ds_read_b128 v[56:59], v9 offset:6144
ds_read_b128 v[60:63], v10 offset:6144
s_waitcnt lgkmcnt(0)
s_mov_b32 s67, 0
s_cmp_ge_i32 s65, 64
s_cbranch_scc0 label_tail_edge_16
s_branch label_tail_chunk_end_18
label_tail_edge_16:
s_cmp_gt_i32 s65, 0
s_cbranch_scc0 label_tail_zero_17
s_or_b32 s67, s67, 1
s_branch label_tail_chunk_end_18
label_tail_zero_17:
v_mov_b32 v32, 0
v_mov_b32 v33, 0
v_mov_b32 v34, 0
v_mov_b32 v35, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
label_tail_chunk_end_18:
s_sub_u32 s68, s65, 64
s_cmp_ge_i32 s68, 64
s_cbranch_scc0 label_tail_edge_19
s_branch label_tail_chunk_end_21
label_tail_edge_19:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_20
s_or_b32 s67, s67, 2
s_branch label_tail_chunk_end_21
label_tail_zero_20:
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
label_tail_chunk_end_21:
s_sub_u32 s68, s65, 128
s_cmp_ge_i32 s68, 64
s_cbranch_scc0 label_tail_edge_22
s_branch label_tail_chunk_end_24
label_tail_edge_22:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_23
s_or_b32 s67, s67, 4
s_branch label_tail_chunk_end_24
label_tail_zero_23:
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
label_tail_chunk_end_24:
s_sub_u32 s68, s65, 192
s_cmp_ge_i32 s68, 64
s_cbranch_scc0 label_tail_edge_25
s_branch label_tail_chunk_end_27
label_tail_edge_25:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_26
s_or_b32 s67, s67, 8
s_branch label_tail_chunk_end_27
label_tail_zero_26:
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
label_tail_chunk_end_27:
s_bitcmp1_b32 s67, 0
s_cbranch_scc0 label_tail_mask_skip_28
v_sub_u32 v21, s65, v2
v_sub_u32 v64, 0, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 4, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v32, v32, v65
v_and_b32 v40, v40, v65
v_sub_u32 v64, 4, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 8, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v33, v33, v65
v_and_b32 v41, v41, v65
v_sub_u32 v64, 8, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 12, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v34, v34, v65
v_and_b32 v42, v42, v65
v_sub_u32 v64, 12, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 16, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v35, v35, v65
v_and_b32 v43, v43, v65
label_tail_mask_skip_28:
s_sub_u32 s68, s65, 64
s_bitcmp1_b32 s67, 1
s_cbranch_scc0 label_tail_mask_skip_29
v_sub_u32 v21, s68, v2
v_sub_u32 v64, 0, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 4, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v36, v36, v65
v_and_b32 v44, v44, v65
v_sub_u32 v64, 4, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 8, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v37, v37, v65
v_and_b32 v45, v45, v65
v_sub_u32 v64, 8, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 12, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v38, v38, v65
v_and_b32 v46, v46, v65
v_sub_u32 v64, 12, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 16, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v39, v39, v65
v_and_b32 v47, v47, v65
label_tail_mask_skip_29:
s_sub_u32 s68, s65, 128
s_bitcmp1_b32 s67, 2
s_cbranch_scc0 label_tail_mask_skip_30
v_sub_u32 v21, s68, v2
v_sub_u32 v64, 0, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 4, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v48, v48, v65
v_and_b32 v56, v56, v65
v_sub_u32 v64, 4, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 8, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v49, v49, v65
v_and_b32 v57, v57, v65
v_sub_u32 v64, 8, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 12, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v50, v50, v65
v_and_b32 v58, v58, v65
v_sub_u32 v64, 12, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 16, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v51, v51, v65
v_and_b32 v59, v59, v65
label_tail_mask_skip_30:
s_sub_u32 s68, s65, 192
s_bitcmp1_b32 s67, 3
s_cbranch_scc0 label_tail_mask_skip_31
v_sub_u32 v21, s68, v2
v_sub_u32 v64, 0, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 4, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v52, v52, v65
v_and_b32 v60, v60, v65
v_sub_u32 v64, 4, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 8, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v53, v53, v65
v_and_b32 v61, v61, v65
v_sub_u32 v64, 8, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 12, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v54, v54, v65
v_and_b32 v62, v62, v65
v_sub_u32 v64, 12, v21
v_ashrrev_i32 v64, 31, v64
v_sub_u32 v65, 16, v21
v_max_i32 v65, 0, v65
v_min_i32 v65, 3, v65
v_lshlrev_b32 v65, 3, v65
v_lshrrev_b32 v65, v65, -1
v_and_b32 v65, v65, v64
v_and_b32 v55, v55, v65
v_and_b32 v63, v63, v65
label_tail_mask_skip_31:
v_sub_u32 v21, s65, v2
v_sub_u32 v21, v21, v2
v_cmp_gt_i32 s[68:69], v21, 0
v_mov_b32 v66, 65535
s_nop 0
v_cndmask_b32 v64, 0, v66, s[68:69]
v_add_u32 v21, -128, v21
v_cmp_gt_i32 s[68:69], v21, 0
v_mov_b32 v66, -65536
s_nop 0
v_cndmask_b32 v21, 0, v66, s[68:69]
v_or_b32 v64, v64, v21
v_and_b32 v65, 2139062143, v64
v_sub_u32 v65, 2139062143, v65
v_and_b32 v26, v26, v64
v_or_b32 v26, v26, v65
v_and_b32 v27, v27, v64
v_or_b32 v27, v27, v65
v_lshrrev_b32 v26, s61, v26
v_lshrrev_b32 v27, s62, v27
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[32:39], v[40:47], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[48:55], v[56:63], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_tail_skip_14:
s_nop 11
s_cmp_eq_u32 s51, 0
s_cbranch_scc1 label_wave_zero_33
ds_write_b128 v11, v[12:15] offset:0
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_tile_end_32
label_wave_zero_33:
s_barrier
ds_read_b128 v[26:29], v11 offset:0
s_waitcnt lgkmcnt(0)
v_add_f32 v12, v12, v26
v_add_f32 v13, v13, v27
v_add_f32 v14, v14, v28
v_add_f32 v15, v15, v29
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_34
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_36
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
s_and_b32 s63, s3, 2147483647
s_cmp_eq_u32 s63, 0
s_cbranch_scc1 label_no_beta_38
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_no_beta_38:
s_add_u32 s63, s59, 0
v_add_u32 v29, s63, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s37
s_add_u32 s63, s60, 0
v_add_u32 v27, s63, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s66, s38
s_lshl_b32 s67, s38, 1
s_mul_i32 s68, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_stored_37
label_workspace_36:
s_mul_i32 s63, s58, s36
s_lshl_b32 s63, s63, 2
s_add_u32 s69, s59, 0
v_add_u32 v29, s69, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s63, v26
s_add_u32 s69, s60, 0
v_add_u32 v27, s69, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v12, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v13, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v14, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v15, v21, s[24:27], 0 offen offset:0
label_stored_37:
s_branch label_epilogue_end_35
label_orientation_b_34:
s_add_u32 s64, s55, 1
s_lshl_b32 s64, s64, 4
s_add_u32 s65, s50, 1
s_lshl_b32 s65, s65, 4
s_cmp_le_u32 s64, s43
s_cselect_b32 s66, 1, 0
s_cmp_le_u32 s65, s44
s_cselect_b32 s67, 1, 0
s_and_b32 s63, s66, s67
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_39
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
s_and_b32 s64, s3, 2147483647
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_no_beta_41
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s63, 1
s_cbranch_scc0 label_c_masked_42
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_and_b32 v26, -65536, v22
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v14, s3, v26
v_and_b32 v26, -65536, v23
v_fmac_f32 v15, s3, v26
s_branch label_c_added_43
label_c_masked_42:
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_c_added_43:
label_no_beta_41:
s_cmp_eq_u32 s63, 1
s_cbranch_scc1 label_vector_44
s_add_u32 s69, s59, 0
v_add_u32 v29, s69, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s37
s_add_u32 s69, s60, 0
v_add_u32 v27, s69, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s66, s38
s_lshl_b32 s67, s38, 1
s_mul_i32 s68, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_tile_stored_45
label_vector_44:
s_add_u32 s69, s59, 0
v_add_u32 v29, s69, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s37
s_add_u32 s69, s60, 0
v_add_u32 v27, s69, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s66, s38
s_lshl_b32 s67, s38, 1
s_mul_i32 s68, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
v_cvt_pk_bf16_f32 v12, v12, v13
v_cvt_pk_bf16_f32 v13, v14, v15
buffer_store_dwordx2 v[12:13], v21, s[20:23], 0 offen offset:0
label_tile_stored_45:
s_branch label_stored_40
label_workspace_39:
s_mul_i32 s69, s58, s36
s_lshl_b32 s69, s69, 2
s_cmp_eq_u32 s63, 1
s_cbranch_scc1 label_vector_46
s_add_u32 s75, s59, 0
v_add_u32 v29, s75, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s69, v26
s_add_u32 s75, s60, 0
v_add_u32 v27, s75, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v12, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s66, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v13, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s67, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v14, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v15, v21, s[24:27], 0 offen offset:0
s_branch label_tile_stored_47
label_vector_46:
s_add_u32 s75, s59, 0
v_add_u32 v29, s75, v1
v_cmp_lt_u32 s[64:65], v29, s43
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s69, v26
s_add_u32 s75, s60, 0
v_add_u32 v27, s75, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s66, s40
s_lshl_b32 s67, s40, 1
s_mul_i32 s68, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
buffer_store_dwordx4 v[12:15], v21, s[24:27], 0 offen offset:0
label_tile_stored_47:
label_stored_40:
label_epilogue_end_35:
label_tile_end_32:
s_add_u32 s2, s2, s46
s_cmp_lt_i32 s2, s52
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 120
  .amdhsa_accum_offset 72
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr 80
  .amdhsa_group_segment_fixed_size 50176
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
            { type: int32, semantic: NumWorkGroupsZ },
            { type: int32, semantic: ConstantZero },
            { type: int32, semantic: ConstantZero } ]
    macrotile: [16, 16, 4096]
    threads: [128, 1, 1]
    grid: [TilesWideSkinnyBatch, One, One]
    kStep: 256
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 32
  AssertTensorBytesLessThan: 2147483648
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 1073741825 }
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
  - .name: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF3_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 120
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 50176
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .sgpr_count: 86
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
