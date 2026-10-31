.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1:
s_load_dword s46, s[0:1], 96
s_load_dword s35, s[0:1], 80
s_load_dword s32, s[0:1], 88
s_load_dword s34, s[0:1], 76
s_load_dword s55, s[0:1], 72
s_load_dword s54, s[0:1], 68
s_load_dword s53, s[0:1], 64
s_load_dword s52, s[0:1], 60
s_load_dword s51, s[0:1], 56
s_load_dwordx2 s[24:25], s[0:1], 40
s_load_dwordx2 s[62:63], s[0:1], 32
s_load_dwordx2 s[60:61], s[0:1], 24
s_load_dwordx2 s[58:59], s[0:1], 16
s_load_dwordx2 s[56:57], s[0:1], 8
s_load_dwordx2 s[20:21], s[0:1], 0
s_load_dword s3, s[0:1], 92
s_load_dword s68, s[0:1], 84
s_load_dwordx2 s[28:29], s[0:1], 48
s_waitcnt lgkmcnt(0)
s_add_u32 s69, s34, 3
s_lshr_b32 s69, s69, 2
s_lshl_b32 s69, s69, 2
s_add_u32 s64, s54, -1
s_mul_i32 s64, s64, s52
s_add_u32 s64, s64, s69
s_add_u32 s65, s55, -1
s_mul_i32 s65, s65, s53
s_add_u32 s65, s65, s69
s_add_u32 s47, s34, 255
s_lshr_b32 s47, s47, 8
s_lshl_b32 s47, s47, 8
s_add_u32 s66, s54, 31
s_lshr_b32 s66, s66, 5
s_mul_i32 s66, s66, s47
s_add_u32 s67, s55, 31
s_lshr_b32 s67, s67, 5
s_mul_i32 s67, s67, s47
s_add_u32 s22, s55, -1
s_mul_i32 s22, s22, s51
s_lshl_b32 s69, s54, 1
s_add_u32 s22, s22, s69
s_add_u32 s30, s55, -1
s_mul_i32 s30, s30, s68
s_lshl_b32 s69, s54, 1
s_add_u32 s30, s30, s69
s_add_u32 s50, s34, s35
s_add_u32 s50, s50, -1
s_abs_i32 s70, s35
s_abs_i32 s71, s50
s_xor_b32 s72, s50, s35
s_ashr_i32 s72, s72, 31
v_cvt_f32_u32 v1, s70
v_rcp_iflag_f32 v1, v1
s_sub_u32 s73, 0, s70
v_mul_f32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32 v1, v1
s_nop 0
v_readfirstlane_b32 s74, v1
s_mul_i32 s75, s73, s74
s_mul_hi_u32 s75, s74, s75
s_add_u32 s74, s74, s75
s_mul_hi_u32 s73, s71, s74
s_mul_i32 s75, s73, s70
s_sub_u32 s75, s71, s75
s_add_u32 s76, s73, 1
s_sub_u32 s77, s75, s70
s_cmp_ge_u32 s75, s70
s_cselect_b32 s73, s76, s73
s_cselect_b32 s75, s77, s75
s_add_u32 s76, s73, 1
s_cmp_ge_u32 s75, s70
s_cselect_b32 s73, s76, s73
s_xor_b32 s73, s73, s72
s_sub_u32 s50, s73, s72
s_mul_i32 s26, s54, s50
s_mul_i32 s26, s26, s55
s_lshl_b32 s26, s26, 2
s_mul_i32 s36, s54, s55
s_lshl_b32 s40, s54, 2
s_cmp_le_u32 s54, s55
s_cselect_b64 s[4:5], s[56:57], s[58:59]
s_cselect_b32 s6, s64, s65
s_cselect_b64 s[8:9], s[58:59], s[56:57]
s_cselect_b32 s10, s65, s64
s_cselect_b64 s[12:13], s[60:61], s[62:63]
s_cselect_b32 s14, s66, s67
s_cselect_b64 s[16:17], s[62:63], s[60:61]
s_cselect_b32 s18, s67, s66
s_cselect_b32 s48, s52, s53
s_cselect_b32 s49, s53, s52
s_cselect_b32 s37, 2, s51
s_cselect_b32 s38, s51, 2
s_cselect_b32 s41, 2, s68
s_cselect_b32 s42, s68, 2
s_cselect_b32 s39, 4, s40
s_cselect_b32 s40, s40, 4
s_cselect_b32 s45, 1, 0
s_min_u32 s43, s54, s55
s_max_u32 s44, s54, s55
s_mov_b32 s7, 131072
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
v_lshrrev_b32 v1, 6, v0
s_nop 4
v_readfirstlane_b32 s51, v1
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s52, s51, 49152
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
s_add_u32 s53, s52, 0
v_add_u32 v9, s53, v14
v_xor_b32 v10, 64, v9
s_add_u32 s53, s51, -1
s_max_i32 s53, s53, 0
s_lshl_b32 s53, s53, 12
s_add_u32 s53, s53, 98304
v_lshlrev_b32 v11, 4, v5
v_add_u32 v11, s53, v11
s_add_u32 s54, s44, 31
s_lshr_b32 s54, s54, 5
s_add_u32 s56, s43, 31
s_lshr_b32 s56, s56, 5
s_mul_i32 s55, s54, s56
s_mul_i32 s53, s55, s50
s_add_u32 s57, s53, 7
s_lshr_b32 s57, s57, 3
s_add_u32 s59, s53, -1
s_and_b32 s58, 7, s59
s_add_u32 s58, s58, 1
s_lshl_b32 s59, s54, 3
s_cmp_lt_i32 s2, s53
s_cbranch_scc0 label_tile_loop_end_1
label_tile_loop_0:
s_waitcnt lgkmcnt(0)
s_barrier
s_lshr_b32 s63, s2, 3
s_and_b32 s64, 7, s2
s_sub_u32 s65, s64, s58
s_max_i32 s65, s65, 0
s_mul_i32 s62, s64, s57
s_add_u32 s62, s62, s63
s_sub_u32 s62, s62, s65
s_abs_i32 s64, s55
s_abs_i32 s65, s62
s_xor_b32 s66, s62, s55
s_ashr_i32 s66, s66, 31
v_cvt_f32_u32 v5, s64
v_rcp_iflag_f32 v5, v5
s_sub_u32 s67, 0, s64
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s68, v5
s_mul_i32 s69, s67, s68
s_mul_hi_u32 s69, s68, s69
s_add_u32 s68, s68, s69
s_mul_hi_u32 s67, s65, s68
s_mul_i32 s69, s67, s64
s_sub_u32 s69, s65, s69
s_add_u32 s70, s67, 1
s_sub_u32 s71, s69, s64
s_cmp_ge_u32 s69, s64
s_cselect_b32 s67, s70, s67
s_cselect_b32 s69, s71, s69
s_add_u32 s70, s67, 1
s_cmp_ge_u32 s69, s64
s_cselect_b32 s67, s70, s67
s_xor_b32 s67, s67, s66
s_sub_u32 s61, s67, s66
s_mul_i32 s63, s61, s55
s_sub_u32 s63, s62, s63
s_abs_i32 s68, s59
s_abs_i32 s69, s63
s_xor_b32 s70, s63, s59
s_ashr_i32 s70, s70, 31
v_cvt_f32_u32 v5, s68
v_rcp_iflag_f32 v5, v5
s_sub_u32 s71, 0, s68
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s72, v5
s_mul_i32 s73, s71, s72
s_mul_hi_u32 s73, s72, s73
s_add_u32 s72, s72, s73
s_mul_hi_u32 s71, s69, s72
s_mul_i32 s73, s71, s68
s_sub_u32 s73, s69, s73
s_add_u32 s74, s71, 1
s_sub_u32 s75, s73, s68
s_cmp_ge_u32 s73, s68
s_cselect_b32 s71, s74, s71
s_cselect_b32 s73, s75, s73
s_add_u32 s74, s71, 1
s_cmp_ge_u32 s73, s68
s_cselect_b32 s71, s74, s71
s_xor_b32 s71, s71, s70
s_sub_u32 s64, s71, s70
s_mul_i32 s65, s64, s59
s_sub_u32 s65, s63, s65
s_lshl_b32 s66, s64, 3
s_sub_u32 s67, s56, s66
s_min_i32 s67, s67, 8
s_abs_i32 s68, s67
s_abs_i32 s69, s65
s_xor_b32 s70, s65, s67
s_ashr_i32 s70, s70, 31
v_cvt_f32_u32 v5, s68
v_rcp_iflag_f32 v5, v5
s_sub_u32 s71, 0, s68
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s72, v5
s_mul_i32 s73, s71, s72
s_mul_hi_u32 s73, s72, s73
s_add_u32 s72, s72, s73
s_mul_hi_u32 s71, s69, s72
s_mul_i32 s73, s71, s68
s_sub_u32 s73, s69, s73
s_add_u32 s74, s71, 1
s_sub_u32 s75, s73, s68
s_cmp_ge_u32 s73, s68
s_cselect_b32 s71, s74, s71
s_cselect_b32 s73, s75, s73
s_add_u32 s74, s71, 1
s_cmp_ge_u32 s73, s68
s_cselect_b32 s71, s74, s71
s_xor_b32 s71, s71, s70
s_sub_u32 s50, s71, s70
s_mul_i32 s60, s50, s67
s_sub_u32 s60, s65, s60
s_add_u32 s60, s60, s66
s_lshl_b32 s62, s60, 5
s_lshl_b32 s63, s50, 5
s_add_u32 s64, s62, 0
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s43
s_nop 1
v_mul_lo_u32 v32, v37, s48
v_add_u32 v32, v32, v7
v_mov_b32 v54, -2147483648
v_cndmask_b32 v32, v54, v32, s[66:67]
s_add_u32 s64, s62, 8
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s43
s_nop 1
v_mul_lo_u32 v33, v37, s48
v_add_u32 v33, v33, v8
v_mov_b32 v54, -2147483648
v_cndmask_b32 v33, v54, v33, s[66:67]
s_add_u32 s64, s62, 16
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s43
s_nop 1
v_mul_lo_u32 v34, v37, s48
v_add_u32 v34, v34, v7
v_mov_b32 v54, -2147483648
v_cndmask_b32 v34, v54, v34, s[66:67]
s_add_u32 s64, s62, 24
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s43
s_nop 1
v_mul_lo_u32 v35, v37, s48
v_add_u32 v35, v35, v8
v_mov_b32 v54, -2147483648
v_cndmask_b32 v35, v54, v35, s[66:67]
s_add_u32 s64, s63, 0
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s44
s_nop 1
v_mul_lo_u32 v28, v37, s49
v_add_u32 v28, v28, v7
v_mov_b32 v54, -2147483648
v_cndmask_b32 v28, v54, v28, s[66:67]
s_add_u32 s64, s63, 8
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s44
s_nop 1
v_mul_lo_u32 v29, v37, s49
v_add_u32 v29, v29, v8
v_mov_b32 v54, -2147483648
v_cndmask_b32 v29, v54, v29, s[66:67]
s_add_u32 s64, s63, 16
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s44
s_nop 1
v_mul_lo_u32 v30, v37, s49
v_add_u32 v30, v30, v7
v_mov_b32 v54, -2147483648
v_cndmask_b32 v30, v54, v30, s[66:67]
s_add_u32 s64, s63, 24
v_add_u32 v37, s64, v6
v_cmp_lt_u32 s[66:67], v37, s44
s_nop 1
v_mul_lo_u32 v31, v37, s49
v_add_u32 v31, v31, v8
v_mov_b32 v54, -2147483648
v_cndmask_b32 v31, v54, v31, s[66:67]
s_lshr_b32 s64, s62, 5
s_mul_i32 s64, s64, s47
v_add_u32 v36, s64, v4
s_add_u32 s64, s63, 0
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
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v24, 0
v_mov_b32 v25, 0
v_mov_b32 v26, 0
v_mov_b32 v27, 0
s_nop 7
s_cmp_eq_u32 s51, 0
s_cbranch_scc0 label_c_prefetch_skip_2
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_c_prefetch_skip_2
s_and_b32 s64, s3, 2147483647
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s65, s62, 0
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s41
s_add_u32 s65, s62, 16
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s41
s_add_u32 s65, s63, 0
v_add_u32 v56, s65, v3
v_mul_lo_u32 v58, v56, s42
s_add_u32 s65, s63, 16
v_add_u32 v57, s65, v3
v_mul_lo_u32 v59, v57, s42
s_mov_b32 s70, s42
s_lshl_b32 s71, s42, 1
s_mul_i32 s72, s42, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v38, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v39, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v40, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v41, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v46, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v47, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v48, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v49, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v42, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v43, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v44, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v45, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v50, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v51, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v52, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v53, v37, s[28:31], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s65, s60, 1
s_lshl_b32 s65, s65, 5
s_add_u32 s66, s50, 1
s_lshl_b32 s66, s66, 5
s_cmp_le_u32 s65, s43
s_cselect_b32 s67, 1, 0
s_cmp_le_u32 s66, s44
s_cselect_b32 s68, 1, 0
s_and_b32 s64, s67, s68
s_add_u32 s65, s62, 0
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s41
s_add_u32 s65, s62, 16
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s41
s_add_u32 s65, s63, 0
v_add_u32 v56, s65, v3
v_mul_lo_u32 v58, v56, s42
s_add_u32 s65, s63, 16
v_add_u32 v57, s65, v3
v_mul_lo_u32 v59, v57, s42
s_mov_b32 s70, s42
s_lshl_b32 s71, s42, 1
s_mul_i32 s72, s42, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v38, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v39, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v40, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v41, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v46, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v47, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v48, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v49, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v42, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v43, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v44, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v45, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v50, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v51, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v52, v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_load_ushort v53, v37, s[28:31], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v37, v54, v58
buffer_load_dwordx2 v[38:39], v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v54, v59
buffer_load_dwordx2 v[40:41], v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v58
buffer_load_dwordx2 v[42:43], v37, s[28:31], 0 offen offset:0
v_add_u32 v37, v55, v59
buffer_load_dwordx2 v[44:45], v37, s[28:31], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_mul_i32 s66, s61, s35
s_sub_u32 s67, s34, s66
s_min_i32 s67, s67, s35
s_lshr_b32 s67, s67, 8
s_add_u32 s68, s67, 1
s_sub_u32 s68, s68, s51
s_lshr_b32 s64, s68, 1
s_lshl_b32 s65, s51, 8
s_add_u32 s65, s65, s66
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
buffer_load_dword v54, v5, s[16:19], s65 offen offset:0
buffer_load_dword v55, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 24576
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 28672
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 25600
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 29696
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 26496
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 30592
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 27520
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 31616
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
buffer_load_dword v56, v5, s[16:19], s65 offen offset:0
buffer_load_dword v57, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 m0, s52, 32768
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 36864
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 40960
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 45056
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 33792
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 37888
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 41984
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 46080
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 34688
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 38784
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 42880
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 46976
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 35712
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 39808
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 43904
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 48000
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
buffer_load_dword v58, v5, s[16:19], s65 offen offset:0
buffer_load_dword v59, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_mul_hi_i32 s66, s64, 1431655766
s_lshr_b32 s67, s66, 31
s_add_u32 s66, s66, s67
s_mul_i32 s66, s66, 3
s_add_u32 s66, s66, -3
s_max_i32 s66, s66, 0
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:0
ds_read_b128 v[64:67], v10 offset:0
ds_read_b128 v[68:71], v9 offset:4096
ds_read_b128 v[72:75], v10 offset:4096
ds_read_b128 v[76:79], v9 offset:8192
ds_read_b128 v[80:83], v10 offset:8192
ds_read_b128 v[84:87], v9 offset:12288
ds_read_b128 v[88:91], v10 offset:12288
s_cmp_gt_i32 s66, 0
s_mov_b32 s67, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v54, v5, s[16:19], s65 offen offset:0
buffer_load_dword v55, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 24576
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 28672
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 25600
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 29696
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 26496
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 30592
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 27520
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 31616
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:32768
ds_read_b128 v[64:67], v10 offset:32768
ds_read_b128 v[68:71], v9 offset:36864
ds_read_b128 v[72:75], v10 offset:36864
ds_read_b128 v[76:79], v9 offset:40960
ds_read_b128 v[80:83], v10 offset:40960
ds_read_b128 v[84:87], v9 offset:45056
ds_read_b128 v[88:91], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v56, v5, s[16:19], s65 offen offset:0
buffer_load_dword v57, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:34816
ds_read_b128 v[96:99], v10 offset:34816
ds_read_b128 v[100:103], v9 offset:38912
ds_read_b128 v[104:107], v10 offset:38912
ds_read_b128 v[108:111], v9 offset:43008
ds_read_b128 v[112:115], v10 offset:43008
ds_read_b128 v[116:119], v9 offset:47104
ds_read_b128 v[120:123], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 32768
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 36864
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 40960
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 45056
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 33792
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 37888
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 41984
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 46080
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 34688
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 38784
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 42880
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 46976
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 35712
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 39808
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 43904
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 48000
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:0
ds_read_b128 v[64:67], v10 offset:0
ds_read_b128 v[68:71], v9 offset:4096
ds_read_b128 v[72:75], v10 offset:4096
ds_read_b128 v[76:79], v9 offset:8192
ds_read_b128 v[80:83], v10 offset:8192
ds_read_b128 v[84:87], v9 offset:12288
ds_read_b128 v[88:91], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v58, v5, s[16:19], s65 offen offset:0
buffer_load_dword v59, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_add_u32 s67, s67, 3
s_cmp_lt_i32 s67, s66
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s67, s64, s66
s_cmp_eq_i32 s67, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s67, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(28)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(18)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s67, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(28)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(18)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(10)
ds_read_b128 v[60:63], v9 offset:32768
ds_read_b128 v[64:67], v10 offset:32768
ds_read_b128 v[68:71], v9 offset:36864
ds_read_b128 v[72:75], v10 offset:36864
ds_read_b128 v[76:79], v9 offset:40960
ds_read_b128 v[80:83], v10 offset:40960
ds_read_b128 v[84:87], v9 offset:45056
ds_read_b128 v[88:91], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[92:95], v9 offset:34816
ds_read_b128 v[96:99], v10 offset:34816
ds_read_b128 v[100:103], v9 offset:38912
ds_read_b128 v[104:107], v10 offset:38912
ds_read_b128 v[108:111], v9 offset:43008
ds_read_b128 v[112:115], v10 offset:43008
ds_read_b128 v[116:119], v9 offset:47104
ds_read_b128 v[120:123], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s67, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v54, v5, s[16:19], s65 offen offset:0
buffer_load_dword v55, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(28)
ds_read_b128 v[60:63], v9 offset:32768
ds_read_b128 v[64:67], v10 offset:32768
ds_read_b128 v[68:71], v9 offset:36864
ds_read_b128 v[72:75], v10 offset:36864
ds_read_b128 v[76:79], v9 offset:40960
ds_read_b128 v[80:83], v10 offset:40960
ds_read_b128 v[84:87], v9 offset:45056
ds_read_b128 v[88:91], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(18)
ds_read_b128 v[92:95], v9 offset:34816
ds_read_b128 v[96:99], v10 offset:34816
ds_read_b128 v[100:103], v9 offset:38912
ds_read_b128 v[104:107], v10 offset:38912
ds_read_b128 v[108:111], v9 offset:43008
ds_read_b128 v[112:115], v10 offset:43008
ds_read_b128 v[116:119], v9 offset:47104
ds_read_b128 v[120:123], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(10)
ds_read_b128 v[60:63], v9 offset:0
ds_read_b128 v[64:67], v10 offset:0
ds_read_b128 v[68:71], v9 offset:4096
ds_read_b128 v[72:75], v10 offset:4096
ds_read_b128 v[76:79], v9 offset:8192
ds_read_b128 v[80:83], v10 offset:8192
ds_read_b128 v[84:87], v9 offset:12288
ds_read_b128 v[88:91], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s67, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v54, v5, s[16:19], s65 offen offset:0
buffer_load_dword v55, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 24576
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 28672
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:0 lds
s_add_u32 m0, s52, 25600
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 29696
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v28, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v30, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 26496
s_nop 0
buffer_load_dwordx4 v32, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 30592
s_nop 0
buffer_load_dwordx4 v34, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v29, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v31, s[8:11], s65 offen offset:128 lds
s_add_u32 m0, s52, 27520
s_nop 0
buffer_load_dwordx4 v33, s[4:7], s65 offen offset:128 lds
s_add_u32 m0, s52, 31616
s_nop 0
buffer_load_dwordx4 v35, s[4:7], s65 offen offset:128 lds
s_waitcnt vmcnt(44)
ds_read_b128 v[60:63], v9 offset:32768
ds_read_b128 v[64:67], v10 offset:32768
ds_read_b128 v[68:71], v9 offset:36864
ds_read_b128 v[72:75], v10 offset:36864
ds_read_b128 v[76:79], v9 offset:40960
ds_read_b128 v[80:83], v10 offset:40960
ds_read_b128 v[84:87], v9 offset:45056
ds_read_b128 v[88:91], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v56, v5, s[16:19], s65 offen offset:0
buffer_load_dword v57, v36, s[12:15], s65 offen offset:0
s_add_u32 s65, s65, 512
s_waitcnt vmcnt(36)
ds_read_b128 v[92:95], v9 offset:34816
ds_read_b128 v[96:99], v10 offset:34816
ds_read_b128 v[100:103], v9 offset:38912
ds_read_b128 v[104:107], v10 offset:38912
ds_read_b128 v[108:111], v9 offset:43008
ds_read_b128 v[112:115], v10 offset:43008
ds_read_b128 v[116:119], v9 offset:47104
ds_read_b128 v[120:123], v10 offset:47104
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(28)
ds_read_b128 v[60:63], v9 offset:0
ds_read_b128 v[64:67], v10 offset:0
ds_read_b128 v[68:71], v9 offset:4096
ds_read_b128 v[72:75], v10 offset:4096
ds_read_b128 v[76:79], v9 offset:8192
ds_read_b128 v[80:83], v10 offset:8192
ds_read_b128 v[84:87], v9 offset:12288
ds_read_b128 v[88:91], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v58, v59 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v58, v59 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v58, v59 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v58, v59 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(18)
ds_read_b128 v[92:95], v9 offset:2048
ds_read_b128 v[96:99], v10 offset:2048
ds_read_b128 v[100:103], v9 offset:6144
ds_read_b128 v[104:107], v10 offset:6144
ds_read_b128 v[108:111], v9 offset:10240
ds_read_b128 v[112:115], v10 offset:10240
ds_read_b128 v[116:119], v9 offset:14336
ds_read_b128 v[120:123], v10 offset:14336
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(10)
ds_read_b128 v[60:63], v9 offset:16384
ds_read_b128 v[64:67], v10 offset:16384
ds_read_b128 v[68:71], v9 offset:20480
ds_read_b128 v[72:75], v10 offset:20480
ds_read_b128 v[76:79], v9 offset:24576
ds_read_b128 v[80:83], v10 offset:24576
ds_read_b128 v[84:87], v9 offset:28672
ds_read_b128 v[88:91], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v54, v55 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v54, v55 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v54, v55 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v54, v55 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[92:95], v9 offset:18432
ds_read_b128 v[96:99], v10 offset:18432
ds_read_b128 v[100:103], v9 offset:22528
ds_read_b128 v[104:107], v10 offset:22528
ds_read_b128 v[108:111], v9 offset:26624
ds_read_b128 v[112:115], v10 offset:26624
ds_read_b128 v[116:119], v9 offset:30720
ds_read_b128 v[120:123], v10 offset:30720
s_nop 7
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[60:67], v[76:83], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[60:67], v[84:91], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:75], v[76:83], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[68:75], v[84:91], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[92:99], v[108:115], v[12:15], v56, v57 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[92:99], v[116:123], v[16:19], v56, v57 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[100:107], v[108:115], v[20:23], v56, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[100:107], v[116:123], v[24:27], v56, v57 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_nop 15
s_nop 15
s_cmp_eq_u32 s51, 0
s_cbranch_scc1 label_wave_zero_15
ds_write_b128 v11, v[12:15] offset:0
ds_write_b128 v11, v[16:19] offset:1024
ds_write_b128 v11, v[20:23] offset:2048
ds_write_b128 v11, v[24:27] offset:3072
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_tile_end_14
label_wave_zero_15:
s_barrier
ds_read_b128 v[54:57], v11 offset:0
ds_read_b128 v[58:61], v11 offset:1024
ds_read_b128 v[62:65], v11 offset:2048
ds_read_b128 v[66:69], v11 offset:3072
s_waitcnt lgkmcnt(0)
v_add_f32 v12, v12, v54
v_add_f32 v13, v13, v55
v_add_f32 v14, v14, v56
v_add_f32 v15, v15, v57
v_add_f32 v16, v16, v58
v_add_f32 v17, v17, v59
v_add_f32 v18, v18, v60
v_add_f32 v19, v19, v61
v_add_f32 v20, v20, v62
v_add_f32 v21, v21, v63
v_add_f32 v22, v22, v64
v_add_f32 v23, v23, v65
v_add_f32 v24, v24, v66
v_add_f32 v25, v25, v67
v_add_f32 v26, v26, v68
v_add_f32 v27, v27, v69
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_16
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_18
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[32:33], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[32:33], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], s[32:33], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[32:33], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[32:33], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], s[32:33], v[26:27] op_sel_hi:[0,1]
s_and_b32 s64, s3, 2147483647
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 label_no_beta_20
s_waitcnt vmcnt(0)
v_lshlrev_b32 v54, 16, v38
v_fmac_f32 v12, s3, v54
v_lshlrev_b32 v54, 16, v39
v_fmac_f32 v13, s3, v54
v_lshlrev_b32 v54, 16, v40
v_fmac_f32 v14, s3, v54
v_lshlrev_b32 v54, 16, v41
v_fmac_f32 v15, s3, v54
v_lshlrev_b32 v54, 16, v46
v_fmac_f32 v20, s3, v54
v_lshlrev_b32 v54, 16, v47
v_fmac_f32 v21, s3, v54
v_lshlrev_b32 v54, 16, v48
v_fmac_f32 v22, s3, v54
v_lshlrev_b32 v54, 16, v49
v_fmac_f32 v23, s3, v54
v_lshlrev_b32 v54, 16, v42
v_fmac_f32 v16, s3, v54
v_lshlrev_b32 v54, 16, v43
v_fmac_f32 v17, s3, v54
v_lshlrev_b32 v54, 16, v44
v_fmac_f32 v18, s3, v54
v_lshlrev_b32 v54, 16, v45
v_fmac_f32 v19, s3, v54
v_lshlrev_b32 v54, 16, v50
v_fmac_f32 v24, s3, v54
v_lshlrev_b32 v54, 16, v51
v_fmac_f32 v25, s3, v54
v_lshlrev_b32 v54, 16, v52
v_fmac_f32 v26, s3, v54
v_lshlrev_b32 v54, 16, v53
v_fmac_f32 v27, s3, v54
label_no_beta_20:
s_add_u32 s75, s62, 0
v_add_u32 v60, s75, v1
v_cmp_lt_u32 s[64:65], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s37
s_add_u32 s75, s62, 16
v_add_u32 v60, s75, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s37
s_add_u32 s75, s63, 0
v_add_u32 v56, s75, v3
v_mul_lo_u32 v58, v56, s38
s_add_u32 s75, s63, 16
v_add_u32 v57, s75, v3
v_mul_lo_u32 v59, v57, s38
s_mov_b32 s68, s38
s_lshl_b32 s69, s38, 1
s_mul_i32 s70, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[76:77], v56, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v12, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v37, v54, v58
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v56, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v13, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v37, v54, v58
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v56, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v14, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v56, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v15, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[76:77], v57, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v20, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v37, v54, v59
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v57, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v21, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v37, v54, v59
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v57, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v22, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v57, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v23, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[76:77], v56, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v16, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v37, v55, v58
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v56, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v17, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v37, v55, v58
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v56, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v18, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v56, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v19, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[76:77], v57, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v24, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v37, v55, v59
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v57, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v25, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v26, 16, v26
v_add_u32 v37, v55, v59
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v57, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v26, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v27, 16, v27
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v57, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_short v27, v37, s[20:23], 0 offen offset:0
s_branch label_stored_19
label_workspace_18:
s_mul_i32 s75, s61, s36
s_lshl_b32 s75, s75, 2
s_add_u32 s76, s62, 0
v_add_u32 v60, s76, v1
v_cmp_lt_u32 s[64:65], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s39
v_add_u32 v54, s75, v54
s_add_u32 s76, s62, 16
v_add_u32 v60, s76, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s39
v_add_u32 v55, s75, v55
s_add_u32 s76, s63, 0
v_add_u32 v56, s76, v3
v_mul_lo_u32 v58, v56, s40
s_add_u32 s76, s63, 16
v_add_u32 v57, s76, v3
v_mul_lo_u32 v59, v57, s40
s_mov_b32 s68, s40
s_lshl_b32 s69, s40, 1
s_mul_i32 s70, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[76:77], v56, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v12, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v56, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v13, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v56, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v14, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v56, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v15, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[76:77], v57, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v20, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v57, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v21, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v57, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v22, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v57, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[64:65]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v23, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[76:77], v56, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v16, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v56, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v17, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v56, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v18, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v56, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v19, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[76:77], v57, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v24, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s68, v37
v_cmp_lt_i32 s[76:77], v57, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v25, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s69, v37
v_cmp_lt_i32 s[76:77], v57, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v26, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[76:77], v57, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v37, -1, v37, s[76:77]
buffer_store_dword v27, v37, s[24:27], 0 offen offset:0
label_stored_19:
s_branch label_epilogue_end_17
label_orientation_b_16:
s_add_u32 s65, s60, 1
s_lshl_b32 s65, s65, 5
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
v_pk_mul_f32 v[20:21], s[32:33], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], s[32:33], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], s[32:33], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], s[32:33], v[26:27] op_sel_hi:[0,1]
s_and_b32 s65, s3, 2147483647
s_cmp_eq_u32 s65, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s64, 1
s_cbranch_scc0 label_c_masked_24
v_lshlrev_b32 v54, 16, v38
v_fmac_f32 v12, s3, v54
v_and_b32 v54, -65536, v38
v_fmac_f32 v13, s3, v54
v_lshlrev_b32 v54, 16, v39
v_fmac_f32 v14, s3, v54
v_and_b32 v54, -65536, v39
v_fmac_f32 v15, s3, v54
v_lshlrev_b32 v54, 16, v40
v_fmac_f32 v20, s3, v54
v_and_b32 v54, -65536, v40
v_fmac_f32 v21, s3, v54
v_lshlrev_b32 v54, 16, v41
v_fmac_f32 v22, s3, v54
v_and_b32 v54, -65536, v41
v_fmac_f32 v23, s3, v54
v_lshlrev_b32 v54, 16, v42
v_fmac_f32 v16, s3, v54
v_and_b32 v54, -65536, v42
v_fmac_f32 v17, s3, v54
v_lshlrev_b32 v54, 16, v43
v_fmac_f32 v18, s3, v54
v_and_b32 v54, -65536, v43
v_fmac_f32 v19, s3, v54
v_lshlrev_b32 v54, 16, v44
v_fmac_f32 v24, s3, v54
v_and_b32 v54, -65536, v44
v_fmac_f32 v25, s3, v54
v_lshlrev_b32 v54, 16, v45
v_fmac_f32 v26, s3, v54
v_and_b32 v54, -65536, v45
v_fmac_f32 v27, s3, v54
s_branch label_c_added_25
label_c_masked_24:
v_lshlrev_b32 v54, 16, v38
v_fmac_f32 v12, s3, v54
v_lshlrev_b32 v54, 16, v39
v_fmac_f32 v13, s3, v54
v_lshlrev_b32 v54, 16, v40
v_fmac_f32 v14, s3, v54
v_lshlrev_b32 v54, 16, v41
v_fmac_f32 v15, s3, v54
v_lshlrev_b32 v54, 16, v46
v_fmac_f32 v20, s3, v54
v_lshlrev_b32 v54, 16, v47
v_fmac_f32 v21, s3, v54
v_lshlrev_b32 v54, 16, v48
v_fmac_f32 v22, s3, v54
v_lshlrev_b32 v54, 16, v49
v_fmac_f32 v23, s3, v54
v_lshlrev_b32 v54, 16, v42
v_fmac_f32 v16, s3, v54
v_lshlrev_b32 v54, 16, v43
v_fmac_f32 v17, s3, v54
v_lshlrev_b32 v54, 16, v44
v_fmac_f32 v18, s3, v54
v_lshlrev_b32 v54, 16, v45
v_fmac_f32 v19, s3, v54
v_lshlrev_b32 v54, 16, v50
v_fmac_f32 v24, s3, v54
v_lshlrev_b32 v54, 16, v51
v_fmac_f32 v25, s3, v54
v_lshlrev_b32 v54, 16, v52
v_fmac_f32 v26, s3, v54
v_lshlrev_b32 v54, 16, v53
v_fmac_f32 v27, s3, v54
label_c_added_25:
label_no_beta_23:
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_vector_26
s_add_u32 s65, s62, 0
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s37
s_add_u32 s65, s62, 16
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s37
s_add_u32 s65, s63, 0
v_add_u32 v56, s65, v3
v_mul_lo_u32 v58, v56, s38
s_add_u32 s65, s63, 16
v_add_u32 v57, s65, v3
v_mul_lo_u32 v59, v57, s38
s_mov_b32 s70, s38
s_lshl_b32 s71, s38, 1
s_mul_i32 s72, s38, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v12, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v13, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v37, v54, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v14, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v37, v54, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v15, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v20, 16, v20
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v20, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v21, 16, v21
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v21, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v22, 16, v22
v_add_u32 v37, v54, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v22, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v23, 16, v23
v_add_u32 v37, v54, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v23, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v16, 16, v16
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v16, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v17, 16, v17
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v17, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v18, 16, v18
v_add_u32 v37, v55, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v18, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v19, 16, v19
v_add_u32 v37, v55, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v19, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v24, 16, v24
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v24, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v25, 16, v25
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v25, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v26, 16, v26
v_add_u32 v37, v55, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v26, v37, s[20:23], 0 offen offset:0
v_lshrrev_b32 v27, 16, v27
v_add_u32 v37, v55, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_short v27, v37, s[20:23], 0 offen offset:0
s_branch label_tile_stored_27
label_vector_26:
s_add_u32 s65, s62, 0
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s37
s_add_u32 s65, s62, 16
v_add_u32 v60, s65, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s37
s_add_u32 s65, s63, 0
v_add_u32 v56, s65, v3
v_mul_lo_u32 v58, v56, s38
s_add_u32 s65, s63, 16
v_add_u32 v57, s65, v3
v_mul_lo_u32 v59, v57, s38
s_mov_b32 s70, s38
s_lshl_b32 s71, s38, 1
s_mul_i32 s72, s38, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v37, v54, v58
v_lshrrev_b32 v12, 16, v12
v_lshrrev_b32 v13, 16, v13
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_pack_b32_f16 v12, v12, v13
v_pack_b32_f16 v13, v14, v15
buffer_store_dwordx2 v[12:13], v37, s[20:23], 0 offen offset:0
v_add_u32 v37, v54, v59
v_lshrrev_b32 v20, 16, v20
v_lshrrev_b32 v21, 16, v21
v_lshrrev_b32 v22, 16, v22
v_lshrrev_b32 v23, 16, v23
v_pack_b32_f16 v20, v20, v21
v_pack_b32_f16 v21, v22, v23
buffer_store_dwordx2 v[20:21], v37, s[20:23], 0 offen offset:0
v_add_u32 v37, v55, v58
v_lshrrev_b32 v16, 16, v16
v_lshrrev_b32 v17, 16, v17
v_lshrrev_b32 v18, 16, v18
v_lshrrev_b32 v19, 16, v19
v_pack_b32_f16 v16, v16, v17
v_pack_b32_f16 v17, v18, v19
buffer_store_dwordx2 v[16:17], v37, s[20:23], 0 offen offset:0
v_add_u32 v37, v55, v59
v_lshrrev_b32 v24, 16, v24
v_lshrrev_b32 v25, 16, v25
v_lshrrev_b32 v26, 16, v26
v_lshrrev_b32 v27, 16, v27
v_pack_b32_f16 v24, v24, v25
v_pack_b32_f16 v25, v26, v27
buffer_store_dwordx2 v[24:25], v37, s[20:23], 0 offen offset:0
label_tile_stored_27:
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s65, s61, s36
s_lshl_b32 s65, s65, 2
s_cmp_eq_u32 s64, 1
s_cbranch_scc1 label_vector_28
s_add_u32 s77, s62, 0
v_add_u32 v60, s77, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s39
v_add_u32 v54, s65, v54
s_add_u32 s77, s62, 16
v_add_u32 v60, s77, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s39
v_add_u32 v55, s65, v55
s_add_u32 s77, s63, 0
v_add_u32 v56, s77, v3
v_mul_lo_u32 v58, v56, s40
s_add_u32 s77, s63, 16
v_add_u32 v57, s77, v3
v_mul_lo_u32 v59, v57, s40
s_mov_b32 s70, s40
s_lshl_b32 s71, s40, 1
s_mul_i32 s72, s40, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v37, v54, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v12, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v13, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v14, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v15, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v20, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v21, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v22, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[66:67]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v23, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_cmp_lt_i32 s[78:79], v56, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v16, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v56, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v17, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v56, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v18, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v56, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v19, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_cmp_lt_i32 s[78:79], v57, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v24, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s70, v37
v_cmp_lt_i32 s[78:79], v57, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v25, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s71, v37
v_cmp_lt_i32 s[78:79], v57, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v26, v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
v_add_u32 v37, s72, v37
v_cmp_lt_i32 s[78:79], v57, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v37, -1, v37, s[78:79]
buffer_store_dword v27, v37, s[24:27], 0 offen offset:0
s_branch label_tile_stored_29
label_vector_28:
s_add_u32 s77, s62, 0
v_add_u32 v60, s77, v1
v_cmp_lt_u32 s[66:67], v60, s43
s_nop 1
v_mul_lo_u32 v54, v60, s39
v_add_u32 v54, s65, v54
s_add_u32 s77, s62, 16
v_add_u32 v60, s77, v1
v_cmp_lt_u32 s[68:69], v60, s43
s_nop 1
v_mul_lo_u32 v55, v60, s39
v_add_u32 v55, s65, v55
s_add_u32 s77, s63, 0
v_add_u32 v56, s77, v3
v_mul_lo_u32 v58, v56, s40
s_add_u32 s77, s63, 16
v_add_u32 v57, s77, v3
v_mul_lo_u32 v59, v57, s40
s_mov_b32 s70, s40
s_lshl_b32 s71, s40, 1
s_mul_i32 s72, s40, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v37, v54, v58
buffer_store_dwordx4 v[12:15], v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v54, v59
buffer_store_dwordx4 v[20:23], v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v58
buffer_store_dwordx4 v[16:19], v37, s[24:27], 0 offen offset:0
v_add_u32 v37, v55, v59
buffer_store_dwordx4 v[24:27], v37, s[24:27], 0 offen offset:0
label_tile_stored_29:
label_stored_22:
label_epilogue_end_17:
label_tile_end_14:
s_add_u32 s2, s2, s46
s_cmp_lt_i32 s2, s53
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 128
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr 80
  .amdhsa_group_segment_fixed_size 102400
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
    macrotile: [32, 2, 4096]
    threads: [128, 1, 1]
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
  - .name: gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM32_TN2_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF3_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 102400
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .sgpr_count: 86
    .vgpr_count: 128
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
