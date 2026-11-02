.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1:
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
s_lshl_b32 s52, s51, 15
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
s_lshl_b32 s53, s53, 10
s_add_u32 s53, s53, 65536
v_lshlrev_b32 v11, 4, v5
v_add_u32 v11, s53, v11
s_add_u32 s54, s44, 15
s_lshr_b32 s54, s54, 4
s_add_u32 s56, s43, 15
s_lshr_b32 s56, s56, 4
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
s_lshl_b32 s62, s60, 4
s_lshl_b32 s63, s50, 4
s_add_u32 s66, s62, 0
v_add_u32 v21, s66, v6
v_cmp_lt_u32 s[68:69], v21, s43
s_nop 1
v_mul_lo_u32 v18, v21, s48
v_add_u32 v18, v18, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v18, v26, v18, s[68:69]
s_add_u32 s66, s62, 8
v_add_u32 v21, s66, v6
v_cmp_lt_u32 s[68:69], v21, s43
s_nop 1
v_mul_lo_u32 v19, v21, s48
v_add_u32 v19, v19, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v19, v26, v19, s[68:69]
s_add_u32 s66, s63, 0
v_add_u32 v21, s66, v6
v_cmp_lt_u32 s[68:69], v21, s44
s_nop 1
v_mul_lo_u32 v16, v21, s49
v_add_u32 v16, v16, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v16, v26, v16, s[68:69]
s_add_u32 s66, s63, 8
v_add_u32 v21, s66, v6
v_cmp_lt_u32 s[68:69], v21, s44
s_nop 1
v_mul_lo_u32 v17, v21, s49
v_add_u32 v17, v17, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v17, v26, v17, s[68:69]
s_lshr_b32 s66, s62, 5
s_mul_i32 s66, s66, s47
v_add_u32 v20, s66, v4
s_and_b32 s65, 16, s62
s_lshr_b32 s65, s65, 1
s_add_u32 s66, s63, 0
s_lshr_b32 s67, s66, 5
s_mul_i32 s67, s67, s47
v_add_u32 v5, s67, v4
s_and_b32 s64, 16, s66
s_lshr_b32 s64, s64, 1
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
s_nop 7
s_cmp_eq_u32 s51, 0
s_cbranch_scc0 label_c_prefetch_skip_2
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_c_prefetch_skip_2
s_and_b32 s66, s3, 2147483647
s_cmp_eq_u32 s66, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s67, s62, 0
v_add_u32 v29, s67, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s41
s_add_u32 s67, s63, 0
v_add_u32 v27, s67, v3
v_mul_lo_u32 v28, v27, s42
s_mov_b32 s70, s42
s_lshl_b32 s71, s42, 1
s_mul_i32 s72, s42, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[78:79], v27, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v22, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[78:79], v27, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v23, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s71, v21
v_cmp_lt_i32 s[78:79], v27, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v24, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s72, v21
v_cmp_lt_i32 s[78:79], v27, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v25, v21, s[28:31], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s67, s60, 1
s_lshl_b32 s67, s67, 4
s_add_u32 s68, s50, 1
s_lshl_b32 s68, s68, 4
s_cmp_le_u32 s67, s43
s_cselect_b32 s69, 1, 0
s_cmp_le_u32 s68, s44
s_cselect_b32 s70, 1, 0
s_and_b32 s66, s69, s70
s_add_u32 s67, s62, 0
v_add_u32 v29, s67, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s41
s_add_u32 s67, s63, 0
v_add_u32 v27, s67, v3
v_mul_lo_u32 v28, v27, s42
s_mov_b32 s70, s42
s_lshl_b32 s71, s42, 1
s_mul_i32 s72, s42, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
s_cmp_eq_u32 s66, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[78:79], v27, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v22, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[78:79], v27, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v23, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s71, v21
v_cmp_lt_i32 s[78:79], v27, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v24, v21, s[28:31], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s72, v21
v_cmp_lt_i32 s[78:79], v27, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_load_ushort v25, v21, s[28:31], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v21, v26, v28
buffer_load_dwordx2 v[22:23], v21, s[28:31], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_mul_i32 s68, s61, s35
s_sub_u32 s69, s34, s68
s_min_i32 s69, s69, s35
s_lshr_b32 s69, s69, 8
s_add_u32 s70, s69, 1
s_sub_u32 s70, s70, s51
s_lshr_b32 s66, s70, 1
s_lshl_b32 s67, s51, 8
s_add_u32 s67, s67, s68
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
buffer_load_dword v26, v5, s[16:19], s67 offen offset:0
buffer_load_dword v27, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
buffer_load_dword v28, v5, s[16:19], s67 offen offset:0
buffer_load_dword v29, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
buffer_load_dword v30, v5, s[16:19], s67 offen offset:0
buffer_load_dword v31, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_add_u32 m0, s52, 24576
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 28672
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 25600
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 29696
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 26496
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 30592
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 27520
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 31616
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
buffer_load_dword v32, v5, s[16:19], s67 offen offset:0
buffer_load_dword v33, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_ashr_i32 s68, s66, 2
s_lshl_b32 s68, s68, 2
s_add_u32 s68, s68, -4
s_max_i32 s68, s68, 0
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
s_cmp_gt_i32 s68, 0
s_mov_b32 s69, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[16:19], s67 offen offset:0
buffer_load_dword v27, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[16:19], s67 offen offset:0
buffer_load_dword v29, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v30, v5, s[16:19], s67 offen offset:0
buffer_load_dword v31, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_lshrrev_b32 v32, s64, v32
v_lshrrev_b32 v33, s65, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 24576
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 28672
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 25600
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 29696
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 26496
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 30592
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 27520
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 31616
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v32, v5, s[16:19], s67 offen offset:0
buffer_load_dword v33, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_add_u32 s69, s69, 4
s_cmp_lt_i32 s69, s68
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s69, s66, s68
s_cmp_eq_i32 s69, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s69, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s69, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s69, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_lshrrev_b32 v32, s64, v32
v_lshrrev_b32 v33, s65, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s69, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[16:19], s67 offen offset:0
buffer_load_dword v27, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_lshrrev_b32 v32, s64, v32
v_lshrrev_b32 v33, s65, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_cmp_eq_i32 s69, 6
s_cbranch_scc0 label_arm_end_14
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[16:19], s67 offen offset:0
buffer_load_dword v27, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[16:19], s67 offen offset:0
buffer_load_dword v29, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_lshrrev_b32 v32, s64, v32
v_lshrrev_b32 v33, s65, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_14:
s_cmp_eq_i32 s69, 7
s_cbranch_scc0 label_arm_end_15
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[16:19], s67 offen offset:0
buffer_load_dword v27, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[16:19], s67 offen offset:0
buffer_load_dword v29, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s52, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:0 lds
s_add_u32 m0, s52, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:0 lds
s_add_u32 m0, s52, 18304
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s67 offen offset:128 lds
s_add_u32 m0, s52, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s67 offen offset:128 lds
s_add_u32 m0, s52, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s67 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v30, v5, s[16:19], s67 offen offset:0
buffer_load_dword v31, v20, s[12:15], s67 offen offset:0
s_add_u32 s67, s67, 512
s_waitcnt vmcnt(30)
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_lshrrev_b32 v32, s64, v32
v_lshrrev_b32 v33, s65, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_lshrrev_b32 v26, s64, v26
v_lshrrev_b32 v27, s65, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_lshrrev_b32 v28, s64, v28
v_lshrrev_b32 v29, s65, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_lshrrev_b32 v30, s64, v30
v_lshrrev_b32 v31, s65, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_15:
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_nop 15
s_nop 15
s_cmp_eq_u32 s51, 0
s_cbranch_scc1 label_wave_zero_17
ds_write_b128 v11, v[12:15] offset:0
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_tile_end_16
label_wave_zero_17:
s_barrier
ds_read_b128 v[26:29], v11 offset:0
s_waitcnt lgkmcnt(0)
v_add_f32 v12, v12, v26
v_add_f32 v13, v13, v27
v_add_f32 v14, v14, v28
v_add_f32 v15, v15, v29
s_cmp_eq_u32 s45, 1
s_cbranch_scc0 label_orientation_b_18
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_20
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
s_and_b32 s66, s3, 2147483647
s_cmp_eq_u32 s66, 0
s_cbranch_scc1 label_no_beta_22
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_no_beta_22:
s_add_u32 s75, s62, 0
v_add_u32 v29, s75, v1
v_cmp_lt_u32 s[66:67], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s37
s_add_u32 s75, s63, 0
v_add_u32 v27, s75, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s68, s38
s_lshl_b32 s69, s38, 1
s_mul_i32 s70, s38, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s69, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_stored_21
label_workspace_20:
s_mul_i32 s75, s61, s36
s_lshl_b32 s75, s75, 2
s_add_u32 s76, s62, 0
v_add_u32 v29, s76, v1
v_cmp_lt_u32 s[66:67], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s75, v26
s_add_u32 s76, s63, 0
v_add_u32 v27, s76, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s68, s40
s_lshl_b32 s69, s40, 1
s_mul_i32 s70, s40, 3
s_sub_u32 s71, s44, 0
s_sub_u32 s72, s44, 1
s_sub_u32 s73, s44, 2
s_sub_u32 s74, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[76:77], v27, s71
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v12, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s68, v21
v_cmp_lt_i32 s[76:77], v27, s72
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v13, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s69, v21
v_cmp_lt_i32 s[76:77], v27, s73
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v14, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[76:77], v27, s74
s_nop 1
s_and_b64 s[76:77], s[76:77], s[66:67]
v_cndmask_b32 v21, -1, v21, s[76:77]
buffer_store_dword v15, v21, s[24:27], 0 offen offset:0
label_stored_21:
s_branch label_epilogue_end_19
label_orientation_b_18:
s_add_u32 s67, s60, 1
s_lshl_b32 s67, s67, 4
s_add_u32 s68, s50, 1
s_lshl_b32 s68, s68, 4
s_cmp_le_u32 s67, s43
s_cselect_b32 s69, 1, 0
s_cmp_le_u32 s68, s44
s_cselect_b32 s70, 1, 0
s_and_b32 s66, s69, s70
s_cmp_le_u32 s34, s35
s_cbranch_scc0 label_workspace_23
v_pk_mul_f32 v[12:13], s[32:33], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[32:33], v[14:15] op_sel_hi:[0,1]
s_and_b32 s67, s3, 2147483647
s_cmp_eq_u32 s67, 0
s_cbranch_scc1 label_no_beta_25
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s66, 1
s_cbranch_scc0 label_c_masked_26
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_and_b32 v26, -65536, v22
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v14, s3, v26
v_and_b32 v26, -65536, v23
v_fmac_f32 v15, s3, v26
s_branch label_c_added_27
label_c_masked_26:
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_c_added_27:
label_no_beta_25:
s_cmp_eq_u32 s66, 1
s_cbranch_scc1 label_vector_28
s_add_u32 s67, s62, 0
v_add_u32 v29, s67, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s37
s_add_u32 s67, s63, 0
v_add_u32 v27, s67, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s70, s38
s_lshl_b32 s71, s38, 1
s_mul_i32 s72, s38, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[78:79], v27, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[78:79], v27, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s71, v21
v_cmp_lt_i32 s[78:79], v27, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s72, v21
v_cmp_lt_i32 s[78:79], v27, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_tile_stored_29
label_vector_28:
s_add_u32 s67, s62, 0
v_add_u32 v29, s67, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s37
s_add_u32 s67, s63, 0
v_add_u32 v27, s67, v3
v_mul_lo_u32 v28, v27, s38
s_mov_b32 s70, s38
s_lshl_b32 s71, s38, 1
s_mul_i32 s72, s38, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v21, v26, v28
v_lshrrev_b32 v12, 16, v12
v_lshrrev_b32 v13, 16, v13
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_pack_b32_f16 v12, v12, v13
v_pack_b32_f16 v13, v14, v15
buffer_store_dwordx2 v[12:13], v21, s[20:23], 0 offen offset:0
label_tile_stored_29:
s_branch label_stored_24
label_workspace_23:
s_mul_i32 s67, s61, s36
s_lshl_b32 s67, s67, 2
s_cmp_eq_u32 s66, 1
s_cbranch_scc1 label_vector_30
s_add_u32 s77, s62, 0
v_add_u32 v29, s77, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s67, v26
s_add_u32 s77, s63, 0
v_add_u32 v27, s77, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s70, s40
s_lshl_b32 s71, s40, 1
s_mul_i32 s72, s40, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[78:79], v27, s73
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_dword v12, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s70, v21
v_cmp_lt_i32 s[78:79], v27, s74
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_dword v13, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s71, v21
v_cmp_lt_i32 s[78:79], v27, s75
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_dword v14, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s72, v21
v_cmp_lt_i32 s[78:79], v27, s76
s_nop 1
s_and_b64 s[78:79], s[78:79], s[68:69]
v_cndmask_b32 v21, -1, v21, s[78:79]
buffer_store_dword v15, v21, s[24:27], 0 offen offset:0
s_branch label_tile_stored_31
label_vector_30:
s_add_u32 s77, s62, 0
v_add_u32 v29, s77, v1
v_cmp_lt_u32 s[68:69], v29, s43
s_nop 1
v_mul_lo_u32 v26, v29, s39
v_add_u32 v26, s67, v26
s_add_u32 s77, s63, 0
v_add_u32 v27, s77, v3
v_mul_lo_u32 v28, v27, s40
s_mov_b32 s70, s40
s_lshl_b32 s71, s40, 1
s_mul_i32 s72, s40, 3
s_sub_u32 s73, s44, 0
s_sub_u32 s74, s44, 1
s_sub_u32 s75, s44, 2
s_sub_u32 s76, s44, 3
v_add_u32 v21, v26, v28
buffer_store_dwordx4 v[12:15], v21, s[24:27], 0 offen offset:0
label_tile_stored_31:
label_stored_24:
label_epilogue_end_19:
label_tile_end_16:
s_add_u32 s2, s2, s46
s_cmp_lt_i32 s2, s53
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 72
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr 80
  .amdhsa_group_segment_fixed_size 66560
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
    macrotile: [16, 1, 4096]
    threads: [128, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
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
  - .name: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF4_GROUP8_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 66560
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
...
.end_amdgpu_metadata
