.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1:
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
s_mul_i32 s54, s53, 49152
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
s_add_u32 s55, s54, 0
v_add_u32 v9, s55, v14
v_xor_b32 v10, 64, v9
s_lshr_b32 s55, s2, 3
s_and_b32 s55, 1, s55
s_and_b32 s56, 7, s2
s_lshl_b32 s56, s56, 1
s_add_u32 s55, s55, s56
s_and_b32 s56, -16, s2
s_add_u32 s55, s55, s56
s_and_b32 s56, -16, s48
s_cmp_lt_u32 s2, s56
s_cselect_b32 s2, s55, s2
s_lshl_b32 s55, s3, 4
s_lshl_b32 s56, s2, 4
s_add_u32 s59, s55, 0
v_add_u32 v24, s59, v6
v_cmp_lt_u32 s[60:61], v24, s45
s_nop 1
v_mul_lo_u32 v18, v24, s51
v_add_u32 v18, v18, v7
v_mov_b32 v25, -2147483648
v_cndmask_b32 v18, v25, v18, s[60:61]
s_add_u32 s59, s55, 8
v_add_u32 v24, s59, v6
v_cmp_lt_u32 s[60:61], v24, s45
s_nop 1
v_mul_lo_u32 v19, v24, s51
v_add_u32 v19, v19, v8
v_mov_b32 v25, -2147483648
v_cndmask_b32 v19, v25, v19, s[60:61]
s_add_u32 s59, s56, 0
v_add_u32 v24, s59, v6
v_cmp_lt_u32 s[60:61], v24, s46
s_nop 1
v_mul_lo_u32 v16, v24, s52
v_add_u32 v16, v16, v7
v_mov_b32 v25, -2147483648
v_cndmask_b32 v16, v25, v16, s[60:61]
s_add_u32 s59, s56, 8
v_add_u32 v24, s59, v6
v_cmp_lt_u32 s[60:61], v24, s46
s_nop 1
v_mul_lo_u32 v17, v24, s52
v_add_u32 v17, v17, v8
v_mov_b32 v25, -2147483648
v_cndmask_b32 v17, v25, v17, s[60:61]
s_lshr_b32 s59, s55, 5
s_mul_i32 s59, s59, s50
v_add_u32 v11, s59, v4
s_and_b32 s58, 16, s55
s_lshr_b32 s58, s58, 1
s_add_u32 s59, s56, 0
s_lshr_b32 s60, s59, 5
s_mul_i32 s60, s60, s50
v_add_u32 v5, s60, v4
s_and_b32 s57, 16, s59
s_lshr_b32 s57, s57, 1
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
s_nop 7
s_cmp_eq_u32 s53, 0
s_cbranch_scc0 label_c_prefetch_skip_0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s59, s5, 2147483647
s_cmp_eq_u32 s59, 0
s_cbranch_scc1 label_c_prefetch_skip_0
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_1
s_add_u32 s69, s55, 0
v_add_u32 v28, s69, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s43
s_add_u32 s69, s56, 0
v_add_u32 v26, s69, v3
v_mul_lo_u32 v27, v26, s44
s_mov_b32 s62, s44
s_lshl_b32 s63, s44, 1
s_mul_i32 s64, s44, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v20, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v21, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v22, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v23, v24, s[32:35], 0 offen offset:0
s_branch label_epilogue_end_2
label_orientation_b_1:
s_add_u32 s60, s3, 1
s_lshl_b32 s60, s60, 4
s_add_u32 s61, s2, 1
s_lshl_b32 s61, s61, 4
s_cmp_le_u32 s60, s45
s_cselect_b32 s62, 1, 0
s_cmp_le_u32 s61, s46
s_cselect_b32 s63, 1, 0
s_and_b32 s59, s62, s63
s_add_u32 s69, s55, 0
v_add_u32 v28, s69, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s43
s_add_u32 s69, s56, 0
v_add_u32 v26, s69, v3
v_mul_lo_u32 v27, v26, s44
s_mov_b32 s62, s44
s_lshl_b32 s63, s44, 1
s_mul_i32 s64, s44, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
s_cmp_eq_u32 s59, 1
s_cbranch_scc1 label_c_vector_3
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v20, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v21, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v22, v24, s[32:35], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_load_ushort v23, v24, s[32:35], 0 offen offset:0
s_branch label_c_loaded_4
label_c_vector_3:
v_add_u32 v24, v25, v27
buffer_load_dwordx2 v[20:21], v24, s[32:35], 0 offen offset:0
label_c_loaded_4:
label_epilogue_end_2:
label_c_prefetch_skip_0:
s_mul_i32 s61, s4, s37
s_sub_u32 s62, s36, s61
s_min_i32 s62, s62, s37
s_lshr_b32 s62, s62, 8
s_add_u32 s63, s62, 0
s_sub_u32 s63, s63, s53
s_mov_b32 s59, s63
s_lshl_b32 s60, s53, 8
s_add_u32 s60, s60, s61
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v28, v5, s[20:23], s60 offen offset:0
buffer_load_dword v29, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v30, v5, s[20:23], s60 offen offset:0
buffer_load_dword v31, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v32, v5, s[20:23], s60 offen offset:0
buffer_load_dword v33, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 m0, s54, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 41984
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 42880
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 43904
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
buffer_load_dword v34, v5, s[20:23], s60 offen offset:0
buffer_load_dword v35, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_mul_hi_i32 s61, s59, 715827883
s_lshr_b32 s62, s61, 31
s_add_u32 s61, s61, s62
s_mul_i32 s61, s61, 6
s_add_u32 s61, s61, -6
s_max_i32 s61, s61, 0
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
s_cmp_gt_i32 s61, 0
s_mov_b32 s62, 0
s_cbranch_scc0 label_kloop_end_6
.p2align 4
label_kloop_5:
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[20:23], s60 offen offset:0
buffer_load_dword v29, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v30, v5, s[20:23], s60 offen offset:0
buffer_load_dword v31, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v32, v5, s[20:23], s60 offen offset:0
buffer_load_dword v33, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 40960
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 45056
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 41984
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 46080
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 42880
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 46976
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 43904
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 48000
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v34, v5, s[20:23], s60 offen offset:0
buffer_load_dword v35, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_add_u32 s62, s62, 6
s_cmp_lt_i32 s62, s61
s_cbranch_scc1 label_kloop_5
label_kloop_end_6:
s_sub_u32 s62, s59, s61
s_cmp_eq_i32 s62, 1
s_cbranch_scc0 label_arm_end_7
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_7:
s_cmp_eq_i32 s62, 2
s_cbranch_scc0 label_arm_end_8
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_8:
s_cmp_eq_i32 s62, 3
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s62, 4
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s62, 5
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s62, 6
s_cbranch_scc0 label_arm_end_12
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s62, 7
s_cbranch_scc0 label_arm_end_13
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_cmp_eq_i32 s62, 8
s_cbranch_scc0 label_arm_end_14
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_14:
s_cmp_eq_i32 s62, 9
s_cbranch_scc0 label_arm_end_15
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[20:23], s60 offen offset:0
buffer_load_dword v29, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_15:
s_cmp_eq_i32 s62, 10
s_cbranch_scc0 label_arm_end_16
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[20:23], s60 offen offset:0
buffer_load_dword v29, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v30, v5, s[20:23], s60 offen offset:0
buffer_load_dword v31, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_16:
s_cmp_eq_i32 s62, 11
s_cbranch_scc0 label_arm_end_17
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 4096
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1024
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 5120
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 1920
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 6016
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 2944
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 7040
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v24, v5, s[20:23], s60 offen offset:0
buffer_load_dword v25, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 8192
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 12288
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 9216
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 13312
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 10112
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 14208
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 11136
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 15232
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v26, v5, s[20:23], s60 offen offset:0
buffer_load_dword v27, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 16384
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 20480
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 17408
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 21504
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 18304
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 22400
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 19328
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 23424
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v28, v5, s[20:23], s60 offen offset:0
buffer_load_dword v29, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 24576
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 28672
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 25600
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 29696
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 26496
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 30592
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 27520
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 31616
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v30, v5, s[20:23], s60 offen offset:0
buffer_load_dword v31, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s54, 32768
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 36864
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 33792
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:0 lds
s_add_u32 m0, s54, 37888
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:0 lds
s_add_u32 m0, s54, 34688
s_nop 0
buffer_load_dwordx4 v16, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 38784
s_nop 0
buffer_load_dwordx4 v18, s[8:11], s60 offen offset:128 lds
s_add_u32 m0, s54, 35712
s_nop 0
buffer_load_dwordx4 v17, s[12:15], s60 offen offset:128 lds
s_add_u32 m0, s54, 39808
s_nop 0
buffer_load_dwordx4 v19, s[8:11], s60 offen offset:128 lds
s_waitcnt vmcnt(54)
ds_read_b128 v[36:39], v9 offset:40960
ds_read_b128 v[40:43], v10 offset:40960
ds_read_b128 v[44:47], v9 offset:45056
ds_read_b128 v[48:51], v10 offset:45056
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dword v32, v5, s[20:23], s60 offen offset:0
buffer_load_dword v33, v11, s[16:19], s60 offen offset:0
s_add_u32 s60, s60, 256
s_waitcnt vmcnt(50)
ds_read_b128 v[52:55], v9 offset:43008
ds_read_b128 v[56:59], v10 offset:43008
ds_read_b128 v[60:63], v9 offset:47104
ds_read_b128 v[64:67], v10 offset:47104
v_lshrrev_b32 v34, s57, v34
v_lshrrev_b32 v35, s58, v35
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(46)
ds_read_b128 v[36:39], v9 offset:0
ds_read_b128 v[40:43], v10 offset:0
ds_read_b128 v[44:47], v9 offset:4096
ds_read_b128 v[48:51], v10 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(40)
ds_read_b128 v[52:55], v9 offset:2048
ds_read_b128 v[56:59], v10 offset:2048
ds_read_b128 v[60:63], v9 offset:6144
ds_read_b128 v[64:67], v10 offset:6144
v_lshrrev_b32 v24, s57, v24
v_lshrrev_b32 v25, s58, v25
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(36)
ds_read_b128 v[36:39], v9 offset:8192
ds_read_b128 v[40:43], v10 offset:8192
ds_read_b128 v[44:47], v9 offset:12288
ds_read_b128 v[48:51], v10 offset:12288
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v24, v25 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(30)
ds_read_b128 v[52:55], v9 offset:10240
ds_read_b128 v[56:59], v10 offset:10240
ds_read_b128 v[60:63], v9 offset:14336
ds_read_b128 v[64:67], v10 offset:14336
v_lshrrev_b32 v26, s57, v26
v_lshrrev_b32 v27, s58, v27
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(26)
ds_read_b128 v[36:39], v9 offset:16384
ds_read_b128 v[40:43], v10 offset:16384
ds_read_b128 v[44:47], v9 offset:20480
ds_read_b128 v[48:51], v10 offset:20480
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(20)
ds_read_b128 v[52:55], v9 offset:18432
ds_read_b128 v[56:59], v10 offset:18432
ds_read_b128 v[60:63], v9 offset:22528
ds_read_b128 v[64:67], v10 offset:22528
v_lshrrev_b32 v28, s57, v28
v_lshrrev_b32 v29, s58, v29
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(16)
ds_read_b128 v[36:39], v9 offset:24576
ds_read_b128 v[40:43], v10 offset:24576
ds_read_b128 v[44:47], v9 offset:28672
ds_read_b128 v[48:51], v10 offset:28672
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(10)
ds_read_b128 v[52:55], v9 offset:26624
ds_read_b128 v[56:59], v10 offset:26624
ds_read_b128 v[60:63], v9 offset:30720
ds_read_b128 v[64:67], v10 offset:30720
v_lshrrev_b32 v30, s57, v30
v_lshrrev_b32 v31, s58, v31
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(6)
ds_read_b128 v[36:39], v9 offset:32768
ds_read_b128 v[40:43], v10 offset:32768
ds_read_b128 v[44:47], v9 offset:36864
ds_read_b128 v[48:51], v10 offset:36864
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(0)
ds_read_b128 v[52:55], v9 offset:34816
ds_read_b128 v[56:59], v10 offset:34816
ds_read_b128 v[60:63], v9 offset:38912
ds_read_b128 v[64:67], v10 offset:38912
v_lshrrev_b32 v32, s57, v32
v_lshrrev_b32 v33, s58, v33
s_nop 7
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[36:43], v[44:51], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[52:59], v[60:67], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_17:
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_nop 15
s_nop 15
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_19
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_21
v_pk_mul_f32 v[12:13], s[6:7], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
s_and_b32 s59, s5, 2147483647
s_cmp_eq_u32 s59, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v22
v_fmac_f32 v14, s5, v25
v_lshlrev_b32 v25, 16, v23
v_fmac_f32 v15, s5, v25
label_no_beta_23:
s_add_u32 s59, s55, 0
v_add_u32 v28, s59, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s39
s_add_u32 s59, s56, 0
v_add_u32 v26, s59, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v12, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v13, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v14, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v15, v24, s[24:27], 0 offen offset:0
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s59, s4, s38
s_lshl_b32 s59, s59, 2
s_add_u32 s69, s55, 0
v_add_u32 v28, s69, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s59, v25
s_add_u32 s69, s56, 0
v_add_u32 v26, s69, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v12, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v13, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v14, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v15, v24, s[28:31], 0 offen offset:0
label_stored_22:
s_branch label_epilogue_end_20
label_orientation_b_19:
s_add_u32 s60, s3, 1
s_lshl_b32 s60, s60, 4
s_add_u32 s61, s2, 1
s_lshl_b32 s61, s61, 4
s_cmp_le_u32 s60, s45
s_cselect_b32 s62, 1, 0
s_cmp_le_u32 s61, s46
s_cselect_b32 s63, 1, 0
s_and_b32 s59, s62, s63
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_24
v_pk_mul_f32 v[12:13], s[6:7], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[6:7], v[14:15] op_sel_hi:[0,1]
s_and_b32 s60, s5, 2147483647
s_cmp_eq_u32 s60, 0
s_cbranch_scc1 label_no_beta_26
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s59, 1
s_cbranch_scc0 label_c_masked_27
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_and_b32 v25, -65536, v20
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v14, s5, v25
v_and_b32 v25, -65536, v21
v_fmac_f32 v15, s5, v25
s_branch label_c_added_28
label_c_masked_27:
v_lshlrev_b32 v25, 16, v20
v_fmac_f32 v12, s5, v25
v_lshlrev_b32 v25, 16, v21
v_fmac_f32 v13, s5, v25
v_lshlrev_b32 v25, 16, v22
v_fmac_f32 v14, s5, v25
v_lshlrev_b32 v25, 16, v23
v_fmac_f32 v15, s5, v25
label_c_added_28:
label_no_beta_26:
s_cmp_eq_u32 s59, 1
s_cbranch_scc1 label_vector_29
s_add_u32 s69, s55, 0
v_add_u32 v28, s69, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s39
s_add_u32 s69, s56, 0
v_add_u32 v26, s69, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_lshrrev_b32 v12, 16, v12
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v12, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v13, 16, v13
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v13, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v14, 16, v14
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v14, v24, s[24:27], 0 offen offset:0
v_lshrrev_b32 v15, 16, v15
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_short v15, v24, s[24:27], 0 offen offset:0
s_branch label_tile_stored_30
label_vector_29:
s_add_u32 s69, s55, 0
v_add_u32 v28, s69, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s39
s_add_u32 s69, s56, 0
v_add_u32 v26, s69, v3
v_mul_lo_u32 v27, v26, s40
s_mov_b32 s62, s40
s_lshl_b32 s63, s40, 1
s_mul_i32 s64, s40, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v24, v25, v27
v_lshrrev_b32 v12, 16, v12
v_lshrrev_b32 v13, 16, v13
v_lshrrev_b32 v14, 16, v14
v_lshrrev_b32 v15, 16, v15
v_pack_b32_f16 v12, v12, v13
v_pack_b32_f16 v13, v14, v15
buffer_store_dwordx2 v[12:13], v24, s[24:27], 0 offen offset:0
label_tile_stored_30:
s_branch label_stored_25
label_workspace_24:
s_mul_i32 s69, s4, s38
s_lshl_b32 s69, s69, 2
s_cmp_eq_u32 s59, 1
s_cbranch_scc1 label_vector_31
s_add_u32 s70, s55, 0
v_add_u32 v28, s70, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s69, v25
s_add_u32 s70, s56, 0
v_add_u32 v26, s70, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v24, v25, v27
v_cmp_lt_i32 s[70:71], v26, s65
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v12, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s62, v24
v_cmp_lt_i32 s[70:71], v26, s66
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v13, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s63, v24
v_cmp_lt_i32 s[70:71], v26, s67
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v14, v24, s[28:31], 0 offen offset:0
v_add_u32 v24, v25, v27
v_add_u32 v24, s64, v24
v_cmp_lt_i32 s[70:71], v26, s68
s_nop 1
s_and_b64 s[70:71], s[70:71], s[60:61]
v_cndmask_b32 v24, -1, v24, s[70:71]
buffer_store_dword v15, v24, s[28:31], 0 offen offset:0
s_branch label_tile_stored_32
label_vector_31:
s_add_u32 s70, s55, 0
v_add_u32 v28, s70, v1
v_cmp_lt_u32 s[60:61], v28, s45
s_nop 1
v_mul_lo_u32 v25, v28, s41
v_add_u32 v25, s69, v25
s_add_u32 s70, s56, 0
v_add_u32 v26, s70, v3
v_mul_lo_u32 v27, v26, s42
s_mov_b32 s62, s42
s_lshl_b32 s63, s42, 1
s_mul_i32 s64, s42, 3
s_sub_u32 s65, s46, 0
s_sub_u32 s66, s46, 1
s_sub_u32 s67, s46, 2
s_sub_u32 s68, s46, 3
v_add_u32 v24, v25, v27
buffer_store_dwordx4 v[12:15], v24, s[28:31], 0 offen offset:0
label_tile_stored_32:
label_stored_25:
label_epilogue_end_20:
label_tile_end_18:
s_endpgm
.Lgemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 72
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr 73
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
    threads: [64, 1, 1]
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
  - .name: gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 79
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
