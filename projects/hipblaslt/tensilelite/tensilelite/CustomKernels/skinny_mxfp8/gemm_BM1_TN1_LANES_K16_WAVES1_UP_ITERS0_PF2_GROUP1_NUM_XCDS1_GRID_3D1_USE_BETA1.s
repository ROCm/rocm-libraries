.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.globl gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.p2align 8
.type gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1,@function
gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1:
s_load_dword s70, s[0:1], 104
s_load_dword s60, s[0:1], 96
s_load_dword s61, s[0:1], 100
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s69, s[0:1], 72
s_load_dword s68, s[0:1], 68
s_load_dword s67, s[0:1], 64
s_load_dword s66, s[0:1], 60
s_load_dword s65, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[78:79], s[0:1], 32
s_load_dwordx2 s[76:77], s[0:1], 24
s_load_dwordx2 s[74:75], s[0:1], 16
s_load_dwordx2 s[72:73], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s83, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
v_and_b32 v81, 15, v0
v_lshrrev_b32 v1, 4, v0
v_and_b32 v77, 7, v81
v_lshrrev_b32 v77, 1, v77
v_lshlrev_b32 v77, 2, v77
v_lshrrev_b32 v82, 3, v81
v_lshlrev_b32 v82, 5, v82
v_add_u32 v77, v77, v82
v_lshrrev_b32 v82, 4, v0
v_add_u32 v77, v77, v82
v_lshlrev_b32 v77, 2, v77
v_add_u32 v78, 64, v77
v_lshrrev_b32 v82, 3, v81
v_lshlrev_b32 v82, 9, v82
v_and_b32 v83, 7, v81
v_lshlrev_b32 v83, 4, v83
v_add_u32 v82, v82, v83
v_lshrrev_b32 v81, 2, v0
v_and_b32 v83, 3, v81
v_lshlrev_b32 v83, 6, v83
v_lshrrev_b32 v84, 3, v81
v_lshlrev_b32 v84, 9, v84
v_add_u32 v83, v83, v84
v_and_b32 v84, 7, v81
v_lshrrev_b32 v84, 2, v84
v_lshlrev_b32 v84, 8, v84
v_add_u32 v83, v83, v84
v_and_b32 v84, 3, v0
v_mov_b32 v81, v84
v_and_b32 v0, 15, v0
s_waitcnt lgkmcnt(0)
s_add_u32 s84, s36, 3
s_lshr_b32 s84, s84, 2
s_lshl_b32 s84, s84, 2
s_add_u32 s71, s68, -1
s_mul_i32 s71, s71, s66
s_add_u32 s71, s71, s84
s_add_u32 s80, s69, -1
s_mul_i32 s80, s80, s67
s_add_u32 s80, s80, s84
s_add_u32 s62, s36, 255
s_lshr_b32 s62, s62, 8
s_lshl_b32 s62, s62, 8
s_add_u32 s81, s68, 31
s_lshr_b32 s81, s81, 5
s_mul_i32 s81, s81, s62
s_add_u32 s82, s69, 31
s_lshr_b32 s82, s82, 5
s_mul_i32 s82, s82, s62
s_add_u32 s26, s69, -1
s_mul_i32 s26, s26, s65
s_lshl_b32 s84, s68, 1
s_add_u32 s26, s26, s84
s_add_u32 s34, s69, -1
s_mul_i32 s34, s34, s83
s_lshl_b32 s84, s68, 1
s_add_u32 s34, s34, s84
s_mul_i32 s30, s68, s70
s_mul_i32 s30, s30, s69
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s68, s69
s_lshl_b32 s42, s68, 2
s_cmp_le_u32 s68, s69
s_cselect_b64 s[8:9], s[72:73], s[74:75]
s_cselect_b32 s10, s71, s80
s_cselect_b64 s[12:13], s[74:75], s[72:73]
s_cselect_b32 s14, s80, s71
s_cselect_b64 s[16:17], s[76:77], s[78:79]
s_cselect_b32 s18, s81, s82
s_cselect_b64 s[20:21], s[78:79], s[76:77]
s_cselect_b32 s22, s82, s81
s_cselect_b32 s63, s66, s67
s_cselect_b32 s64, s67, s66
s_cselect_b32 s39, 2, s65
s_cselect_b32 s40, s65, 2
s_cselect_b32 s43, 2, s83
s_cselect_b32 s44, s83, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s68, s69
s_max_u32 s46, s68, s69
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
s_mul_i32 s52, s4, s37
s_sub_u32 s49, s36, s52
s_min_i32 s49, s49, s37
s_and_b32 s51, 1023, s49
s_ashr_i32 s49, s49, 10
s_mul_i32 s66, s61, s4
s_add_u32 s66, s66, s3
s_mul_i32 s66, s66, s60
s_add_u32 s66, s66, s2
s_add_u32 s67, s37, 1023
s_lshr_b32 s67, s67, 10
s_lshr_b32 s68, s66, 4
s_add_u32 s68, s68, s66
s_abs_i32 s70, s67
s_abs_i32 s71, s68
s_xor_b32 s72, s68, s67
s_ashr_i32 s72, s72, 31
v_cvt_f32_u32 v84, s70
v_rcp_iflag_f32 v84, v84
s_sub_u32 s73, 0, s70
v_mul_f32 v84, 0x4f7ffffe, v84
v_cvt_u32_f32 v84, v84
s_nop 0
v_readfirstlane_b32 s74, v84
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
s_sub_u32 s69, s73, s72
s_mul_i32 s69, s69, s67
s_sub_u32 s65, s68, s69
s_cmp_lt_u32 s65, s49
s_cselect_b32 s50, s65, 0
s_mov_b32 s48, s3
s_lshr_b32 s65, s2, 3
s_and_b32 s65, 7, s65
s_and_b32 s66, 7, s2
s_lshl_b32 s66, s66, 3
s_add_u32 s65, s65, s66
s_and_b32 s66, -64, s2
s_add_u32 s65, s65, s66
s_and_b32 s66, -64, s60
s_cmp_lt_u32 s2, s66
s_cselect_b32 s2, s65, s2
s_lshl_b32 s65, s2, 2
v_add_u32 v1, s65, v1
v_add_u32 v81, s65, v81
s_add_u32 s65, s48, 0
s_cmp_lt_u32 s65, s45
s_cselect_b64 s[56:57], -1, 0
s_mul_i32 s65, s65, s63
v_add_u32 v71, s65, v82
v_cndmask_b32 v71, -1, v71, s[56:57]
v_add_u32 v84, 0, v1
v_cmp_lt_u32 s[58:59], v84, s46
v_mul_lo_u32 v84, v84, s64
v_add_u32 v72, v84, v82
v_cndmask_b32 v72, -1, v72, s[58:59]
v_and_b32 v79, -256, v83
s_add_u32 s65, s48, 0
s_lshr_b32 s66, s65, 5
s_mul_i32 s66, s66, s62
s_and_b32 s65, 15, s65
s_lshl_b32 s65, s65, 2
s_add_u32 s66, s66, s65
v_add_u32 v73, s66, v83
v_cndmask_b32 v73, -1, v73, s[56:57]
v_add_u32 v84, 0, v81
v_cmp_lt_u32 s[66:67], v84, s46
v_lshrrev_b32 v85, 5, v84
v_mul_lo_u32 v85, v85, s62
v_and_b32 v84, 15, v84
v_lshlrev_b32 v84, 2, v84
v_add_u32 v85, v85, v84
v_add_u32 v74, v85, v83
v_cndmask_b32 v74, -1, v74, s[66:67]
s_lshr_b32 s65, s48, 4
s_and_b32 s65, 1, s65
s_mul_i32 s65, s65, 16843009
s_add_u32 s65, s65, 100925952
v_mov_b32 v75, s65
v_lshrrev_b32 v76, 4, v1
v_and_b32 v76, 1, v76
v_mov_b32 v84, 16843009
v_mul_lo_u32 v76, v76, v84
v_add_u32 v76, 100925952, v76
v_mov_b32 v70, 0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s65, s5, 2147483647
s_cmp_eq_u32 s65, 0
s_cbranch_scc1 label_c_prefetch_skip_0
v_cmp_eq_i32 s[66:67], 0, v0
s_mul_i32 s68, s48, s43
v_mul_lo_u32 v82, s44, v1
v_add_u32 v81, s68, v82
s_and_b64 s[72:73], s[56:57], s[58:59]
s_and_b64 s[72:73], s[72:73], s[66:67]
v_cndmask_b32 v81, -1, v81, s[72:73]
buffer_load_ushort v80, v81, s[32:35], 0 offen offset:0
label_c_prefetch_skip_0:
s_add_u32 s65, s49, -1
s_max_i32 s65, s65, 0
s_sub_u32 s67, s50, s49
s_cmp_ge_u32 s50, s49
s_cselect_b32 s66, s67, s50
s_lshl_b32 s67, s66, 10
s_add_u32 s67, s67, s52
buffer_load_dword v66, v73, s[16:19], s67 offen offset:0
buffer_load_dword v67, v74, s[20:23], s67 offen offset:0
buffer_load_dwordx4 v[2:5], v71, s[8:11], s67 offen offset:0
buffer_load_dwordx4 v[6:9], v72, s[12:15], s67 offen offset:0
s_add_u32 s68, s67, 128
buffer_load_dwordx4 v[10:13], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[14:17], v72, s[12:15], s68 offen offset:0
s_add_u32 s68, s67, 256
buffer_load_dwordx4 v[18:21], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[22:25], v72, s[12:15], s68 offen offset:0
s_add_u32 s68, s67, 384
buffer_load_dwordx4 v[26:29], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[30:33], v72, s[12:15], s68 offen offset:0
s_min_u32 s66, s65, 1
s_add_u32 s66, s66, s50
s_sub_u32 s67, s66, s49
s_cmp_ge_u32 s66, s49
s_cselect_b32 s66, s67, s66
s_lshl_b32 s67, s66, 10
s_add_u32 s67, s67, s52
buffer_load_dword v68, v73, s[16:19], s67 offen offset:0
buffer_load_dword v69, v74, s[20:23], s67 offen offset:0
buffer_load_dwordx4 v[34:37], v71, s[8:11], s67 offen offset:0
buffer_load_dwordx4 v[38:41], v72, s[12:15], s67 offen offset:0
s_add_u32 s68, s67, 128
buffer_load_dwordx4 v[42:45], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[46:49], v72, s[12:15], s68 offen offset:0
s_add_u32 s68, s67, 256
buffer_load_dwordx4 v[50:53], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[54:57], v72, s[12:15], s68 offen offset:0
s_add_u32 s68, s67, 384
buffer_load_dwordx4 v[58:61], v71, s[8:11], s68 offen offset:0
buffer_load_dwordx4 v[62:65], v72, s[12:15], s68 offen offset:0
s_ashr_i32 s65, s49, 1
s_lshl_b32 s65, s65, 1
s_add_u32 s65, s65, -2
s_max_i32 s65, s65, 0
s_add_u32 s66, s50, 2
s_add_u32 s53, s52, 128
s_add_u32 s54, s52, 256
s_add_u32 s55, s52, 384
s_add_u32 s67, s50, 3
s_cmp_gt_i32 s65, 0
s_mov_b32 s68, 0
s_cbranch_scc0 label_kloop_end_2
.p2align 4
label_kloop_1:
s_add_u32 s69, s68, s66
s_sub_u32 s70, s69, s49
s_cmp_ge_u32 s69, s49
s_cselect_b32 s69, s70, s69
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v66
ds_bpermute_b32 v66, v78, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v66, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v67, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_lshl_b32 s70, s69, 10
s_add_u32 s70, s70, s52
buffer_load_dword v66, v73, s[16:19], s70 offen offset:0
buffer_load_dword v67, v74, s[20:23], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
buffer_load_dwordx4 v[2:5], v71, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[6:9], v72, s[12:15], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_lshl_b32 s69, s69, 10
s_add_u32 s71, s53, s69
buffer_load_dwordx4 v[10:13], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[14:17], v72, s[12:15], s71 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s71, s54, s69
buffer_load_dwordx4 v[18:21], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[22:25], v72, s[12:15], s71 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s71, s55, s69
buffer_load_dwordx4 v[26:29], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[30:33], v72, s[12:15], s71 offen offset:0
s_add_u32 s69, s68, s67
s_sub_u32 s70, s69, s49
s_cmp_ge_u32 s69, s49
s_cselect_b32 s69, s70, s69
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v68
ds_bpermute_b32 v68, v78, v68
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v69
ds_bpermute_b32 v69, v78, v69
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v68, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v69, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_lshl_b32 s70, s69, 10
s_add_u32 s70, s70, s52
buffer_load_dword v68, v73, s[16:19], s70 offen offset:0
buffer_load_dword v69, v74, s[20:23], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
buffer_load_dwordx4 v[34:37], v71, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[38:41], v72, s[12:15], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_lshl_b32 s69, s69, 10
s_add_u32 s71, s53, s69
buffer_load_dwordx4 v[42:45], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[46:49], v72, s[12:15], s71 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s71, s54, s69
buffer_load_dwordx4 v[50:53], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[54:57], v72, s[12:15], s71 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s71, s55, s69
buffer_load_dwordx4 v[58:61], v71, s[8:11], s71 offen offset:0
buffer_load_dwordx4 v[62:65], v72, s[12:15], s71 offen offset:0
s_add_u32 s68, s68, 2
s_cmp_lt_i32 s68, s65
s_cbranch_scc1 label_kloop_1
label_kloop_end_2:
s_sub_u32 s66, s49, s65
s_cmp_eq_i32 s66, 1
s_cbranch_scc0 label_arm_end_3
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v66
ds_bpermute_b32 v66, v78, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v66, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v67, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(15)
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85
s_waitcnt vmcnt(14)
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(13)
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87
s_waitcnt vmcnt(12)
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(11)
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89
s_waitcnt vmcnt(10)
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
label_arm_end_3:
s_cmp_eq_i32 s66, 2
s_cbranch_scc0 label_arm_end_4
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v66
ds_bpermute_b32 v66, v78, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v66, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v67, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(15)
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85
s_waitcnt vmcnt(14)
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(13)
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87
s_waitcnt vmcnt(12)
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(11)
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89
s_waitcnt vmcnt(10)
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(9)
ds_bpermute_b32 v81, v77, v68
ds_bpermute_b32 v68, v78, v68
s_waitcnt vmcnt(8)
ds_bpermute_b32 v82, v77, v69
ds_bpermute_b32 v69, v78, v69
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v68, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v69, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_waitcnt vmcnt(7)
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83
s_waitcnt vmcnt(6)
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(5)
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85
s_waitcnt vmcnt(4)
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(3)
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87
s_waitcnt vmcnt(2)
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(1)
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89
s_waitcnt vmcnt(0)
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
label_arm_end_4:
s_cmp_eq_i32 s66, 3
s_cbranch_scc0 label_arm_end_5
s_add_u32 s68, s50, s65
s_add_u32 s67, s68, 2
s_sub_u32 s69, s67, s49
s_cmp_ge_u32 s67, s49
s_cselect_b32 s67, s69, s67
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v66
ds_bpermute_b32 v66, v78, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v66, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v67, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_lshl_b32 s69, s67, 10
s_add_u32 s69, s69, s52
buffer_load_dword v66, v73, s[16:19], s69 offen offset:0
buffer_load_dword v67, v74, s[20:23], s69 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
buffer_load_dwordx4 v[2:5], v71, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[6:9], v72, s[12:15], s69 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s70, s69, 128
buffer_load_dwordx4 v[10:13], v71, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[14:17], v72, s[12:15], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s70, s69, 256
buffer_load_dwordx4 v[18:21], v71, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[22:25], v72, s[12:15], s70 offen offset:0
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_add_u32 s70, s69, 384
buffer_load_dwordx4 v[26:29], v71, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[30:33], v72, s[12:15], s70 offen offset:0
s_waitcnt vmcnt(19)
ds_bpermute_b32 v81, v77, v68
ds_bpermute_b32 v68, v78, v68
s_waitcnt vmcnt(18)
ds_bpermute_b32 v82, v77, v69
ds_bpermute_b32 v69, v78, v69
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v68, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v69, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v34, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v38, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v35, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v39, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v36, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v40, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v37, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v41, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(15)
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85
s_waitcnt vmcnt(14)
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v42, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v46, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v43, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v47, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v44, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v48, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v45, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v49, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(13)
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87
s_waitcnt vmcnt(12)
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v50, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v54, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v51, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v55, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v52, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v56, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v53, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v57, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(11)
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89
s_waitcnt vmcnt(10)
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v58, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v62, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v59, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v63, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v60, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v64, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v61, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v65, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(9)
ds_bpermute_b32 v81, v77, v66
ds_bpermute_b32 v66, v78, v66
s_waitcnt vmcnt(8)
ds_bpermute_b32 v82, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v81, v66, v81, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v82, v67, v82, v76
v_lshlrev_b32 v83, 23, v81
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 23, v82
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 15, v81
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 15, v82
v_and_b32 v86, 2139095040, v86
v_lshlrev_b32 v87, 7, v81
v_and_b32 v87, 2139095040, v87
v_lshlrev_b32 v88, 7, v82
v_and_b32 v88, 2139095040, v88
v_lshrrev_b32 v89, 1, v81
v_and_b32 v89, 2139095040, v89
v_lshrrev_b32 v90, 1, v82
v_and_b32 v90, 2139095040, v90
s_waitcnt vmcnt(7)
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83
s_waitcnt vmcnt(6)
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v2, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v6, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v3, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v7, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v4, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v8, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v5, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v9, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(5)
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85
s_waitcnt vmcnt(4)
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v10, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v14, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v11, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v15, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v12, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v16, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v13, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v17, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(3)
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87
s_waitcnt vmcnt(2)
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v18, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v22, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v19, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v23, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v20, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v24, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v21, v87 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v25, v88 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
s_waitcnt vmcnt(1)
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89
s_waitcnt vmcnt(0)
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v26, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v30, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v27, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v31, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v28, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v32, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90
v_dot2c_f32_bf16 v70, v81, v82
v_cvt_scalef32_pk_bf16_fp8 v81, v29, v89 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v82, v33, v90 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v81, v82
label_arm_end_5:
s_cmp_eq_u32 s51, 0
s_cbranch_scc1 label_tail_skip_6
label_tail_begin_7:
s_lshl_b32 s65, s49, 10
s_add_u32 s65, s65, s52
s_sub_u32 s66, s36, s65
v_lshrrev_b32 v81, 3, v0
v_lshlrev_b32 v81, 9, v81
v_and_b32 v82, 7, v0
v_lshlrev_b32 v82, 4, v82
v_add_u32 v81, v81, v82
v_sub_u32 v82, s66, v79
v_add_u32 v82, -1, v82
v_ashrrev_i32 v82, 31, v82
v_or_b32 v83, v73, v82
buffer_load_dword v66, v83, s[16:19], s65 offen offset:0
v_or_b32 v83, v74, v82
buffer_load_dword v67, v83, s[20:23], s65 offen offset:0
s_mov_b32 s67, 0
s_cmp_ge_i32 s51, 640
s_cbranch_scc0 label_tail_edge_8
buffer_load_dwordx4 v[2:5], v71, s[8:11], s65 offen offset:0
buffer_load_dwordx4 v[6:9], v72, s[12:15], s65 offen offset:0
s_branch label_tail_chunk_end_10
label_tail_edge_8:
s_cmp_gt_i32 s51, 0
s_cbranch_scc0 label_tail_zero_9
v_sub_u32 v82, s51, v81
v_add_u32 v82, -1, v82
v_ashrrev_i32 v82, 31, v82
v_or_b32 v83, v71, v82
buffer_load_dwordx4 v[2:5], v83, s[8:11], s65 offen offset:0
v_or_b32 v83, v72, v82
buffer_load_dwordx4 v[6:9], v83, s[12:15], s65 offen offset:0
s_or_b32 s67, s67, 1
s_branch label_tail_chunk_end_10
label_tail_zero_9:
s_waitcnt vmcnt(18)
v_mov_b32 v2, 0
v_mov_b32 v3, 0
v_mov_b32 v4, 0
v_mov_b32 v5, 0
v_mov_b32 v6, 0
v_mov_b32 v7, 0
v_mov_b32 v8, 0
v_mov_b32 v9, 0
label_tail_chunk_end_10:
s_sub_u32 s68, s51, 128
s_cmp_ge_i32 s68, 640
s_cbranch_scc0 label_tail_edge_11
s_add_u32 s69, s65, 128
buffer_load_dwordx4 v[10:13], v71, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[14:17], v72, s[12:15], s69 offen offset:0
s_branch label_tail_chunk_end_13
label_tail_edge_11:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_12
v_sub_u32 v82, s68, v81
v_add_u32 v82, -1, v82
v_ashrrev_i32 v82, 31, v82
s_add_u32 s69, s65, 128
v_or_b32 v83, v71, v82
buffer_load_dwordx4 v[10:13], v83, s[8:11], s69 offen offset:0
v_or_b32 v83, v72, v82
buffer_load_dwordx4 v[14:17], v83, s[12:15], s69 offen offset:0
s_or_b32 s67, s67, 2
s_branch label_tail_chunk_end_13
label_tail_zero_12:
s_waitcnt vmcnt(16)
v_mov_b32 v10, 0
v_mov_b32 v11, 0
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
v_mov_b32 v16, 0
v_mov_b32 v17, 0
label_tail_chunk_end_13:
s_sub_u32 s68, s51, 256
s_cmp_ge_i32 s68, 640
s_cbranch_scc0 label_tail_edge_14
s_add_u32 s69, s65, 256
buffer_load_dwordx4 v[18:21], v71, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[22:25], v72, s[12:15], s69 offen offset:0
s_branch label_tail_chunk_end_16
label_tail_edge_14:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_15
v_sub_u32 v82, s68, v81
v_add_u32 v82, -1, v82
v_ashrrev_i32 v82, 31, v82
s_add_u32 s69, s65, 256
v_or_b32 v83, v71, v82
buffer_load_dwordx4 v[18:21], v83, s[8:11], s69 offen offset:0
v_or_b32 v83, v72, v82
buffer_load_dwordx4 v[22:25], v83, s[12:15], s69 offen offset:0
s_or_b32 s67, s67, 4
s_branch label_tail_chunk_end_16
label_tail_zero_15:
s_waitcnt vmcnt(14)
v_mov_b32 v18, 0
v_mov_b32 v19, 0
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v24, 0
v_mov_b32 v25, 0
label_tail_chunk_end_16:
s_sub_u32 s68, s51, 384
s_cmp_ge_i32 s68, 640
s_cbranch_scc0 label_tail_edge_17
s_add_u32 s69, s65, 384
buffer_load_dwordx4 v[26:29], v71, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[30:33], v72, s[12:15], s69 offen offset:0
s_branch label_tail_chunk_end_19
label_tail_edge_17:
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_tail_zero_18
v_sub_u32 v82, s68, v81
v_add_u32 v82, -1, v82
v_ashrrev_i32 v82, 31, v82
s_add_u32 s69, s65, 384
v_or_b32 v83, v71, v82
buffer_load_dwordx4 v[26:29], v83, s[8:11], s69 offen offset:0
v_or_b32 v83, v72, v82
buffer_load_dwordx4 v[30:33], v83, s[12:15], s69 offen offset:0
s_or_b32 s67, s67, 8
s_branch label_tail_chunk_end_19
label_tail_zero_18:
s_waitcnt vmcnt(12)
v_mov_b32 v26, 0
v_mov_b32 v27, 0
v_mov_b32 v28, 0
v_mov_b32 v29, 0
v_mov_b32 v30, 0
v_mov_b32 v31, 0
v_mov_b32 v32, 0
v_mov_b32 v33, 0
label_tail_chunk_end_19:
s_waitcnt vmcnt(0)
s_bitcmp1_b32 s67, 0
s_cbranch_scc0 label_tail_mask_skip_20
v_sub_u32 v82, s51, v81
v_sub_u32 v83, 0, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 4, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v2, v2, v84
v_and_b32 v6, v6, v84
v_sub_u32 v83, 4, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 8, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v3, v3, v84
v_and_b32 v7, v7, v84
v_sub_u32 v83, 8, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 12, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v4, v4, v84
v_and_b32 v8, v8, v84
v_sub_u32 v83, 12, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 16, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v5, v5, v84
v_and_b32 v9, v9, v84
label_tail_mask_skip_20:
s_sub_u32 s68, s51, 128
s_bitcmp1_b32 s67, 1
s_cbranch_scc0 label_tail_mask_skip_21
v_sub_u32 v82, s68, v81
v_sub_u32 v83, 0, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 4, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v10, v10, v84
v_and_b32 v14, v14, v84
v_sub_u32 v83, 4, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 8, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v11, v11, v84
v_and_b32 v15, v15, v84
v_sub_u32 v83, 8, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 12, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v12, v12, v84
v_and_b32 v16, v16, v84
v_sub_u32 v83, 12, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 16, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v13, v13, v84
v_and_b32 v17, v17, v84
label_tail_mask_skip_21:
s_sub_u32 s68, s51, 256
s_bitcmp1_b32 s67, 2
s_cbranch_scc0 label_tail_mask_skip_22
v_sub_u32 v82, s68, v81
v_sub_u32 v83, 0, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 4, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v18, v18, v84
v_and_b32 v22, v22, v84
v_sub_u32 v83, 4, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 8, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v19, v19, v84
v_and_b32 v23, v23, v84
v_sub_u32 v83, 8, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 12, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v20, v20, v84
v_and_b32 v24, v24, v84
v_sub_u32 v83, 12, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 16, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v21, v21, v84
v_and_b32 v25, v25, v84
label_tail_mask_skip_22:
s_sub_u32 s68, s51, 384
s_bitcmp1_b32 s67, 3
s_cbranch_scc0 label_tail_mask_skip_23
v_sub_u32 v82, s68, v81
v_sub_u32 v83, 0, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 4, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v26, v26, v84
v_and_b32 v30, v30, v84
v_sub_u32 v83, 4, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 8, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v27, v27, v84
v_and_b32 v31, v31, v84
v_sub_u32 v83, 8, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 12, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v28, v28, v84
v_and_b32 v32, v32, v84
v_sub_u32 v83, 12, v82
v_ashrrev_i32 v83, 31, v83
v_sub_u32 v84, 16, v82
v_max_i32 v84, 0, v84
v_min_i32 v84, 3, v84
v_lshlrev_b32 v84, 3, v84
v_lshrrev_b32 v84, v84, -1
v_and_b32 v84, v84, v83
v_and_b32 v29, v29, v84
v_and_b32 v33, v33, v84
label_tail_mask_skip_23:
v_sub_u32 v83, v81, s51
v_mov_b32 v82, 0
v_add_u32 v84, 0, v83
v_ashrrev_i32 v84, 31, v84
v_and_b32 v84, 255, v84
v_or_b32 v82, v82, v84
v_add_u32 v84, 128, v83
v_ashrrev_i32 v84, 31, v84
v_and_b32 v84, 255, v84
v_lshlrev_b32 v84, 8, v84
v_or_b32 v82, v82, v84
v_add_u32 v84, 256, v83
v_ashrrev_i32 v84, 31, v84
v_and_b32 v84, 255, v84
v_lshlrev_b32 v84, 16, v84
v_or_b32 v82, v82, v84
v_add_u32 v84, 384, v83
v_ashrrev_i32 v84, 31, v84
v_and_b32 v84, 255, v84
v_lshlrev_b32 v84, 24, v84
v_or_b32 v82, v82, v84
ds_bpermute_b32 v71, v77, v66
ds_bpermute_b32 v66, v78, v66
ds_bpermute_b32 v72, v77, v67
ds_bpermute_b32 v67, v78, v67
s_waitcnt lgkmcnt(2)
v_perm_b32 v71, v66, v71, v75
s_waitcnt lgkmcnt(0)
v_perm_b32 v72, v67, v72, v76
v_lshlrev_b32 v73, 23, v71
v_and_b32 v73, 2139095040, v73
v_lshlrev_b32 v74, 23, v72
v_and_b32 v74, 2139095040, v74
v_lshlrev_b32 v83, 15, v71
v_and_b32 v83, 2139095040, v83
v_lshlrev_b32 v84, 15, v72
v_and_b32 v84, 2139095040, v84
v_lshlrev_b32 v85, 7, v71
v_and_b32 v85, 2139095040, v85
v_lshlrev_b32 v86, 7, v72
v_and_b32 v86, 2139095040, v86
v_lshrrev_b32 v71, 1, v71
v_and_b32 v71, 2139095040, v71
v_lshrrev_b32 v72, 1, v72
v_and_b32 v72, 2139095040, v72
v_lshlrev_b32 v79, 24, v82
v_ashrrev_i32 v79, 24, v79
v_and_b32 v73, v73, v79
v_and_b32 v74, v74, v79
v_cvt_scalef32_pk_bf16_fp8 v87, v2, v73
v_cvt_scalef32_pk_bf16_fp8 v88, v6, v74
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v2, v73 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v88, v6, v74 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v3, v73
v_cvt_scalef32_pk_bf16_fp8 v88, v7, v74
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v3, v73 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v88, v7, v74 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v4, v73
v_cvt_scalef32_pk_bf16_fp8 v88, v8, v74
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v4, v73 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v88, v8, v74 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v5, v73
v_cvt_scalef32_pk_bf16_fp8 v88, v9, v74
v_dot2c_f32_bf16 v70, v87, v88
v_cvt_scalef32_pk_bf16_fp8 v87, v5, v73 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v88, v9, v74 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v87, v88
s_cmp_le_i32 s51, 128
s_cbranch_scc1 label_tail_dead_chunk_24
v_lshlrev_b32 v73, 16, v82
v_ashrrev_i32 v73, 24, v73
v_and_b32 v83, v83, v73
v_and_b32 v84, v84, v73
v_cvt_scalef32_pk_bf16_fp8 v73, v10, v83
v_cvt_scalef32_pk_bf16_fp8 v74, v14, v84
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v10, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v14, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v11, v83
v_cvt_scalef32_pk_bf16_fp8 v74, v15, v84
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v11, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v15, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v12, v83
v_cvt_scalef32_pk_bf16_fp8 v74, v16, v84
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v12, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v16, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v13, v83
v_cvt_scalef32_pk_bf16_fp8 v74, v17, v84
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v13, v83 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v17, v84 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
label_tail_dead_chunk_24:
s_cmp_le_i32 s51, 256
s_cbranch_scc1 label_tail_dead_chunk_25
v_lshlrev_b32 v73, 8, v82
v_ashrrev_i32 v73, 24, v73
v_and_b32 v85, v85, v73
v_and_b32 v86, v86, v73
v_cvt_scalef32_pk_bf16_fp8 v73, v18, v85
v_cvt_scalef32_pk_bf16_fp8 v74, v22, v86
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v18, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v22, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v19, v85
v_cvt_scalef32_pk_bf16_fp8 v74, v23, v86
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v19, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v23, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v20, v85
v_cvt_scalef32_pk_bf16_fp8 v74, v24, v86
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v20, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v24, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v21, v85
v_cvt_scalef32_pk_bf16_fp8 v74, v25, v86
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v21, v85 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v25, v86 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
label_tail_dead_chunk_25:
s_cmp_le_i32 s51, 384
s_cbranch_scc1 label_tail_dead_chunk_26
v_ashrrev_i32 v73, 24, v82
v_and_b32 v71, v71, v73
v_and_b32 v72, v72, v73
v_cvt_scalef32_pk_bf16_fp8 v73, v26, v71
v_cvt_scalef32_pk_bf16_fp8 v74, v30, v72
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v26, v71 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v30, v72 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v27, v71
v_cvt_scalef32_pk_bf16_fp8 v74, v31, v72
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v27, v71 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v31, v72 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v28, v71
v_cvt_scalef32_pk_bf16_fp8 v74, v32, v72
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v28, v71 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v32, v72 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v29, v71
v_cvt_scalef32_pk_bf16_fp8 v74, v33, v72
v_dot2c_f32_bf16 v70, v73, v74
v_cvt_scalef32_pk_bf16_fp8 v73, v29, v71 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v74, v33, v72 op_sel:[1,0,0]
v_dot2c_f32_bf16 v70, v73, v74
label_tail_dead_chunk_26:
label_tail_skip_6:
s_waitcnt vmcnt(17)
s_nop 2
v_add_f32 v2, v70, v70 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
s_nop 1
v_add_f32 v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
s_nop 1
v_add_f32 v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
s_nop 1
v_add_f32 v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cmp_eq_i32 s[66:67], 0, v0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_27
v_mul_f32 v2, s6, v2
s_and_b32 s65, s5, 2147483647
s_cmp_eq_u32 s65, 0
s_cbranch_scc1 label_no_beta_29
s_waitcnt vmcnt(0)
v_lshlrev_b32 v5, 16, v80
v_fmac_f32 v2, s5, v5
label_no_beta_29:
s_mul_i32 s65, s48, s39
v_mul_lo_u32 v4, s40, v1
v_cvt_pk_bf16_f32 v2, v2, v2
v_add_u32 v3, s65, v4
s_and_b64 s[70:71], s[56:57], s[58:59]
s_and_b64 s[70:71], s[70:71], s[66:67]
v_cndmask_b32 v3, -1, v3, s[70:71]
buffer_store_short v2, v3, s[24:27], 0 offen offset:0
s_branch label_stored_28
label_workspace_27:
s_mul_i32 s70, s4, s38
s_lshl_b32 s70, s70, 2
s_mul_i32 s65, s48, s41
s_add_u32 s65, s65, s70
v_mul_lo_u32 v4, s42, v1
v_add_u32 v3, s65, v4
s_and_b64 s[72:73], s[56:57], s[58:59]
s_and_b64 s[72:73], s[72:73], s[66:67]
v_cndmask_b32 v3, -1, v3, s[72:73]
buffer_store_dword v2, v3, s[28:31], 0 offen offset:0
label_stored_28:
s_endpgm
.Lgemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end:
.size gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1, .Lgemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end - gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 120
  .amdhsa_accum_offset 96
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr 85
  .amdhsa_group_segment_fixed_size 0
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
    macrotile: [4, 1, 4096]
    threads: [64, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
    kStep: 1024
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 32
  AssertTensorBytesLessThan: 4294967296
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 2147482625 }
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
  - .name: gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
    .symbol: gemm_BM1_TN1_LANES_K16_WAVES1_UP_ITERS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1.kd
    .kernarg_segment_size: 120
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 91
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
