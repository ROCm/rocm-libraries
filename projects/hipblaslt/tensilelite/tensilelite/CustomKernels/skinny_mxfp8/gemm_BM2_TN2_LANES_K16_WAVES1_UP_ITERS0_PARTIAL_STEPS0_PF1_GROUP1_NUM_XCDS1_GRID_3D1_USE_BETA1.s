.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.globl gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.p2align 8
.type gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1,@function
gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1:
s_load_dword s73, s[0:1], 104
s_load_dword s55, s[0:1], 96
s_load_dword s64, s[0:1], 100
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s72, s[0:1], 72
s_load_dword s71, s[0:1], 68
s_load_dword s70, s[0:1], 64
s_load_dword s69, s[0:1], 60
s_load_dword s68, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[80:81], s[0:1], 32
s_load_dwordx2 s[78:79], s[0:1], 24
s_load_dwordx2 s[76:77], s[0:1], 16
s_load_dwordx2 s[74:75], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s86, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
v_and_b32 v90, 15, v0
v_lshrrev_b32 v1, 4, v0
v_lshlrev_b32 v1, 1, v1
v_and_b32 v84, 7, v90
v_lshrrev_b32 v84, 1, v84
v_lshlrev_b32 v84, 2, v84
v_lshrrev_b32 v91, 3, v90
v_lshlrev_b32 v91, 5, v91
v_add_u32 v84, v84, v91
v_lshrrev_b32 v91, 4, v0
v_add_u32 v84, v84, v91
v_lshlrev_b32 v84, 2, v84
v_add_u32 v85, 64, v84
v_lshrrev_b32 v91, 3, v90
v_lshlrev_b32 v91, 9, v91
v_and_b32 v92, 7, v90
v_lshlrev_b32 v92, 4, v92
v_add_u32 v91, v91, v92
v_lshrrev_b32 v90, 2, v0
v_and_b32 v92, 3, v90
v_lshlrev_b32 v92, 6, v92
v_lshrrev_b32 v93, 3, v90
v_lshlrev_b32 v93, 9, v93
v_add_u32 v92, v92, v93
v_and_b32 v93, 7, v90
v_lshrrev_b32 v93, 2, v93
v_lshlrev_b32 v93, 8, v93
v_add_u32 v92, v92, v93
v_and_b32 v93, 3, v0
v_mov_b32 v90, v93
v_lshlrev_b32 v90, 1, v90
v_and_b32 v0, 15, v0
s_waitcnt lgkmcnt(0)
s_add_u32 s87, s36, 3
s_lshr_b32 s87, s87, 2
s_lshl_b32 s87, s87, 2
s_add_u32 s82, s71, -1
s_mul_i32 s82, s82, s69
s_add_u32 s82, s82, s87
s_add_u32 s83, s72, -1
s_mul_i32 s83, s83, s70
s_add_u32 s83, s83, s87
s_add_u32 s65, s36, 255
s_lshr_b32 s65, s65, 8
s_lshl_b32 s65, s65, 8
s_add_u32 s84, s71, 31
s_lshr_b32 s84, s84, 5
s_mul_i32 s84, s84, s65
s_add_u32 s85, s72, 31
s_lshr_b32 s85, s85, 5
s_mul_i32 s85, s85, s65
s_add_u32 s26, s72, -1
s_mul_i32 s26, s26, s68
s_lshl_b32 s87, s71, 1
s_add_u32 s26, s26, s87
s_add_u32 s34, s72, -1
s_mul_i32 s34, s34, s86
s_lshl_b32 s87, s71, 1
s_add_u32 s34, s34, s87
s_mul_i32 s30, s71, s73
s_mul_i32 s30, s30, s72
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s71, s72
s_lshl_b32 s42, s71, 2
s_cmp_le_u32 s71, s72
s_cselect_b64 s[8:9], s[74:75], s[76:77]
s_cselect_b32 s10, s82, s83
s_cselect_b64 s[12:13], s[76:77], s[74:75]
s_cselect_b32 s14, s83, s82
s_cselect_b64 s[16:17], s[78:79], s[80:81]
s_cselect_b32 s18, s84, s85
s_cselect_b64 s[20:21], s[80:81], s[78:79]
s_cselect_b32 s22, s85, s84
s_cselect_b32 s66, s69, s70
s_cselect_b32 s67, s70, s69
s_cselect_b32 s39, 2, s68
s_cselect_b32 s40, s68, 2
s_cselect_b32 s43, 2, s86
s_cselect_b32 s44, s86, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s71, s72
s_max_u32 s46, s71, s72
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
s_mul_i32 s51, s4, s37
s_sub_u32 s49, s36, s51
s_min_i32 s49, s49, s37
s_ashr_i32 s49, s49, 10
s_mul_i32 s69, s64, s4
s_add_u32 s69, s69, s3
s_mul_i32 s69, s69, s55
s_add_u32 s69, s69, s2
s_add_u32 s70, s37, 1023
s_lshr_b32 s70, s70, 10
s_lshr_b32 s71, s69, 4
s_add_u32 s71, s71, s69
s_abs_i32 s73, s70
s_abs_i32 s74, s71
s_xor_b32 s75, s71, s70
s_ashr_i32 s75, s75, 31
v_cvt_f32_u32 v93, s73
v_rcp_iflag_f32 v93, v93
s_sub_u32 s76, 0, s73
v_mul_f32 v93, 0x4f7ffffe, v93
v_cvt_u32_f32 v93, v93
s_nop 0
v_readfirstlane_b32 s77, v93
s_mul_i32 s78, s76, s77
s_mul_hi_u32 s78, s77, s78
s_add_u32 s77, s77, s78
s_mul_hi_u32 s76, s74, s77
s_mul_i32 s78, s76, s73
s_sub_u32 s78, s74, s78
s_add_u32 s79, s76, 1
s_sub_u32 s80, s78, s73
s_cmp_ge_u32 s78, s73
s_cselect_b32 s76, s79, s76
s_cselect_b32 s78, s80, s78
s_add_u32 s79, s76, 1
s_cmp_ge_u32 s78, s73
s_cselect_b32 s76, s79, s76
s_xor_b32 s76, s76, s75
s_sub_u32 s72, s76, s75
s_mul_i32 s72, s72, s70
s_sub_u32 s68, s71, s72
s_cmp_lt_u32 s68, s49
s_cselect_b32 s50, s68, 0
s_lshl_b32 s48, s3, 1
s_lshr_b32 s68, s2, 3
s_and_b32 s68, 3, s68
s_and_b32 s69, 7, s2
s_lshl_b32 s69, s69, 2
s_add_u32 s68, s68, s69
s_and_b32 s69, -32, s2
s_add_u32 s68, s68, s69
s_and_b32 s69, -32, s55
s_cmp_lt_u32 s2, s69
s_cselect_b32 s2, s68, s2
s_lshl_b32 s68, s2, 3
v_add_u32 v1, s68, v1
v_add_u32 v90, s68, v90
s_add_u32 s68, s48, 0
s_cmp_lt_u32 s68, s45
s_cselect_b64 s[56:57], -1, 0
s_mul_i32 s68, s68, s66
v_add_u32 v74, s68, v91
v_cndmask_b32 v74, -1, v74, s[56:57]
s_add_u32 s68, s48, 1
s_cmp_lt_u32 s68, s45
s_cselect_b64 s[58:59], -1, 0
s_mul_i32 s68, s68, s66
v_add_u32 v75, s68, v91
v_cndmask_b32 v75, -1, v75, s[58:59]
v_add_u32 v93, 0, v1
v_cmp_lt_u32 s[60:61], v93, s46
v_mul_lo_u32 v93, v93, s67
v_add_u32 v76, v93, v91
v_cndmask_b32 v76, -1, v76, s[60:61]
v_add_u32 v93, 1, v1
v_cmp_lt_u32 s[62:63], v93, s46
v_mul_lo_u32 v93, v93, s67
v_add_u32 v77, v93, v91
v_cndmask_b32 v77, -1, v77, s[62:63]
s_add_u32 s68, s48, 0
s_lshr_b32 s69, s68, 5
s_mul_i32 s69, s69, s65
s_and_b32 s68, 15, s68
s_lshl_b32 s68, s68, 2
s_add_u32 s69, s69, s68
v_add_u32 v78, s69, v92
v_cndmask_b32 v78, -1, v78, s[56:57]
s_add_u32 s68, s48, 1
s_lshr_b32 s69, s68, 5
s_mul_i32 s69, s69, s65
s_and_b32 s68, 15, s68
s_lshl_b32 s68, s68, 2
s_add_u32 s69, s69, s68
v_add_u32 v79, s69, v92
v_cndmask_b32 v79, -1, v79, s[58:59]
v_add_u32 v93, 0, v90
v_cmp_lt_u32 s[68:69], v93, s46
v_lshrrev_b32 v94, 5, v93
v_mul_lo_u32 v94, v94, s65
v_and_b32 v93, 15, v93
v_lshlrev_b32 v93, 2, v93
v_add_u32 v94, v94, v93
v_add_u32 v80, v94, v92
v_cndmask_b32 v80, -1, v80, s[68:69]
v_add_u32 v93, 1, v90
v_cmp_lt_u32 s[68:69], v93, s46
v_lshrrev_b32 v94, 5, v93
v_mul_lo_u32 v94, v94, s65
v_and_b32 v93, 15, v93
v_lshlrev_b32 v93, 2, v93
v_add_u32 v94, v94, v93
v_add_u32 v81, v94, v92
v_cndmask_b32 v81, -1, v81, s[68:69]
s_lshr_b32 s68, s48, 4
s_and_b32 s68, 1, s68
s_mul_i32 s68, s68, 16843009
s_add_u32 s68, s68, 100925952
v_mov_b32 v82, s68
v_lshrrev_b32 v83, 4, v1
v_and_b32 v83, 1, v83
v_mov_b32 v93, 16843009
v_mul_lo_u32 v83, v83, v93
v_add_u32 v83, 100925952, v83
v_mov_b32 v70, 0
v_mov_b32 v71, 0
v_mov_b32 v72, 0
v_mov_b32 v73, 0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s68, s5, 2147483647
s_cmp_eq_u32 s68, 0
s_cbranch_scc1 label_c_prefetch_skip_0
v_cmp_eq_i32 s[68:69], 0, v0
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_1
s_add_u32 s71, s3, 1
s_lshl_b32 s71, s71, 1
s_add_u32 s72, s2, 1
s_lshl_b32 s72, s72, 3
s_cmp_le_u32 s71, s45
s_cselect_b32 s73, 1, 0
s_cmp_le_u32 s72, s46
s_cselect_b32 s74, 1, 0
s_and_b32 s70, s73, s74
s_mul_i32 s71, s48, s43
v_mul_lo_u32 v91, s44, v1
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_c_vector_3
v_add_u32 v90, s71, v91
s_and_b64 s[74:75], s[56:57], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v86, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s44
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[56:57], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v88, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s43
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[58:59], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v87, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s43
s_add_u32 s73, s73, s72
s_mov_b32 s72, s44
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[58:59], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v89, v90, s[32:35], 0 offen offset:0
s_branch label_c_loaded_4
label_c_vector_3:
v_add_u32 v90, s71, v91
v_cndmask_b32 v90, -1, v90, s[68:69]
buffer_load_dword v86, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s44
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
v_cndmask_b32 v90, -1, v90, s[68:69]
buffer_load_dword v88, v90, s[32:35], 0 offen offset:0
label_c_loaded_4:
s_branch label_epilogue_end_2
label_orientation_b_1:
s_add_u32 s71, s3, 1
s_lshl_b32 s71, s71, 1
s_add_u32 s72, s2, 1
s_lshl_b32 s72, s72, 3
s_cmp_le_u32 s71, s45
s_cselect_b32 s73, 1, 0
s_cmp_le_u32 s72, s46
s_cselect_b32 s74, 1, 0
s_and_b32 s70, s73, s74
s_mul_i32 s71, s48, s43
v_mul_lo_u32 v91, s44, v1
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v90, s71, v91
s_and_b64 s[74:75], s[56:57], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v86, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s44
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[56:57], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v87, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s43
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[58:59], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v88, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s43
s_add_u32 s73, s73, s72
s_mov_b32 s72, s44
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
s_and_b64 s[74:75], s[58:59], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v90, -1, v90, s[74:75]
buffer_load_ushort v89, v90, s[32:35], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v90, s71, v91
v_cndmask_b32 v90, -1, v90, s[68:69]
buffer_load_dword v86, v90, s[32:35], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s43
s_add_u32 s73, s73, s72
v_add_u32 v90, s73, v91
v_cndmask_b32 v90, -1, v90, s[68:69]
buffer_load_dword v88, v90, s[32:35], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_2:
label_c_prefetch_skip_0:
s_sub_u32 s69, s50, s49
s_cmp_ge_u32 s50, s49
s_cselect_b32 s68, s69, s50
s_lshl_b32 s69, s68, 10
s_add_u32 s69, s69, s51
buffer_load_dword v66, v78, s[16:19], s69 offen offset:0
buffer_load_dword v67, v79, s[16:19], s69 offen offset:0
buffer_load_dword v68, v80, s[20:23], s69 offen offset:0
buffer_load_dword v69, v81, s[20:23], s69 offen offset:0
buffer_load_dwordx4 v[2:5], v74, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[6:9], v75, s[8:11], s69 offen offset:0
buffer_load_dwordx4 v[10:13], v76, s[12:15], s69 offen offset:0
buffer_load_dwordx4 v[14:17], v77, s[12:15], s69 offen offset:0
s_add_u32 s70, s69, 128
buffer_load_dwordx4 v[18:21], v74, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[22:25], v75, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[26:29], v76, s[12:15], s70 offen offset:0
buffer_load_dwordx4 v[30:33], v77, s[12:15], s70 offen offset:0
s_add_u32 s70, s69, 256
buffer_load_dwordx4 v[34:37], v74, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[38:41], v75, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[42:45], v76, s[12:15], s70 offen offset:0
buffer_load_dwordx4 v[46:49], v77, s[12:15], s70 offen offset:0
s_add_u32 s70, s69, 384
buffer_load_dwordx4 v[50:53], v74, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[54:57], v75, s[8:11], s70 offen offset:0
buffer_load_dwordx4 v[58:61], v76, s[12:15], s70 offen offset:0
buffer_load_dwordx4 v[62:65], v77, s[12:15], s70 offen offset:0
s_add_u32 s68, s49, -1
s_max_i32 s68, s68, 0
s_add_u32 s69, s50, 1
s_add_u32 s52, s51, 128
s_add_u32 s53, s51, 256
s_add_u32 s54, s51, 384
s_cmp_gt_i32 s68, 0
s_mov_b32 s70, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_add_u32 s71, s70, s69
s_sub_u32 s72, s71, s49
s_cmp_ge_u32 s71, s49
s_cselect_b32 s71, s72, s71
s_waitcnt vmcnt(19)
ds_bpermute_b32 v90, v84, v66
ds_bpermute_b32 v66, v85, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v91, v84, v67
ds_bpermute_b32 v67, v85, v67
s_waitcnt vmcnt(17)
ds_bpermute_b32 v92, v84, v68
ds_bpermute_b32 v68, v85, v68
s_waitcnt vmcnt(16)
ds_bpermute_b32 v93, v84, v69
ds_bpermute_b32 v69, v85, v69
s_waitcnt lgkmcnt(6)
v_perm_b32 v90, v66, v90, v82
s_waitcnt lgkmcnt(4)
v_perm_b32 v91, v67, v91, v82
s_waitcnt lgkmcnt(2)
v_perm_b32 v92, v68, v92, v83
s_waitcnt lgkmcnt(0)
v_perm_b32 v93, v69, v93, v83
s_lshl_b32 s72, s71, 10
s_add_u32 s72, s72, s51
buffer_load_dword v66, v78, s[16:19], s72 offen offset:0
buffer_load_dword v67, v79, s[16:19], s72 offen offset:0
buffer_load_dword v68, v80, s[20:23], s72 offen offset:0
buffer_load_dword v69, v81, s[20:23], s72 offen offset:0
v_lshlrev_b32 v94, 23, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 23, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 23, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 23, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v98, v2, v94
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v99, v6, v95
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v100, v10, v96
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v101, v14, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v2, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v6, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v10, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v14, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v3, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v7, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v11, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v15, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v3, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v7, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v11, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v15, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v4, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v8, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v12, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v16, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v4, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v8, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v12, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v16, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v5, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v9, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v13, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v17, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v5, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v9, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v13, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v17, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
buffer_load_dwordx4 v[2:5], v74, s[8:11], s72 offen offset:0
buffer_load_dwordx4 v[6:9], v75, s[8:11], s72 offen offset:0
buffer_load_dwordx4 v[10:13], v76, s[12:15], s72 offen offset:0
buffer_load_dwordx4 v[14:17], v77, s[12:15], s72 offen offset:0
v_lshlrev_b32 v94, 15, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 15, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 15, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 15, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v98, v18, v94
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v99, v22, v95
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v100, v26, v96
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v101, v30, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v18, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v22, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v26, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v30, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v19, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v23, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v27, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v31, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v19, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v23, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v27, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v31, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v20, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v24, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v28, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v32, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v20, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v24, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v28, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v32, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v21, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v25, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v29, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v33, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v21, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v25, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v29, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v33, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
s_lshl_b32 s71, s71, 10
s_add_u32 s73, s52, s71
buffer_load_dwordx4 v[18:21], v74, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[22:25], v75, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[26:29], v76, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[30:33], v77, s[12:15], s73 offen offset:0
v_lshlrev_b32 v94, 7, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 7, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 7, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 7, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v98, v34, v94
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v99, v38, v95
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v100, v42, v96
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v101, v46, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v34, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v38, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v42, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v46, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v35, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v39, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v43, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v47, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v35, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v39, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v43, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v47, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v36, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v40, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v44, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v48, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v36, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v40, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v44, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v48, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v37, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v41, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v45, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v49, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v37, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v41, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v45, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v49, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
s_add_u32 s73, s53, s71
buffer_load_dwordx4 v[34:37], v74, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[38:41], v75, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[42:45], v76, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[46:49], v77, s[12:15], s73 offen offset:0
v_lshrrev_b32 v94, 1, v90
v_and_b32 v94, 2139095040, v94
v_lshrrev_b32 v95, 1, v91
v_and_b32 v95, 2139095040, v95
v_lshrrev_b32 v96, 1, v92
v_and_b32 v96, 2139095040, v96
v_lshrrev_b32 v97, 1, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v98, v50, v94
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v99, v54, v95
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v100, v58, v96
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v101, v62, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v50, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v54, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v58, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v62, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v51, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v55, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v59, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v63, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v51, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v55, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v59, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v63, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v52, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v56, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v60, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v64, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v52, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v56, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v60, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v64, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v53, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v57, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v61, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v65, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v53, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v57, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v61, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v65, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
s_add_u32 s73, s54, s71
buffer_load_dwordx4 v[50:53], v74, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[54:57], v75, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[58:61], v76, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[62:65], v77, s[12:15], s73 offen offset:0
s_add_u32 s70, s70, 1
s_cmp_lt_i32 s70, s68
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s69, s49, s68
s_cmp_eq_i32 s69, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(19)
ds_bpermute_b32 v90, v84, v66
ds_bpermute_b32 v66, v85, v66
s_waitcnt vmcnt(18)
ds_bpermute_b32 v91, v84, v67
ds_bpermute_b32 v67, v85, v67
s_waitcnt vmcnt(17)
ds_bpermute_b32 v92, v84, v68
ds_bpermute_b32 v68, v85, v68
s_waitcnt vmcnt(16)
ds_bpermute_b32 v93, v84, v69
ds_bpermute_b32 v69, v85, v69
s_waitcnt lgkmcnt(6)
v_perm_b32 v90, v66, v90, v82
s_waitcnt lgkmcnt(4)
v_perm_b32 v91, v67, v91, v82
s_waitcnt lgkmcnt(2)
v_perm_b32 v92, v68, v92, v83
s_waitcnt lgkmcnt(0)
v_perm_b32 v93, v69, v93, v83
v_lshlrev_b32 v94, 23, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 23, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 23, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 23, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(15)
v_cvt_scalef32_pk_bf16_fp8 v98, v2, v94
s_waitcnt vmcnt(14)
v_cvt_scalef32_pk_bf16_fp8 v99, v6, v95
s_waitcnt vmcnt(13)
v_cvt_scalef32_pk_bf16_fp8 v100, v10, v96
s_waitcnt vmcnt(12)
v_cvt_scalef32_pk_bf16_fp8 v101, v14, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v2, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v6, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v10, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v14, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v3, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v7, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v11, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v15, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v3, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v7, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v11, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v15, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v4, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v8, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v12, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v16, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v4, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v8, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v12, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v16, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v5, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v9, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v13, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v17, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v5, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v9, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v13, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v17, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_lshlrev_b32 v94, 15, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 15, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 15, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 15, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(11)
v_cvt_scalef32_pk_bf16_fp8 v98, v18, v94
s_waitcnt vmcnt(10)
v_cvt_scalef32_pk_bf16_fp8 v99, v22, v95
s_waitcnt vmcnt(9)
v_cvt_scalef32_pk_bf16_fp8 v100, v26, v96
s_waitcnt vmcnt(8)
v_cvt_scalef32_pk_bf16_fp8 v101, v30, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v18, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v22, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v26, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v30, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v19, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v23, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v27, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v31, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v19, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v23, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v27, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v31, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v20, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v24, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v28, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v32, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v20, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v24, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v28, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v32, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v21, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v25, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v29, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v33, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v21, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v25, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v29, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v33, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_lshlrev_b32 v94, 7, v90
v_and_b32 v94, 2139095040, v94
v_lshlrev_b32 v95, 7, v91
v_and_b32 v95, 2139095040, v95
v_lshlrev_b32 v96, 7, v92
v_and_b32 v96, 2139095040, v96
v_lshlrev_b32 v97, 7, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(7)
v_cvt_scalef32_pk_bf16_fp8 v98, v34, v94
s_waitcnt vmcnt(6)
v_cvt_scalef32_pk_bf16_fp8 v99, v38, v95
s_waitcnt vmcnt(5)
v_cvt_scalef32_pk_bf16_fp8 v100, v42, v96
s_waitcnt vmcnt(4)
v_cvt_scalef32_pk_bf16_fp8 v101, v46, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v34, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v38, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v42, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v46, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v35, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v39, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v43, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v47, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v35, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v39, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v43, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v47, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v36, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v40, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v44, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v48, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v36, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v40, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v44, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v48, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v37, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v41, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v45, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v49, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v37, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v41, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v45, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v49, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_lshrrev_b32 v94, 1, v90
v_and_b32 v94, 2139095040, v94
v_lshrrev_b32 v95, 1, v91
v_and_b32 v95, 2139095040, v95
v_lshrrev_b32 v96, 1, v92
v_and_b32 v96, 2139095040, v96
v_lshrrev_b32 v97, 1, v93
v_and_b32 v97, 2139095040, v97
s_waitcnt vmcnt(3)
v_cvt_scalef32_pk_bf16_fp8 v98, v50, v94
s_waitcnt vmcnt(2)
v_cvt_scalef32_pk_bf16_fp8 v99, v54, v95
s_waitcnt vmcnt(1)
v_cvt_scalef32_pk_bf16_fp8 v100, v58, v96
s_waitcnt vmcnt(0)
v_cvt_scalef32_pk_bf16_fp8 v101, v62, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v50, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v54, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v58, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v62, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v51, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v55, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v59, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v63, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v51, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v55, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v59, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v63, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v52, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v56, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v60, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v64, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v52, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v56, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v60, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v64, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v53, v94
v_cvt_scalef32_pk_bf16_fp8 v99, v57, v95
v_cvt_scalef32_pk_bf16_fp8 v100, v61, v96
v_cvt_scalef32_pk_bf16_fp8 v101, v65, v97
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
v_cvt_scalef32_pk_bf16_fp8 v98, v53, v94 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v99, v57, v95 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v100, v61, v96 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v101, v65, v97 op_sel:[1,0,0]
v_dot2_f32_bf16 v70, v98, v100, v70
v_dot2_f32_bf16 v71, v98, v101, v71
v_dot2_f32_bf16 v72, v99, v100, v72
v_dot2_f32_bf16 v73, v99, v101, v73
label_arm_end_9:
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_10
s_waitcnt vmcnt(14)
v_add_f32 v2, v70, v70 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v71, v71 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v72, v72 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v73, v73 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cmp_eq_i32 s[68:69], 0, v0
s_add_u32 s71, s3, 1
s_lshl_b32 s71, s71, 1
s_add_u32 s72, s2, 1
s_lshl_b32 s72, s72, 3
s_cmp_le_u32 s71, s45
s_cselect_b32 s73, 1, 0
s_cmp_le_u32 s72, s46
s_cselect_b32 s74, 1, 0
s_and_b32 s70, s73, s74
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_12
v_pk_mul_f32 v[2:3], s[6:7], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], s[6:7], v[4:5] op_sel_hi:[0,1]
s_and_b32 s71, s5, 2147483647
s_cmp_eq_u32 s71, 0
s_cbranch_scc1 label_no_beta_14
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s70, 1
s_cbranch_scc0 label_c_masked_15
v_lshlrev_b32 v8, 16, v86
v_fmac_f32 v2, s5, v8
v_lshlrev_b32 v8, 16, v88
v_fmac_f32 v4, s5, v8
v_and_b32 v8, -65536, v86
v_fmac_f32 v3, s5, v8
v_and_b32 v8, -65536, v88
v_fmac_f32 v5, s5, v8
s_branch label_c_added_16
label_c_masked_15:
v_lshlrev_b32 v8, 16, v86
v_fmac_f32 v2, s5, v8
v_lshlrev_b32 v8, 16, v88
v_fmac_f32 v4, s5, v8
v_lshlrev_b32 v8, 16, v87
v_fmac_f32 v3, s5, v8
v_lshlrev_b32 v8, 16, v89
v_fmac_f32 v5, s5, v8
label_c_added_16:
label_no_beta_14:
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_vector_17
s_mul_i32 s71, s48, s39
v_mul_lo_u32 v7, s40, v1
v_lshrrev_b32 v2, 16, v2
v_add_u32 v6, s71, v7
s_and_b64 s[74:75], s[56:57], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v2, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v4, 16, v4
s_mov_b32 s73, s71
s_mov_b32 s72, s40
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[56:57], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v4, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v3, 16, v3
s_mov_b32 s73, s71
s_mov_b32 s72, s39
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[58:59], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v3, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v5, 16, v5
s_mov_b32 s73, s71
s_mov_b32 s72, s39
s_add_u32 s73, s73, s72
s_mov_b32 s72, s40
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[58:59], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v5, v6, s[24:27], 0 offen offset:0
s_branch label_tile_stored_18
label_vector_17:
s_mul_i32 s71, s48, s39
v_mul_lo_u32 v7, s40, v1
v_add_u32 v6, s71, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
v_lshrrev_b32 v2, 16, v2
v_lshrrev_b32 v3, 16, v3
v_pack_b32_f16 v2, v2, v3
buffer_store_dword v2, v6, s[24:27], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s40
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
v_lshrrev_b32 v4, 16, v4
v_lshrrev_b32 v5, 16, v5
v_pack_b32_f16 v4, v4, v5
buffer_store_dword v4, v6, s[24:27], 0 offen offset:0
label_tile_stored_18:
s_branch label_stored_13
label_workspace_12:
s_mul_i32 s74, s4, s38
s_lshl_b32 s74, s74, 2
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_vector_19
s_mul_i32 s71, s48, s41
s_add_u32 s71, s71, s74
v_mul_lo_u32 v7, s42, v1
v_add_u32 v6, s71, v7
s_and_b64 s[76:77], s[56:57], s[60:61]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v2, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s42
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[56:57], s[62:63]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v4, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s41
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[58:59], s[60:61]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v3, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s41
s_add_u32 s73, s73, s72
s_mov_b32 s72, s42
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[58:59], s[62:63]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v5, v6, s[28:31], 0 offen offset:0
s_branch label_tile_stored_20
label_vector_19:
s_mul_i32 s71, s48, s41
s_add_u32 s71, s71, s74
v_mul_lo_u32 v7, s42, v1
v_add_u32 v6, s71, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
buffer_store_dwordx2 v[2:3], v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s42
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
buffer_store_dwordx2 v[4:5], v6, s[28:31], 0 offen offset:0
label_tile_stored_20:
label_stored_13:
s_branch label_epilogue_end_11
label_orientation_b_10:
s_waitcnt vmcnt(14)
v_add_f32 v2, v70, v70 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v71, v71 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v72, v72 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v73, v73 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v4, v4, v4 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32 v5, v5, v5 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cmp_eq_i32 s[68:69], 0, v0
s_add_u32 s71, s3, 1
s_lshl_b32 s71, s71, 1
s_add_u32 s72, s2, 1
s_lshl_b32 s72, s72, 3
s_cmp_le_u32 s71, s45
s_cselect_b32 s73, 1, 0
s_cmp_le_u32 s72, s46
s_cselect_b32 s74, 1, 0
s_and_b32 s70, s73, s74
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_21
v_pk_mul_f32 v[2:3], s[6:7], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], s[6:7], v[4:5] op_sel_hi:[0,1]
s_and_b32 s71, s5, 2147483647
s_cmp_eq_u32 s71, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s70, 1
s_cbranch_scc0 label_c_masked_24
v_lshlrev_b32 v8, 16, v86
v_fmac_f32 v2, s5, v8
v_and_b32 v8, -65536, v86
v_fmac_f32 v3, s5, v8
v_lshlrev_b32 v8, 16, v88
v_fmac_f32 v4, s5, v8
v_and_b32 v8, -65536, v88
v_fmac_f32 v5, s5, v8
s_branch label_c_added_25
label_c_masked_24:
v_lshlrev_b32 v8, 16, v86
v_fmac_f32 v2, s5, v8
v_lshlrev_b32 v8, 16, v87
v_fmac_f32 v3, s5, v8
v_lshlrev_b32 v8, 16, v88
v_fmac_f32 v4, s5, v8
v_lshlrev_b32 v8, 16, v89
v_fmac_f32 v5, s5, v8
label_c_added_25:
label_no_beta_23:
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_vector_26
s_mul_i32 s71, s48, s39
v_mul_lo_u32 v7, s40, v1
v_lshrrev_b32 v2, 16, v2
v_add_u32 v6, s71, v7
s_and_b64 s[74:75], s[56:57], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v2, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v3, 16, v3
s_mov_b32 s73, s71
s_mov_b32 s72, s40
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[56:57], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v3, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v4, 16, v4
s_mov_b32 s73, s71
s_mov_b32 s72, s39
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[58:59], s[60:61]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v4, v6, s[24:27], 0 offen offset:0
v_lshrrev_b32 v5, 16, v5
s_mov_b32 s73, s71
s_mov_b32 s72, s39
s_add_u32 s73, s73, s72
s_mov_b32 s72, s40
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[74:75], s[58:59], s[62:63]
s_and_b64 s[74:75], s[74:75], s[68:69]
v_cndmask_b32 v6, -1, v6, s[74:75]
buffer_store_short v5, v6, s[24:27], 0 offen offset:0
s_branch label_tile_stored_27
label_vector_26:
s_mul_i32 s71, s48, s39
v_mul_lo_u32 v7, s40, v1
v_add_u32 v6, s71, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
v_lshrrev_b32 v2, 16, v2
v_lshrrev_b32 v3, 16, v3
v_pack_b32_f16 v2, v2, v3
buffer_store_dword v2, v6, s[24:27], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s39
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
v_lshrrev_b32 v4, 16, v4
v_lshrrev_b32 v5, 16, v5
v_pack_b32_f16 v4, v4, v5
buffer_store_dword v4, v6, s[24:27], 0 offen offset:0
label_tile_stored_27:
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s74, s4, s38
s_lshl_b32 s74, s74, 2
s_cmp_eq_u32 s70, 1
s_cbranch_scc1 label_vector_28
s_mul_i32 s71, s48, s41
s_add_u32 s71, s71, s74
v_mul_lo_u32 v7, s42, v1
v_add_u32 v6, s71, v7
s_and_b64 s[76:77], s[56:57], s[60:61]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v2, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s42
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[56:57], s[62:63]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v3, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s41
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[58:59], s[60:61]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v4, v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s41
s_add_u32 s73, s73, s72
s_mov_b32 s72, s42
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
s_and_b64 s[76:77], s[58:59], s[62:63]
s_and_b64 s[76:77], s[76:77], s[68:69]
v_cndmask_b32 v6, -1, v6, s[76:77]
buffer_store_dword v5, v6, s[28:31], 0 offen offset:0
s_branch label_tile_stored_29
label_vector_28:
s_mul_i32 s71, s48, s41
s_add_u32 s71, s71, s74
v_mul_lo_u32 v7, s42, v1
v_add_u32 v6, s71, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
buffer_store_dwordx2 v[2:3], v6, s[28:31], 0 offen offset:0
s_mov_b32 s73, s71
s_mov_b32 s72, s41
s_add_u32 s73, s73, s72
v_add_u32 v6, s73, v7
v_cndmask_b32 v6, -1, v6, s[68:69]
buffer_store_dwordx2 v[4:5], v6, s[28:31], 0 offen offset:0
label_tile_stored_29:
label_stored_22:
label_epilogue_end_11:
s_endpgm
.Lgemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end:
.size gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1, .Lgemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end - gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 104
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr 88
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
            { type: int32, semantic: NumWorkGroupsZ } ]
    macrotile: [8, 2, 4096]
    threads: [64, 1, 1]
    grid: [TilesWide, TilesSkinny, Batch]
    kStep: 1024
    noSplitK: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 256
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 2147482625 }
  AssertSizeMultiple: { 3: 1024 }
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
  - .name: gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
    .symbol: gemm_BM2_TN2_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 94
    .vgpr_count: 104
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
