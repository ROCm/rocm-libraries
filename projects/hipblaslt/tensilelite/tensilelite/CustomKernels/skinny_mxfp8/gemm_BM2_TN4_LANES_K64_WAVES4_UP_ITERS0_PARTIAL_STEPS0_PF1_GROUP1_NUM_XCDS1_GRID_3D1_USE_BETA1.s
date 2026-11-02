.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.globl gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.p2align 8
.type gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1,@function
gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1:
s_load_dword s77, s[0:1], 104
s_load_dword s55, s[0:1], 96
s_load_dword s68, s[0:1], 100
s_load_dword s37, s[0:1], 80
s_load_dword s6, s[0:1], 88
s_load_dword s36, s[0:1], 76
s_load_dword s76, s[0:1], 72
s_load_dword s75, s[0:1], 68
s_load_dword s74, s[0:1], 64
s_load_dword s73, s[0:1], 60
s_load_dword s72, s[0:1], 56
s_load_dwordx2 s[28:29], s[0:1], 40
s_load_dwordx2 s[84:85], s[0:1], 32
s_load_dwordx2 s[82:83], s[0:1], 24
s_load_dwordx2 s[80:81], s[0:1], 16
s_load_dwordx2 s[78:79], s[0:1], 8
s_load_dwordx2 s[24:25], s[0:1], 0
s_load_dword s5, s[0:1], 92
s_load_dword s90, s[0:1], 84
s_load_dwordx2 s[32:33], s[0:1], 48
s_waitcnt lgkmcnt(0)
s_add_u32 s91, s36, 3
s_lshr_b32 s91, s91, 2
s_lshl_b32 s91, s91, 2
s_add_u32 s86, s75, -1
s_mul_i32 s86, s86, s73
s_add_u32 s86, s86, s91
s_add_u32 s87, s76, -1
s_mul_i32 s87, s87, s74
s_add_u32 s87, s87, s91
s_add_u32 s69, s36, 255
s_lshr_b32 s69, s69, 8
s_lshl_b32 s69, s69, 8
s_add_u32 s88, s75, 31
s_lshr_b32 s88, s88, 5
s_mul_i32 s88, s88, s69
s_add_u32 s89, s76, 31
s_lshr_b32 s89, s89, 5
s_mul_i32 s89, s89, s69
s_add_u32 s26, s76, -1
s_mul_i32 s26, s26, s72
s_lshl_b32 s91, s75, 1
s_add_u32 s26, s26, s91
s_add_u32 s34, s76, -1
s_mul_i32 s34, s34, s90
s_lshl_b32 s91, s75, 1
s_add_u32 s34, s34, s91
s_mul_i32 s30, s75, s77
s_mul_i32 s30, s30, s76
s_lshl_b32 s30, s30, 2
s_mul_i32 s38, s75, s76
s_lshl_b32 s42, s75, 2
s_cmp_le_u32 s75, s76
s_cselect_b64 s[8:9], s[78:79], s[80:81]
s_cselect_b32 s10, s86, s87
s_cselect_b64 s[12:13], s[80:81], s[78:79]
s_cselect_b32 s14, s87, s86
s_cselect_b64 s[16:17], s[82:83], s[84:85]
s_cselect_b32 s18, s88, s89
s_cselect_b64 s[20:21], s[84:85], s[82:83]
s_cselect_b32 s22, s89, s88
s_cselect_b32 s70, s73, s74
s_cselect_b32 s71, s74, s73
s_cselect_b32 s39, 2, s72
s_cselect_b32 s40, s72, 2
s_cselect_b32 s43, 2, s90
s_cselect_b32 s44, s90, 2
s_cselect_b32 s41, 4, s42
s_cselect_b32 s42, s42, 4
s_cselect_b32 s47, 1, 0
s_min_u32 s45, s75, s76
s_max_u32 s46, s75, s76
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_mov_b32 s31, 131072
s_mov_b32 s35, 131072
v_and_b32 v136, 63, v0
v_lshrrev_b32 v1, 6, v0
v_lshlrev_b32 v1, 2, v1
v_and_b32 v126, 7, v136
v_lshrrev_b32 v126, 1, v126
v_lshrrev_b32 v137, 3, v136
v_lshlrev_b32 v137, 3, v137
v_add_u32 v126, v126, v137
v_lshlrev_b32 v126, 2, v126
v_add_u32 v127, 16, v126
v_lshrrev_b32 v137, 3, v136
v_lshlrev_b32 v137, 9, v137
v_and_b32 v138, 7, v136
v_lshlrev_b32 v138, 4, v138
v_add_u32 v137, v137, v138
v_and_b32 v138, 3, v136
v_lshlrev_b32 v138, 6, v138
v_lshrrev_b32 v139, 3, v136
v_lshlrev_b32 v139, 9, v139
v_add_u32 v138, v138, v139
v_and_b32 v139, 7, v136
v_lshrrev_b32 v139, 2, v139
v_lshlrev_b32 v139, 8, v139
v_add_u32 v138, v138, v139
v_lshrrev_b32 v139, 6, v0
v_lshlrev_b32 v139, 2, v139
v_and_b32 v0, 63, v0
s_mul_i32 s51, s4, s37
s_sub_u32 s49, s36, s51
s_min_i32 s49, s49, s37
s_ashr_i32 s49, s49, 12
s_mul_i32 s73, s68, s4
s_add_u32 s73, s73, s3
s_mul_i32 s73, s73, s55
s_add_u32 s73, s73, s2
s_add_u32 s74, s37, 4095
s_lshr_b32 s74, s74, 12
s_lshr_b32 s75, s73, 4
s_add_u32 s75, s75, s73
s_abs_i32 s77, s74
s_abs_i32 s78, s75
s_xor_b32 s79, s75, s74
s_ashr_i32 s79, s79, 31
v_cvt_f32_u32 v136, s77
v_rcp_iflag_f32 v136, v136
s_sub_u32 s80, 0, s77
v_mul_f32 v136, 0x4f7ffffe, v136
v_cvt_u32_f32 v136, v136
s_nop 0
v_readfirstlane_b32 s81, v136
s_mul_i32 s82, s80, s81
s_mul_hi_u32 s82, s81, s82
s_add_u32 s81, s81, s82
s_mul_hi_u32 s80, s78, s81
s_mul_i32 s82, s80, s77
s_sub_u32 s82, s78, s82
s_add_u32 s83, s80, 1
s_sub_u32 s84, s82, s77
s_cmp_ge_u32 s82, s77
s_cselect_b32 s80, s83, s80
s_cselect_b32 s82, s84, s82
s_add_u32 s83, s80, 1
s_cmp_ge_u32 s82, s77
s_cselect_b32 s80, s83, s80
s_xor_b32 s80, s80, s79
s_sub_u32 s76, s80, s79
s_mul_i32 s76, s76, s74
s_sub_u32 s72, s75, s76
s_cmp_lt_u32 s72, s49
s_cselect_b32 s50, s72, 0
s_lshl_b32 s48, s3, 1
s_lshr_b32 s72, s2, 3
s_and_b32 s72, 1, s72
s_and_b32 s73, 7, s2
s_lshl_b32 s73, s73, 1
s_add_u32 s72, s72, s73
s_and_b32 s73, -16, s2
s_add_u32 s72, s72, s73
s_and_b32 s73, -16, s55
s_cmp_lt_u32 s2, s73
s_cselect_b32 s2, s72, s2
s_lshl_b32 s72, s2, 4
v_add_u32 v1, s72, v1
v_add_u32 v139, s72, v139
s_add_u32 s72, s48, 0
s_cmp_lt_u32 s72, s45
s_cselect_b64 s[56:57], -1, 0
s_mul_i32 s72, s72, s70
v_add_u32 v112, s72, v137
v_cndmask_b32 v112, -1, v112, s[56:57]
s_add_u32 s72, s48, 1
s_cmp_lt_u32 s72, s45
s_cselect_b64 s[58:59], -1, 0
s_mul_i32 s72, s72, s70
v_add_u32 v113, s72, v137
v_cndmask_b32 v113, -1, v113, s[58:59]
v_add_u32 v136, 0, v1
v_cmp_lt_u32 s[60:61], v136, s46
s_nop 1
v_mul_lo_u32 v136, v136, s71
v_add_u32 v114, v136, v137
v_cndmask_b32 v114, -1, v114, s[60:61]
v_add_u32 v136, 1, v1
v_cmp_lt_u32 s[62:63], v136, s46
s_nop 1
v_mul_lo_u32 v136, v136, s71
v_add_u32 v115, v136, v137
v_cndmask_b32 v115, -1, v115, s[62:63]
v_add_u32 v136, 2, v1
v_cmp_lt_u32 s[64:65], v136, s46
s_nop 1
v_mul_lo_u32 v136, v136, s71
v_add_u32 v116, v136, v137
v_cndmask_b32 v116, -1, v116, s[64:65]
v_add_u32 v136, 3, v1
v_cmp_lt_u32 s[66:67], v136, s46
s_nop 1
v_mul_lo_u32 v136, v136, s71
v_add_u32 v117, v136, v137
v_cndmask_b32 v117, -1, v117, s[66:67]
s_add_u32 s72, s48, 0
s_lshr_b32 s73, s72, 5
s_mul_i32 s73, s73, s69
s_and_b32 s72, 15, s72
s_lshl_b32 s72, s72, 2
s_add_u32 s73, s73, s72
v_add_u32 v118, s73, v138
v_cndmask_b32 v118, -1, v118, s[56:57]
s_add_u32 s72, s48, 1
s_lshr_b32 s73, s72, 5
s_mul_i32 s73, s73, s69
s_and_b32 s72, 15, s72
s_lshl_b32 s72, s72, 2
s_add_u32 s73, s73, s72
v_add_u32 v119, s73, v138
v_cndmask_b32 v119, -1, v119, s[58:59]
v_add_u32 v136, 0, v139
v_cmp_lt_u32 s[72:73], v136, s46
s_nop 1
v_lshrrev_b32 v140, 5, v136
v_mul_lo_u32 v140, v140, s69
v_and_b32 v136, 15, v136
v_lshlrev_b32 v136, 2, v136
v_add_u32 v140, v140, v136
v_add_u32 v120, v140, v138
v_cndmask_b32 v120, -1, v120, s[72:73]
v_add_u32 v136, 1, v139
v_cmp_lt_u32 s[72:73], v136, s46
s_nop 1
v_lshrrev_b32 v140, 5, v136
v_mul_lo_u32 v140, v140, s69
v_and_b32 v136, 15, v136
v_lshlrev_b32 v136, 2, v136
v_add_u32 v140, v140, v136
v_add_u32 v121, v140, v138
v_cndmask_b32 v121, -1, v121, s[72:73]
v_add_u32 v136, 2, v139
v_cmp_lt_u32 s[72:73], v136, s46
s_nop 1
v_lshrrev_b32 v140, 5, v136
v_mul_lo_u32 v140, v140, s69
v_and_b32 v136, 15, v136
v_lshlrev_b32 v136, 2, v136
v_add_u32 v140, v140, v136
v_add_u32 v122, v140, v138
v_cndmask_b32 v122, -1, v122, s[72:73]
v_add_u32 v136, 3, v139
v_cmp_lt_u32 s[72:73], v136, s46
s_nop 1
v_lshrrev_b32 v140, 5, v136
v_mul_lo_u32 v140, v140, s69
v_and_b32 v136, 15, v136
v_lshlrev_b32 v136, 2, v136
v_add_u32 v140, v140, v136
v_add_u32 v123, v140, v138
v_cndmask_b32 v123, -1, v123, s[72:73]
s_lshr_b32 s72, s48, 4
s_and_b32 s72, 1, s72
s_mul_i32 s72, s72, 16843009
s_add_u32 s72, s72, 100925952
v_mov_b32 v124, s72
v_lshrrev_b32 v125, 4, v1
v_and_b32 v125, 1, v125
v_mov_b32 v136, 16843009
v_mul_lo_u32 v125, v125, v136
v_add_u32 v125, 100925952, v125
v_mov_b32 v104, 0
v_mov_b32 v105, 0
v_mov_b32 v106, 0
v_mov_b32 v107, 0
v_mov_b32 v108, 0
v_mov_b32 v109, 0
v_mov_b32 v110, 0
v_mov_b32 v111, 0
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_c_prefetch_skip_0
s_and_b32 s72, s5, 2147483647
s_cmp_eq_u32 s72, 0
s_cbranch_scc1 label_c_prefetch_skip_0
v_cmp_eq_i32 s[72:73], 0, v0
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_1
s_add_u32 s75, s3, 1
s_lshl_b32 s75, s75, 1
s_add_u32 s76, s2, 1
s_lshl_b32 s76, s76, 4
s_cmp_le_u32 s75, s45
s_cselect_b32 s77, 1, 0
s_cmp_le_u32 s76, s46
s_cselect_b32 s78, 1, 0
s_and_b32 s74, s77, s78
s_mul_i32 s75, s48, s43
v_mul_lo_u32 v137, s44, v1
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_c_vector_3
v_add_u32 v136, s75, v137
s_and_b64 s[78:79], s[56:57], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v128, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s44
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v130, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s44, 1
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v132, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s44, 3
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v134, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v129, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_mov_b32 s76, s44
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v131, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s44, 1
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v133, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_mul_i32 s76, s44, 3
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v135, v136, s[32:35], 0 offen offset:0
s_branch label_c_loaded_4
label_c_vector_3:
v_add_u32 v136, s75, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dword v128, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s44
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dword v130, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s44, 1
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dword v132, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s44, 3
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dword v134, v136, s[32:35], 0 offen offset:0
label_c_loaded_4:
s_branch label_epilogue_end_2
label_orientation_b_1:
s_add_u32 s75, s3, 1
s_lshl_b32 s75, s75, 1
s_add_u32 s76, s2, 1
s_lshl_b32 s76, s76, 4
s_cmp_le_u32 s75, s45
s_cselect_b32 s77, 1, 0
s_cmp_le_u32 s76, s46
s_cselect_b32 s78, 1, 0
s_and_b32 s74, s77, s78
s_mul_i32 s75, s48, s43
v_mul_lo_u32 v137, s44, v1
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v136, s75, v137
s_and_b64 s[78:79], s[56:57], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v128, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s44
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v129, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s44, 1
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v130, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s44, 3
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[56:57], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v131, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v132, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_mov_b32 s76, s44
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v133, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s44, 1
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v134, v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
s_mul_i32 s76, s44, 3
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
s_and_b64 s[78:79], s[58:59], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v136, -1, v136, s[78:79]
buffer_load_ushort v135, v136, s[32:35], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v136, s75, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dwordx2 v[128:129], v136, s[32:35], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s43
s_add_u32 s77, s77, s76
v_add_u32 v136, s77, v137
v_cndmask_b32 v136, -1, v136, s[72:73]
buffer_load_dwordx2 v[130:131], v136, s[32:35], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_2:
label_c_prefetch_skip_0:
s_sub_u32 s73, s50, s49
s_cmp_ge_u32 s50, s49
s_cselect_b32 s72, s73, s50
s_lshl_b32 s73, s72, 12
s_add_u32 s73, s73, s51
buffer_load_dword v98, v118, s[16:19], s73 offen offset:0
buffer_load_dword v99, v119, s[16:19], s73 offen offset:0
buffer_load_dword v100, v120, s[20:23], s73 offen offset:0
buffer_load_dword v101, v121, s[20:23], s73 offen offset:0
buffer_load_dword v102, v122, s[20:23], s73 offen offset:0
buffer_load_dword v103, v123, s[20:23], s73 offen offset:0
buffer_load_dwordx4 v[2:5], v112, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[6:9], v113, s[8:11], s73 offen offset:0
buffer_load_dwordx4 v[10:13], v114, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[14:17], v115, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[18:21], v116, s[12:15], s73 offen offset:0
buffer_load_dwordx4 v[22:25], v117, s[12:15], s73 offen offset:0
s_add_u32 s74, s73, 128
buffer_load_dwordx4 v[26:29], v112, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[30:33], v113, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[34:37], v114, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[38:41], v115, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[42:45], v116, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[46:49], v117, s[12:15], s74 offen offset:0
s_add_u32 s74, s73, 256
buffer_load_dwordx4 v[50:53], v112, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[54:57], v113, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[58:61], v114, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[62:65], v115, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[66:69], v116, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[70:73], v117, s[12:15], s74 offen offset:0
s_add_u32 s74, s73, 384
buffer_load_dwordx4 v[74:77], v112, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[78:81], v113, s[8:11], s74 offen offset:0
buffer_load_dwordx4 v[82:85], v114, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[86:89], v115, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[90:93], v116, s[12:15], s74 offen offset:0
buffer_load_dwordx4 v[94:97], v117, s[12:15], s74 offen offset:0
s_add_u32 s72, s49, -1
s_max_i32 s72, s72, 0
s_add_u32 s73, s50, 1
s_add_u32 s52, s51, 128
s_add_u32 s53, s51, 256
s_add_u32 s54, s51, 384
s_cmp_gt_i32 s72, 0
s_mov_b32 s74, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_add_u32 s75, s74, s73
s_sub_u32 s76, s75, s49
s_cmp_ge_u32 s75, s49
s_cselect_b32 s75, s76, s75
s_waitcnt vmcnt(29)
ds_bpermute_b32 v136, v126, v98
ds_bpermute_b32 v98, v127, v98
s_waitcnt vmcnt(28)
ds_bpermute_b32 v137, v126, v99
ds_bpermute_b32 v99, v127, v99
s_waitcnt vmcnt(27)
ds_bpermute_b32 v138, v126, v100
ds_bpermute_b32 v100, v127, v100
s_waitcnt vmcnt(26)
ds_bpermute_b32 v139, v126, v101
ds_bpermute_b32 v101, v127, v101
s_waitcnt vmcnt(25)
ds_bpermute_b32 v140, v126, v102
ds_bpermute_b32 v102, v127, v102
s_waitcnt vmcnt(24)
ds_bpermute_b32 v141, v126, v103
ds_bpermute_b32 v103, v127, v103
s_waitcnt lgkmcnt(10)
v_perm_b32 v136, v98, v136, v124
s_waitcnt lgkmcnt(8)
v_perm_b32 v137, v99, v137, v124
s_waitcnt lgkmcnt(6)
v_perm_b32 v138, v100, v138, v125
s_waitcnt lgkmcnt(4)
v_perm_b32 v139, v101, v139, v125
s_waitcnt lgkmcnt(2)
v_perm_b32 v140, v102, v140, v125
s_waitcnt lgkmcnt(0)
v_perm_b32 v141, v103, v141, v125
s_lshl_b32 s76, s75, 12
s_add_u32 s76, s76, s51
buffer_load_dword v98, v118, s[16:19], s76 offen offset:0
buffer_load_dword v99, v119, s[16:19], s76 offen offset:0
buffer_load_dword v100, v120, s[20:23], s76 offen offset:0
buffer_load_dword v101, v121, s[20:23], s76 offen offset:0
buffer_load_dword v102, v122, s[20:23], s76 offen offset:0
buffer_load_dword v103, v123, s[20:23], s76 offen offset:0
v_lshlrev_b32 v142, 23, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 23, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 23, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 23, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 23, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 23, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(29)
v_cvt_scalef32_pk_bf16_fp8 v148, v2, v142
s_waitcnt vmcnt(28)
v_cvt_scalef32_pk_bf16_fp8 v149, v6, v143
s_waitcnt vmcnt(27)
v_cvt_scalef32_pk_bf16_fp8 v150, v10, v144
s_waitcnt vmcnt(26)
v_cvt_scalef32_pk_bf16_fp8 v151, v14, v145
s_waitcnt vmcnt(25)
v_cvt_scalef32_pk_bf16_fp8 v152, v18, v146
s_waitcnt vmcnt(24)
v_cvt_scalef32_pk_bf16_fp8 v153, v22, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v2, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v6, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v10, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v14, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v18, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v22, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v3, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v7, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v11, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v15, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v19, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v23, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v3, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v7, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v11, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v15, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v19, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v23, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v4, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v8, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v12, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v16, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v20, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v24, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v4, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v8, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v12, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v16, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v20, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v24, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v5, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v9, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v13, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v17, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v21, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v25, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v5, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v9, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v13, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v17, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v21, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v25, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
buffer_load_dwordx4 v[2:5], v112, s[8:11], s76 offen offset:0
buffer_load_dwordx4 v[6:9], v113, s[8:11], s76 offen offset:0
buffer_load_dwordx4 v[10:13], v114, s[12:15], s76 offen offset:0
buffer_load_dwordx4 v[14:17], v115, s[12:15], s76 offen offset:0
buffer_load_dwordx4 v[18:21], v116, s[12:15], s76 offen offset:0
buffer_load_dwordx4 v[22:25], v117, s[12:15], s76 offen offset:0
v_lshlrev_b32 v142, 15, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 15, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 15, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 15, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 15, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 15, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(29)
v_cvt_scalef32_pk_bf16_fp8 v148, v26, v142
s_waitcnt vmcnt(28)
v_cvt_scalef32_pk_bf16_fp8 v149, v30, v143
s_waitcnt vmcnt(27)
v_cvt_scalef32_pk_bf16_fp8 v150, v34, v144
s_waitcnt vmcnt(26)
v_cvt_scalef32_pk_bf16_fp8 v151, v38, v145
s_waitcnt vmcnt(25)
v_cvt_scalef32_pk_bf16_fp8 v152, v42, v146
s_waitcnt vmcnt(24)
v_cvt_scalef32_pk_bf16_fp8 v153, v46, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v26, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v30, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v34, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v38, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v42, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v46, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v27, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v31, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v35, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v39, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v43, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v47, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v27, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v31, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v35, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v39, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v43, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v47, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v28, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v32, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v36, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v40, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v44, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v48, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v28, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v32, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v36, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v40, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v44, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v48, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v29, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v33, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v37, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v41, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v45, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v49, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v29, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v33, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v37, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v41, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v45, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v49, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
s_lshl_b32 s75, s75, 12
s_add_u32 s77, s52, s75
buffer_load_dwordx4 v[26:29], v112, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[30:33], v113, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[34:37], v114, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[38:41], v115, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[42:45], v116, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[46:49], v117, s[12:15], s77 offen offset:0
v_lshlrev_b32 v142, 7, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 7, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 7, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 7, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 7, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 7, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(29)
v_cvt_scalef32_pk_bf16_fp8 v148, v50, v142
s_waitcnt vmcnt(28)
v_cvt_scalef32_pk_bf16_fp8 v149, v54, v143
s_waitcnt vmcnt(27)
v_cvt_scalef32_pk_bf16_fp8 v150, v58, v144
s_waitcnt vmcnt(26)
v_cvt_scalef32_pk_bf16_fp8 v151, v62, v145
s_waitcnt vmcnt(25)
v_cvt_scalef32_pk_bf16_fp8 v152, v66, v146
s_waitcnt vmcnt(24)
v_cvt_scalef32_pk_bf16_fp8 v153, v70, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v50, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v54, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v58, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v62, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v66, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v70, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v51, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v55, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v59, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v63, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v67, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v71, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v51, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v55, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v59, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v63, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v67, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v71, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v52, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v56, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v60, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v64, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v68, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v72, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v52, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v56, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v60, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v64, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v68, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v72, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v53, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v57, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v61, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v65, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v69, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v73, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v53, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v57, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v61, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v65, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v69, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v73, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
s_add_u32 s77, s53, s75
buffer_load_dwordx4 v[50:53], v112, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[54:57], v113, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[58:61], v114, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[62:65], v115, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[66:69], v116, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[70:73], v117, s[12:15], s77 offen offset:0
v_lshrrev_b32 v142, 1, v136
v_and_b32 v142, 2139095040, v142
v_lshrrev_b32 v143, 1, v137
v_and_b32 v143, 2139095040, v143
v_lshrrev_b32 v144, 1, v138
v_and_b32 v144, 2139095040, v144
v_lshrrev_b32 v145, 1, v139
v_and_b32 v145, 2139095040, v145
v_lshrrev_b32 v146, 1, v140
v_and_b32 v146, 2139095040, v146
v_lshrrev_b32 v147, 1, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(29)
v_cvt_scalef32_pk_bf16_fp8 v148, v74, v142
s_waitcnt vmcnt(28)
v_cvt_scalef32_pk_bf16_fp8 v149, v78, v143
s_waitcnt vmcnt(27)
v_cvt_scalef32_pk_bf16_fp8 v150, v82, v144
s_waitcnt vmcnt(26)
v_cvt_scalef32_pk_bf16_fp8 v151, v86, v145
s_waitcnt vmcnt(25)
v_cvt_scalef32_pk_bf16_fp8 v152, v90, v146
s_waitcnt vmcnt(24)
v_cvt_scalef32_pk_bf16_fp8 v153, v94, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v74, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v78, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v82, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v86, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v90, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v94, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v75, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v79, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v83, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v87, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v91, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v95, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v75, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v79, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v83, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v87, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v91, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v95, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v76, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v80, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v84, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v88, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v92, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v96, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v76, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v80, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v84, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v88, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v92, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v96, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v77, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v81, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v85, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v89, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v93, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v97, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v77, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v81, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v85, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v89, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v93, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v97, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
s_add_u32 s77, s54, s75
buffer_load_dwordx4 v[74:77], v112, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[78:81], v113, s[8:11], s77 offen offset:0
buffer_load_dwordx4 v[82:85], v114, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[86:89], v115, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[90:93], v116, s[12:15], s77 offen offset:0
buffer_load_dwordx4 v[94:97], v117, s[12:15], s77 offen offset:0
s_add_u32 s74, s74, 1
s_cmp_lt_i32 s74, s72
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s73, s49, s72
s_cmp_eq_i32 s73, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(29)
ds_bpermute_b32 v136, v126, v98
ds_bpermute_b32 v98, v127, v98
s_waitcnt vmcnt(28)
ds_bpermute_b32 v137, v126, v99
ds_bpermute_b32 v99, v127, v99
s_waitcnt vmcnt(27)
ds_bpermute_b32 v138, v126, v100
ds_bpermute_b32 v100, v127, v100
s_waitcnt vmcnt(26)
ds_bpermute_b32 v139, v126, v101
ds_bpermute_b32 v101, v127, v101
s_waitcnt vmcnt(25)
ds_bpermute_b32 v140, v126, v102
ds_bpermute_b32 v102, v127, v102
s_waitcnt vmcnt(24)
ds_bpermute_b32 v141, v126, v103
ds_bpermute_b32 v103, v127, v103
s_waitcnt lgkmcnt(10)
v_perm_b32 v136, v98, v136, v124
s_waitcnt lgkmcnt(8)
v_perm_b32 v137, v99, v137, v124
s_waitcnt lgkmcnt(6)
v_perm_b32 v138, v100, v138, v125
s_waitcnt lgkmcnt(4)
v_perm_b32 v139, v101, v139, v125
s_waitcnt lgkmcnt(2)
v_perm_b32 v140, v102, v140, v125
s_waitcnt lgkmcnt(0)
v_perm_b32 v141, v103, v141, v125
v_lshlrev_b32 v142, 23, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 23, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 23, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 23, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 23, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 23, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(23)
v_cvt_scalef32_pk_bf16_fp8 v148, v2, v142
s_waitcnt vmcnt(22)
v_cvt_scalef32_pk_bf16_fp8 v149, v6, v143
s_waitcnt vmcnt(21)
v_cvt_scalef32_pk_bf16_fp8 v150, v10, v144
s_waitcnt vmcnt(20)
v_cvt_scalef32_pk_bf16_fp8 v151, v14, v145
s_waitcnt vmcnt(19)
v_cvt_scalef32_pk_bf16_fp8 v152, v18, v146
s_waitcnt vmcnt(18)
v_cvt_scalef32_pk_bf16_fp8 v153, v22, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v2, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v6, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v10, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v14, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v18, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v22, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v3, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v7, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v11, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v15, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v19, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v23, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v3, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v7, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v11, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v15, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v19, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v23, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v4, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v8, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v12, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v16, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v20, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v24, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v4, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v8, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v12, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v16, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v20, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v24, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v5, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v9, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v13, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v17, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v21, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v25, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v5, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v9, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v13, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v17, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v21, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v25, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_lshlrev_b32 v142, 15, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 15, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 15, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 15, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 15, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 15, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(17)
v_cvt_scalef32_pk_bf16_fp8 v148, v26, v142
s_waitcnt vmcnt(16)
v_cvt_scalef32_pk_bf16_fp8 v149, v30, v143
s_waitcnt vmcnt(15)
v_cvt_scalef32_pk_bf16_fp8 v150, v34, v144
s_waitcnt vmcnt(14)
v_cvt_scalef32_pk_bf16_fp8 v151, v38, v145
s_waitcnt vmcnt(13)
v_cvt_scalef32_pk_bf16_fp8 v152, v42, v146
s_waitcnt vmcnt(12)
v_cvt_scalef32_pk_bf16_fp8 v153, v46, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v26, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v30, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v34, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v38, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v42, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v46, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v27, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v31, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v35, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v39, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v43, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v47, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v27, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v31, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v35, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v39, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v43, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v47, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v28, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v32, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v36, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v40, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v44, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v48, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v28, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v32, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v36, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v40, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v44, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v48, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v29, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v33, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v37, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v41, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v45, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v49, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v29, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v33, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v37, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v41, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v45, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v49, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_lshlrev_b32 v142, 7, v136
v_and_b32 v142, 2139095040, v142
v_lshlrev_b32 v143, 7, v137
v_and_b32 v143, 2139095040, v143
v_lshlrev_b32 v144, 7, v138
v_and_b32 v144, 2139095040, v144
v_lshlrev_b32 v145, 7, v139
v_and_b32 v145, 2139095040, v145
v_lshlrev_b32 v146, 7, v140
v_and_b32 v146, 2139095040, v146
v_lshlrev_b32 v147, 7, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(11)
v_cvt_scalef32_pk_bf16_fp8 v148, v50, v142
s_waitcnt vmcnt(10)
v_cvt_scalef32_pk_bf16_fp8 v149, v54, v143
s_waitcnt vmcnt(9)
v_cvt_scalef32_pk_bf16_fp8 v150, v58, v144
s_waitcnt vmcnt(8)
v_cvt_scalef32_pk_bf16_fp8 v151, v62, v145
s_waitcnt vmcnt(7)
v_cvt_scalef32_pk_bf16_fp8 v152, v66, v146
s_waitcnt vmcnt(6)
v_cvt_scalef32_pk_bf16_fp8 v153, v70, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v50, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v54, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v58, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v62, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v66, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v70, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v51, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v55, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v59, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v63, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v67, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v71, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v51, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v55, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v59, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v63, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v67, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v71, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v52, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v56, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v60, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v64, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v68, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v72, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v52, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v56, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v60, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v64, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v68, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v72, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v53, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v57, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v61, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v65, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v69, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v73, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v53, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v57, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v61, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v65, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v69, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v73, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_lshrrev_b32 v142, 1, v136
v_and_b32 v142, 2139095040, v142
v_lshrrev_b32 v143, 1, v137
v_and_b32 v143, 2139095040, v143
v_lshrrev_b32 v144, 1, v138
v_and_b32 v144, 2139095040, v144
v_lshrrev_b32 v145, 1, v139
v_and_b32 v145, 2139095040, v145
v_lshrrev_b32 v146, 1, v140
v_and_b32 v146, 2139095040, v146
v_lshrrev_b32 v147, 1, v141
v_and_b32 v147, 2139095040, v147
s_waitcnt vmcnt(5)
v_cvt_scalef32_pk_bf16_fp8 v148, v74, v142
s_waitcnt vmcnt(4)
v_cvt_scalef32_pk_bf16_fp8 v149, v78, v143
s_waitcnt vmcnt(3)
v_cvt_scalef32_pk_bf16_fp8 v150, v82, v144
s_waitcnt vmcnt(2)
v_cvt_scalef32_pk_bf16_fp8 v151, v86, v145
s_waitcnt vmcnt(1)
v_cvt_scalef32_pk_bf16_fp8 v152, v90, v146
s_waitcnt vmcnt(0)
v_cvt_scalef32_pk_bf16_fp8 v153, v94, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v74, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v78, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v82, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v86, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v90, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v94, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v75, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v79, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v83, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v87, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v91, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v95, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v75, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v79, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v83, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v87, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v91, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v95, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v76, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v80, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v84, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v88, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v92, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v96, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v76, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v80, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v84, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v88, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v92, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v96, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v77, v142
v_cvt_scalef32_pk_bf16_fp8 v149, v81, v143
v_cvt_scalef32_pk_bf16_fp8 v150, v85, v144
v_cvt_scalef32_pk_bf16_fp8 v151, v89, v145
v_cvt_scalef32_pk_bf16_fp8 v152, v93, v146
v_cvt_scalef32_pk_bf16_fp8 v153, v97, v147
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
v_cvt_scalef32_pk_bf16_fp8 v148, v77, v142 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v149, v81, v143 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v150, v85, v144 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v151, v89, v145 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v152, v93, v146 op_sel:[1,0,0]
v_cvt_scalef32_pk_bf16_fp8 v153, v97, v147 op_sel:[1,0,0]
v_dot2_f32_bf16 v104, v148, v150, v104
v_dot2_f32_bf16 v105, v148, v151, v105
v_dot2_f32_bf16 v106, v148, v152, v106
v_dot2_f32_bf16 v107, v148, v153, v107
v_dot2_f32_bf16 v108, v149, v150, v108
v_dot2_f32_bf16 v109, v149, v151, v109
v_dot2_f32_bf16 v110, v149, v152, v110
v_dot2_f32_bf16 v111, v149, v153, v111
label_arm_end_9:
s_cmp_eq_u32 s47, 1
s_cbranch_scc0 label_orientation_b_10
s_waitcnt vmcnt(21)
v_add_f32 v2, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v2, v2, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v2, 63
v_nop
v_nop
v_mov_b32 v2, s74
v_add_f32 v4, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v4, v4, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v4, 63
v_nop
v_nop
v_mov_b32 v4, s74
v_add_f32 v6, v106, v106 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v6, v6, v6 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v6, 63
v_nop
v_nop
v_mov_b32 v6, s74
v_add_f32 v8, v107, v107 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v8, v8, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v8, 63
v_nop
v_nop
v_mov_b32 v8, s74
v_add_f32 v3, v108, v108 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v3, v3, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v3, 63
v_nop
v_nop
v_mov_b32 v3, s74
v_add_f32 v5, v109, v109 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v5, v5, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v5, 63
v_nop
v_nop
v_mov_b32 v5, s74
v_add_f32 v7, v110, v110 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v7, v7, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v7, 63
v_nop
v_nop
v_mov_b32 v7, s74
v_add_f32 v9, v111, v111 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v9, v9, v9 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v9, 63
v_nop
v_nop
v_mov_b32 v9, s74
v_cmp_eq_i32 s[72:73], 0, v0
s_add_u32 s75, s3, 1
s_lshl_b32 s75, s75, 1
s_add_u32 s76, s2, 1
s_lshl_b32 s76, s76, 4
s_cmp_le_u32 s75, s45
s_cselect_b32 s77, 1, 0
s_cmp_le_u32 s76, s46
s_cselect_b32 s78, 1, 0
s_and_b32 s74, s77, s78
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_12
v_pk_mul_f32 v[2:3], s[6:7], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], s[6:7], v[4:5] op_sel_hi:[0,1]
v_pk_mul_f32 v[6:7], s[6:7], v[6:7] op_sel_hi:[0,1]
v_pk_mul_f32 v[8:9], s[6:7], v[8:9] op_sel_hi:[0,1]
s_and_b32 s75, s5, 2147483647
s_cmp_eq_u32 s75, 0
s_cbranch_scc1 label_no_beta_14
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s74, 1
s_cbranch_scc0 label_c_masked_15
v_lshlrev_b32 v12, 16, v128
v_fmac_f32 v2, s5, v12
v_lshlrev_b32 v12, 16, v130
v_fmac_f32 v4, s5, v12
v_lshlrev_b32 v12, 16, v132
v_fmac_f32 v6, s5, v12
v_lshlrev_b32 v12, 16, v134
v_fmac_f32 v8, s5, v12
v_and_b32 v12, -65536, v128
v_fmac_f32 v3, s5, v12
v_and_b32 v12, -65536, v130
v_fmac_f32 v5, s5, v12
v_and_b32 v12, -65536, v132
v_fmac_f32 v7, s5, v12
v_and_b32 v12, -65536, v134
v_fmac_f32 v9, s5, v12
s_branch label_c_added_16
label_c_masked_15:
v_lshlrev_b32 v12, 16, v128
v_fmac_f32 v2, s5, v12
v_lshlrev_b32 v12, 16, v130
v_fmac_f32 v4, s5, v12
v_lshlrev_b32 v12, 16, v132
v_fmac_f32 v6, s5, v12
v_lshlrev_b32 v12, 16, v134
v_fmac_f32 v8, s5, v12
v_lshlrev_b32 v12, 16, v129
v_fmac_f32 v3, s5, v12
v_lshlrev_b32 v12, 16, v131
v_fmac_f32 v5, s5, v12
v_lshlrev_b32 v12, 16, v133
v_fmac_f32 v7, s5, v12
v_lshlrev_b32 v12, 16, v135
v_fmac_f32 v9, s5, v12
label_c_added_16:
label_no_beta_14:
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_vector_17
s_mul_i32 s75, s48, s39
v_mul_lo_u32 v11, s40, v1
v_lshrrev_b32 v2, 16, v2
v_add_u32 v10, s75, v11
s_and_b64 s[78:79], s[56:57], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v2, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v4, 16, v4
s_mov_b32 s77, s75
s_mov_b32 s76, s40
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v4, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v6, 16, v6
s_mov_b32 s77, s75
s_lshl_b32 s76, s40, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v6, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v8, 16, v8
s_mov_b32 s77, s75
s_mul_i32 s76, s40, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v8, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v3, 16, v3
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v3, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v5, 16, v5
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_mov_b32 s76, s40
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v5, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v7, 16, v7
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s40, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v7, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v9, 16, v9
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_mul_i32 s76, s40, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v9, v10, s[24:27], 0 offen offset:0
s_branch label_tile_stored_18
label_vector_17:
s_mul_i32 s75, s48, s39
v_mul_lo_u32 v11, s40, v1
v_add_u32 v10, s75, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v2, 16, v2
v_lshrrev_b32 v3, 16, v3
v_pack_b32_f16 v2, v2, v3
buffer_store_dword v2, v10, s[24:27], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s40
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v4, 16, v4
v_lshrrev_b32 v5, 16, v5
v_pack_b32_f16 v4, v4, v5
buffer_store_dword v4, v10, s[24:27], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s40, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v6, 16, v6
v_lshrrev_b32 v7, 16, v7
v_pack_b32_f16 v6, v6, v7
buffer_store_dword v6, v10, s[24:27], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s40, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v8, 16, v8
v_lshrrev_b32 v9, 16, v9
v_pack_b32_f16 v8, v8, v9
buffer_store_dword v8, v10, s[24:27], 0 offen offset:0
label_tile_stored_18:
s_branch label_stored_13
label_workspace_12:
s_mul_i32 s78, s4, s38
s_lshl_b32 s78, s78, 2
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_vector_19
s_mul_i32 s75, s48, s41
s_add_u32 s75, s75, s78
v_mul_lo_u32 v11, s42, v1
v_add_u32 v10, s75, v11
s_and_b64 s[80:81], s[56:57], s[60:61]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v2, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s42
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[62:63]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v4, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s42, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[64:65]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v6, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s42, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[66:67]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v8, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[60:61]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v3, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_mov_b32 s76, s42
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[62:63]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v5, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s42, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[64:65]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v7, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_mul_i32 s76, s42, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[66:67]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v9, v10, s[28:31], 0 offen offset:0
s_branch label_tile_stored_20
label_vector_19:
s_mul_i32 s75, s48, s41
s_add_u32 s75, s75, s78
v_mul_lo_u32 v11, s42, v1
v_add_u32 v10, s75, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx2 v[2:3], v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s42
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx2 v[4:5], v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s42, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx2 v[6:7], v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s42, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx2 v[8:9], v10, s[28:31], 0 offen offset:0
label_tile_stored_20:
label_stored_13:
s_branch label_epilogue_end_11
label_orientation_b_10:
s_waitcnt vmcnt(21)
v_add_f32 v2, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v2, v2, v2 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v2, v2, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v2, 63
v_nop
v_nop
v_mov_b32 v2, s74
v_add_f32 v3, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v3, v3, v3 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v3, v3, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v3, 63
v_nop
v_nop
v_mov_b32 v3, s74
v_add_f32 v4, v106, v106 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v4, v4, v4 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v4, v4, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v4, 63
v_nop
v_nop
v_mov_b32 v4, s74
v_add_f32 v5, v107, v107 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v5, v5, v5 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v5, v5, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v5, 63
v_nop
v_nop
v_mov_b32 v5, s74
v_add_f32 v6, v108, v108 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v6, v6, v6 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v6, v6, v6 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v6, 63
v_nop
v_nop
v_mov_b32 v6, s74
v_add_f32 v7, v109, v109 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v7, v7, v7 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v7, v7, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v7, 63
v_nop
v_nop
v_mov_b32 v7, s74
v_add_f32 v8, v110, v110 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v8, v8, v8 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v8, v8, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v8, 63
v_nop
v_nop
v_mov_b32 v8, s74
v_add_f32 v9, v111, v111 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_nop
v_add_f32 v9, v9, v9 row_bcast:15 row_mask:0xa bank_mask:0xf
v_nop
v_nop
v_add_f32 v9, v9, v9 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_nop
v_readlane_b32 s74, v9, 63
v_nop
v_nop
v_mov_b32 v9, s74
v_cmp_eq_i32 s[72:73], 0, v0
s_add_u32 s75, s3, 1
s_lshl_b32 s75, s75, 1
s_add_u32 s76, s2, 1
s_lshl_b32 s76, s76, 4
s_cmp_le_u32 s75, s45
s_cselect_b32 s77, 1, 0
s_cmp_le_u32 s76, s46
s_cselect_b32 s78, 1, 0
s_and_b32 s74, s77, s78
s_cmp_le_u32 s36, s37
s_cbranch_scc0 label_workspace_21
v_pk_mul_f32 v[2:3], s[6:7], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], s[6:7], v[4:5] op_sel_hi:[0,1]
v_pk_mul_f32 v[6:7], s[6:7], v[6:7] op_sel_hi:[0,1]
v_pk_mul_f32 v[8:9], s[6:7], v[8:9] op_sel_hi:[0,1]
s_and_b32 s75, s5, 2147483647
s_cmp_eq_u32 s75, 0
s_cbranch_scc1 label_no_beta_23
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s74, 1
s_cbranch_scc0 label_c_masked_24
v_lshlrev_b32 v12, 16, v128
v_fmac_f32 v2, s5, v12
v_and_b32 v12, -65536, v128
v_fmac_f32 v3, s5, v12
v_lshlrev_b32 v12, 16, v129
v_fmac_f32 v4, s5, v12
v_and_b32 v12, -65536, v129
v_fmac_f32 v5, s5, v12
v_lshlrev_b32 v12, 16, v130
v_fmac_f32 v6, s5, v12
v_and_b32 v12, -65536, v130
v_fmac_f32 v7, s5, v12
v_lshlrev_b32 v12, 16, v131
v_fmac_f32 v8, s5, v12
v_and_b32 v12, -65536, v131
v_fmac_f32 v9, s5, v12
s_branch label_c_added_25
label_c_masked_24:
v_lshlrev_b32 v12, 16, v128
v_fmac_f32 v2, s5, v12
v_lshlrev_b32 v12, 16, v129
v_fmac_f32 v3, s5, v12
v_lshlrev_b32 v12, 16, v130
v_fmac_f32 v4, s5, v12
v_lshlrev_b32 v12, 16, v131
v_fmac_f32 v5, s5, v12
v_lshlrev_b32 v12, 16, v132
v_fmac_f32 v6, s5, v12
v_lshlrev_b32 v12, 16, v133
v_fmac_f32 v7, s5, v12
v_lshlrev_b32 v12, 16, v134
v_fmac_f32 v8, s5, v12
v_lshlrev_b32 v12, 16, v135
v_fmac_f32 v9, s5, v12
label_c_added_25:
label_no_beta_23:
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_vector_26
s_mul_i32 s75, s48, s39
v_mul_lo_u32 v11, s40, v1
v_lshrrev_b32 v2, 16, v2
v_add_u32 v10, s75, v11
s_and_b64 s[78:79], s[56:57], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v2, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v3, 16, v3
s_mov_b32 s77, s75
s_mov_b32 s76, s40
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v3, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v4, 16, v4
s_mov_b32 s77, s75
s_lshl_b32 s76, s40, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v4, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v5, 16, v5
s_mov_b32 s77, s75
s_mul_i32 s76, s40, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[56:57], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v5, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v6, 16, v6
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[60:61]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v6, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v7, 16, v7
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_mov_b32 s76, s40
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[62:63]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v7, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v8, 16, v8
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s40, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[64:65]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v8, v10, s[24:27], 0 offen offset:0
v_lshrrev_b32 v9, 16, v9
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
s_mul_i32 s76, s40, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[78:79], s[58:59], s[66:67]
s_and_b64 s[78:79], s[78:79], s[72:73]
v_cndmask_b32 v10, -1, v10, s[78:79]
buffer_store_short v9, v10, s[24:27], 0 offen offset:0
s_branch label_tile_stored_27
label_vector_26:
s_mul_i32 s75, s48, s39
v_mul_lo_u32 v11, s40, v1
v_add_u32 v10, s75, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v2, 16, v2
v_lshrrev_b32 v3, 16, v3
v_lshrrev_b32 v4, 16, v4
v_lshrrev_b32 v5, 16, v5
v_pack_b32_f16 v2, v2, v3
v_pack_b32_f16 v3, v4, v5
buffer_store_dwordx2 v[2:3], v10, s[24:27], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s39
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
v_lshrrev_b32 v6, 16, v6
v_lshrrev_b32 v7, 16, v7
v_lshrrev_b32 v8, 16, v8
v_lshrrev_b32 v9, 16, v9
v_pack_b32_f16 v6, v6, v7
v_pack_b32_f16 v7, v8, v9
buffer_store_dwordx2 v[6:7], v10, s[24:27], 0 offen offset:0
label_tile_stored_27:
s_branch label_stored_22
label_workspace_21:
s_mul_i32 s78, s4, s38
s_lshl_b32 s78, s78, 2
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_vector_28
s_mul_i32 s75, s48, s41
s_add_u32 s75, s75, s78
v_mul_lo_u32 v11, s42, v1
v_add_u32 v10, s75, v11
s_and_b64 s[80:81], s[56:57], s[60:61]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v2, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s42
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[62:63]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v3, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_lshl_b32 s76, s42, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[64:65]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v4, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mul_i32 s76, s42, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[56:57], s[66:67]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v5, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[60:61]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v6, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_mov_b32 s76, s42
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[62:63]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v7, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_lshl_b32 s76, s42, 1
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[64:65]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v8, v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
s_mul_i32 s76, s42, 3
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
s_and_b64 s[80:81], s[58:59], s[66:67]
s_and_b64 s[80:81], s[80:81], s[72:73]
v_cndmask_b32 v10, -1, v10, s[80:81]
buffer_store_dword v9, v10, s[28:31], 0 offen offset:0
s_branch label_tile_stored_29
label_vector_28:
s_mul_i32 s75, s48, s41
s_add_u32 s75, s75, s78
v_mul_lo_u32 v11, s42, v1
v_add_u32 v10, s75, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx4 v[2:5], v10, s[28:31], 0 offen offset:0
s_mov_b32 s77, s75
s_mov_b32 s76, s41
s_add_u32 s77, s77, s76
v_add_u32 v10, s77, v11
v_cndmask_b32 v10, -1, v10, s[72:73]
buffer_store_dwordx4 v[6:9], v10, s[28:31], 0 offen offset:0
label_tile_stored_29:
label_stored_22:
label_epilogue_end_11:
s_endpgm
.Lgemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end:
.size gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1, .Lgemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_end - gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 112
  .amdhsa_accum_offset 160
  .amdhsa_next_free_vgpr 160
  .amdhsa_next_free_sgpr 92
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
    macrotile: [4, 4, 4096]
    threads: [256, 1, 1]
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
  - .name: gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1
    .symbol: gemm_BM2_TN4_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1.kd
    .kernarg_segment_size: 112
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .sgpr_count: 98
    .vgpr_count: 160
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
