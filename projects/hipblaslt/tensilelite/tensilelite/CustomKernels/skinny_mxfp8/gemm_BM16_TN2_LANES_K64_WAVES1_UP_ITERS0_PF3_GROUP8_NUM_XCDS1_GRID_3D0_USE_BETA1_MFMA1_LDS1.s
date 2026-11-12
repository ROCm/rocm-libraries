// Generated out of tree, then hand-edited: the m/n/k kernarg names in the HSA metadata became m_size/n_size/k_size and
// the custom.config block was appended (AddCustomConfig). No instruction was edited; an earlier s_nop 11 to 13 edit
// was reverted so the kernels match the generator output.
// Constraints (custom.config keys): AssertSizeEqual (batch 1), AssertSizeGreaterThan and AssertSizeLessThan (m, n and K
// in [1, 2^20]), AssertSummationElementMultiple (K % 32 == 0), AssertStrideAEqual, AssertStrideBEqual,
// AssertStrideCEqual, AssertStrideDEqual (unit strides), AssertStrideAMultiple, AssertStrideBMultiple,
// AssertTensorBytesLessThan (tensors < 2^31 bytes), AssertMXScalesPreSwizzled. No bias, activation or scale-alpha.
// Rule: AssertAnySizeLessThan, min(m, n) <= 4.
// Invariant: SRD num_records is built from the tensor extent in this file and must never be enlarged (for example to
// 0x7fffffff): loads past K in the prologue and tail are safe only because soffset is range-checked against it.
// The 0x80000000 offset for out-of-range A and B rows must stay outside the A and B buffers. Formulas: README.md.
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.protected gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1:
s_load_dword s38, s[0:1], 92
s_load_dword s28, s[0:1], 84
s_load_dword s30, s[0:1], 76
s_load_dword s47, s[0:1], 72
s_load_dword s46, s[0:1], 68
s_load_dword s45, s[0:1], 64
s_load_dword s44, s[0:1], 60
s_load_dword s43, s[0:1], 56
s_load_dwordx2 s[54:55], s[0:1], 32
s_load_dwordx2 s[52:53], s[0:1], 24
s_load_dwordx2 s[50:51], s[0:1], 16
s_load_dwordx2 s[48:49], s[0:1], 8
s_load_dwordx2 s[20:21], s[0:1], 0
s_load_dword s3, s[0:1], 88
s_load_dword s60, s[0:1], 80
s_load_dwordx2 s[24:25], s[0:1], 48
s_mov_b32 s42, 0
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_mul_i32 s61, s42, 36864
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
s_add_u32 s62, s61, 0
v_add_u32 v9, s62, v14
v_xor_b32 v10, 64, v9
s_waitcnt lgkmcnt(0)
s_add_u32 s62, s30, 3
s_lshr_b32 s62, s62, 2
s_lshl_b32 s62, s62, 2
s_add_u32 s56, s46, -1
s_mul_i32 s56, s56, s44
s_add_u32 s56, s56, s62
s_add_u32 s57, s47, -1
s_mul_i32 s57, s57, s45
s_add_u32 s57, s57, s62
s_add_u32 s39, s30, 255
s_lshr_b32 s39, s39, 8
s_lshl_b32 s39, s39, 8
s_add_u32 s58, s46, 31
s_lshr_b32 s58, s58, 5
s_mul_i32 s58, s58, s39
s_add_u32 s59, s47, 31
s_lshr_b32 s59, s59, 5
s_mul_i32 s59, s59, s39
s_add_u32 s22, s47, -1
s_mul_i32 s22, s22, s43
s_lshl_b32 s62, s46, 1
s_add_u32 s22, s22, s62
s_add_u32 s26, s47, -1
s_mul_i32 s26, s26, s60
s_lshl_b32 s62, s46, 1
s_add_u32 s26, s26, s62
s_cmp_le_u32 s46, s47
s_cselect_b64 s[4:5], s[48:49], s[50:51]
s_cselect_b32 s6, s56, s57
s_cselect_b64 s[8:9], s[50:51], s[48:49]
s_cselect_b32 s10, s57, s56
s_cselect_b64 s[12:13], s[52:53], s[54:55]
s_cselect_b32 s14, s58, s59
s_cselect_b64 s[16:17], s[54:55], s[52:53]
s_cselect_b32 s18, s59, s58
s_cselect_b32 s40, s44, s45
s_cselect_b32 s41, s45, s44
s_cselect_b32 s31, 2, s43
s_cselect_b32 s32, s43, 2
s_cselect_b32 s33, 2, s60
s_cselect_b32 s34, s60, 2
s_cselect_b32 s37, 1, 0
s_min_u32 s35, s46, s47
s_max_u32 s36, s46, s47
s_mov_b32 s7, 131072
s_mov_b32 s11, 131072
s_mov_b32 s15, 131072
s_mov_b32 s19, 131072
s_mov_b32 s23, 131072
s_mov_b32 s27, 131072
s_add_u32 s44, s36, 31
s_lshr_b32 s44, s44, 5
s_add_u32 s45, s35, 15
s_lshr_b32 s45, s45, 4
s_mul_i32 s43, s44, s45
s_lshl_b32 s46, s44, 3
s_cmp_lt_i32 s2, s43
s_cbranch_scc0 label_tile_loop_end_1
label_tile_loop_0:
s_abs_i32 s53, s46
s_abs_i32 s54, s2
s_xor_b32 s55, s2, s46
s_ashr_i32 s55, s55, 31
v_cvt_f32_u32 v5, s53
v_rcp_iflag_f32 v5, v5
s_sub_u32 s56, 0, s53
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s57, v5
s_mul_i32 s58, s56, s57
s_mul_hi_u32 s58, s57, s58
s_add_u32 s57, s57, s58
s_mul_hi_u32 s56, s54, s57
s_mul_i32 s58, s56, s53
s_sub_u32 s58, s54, s58
s_add_u32 s59, s56, 1
s_sub_u32 s60, s58, s53
s_cmp_ge_u32 s58, s53
s_cselect_b32 s56, s59, s56
s_cselect_b32 s58, s60, s58
s_add_u32 s59, s56, 1
s_cmp_ge_u32 s58, s53
s_cselect_b32 s56, s59, s56
s_xor_b32 s56, s56, s55
s_sub_u32 s49, s56, s55
s_mul_i32 s50, s49, s46
s_sub_u32 s50, s2, s50
s_lshl_b32 s51, s49, 3
s_sub_u32 s52, s45, s51
s_min_i32 s52, s52, 8
s_abs_i32 s53, s52
s_abs_i32 s54, s50
s_xor_b32 s55, s50, s52
s_ashr_i32 s55, s55, 31
v_cvt_f32_u32 v5, s53
v_rcp_iflag_f32 v5, v5
s_sub_u32 s56, 0, s53
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s57, v5
s_mul_i32 s58, s56, s57
s_mul_hi_u32 s58, s57, s58
s_add_u32 s57, s57, s58
s_mul_hi_u32 s56, s54, s57
s_mul_i32 s58, s56, s53
s_sub_u32 s58, s54, s58
s_add_u32 s59, s56, 1
s_sub_u32 s60, s58, s53
s_cmp_ge_u32 s58, s53
s_cselect_b32 s56, s59, s56
s_cselect_b32 s58, s60, s58
s_add_u32 s59, s56, 1
s_cmp_ge_u32 s58, s53
s_cselect_b32 s56, s59, s56
s_xor_b32 s56, s56, s55
s_sub_u32 s47, s56, s55
s_mul_i32 s48, s47, s52
s_sub_u32 s48, s50, s48
s_add_u32 s48, s48, s51
s_lshl_b32 s49, s48, 4
s_lshl_b32 s50, s47, 5
s_add_u32 s52, s49, 0
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s35
v_mul_lo_u32 v24, v34, s40
v_add_u32 v24, v24, v7
v_mov_b32 v35, -2147483648
v_cndmask_b32 v24, v35, v24, s[54:55]
s_add_u32 s52, s49, 8
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s35
v_mul_lo_u32 v25, v34, s40
v_add_u32 v25, v25, v8
v_mov_b32 v35, -2147483648
v_cndmask_b32 v25, v35, v25, s[54:55]
s_add_u32 s52, s50, 0
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s36
v_mul_lo_u32 v20, v34, s41
v_add_u32 v20, v20, v7
v_mov_b32 v35, -2147483648
v_cndmask_b32 v20, v35, v20, s[54:55]
s_add_u32 s52, s50, 8
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s36
v_mul_lo_u32 v21, v34, s41
v_add_u32 v21, v21, v8
v_mov_b32 v35, -2147483648
v_cndmask_b32 v21, v35, v21, s[54:55]
s_add_u32 s52, s50, 16
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s36
v_mul_lo_u32 v22, v34, s41
v_add_u32 v22, v22, v7
v_mov_b32 v35, -2147483648
v_cndmask_b32 v22, v35, v22, s[54:55]
s_add_u32 s52, s50, 24
v_add_u32 v34, s52, v6
v_cmp_lt_u32 s[54:55], v34, s36
v_mul_lo_u32 v23, v34, s41
v_add_u32 v23, v23, v8
v_mov_b32 v35, -2147483648
v_cndmask_b32 v23, v35, v23, s[54:55]
s_lshr_b32 s52, s49, 5
s_mul_i32 s52, s52, s39
v_add_u32 v11, s52, v4
s_and_b32 s51, 16, s49
s_lshr_b32 s51, s51, 1
s_add_u32 s52, s50, 0
s_lshr_b32 s53, s52, 5
s_mul_i32 s53, s53, s39
v_add_u32 v5, s53, v4
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
v_mov_b32 v16, 0
v_mov_b32 v17, 0
v_mov_b32 v18, 0
v_mov_b32 v19, 0
s_and_b32 s52, s3, 2147483647
s_cmp_eq_u32 s52, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s37, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s53, s49, 0
v_add_u32 v40, s53, v1
v_cmp_lt_u32 s[54:55], v40, s35
v_mul_lo_u32 v35, v40, s33
s_add_u32 s53, s50, 0
v_add_u32 v36, s53, v3
v_mul_lo_u32 v38, v36, s34
s_add_u32 s53, s50, 16
v_add_u32 v37, s53, v3
v_mul_lo_u32 v39, v37, s34
s_mov_b32 s56, s34
s_lshl_b32 s57, s34, 1
s_mul_i32 s58, s34, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_add_u32 v34, v35, v38
v_cmp_lt_i32 s[66:67], v36, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v26, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v36, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v27, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v36, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v28, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v36, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v29, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_cmp_lt_i32 s[66:67], v37, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v30, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v37, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v31, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v37, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v32, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v37, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v33, v34, s[24:27], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s53, s48, 1
s_lshl_b32 s53, s53, 4
s_add_u32 s54, s47, 1
s_lshl_b32 s54, s54, 5
s_cmp_le_u32 s53, s35
s_cselect_b32 s55, 1, 0
s_cmp_le_u32 s54, s36
s_cselect_b32 s56, 1, 0
s_and_b32 s52, s55, s56
s_add_u32 s53, s49, 0
v_add_u32 v40, s53, v1
v_cmp_lt_u32 s[54:55], v40, s35
v_mul_lo_u32 v35, v40, s33
s_add_u32 s53, s50, 0
v_add_u32 v36, s53, v3
v_mul_lo_u32 v38, v36, s34
s_add_u32 s53, s50, 16
v_add_u32 v37, s53, v3
v_mul_lo_u32 v39, v37, s34
s_mov_b32 s56, s34
s_lshl_b32 s57, s34, 1
s_mul_i32 s58, s34, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
s_cmp_eq_u32 s52, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v34, v35, v38
v_cmp_lt_i32 s[66:67], v36, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v26, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v36, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v27, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v36, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v28, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v38
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v36, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v29, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_cmp_lt_i32 s[66:67], v37, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v30, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v37, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v31, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v37, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v32, v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v37, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_load_ushort v33, v34, s[24:27], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v34, v35, v38
buffer_load_dwordx2 v[26:27], v34, s[24:27], 0 offen offset:0
v_add_u32 v34, v35, v39
buffer_load_dwordx2 v[28:29], v34, s[24:27], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_and_b32 s54, s30, 255
s_lshr_b32 s56, s30, 8
s_add_u32 s57, s56, 0
s_sub_u32 s57, s57, s42
s_mov_b32 s52, s57
s_lshl_b32 s53, s42, 8
s_lshl_b32 s55, s56, 8
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 1920
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
buffer_load_dword v34, v5, s[16:19], s53 offen offset:0
buffer_load_dword v35, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 14208
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 18304
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
buffer_load_dword v36, v5, s[16:19], s53 offen offset:0
buffer_load_dword v37, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_add_u32 m0, s61, 24576
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 28672
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 32768
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 25600
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 29696
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 33792
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 26496
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 30592
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 34688
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 27520
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 31616
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 35712
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
buffer_load_dword v38, v5, s[16:19], s53 offen offset:0
buffer_load_dword v39, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_mul_hi_i32 s56, s52, 1431655766
s_lshr_b32 s57, s56, 31
s_add_u32 s56, s56, s57
s_mul_i32 s56, s56, 3
s_add_u32 s56, s56, -3
s_max_i32 s56, s56, 0
s_waitcnt vmcnt(36)
ds_read_b128 v[40:43], v9 offset:0
ds_read_b128 v[44:47], v10 offset:0
ds_read_b128 v[48:51], v9 offset:4096
ds_read_b128 v[52:55], v10 offset:4096
ds_read_b128 v[56:59], v9 offset:8192
ds_read_b128 v[60:63], v10 offset:8192
s_cmp_gt_i32 s56, 0
s_mov_b32 s57, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
buffer_load_dword v34, v5, s[16:19], s53 offen offset:0
buffer_load_dword v35, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 14208
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:24576
ds_read_b128 v[44:47], v10 offset:24576
ds_read_b128 v[48:51], v9 offset:28672
ds_read_b128 v[52:55], v10 offset:28672
ds_read_b128 v[56:59], v9 offset:32768
ds_read_b128 v[60:63], v10 offset:32768
buffer_load_dword v36, v5, s[16:19], s53 offen offset:0
buffer_load_dword v37, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 24576
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 28672
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 32768
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 25600
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 29696
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 33792
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v39, s51, v39
ds_read_b128 v[64:67], v9 offset:26624
ds_read_b128 v[68:71], v10 offset:26624
ds_read_b128 v[72:75], v9 offset:30720
ds_read_b128 v[76:79], v10 offset:30720
ds_read_b128 v[80:83], v9 offset:34816
ds_read_b128 v[84:87], v10 offset:34816
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 30592
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 34688
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 27520
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 31616
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 35712
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:0
ds_read_b128 v[44:47], v10 offset:0
ds_read_b128 v[48:51], v9 offset:4096
ds_read_b128 v[52:55], v10 offset:4096
ds_read_b128 v[56:59], v9 offset:8192
ds_read_b128 v[60:63], v10 offset:8192
buffer_load_dword v38, v5, s[16:19], s53 offen offset:0
buffer_load_dword v39, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_add_u32 s57, s57, 3
s_cmp_lt_i32 s57, s56
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s57, s52, s56
s_cmp_eq_i32 s57, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(28)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s57, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(28)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
s_waitcnt vmcnt(14)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s57, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(28)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
s_waitcnt vmcnt(14)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[40:43], v9 offset:24576
ds_read_b128 v[44:47], v10 offset:24576
ds_read_b128 v[48:51], v9 offset:28672
ds_read_b128 v[52:55], v10 offset:28672
ds_read_b128 v[56:59], v9 offset:32768
ds_read_b128 v[60:63], v10 offset:32768
s_waitcnt vmcnt(0)
v_lshrrev_b32 v39, s51, v39
ds_read_b128 v[64:67], v9 offset:26624
ds_read_b128 v[68:71], v10 offset:26624
ds_read_b128 v[72:75], v9 offset:30720
ds_read_b128 v[76:79], v10 offset:30720
ds_read_b128 v[80:83], v9 offset:34816
ds_read_b128 v[84:87], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s57, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
buffer_load_dword v34, v5, s[16:19], s53 offen offset:0
buffer_load_dword v35, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_waitcnt vmcnt(28)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[40:43], v9 offset:24576
ds_read_b128 v[44:47], v10 offset:24576
ds_read_b128 v[48:51], v9 offset:28672
ds_read_b128 v[52:55], v10 offset:28672
ds_read_b128 v[56:59], v9 offset:32768
ds_read_b128 v[60:63], v10 offset:32768
s_waitcnt vmcnt(14)
v_lshrrev_b32 v39, s51, v39
ds_read_b128 v[64:67], v9 offset:26624
ds_read_b128 v[68:71], v10 offset:26624
ds_read_b128 v[72:75], v9 offset:30720
ds_read_b128 v[76:79], v10 offset:30720
ds_read_b128 v[80:83], v9 offset:34816
ds_read_b128 v[84:87], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[40:43], v9 offset:0
ds_read_b128 v[44:47], v10 offset:0
ds_read_b128 v[48:51], v9 offset:4096
ds_read_b128 v[52:55], v10 offset:4096
ds_read_b128 v[56:59], v9 offset:8192
ds_read_b128 v[60:63], v10 offset:8192
s_waitcnt vmcnt(0)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s57, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 6016
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
buffer_load_dword v34, v5, s[16:19], s53 offen offset:0
buffer_load_dword v35, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 14208
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v20, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v22, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s53 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s53 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s53 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[40:43], v9 offset:24576
ds_read_b128 v[44:47], v10 offset:24576
ds_read_b128 v[48:51], v9 offset:28672
ds_read_b128 v[52:55], v10 offset:28672
ds_read_b128 v[56:59], v9 offset:32768
ds_read_b128 v[60:63], v10 offset:32768
buffer_load_dword v36, v5, s[16:19], s53 offen offset:0
buffer_load_dword v37, v11, s[12:15], s53 offen offset:0
s_add_u32 s53, s53, 256
s_waitcnt vmcnt(28)
v_lshrrev_b32 v39, s51, v39
ds_read_b128 v[64:67], v9 offset:26624
ds_read_b128 v[68:71], v10 offset:26624
ds_read_b128 v[72:75], v9 offset:30720
ds_read_b128 v[76:79], v10 offset:30720
ds_read_b128 v[80:83], v9 offset:34816
ds_read_b128 v[84:87], v10 offset:34816
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v38, v39 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(22)
ds_read_b128 v[40:43], v9 offset:0
ds_read_b128 v[44:47], v10 offset:0
ds_read_b128 v[48:51], v9 offset:4096
ds_read_b128 v[52:55], v10 offset:4096
ds_read_b128 v[56:59], v9 offset:8192
ds_read_b128 v[60:63], v10 offset:8192
s_waitcnt vmcnt(14)
v_lshrrev_b32 v35, s51, v35
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(8)
ds_read_b128 v[40:43], v9 offset:12288
ds_read_b128 v[44:47], v10 offset:12288
ds_read_b128 v[48:51], v9 offset:16384
ds_read_b128 v[52:55], v10 offset:16384
ds_read_b128 v[56:59], v9 offset:20480
ds_read_b128 v[60:63], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v37, s51, v37
ds_read_b128 v[64:67], v9 offset:14336
ds_read_b128 v[68:71], v10 offset:14336
ds_read_b128 v[72:75], v9 offset:18432
ds_read_b128 v[76:79], v10 offset:18432
ds_read_b128 v[80:83], v9 offset:22528
ds_read_b128 v[84:87], v10 offset:22528
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v36, v37 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v36, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_waitcnt 0
s_cmp_eq_u32 s54, 0
s_cbranch_scc1 label_tail_skip_14
label_tail_begin_15:
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s55 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s55 offen offset:0 lds
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s55 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s55 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s55 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s55 offen offset:0 lds
s_add_u32 m0, s61, 1920
s_nop 0
buffer_load_dwordx4 v20, s[8:11], s55 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v22, s[8:11], s55 offen offset:128 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v24, s[4:7], s55 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v21, s[8:11], s55 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v23, s[8:11], s55 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v25, s[4:7], s55 offen offset:128 lds
buffer_load_dword v34, v5, s[16:19], s55 offen offset:0
buffer_load_dword v35, v11, s[12:15], s55 offen offset:0
s_waitcnt vmcnt(0)
ds_read_b128 v[40:43], v9 offset:0
ds_read_b128 v[44:47], v10 offset:0
ds_read_b128 v[48:51], v9 offset:4096
ds_read_b128 v[52:55], v10 offset:4096
ds_read_b128 v[56:59], v9 offset:8192
ds_read_b128 v[60:63], v10 offset:8192
ds_read_b128 v[64:67], v9 offset:2048
ds_read_b128 v[68:71], v10 offset:2048
ds_read_b128 v[72:75], v9 offset:6144
ds_read_b128 v[76:79], v10 offset:6144
ds_read_b128 v[80:83], v9 offset:10240
ds_read_b128 v[84:87], v10 offset:10240
s_waitcnt lgkmcnt(0)
s_mov_b32 s56, 0
s_cmp_ge_i32 s54, 64
s_cbranch_scc0 label_tail_edge_16
s_branch label_tail_chunk_end_17
label_tail_edge_16:
s_or_b32 s56, s56, 1
label_tail_chunk_end_17:
s_sub_u32 s57, s54, 64
s_cmp_ge_i32 s57, 64
s_cbranch_scc0 label_tail_edge_18
s_branch label_tail_chunk_end_20
label_tail_edge_18:
s_cmp_gt_i32 s57, 0
s_cbranch_scc0 label_tail_zero_19
s_or_b32 s56, s56, 2
s_branch label_tail_chunk_end_20
label_tail_zero_19:
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
label_tail_chunk_end_20:
s_sub_u32 s57, s54, 128
s_cmp_ge_i32 s57, 64
s_cbranch_scc0 label_tail_edge_21
s_branch label_tail_chunk_end_23
label_tail_edge_21:
s_cmp_gt_i32 s57, 0
s_cbranch_scc0 label_tail_zero_22
s_or_b32 s56, s56, 4
s_branch label_tail_chunk_end_23
label_tail_zero_22:
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v66, 0
v_mov_b32 v67, 0
v_mov_b32 v72, 0
v_mov_b32 v73, 0
v_mov_b32 v74, 0
v_mov_b32 v75, 0
v_mov_b32 v80, 0
v_mov_b32 v81, 0
v_mov_b32 v82, 0
v_mov_b32 v83, 0
label_tail_chunk_end_23:
s_sub_u32 s57, s54, 192
s_cmp_gt_i32 s57, 0
s_cbranch_scc0 label_tail_zero_24
s_or_b32 s56, s56, 8
s_branch label_tail_chunk_end_25
label_tail_zero_24:
v_mov_b32 v68, 0
v_mov_b32 v69, 0
v_mov_b32 v70, 0
v_mov_b32 v71, 0
v_mov_b32 v76, 0
v_mov_b32 v77, 0
v_mov_b32 v78, 0
v_mov_b32 v79, 0
v_mov_b32 v84, 0
v_mov_b32 v85, 0
v_mov_b32 v86, 0
v_mov_b32 v87, 0
label_tail_chunk_end_25:
s_bitcmp1_b32 s56, 0
s_cbranch_scc0 label_tail_mask_skip_26
v_sub_u32 v88, s54, v2
v_sub_u32 v89, 0, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 4, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v40, v40, v90
v_and_b32 v48, v48, v90
v_and_b32 v56, v56, v90
v_sub_u32 v89, 4, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 8, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v41, v41, v90
v_and_b32 v49, v49, v90
v_and_b32 v57, v57, v90
v_sub_u32 v89, 8, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 12, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v42, v42, v90
v_and_b32 v50, v50, v90
v_and_b32 v58, v58, v90
v_sub_u32 v89, 12, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 16, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v43, v43, v90
v_and_b32 v51, v51, v90
v_and_b32 v59, v59, v90
label_tail_mask_skip_26:
s_sub_u32 s57, s54, 64
s_bitcmp1_b32 s56, 1
s_cbranch_scc0 label_tail_mask_skip_27
v_sub_u32 v88, s57, v2
v_sub_u32 v89, 0, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 4, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v44, v44, v90
v_and_b32 v52, v52, v90
v_and_b32 v60, v60, v90
v_sub_u32 v89, 4, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 8, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v45, v45, v90
v_and_b32 v53, v53, v90
v_and_b32 v61, v61, v90
v_sub_u32 v89, 8, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 12, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v46, v46, v90
v_and_b32 v54, v54, v90
v_and_b32 v62, v62, v90
v_sub_u32 v89, 12, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 16, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v47, v47, v90
v_and_b32 v55, v55, v90
v_and_b32 v63, v63, v90
label_tail_mask_skip_27:
s_sub_u32 s57, s54, 128
s_bitcmp1_b32 s56, 2
s_cbranch_scc0 label_tail_mask_skip_28
v_sub_u32 v88, s57, v2
v_sub_u32 v89, 0, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 4, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v64, v64, v90
v_and_b32 v72, v72, v90
v_and_b32 v80, v80, v90
v_sub_u32 v89, 4, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 8, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v65, v65, v90
v_and_b32 v73, v73, v90
v_and_b32 v81, v81, v90
v_sub_u32 v89, 8, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 12, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v66, v66, v90
v_and_b32 v74, v74, v90
v_and_b32 v82, v82, v90
v_sub_u32 v89, 12, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 16, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v67, v67, v90
v_and_b32 v75, v75, v90
v_and_b32 v83, v83, v90
label_tail_mask_skip_28:
s_sub_u32 s57, s54, 192
s_bitcmp1_b32 s56, 3
s_cbranch_scc0 label_tail_mask_skip_29
v_sub_u32 v88, s57, v2
v_sub_u32 v89, 0, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 4, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v68, v68, v90
v_and_b32 v76, v76, v90
v_and_b32 v84, v84, v90
v_sub_u32 v89, 4, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 8, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v69, v69, v90
v_and_b32 v77, v77, v90
v_and_b32 v85, v85, v90
v_sub_u32 v89, 8, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 12, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v70, v70, v90
v_and_b32 v78, v78, v90
v_and_b32 v86, v86, v90
v_sub_u32 v89, 12, v88
v_ashrrev_i32 v89, 31, v89
v_sub_u32 v90, 16, v88
v_max_i32 v90, 0, v90
v_min_i32 v90, 3, v90
v_lshlrev_b32 v90, 3, v90
v_lshrrev_b32 v90, v90, -1
v_and_b32 v90, v90, v89
v_and_b32 v71, v71, v90
v_and_b32 v79, v79, v90
v_and_b32 v87, v87, v90
label_tail_mask_skip_29:
v_sub_u32 v88, s54, v2
v_sub_u32 v88, v88, v2
v_cmp_gt_i32 s[56:57], v88, 0
v_mov_b32 v91, 65535
s_nop 0
v_cndmask_b32 v89, 0, v91, s[56:57]
v_add_u32 v88, -128, v88
v_cmp_gt_i32 s[56:57], v88, 0
v_mov_b32 v91, -65536
s_nop 0
v_cndmask_b32 v88, 0, v91, s[56:57]
v_or_b32 v89, v89, v88
v_and_b32 v90, 2139062143, v89
v_sub_u32 v90, 2139062143, v90
v_and_b32 v34, v34, v89
v_or_b32 v34, v34, v90
v_and_b32 v35, v35, v89
v_or_b32 v35, v35, v90
v_lshrrev_b32 v35, s51, v35
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:47], v[56:63], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[48:55], v[56:63], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[64:71], v[80:87], v[12:15], v34, v35 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[72:79], v[80:87], v[16:19], v34, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_tail_skip_14:
s_nop 11
s_cmp_eq_u32 s37, 1
s_cbranch_scc0 label_orientation_b_31
v_pk_mul_f32 v[12:13], s[28:29], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[28:29], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[28:29], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[28:29], v[18:19] op_sel_hi:[0,1]
s_and_b32 s52, s3, 2147483647
s_cmp_eq_u32 s52, 0
s_cbranch_scc1 label_no_beta_33
s_waitcnt vmcnt(0)
v_lshlrev_b32 v35, 16, v26
v_fmac_f32 v12, s3, v35
v_lshlrev_b32 v35, 16, v27
v_fmac_f32 v13, s3, v35
v_lshlrev_b32 v35, 16, v28
v_fmac_f32 v14, s3, v35
v_lshlrev_b32 v35, 16, v29
v_fmac_f32 v15, s3, v35
v_lshlrev_b32 v35, 16, v30
v_fmac_f32 v16, s3, v35
v_lshlrev_b32 v35, 16, v31
v_fmac_f32 v17, s3, v35
v_lshlrev_b32 v35, 16, v32
v_fmac_f32 v18, s3, v35
v_lshlrev_b32 v35, 16, v33
v_fmac_f32 v19, s3, v35
label_no_beta_33:
s_add_u32 s62, s49, 0
v_add_u32 v40, s62, v1
v_cmp_lt_u32 s[52:53], v40, s35
v_mul_lo_u32 v35, v40, s31
s_add_u32 s62, s50, 0
v_add_u32 v36, s62, v3
v_mul_lo_u32 v38, v36, s32
s_add_u32 s62, s50, 16
v_add_u32 v37, s62, v3
v_mul_lo_u32 v39, v37, s32
s_mov_b32 s54, s32
s_lshl_b32 s55, s32, 1
s_mul_i32 s56, s32, 3
s_sub_u32 s57, s36, 0
s_sub_u32 s58, s36, 1
s_sub_u32 s59, s36, 2
s_sub_u32 s60, s36, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v34, v35, v38
v_cmp_lt_i32 s[62:63], v36, s57
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v12, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v34, v35, v38
v_add_u32 v34, s54, v34
v_cmp_lt_i32 s[62:63], v36, s58
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v13, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v34, v35, v38
v_add_u32 v34, s55, v34
v_cmp_lt_i32 s[62:63], v36, s59
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v14, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v34, v35, v38
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[62:63], v36, s60
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v15, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v16, v16, v16
v_add_u32 v34, v35, v39
v_cmp_lt_i32 s[62:63], v37, s57
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v16, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v17, v17, v17
v_add_u32 v34, v35, v39
v_add_u32 v34, s54, v34
v_cmp_lt_i32 s[62:63], v37, s58
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v17, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v18, v18, v18
v_add_u32 v34, v35, v39
v_add_u32 v34, s55, v34
v_cmp_lt_i32 s[62:63], v37, s59
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v18, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v19, v19, v19
v_add_u32 v34, v35, v39
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[62:63], v37, s60
s_and_b64 s[62:63], s[62:63], s[52:53]
v_cndmask_b32 v34, -1, v34, s[62:63]
buffer_store_short v19, v34, s[20:23], 0 offen offset:0
s_branch label_epilogue_end_32
label_orientation_b_31:
s_add_u32 s53, s48, 1
s_lshl_b32 s53, s53, 4
s_add_u32 s54, s47, 1
s_lshl_b32 s54, s54, 5
s_cmp_le_u32 s53, s35
s_cselect_b32 s55, 1, 0
s_cmp_le_u32 s54, s36
s_cselect_b32 s56, 1, 0
s_and_b32 s52, s55, s56
v_pk_mul_f32 v[12:13], s[28:29], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[28:29], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], s[28:29], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], s[28:29], v[18:19] op_sel_hi:[0,1]
s_and_b32 s53, s3, 2147483647
s_cmp_eq_u32 s53, 0
s_cbranch_scc1 label_no_beta_34
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s52, 1
s_cbranch_scc0 label_c_masked_35
v_lshlrev_b32 v35, 16, v26
v_fmac_f32 v12, s3, v35
v_and_b32 v35, -65536, v26
v_fmac_f32 v13, s3, v35
v_lshlrev_b32 v35, 16, v27
v_fmac_f32 v14, s3, v35
v_and_b32 v35, -65536, v27
v_fmac_f32 v15, s3, v35
v_lshlrev_b32 v35, 16, v28
v_fmac_f32 v16, s3, v35
v_and_b32 v35, -65536, v28
v_fmac_f32 v17, s3, v35
v_lshlrev_b32 v35, 16, v29
v_fmac_f32 v18, s3, v35
v_and_b32 v35, -65536, v29
v_fmac_f32 v19, s3, v35
s_branch label_c_added_36
label_c_masked_35:
v_lshlrev_b32 v35, 16, v26
v_fmac_f32 v12, s3, v35
v_lshlrev_b32 v35, 16, v27
v_fmac_f32 v13, s3, v35
v_lshlrev_b32 v35, 16, v28
v_fmac_f32 v14, s3, v35
v_lshlrev_b32 v35, 16, v29
v_fmac_f32 v15, s3, v35
v_lshlrev_b32 v35, 16, v30
v_fmac_f32 v16, s3, v35
v_lshlrev_b32 v35, 16, v31
v_fmac_f32 v17, s3, v35
v_lshlrev_b32 v35, 16, v32
v_fmac_f32 v18, s3, v35
v_lshlrev_b32 v35, 16, v33
v_fmac_f32 v19, s3, v35
label_c_added_36:
label_no_beta_34:
s_cmp_eq_u32 s52, 1
s_cbranch_scc1 label_vector_37
s_add_u32 s53, s49, 0
v_add_u32 v40, s53, v1
v_cmp_lt_u32 s[54:55], v40, s35
v_mul_lo_u32 v35, v40, s31
s_add_u32 s53, s50, 0
v_add_u32 v36, s53, v3
v_mul_lo_u32 v38, v36, s32
s_add_u32 s53, s50, 16
v_add_u32 v37, s53, v3
v_mul_lo_u32 v39, v37, s32
s_mov_b32 s56, s32
s_lshl_b32 s57, s32, 1
s_mul_i32 s58, s32, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v34, v35, v38
v_cmp_lt_i32 s[66:67], v36, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v12, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v34, v35, v38
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v36, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v13, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v34, v35, v38
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v36, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v14, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v34, v35, v38
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v36, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v15, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v16, v16, v16
v_add_u32 v34, v35, v39
v_cmp_lt_i32 s[66:67], v37, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v16, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v17, v17, v17
v_add_u32 v34, v35, v39
v_add_u32 v34, s56, v34
v_cmp_lt_i32 s[66:67], v37, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v17, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v18, v18, v18
v_add_u32 v34, v35, v39
v_add_u32 v34, s57, v34
v_cmp_lt_i32 s[66:67], v37, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v18, v34, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v19, v19, v19
v_add_u32 v34, v35, v39
v_add_u32 v34, s58, v34
v_cmp_lt_i32 s[66:67], v37, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v34, -1, v34, s[66:67]
buffer_store_short v19, v34, s[20:23], 0 offen offset:0
s_branch label_tile_stored_38
label_vector_37:
s_add_u32 s53, s49, 0
v_add_u32 v40, s53, v1
v_cmp_lt_u32 s[54:55], v40, s35
v_mul_lo_u32 v35, v40, s31
s_add_u32 s53, s50, 0
v_add_u32 v36, s53, v3
v_mul_lo_u32 v38, v36, s32
s_add_u32 s53, s50, 16
v_add_u32 v37, s53, v3
v_mul_lo_u32 v39, v37, s32
s_mov_b32 s56, s32
s_lshl_b32 s57, s32, 1
s_mul_i32 s58, s32, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_add_u32 v34, v35, v38
v_cvt_pk_bf16_f32 v12, v12, v13
v_cvt_pk_bf16_f32 v13, v14, v15
buffer_store_dwordx2 v[12:13], v34, s[20:23], 0 offen offset:0
v_add_u32 v34, v35, v39
v_cvt_pk_bf16_f32 v16, v16, v17
v_cvt_pk_bf16_f32 v17, v18, v19
buffer_store_dwordx2 v[16:17], v34, s[20:23], 0 offen offset:0
label_tile_stored_38:
label_epilogue_end_32:
label_tile_end_30:
s_add_u32 s2, s2, s38
s_cmp_lt_i32 s2, s43
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 96
  .amdhsa_accum_offset 96
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr 68
  .amdhsa_group_segment_fixed_size 36864
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
    Repository: "Generated out of tree"
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
            { type: int32, semantic: StrideC0Bytes },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int32, semantic: NumWorkGroups } ]
    macrotile: [32, 32, 4096]
    threads: [64, 1, 1]
    grid: [TilesXYBatch, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  AssertSummationElementMultiple: 32
  AssertTensorBytesLessThan: 2147483648
  AssertAnySizeLessThan: { 0: 5, 1: 5 }
  AssertMXScalesPreSwizzled: true
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 0, 1: 0, 3: 0 }
  AssertSizeLessThan: { 0: 1048577, 1: 1048577, 3: 1048577 }
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
  - .name: gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PF3_GROUP8_NUM_XCDS1_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 96
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 36864
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .sgpr_count: 74
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
      - .name: m_size
        .size: 4
        .offset: 68
        .value_kind: by_value
      - .name: n_size
        .size: 4
        .offset: 72
        .value_kind: by_value
      - .name: k_size
        .size: 4
        .offset: 76
        .value_kind: by_value
      - .name: ldC
        .size: 4
        .offset: 80
        .value_kind: by_value
      - .name: alpha
        .size: 4
        .offset: 84
        .value_kind: by_value
      - .name: beta
        .size: 4
        .offset: 88
        .value_kind: by_value
      - .name: numWorkGroupsX
        .size: 4
        .offset: 92
        .value_kind: by_value
...
.end_amdgpu_metadata
