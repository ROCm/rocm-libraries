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
.protected gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.globl gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.p2align 8
.type gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1,@function
gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1:
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
v_lshrrev_b32 v1, 6, v0
s_nop 0
v_readfirstlane_b32 s42, v1
v_and_b32 v5, 63, v0
v_and_b32 v1, 15, v5
v_and_b32 v2, 48, v5
v_and_b32 v3, 48, v5
v_lshrrev_b32 v3, 2, v3
v_lshlrev_b32 v4, 2, v5
v_lshrrev_b32 v6, 3, v5
s_lshl_b32 s61, s42, 15
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
s_add_u32 s62, s42, -1
s_max_i32 s62, s62, 0
s_lshl_b32 s62, s62, 10
s_add_u32 s62, s62, 65536
v_lshlrev_b32 v11, 4, v5
v_add_u32 v11, s62, v11
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
s_add_u32 s44, s36, 15
s_lshr_b32 s44, s44, 4
s_add_u32 s45, s35, 15
s_lshr_b32 s45, s45, 4
s_mul_i32 s43, s44, s45
s_add_u32 s46, s43, 7
s_lshr_b32 s46, s46, 3
s_add_u32 s48, s43, -1
s_and_b32 s47, 7, s48
s_add_u32 s47, s47, 1
s_cmp_lt_i32 s2, s43
s_cbranch_scc0 label_tile_loop_end_1
label_tile_loop_0:
s_barrier
s_lshr_b32 s50, s2, 3
s_and_b32 s51, 7, s2
s_sub_u32 s52, s51, s47
s_max_i32 s52, s52, 0
s_mul_i32 s49, s51, s46
s_add_u32 s49, s49, s50
s_sub_u32 s49, s49, s52
s_abs_i32 s50, s44
s_abs_i32 s51, s49
s_xor_b32 s52, s49, s44
s_ashr_i32 s52, s52, 31
v_cvt_f32_u32 v5, s50
v_rcp_iflag_f32 v5, v5
s_sub_u32 s53, 0, s50
v_mul_f32 v5, 0x4f7ffffe, v5
v_cvt_u32_f32 v5, v5
s_nop 0
v_readfirstlane_b32 s54, v5
s_mul_i32 s55, s53, s54
s_mul_hi_u32 s55, s54, s55
s_add_u32 s54, s54, s55
s_mul_hi_u32 s53, s51, s54
s_mul_i32 s55, s53, s50
s_sub_u32 s55, s51, s55
s_add_u32 s56, s53, 1
s_sub_u32 s57, s55, s50
s_cmp_ge_u32 s55, s50
s_cselect_b32 s53, s56, s53
s_cselect_b32 s55, s57, s55
s_add_u32 s56, s53, 1
s_cmp_ge_u32 s55, s50
s_cselect_b32 s53, s56, s53
s_xor_b32 s53, s53, s52
s_sub_u32 s48, s53, s52
s_mul_i32 s45, s48, s44
s_sub_u32 s45, s49, s45
s_lshl_b32 s49, s48, 4
s_lshl_b32 s50, s45, 4
s_add_u32 s53, s49, 0
v_add_u32 v21, s53, v6
v_cmp_lt_u32 s[54:55], v21, s35
v_mul_lo_u32 v18, v21, s40
v_add_u32 v18, v18, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v18, v26, v18, s[54:55]
s_add_u32 s53, s49, 8
v_add_u32 v21, s53, v6
v_cmp_lt_u32 s[54:55], v21, s35
v_mul_lo_u32 v19, v21, s40
v_add_u32 v19, v19, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v19, v26, v19, s[54:55]
s_add_u32 s53, s50, 0
v_add_u32 v21, s53, v6
v_cmp_lt_u32 s[54:55], v21, s36
v_mul_lo_u32 v16, v21, s41
v_add_u32 v16, v16, v7
v_mov_b32 v26, -2147483648
v_cndmask_b32 v16, v26, v16, s[54:55]
s_add_u32 s53, s50, 8
v_add_u32 v21, s53, v6
v_cmp_lt_u32 s[54:55], v21, s36
v_mul_lo_u32 v17, v21, s41
v_add_u32 v17, v17, v8
v_mov_b32 v26, -2147483648
v_cndmask_b32 v17, v26, v17, s[54:55]
s_lshr_b32 s53, s49, 5
s_mul_i32 s53, s53, s39
v_add_u32 v20, s53, v4
s_and_b32 s52, 16, s49
s_lshr_b32 s52, s52, 1
s_add_u32 s53, s50, 0
s_lshr_b32 s54, s53, 5
s_mul_i32 s54, s54, s39
v_add_u32 v5, s54, v4
s_and_b32 s51, 16, s53
s_lshr_b32 s51, s51, 1
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
s_cmp_eq_u32 s42, 0
s_cbranch_scc0 label_c_prefetch_skip_2
s_and_b32 s53, s3, 2147483647
s_cmp_eq_u32 s53, 0
s_cbranch_scc1 label_c_prefetch_skip_2
s_cmp_eq_u32 s37, 1
s_cbranch_scc0 label_orientation_b_3
s_add_u32 s59, s49, 0
v_add_u32 v29, s59, v1
v_cmp_lt_u32 s[54:55], v29, s35
v_mul_lo_u32 v26, v29, s33
s_add_u32 s59, s50, 0
v_add_u32 v27, s59, v3
v_mul_lo_u32 v28, v27, s34
s_mov_b32 s56, s34
s_lshl_b32 s57, s34, 1
s_mul_i32 s58, s34, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[66:67], v27, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v22, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s56, v21
v_cmp_lt_i32 s[66:67], v27, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v23, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s57, v21
v_cmp_lt_i32 s[66:67], v27, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v24, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s58, v21
v_cmp_lt_i32 s[66:67], v27, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v25, v21, s[24:27], 0 offen offset:0
s_branch label_epilogue_end_4
label_orientation_b_3:
s_add_u32 s54, s48, 1
s_lshl_b32 s54, s54, 4
s_add_u32 s55, s45, 1
s_lshl_b32 s55, s55, 4
s_cmp_le_u32 s54, s35
s_cselect_b32 s56, 1, 0
s_cmp_le_u32 s55, s36
s_cselect_b32 s57, 1, 0
s_and_b32 s53, s56, s57
s_add_u32 s59, s49, 0
v_add_u32 v29, s59, v1
v_cmp_lt_u32 s[54:55], v29, s35
v_mul_lo_u32 v26, v29, s33
s_add_u32 s59, s50, 0
v_add_u32 v27, s59, v3
v_mul_lo_u32 v28, v27, s34
s_mov_b32 s56, s34
s_lshl_b32 s57, s34, 1
s_mul_i32 s58, s34, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
s_cmp_eq_u32 s53, 1
s_cbranch_scc1 label_c_vector_5
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[66:67], v27, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v22, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s56, v21
v_cmp_lt_i32 s[66:67], v27, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v23, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s57, v21
v_cmp_lt_i32 s[66:67], v27, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v24, v21, s[24:27], 0 offen offset:0
v_add_u32 v21, v26, v28
v_add_u32 v21, s58, v21
v_cmp_lt_i32 s[66:67], v27, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_load_ushort v25, v21, s[24:27], 0 offen offset:0
s_branch label_c_loaded_6
label_c_vector_5:
v_add_u32 v21, v26, v28
buffer_load_dwordx2 v[22:23], v21, s[24:27], 0 offen offset:0
label_c_loaded_6:
label_epilogue_end_4:
label_c_prefetch_skip_2:
s_and_b32 s55, s30, 255
s_lshr_b32 s57, s30, 8
s_add_u32 s58, s57, 1
s_sub_u32 s58, s58, s42
s_lshr_b32 s53, s58, 1
s_lshl_b32 s54, s42, 8
s_lshl_b32 s56, s57, 8
s_and_b32 s59, 1, s57
s_cmp_eq_u32 s42, s59
s_cselect_b32 s60, 1, 0
s_cmp_lg_u32 s60, 0
s_cselect_b32 s55, s55, 0
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
buffer_load_dword v26, v5, s[16:19], s54 offen offset:0
buffer_load_dword v27, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 10112
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
buffer_load_dword v28, v5, s[16:19], s54 offen offset:0
buffer_load_dword v29, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 18304
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
buffer_load_dword v30, v5, s[16:19], s54 offen offset:0
buffer_load_dword v31, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_add_u32 m0, s61, 24576
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 28672
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 25600
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 29696
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 26496
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 30592
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 27520
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 31616
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
buffer_load_dword v32, v5, s[16:19], s54 offen offset:0
buffer_load_dword v33, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_ashr_i32 s57, s53, 2
s_lshl_b32 s57, s57, 2
s_add_u32 s57, s57, -4
s_max_i32 s57, s57, 0
s_waitcnt vmcnt(36)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
s_cmp_gt_i32 s57, 0
s_mov_b32 s58, 0
s_cbranch_scc0 label_kloop_end_8
.p2align 4
label_kloop_7:
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s54 offen offset:0
buffer_load_dword v27, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
buffer_load_dword v28, v5, s[16:19], s54 offen offset:0
buffer_load_dword v29, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
buffer_load_dword v30, v5, s[16:19], s54 offen offset:0
buffer_load_dword v31, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 24576
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 28672
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 25600
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 29696
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v32, s51, v32
v_lshrrev_b32 v33, s52, v33
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 26496
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 30592
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 27520
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 31616
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
buffer_load_dword v32, v5, s[16:19], s54 offen offset:0
buffer_load_dword v33, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_add_u32 s58, s58, 4
s_cmp_lt_i32 s58, s57
s_cbranch_scc1 label_kloop_7
label_kloop_end_8:
s_sub_u32 s58, s53, s57
s_cmp_eq_i32 s58, 1
s_cbranch_scc0 label_arm_end_9
s_waitcnt vmcnt(30)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_9:
s_cmp_eq_i32 s58, 2
s_cbranch_scc0 label_arm_end_10
s_waitcnt vmcnt(30)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_10:
s_cmp_eq_i32 s58, 3
s_cbranch_scc0 label_arm_end_11
s_waitcnt vmcnt(30)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
s_waitcnt vmcnt(10)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_11:
s_cmp_eq_i32 s58, 4
s_cbranch_scc0 label_arm_end_12
s_waitcnt vmcnt(30)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
s_waitcnt vmcnt(20)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
s_waitcnt vmcnt(10)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
s_waitcnt vmcnt(0)
v_lshrrev_b32 v32, s51, v32
v_lshrrev_b32 v33, s52, v33
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_12:
s_cmp_eq_i32 s58, 5
s_cbranch_scc0 label_arm_end_13
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s54 offen offset:0
buffer_load_dword v27, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt vmcnt(30)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
s_waitcnt vmcnt(20)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
s_waitcnt vmcnt(10)
v_lshrrev_b32 v32, s51, v32
v_lshrrev_b32 v33, s52, v33
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
s_waitcnt vmcnt(0)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_13:
s_cmp_eq_i32 s58, 6
s_cbranch_scc0 label_arm_end_14
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s54 offen offset:0
buffer_load_dword v27, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
buffer_load_dword v28, v5, s[16:19], s54 offen offset:0
buffer_load_dword v29, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt vmcnt(30)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
s_waitcnt vmcnt(20)
v_lshrrev_b32 v32, s51, v32
v_lshrrev_b32 v33, s52, v33
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
s_waitcnt vmcnt(10)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
s_waitcnt vmcnt(0)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_14:
s_cmp_eq_i32 s58, 7
s_cbranch_scc0 label_arm_end_15
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 1920
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
buffer_load_dword v26, v5, s[16:19], s54 offen offset:0
buffer_load_dword v27, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 8192
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 12288
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 9216
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 13312
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 10112
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 14208
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 11136
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 15232
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
buffer_load_dword v28, v5, s[16:19], s54 offen offset:0
buffer_load_dword v29, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 16384
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 20480
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:0 lds
s_add_u32 m0, s61, 17408
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:0 lds
s_add_u32 m0, s61, 21504
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:0 lds
s_waitcnt vmcnt(34)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
s_add_u32 m0, s61, 18304
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
buffer_load_dwordx4 v16, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 22400
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s54 offen offset:128 lds
s_add_u32 m0, s61, 19328
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s54 offen offset:128 lds
s_add_u32 m0, s61, 23424
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s54 offen offset:128 lds
s_waitcnt vmcnt(34)
ds_read_b128 v[34:37], v9 offset:24576
ds_read_b128 v[38:41], v10 offset:24576
ds_read_b128 v[42:45], v9 offset:28672
ds_read_b128 v[46:49], v10 offset:28672
buffer_load_dword v30, v5, s[16:19], s54 offen offset:0
buffer_load_dword v31, v20, s[12:15], s54 offen offset:0
s_add_u32 s54, s54, 512
s_waitcnt vmcnt(30)
v_lshrrev_b32 v32, s51, v32
v_lshrrev_b32 v33, s52, v33
ds_read_b128 v[50:53], v9 offset:26624
ds_read_b128 v[54:57], v10 offset:26624
ds_read_b128 v[58:61], v9 offset:30720
ds_read_b128 v[62:65], v10 offset:30720
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v32, v33 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(26)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
s_waitcnt vmcnt(20)
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(16)
ds_read_b128 v[34:37], v9 offset:8192
ds_read_b128 v[38:41], v10 offset:8192
ds_read_b128 v[42:45], v9 offset:12288
ds_read_b128 v[46:49], v10 offset:12288
s_waitcnt vmcnt(10)
v_lshrrev_b32 v28, s51, v28
v_lshrrev_b32 v29, s52, v29
ds_read_b128 v[50:53], v9 offset:10240
ds_read_b128 v[54:57], v10 offset:10240
ds_read_b128 v[58:61], v9 offset:14336
ds_read_b128 v[62:65], v10 offset:14336
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v28, v29 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
s_waitcnt vmcnt(6)
ds_read_b128 v[34:37], v9 offset:16384
ds_read_b128 v[38:41], v10 offset:16384
ds_read_b128 v[42:45], v9 offset:20480
ds_read_b128 v[46:49], v10 offset:20480
s_waitcnt vmcnt(0)
v_lshrrev_b32 v30, s51, v30
v_lshrrev_b32 v31, s52, v31
ds_read_b128 v[50:53], v9 offset:18432
ds_read_b128 v[54:57], v10 offset:18432
ds_read_b128 v[58:61], v9 offset:22528
ds_read_b128 v[62:65], v10 offset:22528
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v30, v31 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_arm_end_15:
s_waitcnt 0
s_cmp_eq_u32 s55, 0
s_cbranch_scc1 label_tail_skip_16
label_tail_begin_17:
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s56 offen offset:0 lds
s_add_u32 m0, s61, 4096
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s56 offen offset:0 lds
s_add_u32 m0, s61, 1024
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s56 offen offset:0 lds
s_add_u32 m0, s61, 5120
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s56 offen offset:0 lds
s_add_u32 m0, s61, 1920
s_nop 0
buffer_load_dwordx4 v16, s[8:11], s56 offen offset:128 lds
s_add_u32 m0, s61, 6016
s_nop 0
buffer_load_dwordx4 v18, s[4:7], s56 offen offset:128 lds
s_add_u32 m0, s61, 2944
s_nop 0
buffer_load_dwordx4 v17, s[8:11], s56 offen offset:128 lds
s_add_u32 m0, s61, 7040
s_nop 0
buffer_load_dwordx4 v19, s[4:7], s56 offen offset:128 lds
buffer_load_dword v26, v5, s[16:19], s56 offen offset:0
buffer_load_dword v27, v20, s[12:15], s56 offen offset:0
s_waitcnt vmcnt(0)
ds_read_b128 v[34:37], v9 offset:0
ds_read_b128 v[38:41], v10 offset:0
ds_read_b128 v[42:45], v9 offset:4096
ds_read_b128 v[46:49], v10 offset:4096
ds_read_b128 v[50:53], v9 offset:2048
ds_read_b128 v[54:57], v10 offset:2048
ds_read_b128 v[58:61], v9 offset:6144
ds_read_b128 v[62:65], v10 offset:6144
s_waitcnt lgkmcnt(0)
s_mov_b32 s57, 0
s_cmp_ge_i32 s55, 64
s_cbranch_scc0 label_tail_edge_18
s_branch label_tail_chunk_end_19
label_tail_edge_18:
s_or_b32 s57, s57, 1
label_tail_chunk_end_19:
s_sub_u32 s58, s55, 64
s_cmp_ge_i32 s58, 64
s_cbranch_scc0 label_tail_edge_20
s_branch label_tail_chunk_end_22
label_tail_edge_20:
s_cmp_gt_i32 s58, 0
s_cbranch_scc0 label_tail_zero_21
s_or_b32 s57, s57, 2
s_branch label_tail_chunk_end_22
label_tail_zero_21:
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
label_tail_chunk_end_22:
s_sub_u32 s58, s55, 128
s_cmp_ge_i32 s58, 64
s_cbranch_scc0 label_tail_edge_23
s_branch label_tail_chunk_end_25
label_tail_edge_23:
s_cmp_gt_i32 s58, 0
s_cbranch_scc0 label_tail_zero_24
s_or_b32 s57, s57, 4
s_branch label_tail_chunk_end_25
label_tail_zero_24:
v_mov_b32 v50, 0
v_mov_b32 v51, 0
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
label_tail_chunk_end_25:
s_sub_u32 s58, s55, 192
s_cmp_gt_i32 s58, 0
s_cbranch_scc0 label_tail_zero_26
s_or_b32 s57, s57, 8
s_branch label_tail_chunk_end_27
label_tail_zero_26:
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
label_tail_chunk_end_27:
s_bitcmp1_b32 s57, 0
s_cbranch_scc0 label_tail_mask_skip_28
v_sub_u32 v21, s55, v2
v_sub_u32 v66, 0, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 4, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v34, v34, v67
v_and_b32 v42, v42, v67
v_sub_u32 v66, 4, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 8, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v35, v35, v67
v_and_b32 v43, v43, v67
v_sub_u32 v66, 8, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 12, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v36, v36, v67
v_and_b32 v44, v44, v67
v_sub_u32 v66, 12, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 16, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v37, v37, v67
v_and_b32 v45, v45, v67
label_tail_mask_skip_28:
s_sub_u32 s58, s55, 64
s_bitcmp1_b32 s57, 1
s_cbranch_scc0 label_tail_mask_skip_29
v_sub_u32 v21, s58, v2
v_sub_u32 v66, 0, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 4, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v38, v38, v67
v_and_b32 v46, v46, v67
v_sub_u32 v66, 4, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 8, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v39, v39, v67
v_and_b32 v47, v47, v67
v_sub_u32 v66, 8, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 12, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v40, v40, v67
v_and_b32 v48, v48, v67
v_sub_u32 v66, 12, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 16, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v41, v41, v67
v_and_b32 v49, v49, v67
label_tail_mask_skip_29:
s_sub_u32 s58, s55, 128
s_bitcmp1_b32 s57, 2
s_cbranch_scc0 label_tail_mask_skip_30
v_sub_u32 v21, s58, v2
v_sub_u32 v66, 0, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 4, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v50, v50, v67
v_and_b32 v58, v58, v67
v_sub_u32 v66, 4, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 8, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v51, v51, v67
v_and_b32 v59, v59, v67
v_sub_u32 v66, 8, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 12, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v52, v52, v67
v_and_b32 v60, v60, v67
v_sub_u32 v66, 12, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 16, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v53, v53, v67
v_and_b32 v61, v61, v67
label_tail_mask_skip_30:
s_sub_u32 s58, s55, 192
s_bitcmp1_b32 s57, 3
s_cbranch_scc0 label_tail_mask_skip_31
v_sub_u32 v21, s58, v2
v_sub_u32 v66, 0, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 4, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v54, v54, v67
v_and_b32 v62, v62, v67
v_sub_u32 v66, 4, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 8, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v55, v55, v67
v_and_b32 v63, v63, v67
v_sub_u32 v66, 8, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 12, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v56, v56, v67
v_and_b32 v64, v64, v67
v_sub_u32 v66, 12, v21
v_ashrrev_i32 v66, 31, v66
v_sub_u32 v67, 16, v21
v_max_i32 v67, 0, v67
v_min_i32 v67, 3, v67
v_lshlrev_b32 v67, 3, v67
v_lshrrev_b32 v67, v67, -1
v_and_b32 v67, v67, v66
v_and_b32 v57, v57, v67
v_and_b32 v65, v65, v67
label_tail_mask_skip_31:
v_sub_u32 v21, s55, v2
v_sub_u32 v21, v21, v2
v_cmp_gt_i32 s[58:59], v21, 0
v_mov_b32 v68, 65535
s_nop 0
v_cndmask_b32 v66, 0, v68, s[58:59]
v_add_u32 v21, -128, v21
v_cmp_gt_i32 s[58:59], v21, 0
v_mov_b32 v68, -65536
s_nop 0
v_cndmask_b32 v21, 0, v68, s[58:59]
v_or_b32 v66, v66, v21
v_and_b32 v67, 2139062143, v66
v_sub_u32 v67, 2139062143, v67
v_and_b32 v26, v26, v66
v_or_b32 v26, v26, v67
v_and_b32 v27, v27, v66
v_or_b32 v27, v27, v67
v_lshrrev_b32 v26, s51, v26
v_lshrrev_b32 v27, s52, v27
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[34:41], v[42:49], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0
v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:57], v[58:65], v[12:15], v26, v27 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:0
label_tail_skip_16:
s_nop 11
s_cmp_eq_u32 s42, 0
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
s_cmp_eq_u32 s37, 1
s_cbranch_scc0 label_orientation_b_34
v_pk_mul_f32 v[12:13], s[28:29], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[28:29], v[14:15] op_sel_hi:[0,1]
s_and_b32 s53, s3, 2147483647
s_cmp_eq_u32 s53, 0
s_cbranch_scc1 label_no_beta_36
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_no_beta_36:
s_add_u32 s53, s49, 0
v_add_u32 v29, s53, v1
v_cmp_lt_u32 s[54:55], v29, s35
v_mul_lo_u32 v26, v29, s31
s_add_u32 s53, s50, 0
v_add_u32 v27, s53, v3
v_mul_lo_u32 v28, v27, s32
s_mov_b32 s56, s32
s_lshl_b32 s57, s32, 1
s_mul_i32 s58, s32, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[66:67], v27, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s56, v21
v_cmp_lt_i32 s[66:67], v27, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s57, v21
v_cmp_lt_i32 s[66:67], v27, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s58, v21
v_cmp_lt_i32 s[66:67], v27, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_epilogue_end_35
label_orientation_b_34:
s_add_u32 s54, s48, 1
s_lshl_b32 s54, s54, 4
s_add_u32 s55, s45, 1
s_lshl_b32 s55, s55, 4
s_cmp_le_u32 s54, s35
s_cselect_b32 s56, 1, 0
s_cmp_le_u32 s55, s36
s_cselect_b32 s57, 1, 0
s_and_b32 s53, s56, s57
v_pk_mul_f32 v[12:13], s[28:29], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], s[28:29], v[14:15] op_sel_hi:[0,1]
s_and_b32 s54, s3, 2147483647
s_cmp_eq_u32 s54, 0
s_cbranch_scc1 label_no_beta_37
s_waitcnt vmcnt(0)
s_cmp_eq_u32 s53, 1
s_cbranch_scc0 label_c_masked_38
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_and_b32 v26, -65536, v22
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v14, s3, v26
v_and_b32 v26, -65536, v23
v_fmac_f32 v15, s3, v26
s_branch label_c_added_39
label_c_masked_38:
v_lshlrev_b32 v26, 16, v22
v_fmac_f32 v12, s3, v26
v_lshlrev_b32 v26, 16, v23
v_fmac_f32 v13, s3, v26
v_lshlrev_b32 v26, 16, v24
v_fmac_f32 v14, s3, v26
v_lshlrev_b32 v26, 16, v25
v_fmac_f32 v15, s3, v26
label_c_added_39:
label_no_beta_37:
s_cmp_eq_u32 s53, 1
s_cbranch_scc1 label_vector_40
s_add_u32 s59, s49, 0
v_add_u32 v29, s59, v1
v_cmp_lt_u32 s[54:55], v29, s35
v_mul_lo_u32 v26, v29, s31
s_add_u32 s59, s50, 0
v_add_u32 v27, s59, v3
v_mul_lo_u32 v28, v27, s32
s_mov_b32 s56, s32
s_lshl_b32 s57, s32, 1
s_mul_i32 s58, s32, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_cvt_pk_bf16_f32 v12, v12, v12
v_add_u32 v21, v26, v28
v_cmp_lt_i32 s[66:67], v27, s62
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v12, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v13, v13, v13
v_add_u32 v21, v26, v28
v_add_u32 v21, s56, v21
v_cmp_lt_i32 s[66:67], v27, s63
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v13, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v14, v14, v14
v_add_u32 v21, v26, v28
v_add_u32 v21, s57, v21
v_cmp_lt_i32 s[66:67], v27, s64
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v14, v21, s[20:23], 0 offen offset:0
v_cvt_pk_bf16_f32 v15, v15, v15
v_add_u32 v21, v26, v28
v_add_u32 v21, s58, v21
v_cmp_lt_i32 s[66:67], v27, s65
s_and_b64 s[66:67], s[66:67], s[54:55]
v_cndmask_b32 v21, -1, v21, s[66:67]
buffer_store_short v15, v21, s[20:23], 0 offen offset:0
s_branch label_tile_stored_41
label_vector_40:
s_add_u32 s59, s49, 0
v_add_u32 v29, s59, v1
v_cmp_lt_u32 s[54:55], v29, s35
v_mul_lo_u32 v26, v29, s31
s_add_u32 s59, s50, 0
v_add_u32 v27, s59, v3
v_mul_lo_u32 v28, v27, s32
s_mov_b32 s56, s32
s_lshl_b32 s57, s32, 1
s_mul_i32 s58, s32, 3
s_sub_u32 s62, s36, 0
s_sub_u32 s63, s36, 1
s_sub_u32 s64, s36, 2
s_sub_u32 s65, s36, 3
v_add_u32 v21, v26, v28
v_cvt_pk_bf16_f32 v12, v12, v13
v_cvt_pk_bf16_f32 v13, v14, v15
buffer_store_dwordx2 v[12:13], v21, s[20:23], 0 offen offset:0
label_tile_stored_41:
label_epilogue_end_35:
label_tile_end_32:
s_add_u32 s2, s2, s38
s_cmp_lt_i32 s2, s43
s_cbranch_scc1 label_tile_loop_0
label_tile_loop_end_1:
s_endpgm
.Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end:
.size gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1, .Lgemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1_end - gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
.rodata
.p2align 6
.amdhsa_kernel gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 96
  .amdhsa_accum_offset 72
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr 68
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
    macrotile: [16, 16, 4096]
    threads: [128, 1, 1]
    grid: [TilesXYBatch, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
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
  - .name: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1
    .symbol: gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PF4_GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1.kd
    .kernarg_segment_size: 96
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 66560
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .sgpr_count: 74
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
