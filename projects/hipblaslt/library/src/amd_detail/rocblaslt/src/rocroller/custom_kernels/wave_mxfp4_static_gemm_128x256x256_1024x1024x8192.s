.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text

.protected wave_mxfp4_static_gemm_128x256x256_1024x1024x8192
.globl wave_mxfp4_static_gemm_128x256x256_1024x1024x8192
.p2align 8
.type wave_mxfp4_static_gemm_128x256x256_1024x1024x8192,@function
wave_mxfp4_static_gemm_128x256x256_1024x1024x8192:
  ; SRD setup prologue
  s_load_dwordx2 s[8:9], s[0:1], 0
  s_load_dwordx2 s[12:13], s[0:1], 8
  s_load_dwordx2 s[16:17], s[0:1], 16
  s_load_dwordx2 s[20:21], s[0:1], 24
  s_load_dwordx2 s[24:25], s[0:1], 32
  s_load_dwordx2 s[28:29], s[0:1], 40
  s_load_dwordx2 s[30:31], s[0:1], 48
  s_load_dwordx2 s[32:33], s[0:1], 56
  s_load_dwordx2 s[34:35], s[0:1], 64
  s_load_dwordx2 s[36:37], s[0:1], 72
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s10, 0x7FFFFFFE
  s_mov_b32 s11, 0x20000
  s_mov_b32 s14, 0x7FFFFFFE
  s_mov_b32 s15, 0x20000
  s_mov_b32 s18, 0x20000000
  s_mov_b32 s19, 0x20000
  s_mov_b32 s22, 0x20000000
  s_mov_b32 s23, 0x20000
  s_mov_b32 s26, 0x7FFFFFFC
  s_mov_b32 s27, 0x20000
  v_mov_b32 v5, s28
  v_mov_b32 v6, s30
  v_mov_b32 v7, s32
  v_mov_b32 v8, s34
  v_mov_b32 v9, s36
  ; End SRD setup
  v_bfe_u32 v1, v0, 0, 10
  v_bfe_u32 v2, v0, 10, 10
  v_lshlrev_b32 v3, 3, v2
  v_lshlrev_b32 v4, 8, s3
  v_lshl_or_b32 v5, v2, 3, v4
  v_lshrrev_b32 v6, 3, v1
  v_or_b32 v10, v5, v6
  v_lshl_or_b32 v5, v2, 3, v6
  v_lshlrev_b32 v11, 7, 0
  v_sub_u32 v12, v10, v11
  v_lshrrev_b32 v13, 2, s2
  v_lshlrev_b32 v14, 7, v13
  v_add_u32 v16, v12, v14
  v_and_b32 v12, v6, 7
  v_and_b32 v6, v1, 7
  v_xor_b32 v17, v6, v12
  v_lshlrev_b32 v12, 4, v17
  v_sub_u32 v17, v3, v11
  s_nop 0
  v_readfirstlane_b32 s0, v17
  v_lshlrev_b32 v11, 12, v16
  v_add_u32 v18, v11, v12
  s_mov_b32 s40, s8
  s_and_b32 s41, s9, 0xffff
  s_or_b32 s41, s41, 0x40400000
  s_mov_b32 s42, 0xFFFFFFFF
  s_mov_b32 s43, 0x27000
  s_lshl_b32 s1, s0, 7
  s_add_u32 s4, s1, 18432
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v18, s[40:43], 0 offen lds
  v_add_u32 v18, 128, v12
  v_add_u32 v19, v11, v18
  s_add_u32 s5, s1, 2048
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v19, s[40:43], 0 offen lds
  v_add_u32 v19, v16, 32
  v_add_u32 v16, v17, 32
  s_nop 0
  v_readfirstlane_b32 s0, v16
  v_lshlrev_b32 v16, 12, v19
  v_add_u32 v17, v16, v12
  s_lshl_b32 s1, s0, 7
  s_add_u32 s6, s1, 18432
  s_mov_b32 m0, s6
  buffer_load_dwordx4 v17, s[40:43], 0 offen lds
  v_add_u32 v17, v16, v18
  s_add_u32 s7, s1, 2048
  s_mov_b32 m0, s7
  buffer_load_dwordx4 v17, s[40:43], 0 offen lds
  v_add_u32 v17, v5, 64
  v_and_b32 v19, 4294967168, v17
  v_sub_u32 v17, v10, v19
  v_add_u32 v19, v17, v14
  v_add_u32 v17, v19, 64
  v_add_u32 v19, v2, 8
  v_lshrrev_b32 v20, 4, v19
  v_lshlrev_b32 v19, 7, v20
  v_sub_u32 v20, v3, v19
  v_add_u32 v19, v20, 64
  s_nop 0
  v_readfirstlane_b32 s0, v19
  v_lshlrev_b32 v19, 12, v17
  v_add_u32 v17, v19, v12
  s_lshl_b32 s1, s0, 7
  s_add_u32 s8, s1, 18432
  s_mov_b32 m0, s8
  buffer_load_dwordx4 v17, s[40:43], 0 offen lds
  v_add_u32 v17, v19, v18
  s_add_u32 s9, s1, 2048
  s_mov_b32 m0, s9
  buffer_load_dwordx4 v17, s[40:43], 0 offen lds
  v_add_u32 v17, 96, v5
  v_and_b32 v5, 4294967168, v17
  v_sub_u32 v17, v10, v5
  v_add_u32 v5, v17, v14
  v_add_u32 v10, 96, v5
  v_add_u32 v5, v2, 12
  v_lshrrev_b32 v17, 4, v5
  v_lshlrev_b32 v5, 7, v17
  v_sub_u32 v17, v3, v5
  v_add_u32 v3, 96, v17
  s_nop 0
  v_readfirstlane_b32 s0, v3
  v_lshlrev_b32 v3, 12, v10
  v_add_u32 v5, v3, v12
  s_lshl_b32 s1, s0, 7
  s_add_u32 s10, s1, 18432
  s_mov_b32 m0, s10
  buffer_load_dwordx4 v5, s[40:43], 0 offen lds
  v_add_u32 v5, v3, v18
  s_add_u32 s11, s1, 2048
  s_mov_b32 m0, s11
  buffer_load_dwordx4 v5, s[40:43], 0 offen lds
  v_min_i32 v5, v2, 3
  v_lshl_add_u32 v10, v5, 5, v4
  v_add_u32 v17, v10, v14
  v_lshlrev_b32 v18, 2, v1
  v_lshlrev_b32 v20, 8, 0
  v_sub_u32 v21, v18, v20
  v_lshlrev_b32 v20, 8, v5
  s_nop 0
  v_readfirstlane_b32 s0, v20
  v_lshl_add_u32 v5, v17, 8, v21
  s_mov_b32 s48, s12
  s_and_b32 s49, s13, 0xffff
  s_or_b32 s49, s49, 0x40400000
  s_mov_b32 s50, 0xFFFFFFFF
  s_mov_b32 s51, 0x27000
  s_add_u32 s12, s0, 1024
  s_mov_b32 m0, s12
  buffer_load_dword v5, s[48:51], 0 offen lds
  v_or_b32 v5, v17, 1
  v_lshl_add_u32 v17, v5, 8, v21
  s_mov_b32 m0, s0
  buffer_load_dword v17, s[48:51], 0 offen lds
  v_lshlrev_b32 v5, 6, v2
  v_lshl_or_b32 v17, s2, 8, v5
  v_and_b32 v5, v1, 15
  v_and_b32 v20, v1, 63
  v_lshrrev_b32 v22, 4, v20
  v_and_b32 v23, 4294967280, v20
  v_add_u32 v20, v5, v23
  v_or_b32 v24, v17, 0
  v_lshlrev_b32 v25, 10, v13
  v_sub_u32 v13, 0, v25
  v_sub_u32 v26, v24, v25
  v_lshrrev_b32 v24, 4, v1
  v_lshlrev_b32 v27, 8, v24
  v_sub_u32 v28, 0, v27
  v_lshl_add_u32 v29, v1, 4, v28
  v_lshlrev_b32 v28, 8, v22
  v_add_u32 v30, v29, v28
  v_lshlrev_b32 v29, 12, 0
  v_sub_u32 v31, v30, v29
  v_mul_lo_u32 v29, v26, v7
  v_add_u32 v32, v29, v31
  buffer_load_dwordx4 v[36:39], v32, s[16:19], 0 offen
  buffer_load_dwordx4 v[40:43], v32, s[16:19], 0 offen offset:1024
  v_add_u32 v29, v26, 16
  v_mul_lo_u32 v32, v29, v7
  v_add_u32 v29, v32, v31
  buffer_load_dwordx4 v[32:35], v29, s[16:19], 0 offen
  buffer_load_dwordx4 v[44:47], v29, s[16:19], 0 offen offset:1024
  v_add_u32 v29, v26, 32
  v_mul_lo_u32 v48, v29, v7
  v_add_u32 v29, v48, v31
  buffer_load_dwordx4 v[48:51], v29, s[16:19], 0 offen
  buffer_load_dwordx4 v[52:55], v29, s[16:19], 0 offen offset:1024
  v_add_u32 v29, v26, 48
  v_mul_lo_u32 v26, v29, v7
  v_add_u32 v29, v26, v31
  buffer_load_dwordx4 v[56:59], v29, s[16:19], 0 offen
  buffer_load_dwordx4 v[60:63], v29, s[16:19], 0 offen offset:1024
  ; unhandled: rocdl.sched.barrier
  v_lshrrev_b32 v26, 6, v20
  v_or_b32 v29, v17, v26
  v_sub_u32 v31, v29, v25
  v_lshlrev_b32 v29, 6, v24
  v_sub_u32 v24, v18, v29
  v_lshl_add_u32 v29, v22, 6, v24
  v_lshlrev_b32 v24, 8, v26
  v_sub_u32 v64, v29, v24
  v_mul_lo_u32 v24, v31, v8
  v_add_u32 v29, v24, v64
  buffer_load_dword v24, v29, s[20:23], 0 offen
  v_add_u32 v29, v31, 32
  v_mul_lo_u32 v65, v29, v8
  v_add_u32 v29, v65, v64
  buffer_load_dword v65, v29, s[20:23], 0 offen
  ; unhandled: rocdl.sched.barrier
  s_waitcnt vmcnt(0)
  s_barrier
  ; unhandled: rocdl.sched.barrier
  v_lshl_or_b32 v29, 0, 7, v5
  v_xor_b32 v5, v22, v6
  v_lshlrev_b32 v66, 4, v5
  v_lshlrev_b32 v5, 7, v29
  v_add_u32 v29, v5, v66
  ds_read_b128 v[68:71], v29 offset:18432
  v_add_u32 v66, v22, 4
  v_xor_b32 v67, v66, v6
  v_lshlrev_b32 v6, 4, v67
  v_add_u32 v66, v5, v6
  ds_read_b128 v[72:75], v66 offset:18432
  ds_read_b128 v[76:79], v29 offset:20480
  ds_read_b128 v[80:83], v66 offset:20480
  ds_read_b128 v[84:87], v29 offset:22528
  ds_read_b128 v[88:91], v66 offset:22528
  ds_read_b128 v[92:95], v29 offset:24576
  ds_read_b128 v[96:99], v66 offset:24576
  ds_read_b32 v5, v18 offset:1024
  ds_read_b32 v6, v18 offset:1280
  s_mov_b32 s1, 0
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a0, v15
  v_accvgpr_write_b32 a1, v15
  v_accvgpr_write_b32 a2, v15
  v_accvgpr_write_b32 a3, v15
  v_and_b32 v67, 4294967280, v1
  v_lshl_add_u32 v100, s1, 7, v12
  v_add_u32 v12, 256, v100
  v_add_u32 v101, v11, v12
  v_add_u32 v102, v16, v12
  v_add_u32 v103, v19, v12
  v_add_u32 v104, v3, v12
  v_add_u32 v12, v10, s1
  v_add_u32 v10, v12, v14
  v_add_u32 v12, v10, 2
  v_lshl_add_u32 v105, v12, 8, v21
  v_add_u32 v12, 384, v100
  v_add_u32 v100, v11, v12
  v_add_u32 v11, v16, v12
  v_add_u32 v16, v19, v12
  v_add_u32 v19, v3, v12
  v_add_u32 v3, v10, 3
  v_lshl_add_u32 v10, v3, 8, v21
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a4, v15
  v_accvgpr_write_b32 a5, v15
  v_accvgpr_write_b32 a6, v15
  v_accvgpr_write_b32 a7, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a8, v15
  v_accvgpr_write_b32 a9, v15
  v_accvgpr_write_b32 a10, v15
  v_accvgpr_write_b32 a11, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a12, v15
  v_accvgpr_write_b32 a13, v15
  v_accvgpr_write_b32 a14, v15
  v_accvgpr_write_b32 a15, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a16, v15
  v_accvgpr_write_b32 a17, v15
  v_accvgpr_write_b32 a18, v15
  v_accvgpr_write_b32 a19, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a20, v15
  v_accvgpr_write_b32 a21, v15
  v_accvgpr_write_b32 a22, v15
  v_accvgpr_write_b32 a23, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a24, v15
  v_accvgpr_write_b32 a25, v15
  v_accvgpr_write_b32 a26, v15
  v_accvgpr_write_b32 a27, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a28, v15
  v_accvgpr_write_b32 a29, v15
  v_accvgpr_write_b32 a30, v15
  v_accvgpr_write_b32 a31, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a32, v15
  v_accvgpr_write_b32 a33, v15
  v_accvgpr_write_b32 a34, v15
  v_accvgpr_write_b32 a35, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a36, v15
  v_accvgpr_write_b32 a37, v15
  v_accvgpr_write_b32 a38, v15
  v_accvgpr_write_b32 a39, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a40, v15
  v_accvgpr_write_b32 a41, v15
  v_accvgpr_write_b32 a42, v15
  v_accvgpr_write_b32 a43, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a44, v15
  v_accvgpr_write_b32 a45, v15
  v_accvgpr_write_b32 a46, v15
  v_accvgpr_write_b32 a47, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a48, v15
  v_accvgpr_write_b32 a49, v15
  v_accvgpr_write_b32 a50, v15
  v_accvgpr_write_b32 a51, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a52, v15
  v_accvgpr_write_b32 a53, v15
  v_accvgpr_write_b32 a54, v15
  v_accvgpr_write_b32 a55, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a56, v15
  v_accvgpr_write_b32 a57, v15
  v_accvgpr_write_b32 a58, v15
  v_accvgpr_write_b32 a59, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a60, v15
  v_accvgpr_write_b32 a61, v15
  v_accvgpr_write_b32 a62, v15
  v_accvgpr_write_b32 a63, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a64, v15
  v_accvgpr_write_b32 a65, v15
  v_accvgpr_write_b32 a66, v15
  v_accvgpr_write_b32 a67, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a68, v15
  v_accvgpr_write_b32 a69, v15
  v_accvgpr_write_b32 a70, v15
  v_accvgpr_write_b32 a71, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a72, v15
  v_accvgpr_write_b32 a73, v15
  v_accvgpr_write_b32 a74, v15
  v_accvgpr_write_b32 a75, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a76, v15
  v_accvgpr_write_b32 a77, v15
  v_accvgpr_write_b32 a78, v15
  v_accvgpr_write_b32 a79, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a80, v15
  v_accvgpr_write_b32 a81, v15
  v_accvgpr_write_b32 a82, v15
  v_accvgpr_write_b32 a83, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a84, v15
  v_accvgpr_write_b32 a85, v15
  v_accvgpr_write_b32 a86, v15
  v_accvgpr_write_b32 a87, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a88, v15
  v_accvgpr_write_b32 a89, v15
  v_accvgpr_write_b32 a90, v15
  v_accvgpr_write_b32 a91, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a92, v15
  v_accvgpr_write_b32 a93, v15
  v_accvgpr_write_b32 a94, v15
  v_accvgpr_write_b32 a95, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a96, v15
  v_accvgpr_write_b32 a97, v15
  v_accvgpr_write_b32 a98, v15
  v_accvgpr_write_b32 a99, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a100, v15
  v_accvgpr_write_b32 a101, v15
  v_accvgpr_write_b32 a102, v15
  v_accvgpr_write_b32 a103, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a104, v15
  v_accvgpr_write_b32 a105, v15
  v_accvgpr_write_b32 a106, v15
  v_accvgpr_write_b32 a107, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a108, v15
  v_accvgpr_write_b32 a109, v15
  v_accvgpr_write_b32 a110, v15
  v_accvgpr_write_b32 a111, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a112, v15
  v_accvgpr_write_b32 a113, v15
  v_accvgpr_write_b32 a114, v15
  v_accvgpr_write_b32 a115, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a116, v15
  v_accvgpr_write_b32 a117, v15
  v_accvgpr_write_b32 a118, v15
  v_accvgpr_write_b32 a119, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a120, v15
  v_accvgpr_write_b32 a121, v15
  v_accvgpr_write_b32 a122, v15
  v_accvgpr_write_b32 a123, v15
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a124, v15
  v_accvgpr_write_b32 a125, v15
  v_accvgpr_write_b32 a126, v15
  v_accvgpr_write_b32 a127, v15
  s_mov_b32 s13, 0
  s_mov_b32 s14, 0
  s_waitcnt lgkmcnt(0)
L_loop_0:
  s_waitcnt vmcnt(5) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:71], v[36:39], a[0:3], v5, v24 cbsz:4 blgp:4
  v_lshl_or_b32 v3, s1, 7, v1
  v_sub_u32 v12, v3, v67
  v_add_u32 v3, v12, v23
  v_add_u32 v12, 128, v3
  v_lshrrev_b32 v21, 8, v12
  v_add_u32 v12, v17, v21
  v_sub_u32 v106, v12, v25
  v_lshlrev_b32 v12, 11, s1
  v_lshl_or_b32 v107, v1, 4, v12
  v_sub_u32 v12, v107, v27
  v_add_u32 v107, v12, v28
  v_lshlrev_b32 v12, 12, v21
  v_sub_u32 v21, v107, v12
  v_mul_lo_u32 v12, v106, v7
  v_add_u32 v108, v21, v12
  buffer_load_dwordx4 v[112:115], v108, s[16:19], 0 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v5, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:71], v[32:35], a[4:7], v5, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v5, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[108:111], v29 offset:26624
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[48:51], a[8:11], v5, v65 cbsz:4 blgp:4
  v_add_u32 v12, 192, v3
  v_lshrrev_b32 v116, 8, v12
  v_add_u32 v12, v17, v116
  v_sub_u32 v117, v12, v25
  v_lshlrev_b32 v12, 12, v116
  v_sub_u32 v116, v107, v12
  v_mul_lo_u32 v12, v117, v7
  v_add_u32 v118, v116, v12
  buffer_load_dwordx4 v[120:123], v118, s[16:19], 0 offen offset:3072
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[52:55], a[8:11], v5, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[56:59], a[12:15], v5, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[60:63], a[12:15], v5, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[124:127], v66 offset:26624
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[36:39], a[16:19], v5, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v12, v106, 16
  v_mul_lo_u32 v118, v12, v7
  v_add_u32 v12, v21, v118
  buffer_load_dwordx4 v[128:131], v12, s[16:19], 0 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[40:43], a[16:19], v5, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[32:35], a[20:23], v5, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[44:47], a[20:23], v5, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[132:135], v29 offset:28672
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[48:51], a[24:27], v5, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v12, v117, 16
  v_mul_lo_u32 v118, v12, v7
  v_add_u32 v12, v116, v118
  buffer_load_dwordx4 v[136:139], v12, s[16:19], 0 offen offset:3072
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[52:55], a[24:27], v5, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v5, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[60:63], a[28:31], v5, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[140:143], v66 offset:28672
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[36:39], a[32:35], v6, v24 cbsz:4 blgp:4
  v_add_u32 v12, v106, 32
  v_mul_lo_u32 v118, v12, v7
  v_add_u32 v12, v21, v118
  buffer_load_dwordx4 v[144:147], v12, s[16:19], 0 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[40:43], a[32:35], v6, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[32:35], a[36:39], v6, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[44:47], a[36:39], v6, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[148:151], v29 offset:30720
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[48:51], a[40:43], v6, v65 cbsz:4 blgp:4
  v_add_u32 v12, v117, 32
  v_mul_lo_u32 v118, v12, v7
  v_add_u32 v12, v116, v118
  buffer_load_dwordx4 v[152:155], v12, s[16:19], 0 offen offset:3072
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[52:55], a[40:43], v6, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[56:59], a[44:47], v6, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[60:63], a[44:47], v6, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[156:159], v66 offset:30720
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[36:39], a[48:51], v6, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v12, v106, 48
  v_mul_lo_u32 v106, v12, v7
  v_add_u32 v12, v21, v106
  buffer_load_dwordx4 v[160:163], v12, s[16:19], 0 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[40:43], a[48:51], v6, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[32:35], a[52:55], v6, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[44:47], a[52:55], v6, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[164:167], v29 offset:32768
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v6, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v12, v117, 48
  v_mul_lo_u32 v21, v12, v7
  v_add_u32 v12, v116, v21
  buffer_load_dwordx4 v[116:119], v12, s[16:19], 0 offen offset:3072
  v_add_u32 v12, v17, s1
  v_add_u32 v21, v12, v26
  v_sub_u32 v12, v21, v25
  v_add_u32 v21, v12, 1
  v_mul_lo_u32 v106, v21, v8
  v_add_u32 v21, v106, v64
  buffer_load_dword v106, v21, s[20:23], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[52:55], a[56:59], v6, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[56:59], a[60:63], v6, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[60:63], a[60:63], v6, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[168:171], v66 offset:32768
  ds_read_b32 v21, v18 offset:1536
  ds_read_b32 v172, v18 offset:1792
  v_add_u32 v173, v12, 33
  v_mul_lo_u32 v174, v173, v8
  v_add_u32 v173, v174, v64
  buffer_load_dword v174, v173, s[20:23], 0 offen
  s_waitcnt vmcnt(10) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:111], v[36:39], a[64:67], v21, v24 cbsz:4 blgp:4
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v101, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[124:127], v[40:43], a[64:67], v21, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[32:35], a[68:71], v21, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[44:47], a[68:71], v21, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[176:179], v29 offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[48:51], a[72:75], v21, v65 cbsz:4 blgp:4
  s_mov_b32 m0, s6
  buffer_load_dwordx4 v102, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[52:55], a[72:75], v21, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[56:59], a[76:79], v21, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[60:63], a[76:79], v21, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[108:111], v66 offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[132:135], v[36:39], a[80:83], v21, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s8
  buffer_load_dwordx4 v103, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[40:43], a[80:83], v21, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[32:35], a[84:87], v21, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[44:47], a[84:87], v21, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[124:127], v29 offset:4096
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[48:51], a[88:91], v21, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s10
  buffer_load_dwordx4 v104, s[40:43], s13 offen lds
  s_mov_b32 m0, s12
  buffer_load_dword v105, s[48:51], s14 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[52:55], a[88:91], v21, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[56:59], a[92:95], v21, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[60:63], a[92:95], v21, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[132:135], v66 offset:4096
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:151], v[36:39], a[96:99], v172, v24 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[156:159], v[40:43], a[96:99], v172, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[32:35], a[100:103], v172, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[156:159], v[44:47], a[100:103], v172, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[140:143], v29 offset:6144
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[48:51], a[104:107], v172, v65 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[52:55], a[104:107], v172, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[56:59], a[108:111], v172, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:159], v[60:63], a[108:111], v172, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[148:151], v66 offset:6144
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[36:39], a[112:115], v172, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[168:171], v[40:43], a[112:115], v172, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[32:35], a[116:119], v172, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:171], v[44:47], a[116:119], v172, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[156:159], v29 offset:8192
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[48:51], a[120:123], v172, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[168:171], v[52:55], a[120:123], v172, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[56:59], a[124:127], v172, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[168:171], v[60:63], a[124:127], v172, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[164:167], v66 offset:8192
  ds_read_b32 v21, v18
  ds_read_b32 v168, v18 offset:256
  s_waitcnt vmcnt(5) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[112:115], a[0:3], v21, v106 cbsz:4 blgp:4
  v_lshrrev_b32 v169, 8, v3
  v_add_u32 v170, v17, v169
  v_sub_u32 v171, v170, v25
  v_add_u32 v170, v171, 1
  v_lshlrev_b32 v172, 12, v169
  v_sub_u32 v169, v107, v172
  v_mul_lo_u32 v172, v170, v7
  v_add_u32 v170, v172, v169
  buffer_load_dwordx4 v[36:39], v170, s[16:19], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[120:123], a[0:3], v21, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[128:131], a[4:7], v21, v106 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[136:139], a[4:7], v21, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[180:183], v29 offset:10240
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[176:179], v[144:147], a[8:11], v21, v174 cbsz:4 blgp:4
  v_add_u32 v170, 320, v3
  v_lshrrev_b32 v172, 8, v170
  v_add_u32 v170, v17, v172
  v_sub_u32 v172, v170, v25
  v_add_u32 v170, v3, 64
  v_lshrrev_b32 v3, 8, v170
  v_lshlrev_b32 v170, 12, v3
  v_sub_u32 v3, v107, v170
  v_mul_lo_u32 v107, v172, v7
  v_add_u32 v170, v3, v107
  buffer_load_dwordx4 v[40:43], v170, s[16:19], 0 offen offset:1024
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[152:155], a[8:11], v21, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[176:179], v[160:163], a[12:15], v21, v174 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[116:119], a[12:15], v21, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[108:111], v66 offset:10240
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[112:115], a[16:19], v21, v106 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v107, v171, 17
  v_mul_lo_u32 v170, v107, v7
  v_add_u32 v107, v170, v169
  buffer_load_dwordx4 v[32:35], v107, s[16:19], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[120:123], a[16:19], v21, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[128:131], a[20:23], v21, v106 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[136:139], a[20:23], v21, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[176:179], v29 offset:12288
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[144:147], a[24:27], v21, v174 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v107, v172, 16
  v_mul_lo_u32 v170, v107, v7
  v_add_u32 v107, v3, v170
  buffer_load_dwordx4 v[44:47], v107, s[16:19], 0 offen offset:1024
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[152:155], a[24:27], v21, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[160:163], a[28:31], v21, v174 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[116:119], a[28:31], v21, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[124:127], v66 offset:12288
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[112:115], a[32:35], v168, v106 cbsz:4 blgp:4
  v_add_u32 v21, v171, 33
  v_mul_lo_u32 v107, v21, v7
  v_add_u32 v21, v107, v169
  buffer_load_dwordx4 v[48:51], v21, s[16:19], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:151], v[120:123], a[32:35], v168, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[128:131], a[36:39], v168, v106 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:151], v[136:139], a[36:39], v168, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[132:135], v29 offset:14336
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[144:147], a[40:43], v168, v174 cbsz:4 blgp:4
  v_add_u32 v21, v172, 32
  v_mul_lo_u32 v107, v21, v7
  v_add_u32 v21, v3, v107
  buffer_load_dwordx4 v[52:55], v21, s[16:19], 0 offen offset:1024
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[152:155], a[40:43], v168, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[160:163], a[44:47], v168, v174 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[116:119], a[44:47], v168, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[140:143], v66 offset:14336
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[112:115], a[48:51], v168, v106 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v21, v171, 49
  v_mul_lo_u32 v107, v21, v7
  v_add_u32 v21, v107, v169
  buffer_load_dwordx4 v[56:59], v21, s[16:19], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[164:167], v[120:123], a[48:51], v168, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[128:131], a[52:55], v168, v106 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[136:139], a[52:55], v168, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[148:151], v29 offset:16384
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[144:147], a[56:59], v168, v174 op_sel:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v21, v172, 48
  v_mul_lo_u32 v107, v21, v7
  v_add_u32 v21, v3, v107
  buffer_load_dwordx4 v[60:63], v21, s[16:19], 0 offen offset:1024
  v_add_u32 v3, v12, 2
  v_mul_lo_u32 v21, v3, v8
  v_add_u32 v3, v21, v64
  buffer_load_dword v24, v3, s[20:23], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[152:155], a[56:59], v168, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[160:163], a[60:63], v168, v174 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[116:119], a[60:63], v168, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[156:159], v66 offset:16384
  ds_read_b32 v3, v18 offset:512
  ds_read_b32 v21, v18 offset:768
  v_add_u32 v107, v12, 34
  v_mul_lo_u32 v12, v107, v8
  v_add_u32 v107, v12, v64
  buffer_load_dword v65, v107, s[20:23], 0 offen
  s_waitcnt vmcnt(10) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[180:183], v[112:115], a[64:67], v3, v106 cbsz:4 blgp:4
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v100, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[108:111], v[120:123], a[64:67], v3, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[180:183], v[128:131], a[68:71], v3, v106 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[136:139], a[68:71], v3, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[68:71], v29 offset:18432
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[180:183], v[144:147], a[72:75], v3, v174 cbsz:4 blgp:4
  s_mov_b32 m0, s7
  buffer_load_dwordx4 v11, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[152:155], a[72:75], v3, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[180:183], v[160:163], a[76:79], v3, v174 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[116:119], a[76:79], v3, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[72:75], v66 offset:18432
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[112:115], a[80:83], v3, v106 op_sel:[1,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s9
  buffer_load_dwordx4 v16, s[40:43], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[120:123], a[80:83], v3, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[176:179], v[128:131], a[84:87], v3, v106 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[136:139], a[84:87], v3, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[76:79], v29 offset:20480
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[176:179], v[144:147], a[88:91], v3, v174 op_sel:[1,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s11
  buffer_load_dwordx4 v19, s[40:43], s13 offen lds
  s_mov_b32 m0, s0
  buffer_load_dword v10, s[48:51], s14 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[152:155], a[88:91], v3, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[176:179], v[160:163], a[92:95], v3, v174 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[116:119], a[92:95], v3, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[80:83], v66 offset:20480
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[112:115], a[96:99], v21, v106 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[120:123], a[96:99], v21, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[132:135], v[128:131], a[100:103], v21, v106 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[140:143], v[136:139], a[100:103], v21, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[84:87], v29 offset:22528
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[144:147], a[104:107], v21, v174 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[152:155], a[104:107], v21, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[132:135], v[160:163], a[108:111], v21, v174 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[116:119], a[108:111], v21, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[88:91], v66 offset:22528
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:151], v[112:115], a[112:115], v21, v106 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[156:159], v[120:123], a[112:115], v21, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[128:131], a[116:119], v21, v106 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[156:159], v[136:139], a[116:119], v21, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[92:95], v29 offset:24576
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[144:147], a[120:123], v21, v174 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:159], v[152:155], a[120:123], v21, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[160:163], a[124:127], v21, v174 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[116:119], a[124:127], v21, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[96:99], v66 offset:24576
  ds_read_b32 v5, v18 offset:1024
  ds_read_b32 v6, v18 offset:1280
  s_waitcnt vmcnt(5) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  s_add_u32 s1, s1, 2
  s_add_u32 s13, s13, 256
  s_add_u32 s14, s14, 512
  s_cmp_lt_u32 s1, 30
  s_cbranch_scc1 L_loop_0
  s_waitcnt vmcnt(0) & lgkmcnt(0)
  s_barrier
  ds_read_b32 v3, v18 offset:1536
  ds_read_b32 v10, v18 offset:1792
  ds_read_b128 v[100:103], v29 offset:26624
  ds_read_b128 v[104:107], v66 offset:26624
  ds_read_b128 v[108:111], v29 offset:28672
  ds_read_b128 v[112:115], v66 offset:28672
  ds_read_b128 v[116:119], v29 offset:30720
  ds_read_b128 v[120:123], v66 offset:30720
  ds_read_b128 v[124:127], v29 offset:32768
  ds_read_b128 v[128:131], v66 offset:32768
  v_add_u32 v11, 3968, v20
  v_lshrrev_b32 v12, 8, v11
  v_or_b32 v11, v17, v12
  v_sub_u32 v12, v11, v25
  v_add_u32 v11, 128, v20
  v_lshrrev_b32 v16, 8, v11
  v_lshlrev_b32 v11, 12, v16
  v_sub_u32 v16, v30, v11
  v_mul_lo_u32 v11, v12, v7
  v_add_u32 v19, v16, v11
  buffer_load_dwordx4 v[132:135], v19, s[16:19], 0 offen offset:2048
  v_add_u32 v11, 4032, v20
  v_lshrrev_b32 v19, 8, v11
  v_or_b32 v11, v17, v19
  v_sub_u32 v17, v11, v25
  v_add_u32 v11, 192, v20
  v_lshrrev_b32 v19, 8, v11
  v_lshlrev_b32 v11, 12, v19
  v_sub_u32 v19, v30, v11
  v_mul_lo_u32 v11, v17, v7
  v_add_u32 v20, v19, v11
  buffer_load_dwordx4 v[136:139], v20, s[16:19], 0 offen offset:3072
  v_add_u32 v11, v12, 16
  v_mul_lo_u32 v20, v11, v7
  v_add_u32 v11, v16, v20
  buffer_load_dwordx4 v[140:143], v11, s[16:19], 0 offen offset:2048
  v_add_u32 v11, v17, 16
  v_mul_lo_u32 v20, v11, v7
  v_add_u32 v11, v19, v20
  buffer_load_dwordx4 v[144:147], v11, s[16:19], 0 offen offset:3072
  v_add_u32 v11, v12, 32
  v_mul_lo_u32 v20, v11, v7
  v_add_u32 v11, v16, v20
  buffer_load_dwordx4 v[148:151], v11, s[16:19], 0 offen offset:2048
  v_add_u32 v11, v17, 32
  v_mul_lo_u32 v20, v11, v7
  v_add_u32 v11, v19, v20
  buffer_load_dwordx4 v[152:155], v11, s[16:19], 0 offen offset:3072
  v_add_u32 v11, v12, 48
  v_mul_lo_u32 v12, v11, v7
  v_add_u32 v11, v16, v12
  buffer_load_dwordx4 v[156:159], v11, s[16:19], 0 offen offset:2048
  v_add_u32 v11, v17, 48
  v_mul_lo_u32 v12, v11, v7
  v_add_u32 v11, v19, v12
  buffer_load_dwordx4 v[160:163], v11, s[16:19], 0 offen offset:3072
  v_add_u32 v11, v31, 31
  v_mul_lo_u32 v12, v11, v8
  v_add_u32 v11, v12, v64
  buffer_load_dword v12, v11, s[20:23], 0 offen
  v_add_u32 v11, v31, 63
  v_mul_lo_u32 v16, v11, v8
  v_add_u32 v11, v16, v64
  buffer_load_dword v16, v11, s[20:23], 0 offen
  ; unhandled: rocdl.sched.barrier
  ds_read_b32 v11, v18
  ds_read_b32 v17, v18 offset:256
  ds_read_b128 v[164:167], v29 offset:2048
  ds_read_b128 v[168:171], v66 offset:2048
  ds_read_b128 v[172:175], v29 offset:4096
  ds_read_b128 v[176:179], v66 offset:4096
  ds_read_b128 v[180:183], v29 offset:6144
  ds_read_b128 v[184:187], v66 offset:6144
  ds_read_b128 v[188:191], v29 offset:8192
  ds_read_b128 v[192:195], v66 offset:8192
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:71], v[36:39], a[0:3], v5, v24 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v5, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:71], v[32:35], a[4:7], v5, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v5, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[48:51], a[8:11], v5, v65 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[52:55], a[8:11], v5, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[56:59], a[12:15], v5, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[60:63], a[12:15], v5, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[36:39], a[16:19], v5, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[40:43], a[16:19], v5, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[32:35], a[20:23], v5, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[44:47], a[20:23], v5, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[48:51], a[24:27], v5, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[52:55], a[24:27], v5, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v5, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[60:63], a[28:31], v5, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[36:39], a[32:35], v6, v24 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[40:43], a[32:35], v6, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[32:35], a[36:39], v6, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[44:47], a[36:39], v6, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[48:51], a[40:43], v6, v65 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[52:55], a[40:43], v6, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[56:59], a[44:47], v6, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[60:63], a[44:47], v6, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[36:39], a[48:51], v6, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[40:43], a[48:51], v6, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[32:35], a[52:55], v6, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[44:47], a[52:55], v6, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v6, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[52:55], a[56:59], v6, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[56:59], a[60:63], v6, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[60:63], a[60:63], v6, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[36:39], a[64:67], v3, v24 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[40:43], a[64:67], v3, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[32:35], a[68:71], v3, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[44:47], a[68:71], v3, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[48:51], a[72:75], v3, v65 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[52:55], a[72:75], v3, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[56:59], a[76:79], v3, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[60:63], a[76:79], v3, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[36:39], a[80:83], v3, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[40:43], a[80:83], v3, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:111], v[32:35], a[84:87], v3, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[44:47], a[84:87], v3, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[48:51], a[88:91], v3, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[112:115], v[52:55], a[88:91], v3, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[56:59], a[92:95], v3, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[60:63], a[92:95], v3, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:119], v[36:39], a[96:99], v10, v24 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[40:43], a[96:99], v10, v24 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:119], v[32:35], a[100:103], v10, v24 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[44:47], a[100:103], v10, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:119], v[48:51], a[104:107], v10, v65 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[52:55], a[104:107], v10, v65 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[56:59], a[108:111], v10, v65 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[60:63], a[108:111], v10, v65 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[36:39], a[112:115], v10, v24 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[40:43], a[112:115], v10, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[32:35], a[116:119], v10, v24 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[44:47], a[116:119], v10, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[48:51], a[120:123], v10, v65 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[52:55], a[120:123], v10, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[56:59], a[124:127], v10, v65 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[60:63], a[124:127], v10, v65 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ; unhandled: rocdl.sched.barrier
  ds_read_b32 v3, v18 offset:512
  ds_read_b32 v5, v18 offset:768
  ds_read_b128 v[24:27], v29 offset:10240
  ds_read_b128 v[32:35], v66 offset:10240
  ds_read_b128 v[36:39], v29 offset:12288
  ds_read_b128 v[40:43], v66 offset:12288
  ds_read_b128 v[44:47], v29 offset:14336
  ds_read_b128 v[48:51], v66 offset:14336
  ds_read_b128 v[52:55], v29 offset:16384
  ds_read_b128 v[28:31], v66 offset:16384
  ; unhandled: rocdl.sched.barrier
  s_waitcnt vmcnt(1)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[132:135], a[0:3], v11, v12 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[136:139], a[0:3], v11, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[140:143], a[4:7], v11, v12 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[144:147], a[4:7], v11, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_waitcnt vmcnt(0)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[148:151], a[8:11], v11, v16 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[152:155], a[8:11], v11, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[164:167], v[156:159], a[12:15], v11, v16 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[168:171], v[160:163], a[12:15], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[172:175], v[132:135], a[16:19], v11, v12 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[176:179], v[136:139], a[16:19], v11, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[172:175], v[140:143], a[20:23], v11, v12 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[176:179], v[144:147], a[20:23], v11, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[172:175], v[148:151], a[24:27], v11, v16 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[176:179], v[152:155], a[24:27], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[172:175], v[156:159], a[28:31], v11, v16 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[176:179], v[160:163], a[28:31], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[180:183], v[132:135], a[32:35], v17, v12 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[184:187], v[136:139], a[32:35], v17, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[180:183], v[140:143], a[36:39], v17, v12 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[144:147], a[36:39], v17, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[180:183], v[148:151], a[40:43], v17, v16 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[184:187], v[152:155], a[40:43], v17, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[180:183], v[156:159], a[44:47], v17, v16 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[184:187], v[160:163], a[44:47], v17, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[188:191], v[132:135], a[48:51], v17, v12 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[192:195], v[136:139], a[48:51], v17, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[188:191], v[140:143], a[52:55], v17, v12 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[192:195], v[144:147], a[52:55], v17, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[148:151], a[56:59], v17, v16 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[192:195], v[152:155], a[56:59], v17, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[188:191], v[156:159], a[60:63], v17, v16 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[192:195], v[160:163], a[60:63], v17, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:27], v[132:135], a[64:67], v3, v12 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[32:35], v[136:139], a[64:67], v3, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[24:27], v[140:143], a[68:71], v3, v12 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[32:35], v[144:147], a[68:71], v3, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[24:27], v[148:151], a[72:75], v3, v16 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[32:35], v[152:155], a[72:75], v3, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:27], v[156:159], a[76:79], v3, v16 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[32:35], v[160:163], a[76:79], v3, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[36:39], v[132:135], a[80:83], v3, v12 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[40:43], v[136:139], a[80:83], v3, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:39], v[140:143], a[84:87], v3, v12 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[40:43], v[144:147], a[84:87], v3, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[36:39], v[148:151], a[88:91], v3, v16 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[40:43], v[152:155], a[88:91], v3, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[36:39], v[156:159], a[92:95], v3, v16 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[40:43], v[160:163], a[92:95], v3, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[44:47], v[132:135], a[96:99], v5, v12 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:51], v[136:139], a[96:99], v5, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[44:47], v[140:143], a[100:103], v5, v12 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[48:51], v[144:147], a[100:103], v5, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[44:47], v[148:151], a[104:107], v5, v16 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[48:51], v[152:155], a[104:107], v5, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[44:47], v[156:159], a[108:111], v5, v16 op_sel:[0,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[48:51], v[160:163], a[108:111], v5, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[52:55], v[132:135], a[112:115], v5, v12 op_sel:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(0)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[28:31], v[136:139], a[112:115], v5, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[52:55], v[140:143], a[116:119], v5, v12 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[28:31], v[144:147], a[116:119], v5, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[52:55], v[148:151], a[120:123], v5, v16 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[28:31], v[152:155], a[120:123], v5, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[52:55], v[156:159], a[124:127], v5, v16 op_sel:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[28:31], v[160:163], a[124:127], v5, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_add_u32 v3, v4, v14
  v_lshl_add_u32 v4, s2, 8, v13
  v_lshlrev_b32 v5, 2, v22
  v_lshl_or_b32 v6, 0, 7, v5
  v_lshl_or_b32 v5, v2, 6, v1
  v_sub_u32 v1, v5, v67
  v_mul_lo_u32 v2, v3, v9
  v_mul_lo_u32 v3, v6, v9
  v_add_u32 v5, v2, v4
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  s_mov_b64 s[56:57], s[24:25]
  v_readfirstlane_b32 s59, v5
  s_mul_hi_u32 s58, s59, 2
  s_mul_i32 s60, s59, 2
  s_add_u32 s56, s56, s60
  s_addc_u32 s57, s57, s58
  s_mov_b32 s58, 0x7FFFFFFC
  s_mov_b32 s59, 0x20000
  v_accvgpr_read_b32 v2, a0
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_or_b32 v2, v6, 1
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a1
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_or_b32 v2, v6, 2
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a2
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_or_b32 v2, v6, 3
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a3
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a4
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a5
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a6
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a7
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a8
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a9
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a10
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a11
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a12
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a13
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a14
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a15
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, v6, 16
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a16
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, v6, 17
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a17
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, v6, 18
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a18
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, v6, 19
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a19
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a20
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a21
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a22
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a23
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a24
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a25
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a26
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a27
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a28
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a29
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a30
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a31
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, v6, 32
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a32
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, v6, 33
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a33
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, v6, 34
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a34
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, v6, 35
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a35
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a36
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a37
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a38
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a39
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a40
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a41
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a42
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a43
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a44
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a45
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a46
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a47
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, v6, 48
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a48
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, v6, 49
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a49
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, v6, 50
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a50
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, v6, 51
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a51
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a52
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a53
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a54
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a55
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a56
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a57
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a58
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a59
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a60
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a61
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a62
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a63
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, v6, 64
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a64
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, 65, v6
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a65
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, 66, v6
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a66
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, 67, v6
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a67
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a68
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a69
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a70
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a71
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a72
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a73
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a74
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a75
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a76
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a77
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a78
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a79
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, 80, v6
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a80
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, 81, v6
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a81
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, 82, v6
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a82
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, 83, v6
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a83
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a84
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a85
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a86
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a87
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a88
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a89
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a90
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a91
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a92
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a93
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a94
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a95
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, 96, v6
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a96
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, 97, v6
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a97
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, 98, v6
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a98
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, 99, v6
  v_mul_lo_u32 v10, v2, v9
  v_add_u32 v2, v10, v1
  v_lshlrev_b32 v10, 1, v2
  v_accvgpr_read_b32 v2, a99
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a100
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a101
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a102
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a103
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a104
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a105
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a106
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a107
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v10, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a108
  v_cvt_pk_bf16_f32 v11, v2, 0
  buffer_store_short v11, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a109
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a110
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a111
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v10, s[56:59], 0 offen offset:96
  v_add_u32 v2, 112, v6
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v1
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a112
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[56:59], 0 offen
  v_add_u32 v2, 113, v6
  v_mul_lo_u32 v4, v2, v9
  v_add_u32 v2, v4, v1
  v_lshlrev_b32 v4, 1, v2
  v_accvgpr_read_b32 v2, a113
  v_cvt_pk_bf16_f32 v5, v2, 0
  buffer_store_short v5, v4, s[56:59], 0 offen
  v_add_u32 v2, 114, v6
  v_mul_lo_u32 v5, v2, v9
  v_add_u32 v2, v5, v1
  v_lshlrev_b32 v5, 1, v2
  v_accvgpr_read_b32 v2, a114
  v_cvt_pk_bf16_f32 v10, v2, 0
  buffer_store_short v10, v5, s[56:59], 0 offen
  v_add_u32 v2, 115, v6
  v_mul_lo_u32 v6, v2, v9
  v_add_u32 v2, v6, v1
  v_lshlrev_b32 v1, 1, v2
  v_accvgpr_read_b32 v2, a115
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v1, s[56:59], 0 offen
  v_accvgpr_read_b32 v2, a116
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v3, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a117
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v4, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a118
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v5, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a119
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v1, s[56:59], 0 offen offset:32
  v_accvgpr_read_b32 v2, a120
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v3, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a121
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v4, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a122
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v5, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a123
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v1, s[56:59], 0 offen offset:64
  v_accvgpr_read_b32 v2, a124
  v_cvt_pk_bf16_f32 v6, v2, 0
  buffer_store_short v6, v3, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a125
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v4, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a126
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v5, s[56:59], 0 offen offset:96
  v_accvgpr_read_b32 v2, a127
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v1, s[56:59], 0 offen offset:96
  s_endpgm

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel wave_mxfp4_static_gemm_128x256x256_1024x1024x8192
  .amdhsa_group_segment_fixed_size 34816
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_private_segment_size 0
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_enable_private_segment 0
  .amdhsa_accum_offset 200
  .amdhsa_next_free_vgpr 328
  .amdhsa_next_free_sgpr 64
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 1
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: wave_mxfp4_static_gemm_128x256x256_1024x1024x8192
    .symbol: wave_mxfp4_static_gemm_128x256x256_1024x1024x8192.kd
    .args:
      - .name:       arg0_ptr
        .offset:     0
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg1_ptr
        .offset:     8
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg2_ptr
        .offset:     16
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg3_ptr
        .offset:     24
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg4_ptr
        .offset:     32
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg5_ptr
        .offset:     40
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg6_ptr
        .offset:     48
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg7_ptr
        .offset:     56
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg8_ptr
        .offset:     64
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg9_ptr
        .offset:     72
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
    .kernarg_segment_size: 80
    .group_segment_fixed_size: 34816
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 61
    .vgpr_count: 196
    .agpr_count: 128
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
