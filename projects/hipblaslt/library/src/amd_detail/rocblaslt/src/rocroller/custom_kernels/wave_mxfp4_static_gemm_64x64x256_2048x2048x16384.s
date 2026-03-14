.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text

.protected wave_mxfp4_static_gemm_64x64x256_2048x2048x16384
.globl wave_mxfp4_static_gemm_64x64x256_2048x2048x16384
.p2align 8
.type wave_mxfp4_static_gemm_64x64x256_2048x2048x16384,@function
wave_mxfp4_static_gemm_64x64x256_2048x2048x16384:
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
  v_lshlrev_b32 v4, 6, s3
  v_lshl_or_b32 v5, v2, 3, v4
  v_lshrrev_b32 v6, 3, v1
  v_or_b32 v10, v5, v6
  v_lshl_or_b32 v5, v2, 3, v6
  v_lshlrev_b32 v11, 6, 0
  v_sub_u32 v12, v10, v11
  v_lshrrev_b32 v13, 5, s2
  v_lshlrev_b32 v14, 6, v13
  v_add_u32 v16, v12, v14
  v_add_u32 v12, s3, v13
  v_lshrrev_b32 v17, 5, v12
  v_lshlrev_b32 v12, 11, v17
  v_sub_u32 v17, v16, v12
  v_and_b32 v16, v6, 7
  v_and_b32 v6, v1, 7
  v_xor_b32 v18, v6, v16
  v_lshlrev_b32 v16, 4, v18
  v_sub_u32 v18, v3, v11
  s_nop 0
  v_readfirstlane_b32 s0, v18
  v_lshlrev_b32 v18, 13, v17
  v_add_u32 v17, v18, v16
  s_lshl_b32 s1, s0, 7
  s_add_u32 s4, s1, 9216
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v17, s[8:11], 0 offen lds
  v_add_u32 v17, 128, v16
  v_add_u32 v19, v18, v17
  s_add_u32 s5, s1, 1024
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v19, s[8:11], 0 offen lds
  v_add_u32 v19, v5, 32
  v_and_b32 v5, 4294967232, v19
  v_sub_u32 v19, v10, v5
  v_add_u32 v5, v19, v14
  v_sub_u32 v10, v5, v12
  v_add_u32 v5, v10, 32
  v_add_u32 v10, v2, 4
  v_lshrrev_b32 v19, 3, v10
  v_lshlrev_b32 v10, 6, v19
  v_sub_u32 v19, v3, v10
  v_add_u32 v3, v19, 32
  s_nop 0
  v_readfirstlane_b32 s0, v3
  v_lshlrev_b32 v3, 13, v5
  v_add_u32 v5, v3, v16
  s_lshl_b32 s1, s0, 7
  s_add_u32 s6, s1, 9216
  s_mov_b32 m0, s6
  buffer_load_dwordx4 v5, s[8:11], 0 offen lds
  v_add_u32 v5, v3, v17
  s_add_u32 s7, s1, 1024
  s_mov_b32 m0, s7
  buffer_load_dwordx4 v5, s[8:11], 0 offen lds
  v_min_i32 v5, v2, 1
  v_lshl_add_u32 v10, v5, 5, v4
  v_add_u32 v17, v10, v14
  v_sub_u32 v19, v17, v12
  v_lshlrev_b32 v17, 2, v1
  v_lshlrev_b32 v20, 8, 0
  v_sub_u32 v21, v17, v20
  v_lshlrev_b32 v22, 8, v5
  s_nop 0
  v_readfirstlane_b32 s0, v22
  v_lshl_add_u32 v5, v19, 9, v21
  s_mov_b32 s40, s12
  s_and_b32 s41, s13, 0xffff
  s_or_b32 s41, s41, 0x40400000
  s_mov_b32 s42, 0xFFFFFFFF
  s_mov_b32 s43, 0x27000
  s_add_u32 s12, s0, 512
  s_mov_b32 m0, s12
  buffer_load_dword v5, s[40:43], 0 offen lds
  v_and_b32 v5, v1, 63
  v_add_u32 v19, v5, 64
  v_lshrrev_b32 v22, 7, v19
  v_or_b32 v19, v10, v22
  v_add_u32 v23, v19, v14
  v_sub_u32 v19, v23, v12
  v_lshlrev_b32 v23, 9, v22
  v_sub_u32 v22, v21, v23
  v_add_u32 v21, 256, v22
  v_lshl_add_u32 v22, v19, 9, v21
  s_mov_b32 m0, s0
  buffer_load_dword v22, s[40:43], 0 offen lds
  v_lshlrev_b32 v19, 6, s2
  v_lshlrev_b32 v21, 4, v2
  v_lshl_or_b32 v2, s2, 6, v21
  v_and_b32 v22, v1, 15
  v_lshrrev_b32 v23, 4, v5
  v_and_b32 v24, 4294967280, v5
  v_add_u32 v5, v22, v24
  v_or_b32 v25, v2, 0
  v_lshlrev_b32 v26, 11, v13
  v_sub_u32 v13, v25, v26
  v_add_u32 v25, v13, v12
  v_lshrrev_b32 v13, 4, v1
  v_lshlrev_b32 v27, 8, v13
  v_sub_u32 v28, 0, v27
  v_lshl_add_u32 v29, v1, 4, v28
  v_lshlrev_b32 v28, 8, v23
  v_add_u32 v30, v29, v28
  v_lshlrev_b32 v29, 13, 0
  v_sub_u32 v31, v30, v29
  v_mul_lo_u32 v29, v25, v7
  v_add_u32 v25, v29, v31
  buffer_load_dwordx4 v[32:35], v25, s[16:19], 0 offen
  buffer_load_dwordx4 v[36:39], v25, s[16:19], 0 offen offset:1024
  ; unhandled: rocdl.sched.barrier
  v_add_u32 v25, v1, v21
  v_and_b32 v21, 4294967280, v1
  v_sub_u32 v29, v25, v21
  v_and_b32 v25, 4294967264, v29
  v_add_u32 v31, v19, v25
  v_lshlrev_b32 v25, 6, v13
  v_sub_u32 v13, v17, v25
  v_and_b32 v17, v29, 31
  v_lshrrev_b32 v40, 4, v17
  v_add_u32 v17, v13, v40
  v_lshlrev_b32 v13, 6, v23
  v_add_u32 v41, v17, v13
  v_lshrrev_b32 v17, 9, v41
  v_add_u32 v42, v31, v17
  v_sub_u32 v17, v42, v26
  v_add_u32 v42, v17, v12
  v_and_b32 v17, 511, v41
  v_mul_lo_u32 v43, v42, v8
  v_add_u32 v42, v43, v17
  buffer_load_ubyte v17, v42, s[20:23], 0 offen
  v_add_u32 v42, v23, 4
  v_lshrrev_b32 v43, 3, v42
  v_mov_b32 v15, 252
  v_mul_lo_u32 v44, v43, v15
  v_add_u32 v45, v41, v44
  v_add_u32 v44, v45, 2
  v_lshrrev_b32 v46, 9, v44
  v_add_u32 v47, v31, v46
  v_sub_u32 v46, v47, v26
  v_add_u32 v47, v46, v12
  v_and_b32 v46, 4294966784, v44
  v_sub_u32 v44, v45, v46
  v_mul_lo_u32 v45, v47, v8
  v_add_u32 v46, v44, v45
  buffer_load_ubyte v44, v46, s[20:23], 0 offen offset:2
  ; unhandled: rocdl.sched.barrier
  s_waitcnt vmcnt(0)
  s_barrier
  ; unhandled: rocdl.sched.barrier
  v_and_b32 v45, 4294967232, v1
  v_add_u32 v46, v22, v45
  v_xor_b32 v22, v23, v6
  v_lshlrev_b32 v47, 4, v22
  v_lshlrev_b32 v22, 7, v46
  v_add_u32 v46, v22, v47
  ds_read_b128 v[48:51], v46 offset:9216
  v_xor_b32 v47, v42, v6
  v_lshlrev_b32 v6, 4, v47
  v_add_u32 v42, v22, v6
  ds_read_b128 v[52:55], v42 offset:9216
  ds_read_b128 v[56:59], v46 offset:11264
  ds_read_b128 v[60:63], v42 offset:11264
  v_lshl_or_b32 v6, v1, 2, v20
  ds_read_b32 v22, v6 offset:512
  s_mov_b32 s1, 0
  v_mov_b32 v15, 0
  v_accvgpr_write_b32 a0, v15
  v_accvgpr_write_b32 a1, v15
  v_accvgpr_write_b32 a2, v15
  v_accvgpr_write_b32 a3, v15
  v_lshlrev_b32 v47, 2, v43
  v_add_u32 v43, v23, 12
  v_lshrrev_b32 v64, 3, v43
  v_lshlrev_b32 v43, 8, v64
  v_add_u32 v64, v23, 20
  v_lshrrev_b32 v65, 3, v64
  v_lshlrev_b32 v64, 8, v65
  v_lshl_add_u32 v65, s1, 7, v16
  v_add_u32 v16, 256, v65
  v_add_u32 v66, v18, v16
  v_add_u32 v67, v3, v16
  v_add_u32 v16, 384, v65
  v_add_u32 v65, v18, v16
  v_add_u32 v18, v3, v16
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
  s_mov_b32 s13, 0
  s_waitcnt lgkmcnt(0)
L_loop_0:
  s_waitcnt vmcnt(3) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[32:35], a[0:3], v22, v17 cbsz:4 blgp:4
  v_lshl_or_b32 v16, s1, 7, v1
  v_sub_u32 v68, v16, v21
  v_add_u32 v16, v68, v24
  v_add_u32 v68, 128, v16
  v_lshrrev_b32 v69, 9, v68
  v_add_u32 v68, v2, v69
  v_sub_u32 v70, v68, v26
  v_add_u32 v68, v70, v12
  v_lshlrev_b32 v70, 11, s1
  v_lshl_or_b32 v71, v1, 4, v70
  v_sub_u32 v70, v71, v27
  v_add_u32 v71, v70, v28
  v_lshlrev_b32 v70, 13, v69
  v_sub_u32 v69, v71, v70
  v_mul_lo_u32 v70, v68, v7
  v_add_u32 v68, v69, v70
  buffer_load_dwordx4 v[72:75], v68, s[16:19], 0 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v22, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[32:35], a[4:7], v22, v17 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:63], v[36:39], a[4:7], v22, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  ds_read_b128 v[76:79], v46 offset:13312
  v_add_u32 v69, 192, v16
  v_lshrrev_b32 v70, 9, v69
  v_add_u32 v69, v2, v70
  v_sub_u32 v80, v69, v26
  v_add_u32 v69, v80, v12
  v_lshlrev_b32 v80, 13, v70
  v_sub_u32 v70, v71, v80
  v_mul_lo_u32 v80, v69, v7
  v_add_u32 v69, v70, v80
  buffer_load_dwordx4 v[80:83], v69, s[16:19], 0 offen offset:3072
  ds_read_b128 v[84:87], v42 offset:13312
  ds_read_b128 v[88:91], v46 offset:15360
  ds_read_b128 v[92:95], v42 offset:15360
  ds_read_b32 v69, v6 offset:768
  v_lshlrev_b32 v70, 8, s1
  v_lshl_or_b32 v96, v1, 2, v70
  v_sub_u32 v70, v96, v25
  v_add_u32 v97, v70, v40
  v_add_u32 v70, v97, v13
  v_add_u32 v97, 256, v70
  v_lshrrev_b32 v98, 9, v97
  v_add_u32 v99, v31, v98
  v_sub_u32 v98, v99, v26
  v_add_u32 v99, v98, v12
  v_and_b32 v98, 4294966784, v97
  v_sub_u32 v97, v70, v98
  v_mul_lo_u32 v98, v99, v8
  v_add_u32 v99, v97, v98
  buffer_load_ubyte v97, v99, s[20:23], 0 offen offset:256
  v_sub_u32 v98, v70, v47
  v_add_u32 v99, v98, v43
  v_add_u32 v100, v99, 2
  v_lshrrev_b32 v99, 9, v100
  v_add_u32 v100, v31, v99
  v_sub_u32 v99, v100, v26
  v_add_u32 v100, v99, v12
  v_add_u32 v99, v70, v43
  v_sub_u32 v101, v99, v47
  v_add_u32 v99, v101, 2
  v_and_b32 v102, 4294966784, v99
  v_sub_u32 v99, v101, v102
  v_mul_lo_u32 v101, v100, v8
  v_add_u32 v100, v99, v101
  buffer_load_ubyte v99, v100, s[20:23], 0 offen offset:2
  s_waitcnt vmcnt(2) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[32:35], a[8:11], v69, v17 cbsz:4 blgp:4
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v66, s[8:11], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[36:39], a[8:11], v69, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[32:35], a[12:15], v69, v17 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[36:39], a[12:15], v69, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  ds_read_b128 v[76:79], v46 offset:1024
  s_mov_b32 m0, s6
  buffer_load_dwordx4 v67, s[8:11], s13 offen lds
  ds_read_b128 v[84:87], v42 offset:1024
  ds_read_b128 v[88:91], v46 offset:3072
  ds_read_b128 v[92:95], v42 offset:3072
  ds_read_b32 v3, v6
  v_lshl_or_b32 v68, s1, 6, v1
  v_sub_u32 v69, v68, v11
  v_lshrrev_b32 v68, 7, v69
  v_add_u32 v100, v10, v68
  v_add_u32 v101, v100, v14
  v_sub_u32 v100, v101, v12
  v_add_u32 v101, v100, 1
  v_sub_u32 v100, v96, v20
  v_lshlrev_b32 v96, 9, v68
  v_sub_u32 v68, v100, v96
  v_lshl_add_u32 v96, v101, 9, v68
  s_mov_b32 m0, s12
  buffer_load_dword v96, s[40:43], 0 offen lds
  s_waitcnt vmcnt(3) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[72:75], a[0:3], v3, v97 cbsz:4 blgp:4
  v_add_u32 v76, 256, v16
  v_lshrrev_b32 v77, 9, v76
  v_add_u32 v76, v2, v77
  v_sub_u32 v78, v76, v26
  v_add_u32 v76, v78, v12
  v_lshlrev_b32 v78, 13, v77
  v_sub_u32 v77, v71, v78
  v_mul_lo_u32 v78, v76, v7
  v_add_u32 v76, v77, v78
  v_add_u32 v77, 4096, v76
  buffer_load_dwordx4 v[32:35], v77, s[16:19], 0 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[80:83], a[0:3], v3, v99 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[72:75], a[4:7], v3, v97 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:95], v[80:83], a[4:7], v3, v99 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  ds_read_b128 v[84:87], v46 offset:5120
  v_add_u32 v3, 320, v16
  v_lshrrev_b32 v16, 9, v3
  v_add_u32 v3, v2, v16
  v_sub_u32 v77, v3, v26
  v_add_u32 v3, v77, v12
  v_lshlrev_b32 v77, 13, v16
  v_sub_u32 v16, v71, v77
  v_mul_lo_u32 v71, v3, v7
  v_add_u32 v3, v16, v71
  v_add_u32 v16, 5120, v3
  buffer_load_dwordx4 v[36:39], v16, s[16:19], 0 offen
  ds_read_b128 v[88:91], v42 offset:5120
  ds_read_b128 v[92:95], v46 offset:7168
  ds_read_b128 v[104:107], v42 offset:7168
  ds_read_b32 v3, v6 offset:256
  v_lshrrev_b32 v16, 9, v70
  v_add_u32 v71, v31, v16
  v_sub_u32 v16, v71, v26
  v_add_u32 v71, v16, v12
  v_add_u32 v16, v71, 1
  v_and_b32 v71, 511, v70
  v_mul_lo_u32 v77, v16, v8
  v_add_u32 v16, v77, v71
  buffer_load_ubyte v17, v16, s[20:23], 0 offen
  v_add_u32 v16, v98, v64
  v_add_u32 v71, v16, 2
  v_lshrrev_b32 v16, 9, v71
  v_add_u32 v71, v31, v16
  v_sub_u32 v16, v71, v26
  v_add_u32 v71, v16, v12
  v_add_u32 v16, v70, v64
  v_sub_u32 v70, v16, v47
  v_add_u32 v16, v70, 2
  v_and_b32 v77, 4294966784, v16
  v_sub_u32 v16, v70, v77
  v_mul_lo_u32 v70, v71, v8
  v_add_u32 v71, v16, v70
  buffer_load_ubyte v44, v71, s[20:23], 0 offen offset:2
  s_waitcnt vmcnt(2) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[72:75], a[8:11], v3, v97 cbsz:4 blgp:4
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v65, s[8:11], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[80:83], a[8:11], v3, v99 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[72:75], a[12:15], v3, v97 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[80:83], a[12:15], v3, v99 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  ds_read_b128 v[48:51], v46 offset:9216
  s_mov_b32 m0, s7
  buffer_load_dwordx4 v18, s[8:11], s13 offen lds
  ds_read_b128 v[52:55], v42 offset:9216
  ds_read_b128 v[56:59], v46 offset:11264
  ds_read_b128 v[60:63], v42 offset:11264
  ds_read_b32 v22, v6 offset:512
  v_add_u32 v3, 192, v69
  v_lshrrev_b32 v16, 7, v3
  v_add_u32 v3, v10, v16
  v_add_u32 v16, v3, v14
  v_sub_u32 v3, v16, v12
  v_add_u32 v16, v69, 64
  v_lshrrev_b32 v68, 7, v16
  v_lshlrev_b32 v16, 9, v68
  v_sub_u32 v68, v100, v16
  v_add_u32 v16, 256, v68
  v_lshl_add_u32 v68, v3, 9, v16
  s_mov_b32 m0, s0
  buffer_load_dword v68, s[40:43], 0 offen lds
  s_waitcnt vmcnt(3) & lgkmcnt(0)
  s_waitcnt vmcnt(0)
  s_barrier
  s_add_u32 s1, s1, 2
  s_add_u32 s13, s13, 256
  s_cmp_lt_u32 s1, 62
  s_cbranch_scc1 L_loop_0
  s_waitcnt vmcnt(0) & lgkmcnt(0)
  s_barrier
  ds_read_b32 v1, v6 offset:768
  ds_read_b128 v[64:67], v46 offset:13312
  ds_read_b128 v[68:71], v42 offset:13312
  ds_read_b128 v[72:75], v46 offset:15360
  ds_read_b128 v[76:79], v42 offset:15360
  v_add_u32 v3, 8064, v5
  v_lshrrev_b32 v10, 9, v3
  v_add_u32 v3, v2, v10
  v_sub_u32 v10, v3, v26
  v_add_u32 v3, v10, v12
  v_add_u32 v10, 384, v5
  v_lshrrev_b32 v11, 9, v10
  v_lshlrev_b32 v10, 13, v11
  v_sub_u32 v11, v30, v10
  v_mul_lo_u32 v10, v3, v7
  v_add_u32 v3, v11, v10
  v_add_u32 v10, 6144, v3
  buffer_load_dwordx4 v[80:83], v10, s[16:19], 0 offen
  v_add_u32 v3, 8128, v5
  v_lshrrev_b32 v10, 9, v3
  v_add_u32 v3, v2, v10
  v_sub_u32 v2, v3, v26
  v_add_u32 v3, v2, v12
  v_add_u32 v2, 448, v5
  v_lshrrev_b32 v5, 9, v2
  v_lshlrev_b32 v2, 13, v5
  v_sub_u32 v5, v30, v2
  v_mul_lo_u32 v2, v3, v7
  v_add_u32 v3, v5, v2
  v_add_u32 v2, 7168, v3
  buffer_load_dwordx4 v[84:87], v2, s[16:19], 0 offen
  v_add_u32 v2, 16128, v41
  v_lshrrev_b32 v3, 9, v2
  v_add_u32 v2, v31, v3
  v_sub_u32 v3, v2, v26
  v_add_u32 v2, v3, v12
  v_add_u32 v3, 256, v41
  v_and_b32 v5, 4294966784, v3
  v_sub_u32 v3, v41, v5
  v_mul_lo_u32 v5, v2, v8
  v_add_u32 v2, v3, v5
  buffer_load_ubyte v3, v2, s[20:23], 0 offen offset:256
  v_sub_u32 v2, v41, v47
  v_add_u32 v5, 508, v23
  v_lshrrev_b32 v10, 3, v5
  v_lshlrev_b32 v5, 8, v10
  v_add_u32 v10, v2, v5
  v_add_u32 v2, v10, 2
  v_lshrrev_b32 v10, 9, v2
  v_add_u32 v2, v31, v10
  v_sub_u32 v10, v2, v26
  v_add_u32 v2, v10, v12
  v_add_u32 v10, v41, v5
  v_sub_u32 v5, v10, v47
  v_add_u32 v10, v5, 2
  v_and_b32 v11, 4294966784, v10
  v_sub_u32 v10, v5, v11
  v_mul_lo_u32 v5, v2, v8
  v_add_u32 v2, v10, v5
  buffer_load_ubyte v5, v2, s[20:23], 0 offen offset:2
  ; unhandled: rocdl.sched.barrier
  ds_read_b32 v2, v6
  ds_read_b128 v[88:91], v46 offset:1024
  ds_read_b128 v[92:95], v42 offset:1024
  ds_read_b128 v[96:99], v46 offset:3072
  ds_read_b128 v[100:103], v42 offset:3072
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[32:35], a[0:3], v22, v17 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v22, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[32:35], a[4:7], v22, v17 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:63], v[36:39], a[4:7], v22, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[32:35], a[8:11], v1, v17 cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[36:39], a[8:11], v1, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[32:35], a[12:15], v1, v17 op_sel:[1,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[36:39], a[12:15], v1, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  ; unhandled: rocdl.sched.barrier
  ds_read_b32 v1, v6 offset:256
  ds_read_b128 v[32:35], v46 offset:5120
  ds_read_b128 v[36:39], v42 offset:5120
  ds_read_b128 v[48:51], v46 offset:7168
  ds_read_b128 v[52:55], v42 offset:7168
  ; unhandled: rocdl.sched.barrier
  s_waitcnt vmcnt(1)
  s_waitcnt lgkmcnt(8)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[80:83], a[0:3], v2, v3 cbsz:4 blgp:4
  s_waitcnt vmcnt(0)
  s_waitcnt lgkmcnt(7)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:95], v[84:87], a[0:3], v2, v5 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(6)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[80:83], a[4:7], v2, v3 op_sel:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(5)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[100:103], v[84:87], a[4:7], v2, v5 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(3)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:35], v[80:83], a[8:11], v1, v3 cbsz:4 blgp:4
  s_waitcnt lgkmcnt(2)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:39], v[84:87], a[8:11], v1, v5 op_sel_hi:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(1)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[80:83], a[12:15], v1, v3 op_sel:[1,0,0] cbsz:4 blgp:4
  s_waitcnt lgkmcnt(0)
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[84:87], a[12:15], v1, v5 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
  v_add_u32 v1, v4, v14
  v_sub_u32 v2, v1, v12
  v_sub_u32 v1, v19, v26
  v_add_u32 v3, v1, v12
  v_lshl_add_u32 v1, v23, 2, v45
  v_mul_lo_u32 v4, v2, v9
  v_mul_lo_u32 v2, v1, v9
  v_add_u32 v5, v4, v3
  v_add_u32 v3, v2, v29
  v_lshlrev_b32 v2, 1, v3
  s_mov_b64 s[48:49], s[24:25]
  v_readfirstlane_b32 s51, v5
  s_mul_hi_u32 s50, s51, 2
  s_mul_i32 s52, s51, 2
  s_add_u32 s48, s48, s52
  s_addc_u32 s49, s49, s50
  s_mov_b32 s50, 0x7FFFFFFC
  s_mov_b32 s51, 0x20000
  v_accvgpr_read_b32 v3, a0
  v_cvt_pk_bf16_f32 v4, v3, 0
  buffer_store_short v4, v2, s[48:51], 0 offen
  v_add_u32 v2, v1, 1
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a1
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 2
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a2
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 3
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a3
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 16
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a4
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 17
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a5
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 18
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a6
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 19
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a7
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 32
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a8
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 33
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a9
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 34
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a10
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 35
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a11
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 48
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a12
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 49
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a13
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 50
  v_mul_lo_u32 v3, v2, v9
  v_add_u32 v2, v3, v29
  v_lshlrev_b32 v3, 1, v2
  v_accvgpr_read_b32 v2, a14
  v_cvt_pk_bf16_f32 v4, v2, 0
  buffer_store_short v4, v3, s[48:51], 0 offen
  v_add_u32 v2, v1, 51
  v_mul_lo_u32 v1, v2, v9
  v_add_u32 v2, v1, v29
  v_lshlrev_b32 v1, 1, v2
  v_accvgpr_read_b32 v2, a15
  v_cvt_pk_bf16_f32 v3, v2, 0
  buffer_store_short v3, v1, s[48:51], 0 offen
  s_endpgm

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_2048x2048x16384
  .amdhsa_group_segment_fixed_size 17408
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_private_segment_size 0
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_enable_private_segment 0
  .amdhsa_accum_offset 112
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr 56
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
  - .name: wave_mxfp4_static_gemm_64x64x256_2048x2048x16384
    .symbol: wave_mxfp4_static_gemm_64x64x256_2048x2048x16384.kd
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
    .group_segment_fixed_size: 17408
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 53
    .vgpr_count: 108
    .agpr_count: 16
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
