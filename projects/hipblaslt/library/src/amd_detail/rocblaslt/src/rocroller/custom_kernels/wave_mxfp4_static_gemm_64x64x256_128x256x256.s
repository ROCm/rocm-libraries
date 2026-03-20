; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_128x256x256
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_128x256x256,@function
wave_mxfp4_static_gemm_64x64x256_128x256x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v4, v0, 10, 10
	v_lshrrev_b32_e32 v2, 6, v1
	v_lshlrev_b32_e32 v0, 5, v4
	v_lshl_or_b32 v3, v2, 3, v0
	s_mov_b64 s[24:25], s[6:7]
	s_mov_b64 s[20:21], s[2:3]
	v_readfirstlane_b32 s7, v3
	v_lshrrev_b32_e32 v3, 3, v1
	v_or_b32_e32 v5, v3, v0
	s_lshl_b32 s6, s16, 6
	v_bitop3_b32 v6, v3, 7, v1 bitop3:0x48
	s_and_b32 s12, s21, 0xffff
	v_or_b32_e32 v5, s6, v5
	v_lshlrev_b32_e32 v7, 4, v6
	s_or_b32 s21, s12, 4.0
	s_lshl_b32 s12, s7, 7
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_lshl_or_b32 v5, v5, 7, v7
	s_mov_b32 m0, s12
	v_bfe_u32 v7, v1, 6, 1
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	v_bitop3_b32 v5, v3, 48, v0 bitop3:0xc8
	v_lshl_or_b32 v5, v6, 1, v5
	v_lshlrev_b32_e32 v6, 1, v1
	s_lshl_b32 s7, s17, 6
	v_lshlrev_b32_e32 v3, 4, v3
	v_and_b32_e32 v6, 0x180, v6
	v_or3_b32 v5, v5, v7, s7
	v_sub_u32_e32 v3, v3, v6
	v_lshl_add_u32 v3, v5, 7, v3
	s_add_i32 m0, s12, 0x2000
	v_lshlrev_b32_e32 v10, 2, v1
	v_lshrrev_b32_e32 v5, 4, v1
	s_movk_i32 s12, 0xffc0
	s_and_b32 s13, s25, 0xffff
	v_mad_i32_i24 v12, v5, s12, v10
	s_or_b32 s25, s13, 4.0
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_ashrrev_i32_e32 v14, 31, v12
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_bitop3_b32 v3, v12, v14, v7 bitop3:0x36
	v_ashrrev_i32_e32 v6, 31, v3
	v_lshrrev_b32_e32 v6, 29, v6
	s_mul_i32 s15, s15, s6
	s_mul_hi_u32 s12, s14, s6
	v_add_u32_e32 v3, v3, v6
	s_or_b32 s12, s12, s15
	s_mul_i32 s13, s14, s6
	v_ashrrev_i32_e32 v6, 3, v3
	v_bfe_u32 v3, v1, 4, 2
	s_add_u32 s20, s4, s13
	v_lshlrev_b32_e32 v15, 3, v3
	v_lshrrev_b32_e32 v8, 2, v1
	s_addc_u32 s4, s5, s12
	s_and_b32 s5, s14, 0x3fff
	v_and_or_b32 v16, v8, 32, v15
	s_bitset1_b32 s5, 14
	v_xad_u32 v8, v6, v14, v16
	v_bitop3_b32 v6, v12, 5, v7 bitop3:0xc8
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_or_b32_e32 v13, v12, v7
	s_or_b32 s21, s4, s5
	v_mad_u64_u32 v[8:9], s[4:5], s14, v8, v[6:7]
	v_or_b32_e32 v6, 2, v13
	v_bitop3_b32 v9, v12, -3, v7 bitop3:0x36
	v_cmp_gt_i32_e32 vcc, 0, v12
	s_load_dwordx2 s[2:3], s[0:1], 0x40
	v_mul_i32_i24_e32 v11, 0xffffffc0, v5
	v_cndmask_b32_e32 v6, v6, v9, vcc
	v_ashrrev_i32_e32 v9, 31, v6
	v_lshrrev_b32_e32 v9, 29, v9
	v_add_u32_e32 v6, v6, v9
	v_ashrrev_i32_e32 v6, 3, v6
	v_xor_b32_e32 v6, v6, v14
	v_add_u32_e32 v9, v16, v6
	v_mul_lo_u32 v9, v9, s14
	v_lshlrev_b32_e32 v6, 3, v6
	v_sub_u32_e32 v6, v9, v6
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s3, s7
	s_mul_hi_u32 s4, s2, s7
	v_add3_u32 v9, v6, v7, v12
	buffer_load_ubyte v7, v8, s[20:23], 0 offen
	buffer_load_ubyte v6, v9, s[20:23], 0 offen offset:2
	v_bfe_u32 v8, v1, 1, 3
	s_add_i32 s4, s4, s3
	s_mul_i32 s3, s2, s7
	v_or3_b32 v9, v8, v15, v0
	s_add_u32 s20, s8, s3
	v_mul_lo_u32 v9, s2, v9
	s_addc_u32 s3, s9, s4
	s_and_b32 s2, s2, 0x3fff
	v_lshlrev_b32_e32 v8, 3, v8
	s_bitset1_b32 s2, 14
	v_sub_u32_e32 v8, v11, v8
	s_and_b32 s3, s3, 0xffff
	s_lshl_b32 s2, s2, 16
	s_or_b32 s21, s3, s2
	v_add3_u32 v8, v9, v8, v10
	buffer_load_dword v8, v8, s[20:23], 0 offen
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_b64 s[2:3], vcc, exec
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_scc1 .LBB0_4
	s_barrier
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	s_cbranch_execnz .LBB0_3
.LBB0_2:
	s_barrier
.LBB0_3:
	v_lshlrev_b32_e32 v11, 7, v1
	v_bitop3_b32 v10, v3, v1, 7 bitop3:0x78
	v_mul_i32_i24_e32 v5, 0xfffff800, v5
	v_lshl_add_u32 v4, v4, 12, v11
	s_movk_i32 s2, 0x2000
	v_lshlrev_b32_e32 v13, 11, v2
	v_lshlrev_b32_e32 v10, 4, v10
	v_add3_u32 v4, v4, v5, s2
	v_add3_u32 v5, v13, v5, v11
	v_or_b32_e32 v12, v4, v10
	v_or_b32_e32 v10, v5, v10
	v_and_b32_e32 v9, 7, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[16:19], v10
	ds_read_b128 v[20:23], v12
	v_bitop3_b32 v9, v3, v9, 4 bitop3:0x36
	v_lshlrev_b32_e32 v9, 4, v9
	v_or_b32_e32 v5, v5, v9
	v_or_b32_e32 v4, v4, v9
	ds_read_b128 v[24:27], v5
	ds_read_b128 v[28:31], v12 offset:2048
	v_and_b32_e32 v9, 0xffff, v7
	v_and_b32_e32 v14, 0xffff, v6
	s_movk_i32 s2, 0x7fff
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[16:19], v[20:23], 0, v9, v8 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v4
	ds_read_b128 v[32:35], v4 offset:2048
	s_mul_i32 s1, s1, s6
	v_lshlrev_b32_e32 v2, 4, v2
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[24:27], v[20:23], v[10:13], v14, v8 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_lshl_or_b32 v2, v3, 2, v2
	v_and_b32_e32 v1, 15, v1
	v_lshlrev_b32_e32 v0, 1, v0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[16:19], v[28:31], 0, v9, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v16, 0x7fc0
	s_nop 2
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v15, v4, 16, 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[24:27], v[32:35], v[10:13], v14, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_bfe_u32 v14, v5, 16, 1
	v_add3_u32 v14, v5, v14, s2
	v_add3_u32 v15, v4, v15, s2
	v_bfe_u32 v12, v7, 16, 1
	v_bfe_u32 v13, v6, 16, 1
	v_add3_u32 v12, v7, v12, s2
	v_add3_u32 v13, v6, v13, s2
	v_lshrrev_b32_e32 v12, 16, v12
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v7, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_lshrrev_b32_e32 v14, 16, v14
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v6, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v12, v11, 16, 1
	v_bfe_u32 v13, v10, 16, 1
	v_cndmask_b32_e32 v5, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v14, v9, 16, 1
	v_add3_u32 v14, v9, v14, s2
	v_cndmask_b32_e32 v4, v16, v15, vcc
	v_bfe_u32 v15, v8, 16, 1
	v_add3_u32 v15, v8, v15, s2
	v_add3_u32 v13, v10, v13, s2
	v_add3_u32 v12, v11, v12, s2
	s_mul_hi_u32 s2, s0, s6
	s_add_i32 s3, s2, s1
	s_mul_i32 s2, s0, s6
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s1, s10, s2
	s_addc_u32 s2, s11, s3
	s_lshl_b32 s3, s7, 1
	v_mul_lo_u32 v2, s0, v2
	s_add_u32 s4, s1, s3
	v_lshl_add_u32 v0, v1, 1, v0
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v11, v11
	s_addc_u32 s1, s2, 0
	s_and_b32 s2, s0, 0x3fff
	v_lshl_add_u32 v0, v2, 1, v0
	s_lshl_b32 s0, s0, 1
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v11, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_add_u32_e32 v1, s0, v0
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v10, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	s_or_b32 s1, s2, s1
	v_add_u32_e32 v2, s0, v1
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v9, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	s_or_b32 s5, s1, 2.0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffd
	v_add_u32_e32 v3, s0, v2
	v_cndmask_b32_e32 v8, v16, v15, vcc
	buffer_store_short v4, v0, s[4:7], 0 offen
	buffer_store_short v5, v1, s[4:7], 0 offen
	buffer_store_short v6, v2, s[4:7], 0 offen
	buffer_store_short v7, v3, s[4:7], 0 offen
	buffer_store_short v8, v0, s[4:7], 0 offen offset:32
	buffer_store_short v9, v1, s[4:7], 0 offen offset:32
	buffer_store_short v10, v2, s[4:7], 0 offen offset:32
	buffer_store_short v11, v3, s[4:7], 0 offen offset:32
	s_endpgm
.LBB0_4:
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	s_branch .LBB0_2
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_128x256x256
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 36
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	wave_mxfp4_static_gemm_64x64x256_128x256x256, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_128x256x256

	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.num_vgpr, 36
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.numbered_sgpr, 28
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_128x256x256.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  generic
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_static_gemm_64x64x256_128x256x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_128x256x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
