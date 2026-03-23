; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr='+wavefrontsize32' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_dispatch_0_matmul_16x16x16_f16xf16xf32
	.p2align	8
	.type	matmul_dispatch_0_matmul_16x16x16_f16xf16xf32,@function
matmul_dispatch_0_matmul_16x16x16_f16xf16xf32:
	s_load_b128 s[8:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v10, 1, v0
	s_mov_b32 s7, 0x31027000
	s_movk_i32 s6, 0x200
	s_mov_b32 s15, s7
	v_and_b32_e32 v9, 8, v1
	s_mov_b32 s14, s6
	v_lshlrev_b32_e32 v12, 1, v0
	v_and_b32_e32 v11, 15, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 1, v9
	s_movk_i32 s2, 0x400
	s_mov_b32 s3, s7
	v_lshlrev_b32_e32 v17, 1, v11
	v_lshlrev_b32_e32 v0, 2, v0
	v_lshl_or_b32 v5, v10, 5, v1
	v_mul_u32_u24_e32 v10, 20, v10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s9, 0xffff
	s_mov_b32 s4, s8
	s_and_b32 s13, s11, 0xffff
	s_mov_b32 s12, s10
	buffer_load_b128 v[1:4], v5, s[4:7], 0 offen
	buffer_load_b128 v[5:8], v5, s[12:15], 0 offen
	v_add_lshl_u32 v9, v10, v9, 1
	v_or_b32_e32 v10, 0xa0, v12
	v_or_b32_e32 v12, 0x1e0, v12
	s_waitcnt vmcnt(1)
	ds_store_2addr_b64 v9, v[1:2], v[3:4] offset0:80 offset1:81
	s_waitcnt vmcnt(0)
	ds_store_2addr_b64 v9, v[5:6], v[7:8] offset1:1
	v_mul_u32_u24_e32 v9, 40, v11
	s_and_b32 s1, s1, 0xffff
	s_waitcnt lgkmcnt(0)
	ds_load_u16_d16 v3, v10
	ds_load_u16_d16 v7, v12
	ds_load_u16_d16 v1, v17
	ds_load_u16_d16 v2, v17 offset:80
	ds_load_u16_d16 v5, v17 offset:320
	ds_load_u16_d16 v4, v17 offset:240
	ds_load_u16_d16 v8, v17 offset:560
	ds_load_u16_d16 v6, v17 offset:400
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v3, v17 offset:200
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v7, v17 offset:520
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v1, v17 offset:40
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v2, v17 offset:120
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v5, v17 offset:360
	ds_load_2addr_b64 v[13:16], v9 offset0:82 offset1:83
	ds_load_2addr_b64 v[9:12], v9 offset0:80 offset1:81
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v4, v17 offset:280
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v8, v17 offset:600
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v6, v17 offset:440
	v_mov_b32_e32 v17, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v21, v17
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_mov_b32_e32 v24, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[9:16], v[1:8], v[17:24]
	s_clause 0x7
	buffer_store_b32 v17, v0, s[0:3], 0 offen
	buffer_store_b32 v18, v0, s[0:3], 0 offen offset:128
	buffer_store_b32 v19, v0, s[0:3], 0 offen offset:256
	buffer_store_b32 v20, v0, s[0:3], 0 offen offset:384
	buffer_store_b32 v21, v0, s[0:3], 0 offen offset:512
	buffer_store_b32 v22, v0, s[0:3], 0 offen offset:640
	buffer_store_b32 v23, v0, s[0:3], 0 offen offset:768
	buffer_store_b32 v24, v0, s[0:3], 0 offen offset:896
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_dispatch_0_matmul_16x16x16_f16xf16xf32
		.amdhsa_group_segment_fixed_size 1280
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 4
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
	.size	matmul_dispatch_0_matmul_16x16x16_f16xf16xf32, .Lfunc_end0-matmul_dispatch_0_matmul_16x16x16_f16xf16xf32

	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.num_vgpr, 25
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.num_agpr, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.numbered_sgpr, 16
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.num_named_barrier, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.private_seg_size, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.uses_vcc, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.uses_flat_scratch, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.has_dyn_sized_stack, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.has_recursion, 0
	.set matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1280
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name:           matmul_dispatch_0_matmul_16x16x16_f16xf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         matmul_dispatch_0_matmul_16x16x16_f16xf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     25
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
