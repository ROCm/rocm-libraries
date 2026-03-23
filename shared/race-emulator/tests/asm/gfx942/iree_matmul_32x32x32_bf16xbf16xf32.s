; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32
	.p2align	8
	.type	matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32,@function
matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v2, 28, v2
	s_mov_b64 s[8:9], s[2:3]
	v_lshrrev_b32_e32 v1, 3, v0
	v_lshlrev_b32_e32 v3, 1, v2
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_movk_i32 s10, 0x800
	v_lshl_or_b32 v3, v1, 6, v3
	s_mov_b64 s[0:1], s[6:7]
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, s10
	s_mov_b32 s7, s11
	buffer_load_dwordx2 v[6:7], v3, s[8:11], 0 offen
	buffer_load_dwordx2 v[10:11], v3, s[4:7], 0 offen
	v_and_b32_e32 v3, 15, v0
	v_lshrrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v5, 16, v1
	v_mul_u32_u24_e32 v1, 36, v1
	v_and_b32_e32 v4, 12, v0
	v_add_lshl_u32 v1, v1, v2, 1
	v_or_b32_e32 v2, v5, v3
	v_and_or_b32 v0, v0, 16, v3
	v_mul_u32_u24_e32 v3, 0x48, v4
	v_mul_u32_u24_e32 v2, 0x48, v2
	v_or_b32_e32 v8, v5, v4
	v_lshlrev_b32_e32 v9, 2, v0
	v_lshl_add_u32 v5, v0, 1, v3
	v_lshl_add_u32 v0, v4, 1, v2
	v_add_u32_e32 v0, 0x800, v0
	s_mov_b32 s4, 0x5040100
	s_movk_i32 s2, 0x1000
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, s11
	v_lshl_or_b32 v8, v8, 7, v9
	s_waitcnt vmcnt(1)
	ds_write_b64 v1, v[6:7] offset:2304
	s_waitcnt vmcnt(0)
	ds_write_b64 v1, v[10:11]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[0:3], v0 offset0:32 offset1:36
	ds_read_u16 v4, v5
	ds_read_u16 v6, v5 offset:72
	ds_read_u16 v7, v5 offset:144
	ds_read_u16 v10, v5 offset:216
	ds_read_u16 v11, v5 offset:1152
	ds_read_u16 v12, v5 offset:1224
	ds_read_u16 v13, v5 offset:1296
	ds_read_u16 v14, v5 offset:1368
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v5, v10, v7, s4
	v_perm_b32 v4, v6, v4, s4
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[0:1], v[4:5], 0
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v1, v14, v13, s4
	v_perm_b32 v0, v12, v11, s4
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[2:3], v[0:1], v[4:7]
	s_nop 6
	buffer_store_dword v0, v8, s[0:3], 0 offen
	buffer_store_dword v1, v8, s[0:3], 0 offen offset:128
	buffer_store_dword v2, v8, s[0:3], 0 offen offset:256
	buffer_store_dword v3, v8, s[0:3], 0 offen offset:384
	s_barrier
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32
		.amdhsa_group_segment_fixed_size 4608
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 6
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 15
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 16
		.amdhsa_reserve_vcc 0
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
	.size	matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32, .Lfunc_end0-matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32

	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.num_vgpr, 15
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.num_agpr, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.numbered_sgpr, 12
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.num_named_barrier, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.private_seg_size, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.uses_vcc, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.uses_flat_scratch, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.has_dyn_sized_stack, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.has_recursion, 0
	.set matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.has_indirect_call, 0
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
    .group_segment_fixed_size: 4608
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         matmul_dispatch_0_matmul_32x32x32_bf16xbf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     15
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
