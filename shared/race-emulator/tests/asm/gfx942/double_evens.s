	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z12double_evensPff     ; -- Begin function _Z12double_evensPff
	.globl	_Z12double_evensPff
	.p2align	8
	.type	_Z12double_evensPff,@function
_Z12double_evensPff:                    ; @_Z12double_evensPff
; %bb.0:
	s_load_dword s3, s[0:1], 0x1c
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dword s6, s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s3, 0xffff
	s_mul_i32 s2, s2, s0
	v_add_lshl_u32 v0, s2, v0, 5
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[4:5]
	global_load_dword v2, v[0:1], off
	global_load_dword v3, v[0:1], off offset:8
	global_load_dword v4, v[0:1], off offset:16
	global_load_dword v5, v[0:1], off offset:24
	global_load_dword v6, v[0:1], off offset:32
	global_load_dword v7, v[0:1], off offset:40
	global_load_dword v8, v[0:1], off offset:48
	global_load_dword v9, v[0:1], off offset:56
	global_load_dword v10, v[0:1], off offset:64
	global_load_dword v11, v[0:1], off offset:72
	global_load_dword v12, v[0:1], off offset:80
	global_load_dword v13, v[0:1], off offset:88
	global_load_dword v14, v[0:1], off offset:96
	global_load_dword v15, v[0:1], off offset:104
	global_load_dword v16, v[0:1], off offset:112
	global_load_dword v17, v[0:1], off offset:120
	s_waitcnt vmcnt(15)
	v_mul_f32_e32 v2, s6, v2
	s_waitcnt vmcnt(14)
	v_mul_f32_e32 v3, s6, v3
	s_waitcnt vmcnt(13)
	v_mul_f32_e32 v4, s6, v4
	s_waitcnt vmcnt(12)
	v_mul_f32_e32 v5, s6, v5
	s_waitcnt vmcnt(11)
	v_mul_f32_e32 v6, s6, v6
	s_waitcnt vmcnt(10)
	v_mul_f32_e32 v7, s6, v7
	s_waitcnt vmcnt(9)
	v_mul_f32_e32 v8, s6, v8
	s_waitcnt vmcnt(8)
	v_mul_f32_e32 v9, s6, v9
	s_waitcnt vmcnt(7)
	v_mul_f32_e32 v10, s6, v10
	s_waitcnt vmcnt(6)
	v_mul_f32_e32 v11, s6, v11
	s_waitcnt vmcnt(5)
	v_mul_f32_e32 v12, s6, v12
	s_waitcnt vmcnt(4)
	v_mul_f32_e32 v13, s6, v13
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v14, s6, v14
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v15, s6, v15
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v16, s6, v16
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v17, s6, v17
	global_store_dword v[0:1], v2, off
	global_store_dword v[0:1], v3, off offset:8
	global_store_dword v[0:1], v4, off offset:16
	global_store_dword v[0:1], v5, off offset:24
	global_store_dword v[0:1], v6, off offset:32
	global_store_dword v[0:1], v7, off offset:40
	global_store_dword v[0:1], v8, off offset:48
	global_store_dword v[0:1], v9, off offset:56
	global_store_dword v[0:1], v10, off offset:64
	global_store_dword v[0:1], v11, off offset:72
	global_store_dword v[0:1], v12, off offset:80
	global_store_dword v[0:1], v13, off offset:88
	global_store_dword v[0:1], v14, off offset:96
	global_store_dword v[0:1], v15, off offset:104
	global_store_dword v[0:1], v16, off offset:112
	global_store_dword v[0:1], v17, off offset:120
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12double_evensPff
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 18
		.amdhsa_next_free_sgpr 7
		.amdhsa_accum_offset 20
		.amdhsa_reserve_vcc 0
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
	.size	_Z12double_evensPff, .Lfunc_end0-_Z12double_evensPff
                                        ; -- End function
	.set _Z12double_evensPff.num_vgpr, 18
	.set _Z12double_evensPff.num_agpr, 0
	.set _Z12double_evensPff.numbered_sgpr, 7
	.set _Z12double_evensPff.private_seg_size, 0
	.set _Z12double_evensPff.uses_vcc, 0
	.set _Z12double_evensPff.uses_flat_scratch, 0
	.set _Z12double_evensPff.has_dyn_sized_stack, 0
	.set _Z12double_evensPff.has_recursion, 0
	.set _Z12double_evensPff.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 448
; TotalNumSgprs: 13
; NumVgprs: 18
; NumAgprs: 0
; TotalNumVgprs: 18
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 13
; NumVGPRsForWavesPerEU: 18
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_9d4c6f91a68c64fd,@object ; @__hip_cuid_9d4c6f91a68c64fd
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9d4c6f91a68c64fd
__hip_cuid_9d4c6f91a68c64fd:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9d4c6f91a68c64fd, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9d4c6f91a68c64fd
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12double_evensPff
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .sgpr_spill_count: 0
    .symbol:         _Z12double_evensPff.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
