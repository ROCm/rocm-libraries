	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z13lds_transposePiPKiii ; -- Begin function _Z13lds_transposePiPKiii
	.globl	_Z13lds_transposePiPKiii
	.p2align	8
	.type	_Z13lds_transposePiPKiii,@function
_Z13lds_transposePiPKiii:               ; @_Z13lds_transposePiPKiii
; %bb.0:
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v2, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s3, v2
	v_cmp_gt_i32_e64 s[0:1], s2, v0
	s_and_b64 s[8:9], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mov_b32_e32 v4, s6
	v_mov_b32_e32 v5, s7
	v_mad_u64_u32 v[6:7], s[6:7], s3, v0, v[2:3]
	v_mov_b32_e32 v7, 0
	v_lshl_add_u64 v[4:5], v[6:7], 2, v[4:5]
	global_load_dword v1, v[4:5], off
	v_lshl_add_u32 v3, v6, 2, 0
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v1
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_i32_e32 vcc, s2, v2
	v_cmp_gt_i32_e64 s[0:1], s3, v0
	s_and_b64 s[0:1], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_mad_u64_u32 v[6:7], s[0:1], s3, v2, v[0:1]
	v_lshl_add_u32 v1, v6, 2, 0
	ds_read_b32 v3, v1
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[0:1], s[0:1], s2, v0, v[2:3]
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], v[0:1], 2, v[4:5]
	global_store_dword v[0:1], v3, off
.LBB0_4:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13lds_transposePiPKiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
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
	.size	_Z13lds_transposePiPKiii, .Lfunc_end0-_Z13lds_transposePiPKiii
                                        ; -- End function
	.set _Z13lds_transposePiPKiii.num_vgpr, 8
	.set _Z13lds_transposePiPKiii.num_agpr, 0
	.set _Z13lds_transposePiPKiii.numbered_sgpr, 10
	.set _Z13lds_transposePiPKiii.private_seg_size, 0
	.set _Z13lds_transposePiPKiii.uses_vcc, 1
	.set _Z13lds_transposePiPKiii.uses_flat_scratch, 0
	.set _Z13lds_transposePiPKiii.has_dyn_sized_stack, 0
	.set _Z13lds_transposePiPKiii.has_recursion, 0
	.set _Z13lds_transposePiPKiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 220
; TotalNumSgprs: 16
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 8
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_ebc37234484d2e21,@object ; @__hip_cuid_ebc37234484d2e21
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ebc37234484d2e21
__hip_cuid_ebc37234484d2e21:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ebc37234484d2e21, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ebc37234484d2e21
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z13lds_transposePiPKiii
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z13lds_transposePiPKiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
