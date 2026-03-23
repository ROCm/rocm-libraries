	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z13ragged_kernelPiPKi  ; -- Begin function _Z13ragged_kernelPiPKi
	.globl	_Z13ragged_kernelPiPKi
	.p2align	8
	.type	_Z13ragged_kernelPiPKi,@function
_Z13ragged_kernelPiPKi:                 ; @_Z13ragged_kernelPiPKi
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_load_b32 v2, v0, s[2:3]
	s_mov_b32 s2, exec_lo
	s_waitcnt vmcnt(0)
	v_cmpx_lt_i32_e32 0, v2
; %bb.1:
	v_add_nc_u32_e32 v3, -1, v2
	v_add_nc_u32_e32 v1, -2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[1:2], null, v3, v1, 0
	v_alignbit_b32 v1, v2, v1, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v3, v1
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s2
	global_store_b32 v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13ragged_kernelPiPKi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
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
	.size	_Z13ragged_kernelPiPKi, .Lfunc_end0-_Z13ragged_kernelPiPKi
                                        ; -- End function
	.set _Z13ragged_kernelPiPKi.num_vgpr, 4
	.set _Z13ragged_kernelPiPKi.num_agpr, 0
	.set _Z13ragged_kernelPiPKi.numbered_sgpr, 4
	.set _Z13ragged_kernelPiPKi.private_seg_size, 0
	.set _Z13ragged_kernelPiPKi.uses_vcc, 0
	.set _Z13ragged_kernelPiPKi.uses_flat_scratch, 0
	.set _Z13ragged_kernelPiPKi.has_dyn_sized_stack, 0
	.set _Z13ragged_kernelPiPKi.has_recursion, 0
	.set _Z13ragged_kernelPiPKi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 92
; TotalNumSgprs: 4
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 4
; NumVGPRsForWavesPerEU: 4
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_773283413608c278,@object ; @__hip_cuid_773283413608c278
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_773283413608c278
__hip_cuid_773283413608c278:
	.byte	0                               ; 0x0
	.size	__hip_cuid_773283413608c278, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_773283413608c278
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z13ragged_kernelPiPKi
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .sgpr_spill_count: 0
    .symbol:         _Z13ragged_kernelPiPKi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
