	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z21ragged_complex_kernelPiPKi ; -- Begin function _Z21ragged_complex_kernelPiPKi
	.globl	_Z21ragged_complex_kernelPiPKi
	.p2align	8
	.type	_Z21ragged_complex_kernelPiPKi,@function
_Z21ragged_complex_kernelPiPKi:         ; @_Z21ragged_complex_kernelPiPKi
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	global_load_b32 v1, v0, s[2:3]
	s_mov_b32 s3, 0
	s_mov_b32 s2, exec_lo
	s_waitcnt vmcnt(0)
	v_cmpx_lt_i32_e32 0, v1
	s_cbranch_execz .LBB0_4
; %bb.1:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_mul_hi_u32 s6, s4, 0xaaaaaaab
	s_add_i32 s7, s4, 2
	s_lshr_b32 s6, s6, 1
	s_add_i32 s5, s4, s5
	s_mul_i32 s6, s6, -3
	s_mul_i32 s7, s7, s4
	s_add_i32 s4, s4, 1
	s_add_i32 s5, s5, s6
	v_cmp_eq_u32_e32 vcc_lo, s4, v1
	s_add_i32 s5, s5, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b32_e32 v2, s5
	s_or_b32 s3, vcc_lo, s3
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execnz .LBB0_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s2
	global_store_b32 v0, v2, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21ragged_complex_kernelPiPKi
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
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
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
	.size	_Z21ragged_complex_kernelPiPKi, .Lfunc_end0-_Z21ragged_complex_kernelPiPKi
                                        ; -- End function
	.set _Z21ragged_complex_kernelPiPKi.num_vgpr, 3
	.set _Z21ragged_complex_kernelPiPKi.num_agpr, 0
	.set _Z21ragged_complex_kernelPiPKi.numbered_sgpr, 8
	.set _Z21ragged_complex_kernelPiPKi.private_seg_size, 0
	.set _Z21ragged_complex_kernelPiPKi.uses_vcc, 1
	.set _Z21ragged_complex_kernelPiPKi.uses_flat_scratch, 0
	.set _Z21ragged_complex_kernelPiPKi.has_dyn_sized_stack, 0
	.set _Z21ragged_complex_kernelPiPKi.has_recursion, 0
	.set _Z21ragged_complex_kernelPiPKi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
; TotalNumSgprs: 10
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 3
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
	.type	__hip_cuid_d8c15c6d0d28042,@object ; @__hip_cuid_d8c15c6d0d28042
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d8c15c6d0d28042
__hip_cuid_d8c15c6d0d28042:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d8c15c6d0d28042, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d8c15c6d0d28042
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
    .name:           _Z21ragged_complex_kernelPiPKi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z21ragged_complex_kernelPiPKi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
