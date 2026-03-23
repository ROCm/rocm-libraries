	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z13lds_transposePiPKiii ; -- Begin function _Z13lds_transposePiPKiii
	.globl	_Z13lds_transposePiPKiii
	.p2align	8
	.type	_Z13lds_transposePiPKiii,@function
_Z13lds_transposePiPKiii:               ; @_Z13lds_transposePiPKiii
; %bb.0:
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x10
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s3, v1
	v_cmp_gt_i32_e64 s0, s2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s1, s0, vcc_lo
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mad_u64_u32 v[2:3], null, s3, v0, v[1:2]
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64 v[3:4], 2, v[2:3]
	v_lshl_add_u32 v2, v2, 2, 0
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e32 v4, vcc_lo, s7, v4, vcc_lo
	global_load_b32 v3, v[3:4], off
	s_waitcnt vmcnt(0)
	ds_store_b32 v2, v3
.LBB0_2:
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	v_cmp_gt_i32_e64 s0, s3, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_mad_u64_u32 v[2:3], null, s3, v1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v2, v2, 2, 0
	v_mad_u64_u32 v[0:1], null, s2, v0, v[1:2]
	ds_load_b32 v2, v2
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v1, vcc_lo, s5, v1, vcc_lo
	s_waitcnt lgkmcnt(0)
	global_store_b32 v[0:1], v2, off
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
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 5
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
	.size	_Z13lds_transposePiPKiii, .Lfunc_end0-_Z13lds_transposePiPKiii
                                        ; -- End function
	.set _Z13lds_transposePiPKiii.num_vgpr, 5
	.set _Z13lds_transposePiPKiii.num_agpr, 0
	.set _Z13lds_transposePiPKiii.numbered_sgpr, 8
	.set _Z13lds_transposePiPKiii.private_seg_size, 0
	.set _Z13lds_transposePiPKiii.uses_vcc, 1
	.set _Z13lds_transposePiPKiii.uses_flat_scratch, 0
	.set _Z13lds_transposePiPKiii.has_dyn_sized_stack, 0
	.set _Z13lds_transposePiPKiii.has_recursion, 0
	.set _Z13lds_transposePiPKiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 272
; TotalNumSgprs: 10
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 5
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_f5db3cb68153c0b6,@object ; @__hip_cuid_f5db3cb68153c0b6
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f5db3cb68153c0b6
__hip_cuid_f5db3cb68153c0b6:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f5db3cb68153c0b6, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f5db3cb68153c0b6
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
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z13lds_transposePiPKiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
