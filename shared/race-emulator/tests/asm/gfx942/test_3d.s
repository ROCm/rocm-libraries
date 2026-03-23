	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z14test_3d_kernelPf    ; -- Begin function _Z14test_3d_kernelPf
	.globl	_Z14test_3d_kernelPf
	.p2align	8
	.type	_Z14test_3d_kernelPf,@function
_Z14test_3d_kernelPf:                   ; @_Z14test_3d_kernelPf
; %bb.0:
	v_bfe_u32 v1, v0, 10, 10
	v_bfe_u32 v2, v0, 20, 10
	v_lshlrev_b32_e32 v1, 1, v1
	v_lshlrev_b32_e32 v2, 3, v2
	v_and_b32_e32 v0, 0x3ff, v0
	v_add3_u32 v0, v1, v0, v2
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_add_u32_e32 v1, 1, v0
	v_cvt_f32_u32_e32 v1, v1
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14test_3d_kernelPf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
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
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
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
	.size	_Z14test_3d_kernelPf, .Lfunc_end0-_Z14test_3d_kernelPf
                                        ; -- End function
	.set _Z14test_3d_kernelPf.num_vgpr, 3
	.set _Z14test_3d_kernelPf.num_agpr, 0
	.set _Z14test_3d_kernelPf.numbered_sgpr, 2
	.set _Z14test_3d_kernelPf.private_seg_size, 0
	.set _Z14test_3d_kernelPf.uses_vcc, 0
	.set _Z14test_3d_kernelPf.uses_flat_scratch, 0
	.set _Z14test_3d_kernelPf.has_dyn_sized_stack, 0
	.set _Z14test_3d_kernelPf.has_recursion, 0
	.set _Z14test_3d_kernelPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 76
; TotalNumSgprs: 8
; NumVgprs: 3
; NumAgprs: 0
; TotalNumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 3
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_4dc857bf44881d74,@object ; @__hip_cuid_4dc857bf44881d74
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4dc857bf44881d74
__hip_cuid_4dc857bf44881d74:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4dc857bf44881d74, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4dc857bf44881d74
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z14test_3d_kernelPf
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         _Z14test_3d_kernelPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
