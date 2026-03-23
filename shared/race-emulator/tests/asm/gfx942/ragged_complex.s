	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z21ragged_complex_kernelPiPKi ; -- Begin function _Z21ragged_complex_kernelPiPKi
	.globl	_Z21ragged_complex_kernelPiPKi
	.p2align	8
	.type	_Z21ragged_complex_kernelPiPKi,@function
_Z21ragged_complex_kernelPiPKi:         ; @_Z21ragged_complex_kernelPiPKi
; %bb.0:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v0, 0
	s_waitcnt lgkmcnt(0)
	global_load_dword v3, v2, s[2:3]
	s_waitcnt vmcnt(0)
	v_cmp_lt_i32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_10
; %bb.1:
	s_mov_b32 s10, 1
	v_cmp_ne_u32_e32 vcc, 1, v3
	s_mov_b64 s[6:7], -1
	s_mov_b32 s11, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	v_and_b32_e32 v4, 0x7ffffffe, v3
	s_mov_b64 s[6:7], 0
	v_mov_b32_e32 v5, v4
	s_mov_b32 s8, 0
	s_mov_b32 s9, s11
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_mul_hi_u32 s14, s11, 0xaaaaaaab
	s_mul_hi_u32 s15, s10, 0xaaaaaaab
	s_lshr_b32 s14, s14, 1
	s_lshr_b32 s15, s15, 1
	s_mul_i32 s14, s14, 3
	s_mul_i32 s15, s15, 3
	s_add_i32 s12, s10, 2
	s_add_i32 s13, s11, 2
	s_sub_i32 s14, s11, s14
	s_sub_i32 s15, s10, s15
	v_add_u32_e32 v5, -2, v5
	s_mul_i32 s16, s13, s11
	s_mul_i32 s17, s12, s10
	s_add_i32 s8, s14, s8
	s_add_i32 s9, s15, s9
	v_cmp_eq_u32_e32 vcc, 0, v5
	s_add_i32 s9, s9, s17
	s_add_i32 s8, s8, s16
	s_mov_b32 s11, s13
	s_mov_b32 s10, s12
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[0:1], s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, v3, v4
	v_add_u32_e32 v0, v0, v1
	s_orn2_b64 s[6:7], vcc, exec
.LBB0_5:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_9
; %bb.6:
	s_mov_b64 s[6:7], 0
	s_mov_b32 s8, 0xaaaaaaab
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_mul_hi_u32 v1, v4, s8
	v_add_u32_e32 v5, 2, v4
	v_lshrrev_b32_e32 v1, 1, v1
	v_mul_lo_u32 v5, v5, v4
	v_add_u32_e32 v0, v4, v0
	v_add_u32_e32 v4, 1, v4
	v_mul_lo_u32 v1, v1, -3
	v_cmp_eq_u32_e32 vcc, v3, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	v_add3_u32 v0, v0, v1, v5
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_or_b64 exec, exec, s[6:7]
.LBB0_9:
	s_or_b64 exec, exec, s[4:5]
.LBB0_10:
	s_or_b64 exec, exec, s[2:3]
	global_store_dword v2, v0, s[0:1]
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 18
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
	.size	_Z21ragged_complex_kernelPiPKi, .Lfunc_end0-_Z21ragged_complex_kernelPiPKi
                                        ; -- End function
	.set _Z21ragged_complex_kernelPiPKi.num_vgpr, 6
	.set _Z21ragged_complex_kernelPiPKi.num_agpr, 0
	.set _Z21ragged_complex_kernelPiPKi.numbered_sgpr, 18
	.set _Z21ragged_complex_kernelPiPKi.private_seg_size, 0
	.set _Z21ragged_complex_kernelPiPKi.uses_vcc, 1
	.set _Z21ragged_complex_kernelPiPKi.uses_flat_scratch, 0
	.set _Z21ragged_complex_kernelPiPKi.has_dyn_sized_stack, 0
	.set _Z21ragged_complex_kernelPiPKi.has_recursion, 0
	.set _Z21ragged_complex_kernelPiPKi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 332
; TotalNumSgprs: 24
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 6
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
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
	.type	__hip_cuid_5fcd4ae33b4dda7f,@object ; @__hip_cuid_5fcd4ae33b4dda7f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5fcd4ae33b4dda7f
__hip_cuid_5fcd4ae33b4dda7f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5fcd4ae33b4dda7f, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5fcd4ae33b4dda7f
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
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _Z21ragged_complex_kernelPiPKi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
