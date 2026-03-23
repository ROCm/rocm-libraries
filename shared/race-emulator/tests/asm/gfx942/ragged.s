	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z12ragged_loopsPiS_    ; -- Begin function _Z12ragged_loopsPiS_
	.globl	_Z12ragged_loopsPiS_
	.p2align	8
	.type	_Z12ragged_loopsPiS_,@function
_Z12ragged_loopsPiS_:                   ; @_Z12ragged_loopsPiS_
; %bb.0:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v6, 2, v0
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v6, s[2:3]
	s_waitcnt vmcnt(0)
	v_cmp_lt_i32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_10
; %bb.1:                                ; %.lr.ph.preheader
	v_cmp_lt_u32_e32 vcc, 3, v1
	s_mov_b64 s[8:9], -1
	s_mov_b32 s6, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:                                ; %vector.ph
	v_and_b32_e32 v0, 0x7ffffffc, v1
	s_add_u32 s12, s0, 8
	s_addc_u32 s13, s1, 0
	s_mov_b64 s[8:9], 0
	v_mov_b32_e32 v7, v0
	s_mov_b32 s7, s6
	s_mov_b32 s10, s6
	s_mov_b32 s11, s6
.LBB0_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	s_add_u32 s14, s12, -8
	s_addc_u32 s15, s13, -1
	s_load_dwordx4 s[16:19], s[14:15], 0x0
	v_add_u32_e32 v7, -4, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_waitcnt lgkmcnt(0)
	s_add_i32 s7, s17, s7
	s_add_i32 s6, s16, s6
	s_add_i32 s11, s19, s11
	s_add_i32 s10, s18, s10
	s_add_u32 s12, s12, 16
	s_addc_u32 s13, s13, 0
	v_mov_b64_e32 v[2:3], s[6:7]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[4:5], s[10:11]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
; %bb.4:                                ; %middle.block
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, v5, v3
	v_cmp_ne_u32_e32 vcc, v1, v0
	v_add3_u32 v2, v4, v2, v3
	s_orn2_b64 s[8:9], vcc, exec
.LBB0_5:                                ; %Flow44
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[4:5], s[8:9]
	s_cbranch_execz .LBB0_9
; %bb.6:                                ; %.lr.ph.preheader17
	v_sub_u32_e32 v3, v1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
	s_mov_b64 s[6:7], 0
.LBB0_7:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	global_load_dword v4, v[0:1], off
	v_add_u32_e32 v3, -1, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_lshl_add_u64 v[0:1], v[0:1], 0, 4
	s_or_b64 s[6:7], vcc, s[6:7]
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v2, v4, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_7
; %bb.8:                                ; %Flow
	s_or_b64 exec, exec, s[6:7]
.LBB0_9:                                ; %Flow45
	s_or_b64 exec, exec, s[4:5]
.LBB0_10:                               ; %._crit_edge
	s_or_b64 exec, exec, s[2:3]
	global_store_dword v6, v2, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12ragged_loopsPiS_
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
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 20
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
	.size	_Z12ragged_loopsPiS_, .Lfunc_end0-_Z12ragged_loopsPiS_
                                        ; -- End function
	.set _Z12ragged_loopsPiS_.num_vgpr, 8
	.set _Z12ragged_loopsPiS_.num_agpr, 0
	.set _Z12ragged_loopsPiS_.numbered_sgpr, 20
	.set _Z12ragged_loopsPiS_.private_seg_size, 0
	.set _Z12ragged_loopsPiS_.uses_vcc, 1
	.set _Z12ragged_loopsPiS_.uses_flat_scratch, 0
	.set _Z12ragged_loopsPiS_.has_dyn_sized_stack, 0
	.set _Z12ragged_loopsPiS_.has_recursion, 0
	.set _Z12ragged_loopsPiS_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 304
; TotalNumSgprs: 26
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 26
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
    .name:           _Z12ragged_loopsPiS_
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         _Z12ragged_loopsPiS_.kd
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
