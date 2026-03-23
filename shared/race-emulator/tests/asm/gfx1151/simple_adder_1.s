	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z12simple_adderP12hip_bfloat16fi ; -- Begin function _Z12simple_adderP12hip_bfloat16fi
	.globl	_Z12simple_adderP12hip_bfloat16fi
	.p2align	8
	.type	_Z12simple_adderP12hip_bfloat16fi,@function
_Z12simple_adderP12hip_bfloat16fi:      ; @_Z12simple_adderP12hip_bfloat16fi
; %bb.0:
	s_clause 0x1
	s_load_b32 s3, s[0:1], 0x1c
	s_load_b64 s[4:5], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[0:1], null, s2, s3, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_i32_e64 s5, v0
	s_cbranch_execz .LBB0_10
; %bb.1:
	s_and_b32 s2, s4, 0x7f800000
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s2, 0x7f800000
	s_cbranch_scc1 .LBB0_3
; %bb.2:
	s_bfe_u32 s2, s4, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_i32 s2, s4, s2
	s_addk_i32 s2, 0x7fff
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_5
.LBB0_3:
                                        ; implicit-def: $sgpr2
.LBB0_4:
	s_and_b32 s2, s4, 0xffff
	s_or_b32 s3, s4, 0x10000
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, s4, s3
.LBB0_5:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	s_and_b32 s0, s2, 0xffff0000
	global_load_u16 v2, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v2
	v_add_f32_e32 v2, s0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v3
                                        ; implicit-def: $vgpr3
	s_and_saveexec_b32 s0, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s0
; %bb.6:
	v_bfe_u32 v3, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v3, v2, v3, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.7:
	s_and_not1_saveexec_b32 s0, s0
; %bb.8:
	v_and_b32_e32 v3, 0xffff, v2
	v_or_b32_e32 v4, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	v_cndmask_b32_e32 v3, v4, v2, vcc_lo
; %bb.9:
	s_or_b32 exec_lo, exec_lo, s0
	global_store_d16_hi_b16 v[0:1], v3, off
.LBB0_10:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12simple_adderP12hip_bfloat16fi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 6
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
	.size	_Z12simple_adderP12hip_bfloat16fi, .Lfunc_end0-_Z12simple_adderP12hip_bfloat16fi
                                        ; -- End function
	.set _Z12simple_adderP12hip_bfloat16fi.num_vgpr, 5
	.set _Z12simple_adderP12hip_bfloat16fi.num_agpr, 0
	.set _Z12simple_adderP12hip_bfloat16fi.numbered_sgpr, 6
	.set _Z12simple_adderP12hip_bfloat16fi.private_seg_size, 0
	.set _Z12simple_adderP12hip_bfloat16fi.uses_vcc, 1
	.set _Z12simple_adderP12hip_bfloat16fi.uses_flat_scratch, 0
	.set _Z12simple_adderP12hip_bfloat16fi.has_dyn_sized_stack, 0
	.set _Z12simple_adderP12hip_bfloat16fi.has_recursion, 0
	.set _Z12simple_adderP12hip_bfloat16fi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 312
; TotalNumSgprs: 8
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 5
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
	.type	__hip_cuid_dc767cfbdc4b2fae,@object ; @__hip_cuid_dc767cfbdc4b2fae
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_dc767cfbdc4b2fae
__hip_cuid_dc767cfbdc4b2fae:
	.byte	0                               ; 0x0
	.size	__hip_cuid_dc767cfbdc4b2fae, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_dc767cfbdc4b2fae
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
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
    .name:           _Z12simple_adderP12hip_bfloat16fi
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         _Z12simple_adderP12hip_bfloat16fi.kd
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
