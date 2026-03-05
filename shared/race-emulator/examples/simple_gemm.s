	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	SimpleGemm              ; -- Begin function SimpleGemm
	.globl	SimpleGemm
	.p2align	8
	.type	SimpleGemm,@function
SimpleGemm:                             ; @SimpleGemm
; %bb.0:
	s_load_dword s3, s[0:1], 0x44
	s_load_dwordx4 s[12:15], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	s_mul_i32 s2, s15, s14
	v_cmp_gt_i32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_load_dword s2, s[0:1], 0x30
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_4
; %bb.2:
	s_abs_i32 s3, s14
	v_cvt_f32_u32_e32 v2, s3
	v_sub_u32_e32 v3, 0, v0
	s_sub_i32 s1, 0, s3
	v_max_i32_e32 v4, v0, v3
	v_rcp_iflag_f32_e32 v2, v2
	s_ashr_i32 s0, s14, 31
	v_xor_b32_e32 v5, s0, v1
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_mul_lo_u32 v3, s1, v2
	v_mul_hi_u32 v3, v2, v3
	v_add_u32_e32 v2, v2, v3
	v_mad_u64_u32 v[2:3], s[0:1], v4, v2, 0
	v_mul_lo_u32 v2, v3, s3
	v_sub_u32_e32 v2, v4, v2
	v_add_u32_e32 v6, 1, v3
	v_cmp_le_u32_e32 vcc, s3, v2
	v_subrev_u32_e32 v4, s3, v2
	s_nop 0
	v_cndmask_b32_e32 v3, v3, v6, vcc
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_add_u32_e32 v4, 1, v3
	v_cmp_le_u32_e32 vcc, s3, v2
	s_nop 1
	v_cndmask_b32_e32 v2, v3, v4, vcc
	v_xor_b32_e32 v2, v2, v5
	v_sub_u32_e32 v3, v2, v5
	v_mul_lo_u32 v6, v3, s2
	v_mul_lo_u32 v2, v3, s14
	v_mov_b32_e32 v4, s8
	v_mov_b32_e32 v5, s9
	v_ashrrev_i32_e32 v7, 31, v6
	v_sub_u32_e32 v2, v0, v2
	v_lshl_add_u64 v[4:5], v[6:7], 2, v[4:5]
	v_mov_b32_e32 v6, 0
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[8:9], v[2:3], 2, s[6:7]
	global_load_dword v7, v[4:5], off
	global_load_dword v3, v[8:9], off
	s_add_i32 s2, s2, -1
	v_lshl_add_u64 v[4:5], v[4:5], 0, 4
	v_add_u32_e32 v2, s14, v2
	s_cmp_eq_u32 s2, 0
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v6, v3, v7
	s_cbranch_scc0 .LBB0_3
	s_branch .LBB0_5
.LBB0_4:
	v_mov_b32_e32 v6, 0
.LBB0_5:
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_lshl_add_u64 v[2:3], s[10:11], 0, v[0:1]
	global_load_dword v2, v[2:3], off
	v_lshl_add_u64 v[0:1], s[4:5], 0, v[0:1]
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v2, s13, v2
	v_fmac_f32_e32 v2, s12, v6
	global_store_dword v[0:1], v2, off
.LBB0_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel SimpleGemm
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 312
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
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 12
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
	.size	SimpleGemm, .Lfunc_end0-SimpleGemm
                                        ; -- End function
	.set SimpleGemm.num_vgpr, 10
	.set SimpleGemm.num_agpr, 0
	.set SimpleGemm.numbered_sgpr, 16
	.set SimpleGemm.private_seg_size, 0
	.set SimpleGemm.uses_vcc, 1
	.set SimpleGemm.uses_flat_scratch, 0
	.set SimpleGemm.has_dyn_sized_stack, 0
	.set SimpleGemm.has_recursion, 0
	.set SimpleGemm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 384
; TotalNumSgprs: 22
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 10
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_21d9980975232445,@object ; @__hip_cuid_21d9980975232445
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_21d9980975232445
__hip_cuid_21d9980975232445:
	.byte	0                               ; 0x0
	.size	__hip_cuid_21d9980975232445, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_21d9980975232445
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         60
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         68
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         70
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         72
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         74
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         76
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         78
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         120
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 312
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           SimpleGemm
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         SimpleGemm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
