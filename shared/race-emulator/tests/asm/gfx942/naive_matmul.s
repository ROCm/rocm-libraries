	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z12naive_matmulPfPKfS1_iii ; -- Begin function _Z12naive_matmulPfPKfS1_iii
	.globl	_Z12naive_matmulPfPKfS1_iii
	.p2align	8
	.type	_Z12naive_matmulPfPKfS1_iii,@function
_Z12naive_matmulPfPKfS1_iii:            ; @_Z12naive_matmulPfPKfS1_iii
; %bb.0:
	s_load_dwordx4 s[4:7], s[0:1], 0x18
	v_and_b32_e32 v2, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s5, v2
	v_cmp_gt_i32_e64 s[2:3], s4, v3
	s_and_b64 s[2:3], s[2:3], vcc
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_5
; %bb.1:
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	v_mad_u64_u32 v[4:5], s[2:3], s5, v3, v[2:3]
	v_mov_b32_e32 v5, 0
	s_cmp_lt_i32 s6, 1
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s8
	v_mov_b32_e32 v1, s9
	v_lshl_add_u64 v[0:1], v[4:5], 2, v[0:1]
	global_load_dword v6, v[0:1], off
	s_cbranch_scc1 .LBB0_4
; %bb.2:
	s_load_dwordx2 s[0:1], s[0:1], 0x10
	v_lshlrev_b32_e32 v4, 2, v2
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v9, s11
	v_mul_lo_u32 v10, s6, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[4:5]
	s_ashr_i32 s1, s5, 31
	s_mov_b32 s0, s5
	v_mov_b32_e32 v11, v5
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshl_add_u64 v[4:5], v[10:11], 2, v[8:9]
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	global_load_dword v7, v[4:5], off
	global_load_dword v8, v[2:3], off
	s_add_i32 s6, s6, -1
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	v_lshl_add_u64 v[4:5], v[4:5], 0, 4
	s_cmp_eq_u32 s6, 0
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v6, v7, v8
	s_cbranch_scc0 .LBB0_3
.LBB0_4:
	s_waitcnt vmcnt(0)
	global_store_dword v[0:1], v6, off
.LBB0_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12naive_matmulPfPKfS1_iii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
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
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 12
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
	.size	_Z12naive_matmulPfPKfS1_iii, .Lfunc_end0-_Z12naive_matmulPfPKfS1_iii
                                        ; -- End function
	.set _Z12naive_matmulPfPKfS1_iii.num_vgpr, 12
	.set _Z12naive_matmulPfPKfS1_iii.num_agpr, 0
	.set _Z12naive_matmulPfPKfS1_iii.numbered_sgpr, 12
	.set _Z12naive_matmulPfPKfS1_iii.private_seg_size, 0
	.set _Z12naive_matmulPfPKfS1_iii.uses_vcc, 1
	.set _Z12naive_matmulPfPKfS1_iii.uses_flat_scratch, 0
	.set _Z12naive_matmulPfPKfS1_iii.has_dyn_sized_stack, 0
	.set _Z12naive_matmulPfPKfS1_iii.has_recursion, 0
	.set _Z12naive_matmulPfPKfS1_iii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 240
; TotalNumSgprs: 18
; NumVgprs: 12
; NumAgprs: 0
; TotalNumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 12
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
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
	.type	__hip_cuid_5f9ea9c32e404402,@object ; @__hip_cuid_5f9ea9c32e404402
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5f9ea9c32e404402
__hip_cuid_5f9ea9c32e404402:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5f9ea9c32e404402, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5f9ea9c32e404402
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12naive_matmulPfPKfS1_iii
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z12naive_matmulPfPKfS1_iii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
