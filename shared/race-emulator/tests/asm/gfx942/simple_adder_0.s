.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._Z12adder_kernelILi256EEvfPKdPfPKf,"axG",@progbits,_Z12adder_kernelILi256EEvfPKdPfPKf,comdat
	.protected	_Z12adder_kernelILi256EEvfPKdPfPKf ; -- Begin function _Z12adder_kernelILi256EEvfPKdPfPKf
	.globl	_Z12adder_kernelILi256EEvfPKdPfPKf
	.p2align	8
	.type	_Z12adder_kernelILi256EEvfPKdPfPKf,@function
_Z12adder_kernelILi256EEvfPKdPfPKf:     ; @_Z12adder_kernelILi256EEvfPKdPfPKf
; %bb.0:
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x18
	v_lshl_or_b32 v0, s2, 8, v0
	v_mov_b32_e32 v1, 0
	s_load_dword s0, s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s4
	v_mov_b32_e32 v3, s5
	v_lshl_add_u64 v[2:3], v[0:1], 3, v[2:3]
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_lshl_add_u64 v[4:5], s[8:9], 0, v[0:1]
	global_load_dword v8, v[4:5], off
	global_load_dwordx2 v[6:7], v[2:3], off
	v_cvt_f64_f32_e32 v[2:3], s0
  ; print f32 v8 0
  ; print f64 v6 0
  ; print f32 s0 0
  ; print f64 v2 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_waitcnt vmcnt(1)
	v_cvt_f64_f32_e32 v[4:5], v8
	s_waitcnt vmcnt(0)
  v_add_f64 v[4:5], v[6:7], v[4:5]
	v_add_f64 v[2:3], v[4:5], v[2:3]
	v_cvt_f32_f64_e32 v2, v[2:3]
	global_store_dword v[0:1], v2, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12adder_kernelILi256EEvfPKdPfPKf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 12
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
	.section	.text._Z12adder_kernelILi256EEvfPKdPfPKf,"axG",@progbits,_Z12adder_kernelILi256EEvfPKdPfPKf,comdat
.Lfunc_end0:
	.size	_Z12adder_kernelILi256EEvfPKdPfPKf, .Lfunc_end0-_Z12adder_kernelILi256EEvfPKdPfPKf
                                        ; -- End function
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.num_vgpr, 9
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.num_agpr, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.numbered_sgpr, 10
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.private_seg_size, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.uses_vcc, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.uses_flat_scratch, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.has_dyn_sized_stack, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.has_recursion, 0
	.set _Z12adder_kernelILi256EEvfPKdPfPKf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
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
      - .offset:         0
        .size:           4
        .value_kind:     by_value
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z12adder_kernelILi256EEvfPKdPfPKf
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z12adder_kernelILi256EEvfPKdPfPKf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     9
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
