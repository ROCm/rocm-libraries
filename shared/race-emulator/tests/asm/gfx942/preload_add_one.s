; Hand-crafted kernel to test kernarg preloading on gfx942.
;
; Takes a single pointer argument (to an array of int32), preloaded into s[2:3].
; Each thread adds 1 to array[threadId].
;
; The key difference from a normal kernel: the pointer is NOT loaded via
; s_load_dword from the kernarg segment. Instead, it's already in s[2:3]
; thanks to .amdhsa_user_sgpr_kernarg_preload_length 2.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	preload_add_one
	.p2align	8
	.type	preload_add_one,@function
preload_add_one:
	s_branch .Lreal_start
	.p2align	8
.Lreal_start:
	; s[0:1] = kernarg segment pointer (set by hardware)
	; s[2:3] = &array (preloaded from kernarg bytes 0..7)
	; s[4]   = workgroup_id_x (system sgpr, after user_sgpr_count=4)
	; v0     = thread_id_x within workgroup

	; Compute byte offset: v1 = threadId * 4
	v_lshlrev_b32_e32 v1, 2, v0

	; Load array[threadId]: global_load_dword v2, v1, s[2:3]
	global_load_dword v2, v1, s[2:3]
	s_waitcnt vmcnt(0)

	; Add 1
	v_add_u32_e32 v2, 1, v2

	; Store back
	global_store_dword v1, v2, s[2:3]
	s_waitcnt vmcnt(0)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel preload_add_one
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 4
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 2
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 5
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
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
	.size	preload_add_one, .Lfunc_end0-preload_add_one

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 64
    .name:           preload_add_one
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .sgpr_spill_count: 0
    .symbol:         preload_add_one.kd
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
