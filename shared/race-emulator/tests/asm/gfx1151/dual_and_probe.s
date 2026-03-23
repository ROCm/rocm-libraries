; dual_and_probe.s
;
; Directly tests whether v_dual_and_b32 executes correctly.
; Mirrors IREE kernel line 95:
;   v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v138, 0xf0, v108
; where v108 = threadId * 16 (i.e. v_lshlrev_b32_e32 v108, 4, v0).
;
; Expected per lane:
;   out[0] = threadId
;   out[1] = v138 = (threadId * 16) & 0xf0
;              lane 0 → 0x00, lane 1 → 0x10, lane 2 → 0x20, ...
;              lane 8 → 0x80, lane 9 → 0x90, lane 15 → 0xf0
;              lane 16 → 0x00 (wraps: (16*16=256) & 0xf0 = 0)
;   out[2] = v3 (should equal v1 = 0, from the v_dual_mov_b32 side)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.globl	dual_and_probe
	.p2align	8
	.type	dual_and_probe,@function
dual_and_probe:
	s_load_b64 s[2:3], s[0:1], 0x0		; output pointer

	; Set up inputs exactly as in IREE kernel
	v_lshlrev_b32_e32 v108, 4, v0		; v108 = threadId * 16  (IREE line 27)
	v_mov_b32_e32 v1, 0			; v1 = 0  (IREE line 90)

	; The dual-issue instruction under test (IREE line 95):
	v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v138, 0xf0, v108

	s_waitcnt lgkmcnt(0)

	; Write 3 dwords per lane at byte offset threadId * 12
	v_mul_u32_u24_e32 v10, 12, v0
	global_store_b32 v10, v0, s[2:3]		; [0] threadId
	v_add_nc_u32_e32 v11, 4, v10
	global_store_b32 v11, v138, s[2:3]		; [1] v138 = (threadId*16) & 0xf0
	v_add_nc_u32_e32 v11, 8, v10
	global_store_b32 v11, v3, s[2:3]		; [2] v3 (from v_dual_mov side)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dual_and_probe
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
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
		.amdhsa_next_free_vgpr 139		; v108 and v138 are high-numbered
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 0
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
.Lfunc_end4:
	.size	dual_and_probe, .Lfunc_end4-dual_and_probe
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           dual_and_probe
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .sgpr_spill_count: 0
    .symbol:         dual_and_probe.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     139
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
