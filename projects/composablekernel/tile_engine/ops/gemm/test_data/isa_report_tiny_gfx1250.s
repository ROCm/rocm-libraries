; Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
; SPDX-License-Identifier: MIT
;
; Hand-written fixture for test_isa_report.py, laid out like clang -save-temps
; output for gfx1250. It is only parsed, never assembled. tiny_gemm_kernel has
; a prologue, a two-block K loop (one back edge), an epilogue and an
; out-of-line block that branches back into the loop header (not a back edge).
; The epilogue scratch reload must not be reported as a store, and the
; gfx10/11-style two-operand wait keeps its count.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	tiny_gemm_kernel                ; -- Begin function tiny_gemm_kernel
	.p2align	8
	.type	tiny_gemm_kernel,@function
tiny_gemm_kernel:                       ; @tiny_gemm_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_set_vgpr_msb 0x40
	tensor_load_to_lds s[4:7], s[8:15]
	s_cbranch_scc1 .LBB0_4
.LBB0_1:                                ; %k_loop
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[0:3], v100
	ds_load_b128 v[4:7], v100 offset:64
	global_load_async_to_lds_b128 v101, v[102:103], off
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[8:15], v[0:7], v[16:23], v[8:15]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[8:15], v[0:7], v[16:23], v[8:15]
	s_set_vgpr_msb 0
	tensor_load_to_lds s[4:7], s[8:15]
	s_cbranch_execz .LBB0_3
; %bb.2:                                ;   in Loop: Header=BB0_1 Depth=1
	scratch_store_b32 off, v24, s32
.LBB0_3:                                ;   in Loop: Header=BB0_1 Depth=1
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB0_1
; %bb.5:                                ; %epilogue
	s_wait_asynccnt 0x0
	scratch_load_b32 v24, off, s32
	s_waitcnt_vscnt null, 0x0
	v_cvt_pk_bf16_f32 v0, v8, v9
	ds_store_b64 v100, v[0:1]
	s_barrier_signal -1
	s_barrier_wait -1
	global_store_b128 v[102:103], v[0:3], off
	global_store_b128 v[102:103], v[4:7], off offset:16
	s_endpgm
.LBB0_4:                                ; %cold
	s_mov_b32 s2, 4
	s_branch .LBB0_1
.Lfunc_end0:
	.size	tiny_gemm_kernel, .Lfunc_end0-tiny_gemm_kernel
                                        ; -- End function
	.globl	tiny_helper                     ; -- Begin function tiny_helper
	.p2align	2
	.type	tiny_helper,@function
tiny_helper:                            ; @tiny_helper
; %bb.0:
	v_mov_b32_e32 v0, 0
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	tiny_helper, .Lfunc_end1-tiny_helper
                                        ; -- End function
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 139264
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           tiny_gemm_kernel
    .private_segment_fixed_size: 16
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         tiny_gemm_kernel.kd
    .vgpr_count:     128
    .vgpr_spill_count: 3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
