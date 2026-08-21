// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HSA kernel descriptor + metadata for asan_repro_kernel (see
// kernel_body.s). Kept in a separate file so it never gets fed through
// stinkytofu-opt's raw-.s parser -- see kernel_body.s's header comment for
// why. tests/CMakeLists.txt concatenates this onto stinkytofu-opt's
// (stripped) output before assembling with amdclang++.

.rodata
.p2align 6
.amdhsa_kernel asan_repro_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .offset: 8
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 32
    .name: asan_repro_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 14
    .symbol: asan_repro_kernel.kd
    .vgpr_count: 16
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 1
.end_amdgpu_metadata
