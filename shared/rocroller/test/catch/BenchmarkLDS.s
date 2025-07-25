.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
.p2align 8
.type GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel,@function
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel:
ds_read_b128 v[8:11], v7
ds_read_b128 v[12:15], v7 offset:256
ds_read_b128 v[16:19], v7 offset:512
ds_read_b128 v[20:23], v7 offset:768
s_endpgm
.LGEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_end:
.size GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel, .LGEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_end-GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
.rodata
.p2align 6
.amdhsa_kernel GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 32768
  .amdhsa_accum_offset 56
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel


.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
    .symbol: GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 44
    .vgpr_count: 56
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size:
      - 256
      - 1
      - 1
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_8, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 56, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_12, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 88, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}

.end_amdgpu_metadata
