// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// F32 instance builder functions
void add_f32_merged_groups_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_standard_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_16x16_instances(std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_large_tensor_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_comp_instances(std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);
void add_f32_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances);

std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>
DeviceOperationInstanceFactory<DeviceOpGFwdDefault<float>>::GetInstances()
{
    // Adapted from GetInstances() in the composable_kernel project's file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>> instances{};

    add_f32_merged_groups_instances(instances);
    add_f32_standard_instances(instances);
    add_f32_16x16_instances(instances);
    add_f32_large_tensor_instances(instances);
    add_f32_comp_instances(instances);
    add_f32_mem_intra_instances(instances);
    add_f32_mem_inter_instances(instances);

    return instances;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
