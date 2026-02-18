// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/instance_data/xdl.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// constexpr auto FP32 = ckb::DataType::FP32;

constexpr auto create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp

    return std::array<XdlInstance, 0>{};

    // TODO - These instances have a a_block_transfer_src_vector_dim value of 1, which is invalid
    // according to the CK Builder constraints
    /*
    // clang-format off
    std::array result = {
        // Instance 1: NumGroupsToMerge = 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 8),

        // Instance 2: NumGroupsToMerge = 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 16),

        // Instance 3: NumGroupsToMerge = 32
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 32)
    };
    // clang-format on

    return result;
    //*/
}

constexpr auto
create_device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance_data()
{
    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
            2,
            ckb::TensorLayout::NHWGC,
            ckb::TensorLayout::GKYXC,
            ckb::TensorLayout::NHWGK,
            ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter3x3InstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
            2,
            ckb::TensorLayout::NHWGC,
            ckb::TensorLayout::GKYXC,
            ckb::TensorLayout::NHWGK,
            ckb::ConvSpecialization::FILTER_3x3);

    constexpr auto instanceData = concat(defaultInstanceData, filter3x3InstanceData);

    return instanceData;
}

void add_f32_merged_groups_instances(
    std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances)
{
    constexpr auto kernelData =
        create_device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance_data();
    build_kernels<kernelData>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
