// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/instance_data/xdl.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

constexpr auto FP32 = ckb::DataType::FP32;

constexpr auto
create_device_grouped_conv_fwd_xdl_f32_instance_data(std::size_t spatialDim,
                                                     ckb::TensorLayout inLayout,
                                                     ckb::TensorLayout weiLayout,
                                                     ckb::TensorLayout outLayout,
                                                     ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp

    // clang-format off
    std::array result = {
        // Instance 1: Generic instance
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 2: Small conv.K and conv.C
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 4
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 5
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 16, 4, 4, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 6
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 7
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 9
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 10
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 11
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 64, 16, 4, 4, 32, 32, 2, 1,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 12
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 13
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 14
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 15
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 32, 64, 16, 4, 4, 32, 32, 1, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 17
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 192, 16, 4, 4, 32, 32, 2, 3,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32)

        // clang-format on
    };

    return result;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance_data()
{
    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_instance_data(2,
                                                             ckb::TensorLayout::NHWGC,
                                                             ckb::TensorLayout::GKYXC,
                                                             ckb::TensorLayout::NHWGK,
                                                             ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData = create_device_grouped_conv_fwd_xdl_f32_instance_data(
        2,
        ckb::TensorLayout::NHWGC,
        ckb::TensorLayout::GKYXC,
        ckb::TensorLayout::NHWGK,
        ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f32_instance_data(
            2,
            ckb::TensorLayout::NHWGC,
            ckb::TensorLayout::GKYXC,
            ckb::TensorLayout::NHWGK,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto oddCInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_instance_data(2,
                                                             ckb::TensorLayout::NHWGC,
                                                             ckb::TensorLayout::GKYXC,
                                                             ckb::TensorLayout::NHWGK,
                                                             ckb::ConvSpecialization::ODD_C);

    constexpr auto instanceData = concat(defaultInstanceData,
                                         filter1x1Pad0InstanceData,
                                         filter1x1Stride1Pad0InstanceData,
                                         oddCInstanceData);

    return instanceData;
}

void add_f32_standard_instances(std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances)
{
    constexpr auto kernelData =
        create_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance_data();
    build_kernels<kernelData>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
