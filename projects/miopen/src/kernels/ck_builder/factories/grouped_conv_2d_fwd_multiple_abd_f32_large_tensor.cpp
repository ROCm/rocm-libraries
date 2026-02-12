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

constexpr auto create_device_grouped_conv_fwd_xdl_f32_large_tensor_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp

    return std::array<XdlInstance, 0>{};

    // TODO - c_block_transfer_scalar_per_vector = 8 is a valid parameter but isn't accepted by
    // builder API
    // clang-format off
    /*
    std::array result = {
        // Instance 1: Large tensor optimized for big spatial dimensions
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 2, 32, 32, 4, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32),
        
        // Instance 2: Large tensor with different tile configuration
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 32, 8, 2, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 16}, 8,
            FP32, FP32),
        
        // Instance 3: Optimized for very large batch sizes
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 4}, 8,
            FP32, FP32),
        
        // Instance 4: Large spatial with high channel count
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 256, 256, 32, 8, 2, 32, 32, 8, 4,
            {2, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {2, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 16}, 8,
            FP32, FP32),
        
        // Instance 5: Balanced large tensor configuration
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 2, 32, 32, 8, 8,
            {2, 128, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {2, 128, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32),
        
        // Instance 6: Large tensor for extreme dimensions
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 512, 32, 8, 2, 32, 32, 8, 16,
            {2, 128, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {2, 128, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 16}, 8,
            FP32, FP32)

        // clang-format on
    };

    return result;
    */
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance_data()
{
    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_large_tensor_instance_data(2,
                                                                         ckb::TensorLayout::NHWGC,
                                                                         ckb::TensorLayout::GKYXC,
                                                                         ckb::TensorLayout::NHWGK,
                                                                         ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData = create_device_grouped_conv_fwd_xdl_f32_large_tensor_instance_data(
        2,
        ckb::TensorLayout::NHWGC,
        ckb::TensorLayout::GKYXC,
        ckb::TensorLayout::NHWGK,
        ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f32_large_tensor_instance_data(
            2,
            ckb::TensorLayout::NHWGC,
            ckb::TensorLayout::GKYXC,
            ckb::TensorLayout::NHWGK,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto instanceData =
        concat(defaultInstanceData, filter1x1Pad0InstanceData, filter1x1Stride1Pad0InstanceData);

    return instanceData;
}

void add_f32_large_tensor_instances(std::vector<std::unique_ptr<DeviceOpGFwdDefault<float>>>& instances)
{
    constexpr auto kernelData =
        create_device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance_data();
    build_kernels<kernelData>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
