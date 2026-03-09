// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGK,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::Tuple<>,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto NHWGK = ckb::TensorLayout::NHWGK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;
    constexpr auto ConvFwdOddC    = ckb::ConvSpecialization::ODD_C;

    add_device_operation_instances<device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwdDefault)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwd1x1P0)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwd1x1S1P0)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwdOddC)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
