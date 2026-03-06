// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NGCHW,
    ck::tensor_layout::convolution::GKCYX,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NGKHW,
    float,
    float,
    ck::Tuple<>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[n, g, c, hi, wi] * wei[g, k, c, y, x] = out[n, g, k, ho, wo]
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
    constexpr auto GKCYX = ckb::TensorLayout::GKCYX;
    constexpr auto NGKHW = ckb::TensorLayout::NGKHW;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_instances<0>(
        2, NGCHW, GKCYX, {}, NGKHW, ConvFwdDefault)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_instances<0>(
        2, NGCHW, GKCYX, {}, NGKHW, ConvFwd1x1P0)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_instances<0>(
        2, NGCHW, GKCYX, {}, NGKHW, ConvFwd1x1S1P0)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
