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
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NGKHW,
    float,
    float,
    ck::Tuple<>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[n, g, c, hi, wi] * wei[g, k, y, x, c] = out[n, g, k, ho, wo]
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto NGKHW = ckb::TensorLayout::NGKHW;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_generic_instances<0>(
        2, NGCHW, GKYXC, {}, NGKHW, ConvFwdDefault)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
