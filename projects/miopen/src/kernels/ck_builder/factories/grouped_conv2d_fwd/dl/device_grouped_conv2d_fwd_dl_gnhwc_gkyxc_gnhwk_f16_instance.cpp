// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_dl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::GNHWK,
    ck::half_t,
    ck::half_t,
    ck::Tuple<>,
    ck::half_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto GNHWC = ckb::TensorLayout::GNHWC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto GNHWK = ckb::TensorLayout::GNHWK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    // Operation aliases
    constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;

    add_device_operation_instances<device_grouped_conv2d_fwd_dl_f16_instances<0>(
        GNHWC, GKYXC, {}, GNHWK, {}, PassThrough, ConvFwdDefault)>(instances);

    add_device_operation_instances<device_grouped_conv2d_fwd_dl_f16_instances<0>(
        GNHWC, GKYXC, {}, GNHWK, {}, PassThrough, ConvFwd1x1P0)>(instances);

    add_device_operation_instances<device_grouped_conv2d_fwd_dl_f16_instances<0>(
        GNHWC, GKYXC, {}, GNHWK, {}, PassThrough, ConvFwd1x1S1P0)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
