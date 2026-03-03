// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp>
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
    ck::bhalf_t,
    ck::bhalf_t,
    ck::Tuple<>,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
    constexpr auto GKCYX = ckb::TensorLayout::GKCYX;
    constexpr auto NGKHW = ckb::TensorLayout::NGKHW;

    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;

    constexpr auto Intrawave = ckb::PipelineScheduler::INTRAWAVE;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_mem_instances<0>(
        2, NGCHW, GKCYX, {}, NGKHW, ConvFwdDefault, Intrawave)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
