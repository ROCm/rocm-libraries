// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

// Instances found by tuning (tuning/results_*.cpp), 2D, NHWGC/GKYXC/NHWGK.
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_v3_wmma_tuned_v1_f16_instances_part1 = std::tuple<
    // clang-format off
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, float, PassThrough, PassThrough, PassThrough, ConvSpec, 128, 16, 64, 256, 8, 16, 16, 1, 1, S<32, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 1, 8, 1, S<32, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, 1, 1, S<1, 16, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, F16, F16, 1, 1, false, false>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, float, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 16, 128, 512, 8, 16, 16, 1, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 1, 8, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, 1, 1, S<1, 16, 1, 16>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, F16, F16, 1, 1, false, false>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, float, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 64, 64, 512, 8, 16, 16, 2, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, F16, F16, 1, 1, false, false>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, float, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 32, 128, 512, 8, 16, 16, 2, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, S<64, 4, 1>, S<2, 0, 1>, S<2, 0, 1>, 1, 8, 8, 1, 1, 1, S<1, 16, 1, 16>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, F16, F16, 1, 1, false, false>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
