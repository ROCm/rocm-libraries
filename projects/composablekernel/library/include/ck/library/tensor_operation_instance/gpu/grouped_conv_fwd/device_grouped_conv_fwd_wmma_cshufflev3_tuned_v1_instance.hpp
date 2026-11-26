// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

// Instances found by tuning (tuning/results_*.cpp), 2D, NHWGC/GKYXC/NHWGK.
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wmma_cshufflev3_tuned_v1_f16_instances_part1 = std::tuple<
    // clang-format off
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 128, 64, 32, 32, 8, 8, 16, 16, 2, 1, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, 1, 1, S<1, 32, 1, 4>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 256, 64, 64, 32, 8, 8, 16, 16, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, 1, 1, S<1, 64, 1, 4>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 256, 128, 64, 32, 8, 8, 16, 16, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, 1, 1, S<1, 64, 1, 4>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 64, 64, 64, 32, 8, 8, 16, 16, 4, 2, S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, 1, 1, S<1, 16, 1, 4>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 128, 128, 64, 32, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 1, 1, 1, S<1, 32, 1, 4>, 8, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 128, 64, 16, 64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 8, 1, 16>, 1, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, GemmMNKPadding, 64, 64, 16, 64, 8, 8, 16, 16, 2, 1, S<8, 8, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 8, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 4, 1, 16>, 1, BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
