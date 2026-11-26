// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

// Instances found by tuning (tuning/results_*.cpp), 2D, NHWGC/GKYXC/NHWGK.
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wavelet_xdl_tuned_v1_f16_instances_part1 = std::tuple<
    // clang-format off
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 128, 32, 64, 8, 16, 16, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 128, 1, 2>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 128, 32, 128, 8, 16, 16, 2, 2, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 128, 1, 2>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 512, 256, 32, 64, 8, 16, 16, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 256, 1, 2>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 256, 32, 128, 8, 16, 16, 4, 2, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 128, 1, 2>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 256, 32, 128, 8, 16, 16, 4, 2, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 128, 1, 2>, 8, F16, F16, 1, true, true, true>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 128, 64, 128, 8, 16, 16, 4, 2, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 2, 1, S<1, 64, 1, 4>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 512, 256, 64, 64, 8, 16, 16, 2, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 128, 1, 4>, 8, F16, F16, 1, true, true, false>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wavelet_xdl_tuned_v1_f16_instances_part2 = std::tuple<
    // clang-format off
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 128, 64, 64, 8, 16, 16, 2, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 4>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 128, 64, 128, 8, 16, 16, 2, 4, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 4>, 8, F16, F16, 1, true, true, true>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 256, 64, 128, 8, 16, 16, 4, 4, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 4>, 8, F16, F16, 1, true, true, true>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 512, 128, 128, 64, 8, 16, 16, 2, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 8>, 8, F16, F16, 1, true, true, false>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 256, 256, 128, 64, 8, 16, 16, 4, 8, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 4>, 8, F16, F16, 1, true, true, true>,
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F16, F16, float, F16, Empty_Tuple, F16, PassThrough, PassThrough, PassThrough, ConvSpec, 256, 512, 128, 256, 64, 8, 16, 16, 2, 8, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 2, S<1, 64, 1, 8>, 8, F16, F16, 1, true, true, false>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
