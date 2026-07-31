// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

using InDataType       = ck::bhalf_t;
using WeiDataType      = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using OutDataType      = ck::bhalf_t;
using AComputeDataType = InDataType;
using BComputeDataType = AComputeDataType;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial,
    InLayout,
    WeiLayout,
    ck::Tuple<>,
    OutLayout,
    InDataType,
    WeiDataType,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<>,
    OutDataType,
    InElementOp,
    WeiElementOp,
    OutElementOp,
    ConvSpec,
    GemmSpec,
    1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 4>, 8>;

#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[]) {
    return run_convnd_fwd_example(argc, argv) ? 0 : 1;
}
