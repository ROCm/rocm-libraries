// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Correctness validation for T2-01: block-diagonal WMMA packing (GroupsPerWmma > 1)
// for small-channels-per-group grouped_conv2d_fwd.
// See docs_gfx1250/T2-01_BLOCK_DIAGONAL_WMMA_PACKING_DESIGN.md for the design.
//
// Validation shape (per the design doc's roadmap target):
//   grouped_conv2d_fwd, G=32, N=64, C=128 (C_per_group=4), K=128 (K_per_group=4),
//   Y=X=3, Hi=Wi=28, stride=1, pad=1 (GroupsPerWmma=4 packs 4 groups per WMMA).
//
// Verifies the device result against the CPU reference implementation.

#include <cstdlib>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using F16 = ck::half_t;
using F32 = float;

using InDataType       = F16;
using WeiDataType      = F16;
using OutDataType      = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using DsLayout         = ck::Tuple<>;

using ADataType        = InDataType;
using BDataType        = WeiDataType;
using EDataType        = OutDataType;
using AComputeDataType = InDataType;
using BComputeDataType = WeiDataType;

namespace ctl = ck::tensor_layout::convolution;

using ALayout = ctl::NHWGC;
using BLayout = ctl::GKYXC;
using ELayout = ctl::NHWGK;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

static constexpr ck::index_t NDimSpatial      = 2;
static constexpr ck::index_t GroupsPerWmma    = 4;
static constexpr ck::index_t NumGroupsToMerge = 1;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// GemmN per group-cluster is GroupsPerWmma*K_per_group = 4*4 = 16, matching NPerWmma
// exactly (NPerBlock=16, NRepeat=1): one N-block covers a whole cluster's output
// channels. GemmK per cluster is GroupsPerWmma*Y*X*C_per_group = 4*9*4 = 144 (padded
// to KPerBlock=96 multiples by GemmSpec::MNKPadding, i.e. the "K-padding hit" the
// design doc calls out as an accepted, non-goal-blocking cost for a first pass).
//
// Tuned vs. a naive scalar-vector-width baseline (BlockSize=64/MPerBlock=64/
// NPerBlock=16/KPerBlock=64/*ScalarPerVector=1, 4.56 ms / 0.10 TFlops / 5.6 GB/s on
// gfx1250 at the shape below): A's contiguous run is C_per_group=4 (im2col merge,
// C is innermost), so ABlockTransferSrcScalarPerVector=4 is the max safe width. B
// reads from the GroupsPerWmma prepass's densely-packed scratch buffer (a single
// flat, stride-1 axis, not a multi-dim merge), so BBlockTransferSrcScalarPerVector
// can go up to BK1=8. E's merged N axis is a multi-dim merge (gi, K_per_group) so,
// symmetrically with A, vectorization is capped at the innermost sub-dim's extent
// (K_per_group=4) unless the tensor's group stride is verified contiguous with K_per_group
// -- CDEBlockTransferScalarPerVector_NPerBlock=4 stays inside that per-group bound.
// KPerBlock=96 (2 main-loop iterations covering the 144-wide K) measured faster than
// both smaller (32/64, more loop overhead) and larger (128/160/192, 1-iteration
// tail-only pipeline path) KPerBlock choices. Net: 0.048 ms / 9.68 TFlops / 538 GB/s
// -- ~95x faster, same CPU-reference-verified correctness.
using DeviceConvFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
        NDimSpatial,      // NDimSpatial
        ALayout,          // ALayout
        BLayout,          // BLayout
        DsLayout,         // DsLayout
        ELayout,          // ELayout
        ADataType,        // ADataType
        BDataType,        // BDataType
        AccDataType,      // AccDataType
        CShuffleDataType, // CShuffleDataType
        DsDataType,       // DsDataType
        EDataType,        // EDataType
        InElementOp,      // AElementwiseOperation
        WeiElementOp,     // BElementwiseOperation
        OutElementOp,     // CDEElementwiseOperation
        ConvSpec,         // ConvForwardSpecialization
        GemmSpec,         // GemmSpecialization
        64,               // BlockSize
        64,               // MPerBlock
        16,               // NPerBlock
        96,               // KPerBlock
        8,                // AK1
        8,                // BK1
        16,               // MPerWmma
        16,               // NPerWmma
        2,                // MRepeat
        1,                // NRepeat
        S<4, 16, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
        2,                // ABlockTransferSrcVectorDim
        4,                // ABlockTransferSrcScalarPerVector
        8,                // ABlockTransferDstScalarPerVector_AK1
        1,                // ABlockLdsExtraM
        S<4, 16, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
        2,                // BBlockTransferSrcVectorDim
        8,                // BBlockTransferSrcScalarPerVector
        8,                // BBlockTransferDstScalarPerVector_BK1
        1,                // BBlockLdsExtraN
        1,                // CShuffleMRepeatPerShuffle
        1,                // CShuffleNRepeatPerShuffle
        S<1, 16, 1, 4>,   // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4,                // CDEBlockTransferScalarPerVector_NPerBlock
        ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
        true,                                      // UseThreadTileTransfer
        AComputeDataType,                          // AComputeDataType
        BComputeDataType,                          // BComputeDataType
        NumGroupsToMerge,                          // NumGroupsToMerge
        GroupsPerWmma>;                            // GroupsPerWmma (T2-01)

using ReferenceConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              InElementOp,
                                                                              WeiElementOp,
                                                                              OutElementOp>;

int main()
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This kernel supports gfx12 only" << std::endl;
        return 0;
    }

    // Grouped conv2d fwd parameters (see file header for rationale).
    const ck::index_t G             = 32;
    const ck::index_t N             = 64;
    const ck::index_t C_per_group   = 4;
    const ck::index_t K_per_group   = 4;
    const ck::index_t C             = G * C_per_group;
    const ck::index_t K             = G * K_per_group;
    const ck::index_t Y             = 3;
    const ck::index_t X             = 3;
    const ck::index_t Hi            = 28;
    const ck::index_t Wi            = 28;
    const ck::index_t conv_stride   = 1;
    const ck::index_t conv_dilation = 1;
    const ck::index_t in_pad        = 1;

    const ck::index_t Ho = (Hi + 2 * in_pad - conv_dilation * (Y - 1) - 1) / conv_stride + 1;
    const ck::index_t Wo = (Wi + 2 * in_pad - conv_dilation * (X - 1) - 1) / conv_stride + 1;

    // NHWGC: [G, N, C, Hi, Wi] canonical index order, NHWGC physical memory order
    // (N outer, then Hi, Wi, then G, C innermost).
    HostTensorDescriptor in_desc(
        {G, N, C_per_group, Hi, Wi},
        {C_per_group, Hi * Wi * G * C_per_group, 1, Wi * G * C_per_group, G * C_per_group},
        ck::tensor_layout::BypassLayoutVerification{});

    // GKYXC: [G, K, C, Y, X] canonical index order, packed/contiguous per group.
    HostTensorDescriptor wei_desc(
        {G, K_per_group, C_per_group, Y, X},
        {K_per_group * Y * X * C_per_group, Y * X * C_per_group, 1, X * C_per_group, C_per_group},
        ck::tensor_layout::BypassLayoutVerification{});

    // NHWGK: [G, N, K, Ho, Wo] canonical index order, NHWGK physical memory order.
    HostTensorDescriptor out_desc(
        {G, N, K_per_group, Ho, Wo},
        {K_per_group, Ho * Wo * G * K_per_group, 1, Wo * G * K_per_group, G * K_per_group},
        ck::tensor_layout::BypassLayoutVerification{});

    Tensor<InDataType> in(in_desc);
    Tensor<WeiDataType> wei(wei_desc);
    Tensor<OutDataType> out_host(out_desc);
    Tensor<OutDataType> out_device(out_desc);

    in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
    wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1.0, 1.0});

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_dev(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());
    wei_dev.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_lengths{G, N, C_per_group, Hi, Wi};
    std::array<ck::index_t, NDimSpatial + 3> a_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_lengths{G, K_per_group, C_per_group, Y, X};
    std::array<ck::index_t, NDimSpatial + 3> b_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_lengths{G, N, K_per_group, Ho, Wo};
    std::array<ck::index_t, NDimSpatial + 3> e_strides{};

    ck::ranges::copy(in_desc.GetStrides(), a_strides.begin());
    ck::ranges::copy(wei_desc.GetStrides(), b_strides.begin());
    ck::ranges::copy(out_desc.GetStrides(), e_strides.begin());

    std::array<ck::index_t, NDimSpatial> conv_filter_strides{conv_stride, conv_stride};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{conv_dilation, conv_dilation};
    std::array<ck::index_t, NDimSpatial> input_left_pads{in_pad, in_pad};
    std::array<ck::index_t, NDimSpatial> input_right_pads{in_pad, in_pad};

    auto conv = DeviceConvFwdInstance{};

    auto argument = conv.MakeArgument(in_dev.GetDeviceBuffer(),
                                      wei_dev.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_dev.GetDeviceBuffer(),
                                      a_lengths,
                                      a_strides,
                                      b_lengths,
                                      b_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      e_lengths,
                                      e_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this GroupsPerWmma conv problem"
                  << std::endl;
        return 1;
    }

    DeviceMem workspace_buf(argument.GetWorkspaceSizeBytes());
    conv.SetWorkSpacePointer(&argument, workspace_buf.GetDeviceBuffer());

    auto invoker   = conv.MakeInvoker();
    float avg_time = invoker.Run(argument, StreamConfig{nullptr, true});

    std::size_t flop     = std::size_t(2) * N * K * Ho * Wo * C_per_group * Y * X;
    std::size_t num_byte = sizeof(InDataType) * in.mDesc.GetElementSpaceSize() +
                           sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize() +
                           sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_byte / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    out_dev.FromDevice(out_device.mData.data());

    auto ref_conv     = ReferenceConvFwdInstance{};
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(
        in,
        wei,
        out_host,
        std::vector<ck::long_index_t>(conv_filter_strides.begin(), conv_filter_strides.end()),
        std::vector<ck::long_index_t>(conv_filter_dilations.begin(), conv_filter_dilations.end()),
        std::vector<ck::long_index_t>(input_left_pads.begin(), input_left_pads.end()),
        std::vector<ck::long_index_t>(input_right_pads.begin(), input_right_pads.end()),
        InElementOp{},
        WeiElementOp{},
        OutElementOp{});
    ref_invoker.Run(ref_argument);

    return ck::utils::check_err(out_device, out_host, "Error: incorrect results!", 1e-3, 1e-3) ? 0
                                                                                               : 1;
}
