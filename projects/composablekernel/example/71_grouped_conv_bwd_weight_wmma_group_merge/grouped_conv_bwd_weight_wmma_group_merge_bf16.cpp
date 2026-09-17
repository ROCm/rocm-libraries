// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Correctness/perf validation for extending the existing NumGroupsToMerge
// block-diagonal group-packing mechanism (xor+pad trick in
// TransformConvBwdWeightToGemmV2::make_wei_grid_desc, already wired into
// DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3) to depthwise
// (C_per_group = K_per_group = 1) grouped_conv2d_bwd_weight, where the
// currently-registered NumGroupsToMerge=1 instance pads each group's
// 1x(Y*X) GEMM up to a full MPerBlock x NPerBlock WMMA tile.
//
// NOTE ON SCOPE (see docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md Task 4):
// T2-01's fwd mechanism (a *new* K-axis block-diagonal WMMA pack, with a
// prepass kernel densifying the weight operand) does not port to
// bwd_weight: bwd_weight's GEMM-K axis is N*Ho*Wo (batch/spatial) and
// contains zero group/channel content by construction (weight-gradient
// reduction is over batch and space, never over channels), so there is
// nothing "small-channel" to pack block-diagonally into K. The structurally
// analogous inefficiency in bwd_weight is *M/N* padding waste for small
// per-group K/C (both channel axes appear only in M and N, which share one
// group-independent K-reduction) - and CK already has a proven,
// descriptor-only block-diagonal packing mechanism for exactly that shape
// of problem: `NumGroupsToMerge` (xor+pad trick), already implemented in
// TransformConvBwdWeightToGemmV2 and wired into the TwoStage WMMA device op,
// but so far gated to depthwise (Conv_C_==1 && Conv_K_==1) and never
// exercised with NumGroupsToMerge > 1 by any registered instance. This
// example exercises it for the first time, mirroring the roadmap's
// depthwise wrw shapes (see M06/M14/M15/M25 in miopen_wrw_shapes.txt).
//
// Verifies the device result against the CPU reference implementation.

#include <cstdlib>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using BF16 = ck::bhalf_t;
using F32  = float;

using InDataType  = BF16;
using WeiDataType = BF16;
using OutDataType = BF16;
using AccDataType = F32;

namespace ctl = ck::tensor_layout::convolution;

using InLayout  = ctl::NHWGC;
using WeiLayout = ctl::GKYXC;
using OutLayout = ctl::NHWGK;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr ck::index_t NDimSpatial      = 2;
static constexpr ck::index_t NumGroupsToMerge = 16;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Depthwise (C_per_group = K_per_group = 1) 3x3 conv: per-group GemmM = K_per_group = 1,
// GemmN = C_per_group*Y*X = 9. The currently-registered NumGroupsToMerge=1 instance pads
// both to a 16x16 WMMA tile (256x nominal MAC waste). Packing NumGroupsToMerge=16 groups
// block-diagonally (via the existing xor+pad mechanism, unchanged) fills GemmM = 16*1 = 16
// exactly (matches MPerBlock=16, no M padding) and GemmN = 16*9 = 144 exactly (matches
// NPerBlock=144 = NRepeat(9)*NPerWmma(16), no N padding) - i.e. dense MAC utilization
// instead of ~28x nominal waste (16x in M, 1.78x in N) at NumGroupsToMerge=1. All other
// tile parameters are carried over unchanged from the sole registered NumGroupsToMerge=1
// instance (device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp): the A (output)
// operand and the per-CShuffle-pass epilogue tile are both untouched by NPerBlock's
// growth (the epilogue loops NRepeat times over the same 16x16 pass shape), so no operand
// besides B's N-extent actually needed retuning.
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<
        NDimSpatial,
        InLayout,
        WeiLayout,
        OutLayout,
        InDataType,
        WeiDataType,
        OutDataType,
        AccDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvBwdWeightDefault,
        32,            // BlockSize
        16,            // MPerBlock
        144,           // NPerBlock (= NumGroupsToMerge * C_per_group * Y * X = 16 * 1 * 3 * 3)
        32,            // KPerBlock
        8,             // ABK1
        16,            // MPerWmma
        16,            // NPerWmma
        1,             // MRepeat
        9,             // NRepeat (= NPerBlock / NPerWmma)
        S<4, 8, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<2, 0, 1>,    // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,    // ABlockTransferSrcAccessOrder
        1,             // ABlockTransferSrcVectorDim
        1,             // ABlockTransferSrcScalarPerVector
        4,             // ABlockTransferDstScalarPerVector_AK1
        false,         // ABlockLdsAddExtraM
        S<4, 8, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<2, 0, 1>,    // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,    // BBlockTransferSrcAccessOrder
        1,             // BBlockTransferSrcVectorDim
        1,             // BBlockTransferSrcScalarPerVector
        4,             // BBlockTransferDstScalarPerVector_BK1
        false,         // BBlockLdsAddExtraN
        1,             // CShuffleMRepeatPerShuffle
        1,             // CShuffleNRepeatPerShuffle
        S<1, 4, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        1,             // CShuffleBlockTransferScalarPerVector_NPerBlock
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        NumGroupsToMerge>;

using ReferenceConvBwdWeightInstance =
    ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       InElementOp,
                                                       WeiElementOp,
                                                       OutElementOp>;

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This kernel supports gfx12 only" << std::endl;
        return 0;
    }

    // Depthwise grouped_conv2d_bwd_weight parameters, matching the roadmap's M06 shape
    // (miopen_wrw_shapes.txt: G=192, N=42, C_per_group=K_per_group=1, Y=X=3, Hi=120,
    // Wi=160, stride=2, pad=1) by default. Overridable on the command line as
    // "G split_k Hi Wi stride" (Y=X=3 fixed, matching NPerBlock=144's compile-time
    // sizing) to also cover M14/M15/M25 (g=256/256/512, same C=K=1/Y=X=3 shape family,
    // Hi/Wi/stride varying) so smaller correctness-only runs are fast and every
    // roadmap-cited shape can be exercised. G must stay a multiple of NumGroupsToMerge=16.
    ck::index_t G = 192;
    if(argc >= 2)
        G = std::atoi(argv[1]);

    ck::index_t split_k = 1;
    if(argc >= 3)
        split_k = std::atoi(argv[2]);

    ck::index_t Hi = 120;
    if(argc >= 4)
        Hi = std::atoi(argv[3]);

    ck::index_t Wi = 160;
    if(argc >= 5)
        Wi = std::atoi(argv[4]);

    ck::index_t conv_stride = 2;
    if(argc >= 6)
        conv_stride = std::atoi(argv[5]);

    const ck::index_t N             = 42;
    const ck::index_t C_per_group   = 1;
    const ck::index_t K_per_group   = 1;
    const ck::index_t Y             = 3;
    const ck::index_t X             = 3;
    const ck::index_t conv_dilation = 1;
    const ck::index_t in_pad        = 1;

    if(G % NumGroupsToMerge != 0)
    {
        std::cout << "G must be a multiple of NumGroupsToMerge=" << NumGroupsToMerge << std::endl;
        return 1;
    }

    const ck::index_t Ho = (Hi + 2 * in_pad - conv_dilation * (Y - 1) - 1) / conv_stride + 1;
    const ck::index_t Wo = (Wi + 2 * in_pad - conv_dilation * (X - 1) - 1) / conv_stride + 1;

    // NHWGC: [G, N, C, Hi, Wi] canonical index order, NHWGC physical memory order.
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
    Tensor<WeiDataType> wei_host(wei_desc);
    Tensor<WeiDataType> wei_device(wei_desc);
    Tensor<OutDataType> out(out_desc);

    in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
    out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-1.0, 1.0});

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei_host.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_dev(sizeof(WeiDataType) * wei_device.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());
    out_dev.ToDevice(out.mData.data());
    wei_dev.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> in_lengths{G, N, C_per_group, Hi, Wi};
    std::array<ck::index_t, NDimSpatial + 3> in_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{G, K_per_group, C_per_group, Y, X};
    std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> out_lengths{G, N, K_per_group, Ho, Wo};
    std::array<ck::index_t, NDimSpatial + 3> out_strides{};

    ck::ranges::copy(in_desc.GetStrides(), in_strides.begin());
    ck::ranges::copy(wei_desc.GetStrides(), wei_strides.begin());
    ck::ranges::copy(out_desc.GetStrides(), out_strides.begin());

    std::array<ck::index_t, NDimSpatial> conv_filter_strides{conv_stride, conv_stride};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{conv_dilation, conv_dilation};
    std::array<ck::index_t, NDimSpatial> input_left_pads{in_pad, in_pad};
    std::array<ck::index_t, NDimSpatial> input_right_pads{in_pad, in_pad};

    auto conv = DeviceConvBwdWeightInstance{};

    auto argument = conv.MakeArgument(static_cast<const InDataType*>(in_dev.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_dev.GetDeviceBuffer()),
                                      static_cast<const OutDataType*>(out_dev.GetDeviceBuffer()),
                                      in_lengths,
                                      in_strides,
                                      wei_lengths,
                                      wei_strides,
                                      out_lengths,
                                      out_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{},
                                      split_k);

    DeviceMem workspace_buf(argument.GetWorkspaceSizeBytes());
    conv.SetWorkSpacePointer(&argument, workspace_buf.GetDeviceBuffer());

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this NumGroupsToMerge conv problem"
                  << std::endl;
        return 1;
    }

    auto invoker   = conv.MakeInvoker();
    float avg_time = invoker.Run(argument, StreamConfig{nullptr, true});

    std::size_t flop     = std::size_t(2) * G * N * K_per_group * Ho * Wo * C_per_group * Y * X;
    std::size_t num_byte = sizeof(InDataType) * in.mDesc.GetElementSpaceSize() +
                           sizeof(WeiDataType) * wei_device.mDesc.GetElementSpaceSize() +
                           sizeof(OutDataType) * out.mDesc.GetElementSpaceSize();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_byte / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    wei_dev.FromDevice(wei_device.mData.data());

    auto ref_conv     = ReferenceConvBwdWeightInstance{};
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(
        in,
        wei_host,
        out,
        std::vector<ck::long_index_t>(conv_filter_strides.begin(), conv_filter_strides.end()),
        std::vector<ck::long_index_t>(conv_filter_dilations.begin(), conv_filter_dilations.end()),
        std::vector<ck::long_index_t>(input_left_pads.begin(), input_left_pads.end()),
        std::vector<ck::long_index_t>(input_right_pads.begin(), input_right_pads.end()),
        InElementOp{},
        WeiElementOp{},
        OutElementOp{});
    ref_invoker.Run(ref_argument);

    const ck::index_t num_accums         = N * Ho * Wo;
    const ck::index_t num_accums_split_k = split_k;
    double rtol = ck::utils::get_relative_threshold<InDataType, WeiDataType, AccDataType>(
        num_accums / num_accums_split_k);
    double atol = ck::utils::get_absolute_threshold<InDataType, WeiDataType, AccDataType>(
        *std::max_element(wei_host.mData.begin(), wei_host.mData.end()),
        num_accums / num_accums_split_k);

    return ck::utils::check_err(
               wei_device.mData, wei_host.mData, "Error: incorrect results!", rtol, atol)
               ? 0
               : 1;
}
