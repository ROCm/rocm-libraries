// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Task 6 (docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md): DEVICE vs SYSTEM atomic
// scope stress test for wrw split-K's cross-workgroup buffer atomic-add
// (`amd_buffer_addressing.hpp:600-605`, currently DEVICE scope on gfx1250).
//
// Deliberately adversarial shape: G=1, K_per_group=C_per_group=32 (exactly one
// MPerBlock x NPerBlock tile - GemmM=GemmN=32 matches the smallest registered
// one-stage instance exactly, so every split-K workgroup atomic-adds into the
// *same* 32x32 destination range) and a large, CLI-overridable split_k so many
// concurrent workgroups race on the same weight-tensor addresses. This
// exercises `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3`'s
// `InMemoryDataOperationEnum::AtomicAdd` path directly (KBatch > 1), unlike
// `DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` (used elsewhere in this
// investigation), which never uses atomics at all - it writes each split's
// partial sum to a distinct scratch slice and reduces without a race.
//
// Run in a loop (see the doc's Task 6 validation recipe): a correctness
// failure that appears only sometimes is the signature this test is built to
// catch (atomic races are inherently non-deterministic).

#include <cstdlib>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
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

static constexpr ck::index_t NDimSpatial = 2;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// The smallest registered one-stage bf16 instance
// (device_grouped_conv_bwd_weight_v3_wmma_instance.hpp row 1), reused verbatim -
// this test targets the atomic-add split-K path, not a new tile config.
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
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
        64,            // BlockSize
        32,            // MPerBlock
        32,            // NPerBlock
        32,            // KPerBlock
        8,             // ABK1
        16,            // MPerWmma
        16,            // NPerWmma
        2,             // MRepeat
        1,             // NRepeat
        S<4, 8, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<2, 0, 1>,    // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,    // ABlockTransferSrcAccessOrder
        1,             // ABlockTransferSrcVectorDim
        2,             // ABlockTransferSrcScalarPerVector
        2,             // ABlockTransferDstScalarPerVector_AK1
        false,         // ABlockLdsAddExtraM
        S<4, 16, 1>,   // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<2, 0, 1>,    // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,    // BBlockTransferSrcAccessOrder
        1,             // BBlockTransferSrcVectorDim
        2,             // BBlockTransferSrcScalarPerVector
        2,             // BBlockTransferDstScalarPerVector_BK1
        false,         // BBlockLdsAddExtraN
        1,             // CShuffleMRepeatPerShuffle
        1,             // CShuffleNRepeatPerShuffle
        S<1, 8, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        2>;            // CShuffleBlockTransferScalarPerVector_NPerBlock

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

    ck::index_t split_k = 1024;
    if(argc >= 2)
        split_k = std::atoi(argv[1]);

    // GemmM = K_per_group (= MPerBlock=32 by default: one M-tile), GemmN =
    // C_per_group*Y*X = 32 (= NPerBlock): one M/N tile by default, so every
    // split-K workgroup atomic-adds into the same destination range - maximal
    // contention for a fixed split_k. K_per_group overridable (arg 2) to spread
    // across more M-tiles for the "is this specific to a single-tile grid?"
    // diagnostic.
    const ck::index_t G           = 1;
    const ck::index_t N           = 42;
    const ck::index_t C_per_group = 32;
    ck::index_t K_per_group       = 32;
    if(argc >= 3)
        K_per_group = std::atoi(argv[2]);
    const ck::index_t Y             = 1;
    const ck::index_t X             = 1;
    const ck::index_t Hi            = 64;
    const ck::index_t Wi            = 64;
    const ck::index_t conv_stride   = 1;
    const ck::index_t conv_dilation = 1;
    const ck::index_t in_pad        = 0;

    const ck::index_t Ho = (Hi + 2 * in_pad - conv_dilation * (Y - 1) - 1) / conv_stride + 1;
    const ck::index_t Wo = (Wi + 2 * in_pad - conv_dilation * (X - 1) - 1) / conv_stride + 1;

    HostTensorDescriptor in_desc(
        {G, N, C_per_group, Hi, Wi},
        {C_per_group, Hi * Wi * G * C_per_group, 1, Wi * G * C_per_group, G * C_per_group},
        ck::tensor_layout::BypassLayoutVerification{});

    HostTensorDescriptor wei_desc(
        {G, K_per_group, C_per_group, Y, X},
        {K_per_group * Y * X * C_per_group, Y * X * C_per_group, 1, X * C_per_group, C_per_group},
        ck::tensor_layout::BypassLayoutVerification{});

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

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this problem"
                  << std::endl;
        return 1;
    }

    auto invoker = conv.MakeInvoker();
    invoker.Run(argument, StreamConfig{nullptr, false});

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

    bool pass = ck::utils::check_err(
        wei_device.mData, wei_host.mData, "Error: incorrect results!", rtol, atol);
    std::cout << (pass ? "PASS" : "FAIL") << " split_k=" << split_k << std::endl;
    return pass ? 0 : 1;
}
