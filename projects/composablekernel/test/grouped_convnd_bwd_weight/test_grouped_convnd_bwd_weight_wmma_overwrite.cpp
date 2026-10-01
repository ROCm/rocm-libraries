// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NDimSpatial>
using InLayout = std::conditional_t<NDimSpatial == 2, NHWGC, NDHWGC>;
template <ck::index_t NDimSpatial>
using WeiLayout = std::conditional_t<NDimSpatial == 2, GKYXC, GKZYXC>;
template <ck::index_t NDimSpatial>
using OutLayout = std::conditional_t<NDimSpatial == 2, NHWGK, NDHWGK>;

template <ck::index_t NDimSpatial, typename DataType>
using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NDimSpatial,
                                                                          InLayout<NDimSpatial>,
                                                                          WeiLayout<NDimSpatial>,
                                                                          OutLayout<NDimSpatial>,
                                                                          DataType,
                                                                          DataType,
                                                                          DataType,
                                                                          PassThrough,
                                                                          PassThrough,
                                                                          PassThrough>;

enum class Candidate
{
    DirectScalar,
    TwoStageScalar,
    Depthwise
};

bool IsCandidate(const std::string& name, Candidate candidate)
{
    if(candidate == Candidate::Depthwise)
    {
        return name.compare(0, 40, "DeviceGroupedConvBwdWeightDepthwiseBf16<") == 0;
    }
    const std::string direct = "DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<32, 16, 16, 32, Default";
    const std::string two_stage =
        "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 16, 32, Default";
    const std::string suffix =
        candidate == Candidate::DirectScalar ? "_Split1" : "BlkGemmPipelineVersion: v1, 1>";
    const auto& prefix = candidate == Candidate::DirectScalar ? direct : two_stage;
    return name.compare(0, prefix.size(), prefix) == 0 && name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template <ck::index_t NDimSpatial, typename DataType>
void CheckCandidate(const ck::utils::conv::ConvParam& param,
                    Candidate candidate,
                    ck::index_t split,
                    bool expected_support)
{
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout<NDimSpatial>>(
            param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout<NDimSpatial>>(
            param);
    const auto out_desc = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
        OutLayout<NDimSpatial>>(param);

    ck::Tensor<DataType> in(in_desc);
    ck::Tensor<DataType> out(out_desc);
    ck::Tensor<DataType> expected(wei_desc);
    ck::Tensor<DataType> actual(wei_desc);

    // Distinct signs and spatial patterns make neighboring groups' gradients different.
    // Small integral products accumulate exactly in FP32 (including the split-2 path).
    in.ForEach([](auto& tensor, const auto& index) {
        const auto g = index[0], n = index[1], c = index[2];
        std::size_t spatial = 0;
        for(ck::index_t d = 0; d < NDimSpatial; ++d)
        {
            spatial += (d + 1) * index[3 + d];
        }
        const float sign = g % 4 == 0 ? -1.f : 1.f;
        tensor(index)    = ck::type_convert<DataType>(sign * (1.f + (g + n + c + spatial) % 3));
    });
    out.ForEach([](auto& tensor, const auto& index) {
        const auto g = index[0], n = index[1], k = index[2];
        std::size_t spatial = 0;
        for(ck::index_t d = 0; d < NDimSpatial; ++d)
        {
            spatial += (d + 2) * index[3 + d];
        }
        const float sign = g % 3 == 0 ? -1.f : 1.f;
        tensor(index)    = ck::type_convert<DataType>(sign * (1.f + (2 * g + n + k + spatial) % 3));
    });

    ck::DeviceMem in_device(sizeof(DataType) * in.mDesc.GetElementSpaceSize());
    ck::DeviceMem out_device(sizeof(DataType) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(DataType) * actual.mDesc.GetElementSpaceSize());
    in_device.ToDevice(in.mData.data());
    out_device.ToDevice(out.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> in_lengths{}, in_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{}, wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> out_lengths{}, out_strides{};
    std::array<ck::index_t, NDimSpatial> filter_strides{}, filter_dilations{};
    std::array<ck::index_t, NDimSpatial> left_pads{}, right_pads{};
    auto copy = [](const auto& src, auto& dst) { ck::ranges::copy(src, dst.begin()); };
    copy(in_desc.GetLengths(), in_lengths);
    copy(in_desc.GetStrides(), in_strides);
    copy(wei_desc.GetLengths(), wei_lengths);
    copy(wei_desc.GetStrides(), wei_strides);
    copy(out_desc.GetLengths(), out_lengths);
    copy(out_desc.GetStrides(), out_strides);
    copy(param.conv_filter_strides_, filter_strides);
    copy(param.conv_filter_dilations_, filter_dilations);
    copy(param.input_left_pads_, left_pads);
    copy(param.input_right_pads_, right_pads);

    // Use the same channel-last registrations as the factory without pulling in
    // unrelated explicit-GEMM libraries or the whole device_conv_operations umbrella.
    std::vector<std::unique_ptr<DeviceOp<NDimSpatial, DataType>>> instances;
    if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
#ifdef CK_ENABLE_FP16
        if constexpr(NDimSpatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_f16_instances(
                    instances);
            ck::tensor_operation::device::instance::
                add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_f16_pipev1_instances(
                    instances);
        }
        else
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                    instances);
            ck::tensor_operation::device::instance::
                add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_f16_pipev1_instances(
                    instances);
        }
#endif
    }
    else
    {
#ifdef CK_ENABLE_BF16
        if constexpr(NDimSpatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_bf16_instances(
                    instances);
            if(candidate == Candidate::Depthwise)
            {
                ck::tensor_operation::device::instance::
                    add_device_grouped_conv2d_bwd_weight_depthwise_nhwgc_gkyxc_nhwgk_bf16_instances(
                        instances);
            }
            ck::tensor_operation::device::instance::
                add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_bf16_pipev1_instances(
                    instances);
        }
        else
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    instances);
            ck::tensor_operation::device::instance::
                add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_instances(
                    instances);
        }
#endif
    }
    int matches = 0;
    for(const auto& op : instances)
    {
        const std::string name = op->GetTypeString();
        if(!IsCandidate(name, candidate))
        {
            continue;
        }
        ++matches;
        auto arg                   = op->MakeArgumentPointer(in_device.GetDeviceBuffer(),
                                           wei_device.GetDeviceBuffer(),
                                           out_device.GetDeviceBuffer(),
                                           in_lengths,
                                           in_strides,
                                           wei_lengths,
                                           wei_strides,
                                           out_lengths,
                                           out_strides,
                                           filter_strides,
                                           filter_dilations,
                                           left_pads,
                                           right_pads,
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{},
                                           split);
        const auto workspace_bytes = op->GetWorkSpaceSize(arg.get());
        ck::DeviceMem workspace(workspace_bytes);
        if(workspace_bytes != 0)
        {
            op->SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
        }
        const bool supported = op->IsSupportedArgument(arg.get());
        EXPECT_EQ(supported, expected_support) << name << " G=" << param.G_ << " split=" << split;
        if(candidate == Candidate::Depthwise)
        {
            auto dry_arg = op->MakeArgumentPointer(nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   in_lengths,
                                                   in_strides,
                                                   wei_lengths,
                                                   wei_strides,
                                                   out_lengths,
                                                   out_strides,
                                                   filter_strides,
                                                   filter_dilations,
                                                   left_pads,
                                                   right_pads,
                                                   PassThrough{},
                                                   PassThrough{},
                                                   PassThrough{},
                                                   split);
            EXPECT_EQ(op->IsSupportedArgument(dry_arg.get()), expected_support);
            if(expected_support)
            {
                EXPECT_THROW(
                    op->MakeInvokerPointer()->Run(dry_arg.get(), StreamConfig{nullptr, false}),
                    std::runtime_error);
            }
        }
        if(!supported || !expected_support)
        {
            if(candidate == Candidate::DirectScalar && !supported)
                EXPECT_THROW(op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, false}),
                             std::runtime_error);
            continue;
        }
        if(candidate == Candidate::TwoStageScalar)
        {
            ASSERT_GT(workspace_bytes, 0u) << name << " must use its FP32 workspace";
        }

        using Reference = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                             DataType,
                                                                             DataType,
                                                                             DataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>;
        auto ref_arg    = Reference::MakeArgument(in,
                                               expected,
                                               out,
                                               param.conv_filter_strides_,
                                               param.conv_filter_dilations_,
                                               param.input_left_pads_,
                                               param.input_right_pads_,
                                               PassThrough{},
                                               PassThrough{},
                                               PassThrough{});
        Reference::MakeInvoker().Run(ref_arg);
        auto invoker = op->MakeInvokerPointer();
        // Dirty both buffers before EVERY invocation, not just the first. All-bits-one
        // produces NaN for both FP32 workspace lanes and FP16/BF16 output lanes.
        const std::vector<std::uint8_t> dirty_workspace(workspace_bytes, 0xff);
        for(int repetition = 0; repetition < 2; ++repetition)
        {
            std::fill(actual.mData.begin(),
                      actual.mData.end(),
                      ck::type_convert<DataType>(repetition == 0 ? 113.f : -87.f));
            wei_device.ToDevice(actual.mData.data());
            if(workspace_bytes != 0)
            {
                workspace.ToDevice(dirty_workspace.data());
            }
            invoker->Run(arg.get(), StreamConfig{nullptr, false});
            wei_device.FromDevice(actual.mData.data());

            std::size_t checked = 0;
            expected.ForEach([&](const auto& tensor, const auto& index) {
                const float want = ck::type_convert<float>(tensor(index));
                const float got  = ck::type_convert<float>(actual(index));
                ++checked;
                EXPECT_TRUE(std::isfinite(got)) << name << " group=" << index[0];
                EXPECT_EQ(got, want)
                    << name << " group=" << index[0] << " k=" << index[1] << " c=" << index[2]
                    << " split=" << split << " repetition=" << repetition;
            });
            EXPECT_EQ(checked, expected.mDesc.GetElementSize());
        }
    }
    EXPECT_EQ(matches, 1) << "Expected exactly one registered candidate for G=" << param.G_
                          << " split=" << split;
}

const ck::utils::conv::ConvParam odd_channels{
    2, 3, 2, 5, 3, {3, 3}, {5, 6}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};

const ck::utils::conv::ConvParam odd_channels_3d{
    3, 2, 1, 5, 3, {1, 1, 1}, {2, 3, 6}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};

// G>=128 gives at least 144 filter CTAs. R<=min(G,512) limits each
// reduction lane to at most 16 terms; the G127 and R>G controls stay excluded.
const ck::utils::conv::ConvParam depthwise_g127_r121{
    2, 127, 1, 1, 1, {3, 3}, {11, 11}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g128_r121{
    2, 128, 1, 1, 1, {3, 3}, {11, 11}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g128_r128{
    2, 128, 2, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g128_r144{
    2, 128, 1, 1, 1, {3, 3}, {12, 12}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
// Packed-layout analogues of the G176/N1 and G240/N4 3x3 stride-2 corpus shapes.
const ck::utils::conv::ConvParam depthwise_g176_r169{
    2, 176, 1, 1, 1, {3, 3}, {28, 28}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
const ck::utils::conv::ConvParam depthwise_g240_r196{
    2, 240, 4, 1, 1, {3, 3}, {14, 14}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g256_r256{
    2, 256, 4, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g256_asym_r256{
    2, 256, 4, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 0}, {1, 2}};
const ck::utils::conv::ConvParam depthwise_g450_tail_r256{
    2, 450, 4, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g512_stride2_r512{
    2, 512, 8, 1, 1, {3, 3}, {16, 16}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g256_r512{
    2, 256, 8, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
const ck::utils::conv::ConvParam depthwise_g512_r640{
    2, 512, 10, 1, 1, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16DepthwiseShortReduction)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only depthwise candidate";
#ifdef CK_ENABLE_BF16
    CheckCandidate<2, ck::bhalf_t>(depthwise_g127_r121, Candidate::Depthwise, 1, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g128_r121, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g128_r128, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g128_r144, Candidate::Depthwise, 1, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g176_r169, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g240_r196, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g256_r256, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g256_asym_r256, Candidate::Depthwise, 1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g450_tail_r256, Candidate::Depthwise, -1, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g512_stride2_r512, Candidate::Depthwise, 0, true);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g256_r512, Candidate::Depthwise, -1, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g512_r640, Candidate::Depthwise, -1, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g256_r256, Candidate::Depthwise, 2, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g256_r256, Candidate::Depthwise, -2, false);
    CheckCandidate<2, ck::bhalf_t>(depthwise_g176_r169, Candidate::Depthwise, 2, false);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Fp16ScalarOddChannels)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "gfx1250-only scalar candidate";
    }
#ifdef CK_ENABLE_FP16
    CheckCandidate<2, ck::half_t>(odd_channels, Candidate::DirectScalar, 1, true);
    CheckCandidate<2, ck::half_t>(odd_channels, Candidate::DirectScalar, 2, false);
    CheckCandidate<2, ck::half_t>(odd_channels, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<2, ck::half_t>(odd_channels, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "FP16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16ScalarOddChannels)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "gfx1250-only scalar candidate";
    }
#ifdef CK_ENABLE_BF16
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::DirectScalar, 1, true);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::DirectScalar, 2, false);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Fp16ScalarOddChannels3d)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "gfx1250-only scalar candidate";
    }
#ifdef CK_ENABLE_FP16
    CheckCandidate<3, ck::half_t>(odd_channels_3d, Candidate::DirectScalar, 1, true);
    CheckCandidate<3, ck::half_t>(odd_channels_3d, Candidate::DirectScalar, 2, false);
    CheckCandidate<3, ck::half_t>(odd_channels_3d, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<3, ck::half_t>(odd_channels_3d, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "FP16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16ScalarOddChannels3d)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "gfx1250-only scalar candidate";
    }
#ifdef CK_ENABLE_BF16
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::DirectScalar, 1, true);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::DirectScalar, 2, false);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

} // namespace
