// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <tuple>
#include <stdexcept>

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp"
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

template <ck::index_t... Values>
using S = ck::Sequence<Values...>;
template <bool PatchView>
using XCVectorOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<
    2,
    NHWGC,
    GKYXC,
    NHWGK,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::bhalf_t,
    float,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
    128,
    64,
    64,
    128,
    8,
    16,
    16,
    4,
    1,
    S<16, 8, 1>,
    S<2, 0, 1>,
    S<1, 0, 2>,
    1,
    8,
    8,
    false,
    S<16, 8, 1>,
    S<2, 0, 1>,
    S<1, 0, 2>,
    1,
    8,
    8,
    false,
    1,
    1,
    S<1, 16, 1, 8>,
    1,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    1,
    ck::bhalf_t,
    ck::bhalf_t,
    1,
    1,
    false,
    PatchView>;

enum class Candidate
{
    DirectScalar,
    TwoStageScalar,
    TwoStageXCVector,
    TwoStagePatch,
    TwoStagePatchWide,
    TwoStageSkinny
};

bool IsCandidate(const std::string& name, Candidate candidate)
{
    if(candidate == Candidate::TwoStageXCVector || candidate == Candidate::TwoStagePatch)
    {
        const std::string prefix =
            "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<128, 64, 64, 128, Default";
        const bool patch = name.find("NonOverlapPatchView") != std::string::npos;
        return name.compare(0, prefix.size(), prefix) == 0 &&
               patch == (candidate == Candidate::TwoStagePatch);
    }
    if(candidate == Candidate::TwoStagePatchWide)
    {
        const std::string prefix =
            "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<128, 128, 128, 32, Default";
        const bool patch = name.find("NonOverlapPatchView") != std::string::npos;
        return name.compare(0, prefix.size(), prefix) == 0 && patch;
    }
    if(candidate == Candidate::TwoStageSkinny)
    {
        const std::string prefix =
            "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<128, 16, 128, 64, Default";
        return name.compare(0, prefix.size(), prefix) == 0;
    }
    const std::string prefix =
        "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 16, 32, Default";
    const std::string suffix = "BlkGemmPipelineVersion: v1, 1>";
    return name.compare(0, prefix.size(), prefix) == 0 && name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template <std::size_t N>
std::array<ck::long_index_t, N> LongIndexMetadata(const std::array<ck::index_t, N>& values)
{
    std::array<ck::long_index_t, N> result{};
    ck::ranges::copy(values, result.begin());
    return result;
}

[[gnu::noinline]] void OverwriteMetadataStack()
{
    volatile ck::index_t overwritten[4096];
    for(std::size_t i = 0; i < 4096; ++i)
        overwritten[i] = -37;
    static_cast<void>(overwritten[0]);
}

template <ck::index_t NDimSpatial, typename DataType>
void CheckCandidate(const ck::utils::conv::ConvParam& param,
                    Candidate candidate,
                    ck::index_t split,
                    bool expected_support,
                    bool long_index_metadata = false)
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

    if(candidate == Candidate::TwoStageXCVector || candidate == Candidate::TwoStagePatch ||
       candidate == Candidate::TwoStagePatchWide || candidate == Candidate::TwoStageSkinny)
    {
        // Include cancellation and distinguish image, patch, filter-row, and
        // channel coordinates. Products and FP32 partial sums are exact here.
        in.ForEach([](auto& tensor, const auto& index) {
            std::size_t pattern = 7 * index[0] + 11 * index[1] + 13 * index[2];
            for(ck::index_t d = 0; d < NDimSpatial; ++d)
                pattern = pattern * 17 + (d + 3) * index[d + 3];
            const int value = static_cast<int>(pattern % 7) - 3;
            tensor(index)   = ck::type_convert<DataType>((value >= 0 ? value + 1 : value) / 16.f);
        });
        out.ForEach([](auto& tensor, const auto& index) {
            std::size_t pattern = 19 * index[0] + 23 * index[1] + 29 * index[2];
            for(ck::index_t d = 0; d < NDimSpatial; ++d)
                pattern = pattern * 31 + (d + 5) * index[d + 3];
            const int value = static_cast<int>(pattern % 9) - 4;
            tensor(index)   = ck::type_convert<DataType>((value >= 0 ? value + 1 : value) / 32.f);
        });
    }
    else
    {
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
            tensor(index) =
                ck::type_convert<DataType>(sign * (1.f + (2 * g + n + k + spatial) % 3));
        });
    }

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
    if constexpr(NDimSpatial == 2 && std::is_same_v<DataType, ck::bhalf_t>)
    {
        if(candidate == Candidate::TwoStageXCVector || candidate == Candidate::TwoStagePatch)
        {
            // Test the descriptor/lifetime behavior independently of catalog ordering.
            instances.clear();
            if(candidate == Candidate::TwoStageXCVector)
                instances.push_back(std::make_unique<XCVectorOp<false>>());
            else
                instances.push_back(std::make_unique<XCVectorOp<true>>());
        }
    }
    int matches        = 0;
    using ScalarDirect = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
        NDimSpatial,
        InLayout<NDimSpatial>,
        WeiLayout<NDimSpatial>,
        OutLayout<NDimSpatial>,
        DataType,
        DataType,
        DataType,
        float,
        PassThrough,
        PassThrough,
        PassThrough,
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
        32,
        16,
        16,
        32,
        8,
        16,
        16,
        1,
        1,
        S<4, 8, 1>,
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        1,
        4,
        0,
        S<4, 8, 1>,
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        1,
        4,
        0,
        1,
        1,
        S<1, 4, 1, 8>,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1>;
    for(const auto& op : instances)
    {
        const std::string name = op->GetTypeString();
        const bool selected    = candidate == Candidate::DirectScalar
                                     ? dynamic_cast<const ScalarDirect*>(op.get()) != nullptr
                                     : IsCandidate(name, candidate);
        if(!selected)
        {
            continue;
        }
        ++matches;
        const auto make_arg = [&](const auto& il,
                                  const auto& is,
                                  const auto& wl,
                                  const auto& ws,
                                  const auto& ol,
                                  const auto& os,
                                  const auto& strides,
                                  const auto& dilations,
                                  const auto& left,
                                  const auto& right) {
            return op->MakeArgumentPointer(in_device.GetDeviceBuffer(),
                                           wei_device.GetDeviceBuffer(),
                                           out_device.GetDeviceBuffer(),
                                           il,
                                           is,
                                           wl,
                                           ws,
                                           ol,
                                           os,
                                           strides,
                                           dilations,
                                           left,
                                           right,
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{},
                                           split);
        };
        const bool use_long_metadata =
            long_index_metadata || candidate == Candidate::TwoStageXCVector ||
            candidate == Candidate::TwoStagePatch || candidate == Candidate::TwoStagePatchWide ||
            candidate == Candidate::TwoStageSkinny;
        auto arg = use_long_metadata ? make_arg(LongIndexMetadata(in_lengths),
                                                LongIndexMetadata(in_strides),
                                                LongIndexMetadata(wei_lengths),
                                                LongIndexMetadata(wei_strides),
                                                LongIndexMetadata(out_lengths),
                                                LongIndexMetadata(out_strides),
                                                LongIndexMetadata(filter_strides),
                                                LongIndexMetadata(filter_dilations),
                                                LongIndexMetadata(left_pads),
                                                LongIndexMetadata(right_pads))
                                     : make_arg(in_lengths,
                                                in_strides,
                                                wei_lengths,
                                                wei_strides,
                                                out_lengths,
                                                out_strides,
                                                filter_strides,
                                                filter_dilations,
                                                left_pads,
                                                right_pads);
        if(use_long_metadata)
            OverwriteMetadataStack();
        if(candidate == Candidate::DirectScalar && split < 0)
        {
            const auto* scalar = dynamic_cast<const typename ScalarDirect::Argument*>(arg.get());
            ASSERT_NE(scalar, nullptr);
            EXPECT_EQ(scalar->k_batch_, 1);
        }
        const auto workspace_bytes = op->GetWorkSpaceSize(arg.get());
        ck::DeviceMem workspace(workspace_bytes);
        if(workspace_bytes != 0)
        {
            op->SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
        }
        const bool supported = op->IsSupportedArgument(arg.get());
        EXPECT_EQ(supported, expected_support) << name << " G=" << param.G_ << " split=" << split;
        if(!supported || !expected_support)
        {
            if(candidate == Candidate::DirectScalar && !supported)
                EXPECT_THROW(op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, false}),
                             std::runtime_error);
            continue;
        }
        if(candidate != Candidate::DirectScalar)
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

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, TwoStageAutoSplitFillsDeviceCapacity)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250 grouped WRW occupancy regression";
#ifdef CK_ENABLE_FP16
    using OrdinaryTwoStage = std::tuple_element_t<
        0,
        ck::tensor_operation::device::instance::
            device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_instances<
                2,
                NHWGC,
                GKYXC,
                NHWGK,
                ck::half_t,
                ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v1>>;
    // The confirmed long-reduction RGB regression has a small output-tile grid.
    // Only descriptors are needed: planning must not read or allocate tensor payloads.
    const ck::utils::conv::ConvParam param{
        2, 1, 128, 24, 3, {3, 3}, {480, 480}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);
    std::array<ck::index_t, 5> in_lengths{}, in_strides{}, wei_lengths{}, wei_strides{};
    std::array<ck::index_t, 5> out_lengths{}, out_strides{};
    std::array<ck::index_t, 2> strides{}, dilations{}, left{}, right{};
    auto copy = [](const auto& src, auto& dst) { ck::ranges::copy(src, dst.begin()); };
    copy(in_desc.GetLengths(), in_lengths);
    copy(in_desc.GetStrides(), in_strides);
    copy(wei_desc.GetLengths(), wei_lengths);
    copy(wei_desc.GetStrides(), wei_strides);
    copy(out_desc.GetLengths(), out_lengths);
    copy(out_desc.GetStrides(), out_strides);
    copy(param.conv_filter_strides_, strides);
    copy(param.conv_filter_dilations_, dilations);
    copy(param.input_left_pads_, left);
    copy(param.input_right_pads_, right);
    OrdinaryTwoStage op;
    auto arg            = op.MakeArgumentPointer(nullptr,
                                      nullptr,
                                      nullptr,
                                      in_lengths,
                                      in_strides,
                                      wei_lengths,
                                      wei_strides,
                                      out_lengths,
                                      out_strides,
                                      strides,
                                      dilations,
                                      left,
                                      right,
                                      PassThrough{},
                                      PassThrough{},
                                      PassThrough{},
                                      -1);
    const auto* planned = dynamic_cast<const OrdinaryTwoStage::Argument*>(arg.get());
    ASSERT_NE(planned, nullptr);
    const OrdinaryTwoStage::ActiveWorkgroupsPerCU occupancy;
    const ck::tensor_operation::device::DeviceProperties device;
    const ck::long_index_t capacity = ck::long_index_t{occupancy.max_occupancy_} * device.num_cu_;
    const auto& tiles               = planned->c_grid_desc_mblock_mperblock_nblock_nperblock_;
    const ck::long_index_t grid =
        ck::long_index_t{tiles.GetLength(ck::Number<0>{})} * tiles.GetLength(ck::Number<2>{});
    ASSERT_GT(grid, 0);
    // Each requested split can contain a complete reduction tile, even when
    // filling all resident block slots. Thus reduction length cannot justify
    // discarding occupancy here. Permit the unavoidable partial final grid.
    const ck::long_index_t reduction = ck::long_index_t{param.N_} *
                                       param.output_spatial_lengths_[0] *
                                       param.output_spatial_lengths_[1];
    ASSERT_GE(reduction, capacity * 32);
    const ck::long_index_t planned_blocks = grid * planned->k_batch_;
    EXPECT_LE(planned_blocks, capacity);
    EXPECT_LT(capacity - planned_blocks, grid);
#else
    GTEST_SKIP() << "FP16 instances disabled";
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
    CheckCandidate<2, ck::half_t>(odd_channels, Candidate::DirectScalar, -1, true);
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
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::DirectScalar, -1, true);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::DirectScalar, 2, false);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<2, ck::bhalf_t>(odd_channels, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16XCVectorMetadataLifetimeAndPatchTails)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only XC-vector candidate";
#ifdef CK_ENABLE_BF16
    // R=90 and K=160 exercise reduction and output-tile tails; the rectangular
    // patch/image dimensions prevent accidental contiguous patch flattening.
    const ck::utils::conv::ConvParam patch{
        2, 1, 3, 160, 3, {16, 16}, {80, 96}, {16, 16}, {1, 1}, {0, 0}, {0, 0}};
    for(auto candidate : {Candidate::TwoStageXCVector, Candidate::TwoStagePatch})
        for(ck::index_t split : {1, 3, 8})
            CheckCandidate<2, ck::bhalf_t>(patch, candidate, split, true);
    const ck::utils::conv::ConvParam wide_tail{
        2, 1, 3, 132, 3, {16, 16}, {80, 96}, {16, 16}, {1, 1}, {0, 0}, {0, 0}};
    CheckCandidate<2, ck::bhalf_t>(wide_tail, Candidate::TwoStagePatchWide, 3, true);
    const ck::utils::conv::ConvParam four{
        2, 1, 3, 132, 3, {4, 4}, {20, 28}, {4, 4}, {1, 1}, {0, 0}, {0, 0}};
    CheckCandidate<2, ck::bhalf_t>(four, Candidate::TwoStagePatchWide, 3, true);
    const ck::utils::conv::ConvParam eight{
        2, 1, 3, 132, 3, {8, 8}, {40, 56}, {8, 8}, {1, 1}, {0, 0}, {0, 0}};
    CheckCandidate<2, ck::bhalf_t>(eight, Candidate::TwoStagePatchWide, 3, true);
    const ck::utils::conv::ConvParam padded{
        2, 1, 3, 160, 3, {16, 16}, {80, 96}, {16, 16}, {1, 1}, {0, 1}, {0, 1}};
    const ck::utils::conv::ConvParam dilated{
        2, 1, 3, 160, 3, {16, 16}, {96, 112}, {16, 16}, {2, 1}, {0, 0}, {0, 0}};
    const ck::utils::conv::ConvParam overlapping{
        2, 1, 3, 160, 3, {16, 16}, {80, 96}, {8, 16}, {1, 1}, {0, 0}, {0, 0}};
    CheckCandidate<2, ck::bhalf_t>(padded, Candidate::TwoStagePatch, 3, false);
    CheckCandidate<2, ck::bhalf_t>(dilated, Candidate::TwoStagePatch, 3, false);
    CheckCandidate<2, ck::bhalf_t>(overlapping, Candidate::TwoStagePatch, 3, false);
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
    CheckCandidate<3, ck::half_t>(odd_channels_3d, Candidate::DirectScalar, -1, true);
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
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::DirectScalar, -1, true);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::DirectScalar, 2, false);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::TwoStageScalar, 1, true);
    CheckCandidate<3, ck::bhalf_t>(odd_channels_3d, Candidate::TwoStageScalar, 2, true);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16SkinnyOutputSignedSplitTails)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only skinny-output candidate";
#ifdef CK_ENABLE_BF16
    const ck::utils::conv::ConvParam k3{
        2, 1, 2, 3, 32, {3, 3}, {17, 19}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    const ck::utils::conv::ConvParam k5{
        2, 1, 2, 5, 40, {3, 3}, {65, 67}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    const ck::utils::conv::ConvParam rank3{
        3, 1, 2, 3, 16, {3, 3, 3}, {5, 11, 13}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    CheckCandidate<2, ck::bhalf_t>(k3, Candidate::TwoStageSkinny, 3, true);
    CheckCandidate<2, ck::bhalf_t>(k5, Candidate::TwoStageSkinny, 8, true);
    CheckCandidate<3, ck::bhalf_t>(rank3, Candidate::TwoStageSkinny, 3, true);
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

} // namespace

namespace output_contract_regressions {

using namespace ck::tensor_layout::convolution;
using namespace ck::tensor_operation::device;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... Is>
using S                  = ck::Sequence<Is...>;
constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
constexpr auto PipeV1    = ck::BlockGemmPipelineVersion::v1;

// The ordinary vectorized catalog configuration, not a scalar split-one row.
template <ck::index_t D, typename T>
using VectorizedWrw =
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<D,
                                               std::conditional_t<D == 2, NHWGC, NDHWGC>,
                                               std::conditional_t<D == 2, GKYXC, GKZYXC>,
                                               std::conditional_t<D == 2, NHWGK, NDHWGK>,
                                               T,
                                               T,
                                               T,
                                               float,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough,
                                               ConvolutionBackwardWeightSpecialization::Default,
                                               64,
                                               32,
                                               32,
                                               32,
                                               8,
                                               16,
                                               16,
                                               2,
                                               1,
                                               S<4, 8, 1>,
                                               S<2, 0, 1>,
                                               S<1, 0, 2>,
                                               1,
                                               2,
                                               2,
                                               0,
                                               S<4, 16, 1>,
                                               S<2, 0, 1>,
                                               S<1, 0, 2>,
                                               1,
                                               2,
                                               2,
                                               0,
                                               1,
                                               1,
                                               S<1, 8, 1, 8>,
                                               2,
                                               Intrawave,
                                               PipeV1>;

template <ck::index_t D, typename T>
void CheckVectorizedOutputContract(const ck::utils::conv::ConvParam& param, ck::index_t split)
{
    using In  = std::conditional_t<D == 2, NHWGC, NDHWGC>;
    using Wei = std::conditional_t<D == 2, GKYXC, GKZYXC>;
    using Out = std::conditional_t<D == 2, NHWGK, NDHWGK>;
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<In>(param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Wei>(param);
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Out>(param);
    ck::Tensor<T> in(in_desc), out(out_desc), expected(wei_desc), actual(wei_desc);
    in.ForEach([](auto& tensor, const auto& index) {
        std::size_t spatial = 0;
        for(ck::index_t d = 0; d < D; ++d)
            spatial += (d + 1) * index[3 + d];
        tensor(index) = ck::type_convert<T>((index[0] % 2 == 0 ? -1.f : 1.f) *
                                            (1.f + (index[0] + index[1] + index[2] + spatial) % 3));
    });
    out.ForEach([](auto& tensor, const auto& index) {
        std::size_t spatial = 0;
        for(ck::index_t d = 0; d < D; ++d)
            spatial += (d + 2) * index[3 + d];
        tensor(index) =
            ck::type_convert<T>((index[0] % 3 == 0 ? -1.f : 1.f) *
                                (1.f + (2 * index[0] + index[1] + index[2] + spatial) % 3));
    });
    using Reference = ck::tensor_operation::host::
        ReferenceConvBwdWeight<D, T, T, T, PassThrough, PassThrough, PassThrough>;
    auto reference = Reference::MakeArgument(in,
                                             expected,
                                             out,
                                             param.conv_filter_strides_,
                                             param.conv_filter_dilations_,
                                             param.input_left_pads_,
                                             param.input_right_pads_,
                                             PassThrough{},
                                             PassThrough{},
                                             PassThrough{});
    Reference::MakeInvoker().Run(reference);
    ck::DeviceMem in_device(sizeof(T) * in.mDesc.GetElementSpaceSize());
    ck::DeviceMem out_device(sizeof(T) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(T) * actual.mDesc.GetElementSpaceSize());
    in_device.ToDevice(in.mData.data());
    out_device.ToDevice(out.mData.data());
    std::array<ck::index_t, D + 3> il{}, is{}, wl{}, ws{}, ol{}, os{};
    std::array<ck::index_t, D> strides{}, dilations{}, left{}, right{};
    const auto copy = [](const auto& from, auto& to) { ck::ranges::copy(from, to.begin()); };
    copy(in_desc.GetLengths(), il);
    copy(in_desc.GetStrides(), is);
    copy(wei_desc.GetLengths(), wl);
    copy(wei_desc.GetStrides(), ws);
    copy(out_desc.GetLengths(), ol);
    copy(out_desc.GetStrides(), os);
    copy(param.conv_filter_strides_, strides);
    copy(param.conv_filter_dilations_, dilations);
    copy(param.input_left_pads_, left);
    copy(param.input_right_pads_, right);
    VectorizedWrw<D, T> op;
    auto argument = op.MakeArgumentPointer(in_device.GetDeviceBuffer(),
                                           wei_device.GetDeviceBuffer(),
                                           out_device.GetDeviceBuffer(),
                                           il,
                                           is,
                                           wl,
                                           ws,
                                           ol,
                                           os,
                                           strides,
                                           dilations,
                                           left,
                                           right,
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{},
                                           split);
    ASSERT_TRUE(op.IsSupportedArgument(argument.get()));
    EXPECT_EQ(ck::tensor_operation::device::IsOutputFullyDefined(op, argument.get()), split == 1);
    for(float poison : {113.f, -87.f})
    {
        std::fill(actual.mData.begin(), actual.mData.end(), ck::type_convert<T>(poison));
        wei_device.ToDevice(actual.mData.data());
        op.MakeInvokerPointer()->Run(argument.get(), StreamConfig{nullptr, false});
        wei_device.FromDevice(actual.mData.data());
        expected.ForEach([&](const auto& tensor, const auto& index) {
            const float want = ck::type_convert<float>(tensor(index));
            const float got  = ck::type_convert<float>(actual(index));
            EXPECT_TRUE(std::isfinite(got));
            EXPECT_EQ(got, want) << "group=" << index[0] << " k=" << index[1] << " c=" << index[2]
                                 << " split=" << split;
        });
    }
}

template <typename T>
void CheckPackedOutputContracts()
{
    const ck::utils::conv::ConvParam two_d{
        2, 3, 2, 16, 16, {3, 3}, {5, 6}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    const ck::utils::conv::ConvParam three_d{
        3, 3, 2, 16, 16, {1, 1, 1}, {2, 3, 6}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    CheckVectorizedOutputContract<2, T>(two_d, 1);
    CheckVectorizedOutputContract<2, T>(two_d, 2);
    CheckVectorizedOutputContract<3, T>(three_d, 1);
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Fp16PackedOutputContract)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250 grouped WRW regression";
#ifdef CK_ENABLE_FP16
    CheckPackedOutputContracts<ck::half_t>();
#else
    GTEST_SKIP() << "FP16 instances disabled";
#endif
}

TEST(TestGroupedConvndBwdWeightWmmaOverwrite, Bf16PackedOutputContract)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250 grouped WRW regression";
#ifdef CK_ENABLE_BF16
    CheckPackedOutputContracts<ck::bhalf_t>();
#else
    GTEST_SKIP() << "BF16 instances disabled";
#endif
}

} // namespace output_contract_regressions
