// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

#ifdef CK_ENABLE_FP16
namespace {

using F16         = ck::half_t;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
namespace layout  = ck::tensor_layout::convolution;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Match the registered packed tile, reducing vector widths for smaller or odd
// per-group channels and widening GemmN when a cluster has more than 16 outputs.
template <ck::index_t GroupsPerWmma,
          ck::index_t AVector = 4,
          ck::index_t BVector = 8,
          ck::index_t EVector = 4,
          ck::index_t NRepeat = 1>
using PackedConv = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
    2,
    layout::NHWGC,
    layout::GKYXC,
    ck::Tuple<>,
    layout::NHWGK,
    F16,
    F16,
    float,
    float,
    ck::Tuple<>,
    F16,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    64,
    64,
    16 * NRepeat,
    96,
    8,
    8,
    16,
    16,
    2,
    NRepeat,
    S<4, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    AVector,
    8,
    1,
    S<4, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    BVector,
    8,
    1,
    1,
    1,
    S<1, 16, 1, 4>,
    EVector,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    true,
    F16,
    F16,
    1,
    GroupsPerWmma>;

using ReferenceConv = ck::tensor_operation::host::
    ReferenceConvFwd<2, F16, F16, F16, PassThrough, PassThrough, PassThrough>;

struct Problem
{
    ck::utils::conv::ConvParam param;
    ck::HostTensorDescriptor in_desc;
    ck::HostTensorDescriptor wei_desc;
    ck::HostTensorDescriptor out_desc;
    std::array<ck::index_t, 5> in_lengths{}, in_strides{}, wei_lengths{}, wei_strides{},
        out_lengths{}, out_strides{};
    std::array<ck::index_t, 2> strides{}, dilations{}, left_pads{}, right_pads{};

    Problem(ck::index_t groups,
            ck::index_t channels,
            ck::index_t outputs,
            ck::index_t filter,
            ck::index_t stride,
            ck::index_t dilation,
            ck::index_t pad)
        : param(2,
                groups,
                2,
                outputs,
                channels,
                {filter, filter},
                {9, 11},
                {stride, stride},
                {dilation, dilation},
                {pad, pad},
                {pad, pad}),
          in_desc(
              ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<layout::NHWGC>(
                  param)),
          wei_desc(
              ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<layout::GKYXC>(
                  param)),
          out_desc(
              ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<layout::NHWGK>(
                  param))
    {
        auto copy = [](const auto& source, auto& destination) {
            std::copy(source.begin(), source.end(), destination.begin());
        };
        copy(in_desc.GetLengths(), in_lengths);
        copy(in_desc.GetStrides(), in_strides);
        copy(wei_desc.GetLengths(), wei_lengths);
        copy(wei_desc.GetStrides(), wei_strides);
        copy(out_desc.GetLengths(), out_lengths);
        copy(out_desc.GetStrides(), out_strides);
        copy(param.conv_filter_strides_, strides);
        copy(param.conv_filter_dilations_, dilations);
        copy(param.input_left_pads_, left_pads);
        copy(param.input_right_pads_, right_pads);
    }

    template <typename Op>
    auto MakeArgument(const void* input, const void* weight, void* output) const
    {
        return Op::MakeArgument(input,
                                weight,
                                std::array<const void*, 0>{},
                                output,
                                in_lengths,
                                in_strides,
                                wei_lengths,
                                wei_strides,
                                {},
                                {},
                                out_lengths,
                                out_strides,
                                strides,
                                dilations,
                                left_pads,
                                right_pads,
                                PassThrough{},
                                PassThrough{},
                                PassThrough{});
    }
};

class GroupedConvFwdWmmaGroupsPerWmma : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(!ck::is_gfx125_supported())
        {
            GTEST_SKIP() << "Packed WMMA forward convolution requires gfx1250";
        }
    }
};

template <typename Op>
void CheckCorrectness(const Problem& problem)
{
    ck::Tensor<F16> input(problem.in_desc);
    ck::Tensor<F16> weight(problem.wei_desc);
    ck::Tensor<F16> expected(problem.out_desc);
    ck::Tensor<F16> actual(problem.out_desc);
    ck::DeviceMem input_device(input.mData.size() * sizeof(F16));
    ck::DeviceMem weight_device(weight.mData.size() * sizeof(F16));
    ck::DeviceMem output_device(actual.mData.size() * sizeof(F16));

    Op op;
    auto argument              = problem.MakeArgument<Op>(input_device.GetDeviceBuffer(),
                                             weight_device.GetDeviceBuffer(),
                                             output_device.GetDeviceBuffer());
    const auto workspace_bytes = op.GetWorkSpaceSize(&argument);
    ck::DeviceMem workspace(workspace_bytes);
    op.SetWorkSpacePointer(&argument, workspace.GetDeviceBuffer());
    ASSERT_TRUE(op.IsSupportedArgument(&argument));
    EXPECT_EQ(argument.GetRotMemAsTensorSizeBytes()[0], input_device.GetBufferSize());

    std::vector<F16> dirty_workspace(workspace_bytes / sizeof(F16));
    auto invoker = op.MakeInvoker();
    ReferenceConv reference;
    auto reference_invoker = reference.MakeInvoker();

    // Reuse the same argument and allocation, with different input and weights.
    // Nonzero scratch contents expose stale diagonal or off-diagonal entries.
    for(int iteration = 0; iteration < 2; ++iteration)
    {
        SCOPED_TRACE(iteration);
        for(std::size_t i = 0; i < input.mData.size(); ++i)
        {
            const int value = static_cast<int>((i * 7 + iteration * 3) % 17) - 8;
            input.mData[i]  = ck::type_convert<F16>(value * 0.125f);
        }
        for(std::size_t i = 0; i < weight.mData.size(); ++i)
        {
            const int value = static_cast<int>((i * 5 + iteration * 7) % 13) - 6;
            weight.mData[i] = ck::type_convert<F16>(value * 0.125f);
        }
        std::fill(dirty_workspace.begin(),
                  dirty_workspace.end(),
                  ck::type_convert<F16>(iteration == 0 ? -3.0f : 5.0f));
        workspace.ToDevice(dirty_workspace.data());
        input_device.ToDevice(input.mData.data());
        weight_device.ToDevice(weight.mData.data());

        auto reference_argument = reference.MakeArgument(input,
                                                         weight,
                                                         expected,
                                                         problem.param.conv_filter_strides_,
                                                         problem.param.conv_filter_dilations_,
                                                         problem.param.input_left_pads_,
                                                         problem.param.input_right_pads_,
                                                         PassThrough{},
                                                         PassThrough{},
                                                         PassThrough{});
        reference_invoker.Run(reference_argument);
        invoker.Run(argument, StreamConfig{nullptr, false});
        output_device.FromDevice(actual.mData.data());
        EXPECT_TRUE(
            ck::utils::check_err(actual, expected, "Packed convolution mismatch", 1e-3, 1e-3));
    }
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, FourGroupsFourChannels)
{
    for(const auto& geometry : std::array<std::array<ck::index_t, 4>, 4>{
            {{1, 1, 1, 0}, {3, 1, 1, 1}, {3, 2, 1, 0}, {3, 1, 2, 1}}})
    {
        SCOPED_TRACE(::testing::Message() << "filter=" << geometry[0] << " stride=" << geometry[1]
                                          << " dilation=" << geometry[2] << " pad=" << geometry[3]);
        CheckCorrectness<PackedConv<4>>(
            Problem{8, 4, 4, geometry[0], geometry[1], geometry[2], geometry[3]});
    }
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, FourGroupsTwoChannels)
{
    CheckCorrectness<PackedConv<4, 2, 8, 2>>(Problem{8, 2, 2, 1, 1, 1, 0});
    CheckCorrectness<PackedConv<4, 2, 8, 2>>(Problem{8, 2, 2, 3, 2, 2, 1});
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, FourGroupsOddChannels)
{
    CheckCorrectness<PackedConv<4, 1, 4, 1, 2>>(Problem{8, 3, 5, 1, 1, 1, 0});
    CheckCorrectness<PackedConv<4, 1, 4, 1, 2>>(Problem{8, 3, 5, 3, 1, 2, 1});
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, TwoGroupsFourChannels)
{
    CheckCorrectness<PackedConv<2>>(Problem{4, 4, 4, 1, 1, 1, 0});
    CheckCorrectness<PackedConv<2>>(Problem{4, 4, 4, 3, 2, 1, 1});
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, TwoGroupsEightChannels)
{
    CheckCorrectness<PackedConv<2>>(Problem{4, 8, 8, 1, 2, 1, 0});
    CheckCorrectness<PackedConv<2>>(Problem{4, 8, 8, 3, 1, 2, 0});
}

template <ck::index_t GroupsPerWmma, ck::index_t Channels>
void CheckCandidateCorrectness()
{
    constexpr ck::index_t AVector = Channels == 2 ? 2 : 4;
    constexpr ck::index_t BVector = GroupsPerWmma * Channels < 8 ? 4 : 8;
    constexpr ck::index_t EVector = Channels == 2 ? 2 : 4;
    constexpr ck::index_t NRepeat = (GroupsPerWmma * Channels + 15) / 16;
    using Op = PackedConv<GroupsPerWmma, AVector, BVector, EVector, NRepeat>;

    // Two clusters distinguish both neighboring groups and cluster offsets.
    for(const auto& geometry : std::array<std::array<ck::index_t, 4>, 4>{
            {{1, 1, 1, 0}, {3, 1, 1, 1}, {3, 2, 1, 0}, {3, 1, 2, 1}}})
    {
        SCOPED_TRACE(::testing::Message() << "filter=" << geometry[0] << " stride=" << geometry[1]
                                          << " dilation=" << geometry[2] << " pad=" << geometry[3]);
        CheckCorrectness<Op>(Problem{2 * GroupsPerWmma,
                                     Channels,
                                     Channels,
                                     geometry[0],
                                     geometry[1],
                                     geometry[2],
                                     geometry[3]});
    }
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, TwoGroupsTwoChannels)
{
    CheckCandidateCorrectness<2, 2>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, FourGroupsEightChannels)
{
    CheckCandidateCorrectness<4, 8>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, TwoGroupsSixteenChannels)
{
    CheckCandidateCorrectness<2, 16>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, FourGroupsSixteenChannels)
{
    CheckCandidateCorrectness<4, 16>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, EightGroupsTwoChannels)
{
    CheckCandidateCorrectness<8, 2>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, EightGroupsFourChannels)
{
    CheckCandidateCorrectness<8, 4>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, EightGroupsEightChannels)
{
    CheckCandidateCorrectness<8, 8>();
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, EightGroupsSixteenChannels)
{
    CheckCandidateCorrectness<8, 16>();
}

template <ck::index_t GroupsPerWmma>
void CheckRejections()
{
    using Op = PackedConv<GroupsPerWmma>;
    Op op;
    const Problem packed{2 * GroupsPerWmma, 4, 4, 3, 1, 1, 1};
    auto supported = packed.MakeArgument<Op>(nullptr, nullptr, nullptr);
    ck::DeviceMem workspace(op.GetWorkSpaceSize(&supported));

    EXPECT_FALSE(op.IsSupportedArgument(&supported));
    op.SetWorkSpacePointer(&supported, workspace.GetDeviceBuffer());
    ASSERT_TRUE(op.IsSupportedArgument(&supported));

    const Problem tail{2 * GroupsPerWmma + 1, 4, 4, 3, 1, 1, 1};
    auto tail_argument = tail.MakeArgument<Op>(nullptr, nullptr, nullptr);
    op.SetWorkSpacePointer(&tail_argument, workspace.GetDeviceBuffer());
    EXPECT_FALSE(op.IsSupportedArgument(&tail_argument));

    Problem strided = packed;
    for(auto& stride : strided.wei_strides)
    {
        stride *= 2;
    }
    auto strided_argument = strided.MakeArgument<Op>(nullptr, nullptr, nullptr);
    op.SetWorkSpacePointer(&strided_argument, workspace.GetDeviceBuffer());
    EXPECT_FALSE(op.IsSupportedArgument(&strided_argument));
}

TEST_F(GroupedConvFwdWmmaGroupsPerWmma, RejectsGroupTailStridedWeightsAndNullWorkspace)
{
    CheckRejections<4>();
    CheckRejections<2>();
    CheckRejections<8>();
}

} // namespace
#endif // CK_ENABLE_FP16
