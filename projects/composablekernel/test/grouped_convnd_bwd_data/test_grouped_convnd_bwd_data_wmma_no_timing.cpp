// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile the inline launch helper and the concrete WMMA-v3 invoker untimed,
// regardless of the CK_TIME_KERNEL setting used for the instance libraries.
#undef CK_TIME_KERNEL
#define CK_TIME_KERNEL 0

#include <algorithm>
#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

#include "wmma_bwd_data_test_common.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename DataType>
using WmmaOp = ck::test::WmmaBwdDataSplitKOp<DataType>;

template <typename DataType>
void CheckUntimedSplit4()
{
    static_assert(WmmaOp<DataType>::CShuffleBlockTransferScalarPerVector_NPerBlock == 8);
    static_assert(WmmaOp<DataType>::IsGfx125SplitKCandidate);
    const ck::utils::conv::ConvParam param = {
        2, 2, 2, 128, 16, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);

    ck::Tensor<DataType> out(out_desc);
    ck::Tensor<DataType> wei(wei_desc);
    ck::Tensor<DataType> expected(in_desc);
    ck::Tensor<DataType> actual(in_desc);
    ck::Tensor<DataType> poisoned(in_desc);

    // Distinguish both groups, batches and channels with exactly representable inputs.
    out.ForEach([&](auto& tensor, const auto& index) {
        const auto g  = index[0];
        const auto n  = index[1];
        const auto k  = index[2];
        const float x = (g == 0 ? 0.25f : -0.5f) * (1.f + (n + k) % 3);
        tensor(index) = ck::type_convert<DataType>(x);
    });
    wei.ForEach([&](auto& tensor, const auto& index) {
        const auto g  = index[0];
        const auto k  = index[1];
        const auto c  = index[2];
        const float x = (1.f + g) * (1.f + (k + c) % 2);
        tensor(index) = ck::type_convert<DataType>(x);
    });
    std::fill(poisoned.mData.begin(), poisoned.mData.end(), ck::type_convert<DataType>(113.f));

    using Reference = ck::tensor_operation::host::ReferenceConvBwdData<2,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       PassThrough,
                                                                       PassThrough,
                                                                       PassThrough>;
    auto ref_arg    = Reference::MakeArgument(expected,
                                           wei,
                                           out,
                                           param.conv_filter_strides_,
                                           param.conv_filter_dilations_,
                                           param.input_left_pads_,
                                           param.input_right_pads_,
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{});
    Reference::MakeInvoker().Run(ref_arg);

    ck::DeviceMem out_device(sizeof(DataType) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(DataType) * wei.mDesc.GetElementSpaceSize());
    ck::DeviceMem in_device(sizeof(DataType) * actual.mDesc.GetElementSpaceSize());
    out_device.ToDevice(out.mData.data());
    wei_device.ToDevice(wei.mData.data());

    std::array<ck::index_t, 5> out_lengths{}, out_strides{};
    std::array<ck::index_t, 5> wei_lengths{}, wei_strides{};
    std::array<ck::index_t, 5> in_lengths{}, in_strides{};
    std::array<ck::index_t, 2> filter_strides{}, filter_dilations{};
    std::array<ck::index_t, 2> left_pads{}, right_pads{};
    const auto copy = [](const auto& from, auto& to) { ck::ranges::copy(from, to.begin()); };
    copy(out_desc.GetLengths(), out_lengths);
    copy(out_desc.GetStrides(), out_strides);
    copy(wei_desc.GetLengths(), wei_lengths);
    copy(wei_desc.GetStrides(), wei_strides);
    copy(in_desc.GetLengths(), in_lengths);
    copy(in_desc.GetStrides(), in_strides);
    copy(param.conv_filter_strides_, filter_strides);
    copy(param.conv_filter_dilations_, filter_dilations);
    copy(param.input_left_pads_, left_pads);
    copy(param.input_right_pads_, right_pads);

    WmmaOp<DataType> op;
    auto arg = op.MakeArgument(out_device.GetDeviceBuffer(),
                               wei_device.GetDeviceBuffer(),
                               {},
                               in_device.GetDeviceBuffer(),
                               out_lengths,
                               out_strides,
                               wei_lengths,
                               wei_strides,
                               {},
                               {},
                               in_lengths,
                               in_strides,
                               filter_strides,
                               filter_dilations,
                               left_pads,
                               right_pads,
                               PassThrough{},
                               PassThrough{},
                               PassThrough{},
                               4);
    ASSERT_TRUE(op.IsSupportedArgument(arg)) << op.GetTypeString();

    auto invoker = op.MakeInvoker();
    for(int run = 0; run < 2; ++run)
    {
        // Each atomic launch must clear the same nonzero output independently.
        in_device.ToDevice(poisoned.mData.data());
        invoker.Run(arg, StreamConfig{nullptr, false});
        in_device.FromDevice(actual.mData.data());

        std::size_t checked = 0;
        expected.ForEach([&](const auto& tensor, const auto& index) {
            const float want = ck::type_convert<float>(tensor(index));
            const float got  = ck::type_convert<float>(actual(index));
            ++checked;
            EXPECT_TRUE(std::isfinite(got)) << "run=" << run << " group=" << index[0];
            // BF16 split-K atomics round each partial before the CPU reference's
            // final cast; addition order can also vary between GPU launches.
            const float tolerance = std::is_same_v<DataType, ck::bhalf_t> ? 8.f : 0.5f;
            EXPECT_NEAR(got, want, tolerance)
                << "run=" << run << " group=" << index[0] << " batch=" << index[1]
                << " channel=" << index[2] << " h=" << index[3] << " w=" << index[4];
        });
        EXPECT_EQ(checked, expected.mDesc.GetElementSize());
    }
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, Fp16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_FP16
    CheckUntimedSplit4<ck::half_t>();
#else
    GTEST_SKIP() << "FP16 instances are disabled";
#endif
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, Bf16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_BF16
    CheckUntimedSplit4<ck::bhalf_t>();
#else
    GTEST_SKIP() << "BF16 instances are disabled";
#endif
}

} // namespace
