// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile the inline launch helper and the concrete WMMA-v3 invoker untimed,
// regardless of the CK_TIME_KERNEL setting used for the instance libraries.
#undef CK_TIME_KERNEL
#define CK_TIME_KERNEL 0

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_instances.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ConvSpec    = ck::tensor_operation::device::ConvolutionBackwardDataSpecialization;

// Row 2 in each tuple has an eight-element C-shuffle store and a 32-element
// K tile, so each of the four K=128 partials has a full tile to accumulate.
template <typename DataType>
using WmmaOp = std::tuple_element_t<
    2,
    std::conditional_t<
        std::is_same_v<DataType, ck::half_t>,
        ck::tensor_operation::device::instance::device_grouped_conv_bwd_data_wmma_v3_f16_instances<
            2,
            NHWGK,
            GKYXC,
            ck::Tuple<>,
            NHWGC,
            ConvSpec::Filter1x1Stride1Pad0>,
        ck::tensor_operation::device::instance::device_grouped_conv_bwd_data_wmma_v3_bf16_instances<
            2,
            NHWGK,
            GKYXC,
            ck::Tuple<>,
            NHWGC,
            ConvSpec::Filter1x1Stride1Pad0>>>;

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

__global__ void AtomicAddOne(int* value) { atomicAdd(value, 1); }

template <typename Launch>
void CheckUntimedPreprocess(Launch launch)
{
    ck::DeviceMem device(sizeof(int));
    const int dirty = 113;
    device.ToDevice(&dirty);

    hipStream_t stream;
    ck::hip_check_error(hipStreamCreate(&stream));
    const StreamConfig config{stream, false};
    int* value       = static_cast<int*>(device.GetDeviceBuffer());
    const auto clear = [&]() {
        ck::hip_check_error(hipMemsetAsync(value, 0, sizeof(int), config.stream_id_));
    };

    launch(config, clear, value);
    ck::hip_check_error(hipStreamSynchronize(stream));
    int result = 0;
    device.FromDevice(&result);
    ck::hip_check_error(hipStreamDestroy(stream));
    EXPECT_EQ(result, 1);
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, BasicLauncherPreprocess)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::launch_and_time_kernel_with_preprocess(
            config, clear, AtomicAddOne, dim3(1), dim3(1), 0, value);
    });
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, FlushCacheLauncherPreprocess)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::launch_and_time_kernel_with_preprocess_flush_cache(
            config, clear, AtomicAddOne, dim3(1), dim3(1), 0, value);
    });
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, UtilityLauncherPreprocess)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::utility::launch_and_time_kernel_with_preprocess<false>(
            config, clear, AtomicAddOne, dim3(1), dim3(1), 0, value);
    });
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

#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_BF16)
using Vector2PointwiseOp =
    std::tuple_element_t<0,
                         ck::tensor_operation::device::instance::
                             device_grouped_conv2d_bwd_data_wmma_v3_bf16_a_vector2_instances>;
using ScalarPointwiseOp =
    std::tuple_element_t<6,
                         ck::tensor_operation::device::instance::
                             device_grouped_conv_bwd_data_wmma_v3_bf16_large_tiles_instances<
                                 2,
                                 NHWGK,
                                 GKYXC,
                                 ck::Tuple<>,
                                 NHWGC,
                                 ConvSpec::Filter1x1Stride1Pad0>>;

template <typename Op>
void CheckPointwiseReduction(const ck::utils::conv::ConvParam& param,
                             bool expect_supported,
                             ck::index_t split_k = 1)
{
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);

    ck::Tensor<ck::bhalf_t> out(out_desc);
    ck::Tensor<ck::bhalf_t> wei(wei_desc);
    ck::Tensor<ck::bhalf_t> expected(in_desc);
    ck::Tensor<ck::bhalf_t> actual(in_desc);
    ck::Tensor<ck::bhalf_t> poisoned(in_desc);
    ck::DeviceMem out_device(sizeof(ck::bhalf_t) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(ck::bhalf_t) * wei.mDesc.GetElementSpaceSize());
    ck::DeviceMem in_device(sizeof(ck::bhalf_t) * actual.mDesc.GetElementSpaceSize());

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

    Op op;
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
                               split_k);
    ASSERT_EQ(op.IsSupportedArgument(arg), expect_supported)
        << op.GetTypeString() << " K=" << param.K_ << " G=" << param.G_;
    if(!expect_supported)
    {
        return;
    }

    using Reference = ck::tensor_operation::host::ReferenceConvBwdData<2,
                                                                       ck::bhalf_t,
                                                                       ck::bhalf_t,
                                                                       ck::bhalf_t,
                                                                       PassThrough,
                                                                       PassThrough,
                                                                       PassThrough>;
    auto invoker    = op.MakeInvoker();
    for(int run = 0; run < 2; ++run)
    {
        // For K10, only channels 8 and 9 contribute. They have opposite signs,
        // and both inputs and weights change between calls on the same argument.
        out.ForEach([&](auto& tensor, const auto& index) {
            const auto g      = index[0];
            const auto n      = index[1];
            const auto k      = index[2];
            const auto h      = index[3];
            const auto w      = index[4];
            const float sign  = (g + n + h + w) % 2 == 0 ? 1.f : -1.f;
            const float value = k == 8           ? (run == 0 ? 8.f : -4.f)
                                : k == 9         ? (run == 0 ? -6.f : 2.f)
                                : param.K_ == 10 ? 0.f
                                                 : (static_cast<float>(k % 3) - 1.f) * (run + 1);
            tensor(index)     = ck::type_convert<ck::bhalf_t>(sign * value);
        });
        wei.ForEach([&](auto& tensor, const auto& index) {
            const auto g      = index[0];
            const auto k      = index[1];
            const auto c      = index[2];
            const float value = k == 8   ? (c % 4 == 0 ? 2.f : -1.f) * (run == 0 ? 1.f : -1.f)
                                : k == 9 ? (c % 3 == 0 ? 2.f : -1.f)
                                         : static_cast<float>((k + c + g) % 3) - 1.f;
            tensor(index)     = ck::type_convert<ck::bhalf_t>(value);
        });
        auto ref_arg = Reference::MakeArgument(expected,
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
        out_device.ToDevice(out.mData.data());
        wei_device.ToDevice(wei.mData.data());
        std::fill(poisoned.mData.begin(),
                  poisoned.mData.end(),
                  ck::type_convert<ck::bhalf_t>(run == 0 ? 53.f : -29.f));
        in_device.ToDevice(poisoned.mData.data());
        invoker.Run(arg, StreamConfig{nullptr, false});
        in_device.FromDevice(actual.mData.data());

        std::size_t checked = 0;
        expected.ForEach([&](const auto& tensor, const auto& index) {
            const float want = ck::type_convert<float>(tensor(index));
            const float got  = ck::type_convert<float>(actual(index));
            ++checked;
            EXPECT_TRUE(std::isfinite(got));
            EXPECT_NEAR(got, want, 0.5f)
                << op.GetTypeString() << " run=" << run << " G=" << index[0] << " N=" << index[1]
                << " C=" << index[2] << " H=" << index[3] << " W=" << index[4];
        });
        EXPECT_EQ(checked, expected.mDesc.GetElementSize());
    }
}

TEST(TestGroupedConvndBwdDataWmmaNoTiming, Bf16PointwiseAVector2)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
    // The 7x10 case crosses M128; 3x5 tests a partial spatial tile, and
    // the grouped case catches accidental reads across adjacent K10 rows.
    CheckPointwiseReduction<Vector2PointwiseOp>(
        {2, 1, 2, 10, 128, {1, 1}, {7, 10}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, true);
    CheckPointwiseReduction<Vector2PointwiseOp>(
        {2, 2, 2, 10, 128, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, true);
    const ck::utils::conv::ConvParam split_control = {
        2, 2, 2, 10, 128, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    CheckPointwiseReduction<Vector2PointwiseOp>(split_control, false, 2);
    CheckPointwiseReduction<Vector2PointwiseOp>(split_control, false, 4);
    CheckPointwiseReduction<Vector2PointwiseOp>(
        {2, 1, 1, 8, 128, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, true);
    CheckPointwiseReduction<Vector2PointwiseOp>(
        {2, 1, 1, 16, 128, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, true);
    const ck::utils::conv::ConvParam odd_k = {
        2, 1, 1, 11, 128, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    CheckPointwiseReduction<Vector2PointwiseOp>(odd_k, false);
    CheckPointwiseReduction<ScalarPointwiseOp>(odd_k, true);
}
#endif

} // namespace
