// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <type_traits>
#include <tuple>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_exp_gemm_wmma_universal_km_kn_mn_instance.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using namespace ck::tensor_operation::device;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using F16 = ck::half_t;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
constexpr auto Pointwise = ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;
constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
constexpr auto PipeV1 = ck::BlockGemmPipelineVersion::v1;

// Reuse ordinary instance configurations; no scalar split-one registration is needed.
// clang-format off
using SingleStage = DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
    2, NHWGC, GKYXC, NHWGK, F16, F16, F16, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 64, 32, 32, 32, 8, 16, 16, 2, 1,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 2, 2, 0,
    S<4, 16, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 2, 2, 0,
    1, 1, S<1, 8, 1, 8>, 2, Intrawave, PipeV1>;
using TwoStage = std::tuple_element_t<0,
    ck::tensor_operation::device::instance::device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_instances<
        2, NHWGC, GKYXC, NHWGK, F16, Pointwise, Intrawave, PipeV1>>;
using LegacyWmma = DeviceGroupedConvBwdWeight_Wmma_CShuffle<
    2, NHWGC, GKYXC, NHWGK, F16, F16, F16, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 128, 128, 128, 8, 8, 16, 16, 4, 4,
    S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1,
    S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1,
    1, 1, S<1, 32, 1, 4>, 8>;
using Xdl = DeviceGroupedConvBwdWeight_Xdl_CShuffle<
    2, NHWGC, GKYXC, NHWGK, float, float, float, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 64, 64, 64, 4, 4, 32, 32, 2, 2,
    S<1, 4, 16, 1>, S<0, 3, 1, 2>, S<0, 2, 1, 3>, 2, 1, 4, true,
    S<1, 4, 16, 1>, S<0, 3, 1, 2>, S<0, 2, 1, 3>, 2, 1, 4, true,
    1, 1, S<1, 16, 1, 4>, 1>;
template <bool LargeTensors>
using XdlV3 = DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<
    2, NHWGC, GKYXC, NHWGK, float, float, float, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 64, 16, 16, 32, 8, 16, 16, 1, 1,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false,
    1, 1, S<1, 8, 1, 8>, 2, Intrawave, PipeV1, float, float, false, 1, LargeTensors>;
template <bool LargeTensors>
using TwoStageXdl = DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<
    2, NHWGC, GKYXC, NHWGK, F16, F16, F16, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 64, 16, 16, 32, 8, 16, 16, 1, 1,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false,
    1, 1, S<1, 8, 1, 8>, 1, Intrawave, PipeV1, 1, F16, F16, 1, 1, LargeTensors>;
using Dl = DeviceGroupedConvBwdWeight_Dl<
    2, NHWGC, GKYXC, NHWGK, float, float, float, float, PassThrough, PassThrough, PassThrough,
    Pointwise, 256, 128, 128, 16, 1, 4, 4, 1, S<8, 2>, S<8, 2>,
    S<1, 8, 1, 1, 1>, S<1, 2, 1, 128, 1>, S<0, 2, 3, 1, 4>, S<0, 2, 3, 1, 4>,
    S<1, 1, 1, 1, 1>, S<0, 2, 3, 1, 4>, S<1, 1, 1, 1, 1>,
    S<1, 1, 1, 8, 1>, S<1, 16, 1, 16, 1>, S<0, 1, 4, 2, 3>, S<0, 1, 4, 2, 3>,
    S<1, 1, 1, 1, 1>, S<0, 1, 4, 2, 3>, S<1, 1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>, 5, 1>;
using ExplicitGemm = std::tuple_element_t<0,
    ck::tensor_operation::device::instance::device_gemm_wmma_universal_km_kn_mn_GemmMNKPadding_f16_instances>;
using Explicit = DeviceGroupedConvBwdWeight_Explicit<
    2, NHWGC, GKYXC, NHWGK, F16, F16, F16, PassThrough, PassThrough, PassThrough, ExplicitGemm>;
// clang-format on

template <typename Index>
struct Metadata
{
    std::array<Index, 5> in_lengths{2, 2, 128, 4, 4};
    std::array<Index, 5> in_strides{128, 4096, 1, 1024, 256};
    std::array<Index, 5> wei_lengths{2, 128, 128, 1, 1};
    std::array<Index, 5> wei_strides{16384, 128, 1, 128, 128};
    std::array<Index, 5> out_lengths{2, 2, 128, 4, 4};
    std::array<Index, 5> out_strides{128, 4096, 1, 1024, 256};
    std::array<Index, 2> strides{1, 1}, dilations{1, 1}, left{0, 0}, right{0, 0};
};
[[gnu::noinline]] void OverwriteMetadataStack()
{
    volatile ck::index_t overwritten[4096];
    for(std::size_t i = 0; i < 4096; ++i)
        overwritten[i] = -37;
    static_cast<void>(overwritten[0]);
}

template <typename Op, typename DataType, typename Index>
[[gnu::noinline]] auto MakeTemporaryArgument(Op& op, bool pointer, int invalid = -1)
{
    Metadata<Index> metadata;
    if(invalid == 0)
        metadata.strides[0] = 2;
    else if(invalid == 1)
        metadata.left[0] = 1;
    else if(invalid == 2)
        metadata.right[1] = 1;
    // Neither native caller metadata nor factory-local converted arrays survive this call.
    if(pointer)
        return op.MakeArgumentPointer(nullptr, nullptr, nullptr,
                                      metadata.in_lengths, metadata.in_strides,
                                      metadata.wei_lengths, metadata.wei_strides,
                                      metadata.out_lengths, metadata.out_strides,
                                      metadata.strides, metadata.dilations, metadata.left, metadata.right,
                                      PassThrough{}, PassThrough{}, PassThrough{}, 1);
    return std::unique_ptr<ck::tensor_operation::device::BaseArgument>(
        std::make_unique<typename Op::Argument>(op.MakeArgument(
            static_cast<const DataType*>(nullptr), static_cast<DataType*>(nullptr),
            static_cast<const DataType*>(nullptr),
            metadata.in_lengths, metadata.in_strides,
            metadata.wei_lengths, metadata.wei_strides,
            metadata.out_lengths, metadata.out_strides,
            metadata.strides, metadata.dilations, metadata.left, metadata.right,
            PassThrough{}, PassThrough{}, PassThrough{}, 1)));
}

template <typename Op, typename DataType>
void CheckSupportMetadataLifetime()
{
    Op op;
    // A live caller mutation gives deterministic evidence that support checks use the
    // construction-time contract, independently of stack-slot reuse below.
    Metadata<ck::index_t> live;
    auto stable = op.MakeArgumentPointer(nullptr, nullptr, nullptr,
                                         live.in_lengths, live.in_strides,
                                         live.wei_lengths, live.wei_strides,
                                         live.out_lengths, live.out_strides,
                                         live.strides, live.dilations, live.left, live.right,
                                         PassThrough{}, PassThrough{}, PassThrough{}, 1);
    const auto stable_workspace_bytes = op.GetWorkSpaceSize(stable.get());
    ck::DeviceMem stable_workspace(stable_workspace_bytes);
    if(stable_workspace_bytes != 0)
        op.SetWorkSpacePointer(stable.get(), stable_workspace.GetDeviceBuffer());
    ASSERT_TRUE(op.IsSupportedArgument(stable.get()));
    live.strides[0] = 2;
    live.left[0] = 1;
    live.right[1] = 1;
    EXPECT_TRUE(op.IsSupportedArgument(stable.get()));
    for(bool pointer : {false, true})
    {
        for(bool wide : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "pointer=" << pointer << " wide=" << wide);
            auto make = [&](int invalid) {
                return wide ? MakeTemporaryArgument<Op, DataType, ck::long_index_t>(op, pointer, invalid)
                            : MakeTemporaryArgument<Op, DataType, ck::index_t>(op, pointer, invalid);
            };
            auto arg = make(-1);
            const auto workspace_bytes = op.GetWorkSpaceSize(arg.get());
            ck::DeviceMem workspace(workspace_bytes);
            if(workspace_bytes != 0)
                op.SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
            OverwriteMetadataStack();
            ASSERT_TRUE(op.IsSupportedArgument(arg.get()));
            for(int invalid = 0; invalid < 3; ++invalid)
            {
                auto rejected = make(invalid);
                if(workspace_bytes != 0)
                    op.SetWorkSpacePointer(rejected.get(), workspace.GetDeviceBuffer());
                OverwriteMetadataStack();
                EXPECT_FALSE(op.IsSupportedArgument(rejected.get())) << "invalid=" << invalid;
                EXPECT_TRUE(op.IsSupportedArgument(arg.get())) << "earlier argument changed";
            }
        }
    }
}

template <typename Op>
void CheckTemporaryMetadataConsumer()
{
    const ck::utils::conv::ConvParam param{
        2, 2, 2, 16, 16, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    const auto in_desc = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);
    const auto wei_desc = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
    const auto out_desc = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);
    ck::Tensor<F16> in(in_desc), out(out_desc), expected(wei_desc), actual(wei_desc);
    in.ForEach([](auto& tensor, const auto& index) {
        tensor(index) = ck::type_convert<F16>((index[0] == 0 ? 1.f : -1.f) *
                                            (1.f + (index[1] + index[2] + index[3] + index[4]) % 3));
    });
    out.ForEach([](auto& tensor, const auto& index) {
        tensor(index) = ck::type_convert<F16>(1.f + (index[0] + index[2] + 2 * index[4]) % 3);
    });
    using Reference = ck::tensor_operation::host::ReferenceConvBwdWeight<
        2, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
    auto ref_arg = Reference::MakeArgument(in, expected, out, param.conv_filter_strides_,
                                          param.conv_filter_dilations_, param.input_left_pads_,
                                          param.input_right_pads_, PassThrough{}, PassThrough{}, PassThrough{});
    Reference::MakeInvoker().Run(ref_arg);
    ck::DeviceMem in_device(sizeof(F16) * in.mDesc.GetElementSpaceSize());
    ck::DeviceMem out_device(sizeof(F16) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(F16) * actual.mDesc.GetElementSpaceSize());
    in_device.ToDevice(in.mData.data());
    out_device.ToDevice(out.mData.data());
    Op op;
    auto arg = [&]() {
        Metadata<ck::long_index_t> metadata;
        const auto copy = [](const auto& from, auto& to) { ck::ranges::copy(from, to.begin()); };
        copy(in_desc.GetLengths(), metadata.in_lengths);
        copy(in_desc.GetStrides(), metadata.in_strides);
        copy(wei_desc.GetLengths(), metadata.wei_lengths);
        copy(wei_desc.GetStrides(), metadata.wei_strides);
        copy(out_desc.GetLengths(), metadata.out_lengths);
        copy(out_desc.GetStrides(), metadata.out_strides);
        return op.MakeArgumentPointer(in_device.GetDeviceBuffer(), wei_device.GetDeviceBuffer(),
                                      out_device.GetDeviceBuffer(), metadata.in_lengths, metadata.in_strides,
                                      metadata.wei_lengths, metadata.wei_strides,
                                      metadata.out_lengths, metadata.out_strides,
                                      metadata.strides, metadata.dilations, metadata.left, metadata.right,
                                      PassThrough{}, PassThrough{}, PassThrough{}, 1);
    }();
    const auto workspace_bytes = op.GetWorkSpaceSize(arg.get());
    ck::DeviceMem workspace(workspace_bytes);
    if(workspace_bytes != 0)
        op.SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
    OverwriteMetadataStack();
    ASSERT_TRUE(op.IsSupportedArgument(arg.get()));
    // An unadvertised implementation remains conservative even when its dirty
    // invocation below numerically defines the output through workspace conversion.
    if constexpr(std::is_same_v<Op, TwoStage>)
        EXPECT_FALSE(ck::tensor_operation::device::IsOutputFullyDefined(op, arg.get()));
    auto invoker = op.MakeInvokerPointer();
    for(int repetition = 0; repetition < 2; ++repetition)
    {
        std::fill(actual.mData.begin(), actual.mData.end(), ck::type_convert<F16>(113.f));
        wei_device.ToDevice(actual.mData.data());
        invoker->Run(arg.get(), StreamConfig{nullptr, false});
        wei_device.FromDevice(actual.mData.data());
        expected.ForEach([&](const auto& tensor, const auto& index) {
            const float want = ck::type_convert<float>(tensor(index));
            const float got = ck::type_convert<float>(actual(index));
            EXPECT_TRUE(std::isfinite(got));
            EXPECT_EQ(got, want) << "group=" << index[0] << " k=" << index[1]
                                 << " c=" << index[2] << " repetition=" << repetition;
        });
    }
}

// The ordinary vectorized catalog configuration, not a scalar split-one row.
template <ck::index_t D, typename T>
using VectorizedWrw = DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
    D, std::conditional_t<D == 2, NHWGC, NDHWGC>,
    std::conditional_t<D == 2, GKYXC, GKZYXC>, std::conditional_t<D == 2, NHWGK, NDHWGK>,
    T, T, T, float, PassThrough, PassThrough, PassThrough,
    ConvolutionBackwardWeightSpecialization::Default, 64, 32, 32, 32, 8, 16, 16, 2, 1,
    S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 2, 2, 0,
    S<4, 16, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 2, 2, 0,
    1, 1, S<1, 8, 1, 8>, 2, Intrawave, PipeV1>;

template <ck::index_t D, typename T>
void CheckVectorizedOutputContract(const ck::utils::conv::ConvParam& param, ck::index_t split)
{
    using In = std::conditional_t<D == 2, NHWGC, NDHWGC>;
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
        tensor(index) = ck::type_convert<T>((index[0] % 3 == 0 ? -1.f : 1.f) *
                                            (1.f + (2 * index[0] + index[1] + index[2] + spatial) % 3));
    });
    using Reference = ck::tensor_operation::host::ReferenceConvBwdWeight<
        D, T, T, T, PassThrough, PassThrough, PassThrough>;
    auto reference = Reference::MakeArgument(in, expected, out, param.conv_filter_strides_,
                                             param.conv_filter_dilations_, param.input_left_pads_,
                                             param.input_right_pads_, PassThrough{}, PassThrough{}, PassThrough{});
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
    auto argument = op.MakeArgumentPointer(
        in_device.GetDeviceBuffer(), wei_device.GetDeviceBuffer(), out_device.GetDeviceBuffer(),
        il, is, wl, ws, ol, os, strides, dilations, left, right,
        PassThrough{}, PassThrough{}, PassThrough{}, split);
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
            const float got = ck::type_convert<float>(actual(index));
            EXPECT_TRUE(std::isfinite(got));
            EXPECT_EQ(got, want) << "group=" << index[0] << " k=" << index[1]
                                 << " c=" << index[2] << " split=" << split;
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

TEST(TestGroupedConvndBwdWeightMetadataLifetime, SingleStageOwnsTemporaryMetadata)
{
    if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        GTEST_SKIP() << "WMMA architecture required";
    CheckSupportMetadataLifetime<SingleStage, F16>();
    CheckTemporaryMetadataConsumer<SingleStage>();
}

TEST(TestGroupedConvndBwdWeightMetadataLifetime, TwoStageOwnsLongIndexMetadata)
{
    if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        GTEST_SKIP() << "WMMA architecture required";
    CheckSupportMetadataLifetime<TwoStage, F16>();
    CheckTemporaryMetadataConsumer<TwoStage>();
}

TEST(TestGroupedConvndBwdWeightMetadataLifetime, LegacyWmmaOwnsConvertedMetadata)
{
    if(!ck::is_gfx11_supported())
        GTEST_SKIP() << "Legacy WMMA requires gfx11";
    CheckSupportMetadataLifetime<LegacyWmma, F16>();
}

TEST(TestGroupedConvndBwdWeightMetadataLifetime, ExplicitOwnsConvertedMetadata)
{
    if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        GTEST_SKIP() << "WMMA architecture required";
    CheckSupportMetadataLifetime<Explicit, F16>();
}

TEST(TestGroupedConvndBwdWeightMetadataLifetime, DlOwnsConvertedMetadata)
{
    CheckSupportMetadataLifetime<Dl, float>();
}

TEST(TestGroupedConvndBwdWeightMetadataLifetime, XdlOwnsConvertedMetadata)
{
    if(!ck::is_xdl_wmma_supported<float, float, 32, 32, 16, 16>())
        GTEST_SKIP() << "FP32 XDL/WMMA architecture required";
    CheckSupportMetadataLifetime<Xdl, float>();
    CheckSupportMetadataLifetime<XdlV3<false>, float>();
    CheckSupportMetadataLifetime<TwoStageXdl<false>, F16>();
    if(!ck::is_gfx125_supported())
    {
        CheckSupportMetadataLifetime<XdlV3<true>, float>();
        CheckSupportMetadataLifetime<TwoStageXdl<true>, F16>();
    }
}

} // namespace
