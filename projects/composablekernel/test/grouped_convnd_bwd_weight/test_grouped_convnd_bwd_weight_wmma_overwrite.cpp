// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"

namespace {

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

} // namespace
