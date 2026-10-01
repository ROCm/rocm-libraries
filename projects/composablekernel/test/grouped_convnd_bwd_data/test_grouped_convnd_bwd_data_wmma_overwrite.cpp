// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

#include "wmma_bwd_data_test_common.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

enum class InputPattern
{
    Regular,
    Cancellation,
    SubnormalOutput
};

template <ck::index_t NDimSpatial, typename DataType>
void CheckDirtyOutput(const ck::utils::conv::ConvParam& param,
                      bool has_spatial_holes,
                      ck::index_t split_k   = 1,
                      bool expect_rejection = false,
                      bool filter_only      = false,
                      InputPattern pattern  = InputPattern::Regular)
{
    using OutLayout = std::conditional_t<NDimSpatial == 2, NHWGK, NDHWGK>;
    using WeiLayout = std::conditional_t<NDimSpatial == 2, GKYXC, GKZYXC>;
    using InLayout  = std::conditional_t<NDimSpatial == 2, NHWGC, NDHWGC>;
    using DeviceOp  = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                                                                      OutLayout,
                                                                                      WeiLayout,
                                                                                      ck::Tuple<>,
                                                                                      InLayout,
                                                                                      DataType,
                                                                                      DataType,
                                                                                      ck::Tuple<>,
                                                                                      DataType,
                                                                                      PassThrough,
                                                                                      PassThrough,
                                                                                      PassThrough>;

    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(param);
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(param);

    ck::Tensor<DataType> out(out_desc);
    ck::Tensor<DataType> wei(wei_desc);
    ck::Tensor<DataType> expected(in_desc);
    ck::Tensor<DataType> actual(in_desc);
    ck::Tensor<DataType> poisoned(in_desc);

    // Keep groups distinct (including opposite signs) and all products exactly representable.
    out.ForEach([&](auto& tensor, const auto& index) {
        const auto g = index[0];
        const auto n = index[1];
        const auto k = index[2];
        float value;
        if(pattern == InputPattern::Cancellation)
        {
            // Each 32-element split partial is around 8K, never near FP16 overflow;
            // alternating signs leave a nonzero residual after four atomic adds.
            value = (g == 0 ? 1.f : -1.f) * ((k / 32) % 2 == 0 ? 258.f : -256.f);
        }
        else if(pattern == InputPattern::SubnormalOutput)
        {
            const int exponent = std::is_same_v<DataType, ck::half_t> ? -14 : -126;
            value              = (g == 0 ? 1.f : -1.f) * std::ldexp(1.f, exponent);
        }
        else
        {
            value = (g == 0 ? 0.25f : -0.25f * (g + 1)) * (1.f + (n + k) % 3);
        }
        tensor(index) = ck::type_convert<DataType>(value);
    });
    wei.ForEach([&](auto& tensor, const auto& index) {
        const auto g      = index[0];
        const auto k      = index[1];
        const auto c      = index[2];
        const float value = pattern == InputPattern::Cancellation ? (g == 0 ? 1.f : 0.5f)
                            : pattern == InputPattern::SubnormalOutput
                                ? std::ldexp(1.f, -8)
                                : (1.f + g) * (1.f + (k + c) % 2);
        tensor(index)     = ck::type_convert<DataType>(value);
    });
    std::fill(poisoned.mData.begin(), poisoned.mData.end(), ck::type_convert<DataType>(113.f));

    using Reference = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
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
    if(!expect_rejection)
    {
        Reference::MakeInvoker().Run(ref_arg);
    }

    ck::DeviceMem out_device(sizeof(DataType) * out.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(DataType) * wei.mDesc.GetElementSpaceSize());
    ck::DeviceMem in_device(sizeof(DataType) * actual.mDesc.GetElementSpaceSize());
    out_device.ToDevice(out.mData.data());
    wei_device.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> out_lengths{}, out_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{}, wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> in_lengths{}, in_strides{};
    std::array<ck::index_t, NDimSpatial> filter_strides{}, filter_dilations{};
    std::array<ck::index_t, NDimSpatial> left_pads{}, right_pads{};
    auto copy = [](const auto& from, auto& to) { ck::ranges::copy(from, to.begin()); };
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

    const auto instances = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    // Exercise both the specialized and default v3 paths on a full overwrite. The
    // stride-2 control must use Default; the specialized op rejects that geometry.
    const std::vector<std::string> specializations =
        has_spatial_holes ? std::vector<std::string>{"Default"}
        : filter_only     ? std::vector<std::string>{"Filter1x1Stride1Pad0"}
                          : std::vector<std::string>{"Filter1x1Stride1Pad0", "Default"};
    bool supported_at_split_one = false;
    for(const auto& specialization : specializations)
    {
        bool executed = false;
        for(const auto& op : instances)
        {
            const std::string name = op->GetTypeString();
            if(name.find("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<") != 0 ||
               name.find(", " + specialization + ", ") == std::string::npos)
            {
                continue;
            }

            auto make_arg = [&](ck::index_t split) {
                return op->MakeArgumentPointer(out_device.GetDeviceBuffer(),
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
                                               split);
            };
            auto arg = make_arg(split_k);
            if(expect_rejection)
            {
                supported_at_split_one |= op->IsSupportedArgument(make_arg(1).get());
                EXPECT_FALSE(op->IsSupportedArgument(arg.get())) << name << " C=" << param.C_;
                continue;
            }
            if(!op->IsSupportedArgument(arg.get()))
            {
                continue;
            }

            ck::DeviceMem workspace(op->GetWorkSpaceSize(arg.get()));
            op->SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
            auto invoker = op->MakeInvokerPointer();
            for(int run = 0; run < (split_k > 1 ? 2 : 1); ++run)
            {
                // Re-poison the same allocation before both atomic runs. Each launch
                // must clear the old value, not merely accumulate on the previous run.
                in_device.ToDevice(poisoned.mData.data());
                invoker->Run(arg.get(), StreamConfig{nullptr, false});
                in_device.FromDevice(actual.mData.data());
                executed = true;

                std::size_t checked = 0;
                std::size_t holes   = 0;
                expected.ForEach([&](const auto& tensor, const auto& index) {
                    const float want = ck::type_convert<float>(tensor(index));
                    const float got  = ck::type_convert<float>(actual(index));
                    ++checked;
                    if(has_spatial_holes)
                    {
                        bool is_hole = false;
                        for(ck::index_t d = 0; d < NDimSpatial; ++d)
                        {
                            is_hole |= index[3 + d] % 2 != 0;
                        }
                        if(is_hole)
                        {
                            ++holes;
                            EXPECT_EQ(want, 0.f);
                            EXPECT_EQ(got, 0.f) << name << " group=" << index[0];
                        }
                    }
                    EXPECT_TRUE(std::isfinite(want));
                    EXPECT_TRUE(std::isfinite(got)) << name << " group=" << index[0];
                    if(pattern == InputPattern::SubnormalOutput)
                    {
                        // BF16/FP16 subnormal output may flush to zero in the device
                        // atomic path. Require a finite, correctly bounded result;
                        // cancellation below separately requires a nonzero residual.
                        const int exponent  = std::is_same_v<DataType, ck::half_t> ? -15 : -127;
                        const float nominal = std::ldexp(1.f, exponent);
                        EXPECT_LE(std::abs(got), 2.f * nominal);
                        EXPECT_LE(std::abs(got - want), nominal + 0.25f * std::abs(want));
                    }
                    else
                    {
                        if(pattern == InputPattern::Cancellation)
                        {
                            EXPECT_GT(std::abs(want), 32.f);
                            EXPECT_NE(got, 0.f) << name << " lost the cancellation residual";
                        }
                        // Split atomics round intermediate FP16/BF16 sums in an
                        // unspecified order; compare with dtype-appropriate absolute
                        // and relative bounds instead of bitwise equality.
                        const float atol =
                            pattern == InputPattern::Cancellation
                                ? (std::is_same_v<DataType, ck::bhalf_t> ? 16.f : 1.f)
                            : split_k > 1 && std::is_same_v<DataType, ck::bhalf_t> ? 8.f
                                                                                   : 0.5f;
                        const float rtol =
                            pattern == InputPattern::Cancellation
                                ? (std::is_same_v<DataType, ck::bhalf_t> ? 0.125f : 0.02f)
                            : split_k > 1 ? 0.05f
                                          : 0.f;
                        EXPECT_LE(std::abs(got - want), atol + rtol * std::abs(want))
                            << name << " run=" << run << " group=" << index[0]
                            << " batch=" << index[1] << " channel=" << index[2];
                    }
                });
                EXPECT_EQ(checked, expected.mDesc.GetElementSize());
                if(has_spatial_holes)
                {
                    EXPECT_GT(holes, 0u);
                }
            }
            break;
        }
        if(!expect_rejection)
        {
            EXPECT_TRUE(executed) << "No supported WMMA-v3 " << specialization
                                  << " instance executed for " << NDimSpatial << "D, split "
                                  << split_k << (has_spatial_holes ? ", stride-2" : ", 1x1");
        }
    }
    if(expect_rejection)
    {
        EXPECT_TRUE(supported_at_split_one)
            << "Odd-C rejection must be checked against a supported split-1 WMMA-v3 candidate";
    }
}

template <typename DataType>
void Run2d()
{
    CheckDirtyOutput<2, DataType>({2, 2, 2, 16, 16, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                                  false);
    // Group and channel tails require scalar loads/stores; every logical element
    // must still be written despite padded GEMM M/N tiles.
    CheckDirtyOutput<2, DataType>(
        {2, 4, 2, 5, 3, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, false, 1, false, true);
    CheckDirtyOutput<2, DataType>({2, 2, 2, 16, 16, {1, 1}, {4, 6}, {2, 2}, {1, 1}, {0, 0}, {0, 0}},
                                  true);
}

template <ck::index_t NDimSpatial, typename DataType>
void CheckSplitKRejections(const ck::utils::conv::ConvParam& param)
{
    using Selection = ck::test::WmmaBwdDataSplitKInstance<DataType, NDimSpatial>;
    using Op        = typename Selection::type;
    static_assert(Op::IsSplitKSupported);
    static_assert(Op::IsGfx125SplitKCandidate == (NDimSpatial == 2));
    const auto out_desc = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
        typename Selection::OutLayout>(param);
    const auto wei_desc = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
        typename Selection::WeiLayout>(param);
    const auto in_desc = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
        typename Selection::InLayout>(param);

    ck::DeviceMem out_device(sizeof(DataType) * out_desc.GetElementSpaceSize());
    ck::DeviceMem wei_device(sizeof(DataType) * wei_desc.GetElementSpaceSize());
    ck::DeviceMem in_device(sizeof(DataType) * (in_desc.GetElementSpaceSize() + 1));

    std::array<ck::index_t, NDimSpatial + 3> out_lengths{}, out_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{}, wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> in_lengths{}, in_strides{};
    std::array<ck::index_t, NDimSpatial> filter_strides{}, filter_dilations{};
    std::array<ck::index_t, NDimSpatial> left_pads{}, right_pads{};
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
    auto* input         = static_cast<DataType*>(in_device.GetDeviceBuffer());
    const auto make_arg = [&](ck::index_t split, DataType* p_e, const auto& e_strides) {
        return op.MakeArgument(out_device.GetDeviceBuffer(),
                               wei_device.GetDeviceBuffer(),
                               {},
                               p_e,
                               out_lengths,
                               out_strides,
                               wei_lengths,
                               wei_strides,
                               {},
                               {},
                               in_lengths,
                               e_strides,
                               filter_strides,
                               filter_dilations,
                               left_pads,
                               right_pads,
                               PassThrough{},
                               PassThrough{},
                               PassThrough{},
                               split);
    };
    ASSERT_TRUE(op.IsSupportedArgument(make_arg(1, input, in_strides)));
    for(const ck::index_t split : {2, 4})
    {
        auto aligned = make_arg(split, input, in_strides);
        if constexpr(NDimSpatial == 2)
        {
            // Establish a supported candidate before varying only the new split-K constraints.
            ASSERT_TRUE(Op::IsGfx125SplitKArgument(aligned));
            ASSERT_TRUE(op.IsSupportedArgument(aligned));

            auto misaligned = make_arg(split, input + 1, in_strides);
            EXPECT_FALSE(Op::IsGfx125SplitKArgument(misaligned));
            EXPECT_FALSE(op.IsSupportedArgument(misaligned)) << "misaligned E, split=" << split;

            auto padded_strides = in_strides;
            // Pad each pixel by an even number of elements, preserving dword alignment.
            padded_strides[4] += 8;
            padded_strides[3] = padded_strides[4] * in_lengths[4];
            padded_strides[1] = padded_strides[3] * in_lengths[3];
            auto padded       = make_arg(split, input, padded_strides);
            EXPECT_FALSE(Op::IsGfx125SplitKArgument(padded));
            EXPECT_FALSE(op.IsSupportedArgument(padded)) << "padded E, split=" << split;
        }
        else
        {
            EXPECT_FALSE(Op::IsGfx125SplitKArgument(aligned));
            EXPECT_FALSE(op.IsSupportedArgument(aligned)) << "3D, split=" << split;
        }
    }
    if constexpr(NDimSpatial == 2)
    {
        for(const ck::index_t split : {3, 8})
        {
            auto arg = make_arg(split, input, in_strides);
            EXPECT_FALSE(Op::IsGfx125SplitKArgument(arg));
            EXPECT_FALSE(op.IsSupportedArgument(arg)) << "unsupported split=" << split;
        }
    }
}

template <typename DataType>
void RunPairedSplit2d()
{
    // K=128 leaves at least one full 32-element K tile in each split-4 partial.
    const ck::utils::conv::ConvParam aligned = {
        2, 2, 2, 128, 16, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    CheckDirtyOutput<2, DataType>(aligned, false, 2);
    CheckDirtyOutput<2, DataType>(aligned, false, 4);
    CheckDirtyOutput<2, DataType>(aligned, false, 4, false, false, InputPattern::Cancellation);
    CheckDirtyOutput<2, DataType>(aligned, false, 4, false, false, InputPattern::SubnormalOutput);

    // The scalar split-1 fallback exists at odd C, but no paired split-K store may accept it.
    CheckDirtyOutput<2, DataType>(
        {2, 2, 2, 128, 15, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, false, 2, true);

    CheckSplitKRejections<2, DataType>(aligned);
    CheckSplitKRejections<3, DataType>(
        {3, 2, 2, 128, 16, {1, 1, 1}, {2, 3, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
}

template <typename DataType>
void Run3d()
{
    CheckDirtyOutput<3, DataType>(
        {3, 2, 2, 16, 16, {1, 1, 1}, {2, 3, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}}, false);
    CheckDirtyOutput<3, DataType>(
        {3, 2, 2, 16, 16, {1, 1, 1}, {4, 4, 6}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}}, true);
}

TEST(TestGroupedConvndBwdDataWmmaOverwrite, Fp16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_FP16
    Run2d<ck::half_t>();
    Run3d<ck::half_t>();
#else
    GTEST_SKIP() << "FP16 instances are disabled";
#endif
}

TEST(TestGroupedConvndBwdDataWmmaOverwrite, Bf16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_BF16
    Run2d<ck::bhalf_t>();
    Run3d<ck::bhalf_t>();
#else
    GTEST_SKIP() << "BF16 instances are disabled";
#endif
}

TEST(TestGroupedConvndBwdDataWmmaOverwrite, PairedSplitFp16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_FP16
    RunPairedSplit2d<ck::half_t>();
#else
    GTEST_SKIP() << "FP16 instances are disabled";
#endif
}

TEST(TestGroupedConvndBwdDataWmmaOverwrite, PairedSplitBf16)
{
    if(!ck::is_gfx125_supported())
    {
        GTEST_SKIP() << "This regression is specific to gfx1250";
    }
#ifdef CK_ENABLE_BF16
    RunPairedSplit2d<ck::bhalf_t>();
#else
    GTEST_SKIP() << "BF16 instances are disabled";
#endif
}

} // namespace
