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
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace {

using namespace ck::tensor_layout::convolution;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NDimSpatial, typename DataType>
void CheckDirtyOutput(const ck::utils::conv::ConvParam& param, bool has_spatial_holes)
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
        const auto g      = index[0];
        const auto n      = index[1];
        const auto k      = index[2];
        const float value = (g == 0 ? 0.25f : -0.25f * (g + 1)) * (1.f + (n + k) % 3);
        tensor(index)     = ck::type_convert<DataType>(value);
    });
    wei.ForEach([&](auto& tensor, const auto& index) {
        const auto g      = index[0];
        const auto k      = index[1];
        const auto c      = index[2];
        const float value = (1.f + g) * (1.f + (k + c) % 2);
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
    Reference::MakeInvoker().Run(ref_arg);

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
    bool executed = false;
    for(const auto& op : instances)
    {
        auto arg = op->MakeArgumentPointer(out_device.GetDeviceBuffer(),
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
                                           1);
        if(!op->IsSupportedArgument(arg.get()))
        {
            continue;
        }

        if(!ck::tensor_operation::device::IsOutputFullyDefined(*op, arg.get()))
            continue;
        const std::string name = op->GetTypeString(); // Diagnostics only.

        // A non-packed group stride must never receive a capability promise,
        // even if the implementation accepts this descriptor for invocation.
        auto nonpacked_strides = in_strides;
        ++nonpacked_strides[0];
        auto nonpacked_arg = op->MakeArgumentPointer(out_device.GetDeviceBuffer(),
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
                                                     nonpacked_strides,
                                                     filter_strides,
                                                     filter_dilations,
                                                     left_pads,
                                                     right_pads,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{},
                                                     1);
        EXPECT_FALSE(ck::tensor_operation::device::IsOutputFullyDefined(*op, nonpacked_arg.get()));

        ck::DeviceMem workspace(op->GetWorkSpaceSize(arg.get()));
        op->SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
        auto invoker = op->MakeInvokerPointer();
        for(int run = 0; run < 2; ++run)
        {
            // Re-poison the same allocation before each independent launch.
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
                EXPECT_NEAR(got, want, 0.5f) << name << " run=" << run << " group=" << index[0]
                                             << " batch=" << index[1] << " channel=" << index[2];
            });
            EXPECT_EQ(checked, expected.mDesc.GetElementSize());
            if(has_spatial_holes)
            {
                EXPECT_GT(holes, 0u);
            }
        }
    }
    EXPECT_TRUE(executed) << "No supported output-qualified instance executed for " << NDimSpatial
                          << "D" << (has_spatial_holes ? ", stride-2" : ", 1x1");
}

template <typename DataType>
void Run2d()
{
    CheckDirtyOutput<2, DataType>({2, 2, 2, 16, 16, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                                  false);
    // Group and channel tails require scalar loads/stores; every logical element
    // must still be written despite padded GEMM M/N tiles.
    CheckDirtyOutput<2, DataType>({2, 4, 2, 5, 3, {1, 1}, {3, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                                  false);
    CheckDirtyOutput<2, DataType>({2, 2, 2, 16, 16, {1, 1}, {4, 6}, {2, 2}, {1, 1}, {0, 0}, {0, 0}},
                                  true);
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

} // namespace
