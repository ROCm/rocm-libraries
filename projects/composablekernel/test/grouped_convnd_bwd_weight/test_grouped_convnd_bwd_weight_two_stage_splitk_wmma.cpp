// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

// Split-K for DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3: stage one accumulates the
// split-K partial sums atomically into an fp32 workspace, which gfx11 supports, so k_batch > 1
// must be accepted and computed correctly on gfx11 as well as on gfx12.

namespace {

using namespace ck::tensor_layout::convolution;
namespace instance = ck::tensor_operation::device::instance;
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;

template <typename DataType>
using InstanceList = std::conditional_t<
    std::is_same_v<DataType, ck::half_t>,
    instance::device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances<
        2,
        NHWGC,
        GKYXC,
        NHWGK,
        instance::ConvBwdWeightDefault,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1>,
    instance::device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_bf16_instances<
        2,
        NHWGC,
        GKYXC,
        NHWGK,
        instance::ConvBwdWeightDefault,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1>>;

template <typename DataType>
class TestGroupedConvndBwdWeightTwoStageSplitKWmma : public ::testing::Test
{
    protected:
    using DeviceOp      = std::tuple_element_t<0, InstanceList<DataType>>;
    using ReferenceConv = ck::tensor_operation::host::ReferenceConvBwdWeight<2,
                                                                             DataType,
                                                                             DataType,
                                                                             DataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2, 3, 8, 32, 128};

    void SetUp() override
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            GTEST_SKIP();
        }
    }

    bool RunCase(const ck::utils::conv::ConvParam& param, ck::index_t split_k)
    {
        const auto in_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);
        const auto wei_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
        const auto out_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);

        ck::Tensor<DataType> input(in_desc);
        ck::Tensor<DataType> output(out_desc);
        ck::Tensor<DataType> weight_host(wei_desc);
        ck::Tensor<DataType> weight_device(wei_desc);
        input.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        output.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});

        auto ref_conv     = ReferenceConv{};
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight_host,
                                                  output,
                                                  param.conv_filter_strides_,
                                                  param.conv_filter_dilations_,
                                                  param.input_left_pads_,
                                                  param.input_right_pads_,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{},
                                                  {},
                                                  {},
                                                  {});
        ref_conv.MakeInvoker().Run(ref_argument);

        ck::DeviceMem in_buf(sizeof(DataType) * input.mDesc.GetElementSpaceSize());
        ck::DeviceMem out_buf(sizeof(DataType) * output.mDesc.GetElementSpaceSize());
        ck::DeviceMem wei_buf(sizeof(DataType) * weight_device.mDesc.GetElementSpaceSize());
        in_buf.ToDevice(input.mData.data());
        out_buf.ToDevice(output.mData.data());
        wei_buf.SetValue(ck::type_convert<DataType>(1000.f));

        std::array<ck::index_t, 5> input_lengths{};
        std::array<ck::index_t, 5> filter_lengths{};
        std::array<ck::index_t, 5> output_lengths{};
        std::array<ck::index_t, 5> input_strides{};
        std::array<ck::index_t, 5> weights_strides{};
        std::array<ck::index_t, 5> output_strides{};
        std::array<ck::index_t, 2> conv_filter_strides{};
        std::array<ck::index_t, 2> conv_filter_dilations{};
        std::array<ck::index_t, 2> input_left_pads{};
        std::array<ck::index_t, 2> input_right_pads{};

        auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

        range_copy(in_desc.GetLengths(), begin(input_lengths));
        range_copy(in_desc.GetStrides(), begin(input_strides));
        range_copy(wei_desc.GetLengths(), begin(filter_lengths));
        range_copy(wei_desc.GetStrides(), begin(weights_strides));
        range_copy(out_desc.GetLengths(), begin(output_lengths));
        range_copy(out_desc.GetStrides(), begin(output_strides));
        range_copy(param.conv_filter_strides_, begin(conv_filter_strides));
        range_copy(param.conv_filter_dilations_, begin(conv_filter_dilations));
        range_copy(param.input_left_pads_, begin(input_left_pads));
        range_copy(param.input_right_pads_, begin(input_right_pads));

        auto conv     = DeviceOp{};
        auto argument = conv.MakeArgument(static_cast<DataType*>(in_buf.GetDeviceBuffer()),
                                          static_cast<DataType*>(wei_buf.GetDeviceBuffer()),
                                          static_cast<DataType*>(out_buf.GetDeviceBuffer()),
                                          input_lengths,
                                          input_strides,
                                          filter_lengths,
                                          weights_strides,
                                          output_lengths,
                                          output_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          PassThrough{},
                                          PassThrough{},
                                          PassThrough{},
                                          split_k);

        ck::DeviceMem workspace(conv.GetWorkSpaceSize(&argument));
        conv.SetWorkSpacePointer(&argument, workspace.GetDeviceBuffer());

        if(!conv.IsSupportedArgument(argument))
        {
            std::cout << "Unsupported: split_k " << split_k << " " << param << std::endl;
            return false;
        }

        conv.MakeInvoker().Run(argument, StreamConfig{nullptr, false});
        wei_buf.FromDevice(weight_device.mData.data());

        // Same tolerances as profile_grouped_conv_bwd_weight_impl.
        const ck::index_t num_accums = output.GetElementSize() / (param.K_ * param.G_) / split_k;

        const float max_value =
            *std::max_element(weight_host.mData.begin(), weight_host.mData.end());

        const double rtol =
            std::max(ck::utils::get_relative_threshold<DataType, DataType, float>(num_accums),
                     ck::utils::get_relative_threshold<DataType, DataType, DataType>(split_k));

        const double atol = std::max(
            ck::utils::get_absolute_threshold<DataType, DataType, float>(max_value / split_k,
                                                                         num_accums),
            ck::utils::get_absolute_threshold<DataType, DataType, DataType>(max_value, split_k));

        const bool pass = ck::utils::check_err(
            weight_device, weight_host, "Error: Incorrect results!", rtol, atol);
        if(!pass)
        {
            std::cout << "Failed: split_k " << split_k << " " << param << std::endl;
        }
        return pass;
    }

    void Run()
    {
        bool pass = true;
        for(const auto& param : conv_params)
        {
            for(const auto split_k : split_ks)
            {
                pass = RunCase(param, split_k) && pass;
            }
        }
        EXPECT_TRUE(pass);
    }
};

using DataTypes = ::testing::Types<ck::half_t, ck::bhalf_t>;
TYPED_TEST_SUITE(TestGroupedConvndBwdWeightTwoStageSplitKWmma, DataTypes);

TYPED_TEST(TestGroupedConvndBwdWeightTwoStageSplitKWmma, Test2D)
{
    this->conv_params.clear();
    // C = 1, K = 16, 4x4 filter, stride 2: a long GEMM K (N * Ho * Wo) on a 16x16 output.
    this->conv_params.push_back(
        {2, 1, 32, 16, 1, {4, 4}, {60, 80}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 4, 32, 3, {3, 3}, {33, 35}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 8, 16, 16, {3, 3}, {28, 28}, {2, 2}, {1, 1}, {1, 1}, {1, 1}});
    // k_batch larger than the number of K blocks.
    this->conv_params.push_back({2, 1, 2, 5, 1, {2, 2}, {7, 9}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->Run();
}

} // namespace
