// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <array>
#include <string>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
using F16                         = ck::half_t;

template <typename Tuple>
class TestGroupedConvndFwdWavelet : public ::testing::Test
{
    protected:
    using InDataType   = std::tuple_element_t<0, Tuple>;
    using WeiDataType  = std::tuple_element_t<1, Tuple>;
    using OutDataType  = std::tuple_element_t<2, Tuple>;
    using AComputeType = std::tuple_element_t<3, Tuple>;
    using BComputeType = std::tuple_element_t<4, Tuple>;
    using InLayout     = std::tuple_element_t<5, Tuple>;
    using WeiLayout    = std::tuple_element_t<6, Tuple>;
    using OutLayout    = std::tuple_element_t<7, Tuple>;
    using IndexType    = ck::index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;
#if defined(CK_TEST_DISABLE_GPU_VALIDATION)
    static constexpr int verify_ = 1; // CPU reference
#else
    static constexpr int verify_ = 2; // GPU reference
#endif
    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = conv_params[i];
            pass        = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              AComputeType,
                                                                              BComputeType,
                                                                              IndexType>(
                               verify_, // do_verification
                               1,       // init_method: integer value
                               false,   // do_log
                               false,   // time_kernel
                               param,
                               ck::tensor_operation::element_wise::PassThrough{},
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

// Wavelet XDL conv3d fwd instances use NDHWGC layout (NSpatialGC),
// which maps to Wave32Force16MNPerXDL=true on gfx11, so they are valid on gfx9 and gfx11.
using KernelTypes3d = ::testing::Types<std::tuple<F16, F16, F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdWavelet3d : public TestGroupedConvndFwdWavelet<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdWavelet3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdWavelet3d, Test3D)
{
    this->conv_params.clear();

    // Standard 3x3x3 conv
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {3, 3, 3}, {8, 8, 8}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // Strided conv (matching the example's stride=2 pattern in 3D)
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {3, 3, 3}, {16, 16, 16}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // Larger problem resembling the example (N=128, K=256, C=192 scaled down for test speed)
    this->conv_params.push_back(
        {3, 1, 8, 128, 64, {3, 3, 3}, {12, 12, 12}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    // 1x1x1 filter, stride 1
    this->conv_params.push_back(
        {3, 1, 4, 64, 64, {1, 1, 1}, {8, 8, 8}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->template Run<3>();
}

namespace {

template <typename DataType, ck::index_t Dim>
void CheckWaveletPaddedWeights(const ck::utils::conv::ConvParam& param)
{
    using namespace ck::tensor_layout::convolution;
    using InLayout    = ck::conditional_t<Dim == 2, NHWGC, NDHWGC>;
    using WeiLayout   = ck::conditional_t<Dim == 2, GKYXC, GKZYXC>;
    using OutLayout   = ck::conditional_t<Dim == 2, NHWGK, NDHWGK>;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using DeviceOp    = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<Dim,
                                                                                      InLayout,
                                                                                      WeiLayout,
                                                                                      ck::Tuple<>,
                                                                                      OutLayout,
                                                                                      DataType,
                                                                                      DataType,
                                                                                      ck::Tuple<>,
                                                                                      DataType,
                                                                                      PassThrough,
                                                                                      PassThrough,
                                                                                      PassThrough>;
    using Reference   = ck::tensor_operation::host::
        ReferenceConvFwd<Dim, DataType, DataType, DataType, PassThrough, PassThrough, PassThrough>;

    ck::Tensor<DataType> input(
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(param));
    ck::Tensor<DataType> weight(
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(param));
    const auto output_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(param);
    ck::Tensor<DataType> expected(output_desc), actual(output_desc);
    ck::DeviceMem input_device(input.mData.size() * sizeof(DataType));
    ck::DeviceMem weight_device(weight.mData.size() * sizeof(DataType));
    ck::DeviceMem output_device(actual.mData.size() * sizeof(DataType));

    auto dimensions = [](const auto& values) {
        std::array<ck::index_t, Dim + 3> result{};
        std::copy(values.begin(), values.end(), result.begin());
        return result;
    };
    auto spatial = [](const auto& values) {
        std::array<ck::index_t, Dim> result{};
        std::copy(values.begin(), values.end(), result.begin());
        return result;
    };
    const auto input_lengths    = dimensions(input.mDesc.GetLengths());
    const auto input_strides    = dimensions(input.mDesc.GetStrides());
    const auto weight_lengths   = dimensions(weight.mDesc.GetLengths());
    const auto weight_strides   = dimensions(weight.mDesc.GetStrides());
    const auto output_lengths   = dimensions(output_desc.GetLengths());
    const auto output_strides   = dimensions(output_desc.GetStrides());
    const auto filter_strides   = spatial(param.conv_filter_strides_);
    const auto filter_dilations = spatial(param.conv_filter_dilations_);
    const auto left_pads        = spatial(param.input_left_pads_);
    const auto right_pads       = spatial(param.input_right_pads_);
    auto ops = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    bool exercised = false;
    for(const auto& op : ops)
    {
        const auto name = op->GetTypeString();
        if(name.find("Wavelet") == std::string::npos ||
           name.find("DirectLoad") == std::string::npos)
            continue;
        auto argument = op->MakeArgumentPointer(input_device.GetDeviceBuffer(),
                                                weight_device.GetDeviceBuffer(),
                                                {},
                                                output_device.GetDeviceBuffer(),
                                                input_lengths,
                                                input_strides,
                                                weight_lengths,
                                                weight_strides,
                                                {},
                                                {},
                                                output_lengths,
                                                output_strides,
                                                filter_strides,
                                                filter_dilations,
                                                left_pads,
                                                right_pads,
                                                PassThrough{},
                                                PassThrough{},
                                                PassThrough{});
        ck::DeviceMem workspace(op->GetWorkSpaceSize(argument.get()));
        if(workspace.GetBufferSize() != 0)
            op->SetWorkSpacePointer(argument.get(), workspace.GetDeviceBuffer());
        if(!op->IsSupportedArgument(argument.get()))
            continue;
        exercised = true;
        SCOPED_TRACE(name);
        auto invoker = op->MakeInvokerPointer();
        Reference reference;
        auto reference_invoker = reference.MakeInvoker();
        // Reusing the argument/output with changed weights catches stale LDS and missing
        // positive-end zero fills, rather than merely verifying a newly allocated output.
        for(int iteration = 0; iteration < 2; ++iteration)
        {
            SCOPED_TRACE(iteration);
            for(std::size_t i = 0; i < input.mData.size(); ++i)
                input.mData[i] = ck::type_convert<DataType>(
                    (static_cast<int>((i * 7 + iteration * 3) % 17) - 8) * 0.125f);
            for(std::size_t i = 0; i < weight.mData.size(); ++i)
                weight.mData[i] = ck::type_convert<DataType>(
                    (static_cast<int>((i * 5 + iteration * 7) % 13) - 6) * 0.125f);
            std::fill(actual.mData.begin(), actual.mData.end(), ck::type_convert<DataType>(9));
            input_device.ToDevice(input.mData.data());
            weight_device.ToDevice(weight.mData.data());
            output_device.ToDevice(actual.mData.data());
            auto ref_arg = reference.MakeArgument(input,
                                                  weight,
                                                  expected,
                                                  param.conv_filter_strides_,
                                                  param.conv_filter_dilations_,
                                                  param.input_left_pads_,
                                                  param.input_right_pads_,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});
            reference_invoker.Run(ref_arg);
            invoker->Run(argument.get(), StreamConfig{nullptr, false});
            output_device.FromDevice(actual.mData.data());
            EXPECT_TRUE(ck::utils::check_err(actual, expected, "Padded wavelet weight mismatch"));
        }
    }
    ASSERT_TRUE(exercised) << "No supported DirectLoad wavelet instance for the boundary case";
}

template <typename T>
class WaveletPaddedWeights : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(!ck::is_gfx125_supported())
            GTEST_SKIP() << "Tensor weight loads require gfx1250";
    }
};
using WaveletWeightTypes = ::testing::Types<
#ifdef CK_ENABLE_FP32
    float,
#endif
#ifdef CK_ENABLE_BF16
    ck::bhalf_t,
#endif
    ck::half_t>;
TYPED_TEST_SUITE(WaveletPaddedWeights, WaveletWeightTypes);

TYPED_TEST(WaveletPaddedWeights, Grouped2DTailsOverwrite)
{
    // Partial N tiles, odd/even K tails and num_loop=1/2 (no main loop).
    for(const auto& param : std::vector<ck::utils::conv::ConvParam>{
            {2, 2, 2, 72, 8, {3, 3}, {7, 9}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
            {2, 1, 1, 40, 8, {1, 1}, {5, 7}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
            {2, 1, 1, 72, 16, {1, 1}, {5, 7}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}})
        CheckWaveletPaddedWeights<TypeParam, 2>(param);
}

TYPED_TEST(WaveletPaddedWeights, Grouped3DPartialTilesOverwrite)
{
    CheckWaveletPaddedWeights<TypeParam, 3>(
        {3, 2, 1, 48, 16, {3, 3, 3}, {7, 8, 9}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
}

} // namespace
