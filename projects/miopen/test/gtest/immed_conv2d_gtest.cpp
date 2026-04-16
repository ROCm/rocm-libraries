// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"

namespace {

using TestCase = ConvTestBaseTestCase<NamedParameter<size_t>,              // batch_size
                                      NamedParameter<size_t>,              // input_channels
                                      NamedParameter<size_t>,              // output_channels
                                      NamedContainer<std::vector<size_t>>, // spatial_dim_elements
                                      NamedContainer<std::vector<size_t>>, // filter_dims
                                      NamedContainer<std::vector<int>>,    // pads_strides_dilations
                                      NamedContainer<std::vector<int>>,    // trans_output_pads
                                      NamedParameter<std::string>,         // in_layout
                                      NamedParameter<std::string>,         // fil_layout
                                      NamedParameter<std::string>          // out_layout
                                      >;

template <typename T>
auto GenCases(bool smoke_test)
{
    using ct = conv_test<T, ConvApi::Immediate>;
    BaseConvTestParameters<ConvApi::Immediate> baseParams;

    auto batch_size = MakeNamedParameterCollectionValues<size_t>(
        "batch_size", generate_data_limited(ct::get_batch_sizes(), 1, {16}, !smoke_test));

    auto input_channels = MakeNamedParameterCollectionValues<size_t>(
        "input_channels", generate_data_limited(ct::get_input_channels(), 1, {32}, !smoke_test));

    auto output_channels = MakeNamedParameterCollectionValues<size_t>(
        "output_channels", generate_data_limited(ct::get_output_channels(), 1, {32}, !smoke_test));

    auto spatial_dim_elements = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "spatial_dim_elements",
        generate_data_limited(ct::get_2d_spatial_dims(), 1, {56, 56}, !smoke_test));

    auto filter_dims = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "filter_dims", generate_data_limited(ct::get_2d_filter_dims(), 2, {3, 3}, !smoke_test));

    auto pads_strides_dilations = MakeNamedParameterCollectionValues<std::vector<int>>(
        "pads_strides_dilations",
        generate_data_limited(ct::get_2d_pads_strides_dilations(), 2, !smoke_test));

    auto trans_output_pads = MakeNamedParameterCollectionValues<std::vector<int>>(
        "trans_output_pads", generate_data_limited(ct::get_2d_trans_output_pads(), 1, !smoke_test));

    auto in_layout  = MakeNamedParameterValues<std::string>("in_layout", std::string{"NCHW"});
    auto fil_layout = MakeNamedParameterValues<std::string>("fil_layout", std::string{"NCHW"});
    auto out_layout = MakeNamedParameterValues<std::string>("out_layout", std::string{"NCHW"});

    return ct::GenTestParams(baseParams,
                             batch_size,
                             input_channels,
                             output_channels,
                             spatial_dim_elements,
                             filter_dims,
                             pads_strides_dilations,
                             trans_output_pads,
                             in_layout,
                             fil_layout,
                             out_layout);
}

template <typename T>
auto GetCasesFull()
{
    static const auto cases = GenCases<T>(false);
    return cases;
}

template <typename T>
auto GetCasesSmoke()
{
    static const auto cases = GenCases<T>(true);
    return cases;
}

} // namespace

template <class T>
struct immed_conv2d_test : public conv_test<T, ConvApi::Immediate>,
                           public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();

        this->GetTestParams(GetParam(),
                            this->batch_size,
                            this->input_channels,
                            this->output_channels,
                            this->spatial_dim_elements,
                            this->filter_dims,
                            this->pads_strides_dilations,
                            this->trans_output_pads,
                            this->in_layout,
                            this->fil_layout,
                            this->out_layout);
    }
};

using GPU_Immed_Conv2d_FP32  = immed_conv2d_test<float>;
using GPU_Immed_Conv2d_FP16  = immed_conv2d_test<half_float::half>;
using GPU_Immed_Conv2d_BFP16 = immed_conv2d_test<bfloat16>;

TEST_P(GPU_Immed_Conv2d_FP32, TestFloat) { run(); }
TEST_P(GPU_Immed_Conv2d_FP16, TestFloat16) { run(); }
TEST_P(GPU_Immed_Conv2d_BFP16, TestBFloat16) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Immed_Conv2d_FP32,
                         GetCasesSmoke<float>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Immed_Conv2d_FP32,
                         GetCasesFull<float>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Immed_Conv2d_FP16,
                         GetCasesSmoke<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Immed_Conv2d_FP16,
                         GetCasesFull<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Immed_Conv2d_BFP16,
                         GetCasesSmoke<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Immed_Conv2d_BFP16,
                         GetCasesFull<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
