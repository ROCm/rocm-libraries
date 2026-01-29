// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"

namespace
{
auto GenSmallCases()
{
    std::vector<miopen::test::conv::conv_test_input> cases{};
    miopen::test::conv::conv_test_input input{};
    input.batch_size = 1;
    input.input_channels = 32;
    input.output_channels = 64;
    input.spatial_dim_elements = {28, 28};
    input.filter_dims = {3, 3};
    input.pads_strides_dilations = {1, 1, 1, 1, 1, 1};
    input.trans_output_pads = {0, 0};
    input.in_layout = "NCHW";
    input.fil_layout = "NCHW";
    input.out_layout = "NCHW";
    input.deterministic = false;
    input.tensor_vect = 0U;
    input.vector_length = 1U;
    // Only valid for int8 input and weights
    input.output_type = "int32";
    input.int8_vectorize = false;
    cases.push_back(input);

    return cases;
}
} // namespace

template <class T>
struct conv2d_test : miopen::test::conv::conv_test_base<T>
{
};

using GPU_conv_2d_FP32 = conv2d_test<float>;

TEST_P(GPU_conv_2d_FP32, TestFP32) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_conv_2d_FP32, ::testing::ValuesIn(GenSmallCases()));
