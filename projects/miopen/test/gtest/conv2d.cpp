// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"

namespace {

auto GenFullCases()
{
    std::vector<miopen::test::conv::conv_test_input> cases{};
    
    // Get the standard parameter lists from the header
    auto batch_sizes = miopen::test::conv::get_batch_sizes();
    auto spatial_dims = miopen::test::conv::get_2d_spatial_dims();
    auto filter_dims = miopen::test::conv::get_2d_filter_dims();
    auto in_channels = miopen::test::conv::get_input_channels();
    auto out_channels = miopen::test::conv::get_output_channels();
    auto psd = miopen::test::conv::get_2d_pads_strides_dilations();

    // combinatorial generation with a limit to keep it under control
    // (Similar to how CTest uses MIOPEN_TEST_LIMIT)
    size_t count = 0;
    const size_t limit = 50; // Adjust this to increase/decrease test time

    for (auto b : batch_sizes) {
        for (auto s : spatial_dims) {
            for (auto f : filter_dims) {
                for (auto ic : in_channels) {
                    for (auto oc : out_channels) {
                        for (auto p : psd) {
                            miopen::test::conv::conv_test_input input{};
                            input.batch_size = b;
                            input.input_channels = ic;
                            input.output_channels = oc;
                            input.spatial_dim_elements = s;
                            input.filter_dims = f;
                            input.pads_strides_dilations = p;
                            input.trans_output_pads = {0, 0};
                            input.in_layout = "NCHW";
                            input.fil_layout = "NCHW";
                            input.out_layout = "NCHW";
                            input.conv_mode = "CONV";
                            input.pad_mode = "DEFAULT";
                            input.deterministic = false;
                            input.tensor_vect = 0U;
                            input.vector_length = 1U;
                            
                            cases.push_back(input);
                            
                            if (++count >= limit) return cases;
                        }
                    }
                }
            }
        }
    }
    return cases;
}

} // namespace

template <class T>
struct conv2d_test : miopen::test::conv::conv_test_base<T>
{
};

using GPU_conv_2d_FP32 = conv2d_test<float>;

TEST_P(GPU_conv_2d_FP32, TestFP32) { Run(); }

// Instantiate with the larger dataset
INSTANTIATE_TEST_SUITE_P(Full, GPU_conv_2d_FP32, ::testing::ValuesIn(GenFullCases()));
