// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"

namespace {

template <typename T>
std::vector<T> generate_data_limited(const std::vector<T>& dims, int limit_multiplier, T single)
{
    const bool full_set = true; 
    const int limit_set = 2;

    if(full_set)
    {
        if(limit_set > 0)
        {
            auto endpoint = std::min(static_cast<int>(dims.size()), limit_set * limit_multiplier);
            return std::vector<T>(dims.cbegin(), dims.cbegin() + endpoint);
        }
        else
            return dims;
    }
    else
    {
        return {single};
    }
}

template <typename T>
std::vector<T> generate_data(const std::vector<T>& dims)
{
    const bool full_set = true;
    if(full_set)
        return dims;
    else
        return {dims.front()};
}

auto GetDataset()
{
    std::vector<miopen::test::conv::conv_test_input> cases{};

    auto batch_sizes            = generate_data_limited(miopen::test::conv::get_batch_sizes(), 1, std::size_t{1});
    auto input_channels         = generate_data_limited(miopen::test::conv::get_input_channels(), 1, std::size_t{32});
    auto output_channels        = generate_data_limited(miopen::test::conv::get_output_channels(), 1, std::size_t{64});
    auto spatial_dim_elements   = generate_data_limited(miopen::test::conv::get_2d_spatial_dims(), 1, std::vector<std::size_t>{28, 28});
    auto filter_dims            = generate_data_limited(miopen::test::conv::get_2d_filter_dims(), 2, std::vector<std::size_t>{3, 3});
    auto pads_strides_dilations = generate_data_limited(miopen::test::conv::get_2d_pads_strides_dilations(), 2, std::vector<int>{1, 1, 1, 1, 1, 1});
    auto trans_output_pads      = generate_data(miopen::test::conv::get_2d_trans_output_pads());
    auto in_layouts             = generate_data(std::vector<std::string>{"NCHW"});
    auto fil_layouts            = generate_data(std::vector<std::string>{"NCHW"});
    auto out_layouts            = generate_data(std::vector<std::string>{"NCHW"});
    auto deterministics         = generate_data(std::vector<bool>{false});
    auto tensor_vects           = generate_data(std::vector<std::size_t>{0});
    auto vector_lengths         = generate_data(std::vector<std::size_t>{1});
    auto output_types           = generate_data(std::vector<std::string>{"int32"});
    auto int8_vectorizes        = generate_data(std::vector<bool>{false});

    for(auto b : batch_sizes)
        for(auto ic : input_channels)
            for(auto oc : output_channels)
                for(auto s : spatial_dim_elements)
                    for(auto f : filter_dims)
                        for(auto p : pads_strides_dilations)
                            for(auto tp : trans_output_pads)
                                for(auto il : in_layouts)
                                    for(auto fl : fil_layouts)
                                        for(auto ol : out_layouts)
                                            for(auto d : deterministics)
                                                for(auto tv : tensor_vects)
                                                    for(auto vl : vector_lengths)
                                                        for(auto ot : output_types)
                                                            for(auto iv : int8_vectorizes)
                                                            {
                                                                miopen::test::conv::conv_test_input input{};
                                                                input.batch_size             = b;
                                                                input.input_channels         = ic;
                                                                input.output_channels        = oc;
                                                                input.spatial_dim_elements   = s;
                                                                input.filter_dims            = f;
                                                                input.pads_strides_dilations = p;
                                                                input.trans_output_pads      = tp;
                                                                input.in_layout              = il;
                                                                input.fil_layout             = fl;
                                                                input.out_layout             = ol;
                                                                input.deterministic          = d;
                                                                input.tensor_vect            = tv;
                                                                input.vector_length          = vl;
                                                                input.output_type            = ot;
                                                                input.int8_vectorize         = iv;
                                                                input.do_forward             = true;
                                                                input.do_backward_data       = true;
                                                                input.do_backward_weights    = true;
                                                                cases.push_back(input);
                                                            }
    return cases;
}

} // namespace

template <class T>
struct conv2d_find2_test : miopen::test::conv::conv_test_base<T>
{
    void SetUp() override
    {
        miopen::test::conv::conv_test_base<T>::SetUp();
        // Force Find 2.0
        setenv("MIOPEN_FIND_ENFORCE", "4", 1);
    }
};

using GPU_conv_2d_find2_FP32 = conv2d_find2_test<float>;

TEST_P(GPU_conv_2d_find2_FP32, TestFP32) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_conv_2d_find2_FP32, ::testing::ValuesIn(GetDataset()));
