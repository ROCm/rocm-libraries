// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "conv_common_gtest.hpp"

template <typename... TParams>
using Conv2DBaseTestCase =
    ConvTestBaseTestCase<NamedParameter<size_t>,              // batch_size
                         NamedParameter<size_t>,              // input_channels
                         NamedParameter<size_t>,              // output_channels
                         NamedContainer<std::vector<size_t>>, // spatial_dim_elements
                         NamedContainer<std::vector<size_t>>, // filter_dims
                         NamedContainer<std::vector<int>>,    // pads_strides_dilations
                         NamedContainer<std::vector<int>>,    // trans_output_pads
                         NamedParameter<std::string>,         // in_layout
                         NamedParameter<std::string>,         // fil_layout
                         NamedParameter<std::string>,         // out_layout
                         NamedParameter<bool>,                // deterministic
                         NamedParameter<size_t>,              // tensor_vect
                         NamedParameter<size_t>,              // vector_length
                         NamedParameter<std::string>,         // output_type
                         NamedParameter<bool>,                // int8_vectorize
                         TParams...>;

template <typename T, ConvApi api = ConvApi::Find_1_0>
struct Conv2DBaseTestParameters
{
    using ct = conv_test<T>;

    Conv2DBaseTestParameters(bool smoke_test)
        : batch_size(generate_data_limited(ct::get_batch_sizes(), 1, !smoke_test)),
          input_channels(generate_data_limited(ct::get_input_channels(), 1, {32}, !smoke_test)),
          output_channels(generate_data_limited(ct::get_output_channels(), 1, {64}, !smoke_test)),
          spatial_dim_elements(
              generate_data_limited(ct::get_2d_spatial_dims(), 1, {28, 28}, !smoke_test)),
          filter_dims(generate_data_limited(ct::get_2d_filter_dims(), 2, {3, 3}, !smoke_test)),
          pads_strides_dilations(
              generate_data_limited(ct::get_2d_pads_strides_dilations(), 2, !smoke_test)),
          trans_output_pads(
              smoke_test ? std::vector<std::vector<int>>{*ct::get_2d_trans_output_pads().begin()}
                         : ct::get_2d_trans_output_pads()),
          in_layout({std::string{"NCHW"}}),
          fil_layout({std::string{"NCHW"}}),
          out_layout({std::string{"NCHW"}}),
          deterministic({false}),
          tensor_vect({0}),
          vector_length({1}),
          output_type({std::string{"int32"}}),
          int8_vectorize({false})
    {
    }

    BaseConvTestParameters<api> base_params;
    std::vector<size_t> batch_size;
    std::vector<size_t> input_channels;
    std::vector<size_t> output_channels;
    std::vector<std::vector<size_t>> spatial_dim_elements;
    std::vector<std::vector<size_t>> filter_dims;
    std::vector<std::vector<int>> pads_strides_dilations;
    std::vector<std::vector<int>> trans_output_pads;
    std::vector<std::string> in_layout;
    std::vector<std::string> fil_layout;
    std::vector<std::string> out_layout;
    std::vector<bool> deterministic;
    std::vector<size_t> tensor_vect;
    std::vector<size_t> vector_length;
    std::vector<std::string> output_type;
    std::vector<bool> int8_vectorize;
};

template <class T, ConvApi api = ConvApi::Find_1_0>
struct conv2d_test : public conv_test<T, api>, public testing::TestWithParam<Conv2DBaseTestCase<>>
{
    void SetUp() override
    {
        prng::reset_seed();

        this->GetTestParams(this->GetParam(),
                            this->batch_size,
                            this->input_channels,
                            this->output_channels,
                            this->spatial_dim_elements,
                            this->filter_dims,
                            this->pads_strides_dilations,
                            this->trans_output_pads,
                            this->in_layout,
                            this->fil_layout,
                            this->out_layout,
                            this->deterministic,
                            this->tensor_vect,
                            this->vector_length,
                            this->output_type,
                            this->int8_vectorize);
    }

    template <typename... TParams>
    static auto GenTestParams(const Conv2DBaseTestParameters<T, api>& conv2dBaseParams,
                              TParams&&... params)
    {
        return Conv2DBaseTestParameters<T, api>::ct::GenTestParams(
            conv2dBaseParams.base_params,
            MakeNamedParameterCollectionValues<size_t>("batch_size", conv2dBaseParams.batch_size),
            MakeNamedParameterCollectionValues<size_t>("input_channels",
                                                       conv2dBaseParams.input_channels),
            MakeNamedParameterCollectionValues<size_t>("output_channels",
                                                       conv2dBaseParams.output_channels),
            MakeNamedParameterCollectionValues<std::vector<size_t>>(
                "spatial_dim_elements", conv2dBaseParams.spatial_dim_elements),
            MakeNamedParameterCollectionValues<std::vector<size_t>>("filter_dims",
                                                                    conv2dBaseParams.filter_dims),
            MakeNamedParameterCollectionValues<std::vector<int>>(
                "pads_strides_dilations", conv2dBaseParams.pads_strides_dilations),
            MakeNamedParameterCollectionValues<std::vector<int>>(
                "trans_output_pads", conv2dBaseParams.trans_output_pads),
            MakeNamedParameterCollectionValues<std::string>("in_layout",
                                                            conv2dBaseParams.in_layout),
            MakeNamedParameterCollectionValues<std::string>("fil_layout",
                                                            conv2dBaseParams.fil_layout),
            MakeNamedParameterCollectionValues<std::string>("out_layout",
                                                            conv2dBaseParams.out_layout),
            MakeNamedParameterCollectionValues<bool>("deterministic",
                                                     conv2dBaseParams.deterministic),
            MakeNamedParameterCollectionValues<size_t>("tensor_vect", conv2dBaseParams.tensor_vect),
            MakeNamedParameterCollectionValues<size_t>("vector_length",
                                                       conv2dBaseParams.vector_length),
            MakeNamedParameterCollectionValues<std::string>("output_type",
                                                            conv2dBaseParams.output_type),
            MakeNamedParameterCollectionValues<bool>("int8_vectorize",
                                                     conv2dBaseParams.int8_vectorize),
            std::forward<TParams>(params)...);
    }
};
