// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "conv_common_gtest.hpp"
#include <vector>

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
                         NamedParameter<bool>,                // use_input_dims
                         NamedParameter<bool>,                // use_weight_tensor_dims
                         NamedParameter<miopenDataType_t>,    // data_type,
                         NamedParameter<double>,              // tolerance,
                         TParams...>;

template <typename T, ConvApi api = ConvApi::Find_1_0>
struct Conv2DBaseTestParameters
{
    using ct = conv_test<T, Conv2DBaseTestCase<>, api>;

    Conv2DBaseTestParameters(bool smoke_test, int limit_set = 2)
        : batch_size(generate_data_limited(ct::get_batch_sizes(), 1, !smoke_test, limit_set)),
          input_channels(
              generate_data_limited(ct::get_input_channels(), 1, {32}, !smoke_test, limit_set)),
          output_channels(
              generate_data_limited(ct::get_output_channels(), 1, {64}, !smoke_test, limit_set)),
          spatial_dim_elements(generate_data_limited(
              ct::get_2d_spatial_dims(), 1, {28, 28}, !smoke_test, limit_set)),
          filter_dims(
              generate_data_limited(ct::get_2d_filter_dims(), 2, {3, 3}, !smoke_test, limit_set)),
          pads_strides_dilations(generate_data_limited(
              ct::get_2d_pads_strides_dilations(), 2, !smoke_test, limit_set)),
          trans_output_pads(
              smoke_test ? std::vector<std::vector<int>>{*ct::get_2d_trans_output_pads().begin()}
                         : ct::get_2d_trans_output_pads())
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
    std::vector<std::string> in_layout{std::string{"NCHW"}};
    std::vector<std::string> fil_layout{std::string{"NCHW"}};
    std::vector<std::string> out_layout{std::string{"NCHW"}};
    std::vector<bool> deterministic{false};
    std::vector<size_t> tensor_vect{0};
    std::vector<size_t> vector_length{1};
    std::vector<std::string> output_type{std::string{"int32"}};
    std::vector<bool> int8_vectorize{false};

    double tolerance{80.0f};
    miopenDataType_t input_data_type{miopenFloat};

    // Dummy values have to be supplied to avoid empty parameter lists in the test instantiations.
    std::vector<std::vector<std::size_t>> input_dims{{0}};
    std::vector<std::vector<std::size_t>> weight_tensor_dims{{0}};
    bool use_input_dims{false};
    bool use_weight_tensor_dims{false};
};

template <typename T, typename TestCase = Conv2DBaseTestCase<>, ConvApi api = ConvApi::Find_1_0>
struct conv2d_test_base : public conv_test<T, TestCase, api>
{
    void SetUp() override { prng::reset_seed(); }

    template <typename... TParams>
    void GetTestParams(TParams&... params)
    {
        conv_test<T, TestCase, api>::GetTestParams(this->batch_size,
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
                                                   this->int8_vectorize,
                                                   this->use_input_dims,
                                                   this->use_weight_tensor_dims,
                                                   this->input_data_type,
                                                   this->tolerance,
                                                   params...);
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
            MakeNamedParameterCollectionValues<bool>(
                "use_input_dims", std::vector<bool>{conv2dBaseParams.use_input_dims}),
            MakeNamedParameterCollectionValues<bool>(
                "use_weight_tensor_dims",
                std::vector<bool>{conv2dBaseParams.use_weight_tensor_dims}),
            MakeNamedParameterCollectionValues<miopenDataType_t>(
                "input_data_type", std::vector<miopenDataType_t>{conv2dBaseParams.input_data_type}),
            MakeNamedParameterCollectionValues<double>(
                "tolerance", std::vector<double>{conv2dBaseParams.tolerance}),
            std::forward<TParams>(params)...);
    }
};
