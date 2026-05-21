// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "conv_common_gtest.hpp"

template <typename... TParams>
using Conv3DBaseTestCase =
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
                         NamedParameter<double>,              // tolerance,
                         TParams...>;

template <typename T, ConvApi api = ConvApi::Find_1_0>
struct Conv3DBaseTestParameters
{
    using ct = conv_test<T, Conv3DBaseTestCase<>, api>;

    Conv3DBaseTestParameters(bool smoke_test)
        : batch_size(generate_data_limited(ct::get_batch_sizes(), 1, {8}, !smoke_test)),
          input_channels(generate_data_limited(ct::get_input_channels(), 1, {32}, !smoke_test)),
          output_channels(generate_data_limited(ct::get_output_channels(), 1, {32}, !smoke_test)),
          spatial_dim_elements(
              generate_data_limited(ct::get_3d_spatial_dims(), 1, {16, 16, 16}, !smoke_test)),
          filter_dims(generate_data_limited(ct::get_3d_filter_dims(), 2, {3, 3, 3}, !smoke_test)),
          pads_strides_dilations(
              generate_data_limited(ct::get_3d_pads_strides_dilations(), 2, !smoke_test)),
          trans_output_pads(generate_data_limited(ct::get_3d_trans_output_pads(), 1, !smoke_test))
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
    std::vector<double> tolerance{80.0f};
};

template <typename T, typename TestCase = Conv3DBaseTestCase<>, ConvApi api = ConvApi::Find_1_0>
struct conv3d_test_base : public conv_test<T, TestCase, api>
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
                                                   this->tolerance,
                                                   params...);
    }

    template <typename... TParams>
    static auto GenTestParams(const Conv3DBaseTestParameters<T, api>& conv3dBaseParams,
                              TParams&&... params)
    {
        return Conv3DBaseTestParameters<T, api>::ct::GenTestParams(
            conv3dBaseParams.base_params,
            MakeNamedParameterCollectionValues<size_t>("batch_size", conv3dBaseParams.batch_size),
            MakeNamedParameterCollectionValues<size_t>("input_channels",
                                                       conv3dBaseParams.input_channels),
            MakeNamedParameterCollectionValues<size_t>("output_channels",
                                                       conv3dBaseParams.output_channels),
            MakeNamedParameterCollectionValues<std::vector<size_t>>(
                "spatial_dim_elements", conv3dBaseParams.spatial_dim_elements),
            MakeNamedParameterCollectionValues<std::vector<size_t>>("filter_dims",
                                                                    conv3dBaseParams.filter_dims),
            MakeNamedParameterCollectionValues<std::vector<int>>(
                "pads_strides_dilations", conv3dBaseParams.pads_strides_dilations),
            MakeNamedParameterCollectionValues<std::vector<int>>(
                "trans_output_pads", conv3dBaseParams.trans_output_pads),
            MakeNamedParameterCollectionValues<std::string>("in_layout",
                                                            conv3dBaseParams.in_layout),
            MakeNamedParameterCollectionValues<std::string>("fil_layout",
                                                            conv3dBaseParams.fil_layout),
            MakeNamedParameterCollectionValues<std::string>("out_layout",
                                                            conv3dBaseParams.out_layout),
            MakeNamedParameterCollectionValues<bool>("deterministic",
                                                     conv3dBaseParams.deterministic),
            MakeNamedParameterCollectionValues<size_t>("tensor_vect", conv3dBaseParams.tensor_vect),
            MakeNamedParameterCollectionValues<size_t>("vector_length",
                                                       conv3dBaseParams.vector_length),
            MakeNamedParameterCollectionValues<std::string>("output_type",
                                                            conv3dBaseParams.output_type),
            MakeNamedParameterCollectionValues<bool>("int8_vectorize",
                                                     conv3dBaseParams.int8_vectorize),
            MakeNamedParameterCollectionValues<double>("tolerance", conv3dBaseParams.tolerance),
            std::forward<TParams>(params)...);
    }
};
