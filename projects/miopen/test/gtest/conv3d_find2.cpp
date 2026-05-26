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
                                      NamedParameter<std::string>,         // out_layout
                                      NamedParameter<bool>,                // deterministic
                                      NamedParameter<size_t>,              // tensor_vect
                                      NamedParameter<size_t>,              // vector_length
                                      NamedParameter<std::string>,         // output_type
                                      NamedParameter<bool>                 // int8_vectorize
                                      >;

// Dummy unused arguments have been inserted to make the 'INSTANTIATE_MIOPEN_TEST_SUITE' macro
// happy.
template <typename T, typename... Args>
auto GenCases(bool smoke_test, Args&&...)
{
    using ct = conv_test<T, TestCase, ConvApi::Find_2_0>;
    BaseConvTestParameters<ConvApi::Find_2_0> baseParams;

    auto batch_size = MakeNamedParameterCollectionValues<size_t>(
        "batch_size", generate_data_limited(ct::get_batch_sizes(), 1, {8}, !smoke_test));

    auto input_channels = MakeNamedParameterCollectionValues<size_t>(
        "input_channels", generate_data_limited(ct::get_input_channels(), 1, {32}, !smoke_test));

    auto output_channels = MakeNamedParameterCollectionValues<size_t>(
        "output_channels", generate_data_limited(ct::get_output_channels(), 1, {32}, !smoke_test));

    auto spatial_dim_elements = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "spatial_dim_elements",
        generate_data_limited(ct::get_3d_spatial_dims(), 1, {16, 16, 16}, !smoke_test));

    auto filter_dims = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "filter_dims", generate_data_limited(ct::get_3d_filter_dims(), 2, {3, 3, 3}, !smoke_test));

    auto pads_strides_dilations = MakeNamedParameterCollectionValues<std::vector<int>>(
        "pads_strides_dilations",
        generate_data_limited(ct::get_3d_pads_strides_dilations(), 2, !smoke_test));

    auto trans_output_pads = MakeNamedParameterCollectionValues<std::vector<int>>(
        "trans_output_pads", generate_data_limited(ct::get_3d_trans_output_pads(), 1, !smoke_test));

    auto in_layout     = MakeNamedParameterValues<std::string>("in_layout", std::string{"NCDHW"});
    auto fil_layout    = MakeNamedParameterValues<std::string>("fil_layout", std::string{"NCDHW"});
    auto out_layout    = MakeNamedParameterValues<std::string>("out_layout", std::string{"NCDHW"});
    auto deterministic = MakeNamedParameterValues<bool>("deterministic", false);
    auto tensor_vect   = MakeNamedParameterValues<size_t>("tensor_vect", 0);
    auto vector_length = MakeNamedParameterValues<size_t>("vector_length", 1);

    // Only valid for int8 input and weights
    auto output_type = MakeNamedParameterValues<std::string>("output_type", std::string{"int32"});
    auto int8_vectorize = MakeNamedParameterValues<bool>("int8_vectorize", false);

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
                             out_layout,
                             deterministic,
                             tensor_vect,
                             vector_length,
                             output_type,
                             int8_vectorize);
}

} // namespace

template <class T>
struct conv3d_find2_test : public conv_test<T, TestCase, ConvApi::Find_2_0>
{
    void SetUp() override
    {
        prng::reset_seed();

        this->GetTestParams(this->batch_size,
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
};

using MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Find2_) = conv3d_find2_test<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Find2_), MIOPEN_TEST_INFO(Test)) { run(); }

// The last argument is a dummy argument, just to make the 'INSTANTIATE_MIOPEN_TEST_SUITE' macro
// happy. It is not used in the test.
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Find2_),
                              0);
