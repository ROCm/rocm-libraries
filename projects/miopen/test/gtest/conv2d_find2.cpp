// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "gtest/conv2d.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations,
              std::string conv_mode,
              bool enable_forward,
              bool enable_backward_data,
              bool enable_backward_weights)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.conv_mode           = {std::move(conv_mode)};
    baseParams.base_params.do_forward          = {enable_forward};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

} // namespace

template <typename T>
struct conv2d_find2_test : public conv2d_test_base<T, TestCase, ConvApi::Find_2_0>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using MIOPEN_TESTSUITE_NAME(GPU_Conv2d_Find2_) = conv2d_find2_test<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv2d_Find2_), MIOPEN_TEST_INFO(Test)) { run(); }

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_Find2_),
                              {1, 16, 24, 24},
                              {16, 16, 7, 7},
                              {3, 3, 1, 1, 1, 1},
                              "transpose",
                              true,
                              false,
                              false);

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(1),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_Find2_),
                              {64, 64, 28, 28},
                              {64, 64, 1, 1},
                              {0, 0, 1, 1, 1, 1},
                              "transpose",
                              false,
                              true,
                              false);

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(2),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_Find2_),
                              {64, 64, 28, 28},
                              {64, 64, 1, 1},
                              {0, 0, 1, 1, 1, 1},
                              "transpose",
                              false,
                              false,
                              true);
