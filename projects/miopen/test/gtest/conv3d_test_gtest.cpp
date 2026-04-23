// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>

#include <miopen/miopen.h>

#include "conv3d_gtest.hpp"

namespace {

using TestCase = Conv3DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations)
{
    Conv3DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};

    return conv3d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

} // namespace

template <typename T>
struct conv3d_test : public conv3d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv3d_FP32 = conv3d_test<float>;

TEST_P(GPU_Conv3d_FP32, TestFloat) { run(); }

INSTANTIATE_TEST_SUITES(
    0, GPU_Conv3d_FP32, float, {2, 16, 50, 50, 50}, {32, 16, 5, 5, 5}, {0, 0, 0, 1, 1, 1, 1, 1, 1});
INSTANTIATE_TEST_SUITES(
    1, GPU_Conv3d_FP32, float, {2, 16, 50, 50, 50}, {32, 16, 5, 5, 5}, {0, 0, 0, 2, 2, 2, 1, 1, 1});
INSTANTIATE_TEST_SUITES(
    2, GPU_Conv3d_FP32, float, {2, 16, 50, 50, 50}, {32, 16, 5, 5, 5}, {2, 2, 2, 1, 1, 1, 1, 1, 1});
INSTANTIATE_TEST_SUITES(
    3, GPU_Conv3d_FP32, float, {2, 16, 50, 50, 50}, {32, 16, 5, 5, 5}, {0, 0, 0, 1, 1, 1, 2, 2, 2});
INSTANTIATE_TEST_SUITES(4,
                        GPU_Conv3d_FP32,
                        float,
                        {1, 16, 4, 161, 700},
                        {16, 16, 3, 11, 11},
                        {1, 1, 1, 1, 1, 1, 1, 1, 1});
INSTANTIATE_TEST_SUITES(5,
                        GPU_Conv3d_FP32,
                        float,
                        {1, 16, 4, 140, 602},
                        {16, 16, 3, 11, 11},
                        {0, 0, 0, 1, 1, 1, 1, 1, 1});
