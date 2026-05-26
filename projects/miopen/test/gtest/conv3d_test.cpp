// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>

#include <miopen/miopen.h>

#include "gtest/conv3d.hpp"

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

using MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_) = conv3d_test<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_), MIOPEN_TEST_INFO(Test)) { run(); }

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {2, 16, 50, 50, 50},
                              {32, 16, 5, 5, 5},
                              {0, 0, 0, 1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(1),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {2, 16, 50, 50, 50},
                              {32, 16, 5, 5, 5},
                              {0, 0, 0, 2, 2, 2, 1, 1, 1});
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(2),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {2, 16, 50, 50, 50},
                              {32, 16, 5, 5, 5},
                              {2, 2, 2, 1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(3),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {2, 16, 50, 50, 50},
                              {32, 16, 5, 5, 5},
                              {0, 0, 0, 1, 1, 1, 2, 2, 2});
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(4),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {1, 16, 4, 161, 700},
                              {16, 16, 3, 11, 11},
                              {1, 1, 1, 1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(5),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv3d_Test_),
                              {1, 16, 4, 140, 602},
                              {16, 16, 3, 11, 11},
                              {0, 0, 0, 1, 1, 1, 1, 1, 1});
