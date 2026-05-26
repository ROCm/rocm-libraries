// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "gtest/conv3d.hpp"

namespace {

using TestCase = Conv3DBaseTestCase<>;

// Dummy unused arguments have been inserted to make the 'INSTANTIATE_MIOPEN_TEST_SUITE' macro
// happy.
template <typename T, typename... Args>
auto GenCases(bool smoke_test, Args&&...)
{
    return conv3d_test_base<T>::GenTestParams(Conv3DBaseTestParameters<T>(smoke_test));
}

} // namespace

template <typename T>
struct conv3d_test : public conv3d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams();
    }
};

using MIOPEN_TESTSUITE_NAME(GPU_Conv3d_) = conv3d_test<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv3d_), MIOPEN_TEST_INFO(Test)) { run(); }

// The last argument is a dummy argument, just to make the 'INSTANTIATE_MIOPEN_TEST_SUITE' macro
// happy. It is not used in the test.
INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0), MIOPEN_TESTSUITE_NAME(GPU_Conv3d_), 0);
