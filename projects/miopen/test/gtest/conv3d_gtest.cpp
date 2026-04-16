// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "conv3d_gtest.hpp"

namespace {

using TestCase = Conv3DBaseTestCase<>;

template <typename T>
auto GenCases(bool smoke_test)
{
    return conv3d_test<T>::GenTestParams(Conv3DBaseTestParameters<T>(smoke_test));
}

template <typename T>
auto GetCasesFull()
{
    static const auto cases = GenCases<T>(false);
    return cases;
}

template <typename T>
auto GetCasesSmoke()
{
    static const auto cases = GenCases<T>(true);
    return cases;
}

} // namespace

using GPU_Conv3d_FP32  = conv3d_test<float>;
using GPU_Conv3d_FP16  = conv3d_test<half_float::half>;

TEST_P(GPU_Conv3d_FP32, TestFloat) { run(); }
TEST_P(GPU_Conv3d_FP16, TestFloat16) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_FP32,
                         GetCasesSmoke<float>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_FP32,
                         GetCasesFull<float>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_FP16,
                         GetCasesSmoke<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_FP16,
                         GetCasesFull<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});
