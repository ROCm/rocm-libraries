// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "gtest/gtest_common.hpp"
#include <gtest/gtest.h>
#include "conv2d_gtest.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<>;

template <typename T>
auto GenCases(bool smoke_test)
{
    Conv2DBaseTestParameters<T> params(smoke_test);

    return conv2d_test_base<T>::GenTestParams(params);
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

using GPU_Conv2d_FP16  = conv2d_test_base<half_float::half>;
using GPU_Conv2d_FP32  = conv2d_test_base<float>;
using GPU_Conv2d_FP64  = conv2d_test_base<double>;
using GPU_Conv2d_I8    = conv2d_test_base<int8_t>;
using GPU_Conv2d_BFP16 = conv2d_test_base<bfloat16>;

TEST_P(GPU_Conv2d_FP16, TestFloat16)
{
    GetTestParams();
    run();
}

TEST_P(GPU_Conv2d_FP32, TestFloat)
{
    GetTestParams();
    run();
}

TEST_P(GPU_Conv2d_FP64, TestFloat64)
{
    GetTestParams();
    run();
}

TEST_P(GPU_Conv2d_I8, TestInt8)
{
    GetTestParams();
    run();
}

TEST_P(GPU_Conv2d_BFP16, TestBFloat16)
{
    GetTestParams();
    run();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP16,
                         GetCasesSmoke<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP16,
                         GetCasesFull<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP32,
                         GetCasesSmoke<float>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP32,
                         GetCasesFull<float>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP64,
                         GetCasesSmoke<double>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP64,
                         GetCasesFull<double>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_I8,
                         GetCasesSmoke<int8_t>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_I8,
                         GetCasesFull<int8_t>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_BFP16,
                         GetCasesSmoke<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_BFP16,
                         GetCasesFull<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
