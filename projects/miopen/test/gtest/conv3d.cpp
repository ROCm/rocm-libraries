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

#if MIOPEN_GTEST_SUFFIX == FP16
using GPU_Conv3d_FP16 = conv3d_test<half_float::half>;
TEST_P(GPU_Conv3d_FP16, TestFloat16) { run(); }
#elif MIOPEN_GTEST_SUFFIX == FP32
using GPU_Conv3d_FP32 = conv3d_test<float>;
TEST_P(GPU_Conv3d_FP32, TestFloat32) { run(); }
#elif MIOPEN_GTEST_SUFFIX == BFP16
using GPU_Conv3d_BFP16 = conv3d_test<bfloat16>;
TEST_P(GPU_Conv3d_BFP16, TestBFloat16) { run(); }
#elif MIOPEN_GTEST_SUFFIX == I8
using GPU_Conv3d_I8 = conv3d_test<int8_t>;
TEST_P(GPU_Conv3d_I8, TestInt8) { run(); }
#else
#error "Unsupported test input data type"
#endif

#ifdef MIOPEN_GTEST_ALL
#if MIOPEN_GTEST_SUFFIX == FP16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_FP16,
                         GenCases<half_float::half>(false),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == FP32
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_FP32,
                         GenCases<float>(false),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == BFP16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_BFP16,
                         GenCases<bfloat16>(false),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == I8
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_I8,
                         GenCases<int8_t>(false),
                         DefaultTestNameGenerator<TestCase>{});
#endif
#else // MIOPEN_GTEST_ALL
#if MIOPEN_GTEST_SUFFIX == FP16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_FP16,
                         GenCases<half_float::half>(true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == FP32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_FP32,
                         GenCases<float>(true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == BFP16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_BFP16,
                         GenCases<bfloat16>(true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == I8
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_I8,
                         GenCases<int8_t>(true),
                         DefaultTestNameGenerator<TestCase>{});
#endif
#endif // MIOPEN_GTEST_ALL
