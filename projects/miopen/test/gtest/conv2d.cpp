// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef MIOPEN_GTEST_ALL

#include <gtest/gtest.h>

#include "conv_common_gtest.hpp"
#include "gtest/conv2d.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    return (devName != "gfx900" && devName != "gfx906");
}

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

#ifdef MIOPEN_OVERRIDDEN_TOLERANCE
    baseParams.base_params.tolerance = {MIOPEN_OVERRIDDEN_TOLERANCE};
#endif

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

} // namespace

template <typename T>
struct conv2d_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

template <typename T>
struct conv2d_test_regression_issue_2624 : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

#if MIOPEN_GTEST_SUFFIX == FP16
using GPU_Conv2d_FP16                     = conv2d_test<half_float::half>;
using GPU_Conv2d_RegressionIssue2624_FP16 = conv2d_test_regression_issue_2624<half_float::half>;
TEST_P(GPU_Conv2d_FP16, TestFloat16) { run(); }
TEST_P(GPU_Conv2d_RegressionIssue2624_FP16, TestRegressionIssue2624Float16)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    ScopedEnvironment<bool> winograd(MIOPEN_DEBUG_CONV_WINOGRAD, false);
    ScopedEnvironment<bool> fft(MIOPEN_DEBUG_CONV_FFT, false);
    ScopedEnvironment<bool> direct(MIOPEN_DEBUG_CONV_DIRECT, false);
    ScopedEnvironment<bool> gemm(MIOPEN_DEBUG_CONV_GEMM, false);
    ScopedEnvironment<bool> implicit_gemm(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, true);

    run();
}
#elif MIOPEN_GTEST_SUFFIX == FP32
using GPU_Conv2d_FP32                     = conv2d_test<float>;
using GPU_Conv2d_RegressionIssue2624_FP32 = conv2d_test_regression_issue_2624<float>;
TEST_P(GPU_Conv2d_FP32, TestFloat32) { run(); }
TEST_P(GPU_Conv2d_RegressionIssue2624_FP32, TestRegressionIssue2624Float32)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    ScopedEnvironment<bool> winograd(MIOPEN_DEBUG_CONV_WINOGRAD, false);
    ScopedEnvironment<bool> fft(MIOPEN_DEBUG_CONV_FFT, false);
    ScopedEnvironment<bool> direct(MIOPEN_DEBUG_CONV_DIRECT, false);
    ScopedEnvironment<bool> gemm(MIOPEN_DEBUG_CONV_GEMM, false);
    ScopedEnvironment<bool> implicit_gemm(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, true);

    run();
}
#elif MIOPEN_GTEST_SUFFIX == BFP16
using GPU_Conv2d_BFP16                     = conv2d_test<bfloat16>;
using GPU_Conv2d_RegressionIssue2624_BFP16 = conv2d_test_regression_issue_2624<bfloat16>;
TEST_P(GPU_Conv2d_BFP16, TestBFloat16) { run(); }
TEST_P(GPU_Conv2d_RegressionIssue2624_BFP16, TestRegressionIssue2624BFloat16)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    ScopedEnvironment<bool> winograd(MIOPEN_DEBUG_CONV_WINOGRAD, false);
    ScopedEnvironment<bool> fft(MIOPEN_DEBUG_CONV_FFT, false);
    ScopedEnvironment<bool> direct(MIOPEN_DEBUG_CONV_DIRECT, false);
    ScopedEnvironment<bool> gemm(MIOPEN_DEBUG_CONV_GEMM, false);
    ScopedEnvironment<bool> implicit_gemm(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, true);

    run();
}
#elif MIOPEN_GTEST_SUFFIX == I8
using GPU_Conv2d_I8                     = conv2d_test<int8_t>;
using GPU_Conv2d_RegressionIssue2624_I8 = conv2d_test_regression_issue_2624<int8_t>;
TEST_P(GPU_Conv2d_I8, TestInt8) { run(); }
TEST_P(GPU_Conv2d_RegressionIssue2624_I8, TestRegressionIssue2624Int8)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    ScopedEnvironment<bool> winograd(MIOPEN_DEBUG_CONV_WINOGRAD, false);
    ScopedEnvironment<bool> fft(MIOPEN_DEBUG_CONV_FFT, false);
    ScopedEnvironment<bool> direct(MIOPEN_DEBUG_CONV_DIRECT, false);
    ScopedEnvironment<bool> gemm(MIOPEN_DEBUG_CONV_GEMM, false);
    ScopedEnvironment<bool> implicit_gemm(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, true);

    run();
}
#else
#error "Unsupported test input data type"
#endif

#if MIOPEN_GTEST_SUFFIX == FP16
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_Conv2d_RegressionIssue2624_FP16,
    GenCases<half_float::half>(
        false, {2, 1, 22, 22}, {1, 1, 4, 4}, {1, 2, 4, 4, 3, 2}, "conv", true, false, false),
    DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_0,
                         GPU_Conv2d_FP16,
                         GenCases<half_float::half>(false,
                                                    {1, 16, 24, 24},
                                                    {16, 16, 7, 7},
                                                    {3, 3, 1, 1, 1, 1},
                                                    "transpose",
                                                    true,
                                                    false,
                                                    false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_1,
                         GPU_Conv2d_FP16,
                         GenCases<half_float::half>(false,
                                                    {64, 64, 28, 28},
                                                    {64, 64, 1, 1},
                                                    {0, 0, 1, 1, 1, 1},
                                                    "transpose",
                                                    false,
                                                    true,
                                                    false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_2,
                         GPU_Conv2d_FP16,
                         GenCases<half_float::half>(false,
                                                    {64, 64, 28, 28},
                                                    {64, 64, 1, 1},
                                                    {0, 0, 1, 1, 1, 1},
                                                    "transpose",
                                                    false,
                                                    false,
                                                    true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == FP32
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_Conv2d_RegressionIssue2624_FP32,
    GenCases<float>(
        false, {2, 1, 22, 22}, {1, 1, 4, 4}, {1, 2, 4, 4, 3, 2}, "conv", true, false, false),
    DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_0,
                         GPU_Conv2d_FP32,
                         GenCases<float>(false,
                                         {1, 16, 24, 24},
                                         {16, 16, 7, 7},
                                         {3, 3, 1, 1, 1, 1},
                                         "transpose",
                                         true,
                                         false,
                                         false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_1,
                         GPU_Conv2d_FP32,
                         GenCases<float>(false,
                                         {64, 64, 28, 28},
                                         {64, 64, 1, 1},
                                         {0, 0, 1, 1, 1, 1},
                                         "transpose",
                                         false,
                                         true,
                                         false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_2,
                         GPU_Conv2d_FP32,
                         GenCases<float>(false,
                                         {64, 64, 28, 28},
                                         {64, 64, 1, 1},
                                         {0, 0, 1, 1, 1, 1},
                                         "transpose",
                                         false,
                                         false,
                                         true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == BFP16
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_Conv2d_RegressionIssue2624_BFP16,
    GenCases<bfloat16>(
        false, {2, 1, 22, 22}, {1, 1, 4, 4}, {1, 2, 4, 4, 3, 2}, "conv", true, false, false),
    DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_0,
                         GPU_Conv2d_BFP16,
                         GenCases<bfloat16>(false,
                                            {1, 16, 24, 24},
                                            {16, 16, 7, 7},
                                            {3, 3, 1, 1, 1, 1},
                                            "transpose",
                                            true,
                                            false,
                                            false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_1,
                         GPU_Conv2d_BFP16,
                         GenCases<bfloat16>(false,
                                            {64, 64, 28, 28},
                                            {64, 64, 1, 1},
                                            {0, 0, 1, 1, 1, 1},
                                            "transpose",
                                            false,
                                            true,
                                            false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_2,
                         GPU_Conv2d_BFP16,
                         GenCases<bfloat16>(false,
                                            {64, 64, 28, 28},
                                            {64, 64, 1, 1},
                                            {0, 0, 1, 1, 1, 1},
                                            "transpose",
                                            false,
                                            false,
                                            true),
                         DefaultTestNameGenerator<TestCase>{});
#elif MIOPEN_GTEST_SUFFIX == I8
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_Conv2d_I8,
    GenCases<int8_t>(
        false, {2, 1, 22, 22}, {1, 1, 4, 4}, {1, 2, 4, 4, 3, 2}, "conv", true, false, false),
    DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_0,
                         GPU_Conv2d_I8,
                         GenCases<int8_t>(false,
                                          {1, 16, 24, 24},
                                          {16, 16, 7, 7},
                                          {3, 3, 1, 1, 1, 1},
                                          "transpose",
                                          true,
                                          false,
                                          false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_1,
                         GPU_Conv2d_I8,
                         GenCases<int8_t>(false,
                                          {64, 64, 28, 28},
                                          {64, 64, 1, 1},
                                          {0, 0, 1, 1, 1, 1},
                                          "transpose",
                                          false,
                                          true,
                                          false),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Full_2,
                         GPU_Conv2d_I8,
                         GenCases<int8_t>(false,
                                          {64, 64, 28, 28},
                                          {64, 64, 1, 1},
                                          {0, 0, 1, 1, 1, 1},
                                          "transpose",
                                          false,
                                          false,
                                          true),
                         DefaultTestNameGenerator<TestCase>{});
#endif

#endif // MIOPEN_GTEST_ALL
