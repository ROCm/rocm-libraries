// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "conv_common_gtest.hpp"
#include "gtest/conv2d.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

#ifdef MIOPEN_GTEST_ALL
bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    return (devName != "gfx900" && devName != "gfx906");
}

void SetEnvVars(std::vector<std::string>& envvars)
{
    for(auto& elem : envvars)
    {
        putenv(elem.data());
    }
}
#endif // MIOPEN_GTEST_ALL

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

using MIOPEN_TESTSUITE_NAME(GPU_Conv2d_) = conv2d_test<MIOPEN_GTEST_DATA_TYPE>;
using MIOPEN_TESTSUITE_NAME(GPU_Conv2d_RegressionIssue2624_) =
    conv2d_test_regression_issue_2624<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv2d_), MIOPEN_TEST_INFO(Test)) { run(); }

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv2d_RegressionIssue2624_),
       MIOPEN_TEST_INFO(TestRegressionIssue2624))
{
#ifndef MIOPEN_GTEST_ALL
    GTEST_SKIP()
        << "This test is being skipped as it is not intended to be run in 'smoke test' mode.";
#else  // MIOPEN_GTEST_ALL
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    std::vector<std::string> env_vars{"MIOPEN_DEBUG_CONV_WINOGRAD=0",
                                      "MIOPEN_DEBUG_CONV_FFT=0",
                                      "MIOPEN_DEBUG_CONV_DIRECT=0",
                                      "MIOPEN_DEBUG_CONV_GEMM=0",
                                      "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1"};

    SetEnvVars(env_vars);

    run();

    std::vector<std::string> deleted_env_vars{"MIOPEN_DEBUG_CONV_WINOGRAD=",
                                              "MIOPEN_DEBUG_CONV_FFT=",
                                              "MIOPEN_DEBUG_CONV_DIRECT=",
                                              "MIOPEN_DEBUG_CONV_GEMM=",
                                              "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM="};
    SetEnvVars(deleted_env_vars);
#endif // MIOPEN_GTEST_ALL
}

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_RegressionIssue2624_),
                              {2, 1, 22, 22},
                              {1, 1, 4, 4},
                              {1, 2, 4, 4, 3, 2},
                              "conv",
                              true,
                              false,
                              false);

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_),
                              {1, 16, 24, 24},
                              {16, 16, 7, 7},
                              {3, 3, 1, 1, 1, 1},
                              "transpose",
                              true,
                              false,
                              false);

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(1),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_),
                              {64, 64, 28, 28},
                              {64, 64, 1, 1},
                              {0, 0, 1, 1, 1, 1},
                              "transpose",
                              false,
                              true,
                              false);

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(2),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_),
                              {64, 64, 28, 28},
                              {64, 64, 1, 1},
                              {0, 0, 1, 1, 1, 1},
                              "transpose",
                              false,
                              false,
                              true);
