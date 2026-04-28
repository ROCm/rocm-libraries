// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>

#include "conv2d_gtest.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(IMPLICITGEMM_TESTING_ENV)

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    return (devName == "gfx900" || devName == "gfx906" || devName == "gfx908" ||
            devName == "gfx90a" || devName == "gfx942" || miopen::StartsWith(devName, "gfx103") ||
            miopen::StartsWith(devName, "gfx110") || miopen::StartsWith(devName, "gfx115"));
}

void SetEnvVars(std::vector<std::string>& envvars)
{
    for(auto& elem : envvars)
    {
        putenv(elem.data());
    }
}

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

} // namespace

template <typename T>
struct conv_for_implicit_gemm_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_ConvImplicitGemm_BFP16 = conv_for_implicit_gemm_test<bfloat16>;
using GPU_ConvImplicitGemm_FP16  = conv_for_implicit_gemm_test<half_float::half>;

TEST_P(GPU_ConvImplicitGemm_BFP16, TestBFloat16)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    if(!env::enabled(IMPLICITGEMM_TESTING_ENV))
    {
        GTEST_SKIP()
            << "IMPLICITGEMM_TESTING_ENV environment variable is not enabled. Skipping the test.";
    }

    std::vector<std::string> env_vars{"MIOPEN_FIND_MODE=normal",
                                      "MIOPEN_DEBUG_CONV_WINOGRAD=0",
                                      "MIOPEN_DEBUG_CONV_GEMM=0",
                                      "MIOPEN_DEBUG_CONV_DIRECT=0",
                                      "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1",
                                      "MIOPEN_DEBUG_CONV_FFT=0"};

    SetEnvVars(env_vars);
    testing::internal::CaptureStderr();
    run();
    const auto capture = testing::internal::GetCapturedStderr();
    EXPECT_EQ(capture.find("No suitable algorithm was found"), std::string::npos);
};

TEST_P(GPU_ConvImplicitGemm_FP16, TestFloat16)
{
    if(!IsTestSupportedForDevice(get_handle()))
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

    if(!env::enabled(IMPLICITGEMM_TESTING_ENV))
    {
        GTEST_SKIP()
            << "IMPLICITGEMM_TESTING_ENV environment variable is not enabled. Skipping the test.";
    }

    std::vector<std::string> env_vars{"MIOPEN_FIND_MODE=normal",
                                      "MIOPEN_DEBUG_CONV_WINOGRAD=0",
                                      "MIOPEN_DEBUG_CONV_GEMM=0",
                                      "MIOPEN_DEBUG_CONV_DIRECT=0",
                                      "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1",
                                      "MIOPEN_DEBUG_CONV_FFT=0"};

    SetEnvVars(env_vars);
    testing::internal::CaptureStderr();
    run();
    const auto capture = testing::internal::GetCapturedStderr();
    EXPECT_EQ(capture.find("No suitable algorithm was found"), std::string::npos);
};

#define INSTANTIATE_ALL_MIOPEN_TEST_SUITES(id, ...)                                        \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_ConvImplicitGemm_BFP16, bfloat16, __VA_ARGS__); \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_ConvImplicitGemm_FP16, half_float::half, __VA_ARGS__)

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(0, {64, 16, 28, 28}, {192, 16, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(1, {64, 16, 14, 14}, {160, 16, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(2, {64, 16, 7, 7}, {128, 16, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(3, {64, 16, 55, 55}, {96, 16, 1, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(4, {64, 16, 28, 28}, {64, 16, 1, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(5, {64, 16, 14, 14}, {32, 16, 1, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(6, {64, 32, 28, 28}, {192, 32, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(7, {64, 32, 14, 14}, {160, 32, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(8, {64, 32, 7, 7}, {128, 32, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(9, {64, 32, 55, 55}, {96, 32, 1, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(10, {64, 32, 28, 28}, {64, 32, 1, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(11, {64, 32, 14, 14}, {32, 32, 1, 7}, {0, 0, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(12, {64, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(13, {64, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(14, {64, 64, 73, 73}, {80, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(15, {64, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(16, {64, 128, 55, 55}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(17, {64, 128, 28, 28}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(18, {64, 128, 14, 14}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(19, {64, 128, 7, 7}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(20, {16, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(21, {16, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(22, {16, 64, 73, 73}, {80, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(23, {16, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(24, {16, 128, 55, 55}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(25, {16, 128, 28, 28}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(26, {16, 128, 7, 7}, {16, 128, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(27, {64, 64, 55, 55}, {16, 128, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(28, {64, 128, 28, 28}, {16, 128, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(29, {64, 128, 14, 14}, {16, 128, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(30, {64, 128, 7, 7}, {16, 128, 1, 1}, {0, 0, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(31, {64, 128, 28, 28}, {512, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(32, {64, 160, 73, 73}, {64, 160, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(33, {64, 192, 35, 35}, {32, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(34, {64, 192, 35, 35}, {48, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(35, {64, 192, 35, 35}, {64, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(36, {64, 192, 28, 28}, {16, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(37, {64, 192, 28, 28}, {32, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(38, {64, 192, 28, 28}, {64, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(39, {64, 192, 28, 28}, {96, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(40, {64, 256, 35, 35}, {48, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(41, {64, 256, 35, 35}, {64, 256, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(42, {64, 256, 56, 56}, {128, 256, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(43, {64, 256, 56, 56}, {512, 256, 1, 1}, {0, 0, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(44, {64, 256, 56, 56}, {64, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(45, {64, 256, 28, 28}, {128, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(46, {64, 256, 28, 28}, {32, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(47, {64, 256, 28, 28}, {64, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(48, {64, 288, 35, 35}, {48, 288, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(49, {64, 288, 35, 35}, {64, 288, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(50, {64, 384, 35, 35}, {192, 384, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(51, {64, 384, 35, 35}, {64, 384, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(52, {64, 384, 35, 35}, {96, 384, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(53, {64, 480, 14, 14}, {16, 480, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(54, {64, 480, 14, 14}, {192, 480, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(55, {64, 480, 14, 14}, {64, 480, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(56, {64, 480, 14, 14}, {96, 480, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(57, {64, 512, 28, 28}, {128, 512, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(58, {64, 512, 28, 28}, {256, 512, 1, 1}, {0, 0, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(59, {64, 512, 14, 14}, {112, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(60, {64, 512, 14, 14}, {128, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(61, {64, 512, 14, 14}, {144, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(62, {64, 512, 14, 14}, {160, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(63, {64, 512, 14, 14}, {24, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(64, {64, 512, 14, 14}, {32, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(65, {64, 512, 14, 14}, {64, 512, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(66, {128, 832, 7, 7}, {32, 832, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(67, {128, 832, 7, 7}, {192, 832, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(68, {128, 832, 7, 7}, {128, 832, 1, 1}, {0, 0, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(69, {128, 832, 7, 7}, {32, 832, 1, 1}, {0, 0, 1, 1, 2, 2});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(70, {128, 832, 7, 7}, {192, 832, 1, 1}, {0, 0, 1, 1, 2, 2});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(71, {128, 832, 7, 7}, {128, 832, 1, 1}, {0, 0, 1, 1, 2, 2});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(72, {16, 2048, 7, 7}, {192, 2048, 1, 1}, {0, 0, 1, 1, 2, 2});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(73, {64, 32, 28, 28}, {192, 32, 3, 3}, {1, 1, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(74, {8, 16, 14, 14}, {32, 16, 1, 1}, {1, 1, 1, 1, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(75, {64, 32, 14, 14}, {192, 32, 3, 3}, {1, 1, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(76, {64, 32, 7, 7}, {192, 32, 3, 3}, {1, 1, 2, 2, 1, 1});

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(77, {64, 32, 28, 28}, {192, 32, 3, 3}, {2, 2, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(78, {64, 32, 14, 14}, {192, 32, 3, 3}, {2, 2, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(79, {64, 32, 7, 7}, {192, 32, 3, 3}, {2, 2, 2, 2, 1, 1});
