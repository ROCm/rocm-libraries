// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "conv2d_gtest.hpp"
#include "get_handle.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

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

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const std::string devName = handle.GetDeviceName();

    return (devName == "gfx900" || devName == "gfx906" || devName == "gfx908" ||
            devName == "gfx90a" || devName == "gfx942" || miopen::StartsWith(devName, "gfx103") ||
            miopen::StartsWith(devName, "gfx110") || miopen::StartsWith(devName, "gfx115"));
}

} // namespace

template <typename T>
struct miopen_conv_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2d_MIOpenTestConv_FP32 = miopen_conv_test<float>;

TEST_P(GPU_Conv2d_MIOpenTestConv_FP32, TestFloat)
{
    if(IsTestSupportedForDevice(get_handle()))
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(id, ...) \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_Conv2d_MIOpenTestConv_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(0, {1, 3, 32, 32}, {1, 3, 7, 7}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(1, {1, 3, 227, 227}, {1, 3, 7, 7}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(2, {1, 64, 56, 56}, {1, 64, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(3, {1, 3, 32, 32}, {1, 3, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(4, {1, 3, 224, 224}, {1, 3, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(5, {1, 3, 227, 227}, {1, 3, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(6, {1, 3, 231, 231}, {1, 3, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(7, {1, 3, 224, 224}, {1, 3, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(8, {1, 3, 227, 227}, {1, 3, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(9, {1, 3, 231, 231}, {1, 3, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(10, {1, 3, 32, 32}, {1, 3, 7, 7}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(11, {1, 3, 224, 224}, {1, 3, 7, 7}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(12, {1, 3, 227, 227}, {1, 3, 7, 7}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(13, {1, 3, 231, 231}, {1, 3, 7, 7}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(14, {1, 64, 56, 56}, {1, 64, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(15, {1, 64, 112, 112}, {1, 64, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(16, {1, 64, 512, 1024}, {1, 64, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(17, {1, 96, 27, 27}, {1, 96, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(18, {1, 96, 28, 28}, {1, 96, 3, 3}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(19, {1, 3, 32, 32}, {1, 3, 3, 3}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(20, {1, 3, 224, 224}, {1, 3, 3, 3}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(21, {1, 3, 227, 227}, {1, 3, 3, 3}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(22, {1, 3, 231, 231}, {1, 3, 3, 3}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(23, {1, 3, 32, 32}, {1, 3, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(24, {1, 3, 224, 224}, {1, 3, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(25, {1, 3, 227, 227}, {1, 3, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(26, {1, 3, 231, 231}, {1, 3, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(27, {1, 3, 32, 32}, {1, 3, 7, 7}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(28, {1, 3, 224, 224}, {1, 3, 7, 7}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(29, {1, 3, 227, 227}, {1, 3, 7, 7}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(30, {1, 3, 231, 231}, {1, 3, 7, 7}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(31, {1, 16, 14, 14}, {1, 16, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(32, {1, 16, 28, 28}, {1, 16, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(33, {1, 24, 14, 14}, {1, 24, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(34, {1, 32, 7, 7}, {1, 32, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(35, {1, 32, 8, 8}, {1, 32, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(36, {1, 32, 14, 14}, {1, 32, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(37, {1, 32, 16, 16}, {1, 32, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(38, {1, 32, 28, 28}, {1, 32, 5, 5}, {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(39, {1, 48, 7, 7}, {1, 48, 5, 5}, {0, 0, 4, 4, 1, 1});
