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
              std::vector<int> pads_strides_dilations,
              std::string conv_mode,
              std::string pad_mode,
              int group_count = 1)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};
    baseParams.base_params.conv_mode  = {std::move(conv_mode)};
    baseParams.base_params.pad_mode   = {std::move(pad_mode)};
    baseParams.base_params.groupCount = {group_count};

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
struct conv_trans_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_conv_trans_FP32 = conv_trans_test<float>;

TEST_P(GPU_conv_trans_FP32, TestFloat)
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
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_trans_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    0, {8, 128, 28, 28}, {128, 128, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "default");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    1, {8, 256, 28, 28}, {256, 256, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "same");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    2, {8, 32, 28, 28}, {32, 32, 5, 5}, {0, 0, 2, 2, 1, 1}, "trans", "default");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    3, {8, 512, 14, 14}, {512, 512, 1, 1}, {0, 0, 2, 2, 1, 1}, "trans", "same");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    4, {8, 512, 4, 4}, {512, 512, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "valid");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    5, {8, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 2, 2, 1, 1}, "trans", "valid");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    6, {100, 3, 64, 64}, {3, 3, 1, 1}, {2, 2, 1, 1, 1, 1}, "trans", "default");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    7, {100, 6, 4, 4}, {6, 4, 1, 1}, {2, 2, 1, 1, 1, 1}, "trans", "default");
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    8, {8, 128, 28, 28}, {128, 16, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "default", 8);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    9, {8, 256, 28, 28}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "same", 4);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    10, {8, 32, 28, 28}, {32, 1, 5, 5}, {0, 0, 2, 2, 1, 1}, "trans", "default", 32);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    11, {8, 512, 14, 14}, {512, 16, 1, 1}, {0, 0, 2, 2, 1, 1}, "trans", "same", 32);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    12, {8, 512, 4, 4}, {512, 16, 1, 1}, {0, 0, 1, 1, 1, 1}, "trans", "valid", 32);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    13, {8, 64, 56, 56}, {64, 2, 1, 1}, {0, 0, 2, 2, 1, 1}, "trans", "valid", 32);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    14, {100, 3, 64, 64}, {3, 3, 1, 1}, {2, 2, 1, 1, 1, 1}, "trans", "default", 3);
INSTANTIATE_MIOPEN_FLOAT_TEST_SUITES(
    15, {100, 6, 4, 4}, {6, 4, 1, 1}, {2, 2, 1, 1, 1, 1}, "trans", "default", 2);
