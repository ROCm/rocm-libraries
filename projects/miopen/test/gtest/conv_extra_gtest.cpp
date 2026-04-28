// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gtest/gtest_common.hpp>
#include <miopen/env.hpp>
#include <miopen/miopen.h>

#include "conv2d_gtest.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations,
              bool enable_forward = true)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_forward = {enable_forward};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X, Gpu::gfx115X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

template <typename T>
struct conv_extra_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_conv_extra_BFP16 = conv_extra_test<bfloat16>;
using GPU_conv_extra_FP16  = conv_extra_test<half_float::half>;
using GPU_conv_extra_FP32  = conv_extra_test<float>;
using GPU_conv_extra_I8    = conv_extra_test<int8_t>;

TEST_P(GPU_conv_extra_BFP16, TestBFloat16)
{
    if(IsTestSupportedForDevice())
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_conv_extra_FP16, TestFloat16)
{
    if(IsTestSupportedForDevice())
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_conv_extra_FP32, TestFloat32)
{
    if(IsTestSupportedForDevice())
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_conv_extra_I8, TestInt8)
{
    if(IsTestSupportedForDevice())
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

#define INSTANTIATE_ALL_MIOPEN_TEST_SUITES(id, ...)                                         \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_extra_BFP16, bfloat16, __VA_ARGS__);        \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_extra_FP16, half_float::half, __VA_ARGS__); \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_extra_FP32, float, __VA_ARGS__);            \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_extra_I8, int8_t, __VA_ARGS__)

#define INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(id, ...) \
    INSTANTIATE_MIOPEN_TEST_SUITES(id, GPU_conv_extra_FP32, float, __VA_ARGS__)

INSTANTIATE_ALL_MIOPEN_TEST_SUITES(0, {16, 3, 64, 128}, {96, 3, 11, 11}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(1, {16, 3, 32, 32}, {96, 3, 11, 11}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(2, {16, 3, 64, 128}, {96, 3, 11, 11}, {5, 5, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(3, {16, 3, 32, 32}, {96, 3, 11, 11}, {5, 5, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(4, {2, 16, 1024, 2048}, {32, 16, 3, 3}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(5, {2, 16, 1024, 2048}, {32, 16, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(6, {2, 16, 3072, 3072}, {32, 16, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(7, {2, 16, 3072, 3072}, {32, 16, 3, 3}, {2, 2, 2, 2, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(8, {128, 320, 1, 7}, {256, 320, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(9, {128, 1024, 1, 7}, {2048, 1024, 1, 1}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(10, {352, 192, 7, 1}, {320, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_MIOPEN_TEST_SUITES(11, {352, 16, 7, 1}, {32, 16, 1, 1}, {2, 2, 1, 1, 1, 1});

INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(12,
                                              {4, 1, 161, 700},
                                              {4, 1, 5, 20},
                                              {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(13,
                                              {4, 1, 161, 700},
                                              {4, 1, 5, 20},
                                              {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(14,
                                              {4, 32, 79, 341},
                                              {4, 32, 5, 10},
                                              {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(15,
                                              {4, 32, 79, 341},
                                              {4, 32, 5, 10},
                                              {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(16,
                                              {4, 3, 227, 227},
                                              {4, 3, 11, 11},
                                              {0, 0, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(17,
                                              {4, 3, 224, 224},
                                              {4, 3, 11, 11},
                                              {2, 2, 4, 4, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(18,
                                              {16, 1, 48, 480},
                                              {16, 1, 3, 3},
                                              {1, 1, 1, 1, 1, 1});
// Forward disabled since FFT fails verification for the forward direction
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(
    19, {32, 64, 27, 27}, {192, 64, 5, 5}, {2, 2, 1, 1, 1, 1}, false);
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(20,
                                              {4, 96, 14, 14},
                                              {32, 96, 5, 5},
                                              {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(21,
                                              {4, 16, 14, 14},
                                              {4, 16, 5, 5},
                                              {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SPECIFIC_TEST_SUITES(22,
                                              {4, 32, 14, 14},
                                              {4, 32, 5, 5},
                                              {2, 2, 1, 1, 1, 1});
