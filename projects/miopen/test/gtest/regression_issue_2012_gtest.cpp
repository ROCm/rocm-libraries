// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
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
              std::string conv_mode        = "conv",
              std::string pad_mode         = "default",
              int group_count              = 1,
              bool enable_forward          = false,
              bool enable_backward_data    = false,
              bool enable_backward_weights = false)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.conv_mode           = {std::move(conv_mode)};
    baseParams.base_params.pad_mode            = {std::move(pad_mode)};
    baseParams.base_params.groupCount          = {group_count};
    baseParams.base_params.do_forward          = {enable_forward};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

template <typename T>
struct regression_issue_2012 : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_regression_issue_2012_FP32 = regression_issue_2012<float>;

TEST_P(GPU_regression_issue_2012_FP32, TestFloat)
{
    if(IsTestSupportedForDevice())
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_SMOKE_TEST(id, GPU_regression_issue_2012_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(0, {128, 832, 7, 7}, {32, 832, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(1,
                                          {64, 192, 28, 28},
                                          {64, 192, 1, 1},
                                          {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(2,
                                          {64, 256, 28, 28},
                                          {128, 256, 1, 1},
                                          {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(3,
                                          {64, 480, 14, 14},
                                          {64, 480, 1, 1},
                                          {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(4,
                                          {64, 512, 14, 14},
                                          {128, 512, 1, 1},
                                          {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(5,
                                          {64, 512, 28, 28},
                                          {128, 512, 1, 1},
                                          {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(6, {64, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
