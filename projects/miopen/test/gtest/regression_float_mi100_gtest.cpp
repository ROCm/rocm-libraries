// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv2d_gtest.hpp"
#include "gtest_common.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations,
              bool enable_forward,
              bool enable_backward_weights)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_forward          = {enable_forward};
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
struct regression_float_mi100 : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2d_regression_mi100_FP32 = regression_float_mi100<float>;

TEST_P(GPU_Conv2d_regression_mi100_FP32, TestFloat)
{
    if(IsTestSupportedForDevice())
    {
        // Regression test for SWDEV-305815 (issue 1206)
        ScopedEnvironment<bool> debug_conv_wino_grad(MIOPEN_DEBUG_CONV_WINOGRAD, false);
        ScopedEnvironment<bool> debug_conv_fft(MIOPEN_DEBUG_CONV_FFT, false);
        ScopedEnvironment<bool> debug_conv_direct(MIOPEN_DEBUG_CONV_DIRECT, false);
        ScopedEnvironment<bool> debug_conv_gemm(MIOPEN_DEBUG_CONV_GEMM, false);
        ScopedEnvironment<bool> debug_conv_implicit_gemm(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, false);
        ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 1);

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_SMOKE_TEST(id, GPU_Conv2d_regression_mi100_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(
    0, {32, 256, 38, 38}, {256, 256, 1, 1}, {0, 0, 1, 1, 1, 1}, false, false);
