// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
// #include "gtest/gtest_common.hpp"
#include <gtest/gtest.h>
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
    baseParams.tolerance = {MIOPEN_OVERRIDDEN_TOLERANCE};
#endif

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

// template <typename T>
// auto GetCasesFull()
// {
//     static const auto cases = GenCases<T>(false);
//     return cases;
// }

// template <typename T>
// auto GetCasesSmoke()
// {
//     static const auto cases = GenCases<T>(true);
//     return cases;
// }

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

using MIOPEN_TESTSUITE_NAME(GPU_Conv2d_) = conv2d_test<MIOPEN_GTEST_DATA_TYPE>;

TEST_P(MIOPEN_TESTSUITE_NAME(GPU_Conv2d_), MIOPEN_TEST_INFO(Test))
{
    run();
}

INSTANTIATE_MIOPEN_TEST_SUITE(MIOPEN_TESTSUITE_PREFIX(0),
                              MIOPEN_TESTSUITE_NAME(GPU_Conv2d_),
                              {1, 16, 24, 24},
                              {16, 16, 7, 7},
                              {3, 3, 1, 1, 1, 1},
                              "transpose",
                              true,
                              false,
                              false);

// using GPU_Conv2d_FP16  = conv2d_test_base<half_float::half>;
// using GPU_Conv2d_FP32  = conv2d_test_base<float>;
// using GPU_Conv2d_FP64  = conv2d_test_base<double>;
// using GPU_Conv2d_I8    = conv2d_test_base<int8_t>;
// using GPU_Conv2d_BFP16 = conv2d_test_base<bfloat16>;

// TEST_P(GPU_Conv2d_FP16, TestFloat16)
// {
//     GetTestParams();
//     run();
// }

// TEST_P(GPU_Conv2d_FP32, TestFloat)
// {
//     GetTestParams();
//     run();
// }

// TEST_P(GPU_Conv2d_FP64, TestFloat64)
// {
//     GetTestParams();
//     run();
// }

// TEST_P(GPU_Conv2d_I8, TestInt8)
// {
//     GetTestParams();
//     run();
// }

// TEST_P(GPU_Conv2d_BFP16, TestBFloat16)
// {
//     GetTestParams();
//     run();
// }

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_Conv2d_FP16,
//                          GetCasesSmoke<half_float::half>(),
//                          DefaultTestNameGenerator<TestCase>{});
// INSTANTIATE_TEST_SUITE_P(Full,
//                          GPU_Conv2d_FP16,
//                          GetCasesFull<half_float::half>(),
//                          DefaultTestNameGenerator<TestCase>{});

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_Conv2d_FP32,
//                          GetCasesSmoke<float>(),
//                          DefaultTestNameGenerator<TestCase>{});
// INSTANTIATE_TEST_SUITE_P(Full,
//                          GPU_Conv2d_FP32,
//                          GetCasesFull<float>(),
//                          DefaultTestNameGenerator<TestCase>{});

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_Conv2d_FP64,
//                          GetCasesSmoke<double>(),
//                          DefaultTestNameGenerator<TestCase>{});
// INSTANTIATE_TEST_SUITE_P(Full,
//                          GPU_Conv2d_FP64,
//                          GetCasesFull<double>(),
//                          DefaultTestNameGenerator<TestCase>{});

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_Conv2d_I8,
//                          GetCasesSmoke<int8_t>(),
//                          DefaultTestNameGenerator<TestCase>{});
// INSTANTIATE_TEST_SUITE_P(Full,
//                          GPU_Conv2d_I8,
//                          GetCasesFull<int8_t>(),
//                          DefaultTestNameGenerator<TestCase>{});

// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_Conv2d_BFP16,
//                          GetCasesSmoke<bfloat16>(),
//                          DefaultTestNameGenerator<TestCase>{});
// INSTANTIATE_TEST_SUITE_P(Full,
//                          GPU_Conv2d_BFP16,
//                          GetCasesFull<bfloat16>(),
//                          DefaultTestNameGenerator<TestCase>{});
