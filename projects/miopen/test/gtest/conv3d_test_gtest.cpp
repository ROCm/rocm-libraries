// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>

#include <miopen/miopen.h>

#include "conv3d_gtest.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"

namespace {

using TestCase = Conv3DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test)
{
    Conv3DBaseTestParameters<T> baseParams(smoke_test);

    const std::vector<std::vector<size_t>> input_dims_values{
        {2, 16, 50, 50, 50},
        {1, 16, 4, 161, 700},
        {1, 16, 4, 140, 602},
    };

    const std::vector<std::vector<size_t>> weight_tensor_dims_values{
        {32, 16, 5, 5, 5},
        {16, 16, 3, 11, 11},
    };

    baseParams.pads_strides_dilations = {
        {0, 0, 0, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 2, 2, 2, 1, 1, 1},
        {2, 2, 2, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 1, 1, 1, 2, 2, 2},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 1, 1, 1, 1, 1, 1},
    };

    auto input_dims = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "input_dims",
        smoke_test ? std::vector<std::vector<size_t>>{input_dims_values[0]} : input_dims_values);

    auto weight_tensor_dims = MakeNamedParameterCollectionValues<std::vector<size_t>>(
        "weight_tensor_dims",
        smoke_test ? std::vector<std::vector<size_t>>{weight_tensor_dims_values[0]}
                   : weight_tensor_dims_values);

    //     const std::string psd0 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";
    //     const std::string psd1 = " --pads_strides_dilations 0 0 0 2 2 2 1 1 1";
    //     const std::string psd2 = " --pads_strides_dilations 2 2 2 1 1 1 1 1 1";
    //     const std::string psd3 = " --pads_strides_dilations 0 0 0 1 1 1 2 2 2";
    //     const std::string psd4 = " --pads_strides_dilations 1 1 1 1 1 1 1 1 1";
    //     const std::string psd5 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";

    //     return {
    //         // clang-format off
    //     // test_conv3d_extra
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0},
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1},
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2},
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3},
    //     //test_conv3d_extra reduced set
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0 },
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1 },
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2 },
    //     {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3 },
    //     {precision + " --input 1 16 4 161 700 --weights 16 16 3 11 11" + psd4 },
    //     {precision + " --input 1 16 4 161 700 --weights 16 16 3 11 11" + psd5 },
    //     {precision + " --input 1 16 4 140 602 --weights 16 16 3 11 11" + psd4 },
    //     {precision + " --input 1 16 4 140 602 --weights 16 16 3 11 11" + psd5 }
    //         // clang-format on
    //     };

    return conv3d_test_base<T>::GenTestParams(baseParams, input_dims, weight_tensor_dims);
}

template <typename T>
auto GetCasesFull()
{
    static const auto cases = GenCases<T>(false);
    return cases;
}

template <typename T>
auto GetCasesSmoke()
{
    static const auto cases = GenCases<T>(true);
    return cases;
}

} // namespace

template <typename T>
struct conv3d_test : public conv3d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv3d_FP32 = conv3d_test<float>;

TEST_P(GPU_Conv3d_FP32, TestFloat) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3d_FP32,
                         GetCasesSmoke<float>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3d_FP32,
                         GetCasesFull<float>(),
                         DefaultTestNameGenerator<TestCase>{});
