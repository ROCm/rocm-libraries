// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv3d_gtest.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>

namespace {

using TestCase = Conv3DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations,
              size_t groupCount,
              std::string cmode,
              std::string pmode)
{
    Conv3DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};
    baseParams.base_params.groupCount = {groupCount};
    baseParams.base_params.conv_mode  = {std::move(cmode)};
    baseParams.base_params.pad_mode   = {std::move(pmode)};

    return conv3d_test_base<T>::GenTestParams(
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
struct conv_3d_test : public conv3d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv_3d_FP32 = conv_3d_test<float>;

TEST_P(GPU_Conv_3d_FP32, TestFloat)
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

INSTANTIATE_TEST_SUITES(0,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 32, 4, 9, 9},
                        {64, 32, 3, 3, 3},
                        {0, 0, 0, 2, 2, 2, 1, 1, 1},
                        1,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(1,
                        GPU_Conv_3d_FP32,
                        float,
                        {4, 3, 4, 227, 227},
                        {4, 3, 3, 11, 11},
                        {0, 0, 0, 1, 1, 1, 1, 1, 1},
                        1,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(2,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 128, 4, 56, 56},
                        {256, 4, 3, 3, 3},
                        {1, 1, 1, 1, 1, 1, 1, 1, 1},
                        32,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(3,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 128, 56, 56, 56},
                        {256, 4, 3, 3, 3},
                        {1, 2, 3, 1, 1, 1, 1, 2, 3},
                        32,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(4,
                        GPU_Conv_3d_FP32,
                        float,
                        {4, 4, 4, 161, 700},
                        {32, 1, 3, 5, 20},
                        {1, 1, 1, 2, 2, 2, 1, 1, 1},
                        4,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(5,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 4, 28, 28},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 1, 1, 1, 1, 1, 1},
                        4,
                        "conv",
                        "same");
INSTANTIATE_TEST_SUITES(6,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 4, 56, 56},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 2, 2, 2, 1, 1, 1},
                        4,
                        "conv",
                        "same");
INSTANTIATE_TEST_SUITES(7,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 3, 14, 14},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 2, 2, 2, 1, 1, 1},
                        1,
                        "trans",
                        "same");
INSTANTIATE_TEST_SUITES(8,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 64, 3, 4, 4},
                        {64, 32, 1, 3, 3},
                        {0, 0, 0, 2, 2, 2, 1, 1, 1},
                        4,
                        "trans",
                        "default");
INSTANTIATE_TEST_SUITES(9,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 32, 4, 9, 9},
                        {64, 32, 3, 3, 3},
                        {0, 0, 0, 1, 2, 3, 1, 2, 3},
                        1,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(10,
                        GPU_Conv_3d_FP32,
                        float,
                        {4, 3, 4, 227, 227},
                        {4, 3, 3, 11, 11},
                        {0, 0, 0, 1, 1, 1, 1, 2, 3},
                        1,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(11,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 128, 4, 56, 56},
                        {256, 4, 3, 3, 3},
                        {1, 2, 3, 1, 1, 1, 1, 2, 3},
                        32,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(12,
                        GPU_Conv_3d_FP32,
                        float,
                        {4, 4, 4, 161, 700},
                        {32, 1, 3, 5, 20},
                        {1, 2, 3, 1, 2, 3, 1, 2, 3},
                        4,
                        "conv",
                        "default");
INSTANTIATE_TEST_SUITES(13,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 4, 28, 28},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 1, 1, 1, 1, 2, 3},
                        4,
                        "conv",
                        "same");
INSTANTIATE_TEST_SUITES(14,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 4, 56, 56},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 1, 2, 3, 1, 2, 3},
                        4,
                        "conv",
                        "same");
INSTANTIATE_TEST_SUITES(15,
                        GPU_Conv_3d_FP32,
                        float,
                        {8, 512, 3, 14, 14},
                        {512, 128, 1, 1, 1},
                        {0, 0, 0, 1, 2, 3, 1, 2, 3},
                        1,
                        "trans",
                        "same");
INSTANTIATE_TEST_SUITES(16,
                        GPU_Conv_3d_FP32,
                        float,
                        {16, 64, 3, 4, 4},
                        {64, 32, 1, 3, 3},
                        {0, 0, 0, 1, 2, 3, 1, 2, 3},
                        4,
                        "trans",
                        "default");
