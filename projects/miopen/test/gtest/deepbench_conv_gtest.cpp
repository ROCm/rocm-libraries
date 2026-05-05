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

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X, Gpu::gfx115X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

template <typename T>
struct deepbench_conv : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2d_DeepBench_FP32 = deepbench_conv<float>;

TEST_P(GPU_Conv2d_DeepBench_FP32, TestFloat32)
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

#define INSTANTIATE_MIOPEN_FULL_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_FULL_TEST(id, GPU_Conv2d_DeepBench_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FULL_TEST_SUITE(0, {4, 1, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(1, {8, 1, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(2, {16, 1, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(3, {32, 1, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(4, {4, 32, 79, 341}, {32, 32, 5, 10}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(5, {8, 32, 79, 341}, {32, 32, 5, 10}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(6, {16, 32, 79, 341}, {32, 32, 5, 10}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(7, {32, 32, 79, 341}, {32, 32, 5, 10}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(8, {16, 1, 48, 480}, {16, 1, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(9, {16, 16, 24, 240}, {32, 16, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(10, {16, 32, 12, 120}, {64, 32, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(11, {16, 64, 6, 60}, {128, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(12, {8, 3, 108, 108}, {64, 3, 3, 3}, {1, 1, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(13, {8, 64, 54, 54}, {64, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(14, {8, 128, 27, 27}, {128, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(15, {8, 128, 14, 14}, {256, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(16, {8, 256, 7, 7}, {512, 256, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(17, {8, 3, 224, 224}, {64, 3, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(18, {8, 64, 112, 112}, {128, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(19, {8, 128, 56, 56}, {256, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(20, {8, 256, 28, 28}, {512, 256, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(21, {8, 512, 14, 14}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(22, {8, 512, 7, 7}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(23, {16, 3, 224, 224}, {64, 3, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(24, {16, 64, 112, 112}, {128, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(25, {16, 128, 56, 56}, {256, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(26, {16, 256, 28, 28}, {512, 256, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(27, {16, 512, 14, 14}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(28, {16, 512, 7, 7}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(29, {16, 3, 224, 224}, {64, 3, 7, 7}, {3, 3, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(30, {16, 192, 28, 28}, {32, 192, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(31, {16, 512, 14, 14}, {48, 512, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(32, {16, 832, 7, 7}, {128, 832, 5, 5}, {2, 2, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(33, {16, 192, 28, 28}, {32, 192, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(34, {16, 512, 14, 14}, {48, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(35, {16, 832, 7, 7}, {128, 832, 1, 1}, {0, 0, 1, 1, 1, 1});
