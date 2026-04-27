// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "conv2d_gtest.hpp"
#include "gtest/gtest_common.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations,
              size_t group_count,
              bool enable_backward_weights = true)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};
    baseParams.base_params.groupCount          = {group_count};

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
struct conv_group_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_ConvGroup_FP32 = conv_group_test<float>;

TEST_P(GPU_ConvGroup_FP32, TestFloat32)
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

#define INSTANTIATE_FLOAT_SUITES(id, ...) \
    INSTANTIATE_TEST_SUITES(id, GPU_ConvGroup_FP32, float, __VA_ARGS__)

INSTANTIATE_FLOAT_SUITES(0, {16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(1, {16, 256, 56, 56}, {512, 8, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(2, {16, 256, 28, 28}, {512, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(3, {16, 512, 28, 28}, {1024, 16, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(4, {16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(5, {16, 1024, 14, 14}, {2048, 32, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(6, {16, 1024, 7, 7}, {2048, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(7, {32, 128, 56, 56}, {256, 4, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(8, {32, 256, 56, 56}, {512, 8, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
//
// Workaround for "Memory access fault by GPU node" during "HIP Release All" - WrW disabled.
INSTANTIATE_FLOAT_SUITES(9, {32, 256, 28, 28}, {512, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false);
INSTANTIATE_FLOAT_SUITES(10, {32, 512, 28, 28}, {1024, 16, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(11, {32, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(12, {32, 1024, 14, 14}, {2048, 32, 3, 3}, {1, 1, 2, 2, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(13, {32, 1024, 7, 7}, {2048, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 32);
INSTANTIATE_FLOAT_SUITES(14, {4, 4, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 4);
INSTANTIATE_FLOAT_SUITES(15, {8, 2, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(16, {16, 4, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 4);
INSTANTIATE_FLOAT_SUITES(17, {32, 2, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(18, {4, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(19, {8, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(20, {16, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(21, {32, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(22, {16, 4, 48, 480}, {16, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 4);
INSTANTIATE_FLOAT_SUITES(23, {16, 16, 24, 240}, {32, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 16);
INSTANTIATE_FLOAT_SUITES(24, {16, 32, 12, 120}, {64, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 4);
INSTANTIATE_FLOAT_SUITES(25, {16, 64, 6, 60}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 4);
INSTANTIATE_FLOAT_SUITES(26, {8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(27, {8, 64, 54, 54}, {64, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 8);
INSTANTIATE_FLOAT_SUITES(28, {8, 128, 27, 27}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 8);
INSTANTIATE_FLOAT_SUITES(29, {8, 3, 224, 224}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(30, {8, 64, 112, 112}, {128, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 2);
INSTANTIATE_FLOAT_SUITES(31, {16, 9, 224, 224}, {63, 3, 3, 3}, {1, 1, 1, 1, 1, 1}, 3);
//
// Workaround for "Memory access fault by GPU node" during "FP32 gfx908 Hip Release All subset" -
// WrW disabled.
INSTANTIATE_FLOAT_SUITES(32, {16, 64, 112, 112}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 4, false);
INSTANTIATE_FLOAT_SUITES(33, {16, 3, 224, 224}, {63, 1, 7, 7}, {3, 3, 2, 2, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(34, {16, 192, 28, 28}, {32, 12, 5, 5}, {2, 2, 1, 1, 1, 1}, 16);
INSTANTIATE_FLOAT_SUITES(35, {16, 832, 7, 7}, {128, 52, 5, 5}, {2, 2, 1, 1, 1, 1}, 16);
INSTANTIATE_FLOAT_SUITES(36, {16, 192, 28, 28}, {32, 24, 1, 1}, {0, 0, 1, 1, 1, 1}, 8);
INSTANTIATE_FLOAT_SUITES(37, {16, 832, 7, 7}, {128, 104, 1, 1}, {0, 0, 1, 1, 1, 1}, 8);
INSTANTIATE_FLOAT_SUITES(38, {11, 23, 161, 700}, {46, 1, 7, 7}, {1, 1, 2, 2, 1, 1}, 23);
INSTANTIATE_FLOAT_SUITES(39, {8, 7, 224, 224}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 7);
INSTANTIATE_FLOAT_SUITES(40, {8, 7, 224, 224}, {63, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, 7);
INSTANTIATE_FLOAT_SUITES(41, {8, 7, 224, 224}, {63, 1, 3, 3}, {0, 0, 2, 2, 1, 1}, 7);
INSTANTIATE_FLOAT_SUITES(42, {8, 7, 224, 224}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 7);
INSTANTIATE_FLOAT_SUITES(43, {8, 7, 224, 224}, {63, 1, 3, 3}, {2, 2, 2, 2, 1, 1}, 7);
INSTANTIATE_FLOAT_SUITES(44, {8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(45, {8, 3, 108, 108}, {63, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(46, {8, 3, 108, 108}, {63, 1, 3, 3}, {0, 0, 2, 2, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(47, {8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 3);
INSTANTIATE_FLOAT_SUITES(48, {8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2, 2, 2, 1, 1}, 3);
