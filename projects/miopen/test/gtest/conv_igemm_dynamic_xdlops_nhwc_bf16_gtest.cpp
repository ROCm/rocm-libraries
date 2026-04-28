// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "conv2d_gtest.hpp"
#include "get_handle.hpp"
#include "lib_env_var.hpp"

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
              bool enable_backward_data,
              bool enable_backward_weights,
              std::string in_layout,
              std::string fil_layout,
              std::string out_layout)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};
    baseParams.base_params.do_forward          = {enable_forward};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
    baseParams.in_layout                       = {std::move(in_layout)};
    baseParams.fil_layout                      = {std::move(fil_layout)};
    baseParams.out_layout                      = {std::move(out_layout)};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const auto& target        = handle.GetTargetProperties();
    const std::string devName = handle.GetDeviceName();

    if(target.isXnackEnabled())
    {
        return false;
    }

    return (devName == "gfx90a" || devName == "gfx942");
}

} // namespace

template <typename T>
struct conv_igemm_dynamic_xdlops_nhwc_bf16_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv_IGEMM_DYNAMIC_XDLOPS_NHWC_BF16 = conv_igemm_dynamic_xdlops_nhwc_bf16_test<bfloat16>;

TEST_P(GPU_Conv_IGEMM_DYNAMIC_XDLOPS_NHWC_BF16, TestBFloat16)
{
    if(IsTestSupportedForDevice(get_handle()))
    {
        ScopedEnvironment<std::string> find_mode_env1(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver_env(
            MIOPEN_DEBUG_FIND_ONLY_SOLVER,
            "ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC;ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC;"
            "ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC");

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

#define INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(id, ...) \
    INSTANTIATE_MIOPEN_TEST_SUITES(                      \
        id, GPU_Conv_IGEMM_DYNAMIC_XDLOPS_NHWC_BF16, bfloat16, __VA_ARGS__)

// fwd
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(0,
                                        {64, 256, 7, 7},
                                        {128, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(1,
                                        {32, 160, 73, 73},
                                        {64, 160, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(2,
                                        {16, 64, 56, 56},
                                        {64, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(3,
                                        {2, 256, 40, 52},
                                        {256, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(4,
                                        {2, 64, 59, 57},
                                        {12, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(5,
                                        {32, 128, 14, 14},
                                        {64, 128, 1, 1},
                                        {0, 0, 2, 2, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(6,
                                        {64, 64, 17, 17},
                                        {192, 64, 1, 7},
                                        {0, 3, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(7,
                                        {64, 64, 17, 17},
                                        {192, 64, 7, 1},
                                        {3, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(8,
                                        {4, 128, 28, 28},
                                        {128, 128, 2, 2},
                                        {0, 0, 2, 2, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(9,
                                        {32, 128, 8, 8},
                                        {192, 128, 3, 1},
                                        {1, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(10,
                                        {64, 192, 17, 17},
                                        {160, 192, 3, 3},
                                        {0, 0, 2, 2, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(11,
                                        {64, 32, 73, 73},
                                        {64, 32, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(12,
                                        {16, 64, 56, 56},
                                        {64, 64, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(13,
                                        {64, 3, 78, 78},
                                        {64, 3, 7, 7},
                                        {0, 0, 2, 2, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(14,
                                        {16, 192, 17, 17},
                                        {224, 192, 1, 7},
                                        {0, 3, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(15,
                                        {16, 3, 17, 17},
                                        {64, 3, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        true,
                                        false,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");

// nhwc_bwd
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(16,
                                        {64, 256, 7, 7},
                                        {128, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(17,
                                        {32, 160, 73, 73},
                                        {64, 160, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(18,
                                        {16, 64, 56, 56},
                                        {64, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(19,
                                        {2, 256, 40, 52},
                                        {256, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(20,
                                        {2, 64, 32, 28},
                                        {64, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(21,
                                        {32, 128, 14, 14},
                                        {64, 128, 1, 1},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(22,
                                        {64, 64, 17, 17},
                                        {192, 64, 1, 7},
                                        {0, 3, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(23,
                                        {64, 64, 17, 17},
                                        {192, 64, 7, 1},
                                        {3, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(24,
                                        {4, 128, 28, 28},
                                        {128, 128, 2, 2},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(25,
                                        {32, 128, 8, 8},
                                        {192, 128, 3, 1},
                                        {1, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(26,
                                        {64, 192, 17, 17},
                                        {160, 192, 3, 3},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(27,
                                        {64, 32, 73, 73},
                                        {64, 32, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(28,
                                        {16, 64, 56, 56},
                                        {64, 64, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(29,
                                        {16, 16, 25, 25},
                                        {64, 16, 3, 3},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(30,
                                        {15, 256, 1, 1},
                                        {340, 256, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(31,
                                        {15, 128, 10, 10},
                                        {340, 128, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        true,
                                        false,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");

// nhwc_wrw
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(32,
                                        {64, 256, 7, 7},
                                        {128, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(33,
                                        {32, 160, 73, 73},
                                        {64, 160, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(34,
                                        {16, 64, 56, 56},
                                        {64, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(35,
                                        {2, 256, 40, 52},
                                        {256, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(36,
                                        {2, 64, 32, 28},
                                        {64, 64, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(37,
                                        {32, 128, 14, 14},
                                        {64, 128, 1, 1},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(38,
                                        {64, 64, 17, 17},
                                        {192, 64, 1, 7},
                                        {0, 3, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(39,
                                        {64, 64, 17, 17},
                                        {192, 64, 7, 1},
                                        {3, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(40,
                                        {4, 128, 28, 28},
                                        {128, 128, 2, 2},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(41,
                                        {32, 128, 8, 8},
                                        {192, 128, 3, 1},
                                        {1, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(42,
                                        {64, 192, 17, 17},
                                        {160, 192, 3, 3},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(43,
                                        {64, 32, 73, 73},
                                        {64, 32, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(44,
                                        {16, 64, 56, 56},
                                        {64, 64, 3, 3},
                                        {1, 1, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(45,
                                        {16, 16, 25, 25},
                                        {64, 16, 3, 3},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(46,
                                        {4, 32, 79, 141},
                                        {64, 32, 5, 10},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(47,
                                        {400, 256, 7, 7},
                                        {1024, 256, 7, 7},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(48,
                                        {400, 256, 1, 1},
                                        {1024, 256, 1, 1},
                                        {0, 0, 1, 1, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(49,
                                        {1, 3, 32, 32},
                                        {1, 3, 11, 11},
                                        {1, 1, 2, 2, 2, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(50,
                                        {1, 3, 224, 224},
                                        {1, 3, 3, 3},
                                        {0, 0, 1, 1, 2, 2},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(
    51, {1, 1, 8, 8}, {1, 1, 2, 2}, {0, 0, 1, 1, 2, 2}, false, false, true, "NHWC", "NHWC", "NHWC");
INSTANTIATE_MIOPEN_BFLOAT16_TEST_SUITES(52,
                                        {1, 128, 56, 56},
                                        {1, 128, 5, 5},
                                        {0, 0, 2, 2, 1, 1},
                                        false,
                                        false,
                                        true,
                                        "NHWC",
                                        "NHWC",
                                        "NHWC");
