// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "get_handle.hpp"
#include "conv2d_gtest.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              bool enable_forward,
              bool enable_backward_data,
              bool enable_backward_weights,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::string output_type,
              std::string in_layout,
              std::string fil_layout,
              std::string out_layout,
              std::vector<int> pads_strides_dilations)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.output_type                     = {std::move(output_type)};
    baseParams.base_params.do_forward          = {enable_forward};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};
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
    const std::string devName = handle.GetDeviceName();
    return (devName == "gfx908" || devName == "gfx90a" || devName == "gfx942");
}

} // namespace

template <typename T>
struct conv_hip_igemm_xdlops_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_ConvHipIgemmXdlops_I8 = conv_hip_igemm_xdlops_test<int8_t>;

TEST_P(GPU_ConvHipIgemmXdlops_I8, TestInt8)
{
#if MIOPEN_BACKEND_OPENCL
    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";
#else // MIOPEN_BACKEND_HIP, OCL_DISABLED
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(IsTestSupportedForDevice(get_handle()))
    {
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
#else
    GTEST_SKIP() << "MIOPEN_USE_COMPOSABLEKERNEL needed for this test";
#endif
#endif
}

#define INSTANTIATE_INT8_TEST_SUITES(id, ...) \
    INSTANTIATE_TEST_SUITES(id, GPU_ConvHipIgemmXdlops_I8, int8_t, __VA_ARGS__)

INSTANTIATE_INT8_TEST_SUITES(0,
                             true,
                             false,
                             false,
                             {256, 128, 28, 28},
                             {128, 128, 3, 3},
                             "int8",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(1,
                             true,
                             false,
                             false,
                             {128, 512, 7, 7},
                             {512, 512, 3, 3},
                             "int8",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(2,
                             true,
                             false,
                             false,
                             {128, 64, 56, 56},
                             {64, 64, 1, 1},
                             "int8",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(3,
                             true,
                             false,
                             false,
                             {256, 256, 56, 56},
                             {256, 64, 1, 1},
                             "int8",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});

INSTANTIATE_INT8_TEST_SUITES(4,
                             true,
                             false,
                             false,
                             {256, 128, 28, 28},
                             {128, 128, 3, 3},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(5,
                             true,
                             false,
                             false,
                             {128, 512, 7, 7},
                             {512, 512, 3, 3},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(6,
                             true,
                             false,
                             false,
                             {128, 64, 56, 56},
                             {64, 64, 1, 1},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(7,
                             true,
                             false,
                             false,
                             {256, 256, 56, 56},
                             {256, 64, 1, 1},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(8,
                             true,
                             false,
                             false,
                             {256, 128, 28, 28},
                             {128, 128, 3, 3},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(9,
                             true,
                             false,
                             false,
                             {128, 512, 7, 7},
                             {512, 512, 3, 3},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(10,
                             true,
                             false,
                             false,
                             {128, 64, 56, 56},
                             {64, 64, 1, 1},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(11,
                             true,
                             false,
                             false,
                             {256, 256, 56, 56},
                             {256, 64, 1, 1},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});

INSTANTIATE_INT8_TEST_SUITES(12,
                             false,
                             true,
                             false,
                             {256, 128, 28, 28},
                             {128, 128, 3, 3},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(13,
                             false,
                             true,
                             false,
                             {128, 512, 7, 7},
                             {512, 512, 3, 3},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(14,
                             false,
                             true,
                             false,
                             {128, 64, 56, 56},
                             {64, 64, 1, 1},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(15,
                             false,
                             true,
                             false,
                             {256, 256, 56, 56},
                             {256, 64, 1, 1},
                             "fp32",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(16,
                             false,
                             true,
                             false,
                             {256, 128, 28, 28},
                             {128, 128, 3, 3},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(17,
                             false,
                             true,
                             false,
                             {128, 512, 7, 7},
                             {512, 512, 3, 3},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {1, 1, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(18,
                             false,
                             true,
                             false,
                             {128, 64, 56, 56},
                             {64, 64, 1, 1},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
INSTANTIATE_INT8_TEST_SUITES(19,
                             false,
                             true,
                             false,
                             {256, 256, 56, 56},
                             {256, 64, 1, 1},
                             "fp16",
                             "NHWC",
                             "NHWC",
                             "NHWC",
                             {0, 0, 1, 1, 1, 1});
