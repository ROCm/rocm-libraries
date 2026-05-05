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
              bool enable_backward_data,
              bool enable_backward_weights,
              std::string in_layout  = "NCHW",
              std::string fil_layout = "NCHW",
              std::string out_layout = "NCHW",
              int group_count        = 1)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.in_layout                       = {std::move(in_layout)};
    baseParams.fil_layout                      = {std::move(fil_layout)};
    baseParams.out_layout                      = {std::move(out_layout)};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
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
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void SetEnvVars(std::vector<std::string>& envvars)
{
    for(auto& elem : envvars)
    {
        putenv(elem.data());
    }
}

} // namespace

template <typename T>
struct conv_igemm_mlir_xdlops_fwd : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2dMLIRTestIGemmXDLopsFwd_FP32 = conv_igemm_mlir_xdlops_fwd<float>;
using GPU_Conv2dMLIRTestIGemmXDLopsFwd_I8   = conv_igemm_mlir_xdlops_fwd<int8_t>;

TEST_P(GPU_Conv2dMLIRTestIGemmXDLopsFwd_FP32, TestFloat32)
{
    if(IsTestSupportedForDevice())
    {
        std::vector<std::string> env_vars{"MIOPEN_FIND_MODE=normal",
                                          "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwdXdlops"};

        SetEnvVars(env_vars);
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

TEST_P(GPU_Conv2dMLIRTestIGemmXDLopsFwd_I8, TestInt8)
{
    if(IsTestSupportedForDevice())
    {
        std::vector<std::string> env_vars{"MIOPEN_FIND_MODE=normal",
                                          "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwdXdlops"};

        SetEnvVars(env_vars);
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(id, ...)                                          \
    INSTANTIATE_MIOPEN_FULL_TEST(id, GPU_Conv2dMLIRTestIGemmXDLopsFwd_FP32, float, __VA_ARGS__); \
    INSTANTIATE_MIOPEN_FULL_TEST(id, GPU_Conv2dMLIRTestIGemmXDLopsFwd_I8, int8_t, __VA_ARGS__)

INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(
    0, {256, 1024, 14, 14}, {2048, 1024, 1, 1}, {0, 0, 2, 2, 1, 1}, false, false);
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(
    1, {256, 128, 28, 28}, {128, 128, 3, 3}, {1, 1, 1, 1, 1, 1}, false, false);
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(2,
                                       {256, 128, 28, 28},
                                       {128, 128, 3, 3},
                                       {1, 1, 1, 1, 1, 1},
                                       false,
                                       false,
                                       "NHWC",
                                       "NHWC",
                                       "NHWC");
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(3,
                                       {
                                           128,
                                           512,
                                           7,
                                           7,
                                       },
                                       {512, 512, 3, 3},
                                       {1, 1, 1, 1, 1, 1},
                                       false,
                                       false);
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(4,
                                       {
                                           128,
                                           512,
                                           7,
                                           7,
                                       },
                                       {512, 512, 3, 3},
                                       {1, 1, 1, 1, 1, 1},
                                       false,
                                       false,
                                       "NHWC",
                                       "NHWC",
                                       "NHWC");
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(
    5, {128, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1}, false, false);
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(
    6, {128, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1}, false, false, "NHWC", "NHWC", "NHWC");
INSTANTIATE_MIOPEN_FULL_FWD_TEST_SUITE(7,
                                       {256, 256, 56, 56},
                                       {256, 64, 1, 1},
                                       {0, 0, 1, 1, 1, 1},
                                       false,
                                       false,
                                       "NCHW",
                                       "NCHW",
                                       "NCHW",
                                       4);
