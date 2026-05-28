// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gtest_common.hpp"
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
              bool enable_backward_data    = false,
              bool enable_backward_weights = false)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
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
    using d_mask = disabled<Gpu::gfx908>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

template <typename T>
struct conv_ck_igemm_fwd_v6r1_dlops_nchw : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16 =
    conv_ck_igemm_fwd_v6r1_dlops_nchw<half_float::half>;
using GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32 = conv_ck_igemm_fwd_v6r1_dlops_nchw<float>;

TEST_P(GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16, TestFloat16)
{
    if(IsTestSupportedForDevice())
    {
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvCkIgemmFwdV6r1DlopsNchw");
        ScopedEnvironment<int> conv_ck_igemm_fwd_v6r1_dlops_nchw(
            MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW, 1);

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

TEST_P(GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32, TestFloat)
{
    if(IsTestSupportedForDevice())
    {
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvCkIgemmFwdV6r1DlopsNchw");
        ScopedEnvironment<int> conv_ck_igemm_fwd_v6r1_dlops_nchw(
            MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW, 1);

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FULL_TEST_SUITE(id, ...)                                            \
    INSTANTIATE_MIOPEN_FULL_TEST(                                                              \
        id, GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16, half_float::half, __VA_ARGS__); \
    INSTANTIATE_MIOPEN_FULL_TEST(                                                              \
        id, GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FULL_TEST_SUITE(0, {128, 1024, 14, 14}, {2048, 1024, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(1, {128, 256, 14, 14}, {256, 1024, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(2, {128, 1024, 14, 14}, {512, 1024, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(3, {128, 128, 28, 28}, {128, 1024, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(4, {128, 128, 28, 28}, {512, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(5, {128, 128, 58, 58}, {128, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(6, {128, 2048, 7, 7}, {512, 2048, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(7, {128, 256, 14, 14}, {1024, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(8, {128, 256, 14, 14}, {256, 256, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(9, {128, 256, 30, 30}, {256, 256, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(10, {128, 256, 56, 56}, {128, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(11, {128, 256, 56, 56}, {512, 256, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(12, {128, 256, 56, 56}, {64, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(13, {128, 512, 16, 16}, {512, 512, 3, 3}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(14, {128, 512, 28, 28}, {1024, 512, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(15, {128, 512, 28, 28}, {128, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(16, {128, 512, 28, 28}, {256, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(17, {128, 512, 7, 7}, {2048, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(18, {128, 512, 7, 7}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(19, {128, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(20, {128, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_MIOPEN_FULL_TEST_SUITE(21, {128, 64, 56, 56}, {64, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
