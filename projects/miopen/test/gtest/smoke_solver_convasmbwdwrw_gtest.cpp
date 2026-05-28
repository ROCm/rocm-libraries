// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv2d_gtest.hpp"
#include "gtest_common.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS)

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
              bool enable_backward_data)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations       = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_forward       = {enable_forward};
    baseParams.base_params.do_backward_data = {enable_backward_data};

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
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

template <typename T>
struct smoke_solver_convasmbwdwrw : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2dSingleAsmBwdWrw_CompilerRegression_FP32 = smoke_solver_convasmbwdwrw<float>;
using GPU_Conv2dSingleAsmBwdWrw_FP32                    = smoke_solver_convasmbwdwrw<float>;
using GPU_Conv2dSingleAsmBwdWrw_FP16  = smoke_solver_convasmbwdwrw<half_float::half>;
using GPU_Conv2dSingleAsmBwdWrw_BFP16 = smoke_solver_convasmbwdwrw<bfloat16>;

TEST_P(GPU_Conv2dSingleAsmBwdWrw_FP32, TestFloat32)
{
    if(IsTestSupportedForDevice() && !get_handle_xnack())
    {
        ScopedEnvironment<std::string> find_enforce(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
        ScopedEnvironment<int> tuning_iterations_max(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvAsmBwdWrW1x1");

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_Conv2dSingleAsmBwdWrw_FP16, TestFloat16)
{
    if(IsTestSupportedForDevice() && !get_handle_xnack())
    {
        ScopedEnvironment<std::string> find_enforce(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
        ScopedEnvironment<int> tuning_iterations_max(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvAsmBwdWrW1x1");

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_Conv2dSingleAsmBwdWrw_BFP16, TestBFloat16)
{
    if(IsTestSupportedForDevice() && !get_handle_xnack())
    {
        ScopedEnvironment<std::string> find_enforce(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
        ScopedEnvironment<int> tuning_iterations_max(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvAsmBwdWrW1x1");

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
}

TEST_P(GPU_Conv2dSingleAsmBwdWrw_CompilerRegression_FP32, TestCompRegrFloat32)
{
    if(IsTestSupportedForDevice() && !get_handle_xnack())
    {
        ScopedEnvironment<std::string> find_enforce(MIOPEN_FIND_ENFORCE, "SEARCH");
        ScopedEnvironment<int> tuning_iterations_max(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 1);
        ScopedEnvironment<std::string> find_mode(MIOPEN_FIND_MODE, "normal");
        ScopedEnvironment<std::string> conv_direct_perf_vals(
            MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS, "2,8,4,2,4,2,2,4,0,2");
        ScopedEnvironment<std::string> find_only_solver(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                        "ConvAsmBwdWrW1x1");

        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(id, ...)                                 \
    INSTANTIATE_MIOPEN_SMOKE_TEST(id, GPU_Conv2dSingleAsmBwdWrw_FP32, float, __VA_ARGS__); \
    INSTANTIATE_MIOPEN_SMOKE_TEST(                                                         \
        id, GPU_Conv2dSingleAsmBwdWrw_FP16, half_float::half, __VA_ARGS__);                \
    INSTANTIATE_MIOPEN_SMOKE_TEST(id, GPU_Conv2dSingleAsmBwdWrw_BFP16, bfloat16, __VA_ARGS__)

#define INSTANTIATE_MIOPEN_FLOAT_SMOKE_COMP_REGR_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_SMOKE_TEST(                                   \
        id, GPU_Conv2dSingleAsmBwdWrw_CompilerRegression_FP32, float, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT_SMOKE_TEST_SUITE(
    0, {1, 4, 5, 5}, {4, 4, 1, 1}, {0, 0, 2, 2, 1, 1}, false, false);

INSTANTIATE_MIOPEN_FLOAT_SMOKE_COMP_REGR_TEST_SUITE(
    0, {1, 4, 6, 6}, {4, 4, 1, 1}, {0, 0, 1, 1, 1, 1}, false, false);
