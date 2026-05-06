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
              bool enable_forward,
              bool enable_backward_data,
              bool enable_backward_weights)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.base_params.do_forward          = {enable_forward};
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
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::Default>;
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
struct smoke_solver_ConvCkIgemmFwdV6r1DlopsNchw : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2dTuningV6R1_FP16 = smoke_solver_ConvCkIgemmFwdV6r1DlopsNchw<half_float::half>;

TEST_P(GPU_Conv2dTuningV6R1_FP16, TestFloat16)
{
    if(IsTestSupportedForDevice())
    {
        // MIOPEN_DEBUG_TUNING_ITERATIONS_MAX is set to 2 because kernels are very slow to build.
        // MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW is explicitly enabled due to the kernel is
        // disabled by default via #2306
        std::vector<std::string> env_vars{
            "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
            "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=2",
            "MIOPEN_FIND_MODE=normal",
            "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvCkIgemmFwdV6r1DlopsNchw",
            "MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=true"};

        SetEnvVars(env_vars);
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT16_SMOKE_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_SMOKE_TEST(id, GPU_Conv2dTuningV6R1_FP16, bfloat16, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT16_SMOKE_TEST_SUITE(
    0, {128, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1}, true, false, false);
