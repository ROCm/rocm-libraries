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
              std::string in_layout,
              std::string fil_layout,
              std::string out_layout,
              bool enable_backward_data,
              bool enable_backward_weights,
              size_t tensor_vect,
              size_t vector_length)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations          = {std::move(pads_strides_dilations)};
    baseParams.in_layout                       = {std::move(in_layout)};
    baseParams.fil_layout                      = {std::move(fil_layout)};
    baseParams.out_layout                      = {std::move(out_layout)};
    baseParams.base_params.do_backward_data    = {enable_backward_data};
    baseParams.base_params.do_backward_weights = {enable_backward_weights};
    baseParams.tensor_vect                     = {tensor_vect};
    baseParams.vector_length                   = {vector_length};

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
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
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
struct smoke_solver_ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC
    : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using GPU_Conv2dTuningDynamicFwdDlops_FP16 =
    smoke_solver_ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC<half_float::half>;

TEST_P(GPU_Conv2dTuningDynamicFwdDlops_FP16, TestFloat16)
{
    if(IsTestSupportedForDevice() && !get_handle_xnack())
    {
        std::vector<std::string> env_vars{
            "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
            "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
            "MIOPEN_FIND_MODE=normal",
            "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC"};

        SetEnvVars(env_vars);
        run();
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
};

#define INSTANTIATE_MIOPEN_FLOAT16_SMOKE_TEST_SUITE(id, ...) \
    INSTANTIATE_MIOPEN_SMOKE_TEST(                           \
        id, GPU_Conv2dTuningDynamicFwdDlops_FP16, half_float::half, __VA_ARGS__)

INSTANTIATE_MIOPEN_FLOAT16_SMOKE_TEST_SUITE(0,
                                            {64, 256, 7, 7},
                                            {256, 3, 3, 128},
                                            {0, 0, 1, 1, 1, 1},
                                            "NCHW",
                                            "CHWN",
                                            "NCHW",
                                            false,
                                            false,
                                            1,
                                            4);
