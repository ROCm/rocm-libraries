// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/env.hpp>
#include <sstream>

using Problem = miopen::conv::ProblemDescription;
using Config  = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupFwdXdlops;

struct PerfConfigTestCase
{
    struct group_conv::GroupConvTestConfig<3u> conv;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
    std::string arch;
};

std::vector<PerfConfigTestCase> GetPerfConfigTestCases(miopenDataType_t data_type, std::string arch)
{
    return {{{1, 128, 64, 32, {3, 28, 28}, {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
             data_type,
             miopenTensorNCDHW,
             arch},
            {{1, 128, 64, 192, {3, 28, 28}, {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
             data_type,
             miopenTensorNCDHW,
             arch}};
}

static Problem MakeSearchProblem(group_conv::GroupConvTestConfig<3u> conv,
                                 miopenDataType_t data_type,
                                 miopenTensorLayout_t layout)
{
    const auto x_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetInput());
    const auto w_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetWeights());
    const auto conv_desc = conv.GetConv();
    const auto y_desc    = conv_desc.GetForwardOutputTensor(x_desc, w_desc, data_type);

    return Problem(x_desc, w_desc, y_desc, conv_desc, miopen::conv::Direction::Forward);
}

template <miopenDataType_t date_type>
class PerfConfig_HipImplicitGemm3DGroupFwdXdlops
    : public ::testing::TestWithParam<PerfConfigTestCase>
{
protected:
    void TestConfigs()
    {
        auto test_case = GetParam();

        auto&& handle = get_handle();
        miopen::ExecutionContext ctx(&handle);
        if(test_case.arch != ctx.GetStream().GetDeviceName())
            GTEST_SKIP();

        auto input_tensor_desc =
            miopen::TensorDescriptor(test_case.data_type, test_case.conv.GetInput());

        auto weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());

        auto conv_desc = test_case.conv.GetConv();

        auto output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);

        auto problem = miopen::conv::ProblemDescription(input_tensor_desc,
                                                        weights_tensor_desc,
                                                        output_desc,
                                                        conv_desc,
                                                        miopen::conv::Direction::Forward);

        Config cfg;
        cfg.HeuristicInit(ctx, problem);
        EXPECT_TRUE(cfg.index != 0) << "index is 0:" << test_case.conv;
    }
};

using GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16 =
    PerfConfig_HipImplicitGemm3DGroupFwdXdlops<miopenBFloat16>;
using GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16 =
    PerfConfig_HipImplicitGemm3DGroupFwdXdlops<miopenHalf>;

TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16, All) { TestConfigs(); }
TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16, All) { TestConfigs(); }

TEST(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeSearchProblem({1, 64, 32, 32, {28, 28, 28}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                          miopenHalf,
                          miopenTensorNDHWC);

    Config cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front());
    EXPECT_TRUE(cfg.IsValid(problem));
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16,
                         testing::ValuesIn(GetPerfConfigTestCases(miopenBFloat16, "gfx942")));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16,
                         testing::ValuesIn(GetPerfConfigTestCases(miopenHalf, "gfx942")));
