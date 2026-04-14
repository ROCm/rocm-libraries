// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

using Problem = miopen::conv::ProblemDescription;
using Config  = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupBwdXdlops;

static Problem MakeSearchProblem(group_conv::GroupConvTestConfig<3u> conv,
                                 miopenDataType_t data_type,
                                 miopenTensorLayout_t layout)
{
    const auto x_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetInput());
    const auto w_desc    = miopen::TensorDescriptor(data_type, layout, conv.GetWeights());
    const auto conv_desc = conv.GetConv();
    const auto y_desc    = conv_desc.GetForwardOutputTensor(x_desc, w_desc, data_type);

    return Problem(y_desc, w_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardData);
}

TEST(GPU_PerfConfig_HipImplicitGemm3DGroupBwdXdlops_FP16, SearchStartsAtKernelZero)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.empty())
        GTEST_SKIP() << "No ROCm-capable device is detected";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem =
        MakeSearchProblem({1, 1, 4, 4, {8, 28, 28}, {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                          miopenHalf,
                          miopenTensorNDHWC);

    Config cfg(false);
    ASSERT_TRUE(cfg.SetNextValue(problem));
    ASSERT_FALSE(cfg.valid_kernels.empty());
    EXPECT_EQ(cfg.index, 0);
    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels.front());
    EXPECT_TRUE(cfg.IsValid(problem));
}
