/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <gtest/gtest.h>
#include <gtest/gtest_common.hpp>
#include <miopen/miopen.h>
#include <miopen/fusion.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/find_solution.hpp>

#include "tensor_holder.hpp"
#include "get_handle.hpp"
#include "cba.hpp"
#include "gtest_desc_guard.hpp"
#include "../lib_env_var.hpp"

#if MIOPEN_BACKEND_HIP
namespace {
bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X, Gpu::gfx115X>;
    // gfx120X is not enabled due to WORKAROUND_SWDEV_479810
    using d_mask = disabled<Gpu::None>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

bool IsWorkspaceTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X>;
    // requires ConvCKIgemmGrpFwdBiasActivFused solver
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

template <typename T>
class FusionSetArgTest : public ConvBiasActivInferTest<T>
{
public:
    void SetUp() override
    {
        cba<T>::SetUp();
        weights2 = tensor<T>{cba<T>::tensor_layout, cba<T>::conv_config.GetWeights()};
        weights2.generate(tensor_elem_gen_integer{3});
        cba<T>::weights = weights2;
        auto&& handle   = get_handle();
        cba<T>::wei_dev = handle.Write(weights2.data);
        handle.Finish();
    }

    void TearDown() override { cba<T>::TearDown(); }

    template <typename Tp>
    using cba = ConvBiasActivInferTest<Tp>;

    tensor<T> weights2;
    miopen::Allocator::ManageDataPtr wei_dev2;
};

bool SkipTest() { return get_handle_xnack(); }

} // namespace

using GPU_FusionSetArg_FP16 = FusionSetArgTest<float>;

TEST_P(GPU_FusionSetArg_FP16, TestSetArgApiCall)
{
    // Original fusion_plan/args execution happens in cba_infer.cpp
    // Original is checked independently and not sequentially, prior to FusionTestSetArgTest.

    if(SkipTest())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion does not support xnack";
    }
    if(!IsTestSupportedForDevice())
    {
        test_skipped = true;
        GTEST_SKIP() << "CBA fusion_test is not supported for this device";
    }

    using cba_float = cba<float>;

    auto&& handle = get_handle();
    auto convOp   = std::make_shared<miopen::ConvForwardOpDescriptor>(cba_float::conv_desc,
                                                                    cba_float::weights.desc);
    miopenOperatorArgs_t fusion_args = static_cast<miopenOperatorArgs_t>(&(cba_float::params));
    miopenFusionPlanDescriptor_t fusion_plan =
        static_cast<miopenFusionPlanDescriptor_t>(&(cba_float::fusePlanDesc));
    miopenFusionOpDescriptor_t conv_op = static_cast<miopenFusionOpDescriptor_t>(convOp.get());

    size_t workspace_size = 0;
    miopenConvFwdAlgorithm_t algo{}; // not used in GetWorkSpaceSize
    EXPECT_EQ(miopenFusionPlanGetWorkSpaceSize(&handle, fusion_plan, &workspace_size, algo),
              miopenStatusSuccess)
        << "Workspace size for a non-compiled fusion plan failure";

    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args,
                                         conv_op,
                                         &(cba_float::alpha),
                                         &(cba_float::beta),
                                         cba_float::wei_dev.get()),
              0);

    workspace_size = 0;
    EXPECT_EQ(miopenFusionPlanGetWorkSpaceSize(&handle, fusion_plan, &workspace_size, algo),
              miopenStatusSuccess)
        << "Workspace size for a compiled fusion plan failure";

    cba_float::wspace.resize(workspace_size);

    EXPECT_EQ(miopenExecuteFusionPlan_v2(&handle,
                                         fusion_plan,
                                         &(cba_float::input.desc),
                                         cba_float::in_dev.get(),
                                         &(cba_float::output.desc),
                                         cba_float::out_dev.get(),
                                         fusion_args,
                                         cba_float::wspace.ptr(),
                                         cba_float::wspace.size()),
              0);
    handle.Finish();
    using ConvParam       = miopen::fusion::ConvolutionOpInvokeParam;
    ConvParam* conv_param = dynamic_cast<ConvParam*>(miopen::deref(fusion_args).params[0].get());

    ASSERT_EQ(conv_param->weights, wei_dev.get());
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_FusionSetArg_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)));

TEST(CPU_FusionCreateOpConvForward_FP32, TestInvalidConvLayout)
{
    std::vector<int> xDims{4, 4, 4, 4};
    std::vector<int> xStrides{1, 4, 16, 64}; // WHCN order

    std::vector<int> wDims{1, 4, 4, 4};
    std::vector<int> wStrides{16, 4, 1, 1};

    std::vector<int> padding{0, 0};
    std::vector<int> dilation{1, 1};
    std::vector<int> stride{1, 1};

    TensorDescGuard xDesc;
    miopenSetTensorDescriptor(
        xDesc, miopenDataType_t::miopenFloat, xDims.size(), xDims.data(), xStrides.data());

    TensorDescGuard wDesc;
    miopenSetTensorDescriptor(
        wDesc, miopenDataType_t::miopenFloat, wDims.size(), wDims.data(), wStrides.data());

    miopenFusionPlanDescriptor_t fusionPlanDesc;
    miopenCreateFusionPlan(&fusionPlanDesc, miopenVerticalFusion, xDesc);

    ConvDescGuard convDesc;
    miopenInitConvolutionNdDescriptor(convDesc,
                                      2,
                                      padding.data(),
                                      stride.data(),
                                      dilation.data(),
                                      miopenConvolutionMode_t::miopenConvolution);

    miopenFusionOpDescriptor_t convOp;
    auto status = miopenCreateOpConvForward(fusionPlanDesc, &convOp, convDesc, wDesc);
    EXPECT_EQUAL(status, miopenStatusUnknownError);

    miopenDestroyFusionPlan(fusionPlanDesc);
}

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)

// The test uses a specific fusion configuration that triggers the solver.
// ConvCKIgemmGrpFwdBiasActivFused
// The test runs two cases:
// 1) with exact or larger workspace size than required (expected to pass)
// 2) with smaller workspace size than required (expected to return error)

// check if ConvCKIgemmGrpFwdBiasActivFused solver is supported on the device
bool IsConvCKIgemmGrpFwdBiasActivFusedSolverSupported(miopen::FusionPlanDescriptor& fusion_plan)
{
    const auto fusion_problem = miopen::FusionDescription{&fusion_plan};
    auto fusion_ctx           = miopen::FusionContext{get_handle()};

    miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused solv{};
    return solv.IsApplicable(fusion_ctx, fusion_problem);
}

//
// Check that only ConvCKIgemmGrpFwdBiasActivFused solver is applicable for the fusion problem
// This is to ensure that no other solver interferes with the test
//
bool IsOnlyConvCKIgemmGrpFwdBiasActivFusedSolverApplicable(
    miopen::FusionPlanDescriptor& fusion_plan)
{
    const auto fusion_problem = miopen::FusionDescription{&fusion_plan};
    auto ctx                  = miopen::FusionContext{get_handle()};

    const auto target_id = miopen::solver::Id("ConvCKIgemmGrpFwdBiasActivFused");
    std::vector<miopen::solver::Id> ids =
        miopen::debug::GetAllApplicableFusionSolutions(ctx, fusion_problem);
    // find all fusion solvers that are applicable but not expected
    for(auto id : ids)
    {
        if(id.Value() != target_id.Value())
            return false;
    }
    return true;
}

template <typename T>
class GPU_CBAFind2FusionWorkspace : public ConvBiasActivInferTest<T>
{
public:
    using cba_base = ConvBiasActivInferTest<T>;
    // Setup should be extanded to add some specific fields for Fusion
    void SetUp() override
    {
        cba_base::SetUp();
        fusion_args = static_cast<miopenOperatorArgs_t>(&(cba_base::params));
        fusion_plan = static_cast<miopenFusionPlanDescriptor_t>(&(cba_base::fusePlanDesc));
        miopen::deref(fusion_plan).findMode.Set(miopen::FindMode::Values::Normal);
    }

    void RunTest(bool PositiveTest)
    {
        if(IsConvCKIgemmGrpFwdBiasActivFusedSolverSupported(cba_base::fusePlanDesc) == false)
        {
            this->test_skipped = true;
            GTEST_SKIP()
                << "ConvCKIgemmGrpFwdBiasActivFused solver is not supported on this device";
        }
        miopen::solver::debug::TuningIterationScopedLimiter tuning_limit{5};
        auto&& handle         = get_handle();
        size_t workspace_size = 0;
        miopenConvFwdAlgorithm_t algo{}; // not used in GetWorkSpaceSize
        {
            ScopedEnvironment<bool> find_mode_env1(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1, false);
            ASSERT_TRUE(
                IsOnlyConvCKIgemmGrpFwdBiasActivFusedSolverApplicable(cba_base::fusePlanDesc))
                << "Test configuration is invalid as other solvers are applicable. Please update "
                   "the test case.";
            EXPECT_EQ(miopenFusionPlanGetWorkSpaceSize(&handle, fusion_plan, &workspace_size, algo),
                      miopenStatusSuccess)
                << "Workspace size for a non-compiled fusion plan failure";
            EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), miopenStatusSuccess);
        }

        workspace_size = 0;
        EXPECT_EQ(miopenFusionPlanGetWorkSpaceSize(&handle, fusion_plan, &workspace_size, algo),
                  miopenStatusSuccess)
            << "Workspace size for a compiled fusion plan failure";

        // This test requires a case with a non-zero workspace size.
        // If this check fails, the test configuration needs to be updated
        // to a case that requires workspace.
        ASSERT_TRUE(workspace_size > 0)
            << "Test configuration does not require workspace. Please update the test case.";

        if(PositiveTest)
        {
            // Test with exact workspace size
            cba_base::wspace.resize(workspace_size);

            EXPECT_EQ(miopenExecuteFusionPlan_v2(&handle,
                                                 fusion_plan,
                                                 &(cba_base::input.desc),
                                                 cba_base::in_dev.get(),
                                                 &(cba_base::output.desc),
                                                 cba_base::out_dev.get(),
                                                 fusion_args,
                                                 cba_base::wspace.ptr(),
                                                 cba_base::wspace.size()),
                      miopenStatusSuccess);

            // Test with a larger workspace than required
            cba_base::wspace.resize(workspace_size + 10);
            EXPECT_EQ(miopenExecuteFusionPlan_v2(&handle,
                                                 fusion_plan,
                                                 &(cba_base::input.desc),
                                                 cba_base::in_dev.get(),
                                                 &(cba_base::output.desc),
                                                 cba_base::out_dev.get(),
                                                 fusion_args,
                                                 cba_base::wspace.ptr(),
                                                 cba_base::wspace.size()),
                      miopenStatusSuccess);
        }
        else
        {
            // Test with a smaller workspace than required
            // Should return miopenStatusBadParm
            cba_base::wspace.resize(workspace_size - 10);
            EXPECT_EQ(miopenExecuteFusionPlan_v2(&handle,
                                                 fusion_plan,
                                                 &(cba_base::input.desc),
                                                 cba_base::in_dev.get(),
                                                 &(cba_base::output.desc),
                                                 cba_base::out_dev.get(),
                                                 fusion_args,
                                                 cba_base::wspace.ptr(),
                                                 cba_base::wspace.size()),
                      miopenStatusBadParm);
            test_verification = false;
        }
        handle.Finish();
    }

    void TearDown() override
    {
        if(test_verification)
            cba_base::TearDown();
    }
    miopenOperatorArgs_t fusion_args;
    miopenFusionPlanDescriptor_t fusion_plan;
    bool test_verification = true;
};

using GPU_CBAFind2FusionWorkspace_FP32 = GPU_CBAFind2FusionWorkspace<float>;

TEST_P(GPU_CBAFind2FusionWorkspace_FP32, CBAFind2_testFindWorkspace)
{
    if(SkipTest())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion does not support xnack";
    }
    if(!IsWorkspaceTestSupportedForDevice())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion not supported for this device";
    }
    RunTest(true);
}

TEST_P(GPU_CBAFind2FusionWorkspace_FP32, CBAFind2_testWorkspaceInvalidSize)
{
    if(SkipTest())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion does not support xnack";
    }
    if(!IsWorkspaceTestSupportedForDevice())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion not supported for this device";
    }
    RunTest(false);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_CBAFind2FusionWorkspace_FP32,
    testing::Combine(testing::Values(miopenActivationRELU),
                     testing::Values(ConvTestCaseBase{
                         1, 64, 52, 53, 63, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution}),
                     // try to use unique case that uses ConvCKIgemmGrpFwdBiasActivFused solver
                     // to avoid interference with other tests
                     testing::Values(miopenTensorNCHW),
                     testing::Values(0.25f),
                     testing::Values(0.25f),
                     testing::Values(0.25f)));

#endif
