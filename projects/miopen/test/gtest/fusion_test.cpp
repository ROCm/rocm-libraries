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

#include "tensor_holder.hpp"
#include "get_handle.hpp"
#include "cba.hpp"
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
    using d_mask = disabled<Gpu::None>;
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

    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args,
                                         conv_op,
                                         &(cba_float::alpha),
                                         &(cba_float::beta),
                                         cba_float::wei_dev.get()),
              0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle,
                                      fusion_plan,
                                      &(cba_float::input.desc),
                                      cba_float::in_dev.get(),
                                      &(cba_float::output.desc),
                                      cba_float::out_dev.get(),
                                      fusion_args),
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

    miopenTensorDescriptor_t xDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenSetTensorDescriptor(
        xDesc, miopenDataType_t::miopenFloat, xDims.size(), xDims.data(), xStrides.data());

    miopenTensorDescriptor_t wDesc;
    miopenCreateTensorDescriptor(&wDesc);
    miopenSetTensorDescriptor(
        wDesc, miopenDataType_t::miopenFloat, wDims.size(), wDims.data(), wStrides.data());

    miopenFusionPlanDescriptor_t fusionPlanDesc;
    miopenCreateFusionPlan(&fusionPlanDesc, miopenVerticalFusion, xDesc);

    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenInitConvolutionNdDescriptor(convDesc,
                                      2,
                                      padding.data(),
                                      stride.data(),
                                      dilation.data(),
                                      miopenConvolutionMode_t::miopenConvolution);

    miopenFusionOpDescriptor_t convOp;
    auto status = miopenCreateOpConvForward(fusionPlanDesc, &convOp, convDesc, wDesc);
    EXPECT_EQUAL(status, miopenStatusUnknownError);
}

MIOPEN_LIB_ENV_VAR(MIOPEN_FIND_MODE_FUSION)

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
    }

    void RunTest(bool PositiveTest)
    {

        miopen::solver::debug::TuningIterationScopedLimiter tuning_limit{5};

        auto&& handle = get_handle();
        {
            ScopedEnvironment<std::string> find_mode_env1(MIOPEN_FIND_MODE_FUSION,
                                                          std::string("normal"));
            ScopedEnvironment<std::string> find_mode_env2(
                MIOPEN_DEBUG_FIND_ONLY_SOLVER, std::string("ConvCKIgemmGrpFwdBiasActivFused"));

            EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), miopenStatusSuccess);
        }

        size_t workspace_size = 0;
        miopenConvFwdAlgorithm_t algo{}; // not used in GetWorkSpaceSize
        EXPECT_EQ(miopenFusionPlanGetWorkSpaceSize(&handle, fusion_plan, &workspace_size, algo),
                  miopenStatusSuccess);

        // This test requires a case with a non-zero workspace size.
        // If this check fails, the test configuration needs to be updated
        // to a case that requires workspace.
        EXPECT_TRUE(workspace_size > 0);

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
