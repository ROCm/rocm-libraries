/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Regression tests for the workspace contract:
//  - hiptensorEstimateWorkspaceSize only looked at kernels recorded in the preference by an
//    earlier plan, so it returned 0 for a fresh preference (and failed for a null one);
//  - HIPTENSOR_PLAN_REQUIRED_WORKSPACE returned the limit the plan was created with instead of
//    the workspace its kernel needs, and copied as many bytes as the caller asked for.
// Each plan is then executed with exactly the workspace it reports, and checked against a plain
// loop reference computed here.

#include "api_test_helpers.hpp"

#include <cmath>

using hiptensor_test::ApiObjects;
using hiptensor_test::DeviceBuffer;

namespace
{
    constexpr uint64_t largeLimit = uint64_t(1) << 30;

    uint64_t requiredWorkspace(ApiObjects& api, hiptensorPlan_t plan)
    {
        uint64_t required = ~uint64_t(0);
        EXPECT_EQ(hiptensorPlanGetAttribute(api.handle,
                                            plan,
                                            HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                            &required,
                                            sizeof(required)),
                  HIPTENSOR_STATUS_SUCCESS);
        return required;
    }

    struct Workspace
    {
        explicit Workspace(uint64_t size)
            : size(size)
        {
            if(size > 0)
            {
                CHECK_HIP_ERROR(hipMalloc(&ptr, size));
            }
        }
        ~Workspace()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
        void*    ptr = nullptr;
        uint64_t size;
    };
} // namespace

TEST(WorkspaceSizeTest, ContractionReportsTheWorkspaceItsKernelNeeds)
{
    ApiObjects api;

    constexpr int64_t M = 16, N = 12, K = 8;
    auto              descA = api.tensor({M, K}, {1, M}, HIPTENSOR_R_32F);
    auto              descB = api.tensor({N, K}, {1, N}, HIPTENSOR_R_32F);
    auto              descD = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    int32_t const     modeA[] = {'m', 'k'}, modeB[] = {'n', 'k'}, modeD[] = {'m', 'n'};

    hiptensorOperationDescriptor_t op     = nullptr;
    auto                           status = hiptensorCreateContraction(api.handle,
                                                     &op,
                                                     descA,
                                                     modeA,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descB,
                                                     modeB,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descD,
                                                     modeD,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descD,
                                                     modeD,
                                                     HIPTENSOR_COMPUTE_DESC_32F);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "No f32 contraction on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);

    // The estimate does not depend on the preference: a null one, a fresh one, and one that was
    // used to create a plan give the same value.
    uint64_t estimateNull = ~uint64_t(0);
    ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &estimateNull),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlanPreference_t pref = nullptr;
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);
    uint64_t estimateFresh = ~uint64_t(0);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, &estimateFresh),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(estimateFresh, estimateNull);

    uint64_t estimateMin = ~uint64_t(0);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, pref, HIPTENSOR_WORKSPACE_MIN, &estimateMin),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_LE(estimateMin, estimateNull);

    // A plan created with a large limit reports what its kernel needs, not the limit.
    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(hiptensorCreatePlan(api.handle, &plan, op, pref, largeLimit),
              HIPTENSOR_STATUS_SUCCESS);
    api.plans.push_back(plan);
    auto const required = requiredWorkspace(api, plan);
    EXPECT_LE(required, estimateNull);
    EXPECT_LT(required, largeLimit);

    uint64_t estimateUsed = ~uint64_t(0);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, &estimateUsed),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(estimateUsed, estimateNull);
    hiptensorDestroyPlanPreference(pref);

    // The attribute is a uint64_t.
    uint64_t wide[2] = {7, 7};
    EXPECT_EQ(hiptensorPlanGetAttribute(
                  api.handle, plan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, wide, sizeof(wide)),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(wide[1], 7u);

    // Execute with exactly the required workspace.
    std::vector<float> a(M * K), b(N * K), c(M * N);
    for(int64_t i = 0; i < M * K; ++i)
    {
        a[i] = static_cast<float>(i % 5 - 2);
    }
    for(int64_t i = 0; i < N * K; ++i)
    {
        b[i] = static_cast<float>(i % 3 - 1);
    }
    for(int64_t i = 0; i < M * N; ++i)
    {
        c[i] = static_cast<float>(i % 7 - 3);
    }
    DeviceBuffer<float> dA(a), dB(b), dC(c), dD(std::vector<float>(M * N, NAN));
    Workspace           workspace(required);
    float const         alpha = 0.5f, beta = 2.0f;
    ASSERT_EQ(hiptensorContract(api.handle,
                                plan,
                                &alpha,
                                dA.ptr,
                                dB.ptr,
                                &beta,
                                dC.ptr,
                                dD.ptr,
                                workspace.ptr,
                                workspace.size,
                                nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto d = dD.read();
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t m = 0; m < M; ++m)
        {
            double sum = 0;
            for(int64_t k = 0; k < K; ++k)
            {
                sum += double(a[m + M * k]) * b[n + N * k];
            }
            EXPECT_EQ(d[m + M * n], 0.5 * sum + 2.0 * c[m + M * n]) << m << ", " << n;
        }
    }
}

TEST(WorkspaceSizeTest, TrinaryContractionReportsItsIntermediateAndKernelWorkspace)
{
    ApiObjects api;

    // E_{m,n} = alpha * A_{m,k} B_{k,j} C_{j,n}; the intermediate T is {m, j}.
    constexpr int64_t M = 8, K = 4, J = 6, N = 10;
    auto              descA = api.tensor({M, K}, {1, M}, HIPTENSOR_R_32F);
    auto              descB = api.tensor({K, J}, {1, K}, HIPTENSOR_R_32F);
    auto              descC = api.tensor({J, N}, {1, J}, HIPTENSOR_R_32F);
    auto              descE = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    int32_t const     modeA[] = {'m', 'k'}, modeB[] = {'k', 'j'}, modeC[] = {'j', 'n'},
                  modeE[] = {'m', 'n'};

    hiptensorOperationDescriptor_t op     = nullptr;
    auto                           status = hiptensorCreateContractionTrinary(api.handle,
                                                            &op,
                                                            descA,
                                                            modeA,
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descB,
                                                            modeB,
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descC,
                                                            modeC,
                                                            HIPTENSOR_OP_IDENTITY,
                                                            nullptr,
                                                            nullptr,
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descE,
                                                            modeE,
                                                            HIPTENSOR_COMPUTE_DESC_32F);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "No f32 contraction on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);

    uint64_t const intermediateBytes = (M * J * sizeof(float) + 255) / 256 * 256;

    uint64_t estimate = 0;
    ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, op, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_GE(estimate, intermediateBytes);

    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(api.plan(op, plan, largeLimit), HIPTENSOR_STATUS_SUCCESS);
    auto const required = requiredWorkspace(api, plan);
    EXPECT_GE(required, intermediateBytes);
    EXPECT_LE(required, estimate);
    EXPECT_LT(required, largeLimit);

    std::vector<float> a(M * K), b(K * J), c(J * N);
    for(int64_t i = 0; i < M * K; ++i)
    {
        a[i] = static_cast<float>(i % 5 - 2);
    }
    for(int64_t i = 0; i < K * J; ++i)
    {
        b[i] = static_cast<float>(i % 3 - 1);
    }
    for(int64_t i = 0; i < J * N; ++i)
    {
        c[i] = static_cast<float>(i % 4 - 2);
    }
    DeviceBuffer<float> dA(a), dB(b), dC(c), dE(std::vector<float>(M * N, NAN));
    Workspace           workspace(required);
    float const         alpha = 1.5f, beta = 0.0f;
    ASSERT_EQ(hiptensorContractTrinary(api.handle,
                                       plan,
                                       &alpha,
                                       dA.ptr,
                                       dB.ptr,
                                       dC.ptr,
                                       &beta,
                                       nullptr,
                                       dE.ptr,
                                       workspace.ptr,
                                       workspace.size,
                                       nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto e = dE.read();
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t m = 0; m < M; ++m)
        {
            double sum = 0;
            for(int64_t j = 0; j < J; ++j)
            {
                double t = 0;
                for(int64_t k = 0; k < K; ++k)
                {
                    t += double(a[m + M * k]) * b[k + K * j];
                }
                sum += t * c[j + J * n];
            }
            EXPECT_EQ(e[m + M * n], 1.5 * sum) << m << ", " << n;
        }
    }
}

TEST(WorkspaceSizeTest, ElementwiseAndReductionPlansNeedNoWorkspace)
{
    ApiObjects    api;
    auto          desc2   = api.tensor({8, 4}, {}, HIPTENSOR_R_32F);
    auto          desc1   = api.tensor({8}, {}, HIPTENSOR_R_32F);
    int32_t const modes[] = {'m', 'n'};

    hiptensorOperationDescriptor_t permute = nullptr, reduce = nullptr;
    ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                         &permute,
                                         desc2,
                                         modes,
                                         HIPTENSOR_OP_IDENTITY,
                                         desc2,
                                         modes,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(permute);
    ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                       &reduce,
                                       desc2,
                                       modes,
                                       HIPTENSOR_OP_IDENTITY,
                                       desc1,
                                       modes,
                                       HIPTENSOR_OP_IDENTITY,
                                       desc1,
                                       modes,
                                       HIPTENSOR_OP_ADD,
                                       HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(reduce);

    for(auto op : {permute, reduce})
    {
        uint64_t estimate = ~uint64_t(0);
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api.handle, op, nullptr, HIPTENSOR_WORKSPACE_MAX, &estimate),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(estimate, 0u);

        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan, largeLimit), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(requiredWorkspace(api, plan), 0u);
    }

    uint64_t estimate = 0;
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, nullptr, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  api.handle, permute, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
}
