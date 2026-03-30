/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

class PlanLifetimeTest : public ::testing::TestWithParam<
                             std::tuple<bool /*destroyOpDesc*/, bool /*destroyPref*/>>
{
};

TEST_P(PlanLifetimeTest, PermuteAfterDescriptorDestroy)
{
    auto [destroyOpDesc, destroyPref] = GetParam();

    constexpr int64_t M      = 4;
    constexpr int64_t N      = 5;
    constexpr size_t  nelems = M * N;

    hiptensorHandle_t handle{};
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    int64_t lensA[2] = {M, N};
    int64_t lensB[2] = {N, M};
    int32_t modeA[2] = {0, 1};
    int32_t modeB[2] = {1, 0};

    hiptensorTensorDescriptor_t descA{}, descB{};
    ASSERT_EQ(hiptensorCreateTensorDescriptor(
                  handle, &descA, 2, lensA, nullptr, HIPTENSOR_R_32F, 128),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreateTensorDescriptor(
                  handle, &descB, 2, lensB, nullptr, HIPTENSOR_R_32F, 128),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreatePermutation(handle,
                                         &opDesc,
                                         descA,
                                         modeA,
                                         HIPTENSOR_OP_IDENTITY,
                                         descB,
                                         modeB,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlanPreference_t pref{};
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);

    uint64_t wsSize = 0;
    ASSERT_EQ(
        hiptensorEstimateWorkspaceSize(handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
        HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlan_t plan{};
    ASSERT_EQ(hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize), HIPTENSOR_STATUS_SUCCESS);

    if(destroyOpDesc)
    {
        ASSERT_EQ(hiptensorDestroyOperationDescriptor(opDesc), HIPTENSOR_STATUS_SUCCESS);
        opDesc = nullptr;
    }
    if(destroyPref)
    {
        ASSERT_EQ(hiptensorDestroyPlanPreference(pref), HIPTENSOR_STATUS_SUCCESS);
        pref = nullptr;
    }

    std::vector<float> hA(nelems), hB(nelems, 0.0f);
    for(size_t i = 0; i < nelems; ++i)
        hA[i] = static_cast<float>(i + 1);

    float *dA = nullptr, *dB = nullptr;
    ASSERT_EQ(hipMalloc(&dA, sizeof(float) * nelems), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, sizeof(float) * nelems), hipSuccess);
    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(float) * nelems, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dB, 0, sizeof(float) * nelems), hipSuccess);

    hipStream_t stream{};
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    float alpha = 1.0f;
    ASSERT_EQ(hiptensorPermute(handle, plan, &alpha, dA, dB, stream), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(hB.data(), dB, sizeof(float) * nelems, hipMemcpyDeviceToHost), hipSuccess);

    for(int64_t i = 0; i < N; ++i)
        for(int64_t j = 0; j < M; ++j)
            EXPECT_FLOAT_EQ(hB[i + j * N], hA[j + i * M])
                << "Mismatch at B[" << i << "," << j << "]";

    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipStreamDestroy(stream);
    hiptensorDestroyPlan(plan);
    if(opDesc)
        hiptensorDestroyOperationDescriptor(opDesc);
    if(pref)
        hiptensorDestroyPlanPreference(pref);
    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroy(handle);
}

INSTANTIATE_TEST_SUITE_P(PlanLifetime,
                         PlanLifetimeTest,
                         ::testing::Values(std::make_tuple(false, false),
                                           std::make_tuple(true, false),
                                           std::make_tuple(false, true),
                                           std::make_tuple(true, true)),
                         [](const ::testing::TestParamInfo<PlanLifetimeTest::ParamType>& info)
                             -> std::string {
                             bool destroyOp   = std::get<0>(info.param);
                             bool destroyPref  = std::get<1>(info.param);
                             if(!destroyOp && !destroyPref)
                                 return "KeepAll";
                             if(destroyOp && !destroyPref)
                                 return "DestroyOpDesc";
                             if(!destroyOp && destroyPref)
                                 return "DestroyPref";
                             return "DestroyBoth";
                         });
