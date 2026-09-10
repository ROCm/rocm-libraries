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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// End-to-end checks that an elementwise input carrying a subset of the output modes is
// broadcast along the modes it lacks (ROCm#6560). The 02_elementwise suite also covers
// broadcast, but compares the kernel against hipTensor's own CPU reference, which reaches
// the kernel arguments through the same stride-alignment code. The expected values here are
// written out by hand instead, so a fault shared by both paths cannot pass unnoticed.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <vector>

#include "hiptensor_options.hpp"
#include "util.hpp"

constexpr int32_t mMode = 'm';
constexpr int32_t nMode = 'n';

constexpr uint32_t alignment = 256;

constexpr int64_t M = 8;
constexpr int64_t N = 5;

// A rank-2 {m,n} output of extents {M,N}, plus the rank-1 inputs that broadcast into it.
class ElementwiseBroadcastTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);

        int64_t outLengths[2] = {M, N};
        int64_t mLengths[1]   = {M};
        int64_t nLengths[1]   = {N};

        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mDescOut, 2, outLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mDescAlongM, 1, mLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mDescAlongN, 1, nLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);

        // The kernel derives the output strides from its lengths, so the offset of (i,j) in the
        // result follows whichever convention the library is built or configured for.
        auto outStrides = hiptensor::stridesFromLengths(
            std::vector<int64_t>{M, N},
            hiptensor::HiptensorOptions::instance()->isColMajorStrides());
        mStrideM = outStrides[0];
        mStrideN = outStrides[1];

        ASSERT_EQ(hipMalloc(&mDeviceAlongM, M * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&mDeviceAlongN, N * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&mDeviceOut, M * N * sizeof(float)), hipSuccess);

        mHostAlongM.resize(M);
        mHostAlongN.resize(N);
        mHostOut.assign(M * N, 0.0f);
        for(int64_t i = 0; i < M; i++)
        {
            mHostAlongM[i] = static_cast<float>(100 * (i + 1));
        }
        for(int64_t j = 0; j < N; j++)
        {
            mHostAlongN[j] = static_cast<float>(j + 1);
        }
        ASSERT_EQ(
            hipMemcpy(mDeviceAlongM, mHostAlongM.data(), M * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(mDeviceAlongN, mHostAlongN.data(), N * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipFree(mDeviceOut), hipSuccess);
        EXPECT_EQ(hipFree(mDeviceAlongN), hipSuccess);
        EXPECT_EQ(hipFree(mDeviceAlongM), hipSuccess);
        hiptensorDestroyTensorDescriptor(mDescAlongN);
        hiptensorDestroyTensorDescriptor(mDescAlongM);
        hiptensorDestroyTensorDescriptor(mDescOut);
        hiptensorDestroy(mHandle);
    }

    // Runs the plan's output back to the host so a test can index it by (i,j).
    void readOutput()
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(mHostOut.data(), mDeviceOut, M * N * sizeof(float), hipMemcpyDeviceToHost),
            hipSuccess);
    }

    float outputAt(int64_t i, int64_t j) const
    {
        return mHostOut[i * mStrideM + j * mStrideN];
    }

    // Wraps the boilerplate between an operation descriptor and a runnable plan.
    hiptensorPlan_t makePlan(hiptensorOperationDescriptor_t opDesc)
    {
        hiptensorPlanPreference_t planPref{};
        EXPECT_EQ(hiptensorCreatePlanPreference(
                      mHandle, &planPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);
        hiptensorPlan_t plan{};
        EXPECT_EQ(hiptensorCreatePlan(mHandle, &plan, opDesc, planPref, 0),
                  HIPTENSOR_STATUS_SUCCESS);
        hiptensorDestroyPlanPreference(planPref);
        return plan;
    }

    hiptensorHandle_t           mHandle{};
    hiptensorTensorDescriptor_t mDescOut{};
    hiptensorTensorDescriptor_t mDescAlongM{};
    hiptensorTensorDescriptor_t mDescAlongN{};

    void* mDeviceAlongM{};
    void* mDeviceAlongN{};
    void* mDeviceOut{};

    std::vector<float> mHostAlongM;
    std::vector<float> mHostAlongN;
    std::vector<float> mHostOut;

    int64_t mStrideM{};
    int64_t mStrideN{};

    int32_t mOutModes[2] = {mMode, nMode};
    int32_t mModesM[1]   = {mMode};
    int32_t mModesN[1]   = {nMode};
};

// D{m,n} = alpha * A{n}: A lacks mode m, so every row of D repeats the whole of A.
TEST_F(ElementwiseBroadcastTest, PermutationRepeatsTheInputAlongTheMissingMode)
{
    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreatePermutation(mHandle,
                                         &opDesc,
                                         mDescAlongN,
                                         mModesN,
                                         HIPTENSOR_OP_IDENTITY,
                                         mDescOut,
                                         mOutModes,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);

    auto  plan  = makePlan(opDesc);
    float alpha = 2.0f;
    ASSERT_EQ(hiptensorPermute(mHandle, plan, &alpha, mDeviceAlongN, mDeviceOut, nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    readOutput();

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            EXPECT_FLOAT_EQ(outputAt(i, j), alpha * mHostAlongN[j])
                << "at (" << i << ", " << j << ")";
        }
    }

    hiptensorDestroyPlan(plan);
    hiptensorDestroyOperationDescriptor(opDesc);
}

// D{m,n} = alpha * A{n} + gamma * C{m}: the two inputs are broadcast along opposite modes, so
// every element of D pairs a different (A, C) combination.
TEST_F(ElementwiseBroadcastTest, BinaryBroadcastsTwoInputsAlongOppositeModes)
{
    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreateElementwiseBinary(mHandle,
                                               &opDesc,
                                               mDescAlongN,
                                               mModesN,
                                               HIPTENSOR_OP_IDENTITY,
                                               mDescAlongM,
                                               mModesM,
                                               HIPTENSOR_OP_IDENTITY,
                                               mDescOut,
                                               mOutModes,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);

    auto  plan  = makePlan(opDesc);
    float alpha = 2.0f;
    float gamma = 3.0f;
    ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                  mHandle, plan, &alpha, mDeviceAlongN, &gamma, mDeviceAlongM, mDeviceOut, nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    readOutput();

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            EXPECT_FLOAT_EQ(outputAt(i, j), alpha * mHostAlongN[j] + gamma * mHostAlongM[i])
                << "at (" << i << ", " << j << ")";
        }
    }

    hiptensorDestroyPlan(plan);
    hiptensorDestroyOperationDescriptor(opDesc);
}
