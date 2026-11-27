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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Contractions whose tensors have padded (non-packed) strides, planned with the selection
// algorithms that time every candidate kernel on scratch buffers during hiptensorCreatePlan.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

namespace
{
    // Runs a cleanup function when the scope exits, including through a failed ASSERT_* or
    // GTEST_SKIP, which return from the test body early.
    template <typename F>
    struct ScopeExit
    {
        F mFn;
        ~ScopeExit()
        {
            mFn();
        }
    };
    template <typename F>
    ScopeExit(F) -> ScopeExit<F>;

    // Deliberately avoids printing the status name: CTest treats any output containing
    // the name of HIPTENSOR_STATUS_ARCH_MISMATCH as a skip of the whole binary.
    bool isSkipStatus(hiptensorStatus_t status)
    {
        return status == HIPTENSOR_STATUS_ARCH_MISMATCH || status == HIPTENSOR_STATUS_NOT_SUPPORTED;
    }

    std::string algoName(hiptensorAlgo_t algo)
    {
        return algo == HIPTENSOR_ALGO_DEFAULT_PATIENT ? "DefaultPatient" : "Default";
    }
} // namespace

class ContractionPaddedStridesTest : public ::testing::TestWithParam<hiptensorAlgo_t>
{
};

// D[m,n] = alpha * sum_k A[m,k] * B[n,k] + beta * C[m,n] in FP32, column-major (first mode
// fastest), with the second-mode stride of A, B, C and D padded far past the extent of the
// first mode. Each tensor then spans several MB, while its packed size is a few KB.
TEST_P(ContractionPaddedStridesTest, BilinearPaddedLeadingStrides)
{
    hiptensorAlgo_t const algo = GetParam();

    constexpr int64_t M   = 32;
    constexpr int64_t N   = 32;
    constexpr int64_t K   = 32;
    constexpr int64_t ldA = 64 * 1024; // packed: M
    constexpr int64_t ldB = 48 * 1024; // packed: N
    constexpr int64_t ldD = 40 * 1024; // packed: M; C and D share the layout

    // Elements spanned by each tensor: one past the offset of its last element.
    constexpr int64_t spanA = (M - 1) + (K - 1) * ldA + 1;
    constexpr int64_t spanB = (N - 1) + (K - 1) * ldB + 1;
    constexpr int64_t spanD = (M - 1) + (N - 1) * ldD + 1;

    // Values outside the tensors stay NaN, so a kernel that reads the padding is caught.
    float const        pad = std::nanf("");
    std::vector<float> hA(spanA, pad), hB(spanB, pad), hC(spanD, pad), hD(spanD, pad);
    for(int64_t k = 0; k < K; k++)
    {
        for(int64_t m = 0; m < M; m++)
        {
            hA[m + k * ldA] = static_cast<float>((m + 2 * k) % 7 - 3);
        }
        for(int64_t n = 0; n < N; n++)
        {
            hB[n + k * ldB] = static_cast<float>((3 * n + k) % 5 - 2);
        }
    }
    for(int64_t n = 0; n < N; n++)
    {
        for(int64_t m = 0; m < M; m++)
        {
            hC[m + n * ldD] = static_cast<float>((m + n) % 9 - 4);
        }
    }

    float const alpha = 1.5f;
    float const beta  = -0.5f;

    hiptensorHandle_t              handle = nullptr;
    hiptensorTensorDescriptor_t    descA = nullptr, descB = nullptr, descD = nullptr;
    hiptensorOperationDescriptor_t opDesc = nullptr;
    hiptensorPlanPreference_t      pref   = nullptr;
    hiptensorPlan_t                plan   = nullptr;
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dWs = nullptr;

    ScopeExit cleanup{[&]() {
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
        (void)hipFree(dWs);
        if(plan)
            hiptensorDestroyPlan(plan);
        if(pref)
            hiptensorDestroyPlanPreference(pref);
        if(opDesc)
            hiptensorDestroyOperationDescriptor(opDesc);
        if(descA)
            hiptensorDestroyTensorDescriptor(descA);
        if(descB)
            hiptensorDestroyTensorDescriptor(descB);
        if(descD)
            hiptensorDestroyTensorDescriptor(descD);
        if(handle)
            hiptensorDestroy(handle);
    }};

    auto status = hiptensorCreate(&handle);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << "device not supported";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    int64_t lensA[2]    = {M, K};
    int64_t stridesA[2] = {1, ldA};
    int64_t lensB[2]    = {N, K};
    int64_t stridesB[2] = {1, ldB};
    int64_t lensD[2]    = {M, N};
    int64_t stridesD[2] = {1, ldD};
    int32_t modeA[2]    = {'m', 'k'};
    int32_t modeB[2]    = {'n', 'k'};
    int32_t modeD[2]    = {'m', 'n'};

    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descA, 2, lensA, stridesA, HIPTENSOR_R_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descB, 2, lensB, stridesB, HIPTENSOR_R_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descD, 2, lensD, stridesD, HIPTENSOR_R_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);

    status = hiptensorCreateContraction(handle,
                                        &opDesc,
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
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << "contraction not supported on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    ASSERT_EQ(hiptensorCreatePlanPreference(handle, &pref, algo, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);

    uint64_t wsSize = 0;
    ASSERT_EQ(
        hiptensorEstimateWorkspaceSize(handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
        HIPTENSOR_STATUS_SUCCESS);

    // Times every candidate kernel on scratch buffers, which must cover the padded tensors.
    status = hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize);
    if(isSkipStatus(status))
    {
        GTEST_SKIP() << "no contraction kernel for this problem on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMalloc(&dA, sizeof(float) * spanA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, sizeof(float) * spanB), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC, sizeof(float) * spanD), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, sizeof(float) * spanD), hipSuccess);
    if(wsSize > 0)
    {
        ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    }
    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(float) * spanA, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), sizeof(float) * spanB, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dC, hC.data(), sizeof(float) * spanD, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dD, hD.data(), sizeof(float) * spanD, hipMemcpyHostToDevice), hipSuccess);

    ASSERT_EQ(hiptensorContract(handle, plan, &alpha, dA, dB, &beta, dC, dD, dWs, wsSize, nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, sizeof(float) * spanD, hipMemcpyDeviceToHost), hipSuccess);

    // Inputs are small integers, so every partial sum is exact in FP32.
    int mismatches = 0;
    for(int64_t n = 0; n < N; n++)
    {
        for(int64_t m = 0; m < M; m++)
        {
            double acc = 0.0;
            for(int64_t k = 0; k < K; k++)
            {
                acc += double(hA[m + k * ldA]) * double(hB[n + k * ldB]);
            }
            double const expected = alpha * acc + beta * double(hC[m + n * ldD]);
            double const actual   = hD[m + n * ldD];
            if(!(std::abs(actual - expected) <= 1e-5 * (1.0 + std::abs(expected))))
            {
                if(mismatches++ < 10)
                {
                    ADD_FAILURE() << "D[" << m << "," << n << "] = " << actual << ", expected "
                                  << expected;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0);
}

INSTANTIATE_TEST_SUITE_P(ContractionPaddedStrides,
                         ContractionPaddedStridesTest,
                         ::testing::Values(HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT),
                         [](::testing::TestParamInfo<hiptensorAlgo_t> const& info) {
                             return algoName(info.param);
                         });
