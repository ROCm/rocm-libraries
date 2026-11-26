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

// Argument handling of hiptensorCreatePlan and hiptensorEstimateWorkspaceSize: null
// arguments are rejected before anything is allocated, a null plan preference selects the
// defaults, and a failed plan creation leaves the caller's plan handle untouched.

#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

namespace
{
    // Stands in for the caller's previous plan handle. Never dereferenced or destroyed.
    char                  sentinelStorage;
    hiptensorPlan_t const kSentinelPlan = reinterpret_cast<hiptensorPlan_t>(&sentinelStorage);

    constexpr int64_t kM = 2;
    constexpr int64_t kN = 3;

    class CreatePlanArgsTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(mPlan != nullptr && mPlan != kSentinelPlan)
            {
                hiptensorDestroyPlan(mPlan);
            }
            if(mPref != nullptr)
            {
                hiptensorDestroyPlanPreference(mPref);
            }
            if(mOpDesc != nullptr)
            {
                hiptensorDestroyOperationDescriptor(mOpDesc);
            }
            for(auto desc : {mDescA, mDescB, mDescD})
            {
                if(desc != nullptr)
                {
                    hiptensorDestroyTensorDescriptor(desc);
                }
            }
            if(mStream != nullptr)
            {
                static_cast<void>(hipStreamDestroy(mStream));
            }
            static_cast<void>(hipFree(mDevA));
            static_cast<void>(hipFree(mDevB));
            if(mHandle != nullptr)
            {
                hiptensorDestroy(mHandle);
            }
        }

        // B[n, m] = A[m, n] in FP32.
        void createPermutation()
        {
            int64_t const lensA[2] = {kM, kN};
            int64_t const lensB[2] = {kN, kM};
            ASSERT_EQ(hiptensorCreateTensorDescriptor(
                          mHandle, &mDescA, 2, lensA, nullptr, HIPTENSOR_R_32F, 128),
                      HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hiptensorCreateTensorDescriptor(
                          mHandle, &mDescB, 2, lensB, nullptr, HIPTENSOR_R_32F, 128),
                      HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hiptensorCreatePermutation(mHandle,
                                                 &mOpDesc,
                                                 mDescA,
                                                 mModesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mDescB,
                                                 mModesB,
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
        }

        // D[m, n] = A[m, k] * B[n, k] in FP32.
        hiptensorStatus_t createContraction()
        {
            int64_t const lensA[2]  = {kM, kN};
            int64_t const lensB[2]  = {kN, kN};
            int64_t const lensD[2]  = {kM, kN};
            int32_t const modesA[2] = {'m', 'k'};
            int32_t const modesB[2] = {'n', 'k'};
            int32_t const modesD[2] = {'m', 'n'};

            auto status = hiptensorCreateTensorDescriptor(
                mHandle, &mDescA, 2, lensA, nullptr, HIPTENSOR_R_32F, 128);
            if(status == HIPTENSOR_STATUS_SUCCESS)
            {
                status = hiptensorCreateTensorDescriptor(
                    mHandle, &mDescB, 2, lensB, nullptr, HIPTENSOR_R_32F, 128);
            }
            if(status == HIPTENSOR_STATUS_SUCCESS)
            {
                status = hiptensorCreateTensorDescriptor(
                    mHandle, &mDescD, 2, lensD, nullptr, HIPTENSOR_R_32F, 128);
            }
            if(status == HIPTENSOR_STATUS_SUCCESS)
            {
                status = hiptensorCreateContraction(mHandle,
                                                    &mOpDesc,
                                                    mDescA,
                                                    modesA,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mDescB,
                                                    modesB,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    nullptr,
                                                    nullptr,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    mDescD,
                                                    modesD,
                                                    HIPTENSOR_COMPUTE_DESC_32F);
            }
            return status;
        }

        void createPreference(hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT)
        {
            ASSERT_EQ(hiptensorCreatePlanPreference(mHandle, &mPref, algo, HIPTENSOR_JIT_MODE_NONE),
                      HIPTENSOR_STATUS_SUCCESS);
        }

        hiptensorHandle_t              mHandle    = nullptr;
        hiptensorTensorDescriptor_t    mDescA     = nullptr;
        hiptensorTensorDescriptor_t    mDescB     = nullptr;
        hiptensorTensorDescriptor_t    mDescD     = nullptr;
        hiptensorOperationDescriptor_t mOpDesc    = nullptr;
        hiptensorPlanPreference_t      mPref      = nullptr;
        hiptensorPlan_t                mPlan      = nullptr;
        hipStream_t                    mStream    = nullptr;
        float*                         mDevA      = nullptr;
        float*                         mDevB      = nullptr;
        int32_t                        mModesA[2] = {'m', 'n'};
        int32_t                        mModesB[2] = {'n', 'm'};
    };
} // namespace

TEST_F(CreatePlanArgsTest, NullHandleLeavesPlanUnchanged)
{
    ASSERT_NO_FATAL_FAILURE(createPermutation());
    ASSERT_NO_FATAL_FAILURE(createPreference());

    mPlan = kSentinelPlan;
    EXPECT_EQ(hiptensorCreatePlan(nullptr, &mPlan, mOpDesc, mPref, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(mPlan, kSentinelPlan);
}

TEST_F(CreatePlanArgsTest, NullPlanPointerIsRejected)
{
    ASSERT_NO_FATAL_FAILURE(createPermutation());
    ASSERT_NO_FATAL_FAILURE(createPreference());

    EXPECT_EQ(hiptensorCreatePlan(mHandle, nullptr, mOpDesc, mPref, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
}

TEST_F(CreatePlanArgsTest, NullDescriptorLeavesPlanUnchanged)
{
    ASSERT_NO_FATAL_FAILURE(createPreference());

    mPlan = kSentinelPlan;
    EXPECT_EQ(hiptensorCreatePlan(mHandle, &mPlan, nullptr, mPref, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(mPlan, kSentinelPlan);
}

// The plan is allocated before kernel selection fails, so a failure from selection
// is the path where the plan must be freed instead of returned.
TEST_F(CreatePlanArgsTest, FailedSelectionLeavesPlanUnchanged)
{
    auto status = createContraction();
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH || status == HIPTENSOR_STATUS_NOT_SUPPORTED)
    {
        GTEST_SKIP() << "contraction not supported on this device";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    // No selection algorithm has this value, so no kernel is selected.
    auto const unknownAlgo = static_cast<hiptensorAlgo_t>(0);
    if(hiptensorCreatePlanPreference(mHandle, &mPref, unknownAlgo, HIPTENSOR_JIT_MODE_NONE)
       != HIPTENSOR_STATUS_SUCCESS)
    {
        mPref = nullptr;
        GTEST_SKIP() << "the library rejects an unknown algorithm in the preference";
    }
    auto const cacheMode = HIPTENSOR_CACHE_MODE_NONE;
    ASSERT_EQ(
        hiptensorPlanPreferenceSetAttribute(
            mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode)),
        HIPTENSOR_STATUS_SUCCESS);

    mPlan = kSentinelPlan;
    EXPECT_NE(hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, mPref, 0), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(mPlan, kSentinelPlan);
}

TEST_F(CreatePlanArgsTest, NullPreferenceUsesDefaults)
{
    ASSERT_NO_FATAL_FAILURE(createPermutation());

    ASSERT_EQ(hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, nullptr, 0), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_NE(mPlan, nullptr);

    constexpr size_t   nelems = kM * kN;
    std::vector<float> hostA(nelems);
    std::vector<float> hostB(nelems, 0.0f);
    for(size_t i = 0; i < nelems; ++i)
    {
        hostA[i] = static_cast<float>(i + 1);
    }

    ASSERT_EQ(hipMalloc(&mDevA, sizeof(float) * nelems), hipSuccess);
    ASSERT_EQ(hipMalloc(&mDevB, sizeof(float) * nelems), hipSuccess);
    ASSERT_EQ(hipMemcpy(mDevA, hostA.data(), sizeof(float) * nelems, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(mDevB, 0, sizeof(float) * nelems), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&mStream), hipSuccess);

    float alpha = 1.0f;
    ASSERT_EQ(hiptensorPermute(mHandle, mPlan, &alpha, mDevA, mDevB, mStream),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(mStream), hipSuccess);
    ASSERT_EQ(hipMemcpy(hostB.data(), mDevB, sizeof(float) * nelems, hipMemcpyDeviceToHost),
              hipSuccess);

    for(int64_t m = 0; m < kM; ++m)
    {
        for(int64_t n = 0; n < kN; ++n)
        {
            EXPECT_EQ(hostB[n + m * kN], hostA[m + n * kM]) << "Mismatch at m=" << m << ", n=" << n;
        }
    }
}

TEST_F(CreatePlanArgsTest, EstimateWorkspaceSizeRejectsNullArguments)
{
    ASSERT_NO_FATAL_FAILURE(createPermutation());
    ASSERT_NO_FATAL_FAILURE(createPreference());

    uint64_t workspaceSize = 1234;
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  nullptr, mOpDesc, mPref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  mHandle, nullptr, mPref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(workspaceSize, 1234u);
    EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                  mHandle, mOpDesc, mPref, HIPTENSOR_WORKSPACE_DEFAULT, nullptr),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
}
