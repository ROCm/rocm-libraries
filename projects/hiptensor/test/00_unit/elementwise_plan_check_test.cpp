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

// hiptensorCreatePlan checks that an elementwise operation (permutation, binary, trinary) has a
// kernel that can run it. The elementwise kernels serve outputs of rank 2 to 6, so a plan for an
// output of another rank is rejected with HIPTENSOR_STATUS_NOT_SUPPORTED when it is created
// instead of failing when it is executed, and the plan handle is left unchanged. No kernel is
// launched.

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

#include <cstdint>
#include <vector>

namespace
{
    constexpr uint32_t alignment = 256;

    // Stored in the plan handle before hiptensorCreatePlan, to check that a failed call leaves
    // the handle unchanged. It is never destroyed.
    hiptensorPlan_t const kUnsetPlan = reinterpret_cast<hiptensorPlan_t>(0x1);

    class ElementwisePlanCheckTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hiptensorCreatePlanPreference(
                          mHandle, &mPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                      HIPTENSOR_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(mPlan != kUnsetPlan && mPlan != nullptr)
            {
                hiptensorDestroyPlan(mPlan);
            }
            if(mOpDesc != nullptr)
            {
                hiptensorDestroyOperationDescriptor(mOpDesc);
            }
            for(auto desc : mTensors)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
            if(mPref != nullptr)
            {
                hiptensorDestroyPlanPreference(mPref);
            }
            if(mHandle != nullptr)
            {
                hiptensorDestroy(mHandle);
            }
        }

        // A packed FP32 tensor with `rank` modes of extent 2, named 'a', 'b', ...
        hiptensorTensorDescriptor_t makeTensor(uint32_t rank)
        {
            std::vector<int64_t>        lengths(rank, 2);
            hiptensorTensorDescriptor_t desc{};
            EXPECT_EQ(
                hiptensorCreateTensorDescriptor(
                    mHandle, &desc, rank, lengths.data(), nullptr, HIPTENSOR_R_32F, alignment),
                HIPTENSOR_STATUS_SUCCESS);
            mTensors.push_back(desc);
            return desc;
        }

        static std::vector<int32_t> modes(uint32_t rank)
        {
            std::vector<int32_t> result(rank);
            for(uint32_t i = 0; i < rank; i++)
            {
                result[i] = 'a' + static_cast<int32_t>(i);
            }
            return result;
        }

        hiptensorStatus_t createPermutation(uint32_t rank)
        {
            auto desc = makeTensor(rank);
            auto mode = modes(rank);
            EXPECT_EQ(hiptensorCreatePermutation(mHandle,
                                                 &mOpDesc,
                                                 desc,
                                                 mode.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 desc,
                                                 mode.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            return hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, mPref, 0);
        }

        hiptensorStatus_t createBinary(uint32_t rank)
        {
            auto desc = makeTensor(rank);
            auto mode = modes(rank);
            EXPECT_EQ(hiptensorCreateElementwiseBinary(mHandle,
                                                       &mOpDesc,
                                                       desc,
                                                       mode.data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       desc,
                                                       mode.data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       desc,
                                                       mode.data(),
                                                       HIPTENSOR_OP_ADD,
                                                       HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            return hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, mPref, 0);
        }

        hiptensorStatus_t createTrinary(uint32_t rank)
        {
            auto desc = makeTensor(rank);
            auto mode = modes(rank);
            EXPECT_EQ(hiptensorCreateElementwiseTrinary(mHandle,
                                                        &mOpDesc,
                                                        desc,
                                                        mode.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        desc,
                                                        mode.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        desc,
                                                        mode.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        desc,
                                                        mode.data(),
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            return hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, mPref, 0);
        }

        hiptensorHandle_t                        mHandle{};
        hiptensorPlanPreference_t                mPref{};
        hiptensorOperationDescriptor_t           mOpDesc{};
        hiptensorPlan_t                          mPlan = kUnsetPlan;
        std::vector<hiptensorTensorDescriptor_t> mTensors;
    };
} // namespace

TEST_F(ElementwisePlanCheckTest, PermutationRank2IsSupported)
{
    EXPECT_EQ(createPermutation(2), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_NE(mPlan, kUnsetPlan);
    EXPECT_NE(mPlan, nullptr);
}

TEST_F(ElementwisePlanCheckTest, PermutationRank1ReturnsNotSupported)
{
    EXPECT_EQ(createPermutation(1), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(mPlan, kUnsetPlan);
}

TEST_F(ElementwisePlanCheckTest, PermutationRank7ReturnsNotSupported)
{
    EXPECT_EQ(createPermutation(7), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(mPlan, kUnsetPlan);
}

TEST_F(ElementwisePlanCheckTest, BinaryRank6IsSupported)
{
    EXPECT_EQ(createBinary(6), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_NE(mPlan, kUnsetPlan);
    EXPECT_NE(mPlan, nullptr);
}

TEST_F(ElementwisePlanCheckTest, BinaryRank7ReturnsNotSupported)
{
    EXPECT_EQ(createBinary(7), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(mPlan, kUnsetPlan);
}

TEST_F(ElementwisePlanCheckTest, TrinaryRank2IsSupported)
{
    EXPECT_EQ(createTrinary(2), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_NE(mPlan, kUnsetPlan);
    EXPECT_NE(mPlan, nullptr);
}

TEST_F(ElementwisePlanCheckTest, TrinaryRank7ReturnsNotSupported)
{
    EXPECT_EQ(createTrinary(7), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(mPlan, kUnsetPlan);
}
