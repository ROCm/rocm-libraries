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

// An elementwise input carrying only a subset of the output modes would have to
// be broadcast along the modes it lacks. That isn't implemented, and it used to
// surface as HIPTENSOR_STATUS_INTERNAL_ERROR from the execute call long after
// the descriptor and the plan had been accepted. See ROCm#6560.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

constexpr int32_t mMode = 'm';
constexpr int32_t nMode = 'n';

constexpr uint32_t alignment = 256;

// Fixture holding the descriptors shared by every case: a rank-2 {m,n} tensor,
// a rank-2 {n,m} tensor that only reorders those modes, and a rank-1 {n} tensor
// holding a strict subset of them.
class ElementwiseModeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);

        int64_t fullLengths[2]     = {2, 3};
        int64_t permutedLengths[2] = {3, 2};
        int64_t subsetLengths[1]   = {3};

        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mFull, 2, fullLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mPermuted, 2, permutedLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mSubset, 1, subsetLengths, nullptr, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        hiptensorDestroyTensorDescriptor(mSubset);
        hiptensorDestroyTensorDescriptor(mPermuted);
        hiptensorDestroyTensorDescriptor(mFull);
        hiptensorDestroy(mHandle);
    }

    hiptensorHandle_t           mHandle{};
    hiptensorTensorDescriptor_t mFull{};
    hiptensorTensorDescriptor_t mPermuted{};
    hiptensorTensorDescriptor_t mSubset{};

    int32_t mFullModes[2]     = {mMode, nMode};
    int32_t mPermutedModes[2] = {nMode, mMode};
    int32_t mSubsetModes[1]   = {nMode};
};

TEST_F(ElementwiseModeTest, PermutationWithSubsetModesReturnsNotSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreatePermutation(mHandle,
                                         &opDesc,
                                         mSubset,
                                         mSubsetModes,
                                         HIPTENSOR_OP_IDENTITY,
                                         mFull,
                                         mFullModes,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
}

TEST_F(ElementwiseModeTest, PermutationWithReorderedModesIsSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreatePermutation(mHandle,
                                         &opDesc,
                                         mFull,
                                         mFullModes,
                                         HIPTENSOR_OP_IDENTITY,
                                         mPermuted,
                                         mPermutedModes,
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    hiptensorDestroyOperationDescriptor(opDesc);
}

TEST_F(ElementwiseModeTest, BinaryWithSubsetModesReturnsNotSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreateElementwiseBinary(mHandle,
                                               &opDesc,
                                               mSubset,
                                               mSubsetModes,
                                               HIPTENSOR_OP_IDENTITY,
                                               mFull,
                                               mFullModes,
                                               HIPTENSOR_OP_IDENTITY,
                                               mFull,
                                               mFullModes,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
}

TEST_F(ElementwiseModeTest, BinaryWithReorderedModesIsSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreateElementwiseBinary(mHandle,
                                               &opDesc,
                                               mFull,
                                               mFullModes,
                                               HIPTENSOR_OP_IDENTITY,
                                               mFull,
                                               mFullModes,
                                               HIPTENSOR_OP_IDENTITY,
                                               mPermuted,
                                               mPermutedModes,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    hiptensorDestroyOperationDescriptor(opDesc);
}

// The reproducer from issue(https://github.com/ROCm/ROCm/issues/6560): A{n} against B, C and D of {m,n}.
TEST_F(ElementwiseModeTest, TrinaryWithSubsetModesReturnsNotSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    EXPECT_EQ(hiptensorCreateElementwiseTrinary(mHandle,
                                                &opDesc,
                                                mSubset,
                                                mSubsetModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
}

TEST_F(ElementwiseModeTest, TrinaryWithReorderedModesIsSupported)
{
    hiptensorOperationDescriptor_t opDesc{};
    ASSERT_EQ(hiptensorCreateElementwiseTrinary(mHandle,
                                                &opDesc,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mFull,
                                                mFullModes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mPermuted,
                                                mPermutedModes,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    hiptensorDestroyOperationDescriptor(opDesc);
}
